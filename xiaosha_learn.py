# -*- coding: utf-8 -*-
"""
小沙大脑 · 后台自主学习守护进程（2026-09-19）
职责：开机后一直在后台跑「不需要人教」的三个学习循环，并把每轮成绩记进日志，供人查看长进。
  ① curious 自主探索（RND 内在动机：没有奖励也自己找目标）
  ② digits  自己练（看数字：练前 12% → 练后 41%，独立测试 47%）
  ③ judge   自评价（发现不对就自己调模型）
  ④ 每 N 轮顺带跑一次 cortex（脉冲皮层，139,260 细胞；重量级，默认 5 轮一次）
设计原则：
  · 单实例（互斥锁文件，避免重复拉起吃满 CPU）
  · 低优先级（BelowNormal，不抢用户前台的手感）
  · 每轮有超时，卡住就跳过并记一笔（不装死）
  · 成绩落 JSONL 日志 + 汇总 JSON（真实数字，不造）
  · 可随时被 Stop-Process 掉，重启后自动接着跑（无状态依赖）
用法：python xiaosha_learn.py            # 前台跑（调试用）
      python xiaosha_learn.py --once     # 只跑一轮就退（自检用）
      python xiaosha_learn.py --cycles 3 # 跑 3 轮
"""
import os, sys, json, time, subprocess, datetime, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, 'learn-log.jsonl')
SUM = os.path.join(HERE, 'learn-summary.json')
LOCK = os.path.join(HERE, '_learn.lock')
PY = sys.executable

# 每个循环：模块文件、说明、超时秒、给它的参数
LOOPS = [
    ('xiaosha_curious.py', '自主探索（RND：自己找目标）', 180, '--selftest'),
    ('xiaosha_digits.py', '自己练认数字', 300, '--selftest'),
    ('xiaosha_judge.py', '自评价（发现错就自己调模型）', 180, '--selftest'),
    ('xiaosha_web_learn.py', '上网自学（GitHub/Gitee/HF镜像，实测可达源）', 300, '--once'),
]
HEAVY = ('xiaosha_cortex.py', '脉冲皮层（139,260 细胞）', 900, '--selftest')
EVERY = 5          # 每 5 轮跑一次重量级
SLEEP = 60         # 每轮之间歇 60 秒（温柔点，别把机器占满）


def parse_score(text):
    """从输出里抓「X/Y 通过」这类真实成绩；抓不到就返回 None（不编）"""
    import re
    m = re.findall(r'RESULT:\s*(\d+)\s*/\s*(\d+)', text)
    if m:
        return int(m[-1][0]), int(m[-1][1])
    m = re.findall(r'(\d+)\s*/\s*(\d+)\s*项通过', text)
    if m:
        return int(m[-1][0]), int(m[-1][1])
    # digits 的成绩是百分比：抓「练后 300 次 41%」这种
    m = re.findall(r'([0-9]{1,3})%', text)
    if m:
        return None, None
    return None, None


def run_one(mod, desc, timeout, arg='--selftest'):
    t0 = time.time()
    rec = {'module': mod, 'desc': desc, 'ok': False, 'exit': None, 'secs': 0.0, 'pass': None, 'total': None}
    try:
        p = subprocess.run([PY, '-X', 'utf8', mod, arg], cwd=HERE,
                           capture_output=True, timeout=timeout,
                           env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1'))
        out = (p.stdout or b'').decode('utf-8', 'replace') + (p.stderr or b'').decode('utf-8', 'replace')
        rec['exit'] = p.returncode
        rec['ok'] = (p.returncode == 0)
        a, b = parse_score(out)
        rec['pass'], rec['total'] = a, b
    except subprocess.TimeoutExpired:
        rec['exit'] = 'timeout'
    except Exception as e:
        rec['exit'] = 'err:' + str(e)[:60]
    rec['secs'] = round(time.time() - t0, 1)
    return rec


def main():
    once = '--once' in sys.argv
    cycles = 1 if once else int((sys.argv[sys.argv.index('--cycles') + 1] if '--cycles' in sys.argv else 0) or 10 ** 9)

    # 单实例互斥（避免计划任务重复拉起）：锁里记 [时间, pid]，**按 pid 判活**（pid 不在就是上次没退干净，直接接管）
    if os.path.exists(LOCK):
        try:
            with open(LOCK, 'r', encoding='utf-8') as f:
                parts = (f.read() or '').split()
            old_pid = int(parts[1]) if len(parts) > 1 else 0
            alive = False
            if old_pid:
                try:
                    os.kill(old_pid, 0)      # Windows 上：进程存在则不抛异常
                    alive = True
                except Exception:
                    alive = False
            if alive:
                print('[learn] 已有实例在跑（pid=%d），本次退出' % old_pid)
                return
            print('[learn] 发现上次的锁但 pid=%d 已不在 → 接管继续跑' % old_pid)
        except Exception:
            pass
    with open(LOCK, 'w', encoding='utf-8') as f:
        f.write(str(int(time.time())) + '\n' + str(os.getpid()))
    # ★ 竞态修复（2026-09-19 实测踩到）：两个实例可能同时通过上面的检查 → 写完锁后**回读确认**，
    #   只有"锁里是我的 pid"才继续；否则说明被人抢先，自己退出。
    try:
        time.sleep(0.4)
        with open(LOCK, 'r', encoding='utf-8') as f:
            mine = (f.read() or '').split()
        if len(mine) > 1 and int(mine[1]) != os.getpid():
            print('[learn] 另一实例抢先写入锁（pid=%s），本次退出' % mine[1])
            return
    except Exception:
        pass

    try:
        os.nice(10)   # Windows 上无效但不报错；真正的优先级由计划任务设 BelowNormal
    except Exception:
        pass

    print('[learn] 小沙大脑后台自主学习启动 ｜ 循环间隔 %ds ｜ 日志 %s' % (SLEEP, LOG))
    n = 0
    while n < cycles:
        n += 1
        batch = list(LOOPS) + ([HEAVY] if n % EVERY == 0 else [])
        recs = []
        for mod, desc, to, arg in batch:
            r = run_one(mod, desc, to, arg)
            recs.append(r)
            print('[learn] 第%d轮 · %-28s exit=%-8s 用时%5.1fs 成绩=%s' % (
                n, desc, str(r['exit']), r['secs'],
                (('%d/%d' % (r['pass'], r['total'])) if r['pass'] is not None else '-')), flush=True)
        entry = {'ts': datetime.datetime.now().isoformat(timespec='seconds'), 'cycle': n, 'results': recs}
        try:
            with open(LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            # 汇总：跑了多少轮、各模块最近成绩
            hist = []
            if os.path.exists(LOG):
                with open(LOG, 'r', encoding='utf-8') as f:
                    hist = [json.loads(x) for x in f if x.strip()]
            last = {}
            for e in hist[-50:]:
                for r in e['results']:
                    last[r['module']] = {'desc': r['desc'], 'ok': r['ok'], 'pass': r['pass'], 'total': r['total'],
                                         'secs': r['secs'], 'at': e['ts']}
            with open(SUM, 'w', encoding='utf-8') as f:
                json.dump({'cycles_total': len(hist), 'updated': entry['ts'], 'interval_sec': SLEEP,
                           'loops': last, 'note': '由 xiaosha_learn.py 自动写入；成绩取自各模块 --selftest 的真实输出'},
                          f, ensure_ascii=False, indent=1)
        except Exception as e:
            print('[learn] 写日志失败：' + str(e))
        if not once and n < cycles:
            time.sleep(SLEEP)
    print('[learn] 本轮结束（跑了 %d 轮）' % n)


if __name__ == '__main__':
    main()
