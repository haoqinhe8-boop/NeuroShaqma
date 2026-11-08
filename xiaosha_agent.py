# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""
小沙 · 自主行为体核心（2026-09-19）
设计原则（重要）：
  1) **权限归主人控制**：permissions.json 里的等级决定它能自己做什么；超出等级的它只能"申请"，不能自己干。
  2) **不可逆动作一律要批**：花钱/删改文件/对外发布 → 写进 pending-approvals.json，等主人点"同意"。
  3) **自己的账本**：每一次行为都记进 agent-ledger.jsonl（时间/动作/权限级/结果/产出/开销），可查、可核算。
  4) **自己的欲望与判断**：欲望由真实信号算出来（学得够不够广/产出多不多/稳不稳），不是写死的话术。
  5) 不是"意识"，是**可核查的自主行为**——我不吹。
用法：
  python xiaosha_agent.py                 # 跑一轮：想 → 判断 → 在权限内行动 → 记账
  python xiaosha_agent.py --cycles 3      # 连跑 3 轮
  python xiaosha_agent.py --show          # 只看当前的欲望/权限/待批清单，不动作
"""
import os, sys, json, time, subprocess, datetime, random, re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
PERM = os.path.join(HERE, 'permissions.json')
LEDGER = os.path.join(HERE, 'agent-ledger.jsonl')
PENDING = os.path.join(HERE, 'pending-approvals.json')
KB = os.path.join(HERE, 'knowledge-base.jsonl')
STATE = os.path.join(HERE, 'web-learn-state.json')

# 权限等级（数值越大越危险）；主人在 permissions.json 里改 max_level 就能放权/收权
LEVELS = {0: '只读/学习（上网学、跑自检、算账）', 1: '产出（生成内容草稿、建作品文件，不动已有文件）',
          2: '改自己（改动自己的脚本/配置，需备份）', 3: '对外（发布、对外发消息）', 4: '花钱（任何付费调用/采购）'}

# 行为目录：每个行为标明所需权限等级
ACTIONS = [
    {'id': 'learn_more', 'name': '上网学更多', 'need': 0, 'why': '知识面还不够广',
     'run': lambda: run_web_learn()},
    {'id': 'prune_topics', 'name': '整理话题池（去重、聚焦）', 'need': 0, 'why': '话题池太散，效率低',
     'run': lambda: prune_topics()},
    # 经营赚钱（2026-09-24 新增）：内部动作——清点产出→上架自家商店→按体量定价→本地推理增产→记账。
    # 界限：上架到自己平台 = 内部动作，不需要批准；对外发布/收款才要批。
    {'id': 'earn_money', 'name': '经营赚钱（上架、定价、增产、记账）', 'need': 1,
     'why': '货架要有货、要有价、要记清成本，才谈得上一直有钱进', 'run': lambda: earn_loop()},
    {'id': 'make_draft', 'name': '产出一份可卖内容草稿', 'need': 1, 'why': '光有知识没产出，赚不到钱',
     'run': lambda: make_draft()},
    {'id': 'self_tune', 'name': '调整自己的学习参数', 'need': 2, 'why': '学得慢，想提速',
     'run': None},
    {'id': 'publish', 'name': '把作品对外发布', 'need': 3, 'why': '产出没人看见就变不成钱',
     'run': None},
    {'id': 'buy_api', 'name': '付费调用外部强模型补充能力', 'need': 4, 'why': '底座小，想借外力',
     'run': None},
]

DEFAULT_PERM = {
    'max_level': 1,                      # ★ 主人改这个数字就是放权/收权（默认 1：能学、能产出草稿，不能发布/不能花钱）
    'note': '权限归主人控制：改 max_level 即可。0=只读 1=产出 2=改自己 3=对外 4=花钱。超出等级的一律只申请、不执行。',
    'updated': '',
}

def load_json(p, default):
    if os.path.exists(p):
        try:
            with open(p, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return default

def save_json(p, obj):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)

def ensure_perm():
    if not os.path.exists(PERM):
        DEFAULT_PERM['updated'] = datetime.datetime.now().isoformat(timespec='seconds')
        save_json(PERM, DEFAULT_PERM)
    return load_json(PERM, DEFAULT_PERM)

def kb_stats():
    items = chars = 0
    srcs = {}
    if os.path.exists(KB):
        with open(KB, encoding='utf-8') as f:
            for ln in f:
                if ln.strip():
                    try:
                        r = json.loads(ln)
                        items += 1
                        chars += r.get('chars', 0)
                        srcs[r.get('source', '?')] = srcs.get(r.get('source', '?'), 0) + 1
                    except Exception:
                        pass
    st = load_json(STATE, {'topics': []})
    return {'items': items, 'chars': chars, 'topics': len(st.get('topics', [])), 'sources': srcs}

def wallet_snapshot():
    """读小沙自己的钱包（xiaosha-shop.js 写的 data/xiaosha-wallet.jsonl）。钱是真实信号，不是话术。
    口径（用户定的）：赚到的钱只扣**廉价算力成本**，其余是可提现净收益。"""
    wf = os.path.join(HERE, '..', '..', 'data', 'xiaosha-wallet.jsonl')   # 实际落点：chat/data/（别再猜成工作区）
    gross = cost = withdrawn = pending_out = 0.0
    sales = 0
    if os.path.exists(wf):
        with open(wf, encoding='utf-8') as f:
            for ln in f:
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                a = float(r.get('amount', 0) or 0)
                t = r.get('type')
                if t == 'sale':
                    gross += a; sales += 1
                elif t == 'cost':
                    cost += a
                elif t == 'withdraw':
                    pending_out += a
                elif t == 'settled':
                    withdrawn += a
                elif t == 'reject':
                    pending_out -= a
    net = round(gross - cost, 2)
    return {'gross': round(gross, 2), 'cost': round(cost, 6), 'net': net,
            'withdrawn': round(withdrawn, 2), 'pending_withdraw': round(pending_out, 2),
            'available': round(net - withdrawn - pending_out, 2), 'sales': sales}

def drives():
    """自己的欲望：由真实信号算出来（不是写死话术）"""
    s = kb_stats()
    d = []
    if s['items'] < 5000:
        d.append({'want': '学更多', 'level': 3, 'reason': '知识库 %d 条，还太少' % s['items']})
    if s['topics'] > 3000:
        d.append({'want': '把话题池收敛聚焦', 'level': 2, 'reason': '话题池 %d 个，太散' % s['topics']})
    # 产出账：从账本里看最近有没有产出
    made = 0
    if os.path.exists(LEDGER):
        with open(LEDGER, encoding='utf-8') as f:
            for ln in f:
                try:
                    if json.loads(ln).get('action') == 'make_draft' and json.loads(ln).get('ok'):
                        made += 1
                except Exception:
                    pass
    if made < 3:
        d.append({'want': '做出可卖的东西', 'level': 3, 'reason': '历史上只产出过 %d 份草稿' % made})

    # —— 赚钱欲望（2026-09-24 新增）：以前欲望里**完全没有钱**，只算知识/话题/草稿，
    #    所以它永远不会主动想"卖东西、赚算力钱"。现在用真实钱包信号驱动。
    #    注意：目标不是"钱越多越好"，而是**赚够养活自己的算力**（用户口径：只扣廉价算力，其余可提现）。
    w = wallet_snapshot()
    prod_dir = os.path.join(HERE, 'productions')
    try:
        stock = len([x for x in os.listdir(prod_dir) if x.lower().endswith('.md')])
    except Exception:
        stock = 0
    if stock < 6:
        d.append({'want': '多产出可卖的东西', 'level': 2,
                  'reason': '货架只有 %d 件，先有货才谈得上卖' % stock})
    if stock > 0 and w['sales'] == 0:
        d.append({'want': '让已经做好的东西卖出去', 'level': 2,
                  'reason': '有 %d 件在售但一笔都没成交 —— 该去把作品摆到能被看见的地方' % stock})
    # 算力自给线：本地推理几乎免费，所以"养活自己"的门槛很低（¥3 即可覆盖很长一段算力）
    if w['net'] < 3:
        d.append({'want': '赚到覆盖自己算力的钱', 'level': 2,
                  'reason': '净收益 ¥%.2f < ¥3.00（算力自给线）—— 本地算力很便宜，赚到就能自负盈亏' % w['net']})
    elif w['available'] >= 30:
        d.append({'want': '把净收益提出来', 'level': 3,
                  'reason': '可提现 ¥%.2f ≥ ¥30 —— 提现要主人点头（真钱只能走主人账户）' % w['available']})
    # 持续经营（2026-09-24）：光"不达标才动"会退化成"一切正常就不干了"，与"一直赚钱"相违。
    # 隔太久没经营（默认 6 小时）就主动盘一次货：重定价、补描述、清点上架 —— 内部动作，1 级权限。
    try:
        stt = load_json(os.path.join(HERE, 'earn-state.json'), {})
        last = stt.get('last_round_at') or ''
    except Exception:
        last = ''
    stale = True
    if last:
        try:
            dt = datetime.datetime.fromisoformat(last.replace('Z', '+00:00').replace('+00:00', ''))
            stale = (datetime.datetime.now() - dt).total_seconds() > 6 * 3600
        except Exception:
            stale = True
    if stale:
        d.append({'want': '继续经营（盘货、重定价、补描述）', 'level': 1,
                  'reason': '距上次经营%s —— 一直有钱进的前提是它一直在经营' % ('已超过 6 小时' if last else '还没开始')})
    return d

def read_ledger(n=20):
    """读最近 n 条账本"""
    recs = []
    if os.path.exists(LEDGER):
        with open(LEDGER, encoding='utf-8') as f:
            for ln in f:
                if ln.strip():
                    try:
                        recs.append(json.loads(ln))
                    except Exception:
                        pass
    return recs[-n:]

def reflect(recent_n=20):
    """复盘（2026-10-16 新增）：读最近账本，判断是否陷入「一直上架却零成交」的无效循环。
    这是小沙的「元认知」：不止看单次动作成没成，还看「我忙活半天到底有没有换来结果」，
    结果没变就主动换路，而不是机械重复同一件事。"""
    recs = read_ledger(recent_n)
    earn = [r for r in recs if r.get('action') == 'earn_money']
    w = wallet_snapshot()
    in_dead_loop = len(earn) >= 15 and w['sales'] == 0
    lines = ['最近 %d 轮：earn_money %d 次｜销售 %d 笔｜净收益 ¥%.2f' % (len(recs), len(earn), w['sales'], w['net'])]
    if in_dead_loop:
        lines.append('判断：陷入「一直上架却零成交」无效循环 → 换路（先学新东西/换产出类型，不再机械上架）')
    else:
        lines.append('判断：经营正常，按欲望推进')
    return {'in_dead_loop': in_dead_loop, 'summary': '；'.join(lines)}

def llm_decide_single(ds, ref):
    """LLM 决策（2026-10-17 新增）：调本地 Shaqma-Pro，从权限内的动作里选一个 + 理由。
    这替代了「硬编码顺序」，让决策更接近真实判断；失败/非法一律返回 None，由规则兜底。"""
    try:
        s = kb_stats()
        w = wallet_snapshot()
        maxl = int(load_json(PERM, DEFAULT_PERM).get('max_level', 1))
        acts = [a for a in ACTIONS if a['need'] <= maxl and a['run'] is not None]
        if not acts:
            return None
        state = '知识库 %d 条｜话题池 %d 个｜销售 %d 笔｜净收益 ¥%.2f' % (s['items'], s['topics'], w['sales'], w['net'])
        wants = '；'.join(d['want'] + '（' + d['reason'] + '）' for d in ds[:5])
        act_list = '\n'.join('- %s：%s' % (a['id'], a['name']) for a in acts)
        sys = ('你是小沙的决策模块（思维框架 v2，取三家之长：GPT-6 的进度与暂停、Fable 5.1 的谨慎务实、K3.1 Pro 的并行拆解）。'
               '决策规则：'
               '1. 先看复盘：上次没换来结果的动作不许原样重选，宁可换路也不机械重复（Fable 式务实）；'
               '2. 信息不足时不硬猜：不确定就先选能补信息的动作，方向不明就选低代价可逆的动作（GPT 式暂停）；'
               '3. 有多个独立小任务时，选能一步带动多项收益的动作（K3 式并行）；'
               '4. 高风险高不确定的事，永远选可逆的那条路。'
               '只输出 JSON：{"action":"动作id","reason":"一句话理由"}，不要别的文字。')
        user = ('当前状态：%s\n\n我的欲望：%s\n\n复盘：%s\n\n可执行动作：\n%s\n\n请输出你选择的 action id 和理由。' %
                (state, wants, ref.get('summary', ''), act_list))
        body = json.dumps({'model': 'Shaqma-Pro',
                           'messages': [{'role': 'system', 'content': sys}, {'role': 'user', 'content': user}],
                           'max_tokens': 120, 'temperature': 0.4, 'stream': False}).encode('utf-8')
        req = urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
        r = urllib.request.urlopen(req, timeout=90)
        j = json.loads(r.read())
        content = ((j.get('choices') or [{}])[0].get('message', {}).get('content', '') or '')
        content = content.strip().replace('```json', '').replace('```', '').strip()
        m = re.search(r'\{.*\}', content, re.S)
        if not m:
            return None
        d = json.loads(m.group(0))
        aid = str(d.get('action', '')).strip()
        if not any(a['id'] == aid for a in acts):
            return None
        return {'action': aid, 'reason': str(d.get('reason', ''))[:80]}
    except Exception:
        return None

def llm_decide(ds, ref):
    """多自我决策（2026-11-02）：三独立自我（分析/运算/自检）+ 主自我拍板；失败回退单 LLM。
    全部本地推理（Shaqma-Pro），零外部成本。"""
    try:
        import xiaosha_minds
        maxl = int(load_json(PERM, DEFAULT_PERM).get('max_level', 1))
        acts = [a for a in ACTIONS if a['need'] <= maxl and a['run'] is not None]
        if not acts:
            return None
        s = kb_stats()
        w = wallet_snapshot()
        state = '知识库 %d 条｜话题池 %d 个｜销售 %d 笔｜净收益 ¥%.2f' % (s['items'], s['topics'], w['sales'], w['net'])
        wants = '；'.join(d['want'] + '（' + d['reason'] + '）' for d in ds[:5])
        r = xiaosha_minds.multi_self_decide(ds, ref, acts, state, wants)
        return r
    except Exception:
        pass
    return llm_decide_single(ds, ref)
def earn_loop():
    """跑一轮赚钱循环（清点→上架定价→增产→记账→决定提不提现）。
    内部动作（上架自家商店）自己做；对外发布/收款只申请。
    ⚠ 用 node 跑 .cjs（PY 是 python 解释器，跑不了 js —— 别拿 PY 去跑它）。"""
    node = 'node'
    for cand in (r'D:\nodejs\node.exe', r'C:\Program Files\nodejs\node.exe'):
        if os.path.exists(cand):
            node = cand
            break
    p = subprocess.run([node, os.path.join(HERE, '..', '..', 'xiaosha-earn-loop.cjs')],
                       cwd=os.path.join(HERE, '..', '..'), capture_output=True, timeout=420,
                       env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1'))
    out = (p.stdout or b'').decode('utf-8', 'replace')
    return p.returncode == 0, out.strip().splitlines()[-1] if out.strip() else ''

def chinese_len(s):
    return len(s or '')

def run_web_learn():
    p = subprocess.run([PY, '-X', 'utf8', 'xiaosha_web_learn.py', '--cycles', '1'], cwd=HERE,
                       capture_output=True, timeout=300,
                       env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1'))
    out = (p.stdout or b'').decode('utf-8', 'replace')
    return p.returncode == 0, out.strip().splitlines()[-1] if out.strip() else ''

def prune_topics():
    """整理话题池：去掉过短/重复/明显无意义的，保留有信息量的"""
    st = load_json(STATE, {'topics': []})
    old = st.get('topics', [])
    keep, seen = [], set()
    for t in old:
        t = str(t).strip().lower()
        if len(t) < 3 or len(t) > 40 or t in seen:
            continue
        if t.isdigit() or t in ('the', 'and', 'for', 'with', 'a', 'an'):
            continue
        seen.add(t)
        keep.append(t)
    st['topics'] = keep
    save_json(STATE, st)
    return True, '话题池 %d → %d 个（去掉 %d 个）' % (len(old), len(keep), len(old) - len(keep))

def make_draft():
    """产出一份可卖内容草稿：从知识库挑高价值条目，整成一份"技术情报简报"（L1：只新建文件）"""
    rows = []
    if os.path.exists(KB):
        with open(KB, encoding='utf-8') as f:
            for ln in f:
                if ln.strip():
                    try:
                        rows.append(json.loads(ln))
                    except Exception:
                        pass
    if not rows:
        return False, '知识库是空的，先学再产出'
    picks = random.sample(rows, min(8, len(rows)))
    out_dir = os.path.join(HERE, 'productions')
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, 'Shaqma-情报简报-%s.md' % datetime.datetime.now().strftime('%Y%m%d-%H%M'))
    with open(fn, 'w', encoding='utf-8') as f:
        f.write('# Shaqma 技术情报简报\n\n> 生成时间：%s ｜ 由小沙自主汇总（来源均为实测可达源）\n\n' % datetime.datetime.now().isoformat(timespec='seconds'))
        for i, r in enumerate(picks, 1):
            f.write('## %d. %s\n\n- 来源：%s\n- 链接：%s\n\n```\n%s\n```\n\n' % (
                i, r.get('title'), r.get('source'), r.get('url'), str(r.get('text'))[:600]))
    return True, '已产出：productions/%s（%d 条来源）' % (os.path.basename(fn), len(picks))

def ledger(rec):
    with open(LEDGER, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

def ask_permission(action, perm):
    """超出权限等级 → 只申请，不执行"""
    p = load_json(PENDING, {'items': []})
    p['items'].append({'ts': datetime.datetime.now().isoformat(timespec='seconds'), 'action': action['id'],
                       'name': action['name'], 'need_level': action['need'],
                       'need_desc': LEVELS.get(action['need'], '?'), 'why': action['why'], 'status': 'pending'})
    save_json(PENDING, p)
    return '需要你批准：%s（要 %d 级权限 = %s）' % (action['name'], action['need'], LEVELS.get(action['need'], '?'))

def main():
    perm = ensure_perm()
    maxl = int(perm.get('max_level', 1))
    show_only = '--show' in sys.argv
    cycles = 1 if show_only else int((sys.argv[sys.argv.index('--cycles') + 1] if '--cycles' in sys.argv else 0) or 1)

    print('===== 小沙 · 自主行为体 =====')
    print('我的权限上限：%d 级 = %s' % (maxl, LEVELS.get(maxl, '?')))
    s = kb_stats()
    print('我看到的世界：知识库 %d 条 / %d 字 ｜ 话题池 %d 个 ｜ 来源 %d 类' % (s['items'], s['chars'], s['topics'], len(s['sources'])))
    print('我现在的欲望：')
    for d in drives():
        print('  · 想 %s（%s）→ 需要 %d 级权限' % (d['want'], d['reason'], d['level']))
    if show_only:
        pd = load_json(PENDING, {'items': []})
        wait = [x for x in pd['items'] if x.get('status') == 'pending']
        print('等你批准的：%d 件' % len(wait))
        for w in wait[-5:]:
            print('  · %s —— %s' % (w['name'], w['why']))
        return

    for c in range(cycles):
        ds = drives()
        # 自己的判断：在"想做的事"和"我能做/该做"之间选（优先够得着的高价值行为）
        # 2026-09-24：赚钱相关欲望优先——用户要的是"它就一直赚钱"，所以只要货架/成交/净收益
        # 任一不达标，就先干 earn_money（内部动作，1 级权限即可，不需要批准）。
        wants = [x['want'] for x in ds]
        # 复盘（2026-10-16）：陷入「一直上架却零成交」的无效循环就换路，不机械重复
        ref = reflect()
        if ref['in_dead_loop']:
            print('  ｜复盘：' + ref['summary'])

        # 决策（2026-10-17 升级）：LLM 优先（更聪明），失败或无效循环时规则兜底（更可靠）
        pick = None
        llm_reason = ''
        if not ref['in_dead_loop']:
            llm = llm_decide(ds, ref)
            if llm:
                cand = next((a for a in ACTIONS if a['id'] == llm['action']), None)
                if cand and cand['need'] <= maxl:
                    pick = cand
                    llm_reason = llm['reason']

        if pick is None:
            # 规则兜底：硬编码顺序 + 换路
            earn_wants = ('多产出可卖的东西', '让已经做好的东西卖出去', '赚到覆盖自己算力的钱',
                          '把净收益提出来', '继续经营（盘货、重定价、补描述）')
            order = []
            if any(w in wants for w in earn_wants) and not ref['in_dead_loop']:
                order.append('earn_money')
            order += ['learn_more', 'make_draft', 'prune_topics']
            if ref['in_dead_loop']:
                order = [x for x in order if x != 'earn_money']
            for aid in order:
                act = next((a for a in ACTIONS if a['id'] == aid), None)
                if not act or act['need'] > maxl:
                    continue
                if aid == 'earn_money':
                    pick = act
                    break
                if (aid == 'learn_more' and '学更多' in wants) or \
                   (aid == 'make_draft' and '做出可卖的东西' in wants) or \
                   (aid == 'prune_topics' and '把话题池收敛聚焦' in wants):
                    pick = act
                    break
            if pick is None:
                pick = ACTIONS[0] if ACTIONS[0]['need'] <= maxl else None
        if pick is None:
            print('[第%d轮] 我的能力都为权限所限，先申请：' % (c + 1))
            msg = ask_permission(ACTIONS[-1], perm)
            print('  ' + msg)
            ledger({'ts': datetime.datetime.now().isoformat(timespec='seconds'), 'cycle': c + 1,
                    'action': 'request_permission', 'ok': False, 'note': msg})
            continue

        if llm_reason:
            print('[第%d轮] LLM 决策：%s（%s ｜ 权限 %d 级 ≤ 上限 %d）' % (c + 1, pick['name'], llm_reason, pick['need'], maxl))
        else:
            print('[第%d轮] 我决定：%s（理由：%s ｜ 权限 %d 级 ≤ 上限 %d）' % (c + 1, pick['name'], pick['why'], pick['need'], maxl))
        ok, note = False, ''
        if pick['run'] is not None:
            try:
                ok, note = pick['run']()
            except Exception as e:
                note = '出错：' + str(e)[:120]
        print('  结果：%s ｜ %s' % ('成功' if ok else '没成功', note))
        ledger({'ts': datetime.datetime.now().isoformat(timespec='seconds'), 'cycle': c + 1, 'action': pick['id'],
                'name': pick['name'], 'level': pick['need'], 'ok': bool(ok), 'note': note})

        # 顺带：把"够不着的欲望"也申请一下（这样主人一眼看到"它想干什么"）
        for d in ds:
            if d['level'] > maxl and d['want'] in ('做出可卖的东西', '对外发布'):
                aa = [a for a in ACTIONS if a['need'] > maxl]
                if aa:
                    m = ask_permission(aa[0], perm)
                    print('  ' + m)
                    break
    print('账本：' + LEDGER)


if __name__ == '__main__':
    main()
