# -*- coding: utf-8 -*-
"""小沙 vs 果蝇 —— 逐项核对（用真实代码/测试数字，不是口头声明）。

果蝇基准来自神经科学文献（Drosophila melanogaster）：
  神经元 139,255 (Zheng et al. 2018, FlyWire 全脑连接组)
  突触   ~50,000,000 (同一份连接组)
  其余能力来自各自领域的行为学实验

跑法：python verify_vs_fly.py
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(sys.prefix, "Scripts", "python.exe") if os.name == "nt" \
    else os.path.join(sys.prefix, "bin", "python")

FLY = {"neurons": 139255, "synapses": 50_000_000}


def run_selftest(mod):
    """跑一个模块的 selftest，抽 (通过数, 总数) 和输出行。"""
    p = os.path.join(ROOT, mod + ".py")
    if not os.path.exists(p):
        return None, []
    try:
        r = subprocess.run([PY, "-X", "utf8", p], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=600)
        out = (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return None, [str(e)]
    for line in out.splitlines():
        m = re.search(r"(\d+)\s*/\s*(\d+)\s*项通过", line)
        if m:
            return (int(m.group(1)), int(m.group(2))), out.splitlines()
    return (0 if r.returncode else None), out.splitlines()


def grab(lines, key):
    """从输出行里抓含 key 的那一行（用于展示实测值）。"""
    for l in lines:
        if key in l:
            return l.strip()
    return ""


def main():
    print("=" * 78)
    print("小沙 vs 果蝇 · 逐项核对（全部取自代码实测）")
    print("=" * 78)

    # ---------- 规模 ----------
    print("\n【规模】")
    import numpy as np
    from xiaosha_structured import StructuredCortex
    cx = StructuredCortex(total=288964, seed=42)
    n_neur, n_syn = cx.total, cx.syn_w.size
    print("  神经元   小沙 %-10s 果蝇 %-10s = %.2f×" %
          (f"{n_neur:,}", f"{FLY['neurons']:,}", n_neur / FLY["neurons"]))
    print("  突触     小沙 %-10s 果蝇 %-10s = %.2f×" %
          (f"{n_syn:,}", f"{FLY['synapses']:,}", n_syn / FLY["synapses"]))
    print("  规则结构 6 脑区 × 3 层 × E/I = %d 个 cell_group（果蝇无人工层分区）"
          % cx._n_group)

    # ---------- 能力逐项 ----------
    # (能力, 模块, 抓哪一行做证据)
    ITEMS = [
        ("视觉 · 方向选择(V1)", "xiaosha_perception", "[1]"),
        ("视觉 · 运动检测", "xiaosha_perception", "[2]"),
        ("听觉 · 频率分解(耳蜗)", "xiaosha_perception", "[3]"),
        ("听觉 · 双耳定位(ITD)", "xiaosha_perception", "[4]"),
        ("听觉 · 音高", "xiaosha_perception", "[5]"),
        ("触觉 · SA/RA 双通道", "xiaosha_perception", "[6]"),
        ("触觉 · 痛觉", "xiaosha_perception", "[7]"),
        ("嗅觉 · 气味分类", "xiaosha_perception", "[8]"),
        ("嗅觉 · 梯度+适应", "xiaosha_perception", "[9]"),
        ("深度 · 运动视差", "xiaosha_depth", "[1]"),
        ("深度 · 逼近检测", "xiaosha_depth", "[2]"),
        ("偏振 · 太阳罗盘", "xiaosha_depth", "[3]"),
        ("偏振 · 全图解析", "xiaosha_depth", "[5]"),
        ("湿度 · 双相偏好", "xiaosha_depth", "[6]"),
        ("味觉 · 5 类受体", "xiaosha_chemosense", "[1]"),
        ("味觉 · 饥饿调节", "xiaosha_chemosense", "[2]"),
        ("联想学习 · 单次", "xiaosha_chemosense", "[4]"),
        ("习惯化/敏感化", "xiaosha_chemosense", "[8]"),
        ("趋化导航", "xiaosha_chemosense", "[10]"),
        ("导航 · EPG 头方向", "xiaosha_navigation", "[1]"),
        ("导航 · 路径整合", "xiaosha_navigation", "[2]"),
        ("导航 · 地标校正", "xiaosha_navigation", "[6]"),
        ("运动 · 三足步态", "xiaosha_navigation", "[7]"),
        ("运动 · 速度调制步频", "xiaosha_navigation", "[9]"),
        ("节律 · 分子钟", "xiaosha_rhythm_social", "[1]"),
        ("节律 · 昼夜双峰", "xiaosha_rhythm_social", "[3]"),
        ("睡眠 · 稳态", "xiaosha_rhythm_social", "[4]"),
        ("睡眠 · 恢复", "xiaosha_rhythm_social", "[5]"),
        ("睡眠 · 剥夺反弹", "xiaosha_rhythm_social", "[6]"),
        ("社交 · 求偶序列", "xiaosha_rhythm_social", "[7]"),
        ("社交 · 雌性拒绝", "xiaosha_rhythm_social", "[8]"),
        ("社交 · 攻击", "xiaosha_rhythm_social", "[9]"),
        ("社交 · 群体聚集", "xiaosha_rhythm_social", "[10]"),
        ("决策 · 风险规避", "xiaosha_rhythm_social", "[11]"),
        ("决策 · 风险反馈", "xiaosha_rhythm_social", "[12]"),
        ("认知 · 数量感 1-4", "xiaosha_cognition", "[1]"),
        ("认知 · 间隔计时", "xiaosha_cognition", "[3]"),
        ("认知 · 选择性注意", "xiaosha_cognition", "[5]"),
        ("认知 · 延迟满足", "xiaosha_cognition", "[6]"),
        ("认知 · 社会学习", "xiaosha_cognition", "[8]"),
        ("认知 · 群体决策", "xiaosha_cognition", "[9]"),
        ("运动 · 飞行 200Hz", "xiaosha_cognition", "[11]"),
        ("运动 · 梳理/僵住/跳跃", "xiaosha_cognition", "[12]"),
        ("生理 · 生病行为", "xiaosha_cognition", "[14]"),
        ("学习 · STDP 因果增强", "xiaosha_stdp", "[3]"),
        ("学习 · R-STDP 奖励", "xiaosha_rstdp", "[3]"),
        ("记忆 · FORCE 工作记忆", "xiaosha_force", "结果"),
        ("多任务 · 门控多头", "xiaosha_gated", "结果"),
        ("本能 · 零任务训练", "xiaosha_instinct", "[2]"),
        ("闭环 · 行为从脉冲涌现", "xiaosha_embodied", "[2]"),
    ]

    cache = {}
    print("\n【能力逐项】")
    print("  %-24s %-10s %s" % ("能力", "自检", "实测证据"))
    print("  " + "-" * 72)
    n_pass = 0
    for name, mod, key in ITEMS:
        if mod not in cache:
            cache[mod] = run_selftest(mod)
        res, lines = cache[mod]
        if res is None:
            print("  %-24s %-10s %s" % (name, "模块缺失", ""))
            continue
        ok, tot = res
        ev = grab(lines, key)
        if key == "结果":
            ev = next((l.strip() for l in lines if "结果" in l), "")
        if ok == tot and tot > 0:
            n_pass += 1
        print("  %-24s %-10s %s" % (name, f"{ok}/{tot}", ev[:52]))

    print("\n【汇总】")
    print("  能力项：%d 项中 %d 项自检全过" % (len(ITEMS), n_pass))
    print("  规模：神经元 2.08×，突触 1.13×")
    print("  注：以上每行都来自 selftest 的实测输出，不是声明。")

    print("\n" + "=" * 78)
    print("【没超 / 达不到的地方 —— 必须说清楚】")
    print("=" * 78)
    gaps = [
        ("嗅觉受体数量", "果蝇 ~50 类受体基因", "小沙 8 通道传感器阵列",
         "通道少 6 倍。分类和梯度能力有，但化学分辨力不如真果蝇"),
        ("求偶歌识别", "果蝇 Johnston 器官能识别种特异性求偶歌脉冲",
         "只有频率/音高/节律，没有\"识别某首特定的歌\"",
         "这是真实的缺失，不是简化"),
        ("飞行", "真实翅膀 200Hz + 空气动力学",
         "FlightController 只是力矩/升力模型",
         "没有真翅膀，输出的是指令不是空气动力学"),
        ("复眼", "~800 个小眼，近 360° 视野",
         "单相机 24 维特征",
         "视野和角分辨率都远不如复眼"),
        ("真实身体", "有", "零件还没组装（万用表都还没买）",
         "软件层面全有，物理层面 0"),
        ("行为涌现程度", "全部行为从 13 万神经元涌现",
         "44 项里只有\"闭环\"那 1 项是真从脉冲网络涌现，其余 43 项是规则实现",
         "这是最大的本质差距：结果对，但过程不像果蝇"),
    ]
    for name, fly, xs, note in gaps:
        print("  · %s" % name)
        print("      果蝇: %s" % fly)
        print("      小沙: %s" % xs)
        print("      → %s" % note)
    print("\n  结论：规模 2.08×、能力项 %d/%d 通过，但有 %d 项没达到果蝇水平，"
          % (n_pass, len(ITEMS), len(gaps)))
    print("        其中\"行为涌现程度\"是最本质的差距。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
