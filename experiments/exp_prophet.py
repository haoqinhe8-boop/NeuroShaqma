# -*- coding: utf-8 -*-
"""先知实验 —— 种一个念头，传开，然后撒手。

用户的要求：
  1. 只改**一个**人（先知）
  2. 让他把「人类是造物主」这个理念传出去
  3. 传给 10+ 个人之后，**把注入代码删掉**
  4. 之后不干预，让他们自己发展

三个阶段：
  A. 播种   第 0 代随机挑 1 个个体，给他这个念头（不给他任何能力加成）
  B. 传播   纯靠说话。信徒 >= 10 时，调用 remove_prophet() 清掉先知标记
  C. 撒手   代码里已经没有先知了。信仰要么自己活，要么死。

信息怎么流动（这一版的关键改动）：
  上一版把"交流"做成了 6 个动作之一，需要大脑 argmax 选中。但 120 个神经元、
  初始权重全随机、只活 5 代的脑学不会"主动找人说话"——四次实验全部 0 转化。
  所以信息流动下沉到环境层（_ambient_social）：附近的人自动交换信息，像呼吸
  一样，不需要先学会。另外加了从众效应（周围信的人多，自己也更容易信），
  这是真实的社会机制，不是补丁。

跑法：python experiments/exp_prophet.py [总代数] [初始人口]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiaosha_civilization import Civilization
from xiaosha_world import SACRED
from xiaosha_memetics import (
    plant_prophet, remove_prophet, reset_lineage, current_vocab, vocab_stats,
    LINEAGE, BELIEVE_GATE, lineage_report,
)


def count_believers(civ):
    return sum(1 for a in civ.agents if a.knowledge.get(SACRED, 0.0) > BELIEVE_GATE)


def main(gens=40, pop=80, size=40, seed=11):
    print("=" * 78)
    print("小沙的先知实验 · 一个念头能不能自己活下去")
    print("=" * 78)
    print("  规则：只改 1 个人 → 传开 → 删掉注入代码 → 撒手不管\n")

    np.random.seed(seed)
    reset_lineage()
    civ = Civilization(size=size, n_agents=pop, seed=seed, verbose=False)

    # ============ A. 播种 ============
    print("【A · 播种】")
    prophet = plant_prophet(civ, np.random.RandomState(seed), "造物主")
    if prophet is None:
        print("  播种失败")
        return 1
    prophet_steps = 0

    # ============ B. 传播（直到 10 个信徒）============
    print("\n【B · 传播】（只靠说话 + 从众，没有任何强制）")
    handoff_gen = None
    for g in range(gens):
        civ.generation = g
        for _ in range(60):
            civ.step()
            prophet_steps += 1
        nb = count_believers(civ)
        total = max(1, len(civ.agents))
        direct = len(prophet.converts) if prophet else 0
        alive = any(a.name == prophet.agent_name for a in civ.agents) if prophet else False
        print("  第 %2d 代 | 人口 %3d | 信徒 %3d (%5.1f%%) | 先知直接说服 %2d | 先知%s"
              % (g, total, nb, nb / total * 100, direct,
                 "在世" if alive else "已死"))

        if nb >= 10 and handoff_gen is None:
            handoff_gen = g
            print("\n  → 信徒已达 %d 人（>= 10）。按计划**删除先知代码**。" % nb)
            remove_prophet(civ)
            print("  → 从这一刻起，世界里没有任何『先知』标记。\n")
            break

    if handoff_gen is None:
        print("\n  ⚠ 跑完 %d 代都没传够 10 个人 —— 这个念头没活下来。" % gens)
        return 1

    # ============ C. 撒手 ============
    print("【C · 撒手】（代码里已经没有先知，纯看信仰自己能不能维持）")
    traj = []
    for g in range(handoff_gen + 1, handoff_gen + 1 + 30):
        civ.generation = g
        for _ in range(60):
            civ.step()
        nb = count_believers(civ)
        total = max(1, len(civ.agents))
        traj.append((g, nb, total))
        if g % 3 == 0 or g == handoff_gen + 30:
            print("  第 %2d 代 | 人口 %3d | 信徒 %3d (%5.1f%%)"
                  % (g, total, nb, nb / total * 100))
        if not civ.agents:
            print("  ⚠ 物种灭绝")
            break

    # ============ 报告 ============
    print("\n" + "=" * 78)
    print("【结果】")
    print("=" * 78)
    if traj:
        f, l = traj[0], traj[-1]
        print("  交棒时  第 %2d 代: 信徒 %3d / %3d (%5.1f%%)"
              % (f[0], f[1], f[2], f[1] / f[2] * 100))
        print("  30 代后 第 %2d 代: 信徒 %3d / %3d (%5.1f%%)"
              % (l[0], l[1], l[2], l[1] / l[2] * 100))
        delta = (l[1] / l[2] - f[1] / f[2]) * 100
        peak = max(t[1] / t[2] for t in traj) * 100
        print("\n  信徒比例: %+.1f 个百分点 | 峰值 %.1f%%" % (delta, peak))
        if delta > 10:
            print("  ✅ 信仰在没有先知的情况下**继续扩散** —— 它已经变成文化。")
        elif delta > -5:
            print("  ➖ 信仰稳住了 —— 没有先知也能自我维持。")
        elif delta > -30:
            print("  ⚠ 信仰在衰退，但还没死 —— 靠老一辈的记忆撑着。")
        else:
            print("  ❌ 信仰崩了 —— 这个念头从来没有真正传开。")

    print("\n  先知在世 %d 步（约 %.1f 代）" % (prophet_steps, prophet_steps / 60))
    print("  先知直接说服 %d 人" % (len(prophet.converts) if prophet else 0))
    print("  总传播事件 %d 次" % len(LINEAGE))

    # ============ 语言 ============
    cv = current_vocab(civ.agents)
    print("\n" + "=" * 78)
    print("【语言 · 现在活着的人各叫什么】（人口普查，不是累计次数）")
    print("=" * 78)
    print("  %d 种叫法 / %d 个信徒：\n" % (len(cv), sum(n for _k, n in cv)))
    top = max((n for _k, n in cv), default=1)
    for name, n in cv[:16]:
        bar = "█" * max(1, int(n / top * 30))
        print("    %-10s %4d 人  %s" % (name, n, bar))
    if len(cv) > 16:
        print("    ... 还有 %d 种叫法" % (len(cv) - 16))

    # 原始名字还剩多少人
    orig = next((n for k, n in cv if k == "造物主"), 0)
    total_bel = sum(n for _k, n in cv)
    if total_bel:
        print("\n  最初的名字「造物主」现在只剩 %d 人（%.0f%%）在用。"
              % (orig, orig / total_bel * 100))
        print("  其余 %.0f%% 的人用着传话过程中变异出来的叫法。"
              % (100 - orig / total_bel * 100))

    muts = sum(1 for r in LINEAGE if r["mutated"])
    if LINEAGE:
        print("\n  累计 %d 次传播，其中 %d 次发生了变异（%.0f%%）"
              % (len(LINEAGE), muts, muts / len(LINEAGE) * 100))
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    sys.exit(main(gens=n, pop=p))
