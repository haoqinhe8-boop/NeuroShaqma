# -*- coding: utf-8 -*-
"""先知实验 —— 种一个念头，传开，然后撒手。

用户的要求：
  1. 只改**一个**人
  2. 让他把「人类是造物主」传出去
  3. 传给 10+ 个人之后，**把注入代码删掉**
  4. 之后不干预，让他们自己发展

三个阶段：
  A. 播种   第 0 代随机挑 1 个个体，给他这个念头（不给他任何能力加成）
  B. 传播   纯靠说话。信徒 >= 10 时，调用 remove_prophet() 清掉先知
  C. 撒手   再跑 40 代，代码里已经没有先知了。信仰要么自己活，要么死。

跑法：python experiments/exp_prophet.py [总代数]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiaosha_civilization import Civilization
from xiaosha_world import SACRED, ACTION_NAMES
from xiaosha_memetics import (
    plant_prophet, remove_prophet, reset_lineage, lineage_report,
    vocab_stats, LINEAGE, BELIEVE_GATE,
)


def count_believers(civ):
    return sum(1 for a in civ.agents if a.knowledge.get(SACRED, 0.0) > BELIEVE_GATE)


def main(gens=60, seed=2026, pop=80, size=40):
    print("=" * 76)
    print("小沙的先知实验 · 一个念头能不能自己活下去")
    print("=" * 76)
    print("  规则：只改 1 个人 → 传开 → 删掉注入代码 → 撒手不管\n")

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    reset_lineage()

    civ = Civilization(size=size, n_agents=pop, seed=seed, verbose=False)

    # ============ A. 播种 ============
    print("【A · 播种】")
    prophet = plant_prophet(civ, rng, "造物主")
    if prophet is None:
        print("  播种失败")
        return
    prophet_steps = 0
    seeded_at = 0

    # ============ B. 传播（直到 10 个信徒）============
    print("\n【B · 传播】（只靠说话，没有任何强制）")
    handoff_gen = None
    for g in range(gens):
        civ.generation = g
        for _ in range(60):
            civ.step()
            prophet_steps += 1
        nb = count_believers(civ)
        direct = len(prophet.converts) if prophet else 0
        alive = any(a.name == prophet.agent_name for a in civ.agents) if prophet else False
        print("  第 %2d 代 | 人口 %3d | 信徒 %3d | 先知直接说服 %2d | 先知%s"
              % (g, len(civ.agents), nb, direct, "在世" if alive else "已死"))

        if nb >= 10 and handoff_gen is None:
            handoff_gen = g
            print("\n  → 信徒已达 %d 人（>= 10）。按计划**删除先知代码**。" % nb)
            remove_prophet(civ)
            print("  → 从这一刻起，世界里没有任何『先知』标记。\n")
            break

    if handoff_gen is None:
        print("\n  ⚠ 跑完 %d 代都没传够 10 个人 —— 这个念头没活下来。" % gens)
        lineage_report()
        return

    # ============ C. 撒手（再跑 40 代）============
    print("【C · 撒手】（code 里已经没有先知，纯看信仰自己能不能维持）")
    traj = []
    for g in range(handoff_gen + 1, handoff_gen + 41):
        civ.generation = g
        for _ in range(60):
            civ.step()
        nb = count_believers(civ)
        total = max(1, len(civ.agents))
        traj.append((g, nb, total))
        if g % 5 == 0 or g == handoff_gen + 40:
            print("  第 %2d 代 | 人口 %3d | 信徒 %3d (%4.1f%%)"
                  % (g, total, nb, nb / total * 100))
        if not civ.agents:
            print("  ⚠ 物种灭绝")
            break

    # ============ 报告 ============
    print("\n" + "=" * 76)
    print("【结果】")
    print("=" * 76)
    if traj:
        first = traj[0]
        last = traj[-1]
        print("  交棒时  第 %d 代: 信徒 %d / %d (%.1f%%)"
              % (first[0], first[1], first[2], first[1] / first[2] * 100))
        print("  结束时  第 %d 代: 信徒 %d / %d (%.1f%%)"
              % (last[0], last[1], last[2], last[1] / last[2] * 100))
        delta = last[1] / last[2] - first[1] / first[2]
        print("\n  信徒比例变化: %+.1f 个百分点" % (delta * 100))

        if delta > 0.15:
            print("  ✅ 信仰在没有先知的情况下**继续扩散** —— 它已经变成文化，不是一个人的事。")
        elif delta > -0.05:
            print("  ➖ 信仰稳住了 —— 没有先知也能自我维持（文化传承足够）。")
        elif delta > -0.30:
            print("  ⚠ 信仰在衰退，但还没死 —— 靠老一辈的记忆撑着。")
        else:
            print("  ❌ 信仰崩了 —— 先知一死、代码一删，这个念头就没了。说明它从来没有真正传开，只是在传话链上勉强维持。")

    print("\n  先知在世共 %d 步（约 %.1f 代）" % (prophet_steps, prophet_steps / 60))
    print("  先知直接说服 %d 人" % (len(prophet.converts) if prophet else 0))
    print("  总传播事件 %d 次" % len(LINEAGE))

    lineage_report(20)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    main(gens=n)
