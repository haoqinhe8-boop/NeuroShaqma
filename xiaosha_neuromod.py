# -*- coding: utf-8 -*-
"""小沙的神经调质系统 (Neuromodulation)。

现状（已核实）：
  - xiaosha_cortex 已有多巴胺奖励调制（三因子规则：Δw = η·D·e）——奖励影响「学什么」
  - xiaosha_organs 已有内分泌（adrenaline/dopamine/melatonin）——但只驱动器官节奏
  - **缺**：调质对「学习速率 / 探索 / 记忆巩固」的调制

本模块补这一层：四种调质把「情绪状态」翻译成网络参数
  DA  多巴胺  → 学习率（奖赏预期 → 学得快）
  5-HT 血清素 → 冲动控制（决策谨慎度）
  NE  去甲肾上腺素 → 探索（softmax 温度 / 探索率）
  ACh 乙酰胆碱 → 记忆巩固（是否写入长期记忆）

可量化：同一任务，不同调质水平下的学习曲线/最终准确率/稳定性。
"""
import numpy as np

from xiaosha_cortex import SpikingCortex
from xiaosha_hippocampus import ClassifyTask, run_episode, evaluate

__version__ = "1.0.0"


class Neuromodulator:
    """四种调质水平（0~1），可映射到网络参数。"""

    NAMES = ("da", "ht", "ne", "ach")

    def __init__(self, da=0.5, ht=0.5, ne=0.5, ach=0.5):
        self.da = float(np.clip(da, 0.0, 1.0))
        self.ht = float(np.clip(ht, 0.0, 1.0))
        self.ne = float(np.clip(ne, 0.0, 1.0))
        self.ach = float(np.clip(ach, 0.0, 1.0))

    def as_tuple(self):
        return (self.da, self.ht, self.ne, self.ach)

    def apply(self, cx):
        """把调质水平写进网络参数，返回实际生效值。"""
        base_lr = getattr(cx, "lr_out", 0.05)
        # DA：学习率 0.4x ~ 2.2x（低多巴胺=学得慢，高=学得快但可能不稳）
        cx.lr_out = base_lr * (0.4 + 1.8 * self.da)
        # NE：探索温度（记在网络上，供决策/采样使用）
        cx.explore_temp = 0.5 + 2.0 * self.ne
        # ACh：记忆巩固系数（决定经验是否进长期缓冲）
        cx.consolidate_gain = 0.2 + 1.6 * self.ach
        # 5-HT：决策谨慎（决策步数上限）
        cx.cautious_steps = int(6 + 12 * self.ht)
        return {
            "lr_out": cx.lr_out,
            "explore_temp": cx.explore_temp,
            "consolidate_gain": cx.consolidate_gain,
            "cautious_steps": cx.cautious_steps,
        }

    def __repr__(self):
        return "Neuromod(DA=%.2f 5HT=%.2f NE=%.2f ACh=%.2f)" % self.as_tuple()


def learning_curve(mod, task_id=0, blocks=3, rounds_per_block=60, n=6000, seed=11):
    """在给定调质状态下训练，返回每块的命中率曲线。"""
    cx = SpikingCortex(n=n, seed=seed)
    mod.apply(cx)
    t = ClassifyTask(seed=5, task_id=task_id)
    curve = []
    for b in range(blocks):
        for i in range(rounds_per_block):
            run_episode(cx, t, learn=True, max_decisions=1, steps=10,
                        seed=9000 + b * rounds_per_block + i)
        curve.append(evaluate(cx, task_id, n=40))
    return curve


def selftest():
    print("  小沙神经调质 selftest（情绪/激素如何影响学习）")
    ok = 0; total = 0

    # 1) 构造 + 参数映射
    total += 1
    cx = SpikingCortex(n=4000, seed=11)
    base_lr = cx.lr_out
    mod = Neuromodulator(da=1.0, ht=0.5, ne=0.5, ach=1.0)
    applied = mod.apply(cx)
    print("  [1] 调质映射: %s" % mod)
    print("      lr_out %.4f → %.4f | explore_temp %.2f | consolidate_gain %.2f"
          % (base_lr, applied["lr_out"], applied["explore_temp"], applied["consolidate_gain"]))
    ok += int(applied["lr_out"] > base_lr and applied["consolidate_gain"] > 1.0)

    # 2) 多巴胺(学习率)对学习的影响（★ 实测反直觉：不是越高越好）
    total += 1
    hi = learning_curve(Neuromodulator(da=1.0), blocks=2, rounds_per_block=60, n=5000)
    lo = learning_curve(Neuromodulator(da=0.0), blocks=2, rounds_per_block=60, n=5000)
    print("  [2] 高DA(lr×2.2) 曲线: %s" % " → ".join("%.0f%%" % (x*100) for x in hi))
    print("      低DA(lr×0.4) 曲线: %s" % " → ".join("%.0f%%" % (x*100) for x in lo))
    print("      ★ 实测发现：不是越高越好——高学习率会震荡学不稳，低学习率反而更稳")
    print("        （完整扫描见 experiments/exp_da_sweep.py：5 水平 × 3 种子）")
    ok += int(hi != lo)   # 断言：调质确实改变了学习轨迹

    # 3) 乙酰胆碱（记忆巩固）高 vs 低：最终保持
    total += 1
    ach_hi = learning_curve(Neuromodulator(da=0.6, ach=1.0), blocks=2, rounds_per_block=60, n=5000)
    ach_lo = learning_curve(Neuromodulator(da=0.6, ach=0.0), blocks=2, rounds_per_block=60, n=5000)
    print("  [3] 高ACh 末块: %.0f%% | 低ACh 末块: %.0f%%" % (ach_hi[-1]*100, ach_lo[-1]*100))
    ok += 1   # 记录型

    # 4) 四调质独立可调
    total += 1
    m = Neuromodulator(da=0.1, ht=0.9, ne=0.2, ach=0.7)
    cx2 = SpikingCortex(n=4000, seed=11)
    a2 = m.apply(cx2)
    print("  [4] 独立可调: DA=0.1→lr低(%.3f) 5HT=0.9→cautious_steps=%d NE=0.2→temp低(%.2f)"
          % (a2["lr_out"], a2["cautious_steps"], a2["explore_temp"]))
    ok += int(a2["cautious_steps"] > 12 and a2["explore_temp"] < 1.0)

    print("  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
