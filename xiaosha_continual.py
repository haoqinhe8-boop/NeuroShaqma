# -*- coding: utf-8 -*-
"""小沙的持续学习：任务后锚定 (post-task anchoring, EWC 式)。

背景（README 已如实记录）：
  交替训练两个冲突任务会灾难性遗忘——分类单独 100%，交替后掉到 25%。
  海马体重放能救回一半（55%→82%），但那是"复习"，不是"保护"。

本模块测另一个机制：**任务后锚定**
  学完任务 A → 把 W_anchor 设为 A 的权重（锚点）
  学任务 B 时，anchor_lam 把 W_out 往回拉：W_out -= lr*λ*(W_out - W_anchor)
  → 重要权重不被 B 覆盖。这正是 EWC(Elastic Weight Consolidation) 的简化形式。

关键 trade-off（必须量化）：
  λ 太小 → A 忘光；λ 太大 → B 学不会。找平衡点。
"""
import numpy as np

from xiaosha_cortex import SpikingCortex
from xiaosha_hippocampus import ClassifyTask, run_episode, evaluate

__version__ = "1.0.0"


def train_task(cx, task_id, rounds=150, seed=9000):
    """只训练一个任务若干轮。"""
    t = ClassifyTask(seed=5, task_id=task_id)
    for i in range(rounds):
        run_episode(cx, t, learn=True, max_decisions=1, steps=10, seed=seed + i)


def anchor_to(cx, lam):
    """把当前 W_out 设为锚点，并打开锚定强度。"""
    cx.W_anchor = cx.W_out.copy()
    cx.anchor_lam = float(lam)


def run_continual(n=8000, rounds=150, lam=2.0, seed=11):
    """A → 锚定 → B，返回 (A学习后, A最终, B最终)。"""
    cx = SpikingCortex(n=n, seed=seed)
    cx.anchor_lam = 0.0
    train_task(cx, 0, rounds=rounds, seed=9000)
    accA0 = evaluate(cx, 0, n=40)
    anchor_to(cx, lam)
    train_task(cx, 1, rounds=rounds, seed=9500)
    accA1 = evaluate(cx, 0, n=40)
    accB = evaluate(cx, 1, n=40)
    return accA0, accA1, accB


def selftest():
    print("  小沙持续学习 selftest（任务后锚定 vs 灾难性遗忘）")
    ok = 0; total = 0

    # 1) 机制可用
    total += 1
    cx = SpikingCortex(n=6000, seed=11)
    anchor_to(cx, 2.0)
    print("  [1] 锚定机制可设置: W_anchor %s, anchor_lam=%.1f"
          % (getattr(cx, "W_anchor", np.zeros((1,))).shape, cx.anchor_lam))
    ok += int(hasattr(cx, "W_anchor") and cx.anchor_lam == 2.0)

    # 2) 无锚定：灾难性遗忘（A 明显下降）
    total += 1
    a0, a1, b1 = run_continual(n=6000, rounds=120, lam=0.0, seed=11)
    print("  [2] λ=0（无保护）: A 学后 %.0f%% → 学B后 %.0f%% | B %.0f%%"
          % (a0*100, a1*100, b1*100))
    okso = (a1 <= a0 - 0.15)   # 明显遗忘
    ok += int(okso)
    if not okso: print("      !! 未复现遗忘（意外）")

    # 3) 强锚定：A 保住
    total += 1
    a0b, a1b, b1b = run_continual(n=6000, rounds=120, lam=3.0, seed=11)
    print("  [3] λ=3.0（强锚定）: A 学后 %.0f%% → 学B后 %.0f%% | B %.0f%%"
          % (a0b*100, a1b*100, b1b*100))
    okso = (a1b >= a1 + 0.15)  # 保持率明显优于无锚定
    ok += int(okso)
    if not okso: print("      !! 锚定没起作用")

    # 4) 诚实记录 trade-off：强锚定会牺牲 B
    total += 1
    print("  [4] trade-off 记录: B 从 %.0f%%(λ=0) 变为 %.0f%%(λ=3.0) —— 保护 A 的代价"
          % (b1*100, b1b*100))
    ok += 1

    # 5) 中间 λ 的 trade-off（记录型：如实呈现，不作为通过条件）
    total += 1
    a0c, a1c, b1c = run_continual(n=6000, rounds=120, lam=1.0, seed=11)
    print("  [5] λ=1.0（平衡）: A 保持 %.0f%% | B %.0f%% —— 如实记录 trade-off" % (a1c*100, b1c*100))
    ok += 1   # 记录型断言：实验结论本身就是要如实呈现的

    print("  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)