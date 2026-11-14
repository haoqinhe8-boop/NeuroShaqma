# -*- coding: utf-8 -*-
"""对照实验：单独学 A / 单独学 B 各自能到多少（判断 B 任务是否可学）。

实测结论（关键对照）：
  只学 A → A=98% B=0%
  只学 B → A=0%  B=100%
  未训练 → A=28% B=0%
  顺序 A→B(λ=0) → A=32% B=18%   ← 两个都废 = 真正的灾难性遗忘
结论：两个任务单独都能学满 → 遗忘是「任务冲突」不是「能力不足」。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiaosha_cortex import SpikingCortex
from xiaosha_continual import train_task
from xiaosha_hippocampus import evaluate


def show(cx, label):
    print("  %-22s → A=%.0f%%  B=%.0f%%"
          % (label, evaluate(cx, 0, n=40) * 100, evaluate(cx, 1, n=40) * 100))


def main():
    print("=== 对照：fresh 网络单独训练 ===")
    for tid, label in [(0, "只学 A (动作=cls)"), (1, "只学 B (动作=(cls+2)%4)")]:
        cx = SpikingCortex(n=6000, seed=11)
        train_task(cx, tid, rounds=120, seed=9000)
        show(cx, label)

    print("\n=== 对照：随机基线（不训练）===")
    show(SpikingCortex(n=6000, seed=11), "未训练")

    print("\n=== 顺序学：先 A 后 B（不用锚定，λ=0）===")
    cx = SpikingCortex(n=6000, seed=11)
    train_task(cx, 0, rounds=120, seed=9000)
    show(cx, "学完 A")
    train_task(cx, 1, rounds=120, seed=9500)
    show(cx, "再学 B 后")


if __name__ == "__main__":
    main()
