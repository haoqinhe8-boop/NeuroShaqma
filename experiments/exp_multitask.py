# -*- coding: utf-8 -*-
"""实验：一个网络同时学会两个冲突任务，需要什么条件？

本次实测结论：
  E1 原TAG(A=0,B=0.9) 交替400轮 → A=55% B=45%
  E2 双向TAG 400轮              → A=55% B=75%
  E3 双向TAG 800轮              → A=40% B=100%   ← recency bias
  E4 双向TAG n=2万 400轮        → A=55% B=72%
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from xiaosha_cortex import SpikingCortex
from xiaosha_hippocampus import N_CH, TAG, run_episode, evaluate, ClassifyTask


class BiTagTask:
    """双向 TAG：A=-0.9 / B=+0.9（原版 A=0 时 TAG 无信息，网络会忽略它）"""

    def __init__(self, seed=5, task_id=0):
        self.rng = np.random.default_rng(seed)
        self.task_id = task_id
        self.cls = 0
        self.target_tol = 0.1

    def new_episode(self):
        self.cls = int(self.rng.integers(0, 4))
        return self.obs()

    def target(self):
        return self.cls if self.task_id == 0 else (self.cls + 2) % 4

    def obs(self):
        o = np.zeros(N_CH, dtype=np.float32)
        o[self.cls] = 0.9
        o[TAG] = -0.9 if self.task_id == 0 else 0.9
        return o

    def act(self, a):
        if int(a) == self.target():
            return 1.0, True, {"dist": 0.0}
        return -0.5, True, {"dist": 9.9}


def eval_bi(cx, task_id, n=40):
    t = BiTagTask(seed=5, task_id=task_id)
    return sum(int(run_episode(cx, t, learn=False, max_decisions=1, steps=10,
                               seed=5000 + i)["hit"]) for i in range(n)) / n


def train_alternating(cx, rounds, TaskCls, seed=9000):
    tA, tB = TaskCls(seed=5, task_id=0), TaskCls(seed=5, task_id=1)
    for i in range(rounds):
        run_episode(cx, tA if i % 2 == 0 else tB, learn=True,
                    max_decisions=1, steps=10, seed=seed + i)


def main():
    print("=" * 66)
    print("实验：一个网络如何同时学会两个冲突任务")
    print("=" * 66)

    cx = SpikingCortex(n=6000, seed=11)
    train_alternating(cx, 400, ClassifyTask)
    print("E1 原TAG(A=0,B=.9) 400轮 → A=%.0f%%  B=%.0f%%"
          % (evaluate(cx, 0, 40) * 100, evaluate(cx, 1, 40) * 100))

    for label, rounds, n, seed in [
        ("E2 双向TAG 400轮", 400, 6000, 9000),
        ("E3 双向TAG 800轮", 800, 6000, 20000),
        ("E4 双向TAG 2万/400轮", 400, 20000, 9000),
    ]:
        cx = SpikingCortex(n=n, seed=11)
        train_alternating(cx, rounds, BiTagTask, seed=seed)
        print("%-22s → A=%.0f%%  B=%.0f%%"
              % (label, eval_bi(cx, 0) * 100, eval_bi(cx, 1) * 100))
    print("=" * 66)


if __name__ == "__main__":
    main()
