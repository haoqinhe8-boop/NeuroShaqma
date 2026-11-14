# -*- coding: utf-8 -*-
"""实验 E2-E4：双向 TAG 能否让一个网络同时学会两个冲突任务。

已知对照：
  E1 原TAG(A=0,B=0.9) 交替400轮 → A=55% B=45%
  只学A → A98%/B0%；只学B → A0%/B100%
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from xiaosha_cortex import SpikingCortex
from xiaosha_hippocampus import N_CH, TAG, run_episode


class BiTagTask:
    """双向 TAG：A=-0.9, B=+0.9（原版 A=0 时 TAG 无信息，网络会忽略）"""

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


def eval_task(cx, task_id, n=40):
    t = BiTagTask(seed=5, task_id=task_id)
    hits = 0
    for i in range(n):
        r = run_episode(cx, t, learn=False, max_decisions=1, steps=10, seed=5000 + i)
        hits += int(r["hit"])
    return hits / n


def train_alt(cx, rounds, seed=9000):
    tA, tB = BiTagTask(seed=5, task_id=0), BiTagTask(seed=5, task_id=1)
    for i in range(rounds):
        t = tA if i % 2 == 0 else tB
        run_episode(cx, t, learn=True, max_decisions=1, steps=10, seed=seed + i)


if __name__ == "__main__":
    print("=" * 62, flush=True)
    print("E2/E3/E4：双向 TAG（A=-0.9, B=+0.9）", flush=True)
    print("=" * 62, flush=True)
    print("对照 E1 原TAG 400轮 → A=55% B=45%", flush=True)

    cx2 = SpikingCortex(n=6000, seed=11)
    train_alt(cx2, 400)
    print("E2 双向TAG 交替400轮    → A=%.0f%%  B=%.0f%%"
          % (eval_task(cx2, 0) * 100, eval_task(cx2, 1) * 100), flush=True)

    cx3 = SpikingCortex(n=6000, seed=11)
    train_alt(cx3, 800, seed=20000)
    print("E3 双向TAG 交替800轮    → A=%.0f%%  B=%.0f%%"
          % (eval_task(cx3, 0) * 100, eval_task(cx3, 1) * 100), flush=True)

    cx4 = SpikingCortex(n=20000, seed=11)
    train_alt(cx4, 400)
    print("E4 双向TAG n=2万 400轮  → A=%.0f%%  B=%.0f%%"
          % (eval_task(cx4, 0) * 100, eval_task(cx4, 1) * 100), flush=True)
    print("=" * 62, flush=True)
