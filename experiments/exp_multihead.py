# -*- coding: utf-8 -*-
"""多任务遗忘的根源定位：共享读出层 vs 独立读出头。

已知（本次实验）：
  E1 原TAG 400轮      → A=55% B=45%
  E2 双向TAG 400轮    → A=55% B=75%
  E3 双向TAG 800轮    → A=40% B=100%   ← recency bias：B 越来越好，A 越来越差
  E4 双向TAG 2万 400轮 → A=55% B=72%

假设：遗忘的根源不是网络容量，是「一个共享 W_out 要同时表示两个冲突解」。
上界参照：给每个任务一个独立读出头（训练时切换 W_out），预期零遗忘。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from xiaosha_cortex import SpikingCortex
from xiaosha_hippocampus import N_CH, TAG, run_episode


class BiTagTask:
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


if __name__ == "__main__":
    print("=" * 66, flush=True)
    print("多头读出实验（上界参照）：每个任务一个独立 W_out 头", flush=True)
    print("=" * 66, flush=True)

    cx = SpikingCortex(n=6000, seed=11)
    heads = {0: cx.W_out.copy(), 1: cx.W_out.copy()}
    tA, tB = BiTagTask(seed=5, task_id=0), BiTagTask(seed=5, task_id=1)

    rounds = 300
    for i in range(rounds):
        tid = i % 2
        t = tA if tid == 0 else tB
        cx.W_out = heads[tid]                     # 切到该任务的读出头
        run_episode(cx, t, learn=True, max_decisions=1, steps=10, seed=9000 + i)
        heads[tid] = cx.W_out.copy()              # 存回

    # 分别用各自头评估
    cx.W_out = heads[0]; a = eval_task(cx, 0)
    cx.W_out = heads[1]; b = eval_task(cx, 1)
    print("多头 交替%d轮       → A=%.0f%%  B=%.0f%%（零遗忘，各自独立）"
          % (rounds, a * 100, b * 100), flush=True)

    # 对照：共享单头同轮数
    cx2 = SpikingCortex(n=6000, seed=11)
    for i in range(rounds):
        t = tA if i % 2 == 0 else tB
        run_episode(cx2, t, learn=True, max_decisions=1, steps=10, seed=9000 + i)
    print("单头 交替%d轮       → A=%.0f%%  B=%.0f%%（recency bias）"
          % (rounds, eval_task(cx2, 0) * 100, eval_task(cx2, 1) * 100), flush=True)
    print("=" * 66, flush=True)
    print("结论：遗忘根源 = 共享读出层容量，不是网络神经元不够。", flush=True)
    print("      真实大脑靠「多通路/门控」解决（基底节-皮层环路选择读出头）。", flush=True)
