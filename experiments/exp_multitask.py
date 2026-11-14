# -*- coding: utf-8 -*-
"""实验：让一个网络同时学会两个冲突任务，需要什么条件？

对照已知：
  只学A → A98%/B0%；只学B → A0%/B100%；先A后B(λ=0) → A32%/B18%（都废）
诊断假设：单一 W_out 无法表示"同输入不同输出"，除非网络能用某种「任务上下文」条件化。

实验矩阵：
  E1 原任务(TAG: A=0, B=0.9) + 交替训练
  E2 双向TAG(TAG: A=-0.9, B=+0.9) + 交替训练   ← 关键改进：TAG 始终有信息
  E3 双向TAG + 更长训练
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from xiaosha_cortex import SpikingCortex
from xiaosha_hippocampus import N_CH, TAG, run_episode, evaluate, ClassifyTask


class BiTagTask:
    """双向 TAG：A 用 -0.9，B 用 +0.9（原版 A=0 时 TAG 无信息，网络会忽略它）"""
    def __init__(self, seed=5, task_id=0):
        self.rng = np.random.default_rng(seed)
        self.task_id = task_id
        self.cls = 0

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


def eval_bi(cx, task_id, TaskCls, n=40):
    t = TaskCls(seed=5, task_id=task_id)
    hits = 0
    for i in range(n):
        r = run_episode(cx, t, learn=False, max_decisions=1, steps=10, seed=5000 + i)
        hits += int(r["hit"])
    return hits / n


def train_alternating(cx, rounds, TaskCls, seed=9000):
    tA, tB = TaskCls(seed=5, task_id=0), TaskCls(seed=5, task_id=1)
    for i in range(rounds):
        t = tA if i % 2 == 0 else tB
        run_episode(cx, t, learn=True, max_decisions=1, steps=10, seed=seed + i)


print("=" * 66)
print("实验：一个网络如何同时学会两个冲突任务")
print("=" * 66)

# E1: 原任务 + 交替
cx = SpikingCortex(n=6000, seed=11)
train_alternating(cx, 400, ClassifyTask)
print("E1 原TAG(A=0,B=.9) 交替400轮 → A=%.0f%%  B=%.0f%%"
      % (evaluate(cx, 0, 40)*100, evaluate(cx, 1, 40)*100))

# E2: 双向 TAG + 交替
cx2 = SpikingCortex(n=6000, seed=11)
train_alternating(cx2, 400, BiTagTask)
a2, b2 = eval_bi(cx2, 0, BiTagTask), eval_bi(cx2, 1, BiTagTask)
print("E2 双向TAG 交替400轮       → A=%.0f%%  B=%.0f%%" % (a2*100, b2*100))

# E3: 双向 TAG + 更长
cx3 = SpikingCortex(n=6000, seed=11)
train_alternating(cx3, 800, BiTagTask, seed=20000)
a3, b3 = eval_bi(cx3, 0, BiTagTask), eval_bi(cx3, 1, BiTagTask)
print("E3 双向TAG 交替800轮       → A=%.0f%%  B=%.0f%%" % (a3*100, b3*100))

# E4: 双向 TAG + 更大网络
cx4 = SpikingCortex(n=20000, seed=11)
train_alternating(cx4, 400, BiTagTask)
a4, b4 = eval_bi(cx4, 0, BiTagTask), eval_bi(cx4, 1, BiTagTask)
print("E4 双向TAG n=2万 400轮     → A=%.0f%%  B=%.0f%%" % (a4*100, b4*100))
print("=" * 66)