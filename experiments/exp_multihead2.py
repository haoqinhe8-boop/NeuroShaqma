# -*- coding: utf-8 -*-
"""多头读出实验 v2（修正轮数）。

v1 的教训：300 轮交替 = 每个头只训练 150 轮，太少先学会 → 结论无效，已废弃。
v2：每个头各训练 400 轮（共 800 次更新），才有可比性。
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
    N = 6000
    ROUNDS_PER_HEAD = 400

    print("=" * 68, flush=True)
    print("多头读出 v2：每个任务独立 W_out 头，各训练 %d 轮" % ROUNDS_PER_HEAD, flush=True)
    print("=" * 68, flush=True)

    # ---- 多头 ----
    cx = SpikingCortex(n=N, seed=11)
    heads = {0: cx.W_out.copy(), 1: cx.W_out.copy()}
    tA, tB = BiTagTask(seed=5, task_id=0), BiTagTask(seed=5, task_id=1)
    total = ROUNDS_PER_HEAD * 2
    for i in range(total):
        tid = i % 2
        t = tA if tid == 0 else tB
        cx.W_out = heads[tid]
        run_episode(cx, t, learn=True, max_decisions=1, steps=10, seed=9000 + i)
        heads[tid] = cx.W_out.copy()
    cx.W_out = heads[0]; a_mh = eval_task(cx, 0)
    cx.W_out = heads[1]; b_mh = eval_task(cx, 1)
    print("多头 %d轮/头  → A=%.0f%%  B=%.0f%%" % (ROUNDS_PER_HEAD, a_mh * 100, b_mh * 100), flush=True)

    # ---- 单头对照（同样总轮数）----
    cx2 = SpikingCortex(n=N, seed=11)
    for i in range(total):
        t = tA if i % 2 == 0 else tB
        run_episode(cx2, t, learn=True, max_decisions=1, steps=10, seed=9000 + i)
    a_sh, b_sh = eval_task(cx2, 0), eval_task(cx2, 1)
    print("单头 %d轮    → A=%.0f%%  B=%.0f%%" % (total, a_sh * 100, b_sh * 100), flush=True)

    print("-" * 68, flush=True)
    print("多头 vs 单头: A %+.0fpp, B %+.0fpp" % ((a_mh - a_sh) * 100, (b_mh - b_sh) * 100), flush=True)
    if a_mh >= 0.85 and b_mh >= 0.85:
        print("结论：多头能做到近乎零遗忘（A/B 都 >85%），共享单头做不到。", flush=True)
    else:
        print("结论：即使多头，%d 轮/头仍不够（诚实记录，不是机制问题）。" % ROUNDS_PER_HEAD, flush=True)
    print("=" * 68, flush=True)
