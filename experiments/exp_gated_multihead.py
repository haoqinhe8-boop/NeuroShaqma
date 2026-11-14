# -*- coding: utf-8 -*-
"""多任务门控读出（最后攻一次多任务遗忘）。

已知（本次实验）：
  共享单头 800 轮 → A=55% B=72%（recency bias）
  多头 400轮/头   → A=75% B=98%（有改善但 A 不够）
  上界：只学一个任务 → 98-100%

假设：多头方向是对的，只是每头训练轮数不够。本实验给每头 800 轮，
并在推理时用 TAG 做硬门控（TAG<0 用 A 头，TAG>0 用 B 头）。
→ 如果 A/B 都 >90%，说明「多头 + 门控」是可行解（且符合基底节-皮层多通路选择）。
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
    ROUNDS = int(os.environ.get("ROUNDS", "800"))

    print("=" * 70, flush=True)
    print("门控多头实验：每头 %d 轮 + TAG 硬门控" % ROUNDS, flush=True)
    print("=" * 70, flush=True)

    cx = SpikingCortex(n=N, seed=11)
    heads = {0: cx.W_out.copy(), 1: cx.W_out.copy()}
    tA, tB = BiTagTask(seed=5, task_id=0), BiTagTask(seed=5, task_id=1)

    total = ROUNDS * 2
    for i in range(total):
        tid = i % 2
        t = tA if tid == 0 else tB
        cx.W_out = heads[tid]
        run_episode(cx, t, learn=True, max_decisions=1, steps=10, seed=9000 + i)
        heads[tid] = cx.W_out.copy()
        if (i + 1) % 400 == 0:
            print("  进度 %d/%d" % (i + 1, total), flush=True)

    cx.W_out = heads[0]; a = eval_task(cx, 0)
    cx.W_out = heads[1]; b = eval_task(cx, 1)
    print("-" * 70, flush=True)
    print("多头+门控 (%d轮/头): A=%.0f%%  B=%.0f%%" % (ROUNDS, a * 100, b * 100), flush=True)

    # 对照：同总轮数单头
    cx2 = SpikingCortex(n=N, seed=11)
    for i in range(total):
        t = tA if i % 2 == 0 else tB
        run_episode(cx2, t, learn=True, max_decisions=1, steps=10, seed=9000 + i)
    a2, b2 = eval_task(cx2, 0), eval_task(cx2, 1)
    print("单头 (%d轮):        A=%.0f%%  B=%.0f%%" % (total, a2 * 100, b2 * 100), flush=True)
    print("-" * 70, flush=True)
    print("提升: A %+.0fpp, B %+.0fpp" % ((a - a2) * 100, (b - b2) * 100), flush=True)
    if a >= 0.90 and b >= 0.90:
        print("★ 结论：多头+门控 = 可行解（A/B 都 >90%，接近零遗忘）", flush=True)
    else:
        print("结论：多头+门控仍未达标（A=%.0f%% B=%.0f%%），需更强门控或更多训练" % (a * 100, b * 100), flush=True)
    print("=" * 70, flush=True)
