# -*- coding: utf-8 -*-
"""多巴胺(学习率)扫描：小沙的最优学习率区间在哪？

意外发现：高 DA(lr×2.2) 学不好(28%→50%)，低 DA(lr×0.4) 反而好(57%→82%)
假设：学习率过高 → 权重震荡/softmax 饱和 → 不稳定
本实验：扫 5 个 DA 水平，各 3 个独立种子，看最终准确率 + 方差
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from xiaosha_cortex import SpikingCortex
from xiaosha_hippocampus import ClassifyTask, run_episode, evaluate
from xiaosha_neuromod import Neuromodulator


def run(da, seed, rounds=150, n=5000):
    cx = SpikingCortex(n=n, seed=seed)
    Neuromodulator(da=da).apply(cx)
    lr = cx.lr_out
    t = ClassifyTask(seed=5, task_id=0)
    for i in range(rounds):
        run_episode(cx, t, learn=True, max_decisions=1, steps=10, seed=9000 + i)
    return lr, evaluate(cx, 0, n=40)


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("多巴胺(学习率)扫描：5 个水平 × 3 个种子（各 150 轮）", flush=True)
    print("=" * 70, flush=True)
    print("%-8s %-10s %-28s %s" % ("DA", "实际lr", "3种子准确率", "均值±标准差"), flush=True)
    best = None
    for da in [0.0, 0.25, 0.5, 0.75, 1.0]:
        accs = []
        lr = None
        for sd in [11, 22, 33]:
            lr, acc = run(da, sd)
            accs.append(acc)
        m, s = np.mean(accs), np.std(accs)
        print("%-8.2f %-10.4f %-28s %.0f%% ± %.0f%%"
              % (da, lr, " ".join("%.0f%%" % (a * 100) for a in accs), m * 100, s * 100), flush=True)
        if best is None or m > best[1]:
            best = (da, m, s)
    print("-" * 70, flush=True)
    print("最优 DA=%.2f → 均值 %.0f%%（标准差 %.0f%%）" % (best[0], best[1] * 100, best[2] * 100), flush=True)
    print("=" * 70, flush=True)
