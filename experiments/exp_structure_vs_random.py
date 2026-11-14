# -*- coding: utf-8 -*-
"""功能验证：结构化皮层 vs 随机连接皮层 —— 谁真的学得快？

为什么这个实验重要：
  "结构更精细"听起来好听，但必须证明它**有功能优势**，否则只是好看。
  本实验用同一个任务、同一套读出规则、同一规模，只改"连接怎么生成"。

任务：方向分类（4 类）
  输入 10 维 → V1 的 4 个方向通道各代表一个方向（加噪声）
  读出：从网络的池化特征做 4 类分类，用同一个监督学习规则
  对比：达到 80% 准确率需要多少轮

对比组：
  A. StructuredCortex（本节新模块：脑区×层×E/I×规则投射）
  B. SpikingCortex（原版：随机稀疏连接）—— 同规模

判定标准（诚实）：
  - 学习曲线（每 50 轮测一次）
  - 达到 80% 所需轮数
  - 最终准确率
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from xiaosha_cortex import SpikingCortex
from xiaosha_structured import StructuredCortex

N = 60000          # 用中等规模（跑得起、能对比）
N_CLASS = 4
N_ROUNDS = int(os.environ.get('ROUNDS', '400'))
EVAL_EVERY = 50


def make_direction_task(rng, n=64):
    """方向分类任务：4 个方向，输入在 10 维上编码。"""
    X, y = [], []
    for _ in range(n):
        c = int(rng.randint(N_CLASS))
        o = np.zeros(10, dtype=np.float32)
        o[c] = 0.9
        o[(c + 1) % 10] = 0.35
        o += rng.normal(0, 0.06, 10).astype(np.float32)
        np.clip(o, 0, None, out=o)
        X.append(o); y.append(c)
    return X, np.array(y)


def pool_features(cx, obs, steps=8, region=None):
    """跑几步，把发放按神经元 id 分 64 桶池化。

    region=None → 全局池化（随机网络没有解剖区，只能这样）
    region=3    → 只池化 PFC（结构化网络的决策区，信息在这里汇聚）
    """
    counts = np.zeros(256, dtype=np.float32)
    for _ in range(steps):
        cx.step(obs)
        sp = getattr(cx, "_last_spike_ids", None)
        if sp is not None and len(sp):
            if region is not None and hasattr(cx, "region_id"):
                sp = sp[cx.region_id[sp] == region]
            if len(sp):
                counts += np.bincount(sp % 256, minlength=256).astype(np.float32)
    return counts / steps


def train_readout(feat, y, W, lr=0.15):
    """简单 softmax 读出（同一个规则给两组用，保证公平）。"""
    logits = W @ feat
    logits -= logits.max()
    p = np.exp(logits); p /= p.sum()
    grad = p.copy(); grad[y] -= 1.0
    W -= lr * np.outer(grad, feat)
    return W, float(p[y])


def evaluate(cx, W, rng, n=40, region=None):
    X, y = make_direction_task(rng, n)
    hit = 0
    for o, t in zip(X, y):
        f = pool_features(cx, o, steps=8, region=region)
        pred = int(np.argmax(W @ f))
        hit += int(pred == t)
    return hit / n


def run_one(kind, seed=11, verbose=True):
    rng = np.random.RandomState(seed)
    if kind == "structured":
        cx = StructuredCortex(total=N, seed=seed)
    else:
        cx = SpikingCortex(n=N, seed=seed)
    # 池化区：结构化读 PFC（决策区），随机读全局（它没有解剖区）
    pool_region = 3 if kind == "structured" else None
    W = np.zeros((N_CLASS, 256), dtype=np.float64)
    curve = []
    t0 = time.time()
    for rd in range(N_ROUNDS):
        X, y = make_direction_task(rng, 8)
        for o, t in zip(X, y):
            f = pool_features(cx, o, steps=8, region=pool_region)
            W, _ = train_readout(f, t, W)
        if (rd + 1) % EVAL_EVERY == 0:
            acc = evaluate(cx, W, rng, 40, region=pool_region)
            curve.append(((rd + 1), acc))
            if verbose:
                print("  %-11s 轮 %4d → %.0f%%  (%.0fs)" % (kind, rd + 1, acc * 100, time.time() - t0), flush=True)
    # 边界：轮数少于 EVAL_EVERY 时曲线为空，补一次评估（防止下游 curve[-1] 崩）
    if not curve:
        acc = evaluate(cx, W, rng, 40, region=pool_region)
        curve.append((N_ROUNDS, acc))
        if verbose:
            print("  %-11s 轮 %4d → %.0f%%  (%.0fs) [补充评估]" % (kind, N_ROUNDS, acc * 100, time.time() - t0), flush=True)
    return curve, time.time() - t0


if __name__ == "__main__":
    print("=" * 72)
    print("功能验证：结构化 vs 随机连接（同任务/同读出/同规模 %s）" % f"{N:,}")
    print("=" * 72)

    results = {}
    for kind in ["structured", "random"]:
        print("\n[%s]" % kind, flush=True)
        curve, el = run_one(kind, seed=11)
        results[kind] = curve
        print("  用时 %.1fs" % el, flush=True)

    print("\n" + "=" * 72)
    print("对比结果")
    print("=" * 72)
    print("%-12s %s" % ("轮数", "   ".join("%-10s" % k for k in results)))
    for i in range(len(results["structured"])):
        rounds = results["structured"][i][0]
        row = []
        for k in results:
            row.append("%-10s" % ("%.0f%%" % (results[k][i][1] * 100)))
        print("%-12s %s" % (rounds, "   ".join(row)))

    # 达到 80% 的轮数
    print()
    for k, curve in results.items():
        hit80 = next((r for r, a in curve if a >= 0.80), None)
        final = curve[-1][1]
        print("%-12s 达到80%%所需轮数: %s | 最终 %.0f%%"
              % (k, hit80 if hit80 else "未达到", final * 100))
