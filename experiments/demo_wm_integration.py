# -*- coding: utf-8 -*-
"""端到端演示：小沙皮层感知 + FORCE 工作记忆协作。

任务（真实具身场景）：
  1) 小沙"看到"一个颜色线索（皮层处理，输出 32 维特征）
  2) 转身（3 秒延迟，中间无输入）
  3) 看到第二个颜色，判断「和刚才一样吗」

对比：
  - 无工作记忆（只看当前）→ 约 50%（瞎猜）
  - 皮层特征 + FORCE 工作记忆 → 应显著高于 50%

这验证的不是"FORCE 模块能跑"，而是**能接在小沙真实皮层输出上用**。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from xiaosha_cortex import SpikingCortex
from xiaosha_force import ForceNetwork

N_COLOR = 4          # 4 种颜色
N_CH = 10            # 视觉输入通道（必须 = cortex n_in）


def color_obs(c, noise=0.05, rng=None):
    """生成一个颜色的输入（4 个通道强激活 + 噪声）。"""
    o = np.zeros(N_CH, dtype=np.float32)
    o[c] = 1.0
    o[(c + 1) % N_CH] = 0.3
    if rng is not None:
        o += rng.normal(0, noise, N_CH).astype(np.float32)
    return np.clip(o, 0, None)


def cortex_encode(cx, obs, steps=6):
    """让皮层处理一个输入，返回池化特征（真实皮层表征）。"""
    cx.reset_feat()
    for _ in range(steps):
        cx.step(obs)
    return cx._xf.copy()


def trial(cx, net, delay_steps, learn_cortex=False, rng=None):
    """一局：看线索 → 延迟 → 看测试 → 判断是否同色。返回 (输出, 目标)。"""
    c1 = int(rng.randint(N_COLOR))
    c2 = int(rng.randint(N_COLOR))
    target = 1.0 if c1 == c2 else -1.0

    # 阶段1：看线索（皮层编码 → 喂给 FORCE 当输入）
    f1 = cortex_encode(cx, color_obs(c1, rng=rng))
    net.reset()
    # 线索期 60 步
    for _ in range(60):
        net.step(f1[:net.n_in] if net.n_in <= len(f1) else np.resize(f1, net.n_in), 0.0)

    # 阶段2：延迟（无输入）
    for _ in range(delay_steps):
        net.step(np.zeros(net.n_in), 0.0)

    # 阶段3：看测试颜色（皮层编码 → 喂给 FORCE）
    f2 = cortex_encode(cx, color_obs(c2, rng=rng))
    zs = []
    for _ in range(60):
        z, _ = net.step(f2[:net.n_in] if net.n_in <= len(f2) else np.resize(f2, net.n_in), target)
        zs.append(z)
    return float(np.mean(zs)), target


def baseline_no_memory(rng, n=60):
    """无工作记忆基线：只凭当前颜色猜（等价乱猜）。"""
    hits = 0
    for _ in range(n):
        hits += int(rng.random() < 0.5)
    return hits / n


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("端到端：小沙皮层 → FORCE 工作记忆 → 记忆判断", flush=True)
    print("=" * 70, flush=True)

    rng = np.random.RandomState(42)
    cx = SpikingCortex(n=6000, seed=11)
    net = ForceNetwork(n=800, n_in=16, seed=11)  # 用皮层特征前 16 维

    # 训练
    print("训练中（120 局，延迟 300 步 ≈ 3 秒）...", flush=True)
    for i in range(120):
        trial(cx, net, delay_steps=300, rng=rng)

    # 评估短延迟
    hits = 0; n_eval = 60
    for _ in range(n_eval):
        z, tgt = trial(cx, net, delay_steps=300, rng=rng)
        hits += int((z > 0) == (tgt > 0))
    acc = hits / n_eval
    print("\n结果（延迟 3 秒）:", flush=True)
    print("  皮层+FORCE 工作记忆: %.0f%%" % (acc * 100), flush=True)
    print("  无记忆基线(瞎猜)    : 50%%", flush=True)
    print("  提升                : %+.0fpp" % ((acc - 0.5) * 100), flush=True)

    # 更长延迟（6 秒）
    hits = 0
    for _ in range(40):
        z, tgt = trial(cx, net, delay_steps=600, rng=rng)
        hits += int((z > 0) == (tgt > 0))
    print("\n结果（延迟 6 秒）: %.0f%%（更长延迟是否还记住）" % (hits / 40 * 100), flush=True)
    print("=" * 70, flush=True)
