# -*- coding: utf-8 -*-
"""
脉冲神经网络吞吐基准（本机实测）—— 用来回答"想要多少神经细胞、这台机器跑得动多少"
方法：手写 CSR 风格稀疏连接（每神经元 K 个入突触），numpy 向量化，LIF 动力学，1ms 步长。
不引入 scipy/第三方 SNN 引擎，用的是这台机器上真实可用的最简内核。

输出：神经元·步/秒、突触·步/秒，并据此推算：
  · 果蝇全脑规模（139,255 神经元 / 5×10^7 突触，FlyWire）实时仿真需要多少算力
  · 本机（i7-10700T）能实时跑多少神经元
"""
import time
import numpy as np


def build(N, K, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=(N, K), dtype=np.int32)          # 每个神经元的入突触源
    w = rng.normal(0.0, 1.0, size=(N, K)).astype(np.float32)
    return idx, w


def run(N, K, steps=100, dt=1.0, tau=20.0, thresh=1.0, refractory=2):
    idx, w = build(N, K)
    v = np.zeros(N, dtype=np.float32)
    ref = np.zeros(N, dtype=np.int16)
    spk = (np.random.rand(N) < 0.01).astype(np.float32)
    total_spikes = 0
    t0 = time.perf_counter()
    for _ in range(steps):
        I = (w * spk[idx]).sum(axis=1)          # 稀疏突触传播（gather→mul→sum）
        v = v + (-v + I) / tau                   # LIF 漏电积分
        v = np.where(ref > 0, 0.0, v)
        ref = np.maximum(ref - 1, 0)
        fired = v >= thresh
        v = np.where(fired, 0.0, v)
        ref = np.where(fired, refractory, ref)
        spk = fired.astype(np.float32)
        total_spikes += int(fired.sum())
    t1 = time.perf_counter()
    el = t1 - t0
    return {
        "N": N, "K": K, "syn": N * K, "steps": steps, "seconds": el,
        "neuron_steps_per_s": N * steps / el,
        "syn_steps_per_s": N * K * steps / el,
        "spikes_per_s": total_spikes / el,
        "ms_per_step": el / steps * 1e3,
    }


if __name__ == "__main__":
    print("=" * 78)
    print("  脉冲网络吞吐基准（本机实测：i7-10700T，纯 numpy，1ms 步长）")
    print("=" * 78)
    print(f"{'神经元':>10} {'突触':>12} {'每步耗时':>10} {'神经元·步/秒':>16} {'突触·步/秒':>16}")
    print("-" * 78)
    rows = []
    for N, K in ((1_000, 8), (10_000, 8), (100_000, 8), (300_000, 8)):
        r = run(N, K, steps=60)
        rows.append(r)
        print(f"{r['N']:>10,} {r['syn']:>12,} {r['ms_per_step']:>9.2f}ms "
              f"{r['neuron_steps_per_s']:>16,.0f} {r['syn_steps_per_s']:>16,.0f}")

    best = rows[-1]
    print("-" * 78)
    ns = best["neuron_steps_per_s"]
    ss = best["syn_steps_per_s"]
    fly_n, fly_s = 139_255, 5e7
    need_n, need_s = fly_n * 1000.0, fly_s * 1000.0      # 实时 = 1ms 步长 = 1000 步/秒
    print(f"  · 果蝇全脑（{fly_n:,} 神经元 / {fly_s:.0e} 突触）要**实时**跑，需要：")
    print(f"      神经元-步/秒 {need_n:>16,.0f}   本机 {ns:>16,.0f}   "
          f"→ 只能达到实时的 {ns/need_n*100:5.1f}%")
    print(f"      突触-步/秒   {need_s:>16,.0f}   本机 {ss:>16,.0f}   "
          f"→ 只能达到实时的 {ss/need_s*100:5.2f}%（差 {need_s/ss:,.0f} 倍）")
    print("      ★ 瓶颈在突触：同样神经元数下，连接越密越跑不动")
    rt_syn = ss / 1000.0
    print(f"  · 本机（纯 numpy，单线程）实时可行规模 ≈ {rt_syn:,.0f} 个突触"
          f"（K=8 → 约 {rt_syn/8:,.0f} 个神经元）")
    print(f"  · 对照：沙琪玛写的纯 Python 版 bench_6000.py 实测 57.0ms/步（6000 神经元）"
          f"→ 约 10.5 万神经元-步/秒，比这里的 numpy 版慢约 {ns/105_000:,.0f} 倍")
    print("  · 要跑到 5×10^7 突触实时 → 需要 GPU / 多核 / 稀疏专用内核；"
          "本轮**没有**实测 GPU，不给数字。")
    print("=" * 78)
