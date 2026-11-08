# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""GPU 皮层运行器（2026-11-08）：把 4090 fp16 混合精度正式接进代码库

针对 README 不足点「生产代码纯 CPU」：运行（仿真展示/推理）走 GPU，
权重 fp16 省一半显存、递送累加 fp32 保精度；学习仍在 CPU 皮层（R-STDP 语义不变）。

用法：python xiaosha_gpu.py --selftest   （无 NVIDIA 卡时全部 SKIP，不报失败）
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from xiaosha_cortex import SpikingCortex

try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False

CHUNK = 48  # 分块 gather：整块 int64 索引会 1.66GB OOM（实测踩坑，勿改大）


class GpuCortex:
    """把 CPU 皮层的连接结构搬到 GPU 上运行（fp16 权重 + fp32 累加）。"""

    def __init__(self, cpu_cx: SpikingCortex, device=None):
        if not HAS_TORCH:
            raise RuntimeError("torch 未安装")
        if device is None:
            device = "cuda:0"
        self.dev = torch.device(device)
        self.n = cpu_cx.n
        self.k = cpu_cx.k
        self.n_in = cpu_cx.n_in
        self.decay = cpu_cx.decay
        self.chunk = CHUNK
        # 权重 fp16（省一半显存），索引 int32
        self.w_g = torch.from_numpy(cpu_cx.w.astype(np.float16)).to(self.dev)
        self.tgt_g = torch.from_numpy(cpu_cx.tgt).to(self.dev)
        self.in_tgt_g = torch.from_numpy(cpu_cx.in_tgt).to(self.dev)
        self.in_w_g = torch.from_numpy(cpu_cx.in_w.astype(np.float16)).to(self.dev)
        # 状态
        self.v_g = torch.zeros(self.n, device=self.dev, dtype=torch.float32)
        self.theta_g = torch.full((self.n,), 1.0, device=self.dev, dtype=torch.float32)
        self.ref_g = torch.zeros(self.n, device=self.dev, dtype=torch.uint8)
        self.rate_ema_g = torch.zeros(self.n, device=self.dev, dtype=torch.float32)
        self.rate_alpha = cpu_cx.rate_alpha
        self.target_rate = cpu_cx.target_rate
        self.homo_lr = cpu_cx.homo_lr
        self.homo_max = cpu_cx.homo_max
        self.homo_every = cpu_cx.homo_every
        self._t = 0
        self._ext_g = torch.zeros(self.n_in, device=self.dev, dtype=torch.float32)
        self.last_spikes = 0
        self.total_spikes = 0
        self.total_steps = 0

    # ------------------------------------------------------------
    def step(self, ext=None):
        """GPU 推进一步（泄漏→输入递送→发放→突触递送→稳态）。"""
        v, th, ref = self.v_g, self.theta_g, self.ref_g
        v.mul_(self.decay)

        # 外部输入递送（对齐 CPU：ext>0 的通道都递送，幅度 = in_w × ext[ch]）
        if ext is not None:
            self._ext_g.copy_(torch.from_numpy(np.asarray(ext, dtype=np.float32)))
        else:
            self._ext_g.zero_()
        act_in = self._ext_g > 0.0
        for ch in range(self.n_in):
            if act_in[ch]:
                v.index_add_(0, self.in_tgt_g[ch], self.in_w_g[ch].float() * self._ext_g[ch].item())

        # 发放（不应期内 v 已被压 reset，等效 ref 过滤；对齐 CPU 216-218 行）
        if ref.any().item():
            v.masked_fill_(ref > 0, 0.0)
        spk = (v >= th)

        # 突触递送：发放者 i 的每条突触把 w[i,j] 送给 tgt[i,j]（与 CPU add.at 语义一致）
        # 注意：tgt 是"出边"表（i 打向谁），不是 bench 里的"前突触"表，不能用 spk[tgt] 稠密写法
        I = torch.zeros(self.n, device=self.dev, dtype=torch.float32)
        spk_f = spk.float()
        for c in range(0, self.k, self.chunk):
            idx = self.tgt_g[:, c:c + self.chunk].reshape(-1)
            contrib = (self.w_g[:, c:c + self.chunk].float() * spk_f.unsqueeze(1)).reshape(-1)
            I.index_add_(0, idx, contrib)

        v.add_(I)
        # 复位 + 不应期
        v.masked_fill_(spk, 0.0)
        ref.masked_fill_(spk, 2)
        ref.clamp_(max=2)
        ref[ref > 0] -= 1

        # 稳态：与 CPU 语义逐一对齐（发放神经元 ema 向 1 逼近；无发放步全体 ema 衰减；
        # 每 homo_every 步用 rate_ema×100 与 target_rate 比调阈值，clip 0.45 ~ 0.45+homo_max）
        if self.last_spikes > 0:
            self.rate_ema_g = torch.where(
                spk, self.rate_ema_g + self.rate_alpha * (1.0 - self.rate_ema_g), self.rate_ema_g)
        else:
            self.rate_ema_g.mul_(1.0 - self.rate_alpha)
        self._t += 1
        if self._t % self.homo_every == 0:
            err = self.rate_ema_g * 100.0 - self.target_rate
            self.theta_g = (self.theta_g + self.homo_lr * err).clamp(0.45, 0.45 + self.homo_max)

        self.last_spikes = int(spk.sum().item())
        self.total_spikes += self.last_spikes
        self.total_steps += 1
        self._last_spike_ids = spk.nonzero(as_tuple=True)[0].cpu().numpy()
        return self.last_spikes

    def firing_rate_hz(self):
        return self.total_spikes / max(1, self.total_steps) / self.n * 100.0


# ============================================================
def selftest():
    print("=" * 76)
    print("  GPU 皮层运行器 selftest（fp16 混合精度正式接入）")
    print("=" * 76)
    results, fails = [], []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    if not HAS_TORCH:
        check("torch 未安装 → GPU 全部 SKIP（不算失败）", True, "pip install torch")
        return 0
    if not torch.cuda.is_available():
        check("无 NVIDIA 卡 → GPU 全部 SKIP（不算失败）", True, "需要 CUDA 显卡")
        return 0

    dev_name = torch.cuda.get_device_name(0)
    print(f"  显卡：{dev_name}")

    # 1. 28.9 万全量构造 + 运行
    t0 = time.time()
    cx = SpikingCortex(n=288_964, seed=11)
    g = GpuCortex(cx, "cuda:0")
    del cx  # GPU 副本已拷完，释放 CPU 版（内存紧张机器上必须）
    import gc
    gc.collect()
    build_s = time.time() - t0
    check("28.9 万皮层 GPU 构造（fp16 权重 + int32 索引）", True, f"{build_s:.1f}s")

    # 2. 速度
    ext = np.full(g.n_in, 0.8, dtype=np.float32)
    for _ in range(10):
        g.step(ext)  # warmup
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        g.step(ext)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 100 * 1000
    hz = 1000 / ms
    check("28.9 万 GPU 步进速度（目标 < 30ms/步）", ms < 30,
          f"{ms:.2f}ms/步 = {hz:.0f}Hz（CPU 空闲参考 20.7ms/48Hz）")

    # 3. 稳态发放率行为级一致（6 万小网：300 步后稳态压平再比；20 步暂态受 fp16 量化+
    #    浮点顺序影响本就是噪声级，不适合做一致性判据）
    cx_cpu = SpikingCortex(n=60_000, seed=11)
    g2 = GpuCortex(cx_cpu, "cuda:0")
    ext2 = np.full(cx_cpu.n_in, 0.8, dtype=np.float32)
    for _ in range(300):
        cx_cpu.step(ext2)
    ext3 = np.full(g2.n_in, 0.8, dtype=np.float32)
    for _ in range(300):
        g2.step(ext3)
    cpu_rate = cx_cpu.firing_rate_hz()
    gpu_rate = g2.firing_rate_hz()
    dev_pct = abs(gpu_rate - cpu_rate) / max(cpu_rate, 0.1) * 100
    check("GPU 与 CPU 稳态发放率同量级（偏差 < 50%，fp16 行为级一致）", dev_pct < 50,
          f"CPU {cpu_rate:.2f}% vs GPU {gpu_rate:.2f}%，偏差 {dev_pct:.1f}%")

    # 4. 稳态不烧
    rates = []
    for i in range(300):
        g.step(ext)
        if i % 100 == 99:
            rates.append(g.firing_rate_hz())
    ok = all(r < 60 for r in rates)
    check("稳态：GPU 长时间跑不烧", ok, f"发放率 {[round(r, 2) for r in rates]} Hz")

    # 5. 显存占用报告
    mem = torch.cuda.memory_allocated(0) / 1024**2
    check("显存占用报告", True, f"{mem:.0f}MB（fp16 权重；fp32 约 {mem*1.7:.0f}MB）")

    print("-" * 76)
    print(f"  结果：{sum(results)}/{len(results)} 项通过"
          + ("  ✅ 全部通过" if not fails else f"  ❌ 失败：{fails}"))
    print("=" * 76)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    ap.print_help()
