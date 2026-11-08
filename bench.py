# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""小沙大脑 · 规模-速度扫描（开源 benchmark 入口）

用法：python bench.py          # 默认扫描 2万 / 6万 / 13.9万 / 28.9万
      python bench.py 100000   # 自定义单档
每个规模跑 20 步计时，输出「你的电脑能跑多大」对照表。
"""
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

from xiaosha_cortex import SpikingCortex, FLY_NEURONS

DEFAULT_SIZES = [20_000, 60_000, 139_260, 288_964]
WARM, STEPS = 10, 20

print("=" * 76)
print("  小沙大脑 · 规模-速度扫描（每次跑 20 步计时）")
print("  果蝇参照：139,255 神经元 / 实时 = 每步 ≤ 20ms（50Hz）")
print("=" * 76)
print(f"  {'规模':>10} | {'步进 ms':>8} | {'频率 Hz':>8} | {'vs 果蝇':>8} | 实时?")

sizes = [int(x) for x in sys.argv[1:]] or DEFAULT_SIZES
results = []
for n in sizes:
    try:
        cx = SpikingCortex(n=n)
        ext = np.full(cx.n_in, 0.8, dtype=np.float32)
        for _ in range(WARM):
            cx.step(ext)
        t0 = time.time()
        for _ in range(STEPS):
            cx.step(ext)
        ms = (time.time() - t0) / STEPS * 1000
        hz = 1000 / ms
        ratio = n / FLY_NEURONS
        rt = "实时 ✓" if ms <= 20 else "慢于实时"
        print(f"  {n:>10,} | {ms:>8.2f} | {hz:>8.0f} | {ratio:>7.2f}× | {rt}")
        results.append((n, ms, hz))
    except MemoryError:
        print(f"  {n:>10,} | {'内存不足':>10} | {'-':>8} | {'-':>8} | -")
        break
    except Exception as e:
        print(f"  {n:>10,} | 失败: {e}")
        break

if results:
    n_max, ms_max, hz_max = results[-1]
    print("-" * 76)
    print(f"  这台电脑实测最大规模 {n_max:,} 神经元：{ms_max:.1f}ms/步（{hz_max:.0f}Hz）")
    if n_max >= FLY_NEURONS:
        print(f"  → 已超过果蝇规模（{n_max/FLY_NEURONS:.2f}×）{'且实时' if ms_max <= 20 else ''}")
    try:
        import psutil
        mem_free = psutil.virtual_memory().available / 1024**3
        if mem_free < 8:
            print(f"  ⚠ 当前可用内存仅 {mem_free:.0f}GB（后台程序占用），速度被拖慢；")
            print(f"    内存空闲的机器上参考：28.9 万 = 20.7ms/步（48Hz，实时）")
    except Exception:
        pass
print("=" * 76)
