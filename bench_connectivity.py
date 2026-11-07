# -*- coding: utf-8 -*-
"""连接密度基准：细胞数固定 139,255，只提 K（每个细胞的连接数）→ 突触总数跟着涨。
用途：回答"连接数还能不能往上加、加到多少还能实时跑"。

结论（本机 i7-10700T，numpy 单线程，10ms 步长，稳态，默认档 homo_max=2.8）：
  K=8   ：1,114,040 突触 = 果蝇 2.2%，2.0 ms/步 → 505 Hz
  K=32  ：4,456,160 突触 = 果蝇 8.9%，约 2.1 ms/步 → 464~481 Hz
  K=128 ：17,824,640 突触 = 果蝇 35.6%，约 2.9~3.3 ms/步 → 307~348 Hz
  K=360 ：50,131,800 突触 = 果蝇 100.3%（追平），4.3~5.4 ms/步 → 185~234 Hz（实时，余量 3~4×）

能这么加连接的前提是**三条一起做**：
  ① 权重按 1/√K 归一化（保住输入涨落）
  ② 兴奋/抑制平衡（净均值≈0；只做①时 K=32 就烧到 41.6% 活跃、只有 17.7 Hz）
  ③ 放开稳态阈值上限（θ 上限 1.35 → 3.25：活跃度 4.83% → 0.28%，事件数降 17 倍）
⚠️ 稀疏要配足够的连接：K=32 压到 0.37% 活跃学习会崩（3/30），K=360 + 0.42% 反而最好（21/30）。
"""
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\Apps\MyModel\chat\workspace\xiaosha")
import xiaosha_cortex as C

if __name__ == "__main__":
    print("=" * 100)
    print("  连接密度基准：细胞数固定 139,255，只加连接（E/I 平衡 + 1/√K 归一化）")
    print("=" * 100)
    print(f"  {'K':>4} {'突触总数':>13} {'占果蝇':>7} {'ms/步':>8} {'最高Hz':>8} "
          f"{'活跃%/步':>9} {'大表内存':>9}")
    print("-" * 100)
    for k in (8, 16, 32, 64, 128, 256, 360):
        cx = C.SpikingCortex(n=C.FLY_NEURONS, k=k, seed=3)
        ext = np.full(cx.n_in, 0.8, dtype=np.float32)
        for _ in range(150):                    # 热身：让稳态调节收敛
            cx.step(ext)
        cx.total_spikes = cx.total_steps = 0
        t0 = time.perf_counter()
        steps = 15
        for _ in range(steps):
            cx.step(ext)
        per = (time.perf_counter() - t0) / steps
        mem = (cx.w.nbytes + cx.elig.nbytes + cx.tgt.nbytes) / 1e6
        print(f"  {k:>4} {cx.synapses():>13,} {cx.synapses()/C.FLY_SYNAPSES*100:>6.1f}% "
              f"{per*1000:>8.2f} {1/per:>8.1f} {cx.active_frac()*100:>8.2f}% {mem:>7.0f}MB",
              flush=True)
    print("-" * 100)
    print(f"  果蝇口径：139,255 细胞 / {C.FLY_SYNAPSES:,} 突触（K≈360 追平突触总数）")
    print("  注：'大表内存' = 权重表 + 资格迹表 + 连接索引，是加连接必须付的内存代价")
    print("=" * 100)
