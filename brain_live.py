# -*- coding: utf-8 -*-
"""小沙大脑 · 实时直播服务（Python 仿真端）。

设计：
  - 跑真实 SpikingCortex（288,964 神经元），每步约 30ms（33 步/秒）
  - 每步输出一行 JSON 到 stdout，由 Node.js 读取并 SSE 广播给网页
  - 无限循环：**网页关了大脑也在跑**，重新打开接着看（真直播）

stdout 每行格式：
  {"t":1234,"n":867,"rate":0.0030,"sp":[12,345,...],"reg":[...],"up":12.3}

布局文件（layout.bin）：
  288964 × 3 × float32 = (x, y, region)，前端下载一次用于渲染。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from xiaosha_structured import StructuredCortex, REGIONS

# ---------------- 配置 ----------------
N = 288964            # 真实神经元规模
SEED = 42
MAX_SPIKES_SENT = 600  # 每帧最多发多少个索引（视觉上够，省带宽）
PUSH_EVERY = 1         # 每 N 步推一次
LAYOUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain_layout.bin")

# 脑区定义：(cx, cy, sx, sy, rot, 颜色索引)  —— 前端用同一份参数绘制
REGIONS = [
    (0.00, 0.00, 0.62, 0.42, 0.0, 0),
    (-0.52, 0.28, 0.24, 0.20, 1.2, 1),
    (0.55, 0.22, 0.22, 0.22, -0.9, 2),
    (-0.48, -0.35, 0.26, 0.17, -0.6, 3),
    (0.50, -0.32, 0.24, 0.19, 0.8, 4),
    (0.02, -0.62, 0.32, 0.13, 0.0, 5),
]
REGION_W = [0.42, 0.13, 0.12, 0.11, 0.11, 0.11]


def build_layout(cx):
    """从结构化皮层导出**真实**布局：脑区位置 + 层结构（不再是虚构团簇）。"""
    layout = np.column_stack([cx.xy, cx.region_id.astype(np.float32)]).astype(np.float32)
    layout.tofile(LAYOUT)
    return layout


def make_input(t):
    """感觉输入波形（结构：双通道扫动 + 周期事件 + 稀疏噪声）。

    注意：绝对强度由主循环的 AGC（自动增益）动态决定，这里只给"形状"。
    原因：这个 SNN 没有内稳态，固定强度要么爆炸（17% 活跃、单步 1s 卡死）
    要么静息（0.005%）；必须闭环控制。
    """
    obs = np.zeros(10, dtype=np.float32)
    obs[int(t * 0.03) % 10] = 1.0
    obs[int(t * 0.03 + 5) % 10] = 0.85
    if t % 50 < 2:
        obs[(t // 50) % 10] = 1.0
    if t % 5 == 0:
        obs[(t * 3) % 10] = 0.6
    return obs


def main():
    sys.stderr.write("[brain-live] 构造结构化皮层 %d 神经元（6 脑区 x 3 层 x E/I）...\n" % N)
    cx = StructuredCortex(total=N, seed=SEED)
    layout = build_layout(cx)
    sys.stderr.write("[brain-live] 构造完成: %s 突触，开始直播\n" % f"{cx.n_syn:,}")
    print(json.dumps({"type": "ready", "n": N, "syn": cx.n_syn,
                      "regions": [r[0] for r in REGIONS]}), flush=True)

    t = 0
    t_start = time.time()
    last_push = 0.0
    # 注意：StructuredCortex 内建自适应阈值（稳态），不需要外层 AGC
    while True:
        obs = make_input(t)
        cx.step(obs)
        t += 1
        sp = cx._last_spike_ids
        now = time.time()
        if (t % PUSH_EVERY == 0) and (now - last_push) > 0.02:  # 最多 50 帧/秒
            last_push = now
            idx = sp
            if idx.size > MAX_SPIKES_SENT:
                sel = np.random.choice(idx.size, MAX_SPIKES_SENT, replace=False)
                idx = idx[sel]
            # 各脑区活跃数（真实脑区：V1/V2/IT/PFC/M1/HP）
            rid_of_sp = cx.region_id[sp] if sp.size else np.zeros(0, dtype=np.int8)
            reg_counts = [int(np.count_nonzero(rid_of_sp == r)) for r in range(len(REGIONS))]
            out = {
                "t": t,
                "n": int(sp.size),
                "rate": round(float(sp.size) / N, 6),
                "sp": idx.tolist(),
                "reg": reg_counts,
                "up": round(now - t_start, 1),
            }
            print(json.dumps(out, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("[brain-live] 收到中断，退出\n")
