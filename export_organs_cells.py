# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""小沙类脑器官细胞统计 + 分区发放数据导出（2026-11-07）

把 288,964 个皮层神经元按「器官=脑区」分区（真实脑科学功能映射的比例），
跑真实仿真记录各脑区发放，导出给 3D 工作动画使用。
分区（用户拍板器官 × 神经科学脑区对应）：
  鼻子=嗅球 3% ｜ 眼睛=视觉皮层 18% ｜ 耳朵=听觉皮层 8% ｜ 嘴巴=布洛卡语言区 6%
  脚=运动皮层·下肢 10% ｜ 手=运动皮层·手区 12% ｜ 其余皮层 43%
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\Apps\MyModel\chat\workspace\xiaosha")
from xiaosha_cortex import SpikingCortex

REGIONS = [
    ("鼻子·嗅球", 0.03, "#63e6be"),
    ("眼睛·视觉皮层", 0.18, "#4dabf7"),
    ("耳朵·听觉皮层", 0.08, "#f7b955"),
    ("嘴巴·语言区", 0.06, "#ff8f9e"),
    ("脚·运动皮层", 0.10, "#9ee6a0"),
    ("手·运动皮层", 0.12, "#e8d06a"),
    ("其他皮层", 0.43, "#c9a8a0"),
]
SHOW_PER_REGION = 900          # 每脑区展示点抽样数（3D 显示用）
WARM, REC = 300, 400           # 热身 / 记录步数
N_MONITOR = 60000              # 监控仿真规模（提交内存只余 2.5GB，完整 288,964 构造需 3.5GB）

# ---- 细胞体系（2026-11-07 联网查证后按人体真实比例重做）----
# 真实人体数据（OpenAlex/PubMed 查证 + 教科书事实）：
#   皮层 ~102 亿神经元（2023 论文）｜ 小脑 ~690 亿神经元（Herculano-Houzel 2018，= 皮层 4~7 倍）
#   视网膜每眼 1.07 亿视杆 + 460 万视锥（Curcio 1990 经典）｜ 耳蜗每耳 1.5 万毛细胞
#   嗅上皮每侧 300 万感觉神经元（双侧 600 万）｜ 脊髓 α 运动神经元 ~35 万
# 小沙按人体比例缩放（1/100 量级），器官 = 脑区 + 外周感觉/运动细胞
ORGAN_CELLS = {
    "鼻子·嗅球":      {"cortex": 8669,  "peripheral": 600000, "label": "嗅上皮感觉神经元（人体 600 万）"},
    "眼睛·视觉皮层":  {"cortex": 52014, "peripheral": 1120000, "label": "感光细胞（人体每眼 1.12 亿）"},
    "耳朵·听觉皮层":  {"cortex": 23117, "peripheral": 30000, "label": "耳蜗毛细胞（人体每耳 1.5 万）"},
    "嘴巴·语言区":    {"cortex": 17338, "peripheral": 35000, "label": "发声运动神经元"},
    "脚·运动皮层":    {"cortex": 28896, "peripheral": 170000, "label": "脊髓前角运动神经元（人体 α 运动 ~35 万）"},
    "手·运动皮层":    {"cortex": 34676, "peripheral": 350000, "label": "手部运动神经元"},
    "其他皮层":       {"cortex": 124254, "peripheral": 0, "label": ""},
}
CNS_EXTRA = {
    "小脑": 1200000,   # 人体小脑神经元 = 皮层 4~7 倍（690 亿 vs 102 亿），取 4.15×
    "脑干": 42000,
    "脊髓": 150000,    # 含脊髓灰质神经元
}
ORGANS_TOTAL = sum(v["cortex"] + v["peripheral"] for v in ORGAN_CELLS.values())
CNS_TOTAL = ORGANS_TOTAL + sum(CNS_EXTRA.values())

t0 = time.time()
cx = SpikingCortex(n=N_MONITOR)           # 监控抽样仿真（内存受限时用 6 万；完整统计仍按 288,964 报）
n = cx.n
print(f"[1/4] 皮层就位：监控仿真 {n:,} 神经元（完整皮层 288,964，统计见下）")

# ---- 分区：统计按细胞体系报（器官=脑区+外周，中枢补全）；仿真分区按比例分配 ----
TOTAL_N = 288_964
rng = np.random.default_rng(42)
perm = rng.permutation(n)
region_of = np.zeros(n, dtype=np.int8)
stats, cum = [], 0
lo = 0
for rid, (name, frac, color) in enumerate(REGIONS):
    cnt_full = int(round(frac * TOTAL_N))
    if rid == len(REGIONS) - 1:
        cnt_full = TOTAL_N - cum
    stats.append({"name": name, "cells": cnt_full, "pct": round(frac * 100, 1), "color": color})
    cum += cnt_full
    hi = int(round(sum(f * n for _, f, _ in REGIONS[:rid + 1])))
    if rid == len(REGIONS) - 1:
        hi = n
    region_of[perm[lo:hi]] = rid
    lo = hi
assert cum == TOTAL_N

# 器官完整细胞 = 脑区 + 外周
organ_stats = []
for s in stats:
    oc = ORGAN_CELLS[s["name"]]
    organ_stats.append({
        "name": s["name"], "cells": s["cells"] + oc["peripheral"],
        "cortex": s["cells"], "peripheral": oc["peripheral"],
        "pct": round(s["pct"], 1), "color": s["color"],
        "label": oc["label"],
    })
print("[2/4] 细胞体系统计（器官 = 脑区细胞 + 外周细胞；中枢 = 皮层 + 小脑 + 脑干 + 脊髓）：")
for s in organ_stats:
    extra = f"（脑区 {s['cortex']:,} + {s['label']} {s['peripheral']:,}）" if s["peripheral"] else "（纯脑区）"
    print(f"    {s['name']:　<10} {s['cells']:>8,} 细胞 {extra}")
for name, c in CNS_EXTRA.items():
    print(f"    {name}　　　　　　　 {c:>8,} 细胞")
print(f"    中枢神经系统合计 {CNS_TOTAL:,} 细胞（= 果蝇 139,255 的 {CNS_TOTAL/139255:.2f}×）")

# ---- 跑仿真，记录各脑区发放（每步：该区发放的神经元全局 id → 展示点索引）----
print(f"[3/4] 跑仿真 {WARM}+{REC} 步，记录各器官细胞发放 ...")
show_idx = []                       # 每区展示点：全局 id 列表
show_pos_of = np.full(n, -1, dtype=np.int32)
for rid in range(len(REGIONS)):
    ids = np.where(region_of == rid)[0]
    pick = np.sort(rng.choice(ids, size=min(SHOW_PER_REGION, len(ids)), replace=False))
    show_idx.append(pick)
    show_pos_of[pick] = np.arange(len(pick))
    # 展示点位置：按脑区空间摆放（嗅球前下、视觉后、听觉两侧、语言左前、运动头顶带）
    m = len(pick)
    if REGIONS[rid][0] == "鼻子·嗅球":
        pos = np.column_stack([0.06 * rng.standard_normal(m), 0.05 * rng.standard_normal(m) + 0.25,
                               -0.55 - 0.04 * np.abs(rng.standard_normal(m))])
    elif REGIONS[rid][0] == "眼睛·视觉皮层":
        th = rng.uniform(0, 2 * np.pi, m); ph = rng.uniform(-0.4, 0.4, m); rr = 0.72 + 0.1 * rng.random(m)
        pos = np.column_stack([rr * np.cos(th) * np.cos(ph), rr * np.sin(ph), 0.35 + rr * np.sin(th) * np.cos(ph) * 0.55])
    elif REGIONS[rid][0] == "耳朵·听觉皮层":
        side = np.where(rng.random(m) < 0.5, -1.0, 1.0)
        pos = np.column_stack([side * (0.62 + 0.08 * rng.random(m)), rng.uniform(-0.15, 0.25, m), rng.uniform(-0.2, 0.2, m)])
    elif REGIONS[rid][0] == "嘴巴·语言区":
        pos = np.column_stack([-0.28 + 0.1 * rng.standard_normal(m), rng.uniform(-0.1, 0.3, m), -0.5 + 0.08 * rng.standard_normal(m)])
    elif REGIONS[rid][0] == "脚·运动皮层":
        th = rng.uniform(0, 2 * np.pi, m); ph = rng.uniform(0.25, 0.5, m); rr = 0.8 + 0.06 * rng.random(m)
        pos = np.column_stack([rr * np.cos(th) * np.cos(ph), rr * np.sin(ph), rr * np.sin(th) * np.cos(ph) * 0.55])
    elif REGIONS[rid][0] == "手·运动皮层":
        th = rng.uniform(0, 2 * np.pi, m); ph = rng.uniform(0.25, 0.5, m); rr = 0.8 + 0.06 * rng.random(m)
        pos = np.column_stack([rr * np.cos(th + 1.6) * np.cos(ph), rr * np.sin(ph), rr * np.sin(th + 1.6) * np.cos(ph) * 0.55])
    else:
        th = rng.uniform(0, 2 * np.pi, m); ph = rng.uniform(-1.2, 1.2, m); rr = np.cbrt(rng.random(m)) * 0.72
        pos = np.column_stack([rr * np.cos(th) * np.cos(ph), rr * np.sin(ph), rr * np.sin(th) * np.cos(ph) * 0.6])
    if rid == 0:
        all_pos = pos.astype(np.float32)
        pos_region = np.full(m, rid, dtype=np.int8)
    else:
        all_pos = np.vstack([all_pos, pos.astype(np.float32)])
        pos_region = np.concatenate([pos_region, np.full(m, rid, dtype=np.int8)])

spike_steps = []                    # 每步：[(regionId, showIdx), ...]
region_counts = np.zeros(len(REGIONS), dtype=np.int64)
for t in range(WARM + REC):
    ph = t / 40.0
    ext = 0.5 + 0.35 * np.sin(ph + np.arange(cx.n_in) * 0.7)   # 峰值 0.85：防热身期活跃暴涨
    cx.step(np.clip(ext, 0.0, 1.0).astype(np.float32))
    if t >= WARM:
        ids = cx._last_spike_ids
        ev = []
        for gid in ids:
            rid = int(region_of[gid])
            region_counts[rid] += 1
            si = int(show_pos_of[gid])
            if si >= 0:
                ev.append([rid, si])
        spike_steps.append(ev)

print("    各器官细胞发放总量（400 步，监控抽样仿真）：")
for s, c in zip(stats, region_counts):
    print(f"    {s['name']:　<12} {c:>8,} 次发放（每步平均 {c/REC:.0f} 次）")

out = {
    "meta": {"total": TOTAL_N, "monitor": n, "steps": REC, "fly": 139255,
             "organs": organ_stats, "cnsExtra": CNS_EXTRA, "cnsTotal": CNS_TOTAL},
    "pos": [[round(float(x), 3), round(float(y), 3), round(float(z), 3), int(r)]
            for x, y, z, r in zip(all_pos[:, 0], all_pos[:, 1], all_pos[:, 2], pos_region)],
    "spikes": spike_steps,
    "regionTotals": [int(c) for c in region_counts],
}
p = r"D:\Apps\MyModel\chat\workspace\organs-cells.json"
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"))
import os
print(f"[4/4] 已导出 {p}（{os.path.getsize(p)/1024:.0f}KB，耗时 {time.time()-t0:.0f}s）")
