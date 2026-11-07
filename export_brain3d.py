# -*- coding: utf-8 -*-
"""导出小沙脉冲皮层 3D 数据（真实仿真，288,964 × 720）→ workspace/brain3d-data.json"""
import sys, json, time
import numpy as np
sys.path.insert(0, r"D:\Apps\MyModel\chat\workspace\xiaosha")
from xiaosha_cortex import SpikingCortex, FLY_NEURONS, FLY_SYNAPSES

N_SHOW = 60000       # 3D 展示神经元数（28.9 万抽样，视觉已饱和）
N_SYN = 30000        # 展示突触数
WARM, REC = 300, 400 # 热身 / 记录步数

t0 = time.time()
cx = SpikingCortex()  # 默认 288,964 × 720
print(f"[1/4] 构造完成 {time.time()-t0:.0f}s：神经元 {cx.n:,} ｜ 突触 {cx.synapses():,}（果蝇 {cx.synapses()/FLY_SYNAPSES*100:.0f}%）")

rng = np.random.default_rng(42)
idx = np.sort(rng.choice(cx.n, size=N_SHOW, replace=False))
show_inh = cx.inh[idx]

# 位置（有结构感的脑布局）：中空椭球壳（皮层）+ 前下感觉簇 + 后上输出核（组池化读出）
z = rng.uniform(-1, 1, N_SHOW)
phi = rng.uniform(0, 2 * np.pi, N_SHOW)
r = np.cbrt(rng.uniform(0.72 ** 3, 1.0))          # 壳分布：r ∈ [0.72, 1.0]（中空，像脑皮层）
st = np.sqrt(np.maximum(0.0, 1 - z * z))
x = (st * np.cos(phi) * r * 1.0)
y = (st * np.sin(phi) * r * 0.82)
z = z * r * 0.68

# 输入区（接收 10 路感觉输入的神经元）→ 前下方"感觉面"簇
in_hit = np.zeros(cx.n, dtype=bool)
in_hit[cx.in_tgt.ravel()] = True
show_in = in_hit[idx]
n_in = int(show_in.sum())
x[show_in] = 0.30 * rng.standard_normal(n_in)
y[show_in] = 0.24 * rng.standard_normal(n_in)
z[show_in] = -0.62 - 0.10 * np.abs(rng.standard_normal(n_in))

# 输出核：32 个特征组各 25 个"读出代表"，聚在后方小球（真实语义 = 组池化读出层）
n_out = 32 * 25
out_pos = np.zeros((n_out, 3), dtype=np.float32)
for g in range(32):
    th = g / 32 * 2 * np.pi
    cx0, cy0 = 0.18 * np.cos(th), 0.14 * np.sin(th)
    for j in range(25):
        out_pos[g * 25 + j] = [cx0 + 0.035 * rng.standard_normal(),
                               cy0 + 0.03 * rng.standard_normal(),
                               0.42 + 0.03 * rng.standard_normal()]
# 后 800 个展示位放输出核
out_slice = slice(N_SHOW - n_out, N_SHOW)
x[out_slice] = out_pos[:, 0]; y[out_slice] = out_pos[:, 1]; z[out_slice] = out_pos[:, 2]
show_out = np.zeros(N_SHOW, dtype=bool)
show_out[out_slice] = True
show_in[show_out] = False

typ = np.where(show_in, 2, np.where(show_out, 3, np.where(show_inh, 1, 0))).astype(np.int8)

# 跑仿真记录发放（喂正弦扫频输入，发放有节奏）
print(f"[2/4] 跑仿真 {WARM}+{REC} 步 ...")
pos_of = np.full(cx.n, -1, dtype=np.int32)
pos_of[idx] = np.arange(N_SHOW)
spike_steps = []
total_events = 0
for t in range(WARM + REC):
    ph = t / 40.0
    ext = 0.55 + 0.45 * np.sin(ph + np.arange(cx.n_in) * 0.7)  # 强波动输入 → 发放更活跃（视觉更好看）
    ext = np.clip(ext, 0.0, 1.0).astype(np.float32)
    cx.step(ext)
    if t >= WARM:
        ids = pos_of[cx._last_spike_ids]
        ids = ids[ids >= 0]
        spike_steps.append([int(v) for v in ids])
        total_events += len(ids)
print(f"    记录 {REC} 步，发放事件 {total_events:,}（每步平均 {total_events//REC} 个）")

# 连接（三类，有方向叙事）：
# ① 输入投影：感觉面 → 皮层壳（真实 in_tgt）
# ② 壳内连接：皮层内部（真实 tgt，少量）
# ③ 组池化：皮层壳 → 输出核同组代表（真实语义 = 32 组池化读出）
print(f"[3/4] 抽样连接（输入投影 / 壳内 / 组池化）...")
input_syn = []
in_disp = np.where(show_in)[0]
for i_ in in_disp:
    gi = idx[i_]
    pj = pos_of[cx.tgt[gi, rng.integers(0, cx.k)]]
    if pj >= 0 and not show_out[pj]:
        input_syn.append([int(i_), int(pj)])

syns = []
tries = 0
while len(syns) < 3000 and tries < 100000:
    tries += 1
    i = int(rng.integers(0, cx.n))
    if pos_of[i] < 0 or show_in[pos_of[i]] or show_out[pos_of[i]]:
        continue
    j = int(cx.tgt[i, rng.integers(0, cx.k)])
    pj = pos_of[j]
    if pj >= 0 and not show_in[pj] and not show_out[pj]:
        syns.append([int(pos_of[i]), int(pj)])

# 组池化线：壳上神经元 → 该组输出代表（group_id = i % 32）
pool_syn = []
group_repr = {g: N_SHOW - n_out + g * 25 for g in range(32)}   # 每组代表在输出核里的索引
cand = np.where(~(show_in | show_out))[0]
rng.shuffle(cand)
for i_ in cand[:5000]:
    gi = idx[i_]
    g = gi % 32
    pool_syn.append([int(i_), group_repr[g]])
print(f"    输入投影 {len(input_syn):,} 条 ｜ 壳内 {len(syns):,} 条 ｜ 组池化 {len(pool_syn):,} 条")

out = {
    "meta": {
        "neurons": int(cx.n), "synapses": int(cx.synapses()),
        "fly_neurons": int(FLY_NEURONS), "fly_synapses": int(FLY_SYNAPSES),
        "steps": REC, "dt": 0.01, "shown": N_SHOW, "shown_syn": len(syns),
        "total_events": total_events,
    },
    "pos": [[round(float(a), 3), round(float(b), 3), round(float(c), 3)] for a, b, c in zip(x, y, z)],
    "typ": [int(v) for v in typ],
    "spikes": spike_steps,
    "syn": syns,
    "insyn": input_syn,
    "poolsyn": pool_syn,
}
p = r"D:\Apps\MyModel\chat\workspace\brain3d-data.json"
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"))
import os
print(f"[4/4] 已导出 {p}（{os.path.getsize(p)/1024/1024:.1f} MB，耗时 {time.time()-t0:.0f}s）")
