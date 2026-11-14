# -*- coding: utf-8 -*-
"""小沙大脑 · 结构化皮层（Structured Cortex）

为什么要有这个模块：
  原来的 `SpikingCortex` 是**随机稀疏连接**——288,964 个神经元，每个随机连 720 个目标。
  它能跑、能学，但没有解剖学意义：任何一个神经元连到谁纯属随机。
  真实大脑不是这样：**几亿年进化出的结构，让每个连接都有分工**——
    · 不同脑区处理不同模态（V1 边缘 / IT 物体 / M1 运动）
    · 皮层分 6 层，层间连接有严格方向（L4 输入 → L2/3 关联 → L5 输出 → L6 反馈）
    · 细胞类型分化（兴奋性锥体 80% / 抑制性中间神经元 20%）
    · 投射有规则（前馈 V1→V2→IT，反馈 IT→V2，不是乱连）

本模块实现这套结构：脑区 × 层 × 细胞类型 × 特定投射规则。
每个连接的意义由 (源区/层/型 → 目标区/层/型) 决定。

用法：
    cx = StructuredCortex(total=288964, seed=42)
    cx.summary()                     # 打印结构统计
    cx.step(obs)                     # 和普通皮层一样的接口
    spikes = cx._last_spike_ids      # 这一步发放的神经元
"""
import numpy as np

# numba JIT 加速（可选；装了就自动用，没装回退纯 numpy）
try:
    from xiaosha_jit import step_jit as _step_jit, inject_jit as _inject_jit
    _HAS_JIT = True
except Exception:
    _HAS_JIT = False

__version__ = "1.0.0"

# ==================== 解剖定义 ====================

# 脑区：(名称, 占比, 功能)
REGIONS = [
    ("V1",  0.20, "视觉初级皮层（边缘/方向）"),
    ("V2",  0.15, "视觉次级皮层（轮廓/纹理）"),
    ("IT",  0.12, "下颞皮层（物体识别）"),
    ("PFC", 0.23, "前额叶（工作记忆/决策）"),
    ("M1",  0.14, "初级运动皮层（动作输出）"),
    ("HP",  0.16, "海马（记忆编码/重放）"),
]

# 皮层分层（简化 3 层，保留方向性）
LAYERS = [
    ("L4",  0.25, "输入层：接收丘脑/上行投射"),
    ("L23", 0.45, "浅层：区内关联 + 向上投射"),
    ("L5",  0.30, "深层：向下输出 + 反馈"),
]

# 细胞类型
CELL_TYPES = [
    ("E", 0.80, "锥体细胞（兴奋性，长程投射）"),
    ("I", 0.20, "中间神经元（抑制性，局部）"),
]

# 前馈通路（上行）：谁投射给谁，占该区输出的比例
FEEDFORWARD = [
    ("V1",  "V2",  0.06, "视觉前馈 V1→V2"),
    ("V2",  "IT",  0.06, "视觉前馈 V2→IT"),
    ("IT",  "PFC", 0.05, "语义前馈 IT→PFC"),
    ("PFC", "M1",  0.08, "决策到动作 PFC→M1"),
    ("V1",  "PFC", 0.01, "快速通路（皮层下捷径）"),
]

# 反馈通路（下行）：认知调控
FEEDBACK = [
    ("PFC", "IT",  0.05, "注意调控 PFC→IT"),
    ("PFC", "V2",  0.03, "注意调控 PFC→V2"),
    ("M1",  "PFC", 0.02, "运动副本 M1→PFC"),
    ("IT",  "V2",  0.03, "预测反馈 IT→V2"),
]

# 海马双向（记忆写入/读出）
HIPPO_LINKS = [
    ("IT",  "HP",  0.04, "写入记忆 IT→HP"),
    ("HP",  "PFC", 0.06, "重放到前额 HP→PFC"),
    ("HP",  "IT",  0.03, "记忆读出 HP→IT"),
]


class StructuredCortex:
    """结构化脉冲皮层：脑区 × 层 × 细胞类型 × 规则投射。"""

    def __init__(self, total=288964, seed=42, dt=10.0,
                 v_thresh=1.0, v_reset=0.0, ref_steps=2):
        self.rng = np.random.RandomState(seed)
        self.total = int(total)
        self.dt = dt
        self.theta = v_thresh
        self._theta0 = float(v_thresh)
        self.v_reset = v_reset
        self.ref_steps = ref_steps
        self.n_in = 10

        self._assign_cells()
        self._build_connections()

        # ---- 输入注入池（预生成 64 组不重复的目标，避免每步调 rng.choice）----
        # 实测：rng.choice 0.33ms/次 × 10 通道 = 3.3ms/步（占总耗时一半）
        #       预生成池 0.002ms/次 → 省掉这 3.3ms
        v1_l4 = self.groups[(self.region_names.index("V1"), 0, 0)]
        if v1_l4.size:
            k_inj = max(1, int(v1_l4.size * 0.02))
            self._inject_pools = [
                v1_l4[self.rng.choice(v1_l4.size, min(k_inj, v1_l4.size), replace=False)]
                for _ in range(256)
            ]
        else:
            self._inject_pools = []
        self._inject_i = 0
        if _HAS_JIT:
            self._sp_buf = np.zeros(self.total, dtype=np.int64)
            self._cnt_buf = np.zeros(len(REGIONS), dtype=np.int64)
            self._max_n = max(1, int(self.total * 0.02))
            self._max_per_region = max(1, int(self.total * 0.004))

        self.v = np.zeros(self.total, dtype=np.float32)
        self.ref = np.zeros(self.total, dtype=np.int16)
        self._last_spike_ids = np.zeros(0, dtype=np.int64)
        self._last_ext = np.zeros(self.n_in, dtype=np.float32)
        self.t = 0

    # ---------- 1) 细胞分配：每个神经元获得 (脑区, 层, 类型) 身份 ----------
    def _assign_cells(self):
        n = self.total
        # 脑区
        region_id = np.zeros(n, dtype=np.int8)
        counts = [int(n * w) for _, w, _ in REGIONS]
        counts[-1] = n - sum(counts[:-1])
        idx = 0
        for rid, c in enumerate(counts):
            region_id[idx:idx + c] = rid
            idx += c
        self.region_id = region_id
        self.region_names = [r[0] for r in REGIONS]
        self.region_counts = counts

        # 层（区内按比例，**随机打散**——层与细胞类型必须独立，否则会耦合）
        layer_id = np.zeros(n, dtype=np.int8)
        for rid, c in enumerate(counts):
            start = sum(counts[:rid])
            lay = np.zeros(c, dtype=np.int8)
            off = 0
            for lid, (_, w, _) in enumerate(LAYERS):
                cc = int(c * w) if lid < len(LAYERS) - 1 else c - off
                lay[off:off + cc] = lid
                off += cc
            self.rng.shuffle(lay)
            layer_id[start:start + c] = lay
        self.layer_id = layer_id
        self.layer_names = [l[0] for l in LAYERS]

        # 细胞类型（区内 80/20，**随机打散**）
        type_id = np.zeros(n, dtype=np.int8)
        for rid, c in enumerate(counts):
            start = sum(counts[:rid])
            t = np.zeros(c, dtype=np.int8)
            t[int(c * CELL_TYPES[0][1]):] = 1        # 后 20% 为 I
            self.rng.shuffle(t)
            type_id[start:start + c] = t
        self.type_id = type_id
        self.type_names = [t[0] for t in CELL_TYPES]

        # 2D 位置（区内按层排布，供可视化）
        xy = np.zeros((n, 2), dtype=np.float32)
        # 脑区画在大圆上，区内呈盘状
        ang = np.linspace(0, 2 * np.pi, len(REGIONS), endpoint=False)
        pos = 0
        for rid, c in enumerate(counts):
            cx0, cy0 = 0.72 * np.cos(ang[rid]), 0.72 * np.sin(ang[rid])
            r_loc = 0.30 * np.sqrt(c / n) * 3.0
            # 层：越靠外越是浅层（L4 在内核）
            lay = self.layer_id[pos:pos + c]
            rr = np.where(lay == 0, 0.35, np.where(lay == 1, 0.68, 1.0)) * r_loc
            th = self.rng.uniform(0, 2 * np.pi, c)
            jitter = self.rng.normal(0, r_loc * 0.10, c)
            xy[pos:pos + c, 0] = cx0 + (rr + jitter) * np.cos(th)
            xy[pos:pos + c, 1] = cy0 + (rr + jitter) * np.sin(th)
            pos += c
        self.xy = xy

    # ---------- 2) 连接生成：每个连接由 (源区层型 → 目标区层型) 规则决定 ----------
    def _build_connections(self):
        n = self.total
        rid, lay, typ = self.region_id, self.layer_id, self.type_id

        # 按 (区, 层, 型) 建索引，方便快速取子集
        self.groups = {}
        for r in range(len(REGIONS)):
            for l in range(len(LAYERS)):
                for t in range(len(CELL_TYPES)):
                    m = np.flatnonzero((rid == r) & (lay == l) & (typ == t))
                    self.groups[(r, l, t)] = m

        src_list, dst_list, w_list, kind_list = [], [], [], []

        def add(src_idx, dst_idx, w, kind):
            if src_idx.size == 0 or dst_idx.size == 0:
                return
            src_list.append(src_idx.astype(np.int32))
            dst_list.append(dst_idx.astype(np.int32))
            w_list.append(np.full(src_idx.size, w, dtype=np.float32))
            kind_list.append(np.full(src_idx.size, kind, dtype=np.int8))

        # ---- A. 区内局部连接（规则：E→E 弱、E→I 强、I→E/I 抑制）----
        # 密度按"每神经元出边数"设定，目标总连接 ~5000 万（内存约 650MB）
        for r in range(len(REGIONS)):
            for l in range(len(LAYERS)):
                E23 = self.groups[(r, l, 0)]
                I23 = self.groups[(r, l, 1)]
                if E23.size and I23.size:
                    # E→E：每个 E 连 150 个随机同类（局部关联，弱权重）
                    if E23.size > 200:
                        k = 150
                        s = np.repeat(E23, k)
                        d = E23[self.rng.randint(0, E23.size, s.size)]
                        keep = s != d
                        add(s[keep], d[keep], 0.007, 1)
                    # E→I：前馈兴奋（强，每 E 连 30 个 I）
                    if I23.size > 10:
                        s = np.repeat(E23, 30)
                        d = I23[self.rng.randint(0, I23.size, s.size)]
                        add(s, d, 0.180, 2)
                    # I→E：局部抑制（每个 I 连 100 个 E）
                    if E23.size > 100:
                        s = np.repeat(I23, 100)
                        d = E23[self.rng.randint(0, E23.size, s.size)]
                        add(s, d, -0.400, 3)
                    # I→I：抑制性互抑
                    if I23.size > 10:
                        s = np.repeat(I23, 20)
                        d = I23[self.rng.randint(0, I23.size, s.size)]
                        add(s, d, -0.220, 3)

        # ---- B. 层间微回路（L4 → L23 → L5，标准皮层方向）----
        for r in range(len(REGIONS)):
            L4, L23, L5 = self.groups[(r, 0, 0)], self.groups[(r, 1, 0)], self.groups[(r, 2, 0)]
            if L4.size and L23.size:
                s = np.repeat(L4, 40); d = L23[self.rng.randint(0, L23.size, s.size)]
                add(s, d, 0.250, 4)   # L4 → L23 上行
            if L23.size and L5.size:
                s = np.repeat(L23, 20); d = L5[self.rng.randint(0, L5.size, s.size)]
                add(s, d, 0.250, 5)   # L23 → L5 下行
            if L5.size and L4.size:
                s = np.repeat(L5, 10); d = L4[self.rng.randint(0, L4.size, s.size)]
                add(s, d, 0.120, 6)   # L5 → L4 反馈（弱）

        # ---- C. 前馈 / 反馈 / 海马投射（区间，特定分工）----
        # 按规模等比缩放（硬编码绝对值会让小网络里区间投射占比失真）
        PER_PATH = int(self.total * 0.76)

        def region_out_pool(rname):
            """L5 锥体 = 长程投射细胞（真实皮层里 L5 是主要输出层）。"""
            r = self.region_names.index(rname)
            m = self.groups[(r, 2, 0)]
            if m.size == 0:
                m = self.groups[(r, 1, 0)]
            return m

        def region_in_pool(rname, layer_idx=0):
            """L4 锥体 = 上行输入靶层。"""
            r = self.region_names.index(rname)
            m = self.groups[(r, layer_idx, 0)]
            if m.size == 0:
                m = np.flatnonzero(self.region_id == r)
            return m

        for src_r, dst_r, ratio, _desc in FEEDFORWARD + FEEDBACK + HIPPO_LINKS:
            sp_ = region_out_pool(src_r)
            dp_ = region_in_pool(dst_r, 0)
            if sp_.size == 0 or dp_.size == 0:
                continue
            ss = self.rng.choice(sp_, PER_PATH, replace=True).astype(np.int32)
            dd = self.rng.choice(dp_, PER_PATH, replace=True).astype(np.int32)
            add(ss, dd, 0.450, 7)

        # ---- 汇合 ----
        src = np.concatenate(src_list) if src_list else np.zeros(0, np.int32)
        dst = np.concatenate(dst_list) if dst_list else np.zeros(0, np.int32)
        w = np.concatenate(w_list) if w_list else np.zeros(0, np.float32)
        kind = np.concatenate(kind_list) if kind_list else np.zeros(0, np.int8)

        # 按源神经元排序（step 时按 id 切片）
        order = np.argsort(src, kind="stable")
        self.syn_src = src[order]
        self.syn_dst = dst[order]
        self.syn_w = w[order]
        self.syn_kind = kind[order]
        # 每个源神经元的连接范围
        self.syn_ptr = np.searchsorted(self.syn_src, np.arange(n + 1))
        self.n_syn = int(self.syn_src.size)

    # ---------- 3) 运行 ----------
    def step(self, ext=None):
        v = self.v
        # 输入（10 维感觉 → 映射到 V1 的 L4）
        if ext is not None:
            ext = np.asarray(ext, dtype=np.float32)[:self.n_in]
            self._last_ext[:] = ext
            if self._inject_pools:
                act = np.flatnonzero(ext > 0)
                for a in act:
                    # 从预生成池轮换取（零随机开销，替代 rng.choice 的 0.33ms/次）
                    tgt = self._inject_pools[self._inject_i]
                    self._inject_i = (self._inject_i + 1) & 255
                    if _HAS_JIT:
                        _inject_jit(v, tgt, float(ext[a]) * 1.4, tgt.size)
                    else:
                        v[tgt] += float(ext[a]) * 1.4

        # ===== JIT 加速路径（漏电/不应期/阈值/限流/递送全在编译内核里）=====
        if _HAS_JIT:
            nsp = _step_jit(v, self.ref, self.theta, 0.90, self.ref_steps,
                            self.syn_ptr, self.syn_dst, self.syn_w,
                            self.region_id, len(REGIONS),
                            self._max_n, self._max_per_region,
                            self._sp_buf, self._cnt_buf)
            sp = self._sp_buf[:nsp].copy()
            rate_now = nsp / self.total
            if rate_now > 0.015:
                self.theta = min(12.0, self.theta * 1.02)
            elif rate_now < 0.002:
                self.theta = max(self._theta0 * 0.6, self.theta * 0.995)
            self._last_spike_ids = sp
            self.t += 1
            return sp

        # ===== 纯 numpy 回退路径 =====
        # 漏电
        v *= 0.90
        # 不应期
        ref = self.ref
        if ref.any():
            v[ref > 0] = self.v_reset
            ref -= 1

        # 发放
        sp = np.flatnonzero(v >= self.theta)
        # 代谢约束：每步发放不超过 2% 神经元（真实大脑受能量限制，这比缓慢的阈值反馈稳定）
        # 光靠阈值慢速反馈会振荡（实测 20%→0%→0.04% 来回摆），硬限流 + 温和阈值才是稳的
        max_n = max(1, int(self.total * 0.02))
        if sp.size > max_n:
            keep = np.argpartition(v[sp], -max_n)[-max_n:]
            sp = np.sort(sp[keep])
        # 温和的稳态调节（让阈值跟着活动水平缓慢漂移，辅助而非主控）
        rate_now = sp.size / self.total
        if rate_now > 0.015:
            self.theta = min(12.0, self.theta * 1.02)
        elif rate_now < 0.002:
            self.theta = max(self._theta0 * 0.6, self.theta * 0.995)
        self._last_spike_ids = sp
        if sp.size:
            # 事件驱动：只处理发放神经元的下游（用 ptr 切片）
            lo = self.syn_ptr[sp]
            hi = self.syn_ptr[sp + 1]
            cnt = hi - lo
            total = int(cnt.sum())
            if total > 0:
                # 展开索引
                offs = np.repeat(lo - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
                idx = np.arange(total, dtype=np.int64) + offs
                tgt = self.syn_dst[idx]
                ww = self.syn_w[idx]
                np.add.at(v, tgt, ww)
            v[sp] = self.v_reset
            ref[sp] = self.ref_steps
        self.t += 1
        return sp

    # ---------- 4) 统计 ----------
    def summary(self):
        print("=" * 68)
        print("结构化皮层 · 结构统计")
        print("=" * 68)
        print("总神经元: %s | 总突触: %s" % (f"{self.total:,}", f"{self.n_syn:,}"))
        print("\n脑区（%d 个）：" % len(REGIONS))
        for i, (name, w, desc) in enumerate(REGIONS):
            m = self.region_id == i
            nsyn = int(np.count_nonzero(m[self.syn_src]))
            print("  %-4s %7s 神经元 (%4.1f%%)  出边 %9s   %s"
                  % (name, f"{m.sum():,}", w * 100, f"{nsyn:,}", desc))
        print("\n层（每区内）：")
        for li, (ln, lw, desc) in enumerate(LAYERS):
            print("  %-4s %4.1f%%  %s" % (ln, lw * 100, desc))
        print("\n细胞类型：")
        for ti, (tn, tw, desc) in enumerate(CELL_TYPES):
            m = self.type_id == ti
            print("  %-2s   %5.1f%%  %9s 个  %s" % (tn, tw * 100, f"{m.sum():,}", desc))
        print("\n连接类型分布：")
        kinds = ["E→E(区内)", "E→I(前馈兴奋)", "I→E/I(抑制)", "L4→L23", "L23→L5", "L5→L4", "区间投射"]
        # 注意：kind 值从 1 开始（add() 里传入 1..7），所以 enumerate 必须 start=1
        for k, name in enumerate(kinds, start=1):
            c = int(np.count_nonzero(self.syn_kind == k))
            if c:
                print("  %-16s %10s  (%.1f%%)" % (name, f"{c:,}", c / self.n_syn * 100))
        print("\n投射通路：")
        for s, d, r, desc in FEEDFORWARD:
            print("  [前馈] %-4s → %-4s  %s" % (s, d, desc))
        for s, d, r, desc in FEEDBACK:
            print("  [反馈] %-4s → %-4s  %s" % (s, d, desc))
        for s, d, r, desc in HIPPO_LINKS:
            print("  [海马] %-4s → %-4s  %s" % (s, d, desc))
        print("=" * 68)


def selftest():
    print("  小沙结构化皮层 selftest（脑区×层×细胞类型×规则投射）")
    ok = 0
    total = 0

    # 1) 构造
    total += 1
    cx = StructuredCortex(total=288964, seed=42)
    print("  [1] 构造: %s 神经元 / %s 突触" % (f"{cx.total:,}", f"{cx.n_syn:,}"))
    ok += int(cx.total == 288964 and cx.n_syn > 40_000_000)

    # 2) 脑区分配正确
    total += 1
    sums = [int(np.count_nonzero(cx.region_id == i)) for i in range(len(REGIONS))]
    print("  [2] 脑区分布: " + " ".join("%s=%s" % (cx.region_names[i], f"{sums[i]:,}") for i in range(len(REGIONS))))
    ok += int(sum(sums) == 288964 and all(s > 0 for s in sums))

    # 3) 细胞类型 80/20
    total += 1
    n_e = int(np.count_nonzero(cx.type_id == 0))
    ratio = n_e / cx.total
    print("  [3] E/I 比例: %.1f%% / %.1f%%（目标 80/20）" % (ratio * 100, (1 - ratio) * 100))
    ok += int(0.78 <= ratio <= 0.82)

    # 4) 抑制性连接为负权重
    total += 1
    neg = cx.syn_w[cx.syn_kind == 3]
    print("  [4] 抑制性突触 %s 条，全为负权重: %s" % (f"{neg.size:,}", bool(neg.size and (neg < 0).all())))
    ok += int(neg.size > 0 and (neg < 0).all())

    # 5) 区间投射存在（前馈/反馈/海马）
    total += 1
    inter = int(np.count_nonzero(cx.syn_kind == 7))
    print("  [5] 区间投射 %s 条（前馈+反馈+海马）" % f"{inter:,}")
    ok += int(inter > 80_000)

    # 6) 能跑（输入 → 放电）
    total += 1
    obs = np.zeros(10, dtype=np.float32); obs[0] = 0.9; obs[3] = 0.7
    fired = 0
    for _ in range(10):
        sp = cx.step(obs)
        fired += int(sp.size)
    print("  [6] 运行 10 步，累计发放 %s 次（平均 %.0f/步）" % (f"{fired:,}", fired / 10))
    ok += int(fired > 0)

    # 7) 结构因果性：V1 输入 → V1 先放电，IT/PFC 后放电
    total += 1
    cx2 = StructuredCortex(total=288964, seed=7)
    # 全通道强刺激（激活足够多的 V1 神经元，才能看到信号向 V2/IT/PFC/M1 级联）
    obs2 = np.ones(10, dtype=np.float32)
    first = {}
    for t in range(150):
        sp = cx2.step(obs2)
        if sp.size:
            for rid, rname in enumerate(cx2.region_names):
                if rname not in first and np.any(cx2.region_id[sp] == rid):
                    first[rname] = t
    order_ok = ("V1" in first and "IT" in first and first["V1"] <= first.get("IT", 999))
    print("  [7] 兴奋传播顺序（越早=越靠上游）: " +
          " ".join("%s@t%d" % (k, v) for k, v in sorted(first.items(), key=lambda x: x[1])))
    ok += int(order_ok)

    print("  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    if "--summary" in sys.argv:
        StructuredCortex(total=288964, seed=42).summary()
        sys.exit(0)
    sys.exit(0 if selftest() else 1)
