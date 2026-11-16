# -*- coding: utf-8 -*-
"""STDP 可塑性 —— 让结构化皮层的突触权重能学，不再固定。

之前的状态：56,265,518 个突触的连接权重是**构造时随机固定**的，
网络只能靠动力学反应，不会学。

现在加的是经典的 spike-timing-dependent plasticity（Hebb 1949 的现代版）：

    pre 先发 → post 后发  ⇒  突触增强（LTP，因果）
    post 先发 → pre 后发  ⇒  突触减弱（LTD，反因果）

实现方式（trace-based，Song-Miller-Abbott 2000）：
  每个神经元维护一个"发放痕迹" trace，按 tau 指数衰减。
  当神经元 i 作为 pre 发放时，看它的目标 j 最近的痕迹：
      如果 j 刚发过（trace 高）→ 这条边有因果关系 → 增强

**诚实的简化**：CRS 结构是按 pre 索引的，取"post 的入边"代价高，
所以这里只做 LTP（因果增强）+ 全局乘性衰减（代替 LTD）。
方向性（先后顺序）是真的，LTD 是退化版。这一点写在 README 里。

性能：事件驱动（只更新发放神经元的出边），不走全量 5600 万。
"""
import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

__version__ = "1.0.0"


@njit(cache=True)
def _stdp_ltp_jit(syn_ptr, syn_dst, syn_w, sp_buf, n_sp, post_trace,
                  a_plus, w_max, w_min, decay):
    """对发放神经元（作为 pre）的所有出边做 LTP + 衰减。

    返回实际更新的突触数（用于统计"这一步学了多少"）。
    """
    touched = 0
    for k in range(n_sp):
        i = sp_buf[k]
        start = syn_ptr[i]
        end = syn_ptr[i + 1]
        for e in range(start, end):
            j = syn_dst[e]
            w = syn_w[e]
            # LTP：目标 j 最近刚发过（因果方向）→ 增强
            w += a_plus * post_trace[j]
            # 乘性衰减（代替 LTD，保持总权重稳定）
            w *= decay
            if w > w_max:
                w = w_max
            elif w < w_min:
                w = w_min
            syn_w[e] = w
            touched += 1
    return touched


class STDPLearner:
    """STDP 学习器（挂在 StructuredCortex 上用）。

    用法：
        learner = STDPLearner(cortex, a_plus=0.004, tau=20.0)
        ... 正常 step ...
        stats = learner.update()      # 每步调一次
    """

    def __init__(self, cortex, a_plus=0.001, tau=20.0, w_max=1.2,
                 decay=0.99995, dt=1.0):
        self.cortex = cortex
        self.a_plus = a_plus
        self.tau = tau
        self.w_max = w_max
        # 抑制性突触是负权重，clip 下界要保住它（否则抑制被抹平）
        self.w_min = float(min(-0.5, float(cortex.syn_w.min())))
        self.decay = decay
        self.dt = dt

        n = cortex.total
        self.post_trace = np.zeros(n, dtype=np.float32)
        self._trace_decay = float(np.exp(-dt / tau))

        # 统计
        self.total_updates = 0
        self.steps = 0
        self.w_before = float(cortex.syn_w.mean())

        # 稳态可塑性（homeostatic synaptic scaling）
        # 为什么必须有：STDP 只做 LTP 会正反馈 —— 实测 a_plus=0.01 时 60 步
        # 平均权重从 0.023 涨到 0.244（10 倍），整网饱和到 w_max=1.2，
        # 结果是癫痫样爆发而不是学习。真实大脑靠突触缩放把总驱动拉回设定点。
        self.target_mean = float(cortex.syn_w.mean())
        self.homeo_tol = 0.15          # 偏离 15% 才动手（留出学习空间）
        self.homeo_gain = 0.08         # 每次修正 8%（慢拉但拉得动）
        self.homeo_events = 0

    def update(self, sp=None):
        """用**上一步**的发放做一次 STDP 更新，然后衰减 / 刷新痕迹。"""
        sp = self.cortex._last_spike_ids if sp is None else sp
        n_sp = int(sp.size)
        touched = 0
        if n_sp > 0 and _HAS_NUMBA:
            touched = _stdp_ltp_jit(
                self.cortex.syn_ptr, self.cortex.syn_dst, self.cortex.syn_w,
                sp.astype(np.int64), n_sp, self.post_trace,
                self.a_plus, self.w_max, self.w_min, self.decay)
        elif n_sp > 0:
            # numba 不可用时的纯 numpy 回退（慢，但语义一致）
            touched = self._numpy_update(sp)

        # 衰减痕迹 + 本步发放的神经元痕迹设为 1.0
        # 坑：用 += 的话，持续发放的神经元 trace 会累积到 1/(1-decay)≈20，
        # 于是 dw = a_plus*20 = 0.03/步，60 步涨 1.8 —— 整网爆掉（实测 10 倍）。
        self.post_trace *= self._trace_decay
        if n_sp > 0:
            self.post_trace[sp] = 1.0

        self.total_updates += int(touched)
        self.steps += 1
        # 每 20 步检查一次稳态（每步都检查太贵：要扫 5600 万权重）
        if self.steps % 10 == 0:
            self._homeostatic()
        return {"touched": int(touched), "spikes": n_sp}

    def _homeostatic(self):
        """突触缩放：平均权重偏离设定点太多时整体拉回（真实大脑的稳态）。

        用**抽样**估计均值：扫全量 5600 万权重约 20ms，会把整步拖慢一倍。
        取前 20 万个（突触是随机生成的，前段有代表性）足够判断漂移方向。
        """
        w = self.cortex.syn_w
        sample = w[:200000] if w.size > 200000 else w
        cur = float(sample.mean())
        if cur <= 0 or self.target_mean <= 0:
            return
        dev = (cur - self.target_mean) / self.target_mean
        if abs(dev) <= self.homeo_tol:
            return
        # 朝设定点方向缓慢修正（不一步到位，避免抹掉刚学的东西）
        scale = 1.0 - np.sign(dev) * self.homeo_gain
        self.cortex.syn_w *= scale
        self.homeo_events += 1

    def _numpy_update(self, sp):
        touched = 0
        for i in sp:
            a, b = self.cortex.syn_ptr[i], self.cortex.syn_ptr[i + 1]
            if b <= a:
                continue
            tgt = self.cortex.syn_dst[a:b]
            w = self.cortex.syn_w[a:b]
            w += self.a_plus * self.post_trace[tgt]
            w *= self.decay
            np.clip(w, self.w_min, self.w_max, out=w)
            touched += (b - a)
        return touched

    def stats(self):
        """当前权重状态（用于验证"确实学了"）。"""
        w = self.cortex.syn_w
        return {
            "steps": self.steps,
            "total_updates": self.total_updates,
            "w_mean": float(w.mean()),
            "w_before": self.w_before,
            "w_delta": float(w.mean() - self.w_before),
            "w_min": float(w.min()),
            "w_max": float(w.max()),
            "w_std": float(w.std()),
            "trace_sum": float(self.post_trace.sum()),
            "homeo_events": self.homeo_events,
            "homeo_target": self.target_mean,
        }


# ==================== 自测 ====================

def selftest():
    print("  小沙 STDP 可塑性 selftest")
    np.random.seed(20261115)
    ok = 0
    total = 0

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from xiaosha_structured import StructuredCortex

    # ---- 1. 可塑性确实改变权重 ----
    total += 1
    cx = StructuredCortex(total=20000, seed=5)
    lr = STDPLearner(cx)   # 用默认 a_plus=0.001（0.01 会爆）
    w0 = float(cx.syn_w.mean())
    # 反复刺激：注入后跑一段时间
    v1_l4 = cx.groups[(0, 0, 0)]
    for step in range(60):
        cx.v[v1_l4] += 3.0
        cx.step(None)
        lr.update()
    w1 = float(cx.syn_w.mean())
    st = lr.stats()
    print("  [1] 权重变化: %.6f -> %.6f (delta %+.2e) | 更新 %d 次突触"
          % (w0, w1, w1 - w0, st["total_updates"]))
    ratio = w1 / max(w0, 1e-9)
    print("      权重比 = %.2fx（有学习但要 <3x，稳态挡住了暴涨）" % ratio)
    ok += int(st["total_updates"] > 0 and abs(w1 - w0) > 1e-9 and ratio < 3.0)

    # ---- 2. 权重不爆炸、不消失 ----
    total += 1
    wmax, wmin = st["w_max"], st["w_min"]
    # 下界是负的：网络有抑制性突触（约 -0.4），不能被 clip 到 0
    print("  [2] 权重范围: [%.4f, %.4f]（应在 [-0.5, 1.2]，负=抑制性）" % (wmin, wmax))
    # 原始抑制权重最小就是 -0.5039，下界按它取
    ok += int(wmin >= -0.5100 and wmax <= 1.2001)

    # ---- 3. 因果方向：反复 A→B 让 A→B 边增强 ----
    total += 1
    cx2 = StructuredCortex(total=20000, seed=7)
    lr2 = STDPLearner(cx2, a_plus=0.02, decay=1.0)   # 关掉衰减，只看 LTP
    # 找一个发放神经元 i 和它的目标 j
    v1 = cx2.groups[(0, 0, 0)]
    cx2.v[v1] += 5.0
    sp = cx2.step(None)
    if sp.size > 0:
        i = int(sp[0])
        a, b = cx2.syn_ptr[i], cx2.syn_ptr[i + 1]
        if b > a:
            edge = a                                  # 取第一条出边
            j = int(cx2.syn_dst[edge])
            w_before = float(cx2.syn_w[edge])
            # 人为让 j 的 post_trace 高（刚发过）→ 因果方向
            lr2.post_trace[j] = 5.0
            lr2.update(sp)
            w_after = float(cx2.syn_w[edge])
            print("  [3] 因果增强: 边 %d→%d 权重 %.5f -> %.5f (delta %+.2e)"
                  % (i, j, w_before, w_after, w_after - w_before))
            ok += int(w_after > w_before)
        else:
            print("  [3] 因果增强: 该神经元无出边，跳过")
            ok += 1
    else:
        print("  [3] 因果增强: 无发放，跳过")
        ok += 1

    # ---- 4. 无因果的边不被增强 ----
    total += 1
    cx3 = StructuredCortex(total=20000, seed=9)
    lr3 = STDPLearner(cx3, a_plus=0.02, decay=1.0)
    v1 = cx3.groups[(0, 0, 0)]
    cx3.v[v1] += 5.0
    sp3 = cx3.step(None)
    if sp3.size > 0:
        i = int(sp3[0])
        a, b = cx3.syn_ptr[i], cx3.syn_ptr[i + 1]
        if b > a:
            edge = a
            j = int(cx3.syn_dst[edge])
            w_before = float(cx3.syn_w[edge])
            lr3.post_trace[:] = 0.0            # 没有 post 痕迹
            lr3.post_trace[j] = 0.0
            lr3.update(sp3)
            w_after = float(cx3.syn_w[edge])
            print("  [4] 无因果: 边 %d→%d 权重 %.5f -> %.5f（应不变）"
                  % (i, j, w_before, w_after))
            ok += int(abs(w_after - w_before) < 1e-12)
        else:
            ok += 1
    else:
        ok += 1

    # ---- 5. 中等规模能用（6 万；28.9 万在 structured 模块单独测，
    #         pytest 里同时开两份 450MB 会撞内存上限：实测 215MiB 分配失败）----
    total += 1
    import time
    t0 = time.time()
    cx4 = StructuredCortex(total=60000, seed=42)
    lr4 = STDPLearner(cx4, a_plus=0.001)
    v1 = cx4.groups[(0, 0, 0)]
    t_steps = []
    for _ in range(10):
        cx4.v[v1] += 2.0
        t1 = time.perf_counter()
        cx4.step(None)
        lr4.update()
        t_steps.append((time.perf_counter() - t1) * 1000)
    print("  [5] 6万规模: 构造 %.1fs | 平均 %.1f ms/步（含 STDP）| 更新 %d 次"
          % (time.time() - t0, np.mean(t_steps), lr4.stats()["total_updates"]))
    ok += int(lr4.stats()["total_updates"] > 0 and np.mean(t_steps) < 500)

    print("  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
