# -*- coding: utf-8 -*-
"""R-STDP —— 奖励调制的 STDP（三因素学习规则）。

普通 STDP 只会因为"一起发放"而增强，它不知道什么是好、什么是坏。
R-STDP 加上第三个因素：**多巴胺（奖励预测误差）**。

真实机制（Schultz 1997 多巴胺 RPE；Izhikevich 2007 三因素规则）：
  1. 突触因为 pre/post 的时序产生"资格痕迹"（eligibility trace）
     —— 相当于"这条边刚才活跃过，正在候选名单上"
  2. 多巴胺（DA = 奖励 - 预期）到达时，候选边的权重才真正改变
     —— DA > 0（比预期好）→ 增强；DA < 0（比预期差）→ 减弱
  3. 没有 DA 时痕迹自己衰减掉 —— 不奖励就不学

这解决了普通 STDP 的根本问题：它只会强化"经常发生的事"，不会强化"有用的事"。

实现用**神经元级**资格痕迹（n 大小，2.3MB），不是突触级（5600 万 × 4B = 224MB）。
奖励到来时按"最近的 pre 集合 × 它们的出边"更新，事件驱动。
"""
import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

__version__ = "1.0.0"


@njit(cache=True)
def _rstdp_apply_jit(syn_ptr, syn_dst, syn_w, pre_ids, n_pre,
                     post_elig, da, a_plus, a_minus, w_max, w_min):
    """奖励到来时，按资格痕迹更新：只动"刚才活跃 + 目标也活跃"的边。"""
    touched = 0
    for k in range(n_pre):
        i = pre_ids[k]
        start = syn_ptr[i]
        end = syn_ptr[i + 1]
        for e in range(start, end):
            j = syn_dst[e]
            pe = post_elig[j]
            if pe <= 0.0:
                continue
            w = syn_w[e]
            if da >= 0.0:
                w += a_plus * da * pe
            else:
                w += a_minus * da * pe        # da<0 -> 减小
            if w > w_max:
                w = w_max
            elif w < w_min:
                w = w_min
            syn_w[e] = w
            touched += 1
    return touched


class RewardSTDP:
    """奖励调制 STDP（三因素规则）。

    用法：
        rl = RewardSTDP(cortex)
        ... 每步 ...
        rl.step()                       # 记录资格痕迹
        rl.reward(+1.0)                 # 吃东西了 -> 学
        rl.reward(-1.0)                 # 撞墙了 -> 反向学
    """

    def __init__(self, cortex, a_plus=0.02, a_minus=0.02, tau_elig=8.0,
                 gamma=0.9, lr_value=0.1, w_max=1.2, dt=1.0):
        self.cortex = cortex
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_max = w_max
        # 关键：这个网络的抑制性突触是**负权重**（约 -0.4 ~ 0）。
        # 不能 clip 到 0 —— 那会把所有抑制性连接抹平，均值假性上升
        # （实测：负奖励后 12.3 万条边从 -0.4 跳到 0，均值反而涨了）。
        self.w_min = float(min(-0.5, float(cortex.syn_w.min())))
        self.gamma = gamma              # 期待奖励的折扣
        self.lr_value = lr_value        # 值函数学习率

        n = cortex.total
        self.pre_elig = np.zeros(n, dtype=np.float32)   # pre 资格痕迹
        self.post_elig = np.zeros(n, dtype=np.float32)  # post 资格痕迹
        self._decay = float(np.exp(-dt / tau_elig))

        # 奖励预测（Baseline / critic）
        self.expected = 0.0

        # 统计
        self.rewards = []
        self.da_history = []
        self.updates = 0

    # ---------- 每步：记录资格痕迹 ----------
    def step(self, sp=None):
        sp = self.cortex._last_spike_ids if sp is None else sp
        # 衰减
        self.pre_elig *= self._decay
        self.post_elig *= self._decay
        if sp.size:
            self.pre_elig[sp] = 1.0
            self.post_elig[sp] = 1.0

    # ---------- 奖励到来：算 RPE 并更新 ----------
    def reward(self, r, sp_recent=None):
        """r: 奖励值（+1 好 / -1 坏 / 0 中性）。

        返回 dict：da（奖励预测误差）、touched（更新了多少突触）。
        """
        r = float(r)
        # 奖励预测误差：比预期好多少
        da = r - self.expected
        # 更新预期（简单 TD(0) 风格）
        self.expected = self.expected + self.lr_value * (r - self.expected)

        touched = 0
        if abs(da) > 1e-6:
            # 用"当前还有资格的神经元"作为 pre 集合（痕迹 > 阈值）
            active = np.flatnonzero(self.pre_elig > 0.05).astype(np.int64)
            if active.size and _HAS_NUMBA:
                touched = int(_rstdp_apply_jit(
                    self.cortex.syn_ptr, self.cortex.syn_dst, self.cortex.syn_w,
                    active, active.size, self.post_elig,
                    float(da), self.a_plus, self.a_minus, self.w_max, self.w_min))
            elif active.size:
                touched = self._numpy_apply(active, da)
            self.updates += touched

        self.rewards.append(r)
        self.da_history.append(da)
        return {"reward": r, "da": float(da), "expected": float(self.expected),
                "touched": touched}

    def _numpy_apply(self, active, da):
        touched = 0
        for i in active:
            a, b = self.cortex.syn_ptr[i], self.cortex.syn_ptr[i + 1]
            if b <= a:
                continue
            tgt = self.cortex.syn_dst[a:b]
            pe = self.post_elig[tgt]
            mask = pe > 0
            if not mask.any():
                continue
            w = self.cortex.syn_w[a:b]
            delta = (self.a_plus if da >= 0 else self.a_minus) * da * pe
            w[mask] += delta[mask]
            np.clip(w, self.w_min, self.w_max, out=w)
            touched += int(mask.sum())
        return touched

    # ---------- 稳态（和 STDP 共用同一套思路）----------
    def homeostatic(self, target_mean=None, tol=0.2, gain=0.06):
        w = self.cortex.syn_w
        sample = w[:200000] if w.size > 200000 else w
        cur = float(sample.mean())
        if target_mean is None:
            target_mean = getattr(self, "_target", None)
            if target_mean is None:
                self._target = cur
                return 0.0
        dev = (cur - target_mean) / max(target_mean, 1e-9)
        if abs(dev) <= tol:
            return 0.0
        scale = 1.0 - np.sign(dev) * gain
        w *= scale
        return scale

    def stats(self):
        w = self.cortex.syn_w
        sample = w[:200000] if w.size > 200000 else w
        return {
            "reward_events": len(self.rewards),
            "mean_reward": float(np.mean(self.rewards)) if self.rewards else 0.0,
            "last_da": float(self.da_history[-1]) if self.da_history else 0.0,
            "expected": float(self.expected),
            "updates": self.updates,
            "w_mean_sampled": float(sample.mean()),
            "elig_pre": float((self.pre_elig > 0.05).sum()),
        }


# ==================== 自测 ====================

def selftest():
    print("  小沙 R-STDP 奖励学习 selftest")
    np.random.seed(20261115)
    ok = 0
    total = 0

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from xiaosha_structured import StructuredCortex

    # ---- 1. 奖励预测误差：第一次奖励 DA 高，重复后降低（习惯化）----
    total += 1
    cx = StructuredCortex(total=20000, seed=5)
    rl = RewardSTDP(cx, a_plus=0.05)
    das = []
    for k in range(8):        # 8 次才降到 <0.5（0.9^8=0.43）
        rl.step()
        out = rl.reward(+1.0)
        das.append(out["da"])
    print("  [1] DA 序列（首高后低=习惯化）: %s" % np.round(das, 3))
    ok += int(das[0] > 0.8 and das[-1] < das[0] * 0.5)

    # ---- 2. 负奖励 → 负 DA → 权重下降 ----
    total += 1
    cx2 = StructuredCortex(total=20000, seed=7)
    rl2 = RewardSTDP(cx2, a_minus=0.05)
    # 先造出资格痕迹
    v1 = cx2.groups[(0, 0, 0)]
    for _ in range(5):
        cx2.v[v1] += 4.0
        cx2.step(None)
        rl2.step()
    w0 = float(cx2.syn_w[:200000].mean())
    out2 = rl2.reward(-1.0)
    w1 = float(cx2.syn_w[:200000].mean())
    print("  [2] 负奖励: DA=%+.3f | 权重 %.6f -> %.6f (delta %+.2e)"
          % (out2["da"], w0, w1, w1 - w0))
    ok += int(out2["da"] < -0.3 and w1 < w0)

    # ---- 3. 正奖励 → 权重上升 ----
    total += 1
    cx3 = StructuredCortex(total=20000, seed=9)
    rl3 = RewardSTDP(cx3, a_plus=0.05)
    for _ in range(5):
        cx3.v[cx3.groups[(0, 0, 0)]] += 4.0
        cx3.step(None)
        rl3.step()
    w0 = float(cx3.syn_w[:200000].mean())
    out3 = rl3.reward(+1.0)
    w1 = float(cx3.syn_w[:200000].mean())
    print("  [3] 正奖励: DA=%+.3f | 权重 %.6f -> %.6f (delta %+.2e)"
          % (out3["da"], w0, w1, w1 - w0))
    ok += int(out3["da"] > 0.3 and w1 > w0)

    # ---- 4. 奖励=0 时不学（没有 DA 就不动）----
    total += 1
    cx4 = StructuredCortex(total=20000, seed=11)
    rl4 = RewardSTDP(cx4, a_plus=0.05)
    rl4.expected = 0.0
    for _ in range(5):
        cx4.v[cx4.groups[(0, 0, 0)]] += 4.0
        cx4.step(None)
        rl4.step()
    w0 = float(cx4.syn_w[:200000].mean())
    out4 = rl4.reward(0.0)         # 中性：da = 0 - 0 = 0
    w1 = float(cx4.syn_w[:200000].mean())
    print("  [4] 中性奖励: DA=%+.3f | 权重变化 %+.2e（应≈0）" % (out4["da"], w1 - w0))
    ok += int(abs(out4["da"]) < 1e-6 and abs(w1 - w0) < 1e-12)

    # ---- 5. 大规模可用（28.9 万）----
    total += 1
    import time
    t0 = time.time()
    cx5 = StructuredCortex(total=288964, seed=42)
    rl5 = RewardSTDP(cx5)
    v1 = cx5.groups[(0, 0, 0)]
    ts = []
    for k in range(12):
        cx5.v[v1] += 2.0
        t1 = time.perf_counter()
        cx5.step(None)
        rl5.step()
        if k % 3 == 2:
            rl5.reward(+1.0 if k < 6 else -0.5)
        ts.append((time.perf_counter() - t1) * 1000)
    st = rl5.stats()
    print("  [5] 28.9万: %.1f ms/步（含R-STDP）| 奖励事件 %d | 更新 %d 次 | 资格神经元 %d"
          % (np.mean(ts), st["reward_events"], st["updates"], st["elig_pre"]))
    cond = (st["reward_events"] == 4, st["updates"] > 0, np.mean(ts) < 300)
    print("      判定: 事件4=%s 更新>0=%s 速度<300ms=%s" % cond)
    ok += int(all(cond))

    print("  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
