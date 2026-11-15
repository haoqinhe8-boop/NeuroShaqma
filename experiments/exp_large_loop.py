# -*- coding: utf-8 -*-
"""28.9 万神经元闭环：性能 + 行为 + STDP 学习效果（综合验证）。

跑法：python experiments/exp_large_loop.py

检查：
  1. 28.9 万规模能否跑起来（含 115 维四感 + 认知调制）
  2. 速度（步/秒，含/不含 STDP）
  3. 行为是否依赖皮层（消融输入 -> 行为消失）
  4. STDP 是否真的改变网络响应（学习前后同一刺激的模式差异）
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from xiaosha_embodied import EmbodiedXiaosha
from xiaosha_stdp import STDPLearner

N = 288964
SCENE = {"threat": 0.15, "goal": 0.7, "battery": 0.9, "n_items": 2}


def bench(em, steps=40, label=""):
    """测速度。"""
    for _ in range(5):
        em.step(SCENE)
    t0 = time.perf_counter()
    sp = []
    for _ in range(steps):
        a, i = em.step(SCENE)
        sp.append(i["n_spikes"])
    el = (time.perf_counter() - t0) / steps
    print("    %-18s %.1f ms/步 = %.1f 步/秒 | 发放 均%.0f (%.1f%%)"
          % (label, el * 1000, 1.0 / el, np.mean(sp), np.mean(sp) / N * 100))
    return el, np.mean(sp), a


def main():
    print("  28.9 万神经元闭环综合验证")
    np.random.seed(20261115)
    ok = 0
    total = 0

    # ---- 1. 构造 + 基础性能 ----
    print("\n【1】构造与速度")
    t0 = time.time()
    em = EmbodiedXiaosha(n_neurons=N, seed=42)
    build = time.time() - t0
    print("    构造耗时 %.1fs | 神经元 %d | 突触 %d"
          % (build, em.cortex.total, em.cortex.syn_w.size))
    total += 1
    ok += int(em.cortex.total == N and em.cortex.syn_w.size > 50_000_000)

    el, nsp, act = bench(em, steps=40, label="纯闭环")
    total += 1
    ok += int(el < 0.1 and nsp > 0)      # <100ms/步 且真的在放电
    print("    动作: %s" % np.round(act, 3))

    # ---- 2. 加 STDP 后的速度 ----
    print("\n【2】加 STDP 后的性能")
    lr = STDPLearner(em.cortex)
    for _ in range(5):
        em.step(SCENE)
        lr.update()
    t0 = time.perf_counter()
    ups = 0
    for _ in range(30):
        a, i = em.step(SCENE)
        r = lr.update()
        ups += r["touched"]
    el2 = (time.perf_counter() - t0) / 30
    print("    闭环 + STDP       %.1f ms/步 = %.1f 步/秒 | 突触更新 %d"
          % (el2 * 1000, 1.0 / el2, ups))
    total += 1
    ok += int(el2 < 0.2 and ups > 0)

    # ---- 3. 行为依赖皮层（消融） ----
    print("\n【3】行为依赖皮层（因果验证）")
    em3 = EmbodiedXiaosha(n_neurons=N, seed=7)
    normal, nsp_a = [], []
    for _ in range(20):
        a, i = em3.step(SCENE)
        normal.append(np.abs(a).sum())
        nsp_a.append(i["n_spikes"])
    base_act, abl_act, base_sp, abl_sp = em3.ablate_input(steps=20, scene=SCENE)
    print("    正常:   动作%.4f 发放%.0f" % (base_act, base_sp))
    print("    消融V1: 动作%.4f 发放%.0f" % (abl_act, abl_sp))
    total += 1
    ok += int(abl_act < base_act * 0.2 and abl_sp < base_sp * 0.2)

    # ---- 4. STDP 学习效果：同一刺激下的响应用变化 ----
    print("\n【4】STDP 学习效果")
    em4 = EmbodiedXiaosha(n_neurons=60000, seed=11)   # 6 万，跑得快
    lr4 = STDPLearner(em4.cortex)
    # 基线：同一刺激的发放模式
    def pattern(em_, steps=8):
        sets = []
        for _ in range(steps):
            em_.step(SCENE)
            sets.append(set(em_.cortex._last_spike_ids.tolist()))
        return sets
    base_sets = pattern(em4)
    w_before = float(em4.cortex.syn_w.mean())
    # 训练：反复给同一场景 + STDP
    for t in range(120):
        em4.step(SCENE)
        lr4.update()
    w_after = float(em4.cortex.syn_w.mean())
    after_sets = pattern(em4)
    # 训练前后的响应用差异（同一输入，看网络是否变了）
    diffs = []
    for s1, s2 in zip(base_sets, after_sets):
        if s1 or s2:
            diffs.append(1 - len(s1.intersection(s2)) / max(1, len(s1.union(s2))))
    drift = float(np.mean(diffs)) if diffs else 0.0
    st = lr4.stats()
    print("    权重 %.5f -> %.5f (%.2fx) | 稳态触发 %d 次"
          % (w_before, w_after, w_after / max(w_before, 1e-9), st["homeo_events"]))
    print("    同一刺激的响应漂移 %.3f（0=完全没学，1=完全变了）" % drift)
    print("    突触更新累计 %d" % st["total_updates"])
    total += 1
    ok += int(st["total_updates"] > 0 and drift > 0.05 and w_after / max(w_before, 1e-9) < 4.0)

    print("\n  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
