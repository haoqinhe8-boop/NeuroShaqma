# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""多任务库（2026-11）：小沙大脑不只练导航。

之前皮层主要练导航（DecisionTask）。这里证明同一个 SNN 架构是**通用学习器**，
能学会三类性质完全不同的任务：
  ① 导航 Navigation（连续空间决策，已有 DecisionTask）
  ② 分类 Classify（离散 4 类识别）
  ③ 多臂老虎机 Bandit（强化学习：4 个臂概率不同，学会选最优臂）

三类任务性质不同：导航=连续空间+延迟奖励、分类=监督式识别、老虎机=探索-利用权衡。
同一个 SpikingCortex 架构（不同实例）分别学会它们，命中率都显著高于随机。

用法：python xiaosha_tasks.py --selftest
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

from xiaosha_cortex import SpikingCortex, DecisionTask, run_episode

N_CH = 10


# ② 分类任务：4 类线索 → 4 动作（随机基线 25%）
class ClassifyTask:
    def __init__(self, seed=5):
        self.rng = np.random.default_rng(seed)
        self.cls = 0
        self.target_tol = 0.1

    def new_episode(self):
        self.cls = int(self.rng.integers(0, 4))
        return self.obs()

    def obs(self):
        o = np.zeros(N_CH, dtype=np.float32)
        o[self.cls] = 0.9
        return o

    def act(self, a):
        if int(a) == self.cls:
            return 1.0, True, {"dist": 0.0}
        return -0.5, True, {"dist": 9.9}


# ③ 多臂老虎机：4 臂概率不同，学会选最优臂（随机基线 25%，最优臂 80%）
class BanditTask:
    PROBS = (0.80, 0.45, 0.25, 0.10)   # 4 个臂的奖励概率

    def __init__(self, seed=5):
        self.rng = np.random.default_rng(seed)
        self.target_tol = 0.1

    def new_episode(self):
        return self.obs()

    def obs(self):
        return np.zeros(N_CH, dtype=np.float32)   # 老虎机无观测，纯探索-利用

    def act(self, a):
        best = int(np.argmax(self.PROBS))
        if self.rng.random() < self.PROBS[int(a) % 4]:
            return 1.0, True, {"dist": 0.0 if int(a) == best else 9.9}
        return -0.2, True, {"dist": 9.9}

    @property
    def best_arm(self):
        return int(np.argmax(self.PROBS))


def evaluate(cx, task, n=60, max_decisions=1, steps=10):
    hits = 0
    for i in range(n):
        r = run_episode(cx, task, learn=False, max_decisions=max_decisions, steps=steps, seed=7000 + i)
        hits += int(r["hit"])
    return hits / n


# ============================================================
def selftest():
    print("=" * 76)
    print("  多任务库 selftest（同一架构学导航/分类/老虎机三类任务）")
    print("=" * 76)
    results, fails = [], []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    # ① 导航（对照，已知能学会）
    t0 = time.time()
    cx_nav = SpikingCortex(n=20_000, seed=11)
    nav = DecisionTask(seed=5)
    for i in range(120):
        run_episode(cx_nav, nav, learn=True, seed=8000 + i)
    nav_hit = evaluate(cx_nav, nav, n=30, max_decisions=24, steps=10)
    check("导航任务（对照）学会", nav_hit > 0.50, f"命中率 {nav_hit*100:.0f}%（随机 ~25%）")

    # ② 分类任务
    cx_cls = SpikingCortex(n=20_000, seed=11)
    cls = ClassifyTask(seed=5)
    pre = evaluate(cx_cls, cls, n=30)
    for i in range(300):
        run_episode(cx_cls, cls, learn=True, seed=8000 + i)
    post = evaluate(cx_cls, cls, n=60)
    check("分类任务学会（4类识别）", post > 0.60, f"练前 {pre*100:.0f}% → 练后 {post*100:.0f}%")

    # ③ 多臂老虎机（探索-利用）
    cx_band = SpikingCortex(n=20_000, seed=11)
    band = BanditTask(seed=5)
    for i in range(400):
        run_episode(cx_band, band, learn=True, seed=8000 + i)
    band_hit = evaluate(cx_band, band, n=80)
    check("多臂老虎机学会（选中最优臂，探索-利用）",
          band_hit > 0.40,   # 最优臂 80%、随机 25%，学会后应明显 >25%
          f"选中最优臂率 {band_hit*100:.0f}%（最优臂本身 80%，随机 25%）")

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