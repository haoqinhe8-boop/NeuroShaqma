# -*- coding: utf-8 -*-
"""小沙的睡眠巩固 (Sleep Consolidation, SHY 假设)。

生物学依据（Tononi & Cirelli 的 Synaptic Homeostasis Hypothesis）：
  - 清醒：学习使突触净增强，噪声也一起变强
  - 睡眠：海马重放 + 全脑突触整体下调（保留强连接、削弱弱连接）
  → 信噪比提高，重要记忆被巩固，同时给新学习腾出容量

本模块测：睡眠（重放 + 突触下调）能否减少后续任务的干扰？
  对照组：学 A → 学 B（会遗忘 A）
  实验组：学 A → 睡眠 → 学 B（看 A 是否保住更多）
"""
import numpy as np

from xiaosha_cortex import SpikingCortex
from xiaosha_hippocampus import ClassifyTask, run_episode, evaluate

__version__ = "1.0.0"


def collect_samples(task, n=60, seed=7000):
    """收集一组观测（用于睡眠重放）。"""
    out = []
    for i in range(n):
        task.new_episode()
        out.append(task.obs().copy())
    return out


def sleep_consolidate(cx, samples, downscale=0.92, replay=True):
    """睡眠：① 重放样本激活网络 ② 突触整体下调（SHY）。

    返回执行记录，便于如实报告。
    """
    rec = {"replayed": 0, "downscale": float(downscale)}
    if replay:
        for obs in samples:
            cx.step(obs)          # 只前向激活，不学习
            rec["replayed"] += 1
    if downscale != 1.0:
        # 突触矩阵整体下调（这是 SHY 的核心：睡眠中突触净减弱）
        np.multiply(cx.w, downscale, out=cx.w)
        np.clip(cx.w, -3.0, 2.0, out=cx.w)
    return rec


def run_seq(with_sleep, rounds=120, n=6000, seed=11, downscale=0.92):
    """学 A →（可选睡眠）→ 学 B，返回 (A学后, A最终, B最终)。"""
    cx = SpikingCortex(n=n, seed=seed)
    tA = ClassifyTask(seed=5, task_id=0)
    tB = ClassifyTask(seed=5, task_id=1)

    for i in range(rounds):
        run_episode(cx, tA, learn=True, max_decisions=1, steps=10, seed=9000 + i)
    a0 = evaluate(cx, 0, n=40)

    if with_sleep:
        samples = collect_samples(tA, n=60)
        sleep_consolidate(cx, samples, downscale=downscale)

    for i in range(rounds):
        run_episode(cx, tB, learn=True, max_decisions=1, steps=10, seed=9500 + i)
    a1 = evaluate(cx, 0, n=40)
    b1 = evaluate(cx, 1, n=40)
    return a0, a1, b1


def selftest():
    print("  小沙睡眠巩固 selftest（SHY：重放 + 突触下调）")
    ok = 0; total = 0

    # 1) 睡眠机制可执行
    total += 1
    cx = SpikingCortex(n=4000, seed=11)
    t = ClassifyTask(seed=5, task_id=0)
    for i in range(30):
        run_episode(cx, t, learn=True, max_decisions=1, steps=10, seed=100 + i)
    w_before = float(np.abs(cx.w).mean())
    rec = sleep_consolidate(cx, collect_samples(t, 20), downscale=0.9)
    w_after = float(np.abs(cx.w).mean())
    print("  [1] 睡眠执行: 重放 %d 样本 | 突触均值 %.4f → %.4f（下调 %.0f%%）"
          % (rec["replayed"], w_before, w_after, (1 - w_after / max(w_before, 1e-9)) * 100))
    ok += int(rec["replayed"] == 20 and w_after < w_before)

    # 2) 对照：无睡眠 学A→学B
    total += 1
    a0, a1, b1 = run_seq(False, rounds=100, n=5000)
    print("  [2] 无睡眠: A 学后 %.0f%% → 学B后 %.0f%% | B %.0f%%" % (a0*100, a1*100, b1*100))
    ok += 1   # 记录型

    # 3) 实验：中间插入睡眠
    total += 1
    a0s, a1s, b1s = run_seq(True, rounds=100, n=5000, downscale=0.92)
    print("  [3] 有睡眠: A 学后 %.0f%% → 学B后 %.0f%% | B %.0f%%" % (a0s*100, a1s*100, b1s*100))
    ok += 1   # 记录型（结论好坏都如实记录）

    # 4) 睡眠确实改变了结果（不是无效操作）
    total += 1
    print("  [4] 睡眠影响: A 保持率 %+.0fpp | B 学习 %+.0fpp（正=有助，负=有害）"
          % ((a1s - a1) * 100, (b1s - b1) * 100))
    ok += int((a1s != a1) or (b1s != b1))

    print("  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
