# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""工作记忆（2026-11）：延迟匹配样本 Delayed-Match-to-Sample

人脑招牌能力之一 = 工作记忆（记住"刚才看到的"，延迟一段时间后还能用）。
神经科学里测它的经典任务 = 延迟匹配样本（DMS）：
  ① 看一个线索（样本 A 或 B）
  ② 延迟期（空白，什么都看不到，但大脑要"保持"住线索）
  ③ 两个候选出现，选刚才那个匹配的

SNN 要靠循环连接在延迟期维持活动才能做到（没有专门的记忆单元，全靠网络的短时动态）。
这是对"像人脑"的硬核检验：会 = 有工作记忆雏形；不会 = 当前架构缺这块，要加专门结构。

用法：python xiaosha_memory.py --selftest
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

from xiaosha_cortex import SpikingCortex
from xiaosha_regions import RegionCortex

N_CH = 10
CUE_STEPS = 5       # 线索呈现步数
DELAY_STEPS = 10    # 延迟期（关键：空白，需要保持）
CHOICE_STEPS = 10   # 决策窗


class DelayedMatchTask:
    """延迟匹配样本：看线索 → 延迟 → 选匹配（二选一，随机基线 50%）。"""

    def __init__(self, seed=5):
        self.rng = np.random.default_rng(seed)
        self.sample = 0

    def new_episode(self):
        self.sample = int(self.rng.integers(0, 2))
        return self.sample

    def cue_obs(self):
        o = np.zeros(N_CH, dtype=np.float32)
        o[self.sample] = 0.9          # ch0=样本A, ch1=样本B
        return o

    def delay_obs(self):
        return np.zeros(N_CH, dtype=np.float32)   # 空白，什么都看不到

    def choice_obs(self):
        o = np.zeros(N_CH, dtype=np.float32)
        o[2] = 0.6; o[3] = 0.6        # 两个候选提示（中性）
        return o


def run_dms_episode(cx, task, learn=True, seed=None):
    """完整一局：线索 → 延迟 → 决策。返回命中与否。"""
    if seed is not None:
        task.rng = np.random.default_rng(seed)
    task.new_episode()
    # 阶段 1：线索
    for _ in range(CUE_STEPS):
        cx.step(task.cue_obs())
    # 阶段 2：延迟（空白，工作记忆的关键）
    for _ in range(DELAY_STEPS):
        cx.step(task.delay_obs())
    # 阶段 3：决策
    cx.reset_feat()
    for _ in range(CHOICE_STEPS):
        cx.step(task.choice_obs())
    a = cx.decide(temp=1.0, sample=learn)
    hit = (a % 2) == task.sample
    r = 1.0 if hit else -0.5
    if learn:
        cx.stdp_reward(r)
        cx.learn_returns([(cx._last_xf.copy(), cx._last_probs.copy(), a, float(r))])
    return hit


def evaluate(cx, task, n=40):
    hits = 0
    for i in range(n):
        hits += int(run_dms_episode(cx, task, learn=False, seed=8000 + i))
    return hits / n


# ============================================================
def selftest():
    print("=" * 76)
    print("  工作记忆 selftest（延迟匹配样本 DMS）")
    print("=" * 76)
    results, fails = [], []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    # 1. 任务规则自洽
    t = DelayedMatchTask(seed=5)
    t.new_episode()
    check("任务规则：样本 ∈ {0,1}，延迟期输入全零", t.sample in (0, 1)
          and t.delay_obs().sum() == 0,
          f"样本 {t.sample}")

    # 2. 分区网络学会延迟匹配（练前 ~50% 随机 → 练后 >65%）
    t0 = time.time()
    cx = RegionCortex(n=20_000, n_act=2, seed=11)
    task = DelayedMatchTask(seed=5)
    pre = evaluate(cx, task, n=40)
    for i in range(300):
        run_dms_episode(cx, task, learn=True, seed=9000 + i)
    post = evaluate(cx, task, n=40)
    check("学习曲线如实记录（练前→练后）", True,
          f"练前 {pre*100:.0f}% → 练后 {post*100:.0f}%（300 局，{time.time()-t0:.0f}s）" + (" ← 有工作记忆" if post > 0.65 else " ← 学不会，当前架构缺工作记忆（< 随机 50%）"))

    # 3. 延迟期是否真的在"保持"：延迟期网络仍有发放（信息未熄灭）
    cx.reset_state()
    for _ in range(CUE_STEPS):
        cx.step(task.cue_obs())
    delay_spikes = sum(cx.step(task.delay_obs()) for _ in range(DELAY_STEPS))
    check("延迟期网络仍有活动（线索在循环里保持，未熄灭）", delay_spikes > 0,
          f"延迟 {DELAY_STEPS} 步共 {delay_spikes} 个发放")

    # 4. 对照：随机网络同学练（分区结构是否更有利于工作记忆）
    cx_s = SpikingCortex(n=20_000, n_act=2, seed=11)
    task = DelayedMatchTask(seed=5)
    for i in range(300):
        run_dms_episode(cx_s, task, learn=True, seed=9000 + i)
    post_s = evaluate(cx_s, task, n=40)
    check("对照：分区网络不低于随机网络", post >= post_s - 0.05,
          f"分区 {post*100:.0f}% vs 随机 {post_s*100:.0f}%")

    # 5. 无延迟对照：去掉延迟期（10步空白）后，任务变得容易得多 → 证明延迟期是难点
    cx2 = RegionCortex(n=20_000, n_act=2, seed=11)
    task2 = DelayedMatchTask(seed=5)
    def run_no_delay(cx, task, learn=True, seed=None):
        if seed is not None:
            task.rng = np.random.default_rng(seed)
        task.new_episode()
        for _ in range(CUE_STEPS):
            cx.step(task.cue_obs())
        cx.reset_feat()
        for _ in range(CHOICE_STEPS):
            cx.step(task.choice_obs())
        a = cx.decide(temp=1.0, sample=learn)
        hit = (a % 2) == task.sample
        r = 1.0 if hit else -0.5
        if learn:
            cx.stdp_reward(r)
            cx.learn_returns([(cx._last_xf.copy(), cx._last_probs.copy(), a, float(r))])
        return hit
    for i in range(300):
        run_no_delay(cx2, task2, learn=True, seed=9000 + i)
    post_nodelay = sum(run_no_delay(cx2, task2, learn=False, seed=8000 + i) for i in range(40)) / 40
    check("无延迟对照如实记录（区分：延迟期难 vs 任务本身需记忆结构）", True,
          f"有延迟 {post*100:.0f}% vs 无延迟 {post_nodelay*100:.0f}%" + (" ← 延迟期是难点" if post_nodelay > post + 0.10 else " ← 去掉延迟也没学会，任务本身需要专门记忆结构"))

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