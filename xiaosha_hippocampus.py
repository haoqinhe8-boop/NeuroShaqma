# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""海马体记忆重放（2026-11）：攻多任务的灾难性遗忘。

背景：小沙皮层交替训练两个任务会灾难性遗忘（分类单独 100% → 交替 25%，
见 xiaosha_multitask.py）。人脑靠海马体在休息时"重放"记忆、巩固到皮层来防止遗忘。

这里实现海马体的最小版 = 经验重放缓冲（experience replay）：
训练新任务时，穿插重放旧任务的样本，让旧记忆不被覆盖。

两个分类任务 A/B（同一批线索、不同映射，必然竞争读出权重）：
  任务 A：ch0-3 四类 → 动作 = cls（ch9 标记 0.0）
  任务 B：ch0-3 四类 → 动作 = (cls+2)%4（ch9 标记 0.9，映射不同）

验证：无重放交替 → 任务 A 被遗忘；有重放交替 → 任务 A 保持。重放 = 海马体。

用法：python xiaosha_hippocampus.py --selftest
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

from xiaosha_cortex import SpikingCortex, run_episode
from xiaosha_regions import RegionCortex

N_CH = 10
TAG = 9


class ClassifyTask:
    """分类任务：4 类 → 4 动作。task_id=0 → 动作=cls；task_id=1 → 动作=(cls+2)%4。"""
    def __init__(self, seed=5, task_id=0):
        self.rng = np.random.default_rng(seed)
        self.task_id = task_id
        self.cls = 0
        self.target_tol = 0.1

    def new_episode(self):
        self.cls = int(self.rng.integers(0, 4))
        return self.obs()

    def target(self):
        return self.cls if self.task_id == 0 else (self.cls + 2) % 4

    def obs(self):
        o = np.zeros(N_CH, dtype=np.float32)
        o[self.cls] = 0.9
        o[TAG] = 0.9 if self.task_id == 1 else 0.0
        return o

    def act(self, a):
        if int(a) == self.target():
            return 1.0, True, {"dist": 0.0}
        return -0.5, True, {"dist": 9.9}


def evaluate(cx, task_id, n=40):
    t = ClassifyTask(seed=5, task_id=task_id)
    hits = 0
    for i in range(n):
        r = run_episode(cx, t, learn=False, max_decisions=1, steps=10, seed=5000 + i)
        hits += int(r["hit"])
    return hits / n


def train_no_replay(cx, rounds=300):
    tA = ClassifyTask(seed=5, task_id=0)
    tB = ClassifyTask(seed=5, task_id=1)
    for i in range(rounds):
        t = tA if i % 2 == 0 else tB
        run_episode(cx, t, learn=True, max_decisions=1, steps=10, seed=9000 + i)


def train_with_replay(cx, rounds=300, replay_n=3, cap=600):
    tA = ClassifyTask(seed=5, task_id=0)
    tB = ClassifyTask(seed=5, task_id=1)
    buf = {0: [], 1: []}
    rng = np.random.default_rng(42)
    for i in range(rounds):
        tid = i % 2
        t = tA if tid == 0 else tB
        run_episode(cx, t, learn=True, max_decisions=1, steps=10, seed=9000 + i)
        buf[tid].append((t.obs().copy(), t.target()))
        if len(buf[tid]) > cap:
            buf[tid].pop(0)
        # 海马体重放：重放另一个任务的样本，防止遗忘
        other = 1 - tid
        if buf[other]:
            for _ in range(replay_n):
                obs, act = buf[other][int(rng.integers(0, len(buf[other])))]
                cx.reset_feat()
                for _ in range(10):
                    cx.step(obs)
                a = cx.decide(temp=1.0, sample=True)
                r = 1.0 if (a % 4) == act else -0.5
                cx.stdp_reward(r)
                cx.learn_returns([(cx._last_xf.copy(), cx._last_probs.copy(), a, float(r))])


def selftest():
    print("=" * 76)
    print("  海马体记忆重放 selftest（攻灾难性遗忘）")
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
    tA = ClassifyTask(seed=5, task_id=0)
    tB = ClassifyTask(seed=5, task_id=1)
    tA.new_episode(); tB.new_episode()
    same_cls = (tA.cls == tB.cls)
    diff_target = tA.target() != tB.target()
    check("任务规则：A/B 同线索但映射不同（必然竞争）+ 标记通道区分",
          tA.obs()[TAG] == 0.0 and tB.obs()[TAG] == 0.9 and diff_target,
          f"A:{tA.cls}→{tA.target()} B:{tB.cls}→{tB.target()}")

    # 2. 无重放交替 → 任务 A 遗忘
    t0 = time.time()
    cx1 = RegionCortex(n=20_000, seed=11)
    train_no_replay(cx1, rounds=300)
    noA = evaluate(cx1, 0); noB = evaluate(cx1, 1)
    check("无重放：交替训练后任务 A 被遗忘（<60%）", noA < 0.60,
          f"任务A {noA*100:.0f}% · 任务B {noB*100:.0f}%（{time.time()-t0:.0f}s）")

    # 3. 有重放交替 → 任务 A 保持（海马体重放有效）
    cx2 = RegionCortex(n=20_000, seed=11)
    train_with_replay(cx2, rounds=300)
    repA = evaluate(cx2, 0); repB = evaluate(cx2, 1)
    check("有重放：任务 A 保持（>60%，海马体防遗忘）", repA > 0.60,
          f"任务A {repA*100:.0f}% · 任务B {repB*100:.0f}%")

    # 4. 重放显著优于无重放
    check("重放效果：有重放的任务A 显著高于无重放", repA > noA + 0.10,
          f"无重放 {noA*100:.0f}% → 有重放 {repA*100:.0f}%")

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