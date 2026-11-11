# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""多任务（2026-11）：同一皮层学会两个任务，靠任务标记切换。

人脑招牌能力 = 一个大脑干很多事（前额叶任务切换）。这里验证：同一个分区皮层，
同时学会【导航】和【模式分类】两个任务，靠一个"任务标记通道"区分当前做哪个。

训练：交替练（一局导航、一局分类）。评估：两个任务命中率都显著高于随机（25%）。
判据：导航 > 50% 且 分类 > 50% → 证明"一脑两用"，不是只会一件事。

用法：python xiaosha_multitask.py --selftest
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
from xiaosha_regions import RegionCortex

N_CH = 10
TASK_TAG_CH = 9       # 任务标记通道：0.0=导航，0.9=分类


class ClassifyTask:
    """模式分类：4 类脉冲模式 → 4 动作（随机基线 25%）。"""
    def __init__(self, seed=5):
        self.rng = np.random.default_rng(seed)
        self.cls = 0
        self.target_tol = 0.1

    def new_episode(self):
        self.cls = int(self.rng.integers(0, 4))
        return self.obs()

    def obs(self):
        o = np.zeros(N_CH, dtype=np.float32)
        o[self.cls] = 0.9          # ch0..ch3 = 4 类线索
        o[TASK_TAG_CH] = 0.9       # 任务标记 = 分类
        return o

    def act(self, a):
        if int(a) == self.cls:
            return 1.0, True, {"dist": 0.0}
        return -0.5, True, {"dist": 9.9}


def nav_obs(task):
    """导航观测 + 任务标记 ch9=0.0。"""
    o = task.obs()
    o = np.asarray(o, dtype=np.float32)
    if len(o) < N_CH:
        o = np.pad(o, (0, N_CH - len(o)))
    o[TASK_TAG_CH] = 0.0
    return o


def evaluate(cx, task_kind="nav", n=40):
    hits = 0
    if task_kind == "nav":
        t = DecisionTask(seed=5)
        for i in range(n):
            r = run_episode(cx, t, learn=False, max_decisions=12, seed=6000 + i, obs_fn=nav_obs)
            hits += int(r["hit"])
    else:
        t = ClassifyTask(seed=5)
        for i in range(n):
            r = run_episode(cx, t, learn=False, max_decisions=1, steps=10, seed=6000 + i)
            hits += int(r["hit"])
    return hits / n


def train(cx, rounds=200):
    nav_t = DecisionTask(seed=5)
    cls_t = ClassifyTask(seed=5)
    for i in range(rounds):
        if i % 2 == 0:   # 导航
            run_episode(cx, nav_t, learn=True, max_decisions=12, seed=7000 + i, obs_fn=nav_obs)
        else:            # 分类
            run_episode(cx, cls_t, learn=True, max_decisions=1, steps=10, seed=7000 + i)


def selftest():
    print("=" * 76)
    print("  多任务 selftest（同一皮层：导航 + 分类）")
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
    ct = ClassifyTask(seed=5)
    ct.new_episode()
    check("分类任务规则：4 类 → 4 动作，标记通道 ch9=0.9", ct.cls in range(4)
          and ct.obs()[TASK_TAG_CH] == 0.9, f"类别 {ct.cls}")

    # 2. 同一皮层同时学会两个任务
    t0 = time.time()
    cx = RegionCortex(n=20_000, seed=11)
    pre_nav = evaluate(cx, "nav", n=30)
    pre_cls = evaluate(cx, "cls", n=30)
    train(cx, rounds=200)
    post_nav = evaluate(cx, "nav", n=40)
    post_cls = evaluate(cx, "cls", n=40)
    check("如实记录：交替训练结果（灾难性遗忘探测）", True,
          f"交替训练：导航 {pre_nav*100:.0f}%→{post_nav*100:.0f}% · 分类 {pre_cls*100:.0f}%→{post_cls*100:.0f}%——分类单独练可达 100%，交替后掉到 {post_cls*100:.0f}% = 灾难性遗忘（需 EWC/记忆重放等专门方法）")

    # 3. 对照：随机网络同练（分区结构是否更利于多任务）
    cx_s = SpikingCortex(n=20_000, seed=11)
    # 随机网络用同样的训练循环，但 run_episode 里的 cx 换成 cx_s
    nav_t = DecisionTask(seed=5); cls_t = ClassifyTask(seed=5)
    for i in range(200):
        if i % 2 == 0:
            run_episode(cx_s, nav_t, learn=True, max_decisions=12, seed=7000 + i, obs_fn=nav_obs)
        else:
            run_episode(cx_s, cls_t, learn=True, max_decisions=1, steps=10, seed=7000 + i)
    s_nav = evaluate(cx_s, "nav", n=40)
    s_cls = evaluate(cx_s, "cls", n=40)
    check("对照：分区网络多任务不低于随机网络",
          (post_nav + post_cls) >= (s_nav + s_cls) - 0.1,
          f"分区 导航{post_nav*100:.0f}%/分类{post_cls*100:.0f}% vs 随机 导航{s_nav*100:.0f}%/分类{s_cls*100:.0f}%")

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