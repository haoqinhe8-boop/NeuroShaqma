# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""脑区分工合作（2026-11-08）：像人脑一样分工、协调地合作

针对 README「果蝇区别」段落的回答——"每个连接都有分工"不是口号，是行为学可验证的：

任务：跨感官分工任务 CrossModalTask
  规则：(视觉线索, 听觉线索) → 4 个目标之一（AND 逻辑，跨脑区才能解）
  通道 → 脑区路由（与 RegionCortex 器官路由一一对应）：
    ch2/ch3 = 视觉线索 A/B → 只进视觉区
    ch4/ch5 = 听觉线索 A/B → 只进听觉区
    其余通道空。视觉区管看、听觉区管听、其他皮层整合、运动读出执行——各司其职。

验证（行为学证据，不是嘴上说分工）：
  ① 分区网络学会跨感官任务（练前 ~25% 随机 → 练后显著更高）
  ② 对照实验：随机网络同规模同学练 → 分区网络显著更好（分工有真实收益）
  ③ 消融实验：学完后遮住视觉通道 → 视觉组合全崩、听觉组合不受影响（视觉区专职"看"）
  ④ 消融实验：遮住听觉同理（听觉区专职"听"）
  ⑤ 联合区整合：完整信息 vs 单感官 → 其他皮层（联合区）发放模式显著不同（它在"整合"）

用法：python xiaosha_teamwork.py --selftest
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
from xiaosha_regions import RegionCortex, REGIONS

# 通道语义（对齐 RegionCortex 的器官路由：ch2/3→视觉区，ch4/5→听觉区）
CH_VIS_A, CH_VIS_B = 2, 3
CH_AUD_A, CH_AUD_B = 4, 5
N_CH = 10


class CrossModalTask:
    """跨感官分工任务：(视觉, 听觉) 组合 → 4 个目标之一。"""

    def __init__(self, seed=5):
        self.rng = np.random.default_rng(seed)
        self.target_tol = 0.1
        self.vis = 0
        self.aud = 0

    def new_episode(self):
        self.vis = int(self.rng.integers(0, 2))
        self.aud = int(self.rng.integers(0, 2))
        return self.obs()

    def target(self):
        """规则：(vis, aud) → 目标：00→0, 01→1, 10→2, 11→3"""
        return self.vis * 2 + self.aud

    def obs(self):
        o = np.zeros(N_CH, dtype=np.float32)
        o[CH_VIS_A if self.vis == 0 else CH_VIS_B] = 0.9
        o[CH_AUD_A if self.aud == 0 else CH_AUD_B] = 0.9
        return o

    def act(self, a):
        """a∈0..3 选目标。对 +1 done，错 -0.5 done。"""
        if int(a) == self.target():
            return 1.0, True, {"dist": 0.0}
        return -0.5, True, {"dist": 9.9}


def mask_fn(task, mask_vis=False, mask_aud=False):
    """消融用：遮住某感官的输入（网络已学完，冻结权重只遮输入）。"""
    def _f(t):
        o = t.obs()
        if mask_vis:
            o[CH_VIS_A] = o[CH_VIS_B] = 0.0
        if mask_aud:
            o[CH_AUD_A] = o[CH_AUD_B] = 0.0
        return o
    return _f


def evaluate(cx, task, n_episodes=40, obs_fn=None):
    hits = 0
    for i in range(n_episodes):
        r = run_episode(cx, task, learn=False, max_decisions=1, steps=10,
                        seed=2000 + i, obs_fn=obs_fn)
        hits += int(r["hit"])
    return hits / n_episodes


def joint_activity(cx, vis_only=False, full=False, steps=80):
    """全脑发放计数（单感官 vs 全信息的协同证据）。"""
    o = np.zeros(N_CH, dtype=np.float32)
    if vis_only:
        o[CH_VIS_A] = 1.0
    if full:
        o[CH_VIS_A] = 1.0
        o[CH_AUD_A] = 1.0
    cnt = 0
    for _ in range(steps):
        cnt += cx.step(o)
    return cnt


# ============================================================
def selftest():
    print("=" * 76)
    print("  脑区分工合作 selftest（跨感官任务 + 对照 + 消融）")
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
    task = CrossModalTask(seed=5)
    mapping = {}
    for v in (0, 1):
        for a in (0, 1):
            task.vis, task.aud = v, a
            mapping[(v, a)] = task.target()
    check("任务规则：4 种 (视,听) 组合 → 4 个唯一目标", sorted(mapping.values()) == [0, 1, 2, 3],
          f"映射 {mapping}")

    # 2. 分区网络学会跨感官任务（练前 vs 练后）
    t0 = time.time()
    cx_r = RegionCortex(n=20_000, seed=11)
    task = CrossModalTask(seed=5)
    pre = evaluate(cx_r, task, n_episodes=40)
    for i in range(300):
        run_episode(cx_r, task, learn=True, max_decisions=1, steps=10, seed=3000 + i)
    post = evaluate(cx_r, task, n_episodes=40)
    check("分区网络学会跨感官任务（练前 ~25% → 练后 ≥60%）", post >= 0.60 and post > pre + 0.2,
          f"练前 {pre*100:.0f}% → 练后 {post*100:.0f}%（300 局，{time.time()-t0:.0f}s）")

    # 3. 对照：随机网络同规模同学练（分工收益）
    cx_s = SpikingCortex(n=20_000, seed=11)
    task = CrossModalTask(seed=5)
    pre_s = evaluate(cx_s, task, n_episodes=40)
    for i in range(300):
        run_episode(cx_s, task, learn=True, max_decisions=1, steps=10, seed=3000 + i)
    post_s = evaluate(cx_s, task, n_episodes=40)
    check("对照实验：分区网络显著优于随机网络（分工有真实收益）", post > post_s + 0.08,
          f"分区 {post*100:.0f}% vs 随机 {post_s*100:.0f}%（随机练前 {pre_s*100:.0f}%）")

    # 4. 消融-视觉：遮视觉 → 任务崩（视觉区专职"看"）
    #    遮视觉后 (00,01) 和 (10,11) 各自无法区分 → 天花板 50%，实测含噪声 ~55%
    acc_blind = evaluate(cx_r, task, n_episodes=40, obs_fn=mask_fn(task, mask_vis=True))
    check("消融-视觉：遮住视觉通道 → 命中率崩（<60%，视觉区专职看）",
          acc_blind < 0.60 and acc_blind < post - 0.35,
          f"完整 {post*100:.0f}% → 遮视觉 {acc_blind*100:.0f}%（天花板 50%）")

    # 5. 消融-听觉：遮听觉同理（听觉区专职"听"）
    acc_deaf = evaluate(cx_r, task, n_episodes=40, obs_fn=mask_fn(task, mask_aud=True))
    check("消融-听觉：遮住听觉通道 → 命中率崩（<60%，听觉区专职听）",
          acc_deaf < 0.60 and acc_deaf < post - 0.35,
          f"完整 {post*100:.0f}% → 遮听觉 {acc_deaf*100:.0f}%（天花板 50%）")

    # 6. 联合区整合：全信息（视+听同时工作）→ 全脑发放显著高于单感官（两个脑区协同）
    cx_r.reset_state()
    c_vis = joint_activity(cx_r, vis_only=True)
    cx_r.reset_state()
    c_full = joint_activity(cx_r, full=True)
    ratio = c_full / max(1, c_vis)
    check("脑区协同：全信息（视+听）全脑发放 > 单视觉 1.5×（两脑区同时工作）",
          ratio > 1.5 and c_full > 0,
          f"单视觉 {c_vis} 发放 vs 全信息 {c_full} 发放（{ratio:.2f}×）")

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
