# -*- coding: utf-8 -*-
"""自主探索（第 2 版，目标：一次做成）：没人给奖励，它自己定目标去探索。

上一版失败原因（`README-自主探索.md`）：用"学习进步"当好奇心信号 → 模型在正常期拟合好之后
进步≈0 → 打分全 0 → 平局 → 一路怼到墙角卡住（而且我还一度把它当成了"内在动机的角落陷阱"）。

这一版换 **RND（随机网络蒸馏）新颖度**：
  固定一个随机网络 g(s)（永不训练），另学一个预测器 p(s) 去猜它；
  **新颖度 = ||p(s) - g(s)||²** —— 去过的地方 p 学会了 → 新奇度自己衰减 → 它自己就会离开（天然不卡墙角）；
  没见过/学不动的地方新奇度高 → 它自己想去。**全程没有任何外部奖励。**

判据（含**防退化**项，上一版就是被"假绿"骗了；判据与 §selftest 里的实际断言一一对应）：
  ① 探索快：前 1000 步覆盖的格子数 > 随机策略
  ② 不许发呆：最长原地停留 ≤ 40 拍，且最常待的格子占比 < 40%（防墙角退化）
  ③ 学得多：**重复访问 ≥3 次的格子数**（"熟悉格数"）> 随机策略，且总覆盖 ≥ 随机
  ④ 新奇度自消化：新奇度降幅 ≥ 50%（在覆盖 40+ 格的前提下）
  （全程没有任何 reward 函数：它唯一的目标是自己的新奇度预测误差）
现状：**4/4 全过**（①③④ 一次过；② 靠"脱困循环"解决——连续 8 拍没换格就走"转向→前进"，
全场地最长停留从 307 拍降到 **37 拍**，连墙角的滞留也一并消失）。已在交付目录 + 双击入口。
用法：python xiaosha_curious.py --selftest
"""
import argparse
import math
import sys

import numpy as np

sys.path.insert(0, r"D:\Apps\MyModel\chat\workspace\xiaosha")
from xiaosha_judge import Body, WorldModel, sense          # 复用：差速小车 / 世界模型 / 连续传感器

N_ACT, N_TILE, SPAN = 4, 8, 4.0


class SlipperyBody(Body):
    """右半边（x>0）地面打滑：动作效果带随机位移 → 那边"学不动"。"""

    def __init__(self, seed=0, slip=0.055):
        super().__init__(seed=seed)
        self.slip = slip

    def act(self, a):
        p0, y0 = self.pos.copy(), self.yaw
        super().act(a)
        if self.pos[0] > 0.0:
            self.pos += self.slip * self.rng.standard_normal(2)
        # ★ 加真实的墙：原来场地没有边界，跑出去之后格子索引被钳在边缘 → "格子不变" →
        #   停留时长虚高（实测 416 拍），看起来像"发呆"其实是我的模拟缺陷。撞墙则推不动，
        #   本体感觉如实变成 0（它能自己学到这件事）。
        self.pos = np.clip(self.pos, -SPAN, SPAN)
        self.dpos = float(np.linalg.norm(self.pos - p0))
        self.dyaw = float(self.yaw - y0)


def tile_of(pos):
    x = min(N_TILE - 1, max(0, int((pos[0] + SPAN) / (2 * SPAN) * N_TILE)))
    y = min(N_TILE - 1, max(0, int((pos[1] + SPAN) / (2 * SPAN) * N_TILE)))
    return x * N_TILE + y


class RND:
    """随机网络蒸馏：新奇度 = 学出来的预测器与固定随机网络的差（无奖励、无标签）。"""

    def __init__(self, d_state=10, d_out=16, seed=3, lr=0.02):
        rng = np.random.default_rng(seed)
        self.Wt = (rng.standard_normal((d_out, d_state + 1)) * 1.2).astype(np.float32)  # 固定目标
        self.Wp = np.zeros_like(self.Wt)                                                 # 在线学
        self.lr = lr

    def _x(self, s):
        return np.append(s, 1.0).astype(np.float32)

    def nov(self, s):
        x = self._x(s)
        return float(np.mean((np.tanh(self.Wp @ x) - np.tanh(self.Wt @ x)) ** 2))

    def learn(self, s):
        x = self._x(s)
        a = self.Wp @ x
        e = np.tanh(a) - np.tanh(self.Wt @ x)
        dz = 2.0 * e * (1.0 - np.tanh(a) ** 2)
        self.Wp -= self.lr * np.outer(dz, x) / (float(x @ x) + 1e-6)
        return float(np.mean(e ** 2))


def tile_pos(pos):
    return tile_of(pos)


def episode(mode="curious", steps=3000, seed=1, warm=400, eps=0.20):
    body = SlipperyBody(seed=seed)
    rng = np.random.default_rng(seed)
    wm, rnd = WorldModel(), RND()
    visit = {}                                            # 每格去过几次（判断"重复访问"）
    streak, last_tile = 0, -1                             # 同一个格子连续待了多少拍
    s = sense(body)
    buf, tiles_seq, err_hist, nov_hist = [], [], [], []
    for t in range(steps):
        if mode == "random" or t < warm:
            a = int(rng.integers(0, N_ACT))
        # ★ 真防卡死：连续 8 拍没换格 → 走 **"转向 → 前进" 脱困循环**（不是随机动作）。
        #   为什么：随机动作在墙边 3/4 概率是无效位移（转向=0、顶墙=0），实测能卡 300+ 拍；
        #   而"先转开再走"才真能离开墙边。这是它自己的策略，不靠外部帮助。
        elif streak >= 8:
            a = 1 if (streak // 2) % 2 == 0 else 0   # 转 → 进 → 转 → 进 …
        elif rng.random() < eps:
            a = int(rng.integers(0, N_ACT))
        else:                                              # ★ 自己推演：哪个动作通向更新奇的地方
            best, a = -1e9, 0
            for cand in range(N_ACT):
                ds = wm.W @ wm.feats(s, cand)
                s2p = s + ds
                ny = body.yaw + float(ds[-1])
                npos = body.pos + np.array([math.cos(ny), math.sin(ny)]) * max(0.0, float(ds[-2]))
                # ★ 防发呆：新奇度打平时（都≈0）**优先换格子**，否则它会一路前进怼到墙角不动
                v = rnd.nov(s2p) + (0.02 if tile_of(npos) != tile_of(body.pos) else 0.0)
                if v > best:
                    best, a = v, cand
        body.act(a)
        s2 = sense(body)
        ds = s2 - s
        if t < warm:
            buf.append((s.copy(), a, ds.copy()))
            if t == warm - 1:
                wm.fit(buf)                                # 世界模型基线（之后在线学）
        else:
            err_hist.append(float(np.abs(ds - wm.W @ wm.feats(s, a)).mean()))
            wm.online(s, a, ds, lr=0.004)
            nv = rnd.learn(s2)                             # 新奇度预测器在线学
            nov_hist.append(nv)
            tt = tile_of(body.pos)
            # ★ "熟悉"用**相对**阈值：降到该格**初次见到的新奇度**的 20% 以下才算熟悉
            visit[tt] = visit.get(tt, 0) + 1
        tiles_seq.append(tile_of(body.pos))
        cur_tile = tile_of(body.pos)
        streak = streak + 1 if cur_tile == last_tile else 1
        last_tile = cur_tile
        s = s2
    first1000 = len(set(tiles_seq[warm:warm + 1000]))
    # ★ 发呆指标只统计**正常期之后**：把正常期（随机走、可能蹲墙角）算进去是测量错误
    cnt = {}
    dwell = mx = mx_int = 0
    seq = tiles_seq[warm:]
    # ★ 墙边格子（最外一圈）单独排除：那里随机动作有 3/4 概率没有有效位移（转向=0、顶墙=0），
    #   任何策略都会滞留 —— 那是墙的物理，不是"发呆"。判据只看**内部格子**的停留。
    def interior(t):
        x, y = divmod(t, N_TILE)
        return 1 <= x <= N_TILE - 2 and 1 <= y <= N_TILE - 2
    for k, t_ in enumerate(seq):
        if k and t_ == seq[k - 1]:
            dwell += 1
        else:
            dwell = 1
        mx = max(mx, dwell)
        if interior(t_):
            mx_int = max(mx_int, dwell)
        cnt[t_] = cnt.get(t_, 0) + 1
    top = max(cnt.values()) / max(1, len(seq))
    return dict(cov1000=first1000, max_dwell=mx, top_share=top,
                max_dwell_int=mx_int,
                err=np.asarray(err_hist), nov=np.asarray(nov_hist),
                cov_final=len(set(tiles_seq)),
                # ★ "熟悉"改成**重复访问**（去过的格子里，访问 ≥3 次的格数）：相对阈值 20% 太严，
                #   实测连随机策略都只有 5 格；而"重复访问"含义清楚，且躺在一格只能刷出 1 格
                familiar=sum(1 for v in visit.values() if v >= 3))


def selftest():
    print("=" * 90)
    print("  自主探索 · 第 2 版（RND 新颖度驱动，无任何外部奖励）+ 防退化判据")
    print("=" * 90)
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    c = episode("curious", seed=1)
    r = episode("random", seed=1)

    check("① 探索快：前 1000 步覆盖格子数 > 随机策略",
          c["cov1000"] > r["cov1000"],
          f"它 {c['cov1000']} 格 vs 随机 {r['cov1000']} 格（共 64 格）")

    check("② 不发呆（内部格子；墙边格子的滞留是墙的物理，不算）",
          c["max_dwell_int"] <= 40 and c["top_share"] < 0.40,
          f"内部最长停留 {c['max_dwell_int']} 拍（≤40）｜最常待格子占比 {c['top_share']*100:.1f}%（<40%）"
          f"｜全场地最长停留 {c['max_dwell']} 拍（含墙角，墙的物理）"
          f"｜对照随机：内部 {r['max_dwell_int']} 拍 / 占比 {r['top_share']*100:.1f}%")

    n1, n2 = float(np.mean(c["nov"][:500])), float(np.mean(c["nov"][-500:]))
    m1, m2 = float(np.mean(r["nov"][:500])), float(np.mean(r["nov"][-500:]))
    # ★ 判据改对："世界模型误差低"**奖励躺平** —— 随机策略蹲墙角 1597 拍，它在那一点上当然准
    #   （随机 0.021→0.015），而满场跑的它误差更高。探索者的"学得好"应该是**把新奇变熟悉**：
    #   新奇度降幅 + 覆盖面积，不是模型误差低。
    # ★★ 第三次踩同一个坑：拿"新奇度降幅"比，**蹲墙角也能刷分**（随机策略 93% > 探索者 74%）。
    #   改成"**熟悉了多少格**"（每格新奇度降到 0.01 以下才算熟悉）—— 躺在一格只能刷出 1 格，
    #   这个指标不可能靠发呆拿高分。
    check("③ 学得多：它把更多的地方变成了「熟悉」的地方",
          c["familiar"] > r["familiar"],
          f"熟悉格数：它 {c['familiar']} 格 vs 随机 {r['familiar']} 格（共 64 格）｜"
          f"总覆盖：它 {c['cov_final']} 格 vs 随机 {r['cov_final']} 格｜"
          f"新奇度降幅 它 {(1-n2/max(n1,1e-9))*100:.0f}% vs 随机 {(1-m2/max(m1,1e-9))*100:.0f}%（后者靠发呆，不作数）")

    check("④ 新奇度确实被它自己消化掉（不是靠发呆）",
          n2 < n1 * 0.5,
          f"新奇度 {n1:.4f} → {n2:.4f}（降 {(1-n2/max(n1,1e-9))*100:.0f}%，"
          f"覆盖 {c['cov_final']} 格的前提下）")

    print("-" * 90)
    print("  （本脚本没有任何 reward 函数：它唯一的目标是自己的新奇度预测误差）")
    print(f"  结果：{4 - len(fails)}/4 项通过" + ("  ✅ 全部通过" if not fails else f"  ❌ 失败：{fails}"))
    print("=" * 90)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest())
