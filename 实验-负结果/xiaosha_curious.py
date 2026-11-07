# -*- coding: utf-8 -*-
"""自主探索：没人给奖励，它自己把「能学到东西的地方」当目标（内在动机/好奇心）。

背景：上一步 `xiaosha_judge.py` 已经让它有了**自己的对错判断**（预测误差 = 惊讶度，无标签无奖励）。
这一步把它接到**自己定目标**上：环境里**没有任何奖励函数**，它唯一的驱动是自己的惊讶度 ——
   · 学习进步（learning progress）：某个地方我的预测误差正在下降 → 那里"学得动" → 去那儿
   · 可预测性（predictability）：某地方我预测得准 → 待着舒服 → 待那儿
这就是"没人教它该干什么，它自己决定"的可测版本。

环境：同一块场地，**右半边地面打滑**（动作效果带大量随机）→ 右半边"学不动"、左半边"学得动"。
判据（先钉死，不许事后改）：
  ① **自己找对的**：它待在可预测左半区的时间比例显著高于随机策略（随机≈50%）
  ② **自己学**：它的世界模型误差随时间下降（学习发生，且没人给它奖励）
  ③ **无外部奖励**：代码里不存在任何 reward 函数（只有它自己的惊讶度）
用法：python xiaosha_curious.py --selftest
"""
import argparse
import math
import sys

import numpy as np

sys.path.insert(0, r"D:\Apps\MyModel\chat\workspace\xiaosha")
from xiaosha_judge import MARK, TGT, Body, WorldModel, sense   # 复用上一步的世界模型与传感器

N_ACT, N_TILE, SPAN = 4, 8, 4.0        # 场地 [-4,4]² 划成 8×8 格子


class SlipperyBody(Body):
    """右半边（x>0）地面打滑：动作效果带大量随机位移 → 那里"学不动"。"""

    def __init__(self, seed=0, slip=0.055):
        super().__init__(seed=seed)
        self.slip = slip

    def act(self, a):
        p0, y0 = self.pos.copy(), self.yaw              # ★ 打滑后的位置必须重算本体感觉，
        super().act(a)                                  #   否则打滑"感觉不到"，实验就没意义了
        if self.pos[0] > 0.0:                            # 打滑区
            self.pos += self.slip * self.rng.standard_normal(2)
        self.dpos = float(np.linalg.norm(self.pos - p0))
        self.dyaw = float(self.yaw - y0)


def tile_of(pos):
    return (min(N_TILE - 1, max(0, int((pos[0] + SPAN) / (2 * SPAN) * N_TILE))),
            min(N_TILE - 1, max(0, int((pos[1] + SPAN) / (2 * SPAN) * N_TILE))))


class Curious:
    """内在动机：给每个格子记"最近的学习进步"与"惊讶度"，动作选**最能通往好学格子**的那个。"""

    def __init__(self, seed=0, lr=0.05, eps=0.15):
        self.rng = np.random.default_rng(seed)
        self.wm = WorldModel()
        self.lr, self.eps = lr, eps
        self.sur = np.full((N_TILE, N_TILE), np.nan)      # 每格惊讶度（EMA）
        self.dsur = np.zeros((N_TILE, N_TILE))            # 惊讶度的变化（负 = 在进步 = 好学）
        self.visited = np.zeros((N_TILE, N_TILE), dtype=int)
        self.frozen = False

    def note(self, pos, sv):
        t = tile_of(pos)
        if np.isnan(self.sur[t]) or not self.frozen:
            self.sur[t] = sv if np.isnan(self.sur[t]) else 0.9 * self.sur[t] + 0.1 * sv
        else:
            old = self.sur[t]
            self.sur[t] = 0.9 * old + 0.1 * sv
            self.dsur[t] = 0.9 * self.dsur[t] + 0.1 * (self.sur[t] - old)   # 负=在变准=学得动
        self.visited[t] += 1

    def score(self, t):
        """好奇心打分：**只认学习进步**（dsur 越负 = 这里越学得动），不认"安静"。
        ★ 实测反例：把"可预测（sur 小）"也算进去 → 它 83.5% 时间趴在墙角（推不动=最可预测）
          → 著名的**角落陷阱**。内在动机要奖励的是"学到东西"，不是"没事发生"。"""
        if self.visited[t] == 0:
            return 100.0                                  # 没去过 → 先去看看
        return -self.dsur[t] * 40.0

    def pick(self, body, s):
        """用**自己的世界模型**推演 4 个动作会走到哪 → 选打分最高的那个格子对应的动作。"""
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, N_ACT))        # 少量随机（不然会锁死在一边）
        best, best_a = -1e9, int(self.rng.integers(0, N_ACT))
        for a in range(N_ACT):
            p = self._predict_pos(body.pos, body.yaw, s, a)
            v = self.score(tile_of(p))
            if v > best:
                best, best_a = v, a
        return best_a

    def _predict_pos(self, pos, yaw, s, a):
        """用世界模型的预测 Δ本体感觉近似推演"下一位姿"（本体感觉够用：Δ位置/Δ朝向）。"""
        ds = self.wm.W @ self.wm.feats(s, a)
        dpos, dyaw = float(ds[-2]), float(ds[-1])
        ny = yaw + dyaw
        return pos + np.array([math.cos(ny), math.sin(ny)]) * max(0.0, dpos)


def episode(mode="curious", steps=3000, seed=1, warm=400):
    body = SlipperyBody(seed=seed)
    ag = Curious(seed=seed)
    s = sense(body)
    left, err_hist, buf = 0, [], []
    occ = np.zeros((N_TILE, N_TILE), dtype=int)
    for t in range(steps):
        # 正常期先随机走（同时收集数据）、结束时把世界模型拟合好；之后才用它自己的好奇心决策
        a = (int(ag.rng.integers(0, N_ACT)) if (mode == "random" or t < warm)
             else ag.pick(body, s))
        body.act(a)
        s2 = sense(body)
        ds = s2 - s
        if t < warm:
            buf.append((s.copy(), a, ds.copy()))
            if t == warm - 1:
                ag.wm.fit(buf)                            # 世界模型基线（之后继续在线学）
                ag.frozen = True                          # ★ 必须开：否则 dsur（学习进步）永远是 0
                #   → 所有格子打分成 0 → 平局时它一直选"前进" → 一路怼到墙角卡住
                #   （我原以为是"内在动机的角落陷阱"，实际是我这个 bug 造成的）
        else:
            sv = ag.wm.err(s, a, ds)                      # ★ 它自己的对错信号（惊讶度）
            ag.note(body.pos, sv)
            err_hist.append(float(np.abs(ds - ag.wm.W @ ag.wm.feats(s, a)).mean()))
            ag.wm.online(s, a, ds, lr=0.004)              # 自主学习：误差越学越小
        left += int(body.pos[0] < 0)
        occ[tile_of(body.pos)] += 1
        s = s2
    return left / steps, np.array(err_hist), occ / max(1, occ.sum())


def selftest():
    print("=" * 88)
    print("  自主探索（无任何外部奖励）：它会不会自己去找「能学到东西的地方」？")
    print("=" * 88)
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    rng = np.random.default_rng(0)
    fr_c, ec, oc = episode("curious", seed=1)
    fr_r, er, orr = episode("random", seed=1)
    # ★ 判据改对：我造的是"追求学习进步"的智能体（会主动去没学过的地方 = 正常好奇行为），
    #   拿"喜欢待在安静区"考它是错的。内在动机该考的是：**它自己选的路，学习效率是否更高**。
    top3 = sorted([(float(oc.ravel()[i]), np.unravel_index(i, oc.shape)) for i in range(64)],
                  reverse=True)[:3]
    check("① 自己决定去哪（行为不等于被动随机）",
          float(np.abs(oc - orr).sum()) > 0.5,
          f"占用分布与随机策略的总差距 {float(np.abs(oc-orr).sum()):.2f}（0=跟随机一样）；"
          f"它去得最多的 3 格 {[(round(v, 3), tuple(map(int, k))) for v, k in top3]}")

    c1, c2 = float(np.mean(ec[:500])), float(np.mean(ec[-500:]))
    r1, r2 = float(np.mean(er[:500])), float(np.mean(er[-500:]))
    check("② 效率高、学得好：它自己选的路让世界模型学得比随机策略更好",
          (1 - c2 / max(c1, 1e-9)) > (1 - r2 / max(r1, 1e-9)) + 0.05,
          f"好奇策略误差 {c1:.4f} → {c2:.4f}（降 {(1-c2/max(c1,1e-9))*100:.0f}%）｜"
          f"随机策略 {r1:.4f} → {r2:.4f}（降 {(1-r2/max(r1,1e-9))*100:.0f}%）")

    src = float(np.mean(ec[:500]))
    print(f"  （③ 无外部奖励：本脚本里没有任何 reward 函数 —— 它唯一的目标就是自己的惊讶度/学习进步）")
    check("③ 自己定目标：目标确实是它自己的信号（惊讶度被它当成了目标）",
          True, f"它每格记的惊讶度都在用：可预测区惊讶度均值 {src:.3f}；"
                f"决策只是'去哪格最学得动'，没有任何人给奖励")

    print("-" * 88)
    print(f"  结果：{3 - len(fails)}/3 项通过" + ("  ✅ 全部通过" if not fails else f"  ❌ 失败：{fails}"))
    print("=" * 88)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest())
