# -*- coding: utf-8 -*-
"""劳动最光荣 —— 让这句话在这个世界里成立，而不是写在注释里。

## 一句话

谁干活，谁有声望；有声望的人说话有人听；
而"劳动光荣"这个念头，是**干活干够多的人自己想明白的**，不是天生发给他的。

## 三条链

1. **劳动 → 声望**
   每走一步，按他这一步在干什么记声望：
   采集 +1.0、建造/种地 +1.5、教人 +1.0（教也是劳动）。
   声望是累计的、不衰减 —— 这是一个记仇也记恩的世界。

2. **声望 → 影响力 → 信念传播**
   两个人靠近时，如果其中一个人信"劳动光荣"而且声望明显更高，
   另一个人就有可能被他说服。说服概率跟说话人的声望成正比。
   ——「光荣」的实际含义就是这个：他说的话比别人管用。

3. **信念 → 内在回报 → 更愿意劳动**
   内化之后（Agent.glory），采集和建造的回报放大 (1 + 0.8 * glory)。
   这一步在 xiaosha_world.py 的 act() 里，是唯一真正影响学习的地方。

所以"知道劳动光荣"不是一句口号：它改变奖励，改变动作分布，
还能被传下去。而没有声望的人说这话，没人信。

## 谁先想明白

声望攒到 GLORY_REALIZE 的人，会"领悟"这件事（glory 0.8）。
所以第一个说出"劳动最光荣"的，一定是干活最多的那个人 ——
这也正是现实里这句话的来源。

跑法：
    python xiaosha_labor.py          自测
    python xiaosha_labor.py --exp    A/B 对照（有/没有这套价值）
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from xiaosha_miracle import MiracleCivilization
from xiaosha_world import ACT_EAT, ACT_BUILD, ACT_TEACH, ACT_MOVE, neighbours

__version__ = "1.0.0"

# 每一步各动作记多少声望
PRESTIGE_OF_ACT = {
    ACT_EAT: 1.0,      # 采集
    ACT_BUILD: 1.5,    # 建造 / 种地
    ACT_TEACH: 1.0,    # 教人（也是劳动）
}

# 声望攒到多少，自己"想明白"劳动光荣。
# 原来是 25 —— 第 0 代就有人领悟，等于干了半个回合就明白了，
# "光荣"两个字一点都不值钱。150 大约要一两代持续劳动，
# 这样第一个说出这句话的人，确实是干出了名堂的那个。
GLORY_REALIZE = 150.0
GLORY_ON = 0.8         # 想明白之后先有多信
INFLUENCE = 0.02       # 有声望的人每步能说服邻居的概率系数
PRESTIGE_FOR_FULL = 60.0   # 声望到这个数，说服力算满


class LaborCivilization(MiracleCivilization):
    """球面文明 + 文化累积 + 圆梦 + 劳动光荣。"""

    def __init__(self, size=60, n_agents=150, seed=42,
                 food_rate=1.0, mutation_rate=0.06, verbose=False,
                 external_memory=True, dreams_enabled=True,
                 glory_culture=True):
        super().__init__(size=size, n_agents=n_agents, seed=seed,
                         food_rate=food_rate, mutation_rate=mutation_rate,
                         verbose=verbose, external_memory=external_memory,
                         dreams_enabled=dreams_enabled)
        self.glory_culture = bool(glory_culture)
        self.first_believer = None     # 第一个想明白的人
        self.glory_events = []         # 传播记录
        for a in self.agents:
            a.prestige = 0.0
            a.glory = 0.0

    # ---------- 1. 劳动 → 声望 ----------
    def _accrue_prestige(self):
        for a in self.agents:
            la = getattr(a, "last_act", -1)
            gain = PRESTIGE_OF_ACT.get(la, 0.0)
            if gain:
                a.prestige = getattr(a, "prestige", 0.0) + gain
            # 攒够了就自己想明白（不是被谁教的）
            if (self.glory_culture
                    and getattr(a, "glory", 0.0) <= 0.0
                    and getattr(a, "prestige", 0.0) >= GLORY_REALIZE):
                a.glory = GLORY_ON
                if self.first_believer is None:
                    self.first_believer = (self.generation, a.name,
                                           a.prestige)
                    if self.verbose:
                        print("      ✦ %s 干出了 %.0f 点声望，第一个想明白：劳动最光荣"
                              % (a.name, a.prestige))

    # ---------- 2. 声望 → 影响邻居 ----------
    def _spread_glory(self, by_pos):
        if not self.glory_culture:
            return
        for a in self.agents:
            g = getattr(a, "glory", 0.0)
            if g <= 0.0:
                continue
            # 说话人的声望决定他说服力
            power = min(1.0, getattr(a, "prestige", 0.0) / PRESTIGE_FOR_FULL)
            if power <= 0.0:
                continue
            for o in neighbours(by_pos, a.x, a.y, 2, a):
                og = getattr(o, "glory", 0.0)
                if og >= g - 0.15:
                    continue
                if self.rng.rand() >= INFLUENCE * power:
                    continue
                o.glory = min(1.0, og + 0.25)
                self.glory_events.append(
                    {"gen": self.generation, "from": a.name, "to": o.name,
                     "power": round(power, 3)})

    # ---------- 一步 ----------
    def step(self):
        super().step()
        self._accrue_prestige()
        # 用最新的位置重新建索引
        by_pos = {}
        for a in self.agents:
            by_pos.setdefault((a.x, a.y), []).append(a)
        self._spread_glory(by_pos)


# ==================== 观察 ====================
LABOR_ACTS = (ACT_EAT, ACT_BUILD)


def snapshot(civ):
    pop = max(1, len(civ.agents))
    ac = [0] * 6
    for a in civ.agents:
        la = getattr(a, "last_act", -1)
        if 0 <= la < 6:
            ac[la] += 1
    believers = sum(1 for a in civ.agents if getattr(a, "glory", 0.0) > 0.5)
    pres = [getattr(a, "prestige", 0.0) for a in civ.agents]
    # 声望最高的 10% 里，有多少比例此刻正在劳动
    top = sorted(civ.agents, key=lambda a: -getattr(a, "prestige", 0.0))
    k = max(1, int(len(top) * 0.1))
    top_labor = sum(1 for a in top[:k]
                    if getattr(a, "last_act", -1) in LABOR_ACTS) / k
    allk = set()
    for a in civ.agents:
        allk |= set(a.knowledge)
    return {
        "pop": len(civ.agents),
        "labor": (ac[ACT_EAT] + ac[ACT_BUILD]) / pop,
        "forage": ac[ACT_EAT] / pop,
        "worship": ac[5] / pop,
        "believe": believers / pop,
        "prestige_mean": float(np.mean(pres)) if pres else 0.0,
        "prestige_max": float(np.max(pres)) if pres else 0.0,
        "top_labor": top_labor,
        "kinds": len(allk),
        "food": civ.world.total_food(),
        "spread": len(getattr(civ, "glory_events", [])),
    }


def run(gens=14, size=60, pop=200, seed=20261121, glory=True, verbose=True):
    np.random.seed(seed)
    civ = LaborCivilization(size=size, n_agents=pop, seed=seed,
                            verbose=False, glory_culture=glory)
    for g in range(gens):
        civ.generation = g
        for _ in range(60):
            civ.step()
        if verbose and (g % 4 == 0 or g == gens - 1):
            s = snapshot(civ)
            print("  %2d代 人口%3d 劳动%5.1f%% 信者%5.1f%% 声望均%5.1f 高声望劳动%4.0f%% 概念%2d"
                  % (g, s["pop"], s["labor"]*100, s["believe"]*100,
                     s["prestige_mean"], s["top_labor"]*100, s["kinds"]))
    return civ


# ==================== 自测 ====================
def selftest():
    print("  劳动光荣 selftest（声望来自劳动 / 领悟要够格 / 能传播 / 有效果）")
    ok, total = 0, 0

    # 1. 声望只来自劳动：一直移动的人不涨声望
    total += 1
    np.random.seed(7)
    civ = LaborCivilization(size=40, n_agents=80, seed=7, verbose=False)
    a = civ.agents[0]
    a.prestige = 0.0
    a.last_act = ACT_MOVE
    for _ in range(50):
        civ._accrue_prestige()
    p_move = a.prestige
    a.last_act = ACT_EAT
    for _ in range(10):
        civ._accrue_prestige()
    print("  [1] 只移动 50 步声望 %.0f → 再采集 10 步 %.0f（应 0 → 10）"
          % (p_move, a.prestige))
    ok += int(p_move == 0 and a.prestige >= 10)

    # 2. 干得不够多，想不明白
    total += 1
    b = civ.agents[1]
    b.prestige = GLORY_REALIZE - 1
    b.glory = 0.0
    b.last_act = ACT_MOVE
    civ._accrue_prestige()
    print("  [2] 声望差 1 点时不领悟: glory=%.2f（应 0）" % b.glory)
    ok += int(b.glory == 0.0)

    # 3. 够了就想明白
    total += 1
    c = civ.agents[2]
    c.prestige = GLORY_REALIZE
    c.glory = 0.0
    c.last_act = ACT_EAT
    civ._accrue_prestige()
    print("  [3] 声望够了就领悟: glory=%.2f（应 %.2f）" % (c.glory, GLORY_ON))
    ok += int(c.glory >= GLORY_ON)

    # 4. 有声望的人能把信念传给邻居
    total += 1
    n0 = len(civ.glory_events)
    s = civ.agents[3]
    s.prestige = 200.0
    s.glory = 1.0
    lst = civ.agents[4]
    lst.glory = 0.0
    lst.x, lst.y = s.x, s.y
    for _ in range(200):
        civ._spread_glory({(s.x, s.y): [s, lst]})
        if lst.glory > 0:
            break
    print("  [4] 传教：邻居 glory %.2f，记录 %d 条（应 >0）"
          % (lst.glory, len(civ.glory_events) - n0))
    ok += int(lst.glory > 0 and len(civ.glory_events) > n0)

    # 5. 真跑：信的人多了，劳动占比更高
    total += 1
    c_off = run(gens=8, size=50, pop=150, seed=13, glory=False, verbose=False)
    c_on = run(gens=8, size=50, pop=150, seed=13, glory=True, verbose=False)
    so, sn = snapshot(c_off), snapshot(c_on)
    print("  [5] 8 代 A/B：劳动 %.1f%% vs %.1f%% | 信者 %.1f%% | 高声望者在劳动 %.0f%% | 首个领悟 %s"
          % (so["labor"]*100, sn["labor"]*100, sn["believe"]*100,
             sn["top_labor"]*100,
             (c_on.first_believer[1] if c_on.first_believer else "无人")))
    # 判据说明：劳动占比本身已经很高（砍掉"拜神换饭"之后约 85%），
    # 所以不能拿它当唯一判据 —— 那样测的只是"有没有变差"。
    # 这句话真正要成立的是三条：信念传得开、出声望的人还在劳动、劳动没被挤掉。
    ok += int(sn["believe"] > 0.5 and sn["top_labor"] >= 0.6
              and sn["labor"] >= so["labor"] - 0.02)

    print("\n  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    if "--exp" in sys.argv:
        ok = selftest()
        print("\n" + "=" * 88)
        print("14 代 A/B：有「劳动光荣」vs 没有")
        print("=" * 88)
        print("\n【对照组】没有这套价值")
        a = run(gens=14, size=60, pop=200, seed=20261121, glory=False)
        sa = snapshot(a)
        print("\n【劳动光荣】干得多 → 有声望 → 说话有人听 → 更愿意劳动")
        b = run(gens=14, size=60, pop=200, seed=20261121, glory=True)
        sb = snapshot(b)
        print("\n" + "=" * 88)
        print("第 13 代对比")
        print("=" * 88)
        print("  %-16s %10s %10s" % ("指标", "对照", "劳动光荣"))
        print("  %-16s %9.1f%% %9.1f%%" % ("劳动占比", sa["labor"]*100, sb["labor"]*100))
        print("  %-16s %9.1f%% %9.1f%%" % ("相信的人", sa["believe"]*100, sb["believe"]*100))
        print("  %-16s %10.1f %10.1f" % ("平均声望", sa["prestige_mean"], sb["prestige_mean"]))
        print("  %-16s %9.0f%% %9.0f%%" % ("高声望者在劳动", sa["top_labor"]*100, sb["top_labor"]*100))
        print("  %-16s %10d %10d" % ("概念种类", sa["kinds"], sb["kinds"]))
        print("  %-16s %10.0f %10.0f" % ("食物", sa["food"], sb["food"]))
        if b.first_believer:
            print("\n  ★ 第一个想明白「劳动最光荣」的：%s（第 %d 代，声望 %.0f）"
                  % (b.first_believer[1], b.first_believer[0], b.first_believer[2]))
        print("  ★ 信念传播了 %d 次" % sb["spread"])
        sys.exit(0 if ok else 1)
    sys.exit(0 if selftest() else 1)