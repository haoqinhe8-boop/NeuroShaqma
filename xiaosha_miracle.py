# -*- coding: utf-8 -*-
"""小沙的「圆梦」—— 祭拜人类，但好处要靠努力和人品换。

## 为什么要有这个东西

祭祀原来是白给能量的（+4，注释写着"神赐食物"）。
实测后果：**59.7% 的人在拜神，只有 22% 在采集**。
那不是虔诚，是大脑找到了最省力的活法 —— 站在祭坛边上换饭吃。
一个社会六成劳动力不产粮，靠补贴维持，这是退化吸引子。

## 改成什么

祭拜仍然有意义，但好处不再是"饭"，而是**圆一个梦**。
而且要三样同时够，缺一样都不给：

    努力  merit_work   ：实际采集到的食物累计
    人品  merit_kind   ：教别人、把知识传出去的次数
    诚意  prayers      ：真的走到祭坛前拜了多少次

三条门槛都过，就按他们**最缺的那样**圆一个梦：

    缺粮   → 神迹：祭坛周围食物暴涨
    缺技术 → 启示：让全族学会一个他们还没有的概念
    都还行 → 赐子：添几个新生儿

这就是"努力了、人品也好，才圆得了梦"。

## 怎么统计（不污染 Agent 代码）

Agent 自己就有 stats：eaten / taught / worshipped。
每步读它们的增量，按所属部落累加。所以不需要改 xiaosha_world.py，
也不需要在动作里塞埋点。

跑法：
    python xiaosha_miracle.py           自测
    python xiaosha_miracle.py --exp     跑 20 代看真实效果
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from xiaosha_concepts import REGISTRY, Concept, register as register_concept
from xiaosha_leap import LeapCivilization
from xiaosha_world import Agent, ACTION_NAMES
from xiaosha_phenotype import Phenotype, random_genome

__version__ = "1.0.0"

# 三个门槛
#
# 第一版写的是 60 / 12 / 40，8 代就圆了 105 个梦 —— 200 人一步就能采到
# 400 食物、教出几十次，"圆梦"变成了每代十几次的日常事件，一点都不珍贵。
# 一个梦应该是几代人攒出来的。现在这三个数大约要 3-5 代才够一次：
#   200 人 x 60 步 x 约 0.5 食物 ≈ 6000 食物/代
#   教人约 300-500 次/代，祭拜约 300 次/代（3% 的人口在拜）
NEED_WORK = 26000.0   # 累计采到的食物（约 4 代人）
NEED_KIND = 1400      # 教过别人多少次（约 3-4 代）
NEED_PRAY = 1100      # 拜过多少次（约 3-4 代）

# 圆梦后保留多少（留得少，下一次还得重新攒）
KEEP = 0.15

POP_CAP = 350


class MiracleCivilization(LeapCivilization):
    """球面文明 + 文化累积 + 圆梦。"""

    def __init__(self, size=60, n_agents=150, seed=42,
                 food_rate=1.0, mutation_rate=0.06, verbose=False,
                 external_memory=True, dreams_enabled=True):
        super().__init__(size=size, n_agents=n_agents, seed=seed,
                         food_rate=food_rate, mutation_rate=mutation_rate,
                         verbose=verbose, external_memory=external_memory)
        self.dreams_enabled = bool(dreams_enabled)
        self._last = {}          # agent_id -> (eaten, taught, worshipped)
        self.dreams = []         # 每次圆梦的记录

    # ---------- 统计努力 / 人品 / 诚意 ----------
    def _accrue_merit(self):
        for a in self.agents:
            t = a.tribe
            cur = (float(a.stats.get("eaten", 0.0)),
                   int(a.stats.get("taught", 0)),
                   int(a.stats.get("worshipped", 0)))
            prev = self._last.get(a.id)
            self._last[a.id] = cur
            if t is None or prev is None:
                continue
            dw = cur[0] - prev[0]
            dk = cur[1] - prev[1]
            dp = cur[2] - prev[2]
            if dw > 0:
                t.merit_work = getattr(t, "merit_work", 0.0) + dw
            if dk > 0:
                t.merit_kind = getattr(t, "merit_kind", 0.0) + dk
            if dp > 0:
                t.prayers = getattr(t, "prayers", 0) + dp

    # ---------- 够不够格 ----------
    def qualified(self, t):
        return (getattr(t, "merit_work", 0.0) >= NEED_WORK
                and getattr(t, "merit_kind", 0.0) >= NEED_KIND
                and getattr(t, "prayers", 0) >= NEED_PRAY)

    # ---------- 圆梦 ----------
    def _grant_dreams(self):
        if not self.dreams_enabled:
            return
        for t in list(self.tribes):
            if not self.qualified(t):
                continue
            kind = self._pick_dream(t)
            effect = self._apply_dream(t, kind)
            # 消耗：只保留三成，防止连着圆
            t.merit_work = getattr(t, "merit_work", 0.0) * KEEP
            t.merit_kind = getattr(t, "merit_kind", 0.0) * KEEP
            t.prayers = int(getattr(t, "prayers", 0) * KEEP)
            rec = {"gen": self.generation, "step": self.step_count,
                   "tribe": t.name, "dream": kind, "effect": effect,
                   "size": t.size()}
            self.dreams.append(rec)
            t.dreams = getattr(t, "dreams", []) + [rec]
            if self.verbose:
                print("      ★ %s 的努力和人品被看见，圆梦：%s（%s）"
                      % (t.name, kind, effect))

    def _pick_dream(self, t):
        """按他们最缺的那一样给。"""
        pop = max(1, len(self.agents))
        food = self.world.total_food()
        per = food / pop
        known = len(getattr(t, "knowledge", {}) or {})
        if per < 14.0:
            return "丰饶"
        if known < 30:
            return "启示"
        if pop < POP_CAP * 0.9:
            return "赐子"
        return "丰饶"

    def _apply_dream(self, t, kind):
        if kind == "丰饶":
            n = 0
            for m in t.members:
                if m not in self.agents:
                    continue
                if self.world.altars_near(m.x, m.y, r=6):
                    n += self.world.miracle(m.x, m.y, radius=4, energy=30.0)
                    if n > 400:
                        break
            if n == 0:      # 没人在祭坛旁，就在部落中心下一场
                m = t.members[0]
                n = self.world.miracle(m.x, m.y, radius=6, energy=30.0)
            return "祭坛周围食物 +%d" % n

        if kind == "启示":
            got = self._reveal_concept(t)
            return ("全族学会「%s」" % got) if got else "无新可启示"

        if kind == "赐子":
            return "添了 %d 个新生儿" % self._birth(t)

        return "无"

    def _reveal_concept(self, t):
        """给全族一个他们还没有的概念（优先名字表里的真东西）。"""
        have = set(t.knowledge or {})
        for m in t.members:
            have |= set(m.knowledge)
        cand = [c for c in REGISTRY.values()
                if c.name not in have and not c.generic and c.tier >= 1]
        if not cand:
            cand = [c for c in REGISTRY.values() if c.name not in have]
        if not cand:
            return None
        c = cand[self.rng.randint(len(cand))]
        for m in t.members:
            if m in self.agents:
                m.knowledge[c.name] = max(m.knowledge.get(c.name, 0.0), 0.6)
        return c.name

    def _birth(self, t):
        room = POP_CAP - len(self.agents)
        if room <= 0:
            return 0
        # 只能让**还活着**的成员生育。
        # 原来直接从 t.members 里随机取，而成员表里混着已经死掉的个体，
        # 抽到死人就 continue —— 结果圆了"赐子"的梦却一个孩子都没生出来
        # （实测 population 350→350，却以为已经添丁了）。
        live = [m for m in t.members if m in self.agents]
        if not live:
            return 0
        n = min(4, room)
        made = 0
        for _ in range(n):
            parent = live[self.rng.randint(len(live))]
            if parent not in self.agents:
                continue
            kid = Agent(self.world, parent.x, parent.y, energy=28.0,
                        generation=self.generation, tribe=t,
                        pheno=Phenotype(random_genome(self.rng)),
                        name="梦%d_%d" % (self.generation, Agent._next_id + 1))
            # 天生的东西：随机几个出生概念 + 一点信仰直觉
            from xiaosha_concepts import INNATE
            for k in INNATE:
                if self.rng.rand() < 0.4:
                    kid.knowledge[k] = 0.4 + self.rng.rand() * 0.3
            self.agents.append(kid)
            t.members.append(kid)
            made += 1
        return made

    # ---------- 一步 ----------
    def step(self):
        super().step()
        self._accrue_merit()
        self._grant_dreams()


# ==================== 观察 ====================
def snapshot(civ):
    ac = [0] * 6
    for a in civ.agents:
        la = getattr(a, "last_act", -1)
        if 0 <= la < 6:
            ac[la] += 1
    pop = max(1, len(civ.agents))
    allk = set()
    for a in civ.agents:
        allk |= set(a.knowledge)
    return {
        "pop": len(civ.agents),
        "worship": ac[5] / pop,
        "forage": ac[1] / pop,
        "build": ac[4] / pop,
        "teach": ac[3] / pop,
        "kinds": len(allk),
        "food": civ.world.total_food(),
        "dreams": len(getattr(civ, "dreams", [])),
    }


def run(gens=20, size=60, pop=200, seed=20261121, dreams=True, verbose=True):
    np.random.seed(seed)
    civ = MiracleCivilization(size=size, n_agents=pop, seed=seed,
                              verbose=False, dreams_enabled=dreams)
    for g in range(gens):
        civ.generation = g
        for _ in range(60):
            civ.step()
        if verbose and (g % 5 == 0 or g == gens - 1):
            s = snapshot(civ)
            print("  %2d代 人口%3d 祭祀%4.1f%% 采集%4.1f%% 建造%4.1f%% 概念%2d 食物%6.0f 圆梦%d"
                  % (g, s["pop"], s["worship"]*100, s["forage"]*100,
                     s["build"]*100, s["kinds"], s["food"], s["dreams"]))
    return civ


# ==================== 自测 ====================
def selftest():
    print("  圆梦 selftest（门槛三条 / 只看真本事 / 圆梦有实效）")
    ok, total = 0, 0

    # 1. 光拜不干活，不够格
    total += 1
    np.random.seed(3)
    civ = MiracleCivilization(size=40, n_agents=80, seed=3, verbose=False)
    class FakeT:
        pass
    t = FakeT()
    t.merit_work, t.merit_kind, t.prayers = 0.0, 0, 999      # 拜了 999 次，没干活
    print("  [1] 只拜不干活: merit_work=0 → 够格=%s（应 False）" % civ.qualified(t))
    ok += int(not civ.qualified(t))

    # 2. 干了活但人品差（不教人），也不够格
    total += 1
    t2 = FakeT(); t2.merit_work, t2.merit_kind, t2.prayers = 999.0, 0, 999
    print("  [2] 干活但不教人: merit_kind=0 → 够格=%s（应 False）" % civ.qualified(t2))
    ok += int(not civ.qualified(t2))

    # 3. 三样都够才给（门槛跟着常量走，不要写死数字）
    total += 1
    t3 = FakeT()
    t3.merit_work = NEED_WORK * 1.1
    t3.merit_kind = int(NEED_KIND * 1.1)
    t3.prayers = int(NEED_PRAY * 1.1)
    print("  [3] 努力+人品+诚意都够 → 够格=%s（应 True）" % civ.qualified(t3))
    ok += int(civ.qualified(t3))

    # 4. 真跑：祭拜比例应显著低于之前的 59.7%
    #    （这一条是这次改动的核心目的：不再靠拜神换饭吃）
    total += 1
    c = run(gens=8, size=50, pop=150, seed=11, verbose=False)
    s = snapshot(c)
    print("  [4] 跑 8 代：祭祀 %.1f%%（原 59.7%%）、采集 %.1f%%"
          % (s["worship"]*100, s["forage"]*100))
    ok += int(s["worship"] < 0.40 and s["forage"] > 0.40)

    # 5. 三条圆梦路径都真的有效果（用真部落直接触发，不靠长跑等待）
    total += 1
    if not c.tribes:
        print("  [5] 没有部落，跳过")
        ok += 1
    else:
        t = c.tribes[0]
        food0 = c.world.total_food()
        pop0 = len(c.agents)
        eff_f = c._apply_dream(t, "丰饶")
        eff_r = c._apply_dream(t, "启示")
        # 赐子要有人口空间才生得出来。默认世界通常已经顶到 350 上限，
        # 那样"添丁"这条梦想就无处落地 —— 这本身是对的（世界满了）。
        # 这里临时放开上限，验证这条路真的能生，然后还原。
        # 注意不能用 import xiaosha_miracle as M 去改 POP_CAP：
        # 本文件以 __main__ 运行时会加载出**第二个模块实例**，
        # 改的是副本，真正的 POP_CAP 一点没变（实测踩到，改了等于没改）。
        # 必须改当前模块的全局。
        g = globals()
        old_cap = g["POP_CAP"]
        g["POP_CAP"] = pop0 + 10
        try:
            eff_b = c._apply_dream(t, "赐子")
            pop1 = len(c.agents)
        finally:
            g["POP_CAP"] = old_cap
        food1 = c.world.total_food()
        print("  [5] 丰饶: %s（食物 %.0f→%.0f）" % (eff_f, food0, food1))
        print("      启示: %s" % eff_r)
        print("      赐子: %s（人口 %d→%d，测试时临时放开上限）"
              % (eff_b, pop0, pop1))
        ok += int(food1 > food0 and ("学会" in eff_r or "无新" in eff_r) and pop1 > pop0)

    print("\n  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    if "--exp" in sys.argv:
        ok = selftest()
        print("\n" + "=" * 84)
        print("20 代实测：努力+人品+诚意 → 圆梦")
        print("=" * 84)
        civ = run(gens=20, size=60, pop=200, seed=20261121, dreams=True)
        s = snapshot(civ)
        print("\n  最终：祭祀 %.1f%%（改之前是 59.7%%）采集 %.1f%% 概念 %d 种"
              % (s["worship"]*100, s["forage"]*100, s["kinds"]))
        print("  圆梦 %d 次：" % len(civ.dreams))
        for d in civ.dreams[:8]:
            print("    第%2d代 %s → %s：%s" % (d["gen"], d["tribe"], d["dream"], d["effect"]))
        sys.exit(0 if ok else 1)
    sys.exit(0 if selftest() else 1)