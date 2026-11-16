# -*- coding: utf-8 -*-
"""小沙的生态系统 —— 食物、衰老、竞争、死亡。

之前的问题：每个个体都能无限繁殖，"淘汰"只是 `del cortex` 释放内存。
没有死亡、没有衰老、没有资源限制，所以：
  · 种群不会受环境约束
  · 没有生态位，没有竞争
  · 适应度只有"繁殖数"一个维度

现在加四件事：

  ① 食物限制 Food
     环境每步只产出固定量的食物。活力高的个体优先吃到，
     吃不饱的饿死 —— 这就是竞争。

  ② 衰老 Senescence
     年龄越大存活率越低（真实现象：Gompertz 死亡率随年龄指数上升）。
     这里用幂函数 `1 - (age/max_age)^1.5`，简单但形状接近。

  ③ 死亡 Death
     两种死法：老死（年龄上限）、饿死（食物不够）。
     死亡不是内存回收，是**个体从种群中消失**。

  ④ 食物再生 Regeneration
     食物每步按固定速率恢复，上限是环境承载力。
     这让种群数量围绕承载力上下波动（逻辑斯蒂增长），而不是无限膨胀。

为什么这些重要：没有死亡就没有选择。达尔文式进化的三个前提是
变异、遗传、**差异化存活**——前两个已经有了，缺的就是第三个。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

__version__ = "1.0.0"


class Ecosystem:
    """一个极简生态系统：食物池 + 衰老 + 竞争取食 + 死亡。

    参数（都对应真实现象）：
      food_capacity    环境承载力（食物上限）
      food_per_agent   每个个体每个时间步的食物需求
      regen            每步食物再生量
      max_age          最大寿命（步）
      senescence_pow   衰老曲线指数（1.0=线性, 1.5=加速衰老, 2.0=急剧）
    """

    def __init__(self, food_capacity=600, food_per_agent=25, regen=45,
                 max_age=25, senescence_pow=1.5, rng=None):
        self.capacity = float(food_capacity)
        self.food = float(food_capacity)
        self.food_per_agent = float(food_per_agent)
        self.regen = float(regen)
        self.max_age = int(max_age)
        self.senescence_pow = float(senescence_pow)
        self.rng = rng or np.random.RandomState()

        # 统计
        self.stats = {
            "steps": 0,
            "births": 0,
            "deaths_starve": 0,
            "deaths_old": 0,
            "deaths_weak": 0,
            "food_history": [],
            "pop_history": [],
        }

    # ---------- 一步 ----------
    def step(self, population, vitalities=None):
        """走一个时间步。

        population: list[Offspring]
        vitalities: dict {individual_name: vitality}（用于竞争排序，可空）

        返回 (alive_list, deaths_list)
        deaths_list 元素 = (individual, cause, survive_prob)
        """
        self.stats["steps"] += 1
        vitalities = vitalities or {}

        # 竞争排序：活力高的优先吃到食物
        order = sorted(
            range(len(population)),
            key=lambda i: -float(vitalities.get(population[i].name, 0.0)),
        )

        alive, deaths = [], []
        for rank, i in enumerate(order):
            ind = population[i]
            ind.age = getattr(ind, "age", 0) + 1

            # ① 衰老因子
            age_frac = min(1.0, ind.age / max(1, self.max_age))
            age_factor = max(0.0, 1.0 - age_frac ** self.senescence_pow)

            # ② 竞争取食
            if self.food >= self.food_per_agent:
                self.food -= self.food_per_agent
                fed = 1.0
            elif self.food > 0:
                fed = self.food / self.food_per_agent
                self.food = 0.0
            else:
                fed = 0.0

            # ③ 存活判定
            survive_prob = float(np.clip(age_factor * fed, 0.0, 1.0))
            if self.rng.rand() < survive_prob:
                alive.append(ind)
            else:
                if ind.age >= self.max_age:
                    cause = "old"
                    self.stats["deaths_old"] += 1
                elif fed < 0.5:
                    cause = "starve"
                    self.stats["deaths_starve"] += 1
                else:
                    cause = "weak"
                    self.stats["deaths_weak"] += 1
                deaths.append((ind, cause, round(survive_prob, 3)))

        # ④ 食物再生
        self.food = min(self.capacity, self.food + self.regen)

        self.stats["food_history"].append(round(self.food, 1))
        self.stats["pop_history"].append(len(alive))
        return alive, deaths

    # ---------- 查询 ----------
    def carrying_capacity(self):
        """逻辑斯蒂承载力：食物收支平衡时能养活多少个。"""
        return int(self.regen / max(self.food_per_agent, 1e-9))

    def summary(self):
        s = dict(self.stats)
        s.pop("food_history", None)
        s.pop("pop_history", None)
        total_deaths = s["deaths_starve"] + s["deaths_old"] + s["deaths_weak"]
        s["deaths_total"] = total_deaths
        s["carrying_capacity"] = self.carrying_capacity()
        return s


class Population:
    """带生态约束的种群：把 Ecosystem 和繁殖循环接起来。"""

    def __init__(self, members=None, ecosystem=None):
        self.members = list(members or [])
        self.eco = ecosystem or Ecosystem()
        self.generation = 0
        self.log = []

    def cull(self, vitalities=None, verbose=False):
        """生态筛选：走一步，移除死者。返回死亡数。"""
        before = len(self.members)
        alive, deaths = self.eco.step(self.members, vitalities)
        self.members = alive
        n_died = before - len(alive)
        if verbose and deaths:
            causes = {}
            for _ind, cause, _p in deaths:
                causes[cause] = causes.get(cause, 0) + 1
            print("    生态筛选: %d → %d（死 %d: %s）"
                  % (before, len(alive), n_died, causes))
        self.log.append({"step": self.eco.stats["steps"], "pop": len(self.members),
                         "food": round(self.eco.food, 1), "died": n_died})
        return n_died

    def add(self, individual):
        individual.age = 0
        self.members.append(individual)
        self.eco.stats["births"] += 1

    def size(self):
        return len(self.members)


# ==================== 自测 ====================

def selftest():
    print("  小沙生态系统 selftest（食物 / 衰老 / 竞争 / 死亡）")
    np.random.seed(20261116)
    ok = 0
    total = 0

    class Fake:
        """轻量替身（不构建真皮层，只测生态逻辑）。"""
        _n = 0

        def __init__(self, name=None, age=0):
            Fake._n += 1
            self.name = name or ("f%d" % Fake._n)
            self.age = age
            self.alive = True

    # ---- 1. 食物充足时不会饿死 ----
    total += 1
    eco = Ecosystem(food_capacity=10000, food_per_agent=10, regen=10000,
                    max_age=1000, rng=np.random.RandomState(1))
    pop = [Fake() for _ in range(10)]
    alive, deaths = eco.step(pop, {p.name: 5000 for p in pop})
    print("  [1] 食物充足: 10 → %d 存活，%d 死亡（应全活）" % (len(alive), len(deaths)))
    ok += int(len(alive) == 10)

    # ---- 2. 食物耗尽时会饿死 ----
    total += 1
    eco2 = Ecosystem(food_capacity=10, food_per_agent=50, regen=0,
                     max_age=1000, rng=np.random.RandomState(2))
    pop2 = [Fake() for _ in range(20)]
    alive2, deaths2 = eco2.step(pop2, {p.name: 1000 for p in pop2})
    starved = sum(1 for _i, c, _p in deaths2 if c == "starve")
    print("  [2] 食物耗尽: 20 → %d 存活，%d 死亡（其中饿死 %d，应大部分死）"
          % (len(alive2), len(deaths2), starved))
    ok += int(len(alive2) <= 2 and starved >= 15)

    # ---- 3. 年龄到上限会老死 ----
    total += 1
    eco3 = Ecosystem(food_capacity=1e6, food_per_agent=1, regen=1e6,
                     max_age=10, senescence_pow=2.0, rng=np.random.RandomState(3))
    old = [Fake("old1", age=10), Fake("old2", age=20)]
    alive3, deaths3 = eco3.step(old, {p.name: 9999 for p in old})
    causes3 = [c for _i, c, _p in deaths3]
    print("  [3] 超龄个体: 2 → %d 存活，死因 %s（应为 old）" % (len(alive3), causes3))
    ok += int(len(alive3) == 0 and all(c == "old" for c in causes3))

    # ---- 4. 衰老：年龄越大存活率越低 ----
    total += 1
    eco4 = Ecosystem(food_capacity=1e6, food_per_agent=1, regen=1e6,
                     max_age=20, senescence_pow=1.5, rng=np.random.RandomState(4))
    young = Fake("y", age=1)
    aged = Fake("a", age=18)
    # 各跑 200 次，统计存活率
    sy = sa = 0
    for _ in range(200):
        y = Fake("y", age=1); a = Fake("a", age=18)
        al, _d = eco4.step([y, a], {"y": 9999, "a": 9999})
        sy += len(al) >= 1
        sa += len(al) == 2
    print("  [4] 衰老: 年轻存活 %d/200 vs 老年存活 %d/200（老年应显著低）" % (sy, sa))
    ok += int(sy > sa + 10)

    # ---- 5. 竞争：活力高的优先吃到食物 ----
    total += 1
    eco5 = Ecosystem(food_capacity=30, food_per_agent=30, regen=0,
                     max_age=1000, rng=np.random.RandomState(5))
    strong = Fake("strong")
    weak = Fake("weak")
    alive5, deaths5 = eco5.step([weak, strong], {"strong": 9000, "weak": 100})
    names5 = [i.name for i in alive5]
    print("  [5] 竞争取食: 食物只够 1 个 → 存活 %s（应为 strong）" % names5)
    ok += int("strong" in names5 and "weak" not in names5)

    # ---- 6. 食物再生让种群围绕承载力波动 ----
    total += 1
    eco6 = Ecosystem(food_capacity=200, food_per_agent=20, regen=40,
                     max_age=10000, rng=np.random.RandomState(6))
    cc = eco6.carrying_capacity()
    print("  [6] 承载力: 每步再生 %.0f / 每人需求 %.0f = %d 个（逻辑斯蒂）"
          % (eco6.regen, eco6.food_per_agent, cc))
    ok += int(cc == 2)

    # ---- 7. 种群不会无限增长 ----
    total += 1
    eco7 = Ecosystem(food_capacity=100, food_per_agent=20, regen=40,
                     max_age=100000, senescence_pow=1.0, rng=np.random.RandomState(7))
    p = Population([Fake() for _ in range(20)], eco7)
    sizes = []
    for _ in range(40):
        p.cull({m.name: 1000 for m in p.members})
        sizes.append(p.size())
        if p.size() == 0:
            break
    print("  [7] 40 步后种群: 20 → %d（承载力 %d，应被资源压住不会爆）"
          % (p.size(), eco7.carrying_capacity()))
    ok += int(p.size() <= eco7.carrying_capacity() + 3)

    # ---- 8. 食物历史被记录（可画曲线）----
    total += 1
    hist = eco7.stats["food_history"]
    print("  [8] 食物历史: 记录 %d 步 | 最后值 %.1f | 均值 %.1f"
          % (len(hist), hist[-1] if hist else 0, float(np.mean(hist)) if hist else 0))
    ok += int(len(hist) >= 10)

    # ---- 9. 死亡原因被分类统计 ----
    total += 1
    s = eco2.summary()
    print("  [9] 统计: 饿死 %d | 老死 %d | 体弱 %d | 总死 %d"
          % (s["deaths_starve"], s["deaths_old"], s["deaths_weak"], s["deaths_total"]))
    ok += int(s["deaths_total"] > 0 and "carrying_capacity" in s)

    # ---- 10. 种群日志可追溯 ----
    total += 1
    print("  [10] 种群日志: %d 条 | 首条 %s" % (len(p.log), p.log[0] if p.log else None))
    ok += int(len(p.log) >= 10)

    print("\n  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    sys.exit(0 if selftest() else 1)
