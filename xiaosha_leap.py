# -*- coding: utf-8 -*-
"""小沙的第一次大进化 —— 文化开始累积。

## 卡在哪里

现在的知识只活在活人脑子里：
  · 老人死时，知识交给**一个**同族幸存者（`_die`），打 85% 折扣
  · 新生儿只有 10% 概率从父母那儿继承一点
  · `Tribe.knowledge` 这个"公共知识池"一直在汇总，但**全项目没有一处把它读回去教人**

所以技术数在 20~28 之间来回摆动，涨不上去 —— 这一代学会的东西，
下一代要靠人活着口耳相传才能留住，链条一断就退回原点。

## 大进化是什么

真实人类跨过的那道坎是：**把知识存到身体之外**。
文字、账本、仓库 —— 从此知识不再随个体死亡而消失，文化开始有棘轮
（ratchet）：这一代会的，下一代不会退回去，只会往上加。

这里把它做成一个真机制，而且**不是白送的**：

  只有当部落里有人发明了「记录」类概念（属性同时含 信息 + 可储，
  比如 记录、文字），部落的知识池才会反过来教人。
  没有记录 = 知识继续随人死光（这就是现状，也是对照组）。

判定看属性，不看名字 —— 所以小沙自己发明出别的"能留存的信息载体"，
一样能触发。

## 怎么证明它真的发生了

同一随机种子，跑两遍：external_memory=False（对照）vs True（大进化）。
比 4 个指标：世界里掌握的概念种类、人均概念数、食物总量、发现次数。

跑法：python xiaosha_leap.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from xiaosha_concepts import (has_props, MAX_HELD, combine,
                             get as get_concept, register as register_concept)
from xiaosha_globe import GlobeCivilization
from xiaosha_civilization import SKILL_GATE

__version__ = "1.0.0"

# 「记录」的判定：信息 + 可储（文字/记录/账本都满足）
RECORD_PROPS = {"信息", "可储"}

# 每次从库里学：概率、掌握度
LEARN_P = 0.10
LEARN_V = 0.34


class LeapCivilization(GlobeCivilization):
    """球面文明 + 文化累积。"""

    def __init__(self, size=60, n_agents=150, seed=42,
                 food_rate=1.0, mutation_rate=0.06, verbose=False,
                 external_memory=True):
        super().__init__(size=size, n_agents=n_agents, seed=seed,
                         food_rate=food_rate, mutation_rate=mutation_rate,
                         verbose=verbose)
        self.external_memory = bool(external_memory)
        self.store_learned = 0          # 从库里学到的次数（累计）
        self.store_write = 0            # 写进库的次数

    # ---------- 这个部落有没有"记录" ----------
    def has_record(self, tribe):
        """部落里有没有人掌握「能留存的信息载体」。

        看属性不看名字：只要有人的某个概念同时含 信息 和 可储 就算。
        所以「记录」「文字」都行，小沙自己长出来的别的名字也行。
        """
        if tribe is None:
            return False
        hit = has_props(tribe.knowledge, RECORD_PROPS)
        if hit is not None:
            return True
        for m in tribe.members:
            if has_props(m.knowledge, RECORD_PROPS) is not None:
                return True
        return False

    # ---------- 从库里学（新增的那一环）----------
    def _read_store(self):
        """有记录的部落：成员可以从公共知识池里学到东西。"""
        if not self.external_memory:
            return 0
        n = 0
        for t in self.tribes:
            if not self.has_record(t) or not t.knowledge:
                continue
            keys = list(t.knowledge.keys())
            for a in t.members:
                if a not in self.agents:
                    continue
                if len(a.knowledge) >= MAX_HELD:
                    continue
                if self.rng.rand() >= LEARN_P:
                    continue
                k = keys[self.rng.randint(len(keys))]
                if a.knowledge.get(k, 0.0) >= t.knowledge[k] * 0.9:
                    continue
                a.knowledge[k] = float(min(0.95, t.knowledge[k] * 0.8))
                n += 1
        self.store_learned += n
        return n

    # ---------- 用库里的知识做组合（真正的棘轮）----------
    def _maybe_discover(self, agent):
        """有记录的部落：可以拿"部落库里存的"知识去组合，不必自己记住。

        这是整套机制的关键一步。第一版我只做了"从库里学"（_read_store），
        A/B 跑出来概念种类两边都是 28 —— 因为技术空间本来就只有 28 个，
        人均已经会 21 个，多记几个也没有新东西可学。瓶颈不在记忆，
        在**能同时摆在桌面上的知识量**。

        真实的文字/账本起的作用正是这个：一个人不需要把整个文明的
        知识背下来，也能在别人的成果上继续组合。所以这里让有记录的
        部落用「个人知识 ∪ 部落库」当组合池 —— 组合对的数量级一下子
        大了一个档次，天花板才有可能被顶开。
        """
        from xiaosha_concepts import Concept  # 局部导入，避免顶层循环依赖

        pool = dict(agent.knowledge)
        if self.external_memory and agent.tribe is not None and self.has_record(agent.tribe):
            for k, v in agent.tribe.knowledge.items():
                if v >= SKILL_GATE and k not in pool:
                    pool[k] = v

        skilled = [k for k, v in pool.items() if v >= SKILL_GATE]
        if len(skilled) < 2:
            return
        p = 0.004 * (1.0 + min(3.0, len(skilled) / 4.0))
        if self.rng.rand() >= p:
            return

        i = self.rng.randint(len(skilled))
        j = self.rng.randint(len(skilled) - 1)
        if j >= i:
            j += 1
        na, nb = skilled[i], skilled[j]
        ca, cb = get_concept(na), get_concept(nb)
        if ca is None or cb is None:
            return
        newc = combine(ca, cb)
        if newc is None:
            return
        tech = newc.name
        if tech in agent.knowledge:
            return
        if len(agent.knowledge) >= MAX_HELD:
            return
        if get_concept(tech) is None:
            register_concept(newc)
        agent.knowledge[tech] = 0.35 + self.rng.rand() * 0.2
        self.discoveries.append({"gen": self.generation, "step": self.step_count,
                                 "who": agent.name, "tech": tech, "from": (na, nb),
                                 "via_store": na not in agent.knowledge or nb not in agent.knowledge})
        if self.verbose:
            print("      ✦ %s 组合出了「%s」（%s + %s）" % (agent.name, tech, na, nb))

    def step(self):
        super().step()
        self._read_store()
        # 统计：库里有多少东西（用于报告）
        self.store_write = sum(len(t.knowledge) for t in self.tribes)


# ==================== 对照实验 ====================
def run_arm(external_memory, gens=30, size=80, pop=250, seed=20261121, verbose=True):
    np.random.seed(seed)
    civ = LeapCivilization(size=size, n_agents=pop, seed=seed,
                           verbose=False, external_memory=external_memory)
    rows = []
    for g in range(gens):
        civ.generation = g
        for _ in range(60):
            civ.step()
        if verbose and (g % 5 == 0 or g == gens - 1):
            rows.append((g, snapshot(civ)))
            print("  %2d 代: %s" % (g, fmt(snapshot(civ))))
    return civ, rows


def snapshot(civ):
    allk = set()
    per = []
    for a in civ.agents:
        allk |= set(a.knowledge)
        per.append(len(a.knowledge))
    return {
        "pop": len(civ.agents),
        "kinds": len(allk),
        "per_agent": float(np.mean(per)) if per else 0.0,
        "food": civ.world.total_food(),
        "disc": len(civ.discoveries),
        "tribes": len(civ.tribes),
        "with_record": sum(1 for t in civ.tribes if civ.has_record(t)),
        "store": sum(len(t.knowledge) for t in civ.tribes),
        "learned": getattr(civ, "store_learned", 0),
    }


def fmt(s):
    return ("人口%3d 概念%2d种 人均%.1f 食物%6.0f 部落%2d 有记录%d 库%2d 学到%d"
            % (s["pop"], s["kinds"], s["per_agent"], s["food"],
               s["tribes"], s["with_record"], s["store"], s["learned"]))


def experiment(gens=30, size=80, pop=250, seed=20261121):
    print("=" * 92)
    print("小沙的第一次大进化 —— 文化累积（A/B 对照，同种子 %d）" % seed)
    print("=" * 92)
    print("\n【对照组】知识只活在活人脑子里（external_memory=False）")
    civ_a, _ = run_arm(False, gens=gens, size=size, pop=pop, seed=seed)
    a = snapshot(civ_a)

    print("\n【大进化】部落有了「记录」，知识池能反过来教人")
    civ_b, _ = run_arm(True, gens=gens, size=size, pop=pop, seed=seed)
    b = snapshot(civ_b)

    print("\n" + "=" * 92)
    print("结果对比（第 %d 代）" % (gens - 1))
    print("=" * 92)
    print("  %-14s %10s %10s %10s" % ("指标", "对照", "大进化", "变化"))
    for key, name, fmt_s in (
            ("kinds", "概念种类", "%d"),
            ("per_agent", "人均概念数", "%.2f"),
            ("food", "食物总量", "%.0f"),
            ("pop", "人口", "%d"),
            ("learned", "从库学到次数", "%d")):
        va, vb = a[key], b[key]
        if key == "per_agent":
            delta = "%+.1f%%" % ((vb - va) / max(1e-9, va) * 100)
        else:
            delta = "%+d" % (vb - va) if isinstance(va, int) else "%+.0f" % (vb - va)
        print("  %-14s %10s %10s %10s"
              % (name, fmt_s % va, fmt_s % vb, delta))
    print("\n  ★ 有记录的部落数: %d / %d" % (b["with_record"], b["tribes"]))
    print("  ★ 知识库里存着 %d 项（对照组的池子只写不读，等于死的）" % b["store"])
    return a, b


# ==================== 自测 ====================
def selftest():
    print("  小沙大进化 selftest（记录判定 / 库能教人 / A/B 有差异）")
    ok, total = 0, 0

    # 1. 没记录时，库里东西不会回流
    total += 1
    np.random.seed(5)
    c0 = LeapCivilization(size=40, n_agents=80, seed=5, verbose=False,
                          external_memory=False)
    for g in range(3):
        c0.generation = g
        for _ in range(60):
            c0.step()
    print("  [1] 对照组：从库学到 %d 次（应为 0）" % c0.store_learned)
    ok += int(c0.store_learned == 0)

    # 2. 有记录 + 库里有东西 → 会发生学习
    total += 1
    np.random.seed(5)
    c1 = LeapCivilization(size=40, n_agents=80, seed=5, verbose=False,
                          external_memory=True)
    for g in range(3):
        c1.generation = g
        for _ in range(60):
            c1.step()
    print("  [2] 大进化：从库学到 %d 次（应 > 0）" % c1.store_learned)
    ok += int(c1.store_learned > 0)

    # 3. 记录的判定看属性，不看名字
    total += 1

    class FakeT:
        def __init__(self, k):
            self.knowledge = k
            self.members = []

    with_rec = c1.has_record(FakeT({"记录": 0.6}))
    fake_rec = c1.has_record(FakeT({"羊皮卷": 0.6}))       # 不在名字表里
    no_rec = c1.has_record(FakeT({"石头": 0.9}))
    # 让"羊皮卷"这个自造概念带上 信息+可储 属性
    from xiaosha_concepts import Concept, register
    register(Concept("羊皮卷", ["信息", "可储"]))
    fake_rec2 = c1.has_record(FakeT({"羊皮卷": 0.6}))
    print("  [3] 记录: 记录=%.0f 石头=%s 羊皮卷(无属性)=%s 羊皮卷(信息+可储)=%s"
          % (with_rec, no_rec, fake_rec, fake_rec2))
    ok += int(with_rec and not no_rec and not fake_rec and fake_rec2)

    # 4. A/B 同种子跑，指标不应完全一样
    total += 1
    a = run_arm(False, gens=6, size=40, pop=80, seed=9, verbose=False)[0]
    b = run_arm(True, gens=6, size=40, pop=80, seed=9, verbose=False)[0]
    sa, sb = snapshot(a), snapshot(b)
    print("  [4] 6 代 A/B: 概念 %d vs %d | 人均 %.2f vs %.2f | 学到 %d vs %d"
          % (sa["kinds"], sb["kinds"], sa["per_agent"], sb["per_agent"],
             sa["learned"], sb["learned"]))
    ok += int(sb["learned"] > 0 and sa["learned"] == 0)

    # 5. 大进化组的人均概念数不该更低
    total += 1
    print("  [5] 人均概念: 对照 %.2f → 大进化 %.2f" % (sa["per_agent"], sb["per_agent"]))
    ok += int(sb["per_agent"] >= sa["per_agent"] - 1e-9)

    print("\n  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    # 默认只跑自测（约 60 秒）。30 代 A/B 实验要显式开：
    #     python xiaosha_leap.py --exp
    # 这是本仓库的约定（make_civ_video / xiaosha_globe 都一样）：
    # 无参调用被测试 runner 用来做自检，不能顺带跑几分钟的长实验 ——
    # 实测无参时会跑 300 秒以上，直接把全量 pytest 拖超时。
    if "--exp" in sys.argv:
        ok = selftest()
        print()
        experiment()
        sys.exit(0 if ok else 1)
    sys.exit(0 if selftest() else 1)