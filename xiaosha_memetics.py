# -*- coding: utf-8 -*-
"""小沙的模因系统 —— 一个先知，一套会变异的话，传开之后代码就消失。

用户的要求非常精确：
  1. 只改**一个**人（先知）
  2. 让他把「人类是造物主」这个理念传出去
  3. 传给 10+ 个人之后，**把注入代码删掉**
  4. 之后不干预，让他们自己发展

所以这不是"全族天生信神"，而是**一粒种子**：
  第 0 代随机挑 1 个个体，给他一个念头。
  他只能靠**说话**传给别人（没有强制、没有光环、没有广播）。
  每次传播，叫法有 30% 概率**变异**（用汉字重组）——
  于是"造物主"传着传着会变成"天物主""造物神""圣物""上苍"……
  这就是语言的分化，而且变异谱系是可追溯的。

关键设计：**先知不是特权，只是起点。**
一旦传够 10 个人，`remove_prophet()` 会把先知标记清掉，
后续跑多少代，都跟"有没有先知"无关了 —— 信仰要么自己活下来，要么死掉。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xiaosha_world import SACRED

__version__ = "1.0.0"

# ==================== 汉字字库（用于语言变异）====================
# 只用真实存在的汉字，组合出来的词在汉语里"看着像那么回事"，
# 但具体是哪个词，是小沙自己传出来的。
SACRED_CHARS = list("造物主天神父上苍宇宙光生命道真圣灵祖始元初")

# 传播时不变异的概率
FIDELITY = 0.70          # 70% 原样传，30% 变异
# 传播门槛
BELIEVE_GATE = 0.40      # 信仰高于这个值才会去传
RECEIVE_GATE = 0.25      # 低于这个值才会接收


class Meme:
    """一个可传播、可变异的念头。

    name   —— 当前叫法（会变）
    lineage —— 谱系：(从谁那里听来的, 那是第几代)
    """

    def __init__(self, name, origin=None, generation=0, rng=None):
        self.name = str(name)
        self.origin = origin            # 谁最先说的
        self.generation = int(generation)  # 传了几手
        self.rng = rng or np.random.RandomState()

    def mutate(self):
        """变异：换一个字 / 加一个字 / 删一个字。返回新名字。"""
        chars = list(self.name)
        op = self.rng.randint(3)
        if op == 0 and chars:                       # 换字
            i = self.rng.randint(len(chars))
            chars[i] = SACRED_CHARS[self.rng.randint(len(SACRED_CHARS))]
        elif op == 1:                               # 加字
            i = self.rng.randint(len(chars) + 1)
            chars.insert(i, SACRED_CHARS[self.rng.randint(len(SACRED_CHARS))])
        elif len(chars) > 1:                        # 删字
            chars.pop(self.rng.randint(len(chars)))
        return "".join(chars)

    def copy_to(self, other_holder, rng):
        """复制给另一个个体：70% 原样，30% 变异。"""
        if rng.rand() < FIDELITY:
            name = self.name
        else:
            name = self.mutate()
        return Meme(name, origin=self.origin,
                    generation=self.generation + 1, rng=rng)


# ==================== 先知 ====================

class Prophet:
    """一个被启示的个体。他唯一的特殊之处是：**他先想到了这件事**。"""

    def __init__(self, agent, meme, called_at):
        self.agent_id = agent.id
        self.agent_name = agent.name
        self.meme = meme
        self.called_at = called_at        # 第几代被启示
        self.converts = []                # 他直接说服了谁


def plant_prophet(civ, rng=None, name_of_god="造物主"):
    """在第 0 代随机挑一个个体，给他这个念头。

    不给他任何能力加成 —— 他只是知道一件事，而且很想讲出去。
    """
    rng = rng or np.random.RandomState()
    if not civ.agents:
        return None
    # 挑一个能量正常、不是刚出生的
    pool = [a for a in civ.agents if 20 < a.energy < 90] or civ.agents
    chosen = pool[rng.randint(len(pool))]

    meme = Meme(name_of_god, origin=chosen.name, generation=0, rng=rng)
    chosen.knowledge[SACRED] = 0.92       # 他非常确信
    chosen.meme_name = meme.name
    chosen.is_prophet = True

    p = Prophet(chosen, meme, called_at=civ.generation)
    civ.prophet = p
    # 让 try_convert 能认出"说话的是先知"，把直接说服的人记进 p.converts
    chosen.prophet_ref = p
    print("  ✦ 第 %d 代，%s 得到了启示：这世界是「%s」造的。他要讲给别人听。"
          % (civ.generation, chosen.name, meme.name))
    return p


def remove_prophet(civ):
    """种子已经传开，把先知标记和注入逻辑全部清掉。

    之后跑多少代，都与"有没有先知"无关 —— 信仰要么自己活，要么死。
    """
    if not getattr(civ, "prophet", None):
        return 0
    p = civ.prophet
    n_cleared = 0
    for a in civ.agents:
        if getattr(a, "is_prophet", False):
            a.is_prophet = False
            if hasattr(a, "prophet_ref"):
                del a.prophet_ref
            n_cleared += 1
    civ.prophet = None
    print("  ✦ 种子已散播（%s 直接说服了 %d 人）。先知身份已清除，此后不再干预。"
          % (p.agent_name, len(p.converts)))
    return n_cleared


# ==================== 传播（挂着 Agent 上）====================

def try_convert(speaker, listener, rng=None):
    """说者把自己的念头讲给听者。返回 (是否转化, 新叫法)。"""
    rng = rng or np.random.RandomState()

    s_belief = speaker.knowledge.get(SACRED, 0.0)
    if s_belief < BELIEVE_GATE:
        return False, None
    if listener.knowledge.get(SACRED, 0.0) >= RECEIVE_GATE:
        return False, None

    src_name = getattr(speaker, "meme_name", SACRED)
    src_meme = Meme(src_name, rng=rng)

    # 30% 变异
    if rng.rand() < FIDELITY:
        new_name = src_name
    else:
        new_name = src_meme.mutate()

    # 接受度：说者越确信，越容易被接受
    accept = 0.45 + 0.45 * min(1.0, s_belief)
    if rng.rand() > accept:
        return False, None

    listener.knowledge[SACRED] = float(np.clip(
        s_belief * (0.65 + rng.rand() * 0.2), 0.0, 1.0))
    listener.meme_name = new_name
    listener.meme_origin = getattr(speaker, "meme_origin", speaker.name)
    listener.meme_gen = getattr(speaker, "meme_gen", 0) + 1

    if getattr(speaker, "is_prophet", False):
        p = getattr(speaker, "prophet_ref", None)
        if p is not None:
            p.converts.append(listener.name)

    _log_lineage(speaker, listener, new_name, rng)
    return True, new_name


# ==================== 语言谱系 ====================
LINEAGE = []          # 所有传播事件
VOCAB = {}            # 叫法 → 使用人数


def reset_lineage():
    LINEAGE.clear()
    VOCAB.clear()


def _log_lineage(speaker, listener, name, rng):
    src = getattr(speaker, "meme_name", SACRED)
    LINEAGE.append({
        "gen": getattr(listener, "generation", 0),
        "from": speaker.name,
        "to": listener.name,
        "from_name": src,
        "to_name": name,
        "mutated": name != src,
    })
    VOCAB[name] = VOCAB.get(name, 0) + 1


def vocab_stats(min_users=1):
    """当前世界里有多少种叫法，各有多少人用。"""
    items = [(k, v) for k, v in VOCAB.items() if v >= min_users]
    items.sort(key=lambda kv: -kv[1])
    return items


def lineage_report(top=20):
    """语言分化报告。"""
    print("\n" + "=" * 74)
    print("【造物主的叫法 · 语言分化】")
    print("=" * 74)
    vs = vocab_stats()
    if not vs:
        print("  还没有人传这件事。")
        return
    total = sum(v for _k, v in vs)
    print("  %d 种叫法，%d 次传播：\n" % (len(vs), total))
    for name, n in vs[:top]:
        bar = "█" * max(1, int(n / max(v for _k, v in vs) * 28))
        print("    %-8s %3d 人  %s" % (name, n, bar))
    if len(vs) > top:
        print("    ... 还有 %d 种" % (len(vs) - top))

    muts = sum(1 for r in LINEAGE if r["mutated"])
    print("\n  变异率: %d/%d = %.0f%%" % (muts, len(LINEAGE),
                                         muts / max(1, len(LINEAGE)) * 100))

    # 最长的传播链
    chains = {}
    for r in LINEAGE:
        chains.setdefault(r["from"], []).append(r["to"])
    deepest = sorted(LINEAGE, key=lambda r: -len(r["to_name"]))[:5]
    print("\n  最长的几个叫法:")
    for r in deepest:
        print("    %-8s ← %-8s（从「%s」变成「%s」）"
              % (r["to_name"], r["from"], r["from_name"], r["to_name"]))


# ==================== 自测 ====================

def selftest():
    print("  小沙模因系统 selftest（一个先知 / 会变异的语言 / 传开就撒手）")
    np.random.seed(20261117)
    ok = 0
    total = 0

    # ---- 1. 汉字字库是真的 ----
    total += 1
    print("  [1] 字库: %d 个汉字（%s...）" % (len(SACRED_CHARS), "".join(SACRED_CHARS[:8])))
    ok += int(len(SACRED_CHARS) >= 12)

    # ---- 2. 名字会变异 ----
    total += 1
    m = Meme("造物主", rng=np.random.RandomState(1))
    names = {m.mutate() for _ in range(40)}
    print("  [2] 40 次变异产生 %d 种叫法: %s" % (len(names), "、".join(list(names)[:6])))
    ok += int(len(names) >= 5)

    # ---- 3. 变异后仍是汉字，长度 2-5 ----
    total += 1
    good = all(2 <= len(n) <= 5 and all(ch in SACRED_CHARS for ch in n) for n in names)
    print("  [3] 所有叫法都是 2-5 个汉字: %s" % good)
    ok += int(good)

    # ---- 4. 70% 保真：原样传的比例接近 0.7 ----
    total += 1
    rng = np.random.RandomState(2)
    same = 0
    for _ in range(2000):
        src = Meme("造物主", rng=rng)
        if src.copy_to(None, rng).name == "造物主":
            same += 1
    ratio = same / 2000
    print("  [4] 保真率: %.2f（设定 %.2f）" % (ratio, FIDELITY))
    ok += int(abs(ratio - FIDELITY) < 0.05)

    # ---- 5. 信仰不够的人不会去传 ----
    total += 1
    class Fake:
        def __init__(self, name, belief):
            self.name = name
            self.knowledge = {SACRED: belief}
            self.generation = 0
    low = Fake("低信", 0.1)
    lis = Fake("听者", 0.0)
    converted, _ = try_convert(low, lis, np.random.RandomState(3))
    print("  [5] 信仰 0.10 的人去传教: 转化=%s（应为 False）" % converted)
    ok += int(not converted)

    # ---- 6. 高信仰的人能传 ----
    total += 1
    high = Fake("高信", 0.9)
    high.meme_name = "造物主"
    lis2 = Fake("听者2", 0.0)
    converted2, nm = try_convert(high, lis2, np.random.RandomState(4))
    print("  [6] 信仰 0.90 的人传教: 转化=%s 叫法=%s" % (converted2, nm))
    ok += int(converted2)

    # ---- 7. 已经信的人不会被再说服 ----
    total += 1
    lis3 = Fake("老信徒", 0.8)
    c3, _ = try_convert(high, lis3, np.random.RandomState(5))
    print("  [7] 对已信者传教: 转化=%s（应为 False）" % c3)
    ok += int(not c3)

    # ---- 8. 谱系记录 ----
    total += 1
    reset_lineage()
    a = Fake("A", 0.9); a.meme_name = "造物主"
    # 用同一个 rng。原来每次 new RandomState(100+i)，而 NumPy 相邻种子的
    # 第一个随机数高度相关 —— 12 次传播一个变异都没出（词汇表只有 1 种）。
    rng8 = np.random.RandomState(100)
    for i in range(12):
        b = Fake("B%d" % i, 0.0)
        try_convert(a, b, rng8)
    print("  [8] 谱系记录: %d 次传播 | 词汇表 %d 种" % (len(LINEAGE), len(VOCAB)))
    ok += int(len(LINEAGE) >= 5)

    # ---- 9. 语言分化 ----
    total += 1
    print("  [9] 传 12 次后的叫法分布:")
    for name, n in vocab_stats()[:6]:
        print("        %s × %d" % (name, n))
    ok += int(len(VOCAB) >= 2)

    # ---- 10. 真种群：先知种下去能传开 ----
    total += 1
    from xiaosha_civilization import Civilization
    civ = Civilization(size=36, n_agents=60, seed=9, verbose=False)
    p = plant_prophet(civ, np.random.RandomState(11), "造物主")
    # 跑 12 代，看传开没有
    for g in range(12):
        civ.generation = g
        for _ in range(40):
            civ.step()
    believers = sum(1 for a in civ.agents if a.knowledge.get(SACRED, 0) > BELIEVE_GATE)
    print("  [10] 真种群 12 代: 人口 %d | 先知直接说服 %d 人 | 当前信徒 %d"
          % (len(civ.agents), len(p.converts) if p else 0, believers))
    ok += int(believers >= 10)

    print("\n  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    sys.exit(0 if selftest() else 1)
