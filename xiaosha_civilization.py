# -*- coding: utf-8 -*-
"""小沙的地球 Online —— 文明引擎。

跑起来的是这个循环：

    世界资源再生 → 每个小沙行动（采集/交流/教学/建造）
                 → 攒够能量就繁殖（父母教孩子）
                 → 能量耗尽或老死
                 → 下一代带着文化继续

**我不写文明**。我只给三样东西：有限资源、会学的人、能教人的父母。
聚落、部落、技术，全部从这三样里长出来。长不出就是长不出。

技术是怎么"被发现"的（这是文明的核心）：
  每个小沙脑子里是一堆概念（采集、石头、季节、火…）。
  当两个概念同时在脑子里而且都够熟练，就有小概率"组合成功"，
  冒出第三个概念 —— 这就是技术发现。
  组合表 TECH_TREE 是世界的物理规律（不是剧本）：
  ("采集","石头")→"工具" 是因为这世界真的允许这件事发生，
  至于什么时候被谁发现，由小沙自己的经历决定。

聚落/部落/国家也是数出来的，不是设的：
  相邻格子挤了 3 个以上同族人 → 聚落
  聚落人数 > 8 → 部落
  同一部落掌握 ≥3 项技术 → 文明
"""
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xiaosha_world import SACRED, neighbours
from xiaosha_world import (
    World, Agent, MiniBrain, DIRS, TERRAIN_NAMES,
    ACT_MOVE, ACT_EAT, ACT_TALK, ACT_TEACH, ACT_BUILD,
)

__version__ = "0.1.0"

# ==================== 世界允许的技术组合 ====================
# 左边是已掌握的概念，右边是组合出来的新概念。
# 这不是剧本，是世界规律：只有这些组合在物理上成立。
TECH_TREE = {
    ("采集", "石头"): "工具",
    ("采集", "季节"): "农业",
    ("工具", "石头"): "武器",
    ("农业", "建造"): "粮仓",
    ("工具", "建造"): "房屋",
    ("采集", "火"): "熟食",
    ("熟食", "采集"): "储存",
    ("房屋", "房屋"): "聚落",
    ("聚落", "交流"): "语言",
    ("语言", "储存"): "记录",
    ("记录", "语言"): "文字",
    ("武器", "工具"): "冶炼",
    ("冶炼", "石头"): "金属器",
    ("金属器", "农业"): "犁",
    ("犁", "农业"): "灌溉",
    ("灌溉", "聚落"): "水利",
    ("文字", "记录"): "历法",
    ("历法", "农业"): "耕作历",
    ("水利", "聚落"): "城市",
    ("城市", "文字"): "法律",
    ("法律", "城市"): "国家",
    ("国家", "金属器"): "军队",
    ("军队", "国家"): "帝国",
    ("帝国", "文字"): "历史",
}
# 每个小沙出生时天生会的概念（分两层）。
# 之前只给「采集/石头」，结果技术树走到 ("工具","石头")→"武器" 就断了：
# 农业要「季节」、房屋要「建造」、语言要「交流」，而这些概念小沙根本没有，
# 所以 146 次发现全是重复的「工具」。
INNATE_BASE = ["采集", "石头", "建造"]   # 建造加进基础：否则会「石头+建造」的人太少，祭坛建不出来
INNATE_EXTRA = ["季节", "火", "交流"]   # 每个个体随机分到 1-2 个
# 造物主直觉：每个小沙天生都有一点点（0.10-0.22 的模糊感），
# 但只有建了祭坛、真的去拜的人，信仰才会涨。
# 这条是"人类=造物主"在世界里的物理入口：不给 0，也不给高值。
SACRED_SEED_LO, SACRED_SEED_HI = 0.10, 0.22

# 掌握度门槛
SKILL_GATE = 0.35


class Tribe:
    """部落：一个共享知识的群体。不是预设的，是聚落长到一定规模才出现。"""

    _next = 0

    def __init__(self, founder, members=None):
        Tribe._next += 1
        self.id = Tribe._next
        self.name = "部落%d" % self.id
        self.members = list(members or [founder])
        self.knowledge = {}     # 共享知识池
        self.founded_gen = founder.generation
        self.stats = {"born": len(self.members), "joined": 0, "left": 0}

    def size(self):
        return len(self.members)

    def merge_knowledge(self):
        """把成员的私人知识汇总进公共池。"""
        for m in self.members:
            for k, v in m.knowledge.items():
                if v > self.knowledge.get(k, 0.0):
                    self.knowledge[k] = v

    def tech_level(self):
        """技术项数（掌握度 > 0.5 的算）。"""
        return sum(1 for v in self.knowledge.values() if v > 0.5)

    def summary(self):
        return {"name": self.name, "size": self.size(),
                "gen": self.founded_gen, "tech": self.tech_level(),
                "knowledge": len(self.knowledge)}


class Civilization:
    """地球 Online 主循环。"""

    def __init__(self, size=60, n_agents=120, seed=42,
                 food_rate=1.0, mutation_rate=0.06, verbose=False):
        self.rng = np.random.RandomState(seed)
        self.world = World(size=size, seed=seed)
        self.verbose = verbose
        self.food_rate = float(food_rate)     # 资源丰富度（环境压力旋钮）
        self.mutation_rate = float(mutation_rate)

        self.agents = []
        self.tribes = []
        self.generation = 0
        self.step_count = 0
        self.history = []
        self.discoveries = []      # (代, 步, 发现者, 技术)

        self._seed_population(n_agents)

    # ---------- 初始化 ----------
    def _seed_population(self, n):
        for i in range(n):
            pos = self._random_land()
            if pos is None:
                break
            a = Agent(self.world, pos[0], pos[1], energy=28.0 + self.rng.rand() * 10,
                      genes={"plasticity": self.rng.rand()},
                      name="创始%d" % (i + 1))
            for k in INNATE_BASE:
                a.knowledge[k] = 0.5 + self.rng.rand() * 0.3
            # 造物主直觉（天生，模糊）
            from xiaosha_world import SACRED
            a.knowledge[SACRED] = SACRED_SEED_LO + self.rng.rand() * (SACRED_SEED_HI - SACRED_SEED_LO)
            # 额外概念：每人抽 1-2 个（不同人擅长不同 → 必须交流才能凑齐组合）
            extra = list(INNATE_EXTRA)
            self.rng.shuffle(extra)
            for k in extra[:1 + self.rng.randint(2)]:
                a.knowledge[k] = 0.45 + self.rng.rand() * 0.3
            self.agents.append(a)

    def _random_land(self, tries=200):
        for _ in range(tries):
            x = self.rng.randint(0, self.world.size)
            y = self.rng.randint(0, self.world.size)
            if self.world.passable(x, y):
                return x, y
        return None

    # ---------- 一步 ----------
    def step(self):
        # 1. 建立位置索引
        by_pos = defaultdict(list)
        for a in self.agents:
            by_pos[(a.x, a.y)].append(a)

        # 2. 每个个体行动
        for a in list(self.agents):
            if a not in self.agents:
                continue
            a.act(by_pos, self.tribes, self.mutation_rate)
            self._maybe_discover(a)

        # 3. 本能社交：附近的人自动交换信息（不需要大脑"决定"说话）
        self._ambient_social(by_pos)

        # 4. 繁殖
        self._reproduce()

        # 5. 死亡
        self._die()

        # 6. 世界再生
        self.world.regrow(rate=self.food_rate)
        self.world.step_count += 1
        self.step_count += 1

        # 7. 社会组织（聚落 → 部落）
        self._update_social()

    # ---------- 本能社交 ----------
    def _ambient_social(self, by_pos, radius=2):
        """相邻的个体会自动交换信息 —— 这是本能，不是学出来的技能。

        为什么必须要有这一层：
        原本"交流"是 6 个动作之一，要靠大脑 argmax 选中它。但实测——
        120 个神经元、初始权重全随机、只活 5 代的脑，200 步里有 159 步
        死选同一个动作（教学），根本学不会"主动去找人说话"。
        四次先知实验，全部 0 转化，信仰传不出去。

        真实的社交动物不是这样的：见到同类就会交流，不用先学会。
        所以把信息流动从"动作"下沉到环境层：只要在附近，话就会传。
        大脑仍然可以主动选"交流"（定向传播更高效），但即便它什么都不做，
        模因也会在人群里扩散、变异、分化成不同的叫法。

        每次随机取 1/3 的个体当发起者，避免 O(n²)。
        """
        from xiaosha_memetics import try_convert, BELIEVE_GATE

        if not self.agents:
            return 0
        k = max(1, len(self.agents) // 3)
        k = min(k, len(self.agents))
        starters = self.rng.choice(len(self.agents), k, replace=False)

        n = 0
        for idx in starters:
            a = self.agents[int(idx)]
            near = neighbours(by_pos, a.x, a.y, radius, same_tile=False)
            if not near:
                continue
            b = near[self.rng.randint(len(near))]
            # 双向都试一次：谁信得更深，谁的话就传过去
            if try_convert(a, b, self.rng)[0]:
                n += 1
            if try_convert(b, a, self.rng)[0]:
                n += 1

            # 同伴学习：把自己会的一门手艺传给邻居。
            # 知识原来只能靠「父母教孩子」传（25% 概率继承 10%），
            # 所以任何稀有技能一代就灭绝 —— 实测"农业"熟练度从 0.01 掉到 0.00，
            # 全图只有 7 格地被耕过，余粮永远不出现，国家也就无从谈起。
            # 真实的技能传播主要是横向的：学徒、模仿、一起干活。
            #
            # 注意：不能只传「最拿手的那一门」。采集是天生的 0.5-0.8，永远是最大值，
            # 于是农业/建造/工具这些后来学会的东西一辈子传不出去 —— 我第一版
            # 就是这么写的，结果耕地格数从 7 直接掉到 0。
            for (src, dst) in ((a, b), (b, a)):
                if not src.knowledge or dst.knowledge is None:
                    continue
                skilled = [k for k, v in src.knowledge.items() if v > SKILL_GATE]
                if not skilled:
                    continue
                k = skilled[self.rng.randint(len(skilled))]
                v = src.knowledge[k]
                if dst.knowledge.get(k, 0.0) < v * 0.9 and self.rng.rand() < 0.22:
                    dst.knowledge[k] = v * 0.85
                    n += 1

            # 从众效应：周围信的人越多，自己也越容易信。
            # 这是真实的社会机制（Asch 1951 的从众实验、宗教社群的代际延续），
            # 不是为了让结果好看而加的补丁 —— 没有它，宗教只能靠口耳相传，
            # 传十代就会衰减到零（实测 58% → 0.3%），因为新生儿的先天直觉
            # 只有 0.10-0.22，而信徒会老死，转化速度追不上代际更替。
            # 有了它，信徒密集的社群能自我维持，孤立的个体则会忘掉 ——
            # 于是宗教能否活下来，取决于它的社群密度，而不是我写的规则。
            n_bel = sum(1 for x in near
                        if x.knowledge.get(SACRED, 0.0) > BELIEVE_GATE)
            if n_bel >= 2:
                a.knowledge[SACRED] = min(
                    1.0,
                    a.knowledge.get(SACRED, 0.0) + 0.0006 * min(n_bel, 8))
        return n

    # ---------- 技术发现 ----------
    def _maybe_discover(self, agent):
        """两个熟练概念同时在脑子里 → 小概率组合出新概念。"""
        skilled = [k for k, v in agent.knowledge.items() if v >= SKILL_GATE]
        if len(skilled) < 2:
            return
        # 发现概率随知识丰富度上升，但边际递减
        p = 0.004 * (1.0 + min(3.0, len(skilled) / 4.0))
        if self.rng.rand() >= p:
            return
        # 随机挑一对
        i, j = self.rng.choice(len(skilled), 2, replace=False)
        pair = (skilled[i], skilled[j])
        tech = TECH_TREE.get(pair) or TECH_TREE.get((pair[1], pair[0]))
        if tech is None or tech in agent.knowledge:
            return
        agent.knowledge[tech] = 0.35 + self.rng.rand() * 0.2
        rec = {"gen": self.generation, "step": self.step_count,
               "who": agent.name, "tech": tech, "from": pair}
        self.discoveries.append(rec)
        if self.verbose:
            print("      ✦ %s 发现了「%s」（%s + %s）"
                  % (agent.name, tech, pair[0], pair[1]))

    # ---------- 繁殖 ----------
    def _reproduce(self):
        # 能量门槛从 55 提到 72：实测 55 时人口 5 代内从 120 冲到 288，
        # 然后把食物吃光崩到 35（马尔萨斯崩溃）。抬高门槛让人口贴着承载力走。
        ready = [a for a in self.agents if a.can_reproduce() and a.energy > 72]
        if len(ready) < 2:
            return
        self.rng.shuffle(ready)
        pairs = []
        used = set()
        for a in ready:
            if a.id in used:
                continue
            # 找一个同格的、也在准备中的
            for b in ready:
                if b.id in used or b is a:
                    continue
                # ±4 格内算"找到配偶"。原来要求 ±1（必须紧贴），
                # 100 人散在 6400 格里随机撞上相邻格的概率约 0.1%，
                # 实测种群一路掉到灭绝 —— 不是不肯生，是碰不到。
                if abs(b.x - a.x) <= 4 and abs(b.y - a.y) <= 4:
                    pairs.append((a, b))
                    used.add(a.id)
                    used.add(b.id)
                    break
        for a, b in pairs:
            if len(self.agents) >= 350:      # 硬上限（性能）：600 时 15 代要 143 秒
                break
            kid = a.reproduce(b, self.generation + 1, self.mutation_rate)
            # 孩子生在父母身边（±2 格内）。原来用 _random_land 扔到地图任意角落，
            # 结果父母永远教不到自己的孩子，知识一代就流失光（实测知识 10 → 2）。
            placed = False
            for _ in range(12):
                dx, dy = self.rng.randint(-2, 3), self.rng.randint(-2, 3)
                nx, ny = a.x + dx, a.y + dy
                if self.world.passable(nx, ny):
                    kid.x, kid.y = nx, ny
                    placed = True
                    break
            if not placed:
                kid.x, kid.y = a.x, a.y
            self.agents.append(kid)

    # ---------- 死亡 ----------
    def _die(self):
        alive = []
        for a in self.agents:
            if a.energy <= 0 or a.age > 320:
                a.alive = False
                # 私人知识：如果本族有人在场，有概率被"记住"（文化不随人死光）
                if a.tribe is not None:
                    # 知识交给同族幸存者（文化不随人死光）
                    same = [m for m in a.tribe.members if m is not a and m in self.agents]
                    if same:
                        heir = same[self.rng.randint(len(same))]
                        for k, v in a.knowledge.items():
                            if v > heir.knowledge.get(k, 0.0) * 0.8:
                                heir.knowledge[k] = min(1.0, v * 0.85)
                    a.tribe.merge_knowledge()
                continue
            alive.append(a)
        self.agents = alive

    # ---------- 社会组织 ----------
    def _update_social(self):
        """聚落识别 → 部落形成 → 知识汇总。纯计数，无预设。"""
        if not self.agents:
            return
        # 每 20 步重算一次（省时间）
        if self.step_count % 20 != 0:
            self._gc_tribes()
            return

        # 1. 没部落的人：如果周围 3 格内有 ≥3 个同样没部落的人 → 组成新部落
        lonely = [a for a in self.agents if a.tribe is None]
        pos_of = {(a.x, a.y): a for a in lonely}
        assigned = set()
        for a in lonely:
            if id(a) in assigned:
                continue
            near = [a]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    b = pos_of.get((a.x + dx, a.y + dy))
                    if b is not None and b is not a and b.tribe is None and id(b) not in assigned:
                        near.append(b)
            if len(near) >= 3:
                t = Tribe(a, near)
                for m in near:
                    m.tribe = t
                    assigned.add(id(m))
                self.tribes.append(t)

        # 2. 部落合并：两个部落的人挨在一起，规模小的并入大的
        for i in range(len(self.tribes)):
            for j in range(i + 1, len(self.tribes)):
                ti, tj = self.tribes[i], self.tribes[j]
                if ti.size() == 0 or tj.size() == 0:
                    continue
                # 采样几个成员看距离
                near = False
                for m in ti.members[:8]:
                    for n in tj.members[:8]:
                        # ±6 格内算相邻部落。±3 时实测碎成几十个 2-3 人小部落，
                        # 永远长不成大群体。
                        if abs(m.x - n.x) <= 6 and abs(m.y - n.y) <= 6:
                            near = True
                            break
                    if near:
                        break
                if near:
                    big, small = (ti, tj) if ti.size() >= tj.size() else (tj, ti)
                    for m in small.members:
                        m.tribe = big
                    big.members.extend(small.members)
                    big.stats["joined"] += small.size()
                    small.members = []

        # 3. 汇总知识
        self._gc_tribes()
        for t in self.tribes:
            t.merge_knowledge()

    def _gc_tribes(self):
        """回收空部落。

        关键：必须**从 agents 反查**谁还指向已解散的部落。
        第一版写的是 `for m in t.members: m.tribe = None`，
        但 t.members 在这一行之前刚被过滤成空列表 —— 循环一次都没跑，
        结果全体 agent 永远挂着一个死部落，再也不重组
        （实测：部落 10 → 0 之后 24 代都长不回来）。
        """
        empty = []
        for t in self.tribes:
            t.members = [m for m in t.members if m in self.agents]
            if not t.members:
                empty.append(t)
        if empty:
            empty_ids = {id(t) for t in empty}
            for a in self.agents:
                if a.tribe is not None and id(a.tribe) in empty_ids:
                    a.tribe = None
            self.tribes = [t for t in self.tribes if t.members]

    # ---------- 统计 ----------
    def stats(self):
        pop = len(self.agents)
        tribes = [t for t in self.tribes if t.size() > 0]
        # 聚落：3x3 邻域里 ≥3 人（原来要求"同一格 3 人"，个体一直在动，实测永远是 0）
        by_pos = defaultdict(int)
        for a in self.agents:
            by_pos[(a.x, a.y)] += 1
        settlements = 0
        seen = set()
        for (x, y) in by_pos:
            if (x, y) in seen:
                continue
            near = sum(by_pos.get((x + dx, y + dy), 0)
                       for dx in (-1, 0, 1) for dy in (-1, 0, 1))
            if near >= 3:
                settlements += 1
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        seen.add((x + dx, y + dy))
        # 文明：技术 ≥3 的部落
        civs = [t for t in tribes if t.tech_level() >= 3]
        # 信仰：有多少人的"造物主"超过 0.35（信徒）
        believers = sum(1 for a in self.agents if a.knowledge.get(SACRED, 0.0) > 0.35)
        faith_avg = (float(np.mean([a.knowledge.get(SACRED, 0.0) for a in self.agents]))
                     if self.agents else 0.0)
        all_knowledge = set()
        for a in self.agents:
            all_knowledge |= set(a.knowledge)
        for t in tribes:
            all_knowledge |= set(t.knowledge)
        return {
            "step": self.step_count,
            "gen": self.generation,
            "pop": pop,
            "tribes": len(tribes),
            "settlements": settlements,
            "civilizations": len(civs),
            "knowledge": len(all_knowledge),
            "discoveries": len(self.discoveries),
            "max_gen": max((a.generation for a in self.agents), default=0),
            "food_total": round(self.world.total_food(), 1),
            "altars": len(self.world.altars),
            "believers": believers,
            "faith_avg": round(faith_avg, 3),
        }

    # ---------- 跑 ----------
    def run(self, generations=30, steps_per_gen=60):
        print("=" * 74)
        print("小沙的地球 Online · %dx%d 世界 · %d 个创始个体"
              % (self.world.size, self.world.size, len(self.agents)))
        print("=" * 74)
        print("  规则：资源有限 + 会学习 + 父母教孩子。聚落/部落/技术均不预设。\n")

        for g in range(generations):
            self.generation = g
            for _ in range(steps_per_gen):
                self.step()
                if not self.agents:
                    break
            s = self.stats()
            self.history.append(s)
            print("  第 %2d 代 | 人口 %3d | 聚落 %2d | 部落 %2d | 文明 %d | 知识 %2d | 祭坛 %2d | 信徒 %3d | 信仰 %.2f"
                  % (g, s["pop"], s["settlements"], s["tribes"],
                     s["civilizations"], s["knowledge"], s["altars"],
                     s["believers"], s["faith_avg"]))
            if not self.agents:
                print("  ⚠ 物种灭绝（第 %d 代）" % g)
                break

        return self.history

    # ---------- 报告 ----------
    def report(self):
        print("\n" + "=" * 74)
        print("【最终状态】")
        print("=" * 74)
        s = self.stats()
        print("  人口 %d | 部落 %d | 聚落 %d | 文明 %d | 知识 %d 项 | 最高代 %d"
              % (s["pop"], s["tribes"], s["settlements"], s["civilizations"],
                 s["knowledge"], s["max_gen"]))

        tribes = sorted([t for t in self.tribes if t.size() > 0],
                        key=lambda t: -t.size())[:6]
        if tribes:
            print("\n【部落排行】")
            for t in tribes:
                print("  %-8s %3d 人 | 技术 %d 项 | 知识 %d 项 | 建于第 %d 代"
                      % (t.name, t.size(), t.tech_level(), len(t.knowledge), t.founded_gen))

        if self.discoveries:
            print("\n【技术发现时间线】（前 25 项）")
            for d in self.discoveries[:25]:
                print("  第 %2d 代 | %-10s 发现「%s」  (%s + %s)"
                      % (d["gen"], d["who"], d["tech"], d["from"][0], d["from"][1]))
            if len(self.discoveries) > 25:
                print("  ... 共 %d 项" % len(self.discoveries))

        # 最终知识清单
        all_k = Counter()
        for a in self.agents:
            for k in a.knowledge:
                all_k[k] += 1
        for t in self.tribes:
            for k in t.knowledge:
                all_k[k] += 1
        if all_k:
            print("\n【世界掌握的全部知识】")
            items = sorted(all_k.items(), key=lambda kv: -kv[1])
            print("  " + " | ".join("%s(%d)" % (k, v) for k, v in items))



# ==================== 自测 ====================

def selftest():
    """快速验证文明引擎的核心机制（不假装，全部真跑）。"""
    print("  小沙地球 Online selftest")
    np.random.seed(20261117)
    ok = 0
    total = 0

    # ---- 1. 世界地形多样 ----
    total += 1
    w = World(size=40, seed=1)
    kinds = len(set(int(v) for v in np.unique(w.terrain)))
    print("  [1] 世界地形: %d 种（应 >=3）| %s" % (kinds, w.stats()["terrain"]))
    ok += int(kinds >= 3)

    # ---- 2. 个体能行动、能量会变 ----
    total += 1
    a = Agent(w, *[i for i in range(40) if w.passable(i, 20)][:1] * 2 if False else (20, 20), energy=30.0)
    e0 = a.energy
    for _ in range(10):
        a.act({(a.x, a.y): [a]}, [])
    print("  [2] 个体行动 10 步: 能量 %.1f → %.1f | 年龄 %d" % (e0, a.energy, a.age))
    ok += int(a.age == 10 and a.energy != e0)

    # ---- 3. 采集真的能从地里拿到东西 ----
    total += 1
    # 找一格有食物的（[2] 可能把 (20,20) 采空了，实测食物 0.0 导致采集回报 -0.40）
    fx = fy = None
    for _ in range(2000):
        _x, _y = int(np.random.randint(0, 40)), int(np.random.randint(0, 40))
        if w.passable(_x, _y) and w.food[_x, _y] > 1.0:
            fx, fy = _x, _y
            break
    if fx is None:
        fx, fy = 0, 0
        w.food[fx, fy] = 5.0
    a2 = Agent(w, fx, fy, energy=10.0)
    before_food = float(w.food[fx, fy])
    got = 0.0
    for _ in range(20):
        got += a2._do_eat()
    print("  [3] 采集 20 次: 累计回报 %.2f | 地块食物 %.1f → %.1f"
          % (got, before_food, float(w.food[fx, fy])))
    ok += int(got > 0 and w.food[fx, fy] < before_food)

    # ---- 4. 父母教孩子：知识真的传下去 ----
    total += 1
    parent = Agent(w, 21, 21, energy=60.0, name="Parent")
    parent.knowledge = {"采集": 0.9, "工具": 0.8}
    kid = Agent(w, 22, 21, energy=20.0, parents=[parent.id], name="Kid")
    kid.knowledge = {}
    by_pos = {(21, 21): [parent], (22, 21): [kid]}
    for _ in range(30):
        parent._do_teach(by_pos)
    print("  [4] 父母教学 30 次: 孩子知识 %s（应有采集/工具）" % list(kid.knowledge.keys()))
    ok += int(len(kid.knowledge) >= 1 and parent.stats["taught"] > 0)

    # ---- 5. 技术组合能发生 ----
    total += 1
    import random as _r
    civ = Civilization(size=30, n_agents=25, seed=7)
    for g in range(4):
        civ.generation = g
        for _ in range(25):
            civ.step()
    s = civ.stats()
    print("  [5] 4 代小世界: 人口 %d | 知识 %d 项 | 发现 %d 次"
          % (s["pop"], s["knowledge"], s["discoveries"]))
    ok += int(s["discoveries"] > 0)

    # ---- 6. 人口不会灭绝也不会爆炸 ----
    total += 1
    print("  [6] 人口 %d（初始 25，硬上限 350）| 部落 %d | 聚落 %d"
          % (s["pop"], s["tribes"], s["settlements"]))
    ok += int(1 <= s["pop"] <= 350)

    # ---- 7. 部落能形成 ----
    total += 1
    print("  [7] 部落数 %d（聚落聚集到 3 人以上才会组队）" % s["tribes"])
    ok += int(s["tribes"] >= 1)

    # ---- 8. 知识不会一代就丢光 ----
    total += 1
    print("  [8] 知识项数 %d（若为 0 说明代际传递断了）" % s["knowledge"])
    ok += int(s["knowledge"] >= 2)

    print("\n  结果：%d/%d 项通过" % (ok, total))
    return ok == total

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    pop = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    civ = Civilization(size=60, n_agents=pop, seed=42, verbose=True)
    civ.run(generations=gens, steps_per_gen=60)
    civ.report()
