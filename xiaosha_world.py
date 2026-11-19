# -*- coding: utf-8 -*-
"""小沙的地球 Online —— 世界 + 智能体 + 亲代教学 + 文化。

设计原则：**不写死文明**。我只提供四样东西：
  1. 一块有地形和资源的地
  2. 会学、会死、会生的小沙
  3. 父母能教孩子（知识真的传下去）
  4. 资源是有限的（所以必须竞争/合作）

部落、语言、技术、国家，都是这些机制跑出来的，不是我写的。
写出来什么算什么，跑不出来就是跑不出来。

三层结构：
  World    100x100 地图，地形 + 食物再生 + 矿物
  Agent    有基因、有脑（真 STDP）、有知识、有能量、有寿命
  Culture  知识在代际之间传递的载体（父母教孩子 / 观察模仿）

关键设计：大脑必须小（100-200 神经元），否则 200 个个体跑不动。
但学习必须是真的（STDP），否则就退化成"规则脚本"。
"""
import os
import sys

import numpy as np

from xiaosha_phenotype import (
    Phenotype, random_genome, crossover, clamp, PHENO_GENES,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

__version__ = "0.1.0"

# 地形
GRASS, FOREST, WATER, MOUNTAIN = 0, 1, 2, 3
TERRAIN_NAMES = {GRASS: "草地", FOREST: "森林", WATER: "水", MOUNTAIN: "山"}

# 每格基础食物产出
BASE_FOOD = {GRASS: 0.35, FOREST: 0.25, WATER: 0.10, MOUNTAIN: 0.02}
# 每格最大食物储量
MAX_FOOD = {GRASS: 6.0, FOREST: 8.0, WATER: 2.0, MOUNTAIN: 1.0}
# 能否通行
PASSABLE = {GRASS: True, FOREST: True, WATER: False, MOUNTAIN: False}

# 动作
ACT_MOVE, ACT_EAT, ACT_TALK, ACT_TEACH, ACT_BUILD, ACT_WORSHIP = 0, 1, 2, 3, 4, 5

# ==================== 宗教 ====================
# 小沙天生带着一个直觉：世界是"被造出来的"，有一个造物主。
# 这不是我硬塞的行为，而是一个**先天概念**：
# 有这个概念的小沙，在造出祭坛之后会自发去祭祀、会把这件事讲给同伴听。
# 至于他们管它叫什么、拜不拜得成、几个世代之后还记不记得，由世界决定。
SACRED = "造物主"
EXPLORE_EPS = 0.15        # ε-greedy：15% 的步数完全随机选动作
FARM_MAX_BOOST = 1.2     # 农业满级时，耕地承载力最多是自然的 2.2 倍
FARM_STEP = 0.25         # 每耕一次抬高多少（乘农业熟练度）
ALTAR_MINERAL = 2.5       # 建一座祭坛需要的矿物（山脚 4.5 够建一座）
WORSHIP_RANGE = 2         # 在祭坛几格内算"在祭祀"
WORSHIP_ENERGY = 4.0      # 祭祀带来的能量（"神迹"的实际好处）
# 祭祀不再增加信仰（原来是 0.040）。
# 原因：只要站在祭坛边就能涨信仰，那"被说服"永远比不过"拜一拜"——
# 实测 350 人里有 242 人信了，但传播事件 0 次：他们全是被祭坛"泡"出来的，
# 没有一个人是被谁说服的。宗教成了环境效应，不是思想传播。
# 现在祭祀只给能量（真实好处），信仰只能靠**别人讲给你听**。
WORSHIP_FAITH = 0.0
MIRACLE_ENERGY = 0.0      # 由外部（人类）降下的神迹，累积在这里
ACTION_NAMES = {ACT_MOVE: "移动", ACT_EAT: "采集", ACT_TALK: "交流",
                ACT_TEACH: "教学", ACT_BUILD: "建造", ACT_WORSHIP: "祭祀"}
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


# ==================== 世界 ====================

class World:
    """一块有限的地。资源会再生，但有上限 —— 所以养不活无限多的人。"""

    def __init__(self, size=100, seed=42):
        self.size = int(size)
        self.rng = np.random.RandomState(seed)

        # 地形：用平滑噪声生成（不用 scipy，用多次随机+卷积近似）
        t = self.rng.rand(self.size, self.size)
        for _ in range(3):                       # 平滑三次 → 成片而不是噪点
            t = (t + np.roll(t, 1, 0) + np.roll(t, -1, 0)
                 + np.roll(t, 1, 1) + np.roll(t, -1, 1)) / 5.0
        # 用分位数分配地形（固定阈值会因为平滑后值挤在 0.5 附近而全部落空 —— 实测水/山都是 0 格）
        terrain = np.full((self.size, self.size), GRASS, dtype=np.int8)
        q = np.quantile(t, [0.63, 0.83, 0.955])
        terrain[t > q[0]] = FOREST
        terrain[t > q[1]] = WATER
        terrain[t > q[2]] = MOUNTAIN
        self.terrain = terrain

        # 资源
        self.food = np.zeros((self.size, self.size), dtype=np.float32)
        # 耕地承载力：每格食物再生的上限。初始 = 该地形的自然上限；
        # 小沙种地会把这块地（和周围 8 格）的上限抬高。
        # 没有这个数组的话 regrow() 每步都按 MAX_FOOD 封顶，
        # 农业抬高多少就被压回去多少 —— 实测农业知识到了 1.0，产量毫无变化。
        self.farm_cap = np.zeros((self.size, self.size), dtype=np.float32)
        for kind, mx in MAX_FOOD.items():
            m = terrain == kind
            self.food[m] = float(mx * 0.6)
            self.farm_cap[m] = float(mx)

        # 矿物（造建筑/工具用，稀有）
        self.minerals = np.zeros((self.size, self.size), dtype=np.float32)
        mm = terrain == MOUNTAIN
        self.minerals[mm] = 8.0
        # 山脚也有一点
        for d in DIRS:
            sh = np.roll(np.roll(mm, d[0], 0), d[1], 1)
            # 4.5 而不是 1.5：建祭坛要 2.5 矿物，而山（8.0）是不可通行的，
            # 山脚只给 1.5 的话小沙永远站不到够矿的格子 —— 实测祭坛数一直是 0。
            self.minerals[sh & (terrain == GRASS)] += 4.5

        # 祭坛：小沙自己建的崇拜场所 (x, y, builder_id, generation)
        self.altars = []
        self.step_count = 0
        self.log = []

    def build_altar(self, x, y, builder):
        """在 (x,y) 建一座祭坛。需要矿物。"""
        if self.minerals[x, y] < ALTAR_MINERAL:
            return False
        self.minerals[x, y] -= ALTAR_MINERAL
        self.altars.append((int(x), int(y), builder.id, builder.generation))
        return True

    def altars_near(self, x, y, r=WORSHIP_RANGE):
        return [a for a in self.altars
                if abs(a[0] - x) <= r and abs(a[1] - y) <= r]

    def miracle(self, x, y, radius=6, energy=25.0):
        """人类降下神迹：一片区域食物暴涨（小沙会当成造物主显灵）。"""
        n = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    t = int(self.terrain[nx, ny])
                    self.food[nx, ny] = min(MAX_FOOD[t] * 2.0, self.food[nx, ny] + energy)
                    n += 1
        self.miracles = getattr(self, "miracles", 0) + 1
        return n

    def can_farm(self, x, y):
        """这块地能不能种（陆地且可通行）。"""
        return bool(PASSABLE.get(int(self.terrain[x, y]), False))

    def natural_food(self, x, y):
        """这块地的自然承载力（不考虑种地加成）。"""
        return float(MAX_FOOD.get(int(self.terrain[x, y]), 6.0))

    def food_norm(self, x, y):
        """这一格食物归一化用的分母（agent 感知用的是 食物/上限）。

        平原和球面的地形编号完全不同（平原 4 种、球面 7 种），
        所以不能各自去查同一张 MAX_FOOD 表 —— 让世界自己回答。
        """
        return MAX_FOOD.get(int(self.terrain[x, y]), 6.0)

    def passable(self, x, y):
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        return PASSABLE[int(self.terrain[x, y])]

    def regrow(self, rate=1.0):
        """食物再生（按地形基础产出，上限是耕地承载力而不是自然上限）。"""
        for kind, base in BASE_FOOD.items():
            m = self.terrain == kind
            self.food[m] = np.minimum(self.farm_cap[m], self.food[m] + base * rate)

    def harvest(self, x, y, amount):
        """采集。返回实际采到的量。"""
        got = float(min(amount, self.food[x, y]))
        self.food[x, y] -= got
        return got

    def total_food(self):
        return float(self.food.sum())

    def stats(self):
        counts = {TERRAIN_NAMES[k]: int((self.terrain == k).sum()) for k in TERRAIN_NAMES}
        return {"size": self.size, "terrain": counts,
                "food_total": round(self.total_food(), 1),
                "step": self.step_count}


# ==================== 邻居查询 ====================

def neighbours(by_pos, x, y, r=2, exclude=None, same_tile=True):
    """±r 格内的其他个体。

    这段 5x5 循环原来在三个地方各写了一遍（交流、教学、本能社交），
    改一次半径就要改三处 —— 抽出来。

    same_tile: 是否包含自己站的那一格。默认 True（站在同格的人当然能交谈），
    但 _ambient_social 原本是跳过同格的，为了这次纯重构不改动任何数字，
    那边显式传 False。将来如果要统一成 True，那是一次独立的行为改动，
    会改变所有随机序列，不能混在重构里。
    """
    out = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if same_tile or dx or dy:
                for o in by_pos.get((x + dx, y + dy), ()):
                    if o is not exclude:
                        out.append(o)
    return out


# ==================== 大脑 ====================

class MiniBrain:
    """小型脉冲网络（N 个神经元），真 STDP 学习。

    为什么不用完整皮层：200 个个体 × 6 万神经元 = 跑不动。
    但学习必须真 —— 用稀疏脉冲 + trace-based STDP，规模小但机制真。

    输入（16 维）：
      0-7   周围 8 格的可见食物量（归一化）
      8-11  周围 8 格的邻居密度（4 个象限）
      12    自己的能量水平
      13    年龄（归一化）
      14    是否是族内成员在场
      15    常驻偏置
    输出（5 维）：移动 / 采集 / 交流 / 教学 / 建造
    """

    N_IN = 16
    N_OUT = 6

    def __init__(self, n=120, seed=None, lr=0.02, k_in=6):
        self.n = int(n)
        self.rng = np.random.RandomState(seed)
        self.lr = float(lr)

        # 输入→网络 的固定随机投射（不学，保持稳定）
        self.w_in = (self.rng.rand(self.N_IN, self.n) < 0.25).astype(np.float32)
        # 网络内部随机连接（小规模，学）
        self.w_rec = (self.rng.randn(self.n, self.n) * 0.15).astype(np.float32)
        np.fill_diagonal(self.w_rec, 0.0)
        # 网络→输出（学）
        self.w_out = (self.rng.randn(self.n, self.N_OUT) * 0.1).astype(np.float32)

        self.v = np.zeros(self.n, dtype=np.float32)
        self.ref = np.zeros(self.n, dtype=np.int16)
        self.theta = 1.0
        self.pre_trace = np.zeros(self.n, dtype=np.float32)
        self.post_trace = np.zeros(self.n, dtype=np.float32)

    def forward(self, obs):
        """走一步，返回 5 个动作的得分。"""
        obs = np.asarray(obs, dtype=np.float32)[:self.N_IN]
        if obs.size < self.N_IN:
            obs = np.pad(obs, (0, self.N_IN - obs.size))

        # 输入电流
        cur = self.w_in.T @ obs
        # 网络电流
        cur = cur + self.w_rec.T @ (self.v > 0).astype(np.float32) * 2.0

        self.v = self.v * 0.90 + cur
        self.v[self.ref > 0] = 0.0
        self.ref = np.maximum(0, self.ref - 1)

        sp = self.v >= self.theta
        self.v[sp] = 0.0
        self.ref[sp] = 2

        # 学习：pre 发放 → 增强它的输出权重
        if sp.any():
            self.post_trace = self.post_trace * 0.95
            self.post_trace[sp] = 1.0

        # 输出：读出层的加权和（软最大化前）
        out = self.w_out.T @ (sp.astype(np.float32) + 0.15 * self.post_trace)
        # 自适应阈值（稳态）
        if sp.sum() > self.n * 0.15:
            self.theta = min(4.0, self.theta * 1.01)
        else:
            self.theta = max(1.0, self.theta * 0.999)
        return out

    def learn(self, sp_mask, action, reward):
        """STDP + 奖励调制：奖励>0 时增强刚才发过火的神经元→动作的连接。"""
        if reward == 0:
            return
        col = self.w_out[:, int(action)]
        delta = self.lr * float(np.clip(reward, -2.0, 2.0)) * sp_mask
        np.clip(col + delta, -3.0, 3.0, out=col)
        self.w_out[:, int(action)] = col


# ==================== 智能体 ====================

class Agent:
    """一个小沙个体。"""

    _next_id = 0

    def __init__(self, world, x, y, genes=None, brain=None, tribe=None,
                 energy=20.0, generation=0, parents=None, name="", pheno=None):
        Agent._next_id += 1
        self.id = Agent._next_id
        self.name = name or ("x%d" % self.id)
        self.world = world
        self.x, self.y = int(x), int(y)
        self.genes = genes if genes is not None else {}
        # 表型：每个个体一套自己的参数（体型/代谢/视力/好奇/寿命/社交/夜行/耐心）
        # 没传就随机生成一套 —— 所以**没有任何两个个体是一样的**。
        self.pheno = pheno if pheno is not None else Phenotype(random_genome())
        # 脑子也用这个个体的基因：学习率是它自己的
        if brain is not None:
            self.brain = brain
        else:
            self.brain = MiniBrain(seed=self.id)
        self.brain.lr = float(self.pheno.learn)
        self.tribe = tribe
        self.energy = float(energy)
        self.age = 0
        self.generation = int(generation)
        self.parents = list(parents or [])
        self.alive = True

        # 知识：概念 → 熟练度（0-1）。这就是"文化"。
        self.knowledge = {}
        # 记忆：最近遇到的人（用于识别族内/族外）
        self.met = set()
        # 统计
        self.stats = {"eaten": 0.0, "taught": 0, "learned": 0,
                      "talked": 0, "built": 0, "worshipped": 0, "altars": 0}
        self.last_reward = 0.0
        self.last_sp = None

    # ---------- 感知 ----------
    def perceive(self, agents_by_pos):
        obs = np.zeros(MiniBrain.N_IN, dtype=np.float32)
        # 0-7 周围 8 格食物
        #
        # 注意两件事：
        # 1) 球面世界里 nx 会绕过去（x=size 就是 x=0），所以坐标要先归一化；
        # 2) 地形编号各不相同（平原 4 种、球面 7 种），不能用同一张 MAX_FOOD 表查，
        #    否则球面的"山地(6)"在平原表里直接 KeyError。
        #    改成问世界要"这格的归一化上限"。
        norm = getattr(self.world, "food_norm", None)
        has_norm = hasattr(self.world, "normalize")
        for i, (dx, dy) in enumerate(DIRS):
            nx, ny = self.x + dx, self.y + dy
            if has_norm:
                nx, ny = self.world.normalize(nx, ny)
            elif not (0 <= nx < self.world.size and 0 <= ny < self.world.size):
                continue
            if norm is not None:
                mx = norm(nx, ny)
            else:
                mx = MAX_FOOD.get(int(self.world.terrain[nx, ny]), 6.0)
            obs[i] = min(1.0, self.world.food[nx, ny] / max(mx, 1e-9))
        # 8-11 邻居密度（四象限）
        # 只看 ±2 格的 5x5 窗口，不要遍历 agents_by_pos 的所有位置 ——
        # 原来是 for (...) in agents_by_pos.items()，350 个个体时每个个体每步
        # 都要扫 350 个位置，一代就是 700 万次比较，是球面版跑 30 代要 339 秒的主因。
        n_near = [0, 0, 0, 0]
        n_in_tribe = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                ax, ay = self.x + dx, self.y + dy
                if has_norm:
                    ax, ay = self.world.normalize(ax, ay)
                elif not (0 <= ax < self.world.size and 0 <= ay < self.world.size):
                    continue
                others = agents_by_pos.get((ax, ay))
                if not others:
                    continue
                q = (0 if dx < 0 else 1) + (0 if dy < 0 else 2)
                n_near[q] += len(others)
                if self.tribe is not None and any(o.tribe is self.tribe for o in others):
                    n_in_tribe += 1
        obs[8:12] = np.clip(np.array(n_near, dtype=np.float32) / 5.0, 0, 1)
        # 12 能量
        obs[12] = float(np.clip(self.energy / 60.0, 0, 1))
        # 13 年龄
        obs[13] = float(np.clip(self.age / 200.0, 0, 1))
        # 14 族人在场
        obs[14] = 1.0 if n_in_tribe > 0 else 0.0
        # 15 祭坛在场（有祭坛的人更容易去祭祀 —— 这是"造物主"直觉的输入）
        obs[15] = 1.0 if self.world.altars_near(self.x, self.y, r=6) else 0.0
        return obs

    # ---------- 行动 ----------
    def act(self, agents_by_pos, tribes, mutation_rate=0.05):
        """感知 → 决策 → 执行。返回动作名。"""
        obs = self.perceive(agents_by_pos)
        scores = self.brain.forward(obs)
        # 加一点随机探索
        scores = scores + self.brain.rng.randn(scores.size) * 0.15

        # ε-greedy：小概率完全随机选一个动作，不管脑子怎么说。
        # 没有这一条，每个个体都会被 w_out 的随机初始值锁死在一个动作上：
        # 实测有 300 步 300 次全是"采集"的个体，也有 80% 时间只会"教学"的先知。
        # 脑子学不动（一步失败只改变权重 0.0002，而初始权重散布 0.1，寿命只有 300 步），
        # 所以它会带着同一个偏好活到死 —— 会种地的人从不选"建造"，农业永远用不上。
        # ε-greedy 是强化学习的标准探索项，在动物身上也真实存在（行为变异/试错），
        # 它保证每个动作都会被采到，学习才有输入。
        # 探索率是**每个个体自己的**（好奇基因），不再是全局常量
        if self.brain.rng.rand() < self.pheno.eps:
            act = int(self.brain.rng.randint(scores.size))
        else:
            act = int(np.argmax(scores))
        sp_mask = (self.brain.post_trace > 0.5).astype(np.float32)

        reward = 0.0
        if act == ACT_MOVE:
            reward = self._do_move(agents_by_pos)
        elif act == ACT_EAT:
            reward = self._do_eat()
        elif act == ACT_TALK:
            reward = self._do_talk(agents_by_pos)
        elif act == ACT_TEACH:
            reward = self._do_teach(agents_by_pos)
        elif act == ACT_BUILD:
            reward = self._do_build()
        elif act == ACT_WORSHIP:
            reward = self._do_worship()
        else:
            reward = -0.01

        # 学习（STDP + 奖励调制）
        # 注意：以前是 `reward if abs(reward) > 0.05 else 0.0`，等于"小失败不算数"。
        # 但最常见的失败回报就是 -0.01（教学没人可教、交流没人可说），
        # 于是那个动作的权重永远压不下来 —— 实测一个个体 200 步里 159 次
        # 死选"教学"（80%），因为它从没学过这动作没用。现在任何反馈都学。
        self.brain.learn(sp_mask, act, reward)
        self.last_reward = reward
        self.last_sp = sp_mask
        self.age += 1

        # 球面世界：移动可能越界，这里把坐标折回（经度环绕、纬度钳制）。
        # 平原世界没有 normalize，这一步自动跳过。
        if hasattr(self.world, "normalize"):
            self.x, self.y = self.world.normalize(self.x, self.y)
            light = self.world.light_at(self.x, self.y)
        else:
            light = 1.0

        # 基础代谢 × 活跃度：睡着的人省能量，跑动的人费能量。
        # 夜色对昼行个体 = 休息，能耗降到 0.55 倍；对夜行个体则相反。
        act_level = self.pheno.activity(light)
        self.energy -= 0.35 * (self.pheno.metab_cost / 0.39) * (0.55 + 0.45 * act_level)
        return ACTION_NAMES[act]

    def _do_move(self, agents_by_pos):
        # 朝圣：附近有祭坛时，有一半概率朝它走。
        # 没有这条，祭坛只是一块石头 —— 350 个小沙里没几个会碰巧站在它旁边，
        # 实测祭祀次数趋近于 0、信仰永远停在出生值 0.16。
        altars = self.world.altars_near(self.x, self.y, r=12)
        if altars and self.brain.rng.rand() < 0.55:
            ax, ay = altars[0][0], altars[0][1]
            dx = 1 if ax > self.x else (-1 if ax < self.x else 0)
            dy = 1 if ay > self.y else (-1 if ay < self.y else 0)
            nx, ny = self.x + dx, self.y + dy
            if self.world.passable(nx, ny) and len(agents_by_pos.get((nx, ny), [])) < 6:
                self.x, self.y = nx, ny
                return 0.02

        d = DIRS[self.brain.rng.randint(len(DIRS))]
        nx, ny = self.x + d[0], self.y + d[1]
        if not self.world.passable(nx, ny):
            return -0.02
        # 目标格太挤（>4 人）就不动
        if len(agents_by_pos.get((nx, ny), [])) >= 4:
            return -0.01
        self.x, self.y = nx, ny
        return 0.01

    def _do_eat(self):
        # 采集效率受光照影响：黑夜里几乎拿不到东西（球面昼夜是真的）
        light = self.world.light_at(self.x, self.y) if hasattr(self.world, "light_at") else 1.0
        eff = 0.15 + 0.85 * self.pheno.activity(light)
        got = self.world.harvest(self.x, self.y,
                                 amount=2.0 * self.pheno.harvest_mult * eff)
        if got <= 0:
            return -0.02
        self.energy = min(100.0, self.energy + got * 3.0)
        self.stats["eaten"] += got
        # 熟能生巧：反复采集让"采集"变熟练
        # （原来知识一旦"发现"就永远停在 0.35-0.55，部落 tech_level 永远是 0）
        if "采集" in self.knowledge:
            self.knowledge["采集"] = min(1.0, self.knowledge["采集"] + 0.004)
        return 0.5 if got > 0.5 else 0.15

    def _do_talk(self, agents_by_pos):
        # 交谈范围 ±2 格，不是同一格。
        # 实测：350 人散在 40x40=1600 格 = 0.22 人/格，要求"站在同一格"
        # 等于永远没人能说话（累计 talk 动作 0 次，传播事件 0 次）。
        others = neighbours(agents_by_pos, self.x, self.y, 2, self)
        if not others:
            return -0.01
        other = others[self.brain.rng.randint(len(others))]
        self.met.add(other.id)
        other.met.add(self.id)
        self.stats["talked"] += 1
        # 模因传播：把「造物主」这件事讲出去。
        # 走 xiaosha_memetics.try_convert，会带语言变异（30% 改叫法），
        # 所以「造物主」传几手之后会变成「天物主」「造物神」「上苍」等等。
        from xiaosha_memetics import try_convert
        converted, new_name = try_convert(self, other, self.brain.rng)
        if converted:
            # 传教成功的奖励从 0.25 提到 1.2。
            # 原来 0.25 太弱：大脑学不到"说话有用"，实测先知 80% 的时间
            # 死选"教学"（而他还没孩子，教学永远失败），移动只占 3%，
            # 周围没人 → 576 次交流动作里没有一次成功传播。
            # 说话能改变别人，对群居动物来说是真实的强回报。
            return 1.2

        # 交流：传递一个自己最好的知识
        if self.knowledge and other.knowledge is not None:
            k = max(self.knowledge.items(), key=lambda kv: kv[1])[0]
            v = self.knowledge[k]
            if other.knowledge.get(k, 0.0) < v * 0.9:
                other.knowledge[k] = min(1.0, v * 0.6)   # 只传 60%（不完整）
                other.stats["learned"] += 1
                # 一边教一边自己也更熟
                self.knowledge[k] = min(1.0, v + 0.003)
                other.knowledge[k] = min(1.0, other.knowledge[k] + 0.003)
                return 0.2
        return 0.03

    def _do_teach(self, agents_by_pos):
        """父母教孩子：把知识完整传下去（比交流效率高）。

        范围 ±2 格，不是同一格 —— 父母和孩子每步都在移动，要求站同一格
        实测几乎教不到（知识代际流失到只剩 1 项）。
        """
        kids = [o for o in neighbours(agents_by_pos, self.x, self.y, 2, self)
                if self.id in o.parents]
        if not kids or not self.knowledge:
            return -0.01
        kid = kids[0]
        n = 0
        for k, v in self.knowledge.items():
            if kid.knowledge.get(k, 0.0) < v * 0.95:
                kid.knowledge[k] = min(1.0, v * 0.95)     # 教学 95%（比交流完整）
                self.knowledge[k] = min(1.0, v + 0.004)   # 教一遍自己也更熟
                # 【关键】教学是信仰的主要传播途径（实测：总传播事件 0 次，
                # 但信徒从 9 涨到 242 —— 全靠父母教孩子）。
                # 所以教学时语言也必须变异，否则"造物主"永远只有一个叫法。
                if k == SACRED:
                    from xiaosha_memetics import Meme, _log_lineage
                    src = getattr(self, "meme_name", SACRED)
                    if self.brain.rng.rand() < 0.70:
                        new_name = src
                    else:
                        new_name = Meme(src, rng=self.brain.rng).mutate()
                    kid.meme_name = new_name
                    kid.meme_origin = src
                    _log_lineage(self, kid, new_name, self.brain.rng)
                n += 1
        self.stats["taught"] += n
        kid.stats["learned"] += n
        return 0.3 if n else -0.01

    def _do_worship(self):
        """祭祀造物主。只有在祭坛附近才有效。

        效果是真的：能量 +WORSHIP_ENERGY（相当于"神赐食物"），
        信仰 +WORSHIP_FAITH（信仰越高，越愿意继续拜、越愿意讲给别人听）。
        """
        near = self.world.altars_near(self.x, self.y)
        if not near:
            return -0.01
        self.energy = min(100.0, self.energy + WORSHIP_ENERGY)
        self.knowledge[SACRED] = min(1.0, self.knowledge.get(SACRED, 0.0) + WORSHIP_FAITH)
        self.stats["worshipped"] = self.stats.get("worshipped", 0) + 1
        # 祭坛越多、在场的人越多，越热闹（但收益递减，防止刷分）
        bonus = min(0.5, 0.1 * (len(near) - 1))
        return 0.35 + bonus

    def _do_build(self):
        """建造：在「建祭坛」和「种地」之间按熟练度选。

        原来的顺序是"祭坛优先，而且每条路径都 return"，于是凡是会
        「石头 + 建造」的人（建造是先天基础技能，均值 0.43，几乎所有人都会）
        每次建造都在盖庙或找矿，永远走不到种地那一段 ——
        实测耕地格数一直是 0，余粮不出现，国家也就无从谈起。
        现在按熟练度选：两个都会的人大部分时间务农，三分之一时间还去盖庙。
        """
        can_altar = (self.knowledge.get("石头", 0.0) > 0.15
                     and self.knowledge.get("建造", 0.0) > 0.10)
        can_farm = self.knowledge.get("农业", 0.0) >= 0.3

        if can_altar and can_farm:
            first_farm = self.brain.rng.rand() < 0.65
        else:
            first_farm = can_farm

        for want_farm in ((True, False) if first_farm else (False, True)):
            r = self._try_farm() if want_farm else self._try_altar()
            if r is not None:
                return r
        return -0.01

    def _try_altar(self):
        """建祭坛，或者为了建祭坛去找矿。做不到返回 None（交给别的技能）。"""
        if not (self.knowledge.get("石头", 0.0) > 0.15
                and self.knowledge.get("建造", 0.0) > 0.10):
            return None
        if (self.world.minerals[self.x, self.y] >= ALTAR_MINERAL
                and not self.world.altars_near(self.x, self.y, r=0)):
            if self.world.build_altar(self.x, self.y, self):
                self.knowledge[SACRED] = min(
                    1.0, self.knowledge.get(SACRED, 0.0) + 0.06)
                self.stats["altars"] = self.stats.get("altars", 0) + 1
                return 0.6
            return None
        # 寻矿：为了建神庙，人自己会去找石料。
        # 没有这条，小沙只会在草地上随机游走 —— 实测 204 个矿区格子里
        # 只站着 1 个人，祭坛永远建不出来。
        # 世界若能直接给出最近的矿就用它（球面版预计算过矿区列表，
        # 比扫 15x15 快两个数量级）；否则退回扫描。
        near_fn = getattr(self.world, "nearest_ore", None)
        if near_fn is not None:
            got = near_fn(self.x, self.y)
            if got is None:
                return None
            dx, dy = got
        else:
            best = None
            for dx in range(-7, 8):
                for dy in range(-7, 8):
                    nx, ny = self.x + dx, self.y + dy
                    if not (0 <= nx < self.world.size and 0 <= ny < self.world.size):
                        continue
                    if (self.world.passable(nx, ny)
                            and self.world.minerals[nx, ny] >= ALTAR_MINERAL):
                        d2 = dx * dx + dy * dy
                        if best is None or d2 < best[0]:
                            best = (d2, dx, dy)
            if best is None:
                return None
            _d2, dx, dy = best
        sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
        sy = 1 if dy > 0 else (-1 if dy < 0 else 0)
        nx, ny = self.x + sx, self.y + sy
        if self.world.passable(nx, ny):
            self.x, self.y = nx, ny
            return 0.15
        return None

    def _try_farm(self):
        """把脚下和周围 8 格垦成耕地。不会农业返回 None。"""
        key = "农业"
        lvl = self.knowledge.get(key, 0.0)
        if lvl < 0.3:
            return None
        # 平原 4 种地形、球面 7 种，编号完全不同，不能直接比 WATER/MOUNTAIN 常量
        if not self.world.can_farm(self.x, self.y):
            return None
        # 种地要的是土地，不是石头 —— 原来要求"脚下有矿物 >= 1.0"，
        # 于是只有站在山脚的人能种地，种一次还得挖走一块石头，不合常理。
        # 一个人耕不了整片地，但一片田能养活一家人 —— 这就是"余粮"的来源。
        natural = self.world.natural_food(self.x, self.y)
        cap_max = natural * (1.0 + FARM_MAX_BOOST * lvl)
        n = 0
        has_norm = hasattr(self.world, "normalize")
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = self.x + dx, self.y + dy
                if has_norm:
                    nx, ny = self.world.normalize(nx, ny)
                elif not (0 <= nx < self.world.size and 0 <= ny < self.world.size):
                    continue
                if not self.world.can_farm(nx, ny):
                    continue
                self.world.farm_cap[nx, ny] = min(
                    cap_max, self.world.farm_cap[nx, ny] + FARM_STEP * lvl)
                n += 1
        if not n:
            return None
        self.world.food[self.x, self.y] = min(
            cap_max, self.world.food[self.x, self.y] + 2.0 * lvl)
        self.stats["built"] += 1
        self.knowledge[key] = min(1.0, lvl + 0.02)
        return 0.4

    # ---------- 生命 ----------
    def can_reproduce(self):
        """能繁殖的年龄段。

        上限从 160 提到 260：每代 60 步，160 步 = 只能繁殖 2.7 代，
        实测第 2 代可繁殖人数从 88 暴跌到 15（全体超龄），
        新生代还没长大就断代灭绝。260 步能覆盖 4 代，种群才稳定。
        """
        return self.energy > 55.0 and self.age > 40 and self.age < self.pheno.life

    def reproduce(self, partner, generation, mutation_rate=0.05):
        """有性生殖：基因交叉 + 大脑结构继承 + 知识部分继承。"""
        # 基因：简单平均（真实版在 xiaosha_reproduction，这里用轻量版）
        child_genes = {}
        for k in set(self.genes) | set(partner.genes):
            a = float(self.genes.get(k, 0.5))
            b = float(partner.genes.get(k, 0.5))
            child_genes[k] = 0.5 * (a + b) + self.brain.rng.randn() * mutation_rate * 0.3
            child_genes[k] = float(np.clip(child_genes[k], 0.0, 1.0))

        # 大脑：小的结构继承（学习率来自基因）
        lr = 0.005 + 0.04 * child_genes.get("plasticity", 0.5)
        brain = MiniBrain(seed=self.brain.rng.randint(0, 2**30), lr=lr)
        # 继承父母的部分读出头（"本能"）
        mask = self.brain.rng.rand(self.brain.n, MiniBrain.N_OUT) < 0.5
        brain.w_out = np.where(mask, self.brain.w_out, partner.brain.w_out).astype(np.float32)

        # 表型：从父母的基因组交叉 + 突变。
        # 这是"每个个体都不一样"能延续下去的关键 —— 没有这一步，
        # 新生个体又会退回同一套随机参数，父子之间毫无关系。
        child_genome = crossover(self.pheno.genome, partner.pheno.genome,
                                 rng=self.brain.rng, mutation=0.18)
        child_pheno = Phenotype(child_genome)

        energy_cost = 18.0
        self.energy -= energy_cost / 2
        partner.energy -= energy_cost / 2

        kid = Agent(self.world, self.x, self.y, genes=child_genes, brain=brain,
                    tribe=self.tribe, energy=18.0, generation=generation,
                    parents=[self.id, partner.id], pheno=child_pheno,
                    name="g%d_%d" % (generation, Agent._next_id + 1))
        # 知识：孩子只继承一点点（文化靠教，不靠血脉）
        for k, v in self.knowledge.items():
            if self.brain.rng.rand() < 0.25:
                kid.knowledge[k] = v * 0.1
        # 但"造物主"直觉是天生的，每个新生儿都有（0.10-0.22），
        # 不从父母那里随机继承 —— 否则一代就丢光（实测信仰 0.00）
        kid.knowledge[SACRED] = 0.10 + self.brain.rng.rand() * 0.12
        return kid

    def summary(self):
        return {"name": self.name, "gen": self.generation, "age": self.age,
                "energy": round(self.energy, 1), "tribe": self.tribe,
                "knowledge": len(self.knowledge),
                "best": max(self.knowledge.items(), key=lambda kv: kv[1])[0]
                        if self.knowledge else None}


if __name__ == "__main__":
    # 冒烟：造世界 + 放 50 个个体跑 30 步
    w = World(size=40, seed=1)
    print("世界:", w.stats())
    agents = []
    for i in range(50):
        while True:
            x, y = np.random.randint(0, 40), np.random.randint(0, 40)
            if w.passable(x, y):
                break
        agents.append(Agent(w, x, y, energy=30.0))
    print("个体: %d 个" % len(agents))
    for step in range(30):
        by_pos = {}
        for a in agents:
            by_pos.setdefault((a.x, a.y), []).append(a)
        for a in agents:
            a.act(by_pos, {})
        w.regrow()
        w.step_count += 1
    alive = [a for a in agents if a.energy > 0]
    print("30 步后: 存活 %d | 平均能量 %.1f | 累计采集 %.1f | 交流 %d"
          % (len(alive), float(np.mean([a.energy for a in alive])) if alive else 0,
             sum(a.stats["eaten"] for a in agents),
             sum(a.stats["talked"] for a in agents)))
