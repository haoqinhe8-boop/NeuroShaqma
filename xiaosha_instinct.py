# -*- coding: utf-8 -*-
"""小沙的本能系统（Instinct System）—— 零任务训练，陌生场景即刻可用。

核心思想（跟"训练几千轮"完全不同的路线）：
  真实动物到一个陌生地方能立刻干活，靠的不是"练过这个场景"，而是三层：
    ① 进化先验（基因编码的结构）—— 脑区/回路天生就位
    ② 本能回路（硬连线的固定动作模式 FAP / 中央模式发生器 CPG）
    ③ 一次经历就记住（在线情景记忆，不是梯度下降几千轮）

本模块实现 ②③ 和跨模态感知融合，①由 xiaosha_structured.py 提供。

架构：
    多感觉输入（视/听/触/本体/内感受）
        ↓ 各通道独立算"显著性"（固定规则，不学）
    显著性地图：威胁图 / 目标图 / 新奇图 / 社交图
        ↓ 7 条本能回路各自给出"动作建议 + 效用"
    本能回路：避障 / 趋近 / 朝向声源 / 平衡 / 探索 / 社交 / 稳态
        ↓ 行为仲裁（效用竞争 + 惯性 + 抑制）
    动作输出（vx, vy, vyaw, 手臂） + 在线记忆写入

实测目标（本模块自带验证）：
  - 陌生场景零样本成功率（障碍/声源/目标全部随机）
  - 决策速度（每步 < 2ms）
  - 行为自然度（无死锁、无抖动、动作平滑）
"""
import json
import os
import time

import numpy as np

__version__ = "1.0.0"


# ==================== 1. 多感觉编码（固定规则，不训练）====================

class Senses:
    """把一个"场景"编码成各通道的显著性。

    真实场景来自传感器；这里用结构化描述（便于测试和复现）。
    每个通道的编码规则是**固定的**（进化来的，不是学的）。
    """

    # 感觉通道的"量程"（先天设定）
    VISION_RANGE = 4.0      # 米
    AUDIO_RANGE = 8.0       # 米
    TOUCH_RANGE = 0.15      # 米（接触距离）
    COMFORT_TEMP = 24.0     # 舒适温度
    COMFORT_LIGHT = 0.6     # 舒适亮度

    @staticmethod
    def encode(scene):
        """scene → 显著性字典。

        scene 字段（都可有可无，缺省视为"没有"）：
          obstacles: [(距离, 方位角)]     障碍（威胁）
          target:    (距离, 方位角)       目标（奖励）
          sound:     (距离, 方位角, 强度) 声源（社交/好奇）
          person:    (距离, 方位角)       人（社交）
          light:     0~1                  环境亮度（趋光/避强光）
          temp:      摄氏度               温度（稳态）
          battery:   0~1                  电量（稳态）
          novelty:   0~1                  场景新奇度（探索）
          touch:     0/1                  是否接触
          tilt:      弧度                 身体倾角（平衡）
        """
        sig = {}

        # ---- 威胁图：障碍越近越威胁（指数衰减，先天规则）----
        threat = 0.0
        threat_dir = 0.0
        for d, ang in scene.get("obstacles", []):
            if d <= 0:
                continue
            s = np.exp(-d / 1.2)          # 1.2m 尺度衰减
            if s > threat:
                threat = s
                threat_dir = ang
        # 强光也算威胁（刺眼）
        light = scene.get("light", 0.6)
        if light > 0.9:
            threat = max(threat, (light - 0.9) * 5.0)
        sig["threat"] = min(1.0, threat)
        sig["threat_dir"] = threat_dir

        # ---- 目标图：目标越近越值得趋近 ----
        goal = 0.0
        goal_dir = 0.0
        if "target" in scene:
            d, ang = scene["target"]
            goal = np.exp(-max(0.0, d - 0.3) / 2.0)
            goal_dir = ang
        sig["goal"] = min(1.0, goal)
        sig["goal_dir"] = goal_dir

        # ---- 社交图：人 + 声源（人声优先）----
        social = 0.0
        social_dir = 0.0
        if "person" in scene:
            d, ang = scene["person"]
            social = np.exp(-d / 3.0) * 0.9
            social_dir = ang
        if "sound" in scene:
            d, ang, loud = scene["sound"]
            s = np.exp(-d / 4.0) * float(loud)
            if s > social:
                social = s
                social_dir = ang
        sig["social"] = min(1.0, social)
        sig["social_dir"] = social_dir

        # ---- 新奇图：探索欲（缺省中等好奇心）----
        sig["novelty"] = float(scene.get("novelty", 0.5))

        # ---- 稳态信号（内感受）----
        battery = float(scene.get("battery", 1.0))
        temp = float(scene.get("temp", Senses.COMFORT_TEMP))
        sig["battery"] = battery
        sig["temp_err"] = abs(temp - Senses.COMFORT_TEMP) / 10.0   # 0=舒适
        sig["tilt"] = abs(float(scene.get("tilt", 0.0))) / 0.5     # 0=直立
        sig["touch"] = 1.0 if scene.get("touch") else 0.0

        # ---- 奖励/惩罚（用于在线学习）----
        sig["_light"] = light
        sig["_scene"] = scene
        return sig


# ==================== 2. 本能回路（硬连线，不用学）====================

class Instinct:
    """一条本能回路：给 (动作建议, 效用)，由固定规则算出。"""

    name = "base"
    priority = 1.0

    def evaluate(self, sig, state):
        """返回 (vx, vy, vyaw, arm, utility) 或 None（不触发）。"""
        raise NotImplementedError


class AvoidObstacle(Instinct):
    """避障反射：威胁图 > 阈值就后退+转向（脊髓级反射，无延迟）。"""
    name = "avoid"
    priority = 10.0        # 生存最高优先级

    def evaluate(self, sig, state):
        t = sig["threat"]
        if t < 0.35:
            return None
        # 威胁越大越急，转向远离威胁方向
        mag = min(1.0, t * 1.5)
        ang = sig["threat_dir"]
        # 后退 + 转向相反方向
        vyaw = -np.sign(ang) * mag * 1.2 if abs(ang) > 0.2 else 1.0 * mag
        vx = -mag * 0.6
        return (vx, 0.0, vyaw, 0.0, self.priority * t)


class ApproachGoal(Instinct):
    """趋近目标：目标图越强越趋近。"""
    name = "approach"
    priority = 5.0

    def evaluate(self, sig, state):
        g = sig["goal"]
        if g < 0.05:
            return None
        ang = sig["goal_dir"]
        return (0.7 * g, 0.0, np.clip(ang, -1.0, 1.0) * 0.8, 0.0, self.priority * g)


class OrientToSound(Instinct):
    """朝向声源/人：社交本能（听到声音就转过去看）。"""
    name = "orient"
    priority = 4.0

    def evaluate(self, sig, state):
        s = sig["social"]
        if s < 0.08:
            return None
        ang = sig["social_dir"]
        return (0.0, 0.0, np.clip(ang, -1.2, 1.2) * 0.9, 0.0, self.priority * s)


class KeepBalance(Instinct):
    """平衡维持：倾角越大越急（前庭反射）。"""
    name = "balance"
    priority = 9.0

    def evaluate(self, sig, state):
        tilt = sig["tilt"]
        if tilt < 0.1:
            return None
        return (0.0, 0.0, 0.0, 0.0, self.priority * min(1.0, tilt))


class Explore(Instinct):
    """探索新奇：没有明确目标时四处看看（好奇本能）。"""
    name = "explore"
    priority = 1.5

    def evaluate(self, sig, state):
        # 有目标/威胁时不探索
        if sig["goal"] > 0.2 or sig["threat"] > 0.15:
            return None
        n = sig["novelty"]
        if n < 0.2:
            return None
        # 缓慢巡视（用状态的相位，避免原地打转）
        phase = (state.get("t", 0) * 0.06) % (2 * np.pi)
        return (0.25, 0.0, 0.5 * np.sin(phase), 0.0, self.priority * n)


class SocialFollow(Instinct):
    """社交跟随：有人且不危险时靠近人（通人性的核心）。"""
    name = "social"
    priority = 3.0

    def evaluate(self, sig, state):
        s = sig["social"]
        if s < 0.25 or sig["threat"] > 0.2:
            return None
        ang = sig["social_dir"]
        return (0.45 * s, 0.0, np.clip(ang, -0.8, 0.8) * 0.6, 0.0, self.priority * s)


class Homeostasis(Instinct):
    """稳态优先：电量低/温度严重不适 → 减少活动找补给（生物本能）。

    注意阈值不能太敏感：真实生物温度偏几度只是"不舒服"，不会立刻瘫下。
    只有电量告急(<20%) 或 温度严重偏离(>±12°C) 才触发。
    """
    name = "homeostasis"
    priority = 8.0

    def evaluate(self, sig, state):
        urg = 0.0
        # 电量告急（真正的硬约束）
        if sig["battery"] < 0.2:
            urg = max(urg, (0.2 - sig["battery"]) * 5.0)
        # 温度严重偏离才管（temp_err 1.0 ≈ 偏离 10°C）
        if sig["temp_err"] > 1.2:
            urg = max(urg, (sig["temp_err"] - 1.2) * 0.8)
        if urg < 0.1:
            return None
        # 动作：放慢 + 朝亮处走（找电源/充电指示），而不是原地抖
        light_pull = (Senses.COMFORT_LIGHT - float(sig.get("_light", 0.5)))
        vyaw = np.clip(light_pull * 1.5, -0.5, 0.5)
        return (0.15, 0.0, vyaw, 0.0, self.priority * min(1.0, urg))


INSTINCTS = [
    AvoidObstacle(), ApproachGoal(), OrientToSound(),
    KeepBalance(), Explore(), SocialFollow(), Homeostasis(),
]


# ==================== 3. 行为仲裁（效用竞争 + 惯性 + 平滑）====================

class Arbiter:
    """多个本能竞争 → 一个动作。规则固定（不需要学）。

    人类/动物也是这样：最强的驱力胜出，但会被次级驱力"染色"，
    并且有**惯性 + 行为持续性**（不会每 100ms 改主意，看起来才自然）。
    """

    def __init__(self, smoothing=0.45, hold_steps=6, switch_margin=1.25):
        self.smoothing = smoothing     # 动作平滑（惯性）
        self.hold_steps = hold_steps   # 当前行为至少保持 N 步（行为持续性）
        self.switch_margin = switch_margin  # 新本能要强 25% 才抢班（滞回）
        self.last = np.zeros(4, dtype=np.float32)
        self.last_winner = None
        self.hold = 0
        self.history = []
        self.switch_count = 0

    def decide(self, sig, state):
        cands = []
        for inst in INSTINCTS:
            out = inst.evaluate(sig, state)
            if out is not None:
                cands.append((inst.name, out, inst.priority))
        if not cands:
            act = np.zeros(4, dtype=np.float32)
            winner = "idle"
        else:
            cands.sort(key=lambda c: c[1][4], reverse=True)
            best_name, best_out, _ = cands[0]
            best_util = best_out[4]

            # ---- 行为持续性（滞回）：当前行为还在保持期内时，新本能必须明显更强才抢班 ----
            if self.last_winner is not None and self.hold > 0:
                cur = next((c for c in cands if c[0] == self.last_winner), None)
                if cur is not None and best_util < cur[1][4] * self.switch_margin:
                    # 保持当前行为
                    best_name, best_out = cur[0], cur[1]
                # 紧急生存本能（avoid/homeostasis）可以强制打断
                elif best_name in ("avoid", "homeostasis") and best_out[4] > 0.6:
                    pass  # 允许抢班
                elif cur is not None:
                    best_name, best_out = cur[0], cur[1]

            winner = best_name
            act = np.array(best_out[:4], dtype=np.float32)
            for name, out, _p in cands[1:3]:
                if name == winner:
                    continue
                act += 0.15 * np.array(out[:4], dtype=np.float32)

        # 惯性平滑
        act = self.smoothing * self.last + (1 - self.smoothing) * act
        self.last = act
        if winner != self.last_winner:
            self.switch_count += 1
            self.last_winner = winner
            self.hold = self.hold_steps   # 新行为重新计时
        elif self.hold > 0:
            self.hold -= 1
        self.history.append(winner)
        return act, winner, [c[0] for c in cands[:2]]


# ==================== 4. 在线情景记忆（一次经历就记住）====================

class EpisodicMemory:
    """情景记忆：把"场景指纹 → 动作/结果"存起来，下次遇到相似场景直接复用。

    这不是梯度下降几千轮——是**一次写入**（像海马的一次性编码）。
    """

    def __init__(self, dim=8, capacity=200):
        self.dim = dim
        self.capacity = capacity
        self.keys = []      # 场景指纹
        self.vals = []      # (动作, 结果分)
        self.hits = 0
        self.misses = 0

    def fingerprint(self, sig):
        """场景指纹（固定编码，不学）。"""
        return np.array([
            sig["threat"], sig["goal"], sig["social"],
            sig["novelty"], sig["battery"], sig["temp_err"],
            sig["tilt"], sig["touch"],
        ], dtype=np.float32)

    def recall(self, sig, thresh=0.18):
        """找最相似的记忆（不需要训练）。"""
        if not self.keys:
            self.misses += 1
            return None
        k = self.fingerprint(sig)
        keys = np.stack(self.keys)
        dist = np.linalg.norm(keys - k, axis=1)
        i = int(np.argmin(dist))
        if dist[i] < thresh:
            self.hits += 1
            return self.vals[i]
        self.misses += 1
        return None

    def store(self, sig, action, score):
        """一次写入（容量满了踢掉最差/最旧）。"""
        self.keys.append(self.fingerprint(sig))
        self.vals.append((np.array(action, dtype=np.float32), float(score)))
        if len(self.keys) > self.capacity:
            self.keys.pop(0)
            self.vals.pop(0)

    def stats(self):
        tot = self.hits + self.misses
        return {"size": len(self.keys), "hits": self.hits, "misses": self.misses,
                "hit_rate": (self.hits / tot) if tot else 0.0}


# ==================== 5. 小沙"本能体" ====================

class InstinctMind:
    """零任务训练的行为体：感知 → 本能 → 仲裁 → 动作（+ 一次记忆）。"""

    def __init__(self, memory_capacity=200):
        self.arbiter = Arbiter()
        self.memory = EpisodicMemory(capacity=memory_capacity)
        self.t = 0
        self.log = []

    def step(self, scene):
        """一步：返回 (动作, 信息)。"""
        t0 = time.perf_counter()
        sig = Senses.encode(scene)
        state = {"t": self.t}

        # ① 先查情景记忆（相似的场景 → 上次怎么做的）
        recalled = self.memory.recall(sig)
        if recalled is not None:
            act, score = recalled
            winner = "memory"
            top2 = ["memory"]
            reused = True
        else:
            # ② 没有记忆 → 本能仲裁
            act, winner, top2 = self.arbiter.decide(sig, state)
            reused = False

        dt = (time.perf_counter() - t0) * 1000
        self.t += 1
        info = {"winner": winner, "top2": top2, "reused_memory": reused,
                "sig": {k: v for k, v in sig.items() if not k.startswith("_")},
                "ms": dt}
        self.log.append(info)
        return act, info

    def learn_from_outcome(self, scene, action, outcome_score):
        """一次经历就记住（outcome_score: +1 好 / -1 坏）。"""
        sig = Senses.encode(scene)
        self.memory.store(sig, action, outcome_score)


# ==================== 6. 自测：陌生场景零样本 ====================

def _random_scene(rng, difficulty=1.0):
    """生成一个**陌生**场景（每次随机，训练时没见过）。"""
    s = {}
    n_obs = rng.randint(0, int(3 * difficulty) + 1)
    s["obstacles"] = [(float(rng.uniform(0.6, 4.0)), float(rng.uniform(-np.pi, np.pi)))
                      for _ in range(n_obs)]
    if rng.rand() < 0.7:
        s["target"] = (float(rng.uniform(0.5, 4.0)), float(rng.uniform(-np.pi, np.pi)))
    if rng.rand() < 0.6:
        s["sound"] = (float(rng.uniform(1.0, 6.0)), float(rng.uniform(-np.pi, np.pi)),
                      float(rng.uniform(0.3, 1.0)))
    if rng.rand() < 0.4:
        s["person"] = (float(rng.uniform(1.0, 3.0)), float(rng.uniform(-np.pi, np.pi)))
    s["light"] = float(rng.uniform(0.2, 1.0))
    s["temp"] = float(rng.uniform(15, 35))
    s["battery"] = float(rng.uniform(0.15, 1.0))
    s["novelty"] = float(rng.uniform(0.0, 1.0))
    s["tilt"] = float(rng.uniform(-0.3, 0.3))
    s["touch"] = bool(rng.rand() < 0.1)
    return s



def _evolve_scene(sc, rng, drift=0.08):
    """场景平滑演化（模拟真实世界里相邻时刻的变化，不是每步全新随机）。"""
    out = dict(sc)
    out["obstacles"] = [(max(0.2, d + rng.normal(0, drift)), ang + rng.normal(0, drift))
                        for d, ang in sc.get("obstacles", [])]
    if "target" in sc:
        d, ang = sc["target"]
        out["target"] = (max(0.3, d + rng.normal(0, drift)), ang + rng.normal(0, drift))
    if "sound" in sc:
        d, ang, l = sc["sound"]
        out["sound"] = (max(0.5, d + rng.normal(0, drift)), ang + rng.normal(0, drift),
                        min(1.0, max(0.1, l + rng.normal(0, drift * 0.3))))
    if "person" in sc:
        d, ang = sc["person"]
        out["person"] = (max(0.5, d + rng.normal(0, drift)), ang + rng.normal(0, drift))
    out["light"] = min(1.0, max(0.0, sc.get("light", 0.6) + rng.normal(0, drift * 0.3)))
    out["temp"] = sc.get("temp", 24) + rng.normal(0, drift * 2)
    out["battery"] = min(1.0, max(0.0, sc.get("battery", 1.0) - 0.0008 + rng.normal(0, 0.002)))
    out["novelty"] = min(1.0, max(0.0, sc.get("novelty", 0.5) * 0.995))
    out["tilt"] = sc.get("tilt", 0.0) + rng.normal(0, drift * 0.3)
    return out


def _behaviour_quality(actions, winners):
    """行为自然度：动作平滑度 + 切换频率（人不该每步突变）。"""
    if len(actions) < 2:
        return 1.0, 0.0
    A = np.stack(actions)
    # 平滑度：相邻动作差的平均（越小越平滑）
    jerk = float(np.mean(np.linalg.norm(np.diff(A, axis=0), axis=1)))
    # 切换频率
    switches = sum(1 for i in range(1, len(winners)) if winners[i] != winners[i - 1])
    return jerk, switches / len(winners)


def selftest():
    print("  小沙本能系统 selftest（零任务训练 / 陌生场景）")
    ok = 0
    total = 0

    # 1) 多感觉编码：威胁/目标/社交/稳态都能算出来
    total += 1
    s = {"obstacles": [(0.4, 0.5)], "target": (1.0, -0.3), "sound": (2.0, 0.8, 0.9),
         "battery": 0.15, "temp": 33, "light": 0.95, "novelty": 0.8, "tilt": 0.2}
    sig = Senses.encode(s)
    print("  [1] 多感觉编码: 威胁=%.2f 目标=%.2f 社交=%.2f 电量=%.2f"
          % (sig["threat"], sig["goal"], sig["social"], sig["battery"]))
    ok += int(sig["threat"] > 0.5 and sig["goal"] > 0.3 and sig["social"] > 0.2)

    # 2) 本能触发：危险场景 → 避障胜出
    total += 1
    mind = InstinctMind()
    act, info = mind.step({"obstacles": [(0.25, 0.0)], "battery": 1.0})
    print("  [2] 危险场景 → 胜出本能: %s (动作 vx=%.2f)" % (info["winner"], act[0]))
    ok += int(info["winner"] == "avoid" and act[0] < 0)

    # 3) 目标场景 → 趋近胜出
    total += 1
    mind = InstinctMind()
    act, info = mind.step({"target": (0.8, 0.2), "battery": 1.0, "novelty": 0.2})
    print("  [3] 目标场景 → 胜出本能: %s" % info["winner"])
    ok += int(info["winner"] == "approach")

    # 4) 社交场景 → 朝向/跟随
    total += 1
    mind = InstinctMind()
    act, info = mind.step({"person": (1.2, 0.6), "battery": 1.0, "novelty": 0.1})
    print("  [4] 有人靠近 → 胜出本能: %s" % info["winner"])
    ok += int(info["winner"] in ("social", "orient"))

    # 5) 低电 → 稳态优先
    total += 1
    mind = InstinctMind()
    act, info = mind.step({"battery": 0.05, "target": (1.0, 0.0)})
    print("  [5] 电量 5%% → 胜出本能: %s" % info["winner"])
    ok += int(info["winner"] == "homeostasis")

    # 6) 陌生场景零样本：200 个随机新场景，全部能给出合法动作
    total += 1
    mind = InstinctMind()
    rng = np.random.RandomState(7)
    valid = 0
    acts = []
    wins = []
    t0 = time.perf_counter()
    N = 200
    for _ in range(N):
        sc = _random_scene(rng)
        a, inf = mind.step(sc)
        acts.append(a)
        wins.append(inf["winner"])
        if np.all(np.isfinite(a)) and np.all(np.abs(a) <= 2.0):
            valid += 1
    el = (time.perf_counter() - t0) / N * 1000
    print("  [6] 200 个陌生场景: 合法动作 %d/%d，平均 %.2f ms/步" % (valid, N, el))
    ok += int(valid == N and el < 5.0)

    # 7) 行为自然度：在**连续演化**的场景里测（真实世界相邻时刻是相似的）
    #    随机场景下切换率高是正常的（环境每步全变），不能拿它判断"跳不跳"
    total += 1
    mind_c = InstinctMind()
    rng_c = np.random.RandomState(11)
    sc_c = _random_scene(rng_c)
    acts_c, wins_c = [], []
    for _ in range(300):
        sc_c = _evolve_scene(sc_c, rng_c)
        a, inf = mind_c.step(sc_c)
        acts_c.append(a); wins_c.append(inf["winner"])
    jerk_c, sw_c = _behaviour_quality(acts_c, wins_c)
    jerk_r, sw_r = _behaviour_quality(acts, wins)
    print("  [7] 行为自然度: 连续场景 jerk=%.3f 切换率=%.1f%% | 随机场景 jerk=%.3f 切换率=%.1f%%"
          % (jerk_c, sw_c * 100, jerk_r, sw_r * 100))
    ok += int(jerk_c < 0.35 and sw_c < 0.25)

    # 8) 多样性：不是永远一个动作（否则"僵死"）
    total += 1
    A = np.stack(acts)
    uniq = len(set(tuple(np.round(a, 2)) for a in A))
    print("  [8] 行为多样性: %d 种不同动作 / %d 步" % (uniq, N))
    ok += int(uniq > 15)

    # 9) 在线记忆：一次经历就能复用
    total += 1
    mind = InstinctMind()
    sc = {"obstacles": [(0.5, 0.3)], "battery": 1.0}
    a1, i1 = mind.step(sc)
    mind.learn_from_outcome(sc, a1, +1.0)
    a2, i2 = mind.step(sc)          # 同一场景再来
    st = mind.memory.stats()
    print("  [9] 在线记忆: 第二次 reused=%s，记忆 %d 条，命中率 %.0f%%"
          % (i2["reused_memory"], st["size"], st["hit_rate"] * 100))
    ok += int(i2["reused_memory"] and st["size"] == 1)

    # 10) 记忆不会拖慢决策
    total += 1
    mind = InstinctMind(memory_capacity=200)
    rng = np.random.RandomState(3)
    for _ in range(150):
        sc = _random_scene(rng)
        a, _ = mind.step(sc)
        mind.learn_from_outcome(sc, a, float(rng.uniform(-1, 1)))
    t0 = time.perf_counter()
    for _ in range(100):
        mind.step(_random_scene(rng))
    el = (time.perf_counter() - t0) / 100 * 1000
    print("  [10] 200 条记忆下决策: %.2f ms/步（要求 <8ms）" % el)
    ok += int(el < 8.0)

    print("  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        # 演示：陌生场景 → 行为
        rng = np.random.RandomState(int(time.time()) % 1000)
        mind = InstinctMind()
        print("=== 陌生场景演示（每步随机新场景）===")
        for step in range(12):
            sc = _random_scene(rng)
            a, inf = mind.step(sc)
            print("步%2d 威胁%.2f 目标%.2f 社交%.2f 电量%.2f → %-12s vx=%+.2f vyaw=%+.2f (%.2fms)"
                  % (step, inf["sig"]["threat"], inf["sig"]["goal"], inf["sig"]["social"],
                     inf["sig"]["battery"], inf["winner"], a[0], a[2], inf["ms"]))
        sys.exit(0)
    sys.exit(0 if selftest() else 1)
