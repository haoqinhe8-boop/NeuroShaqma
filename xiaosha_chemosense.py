# -*- coding: utf-8 -*-
"""小沙的化学感官 + 联想学习（对标果蝇核心能力）。

果蝇（Drosophila）在这三件事上是神经科学的标准模型，我们逐项对齐：

  ① 味觉 Gustation      —— 果蝇有 5 类味觉受体神经元（甜/苦/水/咸/酸），
                            分布在唇瓣、腿、翅缘。甜→取食，苦→拒食。
  ② 联想学习 Associate  —— Tully & Quinn (1985) T 迷宫范式：
                            气味(CS) + 电击(US) → 一次训练就学会躲避该气味。
                            果蝇单次训练记忆可持续数小时（短时记忆）。
  ③ 习惯化/敏感化       —— 非联想学习：重复弱刺激 → 响应下降（习惯化）；
                            强刺激后 → 对所有刺激更敏感（敏感化）。
  ④ 趋化导航 Chemotaxis —— 果蝇找食物：沿气味羽流"之字形"前进（casting），
                            遇到高浓度就直冲（surging）。这是真实的行为策略。

全部零梯度下降（联想学习是一次写入的 Hebbian/dopamine 式更新）。
"""
import numpy as np

__version__ = "1.0.0"


# ==================== ① 味觉 Gustation ====================

class Gustation:
    """味觉：5 类受体（对标果蝇的 Gr 受体家族）。

    果蝇味觉受体神经元（GRN）分类与行为：
      甜（Gr5a/Gr64f）  → 糖 → 取食（伸喙 PER 反射）
      苦（Gr66a 等）    → 毒素 → 拒食（抑制伸喙）
      水（PPK28）       → 缺水时强烈取食
      咸（低浓度取食/高浓度拒食，双相）
      酸（拒食，过高浓度有害）

    输出 8 维：
      0 甜  1 苦  2 水  3 咸  4 酸
      5 总浓度
      6 取食倾向（-1 拒食 ~ +1 取食）
      7 唤醒度（强烈味觉 → 提高警觉）
    """

    # 双相响应：低浓度有利、高浓度有害（咸/酸的典型特性）
    SALTY_OPTIMAL = 0.15
    ACID_OPTIMAL = 0.05

    def __init__(self):
        self.dim = 8
        self.adapt = np.zeros(5, dtype=np.float32)   # 味觉适应（持续吃会腻）

    def encode(self, chemicals, hunger=0.5):
        """chemicals: dict，可含 sweet/bitter/water/salty/sour（0-1）。

        hunger: 0=很饱 1=很饿（饥饿提升甜/水的价值，这是果蝇的稳态调节）
        """
        c = chemicals or {}
        sweet = float(np.clip(c.get("sweet", 0.0), 0, 1))
        bitter = float(np.clip(c.get("bitter", 0.0), 0, 1))
        water = float(np.clip(c.get("water", 0.0), 0, 1))
        salty = float(np.clip(c.get("salty", 0.0), 0, 1))
        sour = float(np.clip(c.get("sour", 0.0), 0, 1))

        # 咸和酸是双相：偏离最适浓度就变有害
        salty_val = np.exp(-((salty - self.SALTY_OPTIMAL) ** 2) / 0.02) if salty > 0 else 0.0
        salty_bad = max(0.0, salty - 0.35) * 1.5
        sour_val = np.exp(-((sour - self.ACID_OPTIMAL) ** 2) / 0.01) if sour > 0 else 0.0
        sour_bad = max(0.0, sour - 0.2) * 1.5

        raw = np.array([sweet, bitter, water, salty, sour], dtype=np.float32)
        # 适应（持续刺激 → 响应下降）
        resp = np.maximum(raw - self.adapt, 0.0)
        self.adapt = 0.85 * self.adapt + 0.15 * raw

        total = float(resp.sum())

        # 取食倾向：甜/水促食，苦/酸/高盐抑食；饥饿放大甜和水的价值
        drive = (resp[0] * (0.6 + 0.8 * hunger) + resp[2] * (0.4 + 0.8 * hunger)
                 - resp[1] * 1.4 - sour_bad - salty_bad + salty_val * 0.3)
        feeding = float(np.tanh(drive))

        arousal = float(np.clip(total * 1.2 + max(bitter, sour_bad, salty_bad), 0, 1))

        return np.array([sweet, bitter, water, salty, sour, total, feeding, arousal],
                        dtype=np.float32)


# ==================== ② 联想学习（Tully-Quinn T 迷宫）====================

class AssociativeMemory:
    """果蝇式嗅觉联想学习（气味 CS + 电击/糖 US）。

    果蝇特性（真实数据）：
      - 单次训练（1 分钟）就能形成记忆
      - 短时记忆持续 1-3 小时
      - 间隔重复训练 → 长时记忆（可数天）
      - 记忆会消退（extinction）
      - 反转学习（reversal）：学会 A 好之后，改成 A 坏，果蝇也能重新学

    实现：每个气味一个"价值"（valence），US 出现时按多巴胺式更新：
      被电击的气味 → 价值下降
      被糖奖励的气味 → 价值上升
      未配对的气味 → 保持中性
    """

    def __init__(self, n_odor=8, lr=0.45, decay=0.002):
        self.n_odor = n_odor
        self.lr = lr                 # 一次学习的效果（果蝇很强 → 0.45）
        self.decay = decay           # 每个时间步的遗忘
        self.valence = np.zeros(n_odor, dtype=np.float32)
        self.exposures = np.zeros(n_odor, dtype=np.int32)
        self.history = []            # (odor, outcome, 更新前后的价值)

    def train(self, odor_id, outcome, novelty_bonus=0.0):
        """一次配对训练。outcome: +1 糖 / -1 电击 / 0 无。

        novelty_bonus: 新奇气味学得更快（果蝇的真实特性）
        """
        if not (0 <= odor_id < self.n_odor):
            return 0.0
        before = float(self.valence[odor_id])
        eff_lr = min(0.9, self.lr + novelty_bonus)
        self.valence[odor_id] += eff_lr * (outcome - self.valence[odor_id] * 0.35)
        self.valence[odor_id] = float(np.clip(self.valence[odor_id], -1.0, 1.0))
        self.exposures[odor_id] += 1
        self.history.append((odor_id, outcome, before, float(self.valence[odor_id])))
        return float(self.valence[odor_id])

    def predict(self, odor_id):
        """预测该气味的结果（-1 坏 ~ +1 好）。"""
        if not (0 <= odor_id < self.n_odor):
            return 0.0
        return float(self.valence[odor_id])

    def choose(self, odor_a, odor_b, noise=0.08, rng=None):
        """T 迷宫二选一：返回值更高的那个（带一点探索噪声）。"""
        r = rng or np.random
        va = self.predict(odor_a) + r.normal(0, noise)
        vb = self.predict(odor_b) + r.normal(0, noise)
        return odor_a if va >= vb else odor_b

    def extinction(self, odor_id, steps=10):
        """消退：反复呈现该气味但不给 US → 价值回归中性。"""
        for _ in range(steps):
            self.valence[odor_id] *= (1 - self.lr * 0.5)

    def tick(self):
        """时间流逝 → 遗忘（真实记忆的衰减）。"""
        self.valence *= (1 - self.decay)

    def stats(self):
        return {"learned": int((np.abs(self.valence) > 0.15).sum()),
                "strongest": float(np.abs(self.valence).max()) if self.valence.size else 0.0,
                "mean_abs": float(np.abs(self.valence).mean())}


# ==================== ③ 习惯化 / 敏感化 ====================

class Habituation:
    """非联想学习（果蝇的两种最基础可塑性）。

    习惯化：重复同样的弱刺激 → 响应递减（不再重要，省能量）
    敏感化：一次强刺激后 → 对所有刺激响应增强（提高警觉）

    真实数据：果蝇的惊跳反射（startle）在重复刺激下快速习惯化，
    受到强刺激后又会敏感化（dishabituation）。
    """

    def __init__(self, n_stim=8, hab_rate=0.25, sens_rate=0.30):
        self.n_stim = n_stim
        self.hab_rate = hab_rate
        self.sens_rate = sens_rate
        self.response_gain = np.ones(n_stim, dtype=np.float32)   # 每个刺激的增益
        self.global_sens = 1.0                                    # 全局敏感化

    def respond(self, stim_id, intensity, is_strong=False):
        """返回实际响应强度（0-1 之间，受习惯化/敏感化调制）。"""
        if not (0 <= stim_id < self.n_stim):
            return float(intensity) * self.global_sens
        # 强刺激 → 全局敏感化
        if is_strong:
            self.global_sens = min(2.0, self.global_sens + self.sens_rate)
            # 强刺激也会部分重置该刺激的习惯化（去习惯化）
            self.response_gain[stim_id] = min(1.0, self.response_gain[stim_id] + 0.3)
        out = float(intensity) * self.response_gain[stim_id] * self.global_sens
        # 重复 → 习惯化（只对弱刺激）
        if not is_strong:
            self.response_gain[stim_id] *= (1 - self.hab_rate * float(intensity))
            self.response_gain[stim_id] = max(0.1, self.response_gain[stim_id])
        # 全局敏感化缓慢回落
        self.global_sens = max(1.0, self.global_sens * 0.995)
        return float(np.clip(out, 0, 2.0))

    def spontaneous_recovery(self, rate=0.05):
        """久不刺激 → 习惯化自然恢复。"""
        self.response_gain = np.minimum(1.0, self.response_gain + rate)


# ==================== ④ 趋化导航（气味羽流追踪）====================

class Chemotaxis:
    """果蝇找食物：沿气味羽流之字形前进（真实行为策略）。

    真实果蝇的趋化算法（Gomez-Marin & Louis 2012）：
      - 逆风定向（upwind）：沿着风往上风方向走
      - 横风摆动（casting）：失去气味时左右摆动搜索
      - 直冲（surging）：闻到高浓度就直冲
      - 陡度感知（temporal gradient）：比较前后时刻的浓度判断"上坡"

    实现：只用**时间梯度**（真实昆虫不比较左右，而是比较前后时刻）。
    """

    def __init__(self, surge_thresh=0.02, cast_interval=8):
        self.surge_thresh = surge_thresh   # 浓度上升超过这个值就直冲
        self.cast_interval = cast_interval # 多久没进展就开始横风摆动
        self.prev_conc = None
        self.no_progress = 0
        self.state = "search"              # search / surge / cast

    def step(self, concentration, heading=0.0, wind_dir=0.0):
        """返回 (前进速度 vx, 转向 vyaw, 状态名)。

        concentration: 当前气味浓度（0-1）
        heading: 当前朝向（弧度）
        wind_dir: 风向（气味羽流方向，弧度）
        """
        if self.prev_conc is None:
            self.prev_conc = concentration
            return 0.5, 0.0, "search"

        delta = concentration - self.prev_conc
        self.prev_conc = concentration

        # 已到达食物源（浓度接近饱和）→ 减速停下取食（真实果蝇行为）
        if concentration > 0.85:
            self.state = "feeding"
            return 0.05, 0.0, "feeding"

        if delta > self.surge_thresh:
            # 浓度上升 → 直冲
            self.state = "surge"
            self.no_progress = 0
            # 朝风来的方向（逆风）前进
            err = np.arctan2(np.sin(wind_dir - heading), np.cos(wind_dir - heading))
            return 1.0, float(np.clip(err * 1.5, -1.0, 1.0)), "surge"

        if delta < -0.01:
            # 浓度下降 → 掉头
            self.state = "cast"
            self.no_progress += 1
            return 0.45, 0.7 * (1 if self.no_progress % 2 == 0 else -1), "cast"

        # 浓度不变 → 横风摆动搜索
        self.no_progress += 1
        if self.no_progress > self.cast_interval:
            self.state = "cast"
            sweep = 0.6 * np.sin(self.no_progress * 0.9)
            return 0.55, float(sweep), "cast"

        self.state = "search"
        err = np.arctan2(np.sin(wind_dir - heading), np.cos(wind_dir - heading))
        return 0.6, float(np.clip(err * 0.8, -0.6, 0.6)), "search"


# ==================== 自测 ====================

def selftest():
    print("  小沙化学感官 + 联想学习 selftest（对标果蝇）")
    ok = 0
    total = 0

    # ---- 1. 味觉：甜→取食，苦→拒食 ----
    total += 1
    g = Gustation()
    f_sugar = g.encode({"sweet": 0.8}, hunger=0.8)
    g2 = Gustation()
    f_quinine = g2.encode({"bitter": 0.8}, hunger=0.8)
    print("  [1] 味觉: 糖→取食倾向 %+.2f | 苦→%+.2f" % (f_sugar[6], f_quinine[6]))
    ok += int(f_sugar[6] > 0.3 and f_quinine[6] < -0.3)

    # ---- 2. 味觉：饥饿调节（饿的时候糖更值钱）----
    total += 1
    g_fed = Gustation(); g_hungry = Gustation()
    fed = g_fed.encode({"sweet": 0.5}, hunger=0.1)[6]
    hungry = g_hungry.encode({"sweet": 0.5}, hunger=0.95)[6]
    print("  [2] 饥饿调节: 饱时 %+.2f → 饿时 %+.2f" % (fed, hungry))
    ok += int(hungry > fed + 0.1)

    # ---- 3. 味觉：双相（盐低浓度好，高浓度坏）----
    total += 1
    g_low = Gustation(); g_high = Gustation()
    low = g_low.encode({"salty": 0.15}, hunger=0.5)[6]
    high = g_high.encode({"salty": 0.8}, hunger=0.5)[6]
    print("  [3] 味觉双相: 低盐 %+.2f → 高盐 %+.2f" % (low, high))
    ok += int(low > high)

    # ---- 4. 联想学习：一次训练就学会（果蝇核心）----
    total += 1
    m = AssociativeMemory()
    m.train(3, -1.0)                  # 气味3 + 电击
    v_after = m.predict(3)
    acc = 0
    rng = np.random.RandomState(0)
    for _ in range(100):
        c = m.choose(3, 5, rng=rng)   # 气味3 vs 中性气味5
        acc += int(c == 5)
    print("  [4] 联想学习(单次): 气味3价值 %+.2f，T迷宫避开率 %d%%" % (v_after, acc))
    ok += int(v_after < -0.4 and acc >= 90)

    # ---- 5. 联想学习：反向（奖励=趋近）----
    total += 1
    m2 = AssociativeMemory()
    m2.train(1, +1.0)                 # 气味1 + 糖
    acc2 = 0
    rng = np.random.RandomState(1)
    for _ in range(100):
        acc2 += int(m2.choose(1, 6, rng=rng) == 1)
    print("  [5] 联想学习(奖励): 气味1价值 %+.2f，T迷宫趋近率 %d%%" % (m2.predict(1), acc2))
    ok += int(m2.predict(1) > 0.4 and acc2 >= 90)

    # ---- 6. 消退（extinction）----
    total += 1
    m3 = AssociativeMemory()
    m3.train(2, -1.0)
    before = m3.predict(2)
    m3.extinction(2, steps=10)
    print("  [6] 消退学习: 消退前 %+.2f → 消退后 %+.2f" % (before, m3.predict(2)))
    ok += int(abs(m3.predict(2)) < abs(before) * 0.5)

    # ---- 7. 反转学习（reversal，果蝇能做）----
    total += 1
    m4 = AssociativeMemory()
    m4.train(0, -1.0)                 # 先学：气味0 坏
    m4.train(0, +1.0)                 # 反转：气味0 好
    m4.train(0, +1.0)
    print("  [7] 反转学习: 坏→好 后价值 %+.2f" % m4.predict(0))
    ok += int(m4.predict(0) > 0.2)

    # ---- 8. 习惯化 ----
    total += 1
    h = Habituation()
    r1 = h.respond(0, 0.5)
    for _ in range(8):
        r_last = h.respond(0, 0.5)
    print("  [8] 习惯化: 第1次 %.2f → 第9次 %.2f（应下降）" % (r1, r_last))
    ok += int(r_last < r1 * 0.6)

    # ---- 9. 敏感化 / 去习惯化 ----
    total += 1
    h2 = Habituation()
    for _ in range(8):
        h2.respond(1, 0.3)
    low_before = h2.respond(1, 0.3)
    h2.respond(2, 1.0, is_strong=True)      # 强刺激
    low_after = h2.respond(1, 0.3)
    print("  [9] 敏感化: 强刺激前 %.3f → 后 %.3f（应上升）" % (low_before, low_after))
    ok += int(low_after > low_before)

    # ---- 10. 趋化：能找到食物源（浓度上升 → 直冲）----
    total += 1
    cx = Chemotaxis()
    # 模拟浓度梯度：越靠近源头浓度越高
    pos = 0.0
    conc = lambda p: float(np.exp(-((1.0 - p) ** 2) / 0.5))
    heading = 0.0
    for _ in range(60):
        c = conc(pos)
        vx, vyaw, st = cx.step(c, heading, wind_dir=0.0)
        heading += vyaw * 0.2
        pos += vx * 0.05 * np.cos(heading)
    print("  [10] 趋化导航: 60 步后位置 %.2f（源头 1.0），浓度 %.2f，状态 %s" % (pos, conc(pos), cx.state))
    ok += int(conc(pos) > 0.8)

    # ---- 11. 趋化：无气味时搜索不失控 ----
    total += 1
    cx2 = Chemotaxis()
    states = []
    for i in range(40):
        _, _, st = cx2.step(0.0, 0.0, 0.0)
        states.append(st)
    uniq = set(states)
    print("  [11] 无气味搜索: 状态种类 %s（应有搜索/摆动）" % sorted(uniq))
    ok += int(len(uniq) >= 1 and "search" in uniq or "cast" in uniq)

    print("  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
