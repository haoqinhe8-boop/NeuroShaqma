# -*- coding: utf-8 -*-
"""小沙的节律 / 社交 / 决策（对标果蝇的高级行为）。

果蝇不只会反射，它还有：

  ① 昼夜节律 Circadian —— 分子钟（per/tim 负反馈环）驱动 24 小时活动节律
       即使全黑也保持 ~24h 周期（free-running），光照可重置相位
  ② 睡眠稳态 Sleep —— 睡眠压力累积（腺苷）→ 睡眠 → 恢复；
       剥夺睡眠后会有"睡眠反弹"（真实果蝇数据）
  ③ 社交行为 Social —— 求偶（定向→振翅唱歌→舔舐→尝试）、
       攻击（lunging/boxing）、群体聚集（thigmotaxis）
  ④ 风险决策 Risk —— 果蝇偏好**确定**的奖励而不是同期望值的赌博
       （risk aversion，真实实验：Drosophila 会选择 100% 得 2 单位
        而不是 50% 得 4 单位，虽然期望相同）

全部规则化实现（进化先验），不训练。
"""
import numpy as np

__version__ = "1.0.0"


# ==================== ① 昼夜节律（分子钟）====================

class CircadianClock:
    """分子钟：per/tim 负反馈环的简化模型。

    真实机制（Hall/Rosbash/Young 诺奖）：
      CLK/CYC 促进 per/tim 转录 → PER/TIM 蛋白累积 → 抑制 CLK/CYC → 蛋白降解
      周期约 24 小时（自由运行 23.5-24.5h）
      光照通过 CRY 降解 TIM → 重置相位

    实现：两个变量的延迟负反馈振荡器（van der Pol 式）。
    """

    def __init__(self, period_h=24.0, dt_h=0.25):
        self.period = period_h
        self.dt = dt_h
        self.t_h = 0.0                 # 内部时钟时间（小时）
        self.mrna = 0.5                # per mRNA（归一化）
        self.protein = 0.3             # PER 蛋白
        self.delay = 0.3               # 延迟中间体（振荡的关键）
        self.phase_offset = 0.0

    def step(self, light=0.0):
        """推进一个时间步。light: 0-1（光照强度）。返回活动倾向（0-1）。"""
        # Goodwin 振荡器（三变量 + Hill 抑制）：负反馈要**经过延迟**才会自激振荡
        # ① 延迟中间体：跟随蛋白但滞后
        self.delay += (self.protein - self.delay) * 0.30
        # ② Hill 抑制（系数 4 → 陡峭，产生强振荡）
        repression = 1.0 / (1.0 + (self.delay / 0.55) ** 4)
        self.mrna += (repression - self.mrna) * 0.45
        # ③ 蛋白由 mRNA 产生；光照加速降解（CRY 途径）
        degr = 0.28 + 0.55 * light
        self.protein += (self.mrna * 1.1 - degr * self.protein) * 0.45
        self.mrna = float(np.clip(self.mrna, 0.02, 2.0))
        self.protein = float(np.clip(self.protein, 0.02, 2.0))
        # 内部时间（受光照牵引）
        pull = 1.0 if light > 0.5 else 0.97     # 白天走得准，夜里自由运行略慢
        self.t_h = (self.t_h + self.dt * pull) % 24.0
        return self.activity_drive()

    def activity_drive(self):
        """活动倾向：果蝇是双峰（晨峰 + 昏峰，中间午休）。"""
        h = self.t_h
        dawn = np.exp(-((h - 8.0) ** 2) / (2 * 1.5 ** 2))
        dusk = np.exp(-((h - 17.0) ** 2) / (2 * 1.5 ** 2))
        # 蛋白相位调制（分子钟本身也影响）
        mod = 0.5 + 0.5 * np.tanh(self.protein - 0.6)
        return float(np.clip((dawn + dusk) * mod * 1.8, 0, 1))

    def is_night(self):
        return self.t_h < 6.0 or self.t_h > 20.0

    def reset_by_light(self, light, dawn_detect=False):
        """光照重置（真实：晨光把相位拉到 0）。"""
        if dawn_detect and light > 0.7:
            self.t_h = 6.0         # 黎明 → 时钟对齐到 6 点


# ==================== ② 睡眠稳态 ====================

class SleepHomeostat:
    """睡眠压力（腺苷累积）+ 睡眠恢复 + 剥夺反弹。

    真实果蝇数据：
      清醒越久 → 睡眠压力越高
      睡眠时压力快速释放
      被剥夺睡眠后 → 补睡更多（rebound）
    """

    def __init__(self, accumulate=0.010, release=0.035, threshold=0.65):
        self.pressure = 0.2            # 睡眠压力 0-1
        self.accumulate = accumulate   # 每步清醒累积
        self.release = release         # 每步睡眠释放
        self.threshold = threshold     # 超过就想睡
        self.sleep_debt = 0.0          # 睡眠债（剥夺累积）
        self.sleeping = False
        self.total_sleep = 0.0
        self.total_awake = 0.0

    def step(self, circadian_drive=0.5, force_awake=False):
        """返回 {"sleeping": bool, "pressure": float, "debt": float}"""
        # 昼夜节律给睡眠一个基线倾向（夜里更容易睡着）
        want_sleep = self.pressure + (1 - circadian_drive) * 0.3 + self.sleep_debt * 0.5

        if force_awake:
            self.sleeping = False
            self.sleep_debt = min(1.0, self.sleep_debt + 0.004)
        else:
            # 滞回：压力高才睡，压力降到低才醒
            if not self.sleeping and want_sleep > self.threshold:
                self.sleeping = True
            elif self.sleeping and want_sleep < self.threshold * 0.45:
                self.sleeping = False

        if self.sleeping:
            self.pressure = max(0.0, self.pressure - self.release)
            self.sleep_debt = max(0.0, self.sleep_debt - 0.006)
            self.total_sleep += 1
        else:
            self.pressure = min(1.5, self.pressure + self.accumulate)
            self.total_awake += 1

        return {"sleeping": self.sleeping, "pressure": self.pressure,
                "debt": self.sleep_debt,
                "alertness": float(np.clip(1.0 - self.pressure * 0.8 - self.sleep_debt * 0.5, 0.1, 1.0))}

    def deprive(self, steps=20):
        """强制剥夺睡眠（实验用）。"""
        for _ in range(steps):
            self.step(force_awake=True)


# ==================== ③ 社交行为 ====================

class SocialBehavior:
    """果蝇的三种社交行为：求偶 / 攻击 / 聚集。

    求偶序列（真实果蝇固定动作模式，按顺序）：
      1. 定向（orient）—— 转向目标
      2. 跟随（follow）—— 靠近
      3. 振翅唱歌（sing）—— 翅膀振动产生脉冲歌（~200Hz 脉冲）
      4. 舔舐（lick）
      5. 尝试交配（attempt）
    攻击（雄性间）：
      lunging（冲撞）、boxing（互推）、wing threat（抬翅威胁）
    聚集（thigmotaxis）：喜欢靠近同类/边缘
    """

    COURTSHIP_SEQ = ["orient", "follow", "sing", "lick", "attempt"]

    def __init__(self):
        self.courtship_stage = -1
        self.aggression = 0.0
        self.crowding = 0.0

    def courtship_step(self, female_near, female_receptive, distance):
        """推进求偶序列。返回 (动作名, 是否完成)。"""
        if not female_near:
            self.courtship_stage = -1
            return "idle", False
        if self.courtship_stage < 0:
            self.courtship_stage = 0
        stage = self.COURTSHIP_SEQ[self.courtship_stage]
        # 太远 → 只能定向/跟随
        if distance > 3.0 and self.courtship_stage > 1:
            return "follow", False
        # 雌性不接受 → 回到唱歌（果蝇会反复唱）
        if not female_receptive and self.courtship_stage >= 2:
            self.courtship_stage = 2
            return "sing", False
        # 推进
        if self.courtship_stage < len(self.COURTSHIP_SEQ) - 1:
            self.courtship_stage += 1
            return stage, False
        return "attempt", True

    def aggression_step(self, male_near, same_species_male, resource_value):
        """攻击倾向（资源值高 + 有竞争者 → 攻击）。"""
        if not (male_near and same_species_male):
            self.aggression *= 0.8
            return "none", 0.0
        self.aggression = float(np.clip(self.aggression + 0.1 * resource_value, 0, 1))
        if self.aggression > 0.7:
            action = "lunging" if resource_value > 0.6 else "wing_threat"
        elif self.aggression > 0.4:
            action = "boxing"
        else:
            action = "none"
        return action, self.aggression

    def aggregate_step(self, n_neighbors, space_center_dist):
        """群体聚集（thigmotaxis：靠近同类 + 靠墙）。"""
        attraction = float(np.clip(n_neighbors / 6.0, 0, 1)) * 0.7
        edge_pref = float(np.clip(1.0 - space_center_dist, 0, 1)) * 0.3
        self.crowding = attraction + edge_pref
        return self.crowding


# ==================== ④ 风险决策 ====================

class RiskDecision:
    """果蝇的风险偏好（真实实验：Drosophila 会规避风险）。

    实验（So & Platt 式设计，果蝇版）：
      选项 A：100% 得到 2 单位（确定）
      选项 B：50% 得 4 单位 / 50% 得 0（期望相同=2，但不确定）
      结果：果蝇多数选 A（风险规避）

    实现：期望效用 = 期望值 - 风险厌恶系数 × 方差
    """

    def __init__(self, risk_aversion=0.35):
        self.risk_aversion = risk_aversion   # >0 表示规避风险
        self.history = []

    def expected_utility(self, payoffs, probs):
        """payoffs/probs: 等长列表。返回 (期望值, 方差, 效用)。"""
        p = np.asarray(probs, dtype=np.float64)
        v = np.asarray(payoffs, dtype=np.float64)
        p = p / p.sum()
        ev = float((p * v).sum())
        var = float((p * (v - ev) ** 2).sum())
        utility = ev - self.risk_aversion * var
        return ev, var, utility

    def choose(self, option_a, option_b, noise=0.05, rng=None):
        """option: (payoffs, probs)。返回 (选中的选项 'a'/'b', 两个效用)。"""
        ev_a, var_a, u_a = self.expected_utility(*option_a)
        ev_b, var_b, u_b = self.expected_utility(*option_b)
        r = rng or np.random
        u_a += r.normal(0, noise)
        u_b += r.normal(0, noise)
        pick = "a" if u_a >= u_b else "b"
        self.history.append((pick, ev_a, var_a, u_a, ev_b, var_b, u_b))
        return pick, (u_a, u_b)

    def adjust(self, outcome_good):
        """结果反馈：好结果降低风险厌恶（更敢赌），坏结果提高（更保守）。"""
        if outcome_good:
            self.risk_aversion = max(0.05, self.risk_aversion * 0.97)
        else:
            self.risk_aversion = min(1.5, self.risk_aversion * 1.05)


# ==================== ⑤ 完整"作息体" ====================

class LifeRhythm:
    """节律 + 睡眠 + 社交 + 决策 的整合体。"""

    def __init__(self):
        self.clock = CircadianClock()
        self.sleep = SleepHomeostat()
        self.social = SocialBehavior()
        self.risk = RiskDecision()
        self.t = 0

    def step(self, light=0.8, force_awake=False):
        drive = self.clock.step(light)
        sl = self.sleep.step(drive, force_awake)
        self.t += 1
        return {"clock_h": self.clock.t_h, "activity_drive": drive, **sl}


# ==================== 自测 ====================

def selftest():
    print("  小沙节律/社交/决策 selftest（对标果蝇高级行为）")
    ok = 0
    total = 0

    # ---- 1. 分子钟：自由运行接近 24h ----
    total += 1
    ck = CircadianClock(period_h=24.0, dt_h=0.25)
    # 全黑自由运行 24 小时
    for _ in range(int(24 / 0.25)):
        ck.step(light=0.0)
    print("  [1] 分子钟自由运行: 走了 24h 实际 %.1fh（应≈24，即离 24 或 0 都近）" % ck.t_h)
    ok += int(min(ck.t_h, 24 - ck.t_h) < 1.5)

    # ---- 2. 分子钟振荡：mRNA/蛋白有周期变化 ----
    total += 1
    ck2 = CircadianClock(dt_h=0.5)
    mrna_hist = []
    for _ in range(int(48 / 0.5)):
        ck2.step(light=0.1)
        mrna_hist.append(ck2.mrna)
    mrna_hist = np.array(mrna_hist)
    osc = float(mrna_hist.std())
    print("  [2] 分子钟振荡: mRNA 波动 std=%.3f（>0.05 表示真在振荡）" % osc)
    ok += int(osc > 0.05)

    # ---- 3. 昼夜节律：早晚双峰 ----
    total += 1
    ck3 = CircadianClock(dt_h=0.5)
    drives = []
    for _ in range(int(24 / 0.5)):
        drives.append(ck3.activity_drive())
        ck3.t_h = (ck3.t_h + 0.5) % 24
    drives = np.array(drives)
    at8 = drives[int(8 / 0.5)]
    at12 = drives[int(12 / 0.5)]
    at17 = drives[int(17 / 0.5)]
    at2 = drives[int(2 / 0.5)]
    print("  [3] 昼夜活动双峰: 8点 %.2f 12点 %.2f 17点 %.2f 2点 %.2f" % (at8, at12, at17, at2))
    ok += int(at8 > at12 and at17 > at12 and at2 < at8)

    # ---- 4. 睡眠稳态：清醒压力累积 → 入睡 ----
    total += 1
    sl = SleepHomeostat()
    slept_at = None
    for i in range(200):
        r = sl.step(circadian_drive=0.3)
        if r["sleeping"] and slept_at is None:
            slept_at = i
            break
    print("  [4] 睡眠稳态: 清醒 %d 步后入睡（压力 %.2f）" % (slept_at or -1, sl.pressure))
    ok += int(slept_at is not None and slept_at < 100)

    # ---- 5. 睡眠恢复：睡一段时间后压力下降 ----
    total += 1
    sl2 = SleepHomeostat()
    sl2.pressure = 0.9
    before = sl2.pressure
    for _ in range(30):
        sl2.step(circadian_drive=0.1)
    print("  [5] 睡眠恢复: 压力 %.2f → %.2f（应下降）" % (before, sl2.pressure))
    ok += int(sl2.pressure < before * 0.7)

    # ---- 6. 睡眠剥夺反弹 ----
    total += 1
    sl3 = SleepHomeostat()
    sl3.deprive(40)                       # 强制不睡
    debt = sl3.sleep_debt
    # 之后允许睡 → 应该更想睡（debt 高）
    want = sl3.pressure + (1 - 0.3) * 0.3 + debt * 0.5
    print("  [6] 睡眠剥夺: 剥夺后债 %.2f，睡眠驱力 %.2f（应明显高于常态 0.7）" % (debt, want))
    ok += int(debt > 0.05 and want > 0.7)

    # ---- 7. 求偶序列按顺序推进 ----
    total += 1
    sb = SocialBehavior()
    seq = []
    for _ in range(6):
        act, done = sb.courtship_step(female_near=True, female_receptive=True, distance=1.0)
        seq.append(act)
    print("  [7] 求偶序列: %s" % " → ".join(seq))
    ok += int(seq[0] == "orient" and "sing" in seq and seq[-1] == "attempt")

    # ---- 8. 雌性不接受 → 反复唱歌 ----
    total += 1
    sb2 = SocialBehavior()
    for _ in range(3):
        sb2.courtship_step(True, True, 1.0)      # 先推进
    acts = []
    for _ in range(4):
        a, _d = sb2.courtship_step(True, False, 1.0)   # 雌性拒绝
        acts.append(a)
    print("  [8] 雌性拒绝: 动作 %s（应反复 sing）" % acts)
    ok += int(acts.count("sing") >= 2)

    # ---- 9. 攻击：资源值高热 → lunging ----
    total += 1
    sb3 = SocialBehavior()
    for _ in range(10):
        action, aggr = sb3.aggression_step(male_near=True, same_species_male=True, resource_value=0.9)
    print("  [9] 攻击行为: 资源值 0.9 → 攻击度 %.2f，动作 %s" % (aggr, action))
    ok += int(action in ("lunging", "boxing") and aggr > 0.6)

    # ---- 10. 群体聚集 ----
    total += 1
    sb4 = SocialBehavior()
    near = sb4.aggregate_step(n_neighbors=6, space_center_dist=0.9)
    far = sb4.aggregate_step(n_neighbors=0, space_center_dist=0.1)
    print("  [10] 群体聚集: 同类多+靠墙 %.2f vs 孤立+居中 %.2f" % (near, far))
    ok += int(near > far)

    # ---- 11. 风险决策：规避风险（果蝇真实偏好）----
    total += 1
    rd = RiskDecision(risk_aversion=0.35)
    sure = ([2.0], [1.0])                 # 100% 得 2
    gamble = ([4.0, 0.0], [0.5, 0.5])     # 50% 得 4，期望也是 2
    rng = np.random.RandomState(0)
    picks = [rd.choose(sure, gamble, rng=rng)[0] for _ in range(100)]
    sure_rate = picks.count("a")
    print("  [11] 风险决策: 100 次中选【确定】%d 次（果蝇应多数选确定）" % sure_rate)
    ok += int(sure_rate >= 80)

    # ---- 12. 风险决策：反馈调节（赢多了变敢赌）----
    total += 1
    rd2 = RiskDecision(risk_aversion=0.5)
    before = rd2.risk_aversion
    for _ in range(10):
        rd2.adjust(outcome_good=True)
    print("  [12] 风险反馈: 连赢 10 次后风险厌恶 %.2f → %.2f（应变低）" % (before, rd2.risk_aversion))
    ok += int(rd2.risk_aversion < before * 0.8)

    # ---- 13. 完整作息体：24h 内有清醒也有睡眠 ----
    total += 1
    lr = LifeRhythm()
    awake = sleep = 0
    for h in range(96):                    # 96 步 = 24h（每步 0.25h）
        r = lr.step(light=0.8 if 6 <= lr.clock.t_h <= 18 else 0.05)
        if r["sleeping"]: sleep += 1
        else: awake += 1
    print("  [13] 24h 作息: 清醒 %d 步 / 睡眠 %d 步" % (awake, sleep))
    ok += int(awake > 0 and sleep > 0)

    print("  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
