# -*- coding: utf-8 -*-
"""小沙的认知能力（对标果蝇的九项高级功能）。

果蝇不只会反射。实验证明它还能：

  ① 数量感 Numerosity       —— 能区分 1/2/3/4 个物体（Bengochea 2023）
  ② 时间感知 IntervalTiming —— 秒级间隔计时（起搏器-累加器模型）
  ③ 注意 SelectiveAttention —— 显著性竞争（biased competition）
  ④ 冲动控制 ImpulseControl —— 延迟满足（双曲折扣，等大奖励）
  ⑤ 社会学习 SocialLearning —— 观察同伴的选择来学
  ⑥ 群体决策 SwarmDecision  —— 法定人数（quorum）触发集体决定
  ⑦ 飞行 Flight             —— 翅膀 200Hz，靠振幅差滚转
  ⑧ 运动模式库 MotorRepertoire —— 梳理/攀爬/跳跃（固定动作模式）
  ⑨ 免疫行为 SicknessBehavior —— 感染后活动下降、避开同类

全部规则化实现（进化先验），零梯度训练。
"""
import numpy as np

__version__ = "1.0.0"


# ==================== ① 数量感 ====================

class NumerositySense:
    """近似数量系统（ANS）—— 果蝇能区分 1~4 个物体。

    生理依据（Bengochea et al. 2023, 果蝇视觉数量感）：
      果蝇能区分 1 vs 2、2 vs 3、3 vs 4 个物体
      但 4 vs 5 就吃力（韦伯定律：比值越接近越难）
      机制：累加器——每个物体贡献一个脉冲，总数 = 累加结果

    实现：累加器 + 韦伯分数（区分精度随数量下降）
    """

    WEBER = 0.22        # 韦伯分数（22%：3vs4 比值0.25 能分，4vs5 比值0.20 分不了，匹配果蝇）

    def __init__(self):
        self.accumulator = 0.0
        self.samples = []

    def accumulate(self, item_count, reliability=1.0):
        """看到一个场景（含 n 个物体）→ 累加一个估计。"""
        # 噪声与数量成正比（韦伯定律）
        noise = np.random.normal(0, max(0.1, item_count * self.WEBER * 0.35))
        est = max(0.0, item_count * reliability + noise)
        self.accumulator += est
        self.samples.append(item_count)
        return est

    def estimate(self):
        """当前累加器估计的数量。"""
        return float(self.accumulator)

    def discriminate(self, n_a, n_b):
        """能否区分 n_a 和 n_b（韦伯定律：|a-b|/max(a,b) > WEBER）。"""
        if max(n_a, n_b) == 0:
            return False
        ratio = abs(n_a - n_b) / max(n_a, n_b)
        return bool(ratio > self.WEBER)

    def reset(self):
        self.accumulator = 0.0
        self.samples = []


# ==================== ② 时间感知 ====================

class IntervalTimer:
    """起搏器-累加器间隔计时（果蝇能做秒级计时）。

    生理依据：
      起搏器（pacemaker）稳定发放脉冲
      开关（switch）在事件开始时打开
      累加器（accumulator）累计脉冲数 = 已过时间
      比较器（comparator）与记忆中的目标时长比较

    果蝇的真实精度：间隔标准差约为均值的 15-20%（标量计时理论）
    """

    def __init__(self, tick_hz=10.0, noise=0.12):
        self.tick_hz = tick_hz
        self.noise = noise
        self.accumulator = 0.0
        self.running = False
        self.t = 0.0

    def start(self):
        self.accumulator = 0.0
        self.running = True

    def tick(self, dt=0.1):
        """推进时间（累加器累加，带噪声）。"""
        if not self.running:
            return self.accumulator
        jitter = max(0.0, np.random.normal(1.0, self.noise))
        self.accumulator += dt * jitter
        self.t += dt
        return self.accumulator

    def elapsed(self):
        return float(self.accumulator)

    def stop(self):
        self.running = False
        return float(self.accumulator)

    def is_due(self, target, tolerance=0.2):
        """是否到达目标时长（±20% 容差，符合果蝇精度）。"""
        return bool(abs(self.accumulator - target) <= target * tolerance)

    def wait_until(self, target, max_steps=1000, dt=0.1):
        """等到目标时长（返回实际用了几步）。"""
        self.start()
        steps = 0
        while steps < max_steps:
            self.tick(dt)
            steps += 1
            if self.accumulator >= target:
                break
        return steps


# ==================== ③ 注意 ====================

class SelectiveAttention:
    """选择性注意（偏置竞争 biased competition）。

    生理依据（果蝇的注意研究）：
      多个刺激竞争有限的神经资源
      任务相关特征获得增益（gain）提升
      赢者通吃（winner-take-all），但弱势刺激仍保留微弱表征

    实现：显著性 × 任务增益 → softmax 竞争
    """

    def __init__(self, n_feature=8, temperature=0.4):
        self.n = n_feature
        self.temp = temperature
        self.task_bias = np.ones(n_feature, dtype=np.float32)

    def set_task(self, feature_id, gain=2.5):
        """设定当前任务：提高某特征的增益（注意焦点）。"""
        self.task_bias = np.ones(self.n, dtype=np.float32)
        if 0 <= feature_id < self.n:
            self.task_bias[feature_id] = gain
        return self.task_bias

    def compete(self, salience):
        """竞争：返回 (赢家索引, 注意权重分布, 抑制比)。"""
        s = np.asarray(salience, dtype=np.float32).ravel()[:self.n]
        if s.size < self.n:
            s = np.pad(s, (0, self.n - s.size))
        boosted = s * self.task_bias
        # softmax（温度控制锐度）
        z = boosted / max(1e-6, self.temp)
        z = z - z.max()
        w = np.exp(z)
        w /= (w.sum() + 1e-12)
        winner = int(np.argmax(w))
        # 抑制比：赢家权重 / 第二名权重
        order = np.argsort(w)[::-1]
        suppress = float(w[order[0]] / (w[order[1]] + 1e-9)) if len(order) > 1 else 99.0
        return winner, w, min(99.0, suppress)


# ==================== ④ 冲动控制 ====================

class ImpulseControl:
    """延迟满足（双曲折扣 hyperbolic discounting）。

    生理依据（果蝇的延迟奖励实验）：
      果蝇会等更大的延迟奖励，但等待意愿随延迟下降
      折扣曲线是双曲的：V = R / (1 + k·D)
      k 越大越冲动（越不耐烦）

    5-HT（血清素）降低冲动性（真实药理证据）。
    """

    def __init__(self, k=0.35, serotonin=0.5):
        self.k = k                       # 折扣率（大=冲动）
        self.serotonin = serotonin       # 0-1，高=耐心

    def discounted_value(self, reward, delay):
        """延迟奖励的主观价值。"""
        k_eff = self.k * (1.6 - 1.2 * self.serotonin)   # 5-HT 降低 k
        return float(reward / (1.0 + k_eff * max(0.0, delay)))

    def choose(self, immediate, delayed):
        """immediate: (reward, delay≈0)；delayed: (reward, delay)。"""
        vi = self.discounted_value(*immediate)
        vd = self.discounted_value(*delayed)
        return ("immediate" if vi >= vd else "delayed"), (vi, vd)

    def patience_score(self):
        """耐心指数（越高越能等）。"""
        return float(self.serotonin / (self.k + 0.05))


# ==================== ⑤ 社会学习 ====================

class SocialLearning:
    """观察学习（果蝇能从同伴的选择中学）。

    生理依据（Danchin 2018 等）：
      观察同类选了哪个选项 → 提高自己对那个选项的偏好
      如果观察到的结果好，偏好提升更多（结果依赖的社会学习）
      蘑菇体参与（与个体学习共享回路）

    实现：观察 → 更新选项价值（比个体试错弱，但有）
    """

    def __init__(self, n_option=4, lr=0.25):
        self.n = n_option
        self.lr = lr
        self.value = np.zeros(n_option, dtype=np.float32)
        self.observations = 0

    def observe(self, option, outcome=0.0, model_reliability=1.0):
        """看到同类选了 option，结果 outcome（+1 好 / -1 坏 / 0 未知）。"""
        if not (0 <= option < self.n):
            return 0.0
        # 结果未知时也有效应（单纯从众），但更弱
        delta = (outcome if outcome != 0 else 0.35) * self.lr * model_reliability
        self.value[option] = float(np.clip(self.value[option] + delta, -1.0, 1.0))
        self.observations += 1
        return float(self.value[option])

    def prefer(self, noise=0.05, rng=None):
        """基于观察到的偏好做选择。"""
        r = rng or np.random
        return int(np.argmax(self.value + r.normal(0, noise, self.n)))

    def stats(self):
        return {"observations": self.observations,
                "max_bias": float(self.value.max()) if self.value.size else 0.0}


# ==================== ⑥ 群体决策 ====================

class SwarmDecision:
    """法定人数决策（quorum sensing）。

    生理依据（果蝇群体行为）：
      个体受邻近同类影响（局部交互）
      当跟随者比例超过阈值（quorum）→ 集体转向
      这是无中心的分布式决策（swarm intelligence）

    实现：每个个体有私有偏好 + 从众倾向，超过 quorum 时集体锁定。
    """

    def __init__(self, n_agents=30, quorum=0.6, conformity=0.35):
        self.n = n_agents
        self.quorum = quorum
        self.conformity = conformity
        self.private = np.random.rand(n_agents)          # 私有偏好 0-1
        self.choice = (self.private > 0.5).astype(np.int8)
        self.committed = False

    def step(self):
        """一轮局部交互。返回 (选择A比例, 是否已锁定集体决定)。"""
        if self.committed:
            frac = float(self.choice.mean())
            return frac, True
        # 每个个体看邻居（简化为全局平均），按从众倾向调整
        frac_a = float(self.choice.mean())
        for i in range(self.n):
            pull = (frac_a - 0.5) * 2 * self.conformity
            p = np.clip(self.private[i] + pull, 0, 1)
            self.choice[i] = int(p > 0.5)
        frac_a = float(self.choice.mean())
        if frac_a >= self.quorum or (1 - frac_a) >= self.quorum:
            self.committed = True
        return frac_a, self.committed

    def consensus(self):
        return float(max(self.choice.mean(), 1 - self.choice.mean()))


# ==================== ⑦ 飞行 ====================

class FlightController:
    """果蝇飞行控制（翅膀 200Hz）。

    生理依据：
      翅膀运动神经元驱动间接飞行肌（IFI），振翅频率 ~200Hz
      转向靠**左右翅膀振幅差**（不是频率差）
      升力靠频率/攻角；爬升靠增大振幅
      转弯时内侧翅膀振幅减小 → 产生滚转力矩

    输出：左右翅振幅/频率 + 机体力/力矩
    """

    NOMINAL_HZ = 200.0

    def __init__(self):
        self.left_amp = 0.5      # 左翅振幅（0-1）
        self.right_amp = 0.5
        self.freq = self.NOMINAL_HZ

    def command(self, thrust=0.5, roll=0.0, dt=0.005):
        """thrust: 0-1 总推力；roll: -1 左滚 ~ +1 右滚。"""
        thrust = float(np.clip(thrust, 0, 1))
        roll = float(np.clip(roll, -1, 1))
        # 振幅差产生滚转；总振幅决定升力
        base = 0.3 + 0.7 * thrust
        diff = 0.35 * roll
        self.left_amp = float(np.clip(base - diff, 0.05, 1.0))
        self.right_amp = float(np.clip(base + diff, 0.05, 1.0))
        # 频率随推力小幅上升（真实果蝇 190-220Hz）
        self.freq = self.NOMINAL_HZ * (0.95 + 0.1 * thrust)
        # 输出：升力（∝ 平均振幅²×频率²）、滚转力矩（∝ 振幅差）、周期
        lift = float(((self.left_amp + self.right_amp) / 2) ** 2 * (self.freq / self.NOMINAL_HZ) ** 2)
        torque = float((self.right_amp - self.left_amp))
        period = 1.0 / self.freq
        return {"lift": lift, "roll_torque": torque,
                "left_amp": self.left_amp, "right_amp": self.right_amp,
                "freq_hz": self.freq, "wing_period_ms": period * 1000}

    def hover(self):
        """悬停（左右对称、推力=重力）。"""
        return self.command(thrust=0.5, roll=0.0)


# ==================== ⑧ 运动模式库 ====================

class MotorRepertoire:
    """固定动作模式库（果蝇的先天行为）。

    真实行为：
      grooming（梳理）：前腿搓头/搓翅，按固定顺序（清洁是生存必需）
      climbing（攀爬）：负趋地性，总是往上
      jumping（跳跃）：逃逸反应，瞬间弹射（腿伸展）
      freezing（僵住）：遇到强威胁时静止（减少被注意）
    """

    PATTERNS = ["groom_head", "groom_wing", "groom_abdomen",
                "climb", "jump", "freeze", "walk"]

    def __init__(self):
        self.dirt = {"head": 0.0, "wing": 0.0, "abdomen": 0.0}
        self.current = "walk"

    def soil(self, part, amount=0.2):
        """积累脏污（灰尘/食物残渣）。"""
        if part in self.dirt:
            self.dirt[part] = min(1.0, self.dirt[part] + amount)

    def select(self, threat=0.0, gravity_dir=0.0, need_escape=False):
        """选动作。返回 (动作名, 动作参数)。"""
        # 优先级：逃逸 > 威胁僵住 > 脏了梳理 > 向上爬 > 走
        if need_escape:
            self.current = "jump"
            return "jump", {"impulse": 1.0}
        if threat > 0.7:
            self.current = "freeze"
            return "freeze", {"duration": 1.0}
        # 最脏的部位先梳理
        dirtiest = max(self.dirt, key=lambda k: self.dirt[k])
        if self.dirt[dirtiest] > 0.3:
            self.current = "groom_" + dirtiest
            amt = self.dirt[dirtiest]
            self.dirt[dirtiest] = max(0.0, self.dirt[dirtiest] - 0.35)
            return self.current, {"duration": amt * 2.0}
        if gravity_dir < -0.3:            # 上坡/朝上 → 爬
            self.current = "climb"
            return "climb", {"rate": 0.6}
        self.current = "walk"
        return "walk", {"speed": 0.5}

    def state(self):
        return {"current": self.current, "dirt": dict(self.dirt)}


# ==================== ⑨ 免疫行为 ====================

class SicknessBehavior:
    """生病行为（sickness behavior）。

    生理依据（果蝇感染研究）：
      感染后活动量下降（省能量给免疫）
      食欲下降（部分）
      避开同类（减少传染 / 避免被攻击）
      体温偏好升高（行为性发热 fever）
      恢复后行为回归正常

    实现：免疫负荷 ↑ → 行为调制（活动↓、社交↓、偏暖）
    """

    def __init__(self):
        self.load = 0.0          # 免疫负荷 0-1
        self.recovering = False

    def infect(self, severity=0.6):
        self.load = float(np.clip(self.load + severity, 0, 1))
        self.recovering = False
        return self.load

    def step(self, immune_clearance=0.03):
        """随时间恢复。"""
        if self.load > 0:
            self.load = max(0.0, self.load - immune_clearance)
            self.recovering = self.load > 0
        return self.state()

    def state(self):
        """返回行为调制量（给本能/运动系统用）。"""
        l = self.load
        return {
            "activity_scale": float(np.clip(1.0 - 0.75 * l, 0.1, 1.0)),   # 活动下降
            "social_avoidance": float(np.clip(l * 1.2, 0, 1)),            # 避开同类
            "preferred_temp": float(24.0 + 6.0 * l),                      # 偏好更暖
            "appetite_scale": float(np.clip(1.0 - 0.4 * l, 0.5, 1.0)),
            "sick": bool(l > 0.25),
        }


# ==================== 自测 ====================

def selftest():
    print("  小沙认知能力 selftest（对标果蝇九项高级功能）")
    np.random.seed(20261115)
    ok = 0
    total = 0

    # ---- 1. 数量感：能区分 1-4 ----
    total += 1
    ns = NumerositySense()
    can_12 = ns.discriminate(1, 2)
    can_23 = ns.discriminate(2, 3)
    can_34 = ns.discriminate(3, 4)
    can_45 = ns.discriminate(4, 5)
    print("  [1] 数量感: 1vs2=%s 2vs3=%s 3vs4=%s 4vs5=%s（果蝇 1-4 能分，4vs5 吃力）"
          % (can_12, can_23, can_34, can_45))
    ok += int(can_12 and can_23 and can_34 and not can_45)

    # ---- 2. 数量感：累加器估计准确 ----
    total += 1
    ns2 = NumerositySense()
    for _ in range(5):
        ns2.accumulate(3)
    est = ns2.estimate() / 5
    print("  [2] 数量累加: 5 次看到 3 个 → 平均估计 %.2f（应≈3）" % est)
    ok += int(abs(est - 3.0) < 0.8)

    # ---- 3. 时间感知：计时精度 ----
    total += 1
    it = IntervalTimer()
    steps = it.wait_until(2.0, max_steps=100)
    actual = it.elapsed()
    print("  [3] 间隔计时: 目标 2.0s → 实际 %.2fs（%d 步）" % (actual, steps))
    ok += int(1.6 <= actual <= 2.6)

    # ---- 4. 时间感知：is_due 判定 ----
    total += 1
    it2 = IntervalTimer()
    it2.start()
    for _ in range(20):
        it2.tick(0.1)
    due = it2.is_due(2.0, tolerance=0.25)
    print("  [4] 计时判定: 累加 %.2fs，目标 2.0s → is_due=%s" % (it2.elapsed(), due))
    ok += int(due)

    # ---- 5. 注意：任务相关特征胜出 ----
    total += 1
    att = SelectiveAttention(n_feature=8)
    salience = np.array([0.3, 0.3, 0.9, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    winner_no, _, _ = att.compete(salience)
    att.set_task(0, gain=6.0)     # 任务要求注意特征0（虽然它显著性低）
    winner_task, weights, suppress = att.compete(salience)
    print("  [5] 选择性注意: 无任务 赢家=特征%d | 任务要求注意特征0 → 赢家=特征%d（抑制比 %.1f）"
          % (winner_no, winner_task, suppress))
    ok += int(winner_no == 2 and winner_task == 0)

    # ---- 6. 冲动控制：耐心时选延迟大奖励 ----
    total += 1
    ic = ImpulseControl(k=0.2, serotonin=0.9)      # 高 5-HT = 耐心
    choice_p, vals_p = ic.choose((2.0, 0.0), (6.0, 5.0))
    print("  [6] 冲动控制(耐心): 立即2 vs 5秒后6 → 选 %s（值 %.2f vs %.2f）"
          % (choice_p, vals_p[0], vals_p[1]))
    ok += int(choice_p == "delayed")

    # ---- 7. 冲动控制：冲动时选立即小奖励 ----
    total += 1
    ic2 = ImpulseControl(k=1.2, serotonin=0.1)     # 低 5-HT = 冲动
    choice_i, vals_i = ic2.choose((2.0, 0.0), (6.0, 5.0))
    print("  [7] 冲动控制(冲动): 同上 → 选 %s（值 %.2f vs %.2f）"
          % (choice_i, vals_i[0], vals_i[1]))
    ok += int(choice_i == "immediate")

    # ---- 8. 社会学习：观察同伴改变偏好 ----
    total += 1
    sl = SocialLearning(n_option=4)
    before = sl.prefer(noise=0.0)
    for _ in range(4):
        sl.observe(2, outcome=+1.0)      # 看到同伴选2且结果好
    after = sl.prefer(noise=0.0)
    print("  [8] 社会学习: 观察前偏好选项%d → 观察同伴选2后 → 选项%d"
          % (before, after))
    ok += int(after == 2)

    # ---- 9. 群体决策：法定人数触发锁定 ----
    total += 1
    np.random.seed(7)
    sw = SwarmDecision(n_agents=30, quorum=0.6, conformity=0.4)
    locked_at = None
    for step in range(30):
        frac, locked = sw.step()
        if locked and locked_at is None:
            locked_at = step
            break
    print("  [9] 群体决策: %s 步达成共识（比例 %.2f，共识度 %.2f）"
          % (locked_at if locked_at is not None else "未", frac, sw.consensus()))
    ok += int(locked_at is not None and sw.consensus() >= 0.6)

    # ---- 10. 飞行：滚转靠振幅差 ----
    total += 1
    fl = FlightController()
    r_right = fl.command(thrust=0.6, roll=+0.8)
    fl2 = FlightController()
    r_left = fl2.command(thrust=0.6, roll=-0.8)
    print("  [10] 飞行: 右滚 力矩%+.2f（右翅%.2f 左翅%.2f）| 左滚 力矩%+.2f"
          % (r_right["roll_torque"], r_right["right_amp"], r_right["left_amp"],
             r_left["roll_torque"]))
    ok += int(r_right["roll_torque"] > 0 and r_left["roll_torque"] < 0)

    # ---- 11. 飞行：振翅频率 200Hz 级 ----
    total += 1
    hov = fl.hover()
    print("  [11] 飞行频率: 悬停 %.0fHz（果蝇约 200Hz）| 周期 %.2fms"
          % (hov["freq_hz"], hov["wing_period_ms"]))
    ok += int(180 <= hov["freq_hz"] <= 230)

    # ---- 12. 运动模式库：脏了会梳理 ----
    total += 1
    mr = MotorRepertoire()
    mr.soil("head", 0.8)
    act, params = mr.select()
    print("  [12] 运动模式: 头部脏了 → 选 %s（%s）" % (act, params))
    ok += int(act == "groom_head")

    # ---- 13. 运动模式库：强威胁 → 僵住/逃逸 ----
    total += 1
    mr2 = MotorRepertoire()
    act2, _ = mr2.select(threat=0.9)
    act3, _ = mr2.select(need_escape=True)
    print("  [13] 威胁反应: 高威胁→%s | 需逃逸→%s" % (act2, act3))
    ok += int(act2 == "freeze" and act3 == "jump")

    # ---- 14. 免疫行为：感染后活动下降 + 避开同类 ----
    total += 1
    sb = SicknessBehavior()
    healthy = sb.state()
    sb.infect(0.8)
    sick = sb.state()
    print("  [14] 免疫行为: 健康活动%.2f → 生病活动%.2f | 社交回避 %.2f | 偏好温度 %.1f°C"
          % (healthy["activity_scale"], sick["activity_scale"],
             sick["social_avoidance"], sick["preferred_temp"]))
    ok += int(sick["activity_scale"] < healthy["activity_scale"]
              and sick["social_avoidance"] > 0.5
              and sick["preferred_temp"] > 24.0)

    # ---- 15. 免疫行为：恢复 ----
    total += 1
    for _ in range(60):
        st = sb.step()
    print("  [15] 免疫恢复: 60 步后负荷 %.2f（应接近 0）" % sb.load)
    ok += int(sb.load < 0.1)

    print("  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
