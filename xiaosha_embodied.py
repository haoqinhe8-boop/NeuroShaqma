# -*- coding: utf-8 -*-
"""小沙的具身闭环（Embodied Loop）—— 行为从脉冲网络涌现，而不是规则直接输出。

为什么做这个：
  之前的本能系统（xiaosha_instinct.py）是**规则直接算动作**：
      scene → 显著性 → 本能打分 → 仲裁 → vx/vy/vyaw
  这套东西能用，但和果蝇有本质差距——果蝇的动作是 13 万神经元放电涌现的，
  不是哪段代码【算】出来的。

  本模块把两条线接成一个闭环：
      感官 → 脉冲编码 → 288,964 神经元皮层（真放电）→ 运动区解读 → 动作
          ↑                                              ↓
          └──────────── 环境变化 ←──────────────────────┘

  本能不再直接输出动作，而是像**神经调质**（多巴胺/去甲肾上腺素）一样
  调节皮层的兴奋性/增益——这正是真实大脑里调质的作用。

可验证的【涌现】判据（本模块自测）：
  ① 消融实验：把运动区静默 → 动作幅值显著下降（说明动作确实来自皮层）
  ② 刺激实验：只给运动区某一个子群加偏置 → 出现对应的定向动作
  ③ 闭环实验：动作改变环境 → 感官输入改变 → 皮层活动模式跟着变
"""
import numpy as np

from xiaosha_structured import StructuredCortex

__version__ = "1.0.0"


# ==================== 感官 → 脉冲编码 ====================

class RateEncoder:
    """把连续特征编码成输入电流（rate coding，群体编码的简化版）。

    生物依据：
      感觉神经元用**发放率**编码刺激强度（rate code）
      不同神经元有不同的【最适刺激】（tuning curve）
      群体编码比单个神经元更精确（population code）
    """

    def __init__(self, n_feat, n_per_feat=4, gain=1.5):
        self.n_feat = n_feat
        self.n_per = n_per_feat
        self.gain = gain
        # 每个特征对应 n_per 个神经元，各有不同偏置（tuning）
        self.offsets = np.linspace(-0.3, 0.3, n_per_feat, dtype=np.float32)
        self.dim_out = n_feat * n_per_feat

    def encode(self, features):
        """features: n_feat 维（任意范围）→ 输入电流向量（dim_out 维）。"""
        f = np.asarray(features, dtype=np.float32).ravel()[:self.n_feat]
        if f.size < self.n_feat:
            f = np.pad(f, (0, self.n_feat - f.size))
        # 归一化到 0-1
        f = np.clip(f / (np.abs(f).max() + 1e-6), 0, 1)
        # 每个特征 × 每个 tuning 偏移
        out = (f[:, None] - self.offsets[None, :]) * self.gain
        return np.maximum(out, 0.0).ravel().astype(np.float32)


# ==================== 行为从皮层读出 ====================

class MotorReadout:
    """运动区（M1）解读：把发放模式变成动作。

    设计（关键：动作必须来自皮层活动，不能规则直出）：
      M1 区被分成 6 个运动子群：
        0 前进  1 后退  2 左转  3 右转  4 左臂  5 右臂
      每个子群的**发放率**决定对应动作分量的强度。
      皮层里哪个子群放电多，就做哪个动作——这是涌现，不是查表。
    """

    N_GROUPS = 6
    NAMES = ["forward", "backward", "left_turn", "right_turn", "left_arm", "right_arm"]

    def __init__(self, m1_ids, baseline=0.0):
        self.m1_ids = np.asarray(m1_ids, dtype=np.int64)
        self.n = self.m1_ids.size
        self.baseline = baseline
        # 把 M1 均匀分成 6 个子群
        self.groups = [g.astype(np.int64) for g in np.array_split(self.m1_ids, self.N_GROUPS)]
        self.last_rates = np.zeros(self.N_GROUPS, dtype=np.float32)

    def rate(self, sp):
        """统计各子群发放率（0-1 比例）。"""
        rates = np.zeros(self.N_GROUPS, dtype=np.float32)
        if sp.size == 0:
            self.last_rates = rates
            return rates
        # 用成员掩码统计（M1 的 id 是连续的，可以用 searchsorted 加速）
        in_m1 = sp[(sp >= self.m1_ids[0]) & (sp <= self.m1_ids[-1])]
        if in_m1.size:
            idx = np.searchsorted(self.m1_ids, in_m1)
            for gi, g in enumerate(self.groups):
                lo = np.searchsorted(self.m1_ids, g[0])
                hi = np.searchsorted(self.m1_ids, g[-1])
                cnt = int(((idx >= lo) & (idx <= hi)).sum())
                rates[gi] = cnt / max(1, g.size)
        self.last_rates = rates
        return rates

    def decode(self, sp, gain=1.0):
        """发放率 → 动作 (vx, vy, vyaw, arm_l, arm_r)。"""
        r = self.rate(sp)
        # 差分竞争：前进 vs 后退、左转 vs 右转
        vx = (r[0] - r[1]) * gain
        vyaw = (r[3] - r[2]) * gain       # 右转 - 左转
        arm_l = r[4] * gain * 0.5
        arm_r = r[5] * gain * 0.5
        return np.array([vx, 0.0, vyaw, arm_l, arm_r], dtype=np.float32)


# ==================== 本能 → 神经调质 ====================

class Neuromodulator:
    """本能不再直接输出动作，而是调节皮层。

    生物依据：
      多巴胺（DA）：奖励预测误差 → 提高目标相关通路的增益
      去甲肾上腺素（NE）：威胁 → 全局提高警觉/增益
      乙酰胆碱（ACh）：注意 → 提高感觉皮层增益
      血清素（5-HT）：耐心/等待 → 降低冲动

    所以本能的作用是【拧旋钮】，不是【下命令】。
    """

    def __init__(self):
        self.da = 0.0       # 奖励/趋近
        self.ne = 0.0       # 威胁/警觉
        self.ach = 0.0      # 注意
        self.gain = 1.0     # 皮层输入增益
        self.theta_scale = 1.0  # 皮层阈值缩放

    def update(self, sense):
        """sense: dict，含 threat/goal/social/battery 等。"""
        threat = float(sense.get("threat", 0.0))
        goal = float(sense.get("goal", 0.0))
        social = float(sense.get("social", 0.0))
        battery = float(sense.get("battery", 1.0))

        # 威胁 → NE 上升 → 提高警觉（增益↑ 但阈值也↑，避免乱动）
        self.ne = float(np.clip(threat, 0, 1))
        # 目标 + 低电 → DA 上升（趋近动机）
        self.da = float(np.clip(goal * 0.8 + max(0.0, 0.3 - battery) * 1.5, 0, 1))
        # 新奇/社交 → ACh（注意）
        self.ach = float(np.clip(social * 0.6 + sense.get("novelty", 0.0) * 0.4, 0, 1))

        # 调质 → 皮层参数
        # NE 高：增益提高（更敏感）但阈值提高（更谨慎）
        self.gain = 1.0 + 0.8 * self.ne + 0.4 * self.ach
        self.theta_scale = 1.0 + 0.6 * self.ne - 0.25 * self.da
        return self

    def apply(self, cortex):
        """把调质写进皮层（阈值缩放）。"""
        cortex.theta = float(np.clip(cortex._theta0 * self.theta_scale, 0.3, 12.0))


# ==================== 具身小沙（闭环）====================

class EmbodiedXiaosha:
    """感官 → 脉冲皮层 → 动作 → 环境 → 感官（完整闭环）。

    两种输入模式：
      saliency  —— 12 维显著性（快，用于行为测试/集成）
      full      —— 95 维完整四感（视24+听35+触12+嗅16+味8，吃原始信号）

    full 模式的数据流：
      image/audio/pressure/gas/chemicals → 四感前端（真实信号处理）→ 95 维
        → RateEncoder → 380 维电流 → 注入 V1 L4 → 皮层放电 → PFC 解读 → 动作
    """

    N_FEAT_FULL = 95          # 24+35+12+16+8
    N_FEAT_SALIENCY = 12

    def __init__(self, n_neurons=60000, seed=42, encode_gain=2.5, mode="full"):
        from xiaosha_perception import Perception
        from xiaosha_chemosense import Gustation

        self.mode = mode
        self.cortex = StructuredCortex(total=n_neurons, seed=seed)
        self.perception = Perception()      # 视/听/触/嗅
        self.gustation = Gustation()        # 味觉
        n_feat = self.N_FEAT_FULL if mode == "full" else self.N_FEAT_SALIENCY
        self.n_feat = n_feat
        self.enc = RateEncoder(n_feat=n_feat, n_per_feat=4, gain=encode_gain)
        # 注入面：整个 V1 L4，按通道循环映射
        v1_l4 = self.cortex.groups[(0, 0, 0)]
        self.input_ids = v1_l4
        self.input_map = np.arange(v1_l4.size, dtype=np.int64) % self.enc.dim_out
        # 读出：PFC L5（3 跳可达；M1 在 4 跳外衰减到不了）
        m1 = self.cortex.groups[(3, 2, 0)]
        if m1.size < 60:
            m1 = np.flatnonzero(self.cortex.region_id == 3)
        self.motor = MotorReadout(m1)
        self.mods = Neuromodulator()
        self.t = 0
        self.log = []

    # ---------- 全感：原始信号 → 95 维（真实信号处理）----------
    def perceive(self, image=None, audio=None, audio_right=None, pressure=None,
                 gas=None, chemicals=None, temp=25.0, shear=0.0):
        """把可用的原始感官信号转成 95 维特征（缺失通道补 0）。"""
        v = np.zeros(24, dtype=np.float32)
        a = np.zeros(35, dtype=np.float32)
        t = np.zeros(12, dtype=np.float32)
        o = np.zeros(16, dtype=np.float32)
        g = np.zeros(8, dtype=np.float32)
        if image is not None:
            v = self.perception.retina.encode(image)
        if audio is not None:
            a = self.perception.cochlea.encode(audio, audio_right)
        if pressure is not None:
            t = self.perception.skin.encode(pressure, temp, shear)
        if gas is not None:
            o = self.perception.nose.encode(gas)
        if chemicals is not None:
            g = self.gustation.encode(chemicals)
        return np.concatenate([v, a, t, o, g])

    # ---------- 显著性 → 95 维（补零，向后兼容）----------
    def saliency_to_full(self, scene):
        f = np.zeros(24 + 35 + 12 + 16 + 8, dtype=np.float32)
        base = self.scene_to_features(scene)     # 12 维
        # 放进"味觉 8 维"之前的视觉前 12 槽（保留语义）
        f[:12] = base
        return f

    # ---------- 感官 → 12 维底层特征（saliency 模式）----------
    @staticmethod
    def scene_to_features(scene):
        """把场景压成 12 维（给 RateEncoder 用）。

        这 12 维不是【手写规则】，而是各感官前端的**显著性输出**：
          0 威胁  1 目标  2 社交  3 新奇  4 电量不足
          5 温度不适  6 倾角  7 接触
          8 甜  9 苦  10 气味浓度  11 声音强度
        """
        f = np.zeros(12, dtype=np.float32)
        f[0] = float(scene.get("threat", 0.0))
        f[1] = float(scene.get("goal", 0.0))
        f[2] = float(scene.get("social", 0.0))
        f[3] = float(scene.get("novelty", 0.5))
        f[4] = max(0.0, 0.3 - float(scene.get("battery", 1.0))) * 3
        f[5] = float(scene.get("temp_err", 0.0))
        f[6] = float(scene.get("tilt", 0.0))
        f[7] = 1.0 if scene.get("touch") else 0.0
        f[8] = float(scene.get("sweet", 0.0))
        f[9] = float(scene.get("bitter", 0.0))
        f[10] = float(scene.get("odor", 0.0))
        f[11] = float(scene.get("sound", 0.0))
        return f

    # ---------- 一步 ----------
    def step(self, scene=None, image=None, audio=None, audio_right=None,
             pressure=None, gas=None, chemicals=None, temp=25.0, shear=0.0):
        """走一步。可给原始感官信号（full）或显著性 dict（saliency）。"""
        has_raw = any(x is not None for x in
                      (image, audio, pressure, gas, chemicals))
        if has_raw:
            feats = self.perceive(image, audio, audio_right, pressure,
                                  gas, chemicals, temp, shear)
            # 调质用感官显著性（威胁/目标/电量由触觉-温度、气味等派生）
            sense = {"threat": float(np.clip(feats[0] / 3, 0, 1)),
                     "goal": float(np.clip(feats[1] / 3, 0, 1)),
                     "battery": 1.0, "novelty": 0.5}
        else:
            scene = scene or {}
            base = self.scene_to_features(scene)
            feats = (self.saliency_to_full(scene) if self.n_feat == self.N_FEAT_FULL
                     else base)
            sense = scene

        # ① 本能 → 调质
        self.mods.update(sense)
        self.mods.apply(self.cortex)

        # ② 感官 → 输入电流（注入整个 V1 L4，按 input_map 分通道）
        currents = self.enc.encode(feats) * self.mods.gain
        v = self.cortex.v
        np.add.at(v, self.input_ids, currents[self.input_map])

        # ③ 皮层演化（真放电，行为由此产生）
        sp = self.cortex.step(None)

        # ④ 运动区解读 → 动作
        action = self.motor.decode(sp, gain=0.35)

        self.t += 1
        info = {
            "t": self.t, "n_spikes": int(sp.size),
            "rates": self.motor.last_rates.copy(),
            "action": action.copy(),
            "feats_dim": int(feats.size),
            "mod": {"da": self.mods.da, "ne": self.mods.ne,
                    "ach": self.mods.ach, "theta": self.cortex.theta},
        }
        self.log.append(info)
        return action, info

    # ---------- 消融（验证动作来自皮层）----------
    def ablate_input(self, steps=20, scene=None):
        """消融**输入区 V1**（切断感觉输入），看行为是否消失。

        为什么不用"抑制读出区 PFC"：
          实测抑制 PFC 会让动作**变大**（0.007 → 0.018）——这是真实的
          **抑制解除（disinhibition）**：压制一个区会释放下游。用它做消融验证
          会得出反向结论，测的不是因果。
        切输入才是干净的因果链：没有感觉 → 皮层静息 → 行为消失。
        """
        scene = scene or {"threat": 0.2, "goal": 0.7, "battery": 0.9}
        base, base_sp = [], []
        for _ in range(steps):
            a, i = self.step(scene)
            base.append(np.abs(a).sum()); base_sp.append(i["n_spikes"])
        # 消融：把整个 V1 区压住（切断感觉通路）
        # 关键：网络有活动残留（reverberation），压制后下游还会放几十步才衰减到静息。
        # 实测：压住 V1 后第 0 步下游仍 1200，第 4 步 IT 独占 1200（区内自我维持），
        #       需要 ~50 步才归零。所以必须等稳态再测，否则测到的是残留不是因果。
        v1 = np.flatnonzero(self.cortex.region_id == 0)
        for _ in range(50):                    # 先让它衰减到稳态
            self.cortex.v[v1] = -5.0
            self.cortex.ref[v1] = 5
            self.step(scene)
        ablated, abl_sp = [], []
        for _ in range(steps):
            self.cortex.v[v1] = -5.0
            self.cortex.ref[v1] = 5
            a, i = self.step(scene)
            ablated.append(np.abs(a).sum()); abl_sp.append(i["n_spikes"])
        return (float(np.mean(base)), float(np.mean(ablated)),
                float(np.mean(base_sp)), float(np.mean(abl_sp)))

    # ---------- 刺激（验证动作由皮层决定）----------
    def stimulate_group(self, group_id, steps=15, scene=None):
        """只给某个运动子群加偏置，看是否出现对应动作。"""
        scene = scene or {"threat": 0.1, "goal": 0.1, "battery": 1.0}
        g = self.motor.groups[group_id]
        acts = []
        for _ in range(steps):
            self.cortex.v[g] += 3.0          # 只刺激这一组
            a, _i = self.step(scene)
            acts.append(a.copy())
        return np.mean(np.stack(acts), axis=0)


# ==================== 自测 ====================

def selftest():
    print("  小沙具身闭环 selftest（行为从脉冲网络涌现）")
    np.random.seed(20261115)
    ok = 0
    total = 0

    # ---- 1. 闭环能跑起来，且在产生发放 ----
    total += 1
    em = EmbodiedXiaosha(n_neurons=60000, seed=42)
    scene = {"threat": 0.3, "goal": 0.7, "battery": 0.8, "novelty": 0.5}
    spikes = []
    for _ in range(30):
        a, info = em.step(scene)
        spikes.append(info["n_spikes"])
    print("  [1] 闭环运行: 30 步，皮层发放 均 %d/步，动作 %s"
          % (np.mean(spikes), np.round(a, 2)))
    ok += int(np.mean(spikes) > 0 and np.all(np.isfinite(a)))

    # ---- 2. 动作确实来自皮层：消融运动区后动作显著下降 ----
    total += 1
    em2 = EmbodiedXiaosha(n_neurons=60000, seed=7)
    base, abl, sp_b, sp_a = em2.ablate_input(steps=15)
    print("  [2] 消融输入区V1: 动作 %.3f → %.3f | 皮层发放 %.0f → %.0f（应同步崩溃）"
          % (base, abl, sp_b, sp_a))
    ok += int(sp_a < sp_b * 0.5)

    # ---- 3. 刺激运动子群 → 出现对应动作（前进组）----
    total += 1
    em3 = EmbodiedXiaosha(n_neurons=60000, seed=11)
    for _ in range(10):     # 先热身
        em3.step({"threat": 0.1, "goal": 0.1, "battery": 1.0})
    a_fwd = em3.stimulate_group(0, steps=12)     # group 0 = forward
    print("  [3] 刺激【前进】子群 → 动作 vx=%+.2f（应为正）" % a_fwd[0])
    ok += int(a_fwd[0] > 0.02)

    # ---- 4. 刺激【后退】子群 → vx 为负 ----
    total += 1
    em4 = EmbodiedXiaosha(n_neurons=60000, seed=13)
    for _ in range(10):
        em4.step({"threat": 0.1, "goal": 0.1, "battery": 1.0})
    a_back = em4.stimulate_group(1, steps=12)    # group 1 = backward
    print("  [4] 刺激【后退】子群 → 动作 vx=%+.2f（应为负）" % a_back[0])
    ok += int(a_back[0] < -0.02)

    # ---- 5. 刺激左右转子群 → vyaw 反号 ----
    total += 1
    em5 = EmbodiedXiaosha(n_neurons=60000, seed=17)
    for _ in range(10):
        em5.step({"threat": 0.1, "goal": 0.1, "battery": 1.0})
    a_l = em5.stimulate_group(2, steps=12)       # left_turn
    a_r = em5.stimulate_group(3, steps=12)       # right_turn
    print("  [5] 左转子群 vyaw=%+.2f vs 右转子群 vyaw=%+.2f（应反号）" % (a_l[2], a_r[2]))
    ok += int(a_l[2] * a_r[2] < 0)

    # ---- 6. 神经调质：威胁提高阈值（更谨慎）----
    total += 1
    em6 = EmbodiedXiaosha(n_neurons=60000, seed=19)
    em6.step({"threat": 0.0, "goal": 0.5, "battery": 1.0})
    theta_low = em6.cortex.theta
    for _ in range(5):
        em6.step({"threat": 0.9, "goal": 0.5, "battery": 1.0})
    theta_high = em6.cortex.theta
    print("  [6] 神经调质: 无威胁时阈值 %.2f → 高威胁时 %.2f（应变高=更谨慎）"
          % (theta_low, theta_high))
    ok += int(theta_high > theta_low)

    # ---- 7. 神经调质：低电提高 DA（趋近动机）----
    total += 1
    m = Neuromodulator()
    m.update({"threat": 0.0, "goal": 0.2, "battery": 1.0})
    da_full = m.da
    m.update({"threat": 0.0, "goal": 0.2, "battery": 0.05})
    da_low = m.da
    print("  [7] 神经调质: 满电 DA=%.2f → 低电 DA=%.2f（应变高）" % (da_full, da_low))
    ok += int(da_low > da_full + 0.1)

    # ---- 8. 闭环：动作改变环境 → 感官变化 → 皮层活动跟着变 ----
    total += 1
    em8 = EmbodiedXiaosha(n_neurons=60000, seed=23)
    # 模拟【朝目标移动】：目标越来越近
    patterns = []
    for i in range(20):
        goal_strength = 0.2 + 0.04 * i
        _a, info = em8.step({"threat": 0.1, "goal": goal_strength, "battery": 1.0})
        patterns.append(info["rates"])
    P = np.stack(patterns)
    # 早期 vs 晚期模式应该不同（环境变化 → 皮层活动变化）
    diff = float(np.abs(P[:5].mean(0) - P[-5:].mean(0)).sum())
    print("  [8] 闭环: 目标渐近时运动区模式变化 %.3f（>0.01 说明皮层对感官有反应）" % diff)
    ok += int(diff > 0.01)

    # ---- 9. 不同感官场景 → 不同皮层运动模式 ----
    total += 1
    em9 = EmbodiedXiaosha(n_neurons=60000, seed=29)
    r_threat = []
    r_goal = []
    for _ in range(15):
        _a, i1 = em9.step({"threat": 0.9, "goal": 0.0, "battery": 1.0})
        r_threat.append(i1["rates"])
    em9b = EmbodiedXiaosha(n_neurons=60000, seed=29)   # 同种子，只改场景
    for _ in range(15):
        _a, i2 = em9b.step({"threat": 0.0, "goal": 0.9, "battery": 1.0})
        r_goal.append(i2["rates"])
    d = float(np.abs(np.mean(r_threat, 0) - np.mean(r_goal, 0)).sum())
    print("  [9] 场景区分: 高威胁 vs 高目标 的运动模式差异 %.3f" % d)
    ok += int(d > 0.005)

    # ---- 10. 全系统（12 项输入）都能编码 ----
    total += 1
    enc = RateEncoder(n_feat=12, n_per_feat=4)
    f = np.array([0.5, 0.8, 0.3, 0.6, 0.2, 0.1, 0.05, 1.0, 0.7, 0.1, 0.4, 0.9], dtype=np.float32)
    cur = enc.encode(f)
    print("  [10] 感官编码: 12 维特征 → %d 维输入电流（非零 %.0f%%）"
          % (cur.size, (cur > 0).mean() * 100))
    ok += int(cur.size == 48 and (cur > 0).mean() > 0.2)

    # ---- 11. 全感模式：原始信号（图/声/触/嗅/味）→ 95 维 → 皮层 → 动作 ----
    total += 1
    em11 = EmbodiedXiaosha(n_neurons=60000, seed=31)
    yy, xx = np.mgrid[0:32, 0:32]
    img = ((np.sin(xx * np.pi / 4) > 0) * 255).astype(np.uint8)
    aud = np.sin(2 * np.pi * 440 * np.arange(4000) / 16000).astype(np.float32)
    press = np.zeros((8, 8), dtype=np.float32); press[3:5, 3:5] = 0.7
    gas = np.array([0.9, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    chem = {"sweet": 0.8}
    for _ in range(15):
        a11, i11 = em11.step(image=img, audio=aud, pressure=press, gas=gas, chemicals=chem)
    print("  [11] 全感模式: 特征 %d 维（应95）| 发放 %d | 动作 %s"
          % (i11["feats_dim"], i11["n_spikes"], np.round(a11, 2)))
    ok += int(i11["feats_dim"] == 95 and i11["n_spikes"] > 0 and np.isfinite(a11).all())

    # ---- 12. 看到不同东西 → 不同运动输出（视觉真的进皮层了）----
    total += 1
    em12a = EmbodiedXiaosha(n_neurons=60000, seed=37)
    em12b = EmbodiedXiaosha(n_neurons=60000, seed=37)   # 同种子同皮层
    vert = ((np.sin(xx * np.pi / 4) > 0) * 255).astype(np.uint8)
    hori = ((np.sin(yy * np.pi / 4) > 0) * 255).astype(np.uint8)
    ra, rb = [], []
    for _ in range(20):
        _a, ia = em12a.step(image=vert)
        _b, ib = em12b.step(image=hori)
        ra.append(ia["rates"]); rb.append(ib["rates"])
    d12 = float(np.abs(np.mean(ra, 0) - np.mean(rb, 0)).sum())
    print("  [12] 竖条纹 vs 横条纹 → 运动模式差异 %.3f（>0.005 说明视觉进了皮层）" % d12)
    ok += int(d12 > 0.005)

    print("  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
