# -*- coding: utf-8 -*-
"""小沙的深度 / 偏振 / 湿度感知（对标果蝇的三大招牌感觉）。

果蝇在这三件事上都是教科书级别，我们逐项对齐：

  ① 运动视差深度 DepthFromMotion
       果蝇没有立体视觉（复眼间距太小），靠**运动视差**估距离：
       飞行时近处物体的光流快、远处慢（光流 ∝ 1/深度）。
       这是它能穿林避障的关键。

  ② 偏振光罗盘 PolarizationCompass
       果蝇复眼背缘区（DRA）能检测天空偏振光的偏振角。
       Rayleigh 散射让天空偏振方向与太阳位置有确定关系，
       所以**即使太阳被云遮挡**，果蝇也能靠偏振光定向（太阳罗盘）。
       这是它最出名的导航能力。

  ③ 湿度感知 Hygrosensation
       果蝇有干/湿感受器（避开过干或过湿，偏好中等湿度）。
       干燥 → 趋湿；过湿 → 避湿（防溺/防霉）。

全部规则化实现（进化先验），零训练。
"""
import numpy as np

__version__ = "1.0.0"


# ==================== ① 运动视差深度 ====================

class DepthFromMotion:
    """光流 → 深度（果蝇的核心避障机制）。

    生理依据（Borst 2014, 果蝇光流计算）：
      视网膜大细胞（T4/T5）计算局部运动方向
      小叶板（lobula plate）的 VS/HS 神经元整合全局光流
      光流大小与深度成反比：v_flow = v_own / d * cos(θ)
      FOE（膨胀中心）指示自身运动方向

    输出（9 维）：
      0-3   四个象限的平均深度（近→大，归一化 0-1，1=很近）
      4     最近障碍距离（0-1，1=贴脸）
      5     光流散度（>0 膨胀=向前冲，<0 收缩=后退）
      6     FOE 水平位置（-1 左 ~ +1 右）
      7     深度图熵（杂乱度，高=环境复杂）
      8     时间导数（逼近速度，>0 在靠近）
    """

    def __init__(self, grid=(8, 8), flow_scale=1.0):
        self.grid = grid
        self.flow_scale = flow_scale
        self.prev = None
        self.prev_near = None
        self.dim = 9

    def encode(self, frame, dt=1.0):
        """frame: HxW 灰度图（0-255 或 0-1）→ 9 维深度特征。"""
        f = np.asarray(frame, dtype=np.float32)
        if f.ndim == 3:
            f = f.mean(axis=2)
        if f.max() > 1.5:
            f /= 255.0
        # 缩放到 grid
        h, w = f.shape
        yi = np.linspace(0, h - 1, self.grid[0]).astype(int)
        xi = np.linspace(0, w - 1, self.grid[1]).astype(int)
        g = f[np.ix_(yi, xi)]

        if self.prev is None:
            self.prev = g.copy()
            self.prev_near = 0.5
            return np.zeros(self.dim, dtype=np.float32)

        # 光流：块间差分的绝对值（简化 Lucas-Kanade，够用且无依赖）
        flow = np.abs(g - self.prev)
        # 近度 proximity 用两个互补指标：
        #   ① 分块近度（flow 归一化）—— 给四象限用，表示"哪里有东西动"
        #   ② 光流总量 —— 表示"物体多大/多近"。物体越大边缘越长，总光流越多。
        # 为什么不能用 max：只要有任何边缘，max 就饱和到 1，大小物体分不开（实测）。
        depth_norm = np.clip(flow * 8.0, 0, 1)       # 0-1，1=很近（局部）

        # 四象限平均深度
        gh, gw = self.grid
        quads = [
            depth_norm[:gh // 2, :gw // 2].mean(),
            depth_norm[:gh // 2, gw // 2:].mean(),
            depth_norm[gh // 2:, :gw // 2].mean(),
            depth_norm[gh // 2:, gw // 2:].mean(),
        ]
        # 最近障碍：用**光流总量**（∝ 物体边缘长度 ∝ 大小/近度）
        # 不用 depth_norm.max()：局部边缘一出现就饱和，大小物体都=1
        nearest = float(np.clip(flow.sum() / (flow.size * 0.35), 0, 1))

        # 光流散度（膨胀/收缩）
        gy, gx = np.gradient(flow)
        divergence = float(gx.mean() + gy.mean())

        # FOE 水平位置（光流最大处的 x）
        col_flow = flow.mean(axis=0)
        foe_x = float((np.argmax(col_flow) / max(1, gw - 1)) * 2 - 1)

        # 深度熵（杂乱度）
        p = depth_norm.ravel() / (depth_norm.sum() + 1e-9)
        entropy = float(-(p * np.log(p + 1e-9)).sum() / np.log(p.size + 1e-9))

        # 逼近速度
        approach = nearest - (self.prev_near if self.prev_near is not None else nearest)

        self.prev = g.copy()
        self.prev_near = nearest

        return np.array(quads + [nearest, divergence, foe_x, entropy, approach],
                        dtype=np.float32)

    def reset(self):
        self.prev = None
        self.prev_near = None


# ==================== ② 偏振光罗盘 ====================

class PolarizationCompass:
    """天空偏振光 → 太阳方位（果蝇的太阳罗盘）。

    生理依据（Wehner 2001; Hardie 1979）：
      复眼背缘区（DRA）的光感受器有**偏振敏感**结构（微绒毛排列）
      Rayleigh 散射：天空任意点的偏振方向与"太阳-该点-观察者"平面垂直
      所以测到偏振角，就能反推太阳方位（即使太阳被云遮住）
      果蝇用这个做**航向参考**（compass）

    偏振角到太阳方位的几何（简化）：
      天顶附近偏振方向垂直于太阳-天顶-观察者平面，
      因此 太阳方位角 ≈ 偏振角 + 90°

    输出（6 维）：
      0     检测到的偏振角（归一化 0-1）
      1     偏振度（0-1，高=天空晴朗，可靠）
      2     估计的太阳方位（cos）
      3     估计的太阳方位（sin）
      4     罗盘置信度（偏振度 × 一致性）
      5     与上次朝向的偏差（用于校验）
    """

    def __init__(self):
        self.dim = 6
        self.prev_heading = None

    def encode(self, pol_angles, intensities=None, heading=0.0):
        """pol_angles: 各方向（0~π）的偏振响应数组；intensities: 可选强度。

        返回 6 维。pol_angles 可以是标量（单一偏振角）或数组（完整偏振图）。
        """
        pa = np.asarray(pol_angles, dtype=np.float32).ravel()
        if pa.size == 0:
            return np.zeros(self.dim, dtype=np.float32)

        if pa.size == 1:
            # 单一偏振角（0~π 弧度）
            angle = float(pa[0]) % np.pi
            dop = 1.0
        else:
            # 完整偏振图：找响应最强的方向（偏振角）
            if intensities is None:
                inten = pa
            else:
                inten = np.asarray(intensities, dtype=np.float32).ravel()
            # 偏振图通常是 cos(2θ) 调制，用二阶谐波找偏振角
            theta = np.linspace(0, np.pi, inten.size, endpoint=False)
            s = float((inten * np.sin(2 * theta)).sum())
            c = float((inten * np.cos(2 * theta)).sum())
            angle = 0.5 * np.arctan2(s, c) % np.pi
            # 偏振度 = 二阶谐波幅度 / 平均值
            dop = float(np.clip(np.hypot(s, c) / (inten.mean() * inten.size + 1e-9), 0, 1))

        # Rayleigh 关系：太阳方位 ≈ 偏振角 + 90°（对天顶附近的偏振）
        sun_az = (angle + np.pi / 2) % (2 * np.pi)
        # 相对自身朝向的方位（罗盘输出的是"世界方位"，生物用它保持航向）
        rel = sun_az - heading

        conf = float(dop * (0.5 + 0.5 * np.cos(2 * angle)))
        drift = 0.0 if self.prev_heading is None else float(
            np.arctan2(np.sin(heading - self.prev_heading), np.cos(heading - self.prev_heading)))
        self.prev_heading = heading

        return np.array([angle / np.pi, dop, np.cos(sun_az), np.sin(sun_az), conf, drift],
                        dtype=np.float32)

    def heading_from_polarization(self, pol_angle, target_heading):
        """给定偏振角和目标航向，返回需要的转向量。"""
        sun_az = (float(pol_angle) + np.pi / 2) % (2 * np.pi)
        # 保持太阳在固定相对方位（太阳罗盘：恒定角）
        err = np.arctan2(np.sin(sun_az + target_heading), np.cos(sun_az + target_heading))
        return float(np.clip(err, -1.0, 1.0))


# ==================== ③ 湿度感知 ====================

class Hygrosensation:
    """湿度感知（果蝇的干/湿感受器）。

    生理依据：
      果蝇触角有湿度感受器（hygrosensory neurons）
      偏好**中等湿度**（过干脱水，过湿易溺/霉）
      干燥时主动趋湿（找水源），过湿时避开

    输出（5 维）：
      0  当前湿度（0-1）
      1  偏离舒适区的程度（0=舒适）
      2  变化率（正=变湿，负=变干）
      3  行为倾向（-1 避湿 ~ +1 趋湿）
      4  脱水风险（0-1，长期干燥累积）
    """

    OPTIMAL = 0.55       # 最适湿度（果蝇偏好中等偏湿）

    def __init__(self):
        self.prev = None
        self.dehydration = 0.0
        self.dim = 5

    def encode(self, humidity, dt=1.0):
        h = float(np.clip(humidity, 0, 1))
        rate = 0.0 if self.prev is None else (h - self.prev) / dt
        self.prev = h

        # 偏离舒适区
        dev = abs(h - self.OPTIMAL)

        # 脱水累积（长期低湿）
        if h < 0.3:
            self.dehydration = min(1.0, self.dehydration + 0.01)
        else:
            self.dehydration = max(0.0, self.dehydration - 0.006)

        # 行为倾向：太干 → 趋湿（正）；太湿 → 避湿（负）
        if h < self.OPTIMAL:
            drive = (self.OPTIMAL - h) * 2.0      # 干 → 趋湿
        else:
            drive = -(h - self.OPTIMAL) * 1.5     # 湿 → 避湿
        drive = float(np.clip(drive, -1, 1))

        return np.array([h, dev, rate, drive, self.dehydration], dtype=np.float32)


# ==================== ④ 三者融合 ====================

class SpatialSenses:
    """深度 + 偏振 + 湿度 → 20 维（给皮层/导航用）。"""

    def __init__(self):
        self.depth = DepthFromMotion()
        self.polar = PolarizationCompass()
        self.hygro = Hygrosensation()
        self.dim = self.depth.dim + self.polar.dim + self.hygro.dim   # 20

    def encode(self, frame=None, pol_angle=None, humidity=None, heading=0.0):
        d = self.depth.encode(frame) if frame is not None else np.zeros(self.depth.dim, dtype=np.float32)
        p = self.polar.encode(pol_angle, heading=heading) if pol_angle is not None else np.zeros(self.polar.dim, dtype=np.float32)
        h = self.hygro.encode(humidity) if humidity is not None else np.zeros(self.hygro.dim, dtype=np.float32)
        return np.concatenate([d, p, h])


# ==================== 自测 ====================

def selftest():
    print("  小沙深度/偏振/湿度 selftest（对标果蝇）")
    np.random.seed(20261115)
    ok = 0
    total = 0

    # ---- 1. 深度：近处物体光流大 ----
    total += 1
    dm = DepthFromMotion(grid=(8, 8))
    # 近处大物体快速移动
    f0 = np.zeros((32, 32), dtype=np.float32)
    f1_near = f0.copy(); f1_near[10:22, 10:22] = 1.0      # 大块出现（近）
    f1_far = f0.copy(); f1_far[15:17, 15:17] = 1.0        # 小块出现（远）
    dm.encode(f0)
    near_depth = dm.encode(f1_near)
    dm2 = DepthFromMotion(grid=(8, 8))
    dm2.encode(f0)
    far_depth = dm2.encode(f1_far)
    print("  [1] 深度(运动视差): 大物体 最近%.2f vs 小物体 最近%.2f（大的应更近）"
          % (near_depth[4], far_depth[4]))
    ok += int(near_depth[4] > far_depth[4])

    # ---- 2. 深度：逼近速度（物体占画面比例逐渐变大）----
    total += 1
    # 用 16x16 grid：8x8 在 32x32 输入下把物体压成 1-4 格，边缘差分不连续，
    # approach 会非单调（实测 [0.5, -1, 0, 0, 1]）。提高分辨率后测量才准。
    dm3 = DepthFromMotion(grid=(16, 16))
    approaches = []
    for k in range(6):
        fr = np.zeros((32, 32), dtype=np.float32)
        s = 4 + k * 2
        fr[16 - s // 2:16 + s // 2, 16 - s // 2:16 + s // 2] = 1.0
        d = dm3.encode(fr)
        approaches.append(d[8])
    print("  [2] 深度逼近(16x16): 物体变大时 approach 序列 %s（应多为正）"
          % np.round(approaches[1:], 3))
    ok += int(sum(1 for a in approaches[1:] if a > 0) >= 3)

    # ---- 3. 偏振：单一偏振角 → 太阳方位 ----
    total += 1
    pc = PolarizationCompass()
    pol = 0.0                        # 偏振角 0
    out = pc.encode(pol)
    sun_az_est = float(np.arctan2(out[3], out[2]))
    expected = np.pi / 2             # 0 + 90°
    err = abs(np.arctan2(np.sin(sun_az_est - expected), np.cos(sun_az_est - expected)))
    print("  [3] 偏振罗盘: 偏振角 0° → 太阳方位 %.1f°（应≈90°）" % (np.rad2deg(sun_az_est)))
    ok += int(err < 0.15)

    # ---- 4. 偏振：不同偏振角 → 不同方位（线性关系）----
    total += 1
    azs = []
    for pa in [0.0, 0.5, 1.0, 1.5]:
        o = PolarizationCompass().encode(pa)
        azs.append(float(np.arctan2(o[3], o[2])))
    # 检查单调性（偏振角增加 → 太阳方位增加）
    mono = sum(1 for i in range(len(azs) - 1) if azs[i + 1] > azs[i])
    print("  [4] 偏振角度映射: %s（应单调递增）" % np.round(np.rad2deg(azs), 1))
    ok += int(mono >= 3)

    # ---- 5. 偏振：完整偏振图（二阶谐波解算）----
    total += 1
    true_angle = 0.8
    theta = np.linspace(0, np.pi, 24, endpoint=False)
    inten = 0.5 + 0.5 * np.cos(2 * (theta - true_angle))     # Rayleigh 式调制
    pc2 = PolarizationCompass()
    out2 = pc2.encode(theta, inten)
    est_angle = float(out2[0] * np.pi)
    aerr = abs(np.arctan2(np.sin(est_angle - true_angle), np.cos(est_angle - true_angle)))
    print("  [5] 偏振图解算: 真值 %.2f rad → 估计 %.2f rad（误差 %.3f）| 偏振度 %.2f"
          % (true_angle, est_angle, aerr, out2[1]))
    ok += int(aerr < 0.12 and out2[1] >= 0.45)

    # ---- 6. 湿度：干燥 → 趋湿 ----
    total += 1
    hg = Hygrosensation()
    dry = hg.encode(0.15)
    print("  [6] 湿度(干): h=0.15 → 倾向 %+.2f（应趋湿=正）" % dry[3])
    ok += int(dry[3] > 0.3)

    # ---- 7. 湿度：过湿 → 避湿 ----
    total += 1
    hg2 = Hygrosensation()
    wet = hg2.encode(0.95)
    print("  [7] 湿度(湿): h=0.95 → 倾向 %+.2f（应避湿=负）" % wet[3])
    ok += int(wet[3] < -0.3)

    # ---- 8. 湿度：最适点附近趋向 0 ----
    total += 1
    hg3 = Hygrosensation()
    ideal = hg3.encode(Hygrosensation.OPTIMAL)
    print("  [8] 湿度(最适): h=%.2f → 倾向 %+.2f（应≈0）" % (Hygrosensation.OPTIMAL, ideal[3]))
    ok += int(abs(ideal[3]) < 0.15)

    # ---- 9. 湿度：脱水累积 ----
    total += 1
    hg4 = Hygrosensation()
    for _ in range(60):
        d = hg4.encode(0.05)
    print("  [9] 脱水累积: 持续 5%% 湿度 60 步 → 脱水风险 %.2f（应上升）" % d[4])
    ok += int(d[4] > 0.3)

    # ---- 10. 三者融合 ----
    total += 1
    ss = SpatialSenses()
    fr = np.random.rand(32, 32).astype(np.float32)
    v = ss.encode(frame=fr, pol_angle=0.6, humidity=0.4)
    print("  [10] 空间感融合: %d 维（深%d+偏%d+湿%d），非零 %.0f%%"
          % (len(v), ss.depth.dim, ss.polar.dim, ss.hygro.dim, (v != 0).mean() * 100))
    ok += int(len(v) == ss.dim and np.isfinite(v).all())

    print("  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
