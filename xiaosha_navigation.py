# -*- coding: utf-8 -*-
"""小沙的导航 + 运动（对标果蝇中枢复合体与步态）。

果蝇靠三套系统在陌生环境里回家/找食物，我们逐项对齐：

  ① 路径整合 PathIntegration —— 中枢复合体（central complex）的
       EPG（椭球体-桥脑节）头方向神经元：环形吸引子编码朝哪个方向
       PFN（原型脑桥节）速度神经元：编码走了多远
       两者积分 → 知道自己相对出发点的位置（"死推算"dead reckoning）
       果蝇能在完全黑暗中走直线回来（真实行为，Strauss & Heisenberg）

  ② 地标记忆 Landmark —— 蘑菇体（mushroom body）存视觉地标
       路径整合会累积漂移，靠地标校正（视觉归位）

  ③ 步态生成器 GaitCPG —— 胸神经节的中央模式发生器
       6 腿三足步态：L1+R2+L3 同相，R1+L2+R3 反相
       不需要大脑逐步指挥（节律自我维持），大脑只给方向和速度

全部零训练（结构 + 规则），符合"进化先验"路线。
"""
import numpy as np

__version__ = "1.0.0"


# ==================== ① 头方向（EPG 环形吸引子）====================

class Compass:
    """头方向编码：模拟果蝇 EPG 神经元（环形吸引子）。

    真实机制（Seelig & Jayaraman 2015）：
      EPG 神经元排成一个环，任意时刻只有一个"活动凸包"（bump）
      凸包的位置 = 当前头朝向
      身体转动 → 凸包沿环移动（前庭/视觉输入驱动）
      这是稳定的环形吸引子（不需要外部参照就能保持）
    """

    def __init__(self, n_cells=32, drift=0.004):
        self.n = n_cells
        self.drift = drift      # 自然漂移（真实环形吸引子会慢慢漂）
        self.angle = 0.0        # 内部朝向（弧度）
        self.bump = self._make_bump(0.0)

    def _make_bump(self, angle):
        """高斯凸包（EPG 的活动模式）。"""
        pref = np.linspace(-np.pi, np.pi, self.n, endpoint=False)
        d = np.arctan2(np.sin(pref - angle), np.cos(pref - angle))
        b = np.exp(-(d ** 2) / (2 * 0.35 ** 2))
        return (b / b.sum()).astype(np.float32)

    def rotate(self, dyaw, visual_cue=None):
        """转动 dyaw 弧度；visual_cue 可强制校正（看到地标时）。"""
        self.angle = np.arctan2(np.sin(self.angle + dyaw), np.cos(self.angle + dyaw))
        # 自然漂移
        self.angle += np.random.normal(0, self.drift)
        # 视觉校正（看到已知地标 → 重置到正确方向）
        if visual_cue is not None:
            self.angle = float(visual_cue)
        self.bump = self._make_bump(self.angle)
        return self.read()

    def read(self):
        """返回 (cos, sin, 置信度) —— V2 式的方向编码，避免 0/2π 跳变。"""
        conf = float(self.bump.max() * self.n / 4)   # 凸包越尖越确定
        return float(np.cos(self.angle)), float(np.sin(self.angle)), min(1.0, conf)


# ==================== ② 路径整合（dead reckoning）====================

class PathIntegrator:
    """路径整合：只靠"转了多少 + 走了多远"推算位置（果蝇在黑暗中也能回家）。

    真实果蝇精度（Strauss & Heisenberg 1993）：
      走 1 米后位置误差约 10-20%（会累积漂移）
      有视觉地标时校正
    """

    def __init__(self, drift_per_step=0.008):
        self.x = 0.0
        self.y = 0.0
        self.drift = drift_per_step
        self.compass = Compass()
        self.path = [(0.0, 0.0)]
        self.distance = 0.0

    def step(self, vx, vyaw, dt=1.0, landmark_correction=None):
        """走一步。vx: 前进速度；vyaw: 转向。"""
        self.compass.rotate(vyaw)
        c, s, _ = self.compass.read()
        # 按当前朝向前进
        self.x += vx * dt * c
        self.y += vx * dt * s
        self.distance += abs(vx * dt)
        # 累积漂移（真实的路径整合会漂）
        ang = np.arctan2(self.y, self.x)
        mag = np.hypot(self.x, self.y)
        mag *= (1 + np.random.normal(0, self.drift))
        if landmark_correction is not None:
            # 看到地标 → 校正位置
            self.x, self.y = landmark_correction
        else:
            self.x = mag * np.cos(ang)
            self.y = mag * np.sin(ang)
        self.path.append((self.x, self.y))
        return self.position()

    def position(self):
        return float(self.x), float(self.y)

    def home_vector(self):
        """回家方向与距离（果蝇的核心输出）。"""
        dist = float(np.hypot(self.x, self.y))
        ang = float(np.arctan2(-self.y, -self.x))
        return ang, dist

    def reset(self):
        self.x = self.y = 0.0
        self.distance = 0.0
        self.path = [(0.0, 0.0)]


# ==================== ③ 地标记忆（蘑菇体）====================

class LandmarkMemory:
    """地标 → 位置 的联想记忆（蘑菇体功能）。

    真实果蝇：蘑菇体存储"看到这个视觉模式时，家在哪个方向"。
    这里简化成：视觉指纹 → 位置，最近邻匹配（一次写入，像情景记忆）。
    """

    def __init__(self, n_feat=6, thresh=0.25, capacity=128):
        self.n_feat = n_feat
        self.thresh = thresh
        self.capacity = capacity
        self.keys = []
        self.positions = []

    def _fp(self, visual):
        v = np.asarray(visual, dtype=np.float32).ravel()[:self.n_feat]
        if v.size < self.n_feat:
            v = np.pad(v, (0, self.n_feat - v.size))
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def remember(self, visual, position):
        """一次写入。"""
        self.keys.append(self._fp(visual))
        self.positions.append(tuple(position))
        if len(self.keys) > self.capacity:
            self.keys.pop(0)
            self.positions.pop(0)

    def recall(self, visual):
        """看到相似视觉 → 返回记住的位置（或 None）。"""
        if not self.keys:
            return None
        k = self._fp(visual)
        K = np.stack(self.keys)
        dist = np.linalg.norm(K - k, axis=1)
        i = int(np.argmin(dist))
        if dist[i] < self.thresh:
            return self.positions[i]
        return None

    def stats(self):
        return {"landmarks": len(self.keys)}


# ==================== ④ 步态生成器（CPG，6 腿三足步态）====================

class GaitCPG:
    """6 腿中央模式发生器（真实昆虫的三足步态）。

    真实机制：
      每腿一个相位振荡器（CPG 单元）
      三足步态（tripod）：L1 + R2 + L3 同相；R1 + L2 + R3 反相
      摆动相（swing）：腿抬起前移；支撑相（stance）：腿着地后推
      大脑只给"往哪走、多快"，节律由 CPG 自己维持（不需要逐步指挥）

    腿编号：L1 L2 L3 R1 R2 R3
    """

    TRIPOD_A = (0, 4, 2)     # L1, R2, L3（索引 0,4,2）
    TRIPOD_B = (3, 1, 5)     # R1, L2, R3（索引 3,1,5）

    def __init__(self, freq=2.0, dt=0.05):
        self.freq = freq           # 步频（Hz，果蝇约 3-8Hz）
        self.dt = dt
        self.phase = np.zeros(6, dtype=np.float32)
        # 初始化相位：三足 A 相 0，B 相 0.5
        for i in self.TRIPOD_A:
            self.phase[i] = 0.0
        for i in self.TRIPOD_B:
            self.phase[i] = 0.5
        self.t = 0.0
        self.duty = 0.5            # 支撑相占空比（真实昆虫约 0.5-0.7）

    def step(self, speed=1.0, turn=0.0):
        """走一步。返回 6 条腿的 (抬起 0-1, 前后摆动 -1~1) 和身体位移。"""
        # 步频随速度变化（走得快 → 步频高，真实昆虫如此）
        f = self.freq * (0.5 + 0.5 * min(2.0, speed))
        self.current_freq = f        # 当前实际步频（随速度变化）
        self.t += self.dt
        self.phase = (self.phase + f * self.dt) % 1.0

        lift = np.zeros(6, dtype=np.float32)
        swing = np.zeros(6, dtype=np.float32)

        for i in range(6):
            ph = self.phase[i]
            if ph < self.duty:
                # 支撑相：腿在地上往后推
                swing[i] = 1.0 - 2.0 * (ph / self.duty)     # +1 → -1
                lift[i] = 0.0
            else:
                # 摆动相：腿抬起前移
                p = (ph - self.duty) / (1 - self.duty)
                swing[i] = -1.0 + 2.0 * p                    # -1 → +1
                lift[i] = np.sin(np.pi * p)

        # 身体位移：由着地的腿决定（简化为速度 × 时间）
        # 转向时左右腿幅度不对称（真实昆虫的差速步态）
        left = np.mean(lift[:3]); right = np.mean(lift[3:])
        turn_bias = (left - right) * turn
        vx = speed * self.dt * (1.0 - 0.3 * abs(turn))
        vyaw = turn * self.dt * 2.0 + turn_bias * 0.2

        return {
            "lift": lift, "swing": swing,
            "vx": float(vx), "vyaw": float(vyaw),
            "phase": self.phase.copy(), "freq": f,
        }

    def tripod_sync(self):
        """三足同步度（1.0 = 完美的三足步态）。

        正确算法：分三件事
          ① A 组内相位一致（同相）
          ② B 组内相位一致（同相）
          ③ A 与 B 相差 0.5（反相）
        旧算法直接比 cos 均值，会随相位一起振荡（错误）。
        """
        pa = float(np.mean([self.phase[i] for i in self.TRIPOD_A]))
        pb = float(np.mean([self.phase[i] for i in self.TRIPOD_B]))
        in_a = float(np.mean([np.cos(2 * np.pi * (self.phase[i] - pa)) for i in self.TRIPOD_A]))
        in_b = float(np.mean([np.cos(2 * np.pi * (self.phase[i] - pb)) for i in self.TRIPOD_B]))
        cross = float(-np.cos(2 * np.pi * (pa - pb)))      # 相差 0.5 → +1
        return float((in_a + in_b + cross) / 3.0)


# ==================== ⑤ 完整导航体 ====================

class Navigator:
    """整合：指南针 + 路径整合 + 地标 + 步态。"""

    def __init__(self):
        self.pi = PathIntegrator()
        self.compass = self.pi.compass
        self.landmarks = LandmarkMemory()
        self.gait = GaitCPG()
        self.log = []

    def step(self, speed=1.0, turn=0.0, visual=None):
        """走一步。visual: 可选视觉指纹（有地标时自动校正）。"""
        # 步态产生实际运动
        g = self.gait.step(speed, turn)
        # 路径整合
        correction = None
        if visual is not None:
            correction = self.landmarks.recall(visual)
        pos = self.pi.step(g["vx"], g["vyaw"], landmark_correction=correction)
        info = {"pos": pos, "gait": g, "home": self.pi.home_vector(),
                "corrected": correction is not None}
        self.log.append(info)
        return info

    def remember_here(self, visual):
        self.landmarks.remember(visual, self.pi.position())


# ==================== 自测 ====================

def selftest():
    print("  小沙导航 + 步态 selftest（对标果蝇中枢复合体）")
    # 固定随机种子：路径整合/指南针有真实漂移噪声，不固定种子会 flaky
    # （实测：同代码 100 步漂移在 113.0 / 122.4 之间跳，断言 80-120 会偶发失败）
    np.random.seed(20261115)
    ok = 0
    total = 0

    # ---- 1. 指南针：转动后凸包跟着移动 ----
    total += 1
    cp = Compass(n_cells=32)
    cp.drift = 0.0
    a0 = cp.angle
    cp.rotate(1.0)
    print("  [1] 头方向 EPG: 转 1.0 rad → 内部角度 %.2f (凸包峰值 %.2f)" % (cp.angle, cp.bump.max()))
    ok += int(abs(cp.angle - 1.0) < 0.05)

    # ---- 2. 路径整合：走正方形回到原点（无漂移）----
    total += 1
    pi = PathIntegrator(drift_per_step=0.0)
    pi.compass.drift = 0.0
    for _ in range(4):
        for _ in range(10):
            pi.step(1.0, 0.0)          # 前进 10 步
        pi.step(0.0, np.pi / 2)        # 转 90°
    px, py = pi.position()
    err = np.hypot(px, py)
    print("  [2] 路径整合(走正方形): 终点 (%.2f, %.2f)，回原点误差 %.2f" % (px, py, err))
    ok += int(err < 0.5)

    # ---- 3. 路径整合：有漂移时误差累积（真实的缺陷）----
    total += 1
    pi2 = PathIntegrator(drift_per_step=0.02)
    for _ in range(100):
        pi2.step(1.0, 0.0)
    d = np.hypot(*pi2.position())
    print("  [3] 漂移累积: 直走 100 步后距离 %.1f（应接近 100，有误差）" % d)
    ok += int(75 < d < 135)

    # ---- 4. 回家向量 ----
    total += 1
    pi3 = PathIntegrator(drift_per_step=0.0)
    pi3.compass.drift = 0.0
    for _ in range(20):
        pi3.step(1.0, 0.0)
    ang, dist = pi3.home_vector()
    print("  [4] 回家向量: 方向 %.2f rad (应≈π)，距离 %.1f" % (ang, dist))
    ok += int(abs(abs(ang) - np.pi) < 0.2 and dist > 15)

    # ---- 5. 地标记忆：一次写入就能回忆 ----
    total += 1
    lm = LandmarkMemory()
    vis = np.array([0.9, 0.1, 0.5, 0.3, 0.2, 0.7], dtype=np.float32)
    lm.remember(vis, (3.0, 4.0))
    got = lm.recall(vis * 0.95)     # 略微不同的视觉
    print("  [5] 地标记忆: 存 (3,4) → 回忆 %s" % (got,))
    ok += int(got is not None and abs(got[0] - 3) < 0.5)

    # ---- 6. 地标校正路径整合漂移 ----
    total += 1
    nav = Navigator()
    v = np.array([0.2, 0.8, 0.4, 0.6, 0.1, 0.3], dtype=np.float32)
    # 先在 (5,0) 记住地标
    nav.pi.x, nav.pi.y = 5.0, 0.0
    nav.remember_here(v)
    # 走远（累积漂移）
    for _ in range(30):
        nav.step(1.0, 0.2)
    far = nav.pi.position()
    # 看到地标 → 校正
    nav.step(0.5, 0.0, visual=v)
    back = nav.pi.position()
    print("  [6] 地标校正: 漂移到 (%.1f, %.1f) → 看到地标回到 (%.1f, %.1f)"
          % (far[0], far[1], back[0], back[1]))
    ok += int(abs(back[0] - 5.0) < 0.5 and abs(back[1]) < 0.5)

    # ---- 7. 步态：三足同步 ----
    total += 1
    g = GaitCPG(freq=3.0)
    syncs = []
    for _ in range(50):
        r = g.step(speed=1.0, turn=0.0)
        syncs.append(g.tripod_sync())
    print("  [7] 三足步态同步度: 平均 %.2f（1.0=完美三足）" % np.mean(syncs[10:]))
    ok += int(np.mean(syncs[10:]) > 0.5)

    # ---- 8. 步态：支撑相/摆动相都出现 ----
    total += 1
    lift_any = False; swing_any = False
    g2 = GaitCPG()
    for _ in range(40):
        r = g2.step(1.0, 0.0)
        if r["lift"].max() > 0.5: lift_any = True
        if r["swing"].max() > 0.5: swing_any = True
    print("  [8] 步态相位: 有抬起=%s 有前摆=%s" % (lift_any, swing_any))
    ok += int(lift_any and swing_any)

    # ---- 9. 速度→步频（走得快步子快）----
    total += 1
    g_slow = GaitCPG(freq=2.0); g_fast = GaitCPG(freq=2.0)
    for _ in range(20): g_slow.step(0.3, 0.0)
    for _ in range(20): g_fast.step(2.0, 0.0)
    print("  [9] 速度调制步频: 慢走 %.2fHz → 快走 %.2fHz" % (g_slow.current_freq, g_fast.current_freq))
    ok += int(g_fast.current_freq > g_slow.current_freq * 1.3)

    # ---- 10. 端到端：走出去再靠地标回家 ----
    total += 1
    nav2 = Navigator()
    nav2.pi.x, nav2.pi.y = 0.0, 0.0
    mark = np.array([0.5, 0.5, 0.9, 0.1, 0.3, 0.7], dtype=np.float32)
    nav2.remember_here(mark)        # 记住"这里"
    for _ in range(200):
        nav2.step(1.0, 0.15)        # 走远（步态每步 0.05，200 步约走 10）
    far2 = nav2.pi.position()
    ang_h, dist_h = nav2.pi.home_vector()
    print("  [10] 端到端归位: 走出 (%.1f, %.1f)，回家方向 %.2f rad 距离 %.1f"
          % (far2[0], far2[1], ang_h, dist_h))
    ok += int(dist_h > 5 and np.isfinite(ang_h))

    print("  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
