# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""
小沙 / 沙琪玛 —— 唤醒-回应-定位-移动 闭环（纯 numpy 自研仿真，无需 PyBullet）

流程：
  ① 听见 "小沙" / "沙琪玛"          → 唤醒词识别
  ② 回应（打印/语音）                → "哎，我在"
  ③ 定位声源                         → 用双麦克风时间差判断方位
  ④ 转过/开过去，到你面前            → 运动学 + 简单的"朝目标点移动"控制

作者：Shaqma（沙琪玛）
"""

import numpy as np
import math
import time


# ============================================================
# ① 虚拟世界：一个 2D 房间 + 小沙（轮式底盘）+ 主人声源
# ============================================================
class World:
    def __init__(self, w=10.0, h=10.0):
        self.w, self.h = w, h
        self.t = 0.0

    def step(self, dt=0.05):
        self.t += dt

    def inside(self, p):
        return 0 <= p[0] <= self.w and 0 <= p[1] <= self.h


class XiaoSha:
    """小沙：轮式底盘机器人
    状态：位置 (x,y)、朝向 theta。双麦克风在左右两侧。
    """
    def __init__(self, x=2.0, y=2.0, theta=0.0):
        self.pos = np.array([x, y], dtype=float)
        self.theta = theta              # 朝向，弧度，0 = 指向 +x
        self.mic_sep = 0.20             # 双麦克风间距 20cm
        self.radius = 0.25              # 底盘半径
        self.speed = 1.2                # 最大线速度 m/s
        self.turn_speed = 2.0           # 最大角速度 rad/s
        self.history = [self.pos.copy()]

    def head_dir(self):
        return np.array([math.cos(self.theta), math.sin(self.theta)])

    def move(self, v, omega, dt):
        """v: 前进速度, omega: 角速度"""
        self.theta += omega * dt
        self.pos = self.pos + np.array([math.cos(self.theta), math.sin(self.theta)]) * v * dt
        self.history.append(self.pos.copy())


# ============================================================
# ② 听觉：唤醒词识别 + 声源定位
# ============================================================
class Hearing:
    WAKE_WORDS = ["小沙", "沙琪玛", "小沙", "shaqma", "xiaosha"]

    def __init__(self):
        self.awake = False

    def listen_text(self, text):
        """模拟'听到了一句话'，判断是不是在叫我"""
        text = text.strip().lower()
        hit = any(w in text for w in self.WAKE_WORDS)
        return hit

    def locate_source(self, robot, source_pos):
        """
        双麦克风测向：声波先到近的一只耳朵。
        用两耳到声源的**距离差**模拟到达时间差(TDOA)。
        返回：相对于机器人正前方的角度（弧度，左正右负）
        """
        left_mic = robot.pos + rot(robot.theta + math.pi/2) * (robot.mic_sep/2)
        right_mic = robot.pos + rot(robot.theta - math.pi/2) * (robot.mic_sep/2)
        dL = np.linalg.norm(source_pos - left_mic)
        dR = np.linalg.norm(source_pos - right_mic)
        # 距离差 → 时间差（声速 343 m/s）
        tdoa = (dR - dL) / 343.0
        # 时间差映射到方位角（近似，小角度线性）
        # tdoa>0 说明右耳更远 → 声源偏左 → 角度为正
        angle = math.asin(np.clip(tdoa * 343.0 / robot.mic_sep, -1.0, 1.0))
        return angle, tdoa


def rot(a):
    """单位方向向量，角度 a"""
    return np.array([math.cos(a), math.sin(a)])


# ============================================================
# ③ 语音回应（无库则打印）
# ============================================================
class Voice:
    def __init__(self):
        self.engine = None
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 210)
        except Exception:
            self.engine = None

    def say(self, text):
        print(f"  🗣  小沙：{text}")
        if self.engine is not None:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception:
                pass


# ============================================================
# ④ 行为控制：先转头对准，再开过去
# ============================================================
class Controller:
    def __init__(self, arrive_dist=1.0):
        self.arrive_dist = arrive_dist   # 到了这个距离就算"到你面前"

    def act(self, robot, source_pos, dt):
        to_src = source_pos - robot.pos
        dist = np.linalg.norm(to_src)
        if dist < 1e-6:
            return 0.0, 0.0, dist

        # 目标朝向
        target_theta = math.atan2(to_src[1], to_src[0])
        err = wrap_angle(target_theta - robot.theta)

        if dist > self.arrive_dist:
            # 还没到：边转边靠近（角度差大就多转少走）
            omega = np.clip(2.5 * err, -robot.turn_speed, robot.turn_speed)
            v = robot.speed * max(0.0, math.cos(err))   # 背对时先转向，不硬冲
        else:
            # 到了：转身面对主人，停下
            omega = np.clip(2.0 * err, -robot.turn_speed, robot.turn_speed)
            v = 0.0
        return v, omega, dist


def wrap_angle(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a


# ============================================================
# ⑤ 主循环：把上面几块拼成闭环
# ============================================================
def run_demo(owner_pos=np.array([8.0, 7.5]), verbose=True):
    world = World()
    robot = XiaoSha(x=2.0, y=2.0, theta=0.0)
    ear = Hearing()
    voice = Voice()
    ctrl = Controller(arrive_dist=1.0)

    if verbose:
        print("=" * 56)
        print("  小沙 · 唤醒-回应-定位-移动 闭环演示")
        print("=" * 56)
        print(f"  主人位置：({owner_pos[0]:.1f}, {owner_pos[1]:.1f})")
        print(f"  小沙初始：({robot.pos[0]:.1f}, {robot.pos[1]:.1f})，朝向 {math.degrees(robot.theta):.0f}°")
        print("-" * 56)

    log = []

    # ---- ① 唤醒 ----
    heard = "小沙，过来一下"
    if verbose: print(f"\n👂 [听到] “{heard}”")
    if not ear.listen_text(heard):
        if verbose: print("   （不是叫我，忽略）")
        return
    ear.awake = True
    if verbose: print("   ✅ 唤醒词命中！")

    # ---- ② 回应 ----
    voice.say("哎，我在，这就过去。")
    log.append(("wake", world.t, robot.pos.copy(), None))

    # ---- ③ 定位声源 ----
    angle, tdoa = ear.locate_source(robot, owner_pos)
    if verbose:
        side = "左" if angle > 0 else "右"
        print(f"\n📍 [定位] 声源在正前方偏{side} {abs(math.degrees(angle)):.1f}°")
        print(f"         双麦时间差 = {tdoa*1e6:.1f} 微秒")

    # ---- ④ 移动 ----
    if verbose: print(f"\n🚗 [移动] 小沙开始朝主人开过去……")
    dt = 0.05
    step = 0
    ticks = 0
    while ticks < 4000:                     # 上限 200 秒，防死循环
        v, omega, dist = ctrl.act(robot, owner_pos, dt)
        robot.move(v, omega, dt)
        world.step(dt)
        ticks += 1

        if verbose and ticks % 40 == 0:
            print(f"    t={world.t:5.1f}s  位置=({robot.pos[0]:4.1f},{robot.pos[1]:4.1f})  "
                  f"朝向={math.degrees(robot.theta)%360:5.1f}°  距主人={dist:4.2f}m")

        if dist <= ctrl.arrive_dist:
            # 稳定一下，确认停住并面向主人
            if abs(wrap_angle(math.atan2(owner_pos[1]-robot.pos[1],
                                        owner_pos[0]-robot.pos[0]) - robot.theta)) < 0.15:
                break

    final_dist = np.linalg.norm(owner_pos - robot.pos)
    if verbose:
        print(f"\n✅ [到达] 用时 {world.t:.1f}s，最终距主人 {final_dist:.2f}m")
        voice.say("我到啦，有什么吩咐？")
        print("-" * 56)

    log.append(("arrive", world.t, robot.pos.copy(), final_dist))
    return {
        "world": world, "robot": robot,
        "time": world.t, "final_pos": robot.pos.copy(),
        "final_dist": final_dist, "log": log,
        "path": np.array(robot.history),
    }


# ============================================================
if __name__ == "__main__":
    import sys
    np.random.seed(0)
    result = run_demo(owner_pos=np.array([8.0, 7.5]))

    # 边界情况测试：主人就在旁边 / 主人在正后方 / 主人在正左
    print("\n" + "=" * 56)
    print("  边界测试")
    print("=" * 56)
    cases = [
        ("主人就在1米内", np.array([2.8, 2.0])),
        ("主人在正后方", np.array([-4.0, 2.0])),
        ("主人在正左方", np.array([2.0, 9.0])),
    ]
    for name, p in cases:
        print(f"\n▶ {name}  目标({p[0]:.1f},{p[1]:.1f})")
        r = run_demo(owner_pos=p, verbose=False)
        ok = r["final_dist"] < 1.05
        print(f"   结果：{'✅ 到达' if ok else '⚠️ 未到达'}  用时{r['time']:.1f}s  终距{r['final_dist']:.2f}m")
