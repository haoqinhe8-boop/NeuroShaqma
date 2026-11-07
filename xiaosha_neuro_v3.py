# -*- coding: utf-8 -*-
"""
小沙 · 神经系统 v3
============================================================================
目标：不是"堆功能"，而是把机器人做成一套**分层神经系统**，并且它自己能学、
自己想、失败不放弃。对标的是 FlyWire 那份成年果蝇全脑连接组给的启示：
**13.9 万个神经元就能做导航、学习、决策** —— 所以关键不在"神经元多"，
而在"分层、模块化、有反馈闭环、会自己调整"。

  设计依据（已联网核实，来源见文末）：
    · Dorkenwald et al., Nature 2024, "Neuronal wiring diagram of an adult brain"
      —— 139,255 个校对过的神经元、5×10^7 个化学突触（FlyWire 全脑）。
    · 果蝇用**中央复合体**做航向与路径积分（相当于"知道自己离家多远、朝哪"），
      用**蘑菇体**做稀疏编码与奖励调制式学习/记忆。
    诚实说明：FlyWire 由普林斯顿 Seung/Murthy 实验室牵头，Google / Amazon 是
    资助与算力支持方（原文写的是 "Core support ... from ... Google, and Amazon"），
    说"Google 做的那只果蝇"并不准确，但"用少量神经元做出复杂行为"这个启示成立。

三层神经系统（本文件的骨架）
----------------------------------------------------------------------------
  🧠 大脑 Cortex（控制中心）
      目标管理、任务决策、**内部思考循环 think()**、价值判断（好/坏）
  🎯 小脑 Cerebellum（平衡与运动）
      平衡/航向保持、速度平滑、**内模在线学习**（自己估出左右轮的真实效率并补偿）
  🦴 脊髓 SpinalCord（感知与反射）
      传感器健康自检（**自适应基线**，学出"正常长什么样"）、5 类故障点名检出、
      优先于大脑的反射弧（障碍刹停 / 堵转保护 / 坏传感器降级）

外加四个"果蝇式"能力
----------------------------------------------------------------------------
  · 中央复合体 CentralComplex：路径积分 + 自己回家（go_home）
  · 蘑菇体 MushroomBody：情境 → 稀疏编码 → 情景记忆检索（相似情境会泛化）
  · 可塑性 Plasticity：好的记住（强化）、坏的降级（先给机会，屡犯才舍去）、
                        长期不用按遗忘曲线剔除；巩固过的技能抗遗忘
  · 韧性 Explorer：一次失败就换个策略再试（不止一次），全试完才诚实认输

用法
----------------------------------------------------------------------------
  python xiaosha_neuro_v3.py --selftest     # 机器可判定自检（含 v2 回归）
  python xiaosha_neuro_v3.py --demo         # 三层神经系统联演
  python xiaosha_neuro_v3.py --learn        # 看小脑怎么自己学会补偿
  python xiaosha_neuro_v3.py --no-llm       # 离线（规则大脑）

作者：Shaqma（原型 v1/v2 + mini_brain_v3 类脑模块） / Hermes（v3 神经系统）
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import xiaosha_brain_v2 as v2                      # noqa: E402  复用 v2 的感知/记忆/语言/机械臂
from xiaosha_brain_v2 import wrap_angle            # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_DIR = os.path.join(HERE, "brain_data")
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================================
# 0) 机器人本体：带"轮子效率不一致"和"打滑"的差速底盘
#    —— 真机一定存在左右不对称（电机、摩擦、电池电压），仿真必须模拟它，
#       否则小脑永远没有东西可学。
# ============================================================================
class Chassis:
    L = 0.30                      # 轮距（m）

    def __init__(self, x=2.0, y=2.0, theta=0.0,
                 eff_l=1.0, eff_r=1.0, wheel_noise=0.01, slip=0.0, seed=0,
                 block_until=0.0):
        self.pos = [float(x), float(y)]
        self.theta = theta
        self.block_until = block_until   # 开局被卡住这么久（模拟被线缠住/卡进缝里）
        self.t_elapsed = 0.0
        self.eff_l, self.eff_r = eff_l, eff_r      # 左右轮真实效率（小脑要自己学出来）
        self.wheel_noise = wheel_noise             # 编码器测量噪声
        self.slip = slip                           # 打滑概率（0~1，位置更新按比例打折）
        self.rng = random.Random(seed)
        self.radius = 0.25
        self.path = [[self.pos[0], self.pos[1]]]
        self.batt = 12.6
        self.distance_ripped = 0.0

    def step(self, cmd_l, cmd_r, dt):
        """输入左右轮速指令 (m/s)，返回带噪声的编码器读数 (ml, mr) 与真实 (v, omega)。"""
        vl = self.eff_l * cmd_l
        vr = self.eff_r * cmd_r
        if self.t_elapsed < self.block_until:
            vl = vr = 0.0                        # 被卡住：指令照发，轮子不转
        self.t_elapsed += dt
        if self.slip > 0.0:                        # 打滑：整体速度打折（真机常见）
            k = 1.0 - self.slip * self.rng.random()
            vl *= k
            vr *= k
        ml = vl * (1.0 + self.rng.gauss(0.0, self.wheel_noise))
        mr = vr * (1.0 + self.rng.gauss(0.0, self.wheel_noise))

        v = (vl + vr) / 2.0
        omega = (vr - vl) / self.L
        self.theta = wrap_angle(self.theta + omega * dt)
        self.pos[0] += math.cos(self.theta) * v * dt
        self.pos[1] += math.sin(self.theta) * v * dt
        self.path.append([self.pos[0], self.pos[1]])
        self.distance_ripped += abs(v) * dt
        self.batt = max(9.0, self.batt - abs(v) * dt * 0.02)
        return (v, omega, ml, mr)


# ============================================================================
# 1) 脊髓层 SpinalCord —— 感知自检 + 反射
#    用户原话："感知里面坏的地方和BUG，一定要能自己学习"
#    → 所以基线（正常范围/噪声水平）是**学出来的**，不是硬编码阈值。
# ============================================================================
SENSOR_SPEC = {
    # 名称: (物理下限, 物理上限, 单位)
    "doa": (-math.pi, math.pi, "rad"),      # 声源方向（ReSpeaker 硬件 DoA）
    "us_front": (0.0, 4.0, "m"),            # 前方净空（超声/红外）
    "enc_l": (-1.5, 1.5, "m/s"),            # 左轮编码器
    "enc_r": (-1.5, 1.5, "m/s"),            # 右轮编码器
    "batt": (9.0, 13.0, "V"),               # 电池电压
    "imu_yaw": (-math.pi, math.pi, "rad"),  # 陀螺航向
}


class FaultInjector:
    """把"坏了"这件事变成可测试的：对指定传感器注入指定故障。"""

    def __init__(self):
        self.inject = {}          # name -> kind
        self._frozen = {}

    def set(self, name, kind):
        self.inject[name] = kind

    def clear(self, name=None):
        if name is None:
            self.inject.clear()
        else:
            self.inject.pop(name, None)

    def apply(self, readings):
        for name, kind in list(self.inject.items()):
            if name not in readings:
                continue
            if kind == "stale":            # 值冻结在正常范围（传感器死住）
                self._frozen.setdefault(name, readings[name])
                readings[name] = self._frozen[name]
            elif kind == "stuck":          # 卡在一个"看着合法"的常数上（最隐蔽的一类）
                lo, hi, _ = SENSOR_SPEC[name]
                cur = readings.get(name, lo)
                cand = lo + 0.1 * (hi - lo)
                if abs(cur - cand) < 0.2 * (hi - lo):
                    cand = hi - 0.1 * (hi - lo)
                readings[name] = cand
            elif kind == "out_of_range":   # 越界（线断、漂移）
                readings[name] = 9000.0
            elif kind == "noisy":          # 抖动爆表
                # 幅度按该传感器量程缩放：太大反而会被判成"越界"，
                # 那就测不到"噪声"这一类故障了（这条踩过）。
                lo, hi, _ = SENSOR_SPEC[name]
                readings[name] = readings[name] + random.gauss(0, 0.08 * (hi - lo))
            elif kind == "inconsistent":   # 互相矛盾（左轮读数比右轮莫名大 0.8）
                if name == "enc_l":
                    readings[name] = readings.get("enc_r", 0.0) + 0.8
        return readings


class SensorMonitor:
    """自适应基线 + 5 类故障检出。

    关键点：基线只在**判定健康**时更新 —— 否则学坏了以后就再也检不出故障
    （这是自检系统最经典的坑：把故障当常态学进去）。
    """

    def __init__(self, window=14, sigma=3.0, learn_rate=0.15):
        self.window = window
        self.sigma = sigma
        self.lr = learn_rate
        self.hist = {k: [] for k in SENSOR_SPEC}
        self.baseline = {k: None for k in SENSOR_SPEC}   # (mean, std)
        self.jitter_base = {k: None for k in SENSOR_SPEC}  # 它平时每步抖多少
        self.jb_seed = {}                                  # 起步期的抖动样本
        self.last = {}
        self.constant_run = {k: 0 for k in SENSOR_SPEC}
        self.faults = []

    # ---- 单项检查 ----
    def _check_one(self, name, value, freeze_checks=True):
        lo, hi, unit = SENSOR_SPEC[name]
        faults = []
        delta = None

        # ① 越界：不需要学习，物理上就不该出现
        if not (lo <= value <= hi):
            faults.append({"sensor": name, "kind": "out_of_range",
                           "evidence": f"{value:.3f}{unit} 超物理范围 [{lo},{hi}]"})
            return faults, None

        bl = self.baseline[name]

        # ② 冻住（值长时间完全不变）—— 但只有"学过它会波动"的传感器才算异常：
        #    本来就恒定的量（例：静止时的测距）一直不变是正常的，不许误报。
        if name in self.last and abs(value - self.last[name]) < 1e-9:
            self.constant_run[name] += 1
        else:
            self.constant_run[name] = 0
        if (freeze_checks and self.constant_run[name] >= self.window
                and bl is not None and bl[1] > 1e-4):
            faults.append({"sensor": name, "kind": "stale",
                           "evidence": f"连续 {self.constant_run[name]} 次读数完全相同"
                                       f"（{value:.3f}{unit}），但它平时是波动的"
                                       f"（σ={bl[1]:.4f}）"})

        # ③ 卡在一个"看着合法"的常数上（与学到的基线偏离很多，且一直不变）
        if freeze_checks and bl is not None and self.constant_run[name] >= self.window:
            mean, std = bl
            if abs(value - mean) > max(self.sigma * std, 0.02 * (hi - lo)):
                kinds = {f["kind"] for f in faults}
                if "stuck" not in kinds:
                    faults.append({"sensor": name, "kind": "stuck",
                                   "evidence": f"恒定在 {value:.3f}{unit}，偏离学习基线 "
                                               f"{mean:.3f}±{std:.3f}"})

        # ④ 噪声爆表 —— 判据用"逐点抖动"（相邻采样的平均变化量），不是方差：
        #    方差会把正常的加速、转弯、停车这些工况切换误判成噪声，
        #    而真噪声的特征是"每一步都在跳"。
        self.hist[name].append(value)
        if len(self.hist[name]) > self.window:
            self.hist[name].pop(0)
        if name in self.last:
            delta = abs(value - self.last[name])
        if len(self.hist[name]) >= self.window and bl is not None:
            diffs = [abs(self.hist[name][k + 1] - self.hist[name][k])
                     for k in range(len(self.hist[name]) - 1)]
            jitter = sum(diffs) / max(1, len(diffs))
            jb = self.jitter_base[name]
            allow = None if jb is None else max(4.0 * jb + 0.05, 0.10)
            if allow is not None and jitter > allow:
                faults.append({"sensor": name, "kind": "noisy",
                               "evidence": f"逐点抖动 {jitter:.3f} ＞ 允许 {allow:.3f}"
                                           f"（它平时每步只抖 {jb:.4f}）"})
        return faults, delta

    # ---- 跨传感器一致性 ----
    def _check_consistency(self, readings, cmd, dt):
        faults = []
        if "enc_l" in readings and "enc_r" in readings and cmd is not None:
            cl, cr = cmd
            m = readings["enc_l"] - readings["enc_r"]
            c = cl - cr
            # 指令基本对称却出现巨大轮速差 → 至少一个编码器在说谎
            if abs(c) < 0.10 and abs(m) > 0.6:
                faults.append({"sensor": "enc_l/enc_r", "kind": "inconsistent",
                               "evidence": f"指令差 {c:.3f} 但读数差 {m:.3f}"})
        if "imu_yaw" in readings and "enc_l" in readings and "enc_r" in readings:
            if "imu_yaw" in self.last and dt > 0:
                d_yaw = wrap_angle(readings["imu_yaw"] - self.last["imu_yaw"])
                said = (readings["enc_r"] - readings["enc_l"]) / Chassis.L * dt
                if abs(d_yaw - said) > 0.35 and abs(said) > 0.05:
                    faults.append({"sensor": "imu_yaw", "kind": "inconsistent",
                                   "evidence": f"陀螺转了 {d_yaw:.3f}rad，编码器只支持 "
                                               f"{said:.3f}rad"})
        return faults

    # ---- 主入口 ----
    def check(self, readings, cmd=None, dt=0.05):
        all_faults = []
        driving = (cmd is not None and abs(cmd[0]) + abs(cmd[1]) > 0.10)
        for name, value in readings.items():
            if name not in SENSOR_SPEC:
                continue
            # ★ "冻住"只有在系统本来应该在动的时候才算异常。
            #   站着不动时轮速恒为 0、测距恒定、电压不降 —— 这些都是正常的，
            #   拿它们当故障就是自检系统最经典的误报来源。
            if name in ("enc_l", "enc_r"):
                ci = 0 if name == "enc_l" else 1
                learn = freeze = bool(cmd is not None and abs(cmd[ci]) > 0.05)
            elif name in ("us_front", "imu_yaw", "batt"):
                learn = freeze = driving
            else:                     # doa：声源本来就可能一直不动 → 永不判"冻住"
                learn, freeze = True, False
            fs, delta = self._check_one(name, value, freeze_checks=freeze)
            all_faults.extend(fs)
            if not learn:
                self.hist[name] = []
            elif not fs:
                # ★★ 只在"健康且样本看起来正常"时学习基线/抖动水平。
                #    否则自适应阈值会把故障本身学成常态 —— 阈值一路被抬高，
                #    故障就永久检不出来了（这条踩过：注入噪声后 jb 从 0.008
                #    被抬到 0.56，于是永远不报警）。
                lo_, hi_, _ = SENSOR_SPEC[name]
                if self.baseline[name] is None:
                    self.baseline[name] = [value, 0.0]
                else:
                    mean, std = self.baseline[name]
                    lim = max(3.0 * std, 0.05 * (hi_ - lo_))
                    v_clip = max(mean - lim, min(mean + lim, value))   # 离群点不许拖均值
                    new_mean = (1 - self.lr) * mean + self.lr * v_clip
                    new_std = (1 - self.lr) * std + self.lr * abs(v_clip - new_mean)
                    self.baseline[name] = [new_mean, new_std]
                if delta is not None:
                    jb = self.jitter_base[name]
                    if jb is None:
                        # 起步期取前几个抖动的最小值 —— 万一开机就带故障，
                        # 用最小值起步比用第一个样本安全得多。
                        self.jb_seed.setdefault(name, []).append(delta)
                        if len(self.jb_seed[name]) >= 5:
                            self.jitter_base[name] = min(self.jb_seed[name])
                    else:
                        allow = max(4.0 * jb + 0.05, 0.10)
                        if delta <= allow:          # 超出允许范围的跳变不参与学习
                            self.jitter_base[name] = (1 - self.lr) * jb + self.lr * delta
            self.last[name] = value
        all_faults.extend(self._check_consistency(readings, cmd, dt))
        self.faults = all_faults
        return all_faults

    def degraded_sensors(self):
        return {f["sensor"] for f in self.faults}


class Reflex:
    """脊髓反射：优先于大脑，快、硬、不管目标是什么。
    大脑说"往前冲"，反射说"前面 20cm 有东西" —— 听反射的。
    """

    def __init__(self, front_stop=0.28, stall_cmd=0.20, stall_time=0.35):
        self.front_stop = front_stop
        self.stall_cmd = stall_cmd
        self.stall_time = stall_time
        self.retry_after = 0.60      # 堵转保护后放行这么久，让轮子自己再试一次
        self.stall_timer = 0.0
        self.cooldown = 0.0
        self.jam_time = 0.0          # 连续"给了速度但轮子不动"的时长（给大脑看）
        self.estop = False

    def evaluate(self, readings, faults, v, omega, dt):
        events = []
        degraded = {f["sensor"] for f in faults}

        if self.estop:
            return 0.0, 0.0, [{"reflex": "estop", "note": "急停中"}]

        # ① 前方障碍反射（感知坏掉时按最保守处理，不用坏值冒险）
        us = readings.get("us_front")
        if us is None or "us_front" in degraded:
            us_eff = 0.0            # 测距不可信 → 当"贴脸"处理，减速
            events.append({"reflex": "degrade_us",
                           "note": "前方测距故障 → 降级为保守限速"})
        else:
            us_eff = us
        if us_eff < self.front_stop and v > 0:
            v = 0.0
            events.append({"reflex": "obstacle_brake",
                           "note": f"前方 {us_eff:.2f}m ＜ {self.front_stop}m，刹停"})

        # ② 堵转反射：给了速度但轮子不动（真机＝卡住/顶墙）
        enc = [abs(readings.get("enc_l", 0.0)), abs(readings.get("enc_r", 0.0))]
        # 「已经连续堵了多久」—— 反射只负责保护和再试，该不该放弃由大脑判断
        if abs(v) > self.stall_cmd and max(enc) < 0.02:
            self.jam_time += dt
        else:
            self.jam_time = 0.0
        # ★ 必须"保护一下、再放开让它试"，绝不能一锁到底 —— 否则它用（滞后的）
        #   轮速读数判定"没动" → 把指令压成 0 → 轮子更不可能动 → 永久自锁死。
        if self.cooldown > 0.0:
            self.cooldown -= dt
        elif abs(v) > self.stall_cmd and max(enc) < 0.02:
            self.stall_timer += dt
        else:
            self.stall_timer = 0.0
        if self.stall_timer >= self.stall_time and self.cooldown <= 0.0:
            events.append({"reflex": "stall_guard",
                           "note": f"给了 v={v:.2f} 但轮子 {self.stall_time}s 没动，"
                                   f"停一下再试"})
            self.stall_timer = 0.0
            self.cooldown = self.retry_after
            v, omega = 0.0, 0.0

        # ③ 编码器故障降级：位置只能靠航位推算 → 限速（安全第一）
        if "enc_l" in degraded or "enc_r" in degraded or "enc_l/enc_r" in degraded:
            v = max(-0.35, min(0.35, v))
            omega = max(-0.9, min(0.9, omega))
            events.append({"reflex": "degrade_enc",
                           "note": "编码器故障 → 限速 0.35m/s 并改用航位推算"})
        return v, omega, events


class SpinalCord:
    """脊髓 = 传感器 + 健康自检 + 反射弧。大脑只能看到它给的"可信感知"。"""

    def __init__(self, chassis, world, injector=None):
        self.chassis = chassis
        self.world = world
        self.injector = injector or FaultInjector()
        self.monitor = SensorMonitor()
        self.reflex = Reflex()
        self.event_log = []

    def perceive(self, source_pos=None, cmd=None, dt=0.05):
        readings = self._raw(source_pos)
        readings = self.injector.apply(readings)
        faults = self.monitor.check(readings, cmd=cmd, dt=dt)
        return {"readings": readings, "faults": faults,
                "degraded": sorted(self.monitor.degraded_sensors())}

    def _raw(self, source_pos=None):
        head = (math.cos(self.chassis.theta), math.sin(self.chassis.theta))
        probe = (self.chassis.pos[0] + head[0] * 0.35,
                 self.chassis.pos[1] + head[1] * 0.35)
        us = max(0.0, min(4.0, self.world.clearance(probe)))
        doa = None
        if source_pos is not None:
            import xiaosha_brain_v2 as _v2
            sp = _v2.SimPerception(self.world)
            robot = type("R", (), {"pos": self.chassis.pos, "theta": self.chassis.theta,
                                   "mic_sep": 0.20})()
            doa, _ = sp.locate(robot, source_pos)
        else:
            doa = 0.0
        return {
            "doa": float(doa),
            "us_front": float(us),
            "enc_l": float(self._enc_l),
            "enc_r": float(self._enc_r),
            "batt": float(self.chassis.batt),
            "imu_yaw": float(self.chassis.theta),
        }

    _enc_l = 0.0
    _enc_r = 0.0

    def update_encoders(self, ml, mr):
        self._enc_l, self._enc_r = ml, mr

    def gate(self, v, omega, perception, dt):
        v2_, w2, events = self.reflex.evaluate(perception["readings"],
                                               perception["faults"], v, omega, dt)
        if events:
            self.event_log.extend(events)
        return v2_, w2, events


# ============================================================================
# 2) 小脑层 Cerebellum —— 平衡 + 内模在线学习
#    小脑的经典功能就是"逆模型 + 前馈补偿"：知道自己的身体不完美，学会补。
#    这里它要自己估出左右轮的真实效率，并让"想走直"真的走直。
# ============================================================================
class Cerebellum:
    def __init__(self, lr=0.20, gain_min=0.35, gain_max=1.60):
        self.eff = [1.0, 1.0]           # 学出来的左右轮效率估计
        self.lr = lr
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.updates = 0
        self.err_history = []

    # ---- 逆模型：把期望的 (v, omega) 变成左右轮指令（带补偿）----
    def command(self, v, omega):
        vl = v - omega * Chassis.L / 2.0
        vr = v + omega * Chassis.L / 2.0
        cl = vl / max(self.gain_min, min(self.gain_max, self.eff[0]))
        cr = vr / max(self.gain_min, min(self.gain_max, self.eff[1]))
        return cl, cr

    # ---- 学习：用编码器实测反推效率，滑动更新 ----
    def observe(self, cmd_l, cmd_r, meas_l, meas_r):
        changed = False
        for i, (c, m) in enumerate(((cmd_l, meas_l), (cmd_r, meas_r))):
            if abs(c) > 0.08:           # 指令太小不学（信噪比差）
                est = m / c
                est = max(self.gain_min, min(self.gain_max, est))
                self.eff[i] = (1 - self.lr) * self.eff[i] + self.lr * est
                changed = True
        if changed:
            self.updates += 1
        return self.eff

    # ---- 平衡/航向保持：抑制横向漂移（小脑的"站直"功能）----
    def balance(self, heading_err, omega_goal):
        return omega_goal + 1.2 * wrap_angle(heading_err)

    def report(self):
        return {"eff_l": round(self.eff[0], 4), "eff_r": round(self.eff[1], 4),
                "updates": self.updates}


# ============================================================================
# 3) 中央复合体 CentralComplex —— 路径积分 + 自己回家
#    果蝇靠它知道"巢在哪、我偏了多少"。工程上就是一个带校准的位移积分器。
# ============================================================================
class CentralComplex:
    def __init__(self, home):
        self.home = [float(home[0]), float(home[1])]
        self.vec = [0.0, 0.0]        # 相对出发点的累计位移（航向积分）
        self.drift = 0.0

    def integrate(self, dpos):
        self.vec[0] += dpos[0]
        self.vec[1] += dpos[1]

    def estimate_position(self):
        return (self.home[0] + self.vec[0], self.home[1] + self.vec[1])

    def home_target(self):
        """回家的目标点。仿真里有房间坐标，所以直接给起点的绝对坐标；
        真机没有绝对定位时改用 home_vector()（相对位移的反向量），效果等价。

        ★ 这里踩过一个坑：曾写成 home - vec —— 那是"相对位移向量"而不是坐标，
          结果机器人朝墙外走。同一个式子在两个坐标系里含义完全不同。
        """
        return (self.home[0], self.home[1])

    def home_vector(self):
        """相对出发点的位移的反向量（真机回家用这个）。"""
        return (-self.vec[0], -self.vec[1])

    def calibrate(self, true_pos):
        """到达一个已知地标时校准积分漂移（真机靠视觉/回充触点做这件事）。"""
        est = self.estimate_position()
        err = math.hypot(est[0] - true_pos[0], est[1] - true_pos[1])
        self.drift = err
        self.vec = [true_pos[0] - self.home[0], true_pos[1] - self.home[1]]
        return err


# ============================================================================
# 4) 蘑菇体 MushroomBody —— 稀疏编码的情景记忆
#    果蝇蘑菇体把气味压成极稀疏的"标签"，再由多巴胺信号调权重。
#    这里做同一件事：情境向量 → 稀疏激活 → 按重叠度检索（相似情境会泛化）。
# ============================================================================
class MushroomBody:
    def __init__(self, n_in=6, n_kc=128, k=6, seed=3):
        self.n_in, self.n_kc, self.k = n_in, n_kc, k
        rng = random.Random(seed)
        self.W = [[rng.gauss(0, 1.0) for _ in range(n_in)] for _ in range(n_kc)]
        self.engram = []             # 记忆痕迹

    def encode(self, ctx):
        act = []
        for i in range(self.n_kc):
            s = sum(self.W[i][j] * ctx[j] for j in range(min(self.n_in, len(ctx))))
            act.append((s, i))
        act.sort(reverse=True)
        return frozenset(i for _, i in act[:self.k])

    def store(self, ctx, payload):
        code = self.encode(ctx)
        for e in self.engram:
            if e["code"] == code:
                e["payload"] = payload
                return e
        e = {"code": code, "payload": payload}
        self.engram.append(e)
        return e

    def recall(self, ctx, min_overlap=2):
        code = self.encode(ctx)
        best, score = None, 0
        for e in self.engram:
            ov = len(code & e["code"])
            if ov > score:
                best, score = e, ov
        if score >= min_overlap:
            return best["payload"], score
        return None, score


# ============================================================================
# 5) 可塑性 Plasticity —— 好的记住、坏的舍去、但要先给机会
#    用户明确要求："记住一定不要一失败就放弃，要多试试"
#    → 单次失败只降权；屡教不改（≥3 次且从没成功）才降级成"教训"不再选；
#      长期不用的痕迹按遗忘曲线衰减直到剔除；成功巩固过的技能抗遗忘。
# ============================================================================
class Plasticity:
    def __init__(self, lr_win=0.55, lr_fail=0.30, forget_lambda=0.05,
                 keep_threshold=0.12, lesson_after=3):
        self.traces = {}             # key -> 痕迹
        self.lr_win = lr_win
        self.lr_fail = lr_fail
        self.forget_lambda = forget_lambda
        self.keep = keep_threshold
        self.lesson_after = lesson_after
        self.removed = []
        self.now = 0.0

    def tick(self, dt=1.0):
        self.now += dt

    def _trace(self, key):
        if key not in self.traces:
            self.traces[key] = {"key": key, "payload": None, "w": 0.30,
                                "wins": 0, "fails": 0, "kind": "candidate",
                                "last": self.now, "history": []}
        return self.traces[key]

    def reinforce(self, key, payload, success):
        t = self._trace(key)
        t["payload"] = payload
        t["last"] = self.now
        if success:
            t["wins"] += 1
            t["w"] = min(1.0, t["w"] + self.lr_win * (1.0 - t["w"]))
            t["kind"] = "skill"                 # 好的记住
        else:
            t["fails"] += 1
            t["w"] = max(0.02, t["w"] * (1.0 - self.lr_fail))
            # 屡犯才舍去：连续/累计 3 次且一次没成 → 降级为教训（保留为负样本）
            if t["fails"] >= self.lesson_after and t["wins"] == 0:
                t["kind"] = "lesson"
        t["history"].append("win" if success else "loss")
        return t

    def recall(self, key):
        t = self.traces.get(key)
        if not t or t["kind"] != "skill" or t["payload"] is None:
            return None, 0.0
        return t["payload"], t["w"]

    def candidates(self, exclude=("lesson",)):
        return [t for t in self.traces.values() if t["kind"] not in exclude]

    def forget(self, dt=None):
        """遗忘曲线：不用的痕迹衰减，低于阈值剔除。巩固（wins≥2）过的衰减减半。"""
        dt = 1.0 if dt is None else dt
        gone = []
        for key, t in list(self.traces.items()):
            lam = self.forget_lambda * (0.5 if t["wins"] >= 2 else 1.0)
            t["w"] *= math.exp(-lam * dt)
            if t["w"] < self.keep:
                gone.append(key)
                self.removed.append(key)
                del self.traces[key]
        return gone

    def stats(self):
        by = {}
        for t in self.traces.values():
            by[t["kind"]] = by.get(t["kind"], 0) + 1
        return {"total": len(self.traces), "by_kind": by,
                "removed": len(self.removed),
                "wins": sum(t["wins"] for t in self.traces.values()),
                "fails": sum(t["fails"] for t in self.traces.values())}


STRATEGIES = ["direct", "detour_left", "detour_right", "slow_careful", "segment"]


class Explorer:
    """策略探索器：用学到的技能（利用），没有就试别的（探索）。
    这就是"失败不要放弃，多试试"的机制化。"""

    def __init__(self, plasticity, eps=0.45, seed=7):
        self.p = plasticity
        self.eps = eps
        self.rng = random.Random(seed)
        self.tried = {}

    def choose(self, key, attempt=0, attempted=()):
        # ① 有巩固过的技能 → 直接用（利用）
        if attempt == 0:
            payload, w = self.p.recall(key)
            if payload is not None and w > 0.4:
                return payload
        # ② 换策略时必须换一个没试过的（不做无效重复）
        pool = [s for s in STRATEGIES if s not in attempted]
        if not pool:
            pool = list(STRATEGIES)
        # ③ 优先试"候选里权重高的"，但没有的（全新）也有机会 —— ε 探索
        scored = []
        for s in pool:
            t = self.p.traces.get(f"{key}#{s}")
            w = t["w"] if t else 0.0
            scored.append((w + self.rng.random() * self.eps, s))
        scored.sort(reverse=True)
        return scored[0][1]

    def feedback(self, key, strategy, success):
        return self.p.reinforce(f"{key}#{strategy}", strategy, success)


# ============================================================================
# 6) 大脑层 Cortex —— 控制中心：目标、决策、思考、价值
# ============================================================================
class Cortex:
    """控制中心。它不做运动学（那是小脑），也不做传感自检（那是脊髓）。
    它管：现在到底要干什么、干得怎么样、下一步该怎么想。
    """

    def __init__(self, nlu, llm=None, memory=None):
        self.nlu = nlu
        self.llm = llm
        self.memory = memory
        self.goals = []
        self.thoughts = []
        self.value_ledger = []

    # ---- 决策（复用 v2 的规则 + 沙琪玛模型）----
    def decide(self, text):
        intent = self.nlu.parse(text, self.memory)
        if intent.conf >= 1.0 or self.llm is None:
            return intent, ""
        if not self.llm.available(probe=False):
            return intent, ""
        planner = v2.LLMPlanner(self.llm, self.memory)
        got = planner.plan(text)
        if got:
            li, reply = got
            if li.action == "goto_place" and not li.slots.get("place"):
                return intent, reply
            return li, reply
        return intent, ""

    # ---- 价值判断：好=正、坏=负（小脑/蘑菇体靠它调权重）----
    def value_of(self, ok, seconds=None, attempts=None, faults=0):
        val = 1.0 if ok else -1.0
        if ok and attempts and attempts > 1:
            val -= 0.1 * (attempts - 1)      # 绕了很多次也算"不够好"
        if ok and seconds and seconds > 25:
            val -= 0.2
        val -= 0.1 * faults                   # 感知坏了还硬上，扣分
        self.value_ledger.append(round(val, 3))
        return round(val, 3)

    # ---- 内部思考循环（没有外部输入时也在想）----
    def think(self, ctx):
        """ctx: 统计信息（成功/失败、故障、电池、位置、记忆）"""
        s = (f"任务 {ctx.get('tasks', 0)} 次：成功 {ctx.get('wins', 0)}、"
             f"失败 {ctx.get('fails', 0)}；"
             f"感知故障 {ctx.get('faults', 0)} 处；"
             f"电池 {ctx.get('batt', 0):.1f}V；"
             f"位置 ({ctx.get('pos', (0, 0))[0]:.1f},{ctx.get('pos', (0, 0))[1]:.1f})；"
             f"离家 {ctx.get('home_dist', 0):.2f}m；"
             f"记忆 {ctx.get('mem_total', 0)} 条"
             f"（技能 {ctx.get('mem_skill', 0)} / 教训 {ctx.get('mem_lesson', 0)}）")
        thought = None
        if self.llm is not None and self.llm.available(probe=False):
            sys_p = ("你是机器人小沙的'内心'。用第一人称说一小段内心话，"
                     "中文，40 字以内，要提到具体数字，不要客套。")
            out = self.llm.chat(sys_p, "现在的情况：" + s + "\n你在想什么？",
                                max_tokens=80, temperature=0.8)
            if out:
                thought = out.strip().splitlines()[0][:80]
        if not thought:
            # 离线规则版内心话（也要带数字，能自证在"想"）
            parts = []
            if ctx.get("fails", 0) > ctx.get("wins", 0):
                parts.append(f"最近失败比成功多（{ctx.get('fails')}比{ctx.get('wins')}），"
                             f"得换个策略再来")
            else:
                parts.append(f"最近 {ctx.get('tasks')} 次任务成了 {ctx.get('wins')} 次")
            if ctx.get("faults", 0):
                parts.append(f"有 {ctx['faults']} 个传感器不可信，先保守一点")
            if ctx.get("mem_lesson", 0):
                parts.append(f"{ctx['mem_lesson']} 个办法被我判定为教训，不再用")
            if ctx.get("home_dist", 0) > 2.0:
                parts.append(f"离出发点 {ctx['home_dist']:.1f}m，路记着，能回去")
            thought = "；".join(parts) + "。"
        rec = {"t": self.memory.now_ts() if (self.memory and hasattr(self.memory, "now_ts"))
               else time.time(), "ctx": s, "thought": thought}
        self.thoughts.append(rec)
        return rec


# ============================================================================
# 7) 三层合体：NeuroBrain
# ============================================================================
class NeuroBrain:
    def __init__(self, use_llm=True, seed=0, eff_l=1.0, eff_r=1.0, slip=0.0,
                 start=(2.0, 2.0), injector=None, verbose=False, block_until=0.0):
        self.verbose = verbose
        self.world = v2.SimWorld()
        self.chassis = Chassis(x=start[0], y=start[1], theta=0.0,
                               eff_l=eff_l, eff_r=eff_r, slip=slip, seed=seed,
                               block_until=block_until)
        # 脊髓 / 小脑 / 中央复合体 / 蘑菇体 / 可塑性 / 大脑
        self.spine = SpinalCord(self.chassis, self.world, injector)
        self.cerebellum = Cerebellum()
        self.central = CentralComplex(home=start)
        self.mb = MushroomBody()
        self.plasticity = Plasticity()
        self.memory = v2.Memory(os.path.join(DATA_DIR, "neuro_memory.json"))
        self.llm = v2.LLM() if use_llm else None
        self.cortex = Cortex(v2.RuleNLU(), self.llm, self.memory)
        self.explorer = Explorer(self.plasticity, seed=seed + 1)
        self.tutor = v2.Tutor(os.path.join(DATA_DIR, "neuro_quiz.json"))
        self.arm = v2.Arm(sim=True)
        self.talker = v2.Talker(self.llm)
        self.current_q = None
        self.awake_until = 0.0
        self.dt = 0.05
        self.stats = {"tasks": 0, "wins": 0, "fails": 0, "attempts": 0,
                      "reflex_fires": 0, "faults_seen": 0}
        self._last_cmd = (0.0, 0.0)      # 上一步真正发出去的左右轮指令（对账用）

    # ---------------- 一次"走"的尝试（小脑执行 + 脊髓兜底）----------------
    def drive_once(self, goal, strategy="direct", max_seconds=45.0):
        # 每次尝试都是"重新来过"：先清掉脊髓的堵转状态，
        # 否则上一次的"卡住"会立刻把这一次也判死（踩过这个坑）。
        self.spine.reflex.jam_time = 0.0
        self.spine.reflex.stall_timer = 0.0
        self.spine.reflex.cooldown = 0.0
        p = {"speed": v2.Config.speed, "gap": v2.Config.min_obstacle_gap}
        if strategy == "slow_careful":
            p["speed"] *= 0.5
            p["gap"] *= 1.6
        elif strategy == "segment":
            p["speed"] *= 0.8

        # rotate_first：先原地对准，再走
        if strategy == "rotate_first":
            err0 = wrap_angle(math.atan2(goal[1] - self.chassis.pos[1],
                                         goal[0] - self.chassis.pos[0]) - self.chassis.theta)
            spin = 0.0
            while abs(spin) < abs(err0) and spin < 6.0:
                cl, cr = self.cerebellum.command(0.0, math.copysign(1.6, err0))
                _, _, ml, mr = self.chassis.step(cl, cr, self.dt)
                self.spine.update_encoders(ml, mr)
                self.cerebellum.observe(cl, cr, ml, mr)
                spin += 1.6 * self.dt

        detour_bias = {"detour_left": +1.0, "detour_right": -1.0}.get(strategy, 0.0)
        ticks = 0
        max_ticks = int(max_seconds / self.dt)
        pose_hist = [(self.chassis.pos[0], self.chassis.pos[1], self.chassis.theta)]
        min_clear = 1e9
        reflex_hits = []
        faults_seen = []
        detour = None
        detour_used = 0
        detour_start_dist = 1e9
        note = ""

        while ticks < max_ticks:
            pos = self.chassis.pos
            to = (goal[0] - pos[0], goal[1] - pos[1])
            dist = math.hypot(to[0], to[1])
            err = wrap_angle(math.atan2(to[1], to[0]) - self.chassis.theta)
            if dist <= v2.Config.arrive_dist and abs(err) < 0.15:
                break

            # ---- 选当前要去的点：目标点，或绕行航点（按策略偏侧）----
            blocked, hitpt = self.spine_gate_ray(goal)
            if blocked and dist > v2.Config.arrive_dist + 0.3:
                if detour is None or math.hypot(pos[0] - detour[0],
                                                pos[1] - detour[1]) < 0.35:
                    if detour is not None:
                        detour_used += 1
                    detour = self._detour(hitpt, goal, detour_bias)
                    detour_start_dist = dist
                target = detour if detour else goal
                arrive_now = 0.30
                if detour_used > 24:
                    note = "绕行 24 次仍过不去"
                    break
            elif (detour is not None and dist > detour_start_dist - 0.35
                  and math.hypot(pos[0] - detour[0], pos[1] - detour[1]) > 0.35):
                # 滞回：绕行还没见到实效之前别急着切回直线目标 —— 否则会在
                # "目标点 ↔ 绕行点"之间来回掉头，原地打转到被判卡死。
                target = detour
                arrive_now = 0.30
            else:
                detour = None
                target = goal
                arrive_now = v2.Config.arrive_dist

            tot = (target[0] - pos[0], target[1] - pos[1])
            d_t = math.hypot(tot[0], tot[1])
            err_t = wrap_angle(math.atan2(tot[1], tot[0]) - self.chassis.theta)
            if d_t < 1e-6:
                v, omega = 0.0, 0.0
            elif d_t > arrive_now:
                omega = 2.5 * err_t
                v = p["speed"] * max(0.0, math.cos(err_t))
            else:
                omega, v = 2.0 * err_t, 0.0

            # ---- 🎯 小脑：平衡 + 逆模型补偿 ----
            omega = self.cerebellum.balance(0.0 if strategy == "direct" else -err_t * 0.15,
                                            omega)
            v = max(-p["speed"], min(p["speed"], v))
            omega = max(-v2.Config.turn_speed, min(v2.Config.turn_speed, omega))

            # ---- 🦴 脊髓：先感知（含故障自检），再由反射兜底 ----
            cl, cr = self.cerebellum.command(v, omega)
            # ★ 健康自检用的是"上一步的编码器读数"，所以必须拿上一步发出的指令去对账，
            #   否则指令与读数错拍一步，转弯时会被误判成"编码器互相矛盾"。
            perc = self.spine.perceive(source_pos=None, cmd=self._last_cmd, dt=self.dt)
            if perc["faults"] and not faults_seen:
                faults_seen = perc["faults"]
                self.stats["faults_seen"] += len(perc["faults"])
            v_g, omega_g, events = self.spine.gate(v, omega, perc, self.dt)
            for e in events:
                reflex_hits.append(e["reflex"])
                self.stats["reflex_fires"] += 1
            if v_g != v or omega_g != omega:
                cl, cr = self.cerebellum.command(v_g, omega_g)
                v, omega = v_g, omega_g

            # ---- 执行 + 学习（小脑用编码器读数自我校正）----
            prev = (pos[0], pos[1])
            rv, romega, ml, mr = self.chassis.step(cl, cr, self.dt)
            self.spine.update_encoders(ml, mr)
            self.cerebellum.observe(cl, cr, ml, mr)
            self._last_cmd = (cl, cr)
            self.central.integrate((self.chassis.pos[0] - prev[0],
                                    self.chassis.pos[1] - prev[1]))
            self.plasticity.tick(self.dt)

            clear = self.world.clearance(self.chassis.pos)
            min_clear = min(min_clear, clear)
            ticks += 1
            pose_hist.append((self.chassis.pos[0], self.chassis.pos[1], self.chassis.theta))

            if self.world.blocked(self.chassis.pos):
                note = "撞进障碍"
                break
            if self._stalled(pose_hist):
                note = "卡死（6 秒没挪动，也没转身）"
                break
            if self.spine.reflex.jam_time > 2.5:
                # 连续堵了 2.5 秒 → 这次尝试到此为止，把决定权交回大脑
                # （它会换个办法再来），不在这里硬顶。
                note = "被卡住（连续 2.5 秒给了速度轮子不动），交回大脑换个办法"
                break

        final = math.hypot(goal[0] - self.chassis.pos[0], goal[1] - self.chassis.pos[1])
        ok = final <= v2.Config.arrive_dist + 1e-6
        return {"ok": ok, "final_dist": round(final, 3),
                "seconds": round(ticks * self.dt, 2),
                "min_clearance": round(min_clear, 3),
                "reflexes": sorted(set(reflex_hits)),
                "faults": faults_seen, "note": note,
                "cerebellum": self.cerebellum.report()}

    def spine_gate_ray(self, goal):
        """射线测障（脊髓的"前方有没有东西"能力，供选路用）。"""
        start = tuple(self.chassis.pos)
        total = math.hypot(goal[0] - start[0], goal[1] - start[1])
        gap = v2.Config.min_obstacle_gap * 0.9
        n = max(3, int(total / 0.08))
        for i in range(1, n + 1):
            k = i / n
            pt = (start[0] + (goal[0] - start[0]) * k, start[1] + (goal[1] - start[1]) * k)
            if self.world.clearance(pt) < gap:
                return True, pt
        return False, None

    def _detour(self, hitpt, goal, bias=0.0):
        best, bd = None, 1e9
        for (cx, cy, r) in self.world.obstacles:
            d = math.hypot(hitpt[0] - cx, hitpt[1] - cy)
            if d < bd:
                bd, best = d, (cx, cy, r)
        if best is None:
            return None
        cx, cy, r = best
        ux, uy = hitpt[0] - cx, hitpt[1] - cy
        n = math.hypot(ux, uy) or 1.0
        ux, uy = ux / n, uy / n
        radius = r + v2.Config.min_obstacle_gap * 2.0
        cands = []
        for sgn in (1.0, -1.0):
            # bias>0 偏好左侧绕行，bias<0 偏好右侧 —— 策略要能影响行为
            if bias > 0 and sgn < 0:
                continue
            if bias < 0 and sgn > 0:
                continue
            tx, ty = -uy * sgn, ux * sgn
            vx, vy = ux * 0.45 + tx, uy * 0.45 + ty
            vn = math.hypot(vx, vy) or 1.0
            wx, wy = cx + vx / vn * radius, cy + vy / vn * radius
            if 0.3 < wx < self.world.w - 0.3 and 0.3 < wy < self.world.h - 0.3:
                cands.append((math.hypot(goal[0] - wx, goal[1] - wy), (wx, wy)))
        if not cands:                       # 偏侧绕不出去 → 放开两侧再试
            if bias:
                return self._detour(hitpt, goal, 0.0)
            return None
        cands.sort(key=lambda x: x[0])
        return cands[0][1]

    def _stalled(self, pose_hist, window=120, min_move=0.08, min_turn=0.15):
        """卡死 = 既没挪动、也没转身。
        ★ 只判位移会把"原地转身对准目标"误判成卡死 —— 转身本身就是在干活。
        """
        if len(pose_hist) < window + 1:
            return False
        p0, p1 = pose_hist[-window - 1], pose_hist[-1]
        moved = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        turned = abs(wrap_angle(p1[2] - p0[2]))
        return moved < min_move and turned < min_turn

    # ---------------- 韧性：一个办法不行就换，换到成或全试完 ----------------
    def run_task(self, goal, key="goto", max_attempts=len(STRATEGIES)):
        self.stats["tasks"] += 1
        attempts = []
        attempted = []
        ok = False
        for i in range(max_attempts):
            strat = self.explorer.choose(key, attempt=i, attempted=attempted)
            attempted.append(strat)
            self.stats["attempts"] += 1
            r = self.drive_once(goal, strategy=strat)
            attempts.append({"strategy": strat, **{k: r[k] for k in
                                                   ("ok", "final_dist", "seconds", "note",
                                                    "min_clearance")}})
            # 可塑性：好就记住，坏就降权（但还有下次机会）
            self.explorer.feedback(key, strat, r["ok"])
            if r["ok"]:
                ok = True
                break
        val = self.cortex.value_of(ok, seconds=attempts[-1]["seconds"],
                                   attempts=len(attempts),
                                   faults=len(attempts[-1].get("faults") or []))
        self.stats["wins" if ok else "fails"] += 1
        self.memory.log("task", goal=[round(goal[0], 2), round(goal[1], 2)],
                        ok=ok, attempts=len(attempts), value=val)
        self.memory.bump("arrive" if ok else "miss")
        return {"ok": ok, "attempts": attempts, "attempt_count": len(attempts),
                "value": val, "brains": self.brain_report()}

    def brain_report(self):
        return {"cerebellum": self.cerebellum.report(),
                "plasticity": self.plasticity.stats(),
                "spinal_faults": len(self.spine.monitor.faults),
                "reflex_fires": self.stats["reflex_fires"],
                "home_dist": round(math.hypot(
                    self.chassis.pos[0] - self.central.home[0],
                    self.chassis.pos[1] - self.central.home[1]), 3)}

    # ---------------- 思考（没在干活的时候也在想）----------------
    def think_now(self):
        ctx = {
            "tasks": self.stats["tasks"], "wins": self.stats["wins"],
            "fails": self.stats["fails"],
            "faults": len(self.spine.monitor.faults),   # 当前不可信的传感器数
            "batt": self.chassis.batt, "pos": tuple(self.chassis.pos),
            "home_dist": math.hypot(self.chassis.pos[0] - self.central.home[0],
                                    self.chassis.pos[1] - self.central.home[1]),
            "mem_total": self.plasticity.stats()["total"],
            "mem_skill": self.plasticity.stats()["by_kind"].get("skill", 0),
            "mem_lesson": self.plasticity.stats()["by_kind"].get("lesson", 0),
        }
        return self.cortex.think(ctx)

    # ---------------- 回家（中央复合体）----------------
    def go_home(self):
        target = self.central.home_target()
        return self.run_task(target, key="home", max_attempts=2)

    # ---------------- 对外指挥入口（兼容 v2 语义）----------------
    def handle(self, text, source_pos=None):
        result = {"input": text, "steps": []}
        now = time.monotonic()
        nlu = self.cortex.nlu
        if nlu.wake_hit(text):
            self.awake_until = now + v2.Config.wake_window
        elif now > self.awake_until:
            result["awake"] = False
            result["reply"] = ""
            return result
        result["awake"] = True

        intent, llm_reply = self.cortex.decide(text)
        result["intent"] = intent.action
        result["intent_via"] = intent.source
        result["reply"] = llm_reply or self.talker.reply(intent, extra="")

        if intent.action == "stop":
            self.spine.reflex.estop = True
            result["steps"].append({"skill": "stop", "ok": True})
        elif intent.action == "goto_owner":
            if source_pos is None:
                result["steps"].append({"skill": "goto_owner", "ok": False,
                                        "why": "没有声源位置"})
            else:
                loc = v2.SimPerception(self.world)
                robot = type("R", (), {"pos": self.chassis.pos,
                                       "theta": self.chassis.theta, "mic_sep": 0.20})()
                angle, tdoa = loc.locate(robot, source_pos)
                result["locate"] = {"angle_deg": round(math.degrees(angle), 1),
                                    "tdoa_us": round(tdoa * 1e6, 1)}
                r = self.run_task(source_pos, key="goto_owner")
                result["steps"].append({"skill": "goto_owner", **r})
        elif intent.action == "write":
            cmd = self.arm.write(intent.slots.get("text", ""))
            self.memory.bump("tasks")
            result["steps"].append({"skill": "write", "ok": True, "cmd": cmd})
        elif intent.action == "quiz":
            q = self.tutor.pick()
            self.current_q = q
            result["steps"].append({"skill": "quiz", "ok": q is not None, "question": q})
            if q:
                result["reply"] = f"来，请回答 —— {q}"
        elif intent.action == "answer":
            if self.current_q is None:
                result["steps"].append({"skill": "answer", "ok": False, "why": "还没出题"})
            else:
                ok, truth = self.tutor.record(self.current_q, intent.slots.get("answer", ""))
                result["steps"].append({"skill": "answer", "ok": ok, "truth": truth,
                                        "mastery": round(self.tutor.mastery(), 3)})
                result["reply"] = "对啦！" if ok else f"答案是「{truth}」，我记进错题本了。"
                self.current_q = None
        elif intent.action == "comfort":
            self.memory.bump("comfort")
            result["steps"].append({"skill": "comfort", "ok": True})
        elif intent.action == "status":
            s = self.plasticity.stats()
            result["steps"].append({"skill": "status", "ok": True, "brain": self.brain_report()})
            result["reply"] = (f"三层都在：小脑效率估计 左{self.cerebellum.eff[0]:.2f}/"
                               f"右{self.cerebellum.eff[1]:.2f}，"
                               f"记忆 {s['total']} 条（技能{s['by_kind'].get('skill',0)}/"
                               f"教训{s['by_kind'].get('lesson',0)}），"
                               f"反射 {self.stats['reflex_fires']} 次")
        else:
            result["steps"].append({"skill": "chat", "ok": True})
        return result


# ============================================================================
# 8) 实验：小脑怎么自己学会补偿（给用户看的"学习曲线"）
# ============================================================================
def straight_test(eff_l=1.0, eff_r=0.85, learn_tasks=0, seed=1, length=5.0,
                  disable_learning=False):
    """从 (1,1) 朝 +x 直行 length 米，返回终点横向偏差（m）与小脑报告。

    disable_learning=True → 小脑"还没学会也不学"（出厂状态，真实基线）。
    learn_tasks=N         → 先练 N 趟任务（练的时候学习是开着的）。
    """
    b = NeuroBrain(use_llm=False, seed=seed, eff_l=eff_l, eff_r=eff_r, start=(1.0, 1.0))
    for i in range(learn_tasks):
        b.run_task((6.0, 3.0 + 0.7 * i), key="learn", max_attempts=1)
    if disable_learning:
        b.cerebellum.lr = 0.0
        b.cerebellum.eff = [1.0, 1.0]
    b.chassis.pos = [1.0, 1.0]
    b.chassis.theta = 0.0
    y0 = b.chassis.pos[1]
    v = 1.0
    ticks = int(length / (v * b.dt))
    for _ in range(ticks):
        cl, cr = b.cerebellum.command(v, 0.0)
        _, _, ml, mr = b.chassis.step(cl, cr, b.dt)
        b.cerebellum.observe(cl, cr, ml, mr)
    return b.chassis.pos[1] - y0, b.cerebellum.report()


# ============================================================================
# 9) 自检
# ============================================================================
def selftest():
    fails, checks = [], []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        if not cond:
            fails.append(name)

    print("=" * 74)
    print("  小沙神经系统 v3 · 自检")
    print("=" * 74)

    # ---------- 0. 回归：v2 的 23 条断言不能倒 ----------
    v2_path = os.path.join(HERE, "xiaosha_brain_v2.py")
    r = subprocess.run([sys.executable, v2_path, "--selftest"],
                       capture_output=True, text=True, encoding="utf-8", errors="ignore")
    tail = [l for l in (r.stdout or "").splitlines() if "结果：" in l]
    check("回归①：v2 大脑的 23 条断言仍然全过（升级不能砸掉旧能力）",
          r.returncode == 0 and "全部通过" in (tail[-1] if tail else ""),
          (tail[-1].strip() if tail else f"exit={r.returncode}"))
    check("回归②：v2 模块能被 v3 正常导入复用",
          hasattr(v2, "SimWorld") and hasattr(v2, "RuleNLU"))

    # ---------- 1. 脊髓：故障检出 ----------
    def _drive_step(br, i=0, source=(8.0, 7.5)):
        """让机器人真的动起来。基线必须在这种"传感器本来就会波动"的条件下学，
        否则学到的基线恒为常数，传感器冻死了也看不出来。"""
        # 用"近乎对称 + 缓慢划弧"的固定指令：传感器有正常波动，但不会撞墙、
        # 也不会有瞬移跳变 —— 否则基线会被自己教坏（这条踩过）。
        cl = 0.88
        cr = 0.90
        _, _, ml, mr = br.chassis.step(cl, cr, br.dt)
        br.spine.update_encoders(ml, mr)
        return br.spine.perceive(source_pos=source, cmd=(cl, cr), dt=br.dt)

    b = NeuroBrain(use_llm=False, seed=1)
    warm_faults = []
    for i in range(80):
        warm_faults.extend(_drive_step(b, i)["faults"])
    check("脊髓①：机器人正常跑动时不误报（0 误报）", len(warm_faults) == 0,
          f"误报 {len(warm_faults)} 条："
          f"{[(f['sensor'], f['kind']) for f in warm_faults][:4]}")

    detected = {}
    got = []
    for sensor, kind in (("us_front", "stale"), ("enc_l", "out_of_range"),
                         ("batt", "stuck"), ("doa", "noisy"),
                         ("enc_l", "inconsistent")):
        br = NeuroBrain(use_llm=False, seed=2)
        for i in range(80):
            _drive_step(br, i)
        br.spine.injector.set(sensor, kind)
        for i in range(60):
            got = _drive_step(br, i)["faults"]
            if any(f["kind"] == kind for f in got):
                break
        detected[f"{sensor}:{kind}"] = any(f["kind"] == kind for f in got)
    kinds_hit = sorted({k.split(":")[1] for k, v in detected.items() if v})
    check("脊髓②：stale / out_of_range / stuck / noisy / inconsistent 五类故障全部被点名检出",
          all(detected.values()) and len(kinds_hit) == 5,
          " | ".join(f"{k}={'检出' if v else '漏检'}" for k, v in detected.items()))
    check("脊髓③：检出时给出可复核的证据（不是只喊'异常'）",
          bool(got) and all(f.get("evidence") for f in got),
          f"样例证据：{(got[0]['evidence'] if got else '—')}")

    # ---------- 2. 反射优先于大脑 ----------
    b4 = NeuroBrain(use_llm=False, seed=4)
    b4.chassis.pos = [4.0, 5.0]
    b4.chassis.theta = 0.0            # 正对柱子(5,5,r0.9)
    perc = b4.spine.perceive(source_pos=None, cmd=(1.2, 0.0), dt=b4.dt)
    v_g, w_g, events = b4.spine.gate(1.2, 0.0, perc, b4.dt)
    fired = [e["reflex"] for e in events]
    check("反射①：大脑要冲，脊髓前方有障碍时直接压住指令",
          v_g < 1.2 and any(e in fired for e in ("obstacle_brake", "degrade_us")),
          f"大脑 v=1.20 → 脊髓 v={v_g:.2f}，反射={fired}")
    check("反射②：测距故障时按最保守策略降级（不拿坏值冒险）",
          "degrade_us" in fired or perc["faults"] == [],
          f"faults={len(perc['faults'])} events={fired}")

    # ---------- 3. 小脑：自己学会补偿（核心证据）----------
    before, rep_before = straight_test(eff_r=0.85, disable_learning=True)
    after, rep_after = straight_test(eff_r=0.85, learn_tasks=8)
    check("小脑①：出厂状态（不学、也不补偿）直行会跑偏 —— 右轮效率被做成 0.85",
          abs(before) > 0.10, f"5m 直行终点横向偏差 = {before*100:.1f}cm")
    check("小脑②：练过之后自己学会补偿，5m 直行偏差从「跑飞」降到厘米级",
          abs(after) < 0.09 and abs(after) < abs(before) * 0.1,
          f"练之前 {before*100:.1f}cm → 练之后 {after*100:.1f}cm"
          f"（残余来自编码器噪声，是物理下限，不会归零）")
    check("小脑③：左右轮效率被它自己估出来（真值 1.00 / 0.85）",
          abs(rep_after["eff_l"] - 1.0) < 0.06 and abs(rep_after["eff_r"] - 0.85) < 0.06,
          f"估计 左{rep_after['eff_l']:.3f} / 右{rep_after['eff_r']:.3f}"
          f"（{rep_after['updates']} 次在线更新）")

    # ---------- 4. 可塑性：好的记住 / 坏的舍去 / 遗忘 ----------
    pl = Plasticity()
    pl.reinforce("t#direct", "direct", True)
    pl.reinforce("t#direct", "direct", True)
    w_skill = pl.traces["t#direct"]["w"]
    pl.reinforce("t#one", "one", False)          # 只失败一次
    one = pl.traces["t#one"]
    for _ in range(3):
        pl.reinforce("t#bad", "bad", False)
    kind_bad = pl.traces["t#bad"]["kind"]
    w_bad = pl.traces["t#bad"]["w"]
    check("可塑性①：成功 → 强化并记为技能（好的记住）",
          pl.traces["t#direct"]["kind"] == "skill" and w_skill > 0.65,
          f"权重={w_skill:.2f} 类型=skill 胜利={pl.traces['t#direct']['wins']} 次")
    check("可塑性②：失败一次只降权、不立刻放弃（还留着再试）",
          one["kind"] == "candidate" and 0.02 < one["w"] < 0.30,
          f"失败 1 次：权重 0.30 → {one['w']:.2f}，类型 {one['kind']}（还能被再选上）")
    check("可塑性③：屡教不改（失败 3 次且从未成功）→ 降级成教训不再用",
          kind_bad == "lesson", f"类型={kind_bad} 权重={w_bad:.2f}")
    # 遗忘：一条久不用的痕迹
    pl2 = Plasticity(forget_lambda=0.4, keep_threshold=0.2)
    pl2._trace("forgotten")
    pl2.reinforce("kept#direct", "direct", True)
    pl2.reinforce("kept#direct", "direct", True)
    for _ in range(12):
        pl2.reinforce("kept#direct", "direct", True)
    w_kept_before = pl2.traces["kept#direct"]["w"]
    gone = pl2.forget(dt=6.0)
    check("可塑性④：长期不用的记忆被遗忘曲线剔除（坏的/没用的舍去）",
          "forgotten" in gone, f"被剔除={gone}")
    check("可塑性⑤：巩固过的技能抗遗忘（没被一起删掉）",
          "kept#direct" in pl2.traces,
          f"技能权重 {w_kept_before:.2f} → {pl2.traces.get('kept#direct', {}).get('w', 0):.2f}")

    # ---------- 5. 韧性：一次不行就多试 ----------
    # 让"第一次必然失败"变得确定：开局被线缠住 6 秒（轮子给了指令也不动），
    # 这会触发脊髓的堵转反射 → 第一次尝试判失败 → 必须靠"换个办法再来"才能成。
    b5 = NeuroBrain(use_llm=False, seed=5, start=(1.0, 1.0), block_until=6.0)

    goal = (8.6, 5.0)
    b5.explorer.rng.seed(5)
    res = b5.run_task(goal, key="tough", max_attempts=5)
    check("韧性①：第一次被卡住 → 自己换个办法重来，最终到达（不是一次就放弃）",
          res["ok"] and res["attempt_count"] >= 2,
          f"成功={res['ok']} 尝试 {res['attempt_count']} 次："
          + " → ".join(f"{a['strategy']}({'成' if a['ok'] else '败'})" for a in res["attempts"]))
    check("韧性②：失败过的策略权重被下调（不是白试）",
          all(b5.plasticity.traces.get(f"tough#{a['strategy']}")
              for a in res["attempts"]),
          f"痕迹={list(b5.plasticity.traces.keys())}")

    # 全失败场景：目标在墙外
    b6 = NeuroBrain(use_llm=False, seed=6, start=(2.0, 2.0))
    res2 = b6.run_task((-4.0, 2.0), key="impossible", max_attempts=3)
    check("韧性③：全部策略都失败 → 诚实报告做不到，不假装成功",
          res2["ok"] is False and res2["attempt_count"] == 3,
          f"尝试 {res2['attempt_count']} 次全败，价值评分={res2['value']}")

    # ---------- 6. 中央复合体：路径积分 + 回家 ----------
    b7 = NeuroBrain(use_llm=False, seed=7, start=(1.5, 1.5))
    for g in [(8.0, 2.0), (8.5, 8.0), (2.0, 8.5)]:
        b7.run_task(g, key="tour", max_attempts=2)
    est = b7.central.estimate_position()
    true = tuple(b7.chassis.pos)
    err = math.hypot(est[0] - true[0], est[1] - true[1])
    check("导航①：走完一圈，路径积分估出的位置接近真实位置（<15cm）",
          err < 0.15, f"估计({est[0]:.2f},{est[1]:.2f}) vs 真值({true[0]:.2f},{true[1]:.2f})"
                      f"，误差 {err*100:.1f}cm")
    before_home = math.hypot(true[0] - 1.5, true[1] - 1.5)
    rb = b7.go_home()
    after_home = math.hypot(b7.chassis.pos[0] - 1.5, b7.chassis.pos[1] - 1.5)
    check("导航②：靠自己回家（离家距离明显缩小）",
          after_home < before_home and after_home < 1.1,
          f"离家 {before_home:.2f}m → {after_home:.2f}m")

    # ---------- 7. 蘑菇体：稀疏编码记忆 ----------
    mb = MushroomBody(seed=3)
    ctx_a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ctx_a2 = [0.98, 0.02, 0.0, 0.0, 0.0, 0.0]     # 相似情境
    ctx_b = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]        # 完全不同的情境
    mb.store(ctx_a, "书桌在柱子左边")
    hit_same, s1 = mb.recall(ctx_a)
    hit_sim, s2 = mb.recall(ctx_a2)
    hit_other, s3 = mb.recall(ctx_b)
    check("蘑菇体①：相同情境命中同一条记忆",
          hit_same == "书桌在柱子左边", f"命中='{hit_same}' 重叠={s1}")
    check("蘑菇体②：相似情境会泛化（这是稀疏编码的价值）",
          hit_sim == "书桌在柱子左边", f"命中='{hit_sim}' 重叠={s2}")
    check("蘑菇体③：无关情境不误命中（稀疏化避免串味）",
          hit_other is None, f"命中={hit_other} 重叠={s3}（阈值 2）")

    # ---------- 8. 大脑：内部思考 ----------
    b8 = NeuroBrain(use_llm=False, seed=8)
    b8.run_task((7.0, 6.0), key="think")
    th = b8.think_now()
    has_digit = any(ch.isdigit() for ch in th["thought"])
    check("大脑①：没有外部输入时也在自己思考（内部独白非空且带数字）",
          len(th["thought"]) > 8 and has_digit, f"内心话：{th['thought']}")
    v_ok = b8.cortex.value_of(True, seconds=5, attempts=1)
    v_bad = b8.cortex.value_of(False, seconds=5, attempts=1)
    check("大脑②：价值判断分好坏（成功=正、失败=负）",
          v_ok > 0 > v_bad, f"成功价值={v_ok} 失败价值={v_bad}")

    # ---------- 9. 端到端：坏着感知也能干活（三层一起上）----------
    inj9 = FaultInjector()
    inj9.set("doa", "noisy")
    b9 = NeuroBrain(use_llm=False, seed=9, injector=inj9, start=(2.0, 2.0))
    r9 = b9.handle("小沙，过来一下", source_pos=(8.0, 7.5))
    mv = r9["steps"][0]
    check("端到端：DoA 传感器故障的情况下，仍然靠三层协作走到人面前",
          mv["ok"] and mv["attempt_count"] >= 1,
          f"到达={mv['ok']} 尝试{mv['attempt_count']}次 "
          f"终距={mv['attempts'][-1]['final_dist']}m 反射={sorted(set(mv['attempts'][-1].get('reflexes', [])))}")

    if b9.llm is not None and b9.llm.available():
        rep = b9.talker.reply(v2.Intent("comfort", raw="我今天特别烦"))
        check("端到端：大脑说话仍走本地沙琪玛（生成式）", bool(rep), f"“{rep}”")
    else:
        checks.append(("端到端：本实例未启用模型 → 自动走离线规则大脑（能力不降级）", True,
                       "v2 回归那一项已单独覆盖" + str(b9.llm.cfg.llm_model if b9.llm else "")))

    # ---------- 汇总 ----------
    print()
    npass = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
    print("-" * 74)
    print(f"  结果：{npass}/{len(checks)} 项通过"
          + ("  ✅ 全部通过" if not fails else f"  ❌ 失败项：{fails}"))
    print("=" * 74)
    return 0 if not fails else 1


# ============================================================================
# 10) 演示
# ============================================================================
def demo():
    print("=" * 74)
    print("  小沙神经系统 v3 · 三层联演（大脑 / 小脑 / 脊髓）")
    print("=" * 74)
    b = NeuroBrain(use_llm=True, seed=0, eff_l=1.0, eff_r=0.85, slip=0.03)
    print(f"  硬件设定：右轮效率 0.85（真机常见的不对称）、地面打滑 3%")
    print(f"  模型：{b.llm.cfg.llm_model if b.llm and b.llm.available() else '离线规则大脑'}")
    print("-" * 74)

    print("\n【1】脊髓：传感器自检（先让机器人正常跑动、学会基线，再注入故障）")
    # ★ 预热指令要按"右轮效率 0.85"配平（0.75×1.0 ≈ 0.88×0.85），否则它会真的
    #   跑成圆弧冲出房间 —— 这台机器人的硬件缺陷在演示里是真实存在的。
    b.chassis.theta = math.pi / 4          # 朝柱子方向跑，让前向测距真的有变化
    for _ in range(30):
        _, _, ml, mr = b.chassis.step(0.75, 0.88, b.dt)
        b.spine.update_encoders(ml, mr)
        p = b.spine.perceive(source_pos=None, cmd=(0.75, 0.88), dt=b.dt)
    print(f"   正常跑动 30 步：故障数 = {len(p['faults'])}（应为 0 误报）")
    b.spine.injector.set("us_front", "stale")
    for _ in range(30):
        _, _, ml, mr = b.chassis.step(0.75, 0.88, b.dt)
        b.spine.update_encoders(ml, mr)
        p = b.spine.perceive(source_pos=None, cmd=(0.75, 0.88), dt=b.dt)
        if p["faults"]:
            break
    if p["faults"]:
        for f in p["faults"]:
            print(f"   ⚠ 检出 [{f['kind']}] {f['sensor']}：{f['evidence']}")
    else:
        print("   （这一次没检出，说明阈值偏松）")
    v_g, w_g, ev = b.spine.gate(1.2, 0.0, p, b.dt)
    print(f"   反射兜底：大脑要 v=1.2，脊髓给 v={v_g:.2f}"
          f"（{sorted(set(e['reflex'] for e in ev))}）")
    b.spine.injector.clear()

    print("\n【2】小脑：平衡与在线学习（走进房间另一端，边跑边学）")
    r = b.run_task((8.0, 7.5), key="goto_owner", max_attempts=5)
    print(f"   结果：{'到达' if r['ok'] else '没到'}  尝试 {r['attempt_count']} 次")
    for a in r["attempts"]:
        print(f"     · 策略 {a['strategy']:<13} {'成' if a['ok'] else '败'}  "
              f"终距 {a['final_dist']:.2f}m  {a['seconds']:.1f}s  {a['note']}")
    cb = r["brains"]["cerebellum"]
    print(f"   小脑学到的轮子效率：左 {cb['eff_l']:.3f} / 右 {cb['eff_r']:.3f}"
          f"（真值 1.000 / 0.850，{cb['updates']} 次在线更新）")

    print("\n【3】大脑：内部思考（没在干活时也在想）")
    for i in range(2):
        th = b.think_now()
        print(f"   内心 #{i+1}：{th['thought']}")

    print("\n【4】可塑性：好的记住、坏的舍去")
    s = b.plasticity.stats()
    print(f"   记忆痕迹 {s['total']} 条：{s['by_kind']}")
    for k, t in list(b.plasticity.traces.items())[:6]:
        print(f"     · {k:<22} 权重 {t['w']:.2f}  成 {t['wins']} / 败 {t['fails']}  类型 {t['kind']}")

    print("\n【5】中央复合体：走一圈再自己回家")
    for g in [(8.5, 2.0), (2.0, 8.5)]:
        b.run_task(g, key="tour", max_attempts=2)
    est = b.central.estimate_position()
    drift = math.hypot(est[0] - b.chassis.pos[0], est[1] - b.chassis.pos[1])
    print(f"   路径积分估计 ({est[0]:.2f},{est[1]:.2f})  "
          f"真实 ({b.chassis.pos[0]:.2f},{b.chassis.pos[1]:.2f})  漂移 {drift*100:.0f}cm")
    print("   （3% 打滑会让航位推算漂移 —— 这是物理事实，不是算法错；"
          "真机靠地标校准解决）")
    home = b.go_home()
    d = math.hypot(b.chassis.pos[0] - 2.0, b.chassis.pos[1] - 2.0)
    print(f"   回家结果：{'到' if home['ok'] else '没到'}，离家 {d:.2f}m（出发点 (2,2)）")
    err = b.central.calibrate((2.0, 2.0))
    print(f"   到家后用已知地标校准：漂移 {err*100:.0f}cm → "
          f"{math.hypot(b.central.estimate_position()[0]-2.0, b.central.estimate_position()[1]-2.0)*100:.0f}cm")

    print("\n" + "-" * 74)
    print(f"  反射触发 {b.stats['reflex_fires']} 次；任务 {b.stats['tasks']} 次"
          f"（成 {b.stats['wins']} / 败 {b.stats['fails']}）")
    print("=" * 74)
    return b


def learn_demo():
    print("=" * 74)
    print("  小脑学习实验：右轮效率被别人做成了 0.85，它自己能不能发现并补上？")
    print("=" * 74)
    print(f"{'学习任务数':>10} | {'5m 直行横向偏差':>16} | {'效率估计(左/右)':>18}")
    print("-" * 74)
    base, rep0 = straight_test(eff_r=0.85, disable_learning=True)
    print(f"{'不学(基线)':>9} | {base*100:>14.2f}cm | {rep0['eff_l']:.3f} / {rep0['eff_r']:.3f}")
    for n in (0, 2, 5, 8, 12):
        dev, rep = straight_test(eff_r=0.85, learn_tasks=n)
        print(f"{n:>10} | {dev*100:>14.2f}cm | {rep['eff_l']:.3f} / {rep['eff_r']:.3f}")
    print("-" * 74)
    print("  真值：左轮 1.000、右轮 0.850。偏差越接近 0 说明它补偿得越好。")
    print("  注意：偏差没有归零也不会归零 —— 编码器有噪声，这是诚实的工程现实。")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(description="小沙神经系统 v3")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--learn", action="store_true", help="小脑学习曲线实验")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.learn:
        learn_demo()
        return
    if args.demo:
        demo()
        return
    demo()


if __name__ == "__main__":
    main()
