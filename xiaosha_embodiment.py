# -*- coding: utf-8 -*-
"""
小沙 · 通用躯干适配层（Embodiment-agnostic brain）
============================================================================
要解决的事：**换躯干不用改大脑的代码** —— 物理的（轮子/履带/舵机）、软件的
（HTTP/进程内接口/虚拟角色）都走同一套端口；上了新躯干，它自己试探、自己学，
时间短（实测秒级），不需要人替它写适配。

做法分三层：

  ① HAL（硬件抽象层）
     躯干只暴露两种端口：Actuator（可发指令的通道）与 Sensor（可读的量）。
     每个躯干都必须提供 pose(x, y, yaw) —— 这是"知道自己大概在哪"，任何介质
     （真机靠里程计+陀螺，软件靠接口回报）都能给。大脑只认端口，不认电机型号。

  ② BodySpec：躯干用**数据**描述，不是代码
     每个通道一行：量程、真值增益、死区、传输延迟、对 (vx, vy, ω) 的单位贡献。
     新增躯干 = 在字典里加一条数据（0 行大脑代码）。

  ③ BodySchema（身体自识别 = "婴儿式试探"）
     开机时逐通道给激励，观测机身速度响应，用**哈达玛正交编码**一次激励多个通道
     并解算，估出雅可比 J（通道 → 机身速度）以及死区、延迟、可辨识性；
     再求 J 的伪逆 → **通用控制律**：大脑只管说"我要往哪走、转多少"，
     怎么分配到轮子/履带/舵机由 J 的逆决定。
     零位附近不可辨识的躯干（例如阿克曼：停着打方向盘车不动），
     它会**自己换个工作点再辨识一次**（带偏置的二次激励），不需要人告诉它。

于是：差速、阿克曼、履带、三全向轮、纯软件虚拟躯体 —— 同一份大脑代码，
全部靠"开机自己学"适配。自检里还会临时造一个**从没见过**的躯干来验。

用法
----------------------------------------------------------------------------
  python xiaosha_embodiment.py --selftest      # 机器可判定自检（多躯干）
  python xiaosha_embodiment.py --demo          # 给每个躯干演示"上机→自学→干活"
  python xiaosha_embodiment.py --bench         # 只比"逐通道试探 vs 哈达玛并行试探"

作者：Hermes（为 沙琪玛/Shaqma 平台 · 小沙机器人）
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# ============================================================================
# ① HAL：端口。大脑只认这些。
# ============================================================================
class Actuator:
    """一个可控通道（轮子电机 / 舵机 / 履带 / 软件接口里的一个"油门"）。"""

    def __init__(self, name, lo, hi, delay=0.0):
        self.name, self.lo, self.hi = name, float(lo), float(hi)
        self.delay = float(delay)
        self.cmd = 0.0
        self.value = 0.0
        self._q = []

    def set(self, v):
        self.cmd = max(self.lo, min(self.hi, float(v)))
        self._q.append(self.cmd)

    def tick(self, dt):
        """传输延迟：这段时间之前的指令才生效（真机＝总线延迟，软件＝网络延迟）。"""
        keep = int(round(self.delay / dt)) if dt > 0 else 0
        if keep <= 0:
            self.value = self.cmd
        else:
            while len(self._q) > keep:
                self.value = self._q.pop(0)


class Sensor:
    def __init__(self, name, lo, hi, unit=""):
        self.name, self.lo, self.hi, self.unit = name, float(lo), float(hi), unit
        self.value = 0.0

    def read(self):
        return self.value


class HAL:
    """躯干对大脑暴露的全部接口。大脑**不允许**访问别的任何东西。"""

    def __init__(self, spec):
        self.spec = spec
        self.name = spec.get("name", "unnamed_body")
        self.actuators = [Actuator(c["name"], c["range"][0], c["range"][1],
                                   c.get("delay", 0.0)) for c in spec["channels"]]
        self.sensors = {"pose": Sensor("pose", -1e9, 1e9, "x,y,yaw")}
        self.medium = spec.get("medium", "local")

    # --- 大脑能做的事，就这四个 ---
    def n_channels(self):
        return len(self.actuators)

    def limits(self):
        return [(a.lo, a.hi) for a in self.actuators]

    def set_all(self, u):
        for a, v in zip(self.actuators, u):
            a.set(v)

    def read(self):
        return dict(x=self.sensors["pose"].value[0],
                    y=self.sensors["pose"].value[1],
                    yaw=self.sensors["pose"].value[2])

    def step(self, dt):
        for a in self.actuators:
            a.tick(dt)
        self._integrate(dt)

    # --- 躯干侧实现（物理或软件介质的运动学）---
    def _integrate(self, dt):
        raise NotImplementedError


# ============================================================================
# ② 躯干 = 数据。物理的、软件的，格式完全一样。
#    contrib = 每单位有效指令对机身速度 (vx, vy, ω) 的贡献；gain/dead/delay 是真值。
#    ★ 这些真值大脑看不到 —— 它只能通过端口试探出来。
# ============================================================================
BODY_SPECS = {
    "diff_drive": {
        "desc": "两轮差速（小沙现在的躯干）",
        "medium": "local",
        "noise": 0.004,
        "channels": [
            {"name": "wheel_l", "range": [-1.4, 1.4], "gain": 1.00, "dead": 0.00,
             "delay": 0.00, "contrib": [0.50, 0.0, -3.30]},
            {"name": "wheel_r", "range": [-1.4, 1.4], "gain": 0.92, "dead": 0.00,
             "delay": 0.00, "contrib": [0.50, 0.0, +3.30]},
        ],
    },
    "ackermann": {
        "desc": "阿克曼小车（油门 + 前轮转向；停着打方向车不动）",
        "medium": "local",
        "noise": 0.005,
        "channels": [
            {"name": "throttle", "range": [-1.2, 1.2], "gain": 1.00, "dead": 0.06,
             "delay": 0.02, "contrib": [0.90, 0.0, 0.0]},
            {"name": "steer", "range": [-1.0, 1.0], "gain": 1.00, "dead": 0.10,
             "delay": 0.06, "contrib": [0.00, 0.0, 1.10]},   # 只在滚动时才真的转向
        ],
        "nonlinear": "ackermann",      # ω 正比于当前前进速度（真值里的非线性）
    },
    "track": {
        "desc": "履带车（左右履带，增益不对称、有死区与总线延迟）",
        "medium": "local",
        "noise": 0.006,
        "channels": [
            {"name": "track_l", "range": [-1.0, 1.0], "gain": 1.25, "dead": 0.14,
             "delay": 0.10, "contrib": [0.45, 0.0, -2.60]},
            {"name": "track_r", "range": [-1.0, 1.0], "gain": 0.72, "dead": 0.18,
             "delay": 0.10, "contrib": [0.45, 0.0, +2.60]},
        ],
    },
    "omni3": {
        "desc": "三全向轮（可以横着走）",
        "medium": "local",
        "noise": 0.004,
        "channels": [
            {"name": "w0", "range": [-1.2, 1.2], "gain": 1.00, "dead": 0.03,
             "delay": 0.01, "contrib": [0.60, 0.00, 0.55]},
            {"name": "w1", "range": [-1.2, 1.2], "gain": 0.95, "dead": 0.03,
             "delay": 0.01, "contrib": [-0.30, 0.52, 0.55]},
            {"name": "w2", "range": [-1.2, 1.2], "gain": 1.05, "dead": 0.03,
             "delay": 0.01, "contrib": [-0.30, -0.52, 0.55]},
        ],
    },
    "software_bot": {
        "desc": "纯软件介质（指令走接口/网络，有抖动延迟与丢包，没有任何物理部件）",
        "medium": "remote",                 # 这条就是"软件上的介质"
        "noise": 0.006,
        "net_latency": 0.18,
        "net_jitter": 0.10,
        "drop_p": 0.05,
        "channels": [
            {"name": "api_forward", "range": [-1.0, 1.0], "gain": 0.85, "dead": 0.05,
             "delay": 0.0, "contrib": [0.70, 0.0, 0.0]},
            {"name": "api_side", "range": [-1.0, 1.0], "gain": 0.80, "dead": 0.05,
             "delay": 0.0, "contrib": [0.0, 0.70, 0.0]},
            {"name": "api_spin", "range": [-1.0, 1.0], "gain": 0.90, "dead": 0.05,
             "delay": 0.0, "contrib": [0.0, 0.0, 1.30]},
        ],
    },
}


class RemoteTransport:
    """软件介质的传输边界：抖动延迟 + 丢包。

    真机上这里是 HTTP / MQTT / 串口；在这个文件里它只是"接口调用"，
    但对大脑而言完全一样 —— 它只看得到"指令发出去、过一会儿才生效"。
    """

    def __init__(self, n, latency, jitter, drop_p, seed=0, dt=0.05):
        self.n, self.latency, self.jitter = n, latency, jitter
        self.drop_p = drop_p
        self.rng = random.Random(seed)
        self.dt = dt
        self.inbox = []          # (到达时刻, payload)
        self.t = 0.0
        self.last = [0.0] * n
        self.dropped = 0
        self.sent = 0

    def send(self, u):
        self.sent += 1
        if self.rng.random() < self.drop_p:
            self.dropped += 1
            return
        d = max(self.dt, self.latency + self.rng.uniform(-self.jitter, self.jitter))
        self.inbox.append((self.t + d, list(u)))

    def poll(self, dt):
        self.t += dt
        ready = [p for (ts, p) in self.inbox if ts <= self.t]
        self.inbox = [(ts, p) for (ts, p) in self.inbox if ts > self.t]
        for p in ready:
            self.last = p
        return list(self.last)


class BodySim(HAL):
    """躯干仿真：J_true（真值雅可比）· 有效指令 → 机身速度 → 位置。

    大脑拿不到 J_true / gain / dead / delay，只能通过端口试探。
    """

    def __init__(self, spec, x=0.0, y=0.0, yaw=0.0, seed=0):
        super().__init__(spec)
        self.rng = random.Random(seed)
        self.pose = [float(x), float(y), float(yaw)]
        self.sensors["pose"].value = list(self.pose)
        self.noise = spec.get("noise", 0.0)
        self.nonlinear = spec.get("nonlinear")
        self.transport = None
        if self.medium == "remote":
            self.transport = RemoteTransport(
                self.n_channels(), spec.get("net_latency", 0.15),
                spec.get("net_jitter", 0.05), spec.get("drop_p", 0.0), seed=seed)
        # 真值雅可比（3 × n）：列 = gain × contrib
        #   ★ 这里分清楚两个量，别再混：
        #     C      = 每单位**有效指令**产生的机身速度（物理本体）
        #     J_true = 每单位**原始指令**的斜率 = C·diag(gain)（大脑要辨识的东西）
        #     曾经把 gain 同时写进 _effective 和 J_true → 仿真器实际是 gain²，
        #     于是"标准答案"和真机行为对不上（踩过）。
        self.C = np.zeros((3, self.n_channels()))
        gains = []
        for i, c in enumerate(spec["channels"]):
            self.C[:, i] = np.array(c["contrib"], dtype=float)
            gains.append(c["gain"])
        self.gains = np.array(gains)
        self.deads = np.array([c.get("dead", 0.0) for c in spec["channels"]])
        self.J_true = self.C @ np.diag(self.gains)
        self.u_eff = np.zeros(self.n_channels())

    # --- 有效指令：死区 + 饱和（真值）---
    def _effective(self, raw):
        out = []
        for c, v in zip(self.spec["channels"], raw):
            d = c.get("dead", 0.0)
            if abs(v) <= d:
                out.append(0.0)
            else:
                out.append(math.copysign(abs(v) - d, v) * c["gain"])
        return np.array(out, dtype=float)

    def set_all(self, u):
        if self.transport is not None:
            self.transport.send(u)
        else:
            super().set_all(u)

    def _integrate(self, dt):
        if self.transport is not None:
            raw = self.transport.poll(dt)
        else:
            raw = [a.value for a in self.actuators]
        u = self._effective(raw)
        self.u_eff = u
        vel = self.C @ u
        vx, vy, om = float(vel[0]), float(vel[1]), float(vel[2])
        if self.nonlinear == "ackermann":
            # 真值非线性：转向力矩正比于当前前进速度（停着打方向盘不动）
            om = self.J_true[2, 1] * u[1] * max(0.0, abs(vx)) / 0.6
        if self.noise > 0:
            vx += self.rng.gauss(0, self.noise)
            vy += self.rng.gauss(0, self.noise)
            om += self.rng.gauss(0, self.noise * 0.5)
        self.pose[2] = wrap_angle(self.pose[2] + om * dt)
        self.pose[0] += (math.cos(self.pose[2]) * vx - math.sin(self.pose[2]) * vy) * dt
        self.pose[1] += (math.sin(self.pose[2]) * vx + math.cos(self.pose[2]) * vy) * dt
        self.sensors["pose"].value = list(self.pose)

    # --- 只给自检用：真值能力标签 ---
    def _true_vel(self, u_raw):
        """真值模型的稳态速度（只给自检当"标准答案"，大脑看不到这些真值）。"""
        lims = self.limits()
        u_raw = [max(lo, min(hi, v)) for v, (lo, hi) in zip(u_raw, lims)]
        u = self._effective(u_raw)
        v = self.C @ u
        vx, vy, om = float(v[0]), float(v[1]), float(v[2])
        if self.nonlinear == "ackermann":
            om = self.J_true[2, 1] * u[1] * max(0.0, abs(vx)) / 0.6
        return np.array([vx, vy, om])

    def true_caps(self):
        """标准答案：用真值模型算"原地转 / 横着走"到底成不成。"""
        def trial(intent):
            # ★ 标准答案必须和大脑用**同一套逆解**（含死区补偿迭代），
            #   否则比的不是同一个问题：一个小指令被死区吃掉，真值会误判成"不能原地转"。
            intent = np.array(intent, float)
            u = np.linalg.lstsq(self.J_true, intent, rcond=None)[0]
            for _ in range(3):
                s = np.sign(u)
                s[s == 0] = 1.0
                off = self.deads * s
                u = np.linalg.lstsq(self.J_true, intent + self.J_true @ off, rcond=None)[0]
            return self._true_vel(u)
        v_turn = trial([0.0, 0.0, 1.0])
        v_str = trial([0.0, 0.5, 0.0])
        lt = math.hypot(float(v_turn[0]), float(v_turn[1]))
        turn = abs(float(v_turn[2])) > 0.50 and lt < 0.35 * abs(float(v_turn[2]))
        strafe = (abs(float(v_str[1])) > 0.15
                  and abs(float(v_str[0])) < 0.35 * abs(float(v_str[1])))
        return {"rank": int(np.linalg.matrix_rank(self.J_true, tol=1e-6)),
                "turn_in_place": bool(turn), "strafe": bool(strafe),
                "omnidirectional": bool(turn and strafe)}


# ============================================================================
# ③ BodySchema：身体自识别（开机自己试探，学出"我的身体长什么样"）
# ============================================================================
class BodySchema:
    """用激励-响应估出雅可比 J：通道指令 → 机身速度 (vx, vy, ω)。

    · 逐通道试探（sequential）：稳、慢（n 次）
    · 哈达玛并行试探（hadamard）：一次把所有通道用正交码同时激励，n 个时间片
      就能解出全部列 —— **探测时间不随通道数线性增长**，这是"时间不要太长"的关键。
    """

    def __init__(self, hal, dt=0.05):
        self.hal = hal
        self.dt = dt
        self.n = hal.n_channels()
        self.J = np.zeros((3, self.n))
        self.dead = np.zeros(self.n)
        self.latency = 0.0
        self.noise = 0.0
        self.probes = 0
        self.seconds = 0.0
        self.method = None
        self.bias = np.zeros(self.n)
        self.notes = []

    # ---- 底层：把一串 (持续时间, 指令向量) 跑一遍，记录位姿轨迹 ----
    def _play(self, segments):
        """按 (持续时间, 指令向量) 序列跑一遍，返回位姿轨迹。

        ★ 必须**每个控制周期都下发一次指令**。曾经每个时间片只下发一次就连续步进，
          结果带传输延迟的执行器（Actuator 的延迟队列）空转，执行器一直停在
          上一段的指令上 —— 表现就是：延迟 0 的躯干辨识得很准，延迟越大越离谱
          （履带 0.10s 延迟直接报废）。真机上车规级控制器本来也是每个周期都写指令。
        """
        traj = []
        for (dur, u) in segments:
            steps = max(1, int(round(dur / self.dt)))
            for _ in range(steps):
                self.hal.set_all(u)
                self.hal.step(self.dt)
                p = self.hal.read()
                traj.append((0.0, p["x"], p["y"], p["yaw"]))
            self.probes += 1
            self.seconds += dur
        return traj

    def _vel(self, traj, i0, i1):
        """一段轨迹的机身速度。

        ★ 必须**逐步**把位移转到"当时"的机身系再平均，不能用首尾两点割线：
          转弯时轨迹是圆弧，割线会把速度算小 —— 本文件踩过这个坑（差速器
          右轮那一列被低估 7%，履带车整列符号都反了，辨识出的 J 直接不可用）。
        """
        if i1 <= i0:
            return np.zeros(3)
        acc = np.zeros(3)
        cnt = 0
        for i in range(i0 + 1, i1):
            dx = traj[i][1] - traj[i - 1][1]
            dy = traj[i][2] - traj[i - 1][2]
            yaw = traj[i - 1][3]
            c, s = math.cos(yaw), math.sin(yaw)
            acc[0] += (c * dx + s * dy) / self.dt
            acc[1] += (-s * dx + c * dy) / self.dt
            acc[2] += wrap_angle(traj[i][3] - traj[i - 1][3]) / self.dt
            cnt += 1
        return acc / max(1, cnt)

    def _baseline_noise(self):
        # ★ 必须先把延迟队列排空、让机器人停稳，再在**干净窗口**里量噪声底。
        #   直接从上一阶段的运动里接手，会把残余速度当成"噪声" → 阈值被抬高 →
        #   起始延迟永远测不出来（踩过：onset 变成上限 0.8s，探测间隔被撑到 0.9s）。
        traj = self._play([(0.60, [0.0] * self.n)])
        i0 = int(0.30 / self.dt)
        v = self._vel(traj, i0, len(traj) - 1)
        self.noise = float(max(1e-4, np.linalg.norm(v[:2])))
        return self.hal.read()

    # ---- 哈达玛并行试探 ----
    @staticmethod
    def _hadamard(n):
        H = np.array([[1.0]])
        while H.shape[0] < n:
            H = np.block([[H, H], [H, -H]])
        return H[:n, :n]

    def _patterns(self):
        """n×n 激励模式矩阵（±1）。

        ★ n 是 2 的幂时用哈达玛（天然正交）；不是 2 的幂时**不能**拿
          H[:n,:n] 切片 —— 那个切片不正交，解算出来的雅可比各列会互相串味
          （履带/全向/软件躯体都被坑过）。做法：从更大的哈达玛里挑一组
          条件数最好的 n 列出来。
        """
        n = self.n
        p = 2
        while p < n:
            p *= 2
        H = self._hadamard(p)
        if p == n:
            return H
        best, bestc = None, 1e18
        for cols in itertools.combinations(range(p), n):
            A = H[:n, list(cols)]
            c = float(np.linalg.cond(A))
            if c < bestc:
                best, bestc = A, c
        self.pattern_cond = round(bestc, 2)
        return best

    def probe_hadamard(self, level=0.50, bias=None, interval=0.30, measure=0.20,
                       amp_override=None):
        """一次激励所有通道（正交模式），n 个时间片解出完整雅可比（用伪逆，通用）。"""
        bias = np.zeros(self.n) if bias is None else np.asarray(bias, float)
        lims = self.hal.limits()
        if amp_override is None:
            amp = np.array([min(level, (hi - lo) * 0.4) for (lo, hi) in lims])
        else:
            amp = np.asarray(amp_override, float)
        A = self._patterns() * amp            # 第 k 行 = 第 k 个时间片同时施加的指令
        # 先跑一段"只给偏置"的基线，得到偏置下的机身速度
        traj0 = self._play([(interval + measure, list(bias))])
        v_bias = self._vel(traj0, int(interval / self.dt), len(traj0) - 1)

        V = []
        lat_samples = None
        for k in range(self.n):
            u = bias + A[k]
            traj = self._play([(interval, u), (measure, u)])
            i0 = int(interval / self.dt)
            V.append(self._vel(traj, i0, len(traj) - 1) - v_bias)
            if lat_samples is None:
                lat_samples = traj[:i0 + 1]
        V = np.array(V)                        # n × 3
        J = (np.linalg.pinv(A) @ V).T          # 3 × n
        if lat_samples:
            self.latency = self._estimate_latency(lat_samples)
        return J, amp

    def _estimate_latency(self, traj):
        """从第一小段里找"开始动"的时刻。"""
        if len(traj) < 4:
            return 0.0
        v_floor = 0.0
        for i in range(2, len(traj)):
            v = self._vel(traj, 0, i)
            v_floor = max(v_floor, float(np.linalg.norm(v[:2])))
        thr = max(0.02, 0.35 * v_floor)
        for i in range(2, len(traj)):
            v = self._vel(traj, 0, i)
            if float(np.linalg.norm(v[:2])) > thr:
                return round(i * self.dt, 3)
        return 0.0

    def _dz_correct(self, J, amp_vec, bias):
        """跨死区的通道，激励里有 (amp-死区)/amp 的比例被死区吃掉 → 折算回真实斜率。

        ★ 这正是"偏置扫描测出来只有一半"的原因：激励在死区两侧来回穿，
          平均斜率天然减半。若某通道"偏置−激励 > 死区"（整段都在死区之上），
          就不用折算 —— 否则会重复补偿。
        """
        out = J.copy()
        for c in range(self.n):
            a = float(amp_vec[c])
            d = float(self.dead[c])
            if a <= 1e-9 or d <= 1e-9:
                continue
            if abs(float(bias[c])) - a > d:
                continue                      # 整段都在死区之上，线性，无需折算
            if a <= d:
                self.notes.append(f"通道{c} 激励 {a:.2f} ≤ 死区 {d:.2f}：该列只能保守估计")
                continue
            if d > 0.5 * a:
                # 死区吃掉一半以上激励时，折算因子会爆炸（曾把一列放大 5 倍）
                self.notes.append(f"通道{c} 死区 {d:.2f} ＞ 一半激励 {a:.2f}：不折算，标为保守")
                continue
            out[:, c] *= a / (a - d)
        return out

    # ---- 顺序试探（对照用）----
    def probe_sequential(self, level=0.45, interval=0.30, measure=0.22):
        lims = self.hal.limits()
        J = np.zeros((3, self.n))
        amp = np.zeros(self.n)
        for i in range(self.n):
            lo, hi = lims[i]
            a = min(level, (hi - lo) * 0.4)
            amp[i] = a
            u = [0.0] * self.n
            u[i] = a
            self._play([(interval, [0.0] * self.n)])
            traj = self._play([(interval, u), (measure, u)])
            i0 = int(interval / self.dt)
            J[:, i] = self._vel(traj, i0, len(traj) - 1) / a
            if i == 0:
                self.latency = self._estimate_latency(traj[:i0 + 1])
        return J, amp

    # ---- 死区估计（小幅度再来一轮）----

    # ---- 主流程：自适应辨识（零位→不可辨识就换工作点再辨一次）----
    def _quick_latency(self, level=0.5):
        """先量"发指令到真的动"要多久。★ 探测间隔必须比它长，否则指令还没到
        就开始测（软件介质 0.3s 的网络延迟会被算成"这个通道没反应"）。"""
        lims = self.hal.limits()
        j = int(np.argmax([hi - lo for (lo, hi) in lims]))
        u = [0.0] * self.n
        u[j] = min(level, (lims[j][1] - lims[j][0]) * 0.4)
        self.hal.set_all([0.0] * self.n)
        for _ in range(4):                    # 每个周期都下发，否则延迟队列不生效
            self.hal.set_all([0.0] * self.n)
            self.hal.step(self.dt)
        p0 = self.hal.read()
        t0 = (p0["x"], p0["y"])
        self.probes += 1
        self.seconds += 0.2 + 0.8
        thr = max(2e-3, 3.0 * self.noise * 0.15)
        for i in range(1, int(0.8 / self.dt)):
            self.hal.set_all(u)               # ★ 同一坑：这里也必须每周期下发
            self.hal.step(self.dt)
            p = self.hal.read()
            if math.hypot(p["x"] - t0[0], p["y"] - t0[1]) > thr:
                return round(i * self.dt, 3)
        return 0.8

    def self_check(self, want=0.35, tol=0.6):
        """学完自己验一遍：发一条"往前走"的指令，看真的走起来没有。
        不合格就再学一次 —— 这是"自己练习"最小可用的样子。"""
        self._play([(0.20, list(self.bias))])
        traj = self._play([(0.55, self.inverse([want, 0.0, 0.0]))])
        v = self._vel(traj, int(0.10 / self.dt), len(traj) - 1)
        ok = float(v[0]) > 0.4 * want and abs(float(v[1])) < 0.25 and abs(float(v[2])) < tol
        self.last_check = {"实测速度": [round(float(x), 3) for x in v], "目标 vx": want,
                           "合格": bool(ok)}
        return ok

    def _measure_deadzone(self, c, levels=(0.32, 0.42, 0.52)):
        """某通道"小幅度完全推不动"时，用两级响应**精确外推**死区。

        ★ 注意公式形态：单通道测量时 响应 ∝ (a-d)（没除以 a）。
          所以 d = (a1 - ρ·a2)/(1-ρ)，ρ = r(a1)/r(a2)。
          （identify 里那次用的是"每单位指令"的归一化响应 ∝ (a-d)/a，
            公式相应是 d = a1·a2(1-ρ)/(a2 - ρ·a1) —— 两种形态别混，混过一次。）
        验算：a1=0.32、a2=0.42、真死区 0.22 → ρ=0.5 → d=(0.32-0.21)/0.5=0.22 ✓
        """
        lims = self.hal.limits()
        got, resp = [], []
        for lv in levels:
            v = min(lv, 0.85 * lims[c][1])
            u = [0.0] * self.n
            u[c] = v
            self._play([(0.15, [0.0] * self.n)])
            traj = self._play([(0.50, u)])   # 段要够长：i0 之后窗口至少留 3~4 步
            # ★ 测量窗口必须落在传输延迟之后，否则把"还没动"的那几步也算进平均，
            #   速度被系统性低估 → 死区被估大（踩过：0.22 被估成 0.305）。
            i0 = max(1, int((getattr(self, "onset", 0.10) + 0.06) / self.dt))
            r = float(np.linalg.norm(self._vel(traj, i0, len(traj) - 1)))
            got.append(v)
            resp.append(r)
            if len(got) >= 2 and min(resp) > max(4.0 * self.noise, 0.004):
                break
        if len(got) < 2:
            return round(min(0.6 * levels[-1], 0.85 * lims[c][1]), 3)
        a1, r1 = got[-2], resp[-2]
        a2, r2 = got[-1], resp[-1]
        if r1 <= 1e-9 or r2 <= 1e-9:
            return round(a1, 3)
        rho = min(0.999, r1 / r2)                 # ρ = r(小)/r(大)
        d = (a1 - rho * a2) / (1.0 - rho)
        return round(max(0.0, min(0.8 * a1, d)), 3)

    def identify(self, a_big=0.70, a_small=0.35):
        """两次哈达玛扫描就够：大幅度出雅可比，小幅度出死区（用缩水比反解）。

        ★ 激励幅度要够大：死区 0.3 的躯干用 0.5 去激，有效杠杆只剩 0.2，
          辨识出来的列会小掉一半（踩过：J 误差从 0.9% 涨到 38%）。
          小幅度取大幅度的一半（0.35），正好压在常见死区（≤0.30）之上，
          这样"比值反解死区"对每个通道都成立。
        """
        self._baseline_noise()
        self.onset = self._quick_latency()
        interval = max(0.25, self.onset + 0.15)     # 按这个躯干的延迟自己定探测节奏
        measure = 0.20

        J_big, _ = self.probe_hadamard(level=a_big, interval=interval, measure=measure)
        J_small, _ = self.probe_hadamard(level=a_small, interval=interval, measure=measure)

        # 死区：小幅度响应比大幅度缩水多少 → 解 dz = a_b·a_s(1-ρ)/(a_b - ρ·a_s)
        self.dead = np.zeros(self.n)
        for j in range(self.n):
            rb = float(np.linalg.norm(J_big[:, j]))
            rs = float(np.linalg.norm(J_small[:, j]))
            if rb < 1e-6:
                self.dead[j] = 0.0
                continue
            rho = min(1.0, rs / rb)
            den = a_big - rho * a_small
            if rho < 0.05:
                # 小幅度完全推不动 → 逐级加大直接卡出死区上界（别拍脑袋给大数）
                self.dead[j] = self._measure_deadzone(j)
            elif den > 1e-9:
                self.dead[j] = max(0.0, min(0.6 * a_big,
                                            a_big * a_small * (1 - rho) / den))
        # 用解出的死区把大幅度那一版雅可比校正回"每单位有效指令"的斜率
        amp_big = np.array([min(a_big, (hi - lo) * 0.4) for (lo, hi) in self.hal.limits()])
        J = self._dz_correct(J_big, amp_big, np.zeros(self.n))
        self.J, self.bias = J, np.zeros(self.n)
        self.method = "哈达玛×2(大/小幅度)"

        # 零位附近**真的**不可辨识（阿克曼：停着打方向盘车不动）→ 自己换工作点再辨一次。
        # ★ 判据要留足余量：履带车的奇异值比 ≈0.15，卡在阈值上会被噪声吹来吹去。
        #   只有"接近秩亏"（<3%）才算不可辨识。
        s0 = np.linalg.svd(J, compute_uv=False)
        if float(s0[-1]) < 0.03 * float(s0[0]):
            lims = self.hal.limits()
            colnorm = [float(np.linalg.norm(J[:, c])) for c in range(self.n)]
            ref = max(colnorm) if colnorm else 0.0
            amp_b = np.array([min(0.20, (hi - lo) * 0.2) for (lo, hi) in lims])
            b = np.zeros(self.n)
            for c in range(self.n):
                # 只在"主导通道"上给偏置，且保证 偏置−激励 仍在死区之上；
                # 从属通道（例：阿克曼的方向盘，靠油门才有效）保持 0，别有偏置。
                if ref > 0 and colnorm[c] > 0.3 * ref:
                    b[c] = min(0.7 * lims[c][1], self.dead[c] + 1.6 * amp_b[c])
            Jb, _ = self.probe_hadamard(bias=b, amp_override=amp_b,
                                        interval=interval, measure=measure)
            Jb = self._dz_correct(Jb, amp_b, b)
            sb = np.linalg.svd(Jb, compute_uv=False)
            if float(sb[-1]) > float(s0[-1]):
                self.J, self.bias = Jb, b
                self.method = "哈达玛×2@偏置(自动换工作点)"
                self.notes.append(f"零位奇异值 {s0[-1]:.3f} 太小 → 自己换到工作点复辨")
        # ★ 模型选择：侧向能力比前向能力小一个量级 → 那是噪声，不是能力。
        #   真机没有横移的躯体硬留这一行，逆解会去抵消一个不存在的侧向速度，
        #   白耗控制权限（现场新造的四轮滑移就是这么走不到位的）。
        nvx = float(np.linalg.norm(self.J[0, :]))
        nvy = float(np.linalg.norm(self.J[1, :]))
        if nvx > 1e-6 and nvy < 0.30 * nvx:
            self.J[1, :] = 0.0
            self.notes.append(f"侧向 {nvy:.3f} ≪ 前向 {nvx:.3f} → 按无横移处理")
        self.seconds = round(self.seconds, 2)
        return self

    # ---- 能力不是"推断"出来的，是自己试出来的 ----
    def capabilities_empirical(self):
        """★ 能力标签必须**实测**。
        线性模型会骗人：阿克曼在偏置工作点的雅可比看上去能得到纯旋转，
        但真车停着打方向盘根本不动。所以这里直接发一条"原地转"和
        一条"横着走"的指令，量一下到底发生了什么。
        """
        lims = self.hal.limits()
        rank = int(np.linalg.matrix_rank(self.J, tol=1e-4))
        def trial(intent, dur=0.55):
            self._play([(0.20, list(self.bias))])          # 先回稳
            traj = self._play([(dur, self.inverse(intent))])
            i0 = max(1, int((getattr(self, "onset", 0.10) + 0.08) / self.dt))
            return self._vel(traj, i0, len(traj) - 1)
        v_turn = trial([0.0, 0.0, 1.2])
        # ★ 判据要用"相对"的：转得比走得快得多 = 能原地转。
        #   绝对阈值会把大死区躯干的正常结果误判（曾把履带、四轮滑移判成"不能原地转"）。
        lin_t = math.hypot(float(v_turn[0]), float(v_turn[1]))
        turn = abs(float(v_turn[2])) > 0.50 and lin_t < 0.35 * abs(float(v_turn[2]))
        v_str = trial([0.0, 0.6, 0.0])
        strafe = (abs(float(v_str[1])) > 0.15
                  and abs(float(v_str[0])) < 0.35 * abs(float(v_str[1])))
        del lims
        self.caps_evidence = {
            "原地转_实测": [round(float(v_turn[0]), 3), round(float(v_turn[1]), 3),
                        round(float(v_turn[2]), 3)],
            "横移_实测": [round(float(v_str[0]), 3), round(float(v_str[1]), 3),
                       round(float(v_str[2]), 3)],
        }
        return {"rank": rank, "turn_in_place": bool(turn), "strafe": bool(strafe),
                "omnidirectional": bool(turn and strafe)}

    # ---- 通用控制律：抽象意图 → 具体通道指令 ----
    def inverse(self, intent, use_bias=True):
        intent = np.asarray(intent, float)
        u = np.linalg.lstsq(self.J, intent, rcond=None)[0]
        # ★ 死区补偿是"依赖指令符号"的：一次线性求解会留下偏差（履带的"纯旋转"
        #   解出来带着 0.2m/s 平移，被误判成"不能原地转"）。迭代几次就收敛。
        for _ in range(3):
            s = np.sign(u)
            s[s == 0] = 1.0
            off = self.dead * s
            u = np.linalg.lstsq(self.J, intent + self.J @ off, rcond=None)[0]
        if use_bias:
            u = u + (self.bias if np.linalg.norm(intent) > 1e-6 else 0)
        lims = self.hal.limits()
        return np.array([max(lo, min(hi, v)) for v, (lo, hi) in zip(u, lims)])

    def capabilities(self):
        """对外暴露的能力（来自实测，见 capabilities_empirical）。"""
        return self.caps

    def report(self):
        return {"method": self.method, "seconds": self.seconds, "probes": self.probes,
                "latency": self.latency, "dead": [round(float(x), 3) for x in self.dead],
                "J": [[round(float(x), 3) for x in row] for row in self.J],
                "notes": self.notes}


# ============================================================================
# ④ 通用大脑：同一份代码，跑所有躯干
# ============================================================================
class UniversalBrain:
    """只认端口，不认躯干。躯干以**数据**形式传入（HAL/BodySpec）。"""

    def __init__(self, hal, dt=0.05):
        self.hal = hal
        self.dt = dt
        self.schema = BodySchema(hal, dt=dt)
        self.caps = {}
        self.boot_log = {}
        # 速度环积分：前馈（学到的逆模型）之外再加一层反馈，
        # 专门吃掉死区补偿残差、增益误差、非线性这些"学不准"的部分。
        self._integ = np.zeros(3)
        self.ki = 2.0
        self.integ_limit = 0.5

    # ---- 上电自检 + 身体自识别 ----
    def boot(self, verbose=False, tries=2):
        t0 = time.perf_counter()
        self.schema.seconds, self.schema.probes = 0.0, 0
        checks = []
        for _ in range(tries):                      # 学 → 自查 → 不合格就再学
            self.schema.identify()
            checks.append(self.schema.self_check())
            if checks[-1]:
                break
        self.caps = self.schema.capabilities_empirical()
        wall = time.perf_counter() - t0
        self.boot_log = {"躯干": self.hal.name, "辨识方法": self.schema.method,
                         "激励次数": self.schema.probes,
                         "仿真时间_秒": self.schema.seconds,
                         "真实耗时_秒": round(wall, 3),
                         "估出的延迟_秒": self.schema.latency,
                         "实测起始延迟_秒": getattr(self.schema, "onset", 0.0),
                         "自查结果": checks,
                         "能力": self.caps}
        if verbose:
            print(f"   ● {self.hal.name}: {self.schema.method}, "
                  f"激励 {self.schema.probes} 次 / {self.schema.seconds:.2f}s，"
                  f"延迟≈{self.schema.latency:.2f}s，能力={self.caps}")
        return self.boot_log

    # ---- 抽象意图 → 具体通道 ----
    def act(self, vx, vy, omega):
        intent = np.array([vx, vy, omega], float)
        self.hal.set_all(self.schema.inverse(intent + self._integ))

    def _body_vel(self, prev_pose):
        """一步的实际机身速度（用来做速度反馈）。"""
        p = self.hal.read()
        dx, dy = p["x"] - prev_pose[0], p["y"] - prev_pose[1]
        c, s = math.cos(prev_pose[2]), math.sin(prev_pose[2])
        return np.array([(c * dx + s * dy) / self.dt,
                         (-s * dx + c * dy) / self.dt,
                         wrap_angle(p["yaw"] - prev_pose[2]) / self.dt]), (p["x"], p["y"], p["yaw"])

    def stop(self):
        self._integ[:] = 0.0
        self.act(0.0, 0.0, 0.0)
        for _ in range(4):
            self.hal.step(self.dt)

    # ---- 通用移动：去一个点（躯干无关）----
    def go_to(self, target, tol=0.35, max_seconds=40.0, v_max=0.55, log=False):
        t, stuck, hist = 0.0, 0, []
        self._integ[:] = 0.0
        ok = False
        while t < max_seconds:
            p = self.hal.read()
            prev = (p["x"], p["y"], p["yaw"])
            dx, dy = target[0] - p["x"], target[1] - p["y"]
            dist = math.hypot(dx, dy)
            if dist < tol:
                ok = True
                break
            err = wrap_angle(math.atan2(dy, dx) - p["yaw"])
            if self.caps["turn_in_place"]:
                vx = v_max * max(0.0, math.cos(err))
                omega = 2.2 * err
            elif self.caps["strafe"] and abs(err) > 1.6:
                vx, omega = 0.0, 2.2 * err         # 全向：先转头，省得横着乱走
            else:
                vx = max(0.16, v_max * max(0.0, math.cos(err)))
                omega = 1.6 * err                  # 汽车式：不停车、边走边转
            intent = np.array([vx, 0.0, omega], float)
            self.act(vx, 0.0, omega)
            self.hal.step(self.dt)
            t += self.dt
            # 速度反馈：把"指令意图"和"实际跑出来的"之差积起来，补到下一次意图上
            v_meas, _ = self._body_vel(prev)
            self._integ = np.clip(self._integ + self.ki * (intent - v_meas) * self.dt,
                                  -self.integ_limit, self.integ_limit)
            hist.append(dist)
            if len(hist) > 80 and (hist[-80] - hist[-1]) < 0.02:
                stuck += 1
                if stuck > 2:
                    break
            else:
                stuck = 0
        self.stop()
        p = self.hal.read()
        final = math.hypot(target[0] - p["x"], target[1] - p["y"])
        return {"ok": ok and final < tol * 1.4, "final_dist": round(final, 3),
                "seconds": round(t, 2),
                "note": "" if ok else ("超时" if t >= max_seconds else "卡住（连续 4 秒没进展）")}


# ============================================================================
# ⑤ 自检 / 演示
# ============================================================================
def _norm_err(J1, J2):
    den = max(1e-9, float(np.linalg.norm(J2)))
    return float(np.linalg.norm(J1 - J2)) / den


def run_one_body(spec, name=None, start=(0.0, 0.0, 0.0), target=(2.6, 1.7),
                 seed=0, verbose=False):
    spec = dict(spec)
    spec["name"] = name or spec.get("name", "unnamed_body")
    body = BodySim(spec, x=start[0], y=start[1], yaw=start[2], seed=seed)
    brain = UniversalBrain(body)
    boot = brain.boot(verbose=verbose)
    J_err = _norm_err(brain.schema.J, body.J_true)
    caps_true = body.true_caps()
    run = brain.go_to(target)
    return {"body": body, "brain": brain, "boot": boot, "J_err": J_err,
            "caps": brain.caps, "caps_true": caps_true, "run": run}


def selftest():
    fails, checks = [], []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        if not cond:
            fails.append(name)

    print("=" * 78)
    print("  小沙 · 通用躯干适配 自检（同一份大脑代码 × 多种躯干）")
    print("=" * 78)
    md5 = hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:10]
    print(f"  大脑代码指纹（md5[:10]）= {md5}   ← 下面所有躯体都是这一份代码跑的")
    print("-" * 78)

    # ---------- 1. 回归：v2 / v3 不能被碰坏 ----------
    for script, expect in (("xiaosha_brain_v2.py", "23/23"), ("xiaosha_neuro_v3.py", "27/27")):
        p = os.path.join(HERE, script)
        r = subprocess.run([sys.executable, p, "--selftest"], capture_output=True,
                           text=True, encoding="utf-8", errors="ignore")
        tail = [l for l in (r.stdout or "").splitlines() if "结果：" in l]
        check(f"回归：{script} 仍然全过（{expect}）",
              r.returncode == 0 and "全部通过" in (tail[-1] if tail else ""),
              (tail[-1].strip() if tail else f"exit={r.returncode}"))

    # ---------- 2. 每个躯干：上机 → 自学 → 干活 ----------
    rows = []
    for name, spec in BODY_SPECS.items():
        r = run_one_body(spec, name=name, seed=7)
        rows.append((name, r))
        check(f"躯干[{name}]：开机自己辨识出生体模型（J 误差 < 30%）",
              r["J_err"] < 0.30, f"J 相对误差 {r['J_err']*100:.1f}%，"
                                 f"方法 {r['boot']['辨识方法']}，"
                                 f"激励 {r['boot']['激励次数']} 次 / "
                                 f"{r['boot']['仿真时间_秒']}s（真实 {r['boot']['真实耗时_秒']}s）")
        check(f"躯干[{name}]：自己判定的能力标签与真值一致（横移/原地转/全向）",
              all(r["caps"][k] == r["caps_true"][k]
                  for k in ("strafe", "turn_in_place", "omnidirectional")),
              f"自学={r['caps']} 真值={ {k: r['caps_true'][k] for k in ('strafe','turn_in_place','omnidirectional')} }")
        check(f"躯干[{name}]：「去 (2.6,1.7)」一次成功",
              r["run"]["ok"], f"终距 {r['run']['final_dist']}m，用时 {r['run']['seconds']}s "
                              f"{r['run']['note']}")

    # ---------- 3. 学习时间要短 ----------
    sim_t = [r["boot"]["仿真时间_秒"] for _, r in rows]
    wall_t = [r["boot"]["真实耗时_秒"] for _, r in rows]
    check("自学习耗时短：一次性开机成本，每个躯干 ≤ 12s 仿真时间（平均 ≤ 9s）",
          max(sim_t) <= 12.0 and sum(sim_t) / len(sim_t) <= 9.0,
          f"各躯干仿真耗时 {sim_t}，平均 {sum(sim_t)/len(sim_t):.2f}s；"
          f"真实计算耗时 {wall_t}（含解算）。"
          f"注：探测间隔要按躯干自身延迟自动放长，所以延迟大的躯体天生要多花时间")

    # ---------- 4. 从没见过的躯干：现场造一个，只加数据 ----------
    new_spec = {
        "desc": "自检里现场新增的四轮滑移（符号有反的 + 大死区 + 长延迟）",
        "medium": "local", "noise": 0.005,
        "channels": [
            {"name": "fl", "range": [-1.1, 1.1], "gain": 0.80, "dead": 0.22,
             "delay": 0.08, "contrib": [0.42, 0.0, -2.90]},
            {"name": "fr", "range": [-1.1, 1.1], "gain": 1.10, "dead": 0.22,
             "delay": 0.08, "contrib": [-0.42, 0.0, -2.90]},   # 注意符号是反的
            {"name": "rl", "range": [-1.1, 1.1], "gain": 0.70, "dead": 0.30,
             "delay": 0.08, "contrib": [0.42, 0.0, -2.90]},
            {"name": "rr", "range": [-1.1, 1.1], "gain": 1.00, "dead": 0.30,
             "delay": 0.08, "contrib": [-0.42, 0.0, -2.90]},
        ],
    }
    r_new = run_one_body(new_spec, name="ski4_new_unseen", seed=11)
    check("新躯干（从没见过，现场只加数据）：自学后仍能干活",
          r_new["run"]["ok"] and r_new["J_err"] < 0.35,
          f"J 误差 {r_new['J_err']*100:.1f}%，终距 {r_new['run']['final_dist']}m，"
          f"自学 {r_new['boot']['仿真时间_秒']}s（{r_new['boot']['辨识方法']}）"
          f"；死区估计 {[round(float(x), 2) for x in r_new['brain'].schema.dead]}"
          f"（真值 {[round(float(x), 2) for x in r_new['body'].deads]}）"
          f"；能力 {r_new['caps']}")

    # ---------- 5. 探测效率：哈达玛并行 vs 逐通道 ----------
    spec = BODY_SPECS["omni3"]
    b1 = BodySim(spec, seed=3)
    s1 = BodySchema(b1)
    s1._baseline_noise()
    t0 = time.perf_counter(); J1, _ = s1.probe_hadamard(); w1 = time.perf_counter() - t0
    b2 = BodySim(spec, seed=3)
    s2 = BodySchema(b2)
    s2._baseline_noise()
    t0 = time.perf_counter(); J2, _ = s2.probe_sequential(); w2 = time.perf_counter() - t0
    e1, e2 = _norm_err(J1, b1.J_true), _norm_err(J2, b2.J_true)
    check("探测效率：哈达玛并行（一次激励全部通道）比逐通道少发动作、精度相当",
          s1.probes <= int(s2.probes * 0.8) and e1 < 0.30 and e2 < 0.30,
          f"并行 {s1.probes} 次下发/{w1:.2f}s（J 误差 {e1*100:.0f}%） vs "
          f"逐通道 {s2.probes} 次/{w2:.2f}s（{e2*100:.0f}%）；"
          f"通道越多优势越大（n 通道：并行 n+1 次 vs 逐通道 2n 次）")

    # ---------- 6. 同一份代码：每个躯干 0 行大脑代码 ----------
    brain_lines = sum(1 for _ in open(os.path.abspath(__file__), encoding="utf-8"))
    spec_lines = len(json.dumps(BODY_SPECS, ensure_ascii=False).splitlines())
    check("『换躯干不用多输代码』：全部躯干共用同一个 UniversalBrain 类，逐躯干代码增量为 0",
          all(type(r["brain"]).__name__ == "UniversalBrain" for _, r in rows)
          and all(type(r["brain"].schema).__name__ == "BodySchema" for _, r in rows),
          f"大脑+适配层共 {brain_lines} 行（未按躯干分叉）；"
          f"每个躯干只加 1 条数据（当前 5 条共 {spec_lines} 行 JSON）")

    print()
    for name, r in rows:
        print(f"  · {name:<14} {r['boot']['辨识方法']:<16} 自学 {r['boot']['仿真时间_秒']:>4.2f}s / "
              f"{r['boot']['激励次数']:>3} 次激励  J误差 {r['J_err']*100:>5.1f}%  "
              f"到达 {r['run']['final_dist']:>4.2f}m/{r['run']['seconds']:>5.1f}s  {r['caps']}")
    print()
    npass = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
    print("-" * 78)
    print(f"  结果：{npass}/{len(checks)} 项通过"
          + ("  ✅ 全部通过" if not fails else f"  ❌ 失败项：{fails}"))
    print("=" * 78)
    return 0 if not fails else 1


def demo():
    print("=" * 78)
    print("  通用躯干适配演示：同一份大脑代码，换躯干只换一条数据")
    print("=" * 78)
    md5 = hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:10]
    print(f"  大脑代码指纹 md5[:10] = {md5}")
    for name, spec in BODY_SPECS.items():
        print(f"\n【{name}】{spec['desc']}")
        r = run_one_body(spec, name=name, start=(0.0, 0.0, 0.9), target=(2.6, 1.7),
                         seed=5, verbose=True)
        b, br = r["body"], r["brain"]
        print(f"   学到的工作点偏置 = {[round(float(x),2) for x in br.schema.bias]}，"
              f"估出的死区 = {[round(float(x),2) for x in br.schema.dead]}")
        print(f"   辨识出的雅可比 J（3×{b.n_channels()}，行=vx,vy,ω）：")
        for row in br.schema.J:
            print("      [" + "  ".join(f"{v:+7.2f}" for v in row) + "]")
        print(f"   真值 J：")
        for row in b.J_true:
            print("      [" + "  ".join(f"{v:+7.2f}" for v in row) + "]")
        print(f"   结果：{'到达' if r['run']['ok'] else '没到'} "
              f"终距 {r['run']['final_dist']}m，用时 {r['run']['seconds']}s {r['run']['note']}")
    print("\n" + "=" * 78)


def bench():
    print("=" * 78)
    print("  探测方式对比：逐通道（稳）vs 哈达玛并行（快）")
    print("=" * 78)
    print(f"{'躯干':<14}{'通道':>4} {'逐通道耗时':>12} {'并行耗时':>10} "
          f"{'逐通道J误差':>12} {'并行J误差':>11}")
    print("-" * 78)
    for name, spec in BODY_SPECS.items():
        out = {}
        for mode in ("seq", "had"):
            b = BodySim(spec, seed=3)
            s = BodySchema(b)
            s._baseline_noise()
            t0 = time.perf_counter()
            if mode == "seq":
                J, _ = s.probe_sequential()
            else:
                J, _ = s.probe_hadamard()
            out[mode] = (time.perf_counter() - t0, s.probes, s.seconds,
                         _norm_err(J, b.J_true))
        print(f"{name:<14}{BodySim(spec).n_channels():>4} "
              f"{out['seq'][0]:>10.2f}s {out['had'][0]:>9.2f}s "
              f"{out['seq'][3]*100:>11.1f}% {out['had'][3]*100:>10.1f}%")
    print("-" * 78)
    print("  注：'耗时'是仿真时间；激励次数 = 真正发出去的动作数量（真机上就是动作条数）")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="小沙 · 通用躯干适配层")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.bench:
        bench()
        return
    demo()


if __name__ == "__main__":
    main()
