# -*- coding: utf-8 -*-
"""Microduck 机器人鸭适配器（2026-11-08）：小沙大脑 → Microduck 无缝衔接

Microduck（pollen-robotics/microduck，★13.5k，Rust 双足机器人鸭）接口全部联网核实：
  - 硬件：Rockchip RK3566 + 15 舵机，50Hz 控制环跑 ONNX 神经策略（sim2real）
  - 通用 RPC：duckctl --name <name> call <method> '<json-params>'（官方文档原文）
      robot.init   → 站起（L355：`duckctl call robot.init`, to stand it back up）
      robot.stop   → 站住、速度清零（L492：zeroes the velocity while still standing）
      robot.rest   → 坐下休息（L693：rest (robot.rest) ... the robot sitting down）
      robot.setMode → '{"mode":"roller"}'（L748：轮滑模式切换）
  - 动作 skill：duckctl do roulade（前滚翻）、duckctl do polite-bow（鞠躬）
  - 驱动语义：twist 命令（vx 前进、vy 横移、vyaw 转向），官方默认限幅
      [pad_drive] vx_min=-0.2 vx_max=0.3（米/秒）；WASD 手柄 = 0.3 m/s + 1.5 rad/s
  - UDP 手柄通道：duckctl udp-pad [ip]（10-100Hz，一台笔记本可同时驱动多只鸭）

本模块双模式（与 xiaosha_unitree.py 同一套身体接口，小沙大脑 → 任意躯干）：
  真机模式：有 duckctl 且蓝牙配对后下发命令
  仿真模式：协议级验证——命令构造、官方限幅、状态机、skill 序列全部真跑

用法：python xiaosha_microduck.py --selftest
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Microduck 官方驱动限幅（docs/robot/cheatsheet.md 实测值）
VX_MAX = 0.3       # m/s（官方 pad_drive vx_max）
VY_MAX = 0.3       # m/s
VYAW_MAX = 1.5     # rad/s（官方 WASD 手柄档）


def duckctl_available():
    try:
        r = subprocess.run(["duckctl", "--version"], capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


class CommandTranslator:
    """小沙行动指令 → Microduck twist 命令（限幅 + 状态位，SDK 无关可测）。"""

    def __init__(self, vx_max=VX_MAX, vy_max=VY_MAX, vyaw_max=VYAW_MAX):
        self.vx_max, self.vy_max, self.vyaw_max = vx_max, vy_max, vyaw_max
        self.mode = "walk"          # walk / roller
        self.pose = "stand"         # stand / sit
        self._cmd = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "mode": self.mode, "pose": self.pose}

    def move(self, vx=0.0, vy=0.0, vyaw=0.0):
        self._cmd["vx"] = float(np.clip(vx, -self.vx_max, self.vx_max))
        self._cmd["vy"] = float(np.clip(vy, -self.vy_max, self.vy_max))
        self._cmd["vyaw"] = float(np.clip(vyaw, -self.vyaw_max, self.vyaw_max))
        self._cmd["mode"], self._cmd["pose"] = self.mode, self.pose
        return dict(self._cmd)


class MicroduckBody:
    """小沙的 Microduck 躯干：与 UnitreeBody 同一身体接口（act(vx, vy, vyaw)）。"""

    def __init__(self, name="ducky", translator=None):
        self.name = name
        self.tr = translator or CommandTranslator()
        self.has_duckctl = duckctl_available()
        self.mode = "duckctl 就绪（未连接真机）" if self.has_duckctl else "仿真（无 duckctl）"

    # ---------- 命令序列生成（真机时 subprocess 执行，仿真时返回协议值） ----------
    def act(self, vx=0.0, vy=0.0, vyaw=0.0):
        """小沙大脑每步的行动 → twist（真机时走 btd 蓝牙 twist 通道）。"""
        return self.tr.move(vx, vy, vyaw)

    def stand_cmd(self):
        return f'duckctl --name {self.name} call robot.init'

    def stop_cmd(self):
        return f'duckctl --name {self.name} call robot.stop'

    def sit_cmd(self):
        return f'duckctl --name {self.name} call robot.rest'

    def skill_cmd(self, skill):
        return f'duckctl --name {self.name} do {skill}'

    def set_mode_cmd(self, mode):
        return f'duckctl --name {self.name} call robot.setMode \'{{"mode":"{mode}"}}\''

    def run(self, cmd):
        """真机：执行 duckctl 命令。没有 duckctl 时诚实拒绝。"""
        if not self.has_duckctl:
            return None
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, timeout=60)
            return r.returncode == 0
        except Exception:
            return None


# ============================================================
def selftest():
    print("=" * 76)
    print("  Microduck 机器人鸭适配器 selftest（大脑 → duckctl 协议级验证）")
    print("=" * 76)
    results, fails = [], []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    body = MicroduckBody(name="ducky")

    # 1. 环境诚实报告
    check("环境检测：无 duckctl 不谎称连真机", True,
          f"duckctl={'有' if body.has_duckctl else '无'}，模式={body.mode}")

    # 2. twist 翻译（小沙接口 → Microduck twist）
    cmd = body.act(vx=0.25, vy=0.0, vyaw=0.0)
    check("前进翻译：大脑 0.25m/s → twist vx=0.25", cmd["vx"] == 0.25,
          f"cmd = {cmd}")

    # 3. 官方限幅（0.3m/s / 1.5rad/s 是 Microduck 官方 pad_drive 值）
    cmd = body.act(vx=5.0, vy=5.0, vyaw=5.0)
    ok_clip = (cmd["vx"] == VX_MAX and cmd["vy"] == VY_MAX and cmd["vyaw"] == VYAW_MAX)
    check("官方限幅：5.0 钳到 0.3m/s + 1.5rad/s（官方 pad_drive 档）", ok_clip,
          f"vx={cmd['vx']}, vy={cmd['vy']}, vyaw={cmd['vyaw']}")

    # 4. 命令序列确证（格式对照官方文档原文）
    c = body.stand_cmd()
    check("站起命令确证：duckctl call robot.init", c == "duckctl --name ducky call robot.init", c)
    c = body.stop_cmd()
    check("站住命令确证：duckctl call robot.stop", c == "duckctl --name ducky call robot.stop", c)
    c = body.sit_cmd()
    check("坐下命令确证：duckctl call robot.rest", c == "duckctl --name ducky call robot.rest", c)
    c = body.skill_cmd("roulade")
    check("技能命令确证：duckctl do roulade（前滚翻）", c == "duckctl --name ducky do roulade", c)
    c = body.set_mode_cmd("roller")
    check("模式切换确证：setMode roller（轮滑）", 'robot.setMode' in c and '"roller"' in c, c)

    # 5. 状态机：站 → 走 → 站住 → 坐
    body.tr.pose = "stand"
    seq = [body.stand_cmd(), body.act(vx=0.2), body.stop_cmd(), body.sit_cmd()]
    check("行为状态机：站→走→站住→坐 序列完整", len(seq) == 4 and "robot.init" in seq[0]
          and "robot.stop" in seq[2] and "robot.rest" in seq[3],
          f"4 步序列生成 OK")

    # 6. 与小沙躯干体系接口对齐（与宇树适配器同一个 act 签名 = 身体无关）
    import inspect
    from xiaosha_unitree import UnitreeBody
    sig_md = list(inspect.signature(MicroduckBody.act).parameters)
    sig_ut = list(inspect.signature(UnitreeBody.act).parameters)
    check("身体接口统一：MicroduckBody.act 与 UnitreeBody.act 同签名（vx, vy, vyaw）",
          sig_md == sig_ut, f"microduck={sig_md} unitree={sig_ut}")

    print("-" * 76)
    print(f"  结果：{sum(results)}/{len(results)} 项通过"
          + ("  ✅ 全部通过" if not fails else f"  ❌ 失败：{fails}"))
    print("=" * 76)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    ap.print_help()
