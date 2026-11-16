# -*- coding: utf-8 -*-
"""brain_live.py 的测试 —— 之前这个服务脚本完全没有测试覆盖。

为什么专门测它：它是常驻服务，跑了 38.7 小时才被发现满核空转。
服务脚本最需要的不是"功能对"，而是"不会偷偷烧资源"，所以这里测的是
**速率上限**和**循环里有 sleep**，而不是"仿真结果对不对"。
"""
import os
import re
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "brain_live.py")
PY = os.path.join(sys.prefix, "Scripts", "python.exe") if os.name == "nt" \
    else os.path.join(sys.prefix, "bin", "python")


def test_has_rate_cap():
    """必须有显式速率上限常量，且值在合理区间（不是满速）。"""
    src = open(SRC, encoding="utf-8").read()
    m = re.search(r"^\s*TARGET_HZ\s*=\s*([\d.]+)", src, re.M)
    assert m, "brain_live.py 缺少 TARGET_HZ 常量 —— 无上限的仿真循环会烧满一个核"
    hz = float(m.group(1))
    assert 1.0 <= hz <= 120.0, f"TARGET_HZ={hz} 不合理（直播只需 30）"


def test_loop_sleeps():
    """主循环必须有 sleep —— 光有常量不 sleep 等于没限速。"""
    src = open(SRC, encoding="utf-8").read()
    assert "time.sleep(" in src, "brain_live.py 主循环没有 sleep，会满核空转"
    # sleep 必须在 while True 之后（不能在死代码里）
    loop = src.find("while True:")
    assert loop > 0, "找不到主循环"
    assert src.find("time.sleep(", loop) > loop, "sleep 不在主循环内"


def test_throttled_runtime():
    """实测：跑 6 秒，帧率必须接近 TARGET_HZ 而不是满速（137 步/秒）。"""
    if not os.path.exists(PY):
        pytest.skip("找不到解释器")
    src = open(SRC, encoding="utf-8").read()
    m = re.search(r"^\s*TARGET_HZ\s*=\s*([\d.]+)", src, re.M)
    target = float(m.group(1)) if m else 30.0

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.Popen([PY, "-X", "utf8", SRC], cwd=HERE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, encoding="utf-8", errors="replace", env=env)
    t0 = time.time()
    frames = []
    try:
        while time.time() - t0 < 9:
            line = p.stdout.readline()
            if not line:
                break
            if '"t":' in line:
                frames.append(time.time() - t0)
            if len(frames) >= 6:
                break
    finally:
        p.kill()
        p.wait(timeout=10)

    assert len(frames) >= 3, f"9 秒内只收到 {len(frames)} 帧（服务没起来？）"
    # 用第 1 到第 3+ 帧的时间差算速率（跳过构造期）
    span = frames[-1] - frames[0]
    n = len(frames) - 1
    assert span > 0, "帧时间戳异常"
    rate = n / span
    # 上限：不超过 target 的 1.5 倍（sleep 精度/调度抖动留余量）
    # 若没限速，288,964 规模下会到 ~137 步/秒，直接超过这个阈值
    assert rate <= target * 1.5, \
        f"实测 {rate:.1f} 步/秒，超过 {target}*1.5 —— 限速没生效"
    # 下限：也不能慢到不可用
    assert rate >= target * 0.3, f"实测只有 {rate:.1f} 步/秒，太慢了（直播会卡）"
