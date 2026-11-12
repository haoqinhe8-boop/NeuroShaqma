# -*- coding: utf-8 -*-
"""pytest 入口：一个命令跑完全部模块自检（开源 CI 友好）

用法：pip install pytest && pytest -v
每个模块的 --selftest 内部断言不变，这里只做统一包装（跑一次、查两样：退出码 + 结果行）。
"""
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (模块, 预期自检项数, 备注)
MODULES = [
    ("xiaosha_spine", 4, "脊椎（神经管/节律/反射）"),
    ("xiaosha_immune", 4, "免疫（查 Bug/围堵）"),
    ("xiaosha_watchdog", 4, "看门狗（服务守护）"),
    ("xiaosha_organs", 6, "内脏八大系统"),
    ("xiaosha_body", 5, "身体五感（嘴巴说→耳朵听闭环）"),
    ("xiaosha_embodiment", 21, "躯干适配（6 种躯干）"),
    ("xiaosha_cortex", 10, "脉冲皮层（28.9 万×720 + 学习）"),
    ("xiaosha_regions", 7, "分区皮层（7 脑区路由）"),
    ("xiaosha_teamwork", 6, "脑区分工合作（跨感官任务+对照+消融）"),
    ("xiaosha_memory", 5, "工作记忆探测（如实记录：当前架构边界）"),
    ("xiaosha_multitask", 3, "多任务探测（如实记录：交替训练灾难性遗忘）"),
    ("xiaosha_hippocampus", 4, "海马体记忆重放（防灾难性遗忘，55%→82%）"),
    ("xiaosha_physiology", 16, "人体生理增强（13器官健康机制，排除衰老退化）"),
    ("xiaosha_tasks", 5, "多任务库（导航/分类/老虎机/XOR/6类）"),
    ("xiaosha_senses", 6, "视觉听觉增强（颜色/多目标/运动/音调/方位/音量）"),
    ("xiaosha_circuits", 8, "果蝇招牌回路（蘑菇体+中央复合体）"),
    ("xiaosha_gpu", 5, "GPU 运行器（fp16 混合精度；无卡自动 SKIP）"),
]


def _run(name):
    r = subprocess.run(
        [sys.executable, "-X", "utf8", os.path.join(HERE, name + ".py"), "--selftest"],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
        timeout=2400, cwd=HERE,
    )
    assert r.returncode == 0, f"{name} selftest 失败：\n{r.stdout[-900:]}"
    return r.stdout


@pytest.mark.parametrize("name,expect,desc", MODULES, ids=[m[0] for m in MODULES])
def test_module(name, expect, desc):
    out = _run(name)
    m = [l for l in out.splitlines() if "结果：" in l]
    assert m, f"{name} 无结果行"
    mm = re.search(r"(\d+)/(\d+)\s*项通过", m[-1])
    assert mm, f"{name} 结果行格式异常：{m[-1] if m else '无'}"
    assert int(mm.group(1)) == int(mm.group(2)), f"{name} 有失败项：{m[-1]}"
