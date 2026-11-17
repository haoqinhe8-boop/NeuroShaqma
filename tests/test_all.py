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
    ("xiaosha_neuromod", 4, "神经调质（DA学习率扫描: lr.0255→82%±0% 最稳, lr.066→61%崩）"),
    ("xiaosha_sleep", 4, "睡眠巩固（SHY: 负结果, A 55%→18% 如实记录）"),
    ("xiaosha_navigation", 10, "导航+步态（EPG头方向、路径整合正方形误差0.00、地标校正、6腿三足CPG同步1.00）"),
    ("xiaosha_chemosense", 11, "化学感官（味觉5类+饥饿调节+双相、联想学习单次100%、习惯化/敏感化、趋化导航）"),
    ("xiaosha_perception", 10, "四感前端（视网膜DoG+Gabor/V1方向、耳蜗频率+双耳定位、皮肤四类机械感受器、鼻子气味指纹）"),
    ("xiaosha_civilization", 8, "地球Online（地形世界、个体行动采集、父母教学传知识、技术组合涌现、部落形成、人口不灭绝不爆炸）"),
    ("xiaosha_ecosystem", 10, "生态系统（食物限制、衰老曲线、竞争取食、老死饿死、逻辑斯蒂承载力、种群不爆）"),
    ("xiaosha_mating", 10, "交配行为（求偶5步序列、雌性好基因偏好、最优交配距离、低活力0%成功率 vs 高活力100%、近交被拒）"),
    ("xiaosha_reproduction", 10, "染色体+生殖（二倍体9基因、减数分裂、受精、后代可独立运行、三代谱系、自然选择）"),
    ("xiaosha_rstdp", 5, "R-STDP 奖励学习（DA习惯化、正/负奖励方向、中性不学、28.9万46.6ms/步）"),
    ("xiaosha_stdp", 5, "STDP 可塑性（因果增强、无因果不变、稳态防爆、28.9万规模36.7ms/步）"),
    ("xiaosha_cognition", 15, "认知九项（数量感1-4、间隔计时、选择性注意、延迟满足、社会学习、群体决策、飞行200Hz、梳理/攀爬/跳跃、生病行为）"),
    ("xiaosha_depth", 10, "深度/偏振/湿度（运动视差、偏振光太阳罗盘、干湿感受器）"),
    ("xiaosha_embodied", 12, "具身闭环（感官→28.9万脉冲皮层→动作；无输入=0发放，消融V1发放-97%动作归零）"),
    ("xiaosha_rhythm_social", 13, "节律/社交/决策（Goodwin分子钟、睡眠剥夺反弹、求偶序列、风险规避）"),
    ("xiaosha_jit", 3, "numba JIT 加速内核（4.5倍提速：196→883 步/秒，行为不变）"),
    ("xiaosha_instinct", 10, "本能系统（零任务训练：7条硬连线本能+多感觉+一次记忆，200陌生场景200/200，0.09ms/步）"),
    ("xiaosha_structured", 7, "结构化皮层（6脑区×3层×E/I 细胞类型，5627万条规则突触，前馈级联 V1→V2→IT→PFC→M1）"),
    ("xiaosha_experiments", 5, "experiments/ 健康检查（9脚本语法/import/无副作用/main保护/INVALID标注）"),
    ("xiaosha_gated", 5, "门控多头多任务（A=100%/B=95%，对比共享单头 A78%/B45%）"),
    ("xiaosha_continual", 5, "持续学习（任务后锚定：λ=3 把旧任务30%→82%，但新任务受限——如实记录）"),
    ("xiaosha_force", 5, "FORCE工作记忆（DMS 48.7%±2.6% → 100%±0.0%，长延迟600ms仍100%）"),
    ("xiaosha_memory", 5, "工作记忆探测（如实记录：当前架构边界）"),
    ("xiaosha_multitask", 3, "多任务探测（如实记录：交替训练灾难性遗忘）"),
    ("xiaosha_hippocampus", 4, "海马体记忆重放（防灾难性遗忘，55%→82%）"),
    ("xiaosha_physiology", 16, "人体生理增强（13器官健康机制，排除衰老退化）"),
    ("xiaosha_tasks", 5, "多任务库（导航/分类/老虎机/XOR/6类）"),
    ("xiaosha_senses", 6, "视觉听觉增强（颜色/多目标/运动/音调/方位/音量）"),
    ("xiaosha_circuits", 8, "果蝇招牌回路（蘑菇体+中央复合体）"),
    ("xiaosha_gpu", 5, "GPU 运行器（fp16 混合精度；无卡自动 SKIP）"),
]


# 关键：Windows 上 venv 的 python.exe 是个 45KB 的重定向器，
# 在它下面 sys.executable 会解析成 **base python**（`pyvenv.cfg` 里的 home）。
# 这个 venv 的 base 是 uv 装的裸 CPython —— **没有 numpy**。
# 于是 test_module 起的子进程缺依赖，卡在 import/IO 上（CPU 0%、十分钟无输出，
# 而同样的模块单独跑只要 2.6 秒）。用 sys.prefix（真正的 venv 路径）拼解释器。
_PY = os.path.join(sys.prefix, "Scripts", "python.exe") if os.name == "nt" \
    else os.path.join(sys.prefix, "bin", "python")
if not os.path.exists(_PY):          # 兜底：非 venv 环境
    _PY = sys.executable


def _run(name):
    r = subprocess.run(
        [_PY, "-X", "utf8", os.path.join(HERE, name + ".py"), "--selftest"],
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
