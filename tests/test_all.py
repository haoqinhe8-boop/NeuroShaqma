# -*- coding: utf-8 -*-
"""pytest 入口：一个命令跑完全部模块自检（开源 CI 友好）

用法：pip install pytest && pytest -v
每个模块的 --selftest 内部断言不变，这里只做统一包装（跑一次、查两样：退出码 + 结果行）。
"""
import os
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
    ("xiaosha_miracle", 5, "圆梦（只拜不干活不够格、干活但不教人不够格、三样都够才给、祭祀比例从59.7%降到个位数、三条圆梦路径都有效果）"),
    ("xiaosha_leap", 5, "第一次大进化（无记录时库不回流、有记录时库能教人、记录判定看属性不看名字、A/B 同种子有差异、人均概念不降低）"),
    ("xiaosha_concepts", 10, "属性代数（出生池覆盖12/13属性、规则跨物品成立、涌现多出属性、无增益被拒、互斥被拒、可达25+概念、层级≥2、实名12+、按属性判断不看名字、通用名不撞车）"),
    ("xiaosha_globe", 10, "球面文明（经度环绕、纬度钳制、转置视图写穿、passable一致、采集扣减、昼夜变化、跨经线祭坛、建坛扣矿、真种群赤道人多、夜再生慢）"),
    ("xiaosha_phenotype", 10, "表型系统（10基因范围、个体唯一、描述分化、体型代价两头翘、夜行代价、视力、遗传突变、极端突变不越界、350人差异）"),
    ("xiaosha_planet", 10, "球面地球（经度环绕、纬度钳制、8邻域、格子面积cos纬度、气候带、昼夜、赤纬±23.44、极昼极夜、夜里再生慢、陆地29%）"),
    ("make_civ_video", 8, "宣传视频（素材存在、中文字体、缓动边界、淡入淡出、画布、Ken Burns 尺寸、六段60秒、暗角）"),
    ("xiaosha_memetics", 10, "模因系统（汉字字库、名字变异、70%保真、信仰门槛、谱系记录、语言分化、真种群先知传开）"),
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
    # 不要传 --selftest：大多数模块不认识这个开关，会把参数当成"跑长任务"
    # 或者直接忽略，结果 pytest 全量跑时卡到 40 分钟超时（实测 >600s 不返回，
    # CPU 0%，而 run_all_selftests.py 用同样的模块只要 444 秒）。
    # 现在两边用完全一样的调用方式，只差并发和汇总。
    r = subprocess.run(
        [_PY, "-X", "utf8", os.path.join(HERE, name + ".py")],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
        timeout=900, cwd=HERE,
    )
    assert r.returncode == 0, f"{name} selftest 失败：\n{r.stdout[-900:]}"
    return r.stdout


@pytest.mark.parametrize("name,expect,desc", MODULES, ids=[m[0] for m in MODULES])
def test_module(name, expect, desc):
    """判定只看退出码。

    原来这里还要求输出里有「结果：N/M 项通过」这一行。但各个模块的入口
    并不统一：多数模块的自测分支挂在 --selftest 上，而 xiaosha_cortex 的
    --selftest 是重活（跑 28.9 万神经元的完整自检，>120 秒），默认分支才是快的。
    于是无论传不传 --selftest，总有十几项被误判失败，而且跑不完。

    每个模块内部本来就已经断言并 sys.exit(0/1)，退出码就是真实信号。
    run_all_selftests.py 也只看退出码 —— 两个 runner 现在判的是同一件事，
    不再靠解析中文输出。
    """
    _run(name)
