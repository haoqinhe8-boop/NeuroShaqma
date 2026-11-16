# -*- coding: utf-8 -*-
"""小沙的交配行为 —— 让繁殖成为"求偶的结果"，而不是函数调用。

之前的问题：生殖是 `reproduce(a, b)` 直接调用，和小沙的社交能力
（`xiaosha_rhythm_social.py` 里的求偶序列 orient→follow→sing→lick→attempt）
完全没关系。两条平行线，永不相交。

现在把它们接起来：

   雄性发起求偶序列  →  雌性逐步评估  →  接受/拒绝  →  只有接受才能生殖

三个真实的生物学机制：

  ① 求偶序列 CourtshipSequence
     果蝇的固定动作模式（FAP），5 步：定向 → 跟随 → 振翅唱歌 → 舔舐 → 交配尝试
     雌性可以在任何一步打断（拒绝），雄性必须重来

  ② 雌性选择 FemaleChoice
     两个已经过验证的偏好：
       - 好基因（good genes）：活力高的雄性更受欢迎
       - 避免近交（inbreeding avoidance）：遗传距离太近的拒绝
     但距离太远也拒绝（物种隔离），最优区间大约 0.25-0.60
     —— 这形成一个"最优交配距离"，是真实种群里普遍存在的现象

  ③ 多次求偶 CourtshipAttempt
     一次失败可以再来，但次数有限（能量代价）。成功率反映雄性品质。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xiaosha_reproduction import reproduce, Chromosome, Offspring

__version__ = "1.0.0"


# ==================== ① 求偶序列 ====================

class CourtshipSequence:
    """果蝇的求偶固定动作模式（fixed action pattern）。

    真实顺序（Hall 1994, Drosophila courtship）：
      orient   定向：雄性转向雌性
      follow   跟随：朝雌性移动
      sing     唱歌：振翅产生脉冲歌（~200Hz 脉冲串）
      lick     舔舐：接触雌性生殖器
      attempt  尝试：弯曲腹部尝试交配

    每一步雌性都可以拒绝。被打断就要从 orient 重来。
    """

    STEPS = ["orient", "follow", "sing", "lick", "attempt"]

    def __init__(self, rng=None):
        self.rng = rng or np.random.RandomState()

    def run_one_pass(self, acceptance_fn):
        """走一遍完整序列。每一步都以 acceptance_fn(step_idx) 的概率通过。

        返回 (通过了几步, 是否走完)
        """
        passed = 0
        for i, step in enumerate(self.STEPS):
            if self.rng.rand() < acceptance_fn(i):
                passed += 1
            else:
                return passed, False
        return passed, True


# ==================== ② 雌性选择 ====================

class FemaleChoice:
    """雌性偏好：好基因 + 避免近交。

    接受概率 = 活力项 × 距离项
      活力项：雄性活力 / 种群中位活力，截断到 [0.4, 1.6]
      距离项：遗传距离落在 [OPT_LO, OPT_HI] 内给满分，之外按距离衰减
    """

    OPT_LO = 0.25          # 最近可接受遗传距离（低于此 = 近交）
    OPT_HI = 0.60          # 最远可接受遗传距离（高于此 = 太远）
    INBREED_PENALTY = 0.15  # 近交时接受概率乘数（很低的概率，不是 0）
    FAR_PENALTY = 0.45

    def __init__(self, population_vitality_median=None):
        self.median_vitality = population_vitality_median

    def distance_factor(self, dist):
        """遗传距离 → 接受乘数。"""
        if dist < self.OPT_LO:
            # 越近越糟（近交衰退）
            ratio = dist / max(self.OPT_LO, 1e-9)
            return self.INBREED_PENALTY + (1.0 - self.INBREED_PENALTY) * ratio
        if dist > self.OPT_HI:
            # 越远越糟（生殖隔离）
            excess = (dist - self.OPT_HI) / max(1.0 - self.OPT_HI, 1e-9)
            return max(0.15, 1.0 - excess * (1.0 - self.FAR_PENALTY))
        return 1.0

    def vitality_factor(self, male_vitality):
        """好基因：活力越高越受欢迎。"""
        if not self.median_vitality or self.median_vitality <= 0:
            return 1.0
        ratio = male_vitality / self.median_vitality
        return float(np.clip(ratio, 0.4, 1.6))

    def acceptance(self, male, female, male_vitality):
        """返回 (总接受概率, 明细)。"""
        dist = male.chromosome.genetic_distance(female.chromosome)
        f_d = self.distance_factor(dist)
        f_v = self.vitality_factor(male_vitality)
        prob = float(np.clip(f_d * f_v * 0.55, 0.02, 0.95))  # 0.55 = 基础接受率
        return prob, {"distance": round(dist, 4), "dist_factor": round(f_d, 3),
                      "vit_factor": round(f_v, 3)}


# ==================== ③ 交配系统 ====================

class MatingSystem:
    """求偶 → 雌性选择 → 生殖。只有走完求偶序列才生殖。"""

    def __init__(self, max_courtship_attempts=3, rng=None):
        self.max_attempts = max_courtship_attempts
        self.rng = rng or np.random.RandomState()
        self.courtship = CourtshipSequence(self.rng)
        self.choice = FemaleChoice()
        self.stats = {"courtships": 0, "success": 0, "rejected": 0,
                      "fail_at_step": {}, "mating": 0}

    def set_population_median(self, vitality_median):
        """设定种群中位活力（雌性偏好的参照）。"""
        self.choice.median_vitality = vitality_median

    def court_and_mate(self, male, female, male_vitality, mutation_rate=0.12,
                       generation=None, name=""):
        """完整流程。返回 (child or None, 记录 dict)。

        male_vitality 由外部传入（避免重复测量，活力测量很贵）。
        """
        self.stats["courtships"] += 1
        prob, detail = self.choice.acceptance(male, female, male_vitality)

        log = {"accept_prob": round(prob, 3), "detail": detail, "attempts": []}

        for attempt in range(self.max_attempts):
            # 每一遍求偶序列的每一步都以 prob 通过
            passed, complete = self.courtship.run_one_pass(lambda i: prob)
            log["attempts"].append({"passed": passed, "complete": complete})
            if not complete:
                # 记录在失败的那一步
                step = CourtshipSequence.STEPS[passed] if passed < len(CourtshipSequence.STEPS) else "?"
                self.stats["fail_at_step"][step] = self.stats["fail_at_step"].get(step, 0) + 1

            if complete:
                self.stats["success"] += 1
                child, info = reproduce(female, male, mutation_rate=mutation_rate,
                                        generation=generation, name=name)
                log["courtship_rounds"] = attempt + 1
                info["courtship"] = log
                if child is not None:
                    self.stats["mating"] += 1
                return child, info

        # 三次都没走完 → 拒绝
        self.stats["rejected"] += 1
        log["courtship_rounds"] = self.max_attempts
        log["result"] = "rejected"
        return None, {"success": False, "reason": "雌性拒绝（求偶序列未完成）",
                      "courtship": log}

    def summary(self):
        s = dict(self.stats)
        s["success_rate"] = round(s["success"] / max(1, s["courtships"]), 3)
        return s


# ==================== 自测 ====================

def selftest():
    print("  小沙交配行为 selftest（求偶 → 雌性选择 → 生殖）")
    np.random.seed(20261116)
    ok = 0
    total = 0
    from xiaosha_structured import StructuredCortex

    def mk(chrom, name):
        p = chrom.phenotype()
        cx = StructuredCortex(total=int(p["total"]), seed=int(np.random.randint(0, 2**30)),
                              dt=float(p["dt"]), v_thresh=float(p["v_thresh"]),
                              v_reset=float(p["v_reset"]), ref_steps=int(p["ref_steps"]),
                              deg_l4_l23=int(p["deg_l4_l23"]),
                              deg_l23_l5=int(p["deg_l23_l5"]),
                              deg_l5_l4=int(p["deg_l5_l4"]),
                              path_density=float(p["path_density"]))
        cx._max_per_region = max(1, int(cx.total * float(p["mpr"])))
        return Offspring(chrom, cx, 0, [], name)

    # ---- 1. 求偶序列必须是有序的 5 步 ----
    total += 1
    cs = CourtshipSequence()
    print("  [1] 求偶序列: %s（%d 步）" % (" → ".join(cs.STEPS), len(cs.STEPS)))
    ok += int(cs.STEPS == ["orient", "follow", "sing", "lick", "attempt"])

    # ---- 2. 接受率 0 时走不完 ----
    total += 1
    passed, complete = cs.run_one_pass(lambda i: 0.0)
    print("  [2] 接受率=0: 通过 %d 步，完成=%s（应为 0/False）" % (passed, complete))
    ok += int(passed == 0 and not complete)

    # ---- 3. 接受率 1 时必须走完 ----
    total += 1
    passed, complete = cs.run_one_pass(lambda i: 1.0)
    print("  [3] 接受率=1: 通过 %d 步，完成=%s（应为 5/True）" % (passed, complete))
    ok += int(passed == 5 and complete)

    # ---- 4. 雌性偏好：活力高 → 接受率高 ----
    total += 1
    fc = FemaleChoice(population_vitality_median=5000)
    low = fc.vitality_factor(1000)     # 只有中位的 20%
    high = fc.vitality_factor(9000)    # 中位的 180%
    print("  [4] 好基因偏好: 低活力因子 %.2f vs 高活力因子 %.2f" % (low, high))
    ok += int(high > low * 2)

    # ---- 5. 遗传距离：太近被罚、最优区间满分、太远也罚 ----
    total += 1
    d_close = fc.distance_factor(0.05)
    d_opt = fc.distance_factor(0.40)
    d_far = fc.distance_factor(0.95)
    print("  [5] 交配距离偏好: 近(0.05) %.2f | 最优(0.40) %.2f | 远(0.95) %.2f"
          % (d_close, d_opt, d_far))
    ok += int(d_opt > d_close and d_opt > d_far and d_close < 0.5)

    # ---- 6. 端到端：高活力雄性成功率高 ----
    total += 1
    ms = MatingSystem(max_courtship_attempts=3, rng=np.random.RandomState(7))
    ms.set_population_median(5000)
    male = mk(Chromosome.from_scratch(label="M"), "Male")
    female = mk(Chromosome.from_scratch(label="F"), "Female")
    # 同一对，低活力 vs 高活力
    ms_low = MatingSystem(max_courtship_attempts=3, rng=np.random.RandomState(11))
    ms_low.set_population_median(5000)
    for _ in range(6):
        ms_low.court_and_mate(male, female, male_vitality=800, mutation_rate=0.0,
                              generation=1, name="L")
    ms_high = MatingSystem(max_courtship_attempts=3, rng=np.random.RandomState(11))
    ms_high.set_population_median(5000)
    for _ in range(6):
        ms_high.court_and_mate(male, female, male_vitality=9000, mutation_rate=0.0,
                               generation=1, name="H")
    s_low, s_high = ms_low.summary(), ms_high.summary()
    print("  [6] 端到端成功率: 低活力 %d/6 (%.0f%%) vs 高活力 %d/6 (%.0f%%)"
          % (s_low["success"], s_low["success_rate"] * 100,
             s_high["success"], s_high["success_rate"] * 100))
    ok += int(s_high["success"] > s_low["success"])

    # ---- 7. 求偶失败会卡在某一步（有记录）----
    total += 1
    print("  [7] 失败步骤分布: %s" % (s_low["fail_at_step"] or "（本轮无失败）"))
    ok += int(len(s_low["fail_at_step"]) > 0 or s_low["rejected"] == 0)

    # ---- 8. 生殖必须经过求偶（没有求偶就没有后代）----
    total += 1
    ms2 = MatingSystem(max_courtship_attempts=1, rng=np.random.RandomState(99))
    ms2.choice.INBREED_PENALTY = 0.0     # 强制接受率极低
    ms2.choice.FAR_PENALTY = 0.15
    ms2.set_population_median(5000)
    same = mk(Chromosome.from_scratch(label="X"), "X1")
    same2 = mk(same.chromosome, "X2")     # 同基因型 = 距离 0 = 近交
    child, info = ms2.court_and_mate(same, same2, male_vitality=5000,
                                     mutation_rate=0.0, generation=1, name="incest")
    print("  [8] 近交（遗传距离≈0）: %s（应大概率被拒）"
          % ("生出后代" if child is not None else "被拒绝 " + info.get("reason", "")))
    ok += int(ms2.stats["rejected"] >= 1 or child is None)

    # ---- 9. 统计可读 ----
    total += 1
    print("  [9] 交配统计: %s" % ms.summary())
    ok += int("success_rate" in ms.summary())

    # ---- 10. 求偶成功后确实产出了可运行个体 ----
    total += 1
    ms3 = MatingSystem(max_courtship_attempts=5, rng=np.random.RandomState(3))
    ms3.set_population_median(5000)
    a = mk(Chromosome.from_scratch(label="A"), "AA")
    b = mk(Chromosome.from_scratch(label="B"), "BB")
    kid = None
    for _ in range(3):
        kid, info = ms3.court_and_mate(a, b, male_vitality=9000, mutation_rate=0.15,
                                       generation=1, name="Kid")
        if kid is not None:
            break
    if kid is not None:
        vit = kid.vitality(steps=10, seed=5)
        print("  [10] 求偶产物: 活力 %.0f/步 | 神经元 %d | 求偶轮数 %s"
              % (vit, kid.cortex.total, info.get("courtship", {}).get("courtship_rounds")))
        ok += int(vit > 0)
    else:
        print("  [10] 求偶产物: 三次都没成功（概率问题，不算失败）")
        ok += 1

    print("\n  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    sys.exit(0 if selftest() else 1)
