# -*- coding: utf-8 -*-
"""多代人工选择实验 —— 看小沙能不能"进化"。

达尔文式（不是拉马克式）：不看父母学到了什么，只看**谁活得好、谁生得多**。

流程（每代）：
  1. 测量：每个个体的活力（固定场景下的发放量）
  2. 选择：活力最高的前 50% 活下来
  3. 繁殖：存活者两两配对，有性生殖 + 突变
  4. 淘汰：其余个体释放（模拟死亡）
  5. 记录：活力分布、遗传多样性、基因均值

要回答的问题：
  · 平均活力会上升吗？（选择真的有效？）
  · 遗传多样性会下降吗？（选择压力的痕迹）
  · 哪个基因在漂移？（自然选择选了哪个性状）

跑法：python experiments/exp_evolution.py [代数] [种群大小]
     默认 20 代 × 6 个体（约 6-8 分钟）

**诚实前提**：如果跑出来不上升，那是真实结果，不会去调参凑好看。
脉冲网络对初始状态敏感，单次活力测量噪声大，所以每个个体测 3 个 seed 取平均。
"""
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from xiaosha_reproduction import (
    Chromosome, Offspring, Ovary, Testis, Oviduct, Uterus, Pedigree,
    GENE_NAMES, GENES, random_genome, reproduce,
)
from xiaosha_structured import StructuredCortex
from xiaosha_mating import MatingSystem
from xiaosha_ecosystem import Ecosystem

SEEDS = (11, 23, 37)          # 活力测量的三个 seed（取平均降噪）


def make_founder(seed, name):
    """造一个奠基者个体。

    seed 只用于"这一次抽基因"的随机性，不写进基因组 ——
    具体连接对象由发育噪声决定，遗传的是 4 个结构基因（轴突数/密度）。
    """
    rng = np.random.RandomState(seed)
    chrom = Chromosome(random_genome(rng), random_genome(rng), label=name)
    p = chrom.phenotype()
    cx = StructuredCortex(total=int(p["total"]), seed=int(rng.randint(0, 2**30)),
                          dt=float(p["dt"]), v_thresh=float(p["v_thresh"]),
                          v_reset=float(p["v_reset"]), ref_steps=int(p["ref_steps"]),
                          deg_l4_l23=int(p["deg_l4_l23"]),
                          deg_l23_l5=int(p["deg_l23_l5"]),
                          deg_l5_l4=int(p["deg_l5_l4"]),
                          path_density=float(p["path_density"]))
    cx._max_per_region = max(1, int(cx.total * float(p["mpr"])))
    cx._max_n = max(1, int(cx.total * 0.15))
    return Offspring(chrom, cx, 0, [], name)


def measure(ind):
    """适应度：3 个 seed 的活力平均。"""
    return float(np.mean([ind.vitality(steps=10, seed=s) for s in SEEDS]))


def diversity(pop):
    """种群遗传多样性：两两遗传距离的平均。"""
    if len(pop) < 2:
        return 0.0
    ds = []
    for i in range(len(pop)):
        for j in range(i + 1, len(pop)):
            ds.append(pop[i].chromosome.genetic_distance(pop[j].chromosome))
    return float(np.mean(ds))


def gene_means(pop):
    """种群各基因的平均表现型。"""
    vals = np.array([ind.chromosome.express_values() for ind in pop])
    return vals.mean(axis=0)


def ascii_curve(values, width=60, height=12, label=""):
    """画 ASCII 曲线（终端里直接看趋势）。"""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return "  (所有值相同: %.2f)" % lo
    rows = []
    for r in range(height, -1, -1):
        thr = lo + (hi - lo) * r / height
        line = ""
        for v in values:
            lv = (v - lo) / (hi - lo) * height
            line += "█" if lv >= r else " "
        rows.append("  %8.1f │%s" % (thr, line))
    rows.append("           └" + "─" * len(values))
    rows.append("            " + "".join(str(i % 10) for i in range(len(values))))
    return "\n".join(rows)


def main(n_gen=20, pop_size=6, survival_ratio=0.5, mutation_rate=0.15, use_eco=True):
    t_start = time.time()
    print("=" * 74)
    print("小沙多代进化实验 · 人工选择（达尔文式）")
    print("=" * 74)
    print("  代数 %d | 种群 %d | 存活率 %.0f%% | 突变率 %.2f | 每代活力测 %d 个 seed"
          % (n_gen, pop_size, survival_ratio * 100, mutation_rate, len(SEEDS)))
    print()

    # ---------- 奠基者种群 ----------
    print("【第 0 代】建立奠基者种群...")
    pop = []
    for i in range(pop_size):
        pop.append(make_founder(1000 + i * 7, "F0_%d" % i))
        print("  %s  神经元 %6d  v_thresh %.2f  sense_gain %.1f"
              % (pop[-1].name, pop[-1].cortex.total,
                 pop[-1].phenotype()["v_thresh"], pop[-1].phenotype()["sense_gain"]))

    history = []
    uterus = Uterus()
    mating = MatingSystem(max_courtship_attempts=3)
    # 生态系统：食物**略紧**（再生只够养活 78% 的个体）→ 真有竞争和饿死。
    # 之前设 1.1 倍时食物永远够，生态筛选一个人都没杀过（实测），
    # 那样"资源限制"只是摆设。
    eco = Ecosystem(food_capacity=pop_size * 30 * 1.2,
                    food_per_agent=30, regen=pop_size * 30 * 0.78,
                    max_age=40, senescence_pow=1.5) if use_eco else None
    eco_log = []
    n_survivors = max(2, int(round(pop_size * survival_ratio)))

    for gen in range(n_gen):
        t_gen = time.time()
        # 1. 测量
        fits = [measure(ind) for ind in pop]
        order = np.argsort(fits)[::-1]
        div = diversity(pop)
        gm = gene_means(pop)
        rec = {
            "gen": gen,
            "max": float(max(fits)),
            "mean": float(np.mean(fits)),
            "min": float(min(fits)),
            "diversity": div,
            "gene_means": gm.copy(),
            "pop_size": len(pop),
        }
        history.append(rec)
        print("  第 %2d 代 | 最大 %7.0f | 平均 %7.0f | 最小 %7.0f | 多样性 %.3f | %d 个体 | %.0fs"
              % (gen, rec["max"], rec["mean"], rec["min"], div, len(pop), time.time() - t_gen))

        if gen == n_gen - 1:
            break

        # 2a. 生态筛选（如果开启）：衰老 + 饥饿 + 竞争，弱者真的死掉
        if eco is not None:
            vit_map = {ind.name: float(f) for ind, f in zip(pop, fits)}
            before = len(pop)
            alive, deaths = eco.step(pop, vit_map)
            causes = {}
            for _ind, cause, _p in deaths:
                causes[cause] = causes.get(cause, 0) + 1
            eco_log.append({"gen": gen, "before": before, "after": len(alive),
                            "causes": causes, "food": round(eco.food, 1)})
            if len(alive) < before:
                print("         生态: %d → %d 存活 (%s) | 食物 %.0f"
                      % (before, len(alive), causes or "无死亡", eco.food))
            # 死者真的移除（释放皮层内存）
            keep_ids = {id(x) for x in alive}
            for ind in pop:
                if id(ind) not in keep_ids:
                    ind.cortex = None
                    ind.alive = False
            pop = alive
            if not pop:
                print("  ⚠ 种群灭绝（第 %d 代，生态筛选后全灭）" % gen)
                break
            # 用残留的 vit_map 重算（不依赖旧的 order 索引）
            fits = [vit_map.get(ind.name, 0.0) for ind in pop]
            order = np.argsort(fits)[::-1]
            n_survivors = max(1, int(round(len(pop) * survival_ratio)))

        # 2b. 选择
        survivors = [pop[i] for i in order[:n_survivors]]

        # 3. 繁殖：必须走完求偶序列才能生殖（不再直接 reproduce）
        # 雌性按"好基因 + 避免近交"选择，所以低活力雄性会被拒
        median_vit = float(np.median(fits))
        mating.set_population_median(median_vit)
        next_pop = []
        attempts = 0
        while len(next_pop) < pop_size and attempts < pop_size * 8:
            attempts += 1
            a = survivors[np.random.randint(len(survivors))]
            b = survivors[np.random.randint(len(survivors))]
            if a is b:
                continue
            # 随机决定谁是雄性（交配方向不影响遗传，只影响选择谁）
            if np.random.rand() < 0.5:
                male, female = a, b
            else:
                male, female = b, a
            male_vit = fits[pop.index(male)] if male in pop else median_vit
            child, _info = mating.court_and_mate(
                male, female, male_vitality=male_vit,
                mutation_rate=mutation_rate,
                generation=gen + 1,
                name="F%d_%d" % (gen + 1, len(next_pop)))
            if child is not None:
                next_pop.append(child)

        # 繁殖失败太多就补一个和存活者同基因的（保证种群不灭）
        while len(next_pop) < pop_size and survivors:
            a = survivors[np.random.randint(len(survivors))]
            b = survivors[np.random.randint(len(survivors))]
            child, _ = reproduce(a, b, uterus=uterus, mutation_rate=0.0,
                                 generation=gen + 1,
                                 name="F%d_rescue_%d" % (gen + 1, len(next_pop)))
            if child is not None:
                next_pop.append(child)
            else:
                break

        # 4. 淘汰（释放内存）
        for ind in pop:
            ind.cortex = None
            ind.alive = False
        del pop
        gc.collect()
        pop = next_pop
        if not pop:
            print("  ⚠ 种群灭绝（第 %d 代）" % (gen + 1))
            break

    # ---------- 报告 ----------
    elapsed = time.time() - t_start
    print("\n" + "=" * 74)
    print("【结果】%d 代，%.0f 秒" % (len(history), elapsed))
    print("=" * 74)

    means = [h["mean"] for h in history]
    maxs = [h["max"] for h in history]
    divs = [h["diversity"] for h in history]

    g0, gN = history[0], history[-1]
    print("\n  平均活力: %.0f → %.0f  (%+.1f%%)"
          % (g0["mean"], gN["mean"],
             (gN["mean"] - g0["mean"]) / max(g0["mean"], 1e-9) * 100))
    print("  最大活力: %.0f → %.0f  (%+.1f%%)"
          % (g0["max"], gN["max"],
             (gN["max"] - g0["max"]) / max(g0["max"], 1e-9) * 100))
    print("  遗传多样性: %.3f → %.3f  (%+.1f%%)"
          % (g0["diversity"], gN["diversity"],
             (gN["diversity"] - g0["diversity"]) / max(g0["diversity"], 1e-9) * 100))

    print("\n【平均活力曲线】")
    print(ascii_curve(means))

    print("\n【遗传多样性曲线】（选择压力会让它下降）")
    print(ascii_curve(divs))

    print("\n【基因漂移】（第 0 代 → 最后一代的平均表现型）")
    print("  %-12s %10s %10s %10s" % ("基因", "第0代", "最后代", "变化"))
    for i, (name, lo, hi, is_int, desc) in enumerate(GENES):
        a, b = g0["gene_means"][i], gN["gene_means"][i]
        span = hi - lo
        pct = (b - a) / span * 100
        mark = ""
        if abs(pct) > 15:
            mark = " ←明显" + ("上升" if pct > 0 else "下降")
        print("  %-12s %10.4f %10.4f %+9.1f%%%s" % (name, a, b, pct, mark))

    print("\n【判断】")
    up = gN["mean"] > g0["mean"] * 1.05
    down = gN["mean"] < g0["mean"] * 0.95
    if up:
        print("  ✅ 平均活力上升 >5%：选择起作用了，基因确实影响适应度。")
    elif down:
        print("  ⚠ 平均活力下降 >5%：可能近交衰退（种群太小，多样性耗尽）。")
    else:
        print("  ➖ 平均活力变化在 ±5% 以内：这个基因范围里，活力主要由")
        print("     网络初始状态的随机性决定，而不是基因。这是真实结果。")
    if gN["diversity"] < g0["diversity"] * 0.9:
        print("  ✅ 遗传多样性下降：有选择压力的痕迹（收敛）。")
    else:
        print("  ➖ 遗传多样性没明显下降：选择压力不够强，或突变补充了变异。")
    if eco_log:
        print("\n【生态统计】")
        total_dead = sum(sum(e["causes"].values()) for e in eco_log)
        cause_all = {}
        for e in eco_log:
            for k, v in e["causes"].items():
                cause_all[k] = cause_all.get(k, 0) + v
        print("  总死亡 %d | 分因 %s | 承载力 %d"
              % (total_dead, cause_all, eco.carrying_capacity()))

    print("\n【求偶统计】")
    ms = mating.summary()
    print("  求偶次数 %d | 成功 %d (%.0f%%) | 被拒 %d | 产仔 %d"
          % (ms["courtships"], ms["success"], ms["success_rate"] * 100,
             ms["rejected"], ms["mating"]))
    if ms["fail_at_step"]:
        print("  卡住的步骤: %s" % ms["fail_at_step"])

    print("\n  注：以上每个数字都来自真跑（%d 代 × %d 个体 × %d seed 活力测量）。"
          % (len(history), pop_size, len(SEEDS)))

    # 保存曲线数据
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evolution_history.npz")
    np.savez(out,
             means=np.array(means), maxs=np.array(maxs), divs=np.array(divs),
             gene_means=np.array([h["gene_means"] for h in history]),
             gene_names=np.array(GENE_NAMES))
    print("  曲线数据已存: %s" % out)
    return history


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    eco = "--no-eco" not in sys.argv
    main(n_gen=n, pop_size=p, use_eco=eco)
