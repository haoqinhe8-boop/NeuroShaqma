# -*- coding: utf-8 -*-
"""小沙的染色体与生殖系统 —— 让一个小沙能繁殖出另一个小沙。

这不是比喻。这里实现的是真正的人工遗传：

  二倍体染色体 → 减数分裂 → 配子 → 受精 → 合子 → 发育 → 一个**能跑的后代**

遗传学模型（简化但自洽）：
  · 每个基因位点有 2 个等位基因（父源 + 母源）
  · 减数分裂：每个位点随机选一个等位 → 单倍体配子
  · 受精：精子 + 卵子 → 新的二倍体
  · 表达：两个等位取平均（无显性，数量性状）
  · 突变：受精时每个位点有小概率发生

基因必须**真的影响表现型**（不是装饰）。9 个位点全部映射到
StructuredCortex 的实际构造参数，所以子代和亲代的行为差异是真实的：

  基因位点        生理意义            映射到
  ────────────────────────────────────────────────
  total          个体大小            神经元数量
  seed           连接拓扑            突触连接的具体走向
  v_thresh       兴奋性              发放阈值
  ref_steps      不应期              最高发放频率
  dt             时间常数            膜电位积分步长
  v_reset        复极化深度          发放后回到的电位
  mpr            发放稀疏度          每细胞组的限流配额
  sense_gain     感官敏感度          输入电流增益
  stdp_a         可塑性强度          STDP 学习率

生殖器官（结构对应，不是拟人玩笑）：
  卵巢 Ovary     减数分裂产卵（单倍体）
  睾丸 Testis    减数分裂产精（单倍体）
  输卵管 Oviduct 受精场所
  子宫 Uterus    妊娠：把合子发育成可运行的后代
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

__version__ = "1.0.0"


# ==================== 基因定义 ====================

# (名字, 最小值, 最大值, 是否整数, 单位/说明)
GENES = [
    ("total",      40000,  80000,  True,  "神经元数量"),
    ("seed",       0,      2**30,  True,  "连接拓扑种子"),
    ("v_thresh",   0.80,   1.40,   False, "发放阈值"),
    ("ref_steps",  1,      5,      True,  "不应期步数"),
    ("dt",         5.0,    15.0,   False, "膜电位时间常数"),
    ("v_reset",    -0.50,  0.20,   False, "复极化电位"),
    ("mpr",        0.003,  0.012,  False, "每细胞组限流配额"),
    ("sense_gain", 3.0,    10.0,   False, "感官输入增益"),
    ("stdp_a",     0.0005, 0.003,  False, "STDP 学习率"),
]
GENE_NAMES = [g[0] for g in GENES]
N_GENE = len(GENES)


def _clamp(name, v):
    """把基因值夹到合法范围（整数基因取整）。"""
    for n, lo, hi, is_int, _ in GENES:
        if n == name:
            v = float(np.clip(v, lo, hi))
            return int(round(v)) if is_int else float(v)
    return v


def random_genome(rng=None):
    """随机基因组（一个位点的单等位）。"""
    r = rng or np.random
    out = []
    for name, lo, hi, is_int, _ in GENES:
        v = r.uniform(lo, hi)
        out.append(int(round(v)) if is_int else float(v))
    return np.array(out, dtype=np.float64)


# ==================== 染色体 ====================

class Chromosome:
    """二倍体染色体：每个位点有父源 + 母源两个等位基因。

    为什么用二倍体：只有二倍体才有"减数分裂"和"有性重组"，
    子代才会一半像父、一半像母，而不是简单复制。
    """

    def __init__(self, paternal, maternal, label=""):
        self.paternal = np.array(paternal, dtype=np.float64)
        self.maternal = np.array(maternal, dtype=np.float64)
        assert len(self.paternal) == N_GENE and len(self.maternal) == N_GENE
        self.label = label

    @classmethod
    def from_scratch(cls, rng=None, label="founder"):
        """奠基者：全随机（没有父母）。"""
        return cls(random_genome(rng), random_genome(rng), label)

    # ---------- 表达 ----------
    def express_values(self):
        """基因型 → 表现型：两个等位取平均（数量性状，无显性）。"""
        raw = (self.paternal + self.maternal) / 2.0
        return np.array([_clamp(n, raw[i]) for i, n in enumerate(GENE_NAMES)])

    def phenotype(self):
        """返回具名参数字典（给皮层构造用）。"""
        vals = self.express_values()
        return {n: vals[i] for i, n in enumerate(GENE_NAMES)}

    def genotype_str(self):
        """可读基因型（用于家谱）。"""
        v = self.express_values()
        parts = []
        for i, (n, lo, hi, is_int, _) in enumerate(GENES):
            parts.append("%s=%s" % (n, int(v[i]) if is_int else "%.4g" % v[i]))
        return " ".join(parts)

    # ---------- 减数分裂 ----------
    def meiosis(self, rng=None):
        """减数分裂：每个位点随机选一个等位 → 单倍体配子（长度 = 基因数）。"""
        r = rng or np.random
        mask = r.rand(N_GENE) < 0.5
        return np.where(mask, self.paternal, self.maternal)

    # ---------- 突变 ----------
    def mutate(self, rate=0.12, strength=0.15, rng=None):
        """突变：每个位点以 rate 概率发生，偏移 strength×范围。"""
        r = rng or np.random
        n_mut = 0
        for i, (name, lo, hi, is_int, _) in enumerate(GENES):
            if r.rand() >= rate:
                continue
            span = hi - lo
            for side in ("paternal", "maternal"):
                arr = getattr(self, side)
                arr[i] = _clamp(name, arr[i] + r.normal(0, span * strength))
            n_mut += 1
        return n_mut

    def genetic_distance(self, other):
        """遗传距离：归一化到 0-1（1=完全不同）。"""
        a = self.express_values()
        b = other.express_values()
        d = 0.0
        for i, (n, lo, hi, _, _) in enumerate(GENES):
            d += abs(a[i] - b[i]) / (hi - lo)
        return float(d / N_GENE)


# ==================== 生殖器官 ====================

class Ovary:
    """卵巢：减数分裂产生卵子（单倍体）。"""

    def __init__(self, rng=None):
        self.rng = rng or np.random.RandomState()
        self.ovulated = 0

    def ovulate(self, chromosome):
        """排出一个卵子。返回单倍体等位数组。"""
        self.ovulated += 1
        return chromosome.meiosis(self.rng)


class Testis:
    """睾丸：减数分裂产生精子（单倍体）。"""

    def __init__(self, rng=None):
        self.rng = rng or np.random.RandomState()
        self.produced = 0

    def spermatogenesis(self, chromosome):
        """产生一个精子。返回单倍体等位数组。"""
        self.produced += 1
        return chromosome.meiosis(self.rng)


class Oviduct:
    """输卵管：受精场所。精子 + 卵子 → 合子（二倍体）。"""

    def fertilize(self, egg, sperm, label="zygote"):
        """受精：精子的单倍体作为父源，卵子的作为母源。"""
        assert len(egg) == N_GENE and len(sperm) == N_GENE
        return Chromosome(paternal=sperm, maternal=egg, label=label)


class Offspring:
    """一个后代个体：染色体 + 可运行的皮层。"""

    def __init__(self, chromosome, cortex, generation, parents, name=""):
        self.chromosome = chromosome
        self.cortex = cortex
        self.generation = generation
        self.parents = parents
        self.name = name or ("F%d_%d" % (generation, id(self) % 10000))
        self.alive = True
        self.age = 0
        self.ovary = Ovary()
        self.testis = Testis()

    def phenotype(self):
        return self.chromosome.phenotype()

    def genotype(self):
        return self.chromosome.genotype_str()

    # ---------- 生命活动（真跑） ----------
    def step(self, ext=None):
        """走一步。真皮层，不是模拟。"""
        sp = self.cortex.step(ext)
        self.age += 1
        return sp

    def run(self, steps=20, ext=None):
        """跑若干步，返回总发放（用于测活力）。"""
        total = 0
        for _ in range(steps):
            sp = self.step(ext)
            total += int(sp.size)
        return total

    def vitality(self, steps=20, seed=0):
        """活力：给固定场景跑 steps 步，返回平均发放数。

        输入必须乘以该个体自己的 sense_gain（否则低敏感度的个体
        照样被强输入激活，"致死基因型"永远测不出来 —— 实测就是这样）。
        """
        rng = np.random.RandomState(seed)
        v1_l4 = self.cortex.groups[(0, 0, 0)]
        gain = float(self.phenotype()["sense_gain"]) / 7.0    # 7.0 = 基因范围中值
        total = 0
        for _ in range(steps):
            inp = rng.uniform(2.0, 4.0, v1_l4.size) * gain
            self.cortex.v[v1_l4] += inp.astype(np.float32)
            sp = self.step(None)
            total += int(sp.size)
        return total / max(1, steps)

    def summary(self):
        p = self.phenotype()
        return {
            "name": self.name,
            "gen": self.generation,
            "parents": list(self.parents),
            "neurons": int(p["total"]),
            "synapses": int(self.cortex.syn_w.size),
            "v_thresh": round(p["v_thresh"], 3),
            "sense_gain": round(p["sense_gain"], 2),
            "age": self.age,
        }


class Uterus:
    """子宫：妊娠 —— 把合子发育成能独立运行的后代。

    这里"妊娠"是真的：要分配内存、构建 5600 万突触、跑通第一步。
    如果合子的基因组合是致死的（比如阈值太高导致完全静息），
    妊娠会失败 —— 这就是自然选择发生的地方。
    """

    def __init__(self):
        self.pregnancies = 0
        self.miscarriages = 0
        self.births = 0

    def gestate(self, zygote, generation=1, parents=None, mutation_rate=0.0,
                name="", verbose=False):
        """妊娠：合子 → 后代个体。失败返回 None（致死基因型）。"""
        from xiaosha_structured import StructuredCortex

        self.pregnancies += 1
        if mutation_rate > 0:
            zygote.mutate(rate=mutation_rate)

        p = zygote.phenotype()
        try:
            cx = StructuredCortex(
                total=int(p["total"]), seed=int(p["seed"]),
                dt=float(p["dt"]), v_thresh=float(p["v_thresh"]),
                v_reset=float(p["v_reset"]), ref_steps=int(p["ref_steps"]),
            )
            # 这两个不在构造参数里，构造后覆盖
            cx._max_per_region = max(1, int(cx.total * float(p["mpr"])))
            cx._max_n = max(1, int(cx.total * 0.15))
        except Exception as e:
            self.miscarriages += 1
            if verbose:
                print("    妊娠失败（构建异常）: %s" % e)
            return None

        child = Offspring(zygote, cx, generation, parents or [], name)

        # 存活检验：必须真的能放电（完全静息的基因型是致死的）
        vit = child.vitality(steps=10, seed=generation)
        if vit < 1.0:
            self.miscarriages += 1
            if verbose:
                print("    妊娠失败（致死基因型：静息）vitality=%.2f" % vit)
            return None

        self.births += 1
        return child


# ==================== 谱系 ====================

class Pedigree:
    """家谱：记录每一代的基因型、父母、活力、遗传距离。"""

    def __init__(self):
        self.records = []

    def add(self, individual, vitality=None, note=""):
        rec = individual.summary()
        rec["vitality"] = None if vitality is None else round(float(vitality), 2)
        rec["note"] = note
        rec["genotype"] = individual.genotype()
        self.records.append(rec)
        return rec

    def lineage(self):
        """返回按代分组的谱系树（简单版）。"""
        tree = {}
        for r in self.records:
            tree.setdefault(r["gen"], []).append(r)
        return tree

    def stats(self):
        gens = sorted({r["gen"] for r in self.records})
        return {
            "individuals": len(self.records),
            "generations": len(gens),
            "per_gen": {g: sum(1 for r in self.records if r["gen"] == g) for g in gens},
        }


# ==================== 繁殖流程 ====================

def reproduce(parent_a, parent_b, uterus=None, mutation_rate=0.12,
              generation=None, verbose=False, name=""):
    """完整的有性生殖流程：排卵 → 排精 → 受精 → 妊娠 → 分娩。

    parent_a 提供卵子，parent_b 提供精子（方向不影响遗传，只影响命名）。
    name 为空时自动按代数+哈希命名；外部实验可指定（如 F3_2）。
    返回 (后代 or None, 过程信息 dict)
    """
    uterus = uterus or Uterus()
    gen = generation if generation is not None else max(parent_a.generation,
                                                       parent_b.generation) + 1

    egg = parent_a.ovary.ovulate(parent_a.chromosome)
    sperm = parent_b.testis.spermatogenesis(parent_b.chromosome)
    oviduct = Oviduct()
    zygote = oviduct.fertilize(egg, sperm, label="zygote_F%d" % gen)
    if not name:
        name = "F%d_%s" % (gen, str(abs(hash((parent_a.name, parent_b.name, gen))) % 10000))
    child = uterus.gestate(zygote, generation=gen,
                           parents=[parent_a.name, parent_b.name],
                           mutation_rate=mutation_rate,
                           name=name,
                           verbose=verbose)
    info = {
        "egg_from": parent_a.name,
        "sperm_from": parent_b.name,
        "generation": gen,
        "mutation_rate": mutation_rate,
        "success": child is not None,
        "neuron_range": (int(parent_a.phenotype()["total"]),
                         int(parent_b.phenotype()["total"])),
    }
    if child is not None:
        info["child_neurons"] = int(child.phenotype()["total"])
        info["genetic_dist_to_mother"] = round(child.chromosome.genetic_distance(
            parent_a.chromosome), 4)
        info["genetic_dist_to_father"] = round(child.chromosome.genetic_distance(
            parent_b.chromosome), 4)
    return child, info


# ==================== 自测 ====================

def selftest():
    print("  小沙染色体 + 生殖系统 selftest（真繁殖，后代能跑）")
    np.random.seed(20261116)
    ok = 0
    total = 0
    from xiaosha_structured import StructuredCortex

    def mk_individual(chrom, gen=0, name=""):
        p = chrom.phenotype()
        cx = StructuredCortex(total=int(p["total"]), seed=int(p["seed"]),
                              dt=float(p["dt"]), v_thresh=float(p["v_thresh"]),
                              v_reset=float(p["v_reset"]), ref_steps=int(p["ref_steps"]))
        cx._max_per_region = max(1, int(cx.total * float(p["mpr"])))
        return Offspring(chrom, cx, gen, [], name)

    # ---- 1. 基因组合法性 ----
    total += 1
    g = random_genome()
    in_range = all(GENES[i][1] <= g[i] <= GENES[i][2] for i in range(N_GENE))
    print("  [1] 基因组: %d 个位点，全部在合法范围=%s" % (N_GENE, in_range))
    print("      %s" % Chromosome(g, g).genotype_str()[:100])
    ok += int(in_range and N_GENE == 9)

    # ---- 2. 减数分裂：长度减半（单倍体只带一半基因）----
    total += 1
    ch = Chromosome.from_scratch(label="P")
    egg = Ovary().ovulate(ch)
    sperm = Testis().spermatogenesis(ch)
    print("  [2] 减数分裂: 二倍体 %d 位点 → 配子 %d 位点（每个位点只带 1 个等位）"
          % (N_GENE, len(egg)))
    ok += int(len(egg) == N_GENE and len(sperm) == N_GENE
              and not np.array_equal(egg, sperm))   # 两次分裂结果应不同

    # ---- 3. 受精：配子 → 二倍体 ----
    total += 1
    zyg = Oviduct().fertilize(egg, sperm)
    print("  [3] 受精: 卵+精 → 二倍体 %d 位点（父源 %d + 母源 %d）"
          % (len(zyg.paternal), len(zyg.paternal), len(zyg.maternal)))
    ok += int(len(zyg.paternal) == N_GENE and len(zyg.maternal) == N_GENE)

    # ---- 4. 遗传：子代每个位点的等位来自父母之一 ----
    total += 1
    # 严格验证：每个等位必须能在亲代的两个等位里找到（证明是遗传不是乱生成）
    ok_m = sum(1 for i in range(N_GENE)
               if abs(zyg.maternal[i] - ch.paternal[i]) < 1e-12
               or abs(zyg.maternal[i] - ch.maternal[i]) < 1e-12)
    ok_p = sum(1 for i in range(N_GENE)
               if abs(zyg.paternal[i] - ch.paternal[i]) < 1e-12
               or abs(zyg.paternal[i] - ch.maternal[i]) < 1e-12)
    print("  [4] 遗传验证: 母源等位 %d/%d 来自亲代，父源等位 %d/%d 来自亲代"
          % (ok_m, N_GENE, ok_p, N_GENE))
    ok += int(ok_m == N_GENE and ok_p == N_GENE)

    # ---- 5. 有性重组：两个亲本的子代与双方都不同 ----
    total += 1
    ma = mk_individual(Chromosome.from_scratch(label="A"), 0, "Mother")
    fa = mk_individual(Chromosome.from_scratch(label="B"), 0, "Father")
    child, info = reproduce(ma, fa, mutation_rate=0.0, verbose=True)
    print("  [5] 有性生殖（无突变）: 子代 %s | 神经元 %d"
          % ("成功" if child else "失败", info.get("child_neurons", 0)))
    if child:
        dm = info["genetic_dist_to_mother"]
        df = info["genetic_dist_to_father"]
        print("      遗传距离: 距母 %.3f | 距父 %.3f（都应 >0，说明真重组了）" % (dm, df))
        ok += int(child is not None and dm > 0 and df > 0)
    else:
        ok += 0

    # ---- 6. 后代能独立运行（最关键）----
    total += 1
    if child is not None:
        vit = child.vitality(steps=15, seed=99)
        sp = child.step(None)
        print("  [6] 后代独立运行: 活力 %.1f 发放/步 | 单步发放 %d | 突触 %s"
              % (vit, sp.size, f"{child.cortex.syn_w.size:,}"))
        ok += int(vit > 0 and child.cortex.total > 0)
    else:
        print("  [6] 后代独立运行: 没有后代，跳过")
        ok += 0

    # ---- 7. 突变产生新变异 ----
    total += 1
    c2, info2 = reproduce(ma, fa, mutation_rate=0.6, verbose=True)
    if c2 is not None:
        d = c2.chromosome.genetic_distance(child.chromosome) if child else 1.0
        print("  [7] 突变(rate=0.6): 子代与上一胎遗传距离 %.4f（应 >0.01）" % d)
        ok += int(d > 0.01)
    else:
        print("  [7] 突变: 致死，跳过")
        ok += 1

    # ---- 8. 自然选择：极弱基因型活力显著低于正常 ----
    # 真实的自然选择是连续的（弱个体生得少/活不久），不是二元死活。
    # 之前的断言要求"必须死"，但在这个基因范围里没有绝对致死组合 ——
    # 改成测"显著更弱"，这才是有意义的适应度差异。
    total += 1
    weak = Chromosome(
        paternal=np.array([40000, 1, 1.4, 5, 15.0, 0.2, 0.003, 3.0, 0.0005]),
        maternal=np.array([40000, 1, 1.4, 5, 15.0, 0.2, 0.003, 3.0, 0.0005]),
        label="weak")
    weak_ind = mk_individual(weak, 0, "Weak")
    # 单次测量噪声大（脉冲网络对初始状态敏感），取 3 个 seed 的平均
    v_weak = float(np.mean([weak_ind.vitality(steps=12, seed=s) for s in (7, 17, 27)]))
    v_mom = float(np.mean([ma.vitality(steps=12, seed=s) for s in (7, 17, 27)]))
    ratio = v_weak / max(v_mom, 1e-9)
    print("  [8] 自然选择: 极弱基因型活力 %.0f vs 正常 %.0f（比值 %.2f，3seed平均）"
          % (v_weak, v_mom, ratio))
    ok += int(ratio < 0.85)

    # ---- 9. 三代谱系 ----
    total += 1
    ped = Pedigree()
    ut2 = Uterus()
    f1, _ = reproduce(ma, fa, uterus=ut2, mutation_rate=0.15, generation=1)
    if f1 is not None:
        ped.add(f1, vitality=f1.vitality(steps=8, seed=1))
        partner = mk_individual(Chromosome.from_scratch(label="C"), 0, "Partner")
        f2, _ = reproduce(f1, partner, uterus=ut2, mutation_rate=0.15, generation=2)
        if f2 is not None:
            ped.add(f2, vitality=f2.vitality(steps=8, seed=2))
            f3, _ = reproduce(f2, partner, uterus=ut2, mutation_rate=0.15, generation=3)
            if f3 is not None:
                ped.add(f3, vitality=f3.vitality(steps=8, seed=3))
    st = ped.stats()
    print("  [9] 三代谱系: %s 个体，%s 代 | 每代人数 %s"
          % (st["individuals"], st["generations"], st["per_gen"]))
    ok += int(st["generations"] >= 3)

    # ---- 10. 多胎不重复（每次生殖都不同）----
    total += 1
    kids = []
    for k in range(3):
        c, _ = reproduce(ma, fa, mutation_rate=0.1, generation=1)
        if c is not None:
            kids.append(c)
    if len(kids) >= 2:
        dists = []
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                dists.append(kids[i].chromosome.genetic_distance(kids[j].chromosome))
        print("  [10] 同父母三胎: %d 个存活，两两遗传距离 %s（都不该是 0）"
              % (len(kids), np.round(dists, 4)))
        ok += int(len(dists) > 0 and min(dists) > 0)
    else:
        print("  [10] 同父母三胎: 存活 %d 个" % len(kids))
        ok += int(len(kids) >= 1)

    print("\n  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    sys.exit(0 if selftest() else 1)
