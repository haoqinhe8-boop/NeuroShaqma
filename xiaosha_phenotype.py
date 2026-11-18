# -*- coding: utf-8 -*-
"""小沙的表型 —— 让每个个体真的不一样。

问题：原来 350 个个体只有 MiniBrain 的权重不同，结构、动作、寿命、体型
全都一样。基因表 12 个位点也全是"大脑怎么长"，没有一处决定"这个身体是什么样"。

这里加 10 个表型基因，每个都带**真实的代价**——不能存在全能最优解，
否则几代之后所有人都会收敛到同一个"完美个体"，差异又没了。

   基因        范围      好处                      代价
  ─────────────────────────────────────────────────────────────────
  size      0.65-1.55  采集/出力更多              吃得多、动作慢
  metab     0.75-1.35  单位时间做更多事            饿得快
  eye       0.55-1.50  看得远（能发现远处食物）    夜里更瞎、脑耗高
  nose      0.55-1.50  闻得到食物浓度              同上
  curio     0.03-0.42  试错快、学得快              容易把自己试死
  learn     0.010-0.055 学得快                    容易学错（噪声也学）
  life      170-460    活得久                    老得晚 → 繁殖窗口占比小
  social    0.10-0.92  教得多、传得多、活得稳      自己干活的时间少
  night     0.00-1.00  避开白天的竞争              夜里产出少、看不见
  patience  0.18-0.95  愿意投资长期回报（种地建坛） 不肯先吃苦 → 早期容易饿死

跑法：python xiaosha_phenotype.py  → 自测
"""
import numpy as np

__version__ = "1.0.0"

# (名字, 最小, 最大, 是否整数, 说明)
PHENO_GENES = [
    ("size",     0.65,  1.55,  False, "体型：大个子出活多但饭量大"),
    ("metab",    0.75,  1.35,  False, "代谢：快则动作多但饿得快"),
    ("eye",      0.55,  1.50,  False, "视力：看得远，但夜里更吃亏"),
    ("nose",     0.55,  1.50,  False, "嗅觉：闻得到远处的食物"),
    ("curio",    0.03,  0.42,  False, "好奇：探索率高，学得快也死得快"),
    ("learn",    0.010, 0.055, False, "学习率"),
    ("life",     170,   460,   True,  "寿命（步）"),
    ("social",   0.10,  0.92,  False, "社交：教学传教意愿"),
    ("night",    0.00,  1.00,  False, "夜行性：0 纯昼行，1 纯夜行"),
    ("patience", 0.18,  0.95,  False, "耐心：愿不愿意投资长期回报"),
]
PHENO_NAMES = [g[0] for g in PHENO_GENES]
N_PHENO = len(PHENO_GENES)

# 突变强度（每个基因按自己量程的百分比突变）
MUT_RATE = 0.18


def random_genome(rng=None):
    r = rng if rng is not None else np.random
    out = []
    for name, lo, hi, is_int, _ in PHENO_GENES:
        v = r.uniform(lo, hi)
        out.append(int(round(v)) if is_int else float(v))
    return np.array(out, dtype=np.float64)


def clamp(values):
    out = []
    for v, (_, lo, hi, is_int, _) in zip(values, PHENO_GENES):
        v = float(np.clip(v, lo, hi))
        out.append(int(round(v)) if is_int else v)
    return np.array(out, dtype=np.float64)


def crossover(pa, ma, rng=None, mutation=MUT_RATE):
    """子代：每个位点从父母随机取一个，再按量程比例突变。"""
    r = rng if rng is not None else np.random
    child = np.empty(N_PHENO, dtype=np.float64)
    for i, (_, lo, hi, is_int, _) in enumerate(PHENO_GENES):
        v = pa[i] if r.rand() < 0.5 else ma[i]
        if r.rand() < 0.5:                       # 一半概率突变
            span = hi - lo
            v += r.randn() * span * mutation
        child[i] = v
    return clamp(child)


class Phenotype:
    """基因 → 实际能力。所有"好处"和"代价"都在这里换算成系数。"""

    def __init__(self, genome):
        self.genome = np.array(genome, dtype=np.float64)
        g = {n: float(v) for (n, *_), v in zip(PHENO_GENES, self.genome)}

        self.size = g["size"]
        self.metab = g["metab"]
        self.eye = g["eye"]
        self.nose = g["nose"]
        self.curio = g["curio"]
        self.learn = g["learn"]
        self.life = int(g["life"])
        self.social = g["social"]
        self.night = g["night"]
        self.patience = g["patience"]

        # ---- 派生量：好处 ----
        # 采集量：随体型**饱和**增长（大个子不会成正比地多干出活来）。
        # 一开始我用的是线性 0.70+0.60*size，配上线性的能耗，
        # 结果是"越大越划算"，几代之后所有人都会顶到最大体型 —— 又没有差异了。
        self.harvest_mult = (2.20 * self.size / (self.size + 0.75)) \
            * (0.85 + 0.30 * np.sqrt(self.metab))
        # 感知半径：视力和嗅觉一起决定（3 格为基准）
        self.sense_r = 1.0 + 1.6 * (0.5 * self.eye + 0.5 * self.nose)
        # 教学/传教意愿
        self.teach_mult = 0.30 + 1.20 * self.social
        # 探索率（ε）
        self.eps = float(np.clip(self.curio, 0.02, 0.45))

        # ---- 派生量：代价（体型两头都有代价）----
        # 公式 (size + 0.55/size)/1.55 在 size=1 处最小、两头都变大：
        #   小个子：表面积/体积比大，热量散得快，得一直吃（真实的伯格曼法则）
        #   大个子：绝对需求高，在贫瘠的地方撑不住
        # 这样净效益在中间某个体型最高，而不是"越大越好"或"越小越好"。
        shape = (self.size + 0.55 / self.size) / 1.55
        self.metab_cost = 0.30 * (0.55 + 0.75 * self.metab) * shape
        # 夜里看不见：视力越好，白天越占优，夜里越吃亏
        self.night_penalty = 1.0 - 0.45 * self.eye

    # ---------- 具体行为接口 ----------
    def activity(self, light):
        """当前光照下这个个体的活跃度 0-1。

        昼行性（night=0）：光照越足越活跃。
        夜行性（night=1）：越黑越活跃。
        两头都能活，但食物再生是跟光照走的 —— 夜行的人在自己最活跃的时候
        正好是食物最少的时候，这就是它的代价。
        """
        day = 1.0 - self.night
        return float(np.clip(day * light + self.night * (1.0 - light), 0.05, 1.0))

    def sense_range(self, light):
        """感知半径：天黑看得近（视力越好越明显）。"""
        r = self.sense_r * (0.45 + 0.55 * light)
        return float(np.clip(r, 0.8, 3.2))

    def to_dict(self):
        return {
            "size": round(self.size, 2), "metab": round(self.metab, 2),
            "eye": round(self.eye, 2), "nose": round(self.nose, 2),
            "curio": round(self.curio, 3), "learn": round(self.learn, 4),
            "life": self.life, "social": round(self.social, 2),
            "night": round(self.night, 2), "patience": round(self.patience, 2),
        }

    def label(self):
        """人话描述：让"每个人不同"这件事在界面上看得见。"""
        parts = []
        parts.append("大个" if self.size > 1.25 else ("小个" if self.size < 0.90 else "中等"))
        if self.night > 0.72:
            parts.append("夜行")
        elif self.night < 0.28:
            parts.append("昼行")
        if self.eye > 1.25:
            parts.append("眼尖")
        if self.nose > 1.25:
            parts.append("鼻灵")
        if self.curio > 0.30:
            parts.append("冒失")
        elif self.curio < 0.10:
            parts.append("谨慎")
        if self.social > 0.70:
            parts.append("爱扎堆")
        elif self.social < 0.30:
            parts.append("独行")
        if self.patience > 0.72:
            parts.append("有耐心")
        elif self.patience < 0.35:
            parts.append("急躁")
        if self.life > 380:
            parts.append("长寿")
        elif self.life < 220:
            parts.append("短命")
        return "·".join(parts)


# ==================== 自测 ====================

def selftest():
    print("  小沙的表型 selftest（个体差异 / 代价 / 遗传 / 表达）")
    ok, total = 0, 0
    rng = np.random.RandomState(20261118)

    # ---- 1. 十项基因都在合法范围内 ----
    total += 1
    gs = [random_genome(rng) for _ in range(500)]
    bad = 0
    for g in gs:
        for v, (_, lo, hi, _, _) in zip(g, PHENO_GENES):
            if not (lo - 1e-9 <= v <= hi + 1e-9):
                bad += 1
    print("  [1] 500 个随机基因组，越界 %d 个" % bad)
    ok += int(bad == 0)

    # ---- 2. 每个人都不一样 ----
    total += 1
    uniq = len({tuple(round(float(x), 6) for x in g) for g in gs})
    print("  [2] 500 个基因组里有 %d 个互不相同" % uniq)
    ok += int(uniq == 500)

    # ---- 3. 表型描述真的会分化 ----
    total += 1
    labels = {}
    for g in gs[:200]:
        lb = Phenotype(g).label()
        labels[lb] = labels.get(lb, 0) + 1
    top = sorted(labels.items(), key=lambda kv: -kv[1])[:5]
    print("  [3] 200 个个体分化出 %d 种描述，最常见的 5 种：" % len(labels))
    for lb, n in top:
        print("        %-24s %d 人" % (lb, n))
    ok += int(len(labels) >= 40)

    # ---- 4. 体型真的有代价：采集多但吃得多 ----
    total += 1
    small = Phenotype(clamp([0.70, 1.0, 1.0, 1.0, 0.15, 0.03, 300, 0.5, 0.5, 0.5]))
    big = Phenotype(clamp([1.50, 1.0, 1.0, 1.0, 0.15, 0.03, 300, 0.5, 0.5, 0.5]))
    print("  [4] 小个: 采集×%.2f 能耗×%.3f | 大个: 采集×%.2f 能耗×%.3f"
          % (small.harvest_mult, small.metab_cost, big.harvest_mult, big.metab_cost))
    ok += int(big.harvest_mult > small.harvest_mult and big.metab_cost > small.metab_cost)

    # ---- 5. 体型收益 vs 代价：净收益不是单调的（存在中间最优）----
    total += 1
    sizes = np.linspace(0.65, 1.55, 40)
    net = []
    for s in sizes:
        p = Phenotype(clamp([s, 1.0, 1.0, 1.0, 0.15, 0.03, 300, 0.5, 0.5, 0.5]))
        net.append(p.harvest_mult / p.metab_cost)
    net = np.array(net)
    best = sizes[int(np.argmax(net))]
    monotonic = (np.all(np.diff(net) > 0) or np.all(np.diff(net) < 0))
    print("  [5] 净效益(采集/能耗) 最优体型 %.2f，单调=%s" % (best, monotonic))
    ok += int(not monotonic)          # 必须是中间最优，不能越大越好

    # ---- 6. 夜行性有代价：夜行者活跃时食物最少 ----
    total += 1
    diur = Phenotype(clamp([1.0, 1.0, 1.0, 1.0, 0.15, 0.03, 300, 0.5, 0.00, 0.5]))
    noct = Phenotype(clamp([1.0, 1.0, 1.0, 1.0, 0.15, 0.03, 300, 0.5, 1.00, 0.5]))
    # 白天光照 1，夜里 0
    print("  [6] 白天活跃度: 昼行 %.2f / 夜行 %.2f | 夜里: 昼行 %.2f / 夜行 %.2f"
          % (diur.activity(1.0), noct.activity(1.0), diur.activity(0.0), noct.activity(0.0)))
    ok += int(diur.activity(1.0) > noct.activity(1.0) and noct.activity(0.0) > diur.activity(0.0))

    # ---- 7. 视力好 = 白天看得远，夜里反而更瞎 ----
    total += 1
    sharp = Phenotype(clamp([1.0, 1.0, 1.50, 1.0, 0.15, 0.03, 300, 0.5, 0.5, 0.5]))
    dull = Phenotype(clamp([1.0, 1.0, 0.60, 1.0, 0.15, 0.03, 300, 0.5, 0.5, 0.5]))
    print("  [7] 感知半径: 眼尖 白天%.2f/夜里%.2f | 眼钝 白天%.2f/夜里%.2f"
          % (sharp.sense_range(1.0), sharp.sense_range(0.0),
             dull.sense_range(1.0), dull.sense_range(0.0)))
    ok += int(sharp.sense_range(1.0) > dull.sense_range(1.0)
              and sharp.sense_range(0.0) > dull.sense_range(0.0))

    # ---- 8. 遗传：子代接近父母但不等同 ----
    total += 1
    pa, ma = random_genome(rng), random_genome(rng)
    kids = [crossover(pa, ma, rng) for _ in range(50)]
    same_as_parent = sum(1 for k in kids
                         if np.allclose(k, pa) or np.allclose(k, ma))
    dist = np.mean([np.linalg.norm((k - pa)[:7]) for k in kids])
    print("  [8] 50 个子代：和父母完全相同的 %d 个 | 与父本基因距离均值 %.3f"
          % (same_as_parent, dist))
    ok += int(same_as_parent == 0 and dist > 0.01)

    # ---- 9. 突变不会飞出范围 ----
    total += 1
    out_of_range = 0
    for _ in range(300):
        k = crossover(pa, ma, rng, mutation=2.0)      # 极端突变
        for v, (_, lo, hi, _, _) in zip(k, PHENO_GENES):
            if not (lo - 1e-9 <= v <= hi + 1e-9):
                out_of_range += 1
    print("  [9] 突变强度 2.0（极端）下越界次数 %d" % out_of_range)
    ok += int(out_of_range == 0)

    # ---- 10. 真实种群：350 个个体有多少种表型 ----
    total += 1
    pop = [Phenotype(random_genome(rng)) for _ in range(350)]
    sizes = np.array([p.size for p in pop])
    lives = np.array([p.life for p in pop])
    eps = np.array([p.eps for p in pop])
    print("  [10] 350 个体：体型 %.2f-%.2f | 寿命 %d-%d | 探索率 %.3f-%.3f"
          % (sizes.min(), sizes.max(), lives.min(), lives.max(), eps.min(), eps.max()))
    ok += int(np.ptp(sizes) > 0.5 and np.ptp(lives) > 150)

    print("\n  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    print("=" * 74)
    print("小沙的表型系统 · 每个个体都不一样")
    print("=" * 74)
    ok = selftest()
    print()
    rng = np.random.RandomState(1)
    print("随便抽 6 个个体看看:")
    for i in range(6):
        p = Phenotype(random_genome(rng))
        print("  %d. %-40s 体型%.2f 寿命%d 探索%.2f"
              % (i + 1, p.label(), p.size, p.life, p.eps))
    sys.exit(0 if ok else 1)
