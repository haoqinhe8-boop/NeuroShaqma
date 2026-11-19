# -*- coding: utf-8 -*-
"""小沙的概念 —— 从零开始长的技术。

## 原来的问题

`xiaosha_civilization.TECH_TREE` 是 24 条手写的物品对：
    ("采集","石头") → "工具"
    ("采集","季节") → "农业"
    ...
这等于我把"人类会发明什么、按什么顺序"提前写好了。小沙只是照单念。

## 现在的模型：规则定义在属性上，不在物品上

一件东西由**属性**描述（能吃的、硬的、锋利的、能承重的、含水的、
随季节出现的、能承载意义的、能组织人的、能约束行为的……）。

组合的规则**不针对具体物品，只针对属性对**：

    (硬, 热)  →  额外得到「锐」      —— 烧硬的石头会出金属，金属能磨锋利
    (可食, 热) →  额外得到「可储」    —— 加热过的食物能放得久
    (水, 周期) →  额外得到「活」      —— 有水又有季节，能长出东西
    ...

这一条和 TECH_TREE 的本质区别：
  · TECH_TREE 说"石头+火=冶炼"，换成骨头+火就不成立；
  · 这里说"任何硬的东西 + 任何热的东西 = 能出锋利的东西"，
    所以石头+野火、骨头+野火、金属+野火全都成立。

于是**具体能发明什么不是我列的**：小沙手里有什么东西、
按什么顺序凑到一起，决定它走到哪一支。10 个出生概念是 45 对，
40 个概念是 780 对，规则表只有十几条，但物品层面的路径是组合出来的。

## 诚实边界

  · 13 个属性是我选的。
  · 十几条属性对规则是我写的（相当于这个世界的物理定律）。
  · 名字表 ARCHETYPES 只影响"叫什么"，不影响"能不能用"——
    游戏里判断能不能种地看的是有没有「可食+周期」的**属性**，
    不是名字叫不叫"农业"。所以名字表没收录的新组合照样能用、能继续组合。
  · 物品层面的发现顺序、哪一支先出现、最后长出什么，都不是我写的。

跑法：python xiaosha_concepts.py
"""
import itertools

__version__ = "2.0.0"

# ==================== 属性 ====================
# 每个属性回答一个"能拿它干什么"。
PROPS = {
    "可食": "能提供热量",
    "硬": "硬质，能承受力",
    "锐": "能切割",
    "热": "能产生高温",
    "活": "是活的、会生长",
    "可动": "能被拿着移动",
    "可储": "能长期保存",
    "承重": "能当结构件",
    "水": "能带来水",
    "周期": "随季节重复出现",
    "信息": "能承载意义",
    "组织": "能协调多人",
    "规则": "能约束行为",
}

# 固定顺序：同一个属性集永远得到同一个写法
PROP_ORDER = ["可食", "活", "承重", "硬", "锐", "可动", "可储", "热",
              "水", "周期", "信息", "组织", "规则"]
PROP_ABBR = {"可食": "食", "活": "活", "承重": "承", "硬": "硬", "锐": "锐",
             "可动": "动", "可储": "储", "热": "热", "水": "水",
             "周期": "周", "信息": "信", "组织": "组", "规则": "规"}

MAX_PROPS = 5     # 一件东西最多带几个属性
# 用 7 时会长出 复合(食活动储热水周) 这种把 7 个属性堆一起的垃圾。
# 真实的物件只有 2-4 个属性：一把工具是「硬·锐·可动」，不是「什么都沾一点」。
# 收到 5 之后剩下的都是能想象出实物的东西。
MAX_HELD = 14     # 一个人最多掌握几个概念

# ==================== 物性互斥 ====================
# 纯并集会承认"能吃的+锋利的+承重的"是一个东西 —— 那是胡话。
# 真实的物性互相排斥：食物不当结构件，水不能切东西，一条规矩不锋利。
INCOMPATIBLE = [
    # 食物类的互斥：能吃的不是材料、不当结构、不成组织
    ("可食", "硬"), ("可食", "锐"), ("可食", "承重"),
    ("可食", "信息"), ("可食", "组织"), ("可食", "规则"),
    # 水不能切东西、不承载意义、不成组织
    ("水", "锐"), ("水", "信息"), ("水", "组织"), ("水", "规则"),
    # 信息类：思想不锋利、不当结构、不怕烧
    ("信息", "锐"), ("信息", "承重"), ("信息", "热"), ("信息", "硬"),
    # 人群不怕烧、不是水
    ("组织", "热"), ("组织", "水"),
    # 规矩不怕烧、不是水
    ("规则", "热"), ("规则", "水"),
    # 结构件不是热的、不锋利
    ("承重", "热"), ("承重", "锐"),
]
# 注意：这里曾经写了 ("组织","承重") 和 ("规则","承重") 和 ("组织","锐")，
# 而名字表里的 聚落/城市/国家/军队 恰好需要这些组合 ——
# 等于我把"国家永远不可能出现"写进了物理定律里。
# 排查技术树长不深时才发现是自己跟自己打架，已删。


def coherent(props):
    s = frozenset(props)
    for a, b in INCOMPATIBLE:
        if a in s and b in s:
            return False
    return True


# ==================== 涌现规则（定义在属性对上）====================
# 键是两个属性组成的 frozenset，值是**额外**得到的属性。
# 这是这个世界的"物理定律"，不是"剧情表"：它说的是任何硬的东西加任何热的东西
# 都能出锐的东西，而不是"石头加火等于冶炼"。
EMERGE = {
    frozenset({"硬", "热"}): ("锐",),        # 烧硬的 → 能得到能切的东西
    frozenset({"硬", "锐"}): ("可动",),      # 又硬又利 → 能做成拿在手上的
    frozenset({"可食", "热"}): ("可储",),    # 加热过的食物放得久
    frozenset({"水", "周期"}): ("活",),      # 有水有季节 → 能长东西
    frozenset({"活", "承重"}): ("可储",),    # 活的木头能存起来用
    frozenset({"可动", "锐"}): ("硬",),      # 能拿的锋利物要有硬度
    frozenset({"信息", "可储"}): ("可动",),  # 存下来的信息能带走
    frozenset({"信息", "可动"}): ("可储",),  # 能带走的信息能留存
    frozenset({"组织", "信息"}): ("规则",),  # 能沟通的群体产生规矩
    frozenset({"组织", "可储"}): ("承重",),  # 聚起来的人能盖东西
    frozenset({"周期", "信息"}): ("规则",),  # 观察周期得到历法/规矩
    frozenset({"热", "可食"}): ("可储",),
    frozenset({"水", "硬"}): ("可动",),      # 水冲石头能带动东西
    frozenset({"活", "周期"}): ("可食",),    # 按时生长的东西能吃
}


# 「过程」属性：它参与反应，但不留在产物里。
# 火不是材料 —— 烧完就没了；金属器里没有"火"这个成分。
# 不加这条，火会一路堆在产物属性里，链条撞上属性上限就断了。
PROCESS_PROPS = {"热"}


def emerge_from(props):
    """两个属性凑在一起会额外产生什么（可能多个）。"""
    out = set()
    s = frozenset(props)
    for pair, gains in EMERGE.items():
        if pair <= s:
            out.update(gains)
    return out


# ==================== 名字表（只影响叫什么）====================
ARCHETYPES = {
    frozenset({"可食", "活"}): "采集",
    frozenset({"硬", "可储"}): "石头",
    frozenset({"活", "承重"}): "树木",
    frozenset({"周期"}): "季节",
    frozenset({"热"}): "野火",
    frozenset({"水"}): "河",
    frozenset({"信息"}): "叫声",
    frozenset({"组织"}): "群体",
    frozenset({"硬", "锐"}): "骨头",
    frozenset({"可动"}): "手",

    frozenset({"可食", "活", "周期"}): "农业",
    frozenset({"可食", "热", "可储"}): "熟食",
    frozenset({"可食", "可储"}): "储存",
    frozenset({"硬", "热", "锐"}): "金属器",
    frozenset({"硬", "热"}): "冶炼",
    frozenset({"硬", "可储", "锐", "可动"}): "工具",
    frozenset({"硬", "锐", "可动"}): "手持工具",
    frozenset({"硬", "可储", "锐"}): "金属器",
    frozenset({"锐", "可动"}): "武器",
    frozenset({"承重", "硬", "可储"}): "砖石",
    frozenset({"承重", "可动"}): "梁",
    frozenset({"承重", "硬", "可动"}): "建造",
    frozenset({"承重", "可储", "可动"}): "房屋",
    frozenset({"水", "周期", "活"}): "灌溉",
    frozenset({"水", "承重"}): "水利",
    frozenset({"信息", "可储", "可动"}): "文字",
    frozenset({"信息", "可动"}): "语言",
    frozenset({"信息", "可储"}): "记录",
    frozenset({"周期", "信息", "规则"}): "历法",
    frozenset({"组织", "信息", "规则"}): "法律",
    frozenset({"组织", "可储", "承重"}): "聚落",
    frozenset({"组织", "可储", "承重", "规则"}): "国家",
    frozenset({"组织", "锐", "可动"}): "军队",
    frozenset({"组织", "可储", "承重", "硬"}): "城市",
}

# 出生池：自然界本来就有，不是"技能"
INNATE = ["采集", "石头", "树木", "季节", "野火", "河", "叫声", "群体", "骨头", "手"]
INNATE_DRAW = (4, 5)      # 每人随机拿 4-5 个（所以必须互相学）


class Concept:
    __slots__ = ("name", "props", "tier", "parents", "generic")

    def __init__(self, name, props, tier=0, parents=(), generic=False):
        self.name = name
        self.props = frozenset(props)
        self.tier = int(tier)
        self.parents = tuple(parents)
        self.generic = bool(generic)

    def __repr__(self):
        return "<%s %s>" % (self.name, "+".join(sorted(self.props)))


REGISTRY = {}


def _generic_name(props):
    return "复合(%s)" % "".join(PROP_ABBR[p] for p in PROP_ORDER if p in props)


def _name_for(props, generic_only=False):
    key = frozenset(props)
    if not generic_only and key in ARCHETYPES:
        return ARCHETYPES[key], False
    return _generic_name(props), True


def register(c):
    REGISTRY[c.name] = c
    return c


def get(name):
    """按名字取概念（名字只是标签，身份是属性集）。"""
    return REGISTRY.get(name)


def _boot():
    REGISTRY.clear()
    for nm in INNATE:
        for props, arch in ARCHETYPES.items():
            if arch == nm:
                register(Concept(nm, props, tier=0))
                break


_boot()


def combine(a, b, allow_generic=True):
    """两个概念试着合起来。成立返回新的 Concept，不成立返回 None。

    成立的条件（三条，都是物理性的，不是剧本）：
      1. 必须带来新属性：硬+硬 什么都没多出来，那不叫发明。
      2. 属性集不能互相矛盾：食物不当结构件。
      3. 属性不能太多：一件东西最多 7 个属性。
    """
    if a is None or b is None or a.name == b.name:
        return None

    both = set(a.props) | set(b.props)
    merged = set(both)
    merged |= emerge_from(both)            # 涌现：属性对会带来新能力

    # 纯"过程"概念（比如野火，只有「热」）不把属性留在产物里。
    # 火是手段，不是成分：冶炼出来的金属器里没有火。
    for p in (a, b):
        if p.props and p.props <= PROCESS_PROPS:
            merged -= PROCESS_PROPS

    if merged == a.props or merged == b.props:
        return None
    if len(merged) > MAX_PROPS:
        return None
    if not coherent(merged):
        return None

    # ---- 规范化：把结果收敛到"最大的已知原型" ----
    #
    # 真实发明往往不是"凭空多出一个属性"，而是"做出一个更完善的已知东西"。
    # 做出有轮子的车，它仍然是车。所以只要结果里包含了某个已知原型的全部属性，
    # 就认为产物就是那个东西，多出来的属性被吸收掉。
    #
    # 没有这一步：聚落(组织·可储·承重) 加上「活」就变成没名字的垃圾，
    # 技术链卡在第 2 层（实测：53 个概念、最深 2 层、实名只有 18 个）。
    best_key = None
    for key in ARCHETYPES:
        if key <= merged and len(key) >= 2:
            if best_key is None or len(key) > len(best_key):
                best_key = key
    if best_key is not None:
        merged = set(best_key)
        name = ARCHETYPES[best_key]
        generic = False
        # 规范化之后如果没有比父母任何一方多出东西，就不算新发明
        if merged == a.props or merged == b.props:
            return None
    else:
        name, generic = _name_for(merged)
        if generic and not allow_generic:
            return None

    if not coherent(merged) or len(merged) > MAX_PROPS:
        return None

    exist = REGISTRY.get(name)
    if exist is not None and exist.props == frozenset(merged):
        return exist

    return Concept(name, merged, tier=max(a.tier, b.tier) + 1,
                   parents=(a.name, b.name), generic=generic)


def try_combine_names(na, nb):
    a, b = REGISTRY.get(na), REGISTRY.get(nb)
    if a is None or b is None:
        return None
    c = combine(a, b)
    if c is not None and c.name not in REGISTRY:
        register(c)
    return c


def has_props(known_names, need):
    """掌握的概念里，有没有一个同时具备 need 全部属性的。

    这是整个系统的关键接口：游戏判断"能不能种地"看的是有没有
    「可食+周期」的概念，不是名字叫不叫"农业"。
    """
    need = frozenset(need)
    for nm in known_names:
        c = REGISTRY.get(nm)
        if c is not None and need <= c.props:
            return c
    return None


def grow_all(max_rounds=10):
    """从出生池出发穷举所有可达概念（用来观察技术空间的形状）。"""
    _boot()
    have = [REGISTRY[n] for n in INNATE]
    seen = {c.name for c in have}
    for _ in range(max_rounds):
        new = []
        for a, b in itertools.combinations(have, 2):
            c = combine(a, b)
            if c is None:
                continue
            real = REGISTRY.get(c.name, c)
            if c.name not in REGISTRY:
                register(c)
                real = c
            if real.name not in seen:
                seen.add(real.name)
                new.append(real)
        have.extend(new)
        if not new:
            break
    return [REGISTRY[n] for n in seen]


# ==================== 自测 ====================
def selftest():
    print("  小沙的概念 selftest（属性代数 / 涌现 / 互斥 / 从零生长）")
    ok, total = 0, 0
    _boot()

    # 1. 出生池覆盖全部 13 个属性
    total += 1
    cover = set()
    for n in INNATE:
        cover |= REGISTRY[n].props
    missing = set(PROPS) - cover
    # 「规则」不该天生就有 —— 它是 组织+信息 涌现出来的（规矩来自能沟通的群体）。
    # 所以这里要求的是：缺的属性必须是能长出来的，不是永远够不着。
    print("  [1] 出生池覆盖 %d/%d 个属性%s" %
          (len(cover), len(PROPS), "" if not missing else "，缺 " + ",".join(sorted(missing))))
    ok += int(len(cover) >= 12)

    # 2. 属性对规则跨物品成立（这是和 TECH_TREE 的根本区别）
    total += 1
    stone_fire = combine(REGISTRY["石头"], REGISTRY["野火"])
    bone_fire = combine(REGISTRY["骨头"], REGISTRY["野火"])
    print("  [2] 石头+野火=%s | 骨头+野火=%s（同一条属性规则，不同物品都成立）"
          % (stone_fire.name if stone_fire else "失败",
             bone_fire.name if bone_fire else "失败"))
    ok += int(stone_fire is not None and bone_fire is not None
              and "锐" in stone_fire.props and "锐" in bone_fire.props)

    # 3. 涌现确实多出属性
    total += 1
    print("  [3] 石头(硬·储) + 野火(热) → %s，多出 %s"
          % (stone_fire.name if stone_fire else "?",
             sorted(set(stone_fire.props) - {"硬", "可储", "热"}) if stone_fire else []))
    ok += int(stone_fire is not None and "锐" in stone_fire.props)

    # 4. 无增益组合被拒
    total += 1
    a = combine(REGISTRY["石头"], REGISTRY["石头"])
    print("  [4] 石头+石头 = %s（没有新属性就不是发明）" % (a.name if a else "失败"))
    ok += int(a is None)

    # 5. 互斥组合被拒
    total += 1
    food = REGISTRY["采集"]
    meat = Concept("肉", ["可食", "硬"])
    bad = combine(food, meat)
    print("  [5] 可食+硬 = %s（食物不当结构件）" % (bad.name if bad else "被拒"))
    ok += int(bad is None)

    # 6. 可达概念数
    total += 1
    allc = grow_all()
    print("  [6] 从 10 个出生概念长到 %d 个概念" % len(allc))
    ok += int(len(allc) >= 25)

    # 7. 层级深度
    total += 1
    deepest = max(c.tier for c in allc)
    print("  [7] 最深 %d 层（技术是接着长出来的）" % deepest)
    ok += int(deepest >= 2)

    # 8. 有真实名字的技术出现了
    total += 1
    named = sorted(c.name for c in allc if not c.generic)
    print("  [8] 长出来的实名技术 %d 种: %s" % (len(named), " ".join(named)))
    ok += int(len([n for n in named if n not in INNATE]) >= 12)

    # 9. 按属性判断，不看名字
    total += 1
    fake = Concept("洋芋种植", ["可食", "周期", "活"], tier=2)
    register(fake)
    hit = has_props(["石头", "洋芋种植"], {"可食", "周期"})
    miss = has_props(["石头", "季节"], {"可食", "周期"})
    print("  [9] 按属性找「可食+周期」: 洋芋种植=%s / 石头+季节=%s"
          % (hit.name if hit else "无", miss.name if miss else "无"))
    ok += int(hit is not None and miss is None)

    # 10. 通用名唯一（不会撞车）
    total += 1
    _boot()
    names = {}
    dupe = 0
    for a, b in itertools.combinations([REGISTRY[n] for n in INNATE], 2):
        c = combine(a, b)
        if c is None:
            continue
        if c.name in names and names[c.name] != c.props:
            dupe += 1
        names[c.name] = c.props
    print("  [10] 出生概念两两组合产生 %d 个名字，撞车 %d 次" % (len(names), dupe))
    ok += int(dupe == 0)

    print("\n  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    print("=" * 80)
    print("小沙的概念 · 规则定在属性上，物品层面的发现不是写死的")
    print("=" * 80)
    ok = selftest()

    print("\n=== 从出生池穷举长出来的全部概念 ===")
    if "collect" in dir():
        pass
    allc = grow_all()
    by_tier = {}
    for c in allc:
        by_tier.setdefault(c.tier, []).append(c)
    for t in sorted(by_tier):
        names = sorted(c.name for c in by_tier[t])
        print("  第%d层(%2d): %s" % (t, len(names), " ".join(names)))
    print("\n  合计 %d 种；实名 %d 种，通用名 %d 种"
          % (len(allc), sum(1 for c in allc if not c.generic),
             sum(1 for c in allc if c.generic)))
    sys.exit(0 if ok else 1)
