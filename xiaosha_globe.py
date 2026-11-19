# -*- coding: utf-8 -*-
"""小沙的地球文明 —— 把 Civilization 放到球面上跑。

做法：不重写文明引擎，只换掉"世界"这一层。

Agent 只用到 world 的 9 个东西：
    passable / harvest / food / farm_cap / terrain / minerals
    size / altars_near / build_altar

所以写一个 GlobeWorld 把球面包装成同样的接口，Civilization 一行不用改
（只加了一个 world= 参数），350 个小沙就站在球面上了。

坐标约定：
    x = 经度列（0..N-1，**环绕**：x=N 回到 x=0）
    y = 纬度行（0=南极，N-1=北极，**钳制**：越界就停在极点）
    对外暴露的 food / terrain / farm_cap 是 Planet 的转置视图，
    所以 agent 用 [x, y] 索引时，实际读写的是 planet[lat=y, lon=x]，
    和 passable(x,y) / harvest(x,y) 的约定一致。

球面带来的真实差异：
  1. 走到最东边会从最西边出来 —— 没有世界边缘这堵墙
  2. 赤道格子面积是极点的 38 倍 —— 同样的格子数，极地能养的人少得多
  3. 昼夜由太阳经度决定：一半白天一半黑夜，夜里几乎采不到东西
  4. 季节由赤纬决定：夏冬的产出不同，两极有极昼极夜

跑法：python xiaosha_globe.py [代数] [人口]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from xiaosha_planet import Planet, MAX_FOOD, REGEN, PASSABLE, TERRAIN_NAMES
from xiaosha_world import ALTAR_MINERAL, WORSHIP_RANGE
from xiaosha_civilization import Civilization

__version__ = "1.0.0"


class GlobeWorld:
    """把 Planet 包装成 World 的接口（方形 N×N，x=经度环绕，y=纬度钳制）。"""

    def __init__(self, size=120, seed=42):
        self.size = int(size)
        self.planet = Planet(n_lat=self.size, n_lon=self.size, seed=seed)
        p = self.planet

        # 转置视图：agent 用 [x, y] 读写 → 实际是 planet[lat=y, lon=x]
        self.food = p.food.T
        self.terrain = p.terrain.T
        # 耕地承载力：初始等于自然上限，种地会把它抬高。
        # Planet 没有这一层（它只有 food_cap 自然上限），所以球面版自己维护一份，
        # 否则农业抬高多少都会被 regrow 按自然上限压回去 —— 平原版踩过这个坑。
        self.farm_cap_ll = p.food_cap.copy().astype(np.float32)     # [lat, lon]
        self.farm_cap = self.farm_cap_ll.T                          # [x, y] 视图

        # 矿物：山上多（但山不可通行），山脚给够建祭坛的量。
        # Planet 没有矿物层，这里按地形算一份。
        m = np.zeros((self.size, self.size), dtype=np.float32)     # [lat, lon]
        mt = p.terrain == 6                                          # MOUNTAIN
        m[mt] = 8.0
        for dlat in (-1, 0, 1):
            for dlon in (-1, 0, 1):
                if dlat == 0 and dlon == 0:
                    continue
                sh = np.roll(np.roll(mt, dlat, 0), dlon, 1)
                m[sh & (p.terrain == 4)] += 4.5                      # 山脚草地
        self.minerals = m

        self.altars = []          # (x, y, builder_id, generation)
        self._build_ore_index()   # 预计算矿区，寻矿不用扫 15x15

    # ---------- 时间 ----------
    @property
    def step_count(self):
        return self.planet.step_count

    @step_count.setter
    def step_count(self, v):
        self.planet.step_count = int(v)

    def daylight(self):
        return self.planet.daylight()

    def light_at(self, x, y):
        """某一格当前的光照 0-1（夜里 ~0）。"""
        i, j = self._row(y), self._col(x)
        return float(self.planet.daylight()[i, j])

    def season(self):
        return self.planet.season()

    def day_number(self):
        return self.planet.step_count // self.planet.steps_per_day

    # ---------- 坐标 ----------
    def _row(self, y):
        """纬度行号：钳制到 [0, size-1]。

        这里**不能**用 np.clip。它会被调用几百万次（每个个体每步都要问
        passable / light_at / food_norm），numpy 的标量包装开销远大于运算本身。
        profile 显示 3 代模拟 68 秒里，np.clip 占了 43.8 秒、被调 347 万次。
        纯 Python 比较快 30 倍以上。
        """
        if y < 0:
            return 0
        if y >= self.size:
            return self.size - 1
        return int(y)

    def _col(self, x):
        """经度列号：环绕。"""
        return int(x) % self.size

    def normalize(self, x, y):
        """经度环绕、纬度钳制。agent 每次移动后都要过一下。"""
        return self._col(x), self._row(y)

    def passable(self, x, y):
        i, j = self._row(y), self._col(x)
        return PASSABLE[int(self.terrain[j, i])]

    def can_farm(self, x, y):
        """球面版：可通行的陆地才能种（海洋/山地不行）。"""
        i, j = self._row(y), self._col(x)
        return bool(PASSABLE[int(self.terrain[j, i])])

    def natural_food(self, x, y):
        """自然承载力（不含种地加成）。"""
        i = self._row(y)
        j = self._col(x)
        return float(self.planet.food_cap[i, j])

    def _build_ore_index(self):
        """把所有可通行、矿物够建祭坛的格子记下来。

        _try_altar 原来为了找最近的矿要扫 15x15=225 格，每格还调一次 passable。
        profile 显示光这条路径在 3 代里就跑了 65 万次 passable、占 20 秒。
        矿不会新增，所以开局算一次，之后只在这个小列表里找。
        """
        self.ore_tiles = []
        for yy in range(self.size):
            for xx in range(self.size):
                if self.minerals[xx, yy] >= ALTAR_MINERAL and self.passable(xx, yy):
                    self.ore_tiles.append((xx, yy))
        return len(self.ore_tiles)

    def nearest_ore(self, x, y, max_d=7):
        """最近的矿区方向（经度按最短弧）。返回 (dx, dy) 或 None。"""
        best = None
        for (ox, oy) in self.ore_tiles:
            if self.minerals[ox, oy] < ALTAR_MINERAL:
                continue
            ax = self._lon_delta(ox, x)
            ay = abs(oy - y)
            if ax > max_d or ay > max_d:
                continue
            d2 = ax * ax + ay * ay
            if best is None or d2 < best[0]:
                raw = (ox - x) % self.size
                sdx = raw if raw <= self.size // 2 else raw - self.size
                best = (d2, sdx, oy - y)
        if best is None:
            return None
        return best[1], best[2]

    def food_norm(self, x, y):
        """感知归一化用的分母：用耕地承载力（种过地会更高）。

        上一次替换把 `def food_norm` 这一行吃掉，它的函数体被并进了
        natural_food，于是 natural_food 返回的是「耕地承载力」——
        种地的加成会一次次叠加到上限上，而且 food_norm 根本不存在，
        感知只能退回默认值 6.0。这里恢复成两个独立的方法。
        """
        i = self._row(y)
        j = self._col(x)
        return float(self.farm_cap[j, i])

    def harvest(self, x, y, amount):
        i, j = self._row(y), self._col(x)
        got = float(min(amount, self.food[j, i]))
        self.food[j, i] -= got
        return got

    # ---------- 再生（不碰 step_count，由 Civilization 统一推进）----------
    def regrow(self, rate=1.0):
        p = self.planet
        light = p.daylight()
        for k, base in REGEN.items():
            m = p.terrain == k
            if not m.any():
                continue
            gain = base * rate * (0.25 + 0.75 * light) * p.cell_area[:, None]
            # 上限用耕地承载力（farm_cap_ll），不是自然上限
            p.food[m] = np.minimum(self.farm_cap_ll[m], p.food[m] + gain[m])

    def total_food(self):
        return float(self.food.sum())

    # ---------- 祭坛（经度环绕，用最短弧）----------
    def _lon_delta(self, a, b):
        d = abs(int(a) - int(b)) % self.size
        return min(d, self.size - d)

    def altars_near(self, x, y, r=WORSHIP_RANGE):
        out = []
        for (ax, ay, _bid, _gen) in self.altars:
            if self._lon_delta(ax, x) <= r and abs(ay - y) <= r:
                out.append((ax, ay, _bid, _gen))
        return out

    def build_altar(self, x, y, builder):
        i, j = self._row(y), self._col(x)
        if self.minerals[j, i] < ALTAR_MINERAL:
            return False
        self.minerals[j, i] -= ALTAR_MINERAL
        self.altars.append((j, i, getattr(builder, "id", -1),
                            getattr(builder, "generation", 0)))
        return True

    def miracle(self, x, y, radius=6, energy=25.0):
        """人类降下神迹：一片区域食物暴涨。"""
        n = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                j, i = self.normalize(x + dx, y + dy)
                k = int(self.terrain[j, i])
                self.food[j, i] = min(MAX_FOOD[k] * 2.0, self.food[j, i] + energy)
                n += 1
        return n

    # ---------- 统计 ----------
    def lat_of(self, y):
        return float(self.planet.lat[self._row(y)])

    def stats(self):
        p = self.planet
        counts = {TERRAIN_NAMES[k]: int((p.terrain == k).sum()) for k in TERRAIN_NAMES}
        land = p.terrain != 0
        land_area = float((p.area_grid * land).sum())
        return {
            "size": self.size,
            "terrain": counts,
            "land_frac": round(land_area / max(1e-9, float(p.area_grid.sum())), 3),
            "food_total": round(float(self.food.sum()), 1),
            "day": self.day_number(),
            "season": self.season(),
            "sun_lon": round(float(p.sun_lon()), 1),
            "declination": round(float(p.declination()), 1),
            "altars": len(self.altars),
        }


class GlobeCivilization(Civilization):
    """球面上的文明。除了换世界，其余逻辑完全继承。"""

    def __init__(self, size=120, n_agents=120, seed=42,
                 food_rate=1.0, mutation_rate=0.06, verbose=False):
        world = GlobeWorld(size=size, seed=seed)
        super().__init__(size=size, n_agents=n_agents, seed=seed,
                         food_rate=food_rate, mutation_rate=mutation_rate,
                         verbose=verbose, world=world)


# ==================== 分析：地理分化 ====================
def latitude_bands(civ, n_bands=6):
    """按纬度带统计人口 / 食物 / 表型。看的是"地理是否真的造成分化"。"""
    w = civ.world
    N = w.size
    rows = [[] for _ in range(n_bands)]
    for a in civ.agents:
        b = min(n_bands - 1, int(a.y / N * n_bands))
        rows[b].append(a)

    out = []
    for b in range(n_bands):
        y_mid = (b + 0.5) / n_bands * N
        lat = w.lat_of(y_mid)
        grp = rows[b]
        out.append({
            "band": b,
            "lat": round(lat, 1),
            "pop": len(grp),
            "food": round(float(w.food[:, int(y_mid) % N].sum()), 1),
            "size": round(float(np.mean([a.pheno.size for a in grp])), 2) if grp else 0,
            "night": round(float(np.mean([a.pheno.night for a in grp])), 2) if grp else 0,
            "life": int(np.mean([a.pheno.life for a in grp])) if grp else 0,
            "light": round(float(w.daylight()[int(y_mid) % N].mean()), 2),
        })
    return out


def main(gens=30, pop=160, size=120, seed=42):
    print("=" * 84)
    print("小沙的地球 · 球面文明")
    print("=" * 84)
    np.random.seed(seed)
    civ = GlobeCivilization(size=size, n_agents=pop, seed=seed, verbose=False)
    s = civ.world.stats()
    print("世界: %dx%d = %d 格 | 陆地 %.0f%% | 地形 %s"
          % (size, size, size * size, s["land_frac"] * 100,
             " ".join("%s%d" % (k, v) for k, v in s["terrain"].items())))
    print("初始: %d 个个体\n" % len(civ.agents))

    print("  代 | 人口 | 天数 | 季节 | 食物 | 祭坛 | 知识 | 部落 | 赤道人 | 极地人")
    for g in range(gens):
        civ.generation = g
        for _ in range(60):
            civ.step()
        if g % 5 == 0 or g == gens - 1:
            bands = latitude_bands(civ, 6)
            equator = bands[2]["pop"] + bands[3]["pop"]
            polar = bands[0]["pop"] + bands[5]["pop"]
            allk = set()
            for a in civ.agents:
                allk.update(a.knowledge)
            print("  %2d | %4d | %4d | %2s | %6.0f | %4d | %3d | %2d | %4d | %4d"
                  % (g, len(civ.agents), civ.world.day_number(), civ.world.season(),
                     civ.world.total_food(), len(civ.world.altars), len(allk),
                     len(civ.tribes), equator, polar))

    print("\n=== 纬度带分化（第 %d 代）===" % (gens - 1))
    print("  纬度 | 人口 | 该带食物 | 平均体型 | 夜行性 | 平均寿命 | 光照")
    for b in latitude_bands(civ, 6):
        print("  %+5.1f° | %4d | %7.0f | %5.2f | %5.2f | %5d | %.2f"
              % (b["lat"], b["pop"], b["food"], b["size"], b["night"],
                 b["life"], b["light"]))

    # 极地 vs 赤道：人口密度（每格）
    bands = latitude_bands(civ, 6)
    eq = (bands[2]["pop"] + bands[3]["pop"]) / (2 * size)
    po = (bands[0]["pop"] + bands[5]["pop"]) / (2 * size)
    print("\n  每格人口: 赤道 %.3f vs 极地 %.3f（比值 %.1f×）"
          % (eq, po, eq / po if po > 1e-9 else float("inf")))
    print("  赤道格子面积是极地的 %.0f× —— 同样的格子数，极地能养的人本来就少"
          % (float(civ.world.planet.cell_area[size // 2])
             / max(1e-9, float(civ.world.planet.cell_area[0]))))
    return civ


# ==================== 自测 ====================
def selftest():
    print("  小沙的球面文明 selftest（环绕 / 钳制 / 转置一致 / 昼夜 / 极地人少）")
    ok, total = 0, 0

    w = GlobeWorld(size=40, seed=7)

    # ---- 1. 经度环绕 ----
    total += 1
    a = w.normalize(40, 10)
    b = w.normalize(-1, 10)
    print("  [1] 环绕: x=40→%d, x=-1→%d" % (a[0], b[0]))
    ok += int(a[0] == 0 and b[0] == 39)

    # ---- 2. 纬度钳制 ----
    total += 1
    hi = w.normalize(5, 100)[1]
    lo = w.normalize(5, -9)[1]
    print("  [2] 钳制: y=100→%d, y=-9→%d" % (hi, lo))
    ok += int(hi == 39 and lo == 0)

    # ---- 3. 转置视图与 planet 一致（写穿检查）----
    total += 1
    w.food[3, 5] = 1.234
    same = abs(float(w.planet.food[5, 3]) - 1.234) < 1e-6
    print("  [3] food[x=3,y=5] 写入后 planet[lat=5,lon=3] = %.3f（转置一致=%s）"
          % (float(w.planet.food[5, 3]), same))
    ok += int(same)

    # ---- 4. passable 与 terrain 同源 ----
    total += 1
    mism = 0
    for _ in range(200):
        x, y = int(np.random.randint(0, 40)), int(np.random.randint(0, 40))
        t = int(w.terrain[x, y])
        if w.passable(x, y) != PASSABLE[t]:
            mism += 1
    print("  [4] passable 与 terrain 一致（200 次抽样，不符 %d）" % mism)
    ok += int(mism == 0)

    # ---- 5. 采集会真的减少那格食物 ----
    total += 1
    cand = [(x, y) for x in range(0, 40, 3) for y in range(0, 40, 3) if w.passable(x, y)]
    x, y = cand[0]
    w.food[x, y] = 5.0
    got = w.harvest(x, y, 2.0)
    print("  [5] 采集: 拿到 %.2f，剩 %.2f" % (got, w.food[x, y]))
    ok += int(abs(got - 2.0) < 1e-6 and abs(w.food[x, y] - 3.0) < 1e-6)

    # ---- 6. 光照随时间变化（昼夜是真的）----
    total += 1
    # 取太阳正下方那一列，才能看出昼夜差异。
    # 我第一次固定取 x=10，结果正午和半夜都是 0.15 —— 因为那格离太阳
    # 正好相隔 90°，两个时刻的太阳高度角一样低，是我测试选错了格子。
    w.step_count = 0
    x_sun_day = w.planet.lon_index(w.planet.sun_lon())
    l1 = w.light_at(x_sun_day, 20)
    w.step_count = w.planet.steps_per_day // 2
    x_sun_night = w.planet.lon_index(w.planet.sun_lon())
    l2 = w.light_at(x_sun_day, 20)      # 同一格，半天之后
    l3 = w.light_at(x_sun_night, 20)    # 此刻太阳正下方那一格
    print("  [6] 同一格: 正午 %.2f → 半夜 %.2f（此时太阳正下方 %.2f）" % (l1, l2, l3))
    ok += int(abs(l1 - l2) > 0.5 and l3 > 0.7)

    # ---- 7. 祭坛距离用最短弧（跨 0 度经线也算近）----
    total += 1
    w.altars = [(39, 20, 1, 0)]
    near_across = len(w.altars_near(0, 20, r=2))      # 39 和 0 相差 1 格
    far = len(w.altars_near(20, 20, r=2))
    print("  [7] 跨经线: 祭坛在 x=39，从 x=0 看 %d 个，从 x=20 看 %d 个"
          % (near_across, far))
    ok += int(near_across == 1 and far == 0)

    # ---- 8. 建祭坛要矿物，建完扣掉 ----
    total += 1
    w2 = GlobeWorld(size=40, seed=7)
    # 矿物最多的格子是山本身（不可通行），真正能建祭坛的是山脚。
    # 所以必须筛"可通行"，不能直接取 argwhere 的第一个。
    mt = [(int(a), int(b)) for a, b in np.argwhere(w2.minerals >= ALTAR_MINERAL)
          if w2.passable(int(a), int(b))]
    good = False
    if mt:
        x, y = mt[0]
        before = float(w2.minerals[x, y])
        good = w2.build_altar(x, y, type("B", (), {"id": 1, "generation": 0})())
        after = float(w2.minerals[x, y])
        good = good and after < before
        print("       找到 %d 个可通行的矿区，用 (x=%d,y=%d) 测试" % (len(mt), x, y))
    print("  [8] 建祭坛: 矿物够的格子里有可通行处并能扣费 = %s" % bool(good))
    ok += int(bool(good))

    # ---- 9. 真种群能跑起来，且赤道人比极地多 ----
    total += 1
    np.random.seed(11)
    civ = GlobeCivilization(size=60, n_agents=120, seed=11, verbose=False)
    for g in range(6):
        civ.generation = g
        for _ in range(60):
            civ.step()
    bands = latitude_bands(civ, 6)
    eq = bands[2]["pop"] + bands[3]["pop"]
    po = bands[0]["pop"] + bands[5]["pop"]
    print("  [9] 6 代后: 人口 %d | 赤道带 %d 人 vs 极地带 %d 人"
          % (len(civ.agents), eq, po))
    ok += int(len(civ.agents) > 20 and eq > po)

    # ---- 10. 食物再生受光照影响（夜里慢）----
    total += 1
    wa, wb = GlobeWorld(size=40, seed=7), GlobeWorld(size=40, seed=7)
    wa.food[:] = 0.0
    wb.food[:] = 0.0
    wa.step_count = 0
    wb.step_count = wa.planet.steps_per_day // 2
    ba, bb = wa.total_food(), wb.total_food()
    wa.regrow(1.0)
    wb.regrow(1.0)
    ga, gb = wa.total_food() - ba, wb.total_food() - bb
    print("  [10] 一步再生: 白天 %.0f vs 夜里 %.0f" % (ga, gb))
    ok += int(ga > gb)

    print("\n  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    # 默认只跑自测（几秒钟）。要跑几十代的长任务，必须显式给代数：
    #     python xiaosha_globe.py 30 160
    # 这是本仓库的约定（make_civ_video 也是这样）：
    # 否则每次全量测试都会白跑几十代，把测试从 7 分钟拖到 20 分钟。
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print("=" * 84)
        print("小沙的地球 · 球面文明")
        print("=" * 84)
        sys.exit(0 if selftest() else 1)
    main(gens=int(args[0]), pop=int(args[1]) if len(args) > 1 else 160)
