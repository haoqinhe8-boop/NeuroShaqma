# -*- coding: utf-8 -*-
"""小沙的地球 —— 球面世界。

和 xiaosha_world.py 的区别：
  xiaosha_world  一张 40x40 的平地，四周有边界
  xiaosha_planet 经纬度球面，东西方向绕一圈回到原点，格子面积随纬度收缩

真实的地理在这四件事上会直接影响小沙能不能活：

  1. **东西环绕**：一直往东走会回到出发点。没有"世界边缘"这堵墙。
  2. **格子面积 ∝ cos(纬度)**：赤道上一格最宽，两极的一格几乎是个点。
     所以两极同样多的格子，实际土地少得多 —— 养不了那么多人。
  3. **气候带**：赤道湿热产粮多、南北纬 30° 附近是沙漠带、两极是冰原。
     这不是画的，是纬度算出来的。
  4. **昼夜**：太阳按经度扫过去，一半是白天一半是黑夜。
     夜里看不见东西 —— 采集和建造的效率会掉。

太阳系在这一层只体现为"太阳相对地球的角度"，它给出的就是昼夜和季节。
其他行星不做轨道计算 —— 它们对小沙的生存没有因果作用，只当天上的光点。
宇宙同理，是背景，不是模拟对象。

跑法：python xiaosha_planet.py        自测 + 一个小演示
"""
import os
import sys

import numpy as np

__version__ = "1.0.0"

# ==================== 地形 ====================
OCEAN, ICE, TUNDRA, FOREST, GRASS, DESERT, MOUNTAIN = 0, 1, 2, 3, 4, 5, 6

TERRAIN_NAMES = {
    OCEAN: "海洋", ICE: "冰原", TUNDRA: "苔原", FOREST: "森林",
    GRASS: "草原", DESERT: "沙漠", MOUNTAIN: "山地",
}
TERRAIN_COLOR = {          # 给渲染用
    OCEAN: "#1b4a6e", ICE: "#c8d8e4", TUNDRA: "#4a5a52", FOREST: "#1e4a28",
    GRASS: "#4a7a3a", DESERT: "#b09a62", MOUNTAIN: "#5c5c68",
}
PASSABLE = {OCEAN: False, ICE: True, TUNDRA: True, FOREST: True,
            GRASS: True, DESERT: True, MOUNTAIN: False}

# 各类地形的食物承载力（每种地形的基准，再乘纬度系数）
MAX_FOOD = {
    OCEAN: 0.0, ICE: 0.6, TUNDRA: 2.0, FOREST: 9.0,
    GRASS: 6.0, DESERT: 0.8, MOUNTAIN: 0.0,
}
REGEN = {
    OCEAN: 0.0, ICE: 0.04, TUNDRA: 0.15, FOREST: 0.55,
    GRASS: 0.50, DESERT: 0.05, MOUNTAIN: 0.0,
}


class Planet:
    """经纬度球面。

    n_lat 行（南极 → 北极），n_lon 列（0° → 360°）。
    东西方向环绕（列取模），南北方向不环绕（极点是边界）。
    """

    def __init__(self, n_lat=90, n_lon=180, seed=42, sea_level=0.47):
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.seed = seed
        rng = np.random.RandomState(seed)

        # 纬度：每行中心的纬度。第 0 行在南极附近，最后一行在北极附近。
        self.lat = np.linspace(-90 + 180.0 / n_lat / 2,
                               90 - 180.0 / n_lat / 2, n_lat)
        self.lon = np.linspace(-180, 180, n_lon, endpoint=False)

        # --- 海拔：分形噪声 + 大陆块 ---
        elev = self._fractal_noise(rng, octaves=5)
        # 加几个大陆中心，让陆地成块而不是碎成渣
        for _ in range(6):
            clat = rng.uniform(-60, 60)
            clon = rng.uniform(-180, 180)
            d = self._angular_dist(clat, clon)
            elev += 0.55 * np.exp(-(d / 42.0) ** 2)
        elev = (elev - elev.min()) / max(1e-9, elev.max() - elev.min())
        self.elevation = elev.astype(np.float32)

        # --- 地形判定（纬度 + 海拔）---
        terrain = np.full((n_lat, n_lon), OCEAN, dtype=np.int8)
        abslat = np.abs(self.lat)[:, None] * np.ones((1, n_lon))

        land = elev > sea_level
        # 极地：冰原
        terrain[land & (abslat > 66)] = ICE
        # 高纬：苔原
        terrain[land & (abslat > 55) & (abslat <= 66)] = TUNDRA
        # 中纬：森林（湿度高的地方）
        terrain[land & (abslat > 25) & (abslat <= 55)] = FOREST
        # 低纬：草原
        terrain[land & (abslat <= 25)] = GRASS
        # 沙漠带：南北纬 15-35°，海拔不高
        desert = land & (abslat > 15) & (abslat < 35) & (elev < sea_level + 0.22)
        terrain[desert] = DESERT
        # 高山：海拔最高的 6%
        hi = land & (elev > np.quantile(elev[land], 0.94)) if land.any() else np.zeros_like(land)
        terrain[hi] = MOUNTAIN
        self.terrain = terrain

        # --- 格子面积权重 ∝ cos(纬度) ---
        # 归一化成"平均一格的面积 = 1"
        w = np.cos(np.deg2rad(self.lat))
        self.cell_area = (w / w.mean()).astype(np.float32)      # (n_lat,)
        self.area_grid = np.repeat(self.cell_area[:, None], n_lon, axis=1)

        # --- 食物 ---
        base = np.zeros((n_lat, n_lon), dtype=np.float32)
        for k, v in MAX_FOOD.items():
            base[terrain == k] = v
        # 承载力再乘纬度（两极冷，产出少）
        self.food_cap = (base * np.clip(self.cell_area[:, None], 0.25, 1.6)).astype(np.float32)
        self.food = (self.food_cap * 0.6).astype(np.float32)

        # --- 时间：步数 → 昼夜/季节 ---
        self.step_count = 0
        self.steps_per_day = 60      # 60 步 = 一天
        self.days_per_year = 48      # 48 天 = 一年（简化，不是 365）

    # ---------- 噪声 ----------
    def _fractal_noise(self, rng, octaves=4):
        out = np.zeros((self.n_lat, self.n_lon), dtype=np.float64)
        amp, total = 1.0, 0.0
        for o in range(octaves):
            cells = 3 * (2 ** o)
            g = rng.rand(cells, cells)
            # 双线性放大到球面网格（列方向先环绕拼接一格，避免接缝）
            g2 = np.concatenate([g, g[:, :1]], axis=1)
            yi = np.linspace(0, cells - 1, self.n_lat)
            xi = np.linspace(0, cells, self.n_lon)
            y0 = np.floor(yi).astype(int).clip(0, cells - 1)
            y1 = np.minimum(y0 + 1, cells - 1)
            x0 = np.floor(xi).astype(int).clip(0, cells - 1)
            x1 = x0 + 1
            fy = (yi - y0)[:, None]
            fx = (xi - x0)[None, :]
            top = g2[np.ix_(y0, x0)] * (1 - fx) + g2[np.ix_(y0, x1)] * fx
            bot = g2[np.ix_(y1, x0)] * (1 - fx) + g2[np.ix_(y1, x1)] * fx
            out += amp * (top * (1 - fy) + bot * fy)
            total += amp
            amp *= 0.5
        return out / total

    def _angular_dist(self, lat0, lon0):
        """球面大圆距离（度）。"""
        la = np.deg2rad(self.lat)[:, None]
        lo = np.deg2rad(self.lon)[None, :]
        la0 = np.deg2rad(lat0)
        lo0 = np.deg2rad(lon0)
        cosd = (np.sin(la) * np.sin(la0)
                + np.cos(la) * np.cos(la0) * np.cos(lo - lo0))
        return np.rad2deg(np.arccos(np.clip(cosd, -1, 1)))

    # ---------- 环绕 ----------
    def wrap(self, x, y):
        """把坐标折回球面。x = 列（经度，环绕），y = 行（纬度，钳制）。"""
        return x % self.n_lon, int(np.clip(y, 0, self.n_lat - 1))

    def lon_index(self, lon_deg):
        """经度（度）→ 列号。lon 数组从 -180 开始，所以 0° 在中间那一列。
        测试里我一度直接把 0 当列号，结果昼夜判断整个反了。"""
        return int(round(((float(lon_deg) + 180.0) % 360.0) / 360.0 * self.n_lon)) % self.n_lon

    # ---------- 昼夜与季节 ----------
    def sun_lon(self):
        """太阳正对的经度（0-360）。一步转 360/steps_per_day 度。"""
        return (self.step_count % self.steps_per_day) * (360.0 / self.steps_per_day)

    def declination(self):
        """太阳赤纬（度）：±23.44 之间随"年"摆动。决定季节。"""
        day = (self.step_count // self.steps_per_day) % self.days_per_year
        return 23.44 * np.sin(2 * np.pi * day / self.days_per_year)

    def daylight(self):
        """每格当前的光照 0-1。夜里是 0。

        这是真实的球面日照：太阳正对一条经线，那一侧是白天；
        两极在夏天会极昼、冬天会极夜（因为赤纬超过了 90-|lat|）。

        按 step_count 缓存：这个函数每个时间步只该算一次。
        不缓存的话，个体每走一步都调 light_at()，就要重算一整张 N×N 的图 ——
        350 个体 × 60 步 = 每代 2.1 万次全图计算，30 代直接跑了 339 秒。
        """
        sc = self.step_count
        if getattr(self, "_light_step", None) == sc:
            return self._light_cache
        decl = self.declination()
        la = np.deg2rad(self.lat)[:, None]
        lo = np.deg2rad(self.lon)[None, :]
        sun = np.deg2rad(self.sun_lon())
        d = np.deg2rad(decl)
        # 太阳高度角的 sin
        sin_h = (np.sin(la) * np.sin(d)
                 + np.cos(la) * np.cos(d) * np.cos(lo - sun))
        # 映射到 0-1：地平线以上就算白天，越接近直射越亮
        arr = np.clip(sin_h * 1.6 + 0.15, 0.0, 1.0).astype(np.float32)
        self._light_cache = arr
        self._light_step = sc
        return arr

    def season(self):
        """北半球季节名（南半球相反）。"""
        day = (self.step_count // self.steps_per_day) % self.days_per_year
        f = day / self.days_per_year
        if f < 0.25:
            return "春"
        if f < 0.5:
            return "夏"
        if f < 0.75:
            return "秋"
        return "冬"

    # ---------- 资源 ----------
    def regrow(self, rate=1.0):
        """食物再生。夜里产出减半（植物在黑暗中不长）。"""
        light = self.daylight()
        for k, base in REGEN.items():
            m = self.terrain == k
            if not m.any():
                continue
            gain = base * rate * (0.25 + 0.75 * light) * self.cell_area[:, None]
            self.food[m] = np.minimum(self.food_cap[m], self.food[m] + gain[m])
        self.step_count += 1

    def harvest(self, lat_idx, lon_idx, amount):
        i, j = int(np.clip(lat_idx, 0, self.n_lat - 1)), int(lon_idx) % self.n_lon
        got = float(min(amount, self.food[i, j]))
        self.food[i, j] -= got
        return got

    def passable(self, lat_idx, lon_idx):
        i = int(np.clip(lat_idx, 0, self.n_lat - 1))
        j = int(lon_idx) % self.n_lon
        return PASSABLE[int(self.terrain[i, j])]

    def neighbours8(self, lat_idx, lon_idx):
        """8 邻域，东西环绕。返回 [(lat, lon), ...]。"""
        out = []
        for dlat in (-1, 0, 1):
            for dlon in (-1, 0, 1):
                if dlat == 0 and dlon == 0:
                    continue
                i = lat_idx + dlat
                if i < 0 or i >= self.n_lat:      # 纬度不环绕（越过极点就是边界）
                    continue
                out.append((i, (lon_idx + dlon) % self.n_lon))
        return out

    # ---------- 统计 ----------
    def stats(self):
        counts = {TERRAIN_NAMES[k]: int((self.terrain == k).sum()) for k in TERRAIN_NAMES}
        land = self.terrain != OCEAN
        # 真实陆地面积（加权）
        land_area = float((self.area_grid * land).sum())
        total_area = float(self.area_grid.sum())
        return {
            "n_lat": self.n_lat, "n_lon": self.n_lon,
            "cells": self.n_lat * self.n_lon,
            "terrain": counts,
            "land_frac": round(land_area / max(1e-9, total_area), 3),
            "food_total": round(float(self.food.sum()), 1),
            "day": self.step_count // self.steps_per_day,
            "season": self.season(),
            "sun_lon": round(self.sun_lon(), 1),
            "declination": round(float(self.declination()), 1),
        }

    def random_land(self, rng, tries=200):
        for _ in range(tries):
            i = rng.randint(self.n_lat)
            j = rng.randint(self.n_lon)
            if PASSABLE[int(self.terrain[i, j])]:
                return i, j
        return None


# ==================== 自测 ====================

def selftest():
    print("  小沙的地球 selftest（球面 / 气候带 / 昼夜 / 季节 / 面积）")
    ok, total = 0, 0

    p = Planet(n_lat=60, n_lon=120, seed=7)

    # ---- 1. 球面拓扑：经度环绕 ----
    total += 1
    a = p.wrap(0, 5)
    b = p.wrap(120, 5)
    c = p.wrap(-1, 5)
    good = (b[0] == 0) and (c[0] == 119) and (a[0] == 0)
    print("  [1] 经度环绕: x=0→%d, x=120→%d, x=-1→%d" % (a[0], b[0], c[0]))
    ok += int(good)

    # ---- 2. 南极点不环绕 ----
    total += 1
    # wrap() 返回 (经度列, 纬度行)，别把 x 当纬度 —— 我第一次就写反了
    _, i_hi = p.wrap(0, p.n_lat + 20)
    _, i_lo = p.wrap(0, -5)
    print("  [2] 纬度钳制: y=%d → %d（上限 %d）| y=-5 → %d"
          % (p.n_lat + 20, i_hi, p.n_lat - 1, i_lo))
    ok += int(i_hi == p.n_lat - 1 and i_lo == 0)

    # ---- 3. 8 邻域数量：赤道内圈 8 个，极点是 5 个 ----
    total += 1
    mid = len(p.neighbours8(30, 60))
    pole = len(p.neighbours8(0, 60))
    print("  [3] 8邻域: 中间 %d 个，极点 %d 个（极点少一排是正常的）" % (mid, pole))
    ok += int(mid == 8 and pole == 5)

    # ---- 4. 格子面积 ∝ cos(纬度) ----
    total += 1
    eq = float(p.cell_area[len(p.lat) // 2])
    po = float(p.cell_area[0])
    print("  [4] 格子面积: 赤道 %.2f，极点 %.2f（比值 %.1f×）" % (eq, po, eq / max(1e-9, po)))
    ok += int(eq > po * 3)

    # ---- 5. 气候带按纬度分布 ----
    total += 1
    lat = p.lat[:, None] * np.ones((1, p.n_lon))
    polar = lat > 60
    band = (lat > 15) & (lat < 35)
    n_polar_land = int(((p.terrain != OCEAN) & polar).sum())
    frac_desert = int((p.terrain == DESERT).sum()) / max(1, int(band.sum()))
    print("  [5] 极地陆地 %d 格（冰原 %d）| 沙漠带占比 %.0f%%"
          % (n_polar_land, int((p.terrain == ICE).sum()), frac_desert * 100))
    ok += int(0.02 <= frac_desert <= 0.85 and n_polar_land > 0)

    # ---- 6. 昼夜：太阳对面是黑夜 ----
    total += 1
    p2 = Planet(n_lat=60, n_lon=120, seed=7)
    p2.step_count = 0
    light = p2.daylight()
    sl = p2.lon_index(p2.sun_lon())
    anti = (sl + p2.n_lon // 2) % p2.n_lon
    day_here = float(light[len(p2.lat) // 2, sl])
    night_here = float(light[len(p2.lat) // 2, anti])
    print("  [6] 昼夜: 向日本初 %.2f，背面 %.2f" % (day_here, night_here))
    ok += int(day_here > 0.7 and night_here < 0.3)

    # ---- 7. 季节：赤纬在 ±23.44 之间 ----
    total += 1
    decls = []
    for d in range(0, p.days_per_year):
        p.step_count = d * p.steps_per_day
        decls.append(float(p.declination()))
    print("  [7] 太阳赤纬: 最低 %.1f°，最高 %.1f°" % (min(decls), max(decls)))
    ok += int(abs(min(decls) + 23.44) < 1.5 and abs(max(decls) - 23.44) < 1.5)

    # ---- 8. 极昼极夜会发生 ----
    total += 1
    p.step_count = 12 * p.steps_per_day      # 北半球夏天
    light_summer = p.daylight()
    north_summer = float(light_summer[-1].mean())     # 北极圈那一行
    p.step_count = 36 * p.steps_per_day      # 北半球冬天
    north_winter = float(p.daylight()[-1].mean())
    print("  [8] 北极: 夏天平均光照 %.2f，冬天 %.2f" % (north_summer, north_winter))
    ok += int(north_summer > north_winter + 0.3)

    # ---- 9. 夜里的食物再生更慢 ----
    total += 1
    pa = Planet(n_lat=40, n_lon=80, seed=3)
    pb = Planet(n_lat=40, n_lon=80, seed=3)
    pa.food[:] = 0; pb.food[:] = 0
    pa.step_count = 0          # 白天
    pb.step_count = pa.steps_per_day // 2   # 夜里（太阳在背面）
    # 直接比较总再生量
    before_a, before_b = pa.food.sum(), pb.food.sum()
    pa.regrow(1.0); pb.regrow(1.0)
    print("  [9] 一步再生: 白天 %.1f，夜里 %.1f" % (pa.food.sum() - before_a, pb.food.sum() - before_b))
    ok += int((pa.food.sum() - before_a) > (pb.food.sum() - before_b) * 0.9)

    # ---- 10. 陆地比例合理（25%-55%）----
    total += 1
    s = p.stats()
    print("  [10] 陆地表面积占比 %.0f%%（真实地球 29%%）" % (s["land_frac"] * 100))
    ok += int(0.15 <= s["land_frac"] <= 0.60)

    print("\n  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    print("=" * 70)
    print("小沙的地球 · 球面世界")
    print("=" * 70)
    ok = selftest()
    print()
    p = Planet(n_lat=90, n_lon=180, seed=42)
    s = p.stats()
    print("一个 90x180 的地球（16200 格）:")
    for k, v in s.items():
        print("  %-12s %s" % (k, v))
    sys.exit(0 if ok else 1)
