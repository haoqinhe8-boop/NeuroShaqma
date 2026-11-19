# -*- coding: utf-8 -*-
"""把球面文明画成一张图 + 导出给网页的数据。

跑 20 代，然后画：
  · 等经纬度地图（地形色 + 食物亮度 + 昼夜阴影）
  · 每个个体的位置（点大小 = 体型，颜色 = 夜行性）
  · 祭坛（青点）
  · 右侧纬度带统计
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from xiaosha_globe import GlobeCivilization, latitude_bands
from xiaosha_planet import TERRAIN_COLOR, TERRAIN_NAMES, MAX_FOOD
from xiaosha_world import SACRED

# 输出目录：默认写到脚本旁边（别人 clone 下来就能跑）。
# 本机上平台站点存在时，再额外写一份过去给网页用。
# 注意别硬编码 D:\ 路径 —— 那样别人 clone 后必然跑不了，
# README 那句"你可以自己跑一遍"就成了假话（视频脚本踩过同样的坑）。
HERE_OUT = HERE
SITE = r"D:\Apps\MyModel\chat\public"
PUB = SITE if os.path.isdir(SITE) else HERE
GIS = 6          # 每格放大倍数


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def run(gens=20, pop=140, size=80, seed=42):
    print("跑 %d 代（%dx%d，%d 个体）..." % (gens, size, size, pop))
    np.random.seed(seed)
    civ = GlobeCivilization(size=size, n_agents=pop, seed=seed, verbose=False)
    for g in range(gens):
        civ.generation = g
        for _ in range(60):
            civ.step()
        if g % 5 == 0:
            print("  第 %d 代: 人口 %d | 食物 %.0f"
                  % (g, len(civ.agents), civ.world.total_food()))
    return civ


def render(civ):
    w = civ.world
    N = w.size
    p = w.planet

    # ---- 地形底图 ----
    base = np.zeros((N, N, 3), dtype=np.float32)      # [lat, lon]
    for k, col in TERRAIN_COLOR.items():
        base[p.terrain == k] = hex2rgb(col)

    # 食物密度 → 亮度
    frac = (p.food / np.maximum(p.food_cap, 1e-6))
    base = base * (0.70 + 0.45 * frac[..., None])
    # 海拔阴影
    base = np.clip(base * (0.85 + 0.3 * p.elevation[..., None]), 0, 255)

    img = Image.fromarray(base.astype(np.uint8), "RGB")
    img = img.resize((N * GIS, N * GIS), Image.NEAREST)

    # ---- 昼夜阴影（太阳在 0 度经线一侧）----
    light = p.daylight()
    mask = np.repeat(np.repeat((1 - light).astype(np.float32), GIS, axis=0), GIS, axis=1)
    mask = np.clip(mask * 0.70, 0, 1) * 255
    dusk = Image.new("RGB", img.size, (0, 0, 10))
    img = Image.composite(dusk, img, Image.fromarray(mask.astype(np.uint8), "L"))

    # ---- 画人和祭坛 ----
    d = ImageDraw.Draw(img, "RGBA")
    W = N * GIS

    # 祭坛：青色小方块
    for (ax, ay, _b, _g) in w.altars:
        cx = int((ax + 0.5) * GIS)
        cy = int((N - 1 - ay + 0.5) * GIS)        # 纬度翻转：北在上
        d.rectangle([cx - 3, cy - 3, cx + 3, cy + 3], fill=(63, 220, 230, 220))

    # 个体：大小 = 体型，颜色 = 夜行性（金=夜行，浅蓝=昼行）
    for a in civ.agents:
        cx = int((a.x + 0.5) * GIS)
        cy = int((N - 1 - a.y + 0.5) * GIS)
        r = 1.2 + 1.9 * (a.pheno.size - 0.65) / 0.9
        n = a.pheno.night
        col = (int(90 + 150 * n), int(170 - 60 * n), int(230 - 150 * n), 235)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)

    # ---- 标注 ----
    try:
        f = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 15)
        fs = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 12)
    except Exception:
        f = fs = ImageFont.load_default()

    # 纬度线
    for latv in (-60, -30, 0, 30, 60):
        y = int((90 - latv) / 180 * W)
        d.line([(0, y), (W, y)], fill=(255, 255, 255, 40), width=1)
        d.text((5, y + 1), "%+d°" % latv, font=fs, fill=(210, 226, 240, 230))

    # 太阳位置
    sx = int((p.sun_lon() / 360) * W)
    d.line([(sx, 0), (sx, W)], fill=(255, 214, 92, 200), width=2)

    # ---- 右侧纬度带统计面板 ----
    panel_w = 300
    out = Image.new("RGB", (W + panel_w, W), (8, 11, 15))
    out.paste(img, (0, 0))
    d2 = ImageDraw.Draw(out)

    bands = latitude_bands(civ, 6)
    x0 = W + 14
    d2.text((x0, 12), "纬度带", font=f, fill=(226, 236, 246))
    d2.text((x0, 34), "人口 / 体型 / 夜行性", font=fs, fill=(130, 150, 170))
    yy = 60
    maxtop = max(1, max(b["pop"] for b in bands))
    for b in bands:
        d2.text((x0, yy), "%+5.1f°" % b["lat"], font=fs, fill=(200, 214, 228))
        d2.text((x0 + 62, yy), "%3d 人" % b["pop"], font=fs, fill=(200, 214, 228))
        barw = int(150 * b["pop"] / maxtop)
        if barw > 0:
            d2.rectangle([x0 + 118, yy + 2, x0 + 118 + barw, yy + 11],
                         fill=(63, 201, 211))
        d2.text((x0, yy + 15), "体型 %.2f  夜行 %.2f  光照 %.2f"
                % (b["size"], b["night"], b["light"]), font=fs, fill=(110, 128, 148))
        yy += 40

    # 统计
    yy += 14
    s = w.stats()
    for line in ["第 %d 天 · %s" % (s["day"], s["season"]),
                 "人口 %d · 部落 %d" % (len(civ.agents), len(civ.tribes)),
                 "祭坛 %d · 食物 %.0f" % (s["altars"], s["food_total"]),
                 "太阳赤纬 %+.1f°" % s["declination"]]:
        d2.text((x0, yy), line, font=fs, fill=(150, 168, 188))
        yy += 20

    # 图例
    yy += 12
    d2.text((x0, yy), "● 点大小=体型  颜色=夜行性(金)", font=fs, fill=(120, 140, 160))
    yy += 18
    d2.text((x0, yy), "■ 青色方块 = 祭坛", font=fs, fill=(63, 220, 230))
    yy += 18
    d2.text((x0, yy), "右半暗 = 当地是夜里", font=fs, fill=(120, 140, 160))

    # 标题
    d2.rectangle([0, 0, W, 40], fill=(6, 10, 16))
    big = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 18) if f else f
    d2.text((10, 8), "小沙的地球 · %dx%d 球面 · 人口 %d · 陆地 22%%"
            % (N, N, len(civ.agents)), font=big, fill=(230, 238, 248))

    return out


def export_json(civ, path):
    w = civ.world
    N = w.size
    p = w.planet
    data = {
        "size": N,
        "terrain": p.terrain.T.astype(int).flatten().tolist(),   # [y*N+x] = [x,y]
        "stats": w.stats(),
        "altars": [[int(a), int(b)] for a, b, _c, _d in w.altars],
        "agents": [[int(a.x), int(a.y),
                    round(float(a.pheno.size), 2),
                    round(float(a.pheno.night), 2),
                    round(float(a.knowledge.get(SACRED, 0)), 2)]
                   for a in civ.agents],
        "bands": latitude_bands(civ, 6),
        "terrain_names": TERRAIN_NAMES,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return len(json.dumps(data)) // 1024


def main(gens=20, pop=140, size=80):
    civ = run(gens, pop, size)
    img = render(civ)

    # 始终写一份到脚本旁边（进仓库、可复现）；站点存在时再写一份给网页
    local_png = os.path.join(HERE_OUT, "globe-civ.png")
    img.save(local_png, optimize=True)
    print("\n图: %s (%dx%d, %.0f KB)"
          % (local_png, img.size[0], img.size[1], os.path.getsize(local_png) / 1024))
    if PUB != HERE_OUT:
        img.save(os.path.join(PUB, "globe-civ.png"), optimize=True)

    local_js = os.path.join(HERE_OUT, "globe-civ.json")
    kb = export_json(civ, local_js)
    print("数据: %s (%d KB)" % (local_js, kb))
    if PUB != HERE_OUT:
        export_json(civ, os.path.join(PUB, "globe-civ.json"))

    print("\n纬度带:")
    for b in latitude_bands(civ, 6):
        print("  %+6.1f° %4d 人 体型%.2f 夜行%.2f 光照%.2f"
              % (b["lat"], b["pop"], b["size"], b["night"], b["light"]))


if __name__ == "__main__":
    g = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(gens=g)
