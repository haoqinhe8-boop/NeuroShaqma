# -*- coding: utf-8 -*-
"""把球面地球画成一张等经纬度图（赤道拉平）。

不是示意图：地形是从 xiaosha_planet.Planet 真实生成的，
颜色按地形类型，亮暗表示当天的日照（南北半球一半是白天一半是黑夜）。
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from xiaosha_planet import Planet, TERRAIN_COLOR, TERRAIN_NAMES

OUT = os.path.join(HERE, "planet-map.png")


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def main():
    p = Planet(n_lat=90, n_lon=180, seed=42)
    nl, no = p.n_lat, p.n_lon

    # 每个地形一个基础色
    base = np.zeros((nl, no, 3), dtype=np.float32)
    for k, hexcol in TERRAIN_COLOR.items():
        base[p.terrain == k] = hex2rgb(hexcol)

    # 食物密度：越满越亮（让地图能看出哪里肥沃）
    foodfrac = (p.food / np.maximum(p.food_cap, 1e-6))[..., None]
    base = base * (0.72 + 0.42 * foodfrac)
    # 海拔阴影
    shade = 0.86 + 0.28 * p.elevation[..., None]
    base = np.clip(base * shade, 0, 255)

    img = Image.fromarray(base.astype(np.uint8), "RGB")
    # 放大 5 倍，横向 180*5=900，纵向 90*5=450
    S = 5
    img = img.resize((no * S, nl * S), Image.NEAREST)

    # 叠加一层日照（太阳在 0 度经线）
    light = p.daylight()
    dusk = Image.new("RGB", img.size, (0, 0, 12))
    mask = np.repeat(np.repeat((1 - light).astype(np.float32), S, axis=0), S, axis=1)
    mask = np.clip(mask * 0.72, 0, 1) * 255
    mimg = Image.fromarray(mask.astype(np.uint8), "L")
    img = Image.composite(dusk, img, mimg)

    # ---- 标注 ----
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 15)
        fs = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 12)
    except Exception:
        f = fs = ImageFont.load_default()

    W, H = img.size
    # 经纬线
    for latv in (-60, -30, 0, 30, 60):
        y = int((90 - latv) / 180 * H)
        d.line([(0, y), (W, y)], fill=(255, 255, 255, 42), width=1)
        d.text((6, y + 2), "%+d°" % latv, font=fs, fill=(210, 225, 240))
    for lonv in (-120, -60, 0, 60, 120):
        x = int((lonv + 180) / 360 * W)
        d.line([(x, 0), (x, H)], fill=(255, 255, 255, 42), width=1)
        d.text((x + 3, H - 18), "%+d°" % lonv, font=fs, fill=(210, 225, 240))

    # 太阳位置
    sl = p.lon_index(p.sun_lon())
    sx = int(sl / no * W)
    d.line([(sx, 0), (sx, H)], fill=(255, 214, 92), width=3)
    d.text((sx + 6, 8), "☀ 正午", font=f, fill=(255, 224, 130))

    # 标题
    land = p.terrain != 0
    d.rectangle([0, 0, W, 44], fill=(6, 10, 16))
    d.text((10, 6), "小沙的地球 · 90×180 球面（%d 格，陆地表面积 %.0f%%）"
           % (nl * no, p.stats()["land_frac"] * 100), font=f, fill=(226, 234, 244))
    d.text((10, 26), "东西方向绕圈 · 赤道格子最宽、两极最窄（38×） · 右半边是夜",
           font=fs, fill=(130, 148, 168))

    # 图例
    lx, ly = 10, H - 30
    for k, name in TERRAIN_NAMES.items():
        d.rectangle([lx, ly, lx + 12, ly + 12], fill=hex2rgb(TERRAIN_COLOR[k]))
        d.text((lx + 16, ly - 2), name, font=fs, fill=(200, 214, 228))
        lx += 16 + 14 + len(name) * 13

    img.save(OUT, optimize=True)
    print("已写出 %s（%dx%d, %.0f KB）" % (OUT, img.size[0], img.size[1],
                                          os.path.getsize(OUT) / 1024))
    s = p.stats()
    print("地形: " + " | ".join("%s %d" % (k, v) for k, v in s["terrain"].items()))


if __name__ == "__main__":
    main()
