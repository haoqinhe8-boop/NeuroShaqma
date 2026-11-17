# -*- coding: utf-8 -*-
"""导出"地球 Online"的可视化快照 —— 给网页用。

跑 N 代，每隔若干代存一张快照（地形 / 食物 / 人口 / 祭坛 / 部落 / 信仰），
写成 JSON 给前端 Canvas 渲染。

用法：python export_civ_snapshots.py [代数] [人口]
输出：D:\\Apps\\MyModel\\chat\\public\\civ-data.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xiaosha_civilization import Civilization
from xiaosha_world import SACRED, TERRAIN_NAMES

OUT = r"D:\Apps\MyModel\chat\public\civ-data.json"


def snapshot(civ, gen):
    world = civ.world
    # 地形（压缩成一行字符串，0-3 各一位）
    terrain = "".join(str(int(v)) for v in world.terrain.flatten())
    # 食物（归一化到 0-255，省体积）
    mx = float(world.food.max()) or 1.0
    food = [int(min(255, v / mx * 255)) for v in world.food.flatten()]

    # 人口
    agents = []
    for a in civ.agents:
        agents.append([
            a.x, a.y,
            a.tribe.id if a.tribe is not None else 0,
            1 if a.knowledge.get(SACRED, 0.0) > 0.35 else 0,
            round(float(a.knowledge.get(SACRED, 0.0)), 2),
            a.generation,
        ])

    # 祭坛
    altars = [[x, y, g] for (x, y, _bid, g) in world.altars]

    # 部落汇总
    tribes = []
    for t in civ.tribes:
        if t.size() == 0:
            continue
        tribes.append({
            "id": t.id, "size": t.size(), "tech": t.tech_level(),
            "knowledge": len(t.knowledge), "gen": t.founded_gen,
        })
    tribes.sort(key=lambda d: -d["size"])

    s = civ.stats()
    return {
        "gen": gen,
        "size": world.size,
        "terrain": terrain,
        "food": food,
        "food_max": round(mx, 2),
        "agents": agents,
        "altars": altars,
        "tribes": tribes[:12],
        "stats": {
            "pop": s["pop"], "tribes": s["tribes"], "settlements": s["settlements"],
            "civilizations": s["civilizations"], "knowledge": s["knowledge"],
            "altars": s["altars"], "believers": s["believers"],
            "faith": s["faith_avg"], "discoveries": s["discoveries"],
        },
        "discoveries": civ.discoveries[-40:],
        "terrain_names": TERRAIN_NAMES,
    }


def main(gens=20, pop=100, every=5, seed=42):
    civ = Civilization(size=40, n_agents=pop, seed=seed, verbose=False)
    snaps = []
    print("跑 %d 代，每 %d 代存一张快照..." % (gens, every))
    for g in range(gens):
        civ.generation = g
        for _ in range(60):
            civ.step()
        if g % every == 0 or g == gens - 1:
            snaps.append(snapshot(civ, g))
            s = snaps[-1]["stats"]
            print("  第 %2d 代: 人口 %3d | 祭坛 %2d | 信徒 %3d | 信仰 %.2f | 部落 %2d"
                  % (g, s["pop"], s["altars"], s["believers"], s["faith"], s["tribes"]))

    data = {"generations": gens, "snapshots": snaps}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print("\n已写出 %s（%.0f KB，%d 张快照）"
          % (OUT, os.path.getsize(OUT) / 1024, len(snaps)))


if __name__ == "__main__":
    g = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    main(gens=g, pop=p)
