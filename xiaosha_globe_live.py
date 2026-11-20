# -*- coding: utf-8 -*-
"""小沙的地球 —— 实时直播进程。

不是导出快照：这个进程一直在跑，每帧把当前真实状态打成一行 JSON 交给
server.js 转发给浏览器。关了浏览器它照跑，连上就立刻看到此刻的世界。

输出协议（每行一个 JSON）：
    {"type":"ready", ...}                          启动就绪，带地形底图
    {"type":"frame","d":天,"s":"夏",...,"a":[...]}   每个时间片一帧

为什么用行 JSON 而不是二进制：数量级不大（350 个体 x 4 个数 = 约 4KB/帧），
行文本在 node 里一行 readline 就能切，出错也看得见。

跑法：
    python xiaosha_globe_live.py              默认 80x80，250 个体，10 帧/秒
    python xiaosha_globe_live.py 100 300 6    世界边长 / 人口 / 每秒帧数
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from xiaosha_globe import GlobeCivilization
from xiaosha_planet import TERRAIN_NAMES

__version__ = "1.0.0"


def _r2(v):
    return int(round(float(v) * 100))


def main(size=80, pop=250, fps=15.0, steps_per_frame=1, seed=20261120):
    np.random.seed(seed)
    civ = GlobeCivilization(size=size, n_agents=pop, seed=seed, verbose=False)
    w = civ.world
    p = w.planet
    gen_steps = 60
    out = sys.stdout

    out.write(json.dumps({
        "type": "ready", "size": size,
        "terrain": p.terrain.T.astype(int).flatten().tolist(),
        "terrain_names": {str(k): v for k, v in TERRAIN_NAMES.items()},
    }, ensure_ascii=False) + "\n")
    out.flush()

    target_dt = 1.0 / max(0.5, float(fps))
    frame_no = 0

    while True:
        t0 = time.time()
        for _ in range(steps_per_frame):
            civ.step()
        if civ.step_count % gen_steps == 0:
            civ.generation += 1

        ag = civ.agents
        known = set()
        for a in ag:
            known.update(a.knowledge)

        frame = {
            "type": "frame", "i": frame_no, "g": civ.generation,
            "d": w.day_number(), "s": w.season(),
            "sl": round(float(p.sun_lon()), 1),
            "dc": round(float(p.declination()), 1),
            "n": len(ag), "f": round(w.total_food(), 1),
            "k": len(known), "t": len(civ.tribes),
            "a": [[int(a.x), int(a.y), _r2(a.pheno.size), _r2(a.pheno.night)] for a in ag],
            "al": [[int(ax), int(ay)] for (ax, ay, _b, _g) in w.altars],
        }
        out.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n")
        out.flush()
        frame_no += 1

        sleep = target_dt - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("-")]
    size = int(a[0]) if len(a) > 0 else 80
    pop = int(a[1]) if len(a) > 1 else 250
    fps = float(a[2]) if len(a) > 2 else 15.0
    try:
        main(size=size, pop=pop, fps=fps)
    except KeyboardInterrupt:
        pass