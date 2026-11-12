# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""视觉听觉增强（2026-11）：提升小沙的视力和听力。

之前：视力=简单取景（xiaosha_vision.Camera）、听力=单纯语音转文字（ASR）。
这里补上更丰富的人类视觉/听觉能力（真机接口留好，仿真用数据验证）：

视力提升 EnhancedEye：
  ① 颜色识别（视锥 RGB 三通道 → 颜色名）
  ② 多目标检测（一帧里多个物体同时定位）
  ③ 运动检测（两帧差分 → 运动区域，视动觉）

听力提升 EnhancedEar：
  ① 音调识别（频率 → 低/中/高音，耳蜗频率分析）
  ② 方位识别（双耳时间差 → 声源左/右/前方）
  ③ 音量感知（振幅 → 轻/中/响）

用法：python xiaosha_senses.py --selftest
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


class EnhancedEye:
    """视力提升：颜色识别 + 多目标检测 + 运动检测。"""

    def see_color(self, rgb):
        """颜色识别：RGB → 颜色名（视锥三通道主导）。"""
        r, g, b = [float(x) for x in rgb]
        mx = max(r, g, b)
        if mx < 0.05:
            return "黑"
        if r > 0.5 and g > 0.5 and b < 0.3:
            return "黄"
        if r > g * 1.5 and r > b * 1.5:
            return "红"
        if g > r * 1.5 and g > b * 1.5:
            return "绿"
        if b > r * 1.5 and b > g * 1.5:
            return "蓝"
        if r > 0.7 and g > 0.7 and b > 0.7:
            return "白"
        return "灰"

    def detect_objects(self, img, threshold=0.5):
        """多目标检测：找出亮度 > 阈值的连通区域，返回每个物体中心坐标。"""
        img = np.asarray(img, dtype=np.float32)
        mask = img > threshold
        objects = []
        # 简单连通域：扫描亮像素，聚类（4 邻域）
        seen = np.zeros_like(mask, dtype=bool)
        h, w = mask.shape
        for i in range(h):
            for j in range(w):
                if mask[i, j] and not seen[i, j]:
                    # BFS 找连通域
                    stack = [(i, j)]
                    pts = []
                    while stack:
                        x, y = stack.pop()
                        if x < 0 or x >= h or y < 0 or y >= w or seen[x, y] or not mask[x, y]:
                            continue
                        seen[x, y] = True
                        pts.append((x, y))
                        stack += [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                    if pts:
                        cy = sum(p[0] for p in pts) / len(pts)
                        cx = sum(p[1] for p in pts) / len(pts)
                        objects.append((round(cx, 1), round(cy, 1)))
        return objects

    def motion(self, frame1, frame2, threshold=0.3):
        """运动检测：两帧差分 → 是否发生运动 + 运动像素数。"""
        f1 = np.asarray(frame1, dtype=np.float32)
        f2 = np.asarray(frame2, dtype=np.float32)
        diff = np.abs(f1 - f2)
        moved = int((diff > threshold).sum())
        return moved > 0, moved


class EnhancedEar:
    """听力提升：音调识别 + 方位识别 + 音量感知。"""

    def pitch(self, freq):
        """音调：频率 → 低/中/高音（耳蜗按位置编码频率）。"""
        if freq < 200:
            return "低音"
        if freq <= 2000:
            return "中音"
        return "高音"

    def direction(self, left_delay, right_delay):
        """方位：双耳时间差 → 声源方向（时间差越小，声源越近该耳）。"""
        dt = right_delay - left_delay   # 右耳延迟 - 左耳延迟
        if dt > 0.1:
            return "偏左"
        if dt < -0.1:
            return "偏右"
        return "正前方"

    def loudness(self, amplitude):
        """音量：振幅 → 轻/中/响。"""
        if amplitude < 0.3:
            return "轻"
        if amplitude <= 0.7:
            return "中"
        return "响"


# ============================================================
def selftest():
    print("=" * 76)
    print("  视觉听觉增强 selftest（视力+听力提升）")
    print("=" * 76)
    results, fails = [], []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    eye = EnhancedEye()
    # ① 颜色识别
    check("视力-颜色识别：RGB → 颜色名",
          eye.see_color((1.0, 0.1, 0.1)) == "红" and eye.see_color((0.1, 1.0, 0.1)) == "绿"
          and eye.see_color((0.1, 0.1, 1.0)) == "蓝",
          f"红={eye.see_color((1,0.1,0.1))} 绿={eye.see_color((0.1,1,0.1))} 蓝={eye.see_color((0.1,0.1,1))}")

    # ② 多目标检测
    img = np.zeros((20, 20), dtype=np.float32)
    img[2:5, 2:5] = 1.0    # 物体1（左上）
    img[14:17, 14:17] = 1.0  # 物体2（右下）
    objs = eye.detect_objects(img)
    check("视力-多目标检测：一帧两个物体同时定位", len(objs) == 2, f"检测到 {len(objs)} 个物体 {objs}")

    # ③ 运动检测
    f1 = np.zeros((10, 10), dtype=np.float32)
    f2 = f1.copy(); f2[4:7, 4:7] = 1.0   # 中间出现物体
    moved, px = eye.motion(f1, f2)
    check("视力-运动检测：两帧差分发现运动", moved and px > 0, f"运动像素 {px} 个")

    ear = EnhancedEar()
    # ④ 音调
    check("听力-音调识别：低/中/高音", ear.pitch(100) == "低音" and ear.pitch(1000) == "中音"
          and ear.pitch(5000) == "高音",
          f"100Hz={ear.pitch(100)} 1000Hz={ear.pitch(1000)} 5000Hz={ear.pitch(5000)}")

    # ⑤ 方位
    check("听力-方位识别：双耳时间差定方向",
          ear.direction(0.0, 0.3) == "偏左" and ear.direction(0.3, 0.0) == "偏右"
          and ear.direction(0.15, 0.15) == "正前方",
          f"左耳快={ear.direction(0,0.3)} 右耳快={ear.direction(0.3,0)} 同={ear.direction(0.15,0.15)}")

    # ⑥ 音量
    check("听力-音量感知：轻/中/响", ear.loudness(0.1) == "轻" and ear.loudness(0.5) == "中"
          and ear.loudness(0.9) == "响",
          f"0.1={ear.loudness(0.1)} 0.5={ear.loudness(0.5)} 0.9={ear.loudness(0.9)}")

    print("-" * 76)
    print(f"  结果：{sum(results)}/{len(results)} 项通过"
          + ("  ✅ 全部通过" if not fails else f"  ❌ 失败：{fails}"))
    print("=" * 76)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    ap.print_help()