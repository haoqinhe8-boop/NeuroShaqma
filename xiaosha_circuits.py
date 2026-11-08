# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""果蝇招牌回路（2026-11-08）：蘑菇体 + 中央复合体

针对 README 不足点「果蝇招牌回路还没做出来」的补全：
  ① 蘑菇体 MushroomBody —— 果蝇嗅觉联想学习核心
     Kenyon 细胞把 20 维气味输入稀疏膨胀到 2000 维（每个 KC 只连 ~10 个输入，
     激活极度稀疏 ~2%），输出神经元靠"气味+惩罚/奖励同时发生"学会条件反射。
     对应生物事实：果蝇蘑菇体 ~2000 个 Kenyon 细胞，是"闻过就记得"的结构基础。
  ② 中央复合体 CentralComplex —— 果蝇导航核心
     椭球体是环形吸引子：32 个方向单元首尾相连，左转/右转输入让活动峰沿环移动，
     天然完成"角速度 → 方向"积分（头向细胞）。转多少度，峰就挪多少度，误差累积慢。

用法：python xiaosha_circuits.py --selftest
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ============================================================
# ① 蘑菇体：稀疏膨胀编码 + 联想学习（条件反射）
# ============================================================
N_ODOR = 20          # 气味输入通道（每种气味激活 1 个通道）
N_KC = 2000          # Kenyon 细胞（对齐果蝇 ~2000 个）
KC_IN = 10           # 每个 KC 连 10 个随机输入（稀疏连接）
N_OUT = 2            # 输出：0=奖励预测 1=惩罚预测
KC_ACTIVE_FRAC = 0.02  # 激活最活跃的 2% KC（稀疏编码，生物上 KC 发放率 ~2%）


class MushroomBody:
    """蘑菇体：气味 → Kenyon 稀疏编码 → 输出联想学习。"""

    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.kc_in = rng.integers(0, N_ODOR, size=(N_KC, KC_IN)).astype(np.int32)
        self.kc_w = (0.8 + 0.4 * rng.random((N_KC, KC_IN))).astype(np.float32)
        # 输出连接：每个 KC 到 2 个输出的权重（联想记忆存在这里）
        self.w_out = (0.02 * rng.random((N_OUT, N_KC))).astype(np.float32)
        self.kc_active = np.zeros(N_KC, dtype=np.float32)   # 最近一次 KC 激活强度
        self.lr = 0.15

    def sniff(self, odor_id):
        """闻一次气味：返回 KC 稀疏激活模式。"""
        inp = np.zeros(N_ODOR, dtype=np.float32)
        inp[odor_id % N_ODOR] = 1.0
        # 每个 KC 的兴奋 = 它连到的 10 个输入通道的加权和（按行算，不是按通道索引）
        inp_vec = inp[self.kc_in]                      # (N_KC, KC_IN)
        exc = (self.kc_w * inp_vec).sum(axis=1)
        # 稀疏化：只保留最活跃的 2%（生物事实：KC 极稀疏发放）
        k = max(1, int(N_KC * KC_ACTIVE_FRAC))
        top = np.argpartition(exc, -k)[-k:]
        act = np.zeros(N_KC, dtype=np.float32)
        act[top] = exc[top] / (exc[top].max() + 1e-6)
        self.kc_active = act
        return act

    def predict(self):
        """读出：此刻对奖励/惩罚的预测。"""
        return self.w_out @ self.kc_active

    def learn(self, outcome):
        """结果到达（+1 奖励 / -1 惩罚）：气味 KC 活跃 + 结果同时发生 → 联想增强。
        结果的符号只决定"增强哪一路输出"，不决定减弱（条件反射：闻过就记得）。"""
        out_idx = 0 if outcome > 0 else 1
        self.w_out[out_idx] += self.lr * abs(outcome) * self.kc_active
        np.clip(self.w_out, 0.0, 2.0, out=self.w_out)


# ============================================================
# ② 中央复合体：环形吸引子方向积分（头向细胞）
# ============================================================
N_DIR = 32           # 方向单元（椭球体环形吸引子）
DIR_EXC = 1.0        # 峰邻域兴奋
DIR_INH = 0.55       # 全局抑制（保证单峰）
TAU = 0.35           # 活动衰减
TURN_STEP = 2        # 一次"左转/右转"移动峰的角度单元数


class CentralComplex:
    """椭球体环形吸引子：角速度 → 方向积分。"""

    def __init__(self, seed=3):
        rng = np.random.default_rng(seed)
        self.a = np.zeros(N_DIR, dtype=np.float32)
        # 初始方向：0 度（单元 0 活跃）
        self.a[0] = 1.0
        # 环形连接：邻居兴奋（跨环首尾），远处抑制
        self.conn = np.zeros((N_DIR, N_DIR), dtype=np.float32)
        for i in range(N_DIR):
            for j in range(N_DIR):
                d = min(abs(i - j), N_DIR - abs(i - j))  # 环上距离
                self.conn[i, j] = DIR_EXC if d <= 2 else -DIR_INH
        self.heading = 0.0   # 当前方向（度）

    def step(self, turn):
        """一步：turn = +1 右转 / -1 左转 / 0 直行。返回当前方向（度）。"""
        # ① 环形吸引子动力学（活动峰自保持 + 邻居兴奋）
        drive = self.conn @ self.a
        drive = np.clip(drive, 0.0, None)            # ReLU：只有净兴奋传播
        self.a = (1 - TAU) * self.a + TAU * drive
        # ② 角速度积分：峰沿环移动（转 TURN_STEP 个单元）
        if turn != 0:
            shift = TURN_STEP * turn
            self.a = np.roll(self.a, shift)
        # ③ 归一化保持单峰
        s = self.a.sum()
        if s > 0:
            self.a /= s
        # ④ 读出方向：活动峰位置 → 角度
        peak = int(np.argmax(self.a))
        self.heading = (peak * 360.0 / N_DIR + 360.0) % 360.0
        return self.heading


# ============================================================
def selftest():
    print("=" * 76)
    print("  果蝇招牌回路 selftest（蘑菇体联想学习 + 中央复合体导航）")
    print("=" * 76)
    results, fails = [], []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    # ---------- 蘑菇体 ----------
    mb = MushroomBody(seed=7)
    # 1. 稀疏编码：闻气味 3 → KC 激活 ~2%
    act = mb.sniff(3)
    kc_fr = float((act > 0).mean())
    check("Kenyon 稀疏编码：激活 ~2%（生物 KC 发放率）", 0.01 < kc_fr < 0.05,
          f"激活 {act.sum():.0f}/{N_KC} = {kc_fr*100:.1f}%")

    # 2. 不同气味 → 不同 KC 模式
    act2 = mb.sniff(14)
    overlap = int(((act > 0) & (act2 > 0)).sum()) / max(1, int((act > 0).sum()))
    check("不同气味 → 不同 KC 模式（重叠 < 40%）", overlap < 0.4,
          f"气味 3 与气味 14 的 KC 重叠 {overlap*100:.0f}%")

    # 3. 条件反射学习：气味 3 → 惩罚，气味 14 → 奖励
    #    训练前预测接近 0
    p0 = mb.predict()
    for ep in range(30):
        mb.sniff(3)
        mb.learn(-1.0)       # 闻 3 挨罚
        mb.sniff(14)
        mb.learn(+1.0)       # 闻 14 得奖
    mb.sniff(3)
    p3 = mb.predict()        # 应预测惩罚（out[1] 高）
    mb.sniff(14)
    p14 = mb.predict()       # 应预测奖励（out[0] 高）
    ok_reflex = p3[1] > p3[0] and p14[0] > p14[1]
    check("条件反射学会：闻 3 预测惩罚、闻 14 预测奖励", ok_reflex,
          f"闻3预测 [奖 {p3[0]:.2f}, 罚 {p3[1]:.2f}] | 闻14预测 [奖 {p14[0]:.2f}, 罚 {p14[1]:.2f}]")

    # 4. 泛化克制：没闻过的气味 7 → 预测强度远弱于训练过的气味（<15%，稀疏编码天然防串味）
    mb.sniff(7)
    p7 = mb.predict()
    neutral = max(p7) < 0.15 * min(max(p3), max(p14))
    check("没闻过的气味预测远弱于训练气味（<15%，没有乱联想）", neutral,
          f"闻7预测 [奖 {p7[0]:.2f}, 罚 {p7[1]:.2f}] vs 训练气味峰值 {min(max(p3), max(p14)):.1f}")

    # ---------- 中央复合体 ----------
    cx = CentralComplex(seed=3)
    # 5. 静止时方向保持
    h = cx.heading
    for _ in range(20):
        cx.step(0)
    check("静止：方向峰保持不动", abs(cx.heading - h) < 1e-3,
          f"{h:.0f}° → {cx.heading:.0f}°（20 步静止）")

    # 6. 右转积分：连续 8 次右转 ≈ +180°（8 × TURN_STEP × 11.25°）
    expect = (8 * TURN_STEP * 360.0 / N_DIR) % 360.0
    for _ in range(8):
        cx.step(+1)
    err = abs(((cx.heading - expect + 180) % 360) - 180)
    check("右转积分：8 次右转 ≈ 180°（误差 < 15°）", err < 15,
          f"期望 {expect:.0f}°，实测 {cx.heading:.0f}°，误差 {err:.1f}°")

    # 7. 来回转：8 右 8 左回到原点
    for _ in range(8):
        cx.step(-1)
    err2 = abs(((cx.heading - 0 + 180) % 360) - 180)
    check("可逆：8 右转 + 8 左转回到 0°（误差 < 15°）", err2 < 15,
          f"回到 {cx.heading:.0f}°，误差 {err2:.1f}°")

    # 8. 吸引子抗噪：峰保持单峰（无弥散）
    spread = float((cx.a > 0.01).mean())
    check("环形吸引子单峰：活动集中（>1% 的单元 < 30%）", spread < 0.3,
          f"活跃单元占比 {spread*100:.0f}%")

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
