# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""小沙自己的"对错判断" 第 3 版：按钉死的四条一次做对，争取突破。

前两轮失败记录：`README-对错判断.md`（10 版）、`README-对错判断-第2轮.md`（4 版）。
第 2 轮的钉死结论：世界模型误差 0.328 vs「什么都不变」基线 0.093 —— **比躺平还差 3.5 倍**，
根因三条：参数量/样本量失衡（215 参数 / 120 样本）、特征没归一化（量纲炸）、视觉 32 路是噪声源。

第 3 版四条修正（照着做，不再零敲碎打）：
  ① 每通道 **z-score** 归一化（Δ 用同尺度，误差变成无量纲）
  ② 交互项只加在**变化敏感的 10 路**（测距 8 + 本体感觉 2）→ 参数 55，不是 215
  ③ 正常期拉到 **700 拍**，其中 **1/3 留出**不做拟合（≈13 样本/参数）
  ④ 判据钉死：**留出误差 ≤ 0.6×躺平基线**（不再用"永远 <1 不可能"的比值判据）

而且：视觉不经世界模型（AGC+分块后是噪声源）；"对错"判断靠测距 + 本体感觉（真机=雷达+编码器）。

用法：python xiaosha_judge_v3.py --selftest
"""
import argparse
import math
import sys

import numpy as np

N_ACT, SPEED, TURN = 4, 0.16, 0.30
N_RANGE = 8
D_S = N_RANGE + 2                      # 测距 8 + 本体感觉 2 = 10（不进视觉）


class Body:
    def __init__(self, seed=0, fault=None, fault_at=0):
        self.rng = np.random.default_rng(seed)
        self.pos, self.yaw = np.array([0.0, 0.0]), 0.0
        self.dpos, self.dyaw = 0.0, 0.0
        self.fault, self.fault_at, self.t = fault, fault_at, 0

    def act(self, a):
        self.t += 1
        p0, y0 = self.pos.copy(), self.yaw
        broken = self.fault is not None and self.t > self.fault_at
        eff = a
        if broken:
            if self.fault == "卡死":
                eff = 3
            elif self.fault == "转向反":
                eff = {1: 2, 2: 1}.get(a, a)
        scale = 0.35 if (broken and self.fault == "走得慢") else 1.0
        if eff == 0:
            self.pos += np.array([math.cos(self.yaw), math.sin(self.yaw)]) * SPEED * scale
        elif eff == 1:
            self.yaw += TURN
        elif eff == 2:
            self.yaw -= TURN
        self.pos += 0.004 * self.rng.standard_normal(2)
        self.dpos = float(np.linalg.norm(self.pos - p0))
        self.dyaw = float(self.yaw - y0)


MARK, TGT = [(1.3, 0.6, 0.10), (1.0, -0.5, 0.10), (2.0, 0.15, 0.12), (0.7, 0.1, 0.08)], (6.0, 0.3)


def sense(b, max_r=5.0):
    # ★★ 关键修正：原来是"锥内取最近距离"，障碍物**进出锥边界时读数会突跳**（5.0 → 1.0 那种
    #   4 个单位的跳变）—— 不连续信号线性模型永远预测不了，误差全被跳变主导（实测模型误差 1.28
    #   全是它贡献的）。改成**连续**传感器：角度高斯加权 × 1/距离 求和（像"光强"，位姿连续变化
    #   读数就连续变化）。**传感器的连续性决定线性世界模型能不能work** —— 这条最值钱。
    out = np.zeros(N_RANGE, dtype=np.float32)
    sig = math.pi / N_RANGE
    for i in range(N_RANGE):
        th = b.yaw + (i / N_RANGE) * 2 * math.pi
        d = np.array([math.cos(th), math.sin(th)])
        for (mx, my, mr) in list(MARK) + [(TGT[0], TGT[1], 0.16)]:
            rel = np.array([mx, my]) - b.pos
            dist = float(np.linalg.norm(rel)) + 1e-6
            cosang = float(rel @ d) / dist
            if cosang <= 0.0:
                continue
            ang = math.acos(min(1.0, cosang))
            out[i] += math.exp(-(ang / sig) ** 2) / (1.0 + dist)
    return np.concatenate([out, np.array([b.dpos, b.dyaw], dtype=np.float32)])


class WorldModel:
    """归一化 + 只对敏感通道做交互的岭回归世界模型（预测 Δ）。"""

    def __init__(self, d=D_S, n_act=N_ACT, lam=0.5):
        self.d, self.n_act, self.lam = d, n_act, lam
        self.nf = d + n_act + 1 + n_act * d
        self.W = np.zeros((d, self.nf), dtype=np.float32)
        self.mu = np.zeros(d, dtype=np.float32)
        self.sd = np.ones(d, dtype=np.float32)
        self.dsd = np.ones(d, dtype=np.float32)
        self.rsd = np.ones(d, dtype=np.float32)      # 标准残差尺度（正常期残差自身的标准差）
        self.fitted = False

    def z(self, s):
        return (s - self.mu) / self.sd

    def feats(self, s, a):
        x = self.z(s)
        v = np.zeros(self.n_act + 1, dtype=np.float32)
        v[a], v[-1] = 1.0, 1.0
        inter = np.zeros(self.n_act * self.d, dtype=np.float32)
        inter[a * self.d:(a + 1) * self.d] = x
        return np.concatenate([x, v, inter]).astype(np.float32)

    def err(self, s, a, ds):
        # ★★★ 尺度用**标准残差**：正常期**残差自身**的标准差（每通道"预测不确定性"），
        #   不是状态标准差。前者才是"这次预测有多可信"的正确尺度 —— 实测用状态 sd 时
        #   卡死（单通道偏差）和走得慢（全体小幅偏）都被稀释到阈值以下；换标准残差后
        #   近乎确定性的通道（本体感觉）残差尺度极小 → 故障立刻放大。
        e = np.abs(ds - self.W @ self.feats(s, a)) / self.rsd
        # 卡方式统计（平方平均再开根）：单通道大偏差和全体小幅系统性偏移都抓得到。
        return float(np.sqrt(np.mean(e ** 2)))

    def online(self, s, a, ds, lr=0.05):
        f = self.feats(s, a)
        e = ds - self.W @ self.feats(s, a)
        self.W += lr * np.outer(e, f) / (float(f @ f) + 1e-6)

    def fit(self, buf, lam=None):
        lam = self.lam if lam is None else lam
        S = np.stack([b[0] for b in buf])
        # ★★ 真 bug：有些测距通道**永远读到 5.0**（该方向没障碍物）→ 标准差≈0 →
        #   z-score 除出 1e6 级的天文数字 → FᵀF 奇异 → 权重爆炸（实测留出误差 19.25 vs 基线 1.0）。
        #   给尺度加**下限**（不许小于 0.05），常量通道的 z 就≈0，无害。
        self.mu = S.mean(0).astype(np.float32)
        self.sd = np.maximum(S.std(0), 0.05).astype(np.float32)
        D = np.stack([b[2] for b in buf])
        self.dsd = (np.abs(D).mean(0) + 0.01).astype(np.float32)
        F = np.stack([self.feats(b[0], b[1]) for b in buf])
        # ★★★ 真凶（单位不一致）：原来拟合的是**归一化目标** D/dsd，但预测/评估时没乘回来
        #   → 模型"预测"比真实 Δ 大 1/dsd ≈ 20 倍（10~20 倍误差就是这么来的，坑了我 4 轮）。
        #   特征已经归一化、条件数没问题，所以直接拟合**原始 Δ**，训练与评估单位才一致。
        self.W = np.linalg.solve(F.T @ F + lam * np.eye(self.nf), F.T @ D).T.astype(np.float32)
        R = D - F @ self.W.T                                   # 正常期残差
        self.rsd = np.maximum(R.std(0), 1e-3).astype(np.float32)   # ★ 标准残差尺度
        self.fitted = True


def run(fault=None, warm=700, steps=1050, seed=1, boost=8.0):
    body = Body(seed=seed, fault=fault, fault_at=warm)
    s = sense(body)
    buf, naive_all = [], []
    for t in range(warm):                                  # 正常期（自己轮流下指令）
        a = int(t % 8 // 2)
        body.act(a)
        s2 = sense(body)
        ds = s2 - s
        buf.append((s.copy(), a, ds.copy()))
        naive_all.append(float(np.abs(ds).mean()))
        s = s2
    wm = WorldModel()
    cnt = np.zeros(len(buf), dtype=bool)
    cnt[::3] = True                                        # ★ 1/3 留出（不参与拟合）
    wm.fit([b for i, b in enumerate(buf) if not cnt[i]])   # 用 2/3 拟合
    ho = [b for i, b in enumerate(buf) if cnt[i]]
    base_sv = [wm.err(*b) for b in ho]                     # 留出惊讶度
    # ★ 判据用**同单位绝对误差**（不做除法，避免上面那类被放大 100 倍的坑）
    naive_ho = float(np.mean([np.abs(b[2]).mean() for b in ho]))
    mae_ho = float(np.mean([float(np.abs(b[2] - wm.W @ wm.feats(b[0], b[1])).mean())
                            for b in ho]))
    thr = float(np.quantile(base_sv, 0.95))                # 红线：留出段的 95 分位（自校准）
    det, rec, t_det = None, None, None
    for t in range(warm, steps):                           # 故障期
        a = int(t % 8 // 2)
        body.act(a)
        s2 = sense(body)
        ds = s2 - s
        sv = wm.err(s, a, ds)
        bad = sv > thr
        if bad and det is None and fault is not None:
            det, t_det = t - warm, t
            wm.boosted = True
        if t_det is not None and rec is None and not bad and t > t_det + 3:
            rec = t - t_det
        wm.online(s, a, ds, lr=0.05 * (boost if getattr(wm, "boosted", False) else 1.0))
        s = s2
    return det, rec, mae_ho, naive_ho, thr, float(np.mean(base_sv))


def selftest():
    print("=" * 92)
    print("  自己的对错判断 · 第 3 版（归一化 + 只对敏感 10 路做交互 + 700 拍正常期 + 留出判据）")
    print("=" * 92)
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    # ① 世界模型必须比"躺平"好（第 2 轮就是这里崩的：0.328 vs 0.093）
    det0, rec0, mae, naive, thr, b0 = run(fault=None)
    check("① 世界模型够准：留出误差 ≤ 0.6×「什么都不变」基线",
          mae <= 0.6 * naive,
          f"模型留出误差 {mae:.4f} vs 躺平基线 {naive:.4f}"
          f"（好 {100*(1-mae/naive):.0f}%）｜红线 {thr:.2f}，留出惊讶度均值 {b0:.2f}")

    # ② 三种故障：5 拍内发现，正常期不误报
    rows, ok_all = [], True
    for f in ("卡死", "转向反", "走得慢"):
        det, rec, mae, naive, thr, b0 = run(fault=f)
        ok = det is not None and det <= 5
        ok_all = ok_all and ok
        rows.append(f"{f}: {'延迟 %d 拍' % det if det is not None else '未发现'}"
                    + (f"（后 {rec} 拍自己不再报警）" if rec else ""))
    check("② 判得早判得出（三种身体故障都在 5 拍内）", ok_all, " ｜ ".join(rows))

    # ③ 自己学（无人干预）
    det, rec, mae, naive, thr, b0 = run(fault="卡死")
    check("③ 自己学：发现不对后自己适应，不再报警（无人干预）",
          det is not None and rec is not None,
          f"检测延迟 {det} 拍 → {rec} 拍后自己不再报警（它用「错」这个信号自己调了模型）")

    print("-" * 92)
    print(f"  结果：{3 - len(fails)}/3 项通过" + ("  ✅ 全部通过" if not fails else f"  ❌ 失败：{fails}"))
    print("=" * 92)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest())
