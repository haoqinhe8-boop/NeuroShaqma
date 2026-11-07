# -*- coding: utf-8 -*-
"""小沙自己的"对错判断"（不靠别人教：没有标签、没有奖励）。

机制：**预测式自评价**（预测编码 / 惊讶度）
  它一边走一边预测"我这步之后应该看到什么"：
      预测准  → 对（符合预期）
      预测错  → 错（出意外了）
  真值就是它**自己下一步的观测**，不来自任何标注、任何奖励函数 —— 所以是"自己的"对错判断。
  阈值也是它自己算的（滑动均值 + 3σ），没人给它定。

三件事一起验：
  ① 判得准：正常时误报率低（别一惊一乍）
  ② 判得早、判得出：注入三种身体故障（轮子卡死 / 转向反了 / 走得慢）→ 命中率和延迟
  ③ 自己学（不要人教）：发现不对之后，**它自己**加快学习、适应新身体，惊讶度自己落回阈值内

用法：python xiaosha_judge.py --selftest
"""
import argparse
import math
import sys

import numpy as np

sys.path.insert(0, r"D:\Apps\MyModel\chat\workspace\xiaosha")
import xiaosha_vision as V

N_ACT = 4            # 0 前进 / 1 左转 / 2 右转 / 3 停
SPEED, TURN = 0.16, 0.30


class Body:
    """差速小车 + 可注入的故障（模拟"身体不听话"）。真值只有测试脚本读，大脑读不到。"""

    def __init__(self, seed=0, fault=None, fault_at=120):
        self.rng = np.random.default_rng(seed)
        self.pos = np.array([0.0, 0.0])
        self.yaw = 0.0
        # ★ 本体感觉：这一步**实际**走了多远、转了多少（真机上是编码器/舵机反馈）。
        #   没有它，"轮子卡死"在体外传感器上几乎不可见（实测惊讶度只有 ×1.3 → 判断不了）。
        self.dpos, self.dyaw = 0.0, 0.0
        self.fault, self.fault_at, self.t = fault, fault_at, 0

    def act(self, a):
        self.t += 1
        p0, y0 = self.pos.copy(), self.yaw
        broken = self.fault is not None and self.t > self.fault_at
        eff = a
        if broken:
            if self.fault == "卡死":
                eff = 3                                   # 前进无效（推不动）
            elif self.fault == "转向反":
                eff = {1: 2, 2: 1}.get(a, a)
            elif self.fault == "走得慢":
                pass
        scale = 0.35 if (broken and self.fault == "走得慢") else 1.0
        if eff == 0:
            self.pos += np.array([math.cos(self.yaw), math.sin(self.yaw)]) * SPEED * scale
        elif eff == 1:
            self.yaw += TURN
        elif eff == 2:
            self.yaw -= TURN
        self.pos += 0.004 * self.rng.standard_normal(2)    # 一点噪声（别让它零误差作弊）
        self.dpos = float(np.linalg.norm(self.pos - p0))
        self.dyaw = float(self.yaw - y0)


class Predictor:
    """自己学"我这一动，下一眼会看到什么"：线性预测 + LMS 在线学习（无标签，真值是自己的下一帧）"""

    def __init__(self, d_code=32, lr=0.05):
        self.lr = lr
        self.W = np.zeros((d_code, d_code + N_ACT + 1), dtype=np.float32)
        # ★ 逐通道误差尺度（自己估的）：不然尺度大的通道（测距）会淹没尺度小的通道（视觉码）
        self.escale = None
        self.frozen = False        # 校准期结束后冻住尺度（见 surprise 的注释）

    def surprise(self, err):
        a = np.abs(err)
        if self.escale is None:
            self.escale = a + 1e-3
        elif not self.frozen:
            # ★ 关键：尺度只在**正常期**自适应，之后**冻住**。否则它会用十几拍把"故障"当成
            #   新常态（自我合理化）→ 峰值被抹平（实测"卡死"只有 ×1.2）。判断"错"需要一个
            #   不随错的参照物 —— 这也是自校准阈值的同一条道理。
            self.escale += 0.02 * (a - self.escale)
        r = a / (self.escale + 1e-3)
        k = min(3, r.size)
        # ★ 取**最异常的 3 个通道**而不是 40 通道平均值：故障往往只改 1~2 个通道，
        #   取平均会被稀释 40 倍（实测"卡死"峰值只有 ×1.0，完全看不见）。
        return float(np.mean(np.sort(r)[-k:]))

    def x(self, code, a):
        v = np.zeros(N_ACT + 1, dtype=np.float32)
        v[a] = 1.0
        v[-1] = 1.0                                        # 偏置
        return np.concatenate([code, v])

    def predict(self, code, a):
        return self.W @ self.x(code, a)

    def learn(self, code, a, nxt):
        xx = self.x(code, a)
        err = nxt - self.W @ xx
        self.W += self.lr * np.outer(err, xx) / (float(xx @ xx) + 1e-6)
        return self.surprise(err)


def range_sensors(pos, yaw, marks, tgt, n=8, max_r=5.0):
    """8 路测距（和皮层用的那套一样的思路）：朝 8 个方向看，取最近障碍的距离。"""
    out = np.full(n, max_r, dtype=np.float32)
    half = math.pi / n                                    # ★ 锥形视场（半宽 22.5°）
    # ★ 原来是"射线 + 0.05 容差"：参照物夹在两条射线之间就永远读不到（实测距离恒为 5.0，
    for i in range(n):
        th = yaw + (i / n) * 2 * math.pi
        d = np.array([math.cos(th), math.sin(th)])
        for (mx, my, mr) in list(marks) + [(tgt[0], tgt[1], 0.16)]:
            rel = np.array([mx, my]) - pos
            along = float(rel @ d)
            if along <= 0:
                continue
            perp = float(abs(rel[0] * d[1] - rel[1] * d[0]))
            if perp <= along * math.tan(half) + mr:       # 锥内就算"看见"
                out[i] = min(out[i], max(0.0, along - mr))
    return out


class Judge:
    """自己的对错阈值：滑动均值 + 3σ（没人给它定阈值）；超了就判"错"，并自己加快学习去适应"""
    def __init__(self, warm=120, q=0.95, ema=0.3, slip=10):
        # ★ 不再用手调阈值（σ/MAD/去抖都是拍脑袋，实测要么误报 12%、要么 5 拍内抓不到）。
        #   改成**自校准**：用自己正常段惊讶度的 q 分位当红线 → 误报率由构造保证 (1-q)。
        #   ★ slip：红线**慢慢放**（每拍抬 0.5%），这样"适应之后"不会一直报警（自己学会新身体）。
        self.warm, self.q, self.ema, self.slip = warm, q, ema, slip
        self.hist, self.bad, self.s_ema, self.thr = [], False, None, None

    def update(self, surprise):
        # ★ 不做 EMA：判断用哪个统计量，校准就必须用同一个（平滑后再跟原始分位比，
        #   实测把基线 95 分位抬到 12+，连 ×17 的故障尖峰都被盖住 → "未发现"）。
        self.s_ema = surprise
        if len(self.hist) < self.warm:
            self.hist.append(surprise)
            return False
        if self.thr is None:
            self.thr = float(np.quantile(np.asarray(self.hist[-self.warm:]), self.q))
        self.bad = surprise > self.thr
        if self.bad:
            self.thr *= 1.0 + 0.005                        # 红线慢慢放（自己适应新身体）
        else:
            self.hist.append(surprise)                     # 正常样本才进基线
            self.hist = self.hist[-self.warm:]
        return self.bad


def run(fault=None, steps=260, seed=0, lr=0.05, boost=6.0):
    """跑一段：返回 (每步惊讶度, 每步是否判错, 检测延迟, 恢复步数)"""
    body = Body(seed=seed, fault=fault)
    cam, eye = V.Camera(), V.VisionFrontend(n_in=32)
    pred, judge = Predictor(d_code=42, lr=lr), Judge()   # 42 = 视觉 32 + 测距 8 + 本体感觉 2
    # ★ 视野里放几块**近处**参照物：不然机器人前进时画面几乎不变，
    #   "轮子卡死（没动）"在视觉上与"动了"完全一样 → 原理上判断不了（实测命中 0%）。
    marks = [(1.3, 0.6, 0.10), (1.0, -0.5, 0.10), (2.0, 0.15, 0.12), (0.7, 0.1, 0.08)]
    tgt = (6.0, 0.3)
    def sense(b):
        return np.concatenate([eye.encode(cam.render(b.pos, b.yaw, tgt, marks)),
                               range_sensors(b.pos, b.yaw, marks, tgt),
                               np.array([b.dpos, b.dyaw], dtype=np.float32)]).astype(np.float32)
    code = sense(body)
    sur, flags, det, rec = [], [], None, None
    t_det = None
    for t in range(steps):
        if t == 120:
            pred.frozen = True                             # 正常期结束 → 冻住误差尺度当参照
        a = int(t % 8 // 2)                                # 轮流下 4 种指令（不用人教）
        body.act(a)
        nxt = sense(body)
        s = pred.learn(code, a, nxt)                       # 学 + 得到惊讶度
        bad = judge.update(s)
        if bad and det is None and fault is not None and t > 120:
            det = t - 120
            t_det = t
            pred.lr = lr * boost                           # ★ 自己加快学习（不要人教）
        if t_det is not None and rec is None and not bad and t > t_det + 3:
            rec = t - t_det                                # 从"发现不对"到"不再报警"用了多少拍
        sur.append(s); flags.append(bad); code = nxt
    return np.array(sur), np.array(flags), det, rec


def selftest():
    print("=" * 84)
    print("  小沙自己的对错判断（无标签、无奖励：靠自己的预测误差）")
    print("=" * 84)
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    # ① 正常时别一惊一乍
    sur0, flag0, _, _ = run(fault=None, steps=260, seed=1)
    fp = float(flag0[120:].mean()) * 100
    check("① 判得准（正常时不乱报警）", fp <= 10.0,
          f"正常段误报率 {fp:.1f}%（{int(flag0[120:].sum())}/{len(flag0)-120} 步）；"
          f"惊讶度均值 {sur0[120:].mean():.4f}")

    # ② 三种故障：判得出、判得早
    rows, ok_all = [], True
    for f in ("卡死", "转向反", "走得慢"):
        sur, flag, det, rec = run(fault=f, steps=260, seed=1)
        base = float(np.mean(sur[60:120]))
        peak = float(np.max(sur[121:]))
        # ★ 判据改对：看"能不能在 5 拍内发现" + "惊讶度峰值相对基线放大了多少"
        #   （原来要求故障段一直报警是错的：适应好之后本就不该再报）
        ok = det is not None and det <= 5 and peak > base * 2.0
        ok_all = ok_all and ok
        rows.append(f"{f}: 延迟 {det if det is not None else '未发现'} 拍 / "
                    f"惊讶度 {base:.4f}→峰值 {peak:.4f}（×{peak/base:.1f}）")
    check("② 判得早判得出（三种身体故障）", ok_all, " ｜ ".join(rows))

    # ③ 自己学（不要人教）：惊讶度自己落回阈值内
    sur, flag, det, rec = run(fault="卡死", steps=420, seed=1)
    tail = float(np.mean(sur[-40:]))
    base = float(np.mean(sur[60:120]))
    check("③ 自己学：发现不对后自己适应，惊讶度落回", rec is not None and tail < base * 2.0,
          f"检测延迟 {det} 拍 → **无人干预**下 {rec} 拍后不再报警；末尾惊讶度 {tail:.4f}"
          f"（故障前 {base:.4f}）")

    print("-" * 84)
    print(f"  结果：{3 - len(fails)}/3 项通过" + ("  ✅ 全部通过" if not fails else f"  ❌ 失败：{fails}"))
    print("=" * 84)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest())
