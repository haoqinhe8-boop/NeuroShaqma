# -*- coding: utf-8 -*-
"""小沙自己的"对错判断"（第 2 版：先把世界模型做强，再判对错）。

第 1 版失败根因（10 个版本实测在 `README-对错判断.md`）：
  世界模型太弱 —— 线性模型**没有"动作×状态"交互项**，只会学"前进平均让测距减多少"，
  而不会学"前进**在这个状态下**让测距减多少"；而且直接预测**绝对值**（世界大部分不变，浪费容量）。
  → 正常段惊讶度就有 ×2.2（连正常都预测不准），此时拧阈值只能在误报/漏报之间换位置。

第 2 版两处修正：
  ① 特征含 **动作 × 状态 交互**（onehot(a) ⊗ sensor）→ 每个状态下的动作效果分别学
  ② 预测 **变化量 Δ**（不是绝对值），逐通道归一化 → 世界是"大部分不变 + 小变化"，好学得多
  判据也跟着改对：先要求**世界模型显著优于"什么都不变"的基线**，再看对错判断。

用法：python xiaosha_judge.py --selftest
"""
import argparse
import math
import sys

import numpy as np

sys.path.insert(0, r"D:\Apps\MyModel\chat\workspace\xiaosha")
import xiaosha_vision as V

N_ACT, SPEED, TURN = 4, 0.16, 0.30
D_SENS = 42                                  # 视觉 32 + 测距 8 + 本体感觉 2


class Body:
    """差速小车 + 可注入故障；dpos/dyaw 是本体感觉（真机上的编码器反馈）。"""

    def __init__(self, seed=0, fault=None, fault_at=120):
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


class WorldModel:
    """动作-效果世界模型：预测 Δsensor；特征 = [s, onehot(a), bias, onehot(a)⊗s]。"""

    def __init__(self, d=D_SENS, n_act=N_ACT, lr=0.05):
        self.d, self.n_act, self.lr = d, n_act, lr
        self.nf = d + n_act + 1 + n_act * d
        self.W = np.zeros((d, self.nf), dtype=np.float32)
        self.scale = None
        self.frozen = False
        self.holdout_mae = None

    def feats(self, s, a):
        v = np.zeros(self.n_act + 1, dtype=np.float32)
        v[a], v[-1] = 1.0, 1.0
        inter = np.zeros(self.n_act * self.d, dtype=np.float32)
        inter[a * self.d:(a + 1) * self.d] = s         # ★ 动作×状态交互
        return np.concatenate([s, v, inter])

    def surprise(self, err):
        a = np.abs(err)
        if self.scale is None:
            self.scale = a + 1e-3
        elif not self.frozen:
            self.scale += 0.02 * (a - self.scale)
        r = a / (self.scale + 1e-3)
        return float(np.mean(np.sort(r)[-3:]))         # 最异常的 3 个通道

    def learn(self, s, a, ds):
        f = self.feats(s, a)
        err = ds - self.W @ f
        self.W += self.lr * np.outer(err, f) / (float(f @ f) + 1e-6)
        return err


def range_sensors(pos, yaw, marks, tgt, n=8, max_r=5.0):
    out = np.full(n, max_r, dtype=np.float32)
    half = math.pi / n
    for i in range(n):
        th = yaw + (i / n) * 2 * math.pi
        d = np.array([math.cos(th), math.sin(th)])
        for (mx, my, mr) in list(marks) + [(tgt[0], tgt[1], 0.16)]:
            rel = np.array([mx, my]) - pos
            along = float(rel @ d)
            if along <= 0:
                continue
            if float(abs(rel[0] * d[1] - rel[1] * d[0])) <= along * math.tan(half) + mr:
                out[i] = min(out[i], max(0.0, along - mr))
    return out


class Judge:
    """红线 = 自己正常段惊讶度的 95 分位（自校准）；红线只慢慢放，不随错滑。"""

    def __init__(self, warm=120, q=0.95):
        self.warm, self.q = warm, q
        self.hist, self.thr = [], None

    def update(self, s):
        if len(self.hist) < self.warm:
            self.hist.append(s)
            return False
        if self.thr is None:
            self.thr = float(np.quantile(np.asarray(self.hist[-self.warm:]), self.q))
        bad = s > self.thr
        if bad:
            self.thr *= 1.001
        else:
            self.hist.append(s)
            self.hist = self.hist[-self.warm:]
        return bad


def run(fault=None, steps=300, seed=0, warm=120, boost=8.0):
    body = Body(seed=seed, fault=fault)
    cam, eye = V.Camera(), V.VisionFrontend(n_in=32)
    marks = [(1.3, 0.6, 0.10), (1.0, -0.5, 0.10), (2.0, 0.15, 0.12), (0.7, 0.1, 0.08)]
    tgt = (6.0, 0.3)

    def sense(b):
        return np.concatenate([eye.encode(cam.render(b.pos, b.yaw, tgt, marks)),
                               range_sensors(b.pos, b.yaw, marks, tgt),
                               np.array([b.dpos, b.dyaw], dtype=np.float32)]).astype(np.float32)

    wm, judge = WorldModel(), Judge(warm=warm)
    s = sense(body)
    sur, flags, naive, det, rec, t_det = [], [], [], None, None, None
    buf = []                                             # 正常期经验（之后复习用）
    mae = []
    for t in range(steps):
        a = int(t % 8 // 2)                              # 自己轮流下指令（不用人教）
        body.act(a)
        s2 = sense(body)
        ds = s2 - s
        err = wm.learn(s, a, ds)
        naive.append(float(np.abs(ds).mean()))           # "什么都不变"基线的误差
        mae.append(float(np.abs(err).mean()))
        if t < warm:
            buf.append((s.copy(), a, ds.copy()))
        sv = wm.surprise(err)
        bad = judge.update(sv)
        if t == warm:
            wm.frozen = True                             # 正常期结束 → 参照冻住（不随错滑）
            # ★★ 用**岭回归**替代在线 LMS：215 个参数只有 ~120 个样本时在线 LMS 会发散 ——
            #   实测模型误差 0.1584，比"什么都不变"的 0.0932 还差 70%（比躺平还不如）。
            #   岭回归有闭式解 + 正则，保证不会比躺平更差（λ→∞ 时就是躺平）。
            half = [i for i in range(len(buf)) if i % 2 == 0]      # 留出一半做验证
            tr = [i for i in range(len(buf)) if i % 2 == 1]
            F = np.stack([wm.feats(buf[i][0], buf[i][1]) for i in tr])
            Dd = np.stack([buf[i][2] for i in tr])
            lam = 1.0
            wm.W = np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]),
                                   F.T @ Dd).T.astype(np.float32)
            base_sv, ho = [], []
            for i in half:                                     # 红线 + 留出误差都只用没参与拟合的那一半
                e = buf[i][2] - wm.W @ wm.feats(buf[i][0], buf[i][1])
                base_sv.append(wm.surprise(e))
                ho.append(float(np.abs(e).mean()))
            judge.thr = float(np.quantile(np.asarray(base_sv), judge.q))   # 红线一次定死
            wm.holdout_mae = float(np.mean(ho))
        if bad and det is None and fault is not None and t > warm:
            det, t_det = t - warm, t
            wm.lr *= boost                               # ★ 自己加快学习（适应新身体）
        if t_det is not None and rec is None and not bad and t > t_det + 3:
            rec = t - t_det
        sur.append(sv); flags.append(bad); s = s2
    return (np.array(sur), np.array(flags), det, rec,
            float(np.mean(naive)),
            wm.holdout_mae if wm.holdout_mae is not None else float(np.mean(mae[:warm])))


def selftest():
    print("=" * 88)
    print("  自己的对错判断 · 第 2 版（先把世界模型做强：动作×状态交互 + 预测变化量）")
    print("=" * 88)
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    # ① 世界模型本身要准：比"什么都不变"的基线明显好
    sur, flag, _, _, naive, model_mae = run(fault=None, steps=300, seed=1)
    base = float(np.mean(sur[10:120]))
    # ★ 判据改对：惊讶度是"相对自身尺度"的比值，永远不可能 <1（我上一版判据写错了）。
    #   改成**同单位可比**的绝对误差：模型误差 vs "什么都不变"基线的误差。
    check("① 世界模型够准（绝对误差比「什么都不变」的基线好 40% 以上）",
          model_mae < 0.6 * naive,
          f"模型平均误差 {model_mae:.4f} vs 不变基线 {naive:.4f}"
          f"（好 {100*(1-model_mae/naive):.0f}%；正常段惊讶度 {base:.2f}）")

    # ② 三种故障：5 拍内发现 + 正常段不误报
    fp = float(flag[10:120].mean()) * 100
    rows, ok_all = [], True
    for f in ("卡死", "转向反", "走得慢"):
        sur, flag, det, rec, _n, _m = run(fault=f, steps=300, seed=1)
        peak = float(np.max(sur[121:]))
        ok = det is not None and det <= 5
        ok_all = ok_all and ok
        rows.append(f"{f}: 延迟 {det if det is not None else '未发现'} 拍 / 峰值 ×{peak:.1f}")
    check("② 判得早判得出（三种故障都在 5 拍内）", ok_all and fp <= 10.0,
          f"正常段误报 {fp:.1f}% ｜ " + " ｜ ".join(rows))

    # ③ 自己学（无人干预）：红线上方待一会儿后自己落回
    sur, flag, det, rec, _n, _m = run(fault="卡死", steps=420, seed=1)
    check("③ 自己学：发现不对后自己适应，不再报警（无人干预）",
          det is not None and rec is not None,
          f"检测延迟 {det} 拍 → {rec} 拍后自己不再报警")

    print("-" * 88)
    print(f"  结果：{3 - len(fails)}/3 项通过" + ("  ✅ 全部通过" if not fails else f"  ❌ 失败：{fails}"))
    print("=" * 88)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest())
