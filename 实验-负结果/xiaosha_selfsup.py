# -*- coding: utf-8 -*-
"""自主学习的第①件：自监督表征（不用标签、不用奖励，自己从数据里学）。

目标（用户定）：**学得对、效率高、学得好**。所以两个判据都要量：
  A 找结构（不用教）：600 张没标注的帧做 k-means，纯度 / NMI 比原始编码高多少
  B 学习效率：把表征喂给脉冲皮层学"认数字"，到 25% 准确率要多少次、800 次的准确率多少

自监督目标（InfoNCE 对比学习）：同一帧的**两种增强视图**（平移/遮挡/噪声/明暗）应相似，
不同帧应远离。全程不用任何标签、不用任何奖励 —— 学习信号来自数据自身的结构。
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\Apps\MyModel\chat\workspace\xiaosha")
import xiaosha_cortex as C
import xiaosha_digits as D
import xiaosha_vision as V


# ============================================================================
# 自监督头：32 → 64 → 32（ReLU + L2 归一化），InfoNCE 目标，手写反向传播
# ============================================================================
class SSLHead:
    def __init__(self, d_in=32, d_hid=64, d_out=32, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((d_hid, d_in)) * 0.3).astype(np.float32)
        self.b1 = np.zeros(d_hid, dtype=np.float32)
        self.W2 = (rng.standard_normal((d_out, d_hid)) * 0.3).astype(np.float32)
        self.b2 = np.zeros(d_out, dtype=np.float32)

    def forward(self, X):
        """X: (B, d_in) → Z: (d_out, B) 单位向量（保留中间量给反向用）"""
        H = np.maximum(self.W1 @ X.T + self.b1[:, None], 0.0)
        P = self.W2 @ H + self.b2[:, None]
        nrm = np.linalg.norm(P, axis=0, keepdims=True) + 1e-8
        return P / nrm, H, nrm

    def encode(self, X):
        return self.forward(np.atleast_2d(X))[0].T.astype(np.float32)   # (B, d_out)

    def ae_step(self, X, lr=0.05):
        """自编码式自监督：从 32 维瓶颈**重建输入**（MSE）。
        为什么换它：InfoNCE 用 L2 归一化 + 平移/遮挡不变性，把"几段亮"这类**幅度信息**丢了
        （实测 NMI 0.30→0.19，下游 30%→16%）。重建目标天然不能丢信息 —— 重建不出来就罚它。"""
        H = np.maximum(self.W1 @ X.T + self.b1[:, None], 0.0)
        Y = self.W2 @ H + self.b2[:, None]
        E = Y - X.T
        dY = 2.0 * E / E.size
        gW2, gb2 = dY @ H.T, dY.sum(1)
        dH = self.W2.T @ dY
        dH[H <= 0.0] = 0.0
        gW1, gb1 = dH @ X, dH.sum(1)
        self.W1 -= lr * gW1; self.b1 -= lr * gb1
        self.W2 -= lr * gW2; self.b2 -= lr * gb2
        return float((E ** 2).mean())

    def step(self, X1, X2, tau=0.15, lr=0.05):
        B = X1.shape[0]
        g = {k: np.zeros_like(v) for k, v in
             (("W1", self.W1), ("b1", self.b1), ("W2", self.W2), ("b2", self.b2))}
        loss = 0.0
        for X in (X1, X2):
            Z, H, nrm = self.forward(X)
            S = (Z.T @ Z.T.T if False else Z.T @ Z) / tau     # 同批内互相为负样本
            S = Z.T @ Z / tau
            S -= S.max(1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(1, keepdims=True)
            loss += float(-np.log(np.diag(P) + 1e-12).mean())
            dS = (P - np.eye(B, dtype=np.float32)) / B / tau   # (B, B)
            dZ = Z @ dS.T                                      # (d, B)
            dpre = (dZ - Z * (Z * dZ).sum(0, keepdims=True)) / nrm
            g["W2"] += dpre @ H.T
            g["b2"] += dpre.sum(1)
            dH = self.W2.T @ dpre
            dH[H <= 0.0] = 0.0
            g["W1"] += dH @ X
            g["b1"] += dH.sum(1)
        self.W1 -= lr * g["W1"]; self.b1 -= lr * g["b1"]
        self.W2 -= lr * g["W2"]; self.b2 -= lr * g["b2"]
        return loss / 2.0


# ============================================================================
# 增强（自监督的"题目"由数据自己出）
# ============================================================================
def augment(frame, rng=None):
    """★ 增强必须**不破坏"这是几"**：第一版我用了平移 ±2 + 遮挡，等于明着教它丢位置和形状，
    结果表征变差（NMI 0.30→0.19）。所以这里只做"明暗/噪声/±1 微抖"这类不影响身份的变化。"""
    f = frame.copy()
    rng = rng or np.random.default_rng()
    n = f.shape[0]
    f = np.roll(f, int(rng.integers(-1, 2)), axis=1)             # ±1 微抖（不再是 ±2）
    f = np.roll(f, int(rng.integers(-1, 2)), axis=0)
    f = f * float(rng.uniform(0.85, 1.15))                       # 明暗
    f = f + 0.03 * rng.standard_normal(f.shape).astype(np.float32)
    return np.clip(f, 0.0, 1.0)


def kmeans(X, k=10, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        for i in range(k):
            if (lab == i).any():
                c[i] = X[lab == i].mean(0)
    return lab


def purity_nmi(lab, y, k=10):
    n = len(y)
    pur = sum(np.bincount(y[lab == i], minlength=k).max()
              for i in range(k) if (lab == i).any()) / n * 100
    pu = np.bincount(lab, minlength=k) / n
    pv = np.bincount(y, minlength=k) / n
    mi = hu = hv = 0.0
    for i in range(k):
        if pu[i] == 0:
            continue
        hu -= pu[i] * np.log(pu[i])
        for j in range(k):
            pij = float(((lab == i) & (y == j)).mean())
            if pij > 0:
                mi += pij * np.log(pij / (pu[i] * pv[j]))
    for j in range(k):
        if pv[j] > 0:
            hv -= pv[j] * np.log(pv[j])
    return pur, mi / np.sqrt(hu * hv + 1e-12)


# ============================================================================
def selftest(n_pool=3000, steps=400, trial_budget=800):
    print("=" * 86)
    print("  自主学习①：自监督表征（不用标签、不用奖励）")
    print("=" * 86)
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    eye = V.VisionFrontend(n_in=32)
    task = D.DigitTask(seed=3)
    rng = np.random.default_rng(7)

    # 1) 采未标注数据 + 两种增强视图（只用来学，不看标签）
    t0 = time.perf_counter()
    base, aug1, aug2 = [], [], []
    for _ in range(n_pool):
        f = task.new_episode()                     # 标签 task.digit 我们**不用**
        base.append(eye.encode(f))
        aug1.append(eye.encode(augment(f, rng)))
        aug2.append(eye.encode(augment(f, rng)))
    base, aug1, aug2 = np.stack(base), np.stack(aug1), np.stack(aug2)
    t_enc = time.perf_counter() - t0

    # 2) 自监督训练
    head = SSLHead(seed=1)
    head_ae = SSLHead(seed=1)                      # 自编码式（同一个头结构，换目标）
    t0 = time.perf_counter()
    first = last = None
    for it in range(steps):
        idx = rng.choice(n_pool, 128, replace=False)
        l = head.step(aug1[idx], aug2[idx])
        first = l if first is None else first
        last = l
    t_ssl = time.perf_counter() - t0
    check("自监督目标真的在优化（InfoNCE 损失下降）",
          last < first * 0.8,
          f"损失 {first:.2f} → {last:.2f}（{steps} 步，{n_pool} 帧无标签数据，耗时 {t_ssl:.0f}s；"
          f"编码 {t_enc:.0f}s）")
    ae_first = ae_last = None
    for it in range(steps):
        idx = rng.choice(n_pool, 128, replace=False)
        l = head_ae.ae_step(base[idx])
        ae_first = l if ae_first is None else ae_first
        ae_last = l
    check("第二个自监督目标（重建）也在优化",
          ae_last < ae_first * 0.6,
          f"重建误差 {ae_first:.4f} → {ae_last:.4f}")

    # 3) A：找结构（没人教）—— k-means 纯度 / NMI
    task2 = D.DigitTask(seed=99)
    fr, Y = [], []
    for _ in range(600):
        fr.append(task2.new_episode())
        Y.append(task2.digit)
    Y = np.asarray(Y)
    Eraw = np.stack([eye.encode(f) for f in fr])
    CAND = (("原始眼编码（无自监督）", Eraw), ("InfoNCE 对比学习", head.encode(Eraw)),
            ("自编码重建", head_ae.encode(Eraw)))
    res, detail = {}, []
    for name, X in CAND:
        Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
        res[name] = purity_nmi(kmeans(Xn), Y)
        detail.append(f"{name}: 纯度 {res[name][0]:.0f}%/NMI {res[name][1]:.2f}")
    nmi0 = res["原始眼编码（无自监督）"][1]
    nmi_best = max(res[n][1] for n, _ in CAND[1:])
    check("A 找结构（无人教）：自监督表征比原始编码分得更清",
          nmi_best > nmi0 + 0.05,
          " ｜ ".join(detail) + "（瞎分 NMI≈0）")

    # 4) B：下游学习效率（脉冲皮层认数字）
    def train_down(mode, trials=trial_budget):
        cx = C.SpikingCortex(n=20_000, k=360, n_in=32, n_act=10, seed=11)
        tk = D.DigitTask(seed=3)
        hist, t25 = [], None
        for i in range(trials):
            f = tk.new_episode()
            e = eye.encode(f).reshape(-1)
            if mode == "raw":
                obs = e
            else:
                h = (head if mode == "nce" else head_ae).encode(e).reshape(-1)
                obs = (h + 1.0) * 0.5                            # 归一化到 0~1 喂皮层
            cx.reset_feat()
            for _ in range(C.DECIDE_STEPS):
                cx.step(obs[:cx.n_in])
            a = cx.decide(temp=1.2, sample=True)
            r = 1.0 if a == tk.digit else -1.0
            cx.stdp_reward(r)
            cx.learn_returns([(cx._last_xf.copy(), cx._last_probs.copy(), a, r)], gamma=0.0)
            hist.append(1 if r > 0 else 0)
            if t25 is None and i >= 200 and np.mean(hist[-200:]) >= 0.25:
                t25 = i
        return float(np.mean(hist[-200:])) * 100, t25

    down = {}
    for mode, name in (("raw", "原始眼编码（无自监督）"), ("nce", "InfoNCE 对比学习"),
                       ("ae", "自编码重建")):
        down[mode] = (name,) + train_down(mode)
    acc_raw = down["raw"][1]
    best_mode = max(("nce", "ae"), key=lambda m: down[m][1])
    acc_best, t25_best = down[best_mode][1], down[best_mode][2]
    check("B 学习效率：自监督表征让下游学得更快/更准",
          acc_best > acc_raw + 5,
          " ｜ ".join(f"{n}: 800 次 {a:.0f}%/到25%用 {t if t else '未达到'} 次"
                      for _m, (n, a, t) in ((m, down[m]) for m in ("raw", "nce", "ae"))))

    print("-" * 86)
    print(f"  结果：{2 - len(fails)}/2 项通过" + ("  ✅ 全部通过" if not fails else f"  ❌ 失败：{fails}"))
    print("=" * 86)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pool", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=400)
    a = ap.parse_args()
    sys.exit(selftest(n_pool=a.pool, steps=a.steps))
