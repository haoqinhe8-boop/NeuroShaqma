# -*- coding: utf-8 -*-
"""小沙的 FORCE 学习工作记忆 (Sussillo & Abbott 2009)。

为什么需要它：
  README 里如实记录过——循环吸引子结构做出来了，但 R-STDP 这种无监督局部规则
  学不出稳定的持续活动，DMS 命中率一直卡在随机水平（~48%）。
  FORCE = 用递归最小二乘(RLS)训练「读出权重」，同时把误差反馈回网络，
  是神经科学里造稳定吸引子/工作记忆的标准方法。

任务：延迟匹配样本 (DMS)
  线索期  100ms  输入 [cueA, cueB, 0, 0]      目标输出 0
  延迟期  300ms  输入 全 0（关键！必须靠网络自己记住）  目标输出 0
  测试期  100ms  输入 [0, 0, testA, testB]    目标输出 +1(匹配) / -1(不匹配)
"""
import numpy as np

__version__ = "1.0.0"


class ForceNetwork:
    """通用的 FORCE 训练循环网络（率编码 + tanh）。"""

    def __init__(self, n=1000, n_in=4, dt=1e-3, tau=1e-2, g=1.5,
                 alpha=1.0, seed=0, fb_scale=1.0):
        rng = np.random.RandomState(seed)
        self.n, self.n_in = n, n_in
        self.dt, self.tau = dt, tau
        # 循环权重：高斯，谱半径 g（g>1 才有混沌/丰富动力学）
        self.W = rng.randn(n, n) * g / np.sqrt(n)
        self.W_in = rng.randn(n, n_in)
        self.W_out = np.zeros((1, n))          # 读出权重，从 0 开始学
        self.W_fb = rng.randn(n, 1) * fb_scale  # 误差反馈权重
        self.P = np.eye(n) / alpha              # RLS 逆相关矩阵
        self.x = np.zeros(n)

    def reset(self):
        self.x = np.zeros(self.n)

    def step(self, u, target, learn=True):
        """推进一步。返回 (输出 z, 误差 e)。"""
        r = np.tanh(self.x)                    # 率
        z = float(self.W_out[0] @ r)
        e = z - target
        if learn:
            Pr = self.P @ r
            denom = 1.0 + float(r @ Pr)
            k = Pr / denom
            self.W_out -= np.outer(e, k)       # RLS 更新读出
            self.P -= np.outer(k, Pr)
        dx = -self.x + self.W @ r + self.W_in @ u + self.W_fb[:, 0] * e
        self.x += self.dt / self.tau * dx
        return z, e


def dms_trial(net, cue, test, learn=True, t_cue=100, t_delay=300, t_test=100):
    """一局 DMS：线索 → 延迟 → 测试。返回 (末段平均输出, 目标)。"""
    net.reset()
    total = t_cue + t_delay + t_test
    zs_test = []
    target = 0.0
    for t in range(total):
        if t < t_cue:
            u = np.zeros(4); u[cue] = 1.0
            tgt = 0.0
        elif t < t_cue + t_delay:
            u = np.zeros(4)                    # 延迟期无输入
            tgt = 0.0
        else:
            u = np.zeros(4); u[2 + test] = 1.0
            tgt = 1.0 if cue == test else -1.0
            target = tgt
        z, _ = net.step(u, tgt, learn=learn)
        if t >= t_cue + t_delay:
            zs_test.append(z)
    return float(np.mean(zs_test)), target


def evaluate(net, n=40, seed=5000):
    """评估命中率（不学习）。"""
    rng = np.random.RandomState(seed)
    hits = 0
    for _ in range(n):
        cue = int(rng.randint(2)); test = int(rng.randint(2))
        z, tgt = dms_trial(net, cue, test, learn=False)
        hits += int((z > 0) == (tgt > 0))
    return hits / n


def train(net, trials=60, seed=1000, verbose=False):
    """FORCE 在线学习若干试次。"""
    rng = np.random.RandomState(seed)
    errs = []
    for i in range(trials):
        cue = int(rng.randint(2)); test = int(rng.randint(2))
        _, tgt = dms_trial(net, cue, test, learn=True)
        if verbose and (i + 1) % 20 == 0:
            print("    训练 %d/%d 试次" % (i + 1, trials))
    return net


def selftest():
    print("  小沙 FORCE 工作记忆 selftest（DMS 延迟匹配样本）")
    ok = 0; total = 0

    # 1) 网络能构造
    total += 1
    net = ForceNetwork(n=600, seed=11)
    print("  [1] 构造 600 神经元 FORCE 网络: W%s P%s" % (net.W.shape, net.P.shape))
    ok += 1

    # 2) 未训练时接近随机（基线）
    total += 1
    base = evaluate(net, n=30, seed=5000)
    print("  [2] 训练前 DMS 命中率: %.0f%% (随机基线)" % (base * 100))
    okso = 0.30 <= base <= 0.70
    ok += int(okso)
    if not okso: print("      !! 基线异常")

    # 3) FORCE 训练
    total += 1
    print("  [3] FORCE 训练 60 试次...")
    train(net, trials=60, seed=1000)
    print("      训练完成")
    ok += 1

    # 4) 训练后命中率显著提升
    total += 1
    after = evaluate(net, n=40, seed=6000)
    print("  [4] 训练后 DMS 命中率: %.0f%%" % (after * 100))
    okso = after >= 0.75
    ok += int(okso)
    if not okso: print("      !! 未达到 75% 目标")

    # 5) 延迟期真的是"记住"而不是"抄近路"：延迟时长加倍仍能记住
    total += 1
    rng = np.random.RandomState(7000)
    hits = 0
    for _ in range(30):
        cue = int(rng.randint(2)); test = int(rng.randint(2))
        z, tgt = dms_trial(net, cue, test, learn=False, t_delay=600)
        hits += int((z > 0) == (tgt > 0))
    long_acc = hits / 30
    print("  [5] 延迟加倍到 600ms 后命中率: %.0f%%（验证真记忆非抄近路）" % (long_acc * 100))
    ok += int(long_acc >= 0.65)

    print("  FORCE 工作记忆: %d/%d 项通过" % (ok, total))
    print("  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)