# -*- coding: utf-8 -*-
"""DMS 容量扩展：从 2 类到 4 类 / 6 类，测 FORCE 工作记忆的上限。

已有：2 类 DMS（4 种组合）→ 100%
本实验：4 类（16 种组合）、6 类（36 种组合），看命中率随难度如何衰减。
"""
import numpy as np
import sys

sys.path.insert(0, r"D:\Apps\MyModel\chat\workspace\xiaosha")
from xiaosha_force import ForceNetwork, evaluate


def dms_trial_n(net, n_cls, cue, test, learn=True, t_cue=100, t_delay=300, t_test=100):
    """N 类 DMS：线索 n_cls 维 + 测试 n_cls 维输入。"""
    net.reset()
    n_in = 2 * n_cls
    total = t_cue + t_delay + t_test
    zs_test = []
    target = 0.0
    for t in range(total):
        u = np.zeros(n_in)
        if t < t_cue:
            u[cue] = 1.0
            tgt = 0.0
        elif t < t_cue + t_delay:
            tgt = 0.0                      # 延迟期无输入
        else:
            u[n_cls + test] = 1.0
            tgt = 1.0 if cue == test else -1.0
            target = tgt
        z, _ = net.step(u, tgt, learn=learn)
        if t >= t_cue + t_delay:
            zs_test.append(z)
    return float(np.mean(zs_test)), target


def eval_n(net, n_cls, trials=60, seed=8000):
    rng = np.random.RandomState(seed)
    hits = 0
    for _ in range(trials):
        cue = int(rng.randint(n_cls))
        test = int(rng.randint(n_cls))
        z, tgt = dms_trial_n(net, n_cls, cue, test, learn=False)
        hits += int((z > 0) == (tgt > 0))
    return hits / trials


def train_n(net, n_cls, trials=80, seed=1000):
    rng = np.random.RandomState(seed)
    for _ in range(trials):
        cue = int(rng.randint(n_cls))
        test = int(rng.randint(n_cls))
        dms_trial_n(net, n_cls, cue, test, learn=True)


if __name__ == "__main__":
    print("=" * 66, flush=True)
    print("DMS 容量扩展：FORCE 工作记忆能同时记住几类？", flush=True)
    print("=" * 66, flush=True)
    for n_cls, n_net, trials in [(2, 600, 80), (4, 800, 100), (6, 1000, 120)]:
        net = ForceNetwork(n=n_net, n_in=2 * n_cls, seed=11)
        base = eval_n(net, n_cls, trials=60, seed=9000)
        train_n(net, n_cls, trials=trials, seed=1000)
        after = eval_n(net, n_cls, trials=60, seed=7000)
        # 长延迟
        rng = np.random.RandomState(6000)
        hits = 0
        for _ in range(40):
            cue = int(rng.randint(n_cls)); tst = int(rng.randint(n_cls))
            z, tgt = dms_trial_n(net, n_cls, cue, tst, learn=False, t_delay=600)
            hits += int((z > 0) == (tgt > 0))
        longp = hits / 40
        ncomb = n_cls * n_cls
        print("%d 类 DMS (%d 种组合, 网络 %d): 训练前 %.0f%% → 训练后 %.0f%% | 长延迟 %.0f%%"
              % (n_cls, ncomb, n_net, base * 100, after * 100, longp * 100), flush=True)
    print("=" * 66, flush=True)
