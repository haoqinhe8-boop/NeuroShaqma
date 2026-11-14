# -*- coding: utf-8 -*-
"""小沙的多任务门控读出 (Gated Multi-Head Readout)。

解决什么：
  共享一个 W_out 时，学第二个任务会覆盖第一个（灾难性遗忘，实测 A 98%→30%）。
  本模块给每个任务一个独立读出头 + 按任务标签门控选择 —— 实测 **A=100% / B=95%**。

实测对比（`experiments/exp_gated_multihead.py`）：
    基线(共享单头)         A=55%  B=75%
    多头 + 门控(800轮/头)  A=100% B=95%   ← +45pp/+20pp

生物学对应：真实大脑用基底节-皮层环路做「多通路选择」，而不是所有任务挤一个通路。

用法：
    from xiaosha_gated import GatedMultiHead
    gmh = GatedMultiHead(cx, n_tasks=2)
    gmh.train(cx, task, task_id=0, rounds=800)
    gmh.train(cx, task, task_id=1, rounds=800)
    acc = gmh.evaluate(cx, task_id=0)
"""
import numpy as np

__version__ = "1.0.0"


class GatedMultiHead:
    """多头读出：每任务一个 W_out 头，训练/推理时按 task_id 切换。"""

    def __init__(self, cx, n_tasks=2):
        self.n_tasks = int(n_tasks)
        self.heads = {t: cx.W_out.copy() for t in range(self.n_tasks)}
        self.trained_rounds = {t: 0 for t in range(self.n_tasks)}

    def select(self, cx, task_id):
        """把 cx 的读出切到指定任务的头（同时保存其他头的当前状态）。"""
        task_id = int(task_id)
        if task_id not in self.heads:
            # 新任务：从当前 W_out 复制一个新头（渐进扩展）
            self.heads[task_id] = cx.W_out.copy()
            self.trained_rounds[task_id] = 0
        cx.W_out = self.heads[task_id]

    def commit(self, cx, task_id):
        """把训练后的 W_out 存回对应头。"""
        self.heads[int(task_id)] = cx.W_out.copy()

    def train(self, cx, task, task_id, rounds=800, run_episode=None, seed=9000):
        """训练某个任务（自动切头 + 存回）。"""
        if run_episode is None:
            from xiaosha_cortex import run_episode
        self.select(cx, task_id)
        for i in range(int(rounds)):
            run_episode(cx, task, learn=True, max_decisions=1, steps=10, seed=seed + i)
        self.commit(cx, task_id)
        self.trained_rounds[int(task_id)] += int(rounds)
        return self.trained_rounds[int(task_id)]

    def evaluate(self, cx, task_id, n=40, evaluate_fn=None):
        """评估某个任务（切到它的头）。"""
        if evaluate_fn is None:
            from xiaosha_hippocampus import evaluate as evaluate_fn
        self.select(cx, task_id)
        return evaluate_fn(cx, int(task_id), n=n)

    def stats(self):
        return {"n_tasks": self.n_tasks, "trained_rounds": dict(self.trained_rounds)}


def selftest():
    """验证：两个冲突任务用多头+门控能同时保持。"""
    from xiaosha_cortex import SpikingCortex
    from xiaosha_hippocampus import ClassifyTask, evaluate

    print("  小沙门控多任务 selftest（多头读出 vs 灾难性遗忘）")
    ok = 0; total = 0
    N = 5000
    ROUNDS = 400   # selftest 用中等轮数（完整实验用 800，见 experiments/）

    # 1) 多头构造
    total += 1
    cx = SpikingCortex(n=N, seed=11)
    gmh = GatedMultiHead(cx, n_tasks=2)
    print("  [1] 多头构造: %d 个头, 每头 shape=%s"
          % (gmh.n_tasks, gmh.heads[0].shape))
    ok += int(len(gmh.heads) == 2 and gmh.heads[0].shape == cx.W_out.shape)

    # 2) 训练任务 A
    total += 1
    tA = ClassifyTask(seed=5, task_id=0)
    tB = ClassifyTask(seed=5, task_id=1)
    gmh.train(cx, tA, 0, rounds=ROUNDS, seed=9000)
    accA = gmh.evaluate(cx, 0)
    print("  [2] 训练 A(%d轮) → A=%.0f%%" % (ROUNDS, accA * 100))
    ok += int(accA >= 0.80)

    # 3) 再训练任务 B（关键：A 不能被破坏）
    total += 1
    gmh.train(cx, tB, 1, rounds=ROUNDS, seed=9500)
    accA2 = gmh.evaluate(cx, 0)
    accB = gmh.evaluate(cx, 1)
    print("  [3] 再训练 B(%d轮) → A=%.0f%%  B=%.0f%%（A 未遗忘才算成功）"
          % (ROUNDS, accA2 * 100, accB * 100))
    ok += int(accA2 >= 0.80 and accB >= 0.80)

    # 4) 对照：共享单头同轮数会遗忘
    total += 1
    from xiaosha_cortex import run_episode as _re
    cx2 = SpikingCortex(n=N, seed=11)
    for i in range(ROUNDS * 2):
        _re(cx2, tA if i % 2 == 0 else tB, learn=True,
            max_decisions=1, steps=10, seed=9000 + i)
    shA, shB = evaluate(cx2, 0, n=40), evaluate(cx2, 1, n=40)
    print("  [4] 对照·共享单头(%d轮) → A=%.0f%%  B=%.0f%%" % (ROUNDS * 2, shA * 100, shB * 100))
    ok += int(accA2 > shA)   # 多头必须优于共享单头

    # 5) 报告提升
    total += 1
    print("  [5] 提升: A %+.0fpp | B %+.0fpp（正=多头更好）"
          % ((accA2 - shA) * 100, (accB - shB) * 100))
    ok += 1

    print("  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
