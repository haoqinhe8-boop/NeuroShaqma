# -*- coding: utf-8 -*-
"""对照实验：单独学 A / 单独学 B 各自能到多少（判 B 任务是否可学）"""
import numpy as np
from xiaosha_cortex import SpikingCortex
from xiaosha_continual import train_task
from xiaosha_hippocampus import evaluate

print("=== 对照：fresh 网络单独训练 ===")
for tid, name in [(0, "只学 A (动作=cls)"), (1, "只学 B (动作=(cls+2)%4)")]:
    cx = SpikingCortex(n=6000, seed=11)
    train_task(cx, tid, rounds=120, seed=9000)
    a = evaluate(cx, 0, n=40)
    b = evaluate(cx, 1, n=40)
    print("  %-22s → A=%.0f%%  B=%.0f%%" % (name, a*100, b*100))

print("\n=== 对照：随机基线（不训练）===")
cx = SpikingCortex(n=6000, seed=11)
print("  未训练                → A=%.0f%%  B=%.0f%%" % (evaluate(cx,0,40)*100, evaluate(cx,1,40)*100))

print("\n=== 顺序学：先A后B（不用锚定，λ=0）===")
cx = SpikingCortex(n=6000, seed=11)
train_task(cx, 0, rounds=120, seed=9000)
print("  学完A                 → A=%.0f%%  B=%.0f%%" % (evaluate(cx,0,40)*100, evaluate(cx,1,40)*100))
train_task(cx, 1, rounds=120, seed=9500)
print("  再学B后               → A=%.0f%%  B=%.0f%%" % (evaluate(cx,0,40)*100, evaluate(cx,1,40)*100))