# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""临时：探大规模上限（50 万 / 100 万神经元，K=8）"""
import sys
sys.path.insert(0, r"D:\Apps\MyModel\chat\workspace\xiaosha")
import xiaosha_cortex as C

print("=" * 78)
print("  规模上限探测（K=8，稳态）")
print("=" * 78)
C.bench(ns=(500_000, 1_000_000), ks=(8,), steps=30)
print(f"  果蝇：{C.FLY_NEURONS:,} 神经元 / {C.FLY_SYNAPSES:,} 突触")
print("=" * 78)
