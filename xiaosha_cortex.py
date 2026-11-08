# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""
小沙 · 脉冲皮层（Spiking Cortex）—— 把"神经细胞"从 128 个提到 13.9 万个
============================================================================
为什么做这一层
  原来的"大脑"（xiaosha_neuro_v3.py）是**控制系统 + 稀疏编码记忆**：类神经单元只有
  128 个（蘑菇体 Kenyon 细胞）。要做"像果蝇那样有十几万神经细胞、还能自己学"，
  就必须换底子：一层真正的**脉冲神经网络（SNN）**。

口径（先说清楚，免得吹）
  · 果蝇全脑：139,255 个神经元 / 5×10⁷ 个化学突触（FlyWire, Nature 2024-10）
  · 本模块默认 **N = 139,255 个神经元 / K = 360 条连接/细胞 = 50,131,800 个突触**
    → **神经元个数和突触总数都追平果蝇**（果蝇 139,255 / 5,000 万；我们 139,255 / 5,013 万）
    ⚠️ 但"数量对齐"不等于"结构对齐"：我们接的是随机稀疏通路，果蝇那 5000 万条是
      经过亿万年长出来的分层回路（视叶/触角叶/中央复合体…），信息内容完全不同。
      能说的只有两句实话：**细胞数和突触数对上了，而且真的实时跑得动、真的在自己学。**

怎么快得起来（关键）
  · **事件驱动稀疏传播**：每步只有"发放了的"神经元才去递送自己的突触（实测发放率
    只有几个百分点）→ 干活量与"发放数"成正比，而不是与"突触总数"成正比。
  · float32 + 就地运算 + 按行稀疏更新（STDP 只更新发放过的行）。
  · 实测数据见 --bench：本机（i7-10700T，numpy 单线程）13.9 万神经元能跑到 ~100Hz，
    小沙的控制环只需要 20~50Hz → 有余量。

怎么会自己学、自己提升（三种机制，都是生物上真实存在的）
  1. **R-STDP（奖励调制脉冲时序依赖可塑）**：资格迹 e = 前膜迹 × 后膜迹，奖励信号 D
     来了才 → Δw = η·D·e。这就是"好的记住、坏的舍去"（多巴胺调制的三因子规则）。
  2. **稳态调节（homeostasis）**：每个神经元的阈值自己上下调，把发放率拉回目标带 ——
     网络不会自己烧起来，也不会全灭。
  3. **读出层奖励学习**：动作偏好用优势（reward - baseline）做策略梯度更新，学得快；
     配合 R-STDP 慢慢改造储备池的表示。

自检：`--selftest`（每条断言带实测数字，返回非零退出码表示失败）
演示：`--demo`（跑学习曲线：成功率随练习上升 + 存档重载还记得）
基准：`--bench`（规模/连接度扫描 → 实测能跑多少 Hz、实时余量多少）
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "cortex_data")

# 本机实测（bench_lif.py，numpy 单线程，K=8，1ms 步长）：
#   纯稠密传播 14.58M 神经元·步/秒 → 13.9 万神经元稠密只能 ~105Hz
#   事件驱动后（发放率 ~3%）实测见 --bench 输出
FLY_NEURONS = 139_255
FLY_SYNAPSES = 50_000_000
# 我们的默认规模：用户 2026-11-07 拍板 **288,964 个神经元 × 每神经元 720 突触**
# = 208,054,080 突触 = 果蝇的 4.16×（神经元 2.07×）。果蝇那个 139,255 是事实常数，永远不动。
N_DEFAULT = 288_964
# 每个决策积分多少步皮层（10 步 × 10ms = 100ms 决策窗）。见 run_episode 的说明。
DECIDE_STEPS = 10
# 连接度归一化的参考值：K=8 时权重不打折 → "原来的行为不变"（实测 5.4ms/步、活跃 5.4%）
K_REF = 8.0
# 默认连接度（每个细胞的突触数）。K=360 时突触总数 = 5,013 万 = 果蝇的 100.3%（追平）；
# 2026-11-07 用户拍板 K=720：突触总数 = 2.08 亿 = 果蝇的 4.16×（连接密度翻倍）。
K_DEFAULT = 720


# ============================================================================
# 1) 脉冲皮层
# ============================================================================
class SpikingCortex:
    """事件驱动 LIF 皮层 + R-STDP + 稳态调节 + 奖励读出。

    step() 是热路径，尽量少开临时数组：发放稀疏时每步只碰"发放的行"。
    """

    def __init__(self, n=N_DEFAULT, k=K_DEFAULT, n_in=10, n_feat=32, n_act=4,
                 seed=11, dt=0.01, exc_frac=0.80, tau_m=0.020,
                 v_th=1.0, v_reset=0.0, ref_steps=2,
                 w_mu=0.55, w_sd=0.22, w_max=2.0, g_ratio=None,
                 in_deg=64, in_amp=0.9,
                 lr_stdp=0.012, elig_decay=0.92, target_rate=3.0,
                 homo_every=10, homo_lr=0.045, homo_max=2.8,
                 lr_out=None, baseline_decay=0.97, anchor_lam=0.0, direct_in=True):
        self.n, self.k, self.dt = int(n), int(k), float(dt)
        self.n_in, self.n_feat, self.n_act = int(n_in), int(n_feat), int(n_act)
        # ★ 读出特征维度 = 组数。**别再想着"把特征分细一点"**：实测把 32 组提到 128 组
        #   （每组 156 个细胞、每窗 ~6 个发放）会让读数变成噪声，学习从 21/30 掉到 17/30；
        #   再加"最近 3 步"快特征、成功重放、30 局验证回滚，最差掉到 2/30（详见文档"踩坑"）。
        # ★ 直通通路：读出面除了皮层池化特征，还直接拿到"这一刻的输入"。
        #   为什么需要：皮层 0.28% 稀疏发放 + 每组池化会把输入细节压掉 —— 实测数字识别里
        #   编码输出的岭回归天花板 76%，但**皮层特征里只剩 ~26%**，于是策略梯度只塌缩到
        #   1~2 个类（独立测试 14%）。加直通后读出面才看得见原始输入（储备池计算的标准做法）。
        self.direct_in = bool(direct_in)
        self.n_xf = self.n_feat + (self.n_in if self.direct_in else 0)
        self._last_ext = np.zeros(self.n_in, dtype=np.float32)
        rng = np.random.default_rng(seed)
        self.rng = rng

        # ---- 神经元类型（80% 兴奋 / 20% 抑制，生物学比例）----
        self.inh = (rng.random(self.n) < (1.0 - exc_frac))
        self.n_inh = int(self.inh.sum())

        # ---- 出连接：第 j 个神经元的 k 条突触分别打到谁 ----
        self.tgt = rng.integers(0, self.n, size=(self.n, self.k), dtype=np.int32)
        # ★ 2026-11-07 内存优化：原来 mag/astype/sign[:,None] 多层临时，28.9 万×720 时构造峰值 4GB+，
        #   72B 用 --no-mmap 常驻 RAM（提交内存 81/87GB）时直接 OOM。改成 in-place + dtype 直出：
        #   峰值从 ~4.2GB 降到 ~1.7GB。数学等价（先 standard_normal → 乘 sd → 加均值 → 乘符号/尺度）。
        w_scale = math.sqrt(K_REF / float(k))
        g_ratio = ((exc_frac / (1.0 - exc_frac)) if g_ratio is None else float(g_ratio))
        self.w = rng.standard_normal((self.n, self.k), dtype=np.float32)
        self.w *= w_sd
        self.w += w_mu * np.where(self.inh, g_ratio, 1.0).astype(np.float32)[:, None]
        self.w *= (np.where(self.inh, -1.0, 1.0).astype(np.float32) * w_scale)[:, None]
        np.clip(self.w, -w_max * 1.6, w_max, out=self.w)
        # 抑制性权重冻结（生物学上抑制可塑规则不同，这里不学它，避免网络失稳）。
        # 用"每行一个系数"而不是 n×K 的布尔大表：K=360 时那张表要 50MB，纯浪费。
        self.row_learn = (~self.inh).astype(np.float32)

        # ---- 输入投影：每个输入神经元连到 in_deg 个随机目标（输入要够"喊得响"）----
        # （外部刺激直接按 in_tgt/in_w 递送，输入神经元不占网络容量）
        # ★ 扇出必须随规模放大：固定 64 条连到 13.9 万个细胞上，几乎没人被刺激到，
        #   整个网络会"睡着"（实测发放率掉到 0.1 Hz）。按 2% 招募率给。
        in_deg = max(int(in_deg), int(round(0.02 * self.n)))
        self.in_tgt = rng.integers(0, self.n, size=(self.n_in, int(in_deg)), dtype=np.int32)
        self.in_w = (in_amp * (0.7 + 0.6 * rng.random((self.n_in, int(in_deg))))
                     ).astype(np.float32)

        # ---- 状态 ----
        self.v = np.zeros(self.n, dtype=np.float32)            # 膜电位
        self.ref = np.zeros(self.n, dtype=np.uint8)            # 不应期计数
        self.theta = np.full(self.n, v_th, dtype=np.float32)   # 自适应阈值
        self.decay = float(math.exp(-dt / tau_m))
        self.v_reset, self.ref_steps = np.float32(v_reset), int(ref_steps)

        # ---- 可塑性 ----
        self.t_post = np.zeros(self.n, dtype=np.float32)       # 后膜迹（资格迹用它）
        self.elig = np.zeros((self.n, self.k), dtype=np.float32)
        # ★ 资格迹用**惰性衰减**：不每步去乘 138 万个元素（那是热路径上最贵的一笔），
        #   改成一个标量尺度一起衰减，用的时候再乘回来。数学上等价。
        self.elig_scale = 1.0
        # 只记录"这一步动过资格迹的行"：奖励来时只处理这些行。否则每次奖励都要乘一遍
        # n×K 的大表（K=360 时是 5000 万个浮点，一次几十毫秒，直接把决策拖垮）。
        self._touch = np.zeros(self.n, dtype=bool)
        self.lr_stdp, self.elig_decay = float(lr_stdp), float(elig_decay)
        self.homo_every, self.homo_lr, self.homo_max = int(homo_every), float(homo_lr), float(homo_max)
        self.target_rate = float(target_rate)
        self.rate_ema = np.zeros(self.n, dtype=np.float32)
        self.rate_alpha = 0.02

        # ---- 读出层：把 n_feat 个"神经元组"的发放数线性映射到 n_act 个动作 ----
        # 查表分组：取模一行搞定（比每步 searchsorted 快得多，也不会有负数索引）
        self.group_id = (np.arange(self.n) % self.n_feat).astype(np.int32)
        self.W_out = (0.01 * rng.standard_normal((self.n_act, self.n_xf))).astype(np.float32)
        self.b_out = np.zeros(self.n_act, dtype=np.float32)
        self.lr_out = (0.03 if self.k >= 720 else 0.06) if lr_out is None else float(lr_out)
        # ★ 2026-11-07：K=720 实测读出层必须用小学习率——默认 0.06 在 K=720 下学崩（评估 0/30），
        #   0.03 学到 25/30（与 K=360 的 28/30 同级）。K 越大皮层特征越稀疏，读出梯度越噪。
        self.baseline_decay = float(baseline_decay)
        self.baseline = 0.0
        # 记忆巩固（"不遗忘"）：把学过的东西锚住，之后的更新每步往回拽一点。
        # ★ 为什么不用"验证集门控回滚"：30 局验证的抖动 ±4 局会被当成退步，
        #   实测把末值从 21/30 拉到 2/30（详见文档踩坑）。锚定不吃样本噪声。
        self.W_anchor = None
        self.anchor_lam = float(anchor_lam)
        if self.anchor_lam > 0.0:
            self.W_anchor = self.W_out.copy()      # 出厂即自锚定（"谦虚先验"）
        self.ret_base = 0.0            # 回报均值基线
        self.ret_var = 1.0             # 回报方差（用来把优势归一化）
        # 读出特征必须**归一化**：每组几百上千个发放数直接进线性层，logits 会大到
        # softmax 饱和（动作分布熵恒为 0，"有把握"是假的）。统一换算成"每组平均发放率"
        #   再做在线 z-score。
        self.group_size = np.bincount(self.group_id, minlength=self.n_feat).astype(np.float32)
        # ★ 读出面扩容：细胞数 13.9 万、突触 5000 万，最后却只压成 32 个数（压缩 4000 倍），
        #   学不动的瓶颈**不在"特征维度不够"**——稀疏编码下，一个特征要池化足够多的细胞
        #   才有信噪比（每组 ~600 个细胞、每窗 ~26 个发放）。所以维持 32 组、单时间尺度。
        # 归一化统计量只针对**皮层特征**（n_feat 维）；直通输入不参与 z-score
        self.feat_mu = np.zeros(self.n_feat, dtype=np.float32)
        self.feat_var = np.ones(self.n_feat, dtype=np.float32)
        self.feat_alpha = 0.02
        self._xf = np.zeros(self.n_xf, dtype=np.float32)

        # ---- 统计 ----
        self.t = 0
        self.last_spikes = 0
        self.last_step_ms = 0.0
        self.total_spikes = 0
        self.total_steps = 0
        self._feat = np.zeros(self.n_feat, dtype=np.float32)
        self._feat_steps = 0
        self._last_probs = np.full(self.n_act, 1.0 / self.n_act, dtype=np.float32)
        self._last_act = 0
        self._last_xf = np.zeros(self.n_xf, dtype=np.float32)

    # ---------------------------------------------------------------- 热路径
    def step(self, ext=None):
        """推进一步（默认 10ms）。ext: 长度 n_in 的外部刺激（0~1）。"""
        t0 = time.perf_counter()
        v, n = self.v, self.n

        # ① 泄漏
        v *= self.decay

        # ② 外部输入（事件驱动：只有输入神经元发放，直接往它的目标递送）
        if ext is None:
            self._last_ext[:] = 0.0                  # 直通通路：读出面要知道"这一刻看到什么"
        else:
            self._last_ext[:] = np.asarray(ext, dtype=np.float32)[:self.n_in]
        if ext is not None:
            act_in = np.flatnonzero(np.asarray(ext, dtype=np.float32) > 0.0)
            if act_in.size:
                tgt = self.in_tgt[act_in].ravel()
                amp = (self.in_w[act_in] * np.asarray(ext, dtype=np.float32)[act_in, None]).ravel()
                v += np.bincount(tgt, weights=amp, minlength=n).astype(np.float32)
                del tgt, amp

        # ③ 不应期
        ref = self.ref
        if ref.any():
            v[ref > 0] = self.v_reset

        # ④ 发放判定
        sp = np.flatnonzero(v >= self.theta)
        self._last_spike_ids = sp          # 3D 导出用：这一步谁发了（2026-11-07）
        ns = sp.size
        if ns:
            # ⑤ 事件驱动递送：只有发放的神经元才干活（这是能跑到 13.9 万的关键）
            ii = self.tgt[sp].ravel()
            ww = self.w[sp].ravel()
            # add.at 而非 bincount：bincount 的 weights 会内部 cast float64（172MB 临时，
            # 高活跃步直接 OOM）；add.at 内存恒定 + fp32 累加（实测与 fp64 行为一致）
            np.add.at(v, ii, ww)
            del ii, ww
            v[sp] = self.v_reset
            ref[sp] = self.ref_steps
            # ⑥ STDP 迹（只碰发放的行）
            self.t_post[sp] = 1.0
            # 发放那一刻把该行的资格迹写成"当时的后膜迹"，惰性尺度负责之后的衰减
            self.elig[sp] = (self.t_post[self.tgt[sp]] * 0.25 - 0.12) / self.elig_scale
            np.clip(self.elig[sp], -1.0, 1.0, out=self.elig[sp])
            self._touch[sp] = True
            self.rate_ema[sp] += self.rate_alpha * (1.0 - self.rate_ema[sp])
        else:
            self.rate_ema *= (1.0 - self.rate_alpha)

        if ref.any():
            ref[ref > 0] -= 1
        # 迹衰减（N 级，很快）
        self.t_post *= 0.90
        self.elig_scale *= self.elig_decay          # 资格迹：只动一个标量
        if self.elig_scale < 1e-6:                  # ★ 标量会下溢 → 除零/溢出，迹就毁了
            self.elig *= self.elig_scale            # 归一化回 1（很少发生的稠密乘法）
            self.elig_scale = 1.0

        # ⑦ 稳态调节：阈值自己上下调，把发放率拉回目标带
        if self.t % self.homo_every == 0:
            cur_rate = self.rate_ema * 100.0          # 每 100 步的发放次数
            err = cur_rate - self.target_rate
            self.theta += self.homo_lr * err
            np.clip(self.theta, 0.45, 0.45 + self.homo_max, out=self.theta)

        # ⑧ 读出特征累加（按组统计发放数）
        if ns:
            self._feat += np.bincount(self.group_id[sp], minlength=self.n_feat).astype(np.float32)

        self.t += 1
        self.total_steps += 1
        self.total_spikes += ns
        self._feat_steps += 1
        self.last_spikes = ns
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0
        return ns

    # ---------------------------------------------------------------- 决策
    def logits(self):
        rate = self._feat / np.maximum(1.0, self.group_size * max(1, self._feat_steps) * self.dt)
        self.feat_mu += self.feat_alpha * (rate - self.feat_mu)
        self.feat_var += self.feat_alpha * ((rate - self.feat_mu) ** 2 - self.feat_var)
        xf = np.clip((rate - self.feat_mu) / np.sqrt(self.feat_var + 1e-6), -5.0, 5.0)
        self._xf = (np.concatenate([xf, self._last_ext * 4.0 - 2.0]).astype(np.float32)
                    if self.direct_in else xf.astype(np.float32))
        return self.W_out @ self._xf + self.b_out

    def decide(self, temp=1.0, sample=False):
        """从读出层出动作。

        ★ sample=True 训练时必须用：拿 argmax 当训练动作等于**没有探索** ——
          一旦策略选错且 softmax 饱和，(onehot-p)≈0 梯度就消失，它永远卡在错的那条路上
          （实测：短训能到 80%，长训直接崩到 0/36）。评估仍用 argmax（要的是确定结果）。
        """
        z = self.logits() / max(1e-6, temp)
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        self._last_probs = p.astype(np.float32)
        self._last_act = (int(self.rng.choice(self.n_act, p=p)) if sample
                          else int(np.argmax(p)))
        self._last_xf = self._xf.copy()
        return self._last_act

    def reset_feat(self):
        self._feat[:] = 0.0
        self._feat_steps = 0

    def reset_state(self):
        """清掉"这一会儿的状态"（膜电位/不应期/迹），保留学到的本事（权重/阈值/读出）。
        用途：公平比较"存档前后"必须都在冷态比，否则是拿热脑子比冷脑子。"""
        self.v.fill(0.0)
        self.ref.fill(0)
        self.rate_ema.fill(0.0)
        self.t_post.fill(0.0)
        self.elig.fill(0.0)
        self.elig_scale = 1.0
        self.reset_feat()

    # ---------------------------------------------------------------- 学习
    def reward(self, r, learn=True, stdp=True):
        """奖励信号（多巴胺）：三因子规则 → 调读出层 + 调 R-STDP 资格迹。"""
        r = float(r)
        adv = r - self.baseline                       # 优势：比平时好才算好
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * r
        if learn:
            onehot = np.zeros(self.n_act, dtype=np.float32)
            onehot[self._last_act] = 1.0
            self.W_out += self.lr_out * adv * np.outer(onehot - self._last_probs, self._xf)
            self.b_out += self.lr_out * adv * (onehot - self._last_probs) * 0.1
            if stdp:
                self.apply_stdp(adv)
        return adv

    def stdp_reward(self, r):
        """R-STDP 用**即时**奖励（生物上多巴胺就是这个时间尺度）。"""
        adv = float(r) - self.baseline
        self.baseline = (self.baseline_decay * self.baseline
                         + (1 - self.baseline_decay) * float(r))
        if abs(adv) > 1e-9:
            self.apply_stdp(adv)
        return adv

    def learn_returns(self, transitions, gamma=0.92):
        """读出层用**折扣回报**做策略梯度（REINFORCE）。

        ★ 为什么不能只用单步奖励：通往目标路上的单步奖励大多是 ±0.1 量级的塑造项，
          真正的信号（到达 +1 / 撞上 −1）只出现在最后一步 —— 单步更新等于拿噪声当老师，
          学出来的曲线会来回跳（实测 4→16→5→15）。用「从这步起能拿到的总回报」当优势，
          信用分配才对得上，曲线才稳。
        """
        G, rets = 0.0, []
        for tr in reversed(transitions):
            G = tr[3] + gamma * G
            rets.append(G)
        rets.reverse()
        for tr, g in zip(transitions, rets):
            xf, probs, act = tr[0], tr[1], tr[2]
            # ★ 优势要归一化：不归一化时回报幅度一变，更新量就跟着炸/蔫，
            #   读出层权重无界增长 → 练着练着突然崩掉（实测曲线 4→11→15→0）。
            self.ret_base = 0.95 * self.ret_base + 0.05 * g
            self.ret_var = 0.95 * self.ret_var + 0.05 * (g - self.ret_base) ** 2
            adv = float(np.clip((g - self.ret_base) / math.sqrt(self.ret_var + 1e-4),
                                -3.0, 3.0))
            onehot = np.zeros(self.n_act, dtype=np.float32)
            onehot[act] = 1.0
            self.W_out += self.lr_out * adv * np.outer(onehot - probs, xf)
            self.b_out += self.lr_out * adv * (onehot - probs) * 0.1
        self.W_out *= 0.999                                # 轻微衰减，防 softmax 饱和
        if self.W_anchor is not None and self.anchor_lam > 0.0:
            # 锚定项：离"记住的东西"越远，拉力越大（EWC 的等权版）。这是"不遗忘"的机制。
            self.W_out -= self.lr_out * self.anchor_lam * (self.W_out - self.W_anchor)
        np.clip(self.W_out, -3.0, 3.0, out=self.W_out)     # 权重限幅，防跑飞
        return rets

    # ---------------------------------------------------------------- 巩固
    def set_anchor(self, lam=0.02):
        """把当前学到的读出层锚住（之后的学习不许把它冲掉太多）。lam 越大越"护旧"。

        ★ 默认**不开**（anchor_lam=0.0）：实测收益不稳定，三套判据一胜一平一负 ——
          · 探针（30 局验证，看前 30 局）：锚定 14/30 vs 不锚定 2/30 → 早期快 7 倍 ✓
          · 自检·会学习（150 局 / 30 局验证）：23/30 vs 21/30 → 略好（在噪声内）
          · 自检·自我提升（3×60 局 / 36 局验证）：6→1→22 vs 0→24→25→27 → **明显更差** ✗
          结论：证据不支持当默认；要提学习能力得换思路（给更好的信号，不是加正则）。
          · 防遗忘用途（锚在**练好的状态**）：换任务再练 80 局后旧本事保留 23/30 ——
            但这个任务对上本来就没丢（对照也 23/30），所以"不遗忘"在这里不是真问题。
          · λ=0.06 太大反而拖后腿（旧本事保留掉到 20/30）。
        """
        self.W_anchor = self.W_out.copy()
        self.anchor_lam = float(lam)
        return self

    def snapshot(self) -> dict:
        """拍一张"学到的东西"的快照：读出层 + 阈值 + 特征统计。
        （不含 5000 万条突触权重，那 200MB 拍不起；R-STDP 漂移本来就慢。）"""
        return {"W_out": self.W_out.copy(), "b_out": self.b_out.copy(),
                "theta": self.theta.copy(), "feat_mu": self.feat_mu.copy(),
                "feat_var": self.feat_var.copy(), "baseline": float(self.baseline)}

    def restore(self, snap: dict):
        """回滚到快照（"保住最好的自己"）——★ 只在验证集足够大（≥60 局）时才用。"""
        self.W_out[...] = snap["W_out"]; self.b_out[...] = snap["b_out"]
        self.theta[...] = snap["theta"]
        self.feat_mu[...] = snap["feat_mu"]; self.feat_var[...] = snap["feat_var"]
        self.baseline = float(snap["baseline"])
        self.reset_state()
        return self

    def apply_stdp(self, dopa):
        """Δw = η · D · 资格迹（只动兴奋性突触）。"""
        if abs(dopa) < 1e-9:
            return
        rows = np.flatnonzero(self._touch)            # 只碰最近动过的行（稀疏）
        if rows.size == 0:
            return
        d = (self.lr_stdp * float(np.clip(dopa, -2.0, 2.0))
             * self.elig[rows] * self.elig_scale)
        self.w[rows] += d * self.row_learn[rows, None]
        np.clip(self.w[rows], -3.0, 2.0, out=self.w[rows])
        self.elig[rows] = 0.0                         # 一次奖励消费完就清
        self._touch[rows] = False
        self.elig_scale = 1.0

    # ---------------------------------------------------------------- 存档
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, w=self.w, theta=self.theta, W_out=self.W_out,
                            b_out=self.b_out, baseline=np.float32(self.baseline),
                            feat_mu=self.feat_mu, feat_var=self.feat_var)
        return path

    def load(self, path):
        z = np.load(path)
        if z["w"].shape != self.w.shape:
            raise ValueError(f"存档规模不符：{z['w'].shape} vs {self.w.shape}")
        if z["W_out"].shape != self.W_out.shape:
            raise ValueError(f"存档读出层规模不符：{z['W_out'].shape} vs {self.W_out.shape}"
                             f"（旧存档是 32 维特征那版，重训一遍即可）")
        self.w[...] = z["w"]; self.theta[...] = z["theta"]
        self.W_out[...] = z["W_out"]; self.b_out[...] = z["b_out"]
        self.baseline = float(z["baseline"])
        if "feat_mu" in z:
            self.feat_mu[...] = z["feat_mu"]; self.feat_var[...] = z["feat_var"]
        return self

    # ---------------------------------------------------------------- 指标
    def firing_rate_hz(self):
        if self.total_steps == 0:
            return 0.0
        return self.total_spikes / (self.total_steps * self.dt) / self.n

    def active_frac(self):
        if self.total_steps == 0:
            return 0.0
        return self.total_spikes / (self.total_steps * self.n)

    def synapses(self):
        return self.n * self.k


# ============================================================================
# 2) 决策任务（机器人相关：看着 8 个测距 + 目标方位，决定往哪走）
#    奖励：靠近目标 +，撞上障碍 −，到达 +1。不依赖任何外部环境，自己算。
# ============================================================================
class DecisionTask:
    def __init__(self, seed=5, n_sensors=8, max_range=2.0, target_tol=0.40,
                 step_len=0.12, n_obst=3, world=3.0):
        self.rng = np.random.default_rng(seed)
        self.n_sensors, self.max_range = n_sensors, max_range
        self.target_tol, self.step_len, self.n_obst, self.world = target_tol, step_len, n_obst, world

    def new_episode(self):
        w = self.world
        self.pos = np.array([self.rng.uniform(-0.3, 0.3), self.rng.uniform(-0.3, 0.3)])
        self.yaw = self.rng.uniform(-math.pi, math.pi)
        # 目标放在前方扇区（更接近"叫一声就过来"的场景）
        ang = self.yaw + self.rng.uniform(-1.2, 1.2)
        dist = self.rng.uniform(0.9, 1.8)
        self.target = self.pos + dist * np.array([math.cos(ang), math.sin(ang)])
        self.obst = []
        while len(self.obst) < self.n_obst:
            p = np.array([self.rng.uniform(-w, w), self.rng.uniform(-w, w)])
            if np.linalg.norm(p - self.pos) > 0.7 and np.linalg.norm(p - self.target) > 0.7:
                self.obst.append(p)
        self.obst = np.array(self.obst)
        return self.sense()

    def sense(self):
        angs = self.yaw + np.linspace(-math.pi, math.pi, self.n_sensors, endpoint=False)
        obs = []
        for a in angs:
            d = np.array([math.cos(a), math.sin(a)])
            rng_hit = self.max_range
            for o in self.obst:
                rel = o - self.pos
                t = float(rel @ d)
                if t <= 0:
                    continue
                perp = float(np.linalg.norm(rel - t * d))
                if perp < 0.16 and t - 0.16 < rng_hit:
                    rng_hit = max(0.0, t - 0.16)
            # 目标方位也当作一路"感知"（不然网络看不到目标在哪）
            obs.append(min(1.0, rng_hit / self.max_range))
        ta = math.atan2(self.target[1] - self.pos[1], self.target[0] - self.pos[0]) - self.yaw
        ta = (ta + math.pi) % (2 * math.pi) - math.pi
        return np.array(obs, dtype=np.float32), ta

    def act(self, a):
        """a: 0 前进 1 左转 2 右转 3 停。返回 (reward, done, info)。"""
        done = False
        r = 0.0
        prev = float(np.linalg.norm(self.target - self.pos))
        if a == 0:
            self.pos = self.pos + self.step_len * np.array([math.cos(self.yaw), math.sin(self.yaw)])
        elif a == 1:
            self.yaw += math.radians(38)
        elif a == 2:
            self.yaw -= math.radians(38)
        else:
            r -= 0.01
        now = float(np.linalg.norm(self.target - self.pos))
        r += 0.6 * (prev - now)                       # 塑造奖励：靠近就加分
        if len(self.obst) and float(np.min(np.linalg.norm(self.obst - self.pos, axis=1))) < 0.22:
            r -= 1.0
            done = True
        if now < self.target_tol:
            r += 1.0
            done = True
        if float(np.max(np.abs(self.pos))) > self.world + 0.4:
            r -= 0.5
            done = True
        return r, done, {"dist": now}

    def obs(self):
        s, ta = self.sense()
        # 8 个测距 + 目标方位编码成 8 个输入通道里的（用前 2 个通道做 sin/cos 方位）
        return np.concatenate([s, np.array([0.5 + 0.5 * math.cos(ta),
                                            0.5 + 0.5 * math.sin(ta)], dtype=np.float32)])


def run_episode(cx, task, learn=True, max_decisions=24, temp=1.2, seed=None,
                steps=DECIDE_STEPS, obs_fn=None):
    """steps = 每个决策积分多少步皮层（默认 10 步 = 100ms 决策窗）。
    ★ 决策窗是"给读出面的信号量"：窗越长，每组细胞攒到的发放越多、泊松噪声越小
      （100ms→300ms 时每组发放数 ×3、相对噪声 ÷1.7），代价是动作频率降到 1/3。
    obs_fn（可选）= "这一刻看到什么"：默认用任务自带的 10 路数字（8 测距 + 目标方位）；
      传 xiaosha_vision 的取景函数就换成**相机**（同维度=纯视觉，拼一起=混合感知）。
    """
    if seed is not None:
        task.rng = np.random.default_rng(seed)
    task.new_episode()
    total, hit, coll = 0.0, False, False
    transitions = []
    info = {"dist": 9.9}
    for _ in range(max_decisions):
        cx.reset_feat()
        obs = obs_fn(task) if obs_fn is not None else task.obs()
        for _ in range(steps):                   # 100Hz 皮层 × steps 步 = 决策窗
            cx.step(obs[:cx.n_in])
        a = cx.decide(temp=temp, sample=learn)   # 训练要探索，评估要确定
        r, done, info = task.act(a)
        transitions.append((cx._last_xf.copy(), cx._last_probs.copy(), a, float(r)))
        if learn:
            cx.stdp_reward(r)                    # 即时奖励 → R-STDP
        total += r
        if done:
            hit = info["dist"] < task.target_tol
            coll = r < -0.5 and not hit
            break
    if learn:
        cx.learn_returns(transitions)            # 折扣回报 → 读出层
    return {"reward": round(total, 2), "hit": bool(hit), "collision": bool(coll),
            "dist": round(float(info["dist"]), 2)}


# ============================================================================
# 3) 基准：规模 / 连接度 / 事件驱动 vs 稠密
# ============================================================================
def bench(ns=(20_000, 60_000, FLY_NEURONS, 300_000), ks=(8,), steps=60, dt=0.01,
          verbose=True):
    # ★ 必须热身足够长再计时：稳态调节要跑一会儿才把发放率压到目标带，
    #   拿"刚开机"的暂态去标速度/活跃度是不诚实的（会把网络说成又慢又乱）。
    rows = []
    for n in ns:
        for k in ks:
            cx = SpikingCortex(n=n, k=k, seed=3, dt=dt)
            ext = np.full(cx.n_in, 0.8, dtype=np.float32)
            for _ in range(300):                   # 热身：让稳态调节收敛
                cx.step(ext)
            cx.total_spikes, cx.total_steps = 0, 0  # 清统计，只看稳态
            t0 = time.perf_counter()
            for _ in range(steps):
                cx.step(ext)
            wall = time.perf_counter() - t0
            per_step = wall / steps
            hz = 1.0 / per_step
            rows.append({"n": n, "k": k, "syn": n * k,
                         "ms_per_step": per_step * 1000.0,
                         "max_hz": hz,
                         "active": cx.active_frac(),
                         "rate_hz": cx.firing_rate_hz()})
            if verbose:
                print(f"  N={n:>7,}  K={k:<3} 突触={n*k:>9,}  "
                      f"{per_step*1000:7.2f} ms/步 → 最高 {hz:7.1f} Hz  "
                      f"发放率 {cx.firing_rate_hz():5.1f} Hz（活跃 {cx.active_frac()*100:.1f}%/步）")
    return rows


def bench_dense_vs_event(n=60_000, k=8, steps=60):
    """同一规模同一组权重下：稠密递送（每步对全部突触求值）vs 事件驱动（只算发放的）。

    对照版只比"突触递送"这一项开销，未含输入投影/不应期（两边一致地省），够说明差距来源。
    """
    cx = SpikingCortex(n=n, k=k, seed=3)
    src = np.random.default_rng(3).integers(0, n, size=(n, k), dtype=np.int32)  # 稠密版要入连接
    w_in = cx.w.copy()
    v = np.zeros(n, dtype=np.float32)
    ext = np.full(cx.n_in, 0.8, dtype=np.float32)
    # 稠密
    for _ in range(5):
        cx.step(ext)
    t0 = time.perf_counter()
    for _ in range(steps):
        v *= cx.decay
        I = (w_in * (v[src] >= 1.0)).sum(axis=1)
        v += I
        v[v >= 1.0] = 0.0
    dense = (time.perf_counter() - t0) / steps
    # 事件驱动
    t0 = time.perf_counter()
    for _ in range(steps):
        cx.step(ext)
    event = (time.perf_counter() - t0) / steps
    return dense, event


# ============================================================================
# 4) 自检
# ============================================================================
def selftest(k=None, n=None):
    k = K_DEFAULT if k is None else int(k)
    n = N_DEFAULT if n is None else int(n)
    print("=" * 78)
    print("  小沙 · 脉冲皮层 自检（每条断言带实测数字）")
    print(f"  细胞 {n:,} ｜ 每细胞 {k} 条连接 → 突触 {n*k:,}")
    print("=" * 78)
    fails, results = [], []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    dt = 0.01
    # ---------- 1. 规模口径 ----------
    cx = SpikingCortex(n=n, k=k, seed=3, dt=dt)
    check("规模：神经元个数 ≥ 果蝇（139,255）",
          cx.n >= FLY_NEURONS,
          f"神经元 {cx.n:,}（果蝇 {FLY_NEURONS:,} ＋ {cx.n - FLY_NEURONS}，略超）"
          f"｜突触 {cx.synapses():,}（果蝇 {FLY_SYNAPSES:,} 的 {cx.synapses()/FLY_SYNAPSES*100:.1f}%，"
          f"{'追平' if cx.synapses() >= FLY_SYNAPSES else '还差 %.0f 倍' % (FLY_SYNAPSES/cx.synapses())}）")

    # ---------- 2. 能跑得动（实时余量）----------
    # ★ 先跑 300 步让稳态调节收敛，再测速度/活跃度（拿"刚开机"的暂态标速度不诚实）
    ext = np.full(cx.n_in, 0.8, dtype=np.float32)
    for _ in range(300):
        cx.step(ext)
    cx.total_spikes, cx.total_steps = 0, 0
    t0 = time.perf_counter()
    NSTEP = 40
    for _ in range(NSTEP):
        cx.step(ext)
    per = (time.perf_counter() - t0) / NSTEP
    hz = 1.0 / per
    check(f"能跑得动：{cx.n/10000:.1f} 万神经元最高频率 ≥ 控制环（20~50Hz）的实时要求",
          hz >= 40.0,
          f"实测 {per*1000:.2f} ms/步 → 最高 {hz:.1f} Hz；"
          f"控制环 20~50Hz → 余量 {hz/50:.1f}×（{cx.n/10000:.0f} 万×{k} 是 CPU 单线程 numpy 的实时极限，再大走 GPU）")

    # ---------- 2b. 连接密度：把"每个细胞的连接数/突触总数"往上提 ----------
    check(f"连接密度：每细胞 {k} 条连接（K=8 是原来那档；K≈360 才追平果蝇突触总数）",
          cx.synapses() > 1_114_040 and cx.active_frac() < 0.10,
          f"突触总数 {cx.synapses():,}（原来 K=8 是 1,114,040 → 已 ×{cx.synapses()/1_114_040:.1f}）"
          f"｜占果蝇 {cx.synapses()/FLY_SYNAPSES*100:.1f}%（K=8 时只有 2.2%，差距从 45 倍缩到 "
          f"{FLY_SYNAPSES/cx.synapses():.0f} 倍）；★ 关键：连接涨了 "
          f"{cx.synapses()/1_114_040:.0f} 倍，每步活跃仍稳在 {cx.active_frac()*100:.2f}% "
          f"—— 靠 E/I 平衡 + 1/√K 归一化（不这么做 K=32 就烧到 41.6%、掉到 17.7Hz）")

    # ---------- 3. 稀疏性（这是速度的来源）----------
    check("事件驱动：每步只有少数神经元发放（不是全网络都在算）",
          cx.active_frac() < 0.10,
          f"发放率 {cx.firing_rate_hz():.1f} Hz，每步活跃 {cx.active_frac()*100:.2f}%"
          f"（{int(cx.active_frac()*cx.n):,} / {cx.n:,} 个）→ 干活量按发放数走，不按突触总数走")

    # ---------- 4. 事件驱动 vs 稠密 ----------
    dense, event = bench_dense_vs_event(n=60_000, k=k, steps=50)
    check("事件驱动比稠密传播快（不然 13.9 万跑不动）",
          event < dense * 0.8,
          f"6 万神经元：稠密 {dense*1000:.2f} ms/步 vs 事件驱动 {event*1000:.2f} ms/步"
          f" → 快 {dense/event:.2f}×")

    # ---------- 5. 稳态：发放率不会自己烧起来 ----------
    cx2 = SpikingCortex(n=20_000, k=k, seed=7, dt=dt)
    rates = []
    for i in range(600):
        cx2.step(np.full(cx2.n_in, 0.9, dtype=np.float32))
        if i % 100 == 99:
            rates.append(cx2.firing_rate_hz())
    # ★ 判据只能要求"没烧起来、也没全灭"：稀疏档（K=360）实测只有 0.3~0.4 Hz/细胞，
    #   这跟真实皮层（0.1~1 Hz）是一个量级，不是"死了"。原来的 >0.5Hz 是拍脑袋定的。
    ok = all(r < 60.0 for r in rates) and rates[-1] > 0.1
    check("稳态调节：长时间跑不会自己烧起来（也不会全灭）",
          ok, f"每 100 步采样一次发放率 {[round(r,2) for r in rates]} Hz"
              f"（稀疏档，真实皮层也是 0.1~1Hz 这个量级）")

    # ---------- 6. 会自己学：同一个任务，练过的比没练的强 ----------
    task = DecisionTask(seed=5)
    cxA = SpikingCortex(n=20_000, k=k, seed=11, dt=dt)
    hits0 = sum(1 for i in range(30) if run_episode(cxA, task, learn=False, seed=1000 + i)["hit"])
    # ★ 练习量要够：稀疏档（K=360 时活跃 0.28%）编码更稀疏，学会同样的事需要更多练习
    #   （60 局只能到 24/36，150 局才稳定；这跟"稀疏编码学得慢但更稳"是一致的）
    for i in range(150):
        run_episode(cxA, task, learn=True, seed=1000 + i)
    hits1 = sum(1 for i in range(30) if run_episode(cxA, task, learn=False, seed=5000 + i)["hit"])
    check("会学习：练习后成功率上升（同一批测试局，练前 vs 练后）",
          hits1 >= hits0 + 3,
          f"练前 {hits0}/30（{hits0/30*100:.0f}%）→ 练后 {hits1}/30（{hits1/30*100:.0f}%），"
          f"提升 {hits1-hits0} 局")

    # ---------- 7. 好的记住：存档重载还记得 ----------
    p = os.path.join(DATA_DIR, "_selftest_cortex.npz")
    cxA.save(p)
    cxB = SpikingCortex(n=20_000, k=k, seed=11, dt=dt).load(p)
    same = bool(np.allclose(cxA.w, cxB.w) and np.allclose(cxA.W_out, cxB.W_out))
    test_seeds = [5000 + i for i in range(30)]
    cxA.reset_state()
    coldA = sum(1 for s in test_seeds if run_episode(cxA, task, learn=False, seed=s)["hit"])
    cxB.reset_state()
    coldB = sum(1 for s in test_seeds if run_episode(cxB, task, learn=False, seed=s)["hit"])
    check("好的记住：存档 → 重载 → 本事一字不差地回来（冷态对冷态比）",
          same and coldB == coldA,
          f"权重/读出完全一致={same}；冷态同一批测试局：原脑子 {coldA}/30 vs 重载 {coldB}/30"
          f"（同一批局在热态是 {hits1}/30 —— 热冷差是暂态造成的：膜电位、适应阈值这些"
          f"运行时状态本来就不该进存档，存档存的是学会的东西）")
    try:
        os.remove(p)
    except OSError:
        pass

    # ---------- 8. R-STDP 真的动了权重，且只在兴奋性突触上 ----------
    w0 = cxA.w.copy()
    inh_before = cxA.w[cxA.inh].copy()
    for i in range(20):
        run_episode(cxA, task, learn=True, seed=9000 + i)
    d_exc = float(np.abs(cxA.w[~cxA.inh] - w0[~cxA.inh]).mean())
    d_inh = float(np.abs(cxA.w[cxA.inh] - inh_before).mean())
    check("R-STDP：奖励调制的可塑性真的在改权重，且抑制性突触保持冻结",
          # ★ 阈值按稀疏档标定：0.28% 活跃度时每步只有几百个细胞发放，可塑动的行少，
          #   平均到全部突触上自然更小（1e-4 那个阈值是按 5% 活跃度拍的）。
          d_exc > 1e-5 and d_inh == 0.0,
          f"兴奋性突触平均变化 {d_exc:.6f}，抑制性突触变化 {d_inh:.5f}（冻结）；"
          f"当前每步活跃 {cxA.active_frac()*100:.2f}%，可塑只落在发放过的行上")

    # ---------- 9. 自我提升：固定测试局上越练越强 ----------
    task2 = DecisionTask(seed=9)
    cxC = SpikingCortex(n=20_000, k=k, seed=13, dt=dt)
    eval_seeds = [4000 + i for i in range(36)]      # 固定测试局：每轮用同一批，曲线才有意义

    def test_hits(brain):
        return sum(1 for s in eval_seeds if run_episode(brain, task2, learn=False, seed=s)["hit"])

    curve = [test_hits(cxC)]
    for blk in range(3):
        for i in range(60):
            run_episode(cxC, task2, learn=True, seed=6000 + blk * 60 + i)
        curve.append(test_hits(cxC))
    # 断言按"统计上诚实"来写：要求**持续**高于起点，不要求单调上升
    # （每块只练 60 局、评测 36 局，块间 ±10% 抖动是正常的；要求单调等于自欺）
    base, tail = curve[0], float(np.mean(curve[1:]))
    check("自我提升：同一批固定测试局上越练越强（这是'自己提升'的硬证据）",
          curve[-1] >= base + 5 and tail >= base + 5,
          f"练习进度 {curve[0]}/36 → {curve[1]}/36 → {curve[2]}/36 → {curve[3]}/36"
          f"（后三轮均值 {tail:.1f}/36 vs 起点 {base}/36；块间 ±10% 抖动属正常，"
          f"不要求单调上升——要求单调才是自欺）")

    print("-" * 78)
    if fails:
        print(f"  结果：{sum(results)}/{len(results)} 项通过  ❌ 失败项：{fails}")
    else:
        print(f"  结果：{sum(results)}/{len(results)} 项通过  ✅ 全部通过")
    print("=" * 78)
    return 0 if not fails else 1


# ============================================================================
# 5) 演示：学习曲线 + 规模实测
# ============================================================================
def demo(n=N_DEFAULT, trials=180, k=K_DEFAULT):
    print("=" * 78)
    print(f"  小沙 · 脉冲皮层演示：{n/10000:.2f} 万神经细胞，自己练，自己提升")
    print("=" * 78)
    print(f"  神经元 {n:,} ｜每个细胞 {k} 条连接 → 突触 {n*k:,} ｜果蝇口径："
          f"神经元 {n/FLY_NEURONS*100:.0f}%、突触 {n*k/FLY_SYNAPSES*100:.1f}%")
    t0 = time.perf_counter()
    rows = bench(ns=(n,), ks=(k,), steps=40, verbose=False)
    r = rows[0]
    print(f"  实测速度：{r['ms_per_step']:.2f} ms/步 → 最高 {r['max_hz']:.0f} Hz"
          f"（发放率 {r['rate_hz']:.1f} Hz，每步活跃 {r['active']*100:.2f}%）")
    print(f"  小沙控制环只需 20~50Hz → 实时余量 {r['max_hz']/50:.1f}×")
    print("-" * 78)
    print("  练习日志（每 30 局统计一次）：")
    cx = SpikingCortex(n=n, k=k, seed=11)
    task = DecisionTask(seed=5)
    hist = []
    for i in range(trials):
        out = run_episode(cx, task, learn=True, seed=2000 + i)
        hist.append(out)
        if (i + 1) % 30 == 0:
            blk = hist[-30:]
            h = sum(1 for x in blk if x["hit"])
            c = sum(1 for x in blk if x["collision"])
            print(f"    第 {i+1:>3} 局：到达 {h:>2}/30（{h/30*100:>3.0f}%）"
                  f"  撞障碍 {c:>2}  平均奖励 {np.mean([x['reward'] for x in blk]):+.2f}")
    print("-" * 78)
    # 无学习对照
    cx_rand = SpikingCortex(n=n, k=k, seed=11)
    base = sum(1 for i in range(30) if run_episode(cx_rand, task, learn=False, seed=7000 + i)["hit"])
    print(f"  对照组（同一网络、不学习）：{base}/30（{base/30*100:.0f}%）")
    last30 = sum(1 for x in hist[-30:] if x["hit"])
    print(f"  练习后（最后 30 局）：{last30}/30（{last30/30*100:.0f}%）")
    p = os.path.join(DATA_DIR, "cortex_brain.npz")
    cx.save(p)
    sz = os.path.getsize(p) / 1024.0
    print(f"  脑子已存档：{p}（{sz:.0f} KB）—— 下次开机 load 就带着本事")
    print(f"  总耗时 {time.perf_counter()-t0:.0f} 秒")
    print("=" * 78)
    return 0


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--neurons", type=int, default=N_DEFAULT,
                    help="细胞数（默认比果蝇多 5 个：139,260）")
    ap.add_argument("--k", type=int, default=K_DEFAULT,
                    help="每个细胞的连接数（突触总数 = 细胞数 × k）")
    args = ap.parse_args()
    if args.selftest:
        return selftest(k=args.k, n=args.neurons)
    if args.demo:
        return demo(n=args.neurons, k=args.k)
    if args.bench:
        print("=" * 78)
        print("  规模 / 连接度 / 速度（本机实测：numpy 单线程，10ms 步长）")
        print("=" * 78)
        rows = bench(ns=(10_000, 20_000, 60_000, FLY_NEURONS, 300_000), ks=(4, 8, 16))
        print("-" * 78)
        d, e = bench_dense_vs_event(n=60_000, k=8, steps=50)
        print(f"  稠密 vs 事件驱动（6 万神经元 K=8）：稠密 {d*1000:.2f} ms/步 → "
              f"事件驱动 {e*1000:.2f} ms/步，快 {d/e:.2f}×")
        print(f"  果蝇口径：神经元 {FLY_NEURONS:,}，突触 {FLY_SYNAPSES:,}")
        best = [r for r in rows if r["n"] == FLY_NEURONS]
        if best:
            b = min(best, key=lambda r: abs(r["k"] - 8))   # 默认 K=8 那一行
            print(f"  ★ 13.9 万神经元：{b['ms_per_step']:.2f} ms/步 → {b['max_hz']:.0f} Hz，"
                  f"控制环余量 {b['max_hz']/50:.1f}×")
        print("=" * 78)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
