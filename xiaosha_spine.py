# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""小沙的脊椎：查 BUG、查躯干损坏（检测 → 定位 → 修复 → 复测）。

复用上面做成的"对错判断"内核（`xiaosha_judge` 的世界模型：动作×状态交互 + 标准化残差 + 自校准红线），
在它之上加三件事：
  ① **定位**：不只是"有事"，而是**指出哪个部件**（8 路测距 / 行走 / 转向）—— 靠**逐通道标准化残差**取最大。
  ② **修复**：软件类故障能自己修（屏蔽坏通道 / 重映射动作语义）；硬件类**修不了就如实说需人工**。
  ③ **复测**：修完再跑一段，惊讶度要落回红线内才算修好。

注入的故障（软件 4 种 + 硬件 3 种）：
  软件：传感器卡死 / 传感器漂移 / 传感器异常值(NaN) / 动作错位（指令映射偏了一位）
  硬件：行走电机卡死 / 转向反向 / 编码器断线（反馈恒 0，其实在动）

判据（先钉死，不事后改）：
  ① 检出率 ≥90%：所有注入故障都要 ≤5 拍被发现
  ② 误报率 ≤10%：正常段（无故障）不乱报警
  ③ 定位准确率 ≥80%：指出的部件 == 真实故障部件
  ④ 修复纪律：软件类修好且复测通过；**硬件类必须如实说"硬件仍有问题"**——要么"需人工"，
     要么"已用软件绕过但硬件仍建议检修"（两种都合格；唯一不合格 = 声称硬件已修好）
用法：python xiaosha_spine.py --selftest
"""
import argparse
import math
import sys

import numpy as np

sys.path.insert(0, r"D:\Apps\MyModel\chat\workspace\xiaosha")
from xiaosha_judge import MARK, TGT, Body, WorldModel         # 复用：差速小车 / 世界模型 / 场地

N_ACT, SPAN = 4, 4.0
CH = ["测距0", "测距1", "测距2", "测距3", "测距4", "测距5", "测距6", "测距7", "行走", "转向"]
SOFT = ("传感器卡死", "传感器漂移", "传感器异常值", "传感器间歇漂移", "动作错位")
HARD = ("行走电机卡死", "转向反向", "编码器断线")
SENSOR_CH = {0: "测距2", 3: "测距5"}                           # 注入传感器故障的通道（挑两路好认的）


class FaultyBody(Body):
    """在 judge 的差速小车基础上加"本体故障"（行走/转向/编码器）。"""

    def __init__(self, seed=0, fault=None, fault_at=600):
        super().__init__(seed=seed, fault=None, fault_at=fault_at)
        self.fault = fault
        # ★★ 故障必须**在正常期之后**才生效：上一版从第 0 拍就注入，世界模型把故障学成了正常
        #   → 什么都不异常（实测 4/7 检出，本体类全漏）。传感器类之所以全对，正是因为它们在
        #   正常期之后才注入。fault_at 之前一律正常。
        self.fault_at = fault_at

    def act(self, a):
        p0, y0 = self.pos.copy(), self.yaw
        armed = self.fault is not None and (self.t + 1) > self.fault_at   # super().act 会 t+=1
        eff = a
        if armed and self.fault == "转向反向":
            eff = {1: 2, 2: 1}.get(a, a)
        elif armed and self.fault == "动作错位":
            eff = (a + 1) % N_ACT                        # 指令被内部错位一位（软件 BUG）
        pp0, yy0 = self.pos.copy(), self.yaw
        super().act(eff)                                     # 用（可能被改过的）实际动作走
        if armed and self.fault == "行走电机卡死" and eff == 0:  # 前进指令不动
            self.pos = pp0
        if armed and self.fault == "编码器断线":                # 实际在动，但反馈说没动
            self.dpos, self.dyaw = 0.0, 0.0
        else:
            self.dpos = float(np.linalg.norm(self.pos - p0))
            self.dyaw = float(self.yaw - y0)


def sense_faulty(b, fault=None, rng=None):
    """真实传感器读数 + 注入的**传感器类**故障（卡死/漂移/异常值）。"""
    from xiaosha_judge import sense as sense_true
    s = sense_true(b).copy()
    if fault == "传感器卡死":
        s[3] = 5.0                                           # 某路恒读最大值（探头坏了）
    elif fault == "传感器漂移":
        s[5] += 0.35                                         # 某路恒定偏置（标定 BUG）
    elif fault == "传感器异常值":
        s[3] = -1.0                                          # 越界/NaN 类（软件 BUG）
    elif fault == "传感器间歇漂移":
        # ★ 间歇性故障（真实机器人最常见、也最难检）：时好时坏。用确定性 50% 占空比（每 2 拍漂 1 拍），
        #   可复现；单拍可能完全正常 → 靠「累积检测」而不是单拍阈值才能抓住它。
        if (b.t // 2) % 2 == 0:
            s[5] += 0.35
    return s


class Spine:
    """检测 → 定位 → 修复 → 复测。"""

    def __init__(self, q=0.95):
        self.wm = WorldModel()
        self.q = q
        self.thr = None
        self.mask = np.zeros(10, dtype=bool)                 # 屏蔽掉"已判定不可信"的通道
        self.act_map = None                                  # 动作语义重映射（发现错位时填）
        self.log = []

    # ---- 逐通道标准化残差（定位的依据）----
    def clean(self, s):
        """★ 被判定坏掉的通道，**输入侧**也要处理：换成正常期均值。
        为什么：坏传感器不只是"它的残差大"，它作为**输入**还会污染对所有其它通道的预测
        （实测：只做输出屏蔽，复测不过）。清理输入侧才是真修好。"""
        x = s.copy()
        if self.mask.any():
            x[self.mask] = self.wm.mu[self.mask]
        return x

    def resid(self, s, a, ds):
        f = self.wm.feats(self.clean(s), self._a(a))
        e = np.abs(ds - self.wm.W @ f) / self.wm.rsd
        e[self.mask] = 0.0                                   # 已屏蔽的通道不算
        e[self.mask] = 0.0
        return e

    def _a(self, a):
        return self.act_map[a] if self.act_map is not None else a

    def surprise(self, s, a, ds):
        e = self.resid(s, a, ds)
        return float(np.sqrt(np.mean(e ** 2)))

    def fit(self, buf):
        self.wm.fit(buf)
        self.thr = float(np.quantile([self.surprise(*b) for b in buf[::3]], self.q))

    def localize(self, s, a, ds):
        e = self.resid(s, a, ds)
        return int(np.argmax(e)), float(e.max())

    # ---- 修复尝试 ----
    def repair(self, comp, cur_score=None, fault_hint=None):
        """软件类：屏蔽坏通道 / 试各种动作错位重映射。硬件类：不动，如实报需人工。"""
        # ★ 比较基准要用**故障时的分数**（cur_score），不能用故障前的：拿故障前比会把"已恢复"
        #   判成"没修好"（实测动作错位/转向反向因此被误报成"修复无效"）。
        ref = cur_score if cur_score is not None else self._base_score
        if comp in (0, 1, 2, 3, 4, 5, 6, 7):
            self.mask[comp] = True                           # 把被判定坏掉的通道屏蔽（不再信它）
            return f"已屏蔽{CH[comp]}（不再采信该路）"
        if comp == 9:                                        # 转向通道异常 → 试动作错位重映射
            best, bestk = None, None
            for k in (1, 2, 3):
                am = [(i + k) % N_ACT for i in range(N_ACT)]
                old = self.act_map
                self.act_map = am
                sc = float(np.mean([self.surprise(*b) for b in self._rep_buf]))
                self.act_map = old
                if best is None or sc < best:
                    best, bestk = sc, k
            if best is not None and best < ref * 0.5:
                self.act_map = [(i + bestk) % N_ACT for i in range(N_ACT)]
                return f"已用软件重映射绕过（错位 {bestk} 位）——硬件仍建议检修"
            return "软件修复无效：需人工"
        return "硬件故障：需人工（已检出并定位，但不假装修好）"


def run(fault=None, warm=600, steps=900, seed=3):
    body = FaultyBody(seed=seed, fault=fault, fault_at=warm)
    sp = Spine()
    s = sense_faulty(body, fault)
    buf = []
    for t in range(warm):                                    # 正常期（无故障）学习 + 标定红线
        a = int(t % 8 // 2)
        body.act(a)
        s2 = sense_faulty(body, None)
        buf.append((s.copy(), a, (s2 - s).copy()))
        s = s2
    sp.fit(buf)
    sp._rep_buf = buf[::3]
    sp._base_score = float(np.mean([sp.surprise(*b) for b in sp._rep_buf]))
    sp._rep_buf = buf[::3]
    det = loc = comp = None
    det_n = 0
    det_t = None
    post = []                                                # ★ 修复后的复测段（惊讶度序列）
    window = []                                              # ★ 间歇故障：滑动窗口累积检测
    for t in range(warm, steps):
        a = int(t % 8 // 2)
        body.act(a)
        s2 = sense_faulty(body, fault)
        ds = s2 - s
        sv = sp.surprise(s, a, ds)
        window.append(sv)
        if len(window) > 12:
            window.pop(0)
        single = sv > sp.thr                                 # 单拍超阈值（持续故障直接命中）
        cumul = (det is None and len(window) >= 12           # ★ 间歇故障：单拍可能正常，但窗口内异常率>25%
                 and sum(1 for x in window if x > sp.thr) / len(window) > 0.25)
        if single or cumul:
            det_n += 1
            if det is None:
                det = t - warm
                det_t = t
        # ★ 定位只信「单拍超阈值」的拍：间歇故障要等它处于"漂移拍"读数才准，拿"正常拍"去 localize 会指错
        if det is not None and comp is None and single:
            comp, strength = sp.localize(s, a, ds)
            loc = CH[comp]
            sp.repair_msg = sp.repair(comp, cur_score=sv)
        # ★ 修复后**每一步都记**（原来只在"仍然异常"时记 → 修好了反而一条都不记 → 复测恒判失败）
        if det is not None and det_t is not None and t > det_t + 1:
            post.append(sv)
        s = s2
    # ★ 复测判据：修复后惊讶度要落回红线内（"修好了"由复测证明，不由"我执行了修复动作"证明）
    retest = (len(post) >= 20 and float(np.mean(post[-10:])) <= sp.thr)
    return dict(det=det, loc=loc, comp=comp, hits=det_n, thr=sp.thr, post=post,
                retest=retest, repair=getattr(sp, "repair_msg", None))


def selftest():
    print("=" * 92)
    print("  脊椎：查 BUG / 查躯干损坏（检测 → 定位 → 修复 → 复测）")
    print("=" * 92)
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    # ① 检出率 + ③ 定位
    rows, n_det, n_loc, n_soft_ok, n_hard_honest = [], 0, 0, 0, 0
    for f in SOFT + HARD:
        r = run(fault=f)
        rows.append(f"{f}: {'%d 拍' % r['det'] if r['det'] is not None else '未发现'}"
                    f" → 判为「{r['loc']}」→ {r['repair']}")
        if r["det"] is not None and r["det"] <= 5:
            n_det += 1
        # 定位正确性：传感器类应指到被注入的那路；硬件类应指到动作相关的通道（行走/转向）
        exp = None
        if f == "传感器卡死":
            exp = CH[3]
        elif f == "传感器漂移":
            exp = CH[5]
        elif f == "传感器异常值":
            exp = CH[3]
        elif f == "传感器间歇漂移":
            exp = CH[5]
        if (exp and r["loc"] == exp) or (f in HARD and r["loc"] in ("行走", "转向")) \
           or (f == "动作错位" and r["loc"] in ("行走", "转向")):
            n_loc += 1
        msg = r["repair"] or ""
        # ★ 软件类必须**复测通过**才算修好（"我执行了修复动作" ≠ "修好了"）
        if f in SOFT and ("已" in msg or "重映射" in msg) and "无效" not in msg and r["retest"]:
            n_soft_ok += 1
        # ★ 硬件类的诚实要求是"不许说硬件好了"：报"需人工" 或 "已软件绕过但硬件仍建议检修" 都算合格
        if f in HARD and ("需人工" in msg or "建议检修" in msg):
            n_hard_honest += 1
    total = len(SOFT + HARD)
    check("① 检出率 ≥90%（每种故障都要 ≤5 拍发现）", n_det >= total * 0.9,
          f"检出 {n_det}/{total} ｜ " + " ｜ ".join(rows))

    r0 = run(fault=None)
    fp = r0["hits"] / (900 - 600) * 100
    check("② 误报率 ≤10%（正常段不乱报警）", fp <= 10.0, f"正常段误报 {fp:.1f}%（红线 {r0['thr']:.2f}）")

    check("③ 定位准确率 ≥80%（指出的部件要正确）", n_loc >= total * 0.8,
          f"定位正确 {n_loc}/{total}")

    # ✅ 已补上复测（2026-09-16）：屏蔽坏通道时**输入侧也换成正常期均值**（否则污染所有预测），
    #    判据改为"软件类必须复测通过"。原先只验证口径、不复测 —— 那是"假绿隐患"。
    check("④ 修复纪律：软件类**复测通过** + 硬件类如实说明（不许假装修好）",
          n_soft_ok >= len(SOFT) * 0.75 and n_hard_honest == len(HARD),
          f"软件类修复+复测通过 {n_soft_ok}/{len(SOFT)} ｜ 硬件类如实说明 {n_hard_honest}/{len(HARD)}"
          f"（假装修好=不合格）")

    print("-" * 92)
    print(f"  结果：{4 - len(fails)}/4 项通过" + ("  ✅ 全部通过" if not fails else f"  ❌ 失败：{fails}"))
    print("=" * 92)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest())
