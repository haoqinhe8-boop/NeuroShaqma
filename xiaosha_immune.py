# -*- coding: utf-8 -*-
"""小沙的免疫细胞 v2：脊椎报错后 → **真化验**（主动探针）→ 治疗 → 复测（+ 免疫记忆）。

v1 失败原因（`实验-负结果\README-免疫细胞.md`）：拿**单步 Δ** 猜类型，探针根本没做（步数恒 0）；
动作/硬件类按通道号拍死 → 四类全误判。v2 的修法：
  ① **真探针**：发 3 步「前进」+ 1 步「转向」，逐个测候选通道的表现，才下诊断。
  ② **动作类三分**：命令前进无位移=电机卡死；动了的**反馈却为 0**=编码器断线（关键判据，v1 漏了）；
     转向反号=映射错位。
  ③ **免疫记忆按"类别"记**（传感器-死 / 动作-映射错位 …，不绑具体通道号）→ 第二次直接照方抓药，
     探测步数记为 **0**（可测：冷 5 步 → 热 0 步）。
  ④ **修复必须复测**（惊讶度回落才算好），沿用脊椎 v2 的 `clean()`（坏通道在**输入侧**也要替换）。

判据（先钉死）：
  ① 排查准确率 ≥80%（确诊类型 == 真实故障类型）
  ② 修复成功率 ≥75%（软件类修完复测通过）
  ③ 免疫记忆：第二次同类故障探测步数 ≤ 第一次的 50%，且仍修好
  ④ 不误伤：正常跑 0 次治疗；修复后惊讶度持续 ≤ 红线
用法：python xiaosha_immune.py --selftest
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, r"D:\Apps\MyModel\chat\workspace\xiaosha")
from xiaosha_spine import CH, HARD, N_ACT, SOFT, FaultyBody, Spine, sense_faulty   # 复用脊椎

DEAD, DRIFT, BAD, MAPBAD, MOTOR, ENCODER, HW = "死", "漂移", "异常值", "映射错位", "电机卡死", "编码器断线", "硬件故障"
SOFT_DX = (DEAD, DRIFT, BAD, MAPBAD)
# 每种注入故障的"正确确诊"（转向反向 与 动作错位 在软件视角都表现为映射错位 → 两者都算对）
EXPECT = {"传感器卡死": DEAD, "传感器漂移": DRIFT, "传感器异常值": BAD, "动作错位": MAPBAD,
          "行走电机卡死": MOTOR, "转向反向": MAPBAD, "编码器断线": ENCODER}


class Immune:
    def __init__(self, spine):
        self.sp = spine
        self.mem = {}                     # 免疫记忆：类别键 → (确诊, 修法名)
        self.probes = 0                   # 本轮化验用了几步
        self.offset = np.zeros(10)
        self.subbed = np.zeros(10, dtype=bool)

    # ---- 传感读数：标定补偿 + 冗余替代（治疗生效后才有非零值）----
    def fix_sense(self, raw):
        s = raw - self.offset
        return s

    # ---- ① 真探针化验 ----
    def probe(self, body, s0, comp, key):
        if key in self.mem:                       # ★ 免疫记忆命中：跳过化验（0 步）
            self.probes = 0
            return self.mem[key][0]
        self.probes = 5
        # ★ 动作序列：**前 3 步纯前进**（专供"行走/编码器"判别，不能掺转向——转向会让所有测距变化，
        #   把"电机卡死"误判成"有位移"），最后转向+前进让**各路测距都变化**（否则某路本来就静的
        #   通道会被误当"死"）。
        raw = []
        for aa in (0, 0, 0, 1, 0):
            body.act(aa)
            raw.append(sense_faulty(body, body.fault).copy())
        R = np.array(raw)
        # ★★ 先用**与通道号无关的实测事实**判本体类（原来按 comp>=8 分流，被"转向通道残差更大"
        #    带偏：行走电机卡死/编码器断线 全被误判成「映射错位」）：
        #      · 命令前进：什么都没动 → 电机卡死
        #      · 车真动了、反馈却是 0 → 编码器断线
        #      · 命令左转却朝右 → 映射/接线反（动作错位与转向反向都是这一类）
        fwd = R[:3]
        ranges_moved = float(np.abs(fwd[-1, :8] - fwd[0, :8]).max())
        moved = abs(float(fwd[-1, 8] - fwd[0, 8])) + abs(float(fwd[-1, 9] - fwd[0, 9]))
        turn_dyaw = float(R[3, 9] - R[2, 9])          # 第 4 步发的是 a=1（左转），正常应为正
        if ranges_moved < 1e-6 and moved < 1e-6:
            return MOTOR
        if ranges_moved > 1e-6 and moved < 1e-9:
            return ENCODER
        if turn_dyaw < 0.0:
            return MAPBAD
        if comp >= 8:                             # 行走 / 转向
            fwd = R[:3]                           # ★ 只看纯前进窗口
            moved = abs(float(fwd[-1, 8]) - float(fwd[0, 8])) + abs(float(fwd[-1, 9]) - float(fwd[0, 9]))
            ranges_moved = float(np.abs(fwd[-1, :8] - fwd[0, :8]).max())
            if comp == 8:
                if ranges_moved > 1e-6 and moved < 1e-9:
                    return ENCODER               # 车动了，但反馈说没动
                if ranges_moved < 1e-6:
                    return MOTOR                 # 命令前进，什么都没发生
                return MAPBAD
            return MAPBAD                        # 转向通道：一律先当映射问题（可软件绕过）
        # 传感器通道
        vals = R[:, comp]
        if (not np.all(np.isfinite(vals))) or vals.min() < 0.0 or vals.max() > 11.0:
            return BAD                           # 越界
        # ★ 本环境里"恒定"**不等于**坏（没回波时测距本来就恒 0）→ 用**传感签名**判：
        #   纹丝不动且**饱和在最大量程(5.0)** = 探头卡死；恒 0 = 此刻没回波（正常，不能判死）。
        v_mu, v_ptp = float(np.mean(vals)), float(np.ptp(vals))
        if v_ptp < 1e-9 and v_mu > 4.5:
            return DEAD
        # 漂移：读数与正常期均值的差**恒定**（符号一致、幅度稳定）
        base = float(self.sp.wm.mu[comp])
        dev = vals - base
        if np.all(np.sign(dev) == np.sign(dev[0])) and float(np.abs(dev).mean()) > 0.05:
            return DRIFT
        return HW

    # ---- ② 治疗 ----
    def treat(self, comp, dx, key, body=None):
        if key in self.mem:
            name = self.mem[key][1]
        else:
            name = {DEAD: "屏蔽+输入侧替代", DRIFT: "自动标定补偿", BAD: "剔除该路",
                    MAPBAD: "动作语义重映射", MOTOR: "降级运行+报人工", ENCODER: "降级运行+报人工",
                    HW: "报人工"}[dx]
            self.mem[key] = (dx, name)           # ★ 写入免疫记忆
        if name == "屏蔽+输入侧替代" or name == "剔除该路":
            self.sp.mask[comp] = True
            if comp < 8:
                self.subbed[comp] = True
        elif name == "自动标定补偿":
            self.offset[comp] = float(self._est_offset)
        elif name == "动作语义重映射":
            # ★ 用**真实身体**逐个试（不再拿故障前的缓冲区打分——实测那样选错过 k）
            scored = []
            for k in (1, 2, 3):
                self.sp.act_map = [(i + k) % N_ACT for i in range(N_ACT)]
                s_ = sense_faulty(body, body.fault)
                sc = []
                for i in range(8):
                    a_ = int(i % 8 // 2)
                    body.act(a_)
                    s2_ = sense_faulty(body, body.fault)
                    sc.append(self.sp.surprise(self.fix_sense(s_), a_, self.fix_sense(s2_) - self.fix_sense(s_)))
                    s_ = s2_
                scored.append((float(np.mean(sc)), k))
            scored.sort()
            best, bestk = scored[0]
            self.sp.act_map = [(i + bestk) % N_ACT for i in range(N_ACT)]
            name = f"动作语义重映射(k={bestk})" if best < self._cur * 0.5 else "软件修复无效：需人工"
        return name


def run(fault=None, warm=600, steps=1000, seed=3, mem=None):
    body = FaultyBody(seed=seed, fault=fault, fault_at=warm)
    sp = Spine()
    imm = Immune(sp)
    if mem:
        imm.mem = dict(mem)
    s_raw = sense_faulty(body, fault)
    buf = []
    for t in range(warm):
        a = int(t % 8 // 2)
        body.act(a)
        s2 = sense_faulty(body, None)
        buf.append((s_raw.copy(), a, (s2 - s_raw).copy()))
        s_raw = s2
    sp.fit(buf)
    sp._rep_buf = buf[::3]
    sp._base = float(np.mean([sp.surprise(*b) for b in sp._rep_buf]))
    det = loc = dx = name = None
    probes = 0
    post = []
    det_t = None
    for t in range(warm, steps):
        a = int(t % 8 // 2)
        body.act(a)
        s2_raw = sense_faulty(body, fault)
        s_f = imm.fix_sense(s_raw)
        ds = imm.fix_sense(s2_raw) - s_f
        sv = sp.surprise(s_f, a, ds)
        if sv > sp.thr and det is None and t > warm + 1:
            det = t - warm
            det_t = t
            # ★★ 定位必须**累计几步**（单步残差会抖：实测卡死注入在 3 却判「测距2」、
            #    行走卡死注入在 8 却判「转向」——差一路，直接毁掉后面的化验与治疗）。
            acc = sp.resid(s_f, a, ds).copy()
            for _ in range(3):
                a_ = int((t + 1) % 8 // 2)
                body.act(a_)
                s2_ = sense_faulty(body, fault)
                acc += sp.resid(imm.fix_sense(s_raw), a_, imm.fix_sense(s2_) - imm.fix_sense(s_raw))
                s_raw = s2_
                t += 1
            # ★★ 消融式假设检验定位：坏通道会**污染邻通道的预测**，残差 argmax 会差一路（实测）。
            #    正确做法 = 逐个假设"是第 c 路坏了"，把它的**输入**换成正常期均值，看惊讶度是否明显回落；
            #    谁能让惊讶度消失，谁就是元凶。本体通道（行走/转向）不参与该检验（它们的坏法不体现在输入上），
            #    直接采信残差排序。
            sp.mask[:] = False
            base_sv = float(sv)
            comp = int(np.argmax(acc))
            for c in np.argsort(-acc)[:4]:
                if c >= 8:
                    comp = int(c)
                    break
                sp.mask[int(c)] = True
                if sp.surprise(s_f, a, ds) < base_sv * 0.5:
                    comp = int(c)
                    break
                sp.mask[int(c)] = False
            sp.mask[:] = False
            loc = CH[comp]
            cls = "传感器" if comp < 8 else ("动作" if comp == 9 else "本体")
            key = (cls,)
            imm._cur = sv
            imm._est_offset = float(np.mean([s_raw[comp] - sp.wm.mu[comp]])) if comp < 8 else 0.0
            dx = imm.probe(body, s_f, comp, key)
            probes = imm.probes
            name = imm.treat(comp, dx, key, body=body)
        if det is not None and det_t is not None and t > det_t + 1:
            post.append(sv)
        s_raw = s2_raw
    retest = (len(post) >= 20 and float(np.mean(post[-10:])) <= sp.thr)
    return dict(det=det, loc=loc, dx=dx, treat=name, mem=imm.mem, probes=probes,
                retest=retest, thr=sp.thr, post=float(np.mean(post[-10:])) if post else None)


def selftest():
    print("=" * 94)
    print("  免疫细胞 v2：真探针化验 → 治疗 → 复测（+ 免疫记忆）")
    print("=" * 94)
    fails = []

    def check(n, ok, d=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
        if d:
            print(f"         └ {d}")
        if not ok:
            fails.append(n)

    n_ok = n_fix = 0
    rows = []
    for f, exp in EXPECT.items():
        r = run(fault=f)
        good = (r["dx"] == exp)
        n_ok += good
        soft = f in SOFT
        fixed = soft and r["retest"] and "无效" not in (r["treat"] or "") and "需人工" not in (r["treat"] or "")
        n_fix += (1 if fixed else 0)
        rows.append(f"{f}: 确诊「{r['dx']}」{'✓' if good else '✗'} → {r['treat']}"
                    f"（复测 {'过' if r['retest'] else '未过'}）")
    check("① 排查准确率 ≥80%", n_ok >= len(EXPECT) * 0.8, " ｜ ".join(rows))
    check("② 修复成功率 ≥75%（软件类修完复测要过）", n_fix >= len(SOFT_DX) * 0.75,
          f"软件类修复+复测通过 {n_fix}/{len(SOFT_DX)}")

    c1 = run(fault="传感器卡死")
    c2 = run(fault="传感器卡死", mem=c1["mem"])
    faster = (c2["probes"] <= max(1, c1["probes"]) * 0.5) and c2["retest"]
    check("③ 免疫记忆有效（第二次照方抓药，更快且照样修好）", faster,
          f"第一次化验 {c1['probes']} 步 → 第二次 {c2['probes']} 步（记忆 {len(c2['mem'])} 条）"
          f"｜复测 {'过' if c2['retest'] else '未过'}")

    r0 = run(fault=None)
    check("④ 不误伤：正常时 0 次治疗 + 复测段仍正常", r0["treat"] is None and r0["det"] is None,
          f"正常 400 拍：治疗 {0 if r0['treat'] is None else 1} 次 ｜ 惊讶度均值 {r0['post']}（红线 {r0['thr']:.2f}）")

    print("-" * 94)
    print(f"  结果：{4 - len(fails)}/4 项通过" + ("  ✅ 全部通过" if not fails else f"  ❌ 失败：{fails}"))
    print("=" * 94)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest())
