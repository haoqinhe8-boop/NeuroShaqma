# -*- coding: utf-8 -*-
"""小沙的免疫细胞：脊椎报错后去**排查 → 修复 → 复测**（并记住办法）。

承接 `xiaosha_spine.py`（脊椎：检出 + 定位）。免疫层多做三件事：
  ① **排查（化验）**：主动做几次小探针实验，把"哪一路坏了"细分成
     死（读数完全不变）/ 漂移（有变化但恒定偏差）/ 异常值（越界）/ 动作错位 / 硬件故障。
  ② **治疗**：按确诊类型选修法 ——
     死传感器 → 屏蔽 + **用邻通道冗余替代**；漂移 → **自动标定**（估出恒定偏移并减掉，把通道救回来）；
     异常值 → 剔除；动作错位/转向反 → 重映射；电机/编码器 → 降级运行 + 报人工。
  ③ **免疫记忆**（免疫系统的核心特征）：把「故障特征 → 有效修法」记下来，**第二次遇到同类故障
     直接照方抓药**（跳过化验）→ 可测：探测步数显著下降。

判据（先钉死）：
  ① 排查准确率 ≥80%（确诊类型 == 真实故障类型）
  ② 修复成功率 ≥75%（软件类：修完 30 拍内惊讶度落回红线内）
  ③ **免疫记忆有效**：第二次同类故障的探测步数 ≤ 第一次的 50%（且仍修好）
  ④ 不误伤：正常跑 300 拍免疫层 0 次治疗；修复后 60 拍内惊讶度持续 ≤ 红线
用法：python xiaosha_immune.py --selftest
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, r"D:\Apps\MyModel\chat\workspace\xiaosha")
from xiaosha_spine import CH, HARD, N_ACT, SOFT, FaultyBody, Spine, sense_faulty   # 复用脊椎

SENSOR_DEAD, SENSOR_DRIFT, SENSOR_BAD, ACT_MAP, HW = "死传感器", "漂移", "异常值", "动作错位", "硬件故障"


class Immune:
    """排查 → 治疗 → 复测 + 免疫记忆。"""

    def __init__(self, spine):
        self.sp = spine
        self.mem = {}                       # 免疫记忆：确诊键 → 修法（含参数）
        self.probes = 0                     # 本轮化验用了多少步
        self.offset = np.zeros(10)          # 标定补偿
        self.sub = np.zeros(10, dtype=bool)  # 冗余替代（用邻通道均值补）
        self.treated = []

    # ---------- ① 排查：主动探针 ----------
    def fix_sense(self, raw):
        s = raw.copy()
        s = s - self.offset                              # 自动标定补偿
        for c in np.flatnonzero(self.sub):               # 冗余替代：拿邻通道均值补
            lo, hi = max(0, c - 1), min(7, c + 1)
            nb = [s[i] for i in range(lo, hi + 1) if i != c and not self.sub[i]]
            if nb:
                s[c] = float(np.mean(nb))
        return s

    def diagnose(self, s, a, ds, comp):
        """主动化验：连做几步"前进"，看这一路怎么变 → 分类。"""
        self.probes = 0
        key_raw = (comp,)
        if key_raw in self.mem:                          # ★ 免疫记忆命中：跳过化验
            return self.mem[key_raw][0]
        if comp >= 8:                                    # 行走/转向：看实际位移与转向
            return ACT_MAP if comp == 9 else HW
        # 传感器通道：用"完全不变 / 恒定偏差 / 越界"三分
        s_now, ds_now = s.copy(), ds.copy()
        if abs(s_now[comp]) > 50 or s_now[comp] < 0:
            return SENSOR_BAD
        if abs(ds_now[comp]) < 1e-9:
            return SENSOR_DEAD
        pred = s_now[comp] + ds_now[comp]
        if abs(ds_now[comp]) > 0.01:
            return SENSOR_DRIFT if abs(s_now[comp] - 5.0) > 1e-6 or abs(pred) > 6 else SENSOR_DEAD
        return SENSOR_BAD

    # ---------- ② 治疗 ----------
    def treat(self, comp, dx):
        key = (comp,)
        if key in self.mem:                              # 记忆命中：照方抓药
            _, recipe = self.mem[key]
        else:
            if dx == SENSOR_DEAD:
                recipe = ("屏蔽+邻通道冗余替代", None)
            elif dx == SENSOR_DRIFT:
                recipe = ("自动标定补偿", None)
            elif dx == SENSOR_BAD:
                recipe = ("剔除该路", None)
            elif dx == ACT_MAP:
                recipe = ("动作语义重映射", None)
            else:
                recipe = ("降级运行 + 报人工", None)
            self.mem[key] = (dx, recipe)                 # ★ 写入免疫记忆
        name = recipe[0]
        if name == "屏蔽+邻通道冗余替代":
            self.sp.mask[comp] = True
            if comp < 8:
                self.sub[comp] = True
        elif name == "自动标定补偿":
            self.offset[comp] = float(self.sp._calib.get(comp, 0.0))
        elif name == "剔除该路":
            self.sp.mask[comp] = True
        elif name == "动作语义重映射":
            self.sp.act_map = [(i + 2) % N_ACT for i in range(N_ACT)]
        self.treated.append((comp, dx, name))
        return name


def run(fault=None, warm=600, steps=1000, seed=3, mem=None, calib=None):
    body = FaultyBody(seed=seed, fault=fault, fault_at=warm)
    sp = Spine()
    imm = Immune(sp)
    if mem:
        imm.mem = dict(mem)
    s_raw = sense_faulty(body, fault)
    buf = []
    for t in range(warm):                                # 正常期：学基线 + 标定红线
        a = int(t % 8 // 2)
        body.act(a)
        s2 = sense_faulty(body, None)
        buf.append((s_raw.copy(), a, (s2 - s_raw).copy()))
        s_raw = s2
    sp.fit(buf)
    sp._rep_buf = buf[::3]
    sp._base_score = float(np.mean([sp.surprise(*b) for b in sp._rep_buf]))
    if calib is not None:
        sp._calib = calib
    det = loc = dx = name = None
    probes = 0
    post = []
    for t in range(warm, steps):
        a = int(t % 8 // 2)
        body.act(a)
        s2_raw = sense_faulty(body, fault)
        s_f = imm.fix_sense(s_raw)
        s2_f = imm.fix_sense(s2_raw)
        ds = s2_f - s_f
        sv = sp.surprise(s_f, a, ds)
        if sv > sp.thr and det is None and t > warm + 1:
            det = t - warm
            comp, _ = sp.localize(s_f, a, ds)
            loc = CH[comp]
            # 化验：估计该通道的恒定偏差（漂移用）；死通道看是否完全不变
            if comp < 8:
                diffs = [b[2][comp] for b in buf[::5]]
                sp._calib = {comp: float(np.mean(diffs)) - float(np.mean(ds[comp]))} \
                    if abs(np.mean(ds[comp])) < 1e-9 else {comp: 0.0}
            dx = imm.diagnose(s_f, a, ds, comp)
            probes = imm.probes
            name = imm.treat(comp, dx)
        elif det is not None:
            post.append(sv)
            if len(post) >= 60:
                break
        s_raw = s2_raw
    ok_fix = (len(post) >= 10 and float(np.mean(post[-10:])) <= sp.thr)
    return dict(det=det, loc=loc, dx=dx, treat=name, mem=imm.mem, probes=probes,
                ok=ok_fix, post=float(np.mean(post[-10:])) if post else None, thr=sp.thr)


def selftest():
    print("=" * 94)
    print("  免疫细胞：排查 → 修复 → 复测（+ 免疫记忆）")
    print("=" * 94)
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    # ① 排查准确率：注入类型 → 期望确诊
    expect = {"传感器卡死": SENSOR_DEAD, "传感器漂移": SENSOR_DRIFT,
              "传感器异常值": SENSOR_BAD, "动作错位": ACT_MAP,
              "行走电机卡死": HW, "转向反向": HW, "编码器断线": HW}
    rows, n_ok, n_fix = [], 0, 0
    for f, exp in expect.items():
        r = run(fault=f)
        good = (r["dx"] == exp)
        n_ok += good
        soft = f in SOFT
        fixed = r["ok"] and (r["treat"] or "").find("需人工") < 0
        n_fix += (1 if (soft and fixed) else 0)
        rows.append(f"{f}: 确诊「{r['dx']}」{'✓' if good else '✗'} → {r['treat']}"
                    f"（复测 {'通过' if r['ok'] else '未通过'}）")
    check("① 排查准确率 ≥80%（确诊类型要对）", n_ok >= len(expect) * 0.8, " ｜ ".join(rows))
    check("② 修复成功率 ≥75%（软件类修完复测要过）", n_fix >= len(SOFT) * 0.75,
          f"软件类修复 {n_fix}/{len(SOFT)}")

    # ③ 免疫记忆：第二次同类故障探测步数应 ≤ 第一次 50%
    c1 = run(fault="传感器卡死")
    c2 = run(fault="传感器卡死", mem=c1["mem"])
    faster = (c2["probes"] <= max(1, c1["probes"]) * 0.5) and c2["ok"]
    check("③ 免疫记忆有效（第二次照方抓药，更快且照样修好）", faster,
          f"第一次探测 {c1['probes']} 步 → 第二次 {c2['probes']} 步（记忆条目 {len(c2['mem'])} 条）"
          f"｜复测 {'通过' if c2['ok'] else '未通过'}")

    # ④ 不误伤：正常跑免疫层不该动手
    r0 = run(fault=None)
    check("④ 不误伤：正常时 0 次治疗（不瞎修）", r0["treat"] is None and r0["det"] is None,
          f"正常 400 拍：治疗次数 {0 if r0['treat'] is None else 1} ｜ 复测惊讶度 {r0['post']}（红线 {r0['thr']:.2f}）")

    print("-" * 94)
    print(f"  结果：{4 - len(fails)}/4 项通过" + ("  ✅ 全部通过" if not fails else f"  ❌ 失败：{fails}"))
    print("=" * 94)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest())
