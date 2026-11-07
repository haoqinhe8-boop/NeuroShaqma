# -*- coding: utf-8 -*-
"""小沙的看门狗：健康心跳 → 卡死检测 → 崩溃自恢复 → 不误杀（清单⑥）。

和脊椎 / 免疫是同一套生命保障：
  脊椎：查出来哪里坏了 ｜ 免疫：排查+修复+复测 ｜ **看门狗：保证"脑子不会死"**。

机制：
  · 大脑每拍发**心跳**（时间戳 + 步数 + 数值健康标志）。
  · 看门狗按固定周期检查：心跳**停滞** / **消失** / **数值污染(NaN/Inf)** / worker **抛异常死亡**
    → 判定故障 → **重启并从最近存档恢复状态** → 继续跑。
  · **不误杀**：正常推进时一次都不重启。
  · **不无限重启装正常**：连续故障达到上限 → 停止重启，**如实升级报警**（不假装系统还好）。

范围说明（诚实标注）：本模块验证的是**看门狗逻辑**（检测/恢复/升级/不误杀）；
真实进程级 kill 与守护由计划任务 / 服务层负责（见技能「Windows 服务守护」）。
用法：python xiaosha_watchdog.py --selftest
"""
import argparse
import hashlib
import json
import sys
import time

import numpy as np

HB_TIMEOUT = 0.25          # 心跳停滞多久算卡死（秒）
CHECK_PERIOD = 0.05        # 检查周期
MAX_RESTARTS = 3           # 连续故障重启上限，超过就升级报警（不无限重启）


def state_hash(st):
    return hashlib.md5(json.dumps(st, sort_keys=True).encode()).hexdigest()[:10]


class Worker:
    """被守护的"大脑进程"（这里用轻量对象模拟；真实场景是独立进程）。"""

    def __init__(self, state=None, mode=None, fault_at=0.0, clock=None):
        self.state = dict(state) if state else {"w": 0.1, "step": 0}
        self.mode = mode                       # hang / raise / nan / die / None
        self.fault_at = fault_at
        self.clock = clock if clock is not None else time.monotonic
        self.t0 = self.clock()
        self.hb_time = self.t0
        self.hb_step = 0
        self.dead = False
        self.healthy = True

    def tick(self):
        if self.mode and (self.clock() - self.t0) >= self.fault_at:
            if self.mode == "hang":
                return                          # ★ 卡死：不推进、不更新心跳（真实卡死的样子）
            if self.mode == "raise":
                self.dead = True
                raise RuntimeError("worker crashed (injected)")
            if self.mode == "die":
                self.dead = True
                return
            if self.mode == "nan":
                self.state["w"] = float("nan")   # 数值污染
                self.healthy = False
                self.hb_time = self.clock()
                self.hb_step += 1
                return
        self.state["step"] += 1
        self.state["w"] = 0.1 + 0.001 * (self.state["step"] % 7)
        self.state["_"] = self.state["step"] * 1.0
        self.hb_time = self.clock()
        self.hb_step = self.state["step"]


class Watchdog:
    def __init__(self, snapshot):
        self.snapshot = dict(snapshot)                 # 最近一次存档
        self.restarts = 0
        self.caught = []
        self.escalated = False

    def check(self, w, now):
        """返回 (动作, 原因)。动作：ok / restart / escalate"""
        if w.dead:
            return self._on_fault("worker 死亡（抛异常/退出）")
        if not w.healthy or not np.all(np.isfinite([w.state.get("w", 0.0)])):
            return self._on_fault("数值污染（NaN/Inf）")
        stalled = (now - w.hb_time) > HB_TIMEOUT
        frozen = (w.hb_step <= self.last_step) if hasattr(self, "last_step") else False
        if stalled or frozen:
            return self._on_fault("心跳停滞（卡死）")
        self.last_step = w.hb_step
        return ("ok", "")

    def _on_fault(self, why):
        if self.restarts >= MAX_RESTARTS:
            self.escalated = True
            return ("escalate", f"连续故障 {self.restarts} 次达上限 → 如实升级报警（不假装正常）：{why}")
        self.restarts += 1
        self.caught.append(why)
        return ("restart", why)

    def respawn(self, state=None):
        """重启：从存档恢复状态（一字不差）。"""
        return Worker(state=dict(state if state is not None else self.snapshot))


def run_case(mode, fault_at=0.10, ticks=40, state0=None, timeout=HB_TIMEOUT):
    st0 = state0 or {"w": 0.1, "step": 0}
    snap = dict(st0)                                   # 存档（健康时拍的）
    w = Worker(state=st0, mode=mode, fault_at=fault_at)
    wd = Watchdog(snap)
    t0 = time.monotonic()
    steps_before = w.state["step"]
    detect_tick = None
    armed_i = None                                     # ★ 故障**生效**的那一拍（判据要量"故障后几个周期"）
    restored = None
    for i in range(ticks):
        time.sleep(0.02)                               # 真实时间推进
        try:
            w.tick()
        except RuntimeError:
            pass
        if armed_i is None and w.mode and (w.clock() - w.t0) >= w.fault_at:
            armed_i = i
        act, why = wd.check(w, time.monotonic())
        if act == "restart":
            if detect_tick is None:
                detect_tick = i
                t_det = time.monotonic() - t0
            w = wd.respawn(snap)                       # ★ 从存档恢复
            restored = state_hash(w.state)
        elif act == "escalate":
            if detect_tick is None:
                detect_tick = i
            break
    rel = (detect_tick - armed_i) if (detect_tick is not None and armed_i is not None) else None
    return dict(act=act, why=why, detect_tick=detect_tick, detect_rel=rel, restarts=wd.restarts,
                restored=restored, snap=state_hash(snap), escalated=wd.escalated,
                steps=w.state["step"], det_s=time.monotonic() - t0,
                steps_before=steps_before)


def selftest():
    print("=" * 94)
    print("  看门狗：心跳 → 卡死检测 → 崩溃自恢复 → 不误杀")
    print("=" * 94)
    fails = []

    def check(n, ok, d=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
        if d:
            print(f"         └ {d}")
        if not ok:
            fails.append(n)

    # ① 检出率：四种故障
    res = {m: run_case(m) for m in ("hang", "raise", "nan", "die")}
    # ★ 判据量的是"**故障生效之后**用了几个检查周期"（量绝对第几步是错的：故障本身就发生在第 5 拍）
    det = sum(1 for m, r in res.items() if r["detect_rel"] is not None and r["detect_rel"] <= 2)
    detail = " ｜ ".join(f"{m}: " + (f"故障后 {r['detect_rel']} 周期（{r['why'][:10]}）" if r["detect_rel"] is not None else "未检出")
                        for m, r in res.items())
    check("① 检出率 ≥90%（四种故障都要 ≤2 个检查周期内发现）", det >= 3.6, f"检出 {det}/4 ｜ {detail}")

    # ② 自恢复：恢复后状态与存档一字不差 + 能继续推进
    ok_restore = sum(1 for r in res.values() if r["restored"] and r["restored"] == r["snap"] and r["steps"] > 0)
    check("② 自恢复率 ≥90%（状态回到存档且继续推进）", ok_restore >= 3.6,
          f"恢复正确 {ok_restore}/4 ｜ 例：恢复后哈希 {res['hang']['restored']} == 存档 {res['hang']['snap']}"
          f"（延迟 {res['hang']['det_s']:.2f}s）")

    # ③ 不误杀
    r_ok = run_case(None)
    check("③ 不误杀：正常运行 0 次重启", r_ok["restarts"] == 0 and r_ok["detect_tick"] is None,
          f"正常跑 40 拍：重启 {r_ok['restarts']} 次 ｜ 步数 {r_ok['steps']}")

    # ④ 连续故障达上限要如实升级（不无限重启装正常）
    r_esc = run_case("hang", ticks=120)
    check("④ 达上限如实升级报警（不无限重启假装正常）", r_esc["escalated"] and r_esc["restarts"] == MAX_RESTARTS,
          f"重启 {r_esc['restarts']} 次后升级：{'是' if r_esc['escalated'] else '否'} ｜ 报警：{r_esc['why'][:34]}")

    print("-" * 94)
    print(f"  结果：{4 - len(fails)}/4 项通过" + ("  ✅ 全部通过" if not fails else f"  ❌ 失败：{fails}"))
    print("=" * 94)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest())
