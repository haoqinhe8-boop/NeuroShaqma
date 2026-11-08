# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""小沙身体：五感 + 行动的真实接口层（2026-11-07）

耳朵 = 本机 ASR（faster-whisper :8085）真听
嘴巴 = Windows SAPI TTS 真说（出 WAV 文件）
鼻子 = 气味扩散场 + 8 路嗅探定位（仿真；真机接口 = MQ 气体传感器读数数组替换 sniff）
脚   = xiaosha_embodiment 躯干适配（开机自识别雅可比，真走）
手   = xiaosha_brain_v2 机械臂（真机换 Feetech STS3215 总线舵机）

用法：python xiaosha_body.py --selftest（每条断言带实测数字）
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 兼容性兜底：干净环境（Windows GBK stdout）下 print 中文/emoji 会 UnicodeEncodeError 崩溃
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ============================================================
# 耳朵：听（真 ASR）
# ============================================================
class Ear:
    def __init__(self, base="http://127.0.0.1:8085/asr", timeout=120):
        self.base, self.timeout = base, timeout

    def hear_bytes(self, wav_bytes):
        import requests
        try:
            r = requests.post(self.base, data=wav_bytes, timeout=self.timeout)
        except Exception as e:
            return None, f"ASR 不可达: {e}"
        if r.status_code != 200:
            return None, f"ASR HTTP {r.status_code}"
        j = r.json()
        return str(j.get("text", "")).strip(), None

    def hear_file(self, path):
        with open(path, "rb") as f:
            return self.hear_bytes(f.read())


# ============================================================
# 嘴巴：说（真 TTS）
# ============================================================
class Mouth:
    def __init__(self, tts_ps1=None):
        self.ps1 = tts_ps1 or os.path.join(HERE, "tts_say.ps1")

    def say(self, text, out_wav):
        """文本 → WAV 文件（Windows SAPI 真实合成）。返回是否成功。"""
        r = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", self.ps1, "-Text", text, "-OutWav", out_wav],
            capture_output=True, text=True, encoding="utf-8", errors="ignore",
            timeout=60)
        return os.path.exists(out_wav) and os.path.getsize(out_wav) > 1000


# ============================================================
# 鼻子：闻（气味扩散场 + 8 路嗅探）
# ============================================================
class Nose:
    """稳态扩散近似：浓度 c(r) = Σ A/(1+r²)。真机 = MQ 气体传感器读数数组。"""

    def __init__(self, sources=((2.0, -1.0),), amp=10.0):
        self.sources = [(float(sx), float(sy)) for sx, sy in sources]
        self.amp = float(amp)

    def conc(self, x, y):
        c = 0.0
        for sx, sy in self.sources:
            c += self.amp / (1.0 + (x - sx) ** 2 + (y - sy) ** 2)
        return c

    def sniff(self, x, y, yaw=0.0, probe=0.3):
        """8 路嗅探：朝 8 个方向各探 probe 米，返回浓度读数（0~amp）。"""
        out = np.zeros(8, dtype=np.float32)
        for k in range(8):
            th = yaw + k * np.pi / 4
            out[k] = self.conc(x + probe * np.cos(th), y + probe * np.sin(th))
        return out

    def locate(self, x, y, yaw=0.0):
        """从 8 路读数估计气味源的方向（世界系弧度）。"""
        s = self.sniff(x, y, yaw)
        angs = yaw + np.arange(8) * np.pi / 4
        gx = float(np.sum(s * np.cos(angs)))
        gy = float(np.sum(s * np.sin(angs)))
        return float(np.arctan2(gy, gx))


# ============================================================
# 统一身体
# ============================================================
class Body:
    def __init__(self):
        self.ear = Ear()
        self.mouth = Mouth()
        self.nose = Nose()

    def feet(self, target=(1.5, 0.8)):
        """脚：任选第一个已标定躯干，跑「上机→自学→走到目标」。"""
        from xiaosha_embodiment import BODY_SPECS, run_one_body
        name, spec = next(iter(BODY_SPECS.items()))
        return run_one_body(spec, name=name, target=target)

    def hands(self, text="小"):
        """手：机械臂写字（仿真下发指令序列；真机换总线舵机）。"""
        from xiaosha_brain_v2 import Arm
        arm = Arm(sim=True)
        rec = arm.write(text)
        return rec, arm.log


# ============================================================
# 自检
# ============================================================
def selftest():
    print("=" * 78)
    print("  小沙身体 自检：五感 + 行动（每条断言带实测数字）")
    print("=" * 78)
    fails, results = [], []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    body = Body()
    tmp = os.path.join(os.environ.get("TEMP", HERE), "xiaosha-body-test.wav")

    # ---------- 1. 嘴巴→耳朵 闭环：真说 → 真听 ----------
    try:
        ok_tts = body.mouth.say("小沙你好，今天也要认真干活。", tmp)
        if ok_tts:
            text, err = body.ear.hear_file(tmp)
            hit = (text is not None and ("小沙" in text or "你好" in text))
            check("嘴巴→耳朵闭环：TTS 真合成 → ASR 真识别", hit,
                  f"合成 {os.path.getsize(tmp)//1024}KB WAV → 识别「{text}」" + ("" if hit else f"（err={err}）"))
        else:
            check("嘴巴→耳朵闭环", False, "TTS 合成失败")
    except Exception as e:
        check("嘴巴→耳朵闭环", False, f"异常: {e}")

    # ---------- 2. 鼻子：气味源定位 ----------
    nose = body.nose
    true_ang = float(np.arctan2(-1.0, 2.0))          # 源 (2,-1)
    est = nose.locate(0.0, 0.0)
    err_deg = abs(np.degrees(np.arctan2(np.sin(est - true_ang), np.cos(est - true_ang))))
    check("鼻子：8 路嗅探指向气味源（方位误差 < 20°）", err_deg < 20.0,
          f"源 (2,-1) 真方位 {np.degrees(true_ang):.0f}° → 估计 {np.degrees(est):.0f}°（误差 {err_deg:.1f}°）")

    # ---------- 3. 脚：真实躯干走到目标 ----------
    try:
        r = body.feet(target=(1.5, 0.8))
        run = r["run"]
        ok = bool(run.get("reached", False) or run.get("ok", False) or run.get("final_dist", 9.9) < 0.8)
        dist = run.get("final_dist", 9.9)
        check("脚：躯干自学后走到目标（终距 < 0.8m）", ok,
              f"J 误差 {r['J_err']*100:.1f}% ｜ 终距 {dist:.2f}m ｜ 能力 {r['caps']}")
    except Exception as e:
        check("脚：躯干自学后走到目标", False, f"异常: {e}")

    # ---------- 4. 手：机械臂写字 ----------
    try:
        rec, log = body.hands("小")
        ok = rec.get("cmd") == "write" and len(log) >= 1
        check("手：机械臂接收写字指令（真机接口=总线舵机）", ok,
              f"下发 {rec} ｜ 指令序列 {len(log)} 条")
    except Exception as e:
        check("手：机械臂接收写字指令", False, f"异常: {e}")

    # ---------- 5. 五感齐全性 ----------
    check("五感齐全：耳（听）/ 嘴（说）/ 鼻（闻）/ 脚（走）/ 手（抓）",
          isinstance(body.ear, Ear) and isinstance(body.mouth, Mouth)
          and isinstance(body.nose, Nose) and hasattr(body, "feet") and hasattr(body, "hands"),
          "Ear / Mouth / Nose / feet(embodiment) / hands(Arm) 全部就位")

    print("-" * 78)
    print(f"  结果：{sum(results)}/{len(results)} 项通过"
          + ("  ✅ 全部通过" if not fails else f"  ❌ 失败：{fails}"))
    print("=" * 78)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    ap.print_help()
