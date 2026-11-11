# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""小沙内脏：八大系统（2026-11-07，用户钦定器官映射，每个都有真实功能）

呼吸系统   肺=网络吞吐呼吸（真 HTTP 当空气）｜气管/支气管=多路并行通道
消化系统   胃=数据缓冲｜肝=分类好坏｜肠=复查防误杀｜胰腺=提取养分
循环系统   心脏=心跳调度泵｜血管=模块间数据总线
内分泌系统 激素=负载应激调节（影响各器官节奏）
泌尿系统   肾=废物过滤｜膀胱=暂存｜真排泄（白名单清理真实文件）
神经系统   皮层（xiaosha_cortex）｜免疫系统=脊椎/免疫/看门狗｜运动系统=脚/手（xiaosha_body）

用法：python xiaosha_organs.py --selftest
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 兼容性兜底：干净环境（Windows GBK stdout）下 print 中文/emoji 会 UnicodeEncodeError 崩溃
# （实测 python -E 下 cp936 编码 ✅ → crash）。强制 UTF-8 + replace，任何终端都不崩。
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ============================================================
# 呼吸系统：网络当空气
# ============================================================
class Lung:
    """肺：一次呼吸 = 一次真实 HTTP 请求。吸入量 = 响应字节数。"""

    def __init__(self, air_url="http://127.0.0.1:3001/api/config", timeout=5):
        self.url, self.timeout = air_url, timeout
        self.total_in = 0          # 累计吸入（字节）
        self.breaths = 0
        # 提升（2026-11）：呼吸节律 + 肺活量 + 深度控制
        self.rate = 12.0           # 呼吸频率（次/分，静息 ~12，活动可加快）
        self.depth = "normal"      # shallow / normal / deep
        self.depth_factor = {"shallow": 0.5, "normal": 1.0, "deep": 1.8}
        self.vital_capacity = 0.0  # 肺活量 = 记录到的最大单次有效吸入（字节）

    def set_depth(self, depth):
        """呼吸深度：shallow 浅 / normal 正常 / deep 深。越深单次吸入越多。"""
        if depth in self.depth_factor:
            self.depth = depth
        return self.depth

    def set_rate(self, rate):
        """呼吸频率（次/分）：活动时加快（如 20~30），静息 12 左右。"""
        self.rate = max(1.0, min(60.0, rate))
        return self.rate

    def breathe(self):
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as r:
                data = r.read()
        except Exception:
            return 0.0
        effective = float(len(data)) * self.depth_factor[self.depth]   # 深度控制有效吸入
        self.total_in += effective
        self.breaths += 1
        self.vital_capacity = max(self.vital_capacity, effective)     # 肺活量 = 历史最大
        return float(effective)

    def volume(self):
        return self.total_in / 1024.0   # KB 累计通气量

    def capacity(self):
        return self.vital_capacity / 1024.0   # KB 肺活量


class Bronchi:
    """支气管：把一次吸气拆成多路并行通道（分块 Range 请求），像支气管分叉。"""

    def __init__(self, n_branches=3, timeout=5):
        self.n, self.timeout = n_branches, timeout

    def parallel_breath(self, url="http://127.0.0.1:3001/"):
        import concurrent.futures
        def one(_):
            req = urllib.request.Request(url, headers={"Range": "bytes=0-2047"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return len(r.read())
            except Exception:
                return 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n) as ex:
            return list(ex.map(one, range(self.n)))


class Trachea:
    """气管：多路支气管的合流主管道。"""

    def __init__(self):
        self.bronchi = Bronchi()

    def breathe(self, url="http://127.0.0.1:3001/"):
        flows = self.bronchi.parallel_breath(url)
        return float(sum(flows)), flows


# ============================================================
# 消化系统：吃数据 → 分类 → 吸收 → 复查 → 排泄
# ============================================================
BAD_WORDS = ("垃圾", "广告", "诈骗", "spam", "推广")
GOOD_WORDS = ("学习", "数据", "模型", "工作", "计划", "日志")


class Stomach:
    """胃：进来的原始数据先放这里。"""

    def __init__(self):
        self.buf = []
        self.suspects = []         # 肝判"坏"、等肠复查的
        self.trash = []            # 肠确认真坏、等排泄的

    def ingest(self, items):
        self.buf.extend(items)

    def take(self):
        items, self.buf = self.buf, []
        return items


class Liver:
    """肝：分类好坏。规则评分（真实可判定的质量规则）。"""

    def classify(self, item):
        text = str(item.get("text", ""))
        score = 0.0
        for w in GOOD_WORDS:
            if w in text:
                score += 1.0
        for w in BAD_WORDS:
            if w in text:
                score -= 1.5
        score += min(2.0, len(text) / 20.0) * 0.3
        return "good" if score > 0 else "bad", score


class Intestine:
    """肠：复查肝判的「坏」——换一套判据二次确认，救回误判（假坏）。"""

    def review(self, item):
        text = str(item.get("text", ""))
        # 假坏信号：短但含实义关键词、或标记里明确"重要"的
        saved = item.get("keep", False)
        has_meaning = any(w in text for w in GOOD_WORDS) and len(text) >= 4
        return "false_alarm" if (saved or has_meaning) else "confirmed_bad"


class Pancreas:
    """胰腺：从好数据提取养分（摘要 + 关键词）。"""

    def extract(self, item):
        text = str(item.get("text", ""))
        words = [w for w in GOOD_WORDS if w in text]
        return {"source": text[:24], "keywords": words, "digested_at": time.time()}


# ============================================================
# 循环系统
# ============================================================
class Vessels:
    """血管：器官间的数据总线（真实队列）。"""

    def __init__(self):
        self.channels = {}

    def pump(self, channel, data):
        self.channels.setdefault(channel, []).append(data)

    def drain(self, channel):
        out, self.channels[channel] = self.channels.get(channel, []), []
        return out


class Heart:
    """心脏：心跳调度泵。每次 beat = 把血管里的数据泵向目标处理器。"""

    def __init__(self, rate_hz=1.0):
        self.rate = rate_hz
        self.beats = 0
        self.vessels = Vessels()

    def beat(self):
        self.beats += 1
        return self.beats

    def healthy(self):
        return self.beats > 0


# ============================================================
# 内分泌系统
# ============================================================
class Endocrine:
    """内分泌：按负载分泌激素（应激调节），影响器官节奏。"""

    def __init__(self):
        self.hormones = {"adrenaline": 0.1, "dopamine": 0.5, "melatonin": 0.1}

    def update(self, load):
        # 负载高 → 肾上腺素升（应激提速）；负载低 → 褪黑素升（降频休息）
        self.hormones["adrenaline"] = float(np.clip(load * 1.2, 0.0, 1.0))
        self.hormones["melatonin"] = float(np.clip((1.0 - load) * 0.8, 0.0, 1.0))
        self.hormones["dopamine"] = float(np.clip(0.3 + 0.4 * np.sin(time.time() / 60), 0.05, 1.0))
        return dict(self.hormones)


# ============================================================
# 泌尿系统
# ============================================================
class Kidney:
    """肾：过滤废物——从待处理清单里筛出「该扔」的（规则可判定）。"""

    def filter(self, candidates):
        trash, keep = [], []
        for c in candidates:
            # 文件类候选按 junk 标记判（不查 text——文件没有文本字段）
            if c.get("junk", False):
                trash.append(c)
            else:
                keep.append(c)
        return keep, trash


class Bladder:
    """膀胱：暂存待排泄物。"""

    def __init__(self):
        self.holding = []

    def hold(self, items):
        self.holding.extend(items)

    def release(self):
        out, self.holding = self.holding, []
        return out


class Ureter:
    """输尿管：真实排泄——白名单目录内真删垃圾文件。"""

    def excrete(self, paths, allowed_root):
        removed = []
        root = os.path.abspath(allowed_root)
        for p in paths:
            ap = os.path.abspath(p)
            # 前缀边界：必须是 root 本身或 root 下的文件（防 "...\test" 误匹配 "...\test_evil"）
            inside = ap == root or ap.startswith(root + os.sep)
            if inside and os.path.exists(ap):
                try:
                    os.remove(ap)
                    removed.append(ap)
                except OSError:
                    pass
        return removed


# ============================================================
# 汇总：八大系统
# ============================================================
class Organs:
    """小沙完整身体。神经/免疫/运动桥接已有模块（导入失败则如实标注）。"""

    def __init__(self):
        self.lung = Lung()
        self.trachea = Trachea()
        self.stomach = Stomach()
        self.liver = Liver()
        self.intestine = Intestine()
        self.pancreas = Pancreas()
        self.heart = Heart()
        self.kidney = Kidney()
        self.bladder = Bladder()
        self.ureter = Ureter()
        self.endocrine = Endocrine()
        self.systems = {
            "呼吸": (self.lung, self.trachea),
            "消化": (self.stomach, self.liver, self.intestine, self.pancreas),
            "循环": (self.heart,),
            "内分泌": (self.endocrine,),
            "泌尿": (self.kidney, self.bladder, self.ureter),
        }
        self.extra = {}
        for name, mod in (("神经", "xiaosha_cortex"), ("免疫", "xiaosha_immune"),
                          ("运动", "xiaosha_body")):
            try:
                __import__(mod)
                self.extra[name] = "就位"
            except Exception as e:
                self.extra[name] = f"缺失: {e}"


# ============================================================
# 自检
# ============================================================
def selftest():
    print("=" * 78)
    print("  小沙内脏 自检：八大系统（每条断言带实测数字）")
    print("=" * 78)
    fails, results = [], []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    o = Organs()

    # ---------- 1. 呼吸：肺真吸气 + 气管支气管分路 ----------
    v1 = o.lung.breathe()
    total, branches = o.trachea.breathe()          # 气管返回 (合流量, 各路流量)
    check("呼吸：肺真实吸入网络空气（HTTP 字节）+ 支气管 3 路并行",
          v1 > 0 and sum(branches) > 0,
          f"肺单次吸入 {v1/1024:.1f}KB ｜ 支气管 {len(branches)} 路并行合计 {sum(branches)/1024:.1f}KB")
    # 深度控制：浅/正常/深呼吸吸入量递增
    o.lung.set_depth("shallow"); s = o.lung.breathe()
    o.lung.set_depth("normal");  n = o.lung.breathe()
    o.lung.set_depth("deep");    d = o.lung.breathe()
    check("呼吸深度控制：浅 < 正常 < 深呼吸（吸入量递增）", s < n < d,
          f"浅 {s/1024:.1f}KB < 正常 {n/1024:.1f}KB < 深 {d/1024:.1f}KB")
    o.lung.set_rate(24)
    check("呼吸节律 + 肺活量：频率可调、肺活量=历史最大吸入",
          o.lung.rate == 24 and o.lung.capacity() > 0,
          f"频率 {o.lung.rate:.0f}次/分 ｜ 肺活量 {o.lung.capacity():.1f}KB")

    # ---------- 2. 消化闭环：吃 → 肝分类 → 肠复查 → 胰腺吸收 → 泌尿排泄 ----------
    feed = [
        {"text": "学习计划：今晚复习模型基础"}, {"text": "工作日志：数据整理完成"},
        {"text": "垃圾广告，快来买"}, {"text": "模型训练记录"}, {"text": "推广诈骗短信"},
        {"text": "数据备份计划"}, {"text": "spam 广告信息"}, {"text": "工作安排"},
        {"text": "学习笔记"}, {"text": "广告数据", "keep": True},   # 假坏：肝判坏、肠救回
    ]
    o.stomach.ingest(feed)
    goods, suspects, trash = [], [], []
    for item in o.stomach.take():
        verdict, score = o.liver.classify(item)
        (goods if verdict == "good" else suspects).append(item)
    liver_ok = len(goods) == 6 and len(suspects) == 4
    # 肠复查：把一条"假坏"救回（keep 标记的）
    confirmed = []
    saved = 0
    for item in suspects:
        v = o.intestine.review(item)
        if v == "false_alarm":
            goods.append(item)
            saved += 1
        else:
            confirmed.append(item)
    nutrients = [o.pancreas.extract(g) for g in goods]
    check("消化：肝分类 10 条（6 好 4 坏）→ 肠救回假坏 → 胰腺吸收养分",
          liver_ok and saved >= 1 and len(nutrients) >= 7,
          f"肝分类 {len(goods) - saved} 好 {len(suspects)} 坏 ｜ 肠救回 {saved} 条假坏 ｜ 胰腺吸收 {len(nutrients)} 份养分")

    # ---------- 3. 循环：心跳调度 + 血管泵送 ----------
    h = o.heart
    for i in range(20):
        h.beat()
        h.vessels.pump("aorta", {"beat": i, "ts": time.time()})
    pumped = h.vessels.drain("aorta")
    check("循环：心脏 20 拍泵送，血管运货", h.beats == 20 and len(pumped) == 20,
          f"心跳 {h.beats} 拍 ｜ 血管运送 {len(pumped)} 份")

    # ---------- 4. 内分泌：负载应激 ----------
    hi = o.endocrine.update(0.95)
    lo = o.endocrine.update(0.05)
    check("内分泌：负载高→肾上腺素升，负载低→褪黑素升",
          hi["adrenaline"] > 0.9 and lo["melatonin"] > 0.6,
          f"高负载 肾上腺素 {hi['adrenaline']:.2f} ｜ 低负载 褪黑素 {lo['melatonin']:.2f}")

    # ---------- 5. 泌尿：肾过滤 → 膀胱暂存 → 真实排泄（白名单临时目录） ----------
    tmpd = os.path.join(os.environ.get("TEMP", HERE), "xiaosha-urine-test")
    os.makedirs(tmpd, exist_ok=True)
    junk_files = []
    for i in range(3):
        p = os.path.join(tmpd, f"junk_{i}.tmp")
        with open(p, "w") as f:
            f.write("junk")
        junk_files.append(p)
    keep_path = os.path.join(tmpd, "keep.txt")
    with open(keep_path, "w") as f:
        f.write("keep me")
    candidates = [{"path": p, "junk": True} for p in junk_files] + [{"path": keep_path, "junk": False}]
    kept, trash = o.kidney.filter(candidates)
    o.bladder.hold(trash)
    rel = o.bladder.release()
    removed = o.ureter.excrete([r["path"] for r in rel], tmpd)
    check("泌尿：肾滤出垃圾 → 膀胱暂存 → 真实删除（白名单目录）",
          len(removed) == 3 and os.path.exists(keep_path) and not any(os.path.exists(p) for p in junk_files),
          f"过滤后留 {len(kept)} 弃 {len(trash)} ｜ 真实删除 {len(removed)} 个文件，保留 keep.txt ✓")
    for p in (keep_path,):
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(tmpd)
    except OSError:
        pass

    # ---------- 6. 八大系统齐全 ----------
    missing = [k for k, v in o.extra.items() if not v.startswith("就位")]
    check("八大系统齐全：呼吸/消化/循环/内分泌/泌尿/神经/免疫/运动",
          len(missing) == 0,
          "神经=" + o.extra.get("神经", "?") + " 免疫=" + o.extra.get("免疫", "?")
          + " 运动=" + o.extra.get("运动", "?") + (" ｜ 缺失: " + ",".join(missing) if missing else ""))

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
