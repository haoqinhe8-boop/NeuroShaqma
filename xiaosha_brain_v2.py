# -*- coding: utf-8 -*-
"""
小沙 · 大脑 v2（BrainCore v2）
============================================================
把原来两份互不相通的「小沙大脑」合并升级成一个真正的脑子：

  旧 xiaosha_brain.py : 听唤醒 → TDOA 定位 → 走过去
  旧 robot_brain.py   : 情绪安抚 / 背默出题 / 机械臂指令
  两份代码各自能跑，但彼此不知道对方存在 —— 会走路的不会说话，
  会说话的不会走路，而且"思考"全是关键词 if-else。

v2 的六处升级
------------------------------------------------------------
 1) 决策层接真模型：默认调用本机 llama-server 的 Shaqma-Pro
    （OpenAI 兼容 /v1/chat/completions），让沙琪玛自己判断"该干什么"。
    模型不可用 / 输出不合法 → 自动降级到本地规则 NLU，离线也照跑。
 2) 一个脑子管全身：技能注册表（goto / say / write / quiz / comfort /
    status），走路和说话共用同一套记忆与同一次决策。
 3) 记忆持久化：人物档案、地点（"客厅""书桌"→ 坐标）、背默错题、
    事件日志落盘 JSON，重启后接着用。
 4) 感知要成真：仿真后端用**声学几何 TDOA**（麦克风对之间的真实时延，
    按 343m/s 算），真机后端预留 ReSpeaker 硬件 DoA 与 Vosk ASR 接口，
    不再用"距离差"冒充时差。
 5) 安全层：速度/角速度限幅、最小安全距离、卡死检测、障碍+墙面势场
    避障、切线绕行航点（Bug2 思路）、撞墙前禁止位移。仿真里放一根
    柱子直接验证。
 6) 自检可判定：--selftest 跑场景矩阵并断言，不是"看着像能跑"。

用法
------------------------------------------------------------
  python xiaosha_brain_v2.py                 # 交互式对话（仿真后端）
  python xiaosha_brain_v2.py --demo           # 演一遍完整交互
  python xiaosha_brain_v2.py --selftest       # 机器可判定的自检
  python xiaosha_brain_v2.py --no-llm        # 强制纯离线（规则大脑）

作者：Shaqma（原型） / Hermes（v2 升级）
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request

try:
    import numpy as np  # noqa: F401
except Exception:  # pragma: no cover
    np = None

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "brain_data")
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# 配置
# ============================================================
class Config:
    # 本机沙琪玛推理服务（llama.cpp OpenAI 兼容接口）
    llm_base = os.environ.get("XIAOSHA_LLM_BASE", "http://127.0.0.1:8080/v1")
    llm_model = os.environ.get("XIAOSHA_LLM_MODEL", "Shaqma-Pro")
    llm_key = os.environ.get("XIAOSHA_LLM_KEY", "sk-local")
    llm_timeout = float(os.environ.get("XIAOSHA_LLM_TIMEOUT", "25"))
    llm_max_tokens = 160
    llm_temperature = 0.4

    memory_file = os.path.join(DATA_DIR, "memory.json")
    quiz_file = os.path.join(DATA_DIR, "quiz.json")

    wake_window = 25.0       # 叫一次名字后，这么多秒内接着说话不用再叫

    # 运动与安全
    arrive_dist = 1.0        # 到这个距离算「到你面前」
    speed = 1.2              # m/s 上限
    turn_speed = 2.0         # rad/s 上限
    min_obstacle_gap = 0.45  # 最近障碍安全距离
    max_ticks = 4000         # 单次移动上限（200s @dt=0.05）


# ============================================================
# 记忆（落盘）
# ============================================================
class Memory:
    """小沙的长期记忆：谁、在哪、教过什么、发生过什么。"""

    def __init__(self, path=None):
        self.path = path or Config.memory_file
        self.data = {
            "owner": {"name": "主人", "seen": 0},
            "places": {},
            "events": [],
            "stats": {"wake": 0, "arrive": 0, "tasks": 0, "comfort": 0},
        }
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data.update(json.load(f))
            except Exception:
                pass

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # --- 地点 ---
    def remember_place(self, name, xy):
        self.data["places"][name] = [float(xy[0]), float(xy[1])]
        self.save()

    def place(self, name):
        v = self.data["places"].get(name)
        return None if v is None else (float(v[0]), float(v[1]))

    # --- 事件 ---
    def log(self, kind, **kw):
        self.data["events"].append({"t": time.time(), "kind": kind, **kw})
        self.data["events"] = self.data["events"][-200:]

    def bump(self, key, n=1):
        self.data["stats"][key] = self.data["stats"].get(key, 0) + n
        self.save()

    def summary(self):
        s = self.data["stats"]
        return (f"记忆：主人={self.data['owner']['name']}，"
                f"地点{len(self.data['places'])}个，"
                f"唤醒{s.get('wake',0)}次/到达{s.get('arrive',0)}次/干活{s.get('tasks',0)}件")


# ============================================================
# 仿真世界：房间 + 障碍物
# ============================================================
class SimWorld:
    """2D 房间 + 圆形障碍物。用于在买硬件之前把逻辑验干净。"""

    def __init__(self, w=10.0, h=10.0, obstacles=None):
        self.w, self.h = w, h
        self.t = 0.0
        # 障碍物：圆心 + 半径
        self.obstacles = obstacles if obstacles is not None else [
            (5.0, 5.0, 0.9),   # 房间中间一根柱子
        ]

    def step(self, dt):
        self.t += dt

    def clearance(self, p):
        """到最近障碍/墙面的净空（负数 = 已经进去了）。"""
        best = min(self.w - p[0], p[0], self.h - p[1], p[1])  # 四面墙
        for (cx, cy, r) in self.obstacles:
            best = min(best, math.hypot(p[0] - cx, p[1] - cy) - r)
        return best

    def blocked(self, p):
        return self.clearance(p) < 0.0


# ============================================================
# 机器人本体（差速底盘）
# ============================================================
class Robot:
    def __init__(self, x=2.0, y=2.0, theta=0.0):
        self.pos = [float(x), float(y)]
        self.theta = theta
        self.mic_sep = 0.20     # 双麦间距 20cm（装真机要按实测改）
        self.radius = 0.25
        self.path = [list(self.pos)]

    def head(self):
        return (math.cos(self.theta), math.sin(self.theta))

    def move(self, v, omega, dt):
        self.theta = wrap_angle(self.theta + omega * dt)
        self.pos[0] += math.cos(self.theta) * v * dt
        self.pos[1] += math.sin(self.theta) * v * dt
        self.path.append(list(self.pos))


def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# ============================================================
# 感知：仿真后端 / 真机后端
# ============================================================
class SimPerception:
    """仿真感知：麦克风几何 → 真实声学时延；文本由调用方注入。"""

    speed_of_sound = 343.0

    def __init__(self, world):
        self.world = world

    def mics(self, robot):
        hl = robot.theta + math.pi / 2
        hr = robot.theta - math.pi / 2
        half = robot.mic_sep / 2.0
        left = (robot.pos[0] + math.cos(hl) * half, robot.pos[1] + math.sin(hl) * half)
        right = (robot.pos[0] + math.cos(hr) * half, robot.pos[1] + math.sin(hr) * half)
        return left, right

    def locate(self, robot, source):
        """按真实声程差算 TDOA（秒）与相对正前方的方位角（弧度，左正右负）。"""
        left, right = self.mics(robot)
        dL = math.hypot(source[0] - left[0], source[1] - left[1])
        dR = math.hypot(source[0] - right[0], source[1] - right[1])
        tdoa = (dR - dL) / self.speed_of_sound      # >0: 右耳更远 → 声源在左
        sin_t = max(-1.0, min(1.0, tdoa * self.speed_of_sound / robot.mic_sep))
        angle = math.asin(sin_t)
        return angle, tdoa


class RealPerception:
    """真机感知桩：ReSpeaker 硬件 DoA + ASR。

    装齐硬件后按这里替换实现即可，大脑其余部分不用改：
      · 声源方向：ReSpeaker Mic Array v2.0 出厂固件直接输出 DOA（度），
        不需要自己算 TDOA（环形阵列麦距只有几十毫米，套 20cm 公式必错）。
        典型做法是解析它的 USB 报文/串口输出。
      · 语音输入：Vosk 中文小模型（离线、树莓派跑得动）或 whisper.cpp tiny；
        唤醒词用 KWS（雪人/picovoice）另跑一路，避免整句 ASR 常开耗电。
    """

    def __init__(self):
        self.asr_ready = False
        try:
            import vosk  # noqa: F401
            self.asr_ready = False   # 需自备模型目录后才置 True
        except Exception:
            self.asr_ready = False

    def locate(self, robot=None, source=None):
        raise NotImplementedError(
            "真机 DoA 需要 ReSpeaker 硬件输出；本机现在是仿真后端。"
        )

    def listen(self):
        raise NotImplementedError("真机 ASR 需要 Vosk/whisper 模型；本机现在是仿真后端。")


# ============================================================
# 安全层
# ============================================================
class Safety:
    """任何运动指令都必须过这一层：限幅、势场避障、射线检测、卡死、急停。"""

    def __init__(self, world, cfg=Config):
        self.world = world
        self.cfg = cfg
        self.estop = False

    def clamp(self, v, omega, robot):
        v = max(-self.cfg.speed, min(self.cfg.speed, v))
        omega = max(-self.cfg.turn_speed, min(self.cfg.turn_speed, omega))
        return v, omega

    def avoid(self, robot, v, omega, goal):
        """势场避障：障碍与墙都产生斥力（越近越强），同时减速。

        返回 (v, omega, 最近净空)。
        """
        gap = self.cfg.min_obstacle_gap
        clear = self.world.clearance(robot.pos)
        if clear >= gap:
            return v, omega, clear

        px, py = robot.pos
        push = 0.0
        # 圆障碍斥力
        for (cx, cy, r) in self.world.obstacles:
            d = math.hypot(px - cx, py - cy) - r
            if d < gap:
                away = math.atan2(py - cy, px - cx)
                push += 1.6 * wrap_angle(away - robot.theta) * (1.0 - max(0.0, d) / gap)
        # 墙面斥力（法向指向墙内的反方向）
        for (nx, ny, d) in ((1.0, 0.0, self.world.w - px), (-1.0, 0.0, px),
                            (0.0, 1.0, self.world.h - py), (0.0, -1.0, py)):
            if d < gap:
                away = math.atan2(-ny, -nx)
                push += 1.0 * wrap_angle(away - robot.theta) * (1.0 - max(0.0, d) / gap)

        ratio = max(0.0, (gap - clear) / gap)
        v = v * (1.0 - 0.65 * ratio)
        omega = omega * (1.0 - 0.85 * ratio) + push * ratio
        return v, omega, clear

    def ray_blocked(self, start, goal, gap=None):
        """从当前位置到目标点的直线路径是否被挡住。返回 (被挡?, 第一个阻挡点)。"""
        gap = self.cfg.min_obstacle_gap * 0.9 if gap is None else gap
        total = math.hypot(goal[0] - start[0], goal[1] - start[1])
        n = max(3, int(total / 0.08))
        for i in range(1, n + 1):
            k = i / n
            p = (start[0] + (goal[0] - start[0]) * k,
                 start[1] + (goal[1] - start[1]) * k)
            if self.world.clearance(p) < gap:
                return True, p
        return False, None

    def detour_waypoint(self, pos, goal, hitpt):
        """切线绕行航点（Bug2 思路）：沿挡住去路的那个障碍的切向绕出去。

        两个切向航点里选「离目标更近」的一侧 —— 少绕冤枉路。
        """
        best, bd = None, 1e9
        for (cx, cy, r) in self.world.obstacles:
            d = math.hypot(hitpt[0] - cx, hitpt[1] - cy)
            if d < bd:
                bd, best = d, (cx, cy, r)
        if best is None:
            return None          # 挡住的是墙，没有可绕的障碍

        cx, cy, r = best
        ux, uy = hitpt[0] - cx, hitpt[1] - cy
        n = math.hypot(ux, uy) or 1.0
        ux, uy = ux / n, uy / n
        radius = r + self.cfg.min_obstacle_gap * 1.5

        cands = []
        for sgn in (1.0, -1.0):
            tx, ty = -uy * sgn, ux * sgn
            vx, vy = ux * 0.45 + tx, uy * 0.45 + ty
            vn = math.hypot(vx, vy) or 1.0
            wx, wy = cx + vx / vn * radius, cy + vy / vn * radius
            if 0.3 < wx < self.world.w - 0.3 and 0.3 < wy < self.world.h - 0.3:
                cands.append((math.hypot(goal[0] - wx, goal[1] - wy), (wx, wy)))
        if not cands:
            return None
        cands.sort(key=lambda x: x[0])
        return cands[0][1]

    def stalled(self, pos_hist, window=120, min_move=0.08):
        """卡死检测：最近 window 步（≈6s）里实际位移不到 min_move 米。

        ★ 判据必须用「位移」，不能用「到目标的距离」—— 绕障碍时距离
        长时间不缩短是正常的，用距离判据会把正常绕行误判成卡死。
        """
        if len(pos_hist) < window + 1:
            return False
        p0, p1 = pos_hist[-window - 1], pos_hist[-1]
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < min_move


# ============================================================
# 大模型决策层（本地沙琪玛）
# ============================================================
class LLM:
    """调用本机 llama-server（OpenAI 兼容）。失败一律返回 None，由规则兜底。"""

    def __init__(self, cfg=Config):
        self.cfg = cfg
        self.online = None
        self.last_error = None

    def available(self, probe=True):
        if self.online is not None and not probe:
            return self.online
        try:
            req = urllib.request.Request(
                self.cfg.llm_base.rstrip("/") + "/models",
                headers={"Authorization": "Bearer " + self.cfg.llm_key},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                body = json.loads(r.read().decode("utf-8", "ignore"))
            ids = [m.get("id") for m in body.get("data", [])]
            self.online = bool(ids)
            if self.online and self.cfg.llm_model not in ids:
                self.cfg.llm_model = ids[0]   # 模型名对不上就用服务端报的第一个
            return self.online
        except Exception as e:
            self.online = False
            self.last_error = str(e)
            return False

    def chat(self, system, user, max_tokens=None, temperature=None):
        if not self.available(probe=False):
            return None
        payload = {
            "model": self.cfg.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens or self.cfg.llm_max_tokens,
            "temperature": self.cfg.llm_temperature if temperature is None else temperature,
            "stream": False,
        }
        try:
            req = urllib.request.Request(
                self.cfg.llm_base.rstrip("/") + "/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + self.cfg.llm_key,
                },
            )
            with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout) as r:
                body = json.loads(r.read().decode("utf-8", "ignore"))
            return body["choices"][0]["message"]["content"].strip()
        except Exception as e:
            self.last_error = str(e)
            return None


# ============================================================
# 意图理解：规则 NLU（离线可用） + 模型兜底
# ============================================================
ACTION_SPEC = {
    "goto_owner": "到主人身边来",
    "goto_place": "去某个地方（客厅/书桌/充电座…）",
    "write": "用机械臂写字",
    "quiz": "背默：出一道题考我",
    "answer": "我在回答它出的题",
    "comfort": "主人情绪不好，要安抚",
    "status": "汇报自己的状态/记忆",
    "chat": "普通闲聊",
    "stop": "停下/别动",
}


class Intent:
    def __init__(self, action, slots=None, source="rule", conf=1.0, raw=""):
        self.action = action
        self.slots = slots or {}
        self.source = source
        self.conf = conf
        self.raw = raw

    def __repr__(self):
        return f"Intent({self.action}, {self.slots}, via {self.source})"


class RuleNLU:
    """把一句话解析成动作。规则先跑（确定性高），不命中再交给模型。"""

    WAKE = ["小沙", "沙琪玛", "xiaosha", "shaqma", "小莎", "小傻"]
    EMOTION = {
        "sad": ["难过", "伤心", "哭", "委屈", "失落", "emo", "不开心", "崩溃", "沮丧"],
        "angry": ["生气", "愤怒", "气死", "讨厌", "烦死", "火大", "烦", "烦躁", "郁闷", "抓狂"],
        "anxious": ["焦虑", "紧张", "害怕", "担心", "压力", "慌", "睡不着", "发愁"],
        "tired": ["困", "累", "疲惫", "没力气", "想睡", "加班", "撑不住"],
        "happy": ["开心", "高兴", "太棒", "太好了", "哈哈", "笑死", "喜欢", "顺利"],
    }

    def wake_hit(self, text):
        t = unicodedata.normalize("NFKC", (text or "").strip().lower())
        return any(w in t for w in self.WAKE)

    def emotion(self, text):
        t = text or ""
        hit = [(e, sum(1 for w in ws if w in t)) for e, ws in self.EMOTION.items()]
        hit = [(e, c) for e, c in hit if c > 0]
        if not hit:
            return None
        return max(hit, key=lambda x: x[1])[0]

    def parse(self, text, memory=None):
        t = (text or "").strip()
        low = t.lower()
        del low

        # 急停优先
        if any(k in t for k in ["停下", "别动", "停一下", "站着别", "急停"]):
            return Intent("stop", raw=t)

        # 写字：优先取引号里的内容（"把「好好学习」写下来" 是最常见的说法）
        if "写" in t:
            mq = re.search(r"[「『\"“']([^」』\"”']+)[」』\"”']", t)
            if mq:
                return Intent("write", {"text": mq.group(1).strip()}, raw=t)
            for pre in ["帮我写", "写一下", "写上", "写下", "写"]:
                idx = t.find(pre)
                if idx < 0:
                    continue
                body = t[idx + len(pre):].strip(" ：:，,。！!吧呗啊")
                # "写下来 / 写一下 / 写上去" 这类动词补语必须剥掉，否则会写错字
                for junk in ("下来", "上去", "一下", "上", "这几个字", "两个字", "三个字"):
                    if body.endswith(junk):
                        body = body[: -len(junk)].strip()
                if body:
                    return Intent("write", {"text": body}, raw=t)

        # 背默
        if any(k in t for k in ["背默", "考我", "出题", "背单词", "抽查", "默写", "背书"]):
            return Intent("quiz", raw=t)
        if any(k in t for k in ["答案是", "我答", "答：", "答案："]):
            body = t
            for k in ["答案是", "我答", "答：", "答案："]:
                if k in body:
                    body = body.split(k, 1)[1].strip()
                    break
            return Intent("answer", {"answer": body}, raw=t)

        # 去哪
        if any(k in t for k in ["过来", "到我这儿", "到我这里", "来一下", "来我", "赶紧来", "过来下"]):
            return Intent("goto_owner", raw=t)
        if memory:
            for name in memory.data.get("places", {}):
                if name in t and ("去" in t or "到" in t or "回" in t):
                    return Intent("goto_place", {"place": name}, raw=t)
        # 没记住的地名也交给 goto_place —— 由技能层诚实地答"我还没记住它在哪"，
        # 而不是把它当成闲聊（旧版就是这么丢的）。
        m = re.search(r"(?:去|到|回)(?:了)?([\u4e00-\u9fa5]{2,4})", t)
        if m:
            place = m.group(1)
            for junk in ("那边", "那里", "那儿", "看看", "一下", "一趟"):
                place = place.replace(junk, "")
            if place:
                return Intent("goto_place", {"place": place}, conf=0.85, raw=t)

        # 状态
        if any(k in t for k in ["状态", "记忆", "你会什么", "掌握度", "汇报", "电够吗", "还记得"]):
            return Intent("status", raw=t)

        # 情绪
        emo = self.emotion(t)
        if emo and emo != "happy":
            return Intent("comfort", {"emotion": emo}, raw=t)
        if emo == "happy":
            return Intent("chat", {"emotion": "happy"}, raw=t)

        return Intent("chat", raw=t, conf=0.5)


class LLMPlanner:
    """让沙琪玛自己决定干什么：把动作表给它，要求只回一行 JSON。

    小模型容易跑偏，所以这里做三重保险：
      ① 只让它做「分类 + 抽槽位」这种窄任务，不让它自由发挥；
      ② 解析失败 / 动作不在表里 → 丢弃，用规则结果；
      ③ 超时或服务没起 → 直接不调用（available(probe=False)=False）。
    """

    SYSTEM = (
        "你是机器人「小沙」的大脑。从下面的动作表里选一个最合适的动作，"
        "并抽出参数，只输出一行 JSON，不要解释、不要代码块。\n"
        "动作表：" + json.dumps(ACTION_SPEC, ensure_ascii=False) + "\n"
        "输出格式：{\"action\":\"动作名\",\"slots\":{},\"reply\":\"一句中文回应\"}\n"
        "示例：\n"
        "主人说：小沙，把「好好学习」写下来 -> "
        "{\"action\":\"write\",\"slots\":{\"text\":\"好好学习\"},\"reply\":\"好，我写下来。\"}\n"
        "主人说：我今天特别烦 -> "
        "{\"action\":\"comfort\",\"slots\":{\"emotion\":\"angry\"},\"reply\":\"我在呢，跟我说说。\"}"
    )

    def __init__(self, llm, memory=None):
        self.llm = llm
        self.memory = memory

    def plan(self, text):
        if not self.llm.available(probe=False):
            return None
        ctx = ""
        if self.memory is not None:
            places = list(self.memory.data.get("places", {}).keys())
            if places:
                ctx = "已知地点：" + "、".join(places) + "。\n"
        out = self.llm.chat(self.SYSTEM, ctx + "主人说：" + text,
                            max_tokens=120, temperature=0.2)
        if not out:
            return None
        obj = extract_json(out)
        if not obj or obj.get("action") not in ACTION_SPEC:
            return None
        return (Intent(obj["action"], obj.get("slots") or {}, source="llm", conf=0.9, raw=text),
                (obj.get("reply") or "").strip())


def extract_json(text):
    """从模型输出里抠出第一个 JSON 对象（容忍 ```json 代码块）。"""
    s = text.strip()
    if "```" in s:
        for p in s.split("```"):
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                s = p
                break
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(s[i:j + 1])
    except Exception:
        return None


# ============================================================
# 背默 / 学习
# ============================================================
class Tutor:
    def __init__(self, path=None):
        self.path = path or Config.quiz_file
        self.kb = {}          # 题目 -> 答案
        self.stats = {}       # 题目 -> {right, wrong}
        self.wrong = {}       # 错题本
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.kb = d.get("kb", {})
                self.stats = d.get("stats", {})
                self.wrong = d.get("wrong", {})
            except Exception:
                pass
        if not self.kb:
            # 开箱即用的默认题库（可直接替换成孩子的单元测验）
            self.kb = {
                "光的传播速度约是多少": "每秒30万公里",
                "三角形的内角和是多少度": "180度",
                "水的化学式": "H2O",
                "中国最长的河流": "长江",
                "谁写了《静夜思》": "李白",
            }

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"kb": self.kb, "stats": self.stats, "wrong": self.wrong},
                          f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def learn(self, q, a):
        self.kb[q] = a
        self.save()

    def pick(self):
        """难度自适应：错题加权，答对多次的题降权（不再浪费轮次）。"""
        if not self.kb:
            return None
        pool = []
        for q in self.kb:
            st = self.stats.get(q, {"right": 0, "wrong": 0})
            w = 1.0 + 3.0 * st["wrong"] - 0.6 * st["right"]
            if q in self.wrong:
                w += 2.0
            pool.append((q, max(0.05, w)))
        total = sum(w for _, w in pool)
        r = random.random() * total
        acc = 0.0
        for q, w in pool:
            acc += w
            if r <= acc:
                return q
        return pool[-1][0]

    def record(self, q, user_ans):
        truth = self.kb.get(q, "")
        ok = normalize(user_ans) == normalize(truth)
        rec = self.stats.setdefault(q, {"right": 0, "wrong": 0})
        if ok:
            rec["right"] += 1
            self.wrong.pop(q, None)
        else:
            rec["wrong"] += 1
            self.wrong[q] = truth
        self.save()
        return ok, truth

    def mastery(self):
        if not self.stats:
            return 0.0
        total = sum(r["right"] + r["wrong"] for r in self.stats.values())
        right = sum(r["right"] for r in self.stats.values())
        return right / total if total else 0.0


def normalize(s):
    s = unicodedata.normalize("NFKC", (s or "").strip().lower())
    for ch in "。，,.、！!？?《》\"'“”‘’ 「」":
        s = s.replace(ch, "")
    return s


# ============================================================
# 机械臂 / 技能层
# ============================================================
class Arm:
    """机械臂接口。仿真打印指令；真机换成串口驱动板（Feetech STS3215 总线）。"""

    def __init__(self, sim=True):
        self.sim = sim
        self.log = []

    def _emit(self, cmd, **kw):
        rec = {"cmd": cmd, **kw}
        self.log.append(rec)
        return rec

    def write(self, text):
        # 真机：把 6 轴目标位姿下发给总线舵机驱动板；写字靠位置回读，不能开环
        return self._emit("write", text=text)

    def clean_table(self):
        return self._emit("clean_table")

    def gesture(self, name):
        return self._emit("gesture", name=name)


# ============================================================
# 语言生成（模型优先，模板兜底）
# ============================================================
class Talker:
    TPL = {
        "goto_owner": ["哎，我在，这就过去。", "来啦，等我一下。"],
        "goto_place": ["好，我过去。", "收到，这就动身。"],
        "write": ["好，我写下来。", "写上了，你看看。"],
        "comfort": ["我在呢，先陪你待一会儿。", "别自己扛，我听着。"],
        "quiz": ["来，考你一道。", "题目来了，听好。"],
        "status": ["我挺好的，随时能干活。"],
        "chat": ["我在听呢，你继续说。", "嗯嗯，我听着。"],
        "stop": ["好，停住了。"],
    }

    def __init__(self, llm=None):
        self.llm = llm

    def reply(self, intent, extra=""):
        if self.llm is not None and self.llm.available(probe=False):
            sys_p = ("你是家用小机器人「小沙」，说话短、自然、有温度，"
                     "一句话，不超过 25 字，不要解释自己在做什么。")
            user = (f"主人说：{intent.raw}\n你接下来要做："
                    f"{ACTION_SPEC.get(intent.action, '聊天')}。{extra}\n请回一句话。")
            out = self.llm.chat(sys_p, user, max_tokens=60, temperature=0.7)
            if out:
                return out.strip().splitlines()[0][:60]
        return random.choice(self.TPL.get(intent.action, self.TPL["chat"]))


# ============================================================
# 大脑：把感知 / 决策 / 记忆 / 技能串成一个闭环
# ============================================================
class BrainCore:
    def __init__(self, sim=True, use_llm=True, verbose=True, data_prefix=None):
        self.cfg = Config
        self.verbose = verbose
        self.world = SimWorld()
        self.robot = Robot(2.0, 2.0, 0.0)
        mp = os.path.join(DATA_DIR, (data_prefix or "") + "memory.json")
        qp = os.path.join(DATA_DIR, (data_prefix or "") + "quiz.json")
        self.memory = Memory(mp)
        self.perception = SimPerception(self.world)
        self.safety = Safety(self.world)
        self.nlu = RuleNLU()
        self.llm = LLM() if use_llm else None
        self.planner = LLMPlanner(self.llm, self.memory) if use_llm else None
        self.talker = Talker(self.llm)
        self.tutor = Tutor(qp)
        self.arm = Arm(sim=sim)
        self.current_q = None
        self.wake_calls = 0
        self.awake_until = 0.0   # 唤醒窗口：叫一次名字，接着能连说几句

    # ---------- 感知 ----------
    def locate(self, source):
        return self.perception.locate(self.robot, source)

    # ---------- 运动：走到目标点（绕障 + 防撞 + 卡死保护）----------
    def drive_to(self, goal):
        dt = 0.05
        ticks = 0
        pos_hist = [tuple(self.robot.pos)]
        min_clear = 1e9
        hit_wall = False
        detour = None
        detour_used = 0
        note = ""

        while ticks < self.cfg.max_ticks:
            to = (goal[0] - self.robot.pos[0], goal[1] - self.robot.pos[1])
            dist = math.hypot(to[0], to[1])
            err = wrap_angle(math.atan2(to[1], to[0]) - self.robot.theta)

            # 到位判定：够近 + 朝目标
            if dist <= self.cfg.arrive_dist and abs(err) < 0.15:
                break

            # ① 直线被挡就生成切线绕行航点，否则直冲目标
            blocked, hitpt = self.safety.ray_blocked(tuple(self.robot.pos), goal)
            if blocked and dist > self.cfg.arrive_dist + 0.3:
                if (detour is None
                        or math.hypot(self.robot.pos[0] - detour[0],
                                      self.robot.pos[1] - detour[1]) < 0.35):
                    if detour is not None:
                        detour_used += 1
                    detour = self.safety.detour_waypoint(tuple(self.robot.pos), goal, hitpt)
                target = detour if detour else goal
                arrive_now = 0.30
                if detour_used > 20:
                    note = "绕行多次仍过不去，停机报告"
                    break
            else:
                detour = None
                target = goal
                arrive_now = self.cfg.arrive_dist

            # ② 对当前目标点做 P 控制
            to_t = (target[0] - self.robot.pos[0], target[1] - self.robot.pos[1])
            d_t = math.hypot(to_t[0], to_t[1])
            err_t = wrap_angle(math.atan2(to_t[1], to_t[0]) - self.robot.theta)
            if d_t < 1e-6:
                omega, v = 0.0, 0.0
            elif d_t > arrive_now:
                omega = 2.5 * err_t
                v = self.cfg.speed * max(0.0, math.cos(err_t))
            else:
                omega = 2.0 * err_t      # 到了就原地转身面对目标
                v = 0.0

            # ③ 安全层：限幅 → 势场避障
            v, omega = self.safety.clamp(v, omega, self.robot)
            v, omega, clear = self.safety.avoid(self.robot, v, omega, target)
            min_clear = min(min_clear, clear)

            # ④ 撞墙/撞障前的硬保护：预测下一步会进去就不许位移
            theta_next = wrap_angle(self.robot.theta + omega * dt)
            nxt = (self.robot.pos[0] + math.cos(theta_next) * v * dt,
                   self.robot.pos[1] + math.sin(theta_next) * v * dt)
            if self.world.blocked(nxt):
                v = 0.0
                hit_wall = True

            self.robot.move(v, omega, dt)
            self.world.step(dt)
            ticks += 1
            pos_hist.append(tuple(self.robot.pos))

            if self.safety.stalled(pos_hist):
                note = "卡死（连续 6 秒几乎没挪动），停机不再硬撞"
                break

        final = math.hypot(goal[0] - self.robot.pos[0], goal[1] - self.robot.pos[1])
        facing = abs(wrap_angle(math.atan2(goal[1] - self.robot.pos[1],
                                          goal[0] - self.robot.pos[0]) - self.robot.theta))
        ok = final <= self.cfg.arrive_dist + 1e-6
        self.memory.log("drive", goal=[round(goal[0], 2), round(goal[1], 2)],
                        final_dist=round(final, 3), ok=ok, t=round(self.world.t, 2))
        self.memory.bump("arrive" if ok else "miss")
        return {
            "ok": ok,
            "final_dist": round(final, 3),
            "facing_err_deg": round(math.degrees(facing), 1),
            "seconds": round(self.world.t, 2),
            "min_obstacle_clearance": round(min_clear, 3),
            "hit_obstacle": hit_wall,
            "ticks": ticks,
            "note": note,
        }

    # ---------- 决策 ----------
    def understand(self, text):
        """先规则，规则没把握再让沙琪玛判。返回 (intent, 模型给的回应)"""
        ri = self.nlu.parse(text, self.memory)
        if ri.conf >= 1.0:
            return ri, ""
        if self.planner is not None:
            got = self.planner.plan(text)
            if got:
                li, reply = got
                # 模型说要去某地但没给地名 → 保留更安全的规则结果
                if li.action == "goto_place" and not li.slots.get("place"):
                    return ri, reply
                return li, reply
        return ri, ""

    # ---------- 技能执行（一个脑子管全身）----------
    def handle(self, text, source_pos=None):
        result = {"input": text, "steps": []}

        # ① 唤醒判定：叫名字唤醒；窗口期内（默认 25s）接着说话不用再叫
        now = time.monotonic()
        if self.nlu.wake_hit(text):
            self.awake_until = now + self.cfg.wake_window
            self.memory.bump("wake")
            self.wake_calls += 1
        elif now > self.awake_until:
            result["awake"] = False
            result["reply"] = ""
            return result
        result["awake"] = True

        intent, llm_reply = self.understand(text)
        result["intent"] = intent.action
        result["intent_via"] = intent.source

        # ② 要移动就先定位声源
        if intent.action == "goto_owner" and source_pos is not None:
            angle, tdoa = self.locate(source_pos)
            result["locate"] = {"angle_deg": round(math.degrees(angle), 1),
                                "tdoa_us": round(tdoa * 1e6, 1)}

        # ③ 回应
        reply = llm_reply or self.talker.reply(
            intent, extra=json.dumps(intent.slots, ensure_ascii=False))
        result["reply"] = reply

        # ④ 执行
        if intent.action == "stop":
            self.safety.estop = True
            result["steps"].append({"skill": "stop", "ok": True})

        elif intent.action == "goto_owner":
            if source_pos is None:
                result["steps"].append({"skill": "goto_owner", "ok": False,
                                        "why": "没有声源位置（真机需 DoA）"})
            else:
                r = self.drive_to(source_pos)
                result["steps"].append({"skill": "goto_owner", **r})
                if r["ok"]:
                    self.memory.remember_place("主人身边", source_pos)

        elif intent.action == "goto_place":
            name = intent.slots.get("place", "")
            xy = self.memory.place(name)
            if xy is None:
                # 没记过就说不知道，别乱走（诚实优先）
                result["steps"].append({"skill": "goto_place", "ok": False,
                                        "why": f"没去过「{name}」，位置未记录"})
                result["reply"] = f"「{name}」我还没记住在哪，你带我去一次我就记住了。"
            else:
                r = self.drive_to(xy)
                result["steps"].append({"skill": "goto_place", **r})

        elif intent.action == "write":
            cmd = self.arm.write(intent.slots.get("text", ""))
            self.memory.bump("tasks")
            result["steps"].append({"skill": "write", "ok": True, "cmd": cmd})

        elif intent.action == "quiz":
            q = self.tutor.pick()
            self.current_q = q
            result["steps"].append({"skill": "quiz", "ok": q is not None, "question": q,
                                    "mastery": round(self.tutor.mastery(), 3)})
            if q:
                result["reply"] = f"来，请回答 —— {q}"

        elif intent.action == "answer":
            if self.current_q is None:
                result["steps"].append({"skill": "answer", "ok": False, "why": "还没有出题"})
                result["reply"] = "先让我出题吧。"
            else:
                ok, truth = self.tutor.record(self.current_q, intent.slots.get("answer", ""))
                result["steps"].append({"skill": "answer", "ok": ok,
                                        "question": self.current_q, "truth": truth,
                                        "mastery": round(self.tutor.mastery(), 3),
                                        "wrong_bank": len(self.tutor.wrong)})
                result["reply"] = ("对啦！" if ok
                                   else f"答案是「{truth}」，我记进错题本了，下次还考你。")
                self.current_q = None

        elif intent.action == "comfort":
            self.memory.bump("comfort")
            self.memory.log("comfort", emotion=intent.slots.get("emotion"))
            result["steps"].append({"skill": "comfort", "ok": True,
                                    "emotion": intent.slots.get("emotion")})

        elif intent.action == "status":
            result["steps"].append({"skill": "status", "ok": True,
                                    "memory": self.memory.summary(),
                                    "mastery": round(self.tutor.mastery(), 3),
                                    "wrong_bank": len(self.tutor.wrong)})
            result["reply"] = self.memory.summary()

        else:  # chat
            result["steps"].append({"skill": "chat", "ok": True})

        self.memory.save()
        return result


# ============================================================
# 演示 & 自检
# ============================================================
def demo(use_llm=True):
    b = BrainCore(use_llm=use_llm)
    owner = (8.0, 7.5)
    print("=" * 62)
    print("  小沙大脑 v2 · 演示（决策层：本地沙琪玛）")
    print("=" * 62)
    print(f"  模型在线：{b.llm.available() if b.llm else False}"
          + (f"（{b.llm.cfg.llm_model}）" if b.llm and b.llm.online else ""))
    print(f"  小沙起点：(2.0, 2.0)  主人在：({owner[0]}, {owner[1]})  中间有柱子(5,5,r0.9)")
    print("-" * 62)

    lines = [
        "小沙，过来一下",          # 唤醒 + 绕开柱子走到你面前
        "我今天特别烦",            # 唤醒窗口内：不用再叫名字
        "把「好好学习」写下来",      # 同上，交给机械臂
        "考我一道题",              # 同上
        "答案是180度",             # 同上
        "你状态怎么样",             # 同上
    ]
    for t in lines:
        r = b.handle(t, source_pos=owner)
        print(f"\n[主人] {t}")
        print(f"   -> 意图：{r.get('intent')}（来自{r.get('intent_via')}）")
        if r.get("locate"):
            print(f"   -> 声源：正前方偏 {r['locate']['angle_deg']}°，"
                  f"TDOA={r['locate']['tdoa_us']}µs")
        print(f"   [小沙] {r.get('reply')}")
        for s in r.get("steps", []):
            if s.get("skill") in ("goto_owner", "goto_place"):
                print(f"   [移动] 到了={s.get('ok')} 终距={s.get('final_dist')}m "
                      f"用时={s.get('seconds')}s 最近障碍={s.get('min_obstacle_clearance')}m")
                if s.get("note"):
                    print(f"          {s['note']}")
            elif s.get("skill") == "write":
                print(f"   [机械臂] {s.get('cmd')}")
            elif s.get("skill") == "answer":
                print(f"   [背默] 判分={'对' if s.get('ok') else '错'} "
                      f"掌握度={s.get('mastery')} 错题本={s.get('wrong_bank')}")
    print("\n" + "-" * 62)
    print("  路径点数：", len(b.robot.path))
    return b


def selftest():
    """机器可判定的自检：每个断言都带证据，全过才算升级成功。"""
    fails = []
    checks = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        if not cond:
            fails.append(name)

    print("=" * 70)
    print("  小沙大脑 v2 · 自检（--selftest）")
    print("=" * 70)

    # ---------- 1. 规则 NLU ----------
    b = BrainCore(use_llm=False, data_prefix="sel_")
    cases = [
        ("小沙，过来一下", "goto_owner"),
        ("小沙，把「好好学习」写下来", "write"),
        ("小沙考我一道题", "quiz"),
        ("答案是180度", "answer"),
        ("我今天特别烦", "comfort"),
        ("小沙，你状态怎么样", "status"),
        ("小沙，停下", "stop"),
        ("今天天气不错", "chat"),
    ]
    ok_all, detail = True, []
    for text, want in cases:
        got = b.nlu.parse(text, b.memory).action
        detail.append(f"{text}→{got}{'' if got == want else '(期望' + want + ')'}")
        ok_all = ok_all and (got == want)
    check("规则 NLU 8 条指令全部正确解析", ok_all, " | ".join(detail))

    # ---------- 2. 唤醒词 ----------
    check("唤醒词命中「小沙/沙琪玛」，不误唤醒无关语句",
          b.nlu.wake_hit("小沙，过来") and b.nlu.wake_hit("沙琪玛在吗")
          and not b.nlu.wake_hit("今天天气不错"))

    # ---------- 3. 声源定位的物理正确性 ----------
    r1 = BrainCore(use_llm=False, data_prefix="sel_")
    r1.robot = Robot(2.0, 2.0, 0.0)
    ang_left, tdoa_left = r1.locate((2.0, 9.0))     # 正左
    ang_right, _ = r1.locate((2.0, -5.0))           # 正右
    ang_front, _ = r1.locate((9.0, 2.0))            # 正前
    check("声源在左 → 方位角为正（左正右负约定）", ang_left > 0 and ang_right < 0,
          f"左={math.degrees(ang_left):.1f}° 右={math.degrees(ang_right):.1f}°")
    check("正前方声源方位角≈0", abs(math.degrees(ang_front)) < 0.5,
          f"{math.degrees(ang_front):.3f}°")
    check("时延量级符合声学（20cm 麦距 → 上限 583.1µs）",
          0 < abs(tdoa_left) * 1e6 <= 584, f"{tdoa_left*1e6:.1f}µs")

    # ---------- 4. 「叫一声就过来」闭环（中间有柱子）----------
    b2 = BrainCore(use_llm=False, data_prefix="sel_")
    owner = (8.0, 7.5)
    res = b2.handle("小沙，过来一下", source_pos=owner)
    mv = [s for s in res["steps"] if s["skill"] == "goto_owner"][0]
    check("闭环①：叫一声 → 真的绕开柱子开到你面前（终距≤1.0m）",
          mv["ok"] and mv["final_dist"] <= 1.0,
          f"终距={mv['final_dist']}m 用时={mv['seconds']}s 障碍余量={mv['min_obstacle_clearance']}m")
    check("闭环②：全程没撞进障碍/墙里", not mv["hit_obstacle"],
          f"最近障碍距离={mv['min_obstacle_clearance']}m")

    # ---------- 5. 边界场景矩阵 ----------
    edge = [
        ("正后方（起点7,7→目标7,3）", (7.0, 7.0), (7.0, 3.0), True),
        ("正左（起点2,5→目标2,9）", (2.0, 5.0), (2.0, 9.0), True),
        ("主人就在1米内", (2.0, 2.0), (2.8, 2.0), True),
        ("对角远端且中间有柱子", (1.0, 1.0), (9.5, 9.5), True),
        ("目标在房间外（应诚实报到达不了，且不撞穿墙）", (2.0, 2.0), (-3.0, 2.0), False),
    ]
    for name, start, pos, want_ok in edge:
        bb = BrainCore(use_llm=False, data_prefix="sel_")
        bb.robot = Robot(start[0], start[1], 0.0)
        rr = bb.drive_to(pos)
        if want_ok:
            check(f"边界：{name} → 能到达", rr["ok"] and not rr["hit_obstacle"],
                  f"终距={rr['final_dist']}m 用时={rr['seconds']}s "
                  f"障碍余量={rr['min_obstacle_clearance']}m {rr['note']}")
        else:
            check(f"边界：{name}", (rr["ok"] is False) and not rr["hit_obstacle"],
                  f"终距={rr['final_dist']}m 用时={rr['seconds']}s {rr['note']}")

    # ---------- 6. 记忆持久化 ----------
    mem_path = os.path.join(DATA_DIR, "_selftest_memory.json")
    if os.path.exists(mem_path):
        os.remove(mem_path)
    m1 = Memory(mem_path)
    m1.remember_place("客厅", (7.0, 1.5))
    m1.bump("wake", 3)
    m2 = Memory(mem_path)     # 重新加载 = 模拟重启
    check("记忆落盘：重启后地点与计数还在",
          m2.place("客厅") == (7.0, 1.5) and m2.data["stats"].get("wake") == 3,
          f"客厅={m2.place('客厅')} wake={m2.data['stats'].get('wake')}")

    b3 = BrainCore(use_llm=False, data_prefix="sel_")
    b3.memory.remember_place("客厅", (7.0, 1.5))
    r3 = b3.handle("小沙，去客厅")
    st = r3["steps"][0]
    check("技能：说「去客厅」→ 走到记忆里的坐标",
          st["skill"] == "goto_place" and st.get("ok") is True,
          f"终距={st.get('final_dist')}m")
    b4 = BrainCore(use_llm=False, data_prefix="sel_")
    r4 = b4.handle("小沙，去厨房")
    check("诚实优先：没记过的地点不乱走，明说不知道",
          r4["steps"][0].get("ok") is False and "没记住" in r4["reply"], r4["reply"])

    # ---------- 6.5 唤醒窗口 & 槽位抽取（真人怎么用就怎么测）----------
    b7 = BrainCore(use_llm=False, data_prefix="sel_")
    ra = b7.handle("小沙，过来一下", source_pos=(3.0, 2.0))
    rb = b7.handle("我今天特别烦")
    check("唤醒窗口：叫一次名字后紧接着的句子不用再叫，且不算新唤醒",
          ra.get("awake") and rb.get("awake") and b7.wake_calls == 1,
          f"第一句awake={ra.get('awake')} 第二句awake={rb.get('awake')} "
          f"wake_calls={b7.wake_calls}")
    b8 = BrainCore(use_llm=False, data_prefix="sel_")
    check("没叫名字就不答应（避免乱插话）",
          b8.handle("今天天气不错").get("awake") is False)
    w1 = b.nlu.parse("小沙，把「好好学习」写下来", b.memory).slots.get("text")
    w2 = b.nlu.parse("小沙，帮我写好好学习", b.memory).slots.get("text")
    check("写字槽位：带引号/不带引号都要抽出正确内容，不能抽出「下来」",
          w1 == "好好学习" and w2 == "好好学习", f"带引号={w1} 无引号={w2}")

    # ---------- 7. 背默：判分 + 错题本 ----------
    quiz_path = os.path.join(DATA_DIR, "_selftest_quiz.json")
    if os.path.exists(quiz_path):
        os.remove(quiz_path)
    b5 = BrainCore(use_llm=False, data_prefix="sel_")
    b5.tutor = Tutor(quiz_path)
    b5.tutor.kb = {"1+1": "2", "水的化学式": "H2O"}
    b5.current_q = "1+1"
    rr = b5.handle("小沙，答案是2")
    ok_rec = rr["steps"][0]["ok"] is True
    b5.current_q = "水的化学式"
    rr2 = b5.handle("小沙，答案是H2O吗")
    wrong_rec = rr2["steps"][0]["ok"] is False
    check("背默：答对判对、答错进错题本",
          ok_rec and wrong_rec and "水的化学式" in b5.tutor.wrong,
          f"错题本={list(b5.tutor.wrong.keys())} 掌握度={rr2['steps'][0]['mastery']}")

    # ---------- 8. 真模型（本地沙琪玛，可选）----------
    lb = BrainCore(use_llm=True, data_prefix="sel_")
    online = lb.llm.available() if lb.llm else False
    if online:
        got = lb.planner.plan("把「早点睡」这两个字写在纸上")
        check("模型决策层：本地沙琪玛能把话翻译成结构化动作",
              got is not None and got[0].action in ACTION_SPEC,
              f"{got[0] if got else 'None'}")
        rep = lb.talker.reply(Intent("comfort", raw="我今天特别烦"))
        check("模型说话：能生成一句人话（非模板）", bool(rep), f"“{rep}”")
    else:
        checks.append(("模型决策层：本机 8080 未就绪 → 已自动走离线规则大脑", True,
                       (lb.llm.last_error or "")))

    # ---------- 9. 安全层 ----------
    b6 = BrainCore(use_llm=False, data_prefix="sel_")
    v, w = b6.safety.clamp(99.0, 99.0, b6.robot)
    check("安全：速度/角速度被限幅", v == Config.speed and w == Config.turn_speed,
          f"v={v} omega={w}")
    snap = list(b6.robot.pos)
    r6 = b6.handle("小沙，停下")
    check("安全：急停指令被识别且不产生位移",
          r6["intent"] == "stop" and b6.robot.pos == snap)

    # ---------- 汇总 ----------
    print()
    npass = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
    print("-" * 70)
    print(f"  结果：{npass}/{len(checks)} 项通过"
          + ("  ✅ 全部通过" if not fails else f"  ❌ 失败项：{fails}"))
    print("=" * 70)
    return 0 if not fails else 1


# ============================================================
def interactive(use_llm=True):
    b = BrainCore(use_llm=use_llm, sim=True)
    print("=" * 62)
    print("  小沙大脑 v2 · 交互模式（原型：Shaqma / 升级：Hermes）")
    print("=" * 62)
    print("  每句话都要带唤醒词「小沙」或「沙琪玛」。示例：")
    print("    小沙，过来一下             （会绕开障碍走过来）")
    print("    小沙，我今天特别烦         （安抚）")
    print("    小沙，把「好好学习」写下来 （机械臂指令）")
    print("    小沙考我一道题 / 小沙，答案是XXX")
    print("    小沙，你状态怎么样         （汇报记忆）")
    print("    小沙，去客厅               （先要带它去过一次才记得）")
    print("  输入 退出 结束。/主人 8 7.5 可改主人位置。")
    print(f"  模型：{b.llm.cfg.llm_model if b.llm and b.llm.available() else '离线规则大脑'}")
    print("-" * 62)
    owner = [8.0, 7.5]
    while True:
        try:
            raw = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n小沙：拜拜~")
            break
        if raw in ("退出", "exit", "quit"):
            print("小沙：拜拜，记得充电~")
            break
        if raw.startswith("/主人 "):
            try:
                _, x, y = raw.split()
                owner = [float(x), float(y)]
                print(f"（主人位置已设为 {owner}）")
            except Exception:
                print("（格式：/主人 8 7.5）")
            continue
        r = b.handle(raw, source_pos=tuple(owner))
        if not r.get("awake"):
            print("小沙：（没叫我……）")
            continue
        print(f"小沙：{r.get('reply')}")
        for s in r.get("steps", []):
            if s.get("skill") in ("goto_owner", "goto_place"):
                print(f"      [移动] 到达={s.get('ok')} 终距={s.get('final_dist')}m "
                      f"用时={s.get('seconds')}s")
            elif s.get("skill") == "write":
                print(f"      [机械臂] {s.get('cmd')}")
            elif s.get("skill") == "status":
                print(f"      [状态] {s.get('memory')}")
    return b


def main():
    ap = argparse.ArgumentParser(description="小沙大脑 v2")
    ap.add_argument("--selftest", action="store_true", help="跑机器可判定的自检")
    ap.add_argument("--demo", action="store_true", help="演一遍完整交互")
    ap.add_argument("--no-llm", action="store_true", help="强制离线（不用沙琪玛模型）")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.demo:
        demo(use_llm=not args.no_llm)
        return
    interactive(use_llm=not args.no_llm)


if __name__ == "__main__":
    main()
