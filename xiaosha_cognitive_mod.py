# -*- coding: utf-8 -*-
"""认知调制层 —— 把九项认知能力接进脉冲闭环（每项可单独消融）。

设计原则（关键）：
  这些能力**不直接产生动作**，而是调制"皮层→动作"这条链路。
  就像真实大脑里前额叶/蘑菇体调节运动皮层，而不是替代它。

  调制链：
      感官 → [注意增益] → 皮层放电 → 原始动作 → [模式选择/生病缩放/飞行转换] → 最终动作

  每一项都有 enabled 开关，关掉后对应调制消失 —— 这是可验证的因果链。

九项能力各自的调制作用：
  ① 数量感 numerosity  —— 目标数量影响"接近强度"（多个目标时更兴奋）
  ② 时间感知 timing    —— 决定动作节奏（计时到点才切换行为）
  ③ 注意 attention     —— 放大被注意的输入通道（biased competition）
  ④ 冲动控制 impulse   —— 调节决策持久度（耐心=保持当前行为更久）
  ⑤ 社会学习 social    —— 观察同伴后偏好偏移（方向偏置）
  ⑥ 群体决策 swarm     —— 从众倾向（方向向群体共识靠拢）
  ⑦ 飞行 flight        —— 动作转成翅膀命令（thrust/roll）
  ⑧ 运动模式库 motor   —— 决定用哪种模式（走/爬/梳理/跳/僵/飞）
  ⑨ 生病行为 sickness  —— 全局活动缩放 + 社交回避
"""
import numpy as np

from xiaosha_cognition import (
    NumerositySense, IntervalTimer, SelectiveAttention, ImpulseControl,
    SocialLearning, SwarmDecision, FlightController, MotorRepertoire,
    SicknessBehavior,
)

__version__ = "1.0.0"


class CognitiveModulator:
    """九项认知能力 → 皮层/动作调制。每项可单独消融。"""

    NAMES = ["numerosity", "timing", "attention", "impulse", "social",
             "swarm", "flight", "motor", "sickness"]

    def __init__(self):
        self.numerosity = NumerositySense()
        self.timer = IntervalTimer()
        self.attention = SelectiveAttention(n_feature=8)
        self.impulse = ImpulseControl()
        self.social = SocialLearning(n_option=4)
        self.swarm = SwarmDecision(n_agents=20)
        self.flight = FlightController()
        self.motor = MotorRepertoire()
        self.sickness = SicknessBehavior()

        self.enabled = {n: True for n in self.NAMES}
        self.t = 0
        self.last_mode = "walk"
        self.mode_hold = 0

    # ---------- 皮层之前：注意增益 ----------
    def pre_cortex(self, feats, task_feature=None):
        """返回输入增益向量（每个特征通道一个增益）。

        只有"注意"会在这里生效：
          显著的特征竞争 → 赢家通道增益提高（biased competition）
        """
        g = np.ones(feats.size, dtype=np.float32)
        if not self.enabled["attention"] or feats.size < 8:
            return g
        head = feats[:8].copy()
        # 归一化到 0-1（避免量纲问题）
        mx = np.abs(head).max()
        if mx > 1e-6:
            head = head / mx
        if task_feature is not None:
            self.attention.set_task(int(task_feature) % 8, gain=2.5)
        winner, w, _ = self.attention.compete(head)
        # 赢家通道放大（最多 2 倍），其余略微压制
        g[:8] = (1.0 + 1.0 * w / (w.max() + 1e-9)).astype(np.float32)
        return g

    # ---------- 皮层之后：模式/缩放/飞行 ----------
    def post_cortex(self, scene, action, sense=None):
        """调制皮层输出。返回 (最终动作, 诊断 dict)。"""
        self.t += 1
        sense = sense or {}
        info = {"mode": "walk", "scales": {}, "ablated": []}
        act = np.asarray(action, dtype=np.float32).copy()

        # ---- ① 数量感：目标多 → 接近强度更大 ----
        if self.enabled["numerosity"]:
            n_items = int(sense.get("n_items", 1))
            est = self.numerosity.accumulate(n_items)
            # 多目标 → 略微增强前进（"这里东西多"）
            boost = float(np.clip((est - 1.0) * 0.08, 0, 0.25))
            if boost > 0:
                act[0] *= (1.0 + boost)
                info["scales"]["numerosity"] = 1.0 + boost
        else:
            info["ablated"].append("numerosity")

        # ---- ② 时间感知：到点才允许切换模式 ----
        if self.enabled["timing"]:
            self.timer.tick(0.1)
            # 每 1.0s 重置一次（决定模式切换的节律）
            if self.timer.elapsed() >= 1.0:
                self.timer.start()
                self.mode_hold = max(0, self.mode_hold - 1)
        else:
            info["ablated"].append("timing")

        # ---- ③ 注意：不在此处（pre_cortex 已处理），只报告 ----
        if not self.enabled["attention"]:
            info["ablated"].append("attention")

        # ---- ④ 冲动控制：耐心 → 保持当前模式更久 ----
        if self.enabled["impulse"]:
            patience = self.impulse.patience_score()
            # 耐心高 → mode_hold 增加（不轻易换行为）
            # 阈值 1.0：默认 serotonin=0.5/k=0.35 算出 patience=1.25，
            # 设 2.0 的话永远触发不了（实测 A/B 切换次数都是 9）
            if patience > 1.0:
                self.mode_hold = min(8, self.mode_hold + 1)
            info["scales"]["patience"] = patience
        else:
            info["ablated"].append("impulse")

        # ---- ⑤ 社会学习：观察到的偏好 → 方向偏置 ----
        if self.enabled["social"]:
            pref = int(np.argmax(self.social.value)) if self.social.observations else -1
            if pref >= 0 and self.social.value[pref] > 0.2:
                # 关键：用**乘法门控**而不是加法偏置。
                # 加法会让皮层静默时依然产生动作，污染"行为来自皮层"的因果链。
                # 乘法 = 放大/缩小皮层已有的方向（从众/偏好只是调制，不替代）
                bias = -0.15 if pref <= 1 else 0.15
                act[2] = float(act[2] * (1.0 + bias))
                info["scales"]["social_bias"] = bias
        else:
            info["ablated"].append("social")

        # ---- ⑥ 群体决策：从众（方向向共识靠拢）----
        if self.enabled["swarm"]:
            frac, locked = self.swarm.step()
            if locked:
                consensus_dir = 1.0 if frac >= 0.5 else -1.0
                # 同样用乘法：方向与共识一致 → 放大；相反 → 缩小
                if abs(act[2]) > 1e-6:
                    same = np.sign(act[2]) == consensus_dir
                    act[2] = float(act[2] * (1.2 if same else 0.75))
                info["scales"]["swarm_consensus"] = consensus_dir
        else:
            info["ablated"].append("swarm")

        # ---- ⑦ 运动模式库：决定用哪种模式 ----
        mode = "walk"
        if self.enabled["motor"]:
            threat = float(sense.get("threat", 0.0))
            uphill = -float(sense.get("tilt", 0.0))
            need_escape = bool(sense.get("escape", False))
            new_mode, params = self.motor.select(threat=threat, gravity_dir=uphill,
                                                 need_escape=need_escape)
            # ④b 冲动控制的真正生效点：耐心高 → 保持上一个模式，不轻易切换
            # （这是行为持续性，真实动物靠它避免"每 100ms 改主意"）
            if (self.enabled["impulse"] and self.mode_hold > 0
                    and new_mode != self.last_mode
                    and new_mode not in ("freeze", "jump")):
                mode = self.last_mode
                self.mode_hold -= 1
                info["scales"]["mode_held"] = mode
            else:
                mode = new_mode
            info["mode"] = mode
            info["mode_params"] = params

            if mode == "freeze":
                act[:] = 0.0                      # 僵住：不动
                info["scales"]["mode"] = "freeze->0"
            elif mode == "jump":
                act[0] = 1.5                      # 弹射
                info["scales"]["mode"] = "jump"
            elif mode == "climb":
                act[0] = abs(act[0]) * 1.2 + 0.2  # 强制向上/前进
                info["scales"]["mode"] = "climb"
            elif mode.startswith("groom"):
                act[:] = 0.0                      # 停下梳理
                act[3] = 0.4                      # 手臂动
                info["scales"]["mode"] = mode
        else:
            info["ablated"].append("motor")
            info["mode"] = "walk(ablated)"

        # ---- ⑧ 飞行：把动作转成翅膀命令 ----
        if self.enabled["flight"] and mode in ("fly", "jump") or (
                self.enabled["flight"] and sense.get("flying", False)):
            cmd = self.flight.command(thrust=float(np.clip(abs(act[0]), 0, 1)),
                                      roll=float(np.clip(act[2], -1, 1)))
            info["flight"] = cmd
            # 飞行时地面速度转成前向推力
            act[0] = cmd["lift"] * np.sign(act[0] if act[0] != 0 else 1.0)
        elif self.enabled["flight"]:
            info["flight"] = self.flight.hover()
        else:
            info["ablated"].append("flight")

        # ---- ⑨ 生病行为：全局活动缩放 ----
        if self.enabled["sickness"]:
            st = self.sickness.state()
            act *= st["activity_scale"]
            info["scales"]["sickness"] = st["activity_scale"]
            info["sick"] = st["sick"]
        else:
            info["ablated"].append("sickness")

        self.last_mode = mode
        return act, info

    # ---------- 外部事件接口 ----------
    def observe_conspecific(self, option, outcome=0.0):
        self.social.observe(option, outcome)

    def infect(self, severity=0.6):
        self.sickness.infect(severity)

    def heal_step(self):
        self.sickness.step()

    def soil(self, part, amount=0.2):
        self.motor.soil(part, amount)

    def task_attention(self, feature_id):
        """设定注意任务（下一次 pre_cortex 生效）。"""
        self.attention.set_task(feature_id)

    def ablate(self, name):
        """消融某一项能力（用于因果验证）。"""
        if name in self.enabled:
            self.enabled[name] = False
            return True
        return False

    def restore_all(self):
        for n in self.NAMES:
            self.enabled[n] = True

    def status(self):
        return {n: self.enabled[n] for n in self.NAMES}
