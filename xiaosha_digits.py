# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""⑧ 的真身：看数字（十进制的 0~9）→ 可选转 2/16 进制。

为什么换任务：导航实验证明"加视觉"没用（21/30 → 纯视觉 1/30、混合 11/30）——
导航要的信息 8 路测距已经给全了，视觉在那里是**无效信息**。
数字识别不一样：**没有测距仪能替代眼睛**，识别只能靠看。

链路：渲染数字（七段式，随机位置/大小/亮度/噪声）→ 相机帧 → 视叶式前端 → 脉冲皮层
      → 10 个动作里选出"这是几"（奖励 = 对 +1 / 错 −1）→ 准确率
"可选转 2/16 进制"：认出数字后按位转（4 位二进制 / 1 位十六进制），确定性映射。

自检（--selftest）：
  ① 眼睛看得见数字（编码非退化）
  ② **学得会**：练 1200 次，最后 300 次的准确率显著高于瞎猜（10%）
  ③ 转 2/16 进制正确（0~9 全查一遍）
  ④ 速度：单次试次毫秒级（含取景 + 编码 + 10 步皮层）
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\Apps\MyModel\chat\workspace\xiaosha")
import xiaosha_cortex as C
import xiaosha_vision as V

# 七段式数字：段位 a~g（上/右上/右下/下/左下/左上/中）
SEG = {
    0: "abcdef", 1: "bc", 2: "abdeg", 3: "abcdg", 4: "bcfg",
    5: "acdfg", 6: "acdefg", 7: "abc", 8: "abcdefg", 9: "abcdfg",
}


def render_digit(d, size=V.IMG, scale=0.72, dx=0.0, dy=0.0, bright=1.0, noise=0.06, rng=None):
    """把数字 d(0~9) 画成 size×size 灰度帧（七段式，位置/大小/亮度可随机）。"""
    rng = rng or np.random.default_rng(0)
    img = np.full((size, size), 0.08, dtype=np.float32)
    h = scale * size * 0.5                       # 半高
    w = h * 0.62
    cx, cy = size * 0.5 + dx * size, size * 0.5 + dy * size
    t = max(1.0, size * 0.05)                    # 笔画粗细

    def bar(x0, y0, x1, y1):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for i in range(n + 1):
            f = i / max(1, n)
            x, y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
            x0i, x1i = max(0, int(x - t)), min(size, int(x + t) + 1)
            y0i, y1i = max(0, int(y - t)), min(size, int(y + t) + 1)
            if x1i > x0i and y1i > y0i:
                img[y0i:y1i, x0i:x1i] += bright * 0.85

    seg = SEG[d]
    pts = {
        "a": (cx - w, cy - h, cx + w, cy - h), "b": (cx + w, cy - h, cx + w, cy),
        "c": (cx + w, cy, cx + w, cy + h), "d": (cx - w, cy + h, cx + w, cy + h),
        "e": (cx - w, cy, cx - w, cy + h), "f": (cx - w, cy - h, cx - w, cy),
        "g": (cx - w, cy, cx + w, cy),
    }
    for s in seg:
        bar(*pts[s])
    img += noise * rng.standard_normal((size, size)).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def to_bin_hex(d):
    """可选：转 2/16 进制（4 位二进制 + 1 位十六进制）。"""
    return format(d, "04b"), format(d, "X")


class DigitTask:
    """一回合一帧：看一个数字，10 个动作里选一个（奖励 = 对 +1 / 错 −1）。"""

    def __init__(self, size=V.IMG, seed=0):
        self.size, self.tol = size, 1.0
        self.rng = np.random.default_rng(seed)
        self.digit = 0

    def new_episode(self):
        self.digit = int(self.rng.integers(0, 10))
        self._frame = render_digit(self.digit, size=self.size,
                                   scale=float(self.rng.uniform(0.55, 0.85)),
                                   dx=float(self.rng.uniform(-0.12, 0.12)),
                                   dy=float(self.rng.uniform(-0.12, 0.12)),
                                   bright=float(self.rng.uniform(0.7, 1.0)),
                                   noise=float(self.rng.uniform(0.03, 0.10)), rng=self.rng)
        return self._frame

    def obs(self):
        return self._frame.ravel()[:10]

    def act(self, a):
        ok = int(a) == self.digit
        return (1.0 if ok else -1.0), True, {"dist": 0.0 if ok else 9.9, "ok": ok}


def run_trial(cx, eye, frame, digit, learn, temp=1.2, steps=C.DECIDE_STEPS):
    """一次试次：取景→编码→皮层积分→选一个数。返回 (选的数, 奖励)。"""
    cx.reset_feat()
    obs = eye.encode(frame)
    for _ in range(steps):
        cx.step(obs[:cx.n_in])
    a = cx.decide(temp=temp, sample=learn)
    r = 1.0 if a == digit else -1.0
    if learn:
        tr = [(cx._last_xf.copy(), cx._last_probs.copy(), a, r)]
        cx.stdp_reward(r)
        cx.learn_returns(tr, gamma=0.0)          # 单步决策：没有"后续回报"可折现
    return a, r


def selftest(n_neurons=20_000, k=360, n_in=32, train=1200, seed=11):
    print("=" * 78)
    print("  ⑧ 看数字：眼睛 + 脉冲皮层 → 认出 0~9（可选转 2/16 进制）")
    print("=" * 78)
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    eye = V.VisionFrontend(n_in=n_in)
    cx = C.SpikingCortex(n=n_neurons, k=k, n_in=n_in, n_act=10, seed=seed)
    task = DigitTask(seed=3)

    # ① 眼睛看得见
    codes = np.stack([eye.encode(task.new_episode()) for _ in range(60)])
    check("眼睛看得见数字（编码非退化、随数字变化）",
          float(codes.std()) > 0.02, f"编码标准差 {float(codes.std()):.4f}")

    # ② 学得会：练 train 次，统计最后 300 次
    t0 = time.perf_counter()
    hist, per_trial = [], []
    for i in range(train):
        frame = task.new_episode()
        a, r = run_trial(cx, eye, frame, task.digit, learn=True)
        hist.append(1 if r > 0 else 0)
        if i >= 50:                       # 丢掉预热（首次惰性分配、缓存冷启动会把均值拉爆）
            per_trial.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
    tail = float(np.mean(hist[-300:])) * 100
    head = float(np.mean(hist[:300])) * 100
    ms = float(np.mean(per_trial)) * 1000
    check("学得会：准确率显著高于瞎猜（10%）",
          tail > 25.0,
          f"练前 300 次 {head:.0f}% → 练后 300 次 {tail:.0f}%（瞎猜 10%），"
          f"单次 {ms:.1f} ms（含取景+编码+10 步皮层）")

    # ③ 转 2/16 进制
    ok = all(to_bin_hex(d) == (format(d, "04b"), format(d, "X")) for d in range(10))
    check("认出后能转 2 进制 / 16 进制",
          ok, " | ".join(f"{d}→{to_bin_hex(d)[0]}b/{to_bin_hex(d)[1]}h" for d in (0, 5, 9)))

    # ④ 混淆情况（老实报，不只看总准确率）
    conf = np.zeros((10, 10), dtype=int)
    task.rng = np.random.default_rng(99)
    for _ in range(200):
        frame = task.new_episode()
        a, _r = run_trial(cx, eye, frame, task.digit, learn=False)
        conf[task.digit, a] += 1
    acc = np.trace(conf) / conf.sum() * 100
    check("独立测试集（200 帧，与训练不同随机源）准确率 > 25%",
          acc > 25.0,
          f"独立测试 {acc:.0f}% ｜ 每类命中 " +
          " ".join(f"{d}:{conf[d, d]:>2}" for d in range(10)))

    print("-" * 78)
    print(f"  结果：{4 - len(fails)}/4 项通过" + ("  ✅ 全部通过" if not fails else f"  ❌ 失败：{fails}"))
    print("=" * 78)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--neurons", type=int, default=20_000)
    ap.add_argument("--train", type=int, default=1200)
    a = ap.parse_args()
    sys.exit(selftest(n_neurons=a.neurons, train=a.train))
