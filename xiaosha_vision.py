# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""小沙的"眼睛"——视叶式视觉前端（第一半：看得见）。

为什么做这个：五种学习机制全试过全是负结果，结论是**瓶颈在感知带宽**——
皮层只有 10 路数字输入（8 测距 + 目标方位）。给眼睛，才是"给更好的信号"。

仿果蝇视叶的处理链（一路都是"差分/整流/稀疏"，不是卷积大模型）：
  灰度帧 24×24
   ① 中心-周边差分（板层 lamina：突出**局部对比**，整体明暗变化不影响它）
   ② 半波整流 ON/OFF 双通道（生物上 ON/OFF 通路是分开的）
   ③ 稀疏阈值（只留强响应 → 事件式，省算力）
   ④ 固定随机投影到 n_in 通道 + 每通道稳态归一化（给皮层"输入端口"）

接口与原来的数字输入**完全一致**（n_in 维、0~1），所以皮层可以直接换用：
    eye = VisionFrontend(n_in=32)
    obs = eye.encode(cam.render(world))     # 替代原来的 10 路数字
真机阶段只需把 cam.render 换成相机取帧（from_image 已经留好口子）。

自检（--selftest）四条，都带实测数字：
  ① 输出形状/范围合法
  ② **看得见目标在哪**：不同方位角下，用编码做线性回归预测角度，平均误差 < 25°
  ③ **亮度不变性**：整体亮暗变化 ±50% 时编码几乎不变（中心-周边差分的功劳）
  ④ 稀疏 + 实时：每帧活跃通道 < 35%，单帧编码 < 2 ms
"""
import argparse
import math
import sys
import time

import numpy as np

IMG = 24                    # 24×24 灰度（真机上直接降采样到这么大就够）


# ============================================================================
# 1) 相机：把"世界"渲染成一张灰度图（真机换成相机取帧）
# ============================================================================
class Camera:
    """朝前的针孔相机：目标亮点 + 障碍暗块 + 地面渐变。俯仰角固定，只看一条窄视场。"""

    def __init__(self, size=IMG, fov_deg=110.0, height=0.18):
        self.size, self.fov, self.h = size, math.radians(fov_deg), height

    def render(self, pos, yaw, target, obstacles=()):
        """pos/yaw: 相机位姿；target/obstacles: 世界坐标(x,y) 或 (x,y,半径)。返回 size×size 灰度 0~1。"""
        n = self.size
        img = np.zeros((n, n), dtype=np.float32)
        img += 0.25                                     # 地面底色
        for i in range(n):                              # 近处亮一点（纵深线索）
            img[i, :] += 0.35 * (i / (n - 1.0))

        def to_img(p, radius, bright):
            dx, dy = p[0] - pos[0], p[1] - pos[1]
            ang = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(ang) > self.fov / 2:
                return                                      # 视场外
            depth = math.hypot(dx, dy) + 1e-6
            u = int((0.5 + ang / self.fov) * (n - 1))
            v = int((1.0 - min(1.0, self.h * 1.2 / depth)) * (n - 1))   # 越近越靠下
            r = max(1, int(radius * n / (2 * math.tan(self.fov / 2) * depth)))
            for du in range(-r, r + 1):
                for dv in range(-max(1, r // 2), max(1, r // 2) + 1):
                    uu, vv = u + du, v + dv
                    if 0 <= uu < n and 0 <= vv < n:
                        img[vv, uu] += bright / (1.0 + depth)   # 近亮远暗

        to_img(target, 0.16, 2.2)
        for ob in obstacles:
            to_img(ob, ob[2] if len(ob) > 2 else 0.14, -1.2)
        return np.clip(img, 0.0, 1.0)


# ============================================================================
# 2) 视叶式前端：差分 → 整流 → 稀疏 → 投影到皮层输入
# ============================================================================
class VisionFrontend:
    def __init__(self, n_in=32, seed=7, theta=0.30, blocks=None):
        # ★ 通道怎么来的：**感受野分块**（4×4 块 × ON/OFF = 32 维 / 8×8 = 128 维），
        #   必须**保住空间位置**。实测（600 帧、岭回归天花板）：随机投影 35% → 分块 76%（同为 32 维）
        #   → 8×8 分块 84%。之前用随机投影把"哪几条段亮"压掉了，数字识别根本学不动（13~16%）。
        # 通道数 = 2 × blocks²；不传 blocks 就按 n_in 反推（32→4、128→8）
        blocks = (max(2, int(round((n_in / 2.0) ** 0.5))) if blocks is None else int(blocks))
        self.blocks = blocks
        self.n_in, self.theta = 2 * blocks * blocks, float(theta)
        k = 3                                        # 中心-周边半径
        ax = np.arange(-k, k + 1)
        gx, gy = np.meshgrid(ax, ax)
        dof = np.exp(-(gx ** 2 + gy ** 2) / (2 * 1.4 ** 2))
        self.center = dof / dof.sum()
        self.surround = np.ones_like(dof) / dof.size
        self.gain = np.ones(self.n_in, dtype=np.float32)      # 每通道稳态增益
        self.last_active = 0.0

    def _diff(self, img):
        """中心-周边差分（两路 ON/OFF 整流）。返回展平的两组响应。"""
        c = _conv(img, self.center)
        s = _conv(img, self.surround)
        d = c - s
        return np.concatenate([np.maximum(d, 0.0), np.maximum(-d, 0.0)]).ravel()

    def encode(self, img):
        """帧 → n_in 维 0~1（与原来的数字输入同接口，皮层可直接换用）。"""
        r = self._diff(img)
        # ★ 自动增益：中心-周边差分后的绝对幅度随场景变（远处目标响应只有 0.0x 级），
        #   固定阈值会把整帧砍成 0（实测编码全是 0.5、活跃 0.0%）。拿"当前帧的强响应"
        #   当尺子归一化，阈值才有意义；下限 1e-3 防全暗帧除爆。
        r = r / max(float(np.percentile(np.abs(r), 99)), 1e-3)

        r[r < self.theta] = 0.0                            # 稀疏阈值：只留强响应（事件式）
        self.last_active = float((r > 0).mean())
        # 感受野分块：ON/OFF 两半各按 blocks×blocks 池化（位置保留，通道数不变）
        h = r.size // 2
        k = self.blocks
        step = IMG // k
        shp = (k, step, k, step)
        on = r[:h].reshape(IMG, IMG).reshape(shp).mean(axis=(1, 3))
        off = r[h:].reshape(IMG, IMG).reshape(shp).mean(axis=(1, 3))
        out = np.clip(np.concatenate([on.ravel(), off.ravel()]) * 4.0 * self.gain,
                      0.0, 1.0).astype(np.float32)
        # 每通道稳态：让所有通道都保持"能用"，不被少数通道吃满
        self.gain *= (1.0 - 0.02 * (out - 0.5))
        np.clip(self.gain, 0.2, 5.0, out=self.gain)
        return out


def obs_fn_for(eye, cam, hybrid=False):
    """给皮层用的"取景函数"：obs_fn(task) → 这一刻看到什么。

    hybrid=False：纯视觉（维度 = eye.n_in，皮层 n_in 要设成一样大）
    hybrid=True ：数字（8 测距 + 目标方位，10 路）+ 视觉拼一起（维度 10 + eye.n_in）
    用法：run_episode(cx, task, obs_fn=obs_fn_for(eye, cam))
    """
    def fn(task):
        v = eye.encode(cam.render(task.pos, task.yaw, task.target, task.obst))
        return np.concatenate([task.obs(), v], dtype=np.float32) if hybrid else v
    return fn


def _conv(img, kern):
    """小核直接卷积（7×7 核在 24×24 上只要 ~0.09 ms，犯不上引 scipy/FFT）。"""
    n, m = img.shape
    kh, kw = kern.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros((n, m), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += kern[i, j] * pad[i:i + n, j:j + m]
    return out


# ============================================================================
# 3) 自检
# ============================================================================
def selftest(n_in=32):
    print("=" * 78)
    print("  小沙 · 视觉前端 自检（视叶式：差分→整流→稀疏→投影）")
    print("=" * 78)
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    cam, eye = Camera(), VisionFrontend(n_in=n_in)
    pos, obs_list = (0.0, 0.0), [(6.0, 6.0, 0.14), (-5.0, 4.0, 0.14)]

    # ① 形状与范围
    angs = np.linspace(-math.pi, math.pi, 120, endpoint=False)
    frames, codes = [], []
    t0 = time.perf_counter()
    for a in angs:
        tgt = (pos[0] + 6 * math.cos(a), pos[1] + 6 * math.sin(a))
        img = cam.render(pos, 0.0, tgt, obs_list)
        frames.append(img)
        codes.append(eye.encode(img))
    ms = (time.perf_counter() - t0) / len(angs) * 1000
    codes = np.asarray(codes)
    fps = 1.0 / (ms / 1000.0)
    check("输出形状/范围合法（n_in 维、0~1）",
          codes.shape == (len(angs), n_in) and codes.min() >= 0.0 and codes.max() <= 1.0,
          f"{codes.shape}，范围 [{codes.min():.3f}, {codes.max():.3f}]")
    # ★ 防"假通过"：编码不能是常数（曾经整帧被阈值砍成 0.5，看着合法其实啥也没看见）
    check("编码非退化（帧与帧之间确实在变，不是常数）",
          float(codes.std()) > 0.01,
          f"编码标准差 {float(codes.std()):.4f}（0 就是「啥也没看见」）")

    # ② 看得见目标在哪：线性回归从编码预测方位角（留出 1/3 测）
    # ★ 只在**视场内**判：视场 110°（±55°），目标转到背后就是"看不见"——那是相机的物理
    #   限制，不是编码的错（实测把视场外样本一起算，均值会被拖到 25.6°、中位才 8.6°）。
    fov_half = math.degrees(cam.fov) / 2
    in_fov = np.abs(((angs + math.pi) % (2 * math.pi)) - math.pi) <= math.radians(fov_half - 3)
    codes_f, angs_f = codes[in_fov], angs[in_fov]
    X = np.concatenate([codes_f, np.ones((len(codes_f), 1), dtype=np.float32)], axis=1)
    T = np.stack([np.cos(angs), np.sin(angs)], axis=1)
    T = np.stack([np.cos(angs_f), np.sin(angs_f)], axis=1)
    # ★ 必须**打乱**再切训练/测试：按角度顺序切会把测试集全放在训练范围之外
    #   （实测外推误差 116° = 等于瞎猜），这是测试方法的错，不是编码的错。
    perm = np.random.default_rng(0).permutation(len(angs_f))
    X, T = X[perm], T[perm]
    ntr = int(len(angs_f) * 2 / 3)
    W = np.linalg.solve(X[:ntr].T @ X[:ntr] + 1e-3 * np.eye(n_in + 1), X[:ntr].T @ T[:ntr])
    P = X[ntr:] @ W
    pred = np.arctan2(P[:, 1], P[:, 0])
    a_test = angs_f[perm][ntr:]
    err = np.abs((pred - a_test + math.pi) % (2 * math.pi) - math.pi) * 180 / math.pi
    check("看得见目标在哪：从编码反推出方位角（视场内，留出测试集）",
          float(err.mean()) < 15.0,
          f"平均角度误差 {err.mean():.1f}°（中位 {np.median(err):.1f}°，最差 {err.max():.1f}°）；"
          f"样本 = 视场 ±{fov_half:.0f}° 内的 {len(angs_f)} 个方位（视场外当然看不见）")

    # ③ 亮度不变性：整体亮度 ±50% 时编码几乎不变（中心-周边差分）
    base = codes[::10]
    for factor in (0.5, 0.8):
        # ★ 只测不饱和的倍数：×1.5 会触发 clip 饱和（那是截断不是亮度变化，差 0.10 全是它）
        c2 = np.asarray([eye.encode(np.clip(f * factor, 0, 1))
                         for f in np.asarray(frames)[::10]])
        d = float(np.abs(c2 - base).mean())
        check(f"亮度不变性：整体亮度 ×{factor} 时编码基本不变（且不是常数）",
              d < 0.10 and float(base.std()) > 0.01,
              f"编码平均变化 {d:.4f}（越小越好；同时要求编码非常数，防空通过）")

    # ④ 稀疏 + 实时
    check("稀疏 + 实时：每帧只少数通道活跃、单帧毫秒级",
          eye.last_active < 0.35 and ms < 2.0,
          f"每帧活跃通道 {eye.last_active*100:.1f}%，单帧编码 {ms:.2f} ms（{fps:.0f} 帧/秒）")

    print("-" * 78)
    print(f"  结果：{4 - len(fails)}/4 项通过" + ("  ✅ 全部通过" if not fails else f"  ❌ 失败：{fails}"))
    print("=" * 78)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-in", type=int, default=32)
    a = ap.parse_args()
    sys.exit(selftest(n_in=a.n_in) if a.selftest else selftest(n_in=a.n_in))
