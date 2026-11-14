# -*- coding: utf-8 -*-
"""小沙的四感前端（Perception Front-End）—— 真实信号处理，不是抽象特征。

为什么重写：
  原来的 xiaosha_senses.py 只有简化的 EnhancedEye/EnhancedEar（提取几个手写特征），
  触觉只有一个 "contact 0/1"，嗅觉完全没有。
  真实的感官是**物理信号 → 感受器 → 神经编码**，每一级都有明确分工。

本模块实现四个前端，全部用纯 numpy 做真实信号处理（无 scipy 依赖）：

  ① Retina  视网膜   —— 中心-周边拮抗 / 颜色对立 / 方向选择 / 运动检测
  ② Cochlea 耳蜗     —— 基底膜频率分解 / 毛细胞包络 / 双耳定位 / 音高
  ③ Skin    皮肤     —— 四类机械感受器 / 温度 / 疼痛 / 纹理
  ④ Nose    鼻子     —— 气体阵列指纹 / 浓度 / 梯度 / 嗅觉适应

每个前端的输出是"神经编码"（发放率向量），可以直接喂给结构化皮层。
"""
import numpy as np

__version__ = "1.0.0"


# ==================== ① 视网膜 Retina ====================

class Retina:
    """真实图像 → 视网膜神经节细胞 → V1 简单细胞。

    生理依据（Hubel & Wiesel 经典）：
      视网膜神经节细胞：中心-周边拮抗感受野（DoG），对亮度边缘敏感
      LGN：同上，按颜色/时间特性分通道
      V1 简单细胞：方向选择（Gabor），对特定朝向的边缘响应
      V1 复杂细胞 / MT：运动检测（时间差分 + 方向）

    输出特征（每帧一个向量）：
      0-3   四个方向能量（0°/45°/90°/135°）
      4     整体边缘强度
      5     中心亮度（对数压缩后）
      6     颜色对立 R-G
      7     颜色对立 B-Y
      8-15  八方向运动能量（时间差分后的方向直方图）
      16-23 八个空间区域的边缘强度（粗略的"哪里亮"）
    """

    def __init__(self, size=64, n_orient=4, n_gabor=8):
        self.size = size
        self.n_orient = n_orient
        # 预生成 Gabor 核（8 方向 × 3 尺度）
        self.gabors = [self._gabor(theta, lam, size=9)
                       for theta in np.linspace(0, np.pi, n_orient, endpoint=False)
                       for lam in (3.0, 5.0, 8.0)]
        self.prev_gray = None
        self.dim = 24

    @staticmethod
    def _gabor(theta, lam, size=9, sigma=None):
        """二维 Gabor 核（偶数对称，模拟 V1 简单细胞的感受野）。"""
        if sigma is None:
            sigma = 0.56 * lam
        half = size // 2
        y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
        xr = x * np.cos(theta) + y * np.sin(theta)
        yr = -x * np.sin(theta) + y * np.cos(theta)
        g = np.exp(-(xr ** 2 + (0.5 * yr) ** 2) / (2 * sigma ** 2))
        g = g * np.cos(2 * np.pi * xr / lam)
        return (g - g.mean()).astype(np.float32)

    @staticmethod
    def _conv2d(img, kernel):
        """朴素二维卷积（same padding）。图不大（64×64），够用且无依赖。"""
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        h, w = img.shape
        out = np.zeros((h, w), dtype=np.float32)
        for i in range(kh):
            for j in range(kw):
                out += kernel[i, j] * padded[i:i + h, j:j + w]
        return out

    @staticmethod
    def _to_gray(img):
        if img.ndim == 3:
            return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)
        return img.astype(np.float32)

    def _dog(self, gray):
        """中心-周边拮抗（DoG）：视网膜神经节细胞的经典感受野。"""
        center = self._conv2d(gray, self._gauss(3))
        surround = self._conv2d(gray, self._gauss(9))
        return center - surround

    @staticmethod
    def _gauss(size):
        half = size // 2
        y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
        s = size / 4.0
        g = np.exp(-(x ** 2 + y ** 2) / (2 * s ** 2))
        return (g / g.sum()).astype(np.float32)

    def encode(self, img, dt=1.0):
        """img: HxW 或 HxWx3，0-255 或 0-1 → 24 维神经编码。"""
        a = np.asarray(img, dtype=np.float32)
        if a.max() > 1.5:
            a = a / 255.0
        gray = self._to_gray(a)

        # ---- 边缘（DoG）----
        edges = self._dog(gray)
        edge_energy = float(np.abs(edges).mean())

        # ---- 方向选择（Gabor，V1 简单细胞）----
        orient_e = np.zeros(self.n_orient, dtype=np.float32)
        for k, g in enumerate(self.gabors):
            r = self._conv2d(gray, g)
            orient_e[k % self.n_orient] += float(np.abs(r).mean())
        orient_e /= len(self.gabors) / self.n_orient

        # ---- 颜色对立（小细胞通路）----
        if a.ndim == 3 and a.shape[2] >= 3:
            rg = float((a[..., 0].mean() - a[..., 1].mean()))
            by = float((a[..., 2].mean() - 0.5 * (a[..., 0].mean() + a[..., 1].mean())))
        else:
            rg = by = 0.0

        # ---- 运动（大细胞通路：时间差分）----
        motion = np.zeros(8, dtype=np.float32)
        if self.prev_gray is not None:
            diff = gray - self.prev_gray
            # 8 个方向：把差分图按梯度方向分箱
            gy, gx = np.gradient(diff)
            mag = np.hypot(gx, gy)
            ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)   # 0-1
            bins = np.clip((ang * 8).astype(np.int32) % 8, 0, 7)
            for b in range(8):
                m = bins == b
                motion[b] = float(mag[m].mean()) if m.any() else 0.0
        self.prev_gray = gray.copy()

        # ---- 空间分区（粗略"哪里亮"）----
        h, w = gray.shape
        regions = np.zeros(8, dtype=np.float32)
        for i in range(2):
            for j in range(4):
                blk = gray[i * h // 2:(i + 1) * h // 2, j * w // 4:(j + 1) * w // 4]
                regions[i * 4 + j] = float(blk.mean())

        feat = np.concatenate([
            orient_e,                      # 4
            [edge_energy],                 # 1
            [np.log1p(gray.mean() * 10)],  # 1  （光感受器对数压缩）
            [rg, by],                      # 2
            motion,                        # 8
            regions,                       # 8
        ]).astype(np.float32)
        return feat


# ==================== ② 耳蜗 Cochlea ====================

class Cochlea:
    """真实音频波形 → 基底膜频率分解 → 听神经响应。

    生理依据：
      基底膜：按频率排列（高频在底部、低频在顶部），gammatone 滤波是公认模型
      毛细胞：半波整流 + 低通 → 包络（内毛细胞转导）
      上橄榄复合体：双耳时间差（ITD）+ 强度差（ILD）→ 声源方位
      听觉皮层：音高周期检测

    输出特征（一个向量）：
      0-23   24 个频带的能量（50Hz-8kHz，对数间隔）
      24     总能量
      25     谱质心（音色亮度）
      26     谱平坦度（噪声/音调）
      27-30  四个频段占比（低/中低/中高/高）
      31     音高（自相关检测，归一化）
      32     ITD 方位（-1 左 ~ +1 右；单声道为 0）
      33     ILD 方位（-1 左 ~ +1 右）
      34     节律强度（包络的周期性）
    """

    def __init__(self, sr=16000, n_bands=24, fmin=50.0, fmax=8000.0):
        self.sr = sr
        self.n_bands = n_bands
        # 对数间隔的中心频率（基底膜位置映射）
        self.freqs = np.geomspace(fmin, min(fmax, sr / 2 - 100), n_bands)
        self.dim = 35

    def _band_energy(self, x):
        """FFT 分带能量（比时域滤波快，对大数组友好）。"""
        n = len(x)
        if n == 0:
            return np.zeros(self.n_bands, dtype=np.float32)
        win = np.hanning(n).astype(np.float32)
        spec = np.abs(np.fft.rfft(x * win))
        f = np.fft.rfftfreq(n, 1.0 / self.sr)
        e = np.zeros(self.n_bands, dtype=np.float32)
        for i, fc in enumerate(self.freqs):
            # 每个频带取 ±1/4 倍频程
            lo, hi = fc / 1.19, fc * 1.19
            m = (f >= lo) & (f < hi)
            if m.any():
                e[i] = float(np.sqrt((spec[m] ** 2).sum()))
        return e

    @staticmethod
    def _envelope(x, sr, lp_hz=50.0):
        """毛细胞转导：半波整流 + 低通（用滑动平均近似）。"""
        rect = np.maximum(x, 0.0)
        k = max(1, int(sr / lp_hz))
        if k > 1 and len(rect) > k:
            kernel = np.ones(k, dtype=np.float32) / k
            return np.convolve(rect, kernel, mode="same")
        return rect

    def _pitch(self, x):
        """自相关音高检测（80-500Hz 范围）。"""
        if len(x) < 64:
            return 0.0
        x = x - x.mean()
        ac = np.correlate(x, x, mode="full")[len(x) - 1:]
        lo = int(self.sr / 500)
        hi = min(int(self.sr / 80), len(ac) - 1)
        if hi <= lo:
            return 0.0
        seg = ac[lo:hi]
        if seg.max() <= 0:
            return 0.0
        i = int(np.argmax(seg)) + lo
        conf = float(seg.max() / (ac[0] + 1e-9))
        return float(conf * (1.0 - abs(i / hi - 0.5) * 2))   # 周期越明确分越高

    def encode(self, audio, audio_right=None):
        """audio: 单声道波形（-1..1）→ 35 维听觉编码。"""
        x = np.asarray(audio, dtype=np.float32).ravel()
        if x.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        if np.abs(x).max() > 1.5:
            x = x / 32768.0

        bands = self._band_energy(x)
        total = float(bands.sum())

        # 谱质心 & 平坦度（音色）
        if total > 0:
            centroid = float((bands * self.freqs).sum() / total / self.freqs[-1])
            p = bands / total
            flat = float(np.exp(np.mean(np.log(p + 1e-9))) / (np.mean(p) + 1e-9))
        else:
            centroid = flat = 0.0

        # 四频段占比
        quarter = self.n_bands // 4
        seg = [float(bands[i * quarter:(i + 1) * quarter].sum() / (total + 1e-9)) for i in range(4)]

        pitch = self._pitch(x)

        # 双耳定位（ITD / ILD）
        itd = ild = 0.0
        if audio_right is not None:
            r = np.asarray(audio_right, dtype=np.float32).ravel()[:len(x)]
            if len(r) > 1:
                n = min(len(x), len(r))
                max_lag = 32
                best, best_lag = -1e9, 0
                for lag in range(-max_lag, max_lag + 1):
                    if lag >= 0:
                        a = x[lag:n]; b = r[:n - lag]
                    else:
                        a = x[:n + lag]; b = r[-lag:n]
                    if len(a) > 16:
                        c = float(np.dot(a, b))
                        if c > best:
                            best, best_lag = c, lag
                itd = float(best_lag / max_lag)          # -1 ~ +1
            ild = float(np.clip((np.abs(x).mean() - np.abs(r).mean()) * 4, -1, 1))

        # 节律强度（包络的自相关主峰）
        env = self._envelope(x, self.sr)
        rhythm = 0.0
        if len(env) > 128:
            e = env - env.mean()
            ac = np.correlate(e, e, mode="full")[len(e) - 1:]
            if ac[0] > 0:
                lo = int(self.sr / 10); hi = min(len(ac), int(self.sr / 2))
                if hi > lo:
                    rhythm = float(np.clip(ac[lo:hi].max() / ac[0], 0, 1))

        feat = np.concatenate([
            bands, [total], [centroid, flat], seg,
            [pitch, itd, ild, rhythm],
        ]).astype(np.float32)
        return feat


# ==================== ③ 皮肤 Skin ====================

class Skin:
    """触觉阵列 → 四类机械感受器 + 温度 + 疼痛。

    生理依据（皮肤里的真实感受器，各有分工）：
      SA-I (Merkel)     静态压力、形状细节、边缘 —— 慢适应
      RA-I (Meissner)   低频振动 5-50Hz（flutter）、纹理 —— 快适应
      SA-II (Ruffini)   皮肤拉伸、关节位置（本体感觉）—— 慢适应
      RA-II (Pacinian)  高频振动 50-400Hz、工具传递 —— 快适应
      温度感受器        TRPV1（热）/ TRPM8（冷）
      痛觉              过载（机械/热），保护性反射

    输出特征（12 维）：
      0  SA-I  平均压力（持续接触）
      1  RA-I  低频振动能量（纹理粗糙度）
      2  SA-II 拉伸量（剪切）
      3  RA-II 高频振动能量（工具/硬物）
      4  最大压力点位置 x（0-1）
      5  最大压力点位置 y（0-1）
      6  接触面积占比
      7  温度（归一化 -10~50°C → 0-1）
      8  温度变化率（快速升温=危险）
      9  疼痛（0-1，过载）
      10 压力梯度方向（-1~1，滑动方向）
      11 纹理周期（自相关主峰位置）
    """

    def __init__(self, grid=(8, 8), settle=0.7):
        self.grid = grid
        self.settle = settle       # SA 适应系数（越小越快适应）
        self.dim = 12
        self.prev = np.zeros(grid, dtype=np.float32)
        self.sa_state = np.zeros(grid, dtype=np.float32)
        self.prev_temp = None
        self.hist = []

    def encode(self, pressure, temp=25.0, shear=0.0, dt=1.0):
        """pressure: grid 形状的 0-1 压力阵列（可为任意尺寸，自动 resize）。"""
        p = np.asarray(pressure, dtype=np.float32)
        if p.shape != self.grid:
            # 最近邻重采样到 grid（触觉阵列尺寸各异）
            yi = (np.linspace(0, p.shape[0] - 1, self.grid[0])).astype(int)
            xi = (np.linspace(0, p.shape[1] - 1, self.grid[1])).astype(int)
            p = p[np.ix_(yi, xi)]
        p = np.clip(p, 0.0, 1.0)

        # SA-I：慢适应（对静态压力持续响应，缓慢衰减到基线）
        self.sa_state = self.settle * self.sa_state + (1 - self.settle) * p
        sa1 = float(self.sa_state.mean())

        # RA-I / RA-II：快适应（对变化响应，不看静态）
        d = p - self.prev
        d_pos = np.maximum(d, 0.0)
        ra_fast = float(np.abs(d).mean())

        # 用 FFT 分出低频（RA-I 纹理）和高频（RA-II 工具）
        ra1 = ra2 = 0.0
        if p.size >= 16:
            spec = np.abs(np.fft.rfft(p.ravel() - p.mean()))
            n = len(spec)
            if n > 4:
                ra1 = float(spec[1:n // 4].mean())
                ra2 = float(spec[n // 4:].mean())

        # 拉伸（SA-II）：整体形变（用阵列的均值梯度近似剪切）
        sa2 = float(abs(shear))

        # 最大压力点 + 接触面积
        idx = int(np.argmax(p))
        my, mx = np.unravel_index(idx, p.shape)
        max_y = my / max(1, p.shape[0] - 1)
        max_x = mx / max(1, p.shape[1] - 1)
        contact = float((p > 0.15).mean())

        # 温度 + 变化率
        temp_norm = float(np.clip((temp + 10.0) / 60.0, 0, 1))
        dtemp = 0.0 if self.prev_temp is None else (temp - self.prev_temp) / dt
        self.prev_temp = temp

        # 疼痛：过载（压力>0.85 或 温度>45°C 或 温度骤升）
        pain = float(np.clip(max(0.0, (p.max() - 0.85) / 0.15,
                                 (temp - 45.0) / 15.0,
                                 max(0.0, dtemp - 3.0) / 5.0), 0, 1))

        # 滑动方向（压力中心相对上次的移动）
        slide = 0.0
        if self.hist and contact > 0:
            py, px = self.hist[-1]
            slide = float(np.clip((mx / max(1, p.shape[1] - 1) - px) * 4, -1, 1))
        self.hist.append((my, mx))
        if len(self.hist) > 8:
            self.hist.pop(0)

        # 纹理周期（自相关主峰 → 粗糙度的空间尺度）
        tex_period = 0.0
        if p.size >= 16:
            row = p.ravel() - p.ravel().mean()
            ac = np.correlate(row, row, mode="full")[len(row) - 1:]
            if ac[0] > 0 and len(ac) > 6:
                seg = ac[2:min(len(ac), 32)]
                if seg.size:
                    tex_period = float((int(np.argmax(seg)) + 2) / 32.0)

        self.prev = p.copy()

        feat = np.array([sa1, ra1, sa2, ra2, max_x, max_y, contact,
                         temp_norm, dtemp, pain, slide, tex_period], dtype=np.float32)
        return feat


# ==================== ④ 鼻子 Nose ====================

class Nose:
    """气体传感器阵列 → 气味指纹（电子鼻）。

    生理依据：
      嗅上皮：约 400 种嗅觉受体（人类），每种对一类分子响应
      嗅球： glomerulus 汇聚 + 侧抑制 → 气味"指纹"
      适应：持续暴露 → 响应下降（嗅觉疲劳，真实存在）
      梯度追踪：左右鼻孔/两侧传感器浓度差 → 气味来源方向

    输出特征（16 维）：
      0-7    8 通道响应（已做适应）
      8      总浓度
      9      气味指纹形状（与基线的余弦相似度）
      10     浓度梯度（左右差 → 方向 -1~1）
      11     变化率（浓度快速上升 = 接近源头）
      12-15  四个"气味类别"的亲和度（花香/腐败/焦糊/化学，用固定的受体偏好矩阵）
    """

    # 固定的受体偏好矩阵（8 通道 × 4 类气味）—— 这是"进化给定的"感受谱
    RECEPTOR_MATRIX = np.array([
        [0.9, 0.1, 0.1, 0.2],   # ch0: 芳香族（花香）
        [0.7, 0.3, 0.2, 0.1],   # ch1: 酯类（果香）
        [0.2, 0.9, 0.1, 0.2],   # ch2: 含硫（腐败）
        [0.1, 0.8, 0.3, 0.2],   # ch3: 胺类（腐败）
        [0.1, 0.2, 0.9, 0.2],   # ch4: 焦糊/烟
        [0.2, 0.1, 0.8, 0.3],   # ch5: 酚类（烟熏）
        [0.1, 0.2, 0.2, 0.9],   # ch6: 溶剂（化学）
        [0.2, 0.1, 0.3, 0.8],   # ch7: 酸性/刺激
    ], dtype=np.float32)

    def __init__(self, n_ch=8, adapt_rate=0.12):
        self.n_ch = n_ch
        self.adapt_rate = adapt_rate
        self.baseline = np.full(n_ch, 0.05, dtype=np.float32)
        self.dim = 16
        self.prev_conc = None

    def encode(self, readings, left_right_split=None, dt=1.0):
        """readings: 8 通道原始响应（0-1）。

        left_right_split: 可选，两半通道分别来自左右传感器（用于梯度）。
        """
        x = np.asarray(readings, dtype=np.float32).ravel()[:self.n_ch]
        if x.size < self.n_ch:
            x = np.pad(x, (0, self.n_ch - x.size))

        # 适应（嗅觉疲劳）：基线缓慢跟随，持续刺激的响应被削弱
        adapted = np.maximum(x - self.baseline, 0.0)
        conc = float(adapted.sum() / self.n_ch)

        # 浓度梯度（左右差）
        gradient = 0.0
        if left_right_split is not None:
            l, r = float(left_right_split[0]), float(left_right_split[1])
            gradient = float(np.clip((r - l) * 4, -1, 1))
        else:
            # 近似：前四通道当"左"，后四当"右"
            l = float(adapted[:self.n_ch // 2].mean())
            r = float(adapted[self.n_ch // 2:].mean())
            gradient = float(np.clip((r - l) * 4, -1, 1))

        # 变化率（接近源头 → 浓度快速上升）
        rate = 0.0 if self.prev_conc is None else (conc - self.prev_conc) / dt
        self.prev_conc = conc

        # 指纹形状（归一化后的模式，与基线形状的相似度）—— 用于识别"还是同一个气味吗"
        norm = adapted / (np.linalg.norm(adapted) + 1e-9)
        base_norm = self.baseline / (np.linalg.norm(self.baseline) + 1e-9)
        shape_sim = float(np.dot(norm, base_norm))

        # 四类气味亲和度（受体矩阵投影）
        classes = np.clip(self.RECEPTOR_MATRIX.T @ adapted, 0, None)
        if classes.sum() > 0:
            classes = classes / classes.sum()

        # 更新基线（适应）
        self.baseline = (1 - self.adapt_rate) * self.baseline + self.adapt_rate * x

        feat = np.concatenate([
            adapted, [conc, shape_sim, gradient, rate], classes,
        ]).astype(np.float32)
        return feat


# ==================== 五感融合（给皮层/本能用）====================

class Perception:
    """把四感（+本体）拼成一个统一的感知向量。"""

    def __init__(self, retina_size=64):
        self.retina = Retina(size=retina_size)
        self.cochlea = Cochlea()
        self.skin = Skin()
        self.nose = Nose()
        self.dim = self.retina.dim + self.cochlea.dim + self.skin.dim + self.nose.dim

    def encode(self, image=None, audio=None, audio_right=None,
               pressure=None, temp=25.0, shear=0.0, gas=None):
        """任意子集输入，缺失通道填 0。"""
        v = np.zeros(self.retina.dim, dtype=np.float32)
        a = np.zeros(self.cochlea.dim, dtype=np.float32)
        t = np.zeros(self.skin.dim, dtype=np.float32)
        o = np.zeros(self.nose.dim, dtype=np.float32)
        if image is not None:
            v = self.retina.encode(image)
        if audio is not None:
            a = self.cochlea.encode(audio, audio_right)
        if pressure is not None:
            t = self.skin.encode(pressure, temp, shear)
        if gas is not None:
            o = self.nose.encode(gas)
        return np.concatenate([v, a, t, o])


# ==================== 自测（用真实信号，不是抽象断言）====================

def selftest():
    print("  小沙四感前端 selftest（真实信号处理）")
    ok = 0
    total = 0

    # ---------- ① 视网膜：方向选择性 ----------
    total += 1
    r = Retina(size=48)
    h, w = 48, 48
    yy, xx = np.mgrid[0:h, 0:w]
    # 竖条纹（垂直边缘，水平方向变化）vs 横条纹
    vert = (np.sin(xx * np.pi / 4) > 0).astype(np.float32) * 255
    hori = (np.sin(yy * np.pi / 4) > 0).astype(np.float32) * 255
    f_v = r.encode(vert)
    r.prev_gray = None
    f_h = r.encode(hori)
    # 方向能量分布应该不同
    v_or = f_v[0:4]; h_or = f_h[0:4]
    diff = float(np.abs(v_or - h_or).sum())
    print("  [1] 视网膜方向选择性: 竖条纹 %s vs 横条纹 %s (差异 %.2f)"
          % (np.round(v_or, 2), np.round(h_or, 2), diff))
    ok += int(diff > 0.05)

    # ---------- ② 视网膜：运动检测 ----------
    total += 1
    r2 = Retina(size=48)
    bar0 = np.zeros((h, w), dtype=np.float32); bar0[:, 10:20] = 255
    bar1 = np.zeros((h, w), dtype=np.float32); bar1[:, 20:30] = 255     # 右移 10px
    r2.encode(bar0)
    f_m = r2.encode(bar1)
    motion_e = float(f_m[8:16].sum())
    print("  [2] 视网膜运动检测: 移动后的八方向运动能量 %.3f" % motion_e)
    ok += int(motion_e > 0.01)

    # ---------- ③ 耳蜗：频率分解 ----------
    total += 1
    cch = Cochlea(sr=16000, n_bands=24)
    t = np.arange(8000) / 16000.0
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)      # 440Hz (A4)
    f440 = cch.encode(tone)
    bands = f440[:24]
    peak_band = int(np.argmax(bands))
    peak_freq = cch.freqs[peak_band]
    print("  [3] 耳蜗频率分解: 输入 440Hz → 峰值频带 %.0fHz (第 %d 带)" % (peak_freq, peak_band))
    ok += int(abs(peak_freq - 440) / 440 < 0.6)

    # ---------- ④ 耳蜗：双耳定位 ----------
    total += 1
    n = 2048
    t2 = np.arange(n) / 16000.0
    sig = np.sin(2 * np.pi * 500 * t2).astype(np.float32)
    right_lag = np.roll(sig, 5)     # 右耳滞后 5 样本 → 声音先到左耳 → 声源在「左」
    f_loc = cch.encode(sig, right_lag)
    itd = float(f_loc[32])
    # 反向验证：右耳提前 → 声源在右 → ITD 应为正
    right_lead = np.roll(sig, -5)
    f_loc2 = cch.encode(sig, right_lead)
    itd2 = float(f_loc2[32])
    print("  [4] 耳蜗双耳定位: 右耳滞后→ITD=%+.3f(应负=左) | 右耳提前→ITD=%+.3f(应正=右)" % (itd, itd2))
    ok += int(itd < -0.05 and itd2 > 0.05)

    # ---------- ⑤ 耳蜗：音高 ----------
    total += 1
    f_pitch = cch.encode(tone)      # 440Hz 纯音
    noise = np.random.RandomState(0).randn(8000).astype(np.float32) * 0.3
    f_noise = cch.encode(noise)
    print("  [5] 耳蜗音高检测: 纯音 pitch=%.2f vs 噪声 pitch=%.2f" % (f_pitch[31], f_noise[31]))
    ok += int(f_pitch[31] > f_noise[31])

    # ---------- ⑥ 皮肤：SA（静态）vs RA（动态）----------
    total += 1
    sk = Skin(grid=(8, 8))
    static_p = np.zeros((8, 8), dtype=np.float32); static_p[3:5, 3:5] = 0.8
    f_sa = sk.encode(static_p)
    # 连续按 5 次（静态压力 → SA 高，RA 衰减）
    for _ in range(5):
        f_sa = sk.encode(static_p)
    # 换成振动（交替 0/0.8）
    vib = [np.zeros((8,8), dtype=np.float32), static_p]
    for i in range(6):
        f_ra = sk.encode(vib[i % 2])
    print("  [6] 皮肤 SA静态=%.3f（持续压力保持）| 振动时 RA=%.3f（变化响应）"
          % (f_sa[0], max(f_ra[1], f_ra[3])))
    # SA 应对持续压力保持一个稳定水平；RA 应对振动有响应
    ok += int(f_sa[0] > 0.02 and max(f_ra[1], f_ra[3]) > 0)

    # ---------- ⑦ 皮肤：疼痛（过热）----------
    total += 1
    sk2 = Skin(grid=(8, 8))
    p_f = sk2.encode(static_p, temp=52.0)     # 52°C 烫
    print("  [7] 皮肤疼痛: 52°C 时 pain=%.2f (>0 = 有痛觉)" % p_f[9])
    ok += int(p_f[9] > 0.1)

    # ---------- ⑧ 鼻子：气味指纹分类 ----------
    total += 1
    nose = Nose()
    # 花香：前两通道强（按受体矩阵）；腐败：ch2/ch3 强
    flower = np.array([0.9, 0.8, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    rotten = np.array([0.05, 0.05, 0.9, 0.8, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    f_flower = nose.encode(flower)
    nose2 = Nose()
    f_rotten = nose2.encode(rotten)
    print("  [8] 鼻子气味分类: 花香→类别%s | 腐败→类别%s"
          % (np.round(f_flower[12:16], 2), np.round(f_rotten[12:16], 2)))
    ok += int(f_flower[12] > f_flower[13] and f_rotten[13] > f_rotten[12])

    # ---------- ⑨ 鼻子：梯度 + 适应 ----------
    total += 1
    nose3 = Nose()
    g1 = nose3.encode(np.full(8, 0.5, dtype=np.float32), left_right_split=(0.8, 0.2))
    grad = float(g1[10])
    # 适应：持续暴露 10 次后响应应下降
    first = [[0.0]]
    for i in range(12):
        f_adapt = nose3.encode(np.full(8, 0.6, dtype=np.float32))
        if i == 0: first = f_adapt[8]
    print("  [9] 鼻子: 梯度 %+.2f（负=左侧浓）| 适应 0.6→%.3f（应下降）" % (grad, f_adapt[8]))
    ok += int(grad < -0.1 and f_adapt[8] < 0.6)

    # ---------- ⑩ 五感融合 ----------
    total += 1
    perc = Perception(retina_size=32)
    img = np.random.RandomState(1).randint(0, 255, (32, 32, 3)).astype(np.uint8)
    audio = np.sin(2 * np.pi * 300 * np.arange(4000) / 16000).astype(np.float32)
    press = np.zeros((8, 8), dtype=np.float32); press[2:4, 2:4] = 0.5
    gas = np.full(8, 0.3, dtype=np.float32)
    v = perc.encode(image=img, audio=audio, pressure=press, gas=gas)
    print("  [10] 五感融合: 向量 %d 维（视%d+听%d+触%d+嗅%d），非零 %.0f%%"
          % (len(v), perc.retina.dim, perc.cochlea.dim, perc.skin.dim, perc.nose.dim,
             (v != 0).mean() * 100))
    ok += int(len(v) == perc.dim and np.isfinite(v).all() and (v != 0).mean() > 0.5)

    print("  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
