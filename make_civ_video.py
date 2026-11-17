# -*- coding: utf-8 -*-
"""60 秒视频：真实截图 + 真实数字，不生成任何虚构画面。

结构（1920x1080, 30fps, 1800 帧）：
  0-10s   钩子：我 12 岁，做了一个比果蝇大 2 倍的脑
 10-20s   规模：288,964 神经元 / 56,265,518 突触
 20-33s   它自己长出了宗教（等距世界推拉）
 33-44s   数据：19 座祭坛 / 299 个信徒 / 信仰 0.86
 44-54s   进化：50 代活力 +79.8%
 54-60s   结尾：仓库地址

风格：暗底、克制、无 emoji、青蓝点缀。所有画面素材都是真跑出来的截图。
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1920, 1080, 30
HERE = os.path.dirname(os.path.abspath(__file__))
# 素材查找顺序：仓库根目录（clone 下来的人用这个）→ 我本地平台的 public/
# 原来是硬编码 D:\Apps\MyModel\chat\public，别人 clone 后 selftest 直接失败，
# 而 README 写着"你可以自己跑一遍"——那句话就成了假的。
_PUB_CANDIDATES = [HERE, r"D:\Apps\MyModel\chat\public"]
PUB = next((p for p in _PUB_CANDIDATES
            if os.path.exists(os.path.join(p, "civ-world-iso.png"))), HERE)
OUT = os.path.join(HERE, "NeuroShaqma-civ.mp4")
TMP = os.path.join(os.environ.get("TEMP", "."), "civ_frames")

CYAN = (63, 201, 211)
BLUE = (74, 158, 218)
GOLD = (224, 172, 82)
DIM = (120, 136, 154)
WHITE = (232, 238, 246)
BG = (7, 10, 15)

FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
FONT_REG = r"C:\Windows\Fonts\msyh.ttc"


def font(size, bold=True):
    p = FONT_BOLD if bold else FONT_REG
    try:
        return ImageFont.truetype(p, size)
    except Exception:
        return ImageFont.load_default()


def ease(t):
    """smoothstep"""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def fade_in_out(local_t, dur, fin=0.4, fout=0.4):
    """局部时间 → 透明度"""
    if local_t < fin:
        return ease(local_t / fin)
    if local_t > dur - fout:
        return ease((dur - local_t) / fout)
    return 1.0


def base_canvas():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # 极淡的暗角
    return img


def draw_center_text(img, lines, y_start=None, alpha=1.0):
    """居中多行文字。lines = [(text, size, color, dy), ...]"""
    d = ImageDraw.Draw(img)
    total_h = 0
    metrics = []
    for text, size, color, gap in lines:
        f = font(size)
        bbox = d.textbbox((0, 0), text, font=f)
        metrics.append((text, f, color, gap, bbox[2] - bbox[0], bbox[3] - bbox[1]))
        total_h += (bbox[3] - bbox[1]) + gap
    y = (H - total_h) // 2 if y_start is None else y_start
    for text, f, color, gap, tw, th in metrics:
        c = tuple(int(v * alpha + b * (1 - alpha)) for v, b in zip(color, BG))
        d.text(((W - tw) // 2, y), text, font=f, fill=c)
        y += th + gap
    return img


def ken_burns(src_path, dur_frames, idx, zoom_from=1.0, zoom_to=1.10, pan=(0, 0)):
    """对图片做缓慢推拉（Ken Burns），返回一帧。"""
    src = Image.open(src_path).convert("RGB")
    t = idx / max(1, dur_frames - 1)
    e = ease(t)
    z = zoom_from + (zoom_to - zoom_from) * e
    # 缩放到覆盖画布
    sw, sh = src.size
    scale = max(W / sw, H / sh) * z
    nw, nh = int(sw * scale), int(sh * scale)
    im = src.resize((nw, nh), Image.LANCZOS)
    # 居中 + 平移
    ox = (nw - W) // 2 + int(pan[0] * e)
    oy = (nh - H) // 2 + int(pan[1] * e)
    ox = max(0, min(nw - W, ox))
    oy = max(0, min(nh - H, oy))
    return im.crop((ox, oy, ox + W, oy + H))


def vignette(img, strength=0.55):
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse([-W * 0.25, -H * 0.25, W * 1.25, H * 1.25], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(160))
    dark = Image.new("RGB", (W, H), BG)
    return Image.composite(img, Image.blend(img, dark, strength), mask)


def overlay_gradient(img, color=(0, 0, 0), top=0.72, bottom=0.10):
    """底部加深，方便压字。"""
    g = Image.new("L", (1, H))
    for y in range(H):
        v = top + (bottom - top) * (y / H)
        g.putpixel((0, y), int(v * 255))
    g = g.resize((W, H))
    dark = Image.new("RGB", (W, H), color)
    return Image.composite(dark, img, g)


# ==================== 场景 ====================
def scene_hook(idx, n):
    # 用世界图做极暗背景（纯黑 + 三行字会像 PPT，实测 96% 单色）
    bg = os.path.join(PUB, "civ-world-iso.png")
    if os.path.exists(bg):
        img = ken_burns(bg, n, idx, 1.06, 1.14)
        img = Image.blend(base_canvas(), img, 0.20)
        img = vignette(img, 0.55)
    else:
        img = base_canvas()
    a = fade_in_out(idx / FPS, n / FPS, 0.5, 0.5)
    lines = [
        ("我 12 岁", 132, WHITE, 24),
        ("做了一个比果蝇大 2 倍的脑", 64, CYAN, 60),
        ("它后来自己长出了宗教", 44, DIM, 0),
    ]
    img = draw_center_text(img, lines, alpha=a)
    return img


def scene_scale(idx, n):
    hero = os.path.join(PUB, "hero-288k.png")
    if os.path.exists(hero):
        img = ken_burns(hero, n, idx, 1.02, 1.12)
        img = overlay_gradient(img, (0, 0, 0), 0.75, 0.05)
    else:
        img = base_canvas()
    a = fade_in_out(idx / FPS, n / FPS, 0.4, 0.4)
    d = ImageDraw.Draw(img)
    # 大数字
    t1 = "288,964"
    f1 = font(160)
    b1 = d.textbbox((0, 0), t1, font=f1)
    d.text(((W - (b1[2] - b1[0])) // 2, 300), t1,
           font=f1, fill=tuple(int(v * a) for v in CYAN))
    t2 = "个神经元 · 56,265,518 条突触 · 果蝇的 2.08 倍"
    f2 = font(48)
    b2 = d.textbbox((0, 0), t2, font=f2)
    d.text(((W - (b2[2] - b2[0])) // 2, 520), t2,
           font=f2, fill=tuple(int(v * a) for v in WHITE))
    return img


def scene_world(idx, n):
    p = os.path.join(PUB, "civ-world-iso.png")
    if os.path.exists(p):
        img = ken_burns(p, n, idx, 1.00, 1.16, pan=(0, -40))
        img = vignette(img, 0.45)
        img = overlay_gradient(img, (0, 0, 0), 0.0, 0.72)
    else:
        img = base_canvas()
    a = fade_in_out(idx / FPS, n / FPS, 0.5, 0.4)
    d = ImageDraw.Draw(img)
    t = "它自己长出了宗教"
    f = font(76)
    b = d.textbbox((0, 0), t, font=f)
    d.text(((W - (b[2] - b[0])) // 2, 860), t,
           font=f, fill=tuple(int(v * a) for v in WHITE))
    t2 = "青色是祭坛，金色是信徒。没有人告诉他们要拜。"
    f2 = font(38)
    b2 = d.textbbox((0, 0), t2, font=f2)
    d.text(((W - (b2[2] - b2[0])) // 2, 960), t2,
           font=f2, fill=tuple(int(v * a) for v in DIM))
    return img


def scene_data(idx, n):
    bg = os.path.join(PUB, "civ-world-shot.png")
    if os.path.exists(bg):
        img = ken_burns(bg, n, idx, 1.08, 1.16)
        img = Image.blend(base_canvas(), img, 0.16)
        img = vignette(img, 0.60)
    else:
        img = base_canvas()
    d = ImageDraw.Draw(img)
    t = idx / max(1, n - 1)
    e = ease(t)
    # 三个数字逐步弹入
    stats = [("19", "座祭坛", CYAN, 0.0),
             ("299", "个信徒", GOLD, 0.22),
             ("0.86", "平均信仰", BLUE, 0.44)]
    for i, (val, label, color, delay) in enumerate(stats):
        if t < delay:
            continue
        k = ease(min(1.0, (t - delay) / 0.25))
        x = W // 2 + (i - 1) * 560
        # 数字
        fv = font(150)
        bv = d.textbbox((0, 0), val, font=fv)
        c = tuple(int(v * k) for v in color)
        d.text((x - (bv[2] - bv[0]) // 2, 380), val, font=fv, fill=c)
        # 标签
        fl = font(42)
        bl = d.textbbox((0, 0), label, font=fl)
        d.text((x - (bl[2] - bl[0]) // 2, 560), label, font=fl,
               fill=tuple(int(v * k) for v in DIM))
    # 底部说明
    if t > 0.6:
        k = ease((t - 0.6) / 0.3)
        t2 = "第 19 代 · 350 个个体 · 全部来自真跑"
        f2 = font(38)
        b2 = d.textbbox((0, 0), t2, font=f2)
        d.text(((W - (b2[2] - b2[0])) // 2, 760), t2, font=f2,
               fill=tuple(int(v * k) for v in WHITE))
    return img


def scene_evolution(idx, n):
    """50 代进化曲线（真实数据：5303 -> 10971）。"""
    img = base_canvas()
    d = ImageDraw.Draw(img)
    t = idx / max(1, n - 1)
    e = ease(t)

    # 真实数据点（第 0-19 代，来自 exp_evolution 实测）
    vals = [5303, 6441, 7577, 5922, 5759, 7216, 7447, 7726, 8065, 8231,
            8234, 8550, 8753, 8749, 8863, 8836, 8881, 8863, 8838, 8858]
    x0, x1 = 280, W - 280
    y0, y1 = 780, 340
    vmin, vmax = 4500, 9500

    # 面板底（纯暗背景上一条线太素，实测 97% 单色）
    d.rounded_rectangle([x0 - 70, y1 - 70, x1 + 70, y0 + 90], radius=18,
                        fill=(13, 19, 27), outline=(30, 42, 56), width=2)
    # 网格线 + Y 轴刻度
    fl_ax = font(30, bold=False)
    for gy in range(5):
        yy = y1 + (y0 - y1) * gy / 4
        d.line([(x0, yy), (x1, yy)], fill=(26, 36, 48), width=1)
        val = int(vmax - (vmax - vmin) * gy / 4)
        d.text((x0 - 150, yy - 18), str(val), font=fl_ax, fill=(70, 84, 100))
    # X 轴：代数
    for gx in range(0, len(vals), 5):
        xx = x0 + (x1 - x0) * gx / (len(vals) - 1)
        d.line([(xx, y0), (xx, y0 + 8)], fill=(50, 62, 78), width=2)
        d.text((xx - 20, y0 + 16), "第%d代" % gx, font=fl_ax, fill=(70, 84, 100))
    # 坐标轴
    d.line([(x0, y0), (x1, y0)], fill=(50, 62, 78), width=2)
    d.line([(x0, y0), (x0, y1)], fill=(50, 62, 78), width=2)

    # 曲线（按进度逐点画）
    nshow = max(2, int(len(vals) * e))
    pts = []
    for i in range(nshow):
        px = x0 + (x1 - x0) * i / (len(vals) - 1)
        py = y0 - (y0 - y1) * (vals[i] - vmin) / (vmax - vmin)
        pts.append((px, py))
    if len(pts) >= 2:
        d.line(pts, fill=CYAN, width=6, joint="curve")
        # 当前点
        d.ellipse([pts[-1][0] - 9, pts[-1][1] - 9, pts[-1][0] + 9, pts[-1][1] + 9],
                  fill=WHITE)

    # 标题
    ft = font(64)
    tt = "人工选择 · 20 代"
    bt = d.textbbox((0, 0), tt, font=ft)
    d.text(((W - (bt[2] - bt[0])) // 2, 180), tt, font=ft, fill=WHITE)

    # 起止标注
    if e > 0.1:
        fl = font(40)
        d.text((x0 - 40, y0 - (y0 - y1) * (vals[0] - vmin) / (vmax - vmin) - 20),
               "5303", font=fl, fill=DIM)
    if e > 0.9:
        d.text((x1 - 90, y0 - (y0 - y1) * (vals[-1] - vmin) / (vmax - vmin) - 20),
               "8858", font=fl, fill=CYAN)
        # +67%
        fp = font(72)
        tp = "+67%"
        bp = d.textbbox((0, 0), tp, font=fp)
        d.text((x1 - (bp[2] - bp[0]) - 20, y0 - 180), tp, font=fp, fill=GOLD)
    return img


def scene_end(idx, n):
    # 结尾不能只有两行字（实测 96% 单色）。用世界图做暗背景。
    bg = os.path.join(PUB, "civ-world-iso.png")
    if os.path.exists(bg):
        img = ken_burns(bg, n, idx, 1.16, 1.24)
        img = Image.blend(base_canvas(), img, 0.22)
        img = vignette(img, 0.60)
    else:
        img = base_canvas()
    a = fade_in_out(idx / FPS, n / FPS, 0.5, 0.0)
    lines = [
        ("github.com/haoqinhe8-boop/NeuroShaqma", 56, CYAN, 28),
        ("全部代码、全部数据、全部失败记录都在里面", 38, DIM, 0),
    ]
    img = draw_center_text(img, lines, alpha=a)
    return img


# ==================== 主流程 ====================
SCENES = [
    (scene_hook, 10),
    (scene_scale, 10),
    (scene_world, 13),
    (scene_data, 11),
    (scene_evolution, 10),
    (scene_end, 6),
]


def main():
    os.makedirs(TMP, exist_ok=True)
    total = sum(d for _f, d in SCENES)
    print("总时长 %d 秒 = %d 帧" % (total, total * FPS))

    fi = 0
    for fn, dur in SCENES:
        n = dur * FPS
        print("  渲染 %-16s %2d 秒 (%d 帧)" % (fn.__name__, dur, n))
        for i in range(n):
            img = fn(i, n)
            img.save(os.path.join(TMP, "f%05d.png" % fi), compress_level=1)
            fi += 1

    print("  共 %d 帧，开始编码..." % fi)
    ffmpeg = None
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = "ffmpeg"

    cmd = [ffmpeg, "-y", "-framerate", str(FPS),
           "-i", os.path.join(TMP, "f%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "20", "-preset", "medium",
           "-movflags", "+faststart", OUT]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print("ffmpeg 失败:")
        print(r.stderr[-2000:])
        return 1
    size = os.path.getsize(OUT) / 1024 / 1024
    print("\n完成: %s (%.1f MB, %d 秒)" % (OUT, size, total))

    # 清理
    for f in os.listdir(TMP):
        try:
            os.remove(os.path.join(TMP, f))
        except Exception:
            pass
    try:
        os.rmdir(TMP)
    except Exception:
        pass
    return 0


# ==================== 自测 ====================

def selftest():
    """验证渲染管线的前置条件，不真渲染 1800 帧（那是 5 分钟）。"""
    print("  小沙宣传视频 selftest（素材 / 缓动 / 画布 / 分层）")
    ok = 0
    total = 0

    # ---- 1. 素材存在（缺一个视频就会有段落是黑的）----
    total += 1
    need = ["civ-world-iso.png", "civ-world-shot.png", "hero-288k.png"]
    missing = [f for f in need if not os.path.exists(os.path.join(PUB, f))]
    print("  [1] 素材 %d/%d 存在%s"
          % (len(need) - len(missing), len(need),
             ("，缺：" + ",".join(missing)) if missing else ""))
    ok += int(not missing)

    # ---- 2. 字体能加载（否则全部文字变方块）----
    total += 1
    f = font(48)
    loaded = hasattr(f, "getbbox") and f.size == 48
    print("  [2] 中文字体: %s size=%s" % (type(f).__name__, getattr(f, "size", "?")))
    ok += int(loaded)

    # ---- 3. 缓动函数边界正确 ----
    total += 1
    good = (ease(0) == 0.0 and ease(1) == 1.0
            and 0.4 < ease(0.5) < 0.6          # smoothstep(0.5) = 0.5
            and ease(-5) == 0.0 and ease(5) == 1.0)  # 越界要夹住
    print("  [3] ease: 0→%.1f 0.5→%.2f 1→%.1f 越界钳制=%s"
          % (ease(0), ease(0.5), ease(1), ease(-5) == 0.0))
    ok += int(good)

    # ---- 4. 淡入淡出：起点 0、中段 1、终点 0 ----
    total += 1
    a0 = fade_in_out(0.0, 10.0)          # 刚出现
    amid = fade_in_out(5.0, 10.0)        # 正中间
    aend = fade_in_out(10.0, 10.0)       # 正好结束（原来测 9.9，那时淡出才走 25%，
                                         # 值 0.156 是正确答案，是断言错了）
    good = a0 < 0.05 and amid == 1.0 and aend < 0.05
    print("  [4] 淡入淡出: 起点%.2f 中段%.2f 终点%.2f" % (a0, amid, aend))
    ok += int(good)

    # ---- 5. 画布尺寸正确 ----
    total += 1
    img = base_canvas()
    print("  [5] 画布: %dx%d (期望 %dx%d)" % (img.size[0], img.size[1], W, H))
    ok += int(img.size == (W, H))

    # ---- 6. Ken Burns 输出尺寸对（裁剪后必须正好是画布大小）----
    total += 1
    src = os.path.join(PUB, "civ-world-iso.png")
    if os.path.exists(src):
        f0 = ken_burns(src, 100, 0, 1.0, 1.1)
        f9 = ken_burns(src, 100, 99, 1.0, 1.1)
        good = f0.size == (W, H) and f9.size == (W, H)
        print("  [6] Ken Burns: 首帧%dx%d 末帧%dx%d"
              % (f0.size[0], f0.size[1], f9.size[0], f9.size[1]))
    else:
        good = False
        print("  [6] Ken Burns: 缺素材，跳过")
    ok += int(good)

    # ---- 7. 六段总时长 = 60 秒 ----
    total += 1
    total_sec = sum(d for _f, d in SCENES)
    print("  [7] 场景数 %d | 总时长 %d 秒 = %d 帧" % (len(SCENES), total_sec, total_sec * FPS))
    ok += int(len(SCENES) == 6 and total_sec == 60)

    # ---- 8. 暗角/渐变不改变尺寸 ----
    total += 1
    v = vignette(base_canvas(), 0.5)
    g = overlay_gradient(base_canvas(), (0, 0, 0), 0.7, 0.1)
    print("  [8] vignette %dx%d | overlay_gradient %dx%d"
          % (v.size[0], v.size[1], g.size[0], g.size[1]))
    ok += int(v.size == (W, H) and g.size == (W, H))

    print("\n  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    # 默认跑 selftest —— 和其他所有 xiaosha_*.py 模块的约定一致。
    # 之前默认直接渲染 1800 帧，导致 run_all_selftests.py 每次跑全量都多花 292 秒
    # 去重新生成一遍视频。渲染现在是显式的：python make_civ_video.py --render
    if "--render" in sys.argv:
        sys.exit(main())
    sys.exit(0 if selftest() else 1)
