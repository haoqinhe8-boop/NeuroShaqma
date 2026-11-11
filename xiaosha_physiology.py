# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""人体生理增强（2026-11）：器官对照人体真实「健康」机制。

只学人体健康的、高效的、适应性的机制，**明确排除**：
  - 衰老（器官功能随年龄下降）
  - 退化（萎缩、失用性退化）
  - 疾病/病变
学的都是健康成年人的正常高效功能——小沙是 AI，只取人体的优点。

本模块补充 xiaosha_organs.py 的八大系统，补上 5 个人体标志性机制：
  ① 心脏：心率自主调节 + 每搏输出量 + 心输出量 = 心率×每搏量（人体循环核心公式）
  ② 肝脏：解毒（代谢废物→无害）+ 合成血浆蛋白 + 糖原储存
  ③ 肾脏：过滤 + 重吸收（保留有用、只排真废物）+ 水盐平衡
  ④ 内分泌：激素轴 + 负反馈（把偏离拉回稳态，下丘脑-垂体-靶腺式）
  ⑤ 免疫：先天免疫（快速非特异）+ 适应性免疫（特异 + 免疫记忆）

用法：python xiaosha_physiology.py --selftest
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ① 心脏：心率自主调节 + 每搏输出量 + 心输出量
class Heart:
    """人体心脏：窦房结起搏 + 自主神经调心率 + 每搏输出量 → 心输出量。"""
    def __init__(self, stroke_volume=70.0):
        self.stroke_volume = float(stroke_volume)  # 每搏输出量（mL，人体 ~70mL）
        self.bpm = 60.0                            # 静息心率（次/分）
        self.beats = 0

    def set_rate(self, bpm):
        """自主神经调节心率：活动加快、静息放慢（30~200 次/分）。"""
        self.bpm = max(30.0, min(200.0, float(bpm)))
        return self.bpm

    def cardiac_output(self):
        """心输出量 = 心率 × 每搏输出量（人体循环核心公式，单位 mL/分）。"""
        return self.bpm * self.stroke_volume

    def beat(self):
        """搏动一次，泵出每搏量的血。"""
        self.beats += 1
        return self.stroke_volume


# ② 肝脏：解毒 + 合成 + 储存
class Liver:
    """人体肝脏（500+ 功能里的三个代表）：解毒、合成血浆蛋白、储存糖原。"""
    TOXINS = ("氨", "酒精", "药物代谢物")

    def __init__(self):
        self.glycogen = 0.0    # 糖原储备（g）

    def detox(self, substance):
        """解毒：把有毒代谢物转成无害尿素/代谢物（肝最核心的功能）。"""
        return ("已解毒 → 无害", substance) if substance in self.TOXINS else ("无害", substance)

    def synthesize(self, n=1):
        """合成血浆蛋白（白蛋白等，维持渗透压）。"""
        return ["白蛋白"] * n

    def store_glycogen(self, glucose):
        """储存糖原（血糖高时把葡萄糖存成糖原，稳定血糖）。"""
        self.glycogen += float(glucose)
        return self.glycogen


# ③ 肾脏：过滤 + 重吸收 + 水盐平衡
class Kidney:
    """人体肾脏：肾小球过滤 → 肾小管重吸收（保留有用、只排真废物）+ 水盐平衡。"""
    WASTE = ("尿素", "肌酐", "尿酸")

    def __init__(self):
        self.filtered_total = 0

    def filter(self, blood):
        """肾小球过滤：把血液里的东西都滤进原尿（人体每天滤 180L，99% 重吸收）。"""
        self.filtered_total += len(blood)
        return list(blood)

    def reabsorb(self, filtrate):
        """肾小管重吸收：保留有用的（水/葡萄糖/电解质），只排真废物。"""
        keep = [x for x in filtrate if x not in self.WASTE]
        waste = [x for x in filtrate if x in self.WASTE]
        return keep, waste

    def water_balance(self, intake, output):
        """水盐平衡：水少了就少排（重吸收多），水多了就多排。"""
        if intake < output:
            return "抗利尿（重吸收水，尿少而浓）"
        return "水利尿（排出多余水）"


# ④ 内分泌：激素轴 + 负反馈
class Endocrine:
    """人体内分泌：激素轴 + 负反馈（把偏离拉回稳态，下丘脑-垂体-靶腺式）。"""
    def __init__(self):
        self.cortisol = 1.0      # 皮质醇（应激/糖代谢）
        self.thyroid = 1.0       # 甲状腺激素（代谢率）
        self.insulin = 1.0       # 胰岛素（降血糖）

    def negative_feedback(self, level, setpoint, gain=0.5):
        """负反馈：偏离稳态就反向拉回（高了降、低了升）。"""
        error = setpoint - level
        return level + gain * error

    def regulate_glucose(self, glucose, setpoint=5.0):
        """血糖调节：高→胰岛素分泌↑降血糖，低→胰高血糖素↑升血糖（负反馈）。"""
        if glucose > setpoint:
            self.insulin = 1.0 + 0.2 * (glucose - setpoint)
            return "胰岛素↑ 降血糖"
        self.insulin = max(0.1, 1.0 - 0.2 * (setpoint - glucose))
        return "胰高血糖素↑ 升血糖"


# ⑤ 免疫：先天免疫 + 适应性免疫（记忆）
class Immune:
    """人体免疫：先天免疫（快速非特异）+ 适应性免疫（特异 + 记忆，二次接触快速清除）。"""
    def __init__(self):
        self.memory = set()   # 免疫记忆：遇到过的病原特征

    def respond(self, pathogen):
        """遇到病原：首次=先天免疫清除+建立记忆；再次=记忆细胞快速清除。"""
        if pathogen in self.memory:
            return "快速清除（免疫记忆，二次接触秒杀）"
        self.memory.add(pathogen)
        return "首次清除（先天免疫 + 建立记忆）"

    def known(self, pathogen):
        return pathogen in self.memory



# ⑥ 胃：胃酸消化 + 机械搅拌
class Stomach:
    """人体胃：胃酸（pH~2）杀菌 + 分解蛋白，机械搅拌成食糜。"""
    def __init__(self):
        self.ph = 2.0          # 胃酸 pH（强酸杀菌）
    def digest(self, food):
        """胃酸消化：把大分子食物分解成食糜（chyme，半流质）。"""
        return ["已分解的" + f for f in food]


# ⑦ 小肠：绒毛吸收
class SmallIntestine:
    """人体小肠：百万绒毛把吸收面积铺到 ~250㎡，高效吸收营养。"""
    def __init__(self):
        self.villi = 1000000   # 绒毛数量（人体约 100 万）
    def absorb(self, chyme):
        """绒毛吸收：分解后的营养被吸收入血。"""
        return [x for x in chyme if "已分解" in x]


# ⑧ 胰腺：外分泌消化酶 + 内分泌胰岛素
class Pancreas:
    """人体胰腺：外分泌（消化酶进小肠）+ 内分泌（胰岛素/胰高血糖素调血糖）。"""
    def __init__(self):
        self.enzymes = ["淀粉酶", "蛋白酶", "脂肪酶"]
        self.insulin = 1.0
    def secrete_enzymes(self):
        return list(self.enzymes)
    def secrete_insulin(self, glucose):
        """血糖高 → 胰岛素分泌↑（降血糖）。"""
        self.insulin = max(0.1, 1.0 + 0.1 * (glucose - 5.0))
        return self.insulin


# ⑨ 膀胱：储尿 + 排尿反射
class Bladder:
    """人体膀胱：储尿（~500mL）+ 尿意反射 + 排空。"""
    def __init__(self, capacity=500.0):
        self.capacity = float(capacity)
        self.volume = 0.0
    def fill(self, amount):
        self.volume += float(amount)
        return self.volume
    def urge(self):
        """尿意强度 = 充盈比例（越满越想尿）。"""
        return self.volume / self.capacity
    def void(self):
        released = self.volume
        self.volume = 0.0
        return released


# ⑩ 眼睛：视杆（暗视觉）+ 视锥（色觉）
class Eye:
    """人体眼睛：视杆 ~1.2 亿（暗光/亮度）+ 视锥 ~600 万（颜色/细节）。"""
    def __init__(self):
        self.rods = 120000000
        self.cones = 6000000
    def see_dim(self):
        return "暗视觉（视杆主导，亮度敏感）"
    def see_color(self, color):
        return f"色觉：看到{color}（视锥红/绿/蓝三通道）"


# ⑪ 耳朵：耳蜗毛细胞频率分析
class Ear:
    """人体耳朵：耳蜗 ~1.5 万毛细胞按位置编码频率（频谱分析）。"""
    def __init__(self):
        self.hair_cells = 15000
    def hear(self, freq):
        return f"听到 {freq}Hz（毛细胞按位置编码频率）"


# ⑫ 鼻子：嗅上皮嗅受体
class Nose:
    """人体鼻子：~400 种嗅受体组合识别上万种气味。"""
    def __init__(self):
        self.receptors = 400
    def smell(self, odor):
        return f"闻到 {odor}（嗅受体组合识别）"


# ⑬ 舌头：味蕾 5 种基本味觉
class Tongue:
    """人体舌头：~1 万味蕾，5 种基本味觉（甜咸酸苦鲜）。"""
    def __init__(self):
        self.taste_buds = 10000
        self.basic = ["甜", "咸", "酸", "苦", "鲜"]
    def taste(self, food):
        return f"尝到 {food}（味蕾五基本味觉）"


# ============================================================
def selftest():
    print("=" * 76)
    print("  人体生理增强 selftest（只学健康机制，排除衰老/退化/疾病）")
    print("=" * 76)
    results, fails = [], []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         └ {detail}")
        if not ok:
            fails.append(name)

    # ① 心脏
    h = Heart()
    rest = h.cardiac_output()
    h.set_rate(120)                        # 活动时心率加快
    active = h.cardiac_output()
    check("心脏：心率自主调节 + 心输出量=心率×每搏量", active > rest * 1.8,
          f"静息 {rest:.0f}mL/分 → 活动(120bpm) {active:.0f}mL/分（{h.beats} 次搏动后）")

    # ② 肝脏
    l = Liver()
    d = l.detox("氨")
    check("肝脏：解毒（有毒→无害）", d[0].startswith("已解毒"), f"{d}")
    check("肝脏：合成蛋白 + 储存糖原", len(l.synthesize(2)) == 2 and l.store_glycogen(5.0) == 5.0,
          f"合成 {len(l.synthesize(2))} 份白蛋白 ｜ 糖原 {l.glycogen:.1f}g")

    # ③ 肾脏
    k = Kidney()
    blood = ["水", "葡萄糖", "钠", "尿素", "肌酐", "水"]
    filtrate = k.filter(blood)
    keep, waste = k.reabsorb(filtrate)
    check("肾脏：过滤 + 重吸收（保留有用、只排真废物）",
          "尿素" in waste and "肌酐" in waste and "葡萄糖" in keep and "水" in keep,
          f"滤出 {len(filtrate)} 份 ｜ 重吸收保留 {len(keep)} 份 ｜ 排废物 {waste}")
    check("肾脏：水盐平衡（缺水抗利尿、多水利尿）",
          k.water_balance(3, 5) == "抗利尿（重吸收水，尿少而浓）" and k.water_balance(5, 3) == "水利尿（排出多余水）",
          "双向调节正常")

    # ④ 内分泌
    e = Endocrine()
    corrected = e.negative_feedback(8.0, 5.0)   # 偏离稳态，负反馈拉回
    check("内分泌：负反馈（偏离稳态被拉回）", abs(corrected - 5.0) < abs(8.0 - 5.0),
          f"偏离 8.0 → 校正 {corrected:.1f}（稳态 5.0）")
    check("内分泌：血糖双向调节", e.regulate_glucose(8.0) == "胰岛素↑ 降血糖" and e.regulate_glucose(3.0) == "胰高血糖素↑ 升血糖",
          f"高血糖→{e.regulate_glucose(8.0)}")

    # ⑤ 免疫
    im = Immune()
    first = im.respond("流感病毒")
    second = im.respond("流感病毒")
    check("免疫：先天免疫 + 免疫记忆（二次接触快速清除）",
          "首次" in first and "记忆" in second and im.known("流感病毒"),
          f"首次：{first} ｜ 再次：{second}")

    # ⑥ 胃 + 小肠
    st = Stomach()
    chyme = st.digest(["蛋白质", "脂肪"])
    check("胃：胃酸消化（大分子→食糜）", all("已分解" in x for x in chyme), f"{chyme}")
    si = SmallIntestine()
    absorbed = si.absorb(chyme)
    check("小肠：百万绒毛高效吸收", len(absorbed) == 2 and si.villi == 1000000,
          f"吸收 {len(absorbed)} 份 ｜ 绒毛 {si.villi} 根")

    # ⑦ 胰腺
    p = Pancreas()
    check("胰腺：外分泌消化酶 + 内分泌胰岛素",
          "淀粉酶" in p.secrete_enzymes() and p.secrete_insulin(8.0) > 1.0,
          f"酶 {p.secrete_enzymes()} ｜ 高血糖胰岛素 {p.secrete_insulin(8.0):.2f}")

    # ⑧ 膀胱
    b = Bladder()
    b.fill(400)
    urge = b.urge()
    released = b.void()
    check("膀胱：储尿 + 尿意反射 + 排空", urge == 0.8 and released == 400 and b.volume == 0,
          f"充盈 400mL（尿意 {urge:.1f}）→ 排空 {released:.0f}mL")

    # ⑨-⑬ 五感器官
    eye = Eye()
    check("眼睛：视杆暗视觉(1.2亿) + 视锥色觉(600万)", eye.rods > eye.cones and "色觉" in eye.see_color("红色"),
          f"视杆 {eye.rods/1e6:.0f}百万 ｜ 视锥 {eye.cones/1e6:.0f}百万")
    ear = Ear()
    check("耳朵：耳蜗毛细胞频率分析", ear.hair_cells == 15000 and "Hz" in ear.hear(440),
          f"毛细胞 {ear.hair_cells} 根")
    nose = Nose()
    check("鼻子：嗅受体组合识别", nose.receptors == 400 and "闻到" in nose.smell("花香"),
          f"嗅受体 {nose.receptors} 种")
    tongue = Tongue()
    check("舌头：味蕾 5 种基本味觉", len(tongue.basic) == 5 and "甜" in tongue.basic,
          f"味蕾 {tongue.taste_buds} 个 ｜ 基本味 {tongue.basic}")

    print("-" * 76)
    print(f"  结果：{sum(results)}/{len(results)} 项通过"
          + ("  ✅ 全部通过" if not fails else f"  ❌ 失败：{fails}"))
    print("=" * 76)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    ap.print_help()