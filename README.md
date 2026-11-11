# NeuroShaqma · 小沙大脑 / Xiaosha Brain

![stars](https://img.shields.io/github/stars/haoqinhe8-boop/NeuroShaqma?style=flat-square) ![license](https://img.shields.io/github/license/haoqinhe8-boop/NeuroShaqma?style=flat-square) [![讨论](https://img.shields.io/badge/discussions-有问题来聊-blue?style=flat-square)](https://github.com/haoqinhe8-boop/NeuroShaqma/discussions)

> **我 12 岁。我在家里用一台游戏电脑，造出了一个比果蝇大脑更大的脉冲神经网络大脑——它有 2.08 亿个突触，真的会学习，还已经自己活了 49 天。**
>
> **I'm 12. At home, on a gaming PC, I built a spiking neural network bigger than a fruit fly's brain — 208 million synapses, it really learns, and it's been alive for 49 days.**

![小沙大脑 vs 果蝇大脑](brain-vs-fly.png)

![大脑在放电（真实仿真录像）](spike-activity.gif)

上面这张动图是它真实仿真时的样子：每一个亮点都是一个神经元在发放脉冲。它平时 99% 的神经元是安静的，只有 0.3% 在放电——就像果蝇那样，用极少能量干很多事。

*The GIF above is a real simulation: each dot is a neuron firing a spike. 99% of its neurons are quiet at any moment, only 0.3% fire — like a fly, doing a lot with very little energy.*

## 作者 / Author

**Peter**，来自上海市宝山区的一名 12 岁初中生。这个项目完全是在家里的台式机上完成的——没有实验室，没有服务器，没有团队。白天上学，晚上回家写代码。

**Peter**, a 12-year-old middle-schooler from Baoshan District, Shanghai. Built entirely on a home PC — no lab, no server, no team.

- **NVIDIA RTX 2080 Ti（11GB）** —— 日常开发、跑 7B 语言模型 / daily dev, runs a 7B LLM
- **NVIDIA RTX 4090（48GB）** —— 跑 72B 大模型和大脑的大规模仿真 / runs a 72B model and large-scale brain sims

## 开发方式（诚实说明）/ How it was built (honest)

这个项目的**主力是 DeepSeek-V4-Pro**：绝大部分的代码编写、调试、实验设计和参数调整都由它完成。Peter 负责提出想法、拍板关键决策并验收结果。所有数字都是在真实机器上真实跑出来的，不是 AI 编的——每个都能用 `demo.py`、`bench.py` 和 `--selftest` 自己验证。

**The main force is DeepSeek-V4-Pro**: most of the coding, debugging, experiment design and tuning. Peter sets the ideas, makes the key calls and checks the results. Every number was really run on real hardware — verify with `demo.py`, `bench.py` and `--selftest`.

## 这是什么 / What is this

小沙大脑是一个**事件驱动的脉冲神经网络**（Spiking Neural Network），模仿真实大脑的工作方式：神经元平时安静，只有膜电位超过阈值时才发放一个脉冲（spike），信号以脉冲事件的形式在突触间传递。它不靠大矩阵乘法堆算力，而是靠**稀疏发放 + 事件驱动**。

An **event-driven Spiking Neural Network** that mimics a real brain: neurons stay silent, fire a spike only when their membrane potential crosses a threshold, and signals travel as sparse spike events — not dense matrix math.

## 小沙大脑 vs 果蝇大脑 / Xiaosha vs Fruit Fly

| 指标 Metric | 小沙 Xiaosha | 果蝇 Fly | 对比 Ratio |
|---|---|---|---|
| 神经元 Neurons | **288,964** | 139,255 | **2.07×** |
| 突触 Synapses | **208,054,080** | ~50,000,000 | **4.16×** |
| 每神经元连接 Connections/neuron | 720 | ~360 | 2× |
| 全系统细胞 Total cells (皮层+小脑+脑干+脊髓+感觉/运动外周) | **3,985,964** | 139,255 | **28.6×** |

> 果蝇数据参考：果蝇大脑约 139,255 个神经元、约 5,000 万突触（连接组学公开数据）。
> *Fly reference: ~139,255 neurons and ~50M synapses (public connectome data).*

## 能干什么 / What it can do（每个数字都是真跑出来的 / every number really ran）

- **会学习 Learns**：R-STDP —— 练完导航任务，成功率从出厂 33% 涨到 93% / navigation success 33%→93%
- **会自我调节 Self-regulates**：稳态发放率 + E/I 平衡，活跃度稳定 0.28% / homeostasis keeps firing at 0.28%
- **跑得动 Fast**：28.9 万神经元 CPU 20.7ms/步（48Hz 实时）、4090 上 7.26ms/步（138Hz）
- **有身体 Has a body**：耳听 / 嘴说 / 鼻闻 / 脚走 / 手抓（hears, speaks, smells, walks, grabs）
- **有内脏 Has organs**：肺呼吸 / 心脏泵血 / 肝肠吸收 / 泌尿清理 / 免疫查 Bug / 内分泌调节

## 实战记录 / Real-world track record（真跑出来的 / really ran）

小沙已经在这个家里真实自主运行了近两个月。以下数字全部来自运行日志，逐条可查。
*It has run autonomously at home for nearly two months. All numbers are from real logs.*

### 1. 自主运行 49 天，5,171 个决策 0 失败 / 49 days, 5,171 decisions, 0 failures

自 2026-09-19 起连续自主运行（计划任务每 10 分钟唤醒，无人值守）：

| 指标 Metric | 数字 Value |
|---|---|
| 自主决策动作总数 Decisions | 5,171 |
| 成功 / 失败 Success / Fail | 5,171 / 0（100%） |
| 自主内容产出 Content produced | 3,396 次 |
| 话题池整理 Topic curation | 1,741 次 |
| 产品制作 Products made | 18 次 |
| 自主发起学习 Self-initiated learning | 13 次 |

### 2. 自学 26.6 万条知识 / 266k items self-learned（158MB）

- 知识库 266,381 条（158MB），全是从网络自己学来的（GitHub/arXiv/CSDN 等公开源）
- 学习日志 21,413 条 / learning log 21,413 entries
- 自己维护的话题池 34,587 个 / 34,587 self-curated topics

### 3. 自主产出与经营 / Autonomous production

- 内容产出日志 59,040 条 / 59,040 production log entries
- 上架作品 4,252 件（"热点速读"等，定价 2.9 元/件）/ 4,252 listed products
- 支付扫码到账系统真实运行 / real QR-payment scan system

### 4. 身体的真实闭环 / Real body loop

- **听觉与发声**："嘴巴说 → 耳朵听"闭环，一字不差听懂 / speech→ASR loop, word-perfect
- **脚（躯干适配）**：6 种躯干（两轮差速/阿克曼/履带/三全向/四轮滑移/远程介质）开机自识别、自学身体模型、自己走到目标——21/21 自检 / 6 bodies self-identified, 21/21
- **手**：机械臂写字指令下发 / robotic-arm command

### 5. 学习能力实测 / Learning measured

- 导航：出厂 33% → 训练后 93%
- 大规模学习（288,964×720）：评估 0/30 → 25/30（83%）

## 它和果蝇的区别（说实话）/ vs the fly, honestly

**果蝇赢在**：几亿年进化，脑区结构精细到每个连接都有分工。*The fly wins on evolution: every connection has a job.*
**但"分工合作"已不是空话**：小沙做了跨感官分工任务并行为学验证（`xiaosha_teamwork.py`，6/6）：视觉区专职看、听觉区专职听、联合区整合、运动区执行。①分区网络学会 2%→100%，随机网络只到 78% ②消融实验——遮视觉命中率 100%→55%（视觉区专职看），遮听觉同理 ③全信息下全脑发放是单感官的 4.07 倍。*But "division of labour" is now proven: cross-modal task, region network 2%→100% vs random 78%, ablation shows vision/auditory regions are specialized.*
**小沙赢在**：数量和实时性追平甚至超过果蝇；而且它可以随时长大。*Xiaosha wins on scale and realtime, and it can grow anytime.*

## 不足点 / Limitations

1. **规模仍很小**：人脑约 860 亿神经元，小沙只有果蝇量级 / still fly-scale, ~6 orders of magnitude below the human brain
2. 结构已有分工（7 脑区 + 蘑菇体条件反射 + 中央复合体导航），但离精细回路仍有距离
3. **学习仍在 CPU**：运行已 GPU（fp16，实测 91Hz），R-STDP 学习本身还在 CPU
4. **器官部分是仿真**：嗅觉是气味场模型，真机传感器接口留好未接
5. **任务简单**：只验证了导航学习
6. 测试有 pytest 入口，但偏"冒烟级"
7. **Windows 专属**：Linux/macOS 未适配
8. **工作记忆与多任务边界**：延迟匹配（DMS）学不会（42%<随机）；交替训练多任务会灾难性遗忘（分类单独100%→交替25%）。**但已攻下半个**：海马体记忆重放（经验重放）把多任务遗忘的任务A 从 55% 救回 82%（`xiaosha_hippocampus.py`，4/4）；工作记忆仍缺前额叶持续放电结构（已试慢时间常数神经元 slow_frac，48%→52% 仍不够——已实现分群吸引子+侧抑制+选择性路由（Wang 2001 模型，`slow_attractor`+`attractor_groups`），DMS 命中率 48~62% 波动仍不达标——结论：吸引子结构是必要条件但不够，R-STDP（无监督局部规则）训不出稳定持续活动，下一步需 FORCE/监督式训练吸引子）：延迟匹配样本任务（DMS）实测学不会（练后 42% < 随机 50%）——当前架构没有持续保持信息的专门结构，这是下一步方向（慢时间常数神经元/记忆单元）

## 目录 / Structure

```
xiaosha/
├── README.md             # 项目说明 docs
├── LICENSE               # MIT
├── CITATION.cff          # 引用 cite this
├── demo.py               # 一分钟体验 1-minute demo
├── bench.py              # 规模-速度扫描 scale/speed bench
├── xiaosha_cortex.py     # 脉冲皮层 SNN cortex (288,964×720, R-STDP)
├── xiaosha_regions.py    # 分区皮层 7 brain regions
├── xiaosha_teamwork.py   # 脑区分工合作 division of labour
├── xiaosha_circuits.py   # 果蝇回路 mushroom body + central complex
├── xiaosha_gpu.py        # GPU 运行器 fp16 runner
├── xiaosha_embodiment.py # 躯干适配 body-agnostic control
├── xiaosha_body.py       # 身体五感 5 senses
├── xiaosha_organs.py     # 内脏八大系统 8 organ systems
├── xiaosha_spine.py      # 脊椎 spine
├── xiaosha_immune.py     # 免疫 immune
├── xiaosha_watchdog.py   # 看门狗 watchdog
├── xiaosha_judge.py      # 判断器 judge
├── xiaosha_vision.py     # 视觉 vision
└── xiaosha_unitree.py / xiaosha_microduck.py  # 宇树/Microduck 适配器 robot adapters
```

## 有问题 / 想交流？ Questions?

- [GitHub Discussions](https://github.com/haoqinhe8-boop/NeuroShaqma/discussions) —— 提问、交流、提建议（推荐，我在这回复你）/ ask and chat (I reply here)
- [提 Issue](https://github.com/haoqinhe8-boop/NeuroShaqma/issues) —— 报 Bug、要功能 / bugs, feature requests
- 邮箱 Email：haoqinhe8@gmail.com / 2731952033@qq.com

## 使用与署名 / Use / Attribution

**随意使用、借鉴、修改、商用**——只要遵守一条：**保留原作者署名**。
**Free to use, copy, modify, sell** — just keep the author attribution.

- 许可证 License：MIT（见 `LICENSE`）——保留版权声明即可自由使用
- 每个代码文件头部都有署名 / every source file carries: `Copyright (c) 2026 Peter (NeuroShaqma)`
- 写论文引用用 `CITATION.cff`（GitHub 会自动生成"Cite this repository"按钮）
- 原创者 Author：Peter（上海市宝山区，12 岁）——转发请带名字和仓库链接

## 怎么跑（一分钟上手）/ Quick start (1 minute)

```bash
# 1. 下载代码 clone
git clone https://github.com/haoqinhe8-boop/NeuroShaqma.git
cd NeuroShaqma          # ★ 一定要先进项目目录 / must cd in first

# 2. 装依赖 install deps (only numpy)
pip install -r requirements.txt
#    Windows 不想敲命令：双击 run-demo.bat（自动装依赖并启动）

# 3. 一分钟体验 1-minute demo: 造脑→放电→学习→成绩单
python demo.py

# 4. 规模扫描 scale scan
python bench.py
```

报 `ModuleNotFoundError: No module named 'numpy'` 就是第 2 步没做 / = you skipped step 2.

自检 Self-test（每个模块都有 / every module）:

```bash
python xiaosha_cortex.py --selftest     # 皮层 cortex 10 项
python xiaosha_embodiment.py --selftest # 躯干 body 21 项
python xiaosha_body.py --selftest       # 五感 senses 5 项
python xiaosha_organs.py --selftest     # 内脏 organs 6 项
```

依赖 Dependencies：Python 3.9+，NumPy。

## 联系方式 / Contact

- 📧 haoqinhe8@gmail.com
- 📧 2731952033@qq.com

欢迎交流，欢迎提 Issue，欢迎一起把它养大。*Welcome — chat, open issues, help it grow.*

*—— Peter，上海市宝山区 / Baoshan, Shanghai · 2026 年 11 月*