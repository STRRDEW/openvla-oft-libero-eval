# 项目经历（简历 / 作品集 / 面试口述）

> 定位：个人工程评测项目，不是算法创新，也不是论文复现。
> 范围：官方 OpenVLA-OFT 7B 权重 + LIBERO-Spatial，纯推理/评测；
> **没有训练、没有微调、没有 LoRA 训练**。

---

## A. 简历版（可直接粘贴，建议 4–5 条）

**基于 OpenVLA-OFT 与 LIBERO 的语言条件机械臂闭环评测**（个人项目，2026.10）  
技术栈：Python / PyTorch / CUDA / LIBERO / MuJoCo / robosuite / HuggingFace Transformers / Linux

- 在 AutoDL 单卡 RTX 6000D（Blackwell sm_120，84GB）上部署 OpenVLA-OFT 7B
  官方权重与 LIBERO 仿真；解决官方 PyTorch 2.2 无 sm_120 内核、torch 2.7
  checkpoint 加载、LIBERO EGL 无头渲染、TensorFlow/protobuf 版本冲突；
  固定 openvla-oft `e4287e9`、LIBERO `8f1084e`。
- 基于官方评测函数实现逐 episode 闭环评测器：记录任务、初始状态哈希、seed、
  成功标志、env/policy 步数、模型调用次数、`torch.cuda.synchronize()` 同步后的
  推理延迟、仿真/视频/墙钟时间、显存峰值和视频路径；输出 JSONL/CSV/图表，
  支持任务子集与断点续跑。
- 完成 LIBERO-Spatial 10 tasks × 10 episodes 基线：k=8 成功率 **99/100
  （99.0%）**，平均 13.32 次模型调用、126.28 ms/次推理、7.93 s/episode；
  推理峰值 reserved 显存 **15.18GB**。
- 完成动作块执行长度受控对照（模型固定输出 8 步，分别执行前 1/4/8 步再观察）：
  k=4 为 **99/100**、26.51 次调用、9.51 s/episode（调用约 2×）；
  k=1 为 **94/100**、111.76 次调用、20.28 s/episode。更频繁重规划未提升
  这组样本的成功率，主要代价是模型调用次数；100 episode/配置，Wilson 区间
  重叠，未宣称显著。
- 发现官方 `deque(maxlen=k)` 在 k<8 时保留动作块**最后 k 个**而非前 k 个；
  在项目 runner 中显式改为执行前 k 个，保证对照语义（k=8 基线不受影响）。
  产出 62.7 s 真实 rollout 演示、完整视频/JSONL/图表和 4 个失败案例，
  区分“观察到的现象”与“未验证的原因猜测”。

> 如果简历只能放 3 条：保留“部署适配”“10×10 基线 + 评测器”“k=1/4/8 对照 + deque 发现”。

---

## B. 展开版（作品集 / 项目描述）

### 1. 项目背景

VLA（Vision-Language-Action）模型把视觉观察和语言指令映射为机器人动作。
OpenVLA-OFT 在 OpenVLA 基础上使用动作分块、连续动作头、proprio 输入和并行
解码，在 LIBERO 等仿真 benchmark 上取得较强结果。

本项目不做训练，而是回答一个工程评测问题：

> 模型一次预测 8 个动作，每次执行多少步再重新观察，会怎样影响成功率和计算开销？

### 2. 我做了什么

1. **环境适配**：在 Blackwell sm_120 上跑通官方 PyTorch 2.2 不可用的链路，
   改用 PyTorch 2.7.1 + cu128；用 `--no-deps` 安装 openvla-oft，避免依赖
   把 torch 降回 2.2.0；处理 torch 2.7 checkpoint 加载、EGL 无头渲染、
   TensorFlow/protobuf 冲突。
2. **评测工具**：基于官方 `run_libero_eval.py` 的模型加载/观测/动作处理函数，
   编写项目侧 `tools/run_eval.py`，补齐官方脚本没有的逐 episode JSONL、
   初始状态哈希、模型调用计数、同步计时、显存峰值和视频路径。
3. **基线**：LIBERO-Spatial 10 个任务 × 10 个 episode，k=8。
4. **受控对照**：固定模型、任务、初始状态、seed、图像预处理，只改变执行长度
   k=1/4/8；k=8 复用基线，不重复花费。
5. **分析与产出**：逐任务成功率、调用次数、延迟、墙钟、Wilson 区间、
   受控失败对照、真实视频和失败帧。

### 3. 结果

| 配置 | 成功数 | 平均模型调用/episode | 平均单次推理 | 平均墙钟/episode |
|---|---:|---:|---:|---:|
| k=8 基线 | **99/100** | 13.32 | 126.28 ms | 7.93 s |
| k=4 | **99/100** | 26.51 | 125.92 ms | 9.51 s |
| k=1 | **94/100** | 111.76 | 124.43 ms | 20.28 s |

- 单次推理延迟在三种配置下都在 124–126 ms，主要成本来自调用次数；
  k=4 与基线同成功率，但调用约 2 倍；k=1 调用约 8.4 倍。
- k=1 的 6 个失败全部跑到 220 个 policy steps 上限；k=1 每步调用一次模型，
  所以调用数也是 220。k=8 的失败同样跑到 220 步，但只有 28 次调用。
- 最干净的受控对照：task 8 ep4、同一初始状态 `ad6e3d768b047e19`，
  k=8 成功，k=4/k=1 都超时；只能说明执行长度改变了轨迹，不能证明因果。
- 99/100 与 94/100 的 95% Wilson 区间重叠（94.6–99.8% vs 87.5–97.2%），
  不能宣称显著差异。

### 4. 技术难点

- **硬件兼容**：官方 torch 2.2.0 的 CUDA 内核只到 sm_90，Blackwell sm_120
  会报 `no kernel image is available`；需要换 torch 2.7.1 + cu128，同时避免
  项目依赖把 torch 降级。
- **checkpoint 加载**：torch 2.7 默认 `weights_only=True`，官方旧 checkpoint
  需要 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。
- **无头渲染**：LIBERO/MuJoCo 需要 EGL；要区分“真正渲染失败”和
  “解释器退出阶段的 EGL 析构告警”。
- **评测语义**：官方 `deque(maxlen=k)` 在 k<8 时保留最后 k 个动作，不是前 k 个；
  如果不看代码，k<8 的实验会悄悄变成另一个实验。

### 5. 边界与局限

- 10 tasks × 10 episodes，单 seed、单 GPU，属于小规模工程评测，不是论文复现。
- 没有训练/微调；模型是作者发布的 LIBERO-Spatial checkpoint。
- 没有逐帧末端位姿/夹爪/物体位姿日志，失败原因只能给现象和待验证猜测。
- Blackwell + torch 2.7.1 与官方 A100 + torch 2.2.0 不同，绝对值不能和论文表格逐项对比。

---

## C. 面试 30 秒口述

“我在一张 Blackwell 显卡上部署了 OpenVLA-OFT 7B 和 LIBERO，做了
LIBERO-Spatial 10×10 的闭环评测，基线 99/100。然后固定模型、任务、初始状态
和 seed，只改变动作块执行长度 k=1/4/8：k=4 也是 99/100，但模型调用约 2 倍；
k=1 是 94/100，调用约 8.4 倍。结论是更频繁重规划没有提升这组样本的成功率，
主要代价是调用次数。过程中还发现官方 deque 在 k<8 时执行的是动作块最后 k 个，
我在自己的 runner 里改成了前 k 个。没有训练，也没有宣称论文复现。”

---

## D. 不要写 / 不要说的

- 不要写“训练了 OpenVLA-OFT”或“做了 LoRA 微调”——没有。
- 不要写“复现论文 99%”——论文是全量评测，这里是 10×10 小规模。
- 不要写“提出了新算法”——贡献是环境适配、评测工具、受控对照和失败分析。
- 不要把官方 `run_libero_eval.py` 说成自己写的；要明确复用与新增。
- 不要说“k=4 导致失败”或“失败原因是抓取滑落”——没有因果/中间位姿证据。
