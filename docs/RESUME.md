# 简历描述（只写实际完成的工作）

> 真实范围：官方 OpenVLA-OFT 7B 权重 + LIBERO-Spatial 仿真，纯推理/评测，
> **没有训练、没有微调、没有 LoRA 训练**。不要把这些写成训练经历。

## 中文简历（项目条目）

**基于 OpenVLA-OFT 与 LIBERO 的语言条件机械臂闭环评测**（个人项目）

- 在 AutoDL 单卡 NVIDIA RTX 6000D（Blackwell sm_120，84GB）上部署
  OpenVLA-OFT 7B 与 LIBERO；解决官方 PyTorch 2.2 无 sm_120 内核、
  torch 2.7 checkpoint 加载、LIBERO EGL 无头渲染、TensorFlow/protobuf
  版本冲突等问题；固定 openvla-oft `e4287e9` 与 LIBERO `8f1084e`。
- 实现逐 episode 闭环评测工具：记录任务、初始状态哈希、seed、成功标志、
  环境/policy 步数、模型调用次数、`torch.cuda.synchronize()` 同步后的推理延迟、
  显存峰值、总墙钟和视频路径；在 LIBERO-Spatial 10 个任务上完成
  10×10 基线，成功率 **99/100**，推理峰值显存 **15.18GB**。
- 完成“动作块执行长度”受控对照：模型固定输出 8 步动作，分别执行前
  1/4/8 步再重新观察；保持模型、任务、初始状态、seed 和预处理一致，
  对比成功率、模型调用次数、单次推理延迟和总耗时。k=4 成功率 **99/100**、
  平均 26.51 次模型调用/episode、平均墙钟 9.51s；k=1 为 **94/100**、
  111.76 次调用/episode、20.28s。k=4 与基线同成功率，但调用约 2 倍。
- 定位官方评测循环中 `deque(maxlen=k)` 在 k<8 时保留动作块**最后 k 个**、
  而非前 k 个的行为；在项目 runner 中改为显式执行前 k 个动作，使对照
  语义与“执行前 k 个再重新观察”一致（k=8 基线不受影响）。
- 产出真实 rollout 视频、成功/失败帧、逐 episode JSONL、汇总 CSV/图表、
  README、环境记录和失败分析；失败案例包括基线 task 9、k=1 task 1，以及
  **同一初始状态下 k=8 成功而 k=4/k=1 失败**的受控对照，明确区分
  “观察到的现象”与“未验证的原因猜测”。

**技术栈**：Python、PyTorch、CUDA、LIBERO、MuJoCo、robosuite、
HuggingFace Transformers、Linux、Shell、OpenCV。

## 英文简历（可选）

**Language-Conditioned Robotic Manipulation Evaluation with OpenVLA-OFT and LIBERO** (Personal Project)

- Deployed OpenVLA-OFT 7B and LIBERO on a single NVIDIA RTX 6000D
  (Blackwell sm_120, 84GB); resolved sm_120/PyTorch incompatibility,
  torch 2.7 checkpoint loading, EGL headless rendering, and
  TensorFlow/protobuf conflicts; pinned openvla-oft `e4287e9` and
  LIBERO `8f1084e`.
- Built an instrumented closed-loop evaluator recording task, initial-state
  hash, seed, success, environment/policy steps, model calls,
  CUDA-synchronized inference latency, peak VRAM, wall time, and video path.
  Achieved **99/100** success on a 10-task × 10-episode LIBERO-Spatial baseline
  with **15.18GB** peak inference VRAM.
- Ran a controlled action-chunk execution ablation (execute first 1/4/8 of the
  model's 8-action chunk before re-observing), holding model, tasks, initial
  states, seed, and preprocessing fixed. k=4: **99/100**, 26.51 model calls and
  9.51s wall time per episode; k=1: **94/100**, 111.76 calls and 20.28s.
  k=4 matched the baseline success rate at roughly 2x the model calls.
- Found and documented that the official eval loop's `deque(maxlen=k)` keeps the
  **last** k actions for k<8 rather than the first k; the project runner
  explicitly executes the first k actions to match the intended ablation
  semantics (k=8 baseline unaffected).
- Produced real rollout videos, success/failure frames, per-episode JSONL,
  aggregate CSV/plots, README, environment record, and failure analysis with
  observed symptoms separated from unverified hypotheses, including a matched
  initial state where k=8 succeeds but k=4/k=1 fail.

**Stack**: Python, PyTorch, CUDA, LIBERO, MuJoCo, robosuite,
HuggingFace Transformers, Linux, Shell, OpenCV.

## 一句话项目描述（简历顶部/自我介绍）

> 在 Blackwell 单卡上部署 OpenVLA-OFT 7B + LIBERO，完成 LIBERO-Spatial
> 10×10 闭环评测（基线 99/100），并做了动作块执行长度 1/4/8 的受控对照；
> 工作重点是评测工具、环境适配、失败分析和可复现文档，不是算法创新。

## 面试中必须主动澄清的三点

1. **没有训练**：只加载官方微调权重做推理；没有 LoRA 训练、没有梯度更新。
2. **不是论文复现**：10 tasks × 10 episodes 是小规模工程评测；官方论文是更大规模。
3. **不是原创算法**：官方模型、仿真、评测函数都是复用的；新增的是
   runner、日志、对照设计、分析、环境适配和文档。
