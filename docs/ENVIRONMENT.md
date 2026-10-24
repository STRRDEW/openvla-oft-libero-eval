# 环境记录（实际运行）

本文件记录本项目真实使用的机器、版本、固定 commit、关键命令和已知告警。
它对应 2026-10-24 在 AutoDL 上的一次实际运行。

## 1. 机器

| 项目 | 实际值 |
|---|---|
| 平台 | AutoDL 容器实例 |
| 容器 hostname | `autodl-container-2urata8cw5-856ba703` |
| GPU | `NVIDIA RTX 6000D` |
| 显存 | `85,651 MiB`（`nvidia-smi`；torch 报告可用约 `83.05 GiB`） |
| compute capability | `12.0`（Blackwell，sm_120） |
| 驱动 | `595.71.05` |
| 驱动报告的 CUDA | `13.2` |
| 数据盘 | `/dev/md0` 50GB，挂载 `/root/autodl-tmp` |
| 系统 | Ubuntu 22.04.1 LTS |
| CPU / 内存 | 208 vCPU / 约 1.0 TiB |

## 2. 固定版本

### 仓库

```text
openvla-oft  e4287e94541f459edc4feabc4e181f537cd569a8
LIBERO       8f1084e3132a39270c3a13ebe37270a43ece2a01
```

### 权重

```text
repo: moojink/openvla-7b-oft-finetuned-libero-spatial
snapshot: 6d0231af0e48c5985f1ff86908f4674b84bc049b
```

关键文件：

```text
config.json                               60,711 bytes
dataset_statistics.json                    2,957 bytes
action_head--150000_checkpoint.pt        302,242,674 bytes
proprio_projector--150000_checkpoint.pt   67,275,256 bytes
model-00001-of-00004.safetensors       4,925,122,448 bytes
model-00002-of-00004.safetensors       4,947,392,496 bytes
model-00003-of-00004.safetensors       4,947,417,456 bytes
model-00004-of-00004.safetensors         262,668,432 bytes
```

### Python 包

```text
Python                    3.10.22
torch                     2.7.1+cu128
torchvision               0.22.1+cu128
torchaudio                2.7.1+cu128
CUDA runtime              12.8
transformers              4.40.1  (moojink/transformers-openvla-oft fork)
tensorflow                2.15.0
tensorflow-datasets       4.9.3
tensorflow-metadata       1.14.0
protobuf                  4.25.9
accelerate                0.30.1
diffusers                 0.30.3
huggingface-hub           0.23.4
peft                      0.11.1
timm                      0.9.10
tokenizers                0.19.1
numpy                     1.26.4
robosuite                 1.4.1
mujoco                    2.3.7
libero                    0.1.0 (editable)
ImageIO                   2.38.0
```

## 3. 环境变量

`autodl_bundle/env.sh` 的核心内容：

```bash
export PROJECT_ROOT=/root/autodl-tmp/project
export OPENVLA_OFT_ROOT=$PROJECT_ROOT/openvla-oft
export LIBERO_ROOT=$PROJECT_ROOT/LIBERO
export ENV_PREFIX=/root/autodl-tmp/envs/openvla-oft
export HF_HOME=/root/autodl-tmp/hf_cache
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export TMPDIR=/root/autodl-tmp/tmp

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

export LIBERO_CONFIG_PATH=$PROJECT_ROOT/.libero
export PYTHONPATH=$LIBERO_ROOT:$OPENVLA_OFT_ROOT${PYTHONPATH:+:$PYTHONPATH}
```

`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 是 torch 2.7 的兼容设置：官方 checkpoint
是用旧版 torch 保存的，torch 2.7 默认 `weights_only=True` 会拒绝加载。

## 4. 安装命令（Blackwell 路径）

```bash
source /etc/network_turbo   # 访问 GitHub/HuggingFace 前必须开
bash scripts/01_clone.sh
bash scripts/02_install_blackwell.sh
bash scripts/03_model.sh
```

`02_install_blackwell.sh` 的关键步骤：

1. apt 安装 `libegl1 libgl1 libglib2.0-0 libgles2 libglew2.2 libosmesa6 libsm6 libxext6 libxrender1 ffmpeg`。
2. 在 `/root/autodl-tmp/envs/openvla-oft` 建 Python 3.10 conda 环境。
3. 安装 `torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128`。
4. 先做 1024×1024 CUDA matmul；失败则停止，不下载 15GB 权重。
5. `pip install -e openvla-oft --no-deps`，避免官方 `pyproject.toml` 把 torch 降回 2.2.0。
6. 手动安装其余依赖；`transformers` fork 用 `--no-deps`，`dlimp` 正常安装。
7. `pip install -e LIBERO`，再装 `libero_requirements.txt`。
8. import 检查。

## 5. 已知告警与处理

### 5.1 TensorFlow 的 sm_120 告警

```text
TensorFlow binary is optimized to use available CPU instructions...
Unable to register cuDNN/cuFFT/cuBLAS factory...
```

TensorFlow 2.15 没有 sm_120 的预编译 CUDA kernel，可能提示从 PTX JIT。
本项目推理路径不使用 TensorFlow 计算图，未观察到对评测结果的影响；保留为环境告警。

### 5.2 EGL 退出阶段报错

部分运行在解释器退出时出现：

```text
EGL.eglMakeCurrent ... EGL_NOT_INITIALIZED
```

进程退出码仍为 0，视频和成功标志正常。现象已观察，原因未完全验证；
不能据此声称 EGL 已完全修复。

### 5.3 `pip check`

`tensorflow-metadata==1.14.0` 声明的 protobuf 上限与实际安装的
`protobuf==4.25.9` 不一致，`pip check` 会警告。实测 TensorFlow 2.15、
`tensorflow_datasets`、`dlimp` 均可导入并运行。保留该警告，不盲目升级。

### 5.4 网络

AutoDL 上 GitHub/HuggingFace 直连可能卡住。加载模型前执行：

```bash
source /etc/network_turbo
```

否则 `initialize_model` 可能在 HF 连接上阻塞，GPU 已分配但进程不前进。

## 6. 结果目录

```text
results/
  phase1/                  # 单任务 smoke
  phase2/baseline/         # k=8, 10 tasks x 10 episodes
  phase3/k1/               # k=1
  phase3/k4/               # k=4
  analysis/baseline_k8/    # 基线单独分析
  analysis/all/            # 三组对比
```

每个 run 目录包含：

```text
episodes.jsonl     # 每行一个 episode
summary.json       # 聚合指标
summary.csv        # 表格
run_manifest.json  # 环境/命令/commit/时间
rollouts/          # 真实 MP4
logs/              # 官方文本日志
```
