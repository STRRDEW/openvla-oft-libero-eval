#!/usr/bin/env bash
# Phase 1 (Blackwell / sm_120): install a PyTorch 2.7.1 + CUDA 12.8 environment.
#
# Why this exists:
#   The upstream pyproject.toml pins torch==2.2.0. That wheel has no sm_120
#   kernels, so on an RTX 6000D / RTX 50-series Blackwell GPU it fails with
#   "no kernel image is available for execution on the device".
#   We therefore install a Blackwell-capable torch first, then install
#   openvla-oft with --no-deps and install its runtime dependencies manually.
#
# FlashAttention is intentionally skipped: upstream SETUP.md says it is for
# training, and this project only runs the official inference checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$PROJECT_ROOT/logs/02_install_blackwell_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

if [ "${USE_NETWORK_TURBO:-0}" = "1" ]; then
  source /etc/network_turbo 2>/dev/null || true
fi

echo "==== 0. Confirm this is a Blackwell (sm_120) GPU ===="
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)"
echo "Detected compute capability: ${COMPUTE_CAP:-unknown}"
if [ -z "$COMPUTE_CAP" ]; then
  echo "ERROR: could not read compute capability from nvidia-smi."
  exit 2
fi
if ! awk -v c="$COMPUTE_CAP" 'BEGIN { exit !(c + 0 >= 12.0) }'; then
  echo "ERROR: this script is for Blackwell (>= 12.0). Use scripts/02_install.sh instead."
  exit 3
fi
nvidia-smi

echo
echo "==== 1. Headless rendering / video packages ===="
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  libegl1 libgl1 libglib2.0-0 libgles2 libglew2.2 libosmesa6 \
  libsm6 libxext6 libxrender1 ffmpeg ca-certificates

echo
echo "==== 2. Conda environment on the data disk ===="
CONDA_BASE=""
for c in /root/miniconda3 /opt/conda /root/anaconda3; do
  if [ -f "$c/etc/profile.d/conda.sh" ]; then CONDA_BASE="$c"; break; fi
done
if [ -z "$CONDA_BASE" ]; then
  echo "ERROR: conda not found. Use an AutoDL PyTorch image that includes Miniconda."
  exit 4
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  conda create -p "$ENV_PREFIX" python=3.10 -y
else
  echo "Reusing existing env: $ENV_PREFIX"
fi
PY="$ENV_PREFIX/bin/python"
"$PY" --version

echo
echo "==== 3. Python build/runtime basics ===="
"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install packaging ninja

echo
echo "==== 4. PyTorch 2.7.1 (CUDA 12.8 wheels, supports sm_120) ===="
if [ "${SKIP_TORCH:-0}" = "1" ]; then
  echo "SKIP_TORCH=1: reusing already-installed torch"
  "$PY" -c "import torch; print('torch:', torch.__version__)"
else
  "$PY" -m pip install \
    torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple
fi

echo
echo "==== 5. CUDA matmul smoke test (must pass before downloading weights) ===="
"$PY" - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))

x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
y = x @ x
torch.cuda.synchronize()
print("matmul ok, sum =", float(y.sum()))
PY

echo
echo "==== 6. openvla-oft editable install (NO dependency resolution) ===="
# --no-deps is essential: the upstream pyproject.toml would otherwise
# downgrade torch to 2.2.0 and reintroduce the sm_120 failure.
"$PY" -m pip install -e "$OPENVLA_OFT_ROOT" --no-deps

echo
echo "==== 7. Runtime dependencies (torch pins removed) ===="
"$PY" -m pip install \
  "numpy==1.26.4" \
  "accelerate==0.30.1" \
  "draccus==0.8.0" \
  "einops" \
  "huggingface_hub==0.23.4" \
  "json-numpy" \
  "jsonlines" \
  "matplotlib" \
  "peft==0.11.1" \
  "protobuf<5" \
  "rich" \
  "sentencepiece==0.1.99" \
  "timm==0.9.10" \
  "tokenizers==0.19.1" \
  "wandb" \
  "tensorflow==2.15.0" \
  "tensorflow_datasets==4.9.3" \
  "tensorflow_graphics==2021.12.3" \
  "diffusers==0.30.3" \
  "imageio" \
  "mujoco==2.3.7" \
  "opencv-python==4.10.0.84" \
  "uvicorn" \
  "fastapi"

echo
echo "==== 8. openvla-oft's transformers fork (NO deps, then compatible deps) ===="
"$PY" -m pip install --no-deps \
  "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git"
"$PY" -m pip install \
  "safetensors==0.4.3" \
  "regex" \
  "pyyaml" \
  "requests" \
  "filelock" \
  "tqdm>=4.27" \
  "fsspec"

echo
echo "==== 9. dlimp (dataset helper; inference does not download datasets) ===="
"$PY" -m pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla"

echo
echo "==== 10. LIBERO editable install + robot requirements ===="
"$PY" -m pip install -e "$LIBERO_ROOT"
"$PY" -m pip install -r "$OPENVLA_OFT_ROOT/experiments/robot/libero/libero_requirements.txt"

echo
echo "==== 11. Import and CUDA checks ===="
"$PY" - <<'PY'
import os
import torch

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"

import transformers
print("transformers:", transformers.__version__)
print("transformers file:", transformers.__file__)

import tensorflow as tf
print("tensorflow:", tf.__version__)

import libero
print("libero:", getattr(libero, "__version__", "editable"))

import robosuite
print("robosuite:", robosuite.__version__)

import mujoco
print("mujoco:", mujoco.__version__)

import imageio
print("imageio:", imageio.__version__)
print("MUJOCO_GL:", os.environ.get("MUJOCO_GL"))
PY

echo
echo "==== 12. Dependency consistency (informational) ===="
echo "NOTE: pip check may report 'openvla-oft requires torch==2.2.0'. That is expected:"
echo "      we intentionally keep torch 2.7.1 so the sm_120 kernels exist."
"$PY" -m pip check || true

echo
echo "==== 13. Reclaim pip cache ===="
"$PY" -m pip cache purge || true
df -h "$PROJECT_ROOT" || true

echo
echo "NOTE: flash-attn is NOT installed. It is only needed for training in SETUP.md;"
echo "the official inference path loads the HF checkpoint directly and does not require it."
echo "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 is set in env.sh for torch>=2.6 checkpoint loading."
echo
echo "OK. Blackwell installation finished. Log saved to $LOG"
