#!/usr/bin/env bash
# Phase 1: install OS rendering libs, the pinned Python environment, and LIBERO.
# FlashAttention is intentionally skipped: upstream SETUP.md says it is for training,
# and this project only runs official inference checkpoints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

# Blackwell (sm_120) cannot run the upstream torch==2.2.0 pin. Delegate before
# creating the log tee so the child owns its own log file.
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)"
if [ -n "$COMPUTE_CAP" ] && awk -v c="$COMPUTE_CAP" 'BEGIN { exit !(c + 0 >= 12.0) }'; then
  echo "Detected Blackwell compute capability ${COMPUTE_CAP}; using scripts/02_install_blackwell.sh"
  exec bash "$SCRIPT_DIR/02_install_blackwell.sh" "$@"
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$PROJECT_ROOT/logs/02_install_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

source /etc/network_turbo 2>/dev/null || true

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
  exit 2
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
echo "==== 4. PyTorch 2.2.0 (CUDA 12.1 wheels) ===="
"$PY" -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

echo
echo "==== 5. openvla-oft editable install (pulls pinned deps from pyproject.toml) ===="
"$PY" -m pip install -e "$OPENVLA_OFT_ROOT"

echo
echo "==== 6. LIBERO editable install + robot requirements ===="
"$PY" -m pip install -e "$LIBERO_ROOT"
"$PY" -m pip install -r "$OPENVLA_OFT_ROOT/experiments/robot/libero/libero_requirements.txt"
# robosuite 1.4.1 only says mujoco>=2.3.0; pin the version LIBERO was tested
# against instead of silently taking a future mujoco major release.
"$PY" -m pip install "mujoco==2.3.7" "opencv-python==4.10.0.84"

echo
echo "==== 7. Import and CUDA checks ===="
"$PY" - <<'PY'
import os
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"

import transformers
print("transformers:", transformers.__version__)

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
echo "==== 8. Dependency consistency (informational) ===="
"$PY" -m pip check || true

echo
echo "NOTE: flash-attn is NOT installed. It is only needed for training in SETUP.md;"
echo "the official inference path loads the HF checkpoint directly and does not require it."
echo
echo "OK. Installation finished. Log saved to $LOG"
