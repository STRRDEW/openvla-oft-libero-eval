#!/usr/bin/env bash
# Phase 0: verify the AutoDL instance before spending time on installation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$PROJECT_ROOT/logs/00_check_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "OpenVLA-OFT / LIBERO environment check"
echo "time: $(date -Iseconds)"
echo "host: $(hostname)  user: $(whoami)  pwd: $(pwd)"
echo "============================================================"

echo
echo "---- OS / driver / CUDA ----"
cat /etc/os-release | head -5 || true
nvidia-smi || true
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader || true
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)"
echo "Compute capability: ${COMPUTE_CAP:-unknown}"
if [ -n "$COMPUTE_CAP" ] && awk -v c="$COMPUTE_CAP" 'BEGIN { exit !(c + 0 >= 12.0) }'; then
  echo "WARNING: Blackwell (sm_120) detected. The upstream torch==2.2.0 pin has no sm_120 kernels."
  echo "         scripts/02_install.sh will automatically delegate to 02_install_blackwell.sh"
  echo "         (PyTorch 2.7.1 + CUDA 12.8). Do not install torch 2.2.0 manually."
fi
nvcc --version 2>/dev/null | tail -3 || echo "nvcc not on PATH (not fatal for inference)"
echo

echo "---- CPU / RAM ----"
nproc
free -h
echo

echo "---- Disk ----"
df -h / /root /root/autodl-tmp 2>/dev/null || df -h
echo "Free space on /root/autodl-tmp (GB):"
df -BG /root/autodl-tmp | awk 'NR==2 {gsub("G","",$4); print $4}'
echo

echo "---- Toolchain ----"
git --version || true
ffmpeg -version 2>/dev/null | head -1 || echo "ffmpeg missing (install script will add it)"
for c in /root/miniconda3 /opt/conda /root/anaconda3; do
  if [ -f "$c/etc/profile.d/conda.sh" ]; then
    echo "conda candidate: $c"
  fi
done
python3 --version || true
echo

echo "---- Network (GitHub / HuggingFace) ----"
source /etc/network_turbo 2>/dev/null || echo "network_turbo not available"
for url in https://github.com https://huggingface.co; do
  code="$(curl -L -s -o /dev/null -w '%{http_code}' --max-time 20 "$url" || true)"
  echo "$url -> HTTP $code"
done
echo

echo "---- Existing GPU processes ----"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
echo

echo "---- Preconditions ----"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This is not a usable GPU instance."
  exit 2
fi
GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
echo "Detected GPU memory: ${GPU_MEM_MB} MiB"
if [ "${GPU_MEM_MB:-0}" -lt 20000 ]; then
  echo "ERROR: need >= 20 GB VRAM for the 7B OpenVLA-OFT inference path."
  exit 3
fi
FREE_GB="$(df -BG /root/autodl-tmp | awk 'NR==2 {gsub("G","",$4); print $4}')"
echo "Free data-disk space: ${FREE_GB} GB"
if [ "${FREE_GB:-0}" -lt 30 ]; then
  echo "ERROR: need at least 30 GB free on /root/autodl-tmp; 100 GB is recommended."
  exit 4
fi
echo
echo "OK. Log saved to $LOG"
