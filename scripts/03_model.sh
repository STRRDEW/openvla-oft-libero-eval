#!/usr/bin/env bash
# Phase 1: download the official LIBERO-Spatial checkpoint into the HF cache.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$PROJECT_ROOT/logs/03_model_${STAMP}.log"
# Write straight to the log file. A process-substitution tee would keep the
# SSH channel's stdout pipe open after the script is detached.
exec >> "$LOG" 2>&1

source /etc/network_turbo 2>/dev/null || true
PY="$ENV_PREFIX/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: Python env missing. Run scripts/02_install.sh first."
  exit 2
fi

echo "Free space before download:"
df -h "$PROJECT_ROOT"

echo
echo "==== Downloading $MODEL_ID ===="
if ! "$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    cache_dir=os.path.join(os.environ["HF_HOME"], "hub"),
    max_workers=4,
)
PY
then
  echo
  echo "Direct HuggingFace download failed; retrying through hf-mirror.com."
  HF_ENDPOINT=https://hf-mirror.com "$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    cache_dir=os.path.join(os.environ["HF_HOME"], "hub"),
    max_workers=4,
)
PY
fi

echo
echo "==== Verify expected files ===="
"$PY" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    cache_dir=os.path.join(os.environ["HF_HOME"], "hub"),
    local_files_only=True,
))
print("snapshot:", root)
required = [
    "config.json",
    "dataset_statistics.json",
    "proprio_projector--150000_checkpoint.pt",
    "action_head--150000_checkpoint.pt",
]
missing = [name for name in required if not (root / name).exists()]
weights = sorted(root.glob("*.safetensors")) + sorted(root.glob("*.bin"))
print("weight shards:", len(weights))
for p in weights:
    print(f"  {p.name}  {p.stat().st_size / 2**30:.2f} GiB")
if missing or not weights:
    raise SystemExit(f"ERROR: missing files={missing}, weight_shards={len(weights)}")
print("All expected files are present.")
PY

echo
echo "==== Size on disk ===="
du -sh "$HF_HOME/hub" "$PROJECT_ROOT"
echo
echo "OK. Log saved to $LOG"
