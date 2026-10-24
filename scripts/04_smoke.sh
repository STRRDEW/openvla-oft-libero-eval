#!/usr/bin/env bash
# Phase 1: run one LIBERO-Spatial task for one episode, save video/log/manifest.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

STAMP="$(date +%Y%m%d-%H%M%S)"
PHASE1="$PROJECT_ROOT/results/phase1"
RUN_DIR="$PHASE1/run"
mkdir -p "$PHASE1/logs" "$RUN_DIR"

# Keep stdout in a regular file so a detached run does not hold the SSH pipe open.
SMOKE_LOG="$RUN_DIR/smoke.log"
exec >> "$SMOKE_LOG" 2>&1

source /etc/network_turbo 2>/dev/null || true
PY="$ENV_PREFIX/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: Python env missing. Run scripts/02_install.sh first."
  exit 2
fi

cd "$RUN_DIR"

echo "==== GPU before run ===="
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader

# Lightweight GPU sampler; killed on exit.
( while true; do
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader || true
    sleep 1
  done ) > "$RUN_DIR/gpu_monitor.csv" 2>&1 &
MONITOR_PID=$!
cleanup () { kill "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT

set +e
timeout 3600 "$PY" "$PROJECT_ROOT/tools/run_single_task.py" \
  --pretrained_checkpoint "$MODEL_ID" \
  --task_suite_name libero_spatial \
  --task_id 0 \
  --num_trials_per_task 1 \
  --num_open_loop_steps 8 \
  --seed 7 \
  --num_steps_wait 10 \
  --env_img_res 256 \
  --center_crop True \
  --local_log_dir "$PHASE1/logs" \
  --run_id_note "phase1-smoke"
STATUS=$?
set -e

cleanup
trap - EXIT

echo "==== GPU after run ===="
nvidia-smi

echo "==== Artifacts ===="
find "$RUN_DIR" -maxdepth 3 -type f -printf '%p\t%s bytes\n' | sort
find "$RUN_DIR/rollouts" -type f -name '*.mp4' -printf '%p\t%s bytes\n' 2>/dev/null || true

if [ -f "$PHASE1/run_manifest.json" ]; then
  cp "$PHASE1/run_manifest.json" "$PHASE1/run_manifest_${STAMP}.json"
  "$PY" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["PROJECT_ROOT"]) / "results" / "phase1" / "run_manifest.json"
m = json.loads(p.read_text())
keys = ["status","task_description","episodes","successes","success_rate",
        "model_load_seconds","eval_seconds","wall_seconds",
        "model_load_peak_allocated_gb","eval_peak_allocated_gb","eval_peak_reserved_gb",
        "gpu_name","gpu_total_memory_gb","num_open_loop_steps","seed"]
print("PHASE1_SUMMARY")
for k in keys:
    print(f"  {k}: {m.get(k)}")
print(f"  manifest: {p}")
PY
fi

ARCHIVE="$PROJECT_ROOT/results/phase1_smoke_${STAMP}.tar.gz"
tar -czf "$ARCHIVE" -C "$PROJECT_ROOT/results" phase1
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
cat "${ARCHIVE}.sha256"

echo
if [ "$STATUS" -eq 0 ]; then
  echo "OK: phase-1 smoke test finished. Download the tar.gz before releasing the instance."
else
  echo "FAILED: exit status $STATUS. Inspect $RUN_DIR/smoke.log and the manifest traceback."
fi
exit "$STATUS"
