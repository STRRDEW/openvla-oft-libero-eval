#!/usr/bin/env bash
# Phase 3: compare how many predicted actions are executed before re-observing.
#
# The model always predicts an 8-action chunk. This experiment executes the
# FIRST k actions, then re-observes. k=8 is the phase-2 baseline and is reused,
# not rerun, so the same 100 episodes are not paid for twice.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$BUNDLE_DIR/env.sh"
# Loading the HF checkpoint can hang without the AutoDL GitHub/HF accelerator.
source /etc/network_turbo 2>/dev/null || true
cd "$PROJECT_ROOT"

for k in 1 4; do
  out="$PROJECT_ROOT/results/phase3/k${k}"
  if [ -f "$out/summary.json" ]; then
    echo "SKIP k=$k: $out/summary.json already exists"
    continue
  fi
  mkdir -p "$out"
  echo "==== k=$k start $(date -Is) ===="
  python tools/run_eval.py \
    --task_suite_name libero_spatial \
    --task_ids all \
    --num_trials_per_task 10 \
    --num_open_loop_steps "$k" \
    --seed 7 \
    --output_dir "$out" \
    --run_id_note "ablation-k${k}"
  echo "==== k=$k done $(date -Is) ===="
done
