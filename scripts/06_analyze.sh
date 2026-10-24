#!/usr/bin/env bash
# Compare the phase-2 baseline (k=8) with the phase-3 k=1 and k=4 runs.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$BUNDLE_DIR/env.sh"
cd "$PROJECT_ROOT"

python tools/analyze_results.py \
  --run baseline_k8=results/phase2/baseline \
  --run k1=results/phase3/k1 \
  --run k4=results/phase3/k4 \
  --out results/analysis/all
