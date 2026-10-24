#!/usr/bin/env bash
# Phase 0: clone exactly the two required upstream repositories and pin commits.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$PROJECT_ROOT/logs/01_clone_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

source /etc/network_turbo 2>/dev/null || true

clone_pinned () {
  local repo="$1" dir="$2" commit="$3" name="$4"
  echo
  echo "==== $name ===="
  if [ -d "$dir/.git" ]; then
    echo "Directory already exists; verifying remote instead of re-cloning."
    local actual_remote
    actual_remote="$(git -C "$dir" remote get-url origin)"
    echo "remote: $actual_remote"
    if [ "$actual_remote" != "$repo" ]; then
      echo "ERROR: $dir has unexpected origin: $actual_remote"
      exit 2
    fi
  else
    git clone "$repo" "$dir"
  fi

  git -C "$dir" fetch --all --tags
  git -C "$dir" checkout --detach "$commit"
  local head
  head="$(git -C "$dir" rev-parse HEAD)"
  echo "checked out: $head"
  if [ "$head" != "$commit" ]; then
    echo "ERROR: expected $commit, got $head"
    exit 3
  fi
  git -C "$dir" status --short
}

clone_pinned "$OPENVLA_OFT_REPO" "$OPENVLA_OFT_ROOT" "$OPENVLA_OFT_COMMIT" "openvla-oft"
clone_pinned "$LIBERO_REPO" "$LIBERO_ROOT" "$LIBERO_COMMIT" "LIBERO"

echo
echo "==== Repository sizes ===="
du -sh "$OPENVLA_OFT_ROOT" "$LIBERO_ROOT"

echo
echo "==== Pinned revisions ===="
{
  echo "openvla-oft $OPENVLA_OFT_COMMIT $(git -C "$OPENVLA_OFT_ROOT" log -1 --format=%cI)"
  echo "LIBERO      $LIBERO_COMMIT $(git -C "$LIBERO_ROOT" log -1 --format=%cI)"
} | tee "$PROJECT_ROOT/results/phase1/repos.txt"

echo
echo "OK. Only these two repositories are needed for inference."
echo "Do NOT clone the LIBERO dataset or the HuggingFace model as a git repo."
echo "Log saved to $LOG"
