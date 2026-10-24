#!/usr/bin/env bash
# Shared paths and pinned versions for the OpenVLA-OFT + LIBERO project.
# Source this file from every other script: source "$PROJECT_ROOT/env.sh"

export PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/project}"
export OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-$PROJECT_ROOT/openvla-oft}"
export LIBERO_ROOT="${LIBERO_ROOT:-$PROJECT_ROOT/LIBERO}"
export ENV_PREFIX="${ENV_PREFIX:-/root/autodl-tmp/envs/openvla-oft}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pip_cache}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$PROJECT_ROOT/.libero}"
export PYTHONPATH="$LIBERO_ROOT:$OPENVLA_OFT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# torch>=2.6 defaults torch.load(weights_only=True); the official OpenVLA-OFT
# checkpoints were saved with the older pickle format. This flag preserves the
# old loading behavior without patching upstream code.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Pinned upstream revisions. Do not silently upgrade.
export OPENVLA_OFT_REPO="https://github.com/moojink/openvla-oft.git"
export OPENVLA_OFT_COMMIT="e4287e94541f459edc4feabc4e181f537cd569a8"
export LIBERO_REPO="https://github.com/Lifelong-Robot-Learning/LIBERO.git"
export LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"
export MODEL_ID="moojink/openvla-7b-oft-finetuned-libero-spatial"

mkdir -p "$PROJECT_ROOT"/{logs,results/phase1,models,datasets} "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$LIBERO_CONFIG_PATH"

# LIBERO prompts interactively on first import if this file is absent. Write it
# non-interactively so evaluation scripts can run unattended.
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $LIBERO_ROOT/libero/libero
bddl_files: $LIBERO_ROOT/libero/libero/bddl_files
init_states: $LIBERO_ROOT/libero/libero/init_files
datasets: $PROJECT_ROOT/datasets
assets: $LIBERO_ROOT/libero/libero/assets
EOF
fi

# Put the project conda env first when it exists.
if [ -d "$ENV_PREFIX/bin" ]; then
  export PATH="$ENV_PREFIX/bin:$PATH"
fi
