#!/usr/bin/env python3
"""Run exactly one LIBERO task and record a reproducible phase-1 manifest.

This is a project-side runner. It imports the official OpenVLA-OFT evaluation
functions without modifying them, so the upstream code stays attributable.

Official source:
  https://github.com/moojink/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/root/autodl-tmp/project")).resolve()
OPENVLA_OFT_ROOT = Path(os.environ.get("OPENVLA_OFT_ROOT", PROJECT_ROOT / "openvla-oft")).resolve()
sys.path.insert(0, str(OPENVLA_OFT_ROOT))

# Must be set before importing LIBERO / MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    GenerateConfig,
    initialize_model,
    run_task,
    setup_logging,
    validate_config,
)
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere  # noqa: E402
from libero.libero import benchmark  # noqa: E402


def str2bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def git_commit(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def gpu_metrics() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_gb": round(props.total_memory / 2**30, 3),
        "gpu_capability": f"{props.major}.{props.minor}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", default=os.environ.get("MODEL_ID", ""))
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_trials_per_task", type=int, default=1)
    parser.add_argument("--num_open_loop_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--env_img_res", type=int, default=256)
    parser.add_argument("--center_crop", type=str2bool, default=True)
    parser.add_argument("--local_log_dir", default=str(PROJECT_ROOT / "results" / "phase1" / "logs"))
    parser.add_argument("--run_id_note", default="phase1-smoke")
    args = parser.parse_args()

    cfg = GenerateConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=args.center_crop,
        num_open_loop_steps=args.num_open_loop_steps,
        task_suite_name=args.task_suite_name,
        num_steps_wait=args.num_steps_wait,
        num_trials_per_task=args.num_trials_per_task,
        env_img_res=args.env_img_res,
        local_log_dir=args.local_log_dir,
        use_wandb=False,
        seed=args.seed,
        run_id_note=args.run_id_note,
    )
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    started = time.perf_counter()
    run_started_epoch = time.time()
    manifest = {
        "status": "started",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "project_root": str(PROJECT_ROOT),
        "openvla_oft_root": str(OPENVLA_OFT_ROOT),
        "openvla_oft_commit": git_commit(OPENVLA_OFT_ROOT),
        "libero_commit": git_commit(Path(os.environ.get("LIBERO_ROOT", PROJECT_ROOT / "LIBERO"))),
        "model": args.pretrained_checkpoint,
        "task_suite": args.task_suite_name,
        "task_id": args.task_id,
        "num_trials_per_task": args.num_trials_per_task,
        "num_open_loop_steps": args.num_open_loop_steps,
        "seed": args.seed,
        "config": dataclasses.asdict(cfg),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
            "HF_HOME": os.environ.get("HF_HOME"),
        },
        **gpu_metrics(),
    }

    manifest_path = PROJECT_ROOT / "results" / "phase1" / "run_manifest.json"
    log_file = None
    try:
        print("==== Loading official OpenVLA-OFT checkpoint ====", flush=True)
        load_start = time.perf_counter()
        model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
        torch.cuda.synchronize()
        manifest["model_load_seconds"] = round(time.perf_counter() - load_start, 3)
        manifest["model_load_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        manifest["model_load_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        torch.cuda.reset_peak_memory_stats()

        resize_size = get_image_resize_size(cfg)
        log_file, local_log_filepath, run_id = setup_logging(cfg)
        manifest["run_id"] = run_id
        manifest["local_log_filepath"] = local_log_filepath

        task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
        task = task_suite.get_task(args.task_id)
        manifest["task_description"] = task.language
        manifest["num_tasks_in_suite"] = task_suite.n_tasks

        print(f"==== Running task {args.task_id}: {task.language} ====", flush=True)
        eval_start = time.perf_counter()
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            args.task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            0,
            0,
            log_file,
        )
        torch.cuda.synchronize()
        manifest["eval_seconds"] = round(time.perf_counter() - eval_start, 3)
        manifest["eval_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        manifest["eval_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        manifest["episodes"] = int(total_episodes)
        manifest["successes"] = int(total_successes)
        manifest["success_rate"] = round(float(total_successes) / total_episodes, 4) if total_episodes else 0.0
        video_dir = Path.cwd() / "rollouts"
        manifest["video_paths"] = [
            str(p.resolve())
            for p in sorted(video_dir.rglob("*.mp4"))
            if p.stat().st_mtime >= run_started_epoch - 5
        ]
        manifest["status"] = "ok"
    except Exception:
        manifest["status"] = "error"
        manifest["traceback"] = traceback.format_exc()
        print(manifest["traceback"], file=sys.stderr)
        raise
    finally:
        manifest["wall_seconds"] = round(time.perf_counter() - started, 3)
        if log_file is not None:
            log_file.flush()
            log_file.close()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if manifest.get("run_id"):
            run_copy = manifest_path.parent / f"run_manifest_{manifest['run_id']}.json"
            run_copy.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"PHASE1_MANIFEST_PATH: {run_copy}", flush=True)
        print(f"PHASE1_MANIFEST_PATH: {manifest_path}", flush=True)
        print("PHASE1_RESULT_JSON: " + json.dumps(manifest, ensure_ascii=False), flush=True)

    return 0 if manifest.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
