#!/usr/bin/env python3
"""Instrumented LIBERO evaluator for the OpenVLA-OFT project.

Derived from the official evaluation loop:
  https://github.com/moojink/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py

What is new here (not upstream):
  * evaluate a chosen subset of tasks and episodes;
  * write one JSONL record per episode with task, initial-state id, seed,
    success, environment steps, model calls, inference/simulation/wall time;
  * save the per-episode video path;
  * support resuming an interrupted run without re-running completed episodes.

The model, action head, observation preprocessing and action post-processing
are imported from the official code, so the policy path is unchanged.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/root/autodl-tmp/project")).resolve()
OPENVLA_OFT_ROOT = Path(os.environ.get("OPENVLA_OFT_ROOT", PROJECT_ROOT / "openvla-oft")).resolve()
LIBERO_ROOT = Path(os.environ.get("LIBERO_ROOT", PROJECT_ROOT / "LIBERO")).resolve()
sys.path.insert(0, str(OPENVLA_OFT_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    TASK_MAX_STEPS,
    GenerateConfig,
    initialize_model,
    load_initial_states,
    prepare_observation,
    process_action,
    setup_logging,
    validate_config,
)
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    save_rollout_video,
)
from experiments.robot.robot_utils import (  # noqa: E402
    get_action,
    get_image_resize_size,
    set_seed_everywhere,
)
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


def task_ids_from_arg(raw: str, n_tasks: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(n_tasks))
    ids = [int(x) for x in raw.split(",") if x.strip()]
    for tid in ids:
        if tid < 0 or tid >= n_tasks:
            raise ValueError(f"task id {tid} out of range 0..{n_tasks - 1}")
    return ids


def initial_state_digest(initial_state) -> str:
    arr = np.asarray(initial_state, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def run_episode_instrumented(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    initial_state,
    log_file,
    video_idx: int,
    save_video: bool = True,
) -> dict:
    """Same control loop as official run_episode, with measurements added."""
    if not (1 <= cfg.num_open_loop_steps <= 8):
        raise ValueError(
            f"num_open_loop_steps must be in [1, 8] because the model outputs 8 actions; "
            f"got {cfg.num_open_loop_steps}"
        )
    started = time.perf_counter()
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Deliberately not deque(maxlen=k): the official evaluator uses a bounded
    # deque and then extends it with all 8 predicted actions. For k<8 that keeps
    # the LAST k actions, not the FIRST k. This experiment is explicitly about
    # executing the first k actions, so we keep the full chunk and clear it
    # after k executed actions.
    action_queue = deque()
    executed_from_chunk = 0
    max_policy_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    replay_images: list[np.ndarray] = []
    env_steps = 0
    policy_steps = 0
    model_calls = 0
    actions_generated = 0
    inference_seconds = 0.0
    sim_seconds = 0.0
    video_seconds = 0.0
    success = False
    error = None
    video_path = None

    if cfg.num_open_loop_steps != 8:
        print(
            f"WARNING: num_open_loop_steps={cfg.num_open_loop_steps} differs from "
            f"the model's action chunk length (8); only the first "
            f"{cfg.num_open_loop_steps} predicted actions are executed before re-observing.",
            flush=True,
        )

    torch.cuda.reset_peak_memory_stats()
    t = 0
    try:
        while t < max_policy_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                t0 = time.perf_counter()
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                sim_seconds += time.perf_counter() - t0
                env_steps += 1
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if len(action_queue) == 0:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                torch.cuda.synchronize()
                inference_seconds += time.perf_counter() - t0
                model_calls += 1
                actions_generated += len(actions)
                action_queue.extend(actions)
                executed_from_chunk = 0

            action = process_action(action_queue.popleft(), cfg.model_family)
            executed_from_chunk += 1
            if executed_from_chunk >= cfg.num_open_loop_steps:
                action_queue.clear()
            t0 = time.perf_counter()
            obs, _, done, _ = env.step(action.tolist())
            sim_seconds += time.perf_counter() - t0
            env_steps += 1
            policy_steps += 1
            t += 1
            if done:
                success = True
                break
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        if log_file is not None:
            log_file.write("Episode error:\n" + error + "\n")
            log_file.flush()

    if save_video and replay_images:
        try:
            t0 = time.perf_counter()
            video_path = save_rollout_video(
                replay_images,
                video_idx,
                success=success,
                task_description=task_description,
                log_file=log_file,
            )
            video_seconds = time.perf_counter() - t0
        except Exception:
            video_seconds = time.perf_counter() - t0
            print("Video save failed:\n" + traceback.format_exc(), file=sys.stderr, flush=True)

    wall_seconds = time.perf_counter() - started
    return {
        "success": bool(success),
        "error": error,
        "env_steps": env_steps,
        "policy_steps": policy_steps,
        "model_calls": model_calls,
        "actions_generated": actions_generated,
        "inference_seconds": round(inference_seconds, 6),
        "sim_seconds": round(sim_seconds, 6),
        "video_seconds": round(video_seconds, 6),
        "wall_seconds": round(wall_seconds, 6),
        "inference_latency_mean_ms": round(1000 * inference_seconds / model_calls, 3) if model_calls else None,
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 4),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 4),
        "video_path": str(Path(video_path).resolve()) if video_path else None,
        "replay_frames": len(replay_images),
    }


def aggregate(records: list[dict]) -> dict:
    per_task: dict[str, dict] = {}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        grouped[int(rec["task_id"])].append(rec)

    for task_id in sorted(grouped):
        rows = grouped[task_id]
        eps = len(rows)
        succ = sum(1 for r in rows if r.get("success"))
        per_task[str(task_id)] = {
            "task_id": task_id,
            "task": rows[0].get("task"),
            "episodes": eps,
            "successes": succ,
            "success_rate": round(succ / eps, 4) if eps else 0.0,
            "mean_env_steps": round(sum(r.get("env_steps", 0) for r in rows) / eps, 2) if eps else 0.0,
            "mean_policy_steps": round(sum(r.get("policy_steps", 0) for r in rows) / eps, 2) if eps else 0.0,
            "model_calls": sum(r.get("model_calls", 0) for r in rows),
            "inference_seconds": round(sum(r.get("inference_seconds", 0.0) for r in rows), 3),
            "sim_seconds": round(sum(r.get("sim_seconds", 0.0) for r in rows), 3),
            "wall_seconds": round(sum(r.get("wall_seconds", 0.0) for r in rows), 3),
            "successes_with_video": sum(1 for r in rows if r.get("success") and r.get("video_path")),
            "failures_with_video": sum(1 for r in rows if not r.get("success") and r.get("video_path")),
        }

    n = len(records)
    s = sum(1 for r in records if r.get("success"))
    return {
        "episodes": n,
        "successes": s,
        "success_rate": round(s / n, 4) if n else 0.0,
        "num_tasks": len(per_task),
        "per_task": per_task,
        "timing_totals": {
            "inference_seconds": round(sum(r.get("inference_seconds", 0.0) for r in records), 3),
            "sim_seconds": round(sum(r.get("sim_seconds", 0.0) for r in records), 3),
            "video_seconds": round(sum(r.get("video_seconds", 0.0) for r in records), 3),
            "wall_seconds": round(sum(r.get("wall_seconds", 0.0) for r in records), 3),
            "model_calls": sum(r.get("model_calls", 0) for r in records),
            "env_steps": sum(r.get("env_steps", 0) for r in records),
        },
    }


def write_csv(path: Path, summary: dict) -> None:
    fields = [
        "task_id", "task", "episodes", "successes", "success_rate",
        "mean_env_steps", "mean_policy_steps", "model_calls",
        "inference_seconds", "sim_seconds", "wall_seconds",
        "successes_with_video", "failures_with_video",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary["per_task"].values():
            w.writerow({k: row.get(k) for k in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", default=os.environ.get("MODEL_ID", ""))
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_ids", default="all", help="'all' or comma-separated ids")
    parser.add_argument("--num_trials_per_task", type=int, default=10)
    parser.add_argument("--num_open_loop_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--env_img_res", type=int, default=256)
    parser.add_argument("--center_crop", type=str2bool, default=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_id_note", default="eval")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.jsonl"
    if episodes_path.exists() and not args.resume:
        raise SystemExit(
            f"Refusing to mix runs: {episodes_path} exists. Use --resume or a new --output_dir."
        )

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
        local_log_dir=str(output_dir / "logs"),
        use_wandb=False,
        seed=args.seed,
        run_id_note=args.run_id_note,
    )
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    existing: list[dict] = []
    completed: set[tuple[int, int]] = set()
    if args.resume and episodes_path.exists():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                existing.append(rec)
                completed.add((int(rec["task_id"]), int(rec["episode_idx"])))

    started = time.perf_counter()
    manifest = {
        "status": "started",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "project_root": str(PROJECT_ROOT),
        "openvla_oft_root": str(OPENVLA_OFT_ROOT),
        "openvla_oft_commit": git_commit(OPENVLA_OFT_ROOT),
        "libero_root": str(LIBERO_ROOT),
        "libero_commit": git_commit(LIBERO_ROOT),
        "model": args.pretrained_checkpoint,
        "task_suite": args.task_suite_name,
        "task_ids": args.task_ids,
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
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

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
        task_ids = task_ids_from_arg(args.task_ids, task_suite.n_tasks)
        manifest["resolved_task_ids"] = task_ids

        # save_rollout_video writes ./rollouts relative to the current directory.
        os.chdir(output_dir)
        records: list[dict] = list(existing)
        mode = "a" if args.resume else "w"
        run_started_epoch = time.time()
        with episodes_path.open(mode, encoding="utf-8") as episodes_file:
            for task_id in task_ids:
                task = task_suite.get_task(task_id)
                initial_states, _ = load_initial_states(cfg, task_suite, task_id, log_file)
                env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
                print(f"==== Task {task_id}: {task_description} ====", flush=True)
                if log_file is not None:
                    log_file.write(f"\nTask {task_id}: {task_description}\n")
                    log_file.flush()

                try:
                    for episode_idx in range(args.num_trials_per_task):
                        if (task_id, episode_idx) in completed:
                            print(f"skip completed task={task_id} episode={episode_idx}", flush=True)
                            continue
                        initial_state = initial_states[episode_idx]
                        rec = run_episode_instrumented(
                            cfg=cfg,
                            env=env,
                            task_description=task_description,
                            model=model,
                            resize_size=resize_size,
                            processor=processor,
                            action_head=action_head,
                            proprio_projector=proprio_projector,
                            noisy_action_projector=noisy_action_projector,
                            initial_state=initial_state,
                            log_file=log_file,
                            video_idx=len(records) + 1,
                            save_video=not args.no_video,
                        )
                        rec.update(
                            {
                                "run_id": run_id,
                                "task_suite": cfg.task_suite_name,
                                "task_id": task_id,
                                "task": task_description,
                                "episode_idx": episode_idx,
                                "initial_state_idx": episode_idx,
                                "initial_state_sha256_16": initial_state_digest(initial_state),
                                "seed": cfg.seed,
                                "env_seed": 0,
                                "num_open_loop_steps": cfg.num_open_loop_steps,
                                "num_steps_wait": cfg.num_steps_wait,
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        records.append(rec)
                        episodes_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        episodes_file.flush()
                        print(
                            f"task={task_id} episode={episode_idx} success={rec['success']} "
                            f"env_steps={rec['env_steps']} model_calls={rec['model_calls']} "
                            f"inference_s={rec['inference_seconds']:.3f} wall_s={rec['wall_seconds']:.2f}"
                            f" video={rec['video_path']}",
                            flush=True,
                        )
                        if rec.get("error") and args.fail_fast:
                            raise RuntimeError("episode error; see JSONL traceback")
                finally:
                    try:
                        env.close()
                    except Exception:
                        pass

        summary = aggregate(records)
        summary["run_id"] = run_id
        summary["task_suite"] = cfg.task_suite_name
        summary["num_trials_per_task"] = args.num_trials_per_task
        summary["num_open_loop_steps"] = cfg.num_open_loop_steps
        summary["seed"] = cfg.seed
        summary["model"] = args.pretrained_checkpoint
        summary["output_dir"] = str(output_dir)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_csv(output_dir / "summary.csv", summary)

        manifest.update(
            {
                "status": "ok",
                "run_id": run_id,
                "episodes": summary["episodes"],
                "successes": summary["successes"],
                "success_rate": summary["success_rate"],
                "summary": summary,
                "wall_seconds": round(time.perf_counter() - started, 3),
                "output_dir": str(output_dir),
            }
        )
        print("EVAL_SUMMARY_JSON: " + json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception:
        manifest["status"] = "error"
        manifest["traceback"] = traceback.format_exc()
        print(manifest["traceback"], file=sys.stderr, flush=True)
        raise
    finally:
        manifest["wall_seconds"] = round(time.perf_counter() - started, 3)
        if log_file is not None:
            log_file.flush()
            log_file.close()
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"EVAL_MANIFEST_PATH: {manifest_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
