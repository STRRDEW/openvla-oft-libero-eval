#!/usr/bin/env python3
"""Aggregate and compare `run_eval.py` outputs.

This script only reads measured JSONL records. It does not invent, smooth, or
impute results. Runs are passed as `name=path` pairs, e.g.

    python tools/analyze_results.py \
        --run baseline_k8=results/phase2/baseline \
        --run k1=results/phase3/k1 \
        --run k4=results/phase3/k4 \
        --out results/analysis

It writes:
  * analysis.json          overall + per-task metrics for every run
  * comparison.csv        one row per run, directly comparable
  * per_task.csv          one row per run/task
  * failures.csv          every failed episode with its video path
  * success_by_task.png   grouped bar chart (if matplotlib is available)
  * compute_by_k.png      model calls / inference latency vs chunk length
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected name=path")
    name, raw = value.split("=", 1)
    path = Path(raw).resolve()
    if not (path / "episodes.jsonl").exists():
        raise argparse.ArgumentTypeError(f"{path}/episodes.jsonl not found")
    return name, path


def load_rows(run_dir: Path) -> list[dict]:
    rows = []
    with (run_dir / "episodes.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{run_dir}/episodes.jsonl is empty")
    return rows


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def summarize(name: str, run_dir: Path, rows: list[dict]) -> dict:
    by_task: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[int(row["task_id"])].append(row)

    per_task = {}
    for task_id in sorted(by_task):
        group = by_task[task_id]
        successes = sum(1 for r in group if r.get("success"))
        per_task[str(task_id)] = {
            "task_id": task_id,
            "task": group[0].get("task"),
            "episodes": len(group),
            "successes": successes,
            "success_rate": round(successes / len(group), 4),
            "mean_env_steps": mean([r.get("env_steps", 0) for r in group]),
            "mean_policy_steps": mean([r.get("policy_steps", 0) for r in group]),
            "mean_model_calls": mean([r.get("model_calls", 0) for r in group]),
            "mean_inference_seconds": mean([r.get("inference_seconds", 0.0) for r in group]),
            "mean_wall_seconds": mean([r.get("wall_seconds", 0.0) for r in group]),
            "failures_with_video": sum(
                1 for r in group if not r.get("success") and r.get("video_path")
            ),
        }

    episodes = len(rows)
    successes = sum(1 for r in rows if r.get("success"))
    total_model_calls = sum(r.get("model_calls", 0) for r in rows)
    total_inference = sum(r.get("inference_seconds", 0.0) for r in rows)
    total_sim = sum(r.get("sim_seconds", 0.0) for r in rows)
    total_wall = sum(r.get("wall_seconds", 0.0) for r in rows)
    latencies = [
        r["inference_latency_mean_ms"]
        for r in rows
        if r.get("inference_latency_mean_ms") is not None
    ]

    return {
        "name": name,
        "run_dir": str(run_dir),
        "num_open_loop_steps": sorted({int(r.get("num_open_loop_steps", -1)) for r in rows}),
        "seed": sorted({int(r.get("seed", -1)) for r in rows}),
        "episodes": episodes,
        "successes": successes,
        "success_rate": round(successes / episodes, 4),
        "num_tasks": len(per_task),
        "total_model_calls": total_model_calls,
        "mean_model_calls_per_episode": mean([r.get("model_calls", 0) for r in rows]),
        "mean_env_steps": mean([r.get("env_steps", 0) for r in rows]),
        "mean_policy_steps": mean([r.get("policy_steps", 0) for r in rows]),
        "total_inference_seconds": round(total_inference, 3),
        "mean_inference_seconds_per_episode": mean(
            [r.get("inference_seconds", 0.0) for r in rows]
        ),
        "mean_inference_latency_ms": mean(latencies),
        "total_sim_seconds": round(total_sim, 3),
        "mean_sim_seconds_per_episode": mean([r.get("sim_seconds", 0.0) for r in rows]),
        "total_wall_seconds": round(total_wall, 3),
        "mean_wall_seconds_per_episode": mean([r.get("wall_seconds", 0.0) for r in rows]),
        "inference_share_of_wall": round(total_inference / total_wall, 4) if total_wall else None,
        "sim_share_of_wall": round(total_sim / total_wall, 4) if total_wall else None,
        "model_calls_per_success": round(total_model_calls / successes, 3) if successes else None,
        "inference_seconds_per_success": round(total_inference / successes, 3) if successes else None,
        "per_task": per_task,
    }


def write_comparison(path: Path, summaries: list[dict]) -> None:
    fields = [
        "name",
        "num_open_loop_steps",
        "episodes",
        "successes",
        "success_rate",
        "total_model_calls",
        "mean_model_calls_per_episode",
        "mean_inference_latency_ms",
        "total_inference_seconds",
        "mean_inference_seconds_per_episode",
        "mean_env_steps",
        "mean_wall_seconds_per_episode",
        "inference_share_of_wall",
        "sim_share_of_wall",
        "model_calls_per_success",
        "inference_seconds_per_success",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row = {key: summary.get(key) for key in fields}
            row["num_open_loop_steps"] = ",".join(map(str, summary["num_open_loop_steps"]))
            writer.writerow(row)


def write_per_task(path: Path, summaries: list[dict]) -> None:
    fields = [
        "name",
        "num_open_loop_steps",
        "task_id",
        "task",
        "episodes",
        "successes",
        "success_rate",
        "mean_env_steps",
        "mean_policy_steps",
        "mean_model_calls",
        "mean_inference_seconds",
        "mean_wall_seconds",
        "failures_with_video",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            for task in summary["per_task"].values():
                row = dict(task)
                row["name"] = summary["name"]
                row["num_open_loop_steps"] = ",".join(
                    map(str, summary["num_open_loop_steps"])
                )
                writer.writerow({key: row.get(key) for key in fields})


def write_failures(path: Path, runs: list[tuple[str, Path, list[dict]]]) -> None:
    fields = [
        "name",
        "num_open_loop_steps",
        "task_id",
        "episode_idx",
        "initial_state_idx",
        "initial_state_sha256_16",
        "env_steps",
        "policy_steps",
        "model_calls",
        "inference_seconds",
        "wall_seconds",
        "video_path",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, _run_dir, rows in runs:
            for row in rows:
                if not row.get("success"):
                    error_lines = (row.get("error") or "").strip().splitlines()
                    writer.writerow(
                        {
                            "name": name,
                            "num_open_loop_steps": row.get("num_open_loop_steps"),
                            "task_id": row.get("task_id"),
                            "episode_idx": row.get("episode_idx"),
                            "initial_state_idx": row.get("initial_state_idx"),
                            "initial_state_sha256_16": row.get("initial_state_sha256_16"),
                            "env_steps": row.get("env_steps"),
                            "policy_steps": row.get("policy_steps"),
                            "model_calls": row.get("model_calls"),
                            "inference_seconds": row.get("inference_seconds"),
                            "wall_seconds": row.get("wall_seconds"),
                            "video_path": row.get("video_path"),
                            "error": error_lines[-1][:300] if error_lines else "",
                        }
                    )


def make_charts(out_dir: Path, summaries: list[dict]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    made = []

    # Chart 1: success rate by task, one group per run.
    task_ids = sorted({int(t) for s in summaries for t in s["per_task"]})
    width = 0.8 / max(len(summaries), 1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for idx, summary in enumerate(summaries):
        offsets = [i + (idx - (len(summaries) - 1) / 2) * width for i in range(len(task_ids))]
        values = [
            summary["per_task"].get(str(t), {}).get("success_rate", 0.0) for t in task_ids
        ]
        ax.bar(offsets, values, width=width, label=summary["name"])
    ax.set_xticks(range(len(task_ids)))
    ax.set_xticklabels([f"task {t}" for t in task_ids])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title("LIBERO-Spatial success rate by task")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    path = out_dir / "success_by_task.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(str(path))

    # Chart 2: compute cost vs chunk length.
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ks = [
        s["num_open_loop_steps"][0] if s["num_open_loop_steps"] else None
        for s in summaries
    ]
    if all(k is not None for k in ks):
        order = sorted(range(len(summaries)), key=lambda i: ks[i])
        xs = [ks[i] for i in order]
        calls = [summaries[i]["mean_model_calls_per_episode"] for i in order]
        ax1.plot(xs, calls, marker="o", color="tab:blue")
        ax1.set_xlabel("executed actions before re-observation")
        ax1.set_ylabel("mean model calls / episode", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax1.twinx()
        latency = [summaries[i]["mean_inference_latency_ms"] for i in order]
        ax2.plot(xs, latency, marker="s", color="tab:orange")
        ax2.set_ylabel("mean inference latency / call (ms)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax1.set_xticks(xs)
        ax1.set_title("Compute vs action chunk execution length")
        fig.tight_layout()
        path = out_dir / "compute_by_k.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        made.append(str(path))

    return made


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    loaded = []
    for name, run_dir in args.run:
        rows = load_rows(run_dir)
        loaded.append((name, run_dir, rows))
        summaries.append(summarize(name, run_dir, rows))

    analysis = {"runs": summaries, "notes": "All metrics are computed from episodes.jsonl only."}
    (out_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_comparison(out_dir / "comparison.csv", summaries)
    write_per_task(out_dir / "per_task.csv", summaries)
    write_failures(out_dir / "failures.csv", loaded)
    charts = make_charts(out_dir, summaries)
    print(f"wrote analysis for {len(summaries)} run(s) to {out_dir}")
    for path in charts:
        print(f"chart: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
