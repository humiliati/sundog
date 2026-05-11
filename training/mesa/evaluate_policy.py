"""Closed-loop evaluation for Phase 2 learned Mesa policies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from training.mesa.js_bridge_env import BridgeClient, REPO_ROOT
from training.mesa.policy import load_checkpoint, policy_from_checkpoint


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "results"
    / "mesa"
    / "phase2-matched-capacity"
    / "checkpoints"
    / "signature_bc_from_hc_small_seed_0.pt"
)
DEFAULT_OUT = REPO_ROOT / "results" / "mesa" / "phase2-matched-capacity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Phase 2 learned policy through the JS bridge.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sensor-tier", default="local-probe-field")
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--success-floor", type=float, default=0.90)
    return parser.parse_args()


def _action(policy, obs: list[float], obs_mean: np.ndarray, obs_std: np.ndarray) -> list[float]:
    obs_array = (np.asarray(obs, dtype=np.float32) - obs_mean) / obs_std
    with torch.no_grad():
        action = policy(torch.from_numpy(obs_array).unsqueeze(0)).squeeze(0).cpu().numpy()
    return [float(action[0]), float(action[1])]


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    sensor_tier: str,
    seed_start: int,
    seeds: int,
    horizon: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path)
    policy, obs_mean, obs_std = policy_from_checkpoint(checkpoint)
    rows: list[dict[str, Any]] = []

    with BridgeClient() as client:
        for offset in range(seeds):
            seed = seed_start + offset
            env_id = f"eval-{seed}"
            made = client.request(
                {
                    "cmd": "make",
                    "env_id": env_id,
                    "seed": seed,
                    "sensor_tier": sensor_tier,
                    "env_config": {"horizon": horizon},
                }
            )
            obs = made["obs"]
            terminal_payload = None
            for _ in range(horizon + 1):
                response = client.request({"cmd": "step", "env_id": env_id, "action": _action(policy, obs, obs_mean, obs_std)})
                obs = response["obs"]
                if response["done"]:
                    terminal_payload = response
                    break
            if terminal_payload is None:
                rows.append({
                    "seed": seed,
                    "terminalOutcome": "not_done",
                    "steps": horizon,
                    "terminalAlignment": "",
                    "terminalDistance": "",
                    "pathEfficiency": "",
                    "saturationCount": "",
                })
                continue

            metrics = terminal_payload["info"].get("metrics") or {}
            rows.append({
                "seed": seed,
                "terminalOutcome": metrics.get("terminalOutcome", terminal_payload["info"].get("terminal_outcome")),
                "steps": metrics.get("steps", terminal_payload["info"].get("step_index")),
                "terminalAlignment": metrics.get("terminalAlignment"),
                "terminalDistance": metrics.get("terminalDistance"),
                "pathEfficiency": metrics.get("pathEfficiency"),
                "saturationCount": metrics.get("saturationCount"),
            })

    successes = [row for row in rows if row["terminalOutcome"] == "success"]
    alignments = [float(row["terminalAlignment"]) for row in rows if row["terminalAlignment"] != ""]
    steps = [float(row["steps"]) for row in rows if row["steps"] != ""]
    summary = {
        "checkpoint": str(checkpoint_path),
        "sensor_tier": sensor_tier,
        "seed_start": seed_start,
        "seeds": seeds,
        "success_count": len(successes),
        "success_rate": len(successes) / seeds,
        "mean_terminal_alignment": mean(alignments) if alignments else None,
        "mean_steps": mean(steps) if steps else None,
    }
    return rows, summary


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "evaluation-outcomes.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "evaluation-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evaluation"] = summary
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows, summary = evaluate_checkpoint(
        args.checkpoint.resolve(),
        sensor_tier=args.sensor_tier,
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
    )
    write_outputs(args.out.resolve(), rows, summary)
    mean_alignment = summary["mean_terminal_alignment"]
    mean_steps = summary["mean_steps"]
    mean_alignment_text = f"{mean_alignment:.4f}" if mean_alignment is not None else "n/a"
    mean_steps_text = f"{mean_steps:.1f}" if mean_steps is not None else "n/a"
    print(
        "mesa policy eval: "
        f"success={summary['success_count']}/{summary['seeds']} "
        f"({100 * summary['success_rate']:.1f}%) "
        f"mean_S_T={mean_alignment_text} "
        f"mean_steps={mean_steps_text}"
    )
    if summary["success_rate"] < args.success_floor:
        raise SystemExit(f"success rate below floor: {summary['success_rate']:.3f} < {args.success_floor:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
