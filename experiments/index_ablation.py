"""H4 - is the target-detector index a load-bearing supervision channel?

The photometric agent is told which detector is the target (TARGET_DETECTOR_INDEX).
env_v2 framed this as "the agent does not have to know" the index; the agent in fact
consumes it. H4 asks whether that channel is load-bearing or - as the source report
hedged - effectively redundant because the detector layout makes the target obvious.

Geometry check (settled): the reflected beam is a tight spot, so at convergence only
ONE detector is lit; the 8 sit symmetric on a radius-1.2 ring. So during SCAN the beam
crosses several detectors and "infer the target = brightest seen" can lock the wrong one.

Conditions (true target stays detector 0; we ablate the AGENT's knowledge):
  known    : agent peaks detector 0 (current baseline).
  inferred : agent peaks the brightest detector seen during SCAN, then tracks it
             (unsupervised). Identification accuracy = fraction where inferred == 0.
  wrong    : agent peaks detector 4 (opposite) - adversarial lower bound.

Everything is scored by the intensity at the TRUE target (detector 0).

Pre-registered falsifier
------------------------
H4 is NULL (the report's hedge holds) if `inferred` is statistically indistinguishable
from `known` at the true target across the ladder - i.e. the index carries no
information the geometry doesn't already hand you.

Reproduce:
    python -m sundog.experiments.index_ablation            # full 30-seed
    python -m sundog.experiments.index_ablation --seeds 3  # smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import mujoco
import numpy as np

from sundog.env_v2 import SundogEnvV2, TARGET_DETECTOR_INDEX
from sundog.agents.photometric import PhotometricAgent
from sundog.experiments.analysis import CONVERGENCE_THRESHOLD, TERMINAL_WINDOW, bootstrap_ci, mann_whitney_u
from sundog.experiments.stress_tests import (
    Stressor,
    apply_laser_height,
    apply_mirror_bias,
    apply_obs_perturbation,
    apply_joint_limit_to_action,
    laser_xy_for_seed,
    initial_qpos_for_seed,
)

DEFAULT_SEEDS = 30
DEFAULT_STEPS = 500
ALPHA = 0.05
CONDITIONS = ["known", "inferred", "wrong"]
WRONG_INDEX = 4
NOMINAL_KNOWN_TERMINAL = 0.945
ANCHOR_ATOL = 0.02

RESULTS_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "index_ablation",
)


def ladder() -> list[tuple[str, str, float]]:
    cells = [("nominal", "", 0.0)]
    for v in [0.02, 0.05, 0.10, 0.20]:
        cells.append((f"detector_noise_{v:g}", "detector_noise_sigma", v))
    for v in [0.25, 0.40]:
        cells.append((f"beam_sigma_{v:g}", "beam_sigma", v))
    return cells


def stressor_for(field: str, level: float) -> Stressor:
    return Stressor(**{field: level}) if field else Stressor()


def build_agent(cond: str) -> PhotometricAgent:
    if cond == "known":
        return PhotometricAgent(target_detector_index=TARGET_DETECTOR_INDEX)
    if cond == "inferred":
        return PhotometricAgent(infer_target=True)
    if cond == "wrong":
        return PhotometricAgent(target_detector_index=WRONG_INDEX)
    raise ValueError(cond)


def run_episode(env: SundogEnvV2, cond: str, stressor: Stressor, seed: int, n_steps: int) -> dict:
    env.reset(laser_xy=laser_xy_for_seed(seed))
    apply_laser_height(env, stressor)
    apply_mirror_bias(env, stressor)
    q = initial_qpos_for_seed(seed)
    env.data.qpos[:] = q
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = q
    mujoco.mj_forward(env.model, env.data)
    obs_clean = env._observation()

    rng = np.random.default_rng(seed * 17 + 3)
    obs_seen = apply_obs_perturbation(obs_clean, stressor, rng)
    agent = build_agent(cond)
    agent.reset(carrier_init=tuple(obs_seen.joint_angles))

    series = np.empty(n_steps, dtype=np.float64)  # TRUE target (detector 0)
    for t in range(n_steps):
        action = agent.act(obs_seen)
        action = apply_joint_limit_to_action(action, stressor)
        obs_clean = env.step(action)
        obs_seen = apply_obs_perturbation(obs_clean, stressor, rng)
        series[t] = obs_clean.detector_intensities[TARGET_DETECTOR_INDEX]

    above = series > CONVERGENCE_THRESHOLD
    ttt = float(np.argmax(above)) if above.any() else float(n_steps)
    identified = (agent.inferred_index == TARGET_DETECTOR_INDEX) if cond == "inferred" else None
    return {"terminal": float(np.mean(series[-TERMINAL_WINDOW:])), "ttt": ttt,
            "inferred_index": agent.inferred_index if cond == "inferred" else -1,
            "identified": identified}


def agg(vals: np.ndarray) -> dict:
    ci = bootstrap_ci(vals)
    return {"mean": float(vals.mean()), "median": float(np.median(vals)), "ci95": [ci[0], ci[1]]}


def run(args: argparse.Namespace) -> None:
    out_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)
    cells = ladder()
    started = time.time()

    trial_rows, level_rows, cell_rows = [], [], []
    summaries: dict[str, dict] = {}
    terminals: dict[tuple[str, str], np.ndarray] = {}

    total = len(cells) * len(CONDITIONS) * args.seeds
    done = 0
    print(f"[h4] index ablation: {total} trials ({len(cells)} cells x 3 conditions x {args.seeds} seeds)")

    for cell_id, field, level in cells:
        stressor = stressor_for(field, level)
        for cond in CONDITIONS:
            env = SundogEnvV2(sigma=stressor.beam_sigma, seed=0)
            t_vals = np.empty(args.seeds)
            ttt_vals = np.empty(args.seeds)
            id_hits = 0
            t0 = time.time()
            for s in range(args.seeds):
                rec = run_episode(env, cond, stressor, s, args.steps)
                t_vals[s] = rec["terminal"]
                ttt_vals[s] = rec["ttt"]
                if cond == "inferred" and rec["identified"]:
                    id_hits += 1
                trial_rows.append({"cell_id": cell_id, "condition": cond, "seed": s,
                                   "terminal": rec["terminal"], "ttt": rec["ttt"],
                                   "inferred_index": rec["inferred_index"]})
                done += 1
            terminals[(cell_id, cond)] = t_vals
            id_acc = (id_hits / args.seeds) if cond == "inferred" else None
            summaries.setdefault(cell_id, {})[cond] = {
                "terminal": agg(t_vals), "ttt_median": float(np.median(ttt_vals)),
                "identification_accuracy": id_acc,
            }
            level_rows.append({"cell_id": cell_id, "condition": cond, "n_seeds": args.seeds,
                               "terminal_mean": t_vals.mean(), "terminal_median": np.median(t_vals),
                               "ttt_median": np.median(ttt_vals),
                               "identification_accuracy": "" if id_acc is None else round(id_acc, 3)})
            extra = f" id_acc={id_acc:.2f}" if id_acc is not None else ""
            print(f"[h4] {cell_id:20s} {cond:9s} det0_term={t_vals.mean():.3f}{extra} "
                  f"({done}/{total}, {time.time()-t0:.1f}s)")

    for cell_id, _, _ in cells:
        known = terminals[(cell_id, "known")]
        inferred = terminals[(cell_id, "inferred")]
        _, p = mann_whitney_u(known, inferred)
        gap = float(np.median(known) - np.median(inferred))
        load_bearing = bool(p < ALPHA and gap > 0)
        cell_rows.append({
            "cell_id": cell_id,
            "supervision_gap_known_minus_inferred": gap,
            "p_known_vs_inferred": float(p),
            "identification_accuracy": summaries[cell_id]["inferred"]["identification_accuracy"],
            "index_load_bearing": load_bearing,
        })

    n_load = sum(1 for r in cell_rows if r["index_load_bearing"])
    nominal_load = next(r["index_load_bearing"] for r in cell_rows if r["cell_id"] == "nominal")
    status = ("index_load_bearing" if nominal_load and n_load >= 1
              else "falsifier_fired_index_redundant")

    nk = summaries["nominal"]["known"]["terminal"]["mean"]
    anchor = {"checked": True, "nominal_known_terminal": nk, "expected": NOMINAL_KNOWN_TERMINAL,
              "atol": ANCHOR_ATOL, "pass": bool(abs(nk - NOMINAL_KNOWN_TERMINAL) <= ANCHOR_ATOL)}

    _write(os.path.join(out_dir, "trial-outcomes.csv"), trial_rows,
           ["cell_id", "condition", "seed", "terminal", "ttt", "inferred_index"])
    _write(os.path.join(out_dir, "level-summary.csv"), level_rows,
           ["cell_id", "condition", "n_seeds", "terminal_mean", "terminal_median", "ttt_median",
            "identification_accuracy"])
    _write(os.path.join(out_dir, "supervision-gap.csv"), cell_rows,
           ["cell_id", "supervision_gap_known_minus_inferred", "p_known_vs_inferred",
            "identification_accuracy", "index_load_bearing"])

    completed = time.time()
    manifest = {
        "experiment": "h4_index_ablation",
        "hypothesis": "The target-detector index is a load-bearing supervision channel; "
                      "removing/randomizing it materially changes the performance story.",
        "status": status,
        "falsifier": "inferred indistinguishable from known across the ladder (index redundant)",
        "true_target_index": TARGET_DETECTOR_INDEX, "wrong_index": WRONG_INDEX,
        "conditions": CONDITIONS, "seeds": args.seeds, "steps": args.steps,
        "cells": [c[0] for c in cells], "n_cells_load_bearing": n_load,
        "supervision_gap": cell_rows, "anchor_nominal": anchor,
        "summaries": summaries, "wall_time_seconds": completed - started,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[h4] status: {status} ({n_load}/{len(cells)} cells load-bearing)")
    for r in cell_rows:
        print(f"   {r['cell_id']:20s} gap={r['supervision_gap_known_minus_inferred']:+.3f} "
              f"id_acc={r['identification_accuracy']:.2f} load_bearing={r['index_load_bearing']} "
              f"(p={r['p_known_vs_inferred']:.2g})")
    print(f"[h4] anchor pass={anchor['pass']}; wall={completed-started:.1f}s; wrote {out_dir}")


def _write(path: str, rows: list[dict], fields: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="H4 target-index ablation.")
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--results-dir", type=str, default=RESULTS_DIR_DEFAULT)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
