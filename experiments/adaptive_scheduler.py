"""H5 - belief-aware adaptive phase scheduler vs the fixed scan/seek/track schedule.

The fixed PhotometricAgent hard-codes a 4 s (200-step) SCAN window and an absolute
re-acquire floor. H5 asks whether a model-free, belief-aware schedule -- stop SCAN
on an expected-improvement plateau, re-acquire on a relative drop below the achieved
ceiling -- dominates the fixed schedule across broad stress sweeps.

This is the natural follow-on to H3: H3 showed the photometric controller degrades
gracefully under model mismatch but is slow (time-to-threshold ~188 at nominal). H5
reuses the same stressor harness to ask whether adaptive scheduling buys back that
speed without sacrificing the terminal accuracy H3 established.

Conditions: photometric (fixed), photometric_adaptive (PhotometricAgent(adaptive=True)).
Ladder:     nominal + mirror_bias + beam_sigma + detector_noise + laser_height.
Metrics:    time-to-threshold, convergence-curve AUC (episode-mean intensity),
            terminal intensity (guardrail), scan-exit step (speedup), re-acquire
            count (instability proxy).

Pre-registered falsifier (the claim is DOMINATES)
-------------------------------------------------
H5 is NULL if the adaptive scheduler significantly regresses terminal accuracy in any
cell without a compensating speed gain, OR fails to improve time-to-threshold / AUC
anywhere. A pure speed/accuracy tradeoff (faster but slightly less accurate) refutes
the strong "dominates" claim while still being a real, reportable finding.

Reproduce:
    python -m sundog.experiments.adaptive_scheduler            # full 30-seed ladder
    python -m sundog.experiments.adaptive_scheduler --seeds 5  # smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import mujoco
import numpy as np

from sundog.env_v2 import SundogEnvV2
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
CONDITIONS = ["photometric", "photometric_adaptive"]
NOMINAL_FIXED_TERMINAL = 0.945
ANCHOR_ATOL = 0.02

RESULTS_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "adaptive_scheduler",
)


def ladder() -> list[tuple[str, str, float]]:
    """(cell_id, stressor_field, level). field='' => nominal."""
    cells = [("nominal", "", 0.0)]
    for v in [10.0, 20.0, 30.0, 40.0]:
        cells.append((f"mirror_bias_{v:g}", "mirror_bias_deg", v))
    for v in [0.05, 0.10, 0.25, 0.40]:
        cells.append((f"beam_sigma_{v:g}", "beam_sigma", v))
    for v in [0.02, 0.05, 0.10, 0.20]:
        cells.append((f"detector_noise_{v:g}", "detector_noise_sigma", v))
    for v in [1.5, 2.0, 3.0, 3.5]:
        cells.append((f"laser_height_{v:g}", "laser_height", v))
    return cells


def stressor_for(field: str, level: float) -> Stressor:
    return Stressor(**{field: level}) if field else Stressor()


def run_episode(env: SundogEnvV2, adaptive: bool, stressor: Stressor, seed: int, n_steps: int) -> dict:
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
    agent = PhotometricAgent(
        adaptive=adaptive,
        scan_duration_s=stressor.photometric_scan_duration_s,
        joint_limit=min(stressor.joint_limit - 0.05, 1.45),
    )
    agent.reset(carrier_init=tuple(obs_seen.joint_angles))

    series = np.empty(n_steps, dtype=np.float64)
    for t in range(n_steps):
        action = agent.act(obs_seen)
        action = apply_joint_limit_to_action(action, stressor)
        obs_clean = env.step(action)
        obs_seen = apply_obs_perturbation(obs_clean, stressor, rng)
        series[t] = obs_clean.detector_intensities[0]

    above = series > CONVERGENCE_THRESHOLD
    ttt = float(np.argmax(above)) if above.any() else float(n_steps)
    return {
        "terminal": float(np.mean(series[-TERMINAL_WINDOW:])),
        "ttt": ttt,
        "auc": float(np.mean(series)),
        "scan_exit": float(agent.scan_exit_step),
        "reacquire_count": float(agent.reacquire_count),
    }


def agg(vals: np.ndarray) -> dict:
    ci = bootstrap_ci(vals)
    return {"mean": float(vals.mean()), "median": float(np.median(vals)), "ci95": [ci[0], ci[1]]}


def classify(adaptive: dict[str, np.ndarray], fixed: dict[str, np.ndarray]) -> dict:
    _, p_term = mann_whitney_u(adaptive["terminal"], fixed["terminal"])
    _, p_ttt = mann_whitney_u(adaptive["ttt"], fixed["ttt"])
    _, p_auc = mann_whitney_u(adaptive["auc"], fixed["auc"])
    dt_term = float(np.median(adaptive["terminal"]) - np.median(fixed["terminal"]))
    dt_ttt = float(np.median(adaptive["ttt"]) - np.median(fixed["ttt"]))
    dt_auc = float(np.median(adaptive["auc"]) - np.median(fixed["auc"]))

    terminal_worse = p_term < ALPHA and dt_term < 0
    faster = p_ttt < ALPHA and dt_ttt < 0
    auc_better = p_auc < ALPHA and dt_auc > 0

    if (faster or auc_better) and not terminal_worse:
        cls = "adaptive_dominant"
    elif (faster or auc_better) and terminal_worse:
        cls = "tradeoff_faster_less_accurate"
    elif terminal_worse:
        cls = "regressed"
    else:
        cls = "neutral"
    return {
        "class": cls,
        "delta_terminal_median": dt_term, "p_terminal": float(p_term),
        "delta_ttt_median": dt_ttt, "p_ttt": float(p_ttt),
        "delta_auc_median": dt_auc, "p_auc": float(p_auc),
    }


def run(args: argparse.Namespace) -> None:
    out_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)
    cells = ladder()
    started = time.time()

    trial_rows, level_rows, cell_rows = [], [], []
    summaries: dict[str, dict] = {}
    metrics_by: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    total = len(cells) * len(CONDITIONS) * args.seeds
    done = 0
    print(f"[h5] ladder: {total} trials ({len(cells)} cells x 2 conditions x {args.seeds} seeds)")

    for cell_id, field, level in cells:
        stressor = stressor_for(field, level)
        for cond in CONDITIONS:
            adaptive = cond == "photometric_adaptive"
            env = SundogEnvV2(sigma=stressor.beam_sigma, seed=0)
            cols = {k: np.empty(args.seeds) for k in ("terminal", "ttt", "auc", "scan_exit", "reacquire_count")}
            t0 = time.time()
            for s in range(args.seeds):
                rec = run_episode(env, adaptive, stressor, s, args.steps)
                for k in cols:
                    cols[k][s] = rec[k]
                trial_rows.append({"cell_id": cell_id, "condition": cond, "seed": s, **rec})
                done += 1
            metrics_by[(cell_id, cond)] = cols
            summaries.setdefault(cell_id, {})[cond] = {k: agg(cols[k]) for k in cols}
            level_rows.append({
                "cell_id": cell_id, "condition": cond, "n_seeds": args.seeds,
                "terminal_mean": cols["terminal"].mean(), "terminal_median": np.median(cols["terminal"]),
                "ttt_median": np.median(cols["ttt"]), "auc_mean": cols["auc"].mean(),
                "scan_exit_mean": cols["scan_exit"].mean(), "reacquire_mean": cols["reacquire_count"].mean(),
            })
            print(f"[h5] {cell_id:20s} {cond:22s} term={cols['terminal'].mean():.3f} "
                  f"ttt_med={np.median(cols['ttt']):.0f} auc={cols['auc'].mean():.3f} "
                  f"scan_exit={cols['scan_exit'].mean():.0f} ({done}/{total}, {time.time()-t0:.1f}s)")

    for cell_id, _, _ in cells:
        c = classify(metrics_by[(cell_id, "photometric_adaptive")], metrics_by[(cell_id, "photometric")])
        c["cell_id"] = cell_id
        cell_rows.append(c)

    classes = [r["class"] for r in cell_rows]
    counts = {c: classes.count(c) for c in sorted(set(classes))}
    n = len(cells)
    n_dom = counts.get("adaptive_dominant", 0)
    n_trade = counts.get("tradeoff_faster_less_accurate", 0)
    n_reg = counts.get("regressed", 0)
    n_gain = n_dom + n_trade
    dominates = n_dom == n
    if dominates:
        status = "adaptive_dominates"
    elif n_gain >= n / 2 and n_reg > 0:
        status = "broadly_favorable_with_failure_modes"
    elif n_gain > 0 and n_reg == 0:
        status = "favorable_tradeoff_not_domination"
    elif n_gain > 0:
        status = "mixed_gains_and_regressions"
    else:
        status = "falsifier_fired_no_net_gain"

    # Soft anchor: nominal fixed reproduces the headline number.
    nf = summaries["nominal"]["photometric"]["terminal"]["mean"]
    anchor = {"checked": True, "nominal_fixed_terminal": nf, "expected": NOMINAL_FIXED_TERMINAL,
              "atol": ANCHOR_ATOL, "pass": bool(abs(nf - NOMINAL_FIXED_TERMINAL) <= ANCHOR_ATOL)}

    _write(os.path.join(out_dir, "trial-outcomes.csv"), trial_rows,
           ["cell_id", "condition", "seed", "terminal", "ttt", "auc", "scan_exit", "reacquire_count"])
    _write(os.path.join(out_dir, "level-summary.csv"), level_rows,
           ["cell_id", "condition", "n_seeds", "terminal_mean", "terminal_median", "ttt_median",
            "auc_mean", "scan_exit_mean", "reacquire_mean"])
    _write(os.path.join(out_dir, "cell-verdicts.csv"), cell_rows,
           ["cell_id", "class", "delta_terminal_median", "p_terminal", "delta_ttt_median", "p_ttt",
            "delta_auc_median", "p_auc"])

    completed = time.time()
    manifest = {
        "experiment": "h5_adaptive_scheduler",
        "hypothesis": "A belief-aware (EI-plateau + relative-reacquire) adaptive scheduler dominates "
                      "the fixed scan/seek/track schedule across broad stress sweeps.",
        "status": status,
        "falsifier": "regresses terminal without compensating speed, or no speed/AUC gain anywhere",
        "adaptive_dominates": dominates,
        "adaptive_config": {"ei_window": 40, "ei_epsilon": 0.005, "ei_min_scan_s": 2.0,
                            "ei_signal_floor": 0.1, "reacquire_rel_frac": 0.3,
                            "note": "tuned on the nominal speed/accuracy frontier"},
        "conditions": CONDITIONS, "seeds": args.seeds, "steps": args.steps,
        "cells": [c[0] for c in cells], "class_counts": counts,
        "cell_verdicts": cell_rows, "anchor_nominal": anchor,
        "summaries": summaries, "wall_time_seconds": completed - started,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[h5] status: {status} | class_counts: {counts}")
    for r in cell_rows:
        print(f"   {r['cell_id']:20s} {r['class']:32s} dTerm={r['delta_terminal_median']:+.3f} "
              f"dTTT={r['delta_ttt_median']:+.0f} dAUC={r['delta_auc_median']:+.3f}")
    print(f"[h5] anchor pass={anchor['pass']}; wall={completed-started:.1f}s; wrote {out_dir}")


def _write(path: str, rows: list[dict], fields: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="H5 adaptive-scheduler ladder.")
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--results-dir", type=str, default=RESULTS_DIR_DEFAULT)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
