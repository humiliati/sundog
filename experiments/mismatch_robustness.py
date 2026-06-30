"""H3 - robustness under model mismatch (mirror-calibration bias).

Tests the hypothesis that the analytic oracle is only an upper bound in the
NOMINAL geometry: under a model mismatch the oracle cannot see, the closed-loop
photometric controller can match or beat it on terminal accuracy.

Mismatch lever
--------------
A fixed, unmodelled tilt of the mirror's reflecting normal (a warped /
miscalibrated mirror), installed via SundogEnvV2.set_normal_bias(). The analytic
oracle (optics.optimal_joint_angles) and the particle-Bayes baseline both assume
an ideal mirror, so their open-loop / belief solves land off-target by a
bias-induced offset. The photometric controller measures the true target
intensity and climbs to the (shifted) real peak, until the bias pushes that peak
past the joint limit (the "both fail" region).

Conditions: photometric, doa_direct (oracle), doa_noisy, bayes_particle, random.
Axis:       mirror_bias_deg in {0, 2, 5, 10, 15, 20}.
Primary metric: terminal intensity (the flip happens here, NOT on speed).

Pre-registered falsifier
------------------------
If photometric's terminal intensity never significantly exceeds the oracle's at
any bias level before both collapse at the joint-limit wall, H3 is NULL: the
controller merely "degrades earlier", and the oracle's advantage is not just
local. Recorded as status "no_flip_falsifier_fired".

Reproduce:
    python -m sundog.experiments.mismatch_robustness            # full 30-seed
    python -m sundog.experiments.mismatch_robustness --seeds 5  # smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import mujoco
import numpy as np

from sundog import optics
from sundog.env_v2 import SundogEnvV2, Observation, Oracle, tilt_normal
from sundog.agents.photometric import PhotometricAgent
from sundog.agents.baselines import DOADirectAgent, DOANoisyAgent, RandomAgent
from sundog.agents.bayes_particle import BayesParticleAgent
from sundog.experiments.analysis import (
    CONVERGENCE_THRESHOLD,
    TERMINAL_WINDOW,
    bootstrap_ci,
    mann_whitney_u,
)
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
DEFAULT_BIAS_LEVELS = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
CONDITIONS = ["photometric", "doa_direct", "doa_noisy", "bayes_particle", "random"]
# Terminal intensity below this counts as a failed alignment for "both_fail".
SUCCESS_FLOOR = 0.5
# Significance level for the photometric-vs-oracle terminal comparison.
ALPHA = 0.05
# Soft anchor: bias=0 should reproduce the nominal headline numbers.
NOMINAL_PHOTOMETRIC_TERMINAL = 0.945
NOMINAL_ORACLE_TERMINAL = 0.936
ANCHOR_ATOL = 0.02

RESULTS_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "results", "mismatch_robustness",
)


# ---------------------------------------------------------------------------
# Agent construction (mirrors the Phase 6 live-agent factory, all 5 lanes)
# ---------------------------------------------------------------------------

def build_agent(
    condition: str,
    stressor: Stressor,
    seed: int,
    initial_obs: Observation,
    oracle: Oracle,
    env: SundogEnvV2,
    particle_count: int,
):
    if condition == "photometric":
        a = PhotometricAgent(
            scan_duration_s=stressor.photometric_scan_duration_s,
            joint_limit=min(stressor.joint_limit - 0.05, 1.45),
        )
        a.reset(carrier_init=tuple(initial_obs.joint_angles))
        return a
    if condition == "doa_direct":
        a = DOADirectAgent(joint_limit=stressor.joint_limit)
        a.reset(oracle)
        return a
    if condition == "doa_noisy":
        a = DOANoisyAgent(noise_std=0.05, joint_limit=stressor.joint_limit, seed=seed * 71 + 5)
        a.reset(oracle)
        return a
    if condition == "random":
        a = RandomAgent(joint_limit=stressor.joint_limit, seed=seed * 31 + 7)
        a.reset()
        return a
    if condition == "bayes_particle":
        return BayesParticleAgent(
            particle_count=particle_count,
            seed=seed,
            detector_positions=env._detector_positions.copy(),
            target_detector_pos=oracle.target_detector_pos,
            target_detector_index=oracle.target_detector_index,
            joint_limit=stressor.joint_limit,
        )
    raise ValueError(f"unknown condition: {condition}")


def run_episode(
    env: SundogEnvV2,
    condition: str,
    stressor: Stressor,
    seed: int,
    particle_count: int,
    n_steps: int,
) -> np.ndarray:
    """Run one (condition, seed) episode under `stressor`. Returns target_intensity (T,)."""
    laser_xy = laser_xy_for_seed(seed)
    env.reset(laser_xy=laser_xy)
    apply_laser_height(env, stressor)
    apply_mirror_bias(env, stressor)

    initial_qpos = initial_qpos_for_seed(seed)
    env.data.qpos[:] = initial_qpos
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = initial_qpos
    mujoco.mj_forward(env.model, env.data)
    obs_clean = env._observation()

    oracle = env.get_oracle()
    rng = np.random.default_rng(seed * 17 + 3)
    obs_seen = apply_obs_perturbation(obs_clean, stressor, rng)
    agent = build_agent(condition, stressor, seed, obs_seen, oracle, env, particle_count)

    target_intensity = np.empty(n_steps, dtype=np.float64)
    for t in range(n_steps):
        action = agent.act(obs_seen)
        action = apply_joint_limit_to_action(action, stressor)
        obs_clean = env.step(action)
        obs_seen = apply_obs_perturbation(obs_clean, stressor, rng)
        target_intensity[t] = obs_clean.detector_intensities[0]
    return target_intensity


# ---------------------------------------------------------------------------
# Bias-aware achievable ceiling (for regret + the "both_fail" diagnosis)
# ---------------------------------------------------------------------------

def biased_ceiling(
    laser_pos: np.ndarray,
    target_pos: np.ndarray,
    sigma: float,
    bias_rad: float,
    bias_ref: tuple[float, float, float],
    joint_limit: float,
) -> float:
    """Max target intensity a bias-AWARE solver could reach at this geometry.

    Mirrors optics.optimal_joint_angles but applies the calibration bias to the
    reflecting normal (the mirror POSITION still follows the unbiased pole tip,
    matching env_v2._compute_intensities). regret = ceiling - achieved.
    """
    from scipy.optimize import minimize

    def neg(angles: np.ndarray) -> float:
        n = optics.joint_angles_to_mirror_normal(float(angles[0]), float(angles[1]))
        m = optics.POLE_BASE_WORLD + optics.POLE_LENGTH * n
        refl = tilt_normal(n, bias_rad, bias_ref) if abs(bias_rad) > 1e-12 else n
        d_in = m - laser_pos
        dn = float(np.linalg.norm(d_in))
        if dn < 1e-9:
            return 0.0
        r = optics.reflect(d_in / dn, refl)
        h = optics.floor_hit(m, r)
        if h is None:
            return 0.0
        return -optics.gaussian_intensity(h, target_pos, sigma)

    grid = np.linspace(-joint_limit, joint_limit, 21)
    best_val, best_ang = 0.0, np.zeros(2)
    for tx in grid:
        for ty in grid:
            v = neg(np.array([tx, ty]))
            if v < best_val:
                best_val, best_ang = v, np.array([tx, ty])
    if best_val < 0.0:
        res = minimize(neg, best_ang, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 300})
        return float(-neg(res.x))
    return 0.0


# ---------------------------------------------------------------------------
# Metrics / classification
# ---------------------------------------------------------------------------

def terminal_of(series: np.ndarray) -> float:
    return float(np.mean(series[-TERMINAL_WINDOW:]))


def time_to_threshold(series: np.ndarray) -> float:
    above = series > CONVERGENCE_THRESHOLD
    idx = int(np.argmax(above))
    return float(idx) if above[idx] else float(series.shape[0])


def aggregate(terminals: np.ndarray, ttts: np.ndarray, n_steps: int, regrets: np.ndarray) -> dict:
    t_ci = bootstrap_ci(terminals)
    return {
        "n_seeds": int(terminals.shape[0]),
        "terminal_intensity": {
            "mean": float(terminals.mean()),
            "median": float(np.median(terminals)),
            "std": float(terminals.std(ddof=1)) if terminals.shape[0] > 1 else 0.0,
            "ci95": [t_ci[0], t_ci[1]],
        },
        "time_to_threshold": {
            "mean": float(ttts.mean()),
            "median": float(np.median(ttts)),
            "n_failed": int((ttts >= n_steps).sum()),
        },
        "success_fraction": float((terminals >= CONVERGENCE_THRESHOLD).mean()),
        "mean_regret": float(regrets.mean()),
    }


def classify_level(photo_terminals: np.ndarray, oracle_terminals: np.ndarray) -> dict:
    u, p = mann_whitney_u(photo_terminals, oracle_terminals)
    photo_med = float(np.median(photo_terminals))
    oracle_med = float(np.median(oracle_terminals))
    if photo_med < SUCCESS_FLOOR and oracle_med < SUCCESS_FLOOR:
        cls = "both_fail"
    elif p < ALPHA and photo_med > oracle_med:
        cls = "photometric_dominant"
    elif p < ALPHA and oracle_med > photo_med:
        cls = "oracle_dominant"
    else:
        cls = "indistinguishable"
    return {
        "class": cls,
        "photometric_median_terminal": photo_med,
        "oracle_median_terminal": oracle_med,
        "median_lead_photometric_minus_oracle": photo_med - oracle_med,
        "mannwhitney_u": float(u),
        "mannwhitney_p": float(p),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    out_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)
    levels = args.levels if args.levels is not None else DEFAULT_BIAS_LEVELS
    conditions = args.conditions or CONDITIONS
    started = time.time()

    trial_rows: list[dict] = []
    level_rows: list[dict] = []
    boundary_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    # per (level, condition) -> arrays
    terminals: dict[tuple[float, str], np.ndarray] = {}

    total = len(levels) * len(conditions) * args.seeds
    done = 0
    print(f"[h3] mirror_bias sweep: {total} trials "
          f"({len(levels)} levels x {len(conditions)} conditions x {args.seeds} seeds)")

    bias_ref = (0.0, 0.0, 1.0)
    for level in levels:
        bias_rad = np.deg2rad(level)
        # Per-seed bias-aware ceiling (geometry-only; cheap, no MuJoCo stepping).
        ceilings = np.empty(args.seeds, dtype=np.float64)
        scratch = SundogEnvV2(sigma=optics.DEFAULT_SIGMA, seed=0)
        for s in range(args.seeds):
            scratch.reset(laser_xy=laser_xy_for_seed(s))
            apply_laser_height(scratch, Stressor())
            ora = scratch.get_oracle()
            ceilings[s] = biased_ceiling(
                ora.laser_pos, ora.target_detector_pos, optics.DEFAULT_SIGMA,
                bias_rad, bias_ref, optics.JOINT_LIMIT,
            )

        for condition in conditions:
            stressor = Stressor(mirror_bias_deg=level)
            env = SundogEnvV2(sigma=stressor.beam_sigma, seed=0)
            t_vals = np.empty(args.seeds, dtype=np.float64)
            ttt_vals = np.empty(args.seeds, dtype=np.float64)
            t0 = time.time()
            for s in range(args.seeds):
                series = run_episode(env, condition, stressor, s, args.particle_count, args.steps)
                term = terminal_of(series)
                ttt = time_to_threshold(series)
                t_vals[s] = term
                ttt_vals[s] = ttt
                trial_rows.append({
                    "mirror_bias_deg": level,
                    "condition": condition,
                    "seed": s,
                    "terminal_intensity": term,
                    "time_to_threshold": ttt,
                    "censored": bool(ttt >= args.steps),
                    "ceiling": float(ceilings[s]),
                    "regret": float(ceilings[s] - term),
                })
                done += 1
            terminals[(level, condition)] = t_vals
            regrets = ceilings - t_vals
            agg = aggregate(t_vals, ttt_vals, args.steps, regrets)
            summaries.setdefault(str(level), {})[condition] = agg
            level_rows.append({
                "mirror_bias_deg": level,
                "condition": condition,
                "n_seeds": agg["n_seeds"],
                "terminal_mean": agg["terminal_intensity"]["mean"],
                "terminal_median": agg["terminal_intensity"]["median"],
                "terminal_ci95_lo": agg["terminal_intensity"]["ci95"][0],
                "terminal_ci95_hi": agg["terminal_intensity"]["ci95"][1],
                "ttt_median": agg["time_to_threshold"]["median"],
                "n_failed": agg["time_to_threshold"]["n_failed"],
                "success_fraction": agg["success_fraction"],
                "mean_regret": agg["mean_regret"],
            })
            wall = time.time() - t0
            print(f"[h3] bias={level:5.1f}  {condition:15s} "
                  f"term_mean={agg['terminal_intensity']['mean']:.3f} "
                  f"succ={agg['success_fraction']:.2f} "
                  f"({done}/{total}, {wall:.1f}s)")

    # Per-level classification (photometric vs oracle on terminal intensity).
    for level in levels:
        cls = classify_level(terminals[(level, "photometric")], terminals[(level, "doa_direct")])
        cls["mirror_bias_deg"] = level
        cls["ceiling_median"] = float(np.median(
            [r["ceiling"] for r in trial_rows if r["mirror_bias_deg"] == level and r["condition"] == "photometric"]
        ))
        boundary_rows.append(cls)

    classes = [r["class"] for r in boundary_rows]
    flip = any(c == "photometric_dominant" for c in classes)
    status = "flip_confirmed" if flip else "no_flip_falsifier_fired"

    # Soft anchor at bias=0.
    anchor = {"checked": False}
    if 0.0 in levels:
        p0 = summaries["0.0"]["photometric"]["terminal_intensity"]["mean"] if "photometric" in conditions else None
        o0 = summaries["0.0"]["doa_direct"]["terminal_intensity"]["mean"] if "doa_direct" in conditions else None
        anchor = {
            "checked": True,
            "photometric_terminal_bias0": p0,
            "oracle_terminal_bias0": o0,
            "expected_photometric": NOMINAL_PHOTOMETRIC_TERMINAL,
            "expected_oracle": NOMINAL_ORACLE_TERMINAL,
            "atol": ANCHOR_ATOL,
            "pass": bool(
                (p0 is None or abs(p0 - NOMINAL_PHOTOMETRIC_TERMINAL) <= ANCHOR_ATOL)
                and (o0 is None or abs(o0 - NOMINAL_ORACLE_TERMINAL) <= ANCHOR_ATOL)
            ),
        }

    _write_csv(os.path.join(out_dir, "trial-outcomes.csv"), trial_rows,
               ["mirror_bias_deg", "condition", "seed", "terminal_intensity",
                "time_to_threshold", "censored", "ceiling", "regret"])
    _write_csv(os.path.join(out_dir, "level-summary.csv"), level_rows,
               ["mirror_bias_deg", "condition", "n_seeds", "terminal_mean", "terminal_median",
                "terminal_ci95_lo", "terminal_ci95_hi", "ttt_median", "n_failed",
                "success_fraction", "mean_regret"])
    _write_csv(os.path.join(out_dir, "boundary-map.csv"), boundary_rows,
               ["mirror_bias_deg", "class", "photometric_median_terminal", "oracle_median_terminal",
                "median_lead_photometric_minus_oracle", "mannwhitney_u", "mannwhitney_p", "ceiling_median"])

    completed = time.time()
    manifest = {
        "experiment": "h3_mismatch_robustness",
        "hypothesis": "Under model mismatch (mirror-calibration bias) the photometric "
                      "controller matches/beats the analytic oracle on terminal accuracy; "
                      "the oracle is only a local (nominal-geometry) upper bound.",
        "lever": "mirror_normal_calibration_bias_about_world_x",
        "status": status,
        "flip_confirmed": flip,
        "falsifier": "no bias level is photometric_dominant before both_fail",
        "falsifier_fired": not flip,
        "levels": levels,
        "conditions": conditions,
        "seeds": args.seeds,
        "steps": args.steps,
        "particle_count": args.particle_count,
        "success_floor": SUCCESS_FLOOR,
        "alpha": ALPHA,
        "primary_metric": "terminal_intensity",
        "anchor_bias0": anchor,
        "boundary_map": boundary_rows,
        "class_counts": {c: classes.count(c) for c in sorted(set(classes))},
        "summaries": summaries,
        "wall_time_seconds": completed - started,
        "started_at": started,
        "completed_at": completed,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[h3] status: {status}")
    print(f"[h3] boundary map:")
    for r in boundary_rows:
        print(f"      bias={r['mirror_bias_deg']:5.1f}  {r['class']:22s} "
              f"photo_med={r['photometric_median_terminal']:.3f} "
              f"oracle_med={r['oracle_median_terminal']:.3f} (p={r['mannwhitney_p']:.3g})")
    print(f"[h3] anchor(bias=0) pass={anchor.get('pass')}; "
          f"wall={completed - started:.1f}s; wrote {out_dir}")


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="H3 robustness-under-mismatch sweep.")
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--levels", type=float, nargs="+", default=None,
                    help="Mirror-bias levels in degrees (default 0 2 5 10 15 20).")
    ap.add_argument("--conditions", type=str, nargs="+", default=None)
    ap.add_argument("--particle-count", type=int, default=64)
    ap.add_argument("--results-dir", type=str, default=RESULTS_DIR_DEFAULT)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
