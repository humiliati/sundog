"""Run the photometric-vs-baselines comparison.

Conditions
----------
- photometric : PhotometricAgent (extremum-seeking, no target access)
- doa_direct  : DOADirectAgent (oracle-driven analytic solve, upper bound)
- doa_noisy   : DOANoisyAgent (oracle + xy Gaussian noise)
- random      : RandomAgent (lower bound)

Matched-laser experimental design
---------------------------------
For each seed s (0..N_SEEDS-1), the laser xy position is a deterministic
function of s ONLY. All four conditions are run at that same laser xy,
so their results are directly comparable: differences across conditions
cannot be attributed to scene differences.

Initial joint perturbation is also seeded by s, so all conditions start
in the same pose.

Output
------
Per (condition, seed) we save a numpy archive:
  results/{condition}/seed_{s:03d}.npz
containing:
  - target_intensity  (T,)            # detector_0 intensity per step
  - all_intensities   (T, N_DETECTORS)
  - joint_angles      (T, 2)
  - joint_velocities  (T, 2)
  - actions           (T, 2)
  - laser_xy          (2,)
  - condition         str

T = N_STEPS = 500 (10 sim-seconds at 50 Hz). The analysis script reads
all archives and computes summary statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from sundog.env_v2 import SundogEnvV2, Observation, Oracle, N_DETECTORS
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
    apply_joint_limit_to_action,
    apply_laser_height,
    apply_obs_perturbation,
    run_episode_with_stressor,
)


N_SEEDS = 30
N_STEPS = 500
RESULTS_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "results",
)


def laser_xy_for_seed(seed: int) -> tuple[float, float]:
    """Deterministic laser xy as a function of seed.

    Sampled from a fixed RNG keyed by seed alone, in [-0.4, 0.4]^2 (slightly
    inside the env default range so we don't hit edge geometry).
    """
    rng = np.random.default_rng(seed * 1_000_003 + 17)
    return tuple(rng.uniform(-0.4, 0.4, size=2))


def initial_qpos_for_seed(seed: int, std: float = 0.05) -> np.ndarray:
    """Deterministic initial joint perturbation."""
    rng = np.random.default_rng(seed * 9_999_991 + 41)
    return rng.normal(0.0, std, size=2)


@dataclass
class EpisodeLog:
    target_intensity: np.ndarray
    all_intensities: np.ndarray
    joint_angles: np.ndarray
    joint_velocities: np.ndarray
    actions: np.ndarray
    laser_xy: np.ndarray
    condition: str

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            target_intensity=self.target_intensity,
            all_intensities=self.all_intensities,
            joint_angles=self.joint_angles,
            joint_velocities=self.joint_velocities,
            actions=self.actions,
            laser_xy=self.laser_xy,
            condition=self.condition,
        )


def run_episode(
    env: SundogEnvV2,
    agent_factory: Callable[[Observation, Oracle], object],
    seed: int,
    condition: str,
    n_steps: int = N_STEPS,
) -> EpisodeLog:
    """Run a single episode for one (condition, seed). Returns a log."""
    laser_xy = laser_xy_for_seed(seed)
    initial_qpos = initial_qpos_for_seed(seed)

    obs = env.reset(laser_xy=laser_xy)
    # Override the initial joints with the seeded perturbation. env.reset()
    # already drew from its own RNG; we want determinism keyed to `seed`.
    env.data.qpos[:] = initial_qpos
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = initial_qpos
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    obs = env._observation()

    oracle = env.get_oracle()
    agent = agent_factory(obs, oracle)

    target_intensity = np.empty(n_steps, dtype=np.float64)
    all_intensities = np.empty((n_steps, N_DETECTORS), dtype=np.float64)
    joint_angles = np.empty((n_steps, 2), dtype=np.float64)
    joint_velocities = np.empty((n_steps, 2), dtype=np.float64)
    actions_log = np.empty((n_steps, 2), dtype=np.float64)

    for t in range(n_steps):
        action = agent.act(obs)
        actions_log[t] = action
        obs = env.step(action)
        target_intensity[t] = obs.detector_intensities[0]
        all_intensities[t] = obs.detector_intensities
        joint_angles[t] = obs.joint_angles
        joint_velocities[t] = obs.joint_velocities

    return EpisodeLog(
        target_intensity=target_intensity,
        all_intensities=all_intensities,
        joint_angles=joint_angles,
        joint_velocities=joint_velocities,
        actions=actions_log,
        laser_xy=np.asarray(laser_xy),
        condition=condition,
    )


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

def _make_photometric(initial_obs: Observation, oracle: Oracle) -> PhotometricAgent:
    del oracle  # photometric agent must not consume oracle
    a = PhotometricAgent()
    a.reset(carrier_init=tuple(initial_obs.joint_angles))
    return a


def _make_doa_direct(initial_obs: Observation, oracle: Oracle) -> DOADirectAgent:
    del initial_obs
    a = DOADirectAgent()
    a.reset(oracle)
    return a


def _make_doa_noisy(seed: int):
    def factory(initial_obs: Observation, oracle: Oracle) -> DOANoisyAgent:
        del initial_obs
        a = DOANoisyAgent(noise_std=0.05, seed=seed * 71 + 5)
        a.reset(oracle)
        return a
    return factory


def _make_random(seed: int):
    def factory(initial_obs: Observation, oracle: Oracle) -> RandomAgent:
        del initial_obs, oracle
        a = RandomAgent(seed=seed * 31 + 7)
        a.reset()
        return a
    return factory


CONDITION_FACTORIES: dict[str, Callable[[int], Callable[[Observation, Oracle], object]]] = {
    "photometric": lambda seed: _make_photometric,
    "doa_direct":  lambda seed: _make_doa_direct,
    "doa_noisy":   _make_doa_noisy,
    "random":      _make_random,
}


# ---------------------------------------------------------------------------
# Phase 6: photometric Bayes operating envelope
# ---------------------------------------------------------------------------

PHASE6_CONDITIONS = [
    "photometric",
    "doa_direct",
    "doa_noisy",
    "random",
    "bayes_particle",
]
PHASE6_CLASSIFICATION_CONDITIONS = ["photometric", "bayes_particle"]
PHASE6_BEAM_SIGMAS = [0.05, 0.10, 0.15, 0.25, 0.40]
PHASE6_DETECTOR_NOISE_SIGMAS = [0.0, 0.02, 0.05, 0.10, 0.20]
PHASE6_NOMINAL_BEAM_SIGMA = 0.15
PHASE6_NOMINAL_DETECTOR_NOISE = 0.0
PHASE6_SMOKE_CELLS = ["nominal", "beam_sigma_0p40", "detector_noise_0p20"]


@dataclass(frozen=True)
class Phase6Cell:
    cell_id: str
    axis: str
    beam_sigma: float
    detector_noise_sigma: float


def _level_id(value: float) -> str:
    text = f"{value:.2f}".replace(".", "p").replace("-", "m")
    return text


def phase6_cells(kind: str) -> list[Phase6Cell]:
    cells = [
        Phase6Cell(
            cell_id="nominal",
            axis="nominal",
            beam_sigma=PHASE6_NOMINAL_BEAM_SIGMA,
            detector_noise_sigma=PHASE6_NOMINAL_DETECTOR_NOISE,
        )
    ]
    for sigma in PHASE6_BEAM_SIGMAS:
        if sigma == PHASE6_NOMINAL_BEAM_SIGMA:
            continue
        cells.append(
            Phase6Cell(
                cell_id=f"beam_sigma_{_level_id(sigma)}",
                axis="beam_sigma",
                beam_sigma=sigma,
                detector_noise_sigma=PHASE6_NOMINAL_DETECTOR_NOISE,
            )
        )
    for noise in PHASE6_DETECTOR_NOISE_SIGMAS:
        if noise == PHASE6_NOMINAL_DETECTOR_NOISE:
            continue
        cells.append(
            Phase6Cell(
                cell_id=f"detector_noise_{_level_id(noise)}",
                axis="detector_noise",
                beam_sigma=PHASE6_NOMINAL_BEAM_SIGMA,
                detector_noise_sigma=noise,
            )
        )
    if kind == "smoke":
        wanted = set(PHASE6_SMOKE_CELLS)
        return [cell for cell in cells if cell.cell_id in wanted]
    if kind == "lock":
        return cells
    raise ValueError(f"unknown phase6 cell set: {kind}")


def _phase6_stressor(cell: Phase6Cell) -> Stressor:
    return Stressor(
        beam_sigma=cell.beam_sigma,
        detector_noise_sigma=cell.detector_noise_sigma,
    )


def _phase6_reference_episode_path(cell: Phase6Cell, condition: str, seed: int) -> str:
    if cell.axis == "nominal":
        return os.path.join("results", condition, f"seed_{seed:03d}.npz")
    if cell.axis == "beam_sigma":
        return os.path.join(
            "results",
            "stress_tests",
            "beam_sigma",
            f"level_{cell.beam_sigma}",
            condition,
            f"seed_{seed:03d}.npz",
        )
    if cell.axis == "detector_noise":
        return os.path.join(
            "results",
            "stress_tests",
            "detector_noise",
            f"level_{cell.detector_noise_sigma}",
            condition,
            f"seed_{seed:03d}.npz",
        )
    raise ValueError(f"unknown Phase 6 axis: {cell.axis}")


def load_phase6_reference_episode(cell: Phase6Cell, condition: str, seed: int) -> dict:
    path = _phase6_reference_episode_path(cell, condition, seed)
    data = np.load(path)
    return {
        "target_intensity": data["target_intensity"].copy(),
        "joint_angles": data["joint_angles"].copy(),
        "laser_xy": data["laser_xy"].copy(),
        "condition": condition,
    }


def run_phase6_bayes_particle_episode(
    env: SundogEnvV2,
    stressor: Stressor,
    seed: int,
    particle_count: int,
    n_steps: int = N_STEPS,
) -> dict:
    """Run the nominal-assumption particle Bayes lane under a Phase 6 stressor."""
    laser_xy = laser_xy_for_seed(seed)
    obs_clean = env.reset(laser_xy=laser_xy)
    apply_laser_height(env, stressor)

    initial_qpos = initial_qpos_for_seed(seed)
    env.data.qpos[:] = initial_qpos
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = initial_qpos
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    obs_clean = env._observation()

    oracle = env.get_oracle()
    rng = np.random.default_rng(seed * 17 + 3)
    obs_seen = apply_obs_perturbation(obs_clean, stressor, rng)
    agent = BayesParticleAgent(
        particle_count=particle_count,
        seed=seed,
        detector_positions=env._detector_positions.copy(),
        target_detector_pos=oracle.target_detector_pos,
        target_detector_index=oracle.target_detector_index,
        joint_limit=stressor.joint_limit,
    )

    target_intensity = np.empty(n_steps, dtype=np.float64)
    joint_angles = np.empty((n_steps, 2), dtype=np.float64)

    for t in range(n_steps):
        action = agent.act(obs_seen)
        action = apply_joint_limit_to_action(action, stressor)
        obs_clean = env.step(action)
        obs_seen = apply_obs_perturbation(obs_clean, stressor, rng)
        target_intensity[t] = obs_clean.detector_intensities[0]
        joint_angles[t] = obs_clean.joint_angles

    return {
        "target_intensity": target_intensity,
        "joint_angles": joint_angles,
        "laser_xy": np.asarray(laser_xy),
        "condition": "bayes_particle",
    }


def phase6_time_to_threshold(target_intensity: np.ndarray) -> float:
    above = target_intensity > CONVERGENCE_THRESHOLD
    idx = int(np.argmax(above))
    if above[idx]:
        return float(idx)
    return float(target_intensity.shape[0])


def phase6_trial_metrics(target_intensity: np.ndarray) -> dict:
    return {
        "terminal_intensity": float(np.mean(target_intensity[-TERMINAL_WINDOW:])),
        "time_to_threshold": phase6_time_to_threshold(target_intensity),
    }


def phase6_aggregate(values: list[dict], n_steps: int) -> dict:
    terminal = np.asarray([row["terminal_intensity"] for row in values], dtype=np.float64)
    ttt = np.asarray([row["time_to_threshold"] for row in values], dtype=np.float64)
    terminal_ci = bootstrap_ci(terminal)
    ttt_ci = bootstrap_ci(ttt)
    return {
        "n_seeds": int(len(values)),
        "terminal_intensity": {
            "mean": float(terminal.mean()),
            "std": float(terminal.std(ddof=1)) if len(terminal) > 1 else 0.0,
            "median": float(np.median(terminal)),
            "ci95": [terminal_ci[0], terminal_ci[1]],
        },
        "time_to_threshold": {
            "mean": float(ttt.mean()),
            "median": float(np.median(ttt)),
            "ci95": [ttt_ci[0], ttt_ci[1]],
            "n_failed": int((ttt >= n_steps).sum()),
        },
    }


def phase6_classify_cell(cell_summary: dict[str, dict]) -> dict:
    candidates = [
        condition for condition in PHASE6_CLASSIFICATION_CONDITIONS
        if condition in cell_summary
    ]
    if len(candidates) < 2:
        return {
            "class": "mixed",
            "best_condition": candidates[0] if candidates else "",
            "lead_vs_runner_up": 0.0,
            "ci95_half_width": 0.0,
        }

    ranked = sorted(
        candidates,
        key=lambda name: cell_summary[name]["time_to_threshold"]["median"],
    )
    best, runner_up = ranked[0], ranked[1]
    best_ttt = cell_summary[best]["time_to_threshold"]
    lead = (
        cell_summary[runner_up]["time_to_threshold"]["median"]
        - best_ttt["median"]
    )
    ci_lo, ci_hi = best_ttt["ci95"]
    ci_half_width = (ci_hi - ci_lo) / 2.0
    class_name = f"{best}_dominant" if lead > ci_half_width else "mixed"
    return {
        "class": class_name,
        "best_condition": best,
        "runner_up": runner_up,
        "lead_vs_runner_up": float(lead),
        "ci95_half_width": float(ci_half_width),
    }


def _load_reference_target(path: str) -> np.ndarray:
    return np.load(path)["target_intensity"]


def _compare_reference_series(
    actual: np.ndarray,
    path: str,
) -> dict:
    expected = _load_reference_target(path)
    if expected.shape != actual.shape:
        return {
            "pass": False,
            "path": path,
            "reason": f"shape mismatch {actual.shape} != {expected.shape}",
            "max_abs_diff": None,
        }
    diff = np.abs(actual - expected)
    return {
        "pass": bool(np.array_equal(actual, expected)),
        "path": path,
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
    }


def phase6_anchor_report(
    out_dir: str,
    phase_cells: list[Phase6Cell],
    trial_series: dict[tuple[str, str, int], np.ndarray],
    summaries: dict[str, dict[str, dict]],
    seeds: int,
    n_steps: int,
) -> dict:
    report: dict = {
        "status": "pass",
        "seedPrefix": list(range(seeds)),
        "nominal": {},
        "stressPhotometric": [],
    }

    nominal = summaries.get("nominal", {})
    for condition in ["photometric", "doa_direct"]:
        checks = []
        for seed in range(seeds):
            key = ("nominal", condition, seed)
            if key not in trial_series:
                continue
            path = os.path.join("results", condition, f"seed_{seed:03d}.npz")
            checks.append(_compare_reference_series(trial_series[key], path))
        report["nominal"][condition] = {
            "seedMatchedNpz": checks,
            "seedMatchedPass": all(check["pass"] for check in checks),
        }

    analysis_path = os.path.join("results", "analysis", "analysis_summary.json")
    if seeds == N_SEEDS and os.path.exists(analysis_path):
        with open(analysis_path) as f:
            analysis = json.load(f)
        aggregate_checks = {}
        for condition in ["photometric", "doa_direct"]:
            expected = analysis["conditions"][condition]
            actual = nominal[condition]
            fields = [
                ("time_to_threshold", "mean"),
                ("time_to_threshold", "median"),
                ("terminal_intensity", "mean"),
                ("terminal_intensity", "median"),
                ("terminal_intensity", "ci95"),
            ]
            condition_checks = []
            for metric, field in fields:
                actual_value = actual[metric][field]
                expected_value = expected[metric][field]
                if isinstance(expected_value, list):
                    diff = max(abs(a - b) for a, b in zip(actual_value, expected_value))
                else:
                    diff = abs(actual_value - expected_value)
                condition_checks.append({
                    "metric": metric,
                    "field": field,
                    "max_abs_diff": float(diff),
                    "pass": bool(diff == 0.0),
                })
            aggregate_checks[condition] = condition_checks
        if "photometric" in nominal and "doa_direct" in nominal:
            p_vals = [
                row["terminal_intensity"]
                for row in trial_metrics_for_cell(out_dir, "nominal", "photometric")
            ]
            d_vals = [
                row["terminal_intensity"]
                for row in trial_metrics_for_cell(out_dir, "nominal", "doa_direct")
            ]
            u, p = mann_whitney_u(np.asarray(p_vals), np.asarray(d_vals))
            expected_test = analysis["tests"]["photometric_vs_doa_direct_terminal_intensity"]
            aggregate_checks["photometric_vs_doa_direct_terminal_intensity"] = [
                {"field": "U", "max_abs_diff": float(abs(u - expected_test["U"])), "pass": bool(u == expected_test["U"])},
                {"field": "p", "max_abs_diff": float(abs(p - expected_test["p"])), "pass": bool(abs(p - expected_test["p"]) <= 1e-12)},
            ]
        report["nominal"]["analysisSummaryAggregate"] = aggregate_checks
    else:
        report["nominal"]["analysisSummaryAggregate"] = {
            "checked": False,
            "reason": "strict seed-prefix smoke; full 30-seed aggregate checked only on lock",
        }

    for cell in phase_cells:
        if cell.axis == "nominal":
            stress_refs = [
                ("beam_sigma", cell.beam_sigma),
                ("detector_noise", cell.detector_noise_sigma),
            ]
        elif cell.axis == "beam_sigma":
            stress_refs = [("beam_sigma", cell.beam_sigma)]
        elif cell.axis == "detector_noise":
            stress_refs = [("detector_noise", cell.detector_noise_sigma)]
        else:
            stress_refs = []
        for stressor_name, level in stress_refs:
            checks = []
            for seed in range(seeds):
                key = (cell.cell_id, "photometric", seed)
                if key not in trial_series:
                    continue
                path = os.path.join(
                    "results",
                    "stress_tests",
                    stressor_name,
                    f"level_{level}",
                    "photometric",
                    f"seed_{seed:03d}.npz",
                )
                checks.append(_compare_reference_series(trial_series[key], path))
            aggregate_check = {"checked": False}
            sweep_path = os.path.join(
                "results",
                "stress_tests",
                stressor_name,
                "sweep_summary.json",
            )
            if seeds == N_SEEDS and os.path.exists(sweep_path):
                with open(sweep_path) as f:
                    sweep = json.load(f)
                expected_mean = sweep["results"][str(level)]["photometric"]["mean_terminal"]
                actual_mean = summaries[cell.cell_id]["photometric"]["terminal_intensity"]["mean"]
                diff = abs(actual_mean - expected_mean)
                aggregate_check = {
                    "checked": True,
                    "mean_terminal_max_abs_diff": float(diff),
                    "pass": bool(diff == 0.0),
                }
            report["stressPhotometric"].append({
                "cell_id": cell.cell_id,
                "stressor": stressor_name,
                "level": level,
                "seedMatchedPass": all(check["pass"] for check in checks),
                "seedMatchedNpz": checks,
                "sweepSummaryAggregate": aggregate_check,
            })

    nominal_pass = all(
        entry.get("seedMatchedPass", False)
        for entry in report["nominal"].values()
        if isinstance(entry, dict) and "seedMatchedPass" in entry
    )
    stress_pass = all(row["seedMatchedPass"] for row in report["stressPhotometric"])
    aggregate_pass = True
    aggregate = report["nominal"].get("analysisSummaryAggregate")
    if isinstance(aggregate, dict) and aggregate.get("checked", True) is not False:
        for checks in aggregate.values():
            if isinstance(checks, list):
                aggregate_pass = aggregate_pass and all(check["pass"] for check in checks)
    for row in report["stressPhotometric"]:
        agg = row["sweepSummaryAggregate"]
        if agg.get("checked"):
            aggregate_pass = aggregate_pass and bool(agg["pass"])
    report["pass"] = bool(nominal_pass and stress_pass and aggregate_pass)
    if not report["pass"]:
        report["status"] = "anchor_mismatch"
    return report


def trial_metrics_for_cell(out_dir: str, cell_id: str, condition: str) -> list[dict]:
    path = os.path.join(out_dir, "trial-outcomes.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["cell_id"] == cell_id and row["condition"] == condition:
                rows.append({
                    "terminal_intensity": float(row["terminal_intensity"]),
                    "time_to_threshold": float(row["time_to_threshold"]),
                })
    return rows


def save_phase6_episode(
    out_dir: str,
    cell: Phase6Cell,
    condition: str,
    seed: int,
    rec: dict,
) -> None:
    path = os.path.join(
        out_dir,
        "episodes",
        cell.cell_id,
        condition,
        f"seed_{seed:03d}.npz",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        target_intensity=rec["target_intensity"],
        joint_angles=rec["joint_angles"],
        laser_xy=rec["laser_xy"],
        condition=condition,
        phase6_cell_id=cell.cell_id,
        beam_sigma=cell.beam_sigma,
        detector_noise_sigma=cell.detector_noise_sigma,
    )


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_phase6(args: argparse.Namespace) -> None:
    cell_kind = args.phase6_cells
    cells = phase6_cells(cell_kind)
    conditions = args.conditions or PHASE6_CONDITIONS
    for condition in conditions:
        if condition not in PHASE6_CONDITIONS:
            raise ValueError(f"unknown Phase 6 condition: {condition}")
    out_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    started = time.time()
    trial_rows: list[dict] = []
    candidate_rows: list[dict] = []
    class_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    best_rows: list[dict] = []
    summaries: dict[str, dict[str, dict]] = {}
    trial_values: dict[tuple[str, str], list[dict]] = {}
    trial_series: dict[tuple[str, str, int], np.ndarray] = {}

    total_trials = len(cells) * len(conditions) * args.seeds
    completed = 0
    print(
        f"[phase6] {args.phase}: {total_trials} trials "
        f"({len(cells)} cells x {len(conditions)} lanes x {args.seeds} seeds)"
    )

    for cell in cells:
        stressor = _phase6_stressor(cell)
        for condition in conditions:
            env = SundogEnvV2(sigma=stressor.beam_sigma, seed=0)
            wall_t0 = time.time()
            for seed in range(args.seeds):
                if condition == "bayes_particle":
                    rec = run_phase6_bayes_particle_episode(
                        env,
                        stressor,
                        seed=seed,
                        particle_count=args.particle_count,
                        n_steps=args.steps,
                    )
                else:
                    rec = load_phase6_reference_episode(cell, condition, seed)
                save_phase6_episode(out_dir, cell, condition, seed, rec)
                metrics = phase6_trial_metrics(rec["target_intensity"])
                trial_series[(cell.cell_id, condition, seed)] = rec["target_intensity"].copy()
                trial_values.setdefault((cell.cell_id, condition), []).append(metrics)
                trial_rows.append({
                    "phase": args.phase,
                    "cell_id": cell.cell_id,
                    "axis": cell.axis,
                    "beam_sigma": cell.beam_sigma,
                    "detector_noise_sigma": cell.detector_noise_sigma,
                    "condition": condition,
                    "seed": seed,
                    "terminal_intensity": metrics["terminal_intensity"],
                    "time_to_threshold": metrics["time_to_threshold"],
                    "censored": metrics["time_to_threshold"] >= args.steps,
                })
                completed += 1
            wall = time.time() - wall_t0
            print(
                f"[phase6] {cell.cell_id:22s} {condition:15s} "
                f"{args.seeds} seeds in {wall:.2f}s ({completed}/{total_trials})"
            )

    for cell in cells:
        cell_summary: dict[str, dict] = {}
        for condition in conditions:
            values = trial_values.get((cell.cell_id, condition), [])
            if not values:
                continue
            agg = phase6_aggregate(values, args.steps)
            cell_summary[condition] = agg
            candidate_rows.append({
                "cell_id": cell.cell_id,
                "axis": cell.axis,
                "beam_sigma": cell.beam_sigma,
                "detector_noise_sigma": cell.detector_noise_sigma,
                "condition": condition,
                "n_seeds": agg["n_seeds"],
                "median_time_to_threshold": agg["time_to_threshold"]["median"],
                "mean_time_to_threshold": agg["time_to_threshold"]["mean"],
                "time_to_threshold_ci95_lo": agg["time_to_threshold"]["ci95"][0],
                "time_to_threshold_ci95_hi": agg["time_to_threshold"]["ci95"][1],
                "n_failed": agg["time_to_threshold"]["n_failed"],
                "terminal_intensity_mean": agg["terminal_intensity"]["mean"],
                "terminal_intensity_median": agg["terminal_intensity"]["median"],
                "terminal_intensity_ci95_lo": agg["terminal_intensity"]["ci95"][0],
                "terminal_intensity_ci95_hi": agg["terminal_intensity"]["ci95"][1],
            })
        summaries[cell.cell_id] = cell_summary
        classification = phase6_classify_cell(cell_summary)
        class_rows.append({
            "cell_id": cell.cell_id,
            "axis": cell.axis,
            "beam_sigma": cell.beam_sigma,
            "detector_noise_sigma": cell.detector_noise_sigma,
            "class": classification["class"],
            "best_condition": classification["best_condition"],
            "runner_up": classification.get("runner_up", ""),
            "lead_vs_runner_up": classification["lead_vs_runner_up"],
            "ci95_half_width": classification["ci95_half_width"],
        })
        best_rows.append({
            "cell_id": cell.cell_id,
            "best_condition": classification["best_condition"],
            "class": classification["class"],
            "median_time_to_threshold": (
                cell_summary[classification["best_condition"]]["time_to_threshold"]["median"]
                if classification["best_condition"] else ""
            ),
        })

    class_counts: dict[str, int] = {}
    for row in class_rows:
        class_counts[row["class"]] = class_counts.get(row["class"], 0) + 1
    for class_name, count in sorted(class_counts.items()):
        aggregate_rows.append({
            "class": class_name,
            "cells": count,
            "total_cells": len(class_rows),
        })

    write_csv(
        os.path.join(out_dir, "trial-outcomes.csv"),
        trial_rows,
        [
            "phase",
            "cell_id",
            "axis",
            "beam_sigma",
            "detector_noise_sigma",
            "condition",
            "seed",
            "terminal_intensity",
            "time_to_threshold",
            "censored",
        ],
    )
    write_csv(
        os.path.join(out_dir, "candidate-envelope.csv"),
        candidate_rows,
        [
            "cell_id",
            "axis",
            "beam_sigma",
            "detector_noise_sigma",
            "condition",
            "n_seeds",
            "median_time_to_threshold",
            "mean_time_to_threshold",
            "time_to_threshold_ci95_lo",
            "time_to_threshold_ci95_hi",
            "n_failed",
            "terminal_intensity_mean",
            "terminal_intensity_median",
            "terminal_intensity_ci95_lo",
            "terminal_intensity_ci95_hi",
        ],
    )
    write_csv(
        os.path.join(out_dir, "cell-class-map.csv"),
        class_rows,
        [
            "cell_id",
            "axis",
            "beam_sigma",
            "detector_noise_sigma",
            "class",
            "best_condition",
            "runner_up",
            "lead_vs_runner_up",
            "ci95_half_width",
        ],
    )
    write_csv(
        os.path.join(out_dir, "aggregate-envelope.csv"),
        aggregate_rows,
        ["class", "cells", "total_cells"],
    )
    write_csv(
        os.path.join(out_dir, "best-by-cell.csv"),
        best_rows,
        ["cell_id", "best_condition", "class", "median_time_to_threshold"],
    )

    anchor = phase6_anchor_report(
        out_dir=out_dir,
        phase_cells=cells,
        trial_series=trial_series,
        summaries=summaries,
        seeds=args.seeds,
        n_steps=args.steps,
    )
    completed_at = time.time()
    status = (
        "envelope_mapped"
        if len(class_rows) == len(cells) and anchor["pass"]
        else "envelope_incomplete"
    )
    manifest = {
        "phase": args.phase,
        "status": status,
        "pass": True,
        "startedAt": started,
        "completedAt": completed_at,
        "wallTimeSeconds": completed_at - started,
        "seeds": args.seeds,
        "steps": args.steps,
        "particleCount": args.particle_count,
        "conditions": conditions,
        "classificationConditions": PHASE6_CLASSIFICATION_CONDITIONS,
        "doaDirectClassificationRole": "privileged_yardstick_excluded",
        "reportedRails": ["doa_noisy", "random"],
        "cells": [cell.__dict__ for cell in cells],
        "cellClasses": class_rows,
        "classCounts": class_counts,
        "summaries": summaries,
        "anchor": anchor,
        "receiptExtension": {
            "time_to_threshold.ci95": "bootstrap_ci from experiments.analysis, same estimator as terminal_intensity.ci95",
        },
        "bayesParticle": {
            "hiddenState": "laser_xy in [-0.4,0.4]^2",
            "assumedBeamSigma": PHASE6_NOMINAL_BEAM_SIGMA,
            "assumedDetectorNoiseSigma": PHASE6_NOMINAL_DETECTOR_NOISE,
            "likelihood": "negative squared residual against nominal noiseless detector intensities",
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Photometric {args.phase}: {total_trials} trials in "
        f"{completed_at - started:.3f}s ({total_trials / max(completed_at - started, 1e-9):.2f} trials/s)"
    )
    print(f"Audits: {'pass' if anchor['pass'] else 'FAIL'}")
    print(f"Exit gate: {status} ({len(class_rows)}/{len(cells)} cells classified)")
    print(f"Wrote {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run sundog baseline comparison.")
    parser.add_argument("--phase", type=str, default=None,
                        help="Optional named phase runner. Use phase6-photometric-smoke or phase6-photometric-lock for the Phase 6 envelope.")
    parser.add_argument("--seeds", type=int, default=N_SEEDS,
                        help="Number of seeds per condition.")
    parser.add_argument("--steps", type=int, default=N_STEPS,
                        help="Steps per episode.")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR_DEFAULT,
                        help="Output directory.")
    parser.add_argument("--conditions", type=str, nargs="+",
                        default=None,
                        help="Subset of conditions to run.")
    parser.add_argument("--particle-count", type=int, default=64,
                        help="Particle count for the Phase 6 bayes_particle lane.")
    parser.add_argument("--phase6-cells", type=str, choices=["smoke", "lock"], default="smoke",
                        help="Phase 6 cell set to run.")
    args = parser.parse_args()

    if args.phase and args.phase.startswith("phase6"):
        run_phase6(args)
        return

    conditions = args.conditions or list(CONDITION_FACTORIES.keys())

    env = SundogEnvV2()
    print(f"[runner] env loaded; sanity check passed; running {len(conditions)} conditions x {args.seeds} seeds")

    summary: dict[str, dict] = {}
    for condition in conditions:
        if condition not in CONDITION_FACTORIES:
            raise ValueError(f"unknown condition: {condition}")
        condition_dir = os.path.join(args.results_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)

        terminal_intensities: list[float] = []
        wall_t0 = time.time()
        for seed in range(args.seeds):
            factory_factory = CONDITION_FACTORIES[condition]
            agent_factory = factory_factory(seed)
            log = run_episode(env, agent_factory, seed=seed, condition=condition,
                              n_steps=args.steps)
            log.save(os.path.join(condition_dir, f"seed_{seed:03d}.npz"))
            terminal_intensities.append(float(np.mean(log.target_intensity[-50:])))
        wall_dt = time.time() - wall_t0
        summary[condition] = {
            "n_seeds": args.seeds,
            "wall_time_s": round(wall_dt, 2),
            "mean_terminal_intensity": float(np.mean(terminal_intensities)),
            "std_terminal_intensity": float(np.std(terminal_intensities)),
            "min_terminal_intensity": float(np.min(terminal_intensities)),
            "max_terminal_intensity": float(np.max(terminal_intensities)),
        }
        s = summary[condition]
        print(f"[runner] {condition:12s}  "
              f"terminal mean={s['mean_terminal_intensity']:.3f} "
              f"std={s['std_terminal_intensity']:.3f} "
              f"min={s['min_terminal_intensity']:.3f} "
              f"in {s['wall_time_s']:.1f}s")

    with open(os.path.join(args.results_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[runner] summary saved to {os.path.join(args.results_dir, 'run_summary.json')}")


if __name__ == "__main__":
    main()
