"""Stress-test framework: sweep a single perturbation across levels.

Architecture
------------
- `Stressor` dataclass holds all perturbation parameters with sensible defaults.
- `make_env(stressor)` constructs a SundogEnvV2 with stressor-applied env params.
- `make_agent(condition, stressor, seed, initial_obs, oracle)` constructs the
  appropriate agent with stressor-applied agent params.
- `apply_obs_perturbation(obs, stressor, rng)` adds detector noise to the
  observation between env.step() and agent.act() so the agent sees noisy
  intensities while ground-truth target_intensity remains unperturbed for
  metrics.
- `run_sweep(stressor_name, levels, ...)` runs the matrix of (level x condition
  x seed) and saves NPZ files per episode plus a per-level summary CSV.

Stressors implemented
---------------------
- detector_noise   : Gaussian additive noise on obs.detector_intensities
                     (clipped to [0, 1]). Levels: 0.0, 0.02, 0.05, 0.10, 0.20.
- beam_sigma       : Env Gaussian-spot width. Levels: 0.05, 0.10, 0.15, 0.25, 0.40.
- scan_duration    : Photometric SCAN window in seconds. Levels: 1, 2, 4, 8, 16.
- laser_height     : Laser z-coordinate. Levels: 1.5, 2.0, 2.5, 3.0, 3.5.
- joint_limit      : Symmetric joint limit in radians. Levels: 0.8, 1.0, 1.2, 1.5.
- distractor_boost : Multiplicative boost on detector_4 (opposite the target)
                     in the agent-visible obs. Levels: 1.0 (no boost), 1.5,
                     2.0, 3.0, 5.0.

NOTE on distractor_boost (caveat). The current photometric agent reads only
`obs.detector_intensities[TARGET_DETECTOR_INDEX]` (detector_0). Boosting a
non-target detector therefore has *zero effect* on this agent. The stressor
is included for completeness and to enable a planned max-intensity agent
that picks the brightest detector and tries to align to it; that agent would
genuinely fail under distractor boost. Until that agent exists, omitting
distractor_boost from the headline sweep is the right call.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np

from sundog import optics
from sundog.env_v2 import (
    SundogEnvV2,
    Observation,
    Oracle,
    N_DETECTORS,
    TARGET_DETECTOR_INDEX,
)
from sundog.agents.photometric import PhotometricAgent
from sundog.agents.baselines import DOADirectAgent, DOANoisyAgent, RandomAgent

DEFAULT_SEEDS = 30
DEFAULT_STEPS = 500
RESULTS_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "results", "stress_tests",
)


@dataclass
class Stressor:
    """Single perturbation specification.

    Default values reproduce the baseline scene from run_baseline_comparison.
    Each stressor varies exactly one field while the others remain at default.
    """
    detector_noise_sigma: float = 0.0
    beam_sigma: float = optics.DEFAULT_SIGMA   # 0.15
    laser_height: float = 2.5
    joint_limit: float = optics.JOINT_LIMIT     # 1.5
    photometric_scan_duration_s: float = 4.0
    distractor_boost: float = 1.0
    distractor_index: int = 4   # opposite the target on the ring
    # Model-mismatch lever (H3): fixed unmodelled tilt of the mirror's reflecting
    # normal by mirror_bias_deg degrees (in the plane spanned by the normal and
    # mirror_bias_ref). Invisible to the analytic oracle, which assumes an ideal
    # mirror.
    mirror_bias_deg: float = 0.0
    mirror_bias_ref: tuple[float, float, float] = (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Env / agent construction with stressor applied
# ---------------------------------------------------------------------------

def make_env(stressor: Stressor, seed: int) -> SundogEnvV2:
    env = SundogEnvV2(sigma=stressor.beam_sigma, seed=seed)
    return env


def apply_laser_height(env: SundogEnvV2, stressor: Stressor) -> None:
    """Override the laser z position after reset()."""
    site_id = env.model.site("laser_source").id
    env.model.site_pos[site_id, 2] = float(stressor.laser_height)
    mujoco.mj_forward(env.model, env.data)


def apply_mirror_bias(env: SundogEnvV2, stressor: Stressor) -> None:
    """Install the H3 mirror-normal calibration bias on the env (no-op at 0 deg)."""
    env.set_normal_bias(stressor.mirror_bias_deg, stressor.mirror_bias_ref)


def apply_joint_limit_to_action(action: np.ndarray, stressor: Stressor) -> np.ndarray:
    return np.clip(action, -stressor.joint_limit, stressor.joint_limit)


def apply_obs_perturbation(
    obs: Observation,
    stressor: Stressor,
    rng: np.random.Generator,
) -> Observation:
    """Return a perturbed COPY of obs that the agent sees.

    Distractor boost is applied multiplicatively to the named distractor
    detector. Detector noise is additive Gaussian, clipped to [0, 1].
    Ground-truth target intensity (used by the runner for metrics) is read
    from the original obs, NOT this perturbed copy.
    """
    perturbed = obs.detector_intensities.copy()
    if stressor.distractor_boost != 1.0:
        idx = stressor.distractor_index
        if 0 <= idx < perturbed.shape[0]:
            perturbed[idx] *= stressor.distractor_boost
    if stressor.detector_noise_sigma > 0:
        perturbed += rng.normal(0, stressor.detector_noise_sigma, size=perturbed.shape)
    perturbed = np.clip(perturbed, 0.0, 1.0)
    return Observation(
        detector_intensities=perturbed,
        joint_angles=obs.joint_angles,
        joint_velocities=obs.joint_velocities,
        joint_torques=obs.joint_torques,
    )


def make_agent(
    condition: str,
    stressor: Stressor,
    seed: int,
    initial_obs: Observation,
    oracle: Oracle,
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
    raise ValueError(f"unknown condition: {condition}")


# ---------------------------------------------------------------------------
# Episode / sweep
# ---------------------------------------------------------------------------

def laser_xy_for_seed(seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed * 1_000_003 + 17)
    return tuple(rng.uniform(-0.4, 0.4, size=2))


def initial_qpos_for_seed(seed: int, std: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed * 9_999_991 + 41)
    return rng.normal(0.0, std, size=2)


def run_episode_with_stressor(
    env: SundogEnvV2,
    condition: str,
    stressor: Stressor,
    seed: int,
    n_steps: int = DEFAULT_STEPS,
) -> dict:
    laser_xy = laser_xy_for_seed(seed)
    obs_clean = env.reset(laser_xy=laser_xy)
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
    agent = make_agent(condition, stressor, seed, obs_seen, oracle)

    target_intensity = np.empty(n_steps, dtype=np.float64)
    joint_angles = np.empty((n_steps, 2), dtype=np.float64)

    for t in range(n_steps):
        action = agent.act(obs_seen)
        action = apply_joint_limit_to_action(action, stressor)
        obs_clean = env.step(action)
        obs_seen = apply_obs_perturbation(obs_clean, stressor, rng)
        target_intensity[t] = obs_clean.detector_intensities[TARGET_DETECTOR_INDEX]
        joint_angles[t] = obs_clean.joint_angles

    return {
        "target_intensity": target_intensity,
        "joint_angles": joint_angles,
        "laser_xy": np.asarray(laser_xy),
        "condition": condition,
    }


# Stressor presets (which field to vary, default sweep levels).
STRESSOR_SPECS = {
    "detector_noise":   ("detector_noise_sigma",         [0.0, 0.02, 0.05, 0.10, 0.20]),
    "beam_sigma":       ("beam_sigma",                   [0.05, 0.10, 0.15, 0.25, 0.40]),
    "scan_duration":    ("photometric_scan_duration_s",  [1.0, 2.0, 4.0, 8.0, 16.0]),
    "laser_height":     ("laser_height",                 [1.5, 2.0, 2.5, 3.0, 3.5]),
    "joint_limit":      ("joint_limit",                  [0.8, 1.0, 1.2, 1.5]),
    "distractor_boost": ("distractor_boost",             [1.0, 1.5, 2.0, 3.0, 5.0]),
    "mirror_bias":      ("mirror_bias_deg",              [0.0, 2.0, 5.0, 10.0, 15.0, 20.0]),
}


def run_sweep(
    stressor_name: str,
    levels: Optional[list] = None,
    seeds: int = DEFAULT_SEEDS,
    steps: int = DEFAULT_STEPS,
    conditions: Optional[list[str]] = None,
    results_dir: str = RESULTS_DIR_DEFAULT,
) -> dict:
    if stressor_name not in STRESSOR_SPECS:
        raise ValueError(f"unknown stressor: {stressor_name}. Choose from {list(STRESSOR_SPECS.keys())}")
    field_name, default_levels = STRESSOR_SPECS[stressor_name]
    if levels is None:
        levels = default_levels
    if conditions is None:
        conditions = ["photometric", "doa_direct", "doa_noisy", "random"]

    out_root = os.path.join(results_dir, stressor_name)
    os.makedirs(out_root, exist_ok=True)

    summary: dict = {"stressor": stressor_name, "field": field_name, "levels": levels, "results": {}}

    for level in levels:
        kwargs = {field_name: level}
        stressor = Stressor(**kwargs)
        level_dir = os.path.join(out_root, f"level_{level}")
        for cond in conditions:
            cond_dir = os.path.join(level_dir, cond)
            os.makedirs(cond_dir, exist_ok=True)
            terminal_means = []
            t0 = time.time()
            env = SundogEnvV2(sigma=stressor.beam_sigma, seed=0)
            for s in range(seeds):
                rec = run_episode_with_stressor(env, cond, stressor, s, n_steps=steps)
                np.savez(
                    os.path.join(cond_dir, f"seed_{s:03d}.npz"),
                    target_intensity=rec["target_intensity"],
                    joint_angles=rec["joint_angles"],
                    laser_xy=rec["laser_xy"],
                    condition=rec["condition"],
                    stressor_name=stressor_name,
                    stressor_level=level,
                )
                terminal_means.append(float(np.mean(rec["target_intensity"][-50:])))
            wall = time.time() - t0
            mean_terminal = float(np.mean(terminal_means))
            std_terminal = float(np.std(terminal_means, ddof=1))
            print(f"[stress] {stressor_name}={level}  {cond:12s}  "
                  f"mean_terminal={mean_terminal:.3f}  std={std_terminal:.3f}  "
                  f"in {wall:.1f}s")
            summary["results"].setdefault(str(level), {})[cond] = {
                "mean_terminal": mean_terminal,
                "std_terminal": std_terminal,
                "values": terminal_means,
            }

    with open(os.path.join(out_root, "sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[stress] sweep complete; summary -> {os.path.join(out_root, 'sweep_summary.json')}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a stress-test sweep.")
    parser.add_argument("--stressor", type=str, required=True,
                        choices=list(STRESSOR_SPECS.keys()))
    parser.add_argument("--levels", type=float, nargs="+", default=None,
                        help="Override the default level sweep.")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--conditions", type=str, nargs="+", default=None)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR_DEFAULT)
    args = parser.parse_args()
    run_sweep(
        stressor_name=args.stressor,
        levels=args.levels,
        seeds=args.seeds,
        steps=args.steps,
        conditions=args.conditions,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
