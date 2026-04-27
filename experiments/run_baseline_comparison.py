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
import json
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from sundog.env_v2 import SundogEnvV2, Observation, Oracle, N_DETECTORS
from sundog.agents.photometric import PhotometricAgent
from sundog.agents.baselines import DOADirectAgent, DOANoisyAgent, RandomAgent


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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run sundog baseline comparison.")
    parser.add_argument("--seeds", type=int, default=N_SEEDS,
                        help="Number of seeds per condition.")
    parser.add_argument("--steps", type=int, default=N_STEPS,
                        help="Steps per episode.")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR_DEFAULT,
                        help="Output directory.")
    parser.add_argument("--conditions", type=str, nargs="+",
                        default=list(CONDITION_FACTORIES.keys()),
                        help="Subset of conditions to run.")
    args = parser.parse_args()

    env = SundogEnvV2()
    print(f"[runner] env loaded; sanity check passed; running {len(args.conditions)} conditions x {args.seeds} seeds")

    summary: dict[str, dict] = {}
    for condition in args.conditions:
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
