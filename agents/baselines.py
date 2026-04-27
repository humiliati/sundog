"""Baseline agents for the v2 alignment task.

Three baselines:

- DOADirectAgent: full oracle access (laser_pos, target_detector_pos).
  Solves the analytic mirror-orientation problem once at episode start
  via optics.optimal_joint_angles, then commands the result every step.
  This is the upper bound: a target-aware agent with perfect geometry.
  PhotometricAgent should approach but not exceed its convergence speed.

- DOANoisyAgent: same as DOADirectAgent but the oracle is corrupted with
  zero-mean Gaussian noise on (laser_xy, target_xy). Tests how
  sensitive the analytic baseline is to perception error. We expect
  noise_std ~ 0.05 m to be roughly comparable to what the photometric
  agent learns implicitly.

- RandomAgent: returns uniformly random target angles in joint range.
  Lower bound; included so the comparison is grounded.

All baselines return (theta_x_target, theta_y_target) in radians.
"""

from __future__ import annotations

import numpy as np

from sundog import optics
from sundog.env_v2 import Observation, Oracle


class DOADirectAgent:
    """Target-aware analytic baseline.

    Receives an Oracle at reset() and computes optimal joint angles once.
    The same target is commanded every step; the env's PD loop handles
    servoing. Convergence time for this agent is essentially the joint
    settling time.
    """

    def __init__(self, joint_limit: float = optics.JOINT_LIMIT):
        self.joint_limit = float(joint_limit)
        self._target_action: np.ndarray = np.zeros(2)

    def reset(self, oracle: Oracle) -> None:
        tx, ty = optics.optimal_joint_angles(
            laser_pos=oracle.laser_pos,
            target_pos=oracle.target_detector_pos,
        )
        self._target_action = np.array(
            [
                np.clip(tx, -self.joint_limit, self.joint_limit),
                np.clip(ty, -self.joint_limit, self.joint_limit),
            ]
        )

    def act(self, obs: Observation) -> np.ndarray:
        return self._target_action.copy()


class DOANoisyAgent:
    """Target-aware baseline with Gaussian noise on oracle xy.

    Noise is drawn ONCE at reset() and held for the episode (representing a
    static perception error like a miscalibrated camera). This makes
    convergence terminal-error a direct measure of solve sensitivity. If we
    drew noise per step instead, the agent would average over noise and
    look better than it should.
    """

    def __init__(self, noise_std: float = 0.05, joint_limit: float = optics.JOINT_LIMIT, seed: int | None = None):
        self.noise_std = float(noise_std)
        self.joint_limit = float(joint_limit)
        self.rng = np.random.default_rng(seed)
        self._target_action: np.ndarray = np.zeros(2)

    def reset(self, oracle: Oracle) -> None:
        laser_noisy = oracle.laser_pos.copy()
        laser_noisy[:2] += self.rng.normal(0.0, self.noise_std, size=2)
        target_noisy = oracle.target_detector_pos.copy()
        target_noisy[:2] += self.rng.normal(0.0, self.noise_std, size=2)
        tx, ty = optics.optimal_joint_angles(
            laser_pos=laser_noisy,
            target_pos=target_noisy,
        )
        self._target_action = np.array(
            [
                np.clip(tx, -self.joint_limit, self.joint_limit),
                np.clip(ty, -self.joint_limit, self.joint_limit),
            ]
        )

    def act(self, obs: Observation) -> np.ndarray:
        return self._target_action.copy()


class RandomAgent:
    """Uniform random joint targets. Re-sampled every step."""

    def __init__(self, joint_limit: float = optics.JOINT_LIMIT, seed: int | None = None):
        self.joint_limit = float(joint_limit)
        self.rng = np.random.default_rng(seed)

    def reset(self, oracle: Oracle | None = None) -> None:
        # Accept oracle for API symmetry; ignore it.
        del oracle

    def act(self, obs: Observation) -> np.ndarray:
        return self.rng.uniform(-self.joint_limit, self.joint_limit, size=2)
