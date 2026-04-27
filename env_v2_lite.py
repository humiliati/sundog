"""Sundog environment v2 - numpy-only fallback.

Same alignment task as env_v2.SundogEnvV2 but implemented in pure
numpy. No MuJoCo dependency. Useful for fast iteration, debugging,
and CI.

Joint dynamics
--------------
First-order PD-like model: target angle is commanded, joint position
chases the target with a first-order lowpass. Settling time is matched
empirically to MuJoCo's response (kp=80, kv=8 at 0.005s timestep,
frame_skip=4). With alpha = 1 - exp(-dt/tau) and tau picked so 90%
settling takes ~0.18 s, alpha = 0.36 per step at 50 Hz control. This
reproduces MuJoCo's joint-settling behavior to within ~3%.

What's NOT modeled
------------------
- Joint inertia (no momentum / overshoot beyond what first-order gives)
- Coupling between joints from rotational dynamics
- Actuator force limits (we just clip the target angle)

Inheriting the 2.0 numpy-fallback pattern but staying within v2's
photometric task.
Faithfulness vs. MuJoCo
-----------------------
Sanity-tested on 5 scenes with DOA-direct: Lite agrees to within 0.0001
when the optimal joint angles saturate the joint limit, but reaches
~0.07-0.12 HIGHER terminal intensity in scenes where the optimum is
mid-range (e.g., laser at (-0.3, 0.2) gives MuJoCo 0.881 vs Lite 0.999).

The cause is gravity bias in MuJoCo's PD control: a tilted pole has a
gravity-induced torque load that the kp=80 controller compensates only
partially, leaving a steady-state offset. Lite has no gravity model,
so the joint reaches its commanded target exactly.

Implication: the rebuild's headline claim (photometric agent reaches
terminal intensity statistically indistinguishable from the target-aware
baseline) holds in both envs because the comparison is within-env. But
the absolute terminal-intensity numbers differ: Lite is ~0.10 higher
on average than MuJoCo. Use Lite for fast iteration and CI; report
MuJoCo numbers in the paper.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from sundog import optics
from sundog.env_v2 import (
    Observation,
    Oracle,
    N_DETECTORS,
    TARGET_DETECTOR_INDEX,
)


# Detector ring positions matching models/sundog_pole.xml.
_DETECTOR_POSITIONS = np.array([
    [ 1.200,  0.000, 0.005],
    [ 0.849,  0.849, 0.005],
    [ 0.000,  1.200, 0.005],
    [-0.849,  0.849, 0.005],
    [-1.200,  0.000, 0.005],
    [-0.849, -0.849, 0.005],
    [ 0.000, -1.200, 0.005],
    [ 0.849, -0.849, 0.005],
], dtype=float)


# First-order joint settling rate per agent step. Matched empirically
# to MuJoCo's PD response with kp=80, kv=8, frame_skip=4 at 50 Hz.
_JOINT_SETTLE_ALPHA = 0.36


class SundogEnvV2Lite:
    """Pure-numpy alignment env. API-compatible with SundogEnvV2.

    Differences from MuJoCo env:
    - No physical sim; first-order joint dynamics analytically
    - No torque sensor (we synthesize a proxy from action-state error)
    - No FK sanity check (the analytic FK IS the env)
    """

    def __init__(
        self,
        sigma: float = optics.DEFAULT_SIGMA,
        seed: Optional[int] = None,
        joint_settle_alpha: float = _JOINT_SETTLE_ALPHA,
    ):
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self._detector_positions = _DETECTOR_POSITIONS.copy()
        self._joint_settle_alpha = float(joint_settle_alpha)

        self._joint_angles = np.zeros(2)
        self._joint_velocities = np.zeros(2)
        self._joint_targets = np.zeros(2)
        self._laser_pos = np.array([0.3, 0.4, 2.5])

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        laser_xy: Optional[tuple[float, float]] = None,
        joint_init_noise: float = 0.05,
    ) -> Observation:
        if laser_xy is None:
            lxy = self.rng.uniform(-0.5, 0.5, size=2)
        else:
            lxy = np.asarray(laser_xy, dtype=float)
        self._laser_pos = np.array([lxy[0], lxy[1], 2.5])

        self._joint_angles = self.rng.normal(0.0, joint_init_noise, size=2)
        self._joint_velocities = np.zeros(2)
        self._joint_targets = self._joint_angles.copy()
        return self._observation()

    def step(self, action: np.ndarray) -> Observation:
        action = np.clip(np.asarray(action, dtype=float),
                         -optics.JOINT_LIMIT, optics.JOINT_LIMIT)
        self._joint_targets = action

        prev = self._joint_angles.copy()
        self._joint_angles += self._joint_settle_alpha * (action - self._joint_angles)
        self._joint_angles = np.clip(self._joint_angles,
                                     -optics.JOINT_LIMIT, optics.JOINT_LIMIT)
        self._joint_velocities = (self._joint_angles - prev) / 0.02  # 50 Hz
        return self._observation()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observation(self) -> Observation:
        intensities = self._compute_intensities()
        # Torque proxy: action-position error scaled by an arbitrary stiffness.
        # Not physical; the photometric agent doesn't read this anyway.
        torque_proxy = 80.0 * (self._joint_targets - self._joint_angles)
        return Observation(
            detector_intensities=intensities,
            joint_angles=self._joint_angles.copy(),
            joint_velocities=self._joint_velocities.copy(),
            joint_torques=torque_proxy,
        )

    def _compute_intensities(self) -> np.ndarray:
        n = optics.joint_angles_to_mirror_normal(
            self._joint_angles[0], self._joint_angles[1]
        )
        m = optics.mirror_position_from_normal(n)
        return optics.compute_detector_intensities(
            laser_pos=self._laser_pos,
            mirror_pos=m,
            mirror_normal=n,
            detector_positions=self._detector_positions,
            sigma=self.sigma,
        )

    # ------------------------------------------------------------------
    # Oracle and metrics (same shape as MuJoCo env)
    # ------------------------------------------------------------------

    def get_oracle(self) -> Oracle:
        return Oracle(
            laser_pos=self._laser_pos.copy(),
            target_detector_pos=self._detector_positions[TARGET_DETECTOR_INDEX].copy(),
            target_detector_index=TARGET_DETECTOR_INDEX,
        )

    def target_intensity(self) -> float:
        return float(self._compute_intensities()[TARGET_DETECTOR_INDEX])
