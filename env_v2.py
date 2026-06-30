"""Sundog environment v2 - rebuilt to support the indirect-inference claim.

Key differences from v1 (env.py):

- The agent does NOT receive laser_pos in its observation. The only
  external signal is the 8-vector of detector intensities computed by
  the optics module from the (laser, mirror, mirror-normal) geometry.
- bloom_spread and shadow_pos channels are removed entirely. v1's
  versions of those quantities were trivial functions of ground-truth
  target/tip position; we no longer pretend.
- Ground-truth target information (laser position, target detector
  position) is exposed via `get_oracle()` for use by the DOA-direct
  baseline only. The photometric agent does not call get_oracle().
- Episodes randomize laser xy. Initial joint angles are small random
  offsets, not pre-aligned toward the target.
- Includes a startup sanity check that the analytic forward kinematics
  in optics.py agree with MuJoCo's site geometry (to within 1e-6).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

from sundog import optics


_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "sundog_pole.xml",
)


# Number of detectors in the floor ring; must match the XML.
N_DETECTORS = 8

# Index of the detector that the alignment task targets. The photometric
# agent IS told this index (which detector to peak - like an operator aligning
# a laser knows which power meter to maximize; see agents/photometric.py), but
# never its spatial position. The indirect-inference claim is about position,
# not index identity. Only the evaluation / oracle path uses the position.
TARGET_DETECTOR_INDEX = 0


@dataclass
class Observation:
    """Agent-visible observation. Strictly excludes target position."""

    detector_intensities: np.ndarray  # shape (N_DETECTORS,)
    joint_angles: np.ndarray          # shape (2,)
    joint_velocities: np.ndarray      # shape (2,)
    joint_torques: np.ndarray         # shape (2,)

    def as_array(self) -> np.ndarray:
        """Flat representation for vectorized agents."""
        return np.concatenate(
            [
                self.detector_intensities,
                self.joint_angles,
                self.joint_velocities,
                self.joint_torques,
            ]
        )


@dataclass
class Oracle:
    """Ground-truth info for evaluation and for the DOA-direct baseline.

    Photometric agents must not read this. The canonical experiment runner
    constructs the photometric condition through a factory that discards the
    oracle before instantiating the controller; target-aware baselines receive
    this object explicitly.
    """

    laser_pos: np.ndarray             # shape (3,)
    target_detector_pos: np.ndarray   # shape (3,)
    target_detector_index: int


class SundogEnvV2:
    """MuJoCo environment for the Sundog v2 alignment task.

    Step semantics: action is (target_theta_x, target_theta_y) in radians,
    clipped to joint limits. Each step advances physics by `frame_skip`
    physics ticks (default 4 ticks at 0.005 s each = 0.02 s of sim time
    per agent step, i.e., 50 Hz control rate).

    The episode does not auto-terminate; the runner imposes its own step
    budget. There is no reward returned in obs; the runner queries
    `target_intensity()` as needed.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        sigma: float = optics.DEFAULT_SIGMA,
        frame_skip: int = 4,
        seed: Optional[int] = None,
    ):
        path = model_path or _DEFAULT_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"MuJoCo model not found at {path}")
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.sigma = sigma
        self.frame_skip = frame_skip
        self.rng = np.random.default_rng(seed)

        self._detector_names = [f"detector_{i}" for i in range(N_DETECTORS)]
        # Cache detector world positions (they are world-fixed sites).
        mujoco.mj_forward(self.model, self.data)
        self._detector_positions = np.stack(
            [self.data.site(name).xpos.copy() for name in self._detector_names]
        )

        self._sanity_check_kinematics()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        laser_xy: Optional[tuple[float, float]] = None,
        joint_init_noise: float = 0.05,
    ) -> Observation:
        """Reset to a new episode. Randomizes laser position and initial pose."""
        mujoco.mj_resetData(self.model, self.data)

        if laser_xy is None:
            lxy = self.rng.uniform(-0.5, 0.5, size=2)
        else:
            lxy = np.asarray(laser_xy, dtype=float)
        site_id = self.model.site("laser_source").id
        self.model.site_pos[site_id, 0] = float(lxy[0])
        self.model.site_pos[site_id, 1] = float(lxy[1])
        self.model.site_pos[site_id, 2] = 2.5

        self.data.qpos[:] = self.rng.normal(0.0, joint_init_noise, size=2)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.data.qpos[:]

        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray) -> Observation:
        """Advance one agent step. Action is (theta_x_target, theta_y_target)."""
        action = np.clip(np.asarray(action, dtype=float), -optics.JOINT_LIMIT, optics.JOINT_LIMIT)
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        return self._observation()

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _observation(self) -> Observation:
        intensities = self._compute_intensities()
        return Observation(
            detector_intensities=intensities,
            joint_angles=self.data.qpos[:].copy(),
            joint_velocities=self.data.qvel[:].copy(),
            joint_torques=self.data.qfrc_actuator[:].copy(),
        )

    def _compute_intensities(self) -> np.ndarray:
        laser_pos = self.data.site("laser_source").xpos.copy()
        mirror_pos = self.data.site("mirror").xpos.copy()
        normal_tip = self.data.site("mirror_normal_tip").xpos.copy()
        diff = normal_tip - mirror_pos
        normal = diff / max(np.linalg.norm(diff), 1e-12)
        return optics.compute_detector_intensities(
            laser_pos=laser_pos,
            mirror_pos=mirror_pos,
            mirror_normal=normal,
            detector_positions=self._detector_positions,
            sigma=self.sigma,
        )

    # ------------------------------------------------------------------
    # Oracle (for DOA-direct baseline only)
    # ------------------------------------------------------------------

    def get_oracle(self) -> Oracle:
        """Ground-truth info for evaluation and the target-aware baseline."""
        laser_pos = self.data.site("laser_source").xpos.copy()
        target_pos = self._detector_positions[TARGET_DETECTOR_INDEX].copy()
        return Oracle(
            laser_pos=laser_pos,
            target_detector_pos=target_pos,
            target_detector_index=TARGET_DETECTOR_INDEX,
        )

    def target_intensity(self) -> float:
        """Intensity at the target detector. Used by the runner for metrics."""
        return float(self._compute_intensities()[TARGET_DETECTOR_INDEX])

    # ------------------------------------------------------------------
    # Sanity check at startup
    # ------------------------------------------------------------------

    def _sanity_check_kinematics(self) -> None:
        """Verify analytic FK in optics.py matches the MuJoCo model."""
        original_qpos = self.data.qpos.copy()
        original_qvel = self.data.qvel.copy()

        test_angles = [
            (0.0, 0.0),
            (0.3, 0.0),
            (0.0, 0.5),
            (0.4, -0.6),
            (-0.5, 0.7),
            (1.0, -0.8),
        ]
        try:
            for tx, ty in test_angles:
                self.data.qpos[0] = tx
                self.data.qpos[1] = ty
                self.data.qvel[:] = 0.0
                mujoco.mj_forward(self.model, self.data)
                mirror_pos = self.data.site("mirror").xpos.copy()
                tip_pos = self.data.site("mirror_normal_tip").xpos.copy()
                mj_normal = (tip_pos - mirror_pos) / max(
                    np.linalg.norm(tip_pos - mirror_pos), 1e-12
                )
                analytic_normal = optics.joint_angles_to_mirror_normal(tx, ty)
                if not np.allclose(mj_normal, analytic_normal, atol=1e-6):
                    raise AssertionError(
                        f"FK mismatch at (tx={tx:.3f}, ty={ty:.3f}): "
                        f"MuJoCo={mj_normal}, analytic={analytic_normal}"
                    )
        finally:
            self.data.qpos[:] = original_qpos
            self.data.qvel[:] = original_qvel
            mujoco.mj_forward(self.model, self.data)
