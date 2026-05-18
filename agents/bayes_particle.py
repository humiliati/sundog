"""Particle-Bayes baseline for the photometric mirror task.

The hidden state is the laser xy position. The belief model is intentionally
nominal-only: it predicts detector intensities with DEFAULT_SIGMA and no
detector-noise parameter. Stress in the environment therefore enters as model
misspecification, not as an adapted likelihood.
"""

from __future__ import annotations

import math

import numpy as np

from sundog import optics
from sundog.env_v2 import Observation, TARGET_DETECTOR_INDEX


BAYES_PARTICLE_STRUCTURE_FILTERS = ("s0", "s1", "s2", "s3")

_ACTION_CACHE: dict[tuple[int, int, int, float, str], tuple[np.ndarray, np.ndarray]] = {}


class BayesParticleAgent:
    """Static particle posterior over laser xy from detector intensities."""

    def __init__(
        self,
        particle_count: int,
        seed: int,
        detector_positions: np.ndarray,
        target_detector_pos: np.ndarray,
        target_detector_index: int = TARGET_DETECTOR_INDEX,
        joint_limit: float = optics.JOINT_LIMIT,
        assumed_sigma: float = optics.DEFAULT_SIGMA,
        laser_z: float = 2.5,
        structure_filter: str = "s0",
    ):
        self.particle_count = int(particle_count)
        if self.particle_count <= 0:
            raise ValueError("particle_count must be positive")
        self.seed = int(seed)
        self.detector_positions = np.asarray(detector_positions, dtype=float)
        self.target_detector_pos = np.asarray(target_detector_pos, dtype=float)
        self.target_detector_index = int(target_detector_index)
        self.joint_limit = float(joint_limit)
        self.assumed_sigma = float(assumed_sigma)
        self.laser_z = float(laser_z)
        if structure_filter not in BAYES_PARTICLE_STRUCTURE_FILTERS:
            raise ValueError(f"unknown Bayes particle structure filter: {structure_filter}")
        self.structure_filter = structure_filter

        cache_key = (
            self.particle_count,
            self.seed,
            self.target_detector_index,
            round(self.joint_limit, 12),
            self.structure_filter,
        )
        cached = _ACTION_CACHE.get(cache_key)
        if cached is None:
            particles = self._make_particles()
            actions = self._precompute_particle_actions(particles)
            _ACTION_CACHE[cache_key] = (particles, actions)
        else:
            particles, actions = cached
        self.particles = particles.copy()
        self.candidate_actions = actions.copy()
        self._target_matrix = self._precompute_target_matrix()
        self.log_weights = np.full(
            self.particle_count,
            -math.log(self.particle_count),
            dtype=np.float64,
        )

    def _make_particles(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.uniform(-0.4, 0.4, size=(self.particle_count, 2))

    def _precompute_particle_actions(self, particles: np.ndarray) -> np.ndarray:
        actions = np.empty((particles.shape[0], 2), dtype=np.float64)
        for i, xy in enumerate(particles):
            laser_pos = np.array([xy[0], xy[1], self.laser_z], dtype=np.float64)
            tx, ty = optics.optimal_joint_angles(
                laser_pos=laser_pos,
                target_pos=self.target_detector_pos,
            )
            actions[i] = np.array(
                [
                    np.clip(tx, -self.joint_limit, self.joint_limit),
                    np.clip(ty, -self.joint_limit, self.joint_limit),
                ]
            )
        return actions

    def _precompute_target_matrix(self) -> np.ndarray:
        matrix = np.empty((self.particle_count, self.particle_count), dtype=np.float64)
        for truth_idx, xy in enumerate(self.particles):
            laser_pos = np.array([xy[0], xy[1], self.laser_z], dtype=np.float64)
            for action_idx, action in enumerate(self.candidate_actions):
                matrix[truth_idx, action_idx] = self._predict_for_action(
                    laser_pos,
                    action,
                )[self.target_detector_index]
        return matrix

    def _predict_reflected(self, laser_pos: np.ndarray, action: np.ndarray) -> np.ndarray:
        normal = optics.joint_angles_to_mirror_normal(float(action[0]), float(action[1]))
        mirror_pos = optics.mirror_position_from_normal(normal)
        return optics.compute_detector_intensities(
            laser_pos=laser_pos,
            mirror_pos=mirror_pos,
            mirror_normal=normal,
            detector_positions=self.detector_positions,
            sigma=self.assumed_sigma,
        )

    def _predict_no_reflection(self, laser_pos: np.ndarray, action: np.ndarray) -> np.ndarray:
        normal = optics.joint_angles_to_mirror_normal(float(action[0]), float(action[1]))
        mirror_pos = optics.mirror_position_from_normal(normal)
        incident = mirror_pos - laser_pos
        incident_norm = np.linalg.norm(incident)
        if incident_norm < 1e-9:
            return np.zeros(self.detector_positions.shape[0])
        hit = optics.floor_hit(mirror_pos, incident / incident_norm)
        if hit is None:
            return np.zeros(self.detector_positions.shape[0])
        intensities = np.empty(self.detector_positions.shape[0])
        for i in range(self.detector_positions.shape[0]):
            intensities[i] = optics.gaussian_intensity(
                hit,
                self.detector_positions[i],
                self.assumed_sigma,
            )
        return intensities

    def _predict_for_action(self, laser_pos: np.ndarray, action: np.ndarray) -> np.ndarray:
        if self.structure_filter == "s0":
            return self._predict_reflected(laser_pos, action)
        if self.structure_filter == "s1":
            return self._predict_no_reflection(laser_pos, action)
        if self.structure_filter == "s2":
            intensities = np.zeros(self.detector_positions.shape[0])
            intensities[self.target_detector_index] = self._predict_reflected(
                laser_pos,
                action,
            )[self.target_detector_index]
            return intensities
        return np.zeros(self.detector_positions.shape[0])

    def _predict_current(self, xy: np.ndarray, joint_angles: np.ndarray) -> np.ndarray:
        laser_pos = np.array([xy[0], xy[1], self.laser_z], dtype=np.float64)
        return self._predict_for_action(laser_pos, joint_angles)

    def _posterior_probs(self) -> np.ndarray:
        centered = self.log_weights - np.max(self.log_weights)
        probs = np.exp(centered)
        total = float(np.sum(probs))
        if total <= 0.0 or not np.isfinite(total):
            return np.full(self.particle_count, 1.0 / self.particle_count)
        return probs / total

    def _update(self, obs: Observation) -> None:
        observed = np.asarray(obs.detector_intensities, dtype=np.float64)
        for i, xy in enumerate(self.particles):
            predicted = self._predict_current(xy, obs.joint_angles)
            residual = observed - predicted
            self.log_weights[i] -= float(np.dot(residual, residual))
        self.log_weights -= np.max(self.log_weights)

    def act(self, obs: Observation) -> np.ndarray:
        self._update(obs)
        probs = self._posterior_probs()
        expected_target = probs @ self._target_matrix
        action_idx = int(np.argmax(expected_target))
        return self.candidate_actions[action_idx].copy()
