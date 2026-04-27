"""Optics module for the Sundog v2 simulation.

Provides analytic beam-reflection geometry and Gaussian-spot detector intensity
computation. These are the load-bearing physics of the rebuilt simulation: the
agent's only photometric signal flows through these functions.

All vectors are 3D numpy arrays in world frame. Distances in meters. The mirror
is treated as a point reflector with a given orientation (a unit normal vector).

Forward kinematics (matching the MuJoCo model)
----------------------------------------------
The pole has two perpendicular hinges at its base: rx (axis world-x) and ry
(axis world-y). MuJoCo's joint composition for multiple joints on a body
applies the LAST-listed joint as the inner (closest-to-body) rotation and the
FIRST-listed joint as the outer rotation. In our XML rx is listed first and
ry is listed second, so the world-frame rotation is R_rx * R_ry. Applied to
the local +z axis (the pole's tip direction = the mirror normal):

    n(theta_x, theta_y) = (
        sin(theta_y),
        -sin(theta_x) * cos(theta_y),
        cos(theta_x) * cos(theta_y),
    )

This was discovered (and validated) by the env_v2 startup sanity check, which
diffs MuJoCo's site geometry against this analytic expression.
"""

from __future__ import annotations

import numpy as np


POLE_BASE_WORLD = np.array([0.0, 0.0, 0.05])
POLE_LENGTH = 1.2
JOINT_LIMIT = 1.5
DEFAULT_SIGMA = 0.15


def reflect(d_in: np.ndarray, n: np.ndarray) -> np.ndarray:
    """r = d - 2(d.n)n. n must be unit-length; caller's responsibility."""
    return d_in - 2.0 * float(np.dot(d_in, n)) * n


def floor_hit(start: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
    """Intersect ray with z=0. Returns None if direction is upward."""
    if direction[2] >= -1e-9:
        return None
    t = -start[2] / direction[2]
    if t <= 0:
        return None
    return start + t * direction


def gaussian_intensity(
    hit_point: np.ndarray,
    detector_pos: np.ndarray,
    sigma: float = DEFAULT_SIGMA,
) -> float:
    diff = hit_point - detector_pos
    r2 = float(np.dot(diff, diff))
    return float(np.exp(-r2 / (2.0 * sigma * sigma)))


def compute_detector_intensities(
    laser_pos: np.ndarray,
    mirror_pos: np.ndarray,
    mirror_normal: np.ndarray,
    detector_positions: np.ndarray,
    sigma: float = DEFAULT_SIGMA,
) -> np.ndarray:
    """Full pipeline: laser -> mirror -> floor -> per-detector intensities."""
    incident = mirror_pos - laser_pos
    incident_norm = np.linalg.norm(incident)
    if incident_norm < 1e-9:
        return np.zeros(detector_positions.shape[0])
    d_in = incident / incident_norm

    d_out = reflect(d_in, mirror_normal)
    hit = floor_hit(mirror_pos, d_out)
    if hit is None:
        return np.zeros(detector_positions.shape[0])

    n_detectors = detector_positions.shape[0]
    intensities = np.empty(n_detectors)
    for i in range(n_detectors):
        intensities[i] = gaussian_intensity(hit, detector_positions[i], sigma)
    return intensities


def joint_angles_to_mirror_normal(theta_x: float, theta_y: float) -> np.ndarray:
    """Forward kinematics: matches the MuJoCo joint composition R_rx * R_ry."""
    cx, sx = np.cos(theta_x), np.sin(theta_x)
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    return np.array([sy, -sx * cy, cx * cy])


def mirror_normal_to_joint_angles(n: np.ndarray) -> tuple[float, float]:
    """Inverse kinematics. Returns (theta_x, theta_y) in the principal branch."""
    n = np.asarray(n, dtype=float)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        raise ValueError("zero-magnitude normal")
    n = n / norm

    nx = float(n[0])
    if abs(nx) > 1.0 - 1e-12:
        theta_y = np.pi / 2 if nx > 0 else -np.pi / 2
        theta_x = 0.0
        return theta_x, theta_y

    theta_y = float(np.arcsin(nx))
    cy = float(np.cos(theta_y))
    theta_x = float(np.arctan2(-n[1] / cy, n[2] / cy))
    return theta_x, theta_y


def mirror_position_from_normal(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    return POLE_BASE_WORLD + POLE_LENGTH * n


def optimal_joint_angles(
    laser_pos: np.ndarray,
    target_pos: np.ndarray,
    iterations: int = 6,
) -> tuple[float, float]:
    """Find joint angles that send the laser-mirror reflected beam onto target.

    Coupled because mirror position depends on its orientation. We use a coarse
    grid search across the joint space to seed Nelder-Mead local refinement.
    The grid is 9x9 across +/- JOINT_LIMIT and the refinement converges to
    fractional-radian precision in <50 evaluations.

    The original half-vector fixed-point iteration was abandoned because it
    failed to converge (oscillated between far-apart mirror positions) for
    geometries where the mirror traces a substantial arc on the pole sphere.
    Numerical optimization is robust to that case.

    Returns (theta_x, theta_y) clipped to JOINT_LIMIT.
    """
    from scipy.optimize import minimize

    laser_pos = np.asarray(laser_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)

    def neg_intensity(angles: np.ndarray) -> float:
        tx, ty = float(angles[0]), float(angles[1])
        n = joint_angles_to_mirror_normal(tx, ty)
        m = POLE_BASE_WORLD + POLE_LENGTH * n
        d_in_v = m - laser_pos
        d_in_norm = float(np.linalg.norm(d_in_v))
        if d_in_norm < 1e-9:
            return 0.0
        d_in = d_in_v / d_in_norm
        r = reflect(d_in, n)
        h = floor_hit(m, r)
        if h is None:
            return 0.0
        return -gaussian_intensity(h, target_pos, DEFAULT_SIGMA)

    # Coarse grid search.
    grid_n = 11
    best_angles = np.zeros(2)
    best_val = 0.0
    for tx in np.linspace(-JOINT_LIMIT, JOINT_LIMIT, grid_n):
        for ty in np.linspace(-JOINT_LIMIT, JOINT_LIMIT, grid_n):
            v = neg_intensity(np.array([tx, ty]))
            if v < best_val:
                best_val = v
                best_angles = np.array([tx, ty])

    # Local refinement.
    if best_val < 0.0:
        res = minimize(
            neg_intensity,
            best_angles,
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 200},
        )
        tx_opt, ty_opt = float(res.x[0]), float(res.x[1])
    else:
        # No grid point reached the floor at all - return zeros (no solution).
        tx_opt, ty_opt = 0.0, 0.0

    tx_opt = float(np.clip(tx_opt, -JOINT_LIMIT, JOINT_LIMIT))
    ty_opt = float(np.clip(ty_opt, -JOINT_LIMIT, JOINT_LIMIT))
    return tx_opt, ty_opt


