"""Executable smoke workbench for docs/isotrophy/sundog_v_isotrophy.md.

This is deliberately a small harness, not the full K_facet experiment. It
parses the Li-Liao compact supplementary ansatz, integrates selected rows with
DOP853, and evaluates concrete spacetime-isotropy generators through the same
residual gate specified in docs/isotrophy/files.math:

    generator action -> explicit spatial matrix -> SO(3) Procrustes +
    phase minimization.

The smoke target is parser/integrator/gate sanity, especially:
    - sigma3 and residual generators use the same detector path;
    - F_beta is structural for the ansatz;
    - F_delta is tested as emergent, not quotiented away.

It also owns the K1 prediction-freeze receipt: the 21 strict G.2
choreographies are classified against six concrete generators before the
supplementary-B empirical count is run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


SUPPLEMENT_URLS = {
    "A": "https://numericaltank.sjtu.edu.cn/three-body/three-body/gif/supplementary-A.txt",
    "B": "https://numericaltank.sjtu.edu.cn/three-body/three-body/gif/supplementary-B.txt",
}
DEFAULT_RTOL = 1e-12
DEFAULT_ATOL = 1e-12
DEFAULT_MAX_STEP_FRACTION = 0.02
DEFAULT_SIGMA_TOLERANCE = 1e-5
DEFAULT_SIGMA_CLOSURE_MULTIPLE = 3.0
DEFAULT_IDENTITY_ROTATION_TOLERANCE = 1e-6
DEFAULT_EXPECTED_SIGMA_COUNT = None
DEFAULT_CLOSURE_FLOOR = 1e-15
DEFAULT_KFACET_GENERATORS = ("alpha_I", "beta_I", "gamma_Z", "delta_Z", "F_beta", "F_delta")
DEFAULT_TAU12_GENERATORS = ("tau12_I", "tau12_Z")
DEFAULT_KFACET_STRICT_INDICES = (
    62,
    64,
    231,
    264,
    468,
    524,
    574,
    609,
    617,
    623,
    735,
    793,
    941,
    1034,
    1062,
    1114,
    1172,
    1265,
    1414,
    1488,
    1497,
)

# Pre-registered constants for the v0.3h adaptive-floor reprocessor. The
# `kfacet-reprocess-floor` subcommand reads existing sentinel receipts and
# selects, per row, the smallest floor in the ladder that simultaneously
# (a) drives every D3 operator's kernel leakage to <= 1e-3 and
# (b) preserves an order-scale spectral gap at the kernel boundary.
# Selection is deterministic; no per-row knobs. The upper rungs are
# guardrails: a row that only stabilizes at >= 3e-6 is flagged suspicious
# in the receipt even when it technically passes.
ADAPTIVE_FLOOR_LADDER = (1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5)
ADAPTIVE_FLOOR_PROJECTOR_FLOOR = 1e-3
ADAPTIVE_FLOOR_GAP_RATIO_THRESHOLD = 1e-3
ADAPTIVE_FLOOR_FIRST_REJECTED_THRESHOLD = 1e-3
ADAPTIVE_FLOOR_SUSPICIOUS_THRESHOLD = 3e-6
ADAPTIVE_FLOOR_VERSION = "v0.3h-adaptive-floor"

# Pre-registered constants for the v0.3h bridge audit. The `kfacet-bridge-audit`
# subcommand probes rows that the adaptive-floor reprocessor flagged as failed
# due to a bridge singular value of (M_i - I) sitting in the forbidden band
# (1e-7, 1e-3): above the noise/kernel band but below the projector-floor guard.
# The audit is no-integration; it reuses existing M_i.npy + D3_*.npy and asks
# whether the bridge SV is (a) a missed neutral/Hamiltonian direction,
# (b) a Jordan-block root at eigenvalue 1, (c) a defective D3 standard block,
# or (d) genuinely unexplained and worth escalating to a tighter-rtol rerun.
# All thresholds are pre-registered; outcomes are deterministic; nothing tunable.
BRIDGE_AUDIT_LADDER = (1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
BRIDGE_BAND_LOWER = 1e-7
BRIDGE_BAND_UPPER = 1e-3
BRIDGE_FIXED_FLOOR = 1e-3
BRIDGE_NEUTRAL_OVERLAP_THRESHOLD = 1.0 - 1e-3
BRIDGE_JORDAN_CHAIN_THRESHOLD = 1e-3
BRIDGE_EIGENVALUE_NEAR_ONE_BAND = 1e-2
BRIDGE_PROJECTOR_FLOOR = 1e-3
BRIDGE_AUDIT_VERSION = "v0.3h-bridge-audit"

FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
ROW_RE = re.compile(
    rf"O_\{{(?P<index>\d+)\}}\((?P<m3>{FLOAT})\)\s+"
    rf"(?P<z0>{FLOAT})\s+(?P<vx>{FLOAT})\s+(?P<vy>{FLOAT})\s+"
    rf"(?P<vz>{FLOAT})\s+(?P<period>{FLOAT})\s+(?P<stability>[SU])"
)


I3 = np.diag([1.0, 1.0, 1.0])
Z_MIRROR = np.diag([1.0, 1.0, -1.0])
R_PI = np.diag([-1.0, -1.0, 1.0])
POINT_INVERSION = R_PI @ Z_MIRROR

PERMUTATIONS = {
    "identity": (0, 1, 2),
    "swap12": (1, 0, 2),
    # Group-action convention: (P x)_i = x_{p^{-1}(i)}.
    "cycle123": (2, 0, 1),
    "cycle132": (1, 2, 0),
}


@dataclass(frozen=True)
class Generator:
    name: str
    klass: str
    perm: tuple[int, int, int]
    shift_fraction: float
    treverse: bool
    spatial: np.ndarray


GENERATORS = {
    "sigma3": Generator("sigma3", "choreography", PERMUTATIONS["cycle123"], 1.0 / 3.0, False, I3),
    # True inverse of (cycle123, +T/3): inverse permutation and -T/3 == +2T/3.
    "sigma3_inverse": Generator(
        "sigma3_inverse", "choreography", PERMUTATIONS["cycle132"], 2.0 / 3.0, False, I3
    ),
    "sigma3_opposite_orientation": Generator(
        "sigma3_opposite_orientation", "choreography", PERMUTATIONS["cycle132"], 1.0 / 3.0, False, I3
    ),
    # Inverse of the opposite-orientation element (cycle132, +T/3):
    # its square is (cycle123, +2T/3); the two generate the full opposite Z3.
    "sigma3_opposite_inverse": Generator(
        "sigma3_opposite_inverse", "choreography", PERMUTATIONS["cycle123"], 2.0 / 3.0, False, I3
    ),
    "tau12_I": Generator("tau12_I", "case_split", PERMUTATIONS["swap12"], 0.0, False, I3),
    "tau12_Z": Generator("tau12_Z", "case_split", PERMUTATIONS["swap12"], 0.0, False, Z_MIRROR),
    # Back-compat alias for the first proper-parity case-split receipt.
    "tau12_gauge": Generator("tau12_gauge", "case_split", PERMUTATIONS["swap12"], 0.0, False, I3),
    "alpha_I": Generator("alpha_I", "alpha", PERMUTATIONS["swap12"], 1.0 / 2.0, False, I3),
    "beta_I": Generator("beta_I", "beta", PERMUTATIONS["swap12"], 0.0, True, I3),
    "gamma_Z": Generator("gamma_Z", "gamma", PERMUTATIONS["swap12"], 1.0 / 2.0, False, Z_MIRROR),
    "delta_Z": Generator("delta_Z", "delta", PERMUTATIONS["swap12"], 0.0, True, Z_MIRROR),
    "F_beta": Generator("F_beta", "beta", PERMUTATIONS["swap12"], 0.0, True, R_PI),
    "F_delta": Generator("F_delta", "delta", PERMUTATIONS["swap12"], 0.0, True, POINT_INVERSION),
}


@dataclass(frozen=True)
class OrbitRow:
    source: str
    line_no: int
    index: int
    m3: float
    z0: float
    vx: float
    vy: float
    vz: float
    period: float
    stability: str

    @property
    def masses(self) -> np.ndarray:
        return np.array([1.0, 1.0, self.m3], dtype=float)

    @property
    def label(self) -> str:
        return f"O_{{{self.index}}}({self.m3:g})"


@dataclass(frozen=True)
class IntegratedOrbit:
    row: OrbitRow
    solution: object
    y0: np.ndarray
    elapsed_seconds: float
    closure_position_inf: float
    closure_velocity_inf: float
    inertia_degenerate: bool


def read_text(path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")
    return Path(path_or_url).read_text(encoding="utf-8")


def parse_rows(text: str, source: str) -> list[OrbitRow]:
    rows: list[OrbitRow] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        match = ROW_RE.search(line)
        if not match:
            continue
        groups = match.groupdict()
        rows.append(
            OrbitRow(
                source=source,
                line_no=line_no,
                index=int(groups["index"]),
                m3=float(groups["m3"]),
                z0=float(groups["z0"]),
                vx=float(groups["vx"]),
                vy=float(groups["vy"]),
                vz=float(groups["vz"]),
                period=float(groups["period"]),
                stability=groups["stability"],
            )
        )
    return rows


def expand_initial_state(row: OrbitRow, center_com: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masses = row.masses
    x = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, row.z0],
        ],
        dtype=float,
    )
    v = np.array(
        [
            [row.vx, row.vy, row.vz],
            [row.vx, row.vy, -row.vz],
            [-2.0 * row.vx / row.m3, -2.0 * row.vy / row.m3, 0.0],
        ],
        dtype=float,
    )
    if center_com:
        x = x - np.average(x, axis=0, weights=masses)
    return masses, x, v


def pack_state(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.concatenate([x.reshape(9), v.reshape(9)])


def rhs_factory(masses: np.ndarray):
    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        x = y[:9].reshape(3, 3)
        v = y[9:].reshape(3, 3)
        a = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                delta = x[j] - x[i]
                r2 = float(np.dot(delta, delta))
                r = math.sqrt(r2)
                a[i] += masses[j] * delta / (r2 * r)
        return pack_state(v, a)

    return rhs


def sample_positions(integrated: IntegratedOrbit, times: np.ndarray) -> np.ndarray:
    period = integrated.row.period
    wrapped = np.mod(times, period)
    y = integrated.solution.sol(wrapped)
    return y[:9, :].T.reshape(len(wrapped), 3, 3)


def integrate_orbit(row: OrbitRow, rtol: float, atol: float, max_step_fraction: float) -> IntegratedOrbit:
    masses, x0, v0 = expand_initial_state(row, center_com=True)
    y0 = pack_state(x0, v0)
    max_step = row.period * max_step_fraction if max_step_fraction > 0 else np.inf
    started = time.perf_counter()
    solution = solve_ivp(
        rhs_factory(masses),
        (0.0, row.period),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=max_step,
    )
    elapsed = time.perf_counter() - started
    if not solution.success:
        raise RuntimeError(f"integration failed for {row.label}: {solution.message}")
    y_final = solution.y[:, -1]
    closure_position = np.linalg.norm((y_final[:9] - y0[:9]).reshape(3, 3), axis=1).max() / 2.0
    closure_velocity = np.linalg.norm((y_final[9:] - y0[9:]).reshape(3, 3), axis=1).max()
    return IntegratedOrbit(
        row=row,
        solution=solution,
        y0=y0,
        elapsed_seconds=elapsed,
        closure_position_inf=float(closure_position),
        closure_velocity_inf=float(closure_velocity),
        inertia_degenerate=abs(row.m3 * row.z0 * row.z0 - 2.0) < 1e-3,
    )


def acceleration_jacobian(x: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Return d(a_flat)/d(x_flat) for the Newtonian acceleration field."""
    jacobian = np.zeros((9, 9), dtype=float)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            delta = x[j] - x[i]
            r2 = float(np.dot(delta, delta))
            r = math.sqrt(r2)
            block = masses[j] * (np.eye(3) / (r2 * r) - 3.0 * np.outer(delta, delta) / (r2 * r2 * r))
            jacobian[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] -= block
            jacobian[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += block
    return jacobian


def partial_eps_acceleration_jacobian(x: np.ndarray) -> np.ndarray:
    """Return d/d epsilon of d(a_flat)/d(x_flat) at m3 = 1 + epsilon.

    Coordinates are position/velocity coordinates, not canonical momenta. With
    the IC held fixed, only bodies 1 and 2 feel the first-order m3 perturbation;
    body 3's acceleration row is identically zero.
    """
    jacobian = np.zeros((9, 9), dtype=float)

    def tidal_block(delta: np.ndarray) -> np.ndarray:
        r2 = float(np.dot(delta, delta))
        r = math.sqrt(r2)
        return np.eye(3) / (r2 * r) - 3.0 * np.outer(delta, delta) / (r2 * r2 * r)

    d13 = x[2] - x[0]
    k13 = tidal_block(d13)
    jacobian[0:3, 0:3] -= k13
    jacobian[0:3, 6:9] += k13

    d23 = x[2] - x[1]
    k23 = tidal_block(d23)
    jacobian[3:6, 3:6] -= k23
    jacobian[3:6, 6:9] += k23
    return jacobian


def partial_eps_acceleration(x: np.ndarray) -> np.ndarray:
    """Return d(a_flat)/d epsilon at m3 = 1 + epsilon in fixed q/v coordinates."""
    partial = np.zeros((3, 3), dtype=float)
    for body in (0, 1):
        delta = x[2] - x[body]
        r2 = float(np.dot(delta, delta))
        r = math.sqrt(r2)
        partial[body] = delta / (r2 * r)
    return partial.reshape(9)


def tidal_block_directional_derivative(delta: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Directional derivative of K(d)=I/|d|^3-3dd^T/|d|^5 along direction."""
    r2 = float(np.dot(delta, delta))
    r = math.sqrt(r2)
    dot = float(np.dot(delta, direction))
    return (
        -3.0 * dot * np.eye(3) / (r2 * r2 * r)
        - 3.0 * (np.outer(direction, delta) + np.outer(delta, direction)) / (r2 * r2 * r)
        + 15.0 * dot * np.outer(delta, delta) / (r2 * r2 * r2 * r)
    )


def acceleration_jacobian_position_directional(
    x: np.ndarray,
    direction: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """Return D_x[d(a)/d(x)] applied to a position perturbation direction."""
    jacobian = np.zeros((9, 9), dtype=float)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            delta = x[j] - x[i]
            direction_delta = direction[j] - direction[i]
            block = masses[j] * tidal_block_directional_derivative(delta, direction_delta)
            jacobian[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] -= block
            jacobian[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += block
    return jacobian


def variational_rhs_factory(masses: np.ndarray, orbit_solution):
    """Build the homogeneous variational RHS along an integrated orbit."""

    def variational_rhs(t: float, delta_y: np.ndarray) -> np.ndarray:
        y_t = orbit_solution.sol(t)
        x_t = y_t[:9].reshape(3, 3)
        delta_x = delta_y[:9]
        delta_v = delta_y[9:]
        return np.concatenate([delta_v, acceleration_jacobian(x_t, masses) @ delta_x])

    return variational_rhs


def compute_flow_jacobian(
    integrated: IntegratedOrbit,
    duration: float,
    rtol: float,
    atol: float,
    max_step_fraction: float | None = None,
) -> np.ndarray:
    """Compute Dphi_duration(y0) in the 18D position/velocity coordinates."""
    if duration < -1e-14 or duration > integrated.row.period + 1e-14:
        raise ValueError("compute_flow_jacobian expects duration within one nonnegative period")
    variational_rhs = variational_rhs_factory(integrated.row.masses, integrated.solution)
    max_step = (
        integrated.row.period * max_step_fraction
        if max_step_fraction is not None and max_step_fraction > 0
        else np.inf
    )
    flow = np.zeros((18, 18), dtype=float)
    for column in range(18):
        initial = np.zeros(18, dtype=float)
        initial[column] = 1.0
        solution = solve_ivp(
            variational_rhs,
            (0.0, duration),
            initial,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            dense_output=False,
            max_step=max_step,
        )
        if not solution.success:
            raise RuntimeError(
                f"variational integration failed for {integrated.row.label}, column {column}: {solution.message}"
            )
        flow[:, column] = solution.y[:, -1]
    return flow


def compute_monodromy(
    integrated: IntegratedOrbit,
    rtol: float,
    atol: float,
    max_step_fraction: float | None = None,
) -> np.ndarray:
    """Compute M_i = Dphi_T(y0) for the v0.3i sentinel gates."""
    return compute_flow_jacobian(integrated, integrated.row.period, rtol, atol, max_step_fraction)


def compute_partial_eps_monodromy(
    integrated: IntegratedOrbit,
    rtol: float,
    atol: float,
    max_step_fraction: float | None = None,
    return_joint_baseline: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute dM_i/d epsilon for m3 = 1 + epsilon at the fixed m3=1 IC."""
    if abs(integrated.row.m3 - 1.0) > 1e-12:
        raise ValueError("partial-epsilon monodromy is defined only for the m3=1 sentinel row")
    max_step = (
        integrated.row.period * max_step_fraction
        if max_step_fraction is not None and max_step_fraction > 0
        else np.inf
    )
    masses = integrated.row.masses
    period = integrated.row.period
    sensitivity_max_step = max_step

    def sensitivity_rhs(t: float, sensitivity: np.ndarray) -> np.ndarray:
        baseline = integrated.solution.sol(t)
        x_t = baseline[:9].reshape(3, 3)
        sensitivity_x = sensitivity[:9]
        sensitivity_v = sensitivity[9:]
        acceleration_source = acceleration_jacobian(x_t, masses) @ sensitivity_x + partial_eps_acceleration(x_t)
        return np.concatenate([sensitivity_v, acceleration_source])

    sensitivity_solution = solve_ivp(
        sensitivity_rhs,
        (0.0, period),
        np.zeros(18, dtype=float),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=sensitivity_max_step,
    )
    if not sensitivity_solution.success:
        raise RuntimeError(
            f"partial-epsilon orbit sensitivity failed for {integrated.row.label}: "
            f"{sensitivity_solution.message}"
        )

    def joint_rhs(t: float, state: np.ndarray) -> np.ndarray:
        baseline = integrated.solution.sol(t)
        x_t = baseline[:9].reshape(3, 3)
        jacobian = acceleration_jacobian(x_t, masses)
        sensitivity_t = sensitivity_solution.sol(t)
        sensitivity_x = sensitivity_t[:9].reshape(3, 3)
        total_partial_jacobian = partial_eps_acceleration_jacobian(
            x_t
        ) + acceleration_jacobian_position_directional(x_t, sensitivity_x, masses)

        delta = state[:18]
        partial = state[18:]
        delta_x = delta[:9]
        delta_v = delta[9:]
        partial_x = partial[:9]
        partial_v = partial[9:]

        delta_dot = np.concatenate([delta_v, jacobian @ delta_x])
        partial_dot = np.concatenate([partial_v, jacobian @ partial_x + total_partial_jacobian @ delta_x])
        return np.concatenate([delta_dot, partial_dot])

    joint_baseline_monodromy = np.zeros((18, 18), dtype=float)
    partial_monodromy = np.zeros((18, 18), dtype=float)
    for column in range(18):
        initial = np.zeros(36, dtype=float)
        initial[column] = 1.0
        solution = solve_ivp(
            joint_rhs,
            (0.0, period),
            initial,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            dense_output=False,
            max_step=max_step,
        )
        if not solution.success:
            raise RuntimeError(
                f"partial-epsilon variational integration failed for {integrated.row.label}, "
                f"column {column}: {solution.message}"
            )
        joint_baseline_monodromy[:, column] = solution.y[:18, -1]
        partial_monodromy[:, column] = solution.y[18:, -1]
    if return_joint_baseline:
        return partial_monodromy, joint_baseline_monodromy
    return partial_monodromy


def compute_monodromy_at_masses(
    y0: np.ndarray,
    period: float,
    masses: np.ndarray,
    rtol: float,
    atol: float,
    max_step_fraction: float,
) -> np.ndarray:
    """Compute a monodromy from a fixed IC while varying the RHS masses."""
    max_step = period * max_step_fraction if max_step_fraction > 0 else np.inf
    baseline = solve_ivp(
        rhs_factory(masses),
        (0.0, period),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=max_step,
    )
    if not baseline.success:
        raise RuntimeError(f"fixed-IC baseline integration failed for masses={masses.tolist()}: {baseline.message}")

    variational_rhs = variational_rhs_factory(masses, baseline)
    monodromy = np.zeros((18, 18), dtype=float)
    for column in range(18):
        initial = np.zeros(18, dtype=float)
        initial[column] = 1.0
        solution = solve_ivp(
            variational_rhs,
            (0.0, period),
            initial,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            dense_output=False,
            max_step=max_step,
        )
        if not solution.success:
            raise RuntimeError(
                f"fixed-IC variational integration failed for masses={masses.tolist()}, "
                f"column {column}: {solution.message}"
            )
        monodromy[:, column] = solution.y[:, -1]
    return monodromy


def verify_partial_eps_via_finite_difference(
    integrated: IntegratedOrbit,
    h: float,
    rtol: float,
    atol: float,
    max_step_fraction: float,
    k_fd_relative: float,
    reference_monodromy: np.ndarray | None = None,
    k_joint_baseline_relative: float | None = None,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Cross-check closed-form dM/depsilon against a fixed-IC central difference.

    Both sub-checks (finite-difference vs closed-form, and joint-solver baseline
    vs standalone M_i) are CLOSURE-RELATIVE, matching the discipline already in
    use for G.2 (`k_gamma`/`gamma_floor`), the sigma3 candidate gate
    (`sigma_group_to_closure`), and the rotation-angle gate. Absolute floors
    are wrong for the catalog because `||M_i||` ranges across orders of
    magnitude (O_62: 1.6e+4, O_231: 7.6e+2, etc.) and `||partial_eps M_i||`
    is amplified further by orbit-path sensitivity for unstable rows.

    FD check passes iff
        ||closed - FD||_inf / max(||closed||_inf, ||FD||_inf, 1e-300)
            < k_fd_relative

    Joint-baseline check passes iff
        ||joint_baseline - reference_M_i||_inf / max(||reference_M_i||_inf, 1e-300)
            < k_joint_baseline_relative

    `k_fd_relative` and `k_joint_baseline_relative` are PRE-REGISTERED relative
    tolerances (defensible upper bounds, not tuned to the sentinel). Defaults:
    `k_fd_relative = 1e-4` absorbs O(h^2) FD truncation at h=1e-6, rtol=1e-12
    on unstable orbits without admitting algorithmic divergence;
    `k_joint_baseline_relative = 1e-9` matches rtol-relative integrator
    precision between the 36-dim joint solver and the 18-dim standalone path.
    """
    if abs(integrated.row.m3 - 1.0) > 1e-12:
        raise ValueError("partial-epsilon finite-difference check is defined only for m3=1")
    partial_monodromy, joint_baseline_monodromy = compute_partial_eps_monodromy(
        integrated,
        rtol,
        atol,
        max_step_fraction,
        return_joint_baseline=True,
    )
    masses_plus = np.array([1.0, 1.0, 1.0 + h], dtype=float)
    masses_minus = np.array([1.0, 1.0, 1.0 - h], dtype=float)
    m_plus = compute_monodromy_at_masses(
        integrated.y0, integrated.row.period, masses_plus, rtol, atol, max_step_fraction
    )
    m_minus = compute_monodromy_at_masses(
        integrated.y0, integrated.row.period, masses_minus, rtol, atol, max_step_fraction
    )
    finite_difference = (m_plus - m_minus) / (2.0 * h)
    residual = partial_monodromy - finite_difference
    residual_inf = float(np.linalg.norm(residual, ord=np.inf))
    residual_fro = float(np.linalg.norm(residual, ord="fro"))
    closed_form_inf = float(np.linalg.norm(partial_monodromy, ord=np.inf))
    finite_difference_inf = float(np.linalg.norm(finite_difference, ord=np.inf))
    fd_scale = max(closed_form_inf, finite_difference_inf, 1e-300)
    fd_relative_residual = float(residual_inf / fd_scale)
    finite_difference_passed = fd_relative_residual < k_fd_relative

    joint_consistency: dict[str, object] = {
        "enabled": reference_monodromy is not None,
        "k_relative": k_joint_baseline_relative,
        "scale_used": "reference_monodromy_inf",
        "residual_inf": None,
        "residual_relative": None,
        "scale_inf": None,
        "passed": None,
    }
    if reference_monodromy is not None:
        k_jb = k_fd_relative if k_joint_baseline_relative is None else k_joint_baseline_relative
        joint_residual_inf = float(np.linalg.norm(joint_baseline_monodromy - reference_monodromy, ord=np.inf))
        reference_inf = float(np.linalg.norm(reference_monodromy, ord=np.inf))
        jb_scale = max(reference_inf, 1e-300)
        jb_relative = float(joint_residual_inf / jb_scale)
        joint_consistency = {
            "enabled": True,
            "k_relative": k_jb,
            "scale_used": "reference_monodromy_inf",
            "scale_inf": reference_inf,
            "residual_inf": joint_residual_inf,
            "residual_relative": jb_relative,
            "passed": jb_relative < k_jb,
        }
    result = {
        "gate": "partial_epsilon_finite_difference",
        "enabled": True,
        "h": h,
        "k_fd_relative": k_fd_relative,
        "fd_scale_used": "max(closed_form_inf, finite_difference_inf)",
        "residual_inf": residual_inf,
        "residual_fro": residual_fro,
        "residual_relative": fd_relative_residual,
        "closed_form_norm_inf": closed_form_inf,
        "finite_difference_norm_inf": finite_difference_inf,
        "fd_scale_inf": fd_scale,
        "finite_difference_passed": finite_difference_passed,
        "joint_baseline_consistency": joint_consistency,
        "passed": finite_difference_passed and joint_consistency.get("passed") is not False,
    }
    return result, partial_monodromy, finite_difference


def body_permutation_matrix_18(perm: tuple[int, int, int]) -> np.ndarray:
    """Lift a body relabel to the 18D position/velocity coordinates."""
    block9 = np.zeros((9, 9), dtype=float)
    for new_body, old_body in enumerate(perm):
        for axis in range(3):
            block9[3 * new_body + axis, 3 * old_body + axis] = 1.0
    operator = np.zeros((18, 18), dtype=float)
    operator[:9, :9] = block9
    operator[9:, 9:] = block9
    return operator


def spatial_matrix_18(spatial: np.ndarray) -> np.ndarray:
    """Lift a 3D spatial matrix to all bodies in position and velocity blocks."""
    block9 = np.zeros((9, 9), dtype=float)
    for body in range(3):
        block9[3 * body : 3 * body + 3, 3 * body : 3 * body + 3] = spatial
    operator = np.zeros((18, 18), dtype=float)
    operator[:9, :9] = block9
    operator[9:, 9:] = block9
    return operator


def time_reversal_matrix_18() -> np.ndarray:
    """Lift time reversal to the 18D tangent coordinates: positions fixed, velocities negated."""
    operator = np.eye(18)
    operator[9:, 9:] *= -1.0
    return operator


def f_beta_action_v0() -> np.ndarray:
    """Closed-form F_beta tangent action at the ansatz F_beta fixed anchor."""
    return body_permutation_matrix_18(PERMUTATIONS["swap12"]) @ time_reversal_matrix_18() @ spatial_matrix_18(R_PI)


def invert_permutation(perm: tuple[int, int, int]) -> tuple[int, int, int]:
    """Invert the new-body -> old-body permutation convention used by samples."""
    inverse = [0, 0, 0]
    for new_body, old_body in enumerate(perm):
        inverse[old_body] = new_body
    return tuple(inverse)


def construct_sigma_action_v0(
    integrated: IntegratedOrbit,
    perm: tuple[int, int, int],
    shift_fraction: float,
    rtol: float,
    atol: float,
    max_step_fraction: float,
    invert_permutation_for_tangent: bool,
) -> np.ndarray:
    """Construct a sigma-type D3 endomorphism on T_y0 by forward period transport.

    The time component is represented by the forward no-inverse segment
    Dphi_{T-shift}(y0), which is equivalent to back-flow on ker(M_i - I) but
    avoids inverting an ill-conditioned flow matrix. The catalog contains both
    cyclic body-label orientations, so the sentinel runner tests the sample
    permutation convention and its inverse as explicit orientation candidates.
    """
    period = integrated.row.period
    duration = (period - shift_fraction * period) % period
    if duration <= DEFAULT_CLOSURE_FLOOR:
        flow = np.eye(18)
    else:
        flow = compute_flow_jacobian(integrated, duration, rtol, atol, max_step_fraction)
    tangent_perm = invert_permutation(perm) if invert_permutation_for_tangent else perm
    return body_permutation_matrix_18(tangent_perm) @ flow


def construct_d3_elements_v0(
    integrated: IntegratedOrbit,
    rtol: float,
    atol: float,
    max_step_fraction: float = DEFAULT_MAX_STEP_FRACTION,
    invert_permutation_for_tangent: bool = False,
) -> dict[str, np.ndarray]:
    """Construct the six D3 tangent operators for the sentinel implementation gates."""
    identity = np.eye(18)
    sigma3 = construct_sigma_action_v0(
        integrated,
        PERMUTATIONS["cycle123"],
        1.0 / 3.0,
        rtol,
        atol,
        max_step_fraction,
        invert_permutation_for_tangent,
    )
    sigma3_sq = construct_sigma_action_v0(
        integrated,
        PERMUTATIONS["cycle132"],
        2.0 / 3.0,
        rtol,
        atol,
        max_step_fraction,
        invert_permutation_for_tangent,
    )
    f_beta = f_beta_action_v0()
    return {
        "e": identity,
        "sigma3": sigma3,
        "sigma3_sq": sigma3_sq,
        "F_beta": f_beta,
        "F_beta_sigma3": f_beta @ sigma3,
        "F_beta_sigma3_sq": f_beta @ sigma3_sq,
    }


def verify_d3_relations(
    d3_ops: dict[str, np.ndarray],
    M_i: np.ndarray,
    relation_floor: float,
    kernel_floor: float,
) -> dict[str, object]:
    """Verify the typed D3 relations on ker(M_i - I), where M_i acts as identity.

    Sigma is represented by forward period transport, so on full V_0 its cube
    is expected to track the full monodromy M_i, not the identity. Restricting
    tests to ker(M_i - I) gives the D3 relations the v0.3 functional needs and
    avoids inverse-flow amplification in degenerate Floquet eigenspaces.
    Full-space diagnostics record sigma3^3 vs M_i and [M_i, sigma3] so receipt
    readers can distinguish convention drift from kernel-level gate failure.
    """
    identity_18 = np.eye(18)
    sigma3 = d3_ops["sigma3"]
    sigma3_sq = d3_ops["sigma3_sq"]
    f_beta = d3_ops["F_beta"]

    _u, sv, vh = np.linalg.svd(M_i - identity_18)
    kernel_mask = sv < kernel_floor
    kernel_basis = vh.T[:, kernel_mask]
    kernel_dim = int(kernel_basis.shape[1])

    def kernel_residual(matrix: np.ndarray) -> float:
        if kernel_dim == 0:
            return 0.0
        restricted = kernel_basis.T @ matrix @ kernel_basis
        return float(np.linalg.norm(restricted, ord=np.inf))

    checks_kernel = {
        "sigma3_cubed_minus_I": kernel_residual(sigma3 @ sigma3 @ sigma3 - identity_18),
        "sigma3_sq_consistency": kernel_residual(sigma3 @ sigma3 - sigma3_sq),
        "F_beta_squared_minus_I": kernel_residual(f_beta @ f_beta - identity_18),
        "dihedral_relation": kernel_residual(f_beta @ sigma3 @ f_beta - sigma3_sq),
        "F_beta_sigma3_squared_minus_I": kernel_residual(
            (f_beta @ sigma3) @ (f_beta @ sigma3) - identity_18
        ),
    }
    checks_full_diagnostic = {
        "sigma3_cubed_minus_I_full_V0": float(
            np.linalg.norm(sigma3 @ sigma3 @ sigma3 - identity_18, ord=np.inf)
        ),
        "sigma3_cubed_minus_M_i_full_V0": float(
            np.linalg.norm(sigma3 @ sigma3 @ sigma3 - M_i, ord=np.inf)
        ),
        "sigma3_sq_consistency_full_V0": float(np.linalg.norm(sigma3 @ sigma3 - sigma3_sq, ord=np.inf)),
        "sigma3_M_i_commutator_full_V0": float(np.linalg.norm(M_i @ sigma3 - sigma3 @ M_i, ord=np.inf)),
        "F_beta_squared_minus_I_full_V0": float(
            np.linalg.norm(f_beta @ f_beta - identity_18, ord=np.inf)
        ),
        "M_i_minus_I_full_V0": float(np.linalg.norm(M_i - identity_18, ord=np.inf)),
    }
    return {
        "gate": "D3_relations",
        "floor": relation_floor,
        "kernel_floor": kernel_floor,
        "kernel_dim": kernel_dim,
        "checks": checks_kernel,
        "diagnostic_full_V0": checks_full_diagnostic,
        "kernel_singular_values": [float(value) for value in sv],
        "max_residual": max(checks_kernel.values()) if checks_kernel else 0.0,
        "passed": all(value < relation_floor for value in checks_kernel.values()),
    }


def summarize_d3_gate_candidate(gate: dict[str, object]) -> dict[str, object]:
    """Keep candidate-selection diagnostics JSON-friendly and compact."""
    return {
        "max_residual": gate["max_residual"],
        "passed": gate["passed"],
        "checks": gate["checks"],
        "diagnostic_full_V0": gate["diagnostic_full_V0"],
    }


def canonical_omega_18(masses: np.ndarray) -> np.ndarray:
    """Return the canonical symplectic form in (q, v) coordinates.

    In velocity coordinates the canonical momentum is p_i=m_i v_i, so the
    q/v symplectic matrix carries the mass matrix. For the equal-mass sentinel
    this reduces to the usual J.
    """
    mass9 = np.zeros((9, 9), dtype=float)
    for body, mass in enumerate(masses):
        mass9[3 * body : 3 * body + 3, 3 * body : 3 * body + 3] = np.eye(3) * mass
    return np.block([[np.zeros((9, 9)), mass9], [-mass9, np.zeros((9, 9))]])


def vector_subspace_basis(vectors: np.ndarray, floor: float) -> np.ndarray:
    """Return an orthonormal column basis for the span of input columns."""
    if vectors.size == 0 or vectors.shape[1] == 0:
        return np.zeros((vectors.shape[0], 0), dtype=float)
    u, singular_values, _vh = np.linalg.svd(vectors, full_matrices=False)
    rank = int(np.sum(singular_values > floor))
    return u[:, :rank]


def compute_neutral_basis(M_i: np.ndarray, integrated: IntegratedOrbit, floor: float) -> np.ndarray:
    """Build the translation/boost/flow/Jordan neutral basis for the 18D gate quotient."""
    neutral_columns: list[np.ndarray] = []
    rhs = rhs_factory(integrated.row.masses)
    x_h = rhs(0.0, integrated.y0)
    neutral_columns.append(x_h)

    for axis in range(3):
        translation = np.zeros(18, dtype=float)
        for body in range(3):
            translation[3 * body + axis] = 1.0
        neutral_columns.append(translation)

    for axis in range(3):
        boost = np.zeros(18, dtype=float)
        for body in range(3):
            boost[9 + 3 * body + axis] = 1.0
        neutral_columns.append(boost)

    identity = np.eye(18)
    u_energy, *_ = np.linalg.lstsq(M_i - identity, x_h, rcond=None)
    neutral_columns.append(u_energy)
    return vector_subspace_basis(np.column_stack(neutral_columns), floor)


def compute_kernel_basis(M_i: np.ndarray, floor: float) -> tuple[np.ndarray, list[float]]:
    """Return an orthonormal basis for ker(M_i-I) plus all singular values."""
    identity = np.eye(M_i.shape[0])
    _u, singular_values, vh = np.linalg.svd(M_i - identity)
    return vh.T[:, singular_values < floor], [float(value) for value in singular_values]


def compute_k_fib_basis(
    M_i: np.ndarray,
    integrated: IntegratedOrbit,
    closure_floor: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compute a gate-level basis for K_i^fib = ker(M_i-I) modulo neutral directions."""
    kernel_basis, singular_values = compute_kernel_basis(M_i, closure_floor)
    neutral_basis = compute_neutral_basis(M_i, integrated, closure_floor)
    if kernel_basis.shape[1] == 0:
        return np.zeros((18, 0), dtype=float), {
            "kernel_dim": 0,
            "neutral_dim": int(neutral_basis.shape[1]),
            "K_fib_dim": 0,
            "kernel_singular_values": [float(value) for value in singular_values],
        }

    neutral_projector = neutral_basis @ neutral_basis.T if neutral_basis.shape[1] else np.zeros((18, 18))
    quotient_raw = kernel_basis - neutral_projector @ kernel_basis
    k_fib_basis = vector_subspace_basis(quotient_raw, closure_floor)
    return k_fib_basis, {
        "kernel_dim": int(kernel_basis.shape[1]),
        "neutral_dim": int(neutral_basis.shape[1]),
        "K_fib_dim": int(k_fib_basis.shape[1]),
        "kernel_singular_values": [float(value) for value in singular_values],
    }


def verify_reduced_omega(
    M_i: np.ndarray,
    k_fib_basis: np.ndarray,
    omega: np.ndarray,
    nondegeneracy_floor: float,
) -> dict[str, object]:
    """Verify omega antisymmetry and M_i-symplecticity; record K_fib restriction as diagnostic only.

    Why the earlier "non-degeneracy of omega on K_i^fib" check is dropped from
    the pass condition: by v0.3's own decomposition,
        K_i^fib  ~=  (a-1)*T  +  (b-1)*S  +  c*E
    with T and S each 1-dim real irreps of D_3. Any 1-dim subspace of a
    symplectic vector space is automatically Lagrangian -- omega(x, x) = 0 by
    antisymmetry, so omega restricted to any T- or S-isotypic line is the zero
    bilinear form. When K_i^fib carries (a-1) >= 1 or (b-1) >= 1, the min
    singular value of omega|_K_fib is STRUCTURALLY zero, not a gate failure.
    Earlier drafts treating min(svd) > floor as a pass requirement therefore
    failed by construction on any orbit with a structural continuation
    direction or sign-isotypic mode in K_i^fib.

    The basis-independent gate-2 invariants survive:
    (1) omega is antisymmetric, (2) M_i is symplectic w.r.t. omega.

    The restriction singular values are kept as DIAGNOSTICS so a future
    reader can compare against the predicted 2*c_i rank (where c_i is the
    E-isotypic multiplicity from the D_3 decomposition). The full
    rank-against-2c_i gate is deferred until the D_3 decomposition is
    integrated, which itself depends on the kernel-projected D_3 relations
    from verify_d3_relations.
    """
    checks: dict[str, float] = {
        "omega_skew_residual": float(np.linalg.norm(omega + omega.T, ord=np.inf)),
        "omega_rank": float(np.linalg.matrix_rank(omega)),
        "M_i_symplectic_residual": float(np.linalg.norm(M_i.T @ omega @ M_i - omega, ord=np.inf)),
    }
    if k_fib_basis.shape[1] == 0:
        return {
            "gate": "reduced_omega",
            "floor": nondegeneracy_floor,
            "K_fib_dim": 0,
            "checks": checks,
            "passed": checks["omega_skew_residual"] < nondegeneracy_floor
            and checks["M_i_symplectic_residual"] < nondegeneracy_floor,
            "notes": (
                "K_i^fib empty; omega-restriction is vacuous. "
                "Gate enforces omega antisymmetry and M_i symplecticity only."
            ),
        }
    omega_k = k_fib_basis.T @ omega @ k_fib_basis
    restricted_singular_values = np.linalg.svd(omega_k, compute_uv=False)
    rank_above_floor = int(np.sum(restricted_singular_values > nondegeneracy_floor))
    checks.update(
        {
            "K_fib_omega_min_singular": float(restricted_singular_values.min()),
            "K_fib_omega_max_singular": float(restricted_singular_values.max()),
            "K_fib_omega_rank_above_floor": float(rank_above_floor),
        }
    )
    return {
        "gate": "reduced_omega",
        "floor": nondegeneracy_floor,
        "K_fib_dim": int(k_fib_basis.shape[1]),
        "checks": checks,
        "omega_restriction_singular_values": [float(value) for value in restricted_singular_values],
        "passed": checks["omega_skew_residual"] < nondegeneracy_floor
        and checks["M_i_symplectic_residual"] < nondegeneracy_floor,
        "notes": (
            "K_fib_omega_min_singular and K_fib_omega_rank_above_floor are "
            "DIAGNOSTIC only; T- and S-isotypic 1-dim subspaces of K_i^fib are "
            "structurally Lagrangian under omega. A stronger rank gate (= 2*c_i "
            "where c_i is the E-isotypic multiplicity) is deferred until D_3 "
            "isotypic decomposition is integrated into the runner."
        ),
    }


def nullspace_basis(matrix: np.ndarray, floor: float) -> tuple[np.ndarray, list[float]]:
    """Return a right-nullspace basis using an SVD floor."""
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[1], 0), dtype=float), []
    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.sum(singular_values > floor))
    return vh.T[:, rank:], [float(value) for value in singular_values]


def compute_gamma_rank_gate(
    kernel_basis: np.ndarray,
    d3_ops: dict[str, np.ndarray],
    omega: np.ndarray,
    partial_eps_monodromy: np.ndarray,
    rtol: float,
    k_gamma: float,
    k_int: float,
    projector_floor: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Compute the v0.3h Gamma_i rank gate on F_beta-even standard isotypic.

    The D3 decomposition is performed on ker(M_i-I), not on an arbitrary
    orthogonal complement of the neutral quotient. The v0.3 derivation places
    the neutral block in T/S sectors, so the standard-E multiplicity c_i is
    preserved by the quotient; doing the representation theory on the actual
    kernel avoids a non-invariant complement choice.
    """
    k_dim = int(kernel_basis.shape[1])
    if k_dim == 0:
        result = {
            "gate": "Gamma_i_rank",
            "enabled": True,
            "passed": True,
            "kernel_dim": 0,
            "D3_isotypic_dims": {"T": 0, "S": 0, "E": 0, "c_i": 0},
            "d_i": 0,
            "gamma_matrix": [],
            "gamma_singular_values": [],
            "gamma_floor": 0.0,
            "gamma_singular_to_floor": [],
            "gamma_rank_floor": 0,
            "gamma_singular_bimodality_clean": True,
            "notes": "K_i^fib is empty; Gamma_i is the empty matrix and d_i=0.",
        }
        return result, np.zeros((0, 0), dtype=float), np.zeros((18, 0), dtype=float)

    identity_k = np.eye(k_dim)
    ambient_projector = kernel_basis @ kernel_basis.T
    ambient_identity = np.eye(kernel_basis.shape[0])
    leakage = {
        name: float(np.linalg.norm((ambient_identity - ambient_projector) @ operator @ kernel_basis, ord=np.inf))
        for name, operator in d3_ops.items()
    }
    max_leakage = max(leakage.values()) if leakage else 0.0
    kernel_stability = {
        "floor": projector_floor,
        "max_leakage_inf": max_leakage,
        "F_beta_leakage_inf": leakage.get("F_beta", 0.0),
        "F_beta_passed": leakage.get("F_beta", 0.0) <= projector_floor,
        "all_D3_passed": all(value <= projector_floor for value in leakage.values()),
    }
    restricted = {name: kernel_basis.T @ operator @ kernel_basis for name, operator in d3_ops.items()}
    sigma3 = restricted["sigma3"]
    sigma3_sq = restricted["sigma3_sq"]
    f_beta = restricted["F_beta"]
    f_beta_sigma3 = restricted["F_beta_sigma3"]
    f_beta_sigma3_sq = restricted["F_beta_sigma3_sq"]

    projector_t = (identity_k + sigma3 + sigma3_sq + f_beta + f_beta_sigma3 + f_beta_sigma3_sq) / 6.0
    projector_s = (identity_k + sigma3 + sigma3_sq - f_beta - f_beta_sigma3 - f_beta_sigma3_sq) / 6.0
    projector_e = identity_k - projector_t - projector_s

    t_basis_k = vector_subspace_basis(projector_t, projector_floor)
    s_basis_k = vector_subspace_basis(projector_s, projector_floor)
    e_basis_k = vector_subspace_basis(projector_e, projector_floor)
    e_dim = int(e_basis_k.shape[1])
    c_i = e_dim // 2
    e_dim_even = e_dim % 2 == 0

    if e_dim == 0:
        gamma_matrix = np.zeros((0, 0), dtype=float)
        xi_basis = np.zeros((18, 0), dtype=float)
        singular_values: list[float] = []
        gamma_floor = 0.0
        gamma_rank = 0
        bimodality_clean = True
        f_even_dim = 0
        f_even_singular_values: list[float] = []
    else:
        f_beta_e = e_basis_k.T @ f_beta @ e_basis_k
        f_even_coords, f_even_singular_values = nullspace_basis(f_beta_e - np.eye(e_dim), projector_floor)
        f_even_dim = int(f_even_coords.shape[1])
        xi_basis = kernel_basis @ e_basis_k @ f_even_coords
        xi_basis = vector_subspace_basis(xi_basis, projector_floor)
        gamma_matrix = xi_basis.T @ omega @ partial_eps_monodromy @ xi_basis
        singular_values = [float(value) for value in np.linalg.svd(gamma_matrix, compute_uv=False)]
        partial_norm = float(np.linalg.norm(partial_eps_monodromy, ord=np.inf))
        gamma_floor = float(k_int * max(partial_norm, 1.0) * rtol)
        threshold = k_gamma * gamma_floor
        gamma_rank = int(sum(value > threshold for value in singular_values))
        bimodality_clean = all(not (gamma_floor <= value <= threshold) for value in singular_values)

    gamma_singular_to_floor = [
        float(value / max(gamma_floor, 1e-300)) for value in singular_values
    ]
    expected_f_even_dim = c_i if e_dim_even else None
    passed = bool(e_dim_even and f_even_dim == c_i and bimodality_clean and kernel_stability["all_D3_passed"])
    result = {
        "gate": "Gamma_i_rank",
        "enabled": True,
        "gate_version": "v0.3h-rank-matrix",
        "passed": passed,
        "kernel_dim": k_dim,
        "neutral_quotient_note": "Gamma_i decomposes ker(M_i-I); neutral T/S removal preserves the standard-E count.",
        "D3_isotypic_dims": {
            "T": int(t_basis_k.shape[1]),
            "S": int(s_basis_k.shape[1]),
            "E": e_dim,
            "c_i": c_i,
        },
        "E_dim_even": e_dim_even,
        "F_beta_even_E_dim": f_even_dim,
        "F_beta_even_E_dim_expected": expected_f_even_dim,
        "F_beta_even_E_null_singular_values": f_even_singular_values,
        "D3_kernel_stability": kernel_stability,
        "D3_kernel_leakage_inf": leakage,
        "D3_K_fib_leakage_inf": leakage,
        "projector_floor": projector_floor,
        "k_gamma": k_gamma,
        "k_int": k_int,
        "gamma_floor": gamma_floor,
        "gamma_matrix": gamma_matrix.tolist(),
        "gamma_diagonal": [float(gamma_matrix[i, i]) for i in range(gamma_matrix.shape[0])],
        "gamma_singular_values": singular_values,
        "gamma_singular_to_floor": gamma_singular_to_floor,
        "gamma_rank_floor": gamma_rank,
        "d_i": gamma_rank,
        "gamma_singular_bimodality_clean": bimodality_clean,
        "gamma_marginal_band": [gamma_floor, k_gamma * gamma_floor],
        "dE_perturbation_spectral_degeneracy_E": None,
        "notes": (
            "d_i is the SVD rank of Gamma_i on the F_beta-even standard isotypic. "
            "Diagonal entries are basis-dependent diagnostics only."
        ),
    }
    return result, gamma_matrix, xi_basis


def compute_d3_isotypic_summary(
    kernel_basis: np.ndarray,
    d3_ops: dict[str, np.ndarray],
    projector_floor: float,
) -> dict[str, object]:
    """Compute D3 isotypic dimensions on a recovered kernel basis.

    This is the no-integration counterpart of the projector block inside
    compute_gamma_rank_gate. It records the structural payoff of an adaptive
    kernel floor: whether the stabilized extra direction lands in T, S, or E.
    """
    k_dim = int(kernel_basis.shape[1])
    if k_dim == 0:
        return {
            "kernel_dim": 0,
            "D3_isotypic_dims": {"T": 0, "S": 0, "E": 0, "c_i": 0},
            "E_dim_even": True,
            "F_beta_even_E_dim": 0,
            "F_beta_even_E_dim_expected": 0,
            "F_beta_even_E_null_singular_values": [],
            "requires_gamma_recompute": False,
        }

    identity_k = np.eye(k_dim)
    restricted = {name: kernel_basis.T @ operator @ kernel_basis for name, operator in d3_ops.items()}
    sigma3 = restricted["sigma3"]
    sigma3_sq = restricted["sigma3_sq"]
    f_beta = restricted["F_beta"]
    f_beta_sigma3 = restricted["F_beta_sigma3"]
    f_beta_sigma3_sq = restricted["F_beta_sigma3_sq"]

    projector_t = (identity_k + sigma3 + sigma3_sq + f_beta + f_beta_sigma3 + f_beta_sigma3_sq) / 6.0
    projector_s = (identity_k + sigma3 + sigma3_sq - f_beta - f_beta_sigma3 - f_beta_sigma3_sq) / 6.0
    projector_e = identity_k - projector_t - projector_s

    t_basis_k = vector_subspace_basis(projector_t, projector_floor)
    s_basis_k = vector_subspace_basis(projector_s, projector_floor)
    e_basis_k = vector_subspace_basis(projector_e, projector_floor)
    e_dim = int(e_basis_k.shape[1])
    c_i = e_dim // 2
    e_dim_even = e_dim % 2 == 0

    if e_dim == 0:
        f_even_dim = 0
        f_even_singular_values: list[float] = []
    else:
        f_beta_e = e_basis_k.T @ f_beta @ e_basis_k
        f_even_coords, f_even_singular_values = nullspace_basis(f_beta_e - np.eye(e_dim), projector_floor)
        f_even_dim = int(f_even_coords.shape[1])

    return {
        "kernel_dim": k_dim,
        "D3_isotypic_dims": {
            "T": int(t_basis_k.shape[1]),
            "S": int(s_basis_k.shape[1]),
            "E": e_dim,
            "c_i": c_i,
        },
        "E_dim_even": e_dim_even,
        "F_beta_even_E_dim": f_even_dim,
        "F_beta_even_E_dim_expected": c_i if e_dim_even else None,
        "F_beta_even_E_null_singular_values": f_even_singular_values,
        "requires_gamma_recompute": bool(c_i > 0),
        "notes": (
            "No-integration D3 projector readout on the selected adaptive-floor "
            "kernel. If c_i>0, the full Gamma_i rank must be recomputed before "
            "the row can be interpreted."
        ),
    }


def best_so3_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    h = source.T @ target
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    return r


def rotation_angle(rotation: np.ndarray) -> float:
    cos_theta = (float(np.trace(rotation)) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, cos_theta)))


def rotation_matrix_string(rotation: np.ndarray) -> str:
    return ";".join(f"{value:.17g}" for value in rotation.reshape(-1))


def transform_positions(integrated: IntegratedOrbit, generator: Generator, times: np.ndarray) -> np.ndarray:
    period = integrated.row.period
    shift = generator.shift_fraction * period
    eps = -1.0 if generator.treverse else 1.0
    source_times = eps * (times - shift)
    positions = sample_positions(integrated, source_times)
    positions = positions[:, generator.perm, :]
    return positions @ generator.spatial.T


def residual_for_generator(
    integrated: IntegratedOrbit,
    generator: Generator,
    n_samples: int,
    phase_grid: int,
) -> dict[str, float | str | int | bool]:
    period = integrated.row.period
    times = np.linspace(0.0, period, n_samples, endpoint=False)
    generated = transform_positions(integrated, generator, times)
    generated_flat = generated.reshape(-1, 3)

    def objective(phi: float) -> tuple[float, float, np.ndarray]:
        target = sample_positions(integrated, times + phi).reshape(-1, 3)
        rotation = best_so3_rotation(generated_flat, target)
        aligned = generated_flat @ rotation.T
        diffs = target - aligned
        norms = np.linalg.norm(diffs, axis=1)
        return float(norms.max() / 2.0), float(math.sqrt(np.mean(norms * norms)) / 2.0), rotation

    grid = np.linspace(0.0, period, phase_grid, endpoint=False)
    grid_values = [objective(phi)[0] for phi in grid]
    best_index = int(np.argmin(grid_values))
    grid_step = period / phase_grid
    center = float(grid[best_index])

    def scalar(phi: float) -> float:
        return objective(phi)[0]

    local = minimize_scalar(
        scalar,
        bounds=(center - grid_step, center + grid_step),
        method="bounded",
        options={"xatol": max(period * 1e-10, 1e-12), "maxiter": 80},
    )
    phase = float(local.x % period)
    residual_inf, residual_rms, rotation = objective(phase)
    return {
        "label": integrated.row.label,
        "index": integrated.row.index,
        "m3": integrated.row.m3,
        "period": integrated.row.period,
        "generator": generator.name,
        "class": generator.klass,
        "residual_inf": residual_inf,
        "residual_rms": residual_rms,
        "phase": phase,
        "det_rotation": float(np.linalg.det(rotation)),
        "rotation_angle_rad": rotation_angle(rotation),
        "rotation_matrix": rotation_matrix_string(rotation),
        "closure_position_inf": integrated.closure_position_inf,
        "closure_velocity_inf": integrated.closure_velocity_inf,
        "residual_to_closure": residual_inf / max(integrated.closure_position_inf, DEFAULT_CLOSURE_FLOOR),
        "inertia_degenerate": integrated.inertia_degenerate,
        "integration_seconds": integrated.elapsed_seconds,
        "n_samples": n_samples,
        "phase_grid": phase_grid,
    }


def select_rows(
    rows: Iterable[OrbitRow],
    m3: float | None,
    limit: int,
    sort_period: bool,
    indices: set[int] | None = None,
) -> list[OrbitRow]:
    selected = [
        row
        for row in rows
        if (m3 is None or abs(row.m3 - m3) < 5e-13) and (indices is None or row.index in indices)
    ]
    if sort_period:
        selected.sort(key=lambda row: (row.period, row.index))
    if limit > 0:
        selected = selected[:limit]
    return selected


def parse_indices(indices: str | None) -> set[int] | None:
    if not indices:
        return None
    parsed: set[int] = set()
    for part in indices.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            parsed.add(int(part))
        except ValueError as exc:
            raise SystemExit(f"invalid orbit index {part!r} in --indices") from exc
    return parsed


def parse_generator_names(generators: str) -> list[str]:
    names = [name.strip() for name in generators.split(",") if name.strip()]
    missing = [name for name in names if name not in GENERATORS]
    if missing:
        raise SystemExit(f"unknown generators: {', '.join(missing)}")
    return names


def write_outputs(out_dir: Path, summary: dict[str, object], records: list[dict[str, object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not records:
        return
    fieldnames = list(records[0].keys())
    with (out_dir / "residuals.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def scan_record_from_residuals(
    integrated: IntegratedOrbit,
    sigma3: dict[str, object],
    sigma3_inverse: dict[str, object],
    sigma3_opposite: dict[str, object],
    sigma3_opposite_inverse: dict[str, object],
    f_beta: dict[str, object],
    sigma_tolerance: float,
    sigma_closure_multiple: float,
    identity_rotation_tolerance: float,
    closure_floor: float,
) -> dict[str, object]:
    sigma3_residual = float(sigma3["residual_inf"])
    sigma3_inverse_residual = float(sigma3_inverse["residual_inf"])
    sigma3_rotation_angle = float(sigma3["rotation_angle_rad"])
    sigma3_inverse_rotation_angle = float(sigma3_inverse["rotation_angle_rad"])
    if sigma3_residual <= sigma3_inverse_residual:
        sigma_best_generator = "sigma3"
        sigma_best_residual = sigma3_residual
        sigma_best_phase = float(sigma3["phase"])
    else:
        sigma_best_generator = "sigma3_inverse"
        sigma_best_residual = sigma3_inverse_residual
        sigma_best_phase = float(sigma3_inverse["phase"])
    sigma_group_residual = max(sigma3_residual, sigma3_inverse_residual)
    sigma_group_rotation_angle = max(sigma3_rotation_angle, sigma3_inverse_rotation_angle)

    sigma_opposite_residual = float(sigma3_opposite["residual_inf"])
    sigma_opposite_inverse_residual = float(sigma3_opposite_inverse["residual_inf"])
    sigma_opposite_rotation_angle = float(sigma3_opposite["rotation_angle_rad"])
    sigma_opposite_inverse_rotation_angle = float(sigma3_opposite_inverse["rotation_angle_rad"])
    if sigma_opposite_residual <= sigma_opposite_inverse_residual:
        sigma_opposite_best_generator = "sigma3_opposite_orientation"
        sigma_opposite_best_residual = sigma_opposite_residual
        sigma_opposite_best_phase = float(sigma3_opposite["phase"])
    else:
        sigma_opposite_best_generator = "sigma3_opposite_inverse"
        sigma_opposite_best_residual = sigma_opposite_inverse_residual
        sigma_opposite_best_phase = float(sigma3_opposite_inverse["phase"])
    sigma_opposite_group_residual = max(sigma_opposite_residual, sigma_opposite_inverse_residual)
    sigma_opposite_group_rotation_angle = max(
        sigma_opposite_rotation_angle,
        sigma_opposite_inverse_rotation_angle,
    )

    closure_scale = max(integrated.closure_position_inf, closure_floor)
    sigma_group_to_closure = sigma_group_residual / closure_scale
    sigma_opposite_group_to_closure = sigma_opposite_group_residual / closure_scale
    sigma_candidate = sigma_group_to_closure <= sigma_closure_multiple
    sigma_opposite_candidate = sigma_opposite_group_to_closure <= sigma_closure_multiple
    sigma_strict_candidate = sigma_candidate and sigma_group_rotation_angle <= identity_rotation_tolerance
    sigma_opposite_strict_candidate = (
        sigma_opposite_candidate and sigma_opposite_group_rotation_angle <= identity_rotation_tolerance
    )
    f_beta_residual = float(f_beta["residual_inf"])
    return {
        "label": integrated.row.label,
        "index": integrated.row.index,
        "line_no": integrated.row.line_no,
        "m3": integrated.row.m3,
        "z0": integrated.row.z0,
        "period": integrated.row.period,
        "stability": integrated.row.stability,
        "inertia_degenerate": integrated.inertia_degenerate,
        "closure_position_inf": integrated.closure_position_inf,
        "closure_velocity_inf": integrated.closure_velocity_inf,
        "integration_seconds": integrated.elapsed_seconds,
        "sigma3_residual_inf": sigma3_residual,
        "sigma3_inverse_residual_inf": sigma3_inverse_residual,
        "sigma3_opposite_residual_inf": sigma_opposite_residual,
        "sigma3_opposite_inverse_residual_inf": sigma_opposite_inverse_residual,
        "sigma3_rotation_angle_rad": sigma3_rotation_angle,
        "sigma3_inverse_rotation_angle_rad": sigma3_inverse_rotation_angle,
        "sigma3_opposite_rotation_angle_rad": sigma_opposite_rotation_angle,
        "sigma3_opposite_inverse_rotation_angle_rad": sigma_opposite_inverse_rotation_angle,
        "sigma_best_generator": sigma_best_generator,
        "sigma_best_residual_inf": sigma_best_residual,
        "sigma_best_phase": sigma_best_phase,
        "sigma_best_to_closure": sigma_best_residual / closure_scale,
        "sigma_group_residual_inf": sigma_group_residual,
        "sigma_group_to_closure": sigma_group_to_closure,
        "sigma_group_rotation_angle_rad": sigma_group_rotation_angle,
        "sigma_absolute_candidate": sigma_group_residual <= sigma_tolerance,
        "sigma_candidate": sigma_candidate,
        "sigma_strict_single_curve_candidate": sigma_strict_candidate,
        "sigma_opposite_best_generator": sigma_opposite_best_generator,
        "sigma_opposite_best_residual_inf": sigma_opposite_best_residual,
        "sigma_opposite_best_phase": sigma_opposite_best_phase,
        "sigma_opposite_best_to_closure": sigma_opposite_best_residual / closure_scale,
        "sigma_opposite_group_residual_inf": sigma_opposite_group_residual,
        "sigma_opposite_group_to_closure": sigma_opposite_group_to_closure,
        "sigma_opposite_group_rotation_angle_rad": sigma_opposite_group_rotation_angle,
        "sigma_opposite_absolute_candidate": sigma_opposite_group_residual <= sigma_tolerance,
        "sigma_opposite_candidate": sigma_opposite_candidate,
        "sigma_opposite_strict_single_curve_candidate": sigma_opposite_strict_candidate,
        "sigma_any_orientation_candidate": sigma_candidate or sigma_opposite_candidate,
        "sigma_any_strict_single_curve_candidate": sigma_strict_candidate or sigma_opposite_strict_candidate,
        "F_beta_residual_inf": f_beta_residual,
        "F_beta_to_closure": f_beta_residual / closure_scale,
    }


def kfacet_record_from_residuals(
    integrated: IntegratedOrbit,
    residuals: dict[str, dict[str, object]],
    generator_names: list[str],
    sigma_closure_multiple: float,
    sigma_tolerance: float,
    identity_rotation_tolerance: float,
    closure_floor: float,
) -> dict[str, object]:
    closure_scale = max(integrated.closure_position_inf, closure_floor)
    record: dict[str, object] = {
        "label": integrated.row.label,
        "index": integrated.row.index,
        "line_no": integrated.row.line_no,
        "m3": integrated.row.m3,
        "z0": integrated.row.z0,
        "period": integrated.row.period,
        "stability": integrated.row.stability,
        "inertia_degenerate": integrated.inertia_degenerate,
        "closure_position_inf": integrated.closure_position_inf,
        "closure_velocity_inf": integrated.closure_velocity_inf,
        "integration_seconds": integrated.elapsed_seconds,
    }
    so3_s_i: list[str] = []
    strict_s_i: list[str] = []
    for name in generator_names:
        residual = residuals[name]
        residual_inf = float(residual["residual_inf"])
        residual_to_closure = residual_inf / closure_scale
        rotation_angle_rad = float(residual["rotation_angle_rad"])
        candidate = residual_to_closure <= sigma_closure_multiple
        strict_candidate = candidate and rotation_angle_rad <= identity_rotation_tolerance
        if candidate:
            so3_s_i.append(name)
        if strict_candidate:
            strict_s_i.append(name)
        record[f"{name}_residual_inf"] = residual_inf
        record[f"{name}_to_closure"] = residual_to_closure
        record[f"{name}_rotation_angle_rad"] = rotation_angle_rad
        record[f"{name}_absolute_candidate"] = residual_inf <= sigma_tolerance
        record[f"{name}_candidate"] = candidate
        record[f"{name}_strict_candidate"] = strict_candidate

    # K1 freezes the concrete-generator prediction after quotienting away the
    # structural facet generator. Primary K_facet uses the strict
    # single-inertial-curve convention learned from G.2; the SO(3)-gauged
    # closure-only set is preserved as a secondary diagnostic. No other
    # generator-class merges are assumed.
    e_i = [name for name in strict_s_i if name != "F_beta"]
    so3_e_i = [name for name in so3_s_i if name != "F_beta"]
    record["S_i"] = ";".join(strict_s_i)
    record["E_i_mod_F_beta"] = ";".join(e_i)
    record["d_i"] = len(e_i)
    record["SO3_S_i"] = ";".join(so3_s_i)
    record["SO3_E_i_mod_F_beta"] = ";".join(so3_e_i)
    record["SO3_d_i"] = len(so3_e_i)
    record["strict_S_i"] = ";".join(strict_s_i)
    record["strict_E_i_mod_F_beta"] = ";".join(e_i)
    record["strict_d_i"] = len(e_i)
    record["F_beta_precondition"] = "F_beta" in strict_s_i
    record["F_beta_SO3_precondition"] = "F_beta" in so3_s_i
    return record


def conserved_invariant_values(masses: np.ndarray, x: np.ndarray, v: np.ndarray) -> dict[str, float]:
    kinetic = 0.5 * float(np.sum(masses * np.sum(v * v, axis=1)))
    potential = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            potential -= float(masses[i] * masses[j] / np.linalg.norm(x[i] - x[j]))
    angular_vector = np.sum(masses[:, None] * np.cross(x, v), axis=0)
    total_momentum = np.sum(masses[:, None] * v, axis=0)
    center_of_mass = np.average(x, axis=0, weights=masses)
    return {
        "energy": kinetic + potential,
        "kinetic_energy": kinetic,
        "potential_energy": potential,
        "angular_momentum_x": float(angular_vector[0]),
        "angular_momentum_y": float(angular_vector[1]),
        "angular_momentum_z": float(angular_vector[2]),
        "angular_momentum_norm": float(np.linalg.norm(angular_vector)),
        "total_momentum_norm": float(np.linalg.norm(total_momentum)),
        "center_of_mass_norm": float(np.linalg.norm(center_of_mass)),
    }


def invariant_record(row: OrbitRow) -> dict[str, object]:
    masses, x, v = expand_initial_state(row, center_com=True)
    values = conserved_invariant_values(masses, x, v)
    return {
        "label": row.label,
        "index": row.index,
        "line_no": row.line_no,
        "m3": row.m3,
        "z0": row.z0,
        "period": row.period,
        "stability": row.stability,
        **values,
    }


def bare_permutation_invariants(row: OrbitRow, permutation: tuple[int, int, int]) -> dict[str, float]:
    masses, x, v = expand_initial_state(row, center_com=True)
    permuted_x = x[list(permutation)]
    permuted_v = v[list(permutation)]
    return conserved_invariant_values(masses, permuted_x, permuted_v)


def invariant_pair_matches(
    left: dict[str, object],
    right: dict[str, object],
    energy_abs_tol: float,
    energy_rel_tol: float,
    angular_abs_tol: float,
    angular_rel_tol: float,
) -> bool:
    return math.isclose(
        float(left["energy"]),
        float(right["energy"]),
        rel_tol=energy_rel_tol,
        abs_tol=energy_abs_tol,
    ) and math.isclose(
        float(left["angular_momentum_norm"]),
        float(right["angular_momentum_norm"]),
        rel_tol=angular_rel_tol,
        abs_tol=angular_abs_tol,
    )


def cluster_invariant_records(
    records: list[dict[str, object]],
    energy_abs_tol: float,
    energy_rel_tol: float,
    angular_abs_tol: float,
    angular_rel_tol: float,
) -> list[list[dict[str, object]]]:
    groups: list[list[dict[str, object]]] = []
    ordered = sorted(records, key=lambda record: (float(record["energy"]), float(record["angular_momentum_norm"]), int(record["index"])))
    for record in ordered:
        energy = float(record["energy"])
        angular = float(record["angular_momentum_norm"])
        for group in groups:
            representative = group[0]
            if math.isclose(
                energy,
                float(representative["energy"]),
                rel_tol=energy_rel_tol,
                abs_tol=energy_abs_tol,
            ) and math.isclose(
                angular,
                float(representative["angular_momentum_norm"]),
                rel_tol=angular_rel_tol,
                abs_tol=angular_abs_tol,
            ):
                group.append(record)
                break
        else:
            groups.append([record])
    return groups


def command_parse(args: argparse.Namespace) -> int:
    source = args.source.upper()
    text = read_text(args.path or SUPPLEMENT_URLS[source])
    rows = parse_rows(text, source)
    by_m3: dict[float, int] = {}
    for row in rows:
        by_m3[row.m3] = by_m3.get(row.m3, 0) + 1
    summary = {
        "source": source,
        "rows": len(rows),
        "m3_counts": {f"{key:g}": by_m3[key] for key in sorted(by_m3)},
    }
    print(json.dumps(summary, indent=2))
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    source = args.source.upper()
    text = read_text(args.path or SUPPLEMENT_URLS[source])
    rows = parse_rows(text, source)
    selected = select_rows(rows, args.m3, args.limit, args.sort_period, parse_indices(args.indices))
    if not selected:
        raise SystemExit("no rows selected")

    generator_names = parse_generator_names(args.generators)

    started = time.perf_counter()
    records: list[dict[str, object]] = []
    for row in selected:
        integrated = integrate_orbit(row, args.rtol, args.atol, args.max_step_fraction)
        for name in generator_names:
            records.append(residual_for_generator(integrated, GENERATORS[name], args.n_samples, args.phase_grid))

    summary: dict[str, object] = {
        "mode": "smoke",
        "source": source,
        "rows_total": len(rows),
        "rows_selected": len(selected),
        "selected_labels": [row.label for row in selected],
        "generators": generator_names,
        "n_samples": args.n_samples,
        "phase_grid": args.phase_grid,
        "rtol": args.rtol,
        "atol": args.atol,
        "elapsed_seconds": time.perf_counter() - started,
        "note": "Smoke only: residual magnitudes are not binding K_facet evidence.",
    }

    print(json.dumps(summary, indent=2))
    if records:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    if args.out:
        write_outputs(Path(args.out), summary, records)
    return 0


def command_sigma3_scan(args: argparse.Namespace) -> int:
    source = args.source.upper()
    text = read_text(args.path or SUPPLEMENT_URLS[source])
    rows = parse_rows(text, source)
    selected = select_rows(rows, args.m3, args.limit, args.sort_period, parse_indices(args.indices))
    if not selected:
        raise SystemExit("no rows selected")

    started = time.perf_counter()
    records: list[dict[str, object]] = []
    for row_index, row in enumerate(selected, start=1):
        integrated = integrate_orbit(row, args.rtol, args.atol, args.max_step_fraction)
        sigma3 = residual_for_generator(integrated, GENERATORS["sigma3"], args.n_samples, args.phase_grid)
        sigma3_inverse = residual_for_generator(
            integrated,
            GENERATORS["sigma3_inverse"],
            args.n_samples,
            args.phase_grid,
        )
        sigma3_opposite = residual_for_generator(
            integrated,
            GENERATORS["sigma3_opposite_orientation"],
            args.n_samples,
            args.phase_grid,
        )
        sigma3_opposite_inverse = residual_for_generator(
            integrated,
            GENERATORS["sigma3_opposite_inverse"],
            args.n_samples,
            args.phase_grid,
        )
        f_beta = residual_for_generator(integrated, GENERATORS["F_beta"], args.n_samples, args.phase_grid)
        records.append(
            scan_record_from_residuals(
                integrated,
                sigma3,
                sigma3_inverse,
                sigma3_opposite,
                sigma3_opposite_inverse,
                f_beta,
                args.sigma_tolerance,
                args.sigma_closure_multiple,
                args.identity_rotation_tolerance,
                args.closure_floor,
            )
        )
        if args.report_every and row_index % args.report_every == 0:
            candidates = sum(1 for record in records if record["sigma_candidate"])
            opposite_candidates = sum(1 for record in records if record["sigma_opposite_candidate"])
            any_candidates = sum(1 for record in records if record["sigma_any_orientation_candidate"])
            strict_candidates = sum(1 for record in records if record["sigma_any_strict_single_curve_candidate"])
            print(
                f"[isotrophy] scanned {row_index}/{len(selected)} rows; "
                f"sigma_candidates={candidates}; "
                f"sigma_opposite_candidates={opposite_candidates}; "
                f"sigma_any_orientation_candidates={any_candidates}; "
                f"sigma_any_strict_single_curve_candidates={strict_candidates}",
                file=sys.stderr,
                flush=True,
            )

    sigma_candidates = [record for record in records if record["sigma_candidate"]]
    sigma_opposite_candidates = [record for record in records if record["sigma_opposite_candidate"]]
    sigma_any_orientation_candidates = [record for record in records if record["sigma_any_orientation_candidate"]]
    sigma_strict_candidates = [record for record in records if record["sigma_strict_single_curve_candidate"]]
    sigma_opposite_strict_candidates = [
        record for record in records if record["sigma_opposite_strict_single_curve_candidate"]
    ]
    sigma_any_strict_candidates = [record for record in records if record["sigma_any_strict_single_curve_candidate"]]
    expectation_met = (
        None if args.expected_sigma_count is None else len(sigma_candidates) == args.expected_sigma_count
    )
    summary: dict[str, object] = {
        "mode": "sigma3_scan",
        "source": source,
        "rows_total": len(rows),
        "rows_selected": len(selected),
        "m3": args.m3,
        "rtol": args.rtol,
        "atol": args.atol,
        "n_samples": args.n_samples,
        "phase_grid": args.phase_grid,
        "sigma_tolerance": args.sigma_tolerance,
        "sigma_closure_multiple": args.sigma_closure_multiple,
        "identity_rotation_tolerance": args.identity_rotation_tolerance,
        "expected_sigma_count": args.expected_sigma_count,
        "sigma_candidate_count": len(sigma_candidates),
        "sigma_candidate_labels": [record["label"] for record in sigma_candidates],
        "sigma_opposite_candidate_count": len(sigma_opposite_candidates),
        "sigma_opposite_candidate_labels": [record["label"] for record in sigma_opposite_candidates],
        "sigma_any_orientation_candidate_count": len(sigma_any_orientation_candidates),
        "sigma_any_orientation_candidate_labels": [record["label"] for record in sigma_any_orientation_candidates],
        "sigma_strict_single_curve_candidate_count": len(sigma_strict_candidates),
        "sigma_strict_single_curve_candidate_labels": [record["label"] for record in sigma_strict_candidates],
        "sigma_opposite_strict_single_curve_candidate_count": len(sigma_opposite_strict_candidates),
        "sigma_opposite_strict_single_curve_candidate_labels": [
            record["label"] for record in sigma_opposite_strict_candidates
        ],
        "sigma_any_strict_single_curve_candidate_count": len(sigma_any_strict_candidates),
        "sigma_any_strict_single_curve_candidate_labels": [
            record["label"] for record in sigma_any_strict_candidates
        ],
        "expectation_met": expectation_met,
        "elapsed_seconds": time.perf_counter() - started,
        "note": "G.2 scan: not a K_facet result or daughter-family count.",
    }

    print(json.dumps(summary, indent=2))
    if args.print_records and records:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    if args.out:
        write_outputs(Path(args.out), summary, records)
    if args.strict_expected and expectation_met is not True:
        return 2
    return 0


def command_kfacet_predict(args: argparse.Namespace) -> int:
    source = args.source.upper()
    text = read_text(args.path or SUPPLEMENT_URLS[source])
    rows = parse_rows(text, source)
    requested_indices = parse_indices(args.indices)
    selected = select_rows(rows, args.m3, args.limit, args.sort_period, requested_indices)
    if not selected:
        raise SystemExit("no rows selected")
    if requested_indices is not None:
        selected_indices = {row.index for row in selected}
        missing_indices = sorted(requested_indices - selected_indices)
        if missing_indices:
            raise SystemExit(f"requested indices not found: {','.join(str(index) for index in missing_indices)}")

    generator_names = parse_generator_names(args.generators)
    if "F_beta" not in generator_names:
        raise SystemExit("K1 prediction requires F_beta in --generators")

    started = time.perf_counter()
    records: list[dict[str, object]] = []
    for row_index, row in enumerate(selected, start=1):
        integrated = integrate_orbit(row, args.rtol, args.atol, args.max_step_fraction)
        residuals = {
            name: residual_for_generator(integrated, GENERATORS[name], args.n_samples, args.phase_grid)
            for name in generator_names
        }
        records.append(
            kfacet_record_from_residuals(
                integrated,
                residuals,
                generator_names,
                args.sigma_closure_multiple,
                args.sigma_tolerance,
                args.identity_rotation_tolerance,
                args.closure_floor,
            )
        )
        if args.report_every and row_index % args.report_every == 0:
            k_facet_partial = sum(int(record["d_i"]) for record in records)
            print(
                f"[isotrophy] K1 classified {row_index}/{len(selected)} rows; "
                f"K_facet_partial={k_facet_partial}",
                file=sys.stderr,
                flush=True,
            )

    f_beta_failures = [record["label"] for record in records if not record["F_beta_precondition"]]
    k_facet = sum(int(record["d_i"]) for record in records)
    so3_k_facet = sum(int(record["SO3_d_i"]) for record in records)
    per_orbit = [
        {
            "label": record["label"],
            "S_i": record["S_i"],
            "E_i_mod_F_beta": record["E_i_mod_F_beta"],
            "d_i": record["d_i"],
            "SO3_S_i": record["SO3_S_i"],
            "SO3_d_i": record["SO3_d_i"],
        }
        for record in records
    ]
    summary: dict[str, object] = {
        "mode": "kfacet_prediction",
        "source": source,
        "rows_total": len(rows),
        "rows_selected": len(selected),
        "selected_labels": [row.label for row in selected],
        "m3": args.m3,
        "rtol": args.rtol,
        "atol": args.atol,
        "n_samples": args.n_samples,
        "phase_grid": args.phase_grid,
        "sigma_tolerance": args.sigma_tolerance,
        "sigma_closure_multiple": args.sigma_closure_multiple,
        "identity_rotation_tolerance": args.identity_rotation_tolerance,
        "generators": generator_names,
        "quotient_generator": "F_beta",
        "K_facet": k_facet,
        "SO3_gauged_K_facet": so3_k_facet,
        "F_beta_precondition_failure_count": len(f_beta_failures),
        "F_beta_precondition_failures": f_beta_failures,
        "per_orbit": per_orbit,
        "elapsed_seconds": time.perf_counter() - started,
        "note": (
            "K1 prediction freeze: primary K_facet uses strict single-inertial-curve "
            "candidates and quotients only the structural F_beta generator. "
            "SO3_gauged_K_facet is secondary diagnostic output, not the frozen prediction."
        ),
    }

    print(json.dumps(summary, indent=2))
    if args.print_records and records:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    if args.out:
        write_outputs(Path(args.out), summary, records)
    if args.strict_fbeta and f_beta_failures:
        return 2
    return 0


def tau12_spatial_parity(generator_name: str) -> int:
    if generator_name in {"tau12_I", "tau12_gauge"}:
        return 1
    if generator_name == "tau12_Z":
        return -1
    raise ValueError(f"not a tau12 case-split generator: {generator_name}")


def tau12_case_record(
    integrated: IntegratedOrbit,
    residuals: dict[str, dict[str, object]],
    generator_names: list[str],
    closure_floor: float,
) -> dict[str, object]:
    period = integrated.row.period
    closure_scale = max(integrated.closure_position_inf, closure_floor)
    best_name = min(generator_names, key=lambda name: float(residuals[name]["residual_inf"]))
    best = residuals[best_name]
    phase = float(best["phase"])
    record: dict[str, object] = {
        "label": integrated.row.label,
        "index": integrated.row.index,
        "line_no": integrated.row.line_no,
        "m3": integrated.row.m3,
        "z0": integrated.row.z0,
        "period": period,
        "stability": integrated.row.stability,
        "closure_position_inf": integrated.closure_position_inf,
        "closure_velocity_inf": integrated.closure_velocity_inf,
        "integration_seconds": integrated.elapsed_seconds,
        "tau12_best_generator": best_name,
        "tau12_spatial_parity": tau12_spatial_parity(best_name),
        "tau12_residual_inf": float(best["residual_inf"]),
        "tau12_residual_rms": float(best["residual_rms"]),
        "tau12_to_closure": float(best["residual_inf"]) / closure_scale,
        "phi": phase,
        "phi_over_T": phase / period,
        "phi_over_half_T": phase / (0.5 * period),
        "rotation_angle_rad": float(best["rotation_angle_rad"]),
        "det_rotation": float(best["det_rotation"]),
        "R_i": str(best["rotation_matrix"]),
    }
    for name in generator_names:
        residual = residuals[name]
        candidate_phase = float(residual["phase"])
        record[f"{name}_residual_inf"] = float(residual["residual_inf"])
        record[f"{name}_to_closure"] = float(residual["residual_inf"]) / closure_scale
        record[f"{name}_phi"] = candidate_phase
        record[f"{name}_phi_over_T"] = candidate_phase / period
        record[f"{name}_phi_over_half_T"] = candidate_phase / (0.5 * period)
        record[f"{name}_rotation_angle_rad"] = float(residual["rotation_angle_rad"])
        record[f"{name}_det_rotation"] = float(residual["det_rotation"])
        record[f"{name}_R_i"] = str(residual["rotation_matrix"])
    return record


def classify_tau12_records(
    records: list[dict[str, object]],
    closure_multiple: float,
    induced_closure_multiple: float,
    bimodal_gap_ratio: float,
) -> dict[str, object]:
    tight = [record for record in records if float(record["tau12_to_closure"]) <= closure_multiple]
    loose = [record for record in records if float(record["tau12_to_closure"]) > closure_multiple]
    max_tight = max((float(record["tau12_to_closure"]) for record in tight), default=None)
    min_loose = min((float(record["tau12_to_closure"]) for record in loose), default=None)
    gap_ratio = None if max_tight is None or min_loose is None else min_loose / max(max_tight, DEFAULT_CLOSURE_FLOOR)

    if tight and loose and gap_ratio is not None and gap_ratio >= bimodal_gap_ratio:
        status = "bimodal"
    elif not tight and loose and min_loose is not None and min_loose >= induced_closure_multiple:
        status = "all_induced_no_closure_tight_rows"
    elif tight and not loose:
        status = "all_endomorphism_closure_tight_rows"
    else:
        status = "not_bimodal_flag_marginal"

    for record in records:
        to_closure = float(record["tau12_to_closure"])
        if to_closure <= closure_multiple:
            tau12_case = "endomorphism"
        elif status in {"bimodal", "all_induced_no_closure_tight_rows"}:
            tau12_case = "induced"
        else:
            tau12_case = "marginal_review"
        record["tau12_case"] = tau12_case
        record["closure_tight"] = to_closure <= closure_multiple

    return {
        "bimodality_status": status,
        "endomorphism_count": sum(1 for record in records if record["tau12_case"] == "endomorphism"),
        "endomorphism_labels": [record["label"] for record in records if record["tau12_case"] == "endomorphism"],
        "induced_count": sum(1 for record in records if record["tau12_case"] == "induced"),
        "induced_labels": [record["label"] for record in records if record["tau12_case"] == "induced"],
        "marginal_review_count": sum(1 for record in records if record["tau12_case"] == "marginal_review"),
        "marginal_review_labels": [record["label"] for record in records if record["tau12_case"] == "marginal_review"],
        "max_closure_tight_to_closure": max_tight,
        "min_non_tight_to_closure": min_loose,
        "bimodal_gap_ratio": gap_ratio,
    }


def command_tau12_cases(args: argparse.Namespace) -> int:
    source = args.source.upper()
    text = read_text(args.path or SUPPLEMENT_URLS[source])
    rows = parse_rows(text, source)
    requested_indices = parse_indices(args.indices)
    selected = select_rows(rows, args.m3, args.limit, args.sort_period, requested_indices)
    if not selected:
        raise SystemExit("no rows selected")
    if requested_indices is not None:
        selected_indices = {row.index for row in selected}
        missing_indices = sorted(requested_indices - selected_indices)
        if missing_indices:
            raise SystemExit(f"requested indices not found: {','.join(str(index) for index in missing_indices)}")
    generator_names = parse_generator_names(args.generators)
    not_tau12 = [name for name in generator_names if not name.startswith("tau12_")]
    if not_tau12:
        raise SystemExit(f"tau12-cases accepts only tau12_* generators: {', '.join(not_tau12)}")

    started = time.perf_counter()
    records: list[dict[str, object]] = []
    for row_index, row in enumerate(selected, start=1):
        integrated = integrate_orbit(row, args.rtol, args.atol, args.max_step_fraction)
        residuals = {
            name: residual_for_generator(integrated, GENERATORS[name], args.n_samples, args.phase_grid)
            for name in generator_names
        }
        records.append(tau12_case_record(integrated, residuals, generator_names, args.closure_floor))
        if args.report_every and row_index % args.report_every == 0:
            tight = sum(1 for record in records if float(record["tau12_to_closure"]) <= args.sigma_closure_multiple)
            print(
                f"[isotrophy] tau12 case-split classified {row_index}/{len(selected)} rows; "
                f"closure_tight={tight}",
                file=sys.stderr,
                flush=True,
            )

    case_summary = classify_tau12_records(
        records,
        args.sigma_closure_multiple,
        args.induced_closure_multiple,
        args.bimodal_gap_ratio,
    )
    summary: dict[str, object] = {
        "mode": "tau12_case_split",
        "source": source,
        "rows_total": len(rows),
        "rows_selected": len(selected),
        "selected_labels": [row.label for row in selected],
        "m3": args.m3,
        "rtol": args.rtol,
        "atol": args.atol,
        "n_samples": args.n_samples,
        "phase_grid": args.phase_grid,
        "sigma_closure_multiple": args.sigma_closure_multiple,
        "induced_closure_multiple": args.induced_closure_multiple,
        "bimodal_gap_ratio_required": args.bimodal_gap_ratio,
        "generators": generator_names,
        "condition": "exists S in {I,Z}, R in SO(3), free phi in S1: P12 S C_i(t) = R C_i(t + phi)",
        "proper_endomorphism_count": sum(
            1
            for record in records
            if record["tau12_case"] == "endomorphism" and int(record["tau12_spatial_parity"]) == 1
        ),
        "proper_endomorphism_labels": [
            record["label"]
            for record in records
            if record["tau12_case"] == "endomorphism" and int(record["tau12_spatial_parity"]) == 1
        ],
        "improper_endomorphism_count": sum(
            1
            for record in records
            if record["tau12_case"] == "endomorphism" and int(record["tau12_spatial_parity"]) == -1
        ),
        "improper_endomorphism_labels": [
            record["label"]
            for record in records
            if record["tau12_case"] == "endomorphism" and int(record["tau12_spatial_parity"]) == -1
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "note": (
            "v0.3 case-split gate only: endomorphism vs induced-representation selector over explicit spatial parity candidates. "
            "Not a K_facet value and not monodromy evidence."
        ),
        **case_summary,
    }

    print(json.dumps(summary, indent=2))
    if args.print_records and records:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    if args.out:
        write_outputs(Path(args.out), summary, records)
    if args.strict_bimodal and summary["bimodality_status"] == "not_bimodal_flag_marginal":
        return 2
    return 0


def command_invariants(args: argparse.Namespace) -> int:
    source = args.source.upper()
    text = read_text(args.path or SUPPLEMENT_URLS[source])
    rows = parse_rows(text, source)
    selected = select_rows(rows, args.m3, args.limit, args.sort_period, parse_indices(args.indices))
    if not selected:
        raise SystemExit("no rows selected")

    records = [invariant_record(row) for row in selected]
    groups = cluster_invariant_records(
        records,
        args.energy_abs_tol,
        args.energy_rel_tol,
        args.angular_abs_tol,
        args.angular_rel_tol,
    )
    group_summaries: list[dict[str, object]] = []
    for group_index, group in enumerate(groups, start=1):
        energies = [float(record["energy"]) for record in group]
        angulars = [float(record["angular_momentum_norm"]) for record in group]
        group_summaries.append(
            {
                "group": group_index,
                "size": len(group),
                "labels": [str(record["label"]) for record in group],
                "representative_energy": energies[0],
                "representative_angular_momentum_norm": angulars[0],
                "energy_span": max(energies) - min(energies),
                "angular_momentum_norm_span": max(angulars) - min(angulars),
            }
        )

    summary: dict[str, object] = {
        "mode": "invariant_cluster",
        "source": source,
        "rows_total": len(rows),
        "rows_selected": len(selected),
        "m3": args.m3,
        "energy_abs_tol": args.energy_abs_tol,
        "energy_rel_tol": args.energy_rel_tol,
        "angular_abs_tol": args.angular_abs_tol,
        "angular_rel_tol": args.angular_rel_tol,
        "group_count": len(groups),
        "duplicate_group_count": sum(1 for group in groups if len(group) > 1),
        "singleton_group_count": sum(1 for group in groups if len(group) == 1),
        "groups": group_summaries,
        "note": "Zero-integration invariant clustering from t=0 energy and angular-momentum norm.",
    }

    print(json.dumps(summary, indent=2))
    if args.print_records and records:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    if args.out:
        write_outputs(Path(args.out), summary, records)
    return 0


def command_fbeta_pair_id(args: argparse.Namespace) -> int:
    source = args.source.upper()
    text = read_text(args.path or SUPPLEMENT_URLS[source])
    rows = parse_rows(text, source)
    requested_indices = parse_indices(args.indices)
    selected = select_rows(rows, args.m3, args.limit, args.sort_period, requested_indices)
    if not selected:
        raise SystemExit("no rows selected")
    if requested_indices is not None:
        selected_indices = {row.index for row in selected}
        missing_indices = sorted(requested_indices - selected_indices)
        if missing_indices:
            raise SystemExit(f"requested indices not found: {','.join(str(index) for index in missing_indices)}")

    started = time.perf_counter()
    invariant_records = [invariant_record(row) for row in selected]
    invariant_by_index = {int(record["index"]): record for record in invariant_records}
    invariant_groups = cluster_invariant_records(
        invariant_records,
        args.energy_abs_tol,
        args.energy_rel_tol,
        args.angular_abs_tol,
        args.angular_rel_tol,
    )

    records: list[dict[str, object]] = []
    for row_index, row in enumerate(selected, start=1):
        base_invariants = invariant_by_index[row.index]
        bare12_invariants = bare_permutation_invariants(row, PERMUTATIONS["swap12"])
        catalog_partner_labels = [
            str(candidate["label"])
            for candidate in invariant_records
            if int(candidate["index"]) != row.index
            and invariant_pair_matches(
                bare12_invariants,
                candidate,
                args.energy_abs_tol,
                args.energy_rel_tol,
                args.angular_abs_tol,
                args.angular_rel_tol,
            )
        ]
        self_invariant_match = invariant_pair_matches(
            bare12_invariants,
            base_invariants,
            args.energy_abs_tol,
            args.energy_rel_tol,
            args.angular_abs_tol,
            args.angular_rel_tol,
        )

        integrated = integrate_orbit(row, args.rtol, args.atol, args.max_step_fraction)
        f_beta = residual_for_generator(integrated, GENERATORS["F_beta"], args.n_samples, args.phase_grid)
        closure_scale = max(integrated.closure_position_inf, args.closure_floor)
        f_beta_to_closure = float(f_beta["residual_inf"]) / closure_scale
        records.append(
            {
                "label": row.label,
                "index": row.index,
                "line_no": row.line_no,
                "m3": row.m3,
                "period": row.period,
                "stability": row.stability,
                "energy": base_invariants["energy"],
                "angular_momentum_norm": base_invariants["angular_momentum_norm"],
                "bare12_energy": bare12_invariants["energy"],
                "bare12_angular_momentum_norm": bare12_invariants["angular_momentum_norm"],
                "bare12_self_energy_delta": float(bare12_invariants["energy"])
                - float(base_invariants["energy"]),
                "bare12_self_angular_momentum_norm_delta": float(bare12_invariants["angular_momentum_norm"])
                - float(base_invariants["angular_momentum_norm"]),
                "self_invariant_match": self_invariant_match,
                "catalog_partner_count_excluding_self": len(catalog_partner_labels),
                "catalog_partner_labels_excluding_self": ";".join(catalog_partner_labels),
                "catalog_asymmetric": len(catalog_partner_labels) == 0,
                "closure_position_inf": integrated.closure_position_inf,
                "closure_velocity_inf": integrated.closure_velocity_inf,
                "integration_seconds": integrated.elapsed_seconds,
                "F_beta_residual_inf": f_beta["residual_inf"],
                "F_beta_to_closure": f_beta_to_closure,
                "F_beta_rotation_angle_rad": f_beta["rotation_angle_rad"],
                "F_beta_rotation_matrix": f_beta["rotation_matrix"],
                "F_beta_closure_tight": f_beta_to_closure <= args.sigma_closure_multiple,
            }
        )
        if args.report_every and row_index % args.report_every == 0:
            asymmetric = sum(1 for record in records if record["catalog_asymmetric"])
            f_beta_tight = sum(1 for record in records if record["F_beta_closure_tight"])
            print(
                f"[isotrophy] F_beta pair-ID checked {row_index}/{len(selected)} rows; "
                f"catalog_asymmetric={asymmetric}; F_beta_tight={f_beta_tight}",
                file=sys.stderr,
                flush=True,
            )

    partner_rows = [record for record in records if int(record["catalog_partner_count_excluding_self"]) > 0]
    f_beta_failures = [record for record in records if not record["F_beta_closure_tight"]]
    self_failures = [record for record in records if not record["self_invariant_match"]]
    duplicate_groups = [group for group in invariant_groups if len(group) > 1]
    strict_passed = (
        not partner_rows
        and not f_beta_failures
        and not self_failures
        and len(duplicate_groups) == 0
        and len(records) == len(selected)
    )
    summary: dict[str, object] = {
        "mode": "fbeta_pair_id_confirmation",
        "source": source,
        "rows_total": len(rows),
        "rows_selected": len(selected),
        "selected_labels": [row.label for row in selected],
        "m3": args.m3,
        "rtol": args.rtol,
        "atol": args.atol,
        "n_samples": args.n_samples,
        "phase_grid": args.phase_grid,
        "sigma_closure_multiple": args.sigma_closure_multiple,
        "energy_abs_tol": args.energy_abs_tol,
        "energy_rel_tol": args.energy_rel_tol,
        "angular_abs_tol": args.angular_abs_tol,
        "angular_rel_tol": args.angular_rel_tol,
        "strict_catalog_invariant_group_count": len(invariant_groups),
        "strict_catalog_singleton_group_count": sum(1 for group in invariant_groups if len(group) == 1),
        "strict_catalog_duplicate_group_count": len(duplicate_groups),
        "catalog_match_count_excluding_self": sum(
            int(record["catalog_partner_count_excluding_self"]) for record in records
        ),
        "rows_with_catalog_partner_count": len(partner_rows),
        "rows_with_catalog_partner_labels": [record["label"] for record in partner_rows],
        "catalog_asymmetric_count": sum(1 for record in records if record["catalog_asymmetric"]),
        "self_invariant_match_count": sum(1 for record in records if record["self_invariant_match"]),
        "self_invariant_failure_labels": [record["label"] for record in self_failures],
        "F_beta_closure_tight_count": sum(1 for record in records if record["F_beta_closure_tight"]),
        "F_beta_failure_labels": [record["label"] for record in f_beta_failures],
        "structural_cocycle": "F_beta = ((12), tau-active, Rpi)",
        "tau_component": "schema-constant active",
        "per_row_tau_flag": False,
        "partner_orbit_ivp": False,
        "partner_monodromy_relation": "M_(12*C_i) = rho(Rpi) * M_i^-1 * rho(Rpi)^-1",
        "strict_passed": strict_passed,
        "elapsed_seconds": time.perf_counter() - started,
        "note": (
            "Receipt-discipline confirmation for the F_beta chain. Bare (12) invariant matching has no "
            "inside-catalog partner among the strict 21, while structural F_beta is closure-tight for each row. "
            "This does not freeze K_facet_v0.3 and does not run partner-orbit IVPs."
        ),
    }

    print(json.dumps(summary, indent=2))
    if args.print_records and records:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    if args.out:
        write_outputs(Path(args.out), summary, records)
    if args.strict and not strict_passed:
        return 2
    return 0


def command_kfacet_sentinel(args: argparse.Namespace) -> int:
    """Run the v0.3i sentinel implementation gates without authorizing Gamma_i."""
    source = args.source.upper()
    text = read_text(args.path or SUPPLEMENT_URLS[source])
    rows = parse_rows(text, source)
    selected = select_rows(rows, args.m3, 0, False, {args.sentinel_index})
    if not selected:
        raise SystemExit(f"sentinel row O_{{{args.sentinel_index}}}({args.m3:g}) not in catalog")
    row = selected[0]

    print(f"[kfacet-sentinel] integrating {row.label} for gate scaffolding")
    integrated = integrate_orbit(row, args.rtol, args.atol, args.max_step_fraction)
    print(
        "[kfacet-sentinel] orbit closure "
        f"position={integrated.closure_position_inf:.3e} velocity={integrated.closure_velocity_inf:.3e}"
    )

    print("[kfacet-sentinel] computing M_i")
    monodromy = compute_monodromy(integrated, args.rtol, args.atol, args.max_step_fraction)

    print("[kfacet-sentinel] constructing typed D3 operators")
    d3_candidate_inputs = {
        "sample_perm_forward_Tminus": False,
        "inverse_perm_forward_Tminus": True,
    }
    d3_candidate_results: dict[str, tuple[dict[str, np.ndarray], dict[str, object]]] = {}
    for orientation_name, use_inverse_permutation in d3_candidate_inputs.items():
        candidate_ops = construct_d3_elements_v0(
            integrated,
            args.rtol,
            args.atol,
            args.max_step_fraction,
            invert_permutation_for_tangent=use_inverse_permutation,
        )
        candidate_gate = verify_d3_relations(candidate_ops, monodromy, args.relation_floor, args.closure_floor)
        d3_candidate_results[orientation_name] = (candidate_ops, candidate_gate)

    selected_orientation, (d3_ops, d3_gate) = min(
        d3_candidate_results.items(),
        key=lambda item: float(item[1][1]["max_residual"]),
    )
    d3_gate["selected_orientation"] = selected_orientation
    d3_gate["orientation_candidates"] = {
        name: summarize_d3_gate_candidate(gate) for name, (_ops, gate) in d3_candidate_results.items()
    }
    print(
        "[kfacet-sentinel] D3 gate "
        f"{'PASS' if d3_gate['passed'] else 'FAIL'} "
        f"orientation={selected_orientation} "
        f"kernel_dim={d3_gate['kernel_dim']} "
        f"max(kernel-projected)={float(d3_gate['max_residual']):.3e} "
        f"||[M,sigma3]||={float(d3_gate['diagnostic_full_V0']['sigma3_M_i_commutator_full_V0']):.3e}"
    )

    k_fib_basis, k_fib_summary = compute_k_fib_basis(monodromy, integrated, args.closure_floor)
    omega = canonical_omega_18(integrated.row.masses)
    omega_gate = verify_reduced_omega(monodromy, k_fib_basis, omega, args.nondegeneracy_floor)
    print(
        "[kfacet-sentinel] omega gate "
        f"{'PASS' if omega_gate['passed'] else 'FAIL'} K_fib_dim={omega_gate['K_fib_dim']}"
    )

    preliminary_gates_passed = bool(d3_gate["passed"] and omega_gate["passed"])
    partial_eps_gate: dict[str, object] = {
        "gate": "partial_epsilon_finite_difference",
        "enabled": False,
        "skipped": True,
        "reason": "run with --verify-partial-eps to enable the finite-difference cross-check",
    }
    partial_eps_monodromy = None
    partial_eps_finite_difference = None
    if args.verify_partial_eps:
        if not preliminary_gates_passed:
            partial_eps_gate = {
                "gate": "partial_epsilon_finite_difference",
                "enabled": True,
                "skipped": True,
                "reason": "D3/omega gate failed before partial-epsilon verification",
                "passed": False,
            }
        else:
            print("[kfacet-sentinel] verifying partial-epsilon monodromy by fixed-IC finite difference")
            partial_eps_gate, partial_eps_monodromy, partial_eps_finite_difference = (
                verify_partial_eps_via_finite_difference(
                    integrated,
                    args.fd_h,
                    args.rtol,
                    args.atol,
                    args.max_step_fraction,
                    args.fd_floor,
                    reference_monodromy=monodromy,
                    k_joint_baseline_relative=args.joint_baseline_floor,
                )
            )
            jb_rel = partial_eps_gate['joint_baseline_consistency'].get('residual_relative')
            jb_rel_str = f"{float(jb_rel):.3e}" if jb_rel is not None else "n/a"
            print(
                "[kfacet-sentinel] partial-epsilon gate "
                f"{'PASS' if partial_eps_gate['passed'] else 'FAIL'} "
                f"FD_rel={float(partial_eps_gate['residual_relative']):.3e} "
                f"(<{float(partial_eps_gate['k_fd_relative']):.0e}) "
                f"jb_rel={jb_rel_str} "
                f"residual_inf={float(partial_eps_gate['residual_inf']):.3e}"
            )
    else:
        print("[kfacet-sentinel] partial-epsilon gate SKIPPED (no --verify-partial-eps flag)")

    all_gates_passed = bool(
        preliminary_gates_passed and (not args.verify_partial_eps or partial_eps_gate.get("passed") is True)
    )
    gamma_gate: dict[str, object] = {
        "gate": "Gamma_i_rank",
        "enabled": False,
        "skipped": True,
        "reason": "run with --authorize-sentinel-run after --verify-partial-eps to compute Gamma_i",
    }
    gamma_matrix = None
    gamma_xi_basis = None
    if args.authorize_sentinel_run:
        if not all_gates_passed:
            gamma_gate = {
                "gate": "Gamma_i_rank",
                "enabled": True,
                "skipped": True,
                "passed": False,
                "reason": "implementation gates failed before Gamma_i",
            }
        elif partial_eps_monodromy is None:
            gamma_gate = {
                "gate": "Gamma_i_rank",
                "enabled": True,
                "skipped": True,
                "passed": False,
                "reason": "--authorize-sentinel-run requires --verify-partial-eps so partial_epsilon M_i is trusted",
            }
            all_gates_passed = False
        else:
            print("[kfacet-sentinel] computing Gamma_i rank gate")
            kernel_basis, _kernel_singular_values = compute_kernel_basis(monodromy, args.closure_floor)
            gamma_gate, gamma_matrix, gamma_xi_basis = compute_gamma_rank_gate(
                kernel_basis,
                d3_ops,
                omega,
                partial_eps_monodromy,
                args.rtol,
                args.k_gamma,
                args.k_int,
                args.gamma_projector_floor,
            )
            print(
                "[kfacet-sentinel] Gamma_i gate "
                f"{'PASS' if gamma_gate['passed'] else 'FAIL'} "
                f"c_i={gamma_gate['D3_isotypic_dims']['c_i']} "
                f"d_i={gamma_gate['d_i']} "
                f"singulars={gamma_gate['gamma_singular_values']}"
            )
            all_gates_passed = bool(all_gates_passed and gamma_gate["passed"])

    receipt = {
        "mode": "kfacet_sentinel_gates",
        "gate_version": "v0.3h-gamma" if args.authorize_sentinel_run else (
            "v0.3i-partial-eps" if args.verify_partial_eps else "v0.3i-gates"
        ),
        "row_index": row.index,
        "label": row.label,
        "m3": row.m3,
        "period": row.period,
        "rtol": args.rtol,
        "atol": args.atol,
        "max_step_fraction": args.max_step_fraction,
        "closure_position_inf": integrated.closure_position_inf,
        "closure_velocity_inf": integrated.closure_velocity_inf,
        "integration_seconds": integrated.elapsed_seconds,
        "M_i_norm_inf": float(np.linalg.norm(monodromy, ord=np.inf)),
        "kernel_floor_interpretation": {
            "closure_floor": args.closure_floor,
            "sweep_calibration": (
                "The 21-row v0.3h sweep showed closure_floor=1e-8 splits the "
                "noise-floor kernel cluster on several rows; closure_floor=1e-7 "
                "lands inside the observed spectral gap (last kernel <= 7.5e-8, "
                "first non-kernel >= 5.7e-2). Treat c_i readings as structural "
                "only when D3_K_fib/kernel leakage is clean at this floor."
            ),
        },
        "K_fib": k_fib_summary,
        "D3_relations": d3_gate,
        "reduced_omega": omega_gate,
        "partial_epsilon_finite_difference": partial_eps_gate,
        "Gamma_i_rank": gamma_gate,
        "all_gates_passed": all_gates_passed,
        "sentinel_scope": {
            "full_gamma_run_authorized": bool(args.authorize_sentinel_run),
            "full_21_authorized": False,
            "supplementary_B_authorized": False,
            "notes": (
                "This command is sentinel-only. It never authorizes full-21 K_facet "
                "or supplementary-B comparison."
            ),
        },
    }

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "gate_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        np.save(out_dir / "M_i.npy", monodromy)
        np.save(out_dir / "K_fib_basis.npy", k_fib_basis)
        np.save(out_dir / "omega.npy", omega)
        for name, operator in d3_ops.items():
            np.save(out_dir / f"D3_{name}.npy", operator)
        if partial_eps_monodromy is not None and partial_eps_finite_difference is not None:
            np.save(out_dir / "partial_eps_M_i.npy", partial_eps_monodromy)
            np.save(out_dir / "partial_eps_M_i_fd.npy", partial_eps_finite_difference)
        if gamma_matrix is not None and gamma_xi_basis is not None:
            np.save(out_dir / "Gamma_i.npy", gamma_matrix)
            np.save(out_dir / "Gamma_xi_basis.npy", gamma_xi_basis)

    if not all_gates_passed:
        print("[kfacet-sentinel] HALT: implementation gates failed; Gamma_i run remains blocked")
        return 1
    if args.authorize_sentinel_run:
        print("[kfacet-sentinel] sentinel Gamma_i receipt complete; full-21 comparison remains unauthorised")
        return 0
    print("[kfacet-sentinel] gates passed; Gamma_i run remains unauthorised without --authorize-sentinel-run")
    return 0


def reprocess_kernel_floor_for_row(
    row_dir: Path,
) -> dict[str, object]:
    """Reprocess a sentinel receipt directory under the adaptive-floor ladder.

    Inputs: a directory containing ``M_i.npy`` plus ``D3_*.npy`` matrices and
    a ``gate_receipt.json`` from a prior sentinel run. No integration is
    performed.

    For each floor in the pre-registered ladder, this function recovers a
    kernel basis from ``ker(M_i - I)`` at that floor, computes the D3
    operator leakage outside the recovered kernel, and reports whether the
    floor simultaneously satisfies:
      * Leakage: every D3 operator leak <= ADAPTIVE_FLOOR_PROJECTOR_FLOOR.
      * Gap ratio: max(kept SV) / min(rejected SV) <= ADAPTIVE_FLOOR_GAP_RATIO_THRESHOLD.
      * First-rejected absolute: min(rejected SV) >= ADAPTIVE_FLOOR_FIRST_REJECTED_THRESHOLD.

    The selected floor is the smallest in the ladder satisfying all three.
    Floors at or above ADAPTIVE_FLOOR_SUSPICIOUS_THRESHOLD are flagged
    suspicious even when they pass, so the receipt records suspicion as a
    structured outcome rather than silently accepting the boundary.
    """
    m_path = row_dir / "M_i.npy"
    receipt_path = row_dir / "gate_receipt.json"
    if not m_path.is_file() or not receipt_path.is_file():
        return {
            "row_dir": str(row_dir),
            "outcome": "missing_inputs",
            "missing": [str(p) for p in (m_path, receipt_path) if not p.is_file()],
        }
    M_i = np.load(m_path)
    base_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    row_index = base_receipt.get("row_index")
    label = base_receipt.get("label")
    period = base_receipt.get("period")
    m3 = base_receipt.get("m3")

    d3_names = ("e", "sigma3", "sigma3_sq", "F_beta", "F_beta_sigma3", "F_beta_sigma3_sq")
    d3_ops: dict[str, np.ndarray] = {}
    for name in d3_names:
        op_path = row_dir / f"D3_{name}.npy"
        if not op_path.is_file():
            return {
                "row_dir": str(row_dir),
                "row_index": row_index,
                "outcome": "missing_inputs",
                "missing": [str(op_path)],
            }
        d3_ops[name] = np.load(op_path)

    identity = np.eye(M_i.shape[0])
    _u, singular_values, vh = np.linalg.svd(M_i - identity)
    singular_values_desc = [float(value) for value in singular_values]

    ladder_results: list[dict[str, object]] = []
    selected_floor: float | None = None
    selected_entry: dict[str, object] | None = None
    for floor in ADAPTIVE_FLOOR_LADDER:
        mask = singular_values < floor
        kept_indices = np.where(mask)[0]
        rejected_indices = np.where(~mask)[0]
        k_dim = int(kept_indices.size)
        if kept_indices.size == 0:
            last_kept_sv: float | None = None
        else:
            last_kept_sv = float(singular_values[kept_indices].max())
        if rejected_indices.size == 0:
            first_rejected_sv: float | None = None
        else:
            first_rejected_sv = float(singular_values[rejected_indices].min())
        if last_kept_sv is None or first_rejected_sv is None or first_rejected_sv == 0.0:
            gap_ratio: float | None = None
        else:
            gap_ratio = float(last_kept_sv / first_rejected_sv)
        kernel_basis = vh.T[:, mask]
        ambient_projector = kernel_basis @ kernel_basis.T if kernel_basis.size else np.zeros_like(M_i)
        ambient_identity = np.eye(M_i.shape[0])
        d3_leakage = {
            name: float(np.linalg.norm((ambient_identity - ambient_projector) @ operator @ kernel_basis, ord=np.inf))
            if kernel_basis.size
            else 0.0
            for name, operator in d3_ops.items()
        }
        max_leak = max(d3_leakage.values()) if d3_leakage else 0.0
        satisfies_leak = max_leak <= ADAPTIVE_FLOOR_PROJECTOR_FLOOR and k_dim > 0
        satisfies_gap_ratio = (
            gap_ratio is not None and gap_ratio <= ADAPTIVE_FLOOR_GAP_RATIO_THRESHOLD
        )
        satisfies_first_rejected = (
            first_rejected_sv is not None
            and first_rejected_sv >= ADAPTIVE_FLOOR_FIRST_REJECTED_THRESHOLD
        )
        satisfies_all = bool(satisfies_leak and satisfies_gap_ratio and satisfies_first_rejected)
        entry: dict[str, object] = {
            "floor": float(floor),
            "k_dim": k_dim,
            "last_kept_sv": last_kept_sv,
            "first_rejected_sv": first_rejected_sv,
            "gap_ratio": gap_ratio,
            "D3_leakage_inf": d3_leakage,
            "max_D3_leakage_inf": max_leak,
            "satisfies_leak": satisfies_leak,
            "satisfies_gap_ratio": satisfies_gap_ratio,
            "satisfies_first_rejected": satisfies_first_rejected,
            "satisfies_all": satisfies_all,
        }
        ladder_results.append(entry)
        if satisfies_all and selected_floor is None:
            selected_floor = float(floor)
            selected_entry = entry

    if selected_entry is None:
        outcome = "adaptive_floor_failed"
        is_suspicious = False
        selected_isotypic_summary: dict[str, object] | None = None
    elif selected_floor is not None and selected_floor >= ADAPTIVE_FLOOR_SUSPICIOUS_THRESHOLD:
        outcome = "adaptive_floor_suspicious"
        is_suspicious = True
        selected_mask = singular_values < selected_floor
        selected_kernel_basis = vh.T[:, selected_mask]
        selected_isotypic_summary = compute_d3_isotypic_summary(
            selected_kernel_basis,
            d3_ops,
            ADAPTIVE_FLOOR_PROJECTOR_FLOOR,
        )
    else:
        outcome = "adaptive_floor_resolved"
        is_suspicious = False
        selected_mask = singular_values < selected_floor
        selected_kernel_basis = vh.T[:, selected_mask]
        selected_isotypic_summary = compute_d3_isotypic_summary(
            selected_kernel_basis,
            d3_ops,
            ADAPTIVE_FLOOR_PROJECTOR_FLOOR,
        )

    receipt: dict[str, object] = {
        "mode": "kfacet_reprocess_floor",
        "reprocessor_version": ADAPTIVE_FLOOR_VERSION,
        "row_index": row_index,
        "label": label,
        "m3": m3,
        "period": period,
        "input_receipt": str(receipt_path),
        "row_dir": str(row_dir),
        "M_i_norm_inf": float(np.linalg.norm(M_i, ord=np.inf)),
        "kernel_singular_values_desc": singular_values_desc,
        "floor_ladder": [float(value) for value in ADAPTIVE_FLOOR_LADDER],
        "projector_floor": ADAPTIVE_FLOOR_PROJECTOR_FLOOR,
        "gap_ratio_threshold": ADAPTIVE_FLOOR_GAP_RATIO_THRESHOLD,
        "first_rejected_threshold": ADAPTIVE_FLOOR_FIRST_REJECTED_THRESHOLD,
        "suspicious_floor_threshold": ADAPTIVE_FLOOR_SUSPICIOUS_THRESHOLD,
        "ladder_results": ladder_results,
        "selected_floor": selected_floor,
        "selected_k_dim": selected_entry["k_dim"] if selected_entry else None,
        "selected_D3_leakage_inf": selected_entry["D3_leakage_inf"] if selected_entry else None,
        "selected_max_D3_leakage_inf": selected_entry["max_D3_leakage_inf"] if selected_entry else None,
        "selected_gap_ratio": selected_entry["gap_ratio"] if selected_entry else None,
        "selected_first_rejected_sv": selected_entry["first_rejected_sv"] if selected_entry else None,
        "selected_is_suspicious": is_suspicious,
        "selected_D3_isotypic_summary": selected_isotypic_summary,
        "selected_c_i": (
            selected_isotypic_summary["D3_isotypic_dims"]["c_i"]
            if selected_isotypic_summary
            else None
        ),
        "selected_requires_gamma_recompute": (
            selected_isotypic_summary["requires_gamma_recompute"]
            if selected_isotypic_summary
            else None
        ),
        "outcome": outcome,
        "notes": (
            "Pre-registered selection: smallest floor in the ladder with "
            "max D3 leakage <= projector floor, gap ratio <= 1e-3, and first "
            "rejected SV >= 1e-3. Suspicion is raised when the selected floor "
            "is at or above 3e-6. D3 isotypic dimensions are recomputed on the "
            "selected kernel to decide whether the adaptive boundary vector "
            "lands in T/S or creates a standard E sector."
        ),
    }
    return receipt


def command_kfacet_reprocess_floor(args: argparse.Namespace) -> int:
    """No-integration adaptive-floor reprocessor over existing sentinel receipts."""
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"input directory not found: {input_dir}")

    candidate_dirs: list[Path]
    if args.indices:
        wanted = {int(token.strip()) for token in args.indices.split(",") if token.strip()}
        candidate_dirs = []
        for child in sorted(input_dir.iterdir()):
            if not child.is_dir() or not child.name.startswith("O"):
                continue
            try:
                row_index = int(child.name[1:])
            except ValueError:
                continue
            if row_index in wanted:
                candidate_dirs.append(child)
    else:
        candidate_dirs = sorted(
            child for child in input_dir.iterdir()
            if child.is_dir() and child.name.startswith("O") and child.name[1:].isdigit()
        )

    if not candidate_dirs:
        raise SystemExit(f"no O_x subdirectories found under {input_dir}")

    out_dir = Path(args.out) if args.out else input_dir.parent / f"{input_dir.name}-adaptive-floor"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_row_receipts: list[dict[str, object]] = []
    outcome_counts: dict[str, int] = {}
    selected_floor_histogram: dict[str, int] = {}
    for row_dir in candidate_dirs:
        print(f"[kfacet-reprocess-floor] reprocessing {row_dir.name}")
        receipt = reprocess_kernel_floor_for_row(row_dir)
        per_row_receipts.append(receipt)
        outcome = str(receipt.get("outcome"))
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        selected_floor = receipt.get("selected_floor")
        if isinstance(selected_floor, float):
            key = f"{selected_floor:.0e}"
            selected_floor_histogram[key] = selected_floor_histogram.get(key, 0) + 1
        row_out = out_dir / row_dir.name
        row_out.mkdir(parents=True, exist_ok=True)
        (row_out / "adaptive_floor_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        max_leak = receipt.get("selected_max_D3_leakage_inf")
        max_leak_str = f"{float(max_leak):.2e}" if isinstance(max_leak, (int, float)) else "n/a"
        selected_floor_str = (
            f"{float(selected_floor):.0e}" if isinstance(selected_floor, float) else "none"
        )
        print(
            f"[kfacet-reprocess-floor]   row={receipt.get('row_index')} "
            f"outcome={outcome} selected_floor={selected_floor_str} "
            f"k_dim={receipt.get('selected_k_dim')} "
            f"max_D3_leak={max_leak_str}"
        )

    manifest: dict[str, object] = {
        "mode": "kfacet_reprocess_floor_manifest",
        "reprocessor_version": ADAPTIVE_FLOOR_VERSION,
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "floor_ladder": [float(value) for value in ADAPTIVE_FLOOR_LADDER],
        "projector_floor": ADAPTIVE_FLOOR_PROJECTOR_FLOOR,
        "gap_ratio_threshold": ADAPTIVE_FLOOR_GAP_RATIO_THRESHOLD,
        "first_rejected_threshold": ADAPTIVE_FLOOR_FIRST_REJECTED_THRESHOLD,
        "suspicious_floor_threshold": ADAPTIVE_FLOOR_SUSPICIOUS_THRESHOLD,
        "rows": [
            {
                "row_index": receipt.get("row_index"),
                "outcome": receipt.get("outcome"),
                "selected_floor": receipt.get("selected_floor"),
                "selected_k_dim": receipt.get("selected_k_dim"),
                "selected_max_D3_leakage_inf": receipt.get("selected_max_D3_leakage_inf"),
                "selected_gap_ratio": receipt.get("selected_gap_ratio"),
                "selected_first_rejected_sv": receipt.get("selected_first_rejected_sv"),
                "selected_is_suspicious": receipt.get("selected_is_suspicious"),
                "selected_D3_isotypic_dims": (
                    receipt.get("selected_D3_isotypic_summary", {}).get("D3_isotypic_dims")
                    if isinstance(receipt.get("selected_D3_isotypic_summary"), dict)
                    else None
                ),
                "selected_c_i": receipt.get("selected_c_i"),
                "selected_requires_gamma_recompute": receipt.get("selected_requires_gamma_recompute"),
            }
            for receipt in per_row_receipts
        ],
        "summary": {
            "total": len(per_row_receipts),
            "outcome_counts": outcome_counts,
            "selected_floor_histogram": selected_floor_histogram,
            "suspicious_rows": [
                r.get("row_index") for r in per_row_receipts if r.get("selected_is_suspicious")
            ],
            "failed_rows": [
                r.get("row_index") for r in per_row_receipts if r.get("outcome") == "adaptive_floor_failed"
            ],
            "standard_E_rows": [
                r.get("row_index") for r in per_row_receipts if (r.get("selected_c_i") or 0) > 0
            ],
            "structural_zero_rows": [
                r.get("row_index")
                for r in per_row_receipts
                if r.get("outcome") == "adaptive_floor_resolved" and r.get("selected_c_i") == 0
            ],
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[kfacet-reprocess-floor] wrote {len(per_row_receipts)} row receipts and manifest under {out_dir}")
    print(f"[kfacet-reprocess-floor] outcomes: {outcome_counts}")
    print(f"[kfacet-reprocess-floor] selected floor histogram: {selected_floor_histogram}")
    if manifest["summary"]["suspicious_rows"]:
        print(f"[kfacet-reprocess-floor] suspicious rows: {manifest['summary']['suspicious_rows']}")
    if manifest["summary"]["failed_rows"]:
        print(f"[kfacet-reprocess-floor] failed rows: {manifest['summary']['failed_rows']}")
        if not args.allow_failed_rows:
            return 1
        print("[kfacet-reprocess-floor] continuing because --allow-failed-rows was set")
    if manifest["summary"]["standard_E_rows"]:
        print(f"[kfacet-reprocess-floor] standard-E rows require Gamma_i recompute: {manifest['summary']['standard_E_rows']}")
    return 0


def _bridge_ladder_floor_admitting(value: float) -> float | None:
    """Smallest rung in BRIDGE_AUDIT_LADDER strictly greater than ``value``."""
    for rung in BRIDGE_AUDIT_LADDER:
        if value < rung:
            return float(rung)
    return None


def _bridge_projector_readout(
    kernel_basis: np.ndarray,
    d3_ops: dict[str, np.ndarray],
    floor: float,
) -> dict[str, object]:
    """Compute P_T/P_S/P_E spectra and D3 relation residuals on a kernel basis.

    A clean isotypic projector has singular values clustered near {0, 1}; a
    defective E block shows up as marginal P_E singular values in (1e-3, 0.999)
    or an odd E dimension that cannot split into 2D standard irreps. The D3
    relation residuals quantify whether the chosen kernel basis is a valid
    representation of D3 -- if any relation residual is large, the projector
    readout is itself untrustworthy.
    """
    k_dim = int(kernel_basis.shape[1])
    if k_dim == 0:
        return {
            "floor": float(floor),
            "k_dim": 0,
            "D3_isotypic_dims": {"T": 0, "S": 0, "E": 0, "c_i": 0},
            "P_T_singular_values": [],
            "P_S_singular_values": [],
            "P_E_singular_values": [],
            "P_E_marginal_singular_values": [],
            "D3_relation_residuals": {},
            "defective_E_flagged": False,
        }
    identity_k = np.eye(k_dim)
    restricted = {name: kernel_basis.T @ op @ kernel_basis for name, op in d3_ops.items()}
    sigma3 = restricted["sigma3"]
    sigma3_sq = restricted["sigma3_sq"]
    f_beta = restricted["F_beta"]
    f_beta_sigma3 = restricted["F_beta_sigma3"]
    f_beta_sigma3_sq = restricted["F_beta_sigma3_sq"]

    p_t = (identity_k + sigma3 + sigma3_sq + f_beta + f_beta_sigma3 + f_beta_sigma3_sq) / 6.0
    p_s = (identity_k + sigma3 + sigma3_sq - f_beta - f_beta_sigma3 - f_beta_sigma3_sq) / 6.0
    p_e = identity_k - p_t - p_s
    p_t_svs = [float(value) for value in np.linalg.svd(p_t, compute_uv=False)]
    p_s_svs = [float(value) for value in np.linalg.svd(p_s, compute_uv=False)]
    p_e_svs = [float(value) for value in np.linalg.svd(p_e, compute_uv=False)]
    t_basis = vector_subspace_basis(p_t, BRIDGE_PROJECTOR_FLOOR)
    s_basis = vector_subspace_basis(p_s, BRIDGE_PROJECTOR_FLOOR)
    e_basis = vector_subspace_basis(p_e, BRIDGE_PROJECTOR_FLOOR)
    e_dim = int(e_basis.shape[1])
    c_i = e_dim // 2

    sigma3_cubed_residual = float(
        np.linalg.norm(sigma3 @ sigma3 @ sigma3 - identity_k, ord=np.inf)
    )
    f_beta_squared_residual = float(np.linalg.norm(f_beta @ f_beta - identity_k, ord=np.inf))
    # sigma3^{-1} == sigma3^2 on the kernel basis; check F_beta sigma3 F_beta == sigma3^{-1}.
    f_beta_sigma3_f_beta_residual = float(
        np.linalg.norm(f_beta @ sigma3 @ f_beta - sigma3_sq, ord=np.inf)
    )

    marginal_band = [s for s in p_e_svs if 1e-3 < s < (1.0 - 1e-3)]
    defective_E_flagged = (e_dim % 2 != 0) or bool(marginal_band)
    return {
        "floor": float(floor),
        "k_dim": k_dim,
        "D3_isotypic_dims": {
            "T": int(t_basis.shape[1]),
            "S": int(s_basis.shape[1]),
            "E": e_dim,
            "c_i": c_i,
        },
        "P_T_singular_values": p_t_svs,
        "P_S_singular_values": p_s_svs,
        "P_E_singular_values": p_e_svs,
        "P_E_marginal_singular_values": marginal_band,
        "D3_relation_residuals": {
            "sigma3_cubed_minus_I": sigma3_cubed_residual,
            "F_beta_squared_minus_I": f_beta_squared_residual,
            "F_beta_sigma3_F_beta_minus_sigma3_inv": f_beta_sigma3_f_beta_residual,
        },
        "defective_E_flagged": defective_E_flagged,
    }


def _bridge_outcome_notes(outcome: str) -> str:
    notes_map = {
        "no_bridge_present": (
            "No singular values remain in the unresolved bridge band after "
            "respecting the row's adaptive floor. The row is structurally "
            "clean and does not require a bridge audit."
        ),
        "all_neutral_overlap_explained": (
            "All bridge vectors project into the neutral basis with overlap "
            ">= 0.999. The bridge SVs are missed neutral/Hamiltonian "
            "directions; v0.3h needs a neutral-quotient refinement, not "
            "tighter integration."
        ),
        "jordan_block_explained": (
            "At least one bridge vector has ||(M-I)^2 v|| / ||(M-I) v|| < 1e-3, "
            "consistent with a Jordan-block root at eigenvalue 1. Bridge SV "
            "is a generalized-kernel direction; v0.3h needs Jordan-block "
            "accommodation."
        ),
        "defective_E_block_confirmed": (
            "All bridge vectors produce defective E-block projector readouts "
            "at both fixed and ladder bridge floors (odd-dim E or marginal "
            "P_E singular values). This indicates a real defective D3 "
            "representation at the bridge boundary, not a quotient or Jordan "
            "artifact."
        ),
        "jordan_suspected": (
            "Algebraic multiplicity near eigenvalue 1 exceeds geometric "
            "multiplicity at the fixed bridge floor, but no bridge vector "
            "aligns with the (M-I)^2 generalized-kernel chain. Treat as "
            "suspected, not confirmed; schedule generalized-null alignment "
            "work before tighter-rtol rerun."
        ),
        "unexplained_escalate_to_rtol": (
            "Neither neutral overlap, Jordan chain, defective E block, nor "
            "eigenvalue evidence aligns with the bridge vector. Escalate to "
            "a tighter-rtol rerun to test whether the bridge SV is "
            "rtol-dependent."
        ),
    }
    return notes_map.get(outcome, "unrecognized outcome")


def audit_bridge_for_row(
    row_dir: Path,
    catalog_row: OrbitRow,
    d3_ops: dict[str, np.ndarray],
    M_i: np.ndarray,
    adaptive_receipt: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run the no-integration bridge audit on a row's existing matrices."""
    identity = np.eye(M_i.shape[0])
    m_minus_i = M_i - identity
    _u, singular_values, vh = np.linalg.svd(m_minus_i)
    singular_values_desc = [float(value) for value in singular_values]

    adaptive_selected_floor: float | None = None
    adaptive_outcome: str | None = None
    if adaptive_receipt is not None:
        adaptive_outcome_raw = adaptive_receipt.get("outcome")
        adaptive_outcome = str(adaptive_outcome_raw) if adaptive_outcome_raw is not None else None
        maybe_floor = adaptive_receipt.get("selected_floor")
        if isinstance(maybe_floor, (int, float)):
            adaptive_selected_floor = float(maybe_floor)
    effective_bridge_lower = max(
        BRIDGE_BAND_LOWER,
        adaptive_selected_floor if adaptive_selected_floor is not None else BRIDGE_BAND_LOWER,
    )
    bridge_indices = [
        i for i, s in enumerate(singular_values)
        if effective_bridge_lower < s < BRIDGE_BAND_UPPER
    ]

    # Eigenvalue evidence near 1 (diagnostic, NOT the sole classifier per the
    # signed-off tweak: ill-conditioned M_i can produce spurious near-1
    # eigenvalues from np.linalg.eig; we keep this as a flag and gate the
    # 'jordan_block_explained' outcome on SVD/generalized-null alignment.)
    eigenvalues = np.linalg.eigvals(M_i)
    near_one = []
    for lam in eigenvalues:
        if abs(lam - 1.0) < BRIDGE_EIGENVALUE_NEAR_ONE_BAND:
            near_one.append({
                "value_real": float(np.real(lam)),
                "value_imag": float(np.imag(lam)),
                "abs_minus_one": float(abs(lam - 1.0)),
            })
    algebraic_mult_near_one = len(near_one)

    # Reconstruct y0 directly from the catalog row -- NO integration.
    # Design note: this function accepts a `catalog_row` and recomputes y0
    # from the Li-Liao ansatz. A future receipt schema may carry y0 directly;
    # the call site can then bypass the catalog parse.
    masses, x0_pos, v0 = expand_initial_state(catalog_row, center_com=True)
    y0 = pack_state(x0_pos, v0)
    stub = IntegratedOrbit(
        row=catalog_row,
        solution=None,
        y0=y0,
        elapsed_seconds=0.0,
        closure_position_inf=0.0,
        closure_velocity_inf=0.0,
        inertia_degenerate=False,
    )
    neutral_basis = compute_neutral_basis(M_i, stub, BRIDGE_PROJECTOR_FLOOR)
    p_neutral = neutral_basis @ neutral_basis.T if neutral_basis.size else np.zeros_like(M_i)

    bridge_vectors_out: list[dict[str, object]] = []
    for bridge_idx in bridge_indices:
        sv = float(singular_values[bridge_idx])
        v_bridge = vh[bridge_idx, :].astype(float)
        norm_v = float(np.linalg.norm(v_bridge))
        neutral_projected = p_neutral @ v_bridge
        neutral_overlap = float(np.linalg.norm(neutral_projected) / max(norm_v, 1e-300))
        # Jordan chain: applying (M-I) twice to a Jordan root drops the norm
        # by another factor of |lambda - 1| ~ SV; for a generic non-Jordan
        # vector the second application stays the same scale.
        mv1 = m_minus_i @ v_bridge
        mv2 = m_minus_i @ mv1
        r1 = float(np.linalg.norm(mv1))
        r2 = float(np.linalg.norm(mv2))
        jordan_chain_drop = float(r2 / max(r1, 1e-300))

        # Fixed bridge floor admits ALL SVs below 1e-3 (i.e., every bridge).
        mask_fixed = singular_values < BRIDGE_FIXED_FLOOR
        kernel_basis_fixed = vh.T[:, mask_fixed]
        readout_fixed = _bridge_projector_readout(
            kernel_basis_fixed, d3_ops, BRIDGE_FIXED_FLOOR
        )
        # Ladder bridge floor: smallest BRIDGE_AUDIT_LADDER rung > this SV.
        ladder_floor = _bridge_ladder_floor_admitting(sv)
        if ladder_floor is None:
            readout_ladder: dict[str, object] = {
                "floor_unavailable": True,
                "note": "no rung in BRIDGE_AUDIT_LADDER admits this bridge SV",
            }
        else:
            mask_ladder = singular_values < ladder_floor
            kernel_basis_ladder = vh.T[:, mask_ladder]
            readout_ladder = _bridge_projector_readout(
                kernel_basis_ladder, d3_ops, ladder_floor
            )

        bridge_vectors_out.append({
            "sv": sv,
            "index_in_sv_array": int(bridge_idx),
            "right_singular_vector": [float(value) for value in v_bridge],
            "neutral_overlap": neutral_overlap,
            "neutral_in": neutral_overlap >= BRIDGE_NEUTRAL_OVERLAP_THRESHOLD,
            "jordan_chain_image_norm": r1,
            "jordan_chain_image_squared_norm": r2,
            "jordan_chain_drop": jordan_chain_drop,
            "jordan_chain_in": jordan_chain_drop < BRIDGE_JORDAN_CHAIN_THRESHOLD,
            "projector_readout_fixed_floor": readout_fixed,
            "projector_readout_ladder_floor": readout_ladder,
        })

    geometric_mult_at_fixed = int(np.sum(singular_values < BRIDGE_FIXED_FLOOR))
    jordan_defect = max(0, algebraic_mult_near_one - geometric_mult_at_fixed)

    # Pre-registered outcome categorization. Per the signed-off tweak, the
    # 'jordan_block_explained' branch REQUIRES bridge-vector chain alignment
    # (jordan_chain_in), not just an eigenvalue defect; eigenvalue-only
    # evidence downgrades to 'jordan_suspected'.
    all_neutral = bool(bridge_vectors_out) and all(b["neutral_in"] for b in bridge_vectors_out)
    some_jordan_chain = any(b["jordan_chain_in"] for b in bridge_vectors_out)
    all_defective = bool(bridge_vectors_out) and all(
        b["projector_readout_fixed_floor"].get("defective_E_flagged")
        and b["projector_readout_ladder_floor"].get("defective_E_flagged")
        for b in bridge_vectors_out
    )
    if not bridge_vectors_out:
        outcome = "no_bridge_present"
    elif all_neutral:
        outcome = "all_neutral_overlap_explained"
    elif some_jordan_chain:
        outcome = "jordan_block_explained"
    elif all_defective:
        outcome = "defective_E_block_confirmed"
    elif jordan_defect > 0:
        outcome = "jordan_suspected"
    else:
        outcome = "unexplained_escalate_to_rtol"

    return {
        "mode": "kfacet_bridge_audit",
        "audit_version": BRIDGE_AUDIT_VERSION,
        "row_index": catalog_row.index,
        "label": catalog_row.label,
        "m3": catalog_row.m3,
        "period": catalog_row.period,
        "row_dir": str(row_dir),
        "M_i_norm_inf": float(np.linalg.norm(M_i, ord=np.inf)),
        "M_minus_I_singular_values_desc": singular_values_desc,
        "bridge_band_lower": BRIDGE_BAND_LOWER,
        "bridge_band_lower_effective": effective_bridge_lower,
        "bridge_band_upper": BRIDGE_BAND_UPPER,
        "adaptive_floor_outcome": adaptive_outcome,
        "adaptive_selected_floor": adaptive_selected_floor,
        "fixed_bridge_floor": BRIDGE_FIXED_FLOOR,
        "bridge_indices": bridge_indices,
        "bridge_vectors": bridge_vectors_out,
        "eigenvalues_near_one": near_one,
        "algebraic_multiplicity_near_one": algebraic_mult_near_one,
        "geometric_multiplicity_at_fixed_floor": geometric_mult_at_fixed,
        "jordan_defect_diagnostic": jordan_defect,
        "jordan_defect_note": (
            "Diagnostic only. np.linalg.eig on an ill-conditioned M_i can "
            "report spurious eigenvalues near 1; jordan_block_explained is "
            "only reported when a bridge vector also passes the SVD-based "
            "(M-I)^2 chain alignment test."
        ),
        "neutral_basis_dim": int(neutral_basis.shape[1]) if neutral_basis.size else 0,
        "neutral_basis_floor": BRIDGE_PROJECTOR_FLOOR,
        "outcome": outcome,
        "interpretation_notes": _bridge_outcome_notes(outcome),
    }


def command_kfacet_bridge_audit(args: argparse.Namespace) -> int:
    """Receipt-only bridge audit subcommand. Reuses existing M_i.npy + D3_*.npy."""
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"input directory not found: {input_dir}")

    # Decide which rows to audit. Explicit --rows wins; otherwise auto-target
    # the manifest's failed_rows list (the structural-bridge candidates).
    if args.rows:
        wanted = {int(token.strip()) for token in args.rows.split(",") if token.strip()}
    else:
        manifest_path = input_dir / "manifest.json"
        if manifest_path.is_file():
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            wanted = set(m.get("summary", {}).get("failed_rows", []))
        else:
            wanted = set()
    if not wanted:
        raise SystemExit(
            "no rows to audit (pass --rows X,Y,... or ensure manifest.json "
            "has failed_rows under summary)"
        )

    source = args.source.upper()
    text = read_text(args.path or SUPPLEMENT_URLS[source])
    catalog_rows = parse_rows(text, source)
    catalog_by_index = {
        r.index: r for r in catalog_rows if abs(r.m3 - args.m3) < 5e-13
    }

    out_dir = (
        Path(args.out) if args.out else input_dir.parent / f"{input_dir.name}-bridge-audit"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    d3_names = ("e", "sigma3", "sigma3_sq", "F_beta", "F_beta_sigma3", "F_beta_sigma3_sq")
    per_row: list[dict[str, object]] = []
    outcome_counts: dict[str, int] = {}
    for row_index in sorted(wanted):
        row_dir = input_dir / f"O{row_index}"
        if not row_dir.is_dir():
            print(f"[kfacet-bridge-audit] WARN: O_{row_index} subdir not found in {input_dir}")
            continue
        # Resolve M_i.npy + D3_*.npy. They live either in this dir directly
        # (sentinel output) or in the source dir referenced by the adaptive
        # reprocessor's receipt (when input_dir is a reprocessor output).
        m_path = row_dir / "M_i.npy"
        ar_receipt: dict[str, object] | None = None
        if m_path.is_file():
            src_dir = row_dir
        else:
            ar_path = row_dir / "adaptive_floor_receipt.json"
            if not ar_path.is_file():
                print(f"[kfacet-bridge-audit] WARN: O_{row_index} has neither M_i.npy nor adaptive_floor_receipt.json")
                continue
            ar_receipt = json.loads(ar_path.read_text(encoding="utf-8"))
            src_dir = Path(ar_receipt.get("row_dir", str(row_dir)))
            m_path = src_dir / "M_i.npy"
            if not m_path.is_file():
                print(f"[kfacet-bridge-audit] WARN: O_{row_index} M_i.npy not found via adaptive receipt -> {src_dir}")
                continue
        M_i = np.load(m_path)
        d3_ops: dict[str, np.ndarray] = {}
        d3_missing = False
        for name in d3_names:
            op_path = src_dir / f"D3_{name}.npy"
            if not op_path.is_file():
                print(f"[kfacet-bridge-audit] WARN: O_{row_index} missing D3_{name}.npy at {src_dir}")
                d3_missing = True
                break
            d3_ops[name] = np.load(op_path)
        if d3_missing:
            continue
        catalog_row = catalog_by_index.get(row_index)
        if catalog_row is None:
            print(f"[kfacet-bridge-audit] WARN: O_{row_index} not in catalog at m3={args.m3}")
            continue
        print(f"[kfacet-bridge-audit] auditing {catalog_row.label}")
        receipt = audit_bridge_for_row(row_dir, catalog_row, d3_ops, M_i, ar_receipt)
        per_row.append(receipt)
        outcome = str(receipt["outcome"])
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        row_out_dir = out_dir / f"O{row_index}"
        row_out_dir.mkdir(parents=True, exist_ok=True)
        (row_out_dir / "bridge_audit_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[kfacet-bridge-audit]   row={row_index} outcome={outcome} "
            f"n_bridges={len(receipt['bridge_vectors'])} "
            f"alg_mult_near_1={receipt['algebraic_multiplicity_near_one']} "
            f"geom_mult_fixed_floor={receipt['geometric_multiplicity_at_fixed_floor']} "
            f"jordan_defect_diag={receipt['jordan_defect_diagnostic']}"
        )

    manifest = {
        "mode": "kfacet_bridge_audit_manifest",
        "audit_version": BRIDGE_AUDIT_VERSION,
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "fixed_bridge_floor": BRIDGE_FIXED_FLOOR,
        "bridge_band": [BRIDGE_BAND_LOWER, BRIDGE_BAND_UPPER],
        "neutral_overlap_threshold": BRIDGE_NEUTRAL_OVERLAP_THRESHOLD,
        "jordan_chain_threshold": BRIDGE_JORDAN_CHAIN_THRESHOLD,
        "eigenvalue_near_one_band": BRIDGE_EIGENVALUE_NEAR_ONE_BAND,
        "projector_floor": BRIDGE_PROJECTOR_FLOOR,
        "rows": [
            {
                "row_index": r["row_index"],
                "outcome": r["outcome"],
                "n_bridge_vectors": len(r["bridge_vectors"]),
                "adaptive_selected_floor": r.get("adaptive_selected_floor"),
                "bridge_band_lower_effective": r.get("bridge_band_lower_effective"),
                "max_neutral_overlap": max(
                    (b["neutral_overlap"] for b in r["bridge_vectors"]), default=None
                ),
                "min_jordan_chain_drop": min(
                    (b["jordan_chain_drop"] for b in r["bridge_vectors"]), default=None
                ),
                "algebraic_multiplicity_near_one": r["algebraic_multiplicity_near_one"],
                "geometric_multiplicity_at_fixed_floor": r["geometric_multiplicity_at_fixed_floor"],
                "jordan_defect_diagnostic": r["jordan_defect_diagnostic"],
            }
            for r in per_row
        ],
        "summary": {
            "total": len(per_row),
            "outcome_counts": outcome_counts,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[kfacet-bridge-audit] wrote {len(per_row)} row receipts and manifest under {out_dir}")
    print(f"[kfacet-bridge-audit] outcomes: {outcome_counts}")
    return 0


# v0.4b row-z2-sweep: per-row M_i + Z_2 isotypic dim extraction + threshold rule
# Implementation guardrails (signed off by Codex, see
# docs/isotrophy/kfacet/kfacet_v04b_gamma3_form.md):
#  - Compute F_beta_even_dim / F_beta_odd_dim on K_fib (neutral-quotiented).
#  - Record projection_target = "K_fib" explicitly in every receipt.
#  - Store full pre-rule feature set so the aggregator is brutally dumb.
V04B_VERSION = "v0.4b-row-z2-sweep"
V04B_DEFAULT_CLOSURE_FLOOR = 1e-7
V04B_PROJECTOR_FLOOR = 1e-3
V04B_BRIDGE_BAND_LOWER = 1e-7
V04B_BRIDGE_BAND_UPPER = 1e-3


def _v04b_isotypic_dim(projector: np.ndarray, floor: float) -> int:
    """Count orthonormal columns of a projector with singular value above floor."""
    if projector.size == 0:
        return 0
    _u, sv, _vh = np.linalg.svd(projector, full_matrices=False)
    return int(np.sum(sv > floor))


def _v04b_threshold_rule(f_beta_even_dim: int, f_beta_odd_dim: int) -> str:
    """Pre-registered threshold rule: predict S iff even_dim >= odd_dim."""
    return "S" if f_beta_even_dim >= f_beta_odd_dim else "U"


def command_kfacet_row_z2_sweep(args: argparse.Namespace) -> int:
    """v0.4b per-row Z_2 isotypic-dim extraction + threshold-rule prediction.

    For each selected row: integrate, compute M_i, neutral basis, K_fib;
    project F_beta onto K_fib and count F_beta-even / F_beta-odd dims; apply
    the locked threshold rule; record full pre-rule feature set and the
    prediction. Aggregator reads these receipts via v04b_aggregator.py.
    """
    source = args.source.upper()
    text = read_text(args.path or SUPPLEMENT_URLS[source])
    rows_all = parse_rows(text, source)
    if args.indices:
        wanted = {int(token.strip()) for token in args.indices.split(",") if token.strip()}
        selected = [r for r in rows_all if r.index in wanted and abs(r.m3 - args.m3) < 5e-13]
    else:
        selected = [r for r in rows_all if abs(r.m3 - args.m3) < 5e-13]
    if args.limit > 0:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit(f"no rows selected for source={source} m3={args.m3} indices={args.indices}")

    out_dir = Path(args.out) if args.out else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    closure_floor = float(args.closure_floor)
    projector_floor = float(args.projector_floor)
    f_beta = f_beta_action_v0()
    rows_results: list[dict[str, object]] = []
    print(
        f"[kfacet-row-z2-sweep] source={source} m3={args.m3} n_rows={len(selected)} "
        f"closure_floor={closure_floor:.0e} projector_floor={projector_floor:.0e}"
    )
    for row in selected:
        integrated = integrate_orbit(row, args.rtol, args.atol, args.max_step_fraction)
        M_i = compute_monodromy(integrated, args.rtol, args.atol, args.max_step_fraction)
        # Raw kernel dims
        identity = np.eye(18)
        _u, svs_M, _vh = np.linalg.svd(M_i - identity)
        kernel_dim = int(np.sum(svs_M < closure_floor))
        bridge_band_count = int(np.sum(
            (svs_M > V04B_BRIDGE_BAND_LOWER) & (svs_M < V04B_BRIDGE_BAND_UPPER)
        ))
        # Neutral basis + K_fib (the registered projection target)
        neutral_basis = compute_neutral_basis(M_i, integrated, closure_floor)
        k_fib_basis, k_fib_summary = compute_k_fib_basis(M_i, integrated, closure_floor)
        neutral_dim = int(neutral_basis.shape[1])
        k_fib_dim = int(k_fib_basis.shape[1])
        # F_beta leakage outside K_fib (diagnostic)
        if k_fib_dim > 0:
            P_kfib = k_fib_basis @ k_fib_basis.T
            f_beta_leakage = float(
                np.linalg.norm((np.eye(18) - P_kfib) @ f_beta @ k_fib_basis, ord=np.inf)
            )
        else:
            f_beta_leakage = 0.0
        # F_beta isotypic dims on K_fib
        if k_fib_dim > 0:
            identity_k = np.eye(k_fib_dim)
            f_beta_k = k_fib_basis.T @ f_beta @ k_fib_basis
            p_plus = (identity_k + f_beta_k) / 2.0
            p_minus = (identity_k - f_beta_k) / 2.0
            f_beta_even_dim = _v04b_isotypic_dim(p_plus, projector_floor)
            f_beta_odd_dim = _v04b_isotypic_dim(p_minus, projector_floor)
        else:
            f_beta_even_dim = 0
            f_beta_odd_dim = 0
        # Neutral-sector conditioning: how well does neutral_basis stay
        # F_beta-invariant? large value flags structural concern with the
        # quotient. Diagnostic only; not used by the rule.
        if neutral_dim > 0:
            P_neut = neutral_basis @ neutral_basis.T
            neutral_sector_conditioning = float(
                np.linalg.norm((np.eye(18) - P_neut) @ f_beta @ neutral_basis, ord=np.inf)
            )
        else:
            neutral_sector_conditioning = 0.0
        # Apply locked threshold rule
        prediction = _v04b_threshold_rule(f_beta_even_dim, f_beta_odd_dim)
        prediction_correct = (prediction == row.stability)
        row_receipt = {
            "mode": "kfacet_row_z2_sweep",
            "version": V04B_VERSION,
            "projection_target": "K_fib",
            "row_index": row.index,
            "label": row.label,
            "m3": row.m3,
            "z0": row.z0,
            "period": row.period,
            "stability": row.stability,
            "closure_floor": closure_floor,
            "projector_floor": projector_floor,
            "kernel_dim": kernel_dim,
            "neutral_dim": neutral_dim,
            "K_fib_dim": k_fib_dim,
            "F_beta_even_dim": f_beta_even_dim,
            "F_beta_odd_dim": f_beta_odd_dim,
            "bridge_band_count": bridge_band_count,
            "F_beta_leakage_inf": f_beta_leakage,
            "neutral_sector_conditioning": neutral_sector_conditioning,
            "K_fib_summary": k_fib_summary,
            "gamma3_prediction": prediction,
            "observed_stability": row.stability,
            "gamma3_prediction_correct": prediction_correct,
            "integration_seconds": integrated.elapsed_seconds,
            "closure_position_inf": integrated.closure_position_inf,
            "closure_velocity_inf": integrated.closure_velocity_inf,
            "M_i_norm_inf": float(np.linalg.norm(M_i, ord=np.inf)),
        }
        rows_results.append(row_receipt)
        print(
            f"[kfacet-row-z2-sweep] O_{row.index}({row.m3}): "
            f"K_fib_dim={k_fib_dim} F_beta_even={f_beta_even_dim} "
            f"F_beta_odd={f_beta_odd_dim} predict={prediction} "
            f"observed={row.stability} correct={prediction_correct}"
        )
        if out_dir is not None:
            row_out = out_dir / f"O{row.index}"
            row_out.mkdir(parents=True, exist_ok=True)
            (row_out / "row_z2_receipt.json").write_text(
                json.dumps(row_receipt, indent=2) + "\n", encoding="utf-8"
            )
            np.save(row_out / "M_i.npy", M_i)
            np.save(row_out / "F_beta.npy", f_beta)

    if out_dir is not None:
        manifest = {
            "mode": "kfacet_row_z2_sweep_manifest",
            "version": V04B_VERSION,
            "source": source,
            "m3": args.m3,
            "indices_arg": args.indices,
            "limit_arg": args.limit,
            "closure_floor": closure_floor,
            "projector_floor": projector_floor,
            "row_count": len(rows_results),
            "rows": rows_results,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[kfacet-row-z2-sweep] wrote {len(rows_results)} row receipts under {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sundog isotrophy workbench smoke harness")
    sub = parser.add_subparsers(dest="command", required=True)

    parse_cmd = sub.add_parser("parse", help="fetch/parse a supplementary file and report row counts")
    parse_cmd.add_argument("--source", choices=sorted(SUPPLEMENT_URLS), default="A")
    parse_cmd.add_argument("--path", help="local supplementary file instead of the live URL")
    parse_cmd.set_defaults(func=command_parse)

    smoke = sub.add_parser("smoke", help="run a capped integration/residual smoke")
    smoke.add_argument("--source", choices=sorted(SUPPLEMENT_URLS), default="A")
    smoke.add_argument("--path", help="local supplementary file instead of the live URL")
    smoke.add_argument("--m3", type=float, default=1.0)
    smoke.add_argument("--limit", type=int, default=1)
    smoke.add_argument("--indices", help="comma-separated orbit indices to include before limit is applied")
    smoke.add_argument("--sort-period", action="store_true", default=True)
    smoke.add_argument("--no-sort-period", dest="sort_period", action="store_false")
    smoke.add_argument("--generators", default="sigma3,sigma3_inverse,F_beta,F_delta")
    smoke.add_argument("--n-samples", type=int, default=181)
    smoke.add_argument("--phase-grid", type=int, default=25)
    smoke.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    smoke.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    smoke.add_argument("--max-step-fraction", type=float, default=DEFAULT_MAX_STEP_FRACTION)
    smoke.add_argument("--out", help="optional output directory for manifest.json and residuals.csv")
    smoke.set_defaults(func=command_smoke)

    scan = sub.add_parser("sigma3-scan", help="scan selected m3 rows for sigma3 precondition candidates")
    scan.add_argument("--source", choices=sorted(SUPPLEMENT_URLS), default="A")
    scan.add_argument("--path", help="local supplementary file instead of the live URL")
    scan.add_argument("--m3", type=float, default=1.0)
    scan.add_argument("--limit", type=int, default=0)
    scan.add_argument("--indices", help="comma-separated orbit indices to include before limit is applied")
    scan.add_argument("--sort-period", action="store_true", default=True)
    scan.add_argument("--no-sort-period", dest="sort_period", action="store_false")
    scan.add_argument("--n-samples", type=int, default=1009)
    scan.add_argument("--phase-grid", type=int, default=73)
    scan.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    scan.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    scan.add_argument("--max-step-fraction", type=float, default=DEFAULT_MAX_STEP_FRACTION)
    scan.add_argument("--sigma-tolerance", type=float, default=DEFAULT_SIGMA_TOLERANCE)
    scan.add_argument("--sigma-closure-multiple", type=float, default=DEFAULT_SIGMA_CLOSURE_MULTIPLE)
    scan.add_argument("--identity-rotation-tolerance", type=float, default=DEFAULT_IDENTITY_ROTATION_TOLERANCE)
    scan.add_argument("--expected-sigma-count", type=int, default=DEFAULT_EXPECTED_SIGMA_COUNT)
    scan.add_argument("--closure-floor", type=float, default=DEFAULT_CLOSURE_FLOOR)
    scan.add_argument("--strict-expected", action="store_true")
    scan.add_argument("--print-records", action="store_true")
    scan.add_argument("--report-every", type=int, default=25)
    scan.add_argument("--out", help="optional output directory for manifest.json and residuals.csv")
    scan.set_defaults(func=command_sigma3_scan)

    kfacet = sub.add_parser("kfacet-predict", help="freeze the K1 K_facet prediction on strict G.2 rows")
    kfacet.add_argument("--source", choices=sorted(SUPPLEMENT_URLS), default="A")
    kfacet.add_argument("--path", help="local supplementary file instead of the live URL")
    kfacet.add_argument("--m3", type=float, default=1.0)
    kfacet.add_argument("--limit", type=int, default=0)
    kfacet.add_argument(
        "--indices",
        default=",".join(str(index) for index in DEFAULT_KFACET_STRICT_INDICES),
        help="comma-separated strict G.2 orbit indices for the K1 prediction",
    )
    kfacet.add_argument("--sort-period", action="store_true", default=False)
    kfacet.add_argument("--no-sort-period", dest="sort_period", action="store_false")
    kfacet.add_argument("--generators", default=",".join(DEFAULT_KFACET_GENERATORS))
    kfacet.add_argument("--n-samples", type=int, default=1009)
    kfacet.add_argument("--phase-grid", type=int, default=73)
    kfacet.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    kfacet.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    kfacet.add_argument("--max-step-fraction", type=float, default=DEFAULT_MAX_STEP_FRACTION)
    kfacet.add_argument("--sigma-tolerance", type=float, default=DEFAULT_SIGMA_TOLERANCE)
    kfacet.add_argument("--sigma-closure-multiple", type=float, default=DEFAULT_SIGMA_CLOSURE_MULTIPLE)
    kfacet.add_argument("--identity-rotation-tolerance", type=float, default=DEFAULT_IDENTITY_ROTATION_TOLERANCE)
    kfacet.add_argument("--closure-floor", type=float, default=DEFAULT_CLOSURE_FLOOR)
    kfacet.add_argument("--strict-fbeta", action="store_true")
    kfacet.add_argument("--print-records", action="store_true")
    kfacet.add_argument("--report-every", type=int, default=7)
    kfacet.add_argument("--out", help="optional output directory for manifest.json and residuals.csv")
    kfacet.set_defaults(func=command_kfacet_predict)

    tau12 = sub.add_parser("tau12-cases", help="classify strict G.2 rows by the v0.3 tau12 gauge-case split")
    tau12.add_argument("--source", choices=sorted(SUPPLEMENT_URLS), default="A")
    tau12.add_argument("--path", help="local supplementary file instead of the live URL")
    tau12.add_argument("--m3", type=float, default=1.0)
    tau12.add_argument("--limit", type=int, default=0)
    tau12.add_argument(
        "--indices",
        default=",".join(str(index) for index in DEFAULT_KFACET_STRICT_INDICES),
        help="comma-separated strict G.2 orbit indices for the tau12 case split",
    )
    tau12.add_argument("--sort-period", action="store_true", default=False)
    tau12.add_argument("--no-sort-period", dest="sort_period", action="store_false")
    tau12.add_argument("--generators", default=",".join(DEFAULT_TAU12_GENERATORS))
    tau12.add_argument("--n-samples", type=int, default=1009)
    tau12.add_argument("--phase-grid", type=int, default=73)
    tau12.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    tau12.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    tau12.add_argument("--max-step-fraction", type=float, default=DEFAULT_MAX_STEP_FRACTION)
    tau12.add_argument("--sigma-closure-multiple", type=float, default=DEFAULT_SIGMA_CLOSURE_MULTIPLE)
    tau12.add_argument("--induced-closure-multiple", type=float, default=1e5)
    tau12.add_argument("--bimodal-gap-ratio", type=float, default=1e5)
    tau12.add_argument("--closure-floor", type=float, default=DEFAULT_CLOSURE_FLOOR)
    tau12.add_argument("--strict-bimodal", action="store_true")
    tau12.add_argument("--print-records", action="store_true")
    tau12.add_argument("--report-every", type=int, default=7)
    tau12.add_argument("--out", help="optional output directory for manifest.json and residuals.csv")
    tau12.set_defaults(func=command_tau12_cases)

    invariants = sub.add_parser("invariants", help="cluster selected rows by zero-integration conserved invariants")
    invariants.add_argument("--source", choices=sorted(SUPPLEMENT_URLS), default="A")
    invariants.add_argument("--path", help="local supplementary file instead of the live URL")
    invariants.add_argument("--m3", type=float, default=1.0)
    invariants.add_argument("--limit", type=int, default=0)
    invariants.add_argument("--indices", help="comma-separated orbit indices to include before limit is applied")
    invariants.add_argument("--sort-period", action="store_true", default=True)
    invariants.add_argument("--no-sort-period", dest="sort_period", action="store_false")
    invariants.add_argument("--energy-abs-tol", type=float, default=1e-10)
    invariants.add_argument("--energy-rel-tol", type=float, default=1e-10)
    invariants.add_argument("--angular-abs-tol", type=float, default=1e-10)
    invariants.add_argument("--angular-rel-tol", type=float, default=1e-10)
    invariants.add_argument("--print-records", action="store_true")
    invariants.add_argument("--out", help="optional output directory for manifest.json and residuals.csv")
    invariants.set_defaults(func=command_invariants)

    fbeta_pair = sub.add_parser(
        "fbeta-pair-id",
        help="confirm the v0.3 F_beta pair-ID chain on strict G.2 rows",
    )
    fbeta_pair.add_argument("--source", choices=sorted(SUPPLEMENT_URLS), default="A")
    fbeta_pair.add_argument("--path", help="local supplementary file instead of the live URL")
    fbeta_pair.add_argument("--m3", type=float, default=1.0)
    fbeta_pair.add_argument("--limit", type=int, default=0)
    fbeta_pair.add_argument(
        "--indices",
        default=",".join(str(index) for index in DEFAULT_KFACET_STRICT_INDICES),
        help="comma-separated strict G.2 orbit indices for F_beta pair-ID confirmation",
    )
    fbeta_pair.add_argument("--sort-period", action="store_true", default=False)
    fbeta_pair.add_argument("--no-sort-period", dest="sort_period", action="store_false")
    fbeta_pair.add_argument("--n-samples", type=int, default=1009)
    fbeta_pair.add_argument("--phase-grid", type=int, default=73)
    fbeta_pair.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    fbeta_pair.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    fbeta_pair.add_argument("--max-step-fraction", type=float, default=DEFAULT_MAX_STEP_FRACTION)
    fbeta_pair.add_argument("--sigma-closure-multiple", type=float, default=DEFAULT_SIGMA_CLOSURE_MULTIPLE)
    fbeta_pair.add_argument("--closure-floor", type=float, default=DEFAULT_CLOSURE_FLOOR)
    fbeta_pair.add_argument("--energy-abs-tol", type=float, default=1e-10)
    fbeta_pair.add_argument("--energy-rel-tol", type=float, default=1e-10)
    fbeta_pair.add_argument("--angular-abs-tol", type=float, default=1e-10)
    fbeta_pair.add_argument("--angular-rel-tol", type=float, default=1e-10)
    fbeta_pair.add_argument("--strict", action="store_true")
    fbeta_pair.add_argument("--print-records", action="store_true")
    fbeta_pair.add_argument("--report-every", type=int, default=7)
    fbeta_pair.add_argument("--out", help="optional output directory for manifest.json and residuals.csv")
    fbeta_pair.set_defaults(func=command_fbeta_pair_id)

    sentinel = sub.add_parser(
        "kfacet-sentinel",
        help="v0.3i sentinel calibration gate scaffold (D3 and omega checks only)",
    )
    sentinel.add_argument("--source", choices=sorted(SUPPLEMENT_URLS), default="A")
    sentinel.add_argument("--path", help="local supplementary file instead of the live URL")
    sentinel.add_argument("--m3", type=float, default=1.0)
    sentinel.add_argument("--sentinel-index", type=int, default=62)
    sentinel.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    sentinel.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    sentinel.add_argument("--max-step-fraction", type=float, default=DEFAULT_MAX_STEP_FRACTION)
    # relation_floor / nondegeneracy_floor are gate floors for the current
    # kernel-projected v0.3i scaffolding. The sigma operators avoid inverse
    # flow entirely, but unstable or nearly degenerate Floquet structure still
    # amplifies rtol-scale integration noise in composition checks. Closure-
    # relative scaling is the proper refinement once the runner is generalised
    # beyond a single sentinel.
    sentinel.add_argument("--relation-floor", type=float, default=1e-3)
    sentinel.add_argument("--closure-floor", type=float, default=1e-7)
    sentinel.add_argument("--nondegeneracy-floor", type=float, default=1e-3)
    sentinel.add_argument("--verify-partial-eps", action="store_true")
    sentinel.add_argument("--fd-h", type=float, default=1e-6)
    # k_FD: closure-relative tolerance on (||closed - FD||_inf) / max(||closed||,||FD||).
    # Default 1e-4 reflects that central-difference truncation is O(h^2) (h=1e-6
    # gives ~1e-12 truncation) but unstable monodromy norms can amplify rtol=1e-12
    # integration noise to ~1e-5 relative; 1e-4 leaves a margin without masking
    # real structural failure. Replaces the prior absolute 1e-9 floor.
    sentinel.add_argument("--fd-floor", type=float, default=1e-4)
    # k_jb: closure-relative tolerance on the joint-solver-vs-standalone baseline
    # consistency check. The joint solver and standalone M_i integrate the same
    # ODE with the same tolerances, but the 21-row sweep placed most rows in
    # the 1e-9..5e-9 relative band. 1e-8 is the calibrated catalog floor, not
    # a tuning knob for Gamma_i.
    sentinel.add_argument("--joint-baseline-floor", type=float, default=1e-8)
    sentinel.add_argument("--k-gamma", type=float, default=3.0)
    sentinel.add_argument("--k-int", type=float, default=10.0)
    sentinel.add_argument("--gamma-projector-floor", type=float, default=1e-3)
    sentinel.add_argument("--authorize-sentinel-run", action="store_true")
    sentinel.add_argument("--out", help="optional output directory for gate receipt and matrices")
    sentinel.set_defaults(func=command_kfacet_sentinel)

    audit = sub.add_parser(
        "kfacet-bridge-audit",
        help=(
            "v0.3h bridge audit: no-integration probe asking whether a "
            "bridge singular value of (M_i - I) is a missed neutral "
            "direction, a Jordan-block root, or a defective D3 standard block."
        ),
    )
    audit.add_argument(
        "--input-dir",
        required=True,
        help="adaptive-floor reprocessor output dir (or sentinel output dir).",
    )
    audit.add_argument(
        "--rows",
        help="comma-separated row indices to audit (default: failed_rows from manifest.json)",
    )
    audit.add_argument("--source", choices=sorted(SUPPLEMENT_URLS), default="A")
    audit.add_argument(
        "--path",
        help="local supplementary file instead of the live URL",
    )
    audit.add_argument("--m3", type=float, default=1.0)
    audit.add_argument(
        "--out",
        help="optional output directory (default: {input_dir}-bridge-audit)",
    )
    audit.set_defaults(func=command_kfacet_bridge_audit)

    reprocess = sub.add_parser(
        "kfacet-reprocess-floor",
        help=(
            "v0.3h adaptive-floor reprocessor: no-integration kernel/D3 leakage "
            "sweep over existing sentinel receipts (M_i.npy + D3_*.npy)."
        ),
    )
    reprocess.add_argument(
        "--input-dir",
        required=True,
        help="directory containing O_*/ subdirectories with M_i.npy, D3_*.npy, gate_receipt.json",
    )
    reprocess.add_argument(
        "--indices",
        help="optional comma-separated row indices to restrict the sweep (default: all O_x dirs)",
    )
    reprocess.add_argument(
        "--out",
        help="optional output directory (default: {input_dir}-adaptive-floor)",
    )
    reprocess.add_argument(
        "--allow-failed-rows",
        action="store_true",
        help="exit 0 while still recording failed rows in the manifest (used for bridge-case audits)",
    )
    reprocess.set_defaults(func=command_kfacet_reprocess_floor)

    z2_sweep = sub.add_parser(
        "kfacet-row-z2-sweep",
        help=(
            "v0.4b per-row Z_2 isotypic-dim extraction + threshold-rule "
            "prediction. Computes M_i, K_fib, F_beta-even/odd dims, and "
            "the locked gamma_3 threshold-rule prediction (predict S iff "
            "F_beta_even_dim >= F_beta_odd_dim). See "
            "docs/isotrophy/kfacet/kfacet_v04b_gamma3_form.md for the form lock."
        ),
    )
    z2_sweep.add_argument("--source", choices=sorted(SUPPLEMENT_URLS), default="B")
    z2_sweep.add_argument("--path", help="local supplementary file instead of the live URL")
    z2_sweep.add_argument("--m3", type=float, required=True)
    z2_sweep.add_argument("--indices", help="optional comma-separated row indices")
    z2_sweep.add_argument("--limit", type=int, default=0)
    z2_sweep.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    z2_sweep.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    z2_sweep.add_argument("--max-step-fraction", type=float, default=DEFAULT_MAX_STEP_FRACTION)
    z2_sweep.add_argument("--closure-floor", type=float, default=V04B_DEFAULT_CLOSURE_FLOOR)
    z2_sweep.add_argument("--projector-floor", type=float, default=V04B_PROJECTOR_FLOOR)
    z2_sweep.add_argument("--out", help="output directory for per-row receipts and M_i.npy")
    z2_sweep.set_defaults(func=command_kfacet_row_z2_sweep)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
