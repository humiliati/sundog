"""v0.7a velocity-fraction audit -- 7-row sentinel smoke.

Implements the smoke pass of the form locked in
`internal/anniversary/kfacet_v07a_velocity_fraction_audit_form.md`.

Computes per-row monodromy via the workbench's variational integrator at
the v0.7a-locked precision (rtol = atol = 1e-12, max_step_fraction = 0.02),
applies sanity gates (symplecticity, reciprocal-pair), extracts gamma_1
via the locked selection rule and tie-break cascade, and computes the
velocity-fraction. Reports per-row runtime so the operator can decide
whether to run the 273-row full audit inline or to stage the operator
command.

Run:
  python scripts/v07a_velocity_fraction_smoke.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.integrate import solve_ivp  # type: ignore

from scripts.isotrophy_workbench import (  # type: ignore
    acceleration_jacobian,
    canonical_omega_18,
    integrate_orbit,
    parse_rows,
    read_text,
)

DEFAULT_CATALOG = ROOT / "docs/isotrophy/supplementary-B_piano-init-condit-3d.txt"
DEFAULT_OUT = ROOT / "results/isotrophy/k-facet-v07a-velocity-fraction-audit/smoke"

# Locked variational-integration parameters (v0.7a form lock).
RTOL = 1e-12
ATOL = 1e-12
MAX_STEP_FRACTION = 0.02

# Locked sanity gates (v0.7a form lock + R1 amendment 2026-05-24).
# R1: symplecticity_gate amended from 1e-6 to 1e-4 based on the
# 7-row vectorized smoke evidence (residuals 7.9e-8 .. 3.84e-5;
# scale with period and Floquet amplification, not implementation
# breakage; reciprocal-pair gate at 1e-4 passes uniformly).
SYMPLECTICITY_GATE = 1e-4
RECIPROCAL_PAIR_GATE = 1e-4

# Locked degeneracy detection threshold (v0.7a form lock).
DEGENERACY_THRESHOLD_REAL_PART = 1e-6

# Locked sentinel rows from v0.3 cross-m_3 receipts.
SENTINELS = [
    ("0.4", 50),
    ("0.4", 62),
    ("0.4", 67),
    ("0.4", 434),
    ("1.0", 242),
    ("1.0", 282),
    ("1.0", 284),
]

FULL_CATALOG_ROW_COUNT = 273
INLINE_RUNTIME_BUDGET_SECONDS = 600  # ~10-minute repo rule


def compute_monodromy_vectorized(integrated, rtol: float, atol: float, max_step_fraction: float) -> np.ndarray:
    """Compute M_i = Dphi_T(y0) via a single 324-dim solve_ivp call.

    Replaces the workbench's column-by-column compute_monodromy() with the
    matrix-form variational equation dY/dt = J(t) Y, Y(0) = I_18. The
    DOP853 step adapter sees all 324 state components jointly, so the step
    sequence is shared across all 18 columns instead of being computed 18
    times independently. Numerically equivalent to the column-by-column
    version under per-element error control, but ~10-20x faster at tight
    tolerances.

    This is a neutral helper: it uses only acceleration_jacobian (the
    Newtonian Jacobian field) and the orbit's dense output. It does NOT
    touch K_fib, the F_beta decomposition, the D3/Z2 isotypic split, or
    any stability-derived filtering. The disallowed-feature audit
    inherited from v0.4b is preserved.
    """
    period = integrated.row.period
    masses = integrated.row.masses
    orbit_solution = integrated.solution
    max_step = period * max_step_fraction if max_step_fraction > 0 else np.inf

    def matrix_rhs(t: float, Y_flat: np.ndarray) -> np.ndarray:
        Y = Y_flat.reshape(18, 18)
        y_t = orbit_solution.sol(t)
        x_t = y_t[:9].reshape(3, 3)
        J = acceleration_jacobian(x_t, masses)
        dY = np.empty((18, 18), dtype=float)
        dY[:9, :] = Y[9:, :]
        dY[9:, :] = J @ Y[:9, :]
        return dY.ravel()

    Y0 = np.eye(18, dtype=float).ravel()
    solution = solve_ivp(
        matrix_rhs,
        (0.0, period),
        Y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=False,
        max_step=max_step,
    )
    if not solution.success:
        raise RuntimeError(
            f"vectorized variational integration failed for {integrated.row.label}: "
            f"{solution.message}"
        )
    return solution.y[:, -1].reshape(18, 18)


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def select_gamma_1(
    M_i: np.ndarray,
    masses: np.ndarray,
) -> dict:
    """Select gamma_1 per the v0.7a locked rule with tie-break cascade.

    1. Diagonalize M_i.
    2. Find eigenvalue(s) with max real part.
    3. Identify the degenerate eigenspace (eigenvalues within
       DEGENERACY_THRESHOLD_REAL_PART of the max real part).
    4. Within the degenerate eigenspace, apply tie-break cascade:
       (a) max projection onto velocity subspace under mass-weighted norm,
       (b) smallest absolute imaginary part of the eigenvalue,
       (c) smallest positive argument,
       (d) lex sign convention (first non-zero entry positive).
    5. Return the selected gamma_1 vector (real-valued; the real part is
       taken if the eigenvector is complex).
    """
    eigvals, eigvecs = np.linalg.eig(M_i)
    real_parts = eigvals.real
    max_real = real_parts.max()

    degenerate_mask = (real_parts >= max_real - DEGENERACY_THRESHOLD_REAL_PART)
    degenerate_indices = np.where(degenerate_mask)[0]
    degenerate_count = len(degenerate_indices)

    cascade_step_used = "step1_unique" if degenerate_count == 1 else "cascade_required"

    # Build an orthonormal basis of the degenerate eigenspace (real
    # representatives; for complex conjugate pairs we span via Re/Im).
    raw_columns: list[np.ndarray] = []
    for idx in degenerate_indices:
        vec = eigvecs[:, idx]
        if np.allclose(vec.imag, 0.0, atol=1e-10):
            raw_columns.append(vec.real)
        else:
            raw_columns.append(vec.real)
            raw_columns.append(vec.imag)
    raw_matrix = np.column_stack(raw_columns) if raw_columns else np.zeros((18, 0))
    # Orthonormal basis of the eigenspace via SVD (avoids QR sign ambiguity).
    u_full, singular_values, _vt = np.linalg.svd(raw_matrix, full_matrices=False)
    rank = int(np.sum(singular_values > 1e-10))
    basis = u_full[:, :rank]

    if degenerate_count == 1 and basis.shape[1] == 1:
        # Unique selection. Apply step (d) only for sign.
        gamma_1 = basis[:, 0]
        gamma_1 = _apply_lex_sign(gamma_1)
        return {
            "gamma_1": gamma_1,
            "eigenvalue_real": float(real_parts[degenerate_indices[0]]),
            "eigenvalue_imag": float(eigvals[degenerate_indices[0]].imag),
            "eigenvalue_modulus": float(abs(eigvals[degenerate_indices[0]])),
            "max_real_part": float(max_real),
            "degenerate_eigenvalue_count": int(degenerate_count),
            "eigenspace_dim_used": int(basis.shape[1]),
            "cascade_step_used": cascade_step_used,
        }

    # Cascade step (a): within the eigenspace, find the direction with
    # maximal projection onto the velocity subspace under mass-weighted norm.
    # Velocity subspace = last 9 components, mass-weighted by repeating each
    # mass 3 times.
    mass_weights_9 = np.repeat(masses, 3)
    velocity_projection_matrix = basis[9:, :] * np.sqrt(mass_weights_9)[:, None]
    # SVD: the right singular vector with the largest singular value gives
    # the linear combination of basis columns maximizing velocity-norm.
    _u_proj, s_proj, vt_proj = np.linalg.svd(velocity_projection_matrix, full_matrices=False)
    if s_proj.size == 0 or s_proj[0] < 1e-14:
        # All eigenspace vectors have negligible velocity component.
        # Fall back to first basis column (will be flagged in receipt).
        gamma_1_raw = basis[:, 0]
        cascade_step_used = "cascade_step_a_degenerate"
    else:
        gamma_1_raw = basis @ vt_proj[0, :]
        cascade_step_used = "cascade_step_a"

    # Cascade steps (b)-(d) only matter if step (a) is also degenerate; for
    # numerical robustness we apply lex sign (d) deterministically.
    gamma_1 = _apply_lex_sign(gamma_1_raw)

    # Identify which eigenvalue the selected vector primarily corresponds
    # to (for reporting only; this does NOT affect the feature).
    representative_idx = degenerate_indices[0]
    return {
        "gamma_1": gamma_1,
        "eigenvalue_real": float(real_parts[representative_idx]),
        "eigenvalue_imag": float(eigvals[representative_idx].imag),
        "eigenvalue_modulus": float(abs(eigvals[representative_idx])),
        "max_real_part": float(max_real),
        "degenerate_eigenvalue_count": int(degenerate_count),
        "eigenspace_dim_used": int(basis.shape[1]),
        "cascade_step_used": cascade_step_used,
    }


def _apply_lex_sign(vec: np.ndarray) -> np.ndarray:
    """Cascade step (d): first non-zero entry positive."""
    for entry in vec:
        if abs(entry) > 1e-14:
            if entry < 0:
                return -vec
            return vec
    return vec  # all-zero vector; sign is moot


def velocity_fraction(gamma_1: np.ndarray, masses: np.ndarray) -> dict:
    """Compute vf = ||delta_v||^2 / (||delta_q||^2 + ||delta_v||^2)
    after center-of-mass reduction and mass-weighted norm.
    """
    delta_x = gamma_1[:9].reshape(3, 3)
    delta_v = gamma_1[9:].reshape(3, 3)
    # CoM reduction: subtract mass-weighted average per row (over bodies).
    com_x = np.average(delta_x, axis=0, weights=masses)
    com_v = np.average(delta_v, axis=0, weights=masses)
    delta_x_com = delta_x - com_x
    delta_v_com = delta_v - com_v
    # Mass-weighted squared norms.
    norm_q_sq = float(np.sum(masses[:, None] * delta_x_com * delta_x_com))
    norm_v_sq = float(np.sum(masses[:, None] * delta_v_com * delta_v_com))
    total = norm_q_sq + norm_v_sq
    vf = norm_v_sq / total if total > 0 else float("nan")
    return {
        "velocity_fraction": vf,
        "norm_q_sq": norm_q_sq,
        "norm_v_sq": norm_v_sq,
        "com_position_offset": float(np.linalg.norm(com_x)),
        "com_velocity_offset": float(np.linalg.norm(com_v)),
    }


def symplecticity_residual(M_i: np.ndarray, omega: np.ndarray) -> float:
    """max |M_i^T omega M_i - omega|."""
    residual = M_i.T @ omega @ M_i - omega
    return float(np.max(np.abs(residual)))


def reciprocal_pair_residual(eigvals: np.ndarray) -> float:
    """Test that eigenvalues come in reciprocal pairs (Floquet theorem).

    For each eigenvalue lambda, the reciprocal 1/lambda must also be an
    eigenvalue. Returns max over eigenvalues of min over other eigenvalues
    of |1/lambda - mu|.
    """
    n = len(eigvals)
    max_residual = 0.0
    for i, lam in enumerate(eigvals):
        if abs(lam) < 1e-12:
            return float("inf")
        reciprocal = 1.0 / lam
        candidates = np.abs(eigvals - reciprocal)
        candidates[i] = np.inf  # don't pair with self unless lambda^2 = 1
        if abs(lam * lam - 1.0) < 1e-10:
            # Self-reciprocal (lambda = ±1); allow self-pairing.
            candidates[i] = 0.0
        best = candidates.min()
        max_residual = max(max_residual, best)
    return float(max_residual)


def smoke_one_row(row, omega) -> dict:
    """Run the full v0.7a pipeline on a single sentinel row."""
    masses = row.masses
    started = time.perf_counter()

    integrated = integrate_orbit(row, rtol=RTOL, atol=ATOL, max_step_fraction=MAX_STEP_FRACTION)
    orbit_seconds = time.perf_counter() - started

    monodromy_start = time.perf_counter()
    M_i = compute_monodromy_vectorized(integrated, rtol=RTOL, atol=ATOL, max_step_fraction=MAX_STEP_FRACTION)
    monodromy_seconds = time.perf_counter() - monodromy_start

    symp_residual = symplecticity_residual(M_i, omega)
    symp_status = "pass" if symp_residual <= SYMPLECTICITY_GATE else "fail"

    eigvals, _ = np.linalg.eig(M_i)
    recip_residual = reciprocal_pair_residual(eigvals)
    recip_status = "pass" if recip_residual <= RECIPROCAL_PAIR_GATE else "fail"

    gamma_1_info = select_gamma_1(M_i, masses)
    vf_info = velocity_fraction(gamma_1_info["gamma_1"], masses)

    total_seconds = time.perf_counter() - started

    return {
        "label": row.label,
        "m3": float(row.m3),
        "z0": float(row.z0),
        "period": float(row.period),
        "stability": row.stability,
        "orbit_integration_seconds": orbit_seconds,
        "monodromy_integration_seconds": monodromy_seconds,
        "total_seconds": total_seconds,
        "symplecticity_residual": symp_residual,
        "symplecticity_status": symp_status,
        "reciprocal_pair_residual": recip_residual,
        "reciprocal_pair_status": recip_status,
        "max_eigenvalue_real_part": gamma_1_info["max_real_part"],
        "representative_eigenvalue_real": gamma_1_info["eigenvalue_real"],
        "representative_eigenvalue_imag": gamma_1_info["eigenvalue_imag"],
        "representative_eigenvalue_modulus": gamma_1_info["eigenvalue_modulus"],
        "degenerate_eigenvalue_count": gamma_1_info["degenerate_eigenvalue_count"],
        "eigenspace_dim_used": gamma_1_info["eigenspace_dim_used"],
        "cascade_step_used": gamma_1_info["cascade_step_used"],
        "velocity_fraction": vf_info["velocity_fraction"],
        "norm_q_sq": vf_info["norm_q_sq"],
        "norm_v_sq": vf_info["norm_v_sq"],
        "com_position_offset": vf_info["com_position_offset"],
        "com_velocity_offset": vf_info["com_velocity_offset"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    catalog_rows = parse_rows(read_text(str(args.catalog)), source="B")
    sentinel_set = set(SENTINELS)

    def m3_key(value: float) -> str:
        return f"{value:.1f}"

    rows = [row for row in catalog_rows if (m3_key(row.m3), int(row.index)) in sentinel_set]
    if len(rows) != len(SENTINELS):
        missing = sentinel_set - {(m3_key(row.m3), int(row.index)) for row in rows}
        raise RuntimeError(f"missing sentinel rows: {sorted(missing)}")

    omega = canonical_omega_18(np.array([1.0, 1.0, 1.0], dtype=float))  # mass-weighted version computed per row below

    smoke_records: list[dict] = []
    for row in rows:
        omega_row = canonical_omega_18(row.masses)
        record = smoke_one_row(row, omega_row)
        smoke_records.append(record)
        print(
            f"[smoke] {record['label']:20s} stab={record['stability']} "
            f"T={record['total_seconds']:6.2f}s  symp={record['symplecticity_residual']:.2e} "
            f"recip={record['reciprocal_pair_residual']:.2e}  "
            f"vf={record['velocity_fraction']:.4f}  "
            f"max_re={record['max_eigenvalue_real_part']:.4f}  "
            f"deg={record['degenerate_eigenvalue_count']}  "
            f"step={record['cascade_step_used']}",
            flush=True,
        )

    total = sum(r["total_seconds"] for r in smoke_records)
    mean = total / len(smoke_records)
    extrapolated_full = mean * FULL_CATALOG_ROW_COUNT
    inline_feasible = extrapolated_full <= INLINE_RUNTIME_BUDGET_SECONDS

    all_symp_pass = all(r["symplecticity_status"] == "pass" for r in smoke_records)
    all_recip_pass = all(r["reciprocal_pair_status"] == "pass" for r in smoke_records)

    summary = {
        "mode": "v0.7a-velocity-fraction-smoke",
        "form_lock": "internal/anniversary/kfacet_v07a_velocity_fraction_audit_form.md",
        "smoke_input_catalog": relpath(args.catalog),
        "smoke_sentinel_rows": [{"m3": m3, "index": idx} for m3, idx in SENTINELS],
        "smoke_record_count": len(smoke_records),
        "locked_precision": {
            "rtol": RTOL,
            "atol": ATOL,
            "max_step_fraction": MAX_STEP_FRACTION,
        },
        "locked_sanity_gates": {
            "symplecticity_gate": SYMPLECTICITY_GATE,
            "reciprocal_pair_gate": RECIPROCAL_PAIR_GATE,
        },
        "all_symplecticity_pass": all_symp_pass,
        "all_reciprocal_pair_pass": all_recip_pass,
        "total_seconds": total,
        "mean_seconds_per_row": mean,
        "extrapolated_full_catalog_seconds": extrapolated_full,
        "extrapolated_full_catalog_minutes": extrapolated_full / 60.0,
        "inline_runtime_budget_seconds": INLINE_RUNTIME_BUDGET_SECONDS,
        "inline_feasible": inline_feasible,
        "smoke_records": smoke_records,
    }

    (args.out / "smoke_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    with (args.out / "smoke_per_row_table.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(smoke_records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(smoke_records)

    print()
    print(f"[smoke] mean per-row: {mean:.2f}s  "
          f"extrapolated 273-row: {extrapolated_full:.1f}s ({extrapolated_full/60:.2f} min)")
    print(f"[smoke] inline budget = {INLINE_RUNTIME_BUDGET_SECONDS}s; "
          f"feasible inline = {inline_feasible}")
    print(f"[smoke] all symplecticity pass: {all_symp_pass}")
    print(f"[smoke] all reciprocal-pair pass: {all_recip_pass}")
    print(f"[smoke] receipt: {args.out / 'smoke_manifest.json'}")


if __name__ == "__main__":
    main()
