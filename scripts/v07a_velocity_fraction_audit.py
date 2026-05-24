"""v0.7a velocity-fraction audit -- full 273-row supp-B audit.

Implements the form locked in
`internal/anniversary/kfacet_v07a_velocity_fraction_audit_form.md`
with the R1 sanity-gate amendment (symplecticity_gate = 1e-4).

Pipeline per row (catalog-only feature extraction; new compute is
variational integration only):

  1. Parse supp-B catalog ICs.
  2. Integrate the three-body orbit over one period (DOP853, rtol = atol
     = 1e-12, max_step_fraction = 0.02).
  3. Compute M_i = Dphi_T(y0) via vectorized 324-dim variational
     integration (matrix-form dY/dt = J(t) Y, Y(0) = I_18). Single
     solve_ivp call; step adapter shared across all 18 columns.
  4. Sanity gates: symplecticity (M_i^T omega M_i = omega; tol 1e-4
     post-R1 amendment), reciprocal-pair (Floquet pairing; tol 1e-4).
     BLOCK runner if any row fails.
  5. Extract gamma_1 via locked rule: largest-real-part eigenvalue's
     eigenvector with tie-break cascade (max velocity-projection in
     degenerate eigenspace, then smallest |Im(lambda)|, then smallest
     positive argument, then lex sign).
  6. Compute velocity-fraction vf = ||delta_v||^2 /
     (||delta_q||^2 + ||delta_v||^2) under center-of-mass reduction
     with mass-weighted norm.
  7. Compute D1+A sidecar scalar: z-fraction of gamma_1 (mass-weighted
     z-component magnitude over total mass-weighted norm).

Aggregation (after all 273 rows):
  - Constant-feature retirement: if sd(vf) < 0.01, retire.
  - Quartile cutpoints over supp-B distribution.
  - 4 x 2 contingency vs S/U.
  - chi-squared with sparse-cell fallback (permutation seed 20260523,
    n_permutations 10000).
  - Alignment-tightness scalar vs v0.5a branch_label
    (thresholds 0.8 warning, 0.95 severe).
  - Same shape for D1+A sidecar (z-fraction); REPORT-ONLY verdict.

Receipt at:
  results/isotrophy/k-facet-v07a-velocity-fraction-audit/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.isotrophy_workbench import (  # type: ignore
    acceleration_jacobian,
    canonical_omega_18,
    integrate_orbit,
    parse_rows,
    read_text,
)

DEFAULT_CATALOG = ROOT / "docs/isotrophy/supplementary-B_piano-init-condit-3d.txt"
DEFAULT_V05A_PER_ROW = ROOT / "results/isotrophy/k-facet-v05a-branch-map/per_row_table.csv"
DEFAULT_OUT = ROOT / "results/isotrophy/k-facet-v07a-velocity-fraction-audit"

FORM_LOCK = "internal/anniversary/kfacet_v07a_velocity_fraction_audit_form.md"
VERSION = "v0.7a-velocity-fraction-audit-r1"

# Locked precision (v0.7a form lock).
RTOL = 1e-12
ATOL = 1e-12
MAX_STEP_FRACTION = 0.02

# Sanity gates (v0.7a form lock + R1 amendment 2026-05-24).
SYMPLECTICITY_GATE = 1e-4  # R1-amended (was 1e-6).
RECIPROCAL_PAIR_GATE = 1e-4

# Locked degeneracy detection threshold (v0.7a form lock).
DEGENERACY_THRESHOLD_REAL_PART = 1e-6

# Locked statistical thresholds (v0.7a form lock).
CHI2_CRITICAL_AT_P01 = {1: 6.63, 2: 9.21, 3: 11.34}
P_VALUE_THRESHOLD = 0.01
ALIGNMENT_WARNING_THRESHOLD = 0.8
ALIGNMENT_SEVERE_THRESHOLD = 0.95
ASYMPTOTIC_EXPECTED_FLOOR = 5
MIN_OCCUPIED_BIN_COUNT_FOR_TEST = 2
PERMUTATION_SEED = 20260523
N_PERMUTATIONS = 10000

# Locked constant-feature retirement threshold.
CONSTANT_FEATURE_SD_THRESHOLD = 0.01

# Locked quartile method.
QUANTILE_METHOD = "linear"


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"expected boolean text, got {value!r}")


def load_v05a_branch_labels(path: Path) -> dict[tuple[str, int], str]:
    """Load (m3_key, index) -> branch_label from the v0.5a per_row_table."""
    rows: dict[tuple[str, int], str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            key = (f"{float(raw['m3']):.1f}", int(raw["index"]))
            rows[key] = raw["branch_label"]
    return rows


def compute_monodromy_vectorized(integrated, rtol: float, atol: float, max_step_fraction: float) -> np.ndarray:
    """Compute M_i = Dphi_T(y0) via a single 324-dim solve_ivp call.

    Matrix-form variational equation dY/dt = J(t) Y, Y(0) = I_18.
    The DOP853 step adapter sees all 324 state components jointly, so
    the step sequence is shared across all 18 columns instead of being
    computed 18 times independently. Numerically equivalent to the
    column-by-column compute_monodromy() under per-element error
    control, but ~10-20x faster at tight tolerances.

    Neutral helper: uses only acceleration_jacobian and the orbit's
    dense output. Does NOT touch K_fib, F_beta, D3/Z2 isotypic split,
    or stability-derived filtering.
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


def _apply_lex_sign(vec: np.ndarray) -> np.ndarray:
    """Cascade step (d): first non-zero entry positive."""
    for entry in vec:
        if abs(entry) > 1e-14:
            if entry < 0:
                return -vec
            return vec
    return vec  # all-zero vector; sign is moot


def select_gamma_1(M_i: np.ndarray, masses: np.ndarray) -> dict:
    """Select gamma_1 per the v0.7a locked rule with tie-break cascade."""
    eigvals, eigvecs = np.linalg.eig(M_i)
    real_parts = eigvals.real
    max_real = real_parts.max()

    degenerate_mask = (real_parts >= max_real - DEGENERACY_THRESHOLD_REAL_PART)
    degenerate_indices = np.where(degenerate_mask)[0]
    degenerate_count = len(degenerate_indices)

    raw_columns: list[np.ndarray] = []
    for idx in degenerate_indices:
        vec = eigvecs[:, idx]
        if np.allclose(vec.imag, 0.0, atol=1e-10):
            raw_columns.append(vec.real)
        else:
            raw_columns.append(vec.real)
            raw_columns.append(vec.imag)
    raw_matrix = np.column_stack(raw_columns) if raw_columns else np.zeros((18, 0))
    u_full, singular_values, _vt = np.linalg.svd(raw_matrix, full_matrices=False)
    rank = int(np.sum(singular_values > 1e-10))
    basis = u_full[:, :rank]

    if degenerate_count == 1 and basis.shape[1] == 1:
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
            "cascade_step_used": "step1_unique",
        }

    # Cascade step (a): within eigenspace, max velocity-subspace projection
    # under mass-weighted norm. Mass weights repeat each mass 3 times.
    mass_weights_9 = np.repeat(masses, 3)
    velocity_projection_matrix = basis[9:, :] * np.sqrt(mass_weights_9)[:, None]
    _u_proj, s_proj, vt_proj = np.linalg.svd(velocity_projection_matrix, full_matrices=False)
    if s_proj.size == 0 or s_proj[0] < 1e-14:
        gamma_1_raw = basis[:, 0]
        cascade_step_used = "cascade_step_a_degenerate"
    else:
        gamma_1_raw = basis @ vt_proj[0, :]
        cascade_step_used = "cascade_step_a"

    gamma_1 = _apply_lex_sign(gamma_1_raw)

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


def velocity_fraction_and_z_fraction(gamma_1: np.ndarray, masses: np.ndarray) -> dict:
    """Compute vf and z_fraction under CoM reduction + mass-weighted norm.

    vf = ||delta_v||^2 / (||delta_q||^2 + ||delta_v||^2)
    z_fraction = (mass-weighted z-component squared) / total mass-weighted norm
                  -- the D1+A sidecar feature (report-only).
    """
    delta_x = gamma_1[:9].reshape(3, 3)
    delta_v = gamma_1[9:].reshape(3, 3)
    com_x = np.average(delta_x, axis=0, weights=masses)
    com_v = np.average(delta_v, axis=0, weights=masses)
    delta_x_com = delta_x - com_x
    delta_v_com = delta_v - com_v
    norm_q_sq = float(np.sum(masses[:, None] * delta_x_com * delta_x_com))
    norm_v_sq = float(np.sum(masses[:, None] * delta_v_com * delta_v_com))
    total = norm_q_sq + norm_v_sq
    vf = norm_v_sq / total if total > 0 else float("nan")

    # D1+A sidecar: z-component magnitude (mass-weighted) over total norm.
    # z-component is axis index 2 in the per-body 3-vector.
    norm_z_q_sq = float(np.sum(masses * delta_x_com[:, 2] * delta_x_com[:, 2]))
    norm_z_v_sq = float(np.sum(masses * delta_v_com[:, 2] * delta_v_com[:, 2]))
    z_fraction = (norm_z_q_sq + norm_z_v_sq) / total if total > 0 else float("nan")

    return {
        "velocity_fraction": vf,
        "z_fraction": z_fraction,
        "norm_q_sq": norm_q_sq,
        "norm_v_sq": norm_v_sq,
        "norm_z_q_sq": norm_z_q_sq,
        "norm_z_v_sq": norm_z_v_sq,
        "com_position_offset": float(np.linalg.norm(com_x)),
        "com_velocity_offset": float(np.linalg.norm(com_v)),
    }


def symplecticity_residual(M_i: np.ndarray, omega: np.ndarray) -> float:
    residual = M_i.T @ omega @ M_i - omega
    return float(np.max(np.abs(residual)))


def reciprocal_pair_residual(eigvals: np.ndarray) -> float:
    n = len(eigvals)
    max_residual = 0.0
    for i, lam in enumerate(eigvals):
        if abs(lam) < 1e-12:
            return float("inf")
        reciprocal = 1.0 / lam
        candidates = np.abs(eigvals - reciprocal)
        candidates[i] = np.inf
        if abs(lam * lam - 1.0) < 1e-10:
            candidates[i] = 0.0
        best = candidates.min()
        max_residual = max(max_residual, best)
    return float(max_residual)


def chi_square_survival(x: float, df: int) -> float:
    if x < 0:
        return 1.0
    z = x / 2.0
    if df == 1:
        return math.erfc(math.sqrt(z))
    if df == 2:
        return math.exp(-z)
    if df == 3:
        return math.erfc(math.sqrt(z)) + (2.0 * math.sqrt(z) * math.exp(-z) / math.sqrt(math.pi))
    raise NotImplementedError(f"chi-squared survival for df={df} not implemented")


def quantile_cutpoints(values: list[float]) -> dict:
    q25, q50, q75 = np.quantile(np.asarray(values, dtype=float), [0.25, 0.50, 0.75], method=QUANTILE_METHOD)
    return {"q25": float(q25), "q50": float(q50), "q75": float(q75)}


def assign_quartile(value: float, cutpoints: dict) -> int:
    if value <= cutpoints["q25"]:
        return 1
    if value <= cutpoints["q50"]:
        return 2
    if value <= cutpoints["q75"]:
        return 3
    return 4


def contingency_for(rows: list[dict], quartile_key: str) -> list[dict]:
    table = []
    for q in (1, 2, 3, 4):
        selected = [row for row in rows if int(row[quartile_key]) == q]
        counts = Counter(row["stability"] for row in selected)
        n = len(selected)
        s = counts.get("S", 0)
        u = counts.get("U", 0)
        table.append({
            "quartile": q,
            "N": n,
            "S": s,
            "U": u,
            "S_fraction": (s / n) if n else 0.0,
            "occupied": n > 0,
        })
    return table


def chi_squared_with_expected(table: list[dict]) -> tuple[float, list[dict], float]:
    occupied = [row for row in table if row["occupied"]]
    total = sum(row["N"] for row in occupied)
    total_s = sum(row["S"] for row in occupied)
    total_u = sum(row["U"] for row in occupied)
    chi2 = 0.0
    min_expected = math.inf
    enriched = []
    for row in table:
        if not row["occupied"]:
            enriched.append({**row, "expected_S": 0.0, "expected_U": 0.0, "chi2_contribution": 0.0})
            continue
        expected_s = row["N"] * total_s / total if total else 0.0
        expected_u = row["N"] * total_u / total if total else 0.0
        contrib_s = ((row["S"] - expected_s) ** 2 / expected_s) if expected_s else 0.0
        contrib_u = ((row["U"] - expected_u) ** 2 / expected_u) if expected_u else 0.0
        contribution = contrib_s + contrib_u
        chi2 += contribution
        min_expected = min(min_expected, expected_s, expected_u)
        enriched.append({
            **row,
            "expected_S": expected_s,
            "expected_U": expected_u,
            "chi2_contribution": contribution,
        })
    return chi2, enriched, min_expected


def chi_squared_from_arrays(quartiles: np.ndarray, stability_is_s: np.ndarray) -> float:
    occupied_qs = np.unique(quartiles)
    total = len(quartiles)
    if total == 0:
        return 0.0
    total_s = int(stability_is_s.sum())
    total_u = total - total_s
    chi2 = 0.0
    for q in occupied_qs:
        mask = quartiles == q
        n = int(mask.sum())
        if n == 0:
            continue
        s = int(stability_is_s[mask].sum())
        u = n - s
        expected_s = n * total_s / total
        expected_u = n * total_u / total
        if expected_s:
            chi2 += (s - expected_s) ** 2 / expected_s
        if expected_u:
            chi2 += (u - expected_u) ** 2 / expected_u
    return chi2


def permutation_p_value(quartiles, stability_is_s, observed_chi2, seed, n_permutations):
    rng = np.random.default_rng(seed)
    extreme = 0
    labels = stability_is_s.copy()
    for _ in range(n_permutations):
        rng.shuffle(labels)
        stat = chi_squared_from_arrays(quartiles, labels)
        if stat >= observed_chi2:
            extreme += 1
    return extreme / n_permutations, extreme


def alignment_tightness(rows: list[dict], quartile_key: str) -> tuple[float, list[dict]]:
    records = []
    max_fraction = 0.0
    for q in (1, 2, 3, 4):
        selected = [row for row in rows if int(row[quartile_key]) == q]
        counts = Counter(row["branch_label"] for row in selected)
        dominant_branch, dominant_count = ("", 0)
        if counts:
            dominant_branch, dominant_count = max(counts.items(), key=lambda item: (item[1], item[0]))
        fraction = dominant_count / len(selected) if selected else 0.0
        max_fraction = max(max_fraction, fraction)
        records.append({
            "quartile": q,
            "N": len(selected),
            "dominant_branch_label": dominant_branch,
            "dominant_branch_count": dominant_count,
            "dominant_branch_fraction": fraction,
            "branch_counts": dict(sorted(counts.items())),
        })
    return max_fraction, records


def sparse_cell_fallback(table, observed_chi2, occupied_count, min_expected, rows, quartile_key):
    occupied_bins = [row for row in table if row["occupied"]]
    min_bin_count = min((row["N"] for row in occupied_bins), default=0)

    test_branch_taken: str
    p_value = None
    permutation_extreme = None
    critical_value = None
    df = None
    verdict_decision = None

    if min_bin_count < MIN_OCCUPIED_BIN_COUNT_FOR_TEST:
        test_branch_taken = "inconclusive_sparse"
        verdict_decision = "inconclusive_sparse"
    elif min_expected < ASYMPTOTIC_EXPECTED_FLOOR:
        test_branch_taken = "permutation"
        quartiles = np.array([int(row[quartile_key]) for row in rows], dtype=int)
        stability_is_s = np.array([row["stability"] == "S" for row in rows], dtype=bool)
        p_value, permutation_extreme = permutation_p_value(
            quartiles=quartiles,
            stability_is_s=stability_is_s,
            observed_chi2=observed_chi2,
            seed=PERMUTATION_SEED,
            n_permutations=N_PERMUTATIONS,
        )
        verdict_decision = "pass" if p_value <= P_VALUE_THRESHOLD else "fail"
    else:
        test_branch_taken = "chi_squared"
        df = occupied_count - 1
        critical_value = CHI2_CRITICAL_AT_P01.get(df)
        if critical_value is None:
            raise NotImplementedError(f"chi-squared critical for df={df} not registered")
        p_value = chi_square_survival(observed_chi2, df)
        verdict_decision = "pass" if observed_chi2 > critical_value else "fail"

    return {
        "test_branch_taken": test_branch_taken,
        "min_occupied_bin_count": min_bin_count,
        "min_expected_cell": min_expected,
        "df": df,
        "critical": critical_value,
        "p_value": p_value,
        "permutation_seed": PERMUTATION_SEED if test_branch_taken == "permutation" else None,
        "n_permutations": N_PERMUTATIONS if test_branch_taken == "permutation" else None,
        "permutation_extreme_count": permutation_extreme,
        "verdict_decision": verdict_decision,
    }


def feature_audit(rows, quartile_key, branch_key="branch_label"):
    table = contingency_for(rows, quartile_key)
    occupied_bins = [row for row in table if row["occupied"]]
    occupied_count = len(occupied_bins)
    chi2, enriched, min_expected = chi_squared_with_expected(table)
    fallback = sparse_cell_fallback(
        table=enriched,
        observed_chi2=chi2,
        occupied_count=occupied_count,
        min_expected=min_expected,
        rows=rows,
        quartile_key=quartile_key,
    )
    alignment, alignment_records = alignment_tightness(rows, quartile_key) if branch_key else (None, [])
    return {
        "contingency": enriched,
        "chi_squared": chi2,
        "occupied_bin_count": occupied_count,
        **fallback,
        "alignment_tightness_scalar": alignment,
        "alignment_records": alignment_records,
    }


def per_row_pipeline(row, omega) -> dict:
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
    feat = velocity_fraction_and_z_fraction(gamma_1_info["gamma_1"], masses)

    total_seconds = time.perf_counter() - started

    return {
        "label": row.label,
        "index": int(row.index),
        "m3": float(row.m3),
        "m3_key": f"{row.m3:.1f}",
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
        "velocity_fraction": feat["velocity_fraction"],
        "z_fraction": feat["z_fraction"],
        "norm_q_sq": feat["norm_q_sq"],
        "norm_v_sq": feat["norm_v_sq"],
        "norm_z_q_sq": feat["norm_z_q_sq"],
        "norm_z_v_sq": feat["norm_z_v_sq"],
        "com_position_offset": feat["com_position_offset"],
        "com_velocity_offset": feat["com_velocity_offset"],
    }


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_dynamic(path, rows, first_field):
    fields = [first_field, "N"]
    extra = sorted({key for row in rows for key in row if key not in fields})
    fields.extend(extra)
    write_csv(path, rows, fields)


def build_cross_table(rows, row_key, col_key):
    row_values = sorted({row[row_key] for row in rows})
    col_values = sorted({str(row[col_key]) for row in rows})
    table = []
    for rv in row_values:
        selected = [row for row in rows if row[row_key] == rv]
        record = {row_key: rv, "N": len(selected)}
        for cv in col_values:
            record[cv] = sum(1 for row in selected if str(row[col_key]) == cv)
        table.append(record)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--v05a-per-row", type=Path, default=DEFAULT_V05A_PER_ROW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    catalog_rows = parse_rows(read_text(str(args.catalog)), source="B")
    branch_labels = load_v05a_branch_labels(args.v05a_per_row)

    per_row_records: list[dict] = []
    sanity_failures: list[dict] = []
    started_total = time.perf_counter()
    for idx_row, row in enumerate(catalog_rows, start=1):
        omega = canonical_omega_18(row.masses)
        record = per_row_pipeline(row, omega)
        # Attach v0.5a branch_label.
        key = (record["m3_key"], record["index"])
        record["branch_label"] = branch_labels.get(key, "UNKNOWN")
        per_row_records.append(record)
        if record["symplecticity_status"] == "fail" or record["reciprocal_pair_status"] == "fail":
            sanity_failures.append({
                "label": record["label"],
                "symp_status": record["symplecticity_status"],
                "symp_residual": record["symplecticity_residual"],
                "recip_status": record["reciprocal_pair_status"],
                "recip_residual": record["reciprocal_pair_residual"],
            })
        elapsed = time.perf_counter() - started_total
        print(
            f"[{idx_row:3d}/{len(catalog_rows)}] {record['label']:20s} "
            f"stab={record['stability']} "
            f"T={record['total_seconds']:6.2f}s "
            f"symp={record['symplecticity_residual']:.2e} "
            f"recip={record['reciprocal_pair_residual']:.2e} "
            f"vf={record['velocity_fraction']:.4f}  "
            f"z={record['z_fraction']:.4f}  "
            f"max_re={record['max_eigenvalue_real_part']:.3g}  "
            f"deg={record['degenerate_eigenvalue_count']}  "
            f"step={record['cascade_step_used']}  "
            f"(elapsed {elapsed:.0f}s)",
            flush=True,
        )

    total_runtime = time.perf_counter() - started_total
    sanity_blocked = bool(sanity_failures)

    # Constant-feature retirement check.
    vf_values = [r["velocity_fraction"] for r in per_row_records]
    vf_sd = float(np.std(vf_values, ddof=1))
    constant_feature_retired = vf_sd < CONSTANT_FEATURE_SD_THRESHOLD

    if sanity_blocked:
        verdict = "velocity_fraction_blocked_sanity"
    elif constant_feature_retired:
        verdict = "velocity_fraction_retired_near_constant"
    else:
        # Quartile binning + chi-squared + alignment for vf and z_fraction.
        vf_cutpoints = quantile_cutpoints(vf_values)
        z_values = [r["z_fraction"] for r in per_row_records]
        z_cutpoints = quantile_cutpoints(z_values)
        for r in per_row_records:
            r["Q_vf"] = assign_quartile(r["velocity_fraction"], vf_cutpoints)
            r["Q_z"] = assign_quartile(r["z_fraction"], z_cutpoints)

        primary_audit = feature_audit(per_row_records, "Q_vf")
        sidecar_audit = feature_audit(per_row_records, "Q_z")

        alignment_E = primary_audit["alignment_tightness_scalar"]
        if primary_audit["verdict_decision"] == "pass":
            if alignment_E is not None and alignment_E > ALIGNMENT_SEVERE_THRESHOLD:
                verdict = "velocity_fraction_passes_audit_severe_alignment"
            elif alignment_E is not None and alignment_E > ALIGNMENT_WARNING_THRESHOLD:
                verdict = "velocity_fraction_passes_audit_alignment_warning"
            else:
                verdict = "velocity_fraction_passes_audit"
        elif primary_audit["verdict_decision"] == "fail":
            verdict = "velocity_fraction_fails_audit"
        elif primary_audit["verdict_decision"] == "inconclusive_sparse":
            verdict = "velocity_fraction_inconclusive_sparse"
        else:
            verdict = "velocity_fraction_unknown"

    result: dict = {
        "mode": "v0.7a-velocity-fraction-audit",
        "version": VERSION,
        "form_lock": FORM_LOCK,
        "input_catalog": relpath(args.catalog),
        "input_v05a_per_row_table": relpath(args.v05a_per_row),
        "rtol": RTOL,
        "atol": ATOL,
        "max_step_fraction": MAX_STEP_FRACTION,
        "symplecticity_gate_locked": SYMPLECTICITY_GATE,
        "symplecticity_gate_r1_amendment_note": "Amended 2026-05-24 from 1e-6 to 1e-4 per the 7-row smoke evidence; see form-lock Amendment R1.",
        "reciprocal_pair_gate_locked": RECIPROCAL_PAIR_GATE,
        "degeneracy_threshold_real_part": DEGENERACY_THRESHOLD_REAL_PART,
        "chi2_critical_at_p01": CHI2_CRITICAL_AT_P01,
        "p_value_threshold": P_VALUE_THRESHOLD,
        "alignment_warning_threshold": ALIGNMENT_WARNING_THRESHOLD,
        "alignment_severe_threshold": ALIGNMENT_SEVERE_THRESHOLD,
        "asymptotic_expected_floor": ASYMPTOTIC_EXPECTED_FLOOR,
        "min_occupied_bin_count_for_test": MIN_OCCUPIED_BIN_COUNT_FOR_TEST,
        "permutation_seed_locked": PERMUTATION_SEED,
        "n_permutations_locked": N_PERMUTATIONS,
        "constant_feature_sd_threshold": CONSTANT_FEATURE_SD_THRESHOLD,
        "quantile_method": f"numpy_quantile_{QUANTILE_METHOD}",
        "row_count": len(per_row_records),
        "S_count": sum(1 for r in per_row_records if r["stability"] == "S"),
        "U_count": sum(1 for r in per_row_records if r["stability"] == "U"),
        "vf_sd": vf_sd,
        "constant_feature_retired": constant_feature_retired,
        "sanity_failures": sanity_failures,
        "sanity_blocked": sanity_blocked,
        "total_runtime_seconds": total_runtime,
        "verdict": verdict,
    }

    if not sanity_blocked and not constant_feature_retired:
        result["cutpoints_vf"] = vf_cutpoints
        result["cutpoints_z"] = z_cutpoints
        # Strip the gamma_1 vector reference (we don't carry it in per-row).
        result["primary_audit_vf"] = {k: v for k, v in primary_audit.items()}
        result["sidecar_audit_z"] = {k: v for k, v in sidecar_audit.items()}
        # Diagnostic cross-tables.
        result["diagnostic_Q_vf_x_branch"] = build_cross_table(per_row_records, "Q_vf", "branch_label")
        result["diagnostic_Q_vf_x_m3"] = build_cross_table(per_row_records, "Q_vf", "m3_key")
        result["diagnostic_Q_z_x_branch"] = build_cross_table(per_row_records, "Q_z", "branch_label")

    (args.out / "manifest.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")

    per_row_fields = [
        "label", "index", "m3", "m3_key", "z0", "period", "stability", "branch_label",
        "velocity_fraction", "z_fraction",
        "norm_q_sq", "norm_v_sq", "norm_z_q_sq", "norm_z_v_sq",
        "Q_vf", "Q_z",
        "max_eigenvalue_real_part",
        "representative_eigenvalue_real", "representative_eigenvalue_imag", "representative_eigenvalue_modulus",
        "degenerate_eigenvalue_count", "eigenspace_dim_used", "cascade_step_used",
        "symplecticity_residual", "symplecticity_status",
        "reciprocal_pair_residual", "reciprocal_pair_status",
        "orbit_integration_seconds", "monodromy_integration_seconds", "total_seconds",
        "com_position_offset", "com_velocity_offset",
    ]
    write_csv(args.out / "per_row_table.csv", per_row_records, per_row_fields)

    if not sanity_blocked and not constant_feature_retired:
        contingency_fields = [
            "quartile", "N", "S", "U", "S_fraction", "expected_S", "expected_U", "chi2_contribution", "occupied",
        ]
        write_csv(args.out / "contingency_table_vf.csv", primary_audit["contingency"], contingency_fields)
        write_csv(args.out / "contingency_table_z.csv", sidecar_audit["contingency"], contingency_fields)
        write_csv_dynamic(args.out / "velocity_fraction_by_branch.csv", result["diagnostic_Q_vf_x_branch"], "Q_vf")
        write_csv_dynamic(args.out / "velocity_fraction_by_m3.csv", result["diagnostic_Q_vf_x_m3"], "Q_vf")
        write_csv_dynamic(args.out / "z_fraction_by_branch.csv", result["diagnostic_Q_z_x_branch"], "Q_z")

    print()
    print(f"[v07a] verdict: {verdict}")
    print(f"[v07a] rows: {len(per_row_records)}  S={result['S_count']} U={result['U_count']}")
    print(f"[v07a] sanity_blocked={sanity_blocked}  constant_retired={constant_feature_retired}")
    print(f"[v07a] vf_sd={vf_sd:.4f}  threshold={CONSTANT_FEATURE_SD_THRESHOLD}")
    if not sanity_blocked and not constant_feature_retired:
        pa = primary_audit
        sa = sidecar_audit
        print(f"[v07a] primary vf: chi^2={pa['chi_squared']:.6f}  branch={pa['test_branch_taken']}  "
              f"p={pa['p_value']}  critical={pa['critical']}  "
              f"alignment={pa['alignment_tightness_scalar']:.4f}")
        print(f"[v07a] sidecar z:  chi^2={sa['chi_squared']:.6f}  branch={sa['test_branch_taken']}  "
              f"p={sa['p_value']}  critical={sa['critical']}  "
              f"alignment={sa['alignment_tightness_scalar']:.4f}  (report-only)")
    print(f"[v07a] total runtime: {total_runtime/60:.1f} min")
    print(f"[v07a] manifest: {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
