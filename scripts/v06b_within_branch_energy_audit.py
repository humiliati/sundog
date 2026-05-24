"""v0.6b within-branch energy audit (alignment-breaking).

Implements the form locked in
`internal/anniversary/kfacet_v06b_within_branch_energy_audit_form.md`.

Stratum: v0.5a (m_3 < 1, z_0 < 0.3) branch only (113 rows). Q_E and Q_|L|
are read from the v0.6a receipt (no re-binning within branch). Tests
within-branch independence with a pre-registered sparse-cell fallback tree:

    if min_occupied_bin_count < 2:
        verdict = within_branch_energy_inconclusive_sparse  (no p-value)
    elif min(expected cell count) < 5:
        exact permutation test  (seed = 20260523, n_permutations = 10000)
    else:
        asymptotic chi-squared(occupied_bins - 1) at p = 0.01.

The Q_|L| audit under the same shape is REPORT-ONLY (no verdict claim).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_V06A_MANIFEST = ROOT / "results/isotrophy/k-facet-v06a-energy-quartile-audit/manifest.json"
DEFAULT_V06A_PER_ROW = ROOT / "results/isotrophy/k-facet-v06a-energy-quartile-audit/per_row_table.csv"
DEFAULT_OUT = ROOT / "results/isotrophy/k-facet-v06b-within-branch-energy"

FORM_LOCK = "internal/anniversary/kfacet_v06b_within_branch_energy_audit_form.md"
VERSION = "v0.6b-within-branch-energy-audit"

STRATUM_BRANCH_LABEL = "m3_lt_1__z0_lt_0p3"
EXPECTED_STRATUM_N = 113
EXPECTED_STRATUM_S = 63
EXPECTED_STRATUM_U = 50

# Locked thresholds.
P_VALUE_THRESHOLD = 0.01
ASYMPTOTIC_EXPECTED_FLOOR = 5
MIN_OCCUPIED_BIN_COUNT_FOR_TEST = 2

# Locked permutation parameters.
PERMUTATION_SEED = 20260523
N_PERMUTATIONS = 10000

# Locked chi-squared critical values at p = 0.01 for df 1, 2, 3.
CHI2_CRITICAL_AT_P01 = {1: 6.63, 2: 9.21, 3: 11.34}


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


def load_v06a_per_row(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append({
                "label": raw["label"],
                "index": int(raw["index"]),
                "m3": float(raw["m3"]),
                "m3_key": raw["m3_key"],
                "z0": float(raw["z0"]),
                "period": float(raw["period"]),
                "stability": raw["stability"],
                "energy": float(raw["energy"]),
                "angular_momentum_norm": float(raw["angular_momentum_norm"]),
                "Q_E": int(raw["Q_E"]),
                "Q_L": int(raw["Q_L"]),
                "branch_label": raw["branch_label"],
                "b1_m3_lt_1": parse_bool(raw["b1_m3_lt_1"]),
                "b2_z0_lt_0p3": parse_bool(raw["b2_z0_lt_0p3"]),
                "final_Z2_class": raw.get("final_Z2_class") or None,
                "v04a_provenance": raw.get("v04a_provenance") or None,
            })
    return rows


def chi_square_survival(x: float, df: int) -> float:
    """Upper-tail survival function of chi-squared(df) for df in {1, 2, 3}."""
    if x < 0:
        return 1.0
    z = x / 2.0
    if df == 1:
        return math.erfc(math.sqrt(z))
    if df == 2:
        return math.exp(-z)
    if df == 3:
        return math.erfc(math.sqrt(z)) + (2.0 * math.sqrt(z) * math.exp(-z) / math.sqrt(math.pi))
    raise NotImplementedError(f"chi-squared survival for df={df} not implemented in this runner")


def contingency_for(rows: list[dict], quartile_key: str) -> list[dict]:
    """4 x 2 contingency over Q in {1..4} and stability in {S, U}.

    Occupied bins (N > 0) are flagged. Unoccupied bins are kept in the
    receipt for completeness but excluded from the chi-squared and df.
    """
    table = []
    for quartile in (1, 2, 3, 4):
        selected = [row for row in rows if int(row[quartile_key]) == quartile]
        counts = Counter(row["stability"] for row in selected)
        n = len(selected)
        s = counts.get("S", 0)
        u = counts.get("U", 0)
        table.append({
            "quartile": quartile,
            "N": n,
            "S": s,
            "U": u,
            "S_fraction": (s / n) if n else 0.0,
            "occupied": n > 0,
        })
    return table


def chi_squared_with_expected(table: list[dict]) -> tuple[float, list[dict], float]:
    """Chi-squared statistic over occupied cells, with per-cell expected
    counts attached to the returned table. Also returns min expected
    count over occupied cells (or +inf if no occupied bins)."""
    occupied = [row for row in table if row["occupied"]]
    total = sum(row["N"] for row in occupied)
    total_s = sum(row["S"] for row in occupied)
    total_u = sum(row["U"] for row in occupied)
    chi2 = 0.0
    min_expected = math.inf
    enriched = []
    for row in table:
        if not row["occupied"]:
            enriched.append({
                **row,
                "expected_S": 0.0,
                "expected_U": 0.0,
                "chi2_contribution": 0.0,
            })
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
    """Compute the chi-squared statistic over occupied Q bins for paired
    (quartile, stability) arrays. Used inside the permutation loop."""
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


def permutation_p_value(
    quartiles: np.ndarray,
    stability_is_s: np.ndarray,
    observed_chi2: float,
    seed: int,
    n_permutations: int,
) -> tuple[float, int]:
    """One-sided exact permutation p-value: permute stability labels,
    recompute chi-squared, count permutations with stat >= observed."""
    rng = np.random.default_rng(seed)
    extreme = 0
    labels = stability_is_s.copy()
    for _ in range(n_permutations):
        rng.shuffle(labels)
        stat = chi_squared_from_arrays(quartiles, labels)
        if stat >= observed_chi2:
            extreme += 1
    return extreme / n_permutations, extreme


def sparse_cell_fallback(
    table: list[dict],
    observed_chi2: float,
    occupied_count: int,
    min_expected: float,
    rows: list[dict],
    quartile_key: str,
) -> dict:
    """Apply the pre-registered fallback tree."""
    occupied_bins = [row for row in table if row["occupied"]]
    min_bin_count = min((row["N"] for row in occupied_bins), default=0)

    test_branch_taken: str
    p_value: float | None = None
    permutation_extreme: int | None = None
    critical_value: float | None = None
    df: int | None = None
    verdict_decision: str | None = None

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


def stratum_audit(rows: list[dict], quartile_key: str) -> dict:
    """Compute the within-stratum chi-squared / fallback for a given Q."""
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
    return {
        "contingency": enriched,
        "chi_squared": chi2,
        "occupied_bin_count": occupied_count,
        **fallback,
    }


def build_per_other_branch_diagnostic(all_rows: list[dict]) -> dict:
    """Diagnostic per-other-branch audits using Q_E (and Q_L sidecar).

    Report-only. No verdict claim.
    """
    branches = sorted({row["branch_label"] for row in all_rows})
    per_branch: dict[str, dict] = {}
    for branch in branches:
        sub = [row for row in all_rows if row["branch_label"] == branch]
        if branch == STRATUM_BRANCH_LABEL:
            continue
        per_branch[branch] = {
            "N": len(sub),
            "S": sum(1 for row in sub if row["stability"] == "S"),
            "U": sum(1 for row in sub if row["stability"] == "U"),
            "Q_E_audit": stratum_audit(sub, "Q_E") if sub else None,
            "Q_L_audit": stratum_audit(sub, "Q_L") if sub else None,
        }
    return per_branch


def build_joint_table(rows: list[dict], row_key: str, col_key: str) -> list[dict]:
    """Joint cross-tab over (row_key, col_key)."""
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


def run_sanity(stratum_rows: list[dict]) -> dict:
    n = len(stratum_rows)
    s = sum(1 for row in stratum_rows if row["stability"] == "S")
    u = n - s
    pass_count = n == EXPECTED_STRATUM_N and s == EXPECTED_STRATUM_S and u == EXPECTED_STRATUM_U
    return {
        "expected_N": EXPECTED_STRATUM_N,
        "expected_S": EXPECTED_STRATUM_S,
        "expected_U": EXPECTED_STRATUM_U,
        "observed_N": n,
        "observed_S": s,
        "observed_U": u,
        "status": "pass" if pass_count else "fail",
    }


def verdict_for(decision: str, quartile_label: str) -> str:
    base = "within_branch_energy"  # primary verdict family is E-named even though the same code services |L|
    if quartile_label == "Q_E":
        prefix = "within_branch_energy"
    else:
        prefix = "within_branch_angular_momentum"
    if decision == "pass":
        return f"{prefix}_passes_audit"
    if decision == "fail":
        return f"{prefix}_fails_audit"
    if decision == "inconclusive_sparse":
        return f"{prefix}_inconclusive_sparse"
    raise ValueError(f"unknown decision {decision!r}")


def build_result(v06a_manifest_path: Path, v06a_per_row_path: Path) -> dict:
    all_rows = load_v06a_per_row(v06a_per_row_path)

    stratum_rows = [row for row in all_rows if row["branch_label"] == STRATUM_BRANCH_LABEL]
    sanity = run_sanity(stratum_rows)
    sanity_passed = sanity["status"] == "pass"

    primary_audit = stratum_audit(stratum_rows, "Q_E") if stratum_rows else None
    sidecar_audit = stratum_audit(stratum_rows, "Q_L") if stratum_rows else None

    verdict_primary: str
    if not sanity_passed:
        verdict_primary = "within_branch_energy_blocked_sanity"
    else:
        verdict_primary = verdict_for(primary_audit["verdict_decision"], "Q_E")

    sidecar_verdict_label = (
        verdict_for(sidecar_audit["verdict_decision"], "Q_L") if sanity_passed else None
    )

    per_other_branch = build_per_other_branch_diagnostic(all_rows)

    within_branch_Q_E_x_m3 = build_joint_table(stratum_rows, "Q_E", "m3_key")
    within_branch_Q_E_x_Q_L = build_joint_table(stratum_rows, "Q_E", "Q_L")

    cutpoints_E_locked = None
    cutpoints_L_locked = None
    if v06a_manifest_path.exists():
        v06a_manifest = json.loads(v06a_manifest_path.read_text(encoding="utf-8"))
        cutpoints_E_locked = v06a_manifest.get("cutpoints_E")
        cutpoints_L_locked = v06a_manifest.get("cutpoints_L")

    result = {
        "mode": "v0.6b-within-branch-energy-audit",
        "version": VERSION,
        "form_lock": FORM_LOCK,
        "input_v06a_manifest": relpath(v06a_manifest_path),
        "input_v06a_per_row_table": relpath(v06a_per_row_path),
        "stratum": "v0.5a (m_3 < 1, z_0 < 0.3) branch",
        "stratum_branch_label": STRATUM_BRANCH_LABEL,
        "stratum_row_count": len(stratum_rows),
        "stratum_S_count": sanity["observed_S"],
        "stratum_U_count": sanity["observed_U"],
        "sanity_check": sanity,
        "quartile_cutpoints_used_E": cutpoints_E_locked,
        "quartile_cutpoints_used_L": cutpoints_L_locked,
        "p_value_threshold": P_VALUE_THRESHOLD,
        "asymptotic_expected_floor": ASYMPTOTIC_EXPECTED_FLOOR,
        "min_occupied_bin_count_for_test": MIN_OCCUPIED_BIN_COUNT_FOR_TEST,
        "permutation_seed_locked": PERMUTATION_SEED,
        "n_permutations_locked": N_PERMUTATIONS,
        "chi2_critical_at_p01": CHI2_CRITICAL_AT_P01,
        "primary_audit_E": primary_audit,
        "verdict": verdict_primary,
        "sidecar_audit_L": sidecar_audit,
        "sidecar_status": "report_only",
        "sidecar_verdict_label_if_promoted": sidecar_verdict_label,
        "diagnostic_per_other_branch": per_other_branch,
        "diagnostic_within_branch_Q_E_x_m3": within_branch_Q_E_x_m3,
        "diagnostic_within_branch_Q_E_x_Q_L": within_branch_Q_E_x_Q_L,
        "_tables": {
            "stratum_per_row": stratum_rows,
            "contingency_E": primary_audit["contingency"] if primary_audit else [],
            "contingency_L": sidecar_audit["contingency"] if sidecar_audit else [],
            "Q_E_x_m3": within_branch_Q_E_x_m3,
            "Q_E_x_Q_L": within_branch_Q_E_x_Q_L,
        },
    }
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_dynamic(path: Path, rows: list[dict], first_field: str) -> None:
    fields = [first_field, "N"]
    extra = sorted({key for row in rows for key in row if key not in fields})
    fields.extend(extra)
    write_csv(path, rows, fields)


def write_receipts(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = result.pop("_tables")
    try:
        (out_dir / "manifest.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        write_csv(out_dir / "per_row_table.csv", tables["stratum_per_row"], [
            "label", "index", "m3", "m3_key", "z0", "period", "stability",
            "energy", "angular_momentum_norm",
            "Q_E", "Q_L", "branch_label",
            "final_Z2_class", "v04a_provenance",
        ])
        write_csv(out_dir / "contingency_table_within_branch_E.csv", tables["contingency_E"], [
            "quartile", "N", "S", "U", "S_fraction", "expected_S", "expected_U", "chi2_contribution", "occupied",
        ])
        write_csv(out_dir / "contingency_table_within_branch_L.csv", tables["contingency_L"], [
            "quartile", "N", "S", "U", "S_fraction", "expected_S", "expected_U", "chi2_contribution", "occupied",
        ])
        write_csv_dynamic(out_dir / "within_branch_Q_E_by_m3.csv", tables["Q_E_x_m3"], "Q_E")
        write_csv_dynamic(out_dir / "within_branch_Q_E_by_Q_L.csv", tables["Q_E_x_Q_L"], "Q_E")
    finally:
        result["_tables"] = tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v06a-manifest", type=Path, default=DEFAULT_V06A_MANIFEST)
    parser.add_argument("--v06a-per-row", type=Path, default=DEFAULT_V06A_PER_ROW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    result = build_result(args.v06a_manifest.resolve(), args.v06a_per_row.resolve())
    write_receipts(result, args.out)

    primary = result["primary_audit_E"]
    sidecar = result["sidecar_audit_L"]
    print("[v06b-within-branch-energy] verdict:", result["verdict"])
    print(f"  stratum rows: {result['stratum_row_count']}  S={result['stratum_S_count']} U={result['stratum_U_count']}")
    print(f"  sanity:       {result['sanity_check']['status']}")
    if primary:
        p_disp = "n/a" if primary["p_value"] is None else f"{primary['p_value']:.6g}"
        crit_disp = "n/a" if primary["critical"] is None else f"{primary['critical']:.6g}"
        df_disp = "n/a" if primary["df"] is None else str(primary["df"])
        print(
            f"  E primary:    chi^2={primary['chi_squared']:.6f}  "
            f"branch={primary['test_branch_taken']}  df={df_disp}  "
            f"crit={crit_disp}  p={p_disp}  "
            f"min_bin={primary['min_occupied_bin_count']}  min_exp={primary['min_expected_cell']:.3f}"
        )
    if sidecar:
        p_disp = "n/a" if sidecar["p_value"] is None else f"{sidecar['p_value']:.6g}"
        crit_disp = "n/a" if sidecar["critical"] is None else f"{sidecar['critical']:.6g}"
        df_disp = "n/a" if sidecar["df"] is None else str(sidecar["df"])
        print(
            f"  |L| sidecar:  chi^2={sidecar['chi_squared']:.6f}  "
            f"branch={sidecar['test_branch_taken']}  df={df_disp}  "
            f"crit={crit_disp}  p={p_disp}  "
            f"min_bin={sidecar['min_occupied_bin_count']}  min_exp={sidecar['min_expected_cell']:.3f}  (report-only)"
        )
    print(f"  manifest:     {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
