"""v0.7a' minimal-scoped restricted-scope audit.

Implements the form locked in
`internal/anniversary/kfacet_v07a_prime_restricted_scope_form.md`.

Reads the v0.7a per_row_table.csv, filters to the 250 analyzable rows
(integration_blocked == False), re-quartile-bins velocity_fraction and
z_fraction within the subset, computes the 4x2 chi-squared vs S/U with
the v0.6b sparse-cell fallback, computes alignment-tightness vs v0.5a
branch_label, and applies the locked Pass / Partial / Fail outcome tree.

No new variational compute. Runtime: seconds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_V07A_PER_ROW = ROOT / "results/isotrophy/k-facet-v07a-velocity-fraction-audit/per_row_table.csv"
DEFAULT_OUT = ROOT / "results/isotrophy/k-facet-v07a-prime-restricted-scope"

FORM_LOCK = "internal/anniversary/kfacet_v07a_prime_restricted_scope_form.md"
VERSION = "v0.7a-prime-restricted-scope-audit"

# Inherited from v0.7a + v0.6 lineage.
CHI2_CRITICAL_AT_P01 = {1: 6.63, 2: 9.21, 3: 11.34}
P_VALUE_THRESHOLD = 0.01
ALIGNMENT_WARNING_THRESHOLD = 0.8
ALIGNMENT_SEVERE_THRESHOLD = 0.95
ASYMPTOTIC_EXPECTED_FLOOR = 5
MIN_OCCUPIED_BIN_COUNT_FOR_TEST = 2
PERMUTATION_SEED = 20260523
N_PERMUTATIONS = 10000
QUANTILE_METHOD = "linear"

# Locked attrition disclosure thresholds (from v0.7a R2.A; permanent on
# this audit's receipt for transparency).
V07A_FULL_CATALOG_ROW_COUNT = 273


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


def load_v07a_per_row(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            blocked_text = raw.get("integration_blocked", "False")
            blocked = parse_bool(blocked_text) if blocked_text not in ("", None) else False
            row = {
                "label": raw["label"],
                "index": int(raw["index"]),
                "m3": float(raw["m3"]),
                "m3_key": raw["m3_key"],
                "z0": float(raw["z0"]),
                "period": float(raw["period"]),
                "stability": raw["stability"],
                "branch_label": raw["branch_label"],
                "integration_blocked": blocked,
                "symplecticity_status": raw.get("symplecticity_status", ""),
                "reciprocal_pair_status": raw.get("reciprocal_pair_status", ""),
            }
            for k in ("velocity_fraction", "z_fraction",
                      "symplecticity_residual", "reciprocal_pair_residual"):
                v = raw.get(k, "")
                row[k] = None if v in ("", "None", "null", "NaN", "nan") else float(v)
            rows.append(row)
    return rows


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
        selected = [row for row in rows if row.get(quartile_key) == q]
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
        selected = [row for row in rows if row.get(quartile_key) == q]
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


def feature_audit(rows: list[dict], quartile_key: str) -> dict:
    table = contingency_for(rows, quartile_key)
    occupied_bins = [row for row in table if row["occupied"]]
    occupied_count = len(occupied_bins)
    chi2, enriched, min_expected = chi_squared_with_expected(table)
    alignment, alignment_records = alignment_tightness(rows, quartile_key)
    min_bin_count = min((row["N"] for row in occupied_bins), default=0)

    test_branch_taken = None
    p_value = None
    critical = None
    df = None
    permutation_extreme = None

    if min_bin_count < MIN_OCCUPIED_BIN_COUNT_FOR_TEST:
        test_branch_taken = "inconclusive_sparse"
    elif min_expected < ASYMPTOTIC_EXPECTED_FLOOR:
        test_branch_taken = "permutation"
        quartiles = np.array([int(row[quartile_key]) for row in rows], dtype=int)
        stability_is_s = np.array([row["stability"] == "S" for row in rows], dtype=bool)
        p_value, permutation_extreme = permutation_p_value(
            quartiles=quartiles,
            stability_is_s=stability_is_s,
            observed_chi2=chi2,
            seed=PERMUTATION_SEED,
            n_permutations=N_PERMUTATIONS,
        )
    else:
        test_branch_taken = "chi_squared"
        df = occupied_count - 1
        critical = CHI2_CRITICAL_AT_P01.get(df)
        if critical is None:
            raise NotImplementedError(f"chi-squared critical for df={df} not registered")
        p_value = chi_square_survival(chi2, df)

    return {
        "contingency": enriched,
        "chi_squared": chi2,
        "occupied_bin_count": occupied_count,
        "min_occupied_bin_count": min_bin_count,
        "min_expected_cell": min_expected,
        "test_branch_taken": test_branch_taken,
        "df": df,
        "critical": critical,
        "p_value": p_value,
        "permutation_seed": PERMUTATION_SEED if test_branch_taken == "permutation" else None,
        "n_permutations": N_PERMUTATIONS if test_branch_taken == "permutation" else None,
        "permutation_extreme_count": permutation_extreme,
        "alignment_tightness_scalar": alignment,
        "alignment_records": alignment_records,
    }


def verdict_for_primary(audit: dict) -> str:
    """Locked Pass / Partial / Fail outcome tree."""
    if audit["test_branch_taken"] == "inconclusive_sparse":
        return "velocity_fraction_restricted_inconclusive_sparse"
    chi2 = audit["chi_squared"]
    critical = audit["critical"]
    alignment = audit["alignment_tightness_scalar"]
    p_value = audit["p_value"]
    if audit["test_branch_taken"] == "permutation":
        passes = p_value is not None and p_value <= P_VALUE_THRESHOLD
    else:
        passes = critical is not None and chi2 > critical
    if not passes:
        return "velocity_fraction_restricted_fails_audit"
    if alignment is None:
        return "velocity_fraction_restricted_passes_audit"
    if alignment <= ALIGNMENT_WARNING_THRESHOLD:
        return "velocity_fraction_restricted_passes_audit"
    if alignment <= ALIGNMENT_SEVERE_THRESHOLD:
        return "velocity_fraction_restricted_passes_audit_alignment_warning"
    return "velocity_fraction_restricted_passes_audit_severe_alignment"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v07a-per-row", type=Path, default=DEFAULT_V07A_PER_ROW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows = load_v07a_per_row(args.v07a_per_row)
    v07a_total = len(all_rows)
    v07a_blocked_count = sum(1 for r in all_rows if r["integration_blocked"])
    v07a_sanity_failed_count = sum(
        1 for r in all_rows
        if not r["integration_blocked"]
        and (r["symplecticity_status"] == "fail" or r["reciprocal_pair_status"] == "fail")
    )

    analyzable = [r for r in all_rows if not r["integration_blocked"]]
    analyzable_S = sum(1 for r in analyzable if r["stability"] == "S")
    analyzable_U = sum(1 for r in analyzable if r["stability"] == "U")

    vf_values = [r["velocity_fraction"] for r in analyzable if r["velocity_fraction"] is not None]
    z_values = [r["z_fraction"] for r in analyzable if r["z_fraction"] is not None]

    vf_cutpoints = quantile_cutpoints(vf_values)
    z_cutpoints = quantile_cutpoints(z_values)
    for r in analyzable:
        r["Q_vf"] = assign_quartile(r["velocity_fraction"], vf_cutpoints)
        r["Q_z"] = assign_quartile(r["z_fraction"], z_cutpoints)

    primary_audit = feature_audit(analyzable, "Q_vf")
    sidecar_audit = feature_audit(analyzable, "Q_z")

    verdict = verdict_for_primary(primary_audit)
    sidecar_indicative_verdict = verdict_for_primary(sidecar_audit)  # report-only

    result = {
        "mode": "v0.7a-prime-restricted-scope-audit",
        "version": VERSION,
        "form_lock": FORM_LOCK,
        "input_v07a_per_row_table": relpath(args.v07a_per_row),
        "attrition_disclosure": {
            "v07a_full_catalog_row_count": V07A_FULL_CATALOG_ROW_COUNT,
            "v07a_total_in_per_row_table": v07a_total,
            "v07a_integration_blocked_count": v07a_blocked_count,
            "v07a_sanity_failed_count": v07a_sanity_failed_count,
            "v07a_attrition_fraction": v07a_blocked_count / V07A_FULL_CATALOG_ROW_COUNT,
            "v07a_total_data_integrity_fraction": (
                (v07a_blocked_count + v07a_sanity_failed_count) / V07A_FULL_CATALOG_ROW_COUNT
            ),
        },
        "restricted_scope_row_count": len(analyzable),
        "restricted_scope_S_count": analyzable_S,
        "restricted_scope_U_count": analyzable_U,
        "p_value_threshold": P_VALUE_THRESHOLD,
        "alignment_warning_threshold": ALIGNMENT_WARNING_THRESHOLD,
        "alignment_severe_threshold": ALIGNMENT_SEVERE_THRESHOLD,
        "asymptotic_expected_floor": ASYMPTOTIC_EXPECTED_FLOOR,
        "min_occupied_bin_count_for_test": MIN_OCCUPIED_BIN_COUNT_FOR_TEST,
        "permutation_seed_locked": PERMUTATION_SEED,
        "n_permutations_locked": N_PERMUTATIONS,
        "chi2_critical_at_p01": CHI2_CRITICAL_AT_P01,
        "quantile_method": f"numpy_quantile_{QUANTILE_METHOD}",
        "cutpoints_vf_restricted": vf_cutpoints,
        "cutpoints_z_restricted": z_cutpoints,
        "primary_audit_vf": primary_audit,
        "sidecar_audit_z": sidecar_audit,
        "sidecar_status": "report_only",
        "sidecar_indicative_verdict_if_promoted": sidecar_indicative_verdict,
        "verdict": verdict,
    }

    (args.out / "manifest.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")

    per_row_fields = [
        "label", "index", "m3", "m3_key", "z0", "period", "stability", "branch_label",
        "integration_blocked",
        "velocity_fraction", "z_fraction",
        "Q_vf", "Q_z",
        "symplecticity_residual", "symplecticity_status",
        "reciprocal_pair_residual", "reciprocal_pair_status",
    ]
    write_csv(args.out / "per_row_table.csv", analyzable, per_row_fields)

    contingency_fields = [
        "quartile", "N", "S", "U", "S_fraction", "expected_S", "expected_U", "chi2_contribution", "occupied",
    ]
    write_csv(args.out / "contingency_table_vf.csv", primary_audit["contingency"], contingency_fields)
    write_csv(args.out / "contingency_table_z.csv", sidecar_audit["contingency"], contingency_fields)

    print(f"[v07a'] verdict: {verdict}")
    print(f"[v07a'] restricted scope: {len(analyzable)} rows  ({analyzable_S} S / {analyzable_U} U)")
    print(f"[v07a'] attrition disclosure: blocked {v07a_blocked_count}/273 ({100*v07a_blocked_count/273:.1f}%); "
          f"+ sanity_failed {v07a_sanity_failed_count}; "
          f"total data-integrity {(v07a_blocked_count+v07a_sanity_failed_count)/273*100:.1f}%")
    pa = primary_audit
    print(f"[v07a'] primary vf: chi^2={pa['chi_squared']:.6f}  branch={pa['test_branch_taken']}  "
          f"df={pa['df']}  critical={pa['critical']}  p={pa['p_value']}  "
          f"alignment={pa['alignment_tightness_scalar']:.4f}")
    sa = sidecar_audit
    print(f"[v07a'] sidecar z:  chi^2={sa['chi_squared']:.6f}  branch={sa['test_branch_taken']}  "
          f"df={sa['df']}  critical={sa['critical']}  p={sa['p_value']}  "
          f"alignment={sa['alignment_tightness_scalar']:.4f}  (report-only; "
          f"indicative_verdict_if_promoted={sidecar_indicative_verdict})")
    print(f"[v07a'] manifest: {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
