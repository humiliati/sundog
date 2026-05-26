"""v0.9a signed Floquet vf three-zone audit.

Implements the form locked in
`docs/isotrophy/kfacet/kfacet_v09a_signed_vf_three_zone_form.md`.

Reads v0.7a per_row_table.csv, filters to the 250 analyzable rows
(integration_blocked == False), bins vf into three physical zones at
the locked cutpoints {0.25, 0.50}:

  positional-dominant:  vf in [0, 0.25)
  mixed:                vf in [0.25, 0.50)
  velocity-heavy:       vf in [0.50, 1]

Computes a 3 x 2 chi-squared (zone, S/U) vs S/U with the v0.6b
sparse-cell fallback + alignment-tightness scalar vs v0.5a
branch_label, and applies the locked verdict tree:

  Pass iff chi^2 > critical
       AND S_fraction(mixed) < S_fraction(positional)
       AND S_fraction(mixed) < S_fraction(velocity-heavy)

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
DEFAULT_OUT = ROOT / "results/isotrophy/k-facet-v09a-signed-vf-three-zone"

FORM_LOCK = "docs/isotrophy/kfacet/kfacet_v09a_signed_vf_three_zone_form.md"
VERSION = "v0.9a-signed-vf-three-zone-audit"

# Locked physical cutpoints.
CUTPOINT_LOWER = 0.25
CUTPOINT_UPPER = 0.50

# Locked thresholds (inherited from v0.6/v0.7/v0.8).
CHI2_CRITICAL_AT_P01 = {1: 6.63, 2: 9.21, 3: 11.34}
P_VALUE_THRESHOLD = 0.01
ALIGNMENT_WARNING_THRESHOLD = 0.8
ALIGNMENT_SEVERE_THRESHOLD = 0.95
ASYMPTOTIC_EXPECTED_FLOOR = 5
MIN_OCCUPIED_BIN_COUNT_FOR_TEST = 2
PERMUTATION_SEED = 20260523
N_PERMUTATIONS = 10000

ZONE_LABELS = ["positional-dominant", "mixed", "velocity-heavy"]


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


def assign_zone(vf: float) -> str:
    if vf < CUTPOINT_LOWER:
        return "positional-dominant"
    if vf < CUTPOINT_UPPER:
        return "mixed"
    return "velocity-heavy"


def load_v07a_per_row(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            blocked = parse_bool(raw.get("integration_blocked", "False")) if raw.get("integration_blocked") not in ("", None) else False
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
            }
            v = raw.get("velocity_fraction", "")
            row["velocity_fraction"] = None if v in ("", "None", "null") else float(v)
            rows.append(row)
    return rows


def zone_contingency(rows: list[dict]) -> list[dict]:
    table = []
    for zone in ZONE_LABELS:
        selected = [row for row in rows if row.get("zone") == zone]
        counts = Counter(row["stability"] for row in selected)
        n = len(selected)
        s = counts.get("S", 0)
        u = counts.get("U", 0)
        table.append({
            "zone": zone,
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
        enriched.append({**row, "expected_S": expected_s, "expected_U": expected_u, "chi2_contribution": contribution})
    return chi2, enriched, min_expected


def chi_squared_from_arrays(zones: np.ndarray, stability_is_s: np.ndarray) -> float:
    occupied_zones = np.unique(zones)
    total = len(zones)
    if total == 0:
        return 0.0
    total_s = int(stability_is_s.sum())
    total_u = total - total_s
    chi2 = 0.0
    for zone in occupied_zones:
        mask = zones == zone
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


def permutation_p_value(zones, stability_is_s, observed_chi2, seed, n_permutations):
    rng = np.random.default_rng(seed)
    extreme = 0
    labels = stability_is_s.copy()
    for _ in range(n_permutations):
        rng.shuffle(labels)
        stat = chi_squared_from_arrays(zones, labels)
        if stat >= observed_chi2:
            extreme += 1
    return extreme / n_permutations, extreme


def alignment_tightness(rows: list[dict]) -> tuple[float, list[dict]]:
    records = []
    max_fraction = 0.0
    for zone in ZONE_LABELS:
        selected = [row for row in rows if row.get("zone") == zone]
        counts = Counter(row["branch_label"] for row in selected)
        dominant_branch, dominant_count = ("", 0)
        if counts:
            dominant_branch, dominant_count = max(counts.items(), key=lambda item: (item[1], item[0]))
        fraction = dominant_count / len(selected) if selected else 0.0
        max_fraction = max(max_fraction, fraction)
        records.append({
            "zone": zone,
            "N": len(selected),
            "dominant_branch_label": dominant_branch,
            "dominant_branch_count": dominant_count,
            "dominant_branch_fraction": fraction,
            "branch_counts": dict(sorted(counts.items())),
        })
    return max_fraction, records


def feature_audit(rows: list[dict]) -> dict:
    table = zone_contingency(rows)
    occupied_bins = [row for row in table if row["occupied"]]
    occupied_count = len(occupied_bins)
    chi2, enriched, min_expected = chi_squared_with_expected(table)
    alignment, alignment_records = alignment_tightness(rows)
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
        # Encode zone -> integer for permutation
        zone_to_int = {z: i for i, z in enumerate(ZONE_LABELS)}
        zones = np.array([zone_to_int[row["zone"]] for row in rows], dtype=int)
        stability_is_s = np.array([row["stability"] == "S" for row in rows], dtype=bool)
        p_value, permutation_extreme = permutation_p_value(
            zones=zones,
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


def pattern_check(audit: dict) -> dict:
    """Hypothesis-specific verdict augmentation: mixed < both direction-dominant zones."""
    s_pos = next((row["S_fraction"] for row in audit["contingency"] if row["zone"] == "positional-dominant"), None)
    s_mix = next((row["S_fraction"] for row in audit["contingency"] if row["zone"] == "mixed"), None)
    s_vel = next((row["S_fraction"] for row in audit["contingency"] if row["zone"] == "velocity-heavy"), None)
    mixed_lt_positional = (s_mix is not None and s_pos is not None and s_mix < s_pos)
    mixed_lt_velocity = (s_mix is not None and s_vel is not None and s_mix < s_vel)
    u_shape_confirmed = mixed_lt_positional and mixed_lt_velocity
    return {
        "S_fraction_positional": s_pos,
        "S_fraction_mixed": s_mix,
        "S_fraction_velocity_heavy": s_vel,
        "mixed_lt_positional": mixed_lt_positional,
        "mixed_lt_velocity_heavy": mixed_lt_velocity,
        "u_shape_confirmed": u_shape_confirmed,
    }


def verdict_for(audit: dict, pattern: dict) -> str:
    """Locked verdict tree."""
    if audit["test_branch_taken"] == "inconclusive_sparse":
        return "signed_vf_three_zone_inconclusive_sparse"

    chi2 = audit["chi_squared"]
    critical = audit["critical"]
    p_value = audit["p_value"]
    alignment = audit["alignment_tightness_scalar"]
    if audit["test_branch_taken"] == "permutation":
        chi2_passes = p_value is not None and p_value <= P_VALUE_THRESHOLD
    else:
        chi2_passes = critical is not None and chi2 > critical

    if not chi2_passes:
        return "signed_vf_three_zone_fails_audit_chi2"
    if not pattern["u_shape_confirmed"]:
        return "signed_vf_three_zone_fails_audit_pattern"
    if alignment is None:
        return "signed_vf_three_zone_passes_audit"
    if alignment <= ALIGNMENT_WARNING_THRESHOLD:
        return "signed_vf_three_zone_passes_audit"
    if alignment <= ALIGNMENT_SEVERE_THRESHOLD:
        return "signed_vf_three_zone_passes_audit_alignment_warning"
    return "signed_vf_three_zone_passes_audit_severe_alignment"


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
    blocked_count = sum(1 for r in all_rows if r["integration_blocked"])

    analyzable = [r for r in all_rows if not r["integration_blocked"]]
    for r in analyzable:
        if r["velocity_fraction"] is None or not (0.0 <= r["velocity_fraction"] <= 1.0):
            raise RuntimeError(f"out-of-range vf for {r['label']}: {r['velocity_fraction']}")
        r["zone"] = assign_zone(r["velocity_fraction"])

    audit = feature_audit(analyzable)
    pattern = pattern_check(audit)
    verdict = verdict_for(audit, pattern)

    s_count = sum(1 for r in analyzable if r["stability"] == "S")
    u_count = len(analyzable) - s_count

    result = {
        "mode": "v0.9a-signed-vf-three-zone-audit",
        "version": VERSION,
        "form_lock": FORM_LOCK,
        "input_v07a_per_row_table": relpath(args.v07a_per_row),
        "domain_row_count": len(analyzable),
        "domain_S_count": s_count,
        "domain_U_count": u_count,
        "attrition_disclosure": {
            "v07a_full_catalog_row_count": v07a_total,
            "v07a_integration_blocked_count": blocked_count,
        },
        "zone_cutpoints": [CUTPOINT_LOWER, CUTPOINT_UPPER],
        "zone_labels": ZONE_LABELS,
        "non_circular_conditions_broken": [
            "feature_derivation_via_physical_cutpoints",
            "test_derivation_via_pattern_specific_verdict",
        ],
        "p_value_threshold": P_VALUE_THRESHOLD,
        "alignment_warning_threshold": ALIGNMENT_WARNING_THRESHOLD,
        "alignment_severe_threshold": ALIGNMENT_SEVERE_THRESHOLD,
        "asymptotic_expected_floor": ASYMPTOTIC_EXPECTED_FLOOR,
        "min_occupied_bin_count_for_test": MIN_OCCUPIED_BIN_COUNT_FOR_TEST,
        "permutation_seed_locked": PERMUTATION_SEED,
        "n_permutations_locked": N_PERMUTATIONS,
        "chi2_critical_at_p01": CHI2_CRITICAL_AT_P01,
        "audit": audit,
        "pattern_check": pattern,
        "verdict": verdict,
    }

    (args.out / "manifest.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")

    per_row_fields = [
        "label", "index", "m3", "m3_key", "z0", "period", "stability", "branch_label",
        "velocity_fraction", "zone",
    ]
    write_csv(args.out / "per_row_table.csv", analyzable, per_row_fields)

    contingency_fields = [
        "zone", "N", "S", "U", "S_fraction", "expected_S", "expected_U", "chi2_contribution", "occupied",
    ]
    write_csv(args.out / "contingency_table_three_zone.csv", audit["contingency"], contingency_fields)

    print(f"[v09a] verdict: {verdict}")
    print(f"[v09a] domain: {len(analyzable)} rows  ({s_count} S / {u_count} U)")
    print(f"[v09a] attrition disclosure: v07a blocked {blocked_count}/{v07a_total}")
    print(f"[v09a] zones:")
    for row in audit["contingency"]:
        print(f"          {row['zone']:>22s}: N={row['N']:3d}  S={row['S']:3d}  U={row['U']:3d}  "
              f"S_frac={row['S_fraction']:.4f}  chi2_contrib={row['chi2_contribution']:.4f}")
    pa = audit
    print(f"[v09a] chi^2: {pa['chi_squared']:.6f}  branch={pa['test_branch_taken']}  "
          f"df={pa['df']}  critical={pa['critical']}  p={pa['p_value']}  "
          f"alignment={pa['alignment_tightness_scalar']:.4f}")
    pc = pattern
    print(f"[v09a] pattern: S_frac(pos)={pc['S_fraction_positional']:.4f}  "
          f"S_frac(mix)={pc['S_fraction_mixed']:.4f}  S_frac(vel)={pc['S_fraction_velocity_heavy']:.4f}")
    print(f"[v09a] U-shape check: mixed < pos? {pc['mixed_lt_positional']}  "
          f"mixed < vel? {pc['mixed_lt_velocity_heavy']}  "
          f"u_shape_confirmed? {pc['u_shape_confirmed']}")
    print(f"[v09a] manifest: {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
