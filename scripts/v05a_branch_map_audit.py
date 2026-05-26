"""v0.5a branch-shadow audit.

Reads the v0.4a domain-map manifest, supplements catalog-only velocity
diagnostics from supplementary-B, applies the locked branch hash in
`docs/isotrophy/kfacet/kfacet_v05a_branch_map_form.md`, and emits the 4x2
branch_label x stability contingency receipt.

No dynamical compute is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "results/isotrophy/k-facet-v04a-domain-map/manifest.json"
DEFAULT_CATALOG = ROOT / "docs/isotrophy/supplementary-B_piano-init-condit-3d.txt"
DEFAULT_OUT = ROOT / "results/isotrophy/k-facet-v05a-branch-map"
FORM_LOCK = "docs/isotrophy/kfacet/kfacet_v05a_branch_map_form.md"

VERSION = "v0.5a-branch-map-audit"
CHI2_CRITICAL = 11.34  # chi-squared(3), p=0.01
CHI2_MARGINAL_HIGH = 16.0
Z0_THRESHOLD = 0.3
VZ_ZERO_TOL = 1e-6
INERTIA_THRESHOLD = 2.0

FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
ROW_RE = re.compile(
    rf"O_\{{(?P<index>\d+)\}}\((?P<m3>{FLOAT})\)\s+"
    rf"(?P<z0>{FLOAT})\s+"
    rf"(?P<vx>{FLOAT})\s+"
    rf"(?P<vy>{FLOAT})\s+"
    rf"(?P<vz>{FLOAT})\s+"
    rf"(?P<T>{FLOAT})\s+"
    rf"(?P<stability>[SU])"
)


def parse_catalog(path: Path) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ROW_RE.search(line)
        if not match:
            continue
        g = match.groupdict()
        key = (format_m3(float(g["m3"])), int(g["index"]))
        rows[key] = {
            "index": int(g["index"]),
            "m3": float(g["m3"]),
            "z0": float(g["z0"]),
            "vx": float(g["vx"]),
            "vy": float(g["vy"]),
            "vz": float(g["vz"]),
            "period": float(g["T"]),
            "stability": g["stability"],
        }
    return rows


def format_m3(value: float | str) -> str:
    return f"{float(value):.1f}"


def bit_values(row: dict, catalog_row: dict | None) -> dict[str, bool]:
    m3 = float(row["m3"])
    z0 = float(row["z0"])
    vz = float(catalog_row["vz"]) if catalog_row else float("nan")
    return {
        "b1_m3_lt_1": m3 < 1.0,
        "b2_z0_lt_0p3": z0 < Z0_THRESHOLD,
        "b3_abs_vz_lt_1e_minus_6": abs(vz) < VZ_ZERO_TOL if not math.isnan(vz) else False,
        "b4_m3_z0_sq_lt_2": (m3 * z0 * z0) < INERTIA_THRESHOLD,
    }


def chi_square_survival_df3(x: float) -> float:
    """Survival function for chi-square with df=3.

    chi2(df=3) is Gamma(a=3/2, scale=2). For z=x/2:
    Q(3/2, z) = erfc(sqrt(z)) + 2*sqrt(z)*exp(-z)/sqrt(pi).
    """
    if x < 0:
        return 1.0
    z = x / 2.0
    return math.erfc(math.sqrt(z)) + (2.0 * math.sqrt(z) * math.exp(-z) / math.sqrt(math.pi))


def audit(manifest: dict, catalog_rows: dict[tuple[str, int], dict]) -> dict:
    rows = manifest["per_row_table"]
    per_row = []

    bit_names = [
        "b1_m3_lt_1",
        "b2_z0_lt_0p3",
        "b3_abs_vz_lt_1e_minus_6",
        "b4_m3_z0_sq_lt_2",
    ]
    truth_counts = {name: 0 for name in bit_names}

    for row in rows:
        key = (format_m3(row["m3_input"]), int(row["index"]))
        catalog_row = catalog_rows.get(key)
        bits = bit_values(row, catalog_row)
        for name, value in bits.items():
            truth_counts[name] += int(value)
        m3 = float(row["m3"])
        z0 = float(row["z0"])
        vz = float(catalog_row["vz"]) if catalog_row else None
        per_row.append({
            "label": row["label"],
            "index": int(row["index"]),
            "m3": m3,
            "m3_input": format_m3(row["m3_input"]),
            "z0": z0,
            "period": float(row["period"]),
            "stability": row["stability"],
            "vz": vz,
            "abs_vz": abs(vz) if vz is not None else None,
            "m3_z0_squared": m3 * z0 * z0,
            "bits": bits,
        })

    total = len(per_row)
    bit_status = {}
    active_bits = []
    for name in bit_names:
        count = truth_counts[name]
        status = "RETIRED_CONSTANT_FALSE" if count == 0 else (
            "RETIRED_CONSTANT_TRUE" if count == total else "ACTIVE"
        )
        bit_status[name] = {"true": count, "false": total - count, "status": status}
        if status == "ACTIVE":
            active_bits.append(name)

    buckets: dict[str, dict] = {}
    for row in per_row:
        active_tuple = tuple(bool(row["bits"][name]) for name in active_bits)
        label = branch_label(active_bits, active_tuple)
        row["active_bits"] = {name: row["bits"][name] for name in active_bits}
        row["branch_label"] = label
        bucket = buckets.setdefault(label, {
            "branch_label": label,
            "active_tuple": list(active_tuple),
            "N": 0,
            "S": 0,
            "U": 0,
        })
        bucket["N"] += 1
        bucket[row["stability"]] += 1

    total_s = sum(1 for row in per_row if row["stability"] == "S")
    total_u = total - total_s
    p_s = total_s / total if total else 0.0
    p_u = total_u / total if total else 0.0

    chi2 = 0.0
    bucket_list = []
    for label in sorted(buckets):
        bucket = buckets[label]
        expected_s = bucket["N"] * p_s
        expected_u = bucket["N"] * p_u
        contrib_s = ((bucket["S"] - expected_s) ** 2 / expected_s) if expected_s else 0.0
        contrib_u = ((bucket["U"] - expected_u) ** 2 / expected_u) if expected_u else 0.0
        contribution = contrib_s + contrib_u
        chi2 += contribution
        bucket_list.append({
            **bucket,
            "S_fraction": bucket["S"] / bucket["N"] if bucket["N"] else 0.0,
            "expected_S": expected_s,
            "expected_U": expected_u,
            "chi2_contribution": contribution,
        })

    df = max(len(bucket_list) - 1, 0)
    p_value = chi_square_survival_df3(chi2) if df == 3 else None
    if chi2 <= CHI2_CRITICAL:
        verdict = "branch_hash_fails_audit"
    elif chi2 < CHI2_MARGINAL_HIGH:
        verdict = "branch_hash_passes_audit_marginal"
    else:
        verdict = "branch_hash_passes_audit"

    return {
        "mode": "kfacet_v05a_branch_map_audit",
        "version": VERSION,
        "form_lock": FORM_LOCK,
        "input_manifest": str(DEFAULT_MANIFEST.relative_to(ROOT)),
        "catalog_path": str(DEFAULT_CATALOG.relative_to(ROOT)),
        "thresholds": {
            "m3_lt_threshold": 1.0,
            "z0_lt_threshold": Z0_THRESHOLD,
            "abs_vz_zero_tolerance": VZ_ZERO_TOL,
            "m3_z0_squared_threshold": INERTIA_THRESHOLD,
            "chi2_critical": CHI2_CRITICAL,
            "chi2_p": 0.01,
            "marginal_pass_upper": CHI2_MARGINAL_HIGH,
        },
        "summary": {
            "total_rows": total,
            "S": total_s,
            "U": total_u,
            "S_fraction": p_s,
            "active_bits": active_bits,
            "occupied_branch_count": len(bucket_list),
            "df": df,
            "chi_squared": chi2,
            "p_value_df3": p_value,
            "verdict": verdict,
        },
        "bit_truth_counts": bit_status,
        "bucket_counts": bucket_list,
        "per_row_table": per_row,
    }


def branch_label(active_bits: list[str], values: tuple[bool, ...]) -> str:
    if not active_bits:
        return "all_constant"
    parts = []
    for name, value in zip(active_bits, values):
        if name == "b1_m3_lt_1":
            parts.append("m3_lt_1" if value else "m3_ge_1")
        elif name == "b2_z0_lt_0p3":
            parts.append("z0_lt_0p3" if value else "z0_ge_0p3")
        elif name == "b3_abs_vz_lt_1e_minus_6":
            parts.append("vz_zero" if value else "vz_nonzero")
        elif name == "b4_m3_z0_sq_lt_2":
            parts.append("m3_z0sq_lt_2" if value else "m3_z0sq_ge_2")
        else:
            parts.append(f"{name}_{int(value)}")
    return "__".join(parts)


def write_receipts(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    with (out_dir / "contingency_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "branch_label", "N", "S", "U", "S_fraction",
            "expected_S", "expected_U", "chi2_contribution",
        ])
        for b in result["bucket_counts"]:
            writer.writerow([
                b["branch_label"], b["N"], b["S"], b["U"], b["S_fraction"],
                b["expected_S"], b["expected_U"], b["chi2_contribution"],
            ])

    with (out_dir / "per_row_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label", "index", "m3", "z0", "period", "stability",
            "vz", "abs_vz", "m3_z0_squared", "branch_label",
            "b1_m3_lt_1", "b2_z0_lt_0p3",
            "b3_abs_vz_lt_1e_minus_6", "b4_m3_z0_sq_lt_2",
        ])
        for row in result["per_row_table"]:
            writer.writerow([
                row["label"], row["index"], row["m3"], row["z0"], row["period"],
                row["stability"], row["vz"], row["abs_vz"], row["m3_z0_squared"],
                row["branch_label"],
                row["bits"]["b1_m3_lt_1"], row["bits"]["b2_z0_lt_0p3"],
                row["bits"]["b3_abs_vz_lt_1e_minus_6"],
                row["bits"]["b4_m3_z0_sq_lt_2"],
            ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    catalog_rows = parse_catalog(args.catalog)
    result = audit(manifest, catalog_rows)
    # Preserve the actual CLI paths if non-default paths were used.
    result["input_manifest"] = str(args.manifest)
    result["catalog_path"] = str(args.catalog)
    write_receipts(result, args.out)

    summary = result["summary"]
    print("[v05a-branch-map] verdict:", summary["verdict"])
    print(f"  rows:        {summary['total_rows']}  S={summary['S']} U={summary['U']}")
    print(f"  active_bits: {', '.join(summary['active_bits'])}")
    print(f"  buckets:     {summary['occupied_branch_count']}  df={summary['df']}")
    print(f"  chi^2:       {summary['chi_squared']:.6f}  critical={CHI2_CRITICAL}")
    if summary["p_value_df3"] is not None:
        print(f"  p_value:     {summary['p_value_df3']:.6g}")
    print(f"  manifest:    {args.out / 'manifest.json'}")
    print(f"  table:       {args.out / 'contingency_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
