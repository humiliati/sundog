"""v0.6a energy-quartile audit.

Implements the form locked in
`internal/anniversary/kfacet_v06a_energy_quartile_audit_form.md`.

The runner is catalog-only: it parses the published supplementary-B initial
conditions, joins v0.4a/v0.5a receipt metadata, computes E and |L| at the
initial condition, and emits the registered quartile chi-squared receipts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MANIFEST = ROOT / "results/isotrophy/k-facet-v04a-domain-map/manifest.json"
DEFAULT_V05A_TABLE = ROOT / "results/isotrophy/k-facet-v05a-branch-map/per_row_table.csv"
DEFAULT_CATALOG = ROOT / "docs/isotrophy/supplementary-B_piano-init-condit-3d.txt"
DEFAULT_OUT = ROOT / "results/isotrophy/k-facet-v06a-energy-quartile-audit"

FORM_LOCK = "internal/anniversary/kfacet_v06a_energy_quartile_audit_form.md"
VERSION = "v0.6a-energy-quartile-audit"

CHI2_CRITICAL = 11.34
ALIGNMENT_WARNING_THRESHOLD = 0.8
SANITY_THRESHOLD = 1e-6
QUANTILE_METHOD = "linear"

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

SENTINELS = {
    ("0.4", 50),
    ("0.4", 62),
    ("0.4", 67),
    ("0.4", 434),
    ("1.0", 242),
    ("1.0", 282),
    ("1.0", 284),
}


def format_m3(value: str | float) -> str:
    return f"{float(value):.1f}"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"expected boolean text, got {value!r}")


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_catalog(path: Path) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = ROW_RE.search(line)
        if not match:
            continue
        g = match.groupdict()
        m3 = float(g["m3"])
        index = int(g["index"])
        rows[(format_m3(m3), index)] = {
            "label": f"O_{{{index}}}({m3:g})",
            "index": index,
            "line_no": line_no,
            "m3": m3,
            "m3_key": format_m3(m3),
            "z0": float(g["z0"]),
            "vx": float(g["vx"]),
            "vy": float(g["vy"]),
            "vz": float(g["vz"]),
            "period": float(g["T"]),
            "stability": g["stability"],
        }
    return rows


def load_v05a_table(path: Path) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            key = (format_m3(raw["m3"]), int(raw["index"]))
            rows[key] = {
                "label": raw["label"],
                "index": int(raw["index"]),
                "m3": float(raw["m3"]),
                "m3_key": format_m3(raw["m3"]),
                "z0": float(raw["z0"]),
                "period": float(raw["period"]),
                "stability": raw["stability"],
                "branch_label": raw["branch_label"],
                "b1_m3_lt_1": parse_bool(raw["b1_m3_lt_1"]),
                "b2_z0_lt_0p3": parse_bool(raw["b2_z0_lt_0p3"]),
                "b3_abs_vz_lt_1e_minus_6": parse_bool(raw["b3_abs_vz_lt_1e_minus_6"]),
                "b4_m3_z0_sq_lt_2": parse_bool(raw["b4_m3_z0_sq_lt_2"]),
            }
    return rows


def load_v04a_manifest(path: Path) -> dict[tuple[str, int], dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for raw in manifest["per_row_table"]:
        key = (format_m3(raw["m3"]), int(raw["index"]))
        rows[key] = raw
    return rows


def expand_initial_state(row: dict, center_com: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m3 = float(row["m3"])
    masses = np.array([1.0, 1.0, m3], dtype=float)
    x = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, float(row["z0"])],
        ],
        dtype=float,
    )
    v = np.array(
        [
            [float(row["vx"]), float(row["vy"]), float(row["vz"])],
            [float(row["vx"]), float(row["vy"]), -float(row["vz"])],
            [-2.0 * float(row["vx"]) / m3, -2.0 * float(row["vy"]) / m3, 0.0],
        ],
        dtype=float,
    )
    if center_com:
        x = x - np.average(x, axis=0, weights=masses)
    return masses, x, v


def conserved_values(row: dict) -> dict[str, float]:
    masses, x, v = expand_initial_state(row, center_com=True)
    kinetic = 0.5 * float(np.sum(masses * np.sum(v * v, axis=1)))
    potential = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            potential -= float(masses[i] * masses[j] / np.linalg.norm(x[i] - x[j]))
    angular_vector = np.sum(masses[:, None] * np.cross(x, v), axis=0)
    angular_norm = float(np.linalg.norm(angular_vector))
    total_momentum = np.sum(masses[:, None] * v, axis=0)
    center_of_mass = np.average(x, axis=0, weights=masses)
    return {
        "energy": kinetic + potential,
        "kinetic_energy": kinetic,
        "potential_energy": potential,
        "angular_momentum_x": float(angular_vector[0]),
        "angular_momentum_y": float(angular_vector[1]),
        "angular_momentum_z": float(angular_vector[2]),
        "angular_momentum_norm": angular_norm,
        "angular_momentum_z_abs_over_norm": abs(float(angular_vector[2])) / angular_norm if angular_norm else None,
        "total_momentum_norm": float(np.linalg.norm(total_momentum)),
        "center_of_mass_norm": float(np.linalg.norm(center_of_mass)),
    }


def v03_reference_values(row: dict) -> dict[str, float] | None:
    """Use the existing v0.3 invariant implementation as a parity reference.

    The cross-m3 gate receipts in this workspace do not carry scalar E/|L|
    fields, but the v0.3 workbench invariant code is available and predates
    this runner. This check keeps the implementation tied to that convention.
    """
    try:
        import scripts.isotrophy_workbench as iso
    except Exception:
        return None
    orbit_row = iso.OrbitRow(
        source="B",
        line_no=int(row.get("line_no", 0)),
        index=int(row["index"]),
        m3=float(row["m3"]),
        z0=float(row["z0"]),
        vx=float(row["vx"]),
        vy=float(row["vy"]),
        vz=float(row["vz"]),
        period=float(row["period"]),
        stability=str(row["stability"]),
    )
    record = iso.invariant_record(orbit_row)
    return {
        "energy": float(record["energy"]),
        "angular_momentum_norm": float(record["angular_momentum_norm"]),
    }


def quantile_cutpoints(values: list[float]) -> dict[str, float]:
    q25, q50, q75 = np.quantile(np.asarray(values, dtype=float), [0.25, 0.50, 0.75], method=QUANTILE_METHOD)
    return {"q25": float(q25), "q50": float(q50), "q75": float(q75)}


def assign_quartile(value: float, cutpoints: dict[str, float]) -> int:
    if value <= cutpoints["q25"]:
        return 1
    if value <= cutpoints["q50"]:
        return 2
    if value <= cutpoints["q75"]:
        return 3
    return 4


def contingency(rows: list[dict], quartile_key: str) -> list[dict]:
    table = []
    for quartile in (1, 2, 3, 4):
        selected = [row for row in rows if int(row[quartile_key]) == quartile]
        counts = Counter(row["stability"] for row in selected)
        n = len(selected)
        table.append({
            "quartile": quartile,
            "N": n,
            "S": counts.get("S", 0),
            "U": counts.get("U", 0),
            "S_fraction": counts.get("S", 0) / n if n else 0.0,
        })
    return table


def chi_square(table: list[dict]) -> tuple[float, list[dict]]:
    total = sum(row["N"] for row in table)
    total_s = sum(row["S"] for row in table)
    total_u = sum(row["U"] for row in table)
    chi2 = 0.0
    enriched = []
    for row in table:
        expected_s = row["N"] * total_s / total if total else 0.0
        expected_u = row["N"] * total_u / total if total else 0.0
        contrib_s = ((row["S"] - expected_s) ** 2 / expected_s) if expected_s else 0.0
        contrib_u = ((row["U"] - expected_u) ** 2 / expected_u) if expected_u else 0.0
        contribution = contrib_s + contrib_u
        chi2 += contribution
        enriched.append({
            **row,
            "expected_S": expected_s,
            "expected_U": expected_u,
            "chi2_contribution": contribution,
        })
    return chi2, enriched


def chi_square_survival_df3(x: float) -> float:
    if x < 0:
        return 1.0
    z = x / 2.0
    return math.erfc(math.sqrt(z)) + (2.0 * math.sqrt(z) * math.exp(-z) / math.sqrt(math.pi))


def alignment_tightness(rows: list[dict], quartile_key: str) -> tuple[float, list[dict]]:
    records = []
    max_fraction = 0.0
    for quartile in (1, 2, 3, 4):
        selected = [row for row in rows if int(row[quartile_key]) == quartile]
        counts = Counter(row["branch_label"] for row in selected)
        dominant_branch, dominant_count = ("", 0)
        if counts:
            dominant_branch, dominant_count = max(counts.items(), key=lambda item: (item[1], item[0]))
        fraction = dominant_count / len(selected) if selected else 0.0
        max_fraction = max(max_fraction, fraction)
        records.append({
            "quartile": quartile,
            "N": len(selected),
            "dominant_branch_label": dominant_branch,
            "dominant_branch_count": dominant_count,
            "dominant_branch_fraction": fraction,
            "branch_counts": dict(sorted(counts.items())),
        })
    return max_fraction, records


def build_cross_table(rows: list[dict], row_key: str, col_key: str) -> list[dict]:
    row_values = sorted({row[row_key] for row in rows})
    col_values = sorted({row[col_key] for row in rows})
    table = []
    for rv in row_values:
        selected = [row for row in rows if row[row_key] == rv]
        record = {row_key: rv, "N": len(selected)}
        for cv in col_values:
            record[str(cv)] = sum(1 for row in selected if row[col_key] == cv)
        table.append(record)
    return table


def build_rows(catalog_rows: dict[tuple[str, int], dict], v04a_rows: dict[tuple[str, int], dict], v05a_rows: dict[tuple[str, int], dict]) -> list[dict]:
    rows = []
    for key in sorted(v05a_rows, key=lambda item: (float(item[0]), item[1])):
        if key not in catalog_rows:
            raise ValueError(f"missing catalog row for {key}")
        if key not in v04a_rows:
            raise ValueError(f"missing v0.4a manifest row for {key}")
        catalog = catalog_rows[key]
        v05a = v05a_rows[key]
        v04a = v04a_rows[key]
        values = conserved_values(catalog)
        if abs(float(catalog["z0"]) - float(v05a["z0"])) > 1e-12:
            raise ValueError(f"z0 mismatch for {catalog['label']}")
        if str(catalog["stability"]) != str(v05a["stability"]) or str(catalog["stability"]) != str(v04a["stability"]):
            raise ValueError(f"stability mismatch for {catalog['label']}")
        rows.append({
            **catalog,
            **values,
            "branch_label": v05a["branch_label"],
            "b1_m3_lt_1": v05a["b1_m3_lt_1"],
            "b2_z0_lt_0p3": v05a["b2_z0_lt_0p3"],
            "b3_abs_vz_lt_1e_minus_6": v05a["b3_abs_vz_lt_1e_minus_6"],
            "b4_m3_z0_sq_lt_2": v05a["b4_m3_z0_sq_lt_2"],
            "final_Z2_class": v04a.get("final_class"),
            "v04a_provenance": v04a.get("provenance"),
        })
    return rows


def run_sanity(rows: list[dict]) -> dict:
    rows_by_key = {(row["m3_key"], int(row["index"])): row for row in rows}
    sentinel_records = []
    missing = []
    max_energy_residual = 0.0
    max_l_residual = 0.0
    for key in sorted(SENTINELS, key=lambda item: (float(item[0]), item[1])):
        row = rows_by_key.get(key)
        if row is None:
            missing.append({"m3": key[0], "index": key[1]})
            continue
        reference = v03_reference_values(row)
        if reference is None:
            sentinel_records.append({
                "label": row["label"],
                "reference_available": False,
                "energy": row["energy"],
                "angular_momentum_norm": row["angular_momentum_norm"],
            })
            continue
        energy_residual = abs(float(row["energy"]) - reference["energy"])
        l_residual = abs(float(row["angular_momentum_norm"]) - reference["angular_momentum_norm"])
        max_energy_residual = max(max_energy_residual, energy_residual)
        max_l_residual = max(max_l_residual, l_residual)
        sentinel_records.append({
            "label": row["label"],
            "reference_available": True,
            "reference_source": "scripts.isotrophy_workbench.invariant_record",
            "energy": row["energy"],
            "reference_energy": reference["energy"],
            "energy_abs_residual": energy_residual,
            "angular_momentum_norm": row["angular_momentum_norm"],
            "reference_angular_momentum_norm": reference["angular_momentum_norm"],
            "angular_momentum_norm_abs_residual": l_residual,
        })
    pass_status = (
        not missing
        and all(record.get("reference_available") for record in sentinel_records)
        and max_energy_residual < SANITY_THRESHOLD
        and max_l_residual < SANITY_THRESHOLD
    )
    return {
        "pass_threshold": SANITY_THRESHOLD,
        "status": "pass" if pass_status else "fail",
        "reference_note": (
            "Cross-m3 gate receipts in this workspace do not carry scalar E/|L| fields; "
            "sanity check compares this runner against the pre-existing v0.3 isotrophy_workbench "
            "invariant implementation on the seven registered sentinel rows."
        ),
        "missing_sentinel_rows": missing,
        "per_row_residual_E_max": max_energy_residual,
        "per_row_residual_L_max": max_l_residual,
        "sentinel_rows": sentinel_records,
    }


def verdict_for(chi2: float, alignment: float) -> str:
    if chi2 <= CHI2_CRITICAL:
        return "energy_quartile_fails_audit"
    if alignment > ALIGNMENT_WARNING_THRESHOLD:
        return "energy_quartile_passes_audit_alignment_warning"
    return "energy_quartile_passes_audit"


def build_result(manifest_path: Path, v05a_path: Path, catalog_path: Path) -> dict:
    catalog_rows = parse_catalog(catalog_path)
    v04a_rows = load_v04a_manifest(manifest_path)
    v05a_rows = load_v05a_table(v05a_path)
    rows = build_rows(catalog_rows, v04a_rows, v05a_rows)

    energy_cutpoints = quantile_cutpoints([row["energy"] for row in rows])
    l_cutpoints = quantile_cutpoints([row["angular_momentum_norm"] for row in rows])
    for row in rows:
        row["Q_E"] = assign_quartile(float(row["energy"]), energy_cutpoints)
        row["Q_L"] = assign_quartile(float(row["angular_momentum_norm"]), l_cutpoints)
        row["z0_bucket"] = "z0_lt_0p3" if row["b2_z0_lt_0p3"] else "z0_ge_0p3"

    sanity = run_sanity(rows)
    rows_with_e_ge_0 = [
        {"label": row["label"], "energy": row["energy"]}
        for row in rows
        if float(row["energy"]) >= 0.0
    ]
    bound_check = {
        "rows_with_E_ge_0": rows_with_e_ge_0,
        "status": "pass" if not rows_with_e_ge_0 else "fail",
    }

    contingency_e = contingency(rows, "Q_E")
    chi2_e, contingency_e = chi_square(contingency_e)
    alignment_e, alignment_records_e = alignment_tightness(rows, "Q_E")

    contingency_l = contingency(rows, "Q_L")
    chi2_l, contingency_l = chi_square(contingency_l)
    alignment_l, alignment_records_l = alignment_tightness(rows, "Q_L")

    hard_gate_passed = sanity["status"] == "pass" and bound_check["status"] == "pass"
    verdict = verdict_for(chi2_e, alignment_e) if hard_gate_passed else "energy_quartile_blocked_sanity_or_bound"

    total_s = sum(1 for row in rows if row["stability"] == "S")
    total_u = len(rows) - total_s
    result = {
        "mode": "v0.6a-energy-quartile-audit",
        "version": VERSION,
        "form_lock": FORM_LOCK,
        "input_manifest_v04a": relpath(manifest_path),
        "input_per_row_table_v05a": relpath(v05a_path),
        "input_catalog_supplementary_b": relpath(catalog_path),
        "input_note": (
            "v0.4a/v0.5a receipts provide row/stability/branch metadata; "
            "supp-B catalog is re-read for vx/vy/vz IC fields required by E and |L|."
        ),
        "quartile_method": f"numpy_quantile_{QUANTILE_METHOD}",
        "cutpoints_E": energy_cutpoints,
        "cutpoints_L": l_cutpoints,
        "bin_assignment_convention": "right_closed_lower_ties_to_lower",
        "sanity_check_v03_sentinels": sanity,
        "bound_orbit_check": bound_check,
        "summary": {
            "total_rows": len(rows),
            "S": total_s,
            "U": total_u,
            "energy_min": min(row["energy"] for row in rows),
            "energy_max": max(row["energy"] for row in rows),
            "angular_momentum_norm_min": min(row["angular_momentum_norm"] for row in rows),
            "angular_momentum_norm_max": max(row["angular_momentum_norm"] for row in rows),
        },
        "contingency_E": contingency_e,
        "chi_squared_E": chi2_e,
        "p_value_E_df3": chi_square_survival_df3(chi2_e),
        "df_E": 3,
        "critical": CHI2_CRITICAL,
        "alignment_tightness_scalar_E": alignment_e,
        "alignment_tightness_threshold": ALIGNMENT_WARNING_THRESHOLD,
        "alignment_records_E": alignment_records_e,
        "verdict": verdict,
        "contingency_L": contingency_l,
        "chi_squared_L": chi2_l,
        "p_value_L_df3": chi_square_survival_df3(chi2_l),
        "df_L": 3,
        "alignment_tightness_scalar_L": alignment_l,
        "alignment_records_L": alignment_records_l,
        "sidecar_status": "report_only",
        "diagnostic_tables_emitted": [
            "E_vs_m3",
            "E_vs_z0_bucket",
            "E_vs_branch",
            "L_vs_branch",
        ],
        "_tables": {
            "per_row_table": rows,
            "contingency_E": contingency_e,
            "contingency_L": contingency_l,
            "E_vs_m3": build_cross_table(rows, "Q_E", "m3_key"),
            "E_vs_z0_bucket": build_cross_table(rows, "Q_E", "z0_bucket"),
            "E_vs_branch": build_cross_table(rows, "Q_E", "branch_label"),
            "L_vs_branch": build_cross_table(rows, "Q_L", "branch_label"),
        },
    }
    return result


def write_receipts(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = result.pop("_tables")
    try:
        (out_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        write_csv(out_dir / "per_row_table.csv", tables["per_row_table"], [
            "label", "index", "m3", "m3_key", "z0", "period", "stability",
            "energy", "kinetic_energy", "potential_energy", "angular_momentum_norm",
            "angular_momentum_x", "angular_momentum_y", "angular_momentum_z",
            "angular_momentum_z_abs_over_norm", "Q_E", "Q_L", "branch_label",
            "b1_m3_lt_1", "b2_z0_lt_0p3", "final_Z2_class", "v04a_provenance",
        ])
        write_csv(out_dir / "contingency_table_E.csv", tables["contingency_E"], [
            "quartile", "N", "S", "U", "S_fraction", "expected_S", "expected_U", "chi2_contribution",
        ])
        write_csv(out_dir / "contingency_table_L.csv", tables["contingency_L"], [
            "quartile", "N", "S", "U", "S_fraction", "expected_S", "expected_U", "chi2_contribution",
        ])
        write_csv_dynamic(out_dir / "energy_quartile_by_m3.csv", tables["E_vs_m3"], "Q_E")
        write_csv_dynamic(out_dir / "energy_quartile_by_z0_bucket.csv", tables["E_vs_z0_bucket"], "Q_E")
        write_csv_dynamic(out_dir / "energy_quartile_by_branch.csv", tables["E_vs_branch"], "Q_E")
        write_csv_dynamic(out_dir / "l_quartile_by_branch.csv", tables["L_vs_branch"], "Q_L")
    finally:
        result["_tables"] = tables


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--v05a-table", type=Path, default=DEFAULT_V05A_TABLE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    result = build_result(args.manifest.resolve(), args.v05a_table.resolve(), args.catalog.resolve())
    write_receipts(result, args.out)

    print("[v06a-energy-quartile] verdict:", result["verdict"])
    print(f"  rows:        {result['summary']['total_rows']}  S={result['summary']['S']} U={result['summary']['U']}")
    print(f"  E chi^2:     {result['chi_squared_E']:.6f}  p={result['p_value_E_df3']:.6g}  align={result['alignment_tightness_scalar_E']:.6f}")
    print(f"  |L| chi^2:   {result['chi_squared_L']:.6f}  p={result['p_value_L_df3']:.6g}  align={result['alignment_tightness_scalar_L']:.6f}  (sidecar)")
    print(f"  sanity:      {result['sanity_check_v03_sentinels']['status']}  bound: {result['bound_orbit_check']['status']}")
    print(f"  manifest:    {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
