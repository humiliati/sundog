"""v0.4a domain-map aggregator.

Reads Pass 1 and Pass 2 sigma_3-scan receipts under
`results/isotrophy/k-facet-v04a-domain-map/`, applies the pre-registered
four-band classifier on F_beta_residual_inf, and emits the final
table[m_3][stability][class] manifest.

Two modes:
  --phase pass1   Aggregate Pass 1 outputs only; emit `aggregator_manifest.json`
                  with provisional bands and a flagged-row list for Pass 2.
  --phase final   Combine Pass 1 + Pass 2 receipts; emit
                  `manifest.json` with per-row final classification +
                  provenance, plus the table[m_3][stability][class].

All thresholds are pre-registered in
`internal/anniversary/kfacet_v04a_domain_map_preregistration.md`.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Pre-registered classifier bands.
V04A_VERSION = "v0.4a-domain-map"
V04A_PASS1_IDENTITY_ROT_TOL = 1e-6
V04A_PASS1_PHASE_GRID = 73
V04A_PASS2_IDENTITY_ROT_TOL = 1e-9
V04A_PASS2_PHASE_GRID = 361
V04A_BAND_Z2_CLEAN_MAX = 1e-4
V04A_BAND_MARGINAL_MAX = 1e-2
V04A_BAND_SMALLER_MAX = 1.0
# residual > 1.0 OR integration failure -> undefined

ROOT = Path(__file__).resolve().parent.parent
ROOT_DIR = ROOT / "results/isotrophy/k-facet-v04a-domain-map"
PASS1_ROOT = ROOT_DIR / "pass1"
PASS2_ROOT = ROOT_DIR / "pass2"

M3_VALUES = ("0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1",
             "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.9")


def classify(residual: float | None) -> str:
    """Pre-registered four-band classifier on F_beta_residual_inf."""
    if residual is None:
        return "undefined"
    if residual <= V04A_BAND_Z2_CLEAN_MAX:
        return "Z2_clean"
    if residual <= V04A_BAND_MARGINAL_MAX:
        return "marginal_Z2"
    if residual <= V04A_BAND_SMALLER_MAX:
        return "smaller_symmetry"
    return "undefined"


def load_pass1_rows() -> list[dict]:
    """Walk Pass 1 m_3 subdirectories and load every row from each residuals.csv."""
    rows = []
    for m3 in M3_VALUES:
        residuals_csv = PASS1_ROOT / f"m3eq{m3}" / "residuals.csv"
        if not residuals_csv.is_file():
            continue
        with residuals_csv.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "label": r["label"],
                    "index": int(r["index"]),
                    "m3_input": m3,
                    "m3": float(r["m3"]),
                    "z0": float(r["z0"]),
                    "period": float(r["period"]),
                    "stability": r["stability"],
                    "sigma_group_residual_inf": float(r["sigma_group_residual_inf"]),
                    "F_beta_residual_inf": float(r["F_beta_residual_inf"]),
                    "F_beta_to_closure": float(r["F_beta_to_closure"]),
                    "inertia_degenerate": r.get("inertia_degenerate", "False") == "True",
                })
    return rows


def load_pass2_overrides() -> dict[tuple[str, int], dict]:
    """Walk Pass 2 directories and load each row's residual override."""
    overrides = {}
    if not PASS2_ROOT.is_dir():
        return overrides
    for m3_dir in PASS2_ROOT.iterdir():
        if not m3_dir.is_dir() or not m3_dir.name.startswith("m3eq"):
            continue
        m3 = m3_dir.name[4:]
        for row_dir in m3_dir.iterdir():
            if not row_dir.is_dir() or not row_dir.name.startswith("O"):
                continue
            residuals_csv = row_dir / "residuals.csv"
            if not residuals_csv.is_file():
                continue
            with residuals_csv.open("r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    key = (m3, int(r["index"]))
                    overrides[key] = {
                        "F_beta_residual_inf": float(r["F_beta_residual_inf"]),
                        "F_beta_to_closure": float(r["F_beta_to_closure"]),
                        "sigma_group_residual_inf": float(r["sigma_group_residual_inf"]),
                    }
    return overrides


def aggregate_pass1(rows: list[dict]) -> dict:
    """Provisional bands + flagged-row list."""
    flagged = []
    for r in rows:
        band = classify(r["F_beta_residual_inf"])
        r["pass1_class"] = band
        if band != "Z2_clean":
            flagged.append({
                "m3": r["m3_input"],
                "index": r["index"],
                "label": r["label"],
                "pass1_F_beta_residual_inf": r["F_beta_residual_inf"],
                "pass1_class": band,
            })
    return {
        "mode": "kfacet_v04a_pass1_aggregate",
        "version": V04A_VERSION,
        "pass1_tolerances": {
            "identity_rotation_tolerance": V04A_PASS1_IDENTITY_ROT_TOL,
            "phase_grid": V04A_PASS1_PHASE_GRID,
        },
        "thresholds": {
            "Z2_clean_max": V04A_BAND_Z2_CLEAN_MAX,
            "marginal_max": V04A_BAND_MARGINAL_MAX,
            "smaller_max": V04A_BAND_SMALLER_MAX,
        },
        "row_count": len(rows),
        "rows": rows,
        "flagged_for_pass2": flagged,
        "summary": {
            "total_rows": len(rows),
            "Z2_clean": sum(1 for r in rows if r["pass1_class"] == "Z2_clean"),
            "marginal_Z2": sum(1 for r in rows if r["pass1_class"] == "marginal_Z2"),
            "smaller_symmetry": sum(1 for r in rows if r["pass1_class"] == "smaller_symmetry"),
            "undefined": sum(1 for r in rows if r["pass1_class"] == "undefined"),
            "flagged_count": len(flagged),
        },
    }


def aggregate_final(rows: list[dict], overrides: dict) -> dict:
    """Apply Pass 2 overrides, emit final table[m_3][stability][class]."""
    reclassification_count = 0
    for r in rows:
        pass1_band = classify(r["F_beta_residual_inf"])
        r["pass1_class"] = pass1_band
        key = (r["m3_input"], r["index"])
        if key in overrides:
            r["pass2_F_beta_residual_inf"] = overrides[key]["F_beta_residual_inf"]
            pass2_band = classify(r["pass2_F_beta_residual_inf"])
            r["pass2_class"] = pass2_band
            r["final_class"] = pass2_band
            r["provenance"] = "pass2"
            if pass1_band != pass2_band:
                reclassification_count += 1
        else:
            r["pass2_F_beta_residual_inf"] = None
            r["pass2_class"] = None
            r["final_class"] = pass1_band
            r["provenance"] = "pass1"
    # Build the table
    table = {}
    for m3 in M3_VALUES:
        table[m3] = {"S": {}, "U": {}}
        for stab in ("S", "U"):
            table[m3][stab] = {"Z2_clean": 0, "marginal_Z2": 0,
                               "smaller_symmetry": 0, "undefined": 0}
    for r in rows:
        m3 = r["m3_input"]
        stab = r["stability"]
        cls = r["final_class"]
        if m3 in table and stab in table[m3] and cls in table[m3][stab]:
            table[m3][stab][cls] += 1
    # Summary
    class_counts = {"Z2_clean": 0, "marginal_Z2": 0,
                    "smaller_symmetry": 0, "undefined": 0}
    for r in rows:
        class_counts[r["final_class"]] = class_counts.get(r["final_class"], 0) + 1
    # Pre-registered verdict
    if (class_counts["smaller_symmetry"] == 0
            and class_counts["undefined"] == 0
            and class_counts["marginal_Z2"] == 0):
        verdict = "outcome_A_all_Z2_clean"
    elif (class_counts["smaller_symmetry"] == 0
            and class_counts["undefined"] == 0):
        verdict = "outcome_B_marginal_populated"
    else:
        verdict = "outcome_C_smaller_or_undefined_populated"
    return {
        "mode": "kfacet_v04a_domain_map",
        "version": V04A_VERSION,
        "pass1_tolerances": {
            "identity_rotation_tolerance": V04A_PASS1_IDENTITY_ROT_TOL,
            "phase_grid": V04A_PASS1_PHASE_GRID,
        },
        "pass2_tolerances": {
            "identity_rotation_tolerance": V04A_PASS2_IDENTITY_ROT_TOL,
            "phase_grid": V04A_PASS2_PHASE_GRID,
        },
        "thresholds": {
            "Z2_clean_max": V04A_BAND_Z2_CLEAN_MAX,
            "marginal_max": V04A_BAND_MARGINAL_MAX,
            "smaller_max": V04A_BAND_SMALLER_MAX,
        },
        "per_row_table": rows,
        "table_m3_by_stability_by_class": table,
        "summary": {
            "total_rows": len(rows),
            "class_counts": class_counts,
            "pass2_invoked_count": len(overrides),
            "pass2_reclassification_count": reclassification_count,
        },
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pass1", "final"), required=True)
    args = parser.parse_args()
    rows = load_pass1_rows()
    if args.phase == "pass1":
        result = aggregate_pass1(rows)
        out_path = PASS1_ROOT / "aggregator_manifest.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"[v04a-aggregator pass1] {result['row_count']} rows, "
              f"{result['summary']['flagged_count']} flagged for Pass 2")
        print(f"  Z2_clean:         {result['summary']['Z2_clean']}")
        print(f"  marginal_Z2:      {result['summary']['marginal_Z2']}")
        print(f"  smaller_symmetry: {result['summary']['smaller_symmetry']}")
        print(f"  undefined:        {result['summary']['undefined']}")
        print(f"  manifest:         {out_path}")
        # Also emit a flagged-rows CSV for the Pass 2 shell loop
        flagged_csv = PASS1_ROOT / "flagged_for_pass2.csv"
        with flagged_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["m3", "index", "label", "pass1_F_beta_residual_inf", "pass1_class"])
            for r in result["flagged_for_pass2"]:
                writer.writerow([r["m3"], r["index"], r["label"],
                                 r["pass1_F_beta_residual_inf"], r["pass1_class"]])
        print(f"  flagged CSV:      {flagged_csv}")
    else:
        overrides = load_pass2_overrides()
        result = aggregate_final(rows, overrides)
        out_path = ROOT_DIR / "manifest.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"[v04a-aggregator final] verdict: {result['verdict']}")
        print(f"  class_counts: {result['summary']['class_counts']}")
        print(f"  pass2 invoked: {result['summary']['pass2_invoked_count']}")
        print(f"  pass2 reclassifications: {result['summary']['pass2_reclassification_count']}")
        print(f"  manifest: {out_path}")


if __name__ == "__main__":
    main()
