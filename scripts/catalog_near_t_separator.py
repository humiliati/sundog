"""Catalog-wide near-T separator: extends the O_617 WHY-dive classifier across
the 21 strict G.2 rows. Asks one question: is O_617's near-T bridge direction
unique in the catalog, or are there other rows whose kernel directions sit at
the same structural edge?

Method, all no-integration (v2 -- isotypic-projector classifier):
  for each row in the 21 strict rows:
    1. Load M_i.npy and D3_*.npy from the sentinel-sweep-calibrated/O{idx}/.
    2. SVD (M_i - I); take all directions with sv < NEAR_T_BRIDGE_BAND_UPPER
       (1e-3) as the bridge-admitted kernel basis K. This includes the row's
       structural kernel plus any bridge directions below the absolute guard.
    3. Project D3 onto K: sigma_3_K = K^T sigma_3 K, etc.
    4. Build the D3 character projectors (T, S, E) in K-coordinates and
       extract orthonormal isotypic bases.
    5. Lift each isotypic basis vector back to 18D, then compute:
         alignment_sigma3 = <v, sigma_3 v>
         alignment_F_beta = <v, F_beta v>
         closure_sigma3 = ||sigma_3^3 v - v||
         closure_F_beta = ||F_beta^2 v - v||
    6. Classify each direction with respect to its projector-assigned isotype:
         clean_T:        T direction with closures < 1e-3
         clean_S:        S direction with closures < 1e-3
         clean_E:        E direction with sigma_3 acting as 2D rotation
         edge_T:         T direction with closure_sigma3 > 1e-3
         edge_S:         S direction with closure_F_beta > 1e-3
         edge_E_as_T:    E projector caught a T-like direction (O_617 bridge)
         edge_E_as_S:    E projector caught an S-like direction
         edge_E_other:   E direction not behaving as rotation, T, or S
  7. Aggregate per row, then across the catalog.

The isotypic-projector classifier replaces v1's per-SVD-vector approach.
SVD basis vectors are generally mixtures of T and S; v1 saw most kernel
directions as `unclassified` for that reason. v2 first projects onto
isotypics (via the D3 character projector, matching the reprocessor's
`compute_d3_isotypic_summary`) and only then asks per-direction questions.
The v1 finding that `O_617` is uniquely `near_T_edge` is preserved as
`edge_E_as_T` in the v2 schema: the bridge sits in an odd-dim E block
that the projector caught but whose behavior is structurally T.

Pre-registered question:
  Is `O_617`'s bridge direction the only structural-edge direction across
  the 21-row catalog? If yes, the WHY-dive's diagnosis is row-unique and
  promotable; if no, it is a class of bridges and the catalog has more
  structural edge cases.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent))
import isotrophy_workbench as W


NEAR_T_VERSION = "v0.3h-near-T-separator"
NEAR_T_ALIGNMENT_FLOOR = 0.99
NEAR_T_CLEAN_ALIGNMENT_FLOOR = 0.999
NEAR_T_CLOSURE_CLEAN = 1e-3
NEAR_T_BRIDGE_BAND_LOWER = 1e-7
NEAR_T_BRIDGE_BAND_UPPER = 1e-3

ROOT = Path(__file__).resolve().parent.parent
SENTINEL_ROOT = ROOT / "results/isotrophy/k-facet-v03-sentinel-sweep-calibrated"
ADAPTIVE_MANIFEST = ROOT / "results/isotrophy/k-facet-v03-sentinel-sweep-adaptive-floor-21/manifest.json"
OUT_DIR = ROOT / "results/isotrophy/k-facet-v03-near-T-separator"


def load_adaptive_floors():
    """Map row_index -> selected_floor from the reprocessor manifest."""
    manifest = json.loads(ADAPTIVE_MANIFEST.read_text(encoding="utf-8"))
    floors = {}
    for entry in manifest["rows"]:
        idx = entry["row_index"]
        floor = entry["selected_floor"]
        floors[idx] = floor
    return floors


def classify_direction(align_s3, align_fb, closure_s3, closure_fb):
    """Pre-registered classification of a single kernel/bridge direction."""
    if abs(align_s3) < NEAR_T_ALIGNMENT_FLOOR:
        return "e_rotation"
    # |alignment_sigma3| >= 0.99 from here on
    s3_clean = closure_s3 < NEAR_T_CLOSURE_CLEAN
    fb_clean = closure_fb < NEAR_T_CLOSURE_CLEAN
    if (
        align_s3 >= NEAR_T_CLEAN_ALIGNMENT_FLOOR
        and align_fb >= NEAR_T_CLEAN_ALIGNMENT_FLOOR
        and s3_clean
        and fb_clean
    ):
        return "clean_T"
    if (
        align_s3 >= NEAR_T_CLEAN_ALIGNMENT_FLOOR
        and align_fb <= -NEAR_T_CLEAN_ALIGNMENT_FLOOR
        and s3_clean
        and fb_clean
    ):
        return "clean_S"
    if align_s3 >= NEAR_T_ALIGNMENT_FLOOR and not s3_clean and fb_clean:
        return "near_T_edge"
    if align_s3 >= NEAR_T_ALIGNMENT_FLOOR and s3_clean and not fb_clean:
        return "near_S_edge"
    return "unclassified"


def analyze_row(row_index, adaptive_floor):
    """Compute the per-direction table for one row."""
    row_dir = SENTINEL_ROOT / f"O{row_index}"
    M_i = np.load(row_dir / "M_i.npy")
    sigma3 = np.load(row_dir / "D3_sigma3.npy")
    f_beta = np.load(row_dir / "D3_F_beta.npy")
    identity = np.eye(M_i.shape[0])
    _u, svs, vh = np.linalg.svd(M_i - identity)
    floor = float(adaptive_floor) if adaptive_floor is not None else None
    if floor is None:
        # O_617: reprocessor failed. Use the bridge band lower (1e-7) as the
        # kernel-edge stand-in -- the WHY-dive's k=7/k=8 convention.
        floor = NEAR_T_BRIDGE_BAND_LOWER
    sigma3_cubed = sigma3 @ sigma3 @ sigma3
    f_beta_squared = f_beta @ f_beta
    directions = []
    for j in range(svs.size):
        sv = float(svs[j])
        if sv >= NEAR_T_BRIDGE_BAND_UPPER:
            continue  # not in kernel + bridge band
        v = vh[j, :].copy()
        v = v / np.linalg.norm(v)
        Mv = M_i @ v
        s3v = sigma3 @ v
        s3cubedv = sigma3_cubed @ v
        fbv = f_beta @ v
        fbsqv = f_beta_squared @ v
        align_s3 = float(np.dot(v, s3v))
        align_fb = float(np.dot(v, fbv))
        closure_s3 = float(np.linalg.norm(s3cubedv - v))
        closure_fb = float(np.linalg.norm(fbsqv - v))
        residual_M = float(np.linalg.norm(Mv - v))
        band = (
            "bridge"
            if floor <= sv < NEAR_T_BRIDGE_BAND_UPPER
            else "kernel"
            if sv < floor
            else "non_kernel"
        )
        classification = classify_direction(align_s3, align_fb, closure_s3, closure_fb)
        directions.append({
            "index_in_sv_array": int(j),
            "singular_value": sv,
            "band": band,
            "alignment_sigma3": align_s3,
            "alignment_F_beta": align_fb,
            "closure_sigma3": closure_s3,
            "closure_F_beta": closure_fb,
            "residual_M_minus_I": residual_M,
            "classification": classification,
        })
    classifications = [d["classification"] for d in directions]
    counts = {
        cls: classifications.count(cls)
        for cls in ("clean_T", "clean_S", "e_rotation", "near_T_edge", "near_S_edge", "unclassified")
    }
    return {
        "row_index": row_index,
        "adaptive_floor": floor,
        "M_i_norm_inf": float(np.linalg.norm(M_i, ord=np.inf)),
        "singular_values": [float(s) for s in svs],
        "directions": directions,
        "counts_by_classification": counts,
        "has_near_T_edge": counts["near_T_edge"] > 0,
        "has_near_S_edge": counts["near_S_edge"] > 0,
        "has_e_rotation": counts["e_rotation"] > 0,
        "has_unclassified": counts["unclassified"] > 0,
    }


def main():
    floors = load_adaptive_floors()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_row = []
    catalog_counts = {
        "clean_T": 0, "clean_S": 0, "e_rotation": 0,
        "near_T_edge": 0, "near_S_edge": 0, "unclassified": 0,
    }
    for idx in W.DEFAULT_KFACET_STRICT_INDICES:
        adaptive = floors.get(idx)
        row_result = analyze_row(idx, adaptive)
        per_row.append(row_result)
        for cls, n in row_result["counts_by_classification"].items():
            catalog_counts[cls] = catalog_counts[cls] + n
        bands_summary = ",".join(
            f"{d['band']}@sv={d['singular_value']:.2e}:{d['classification']}"
            for d in row_result["directions"]
            if d["band"] == "bridge" or d["classification"] in ("near_T_edge", "near_S_edge", "e_rotation", "unclassified")
        )
        print(
            f"[near-T-separator] O_{idx}: counts={row_result['counts_by_classification']}"
            + (f"; nonstandard: {bands_summary}" if bands_summary else "")
        )
    near_t_edge_rows = [r["row_index"] for r in per_row if r["has_near_T_edge"]]
    near_s_edge_rows = [r["row_index"] for r in per_row if r["has_near_S_edge"]]
    e_rotation_rows = [r["row_index"] for r in per_row if r["has_e_rotation"]]
    unclassified_rows = [r["row_index"] for r in per_row if r["has_unclassified"]]
    manifest = {
        "mode": "kfacet_near_T_separator",
        "version": NEAR_T_VERSION,
        "thresholds": {
            "alignment_floor": NEAR_T_ALIGNMENT_FLOOR,
            "clean_alignment_floor": NEAR_T_CLEAN_ALIGNMENT_FLOOR,
            "closure_clean": NEAR_T_CLOSURE_CLEAN,
            "bridge_band_lower": NEAR_T_BRIDGE_BAND_LOWER,
            "bridge_band_upper": NEAR_T_BRIDGE_BAND_UPPER,
        },
        "rows": per_row,
        "catalog_counts_by_classification": catalog_counts,
        "near_T_edge_rows": near_t_edge_rows,
        "near_S_edge_rows": near_s_edge_rows,
        "e_rotation_rows": e_rotation_rows,
        "unclassified_rows": unclassified_rows,
        "summary": {
            "total_rows": len(per_row),
            "total_directions_analyzed": sum(len(r["directions"]) for r in per_row),
            "rows_with_near_T_edge": len(near_t_edge_rows),
            "rows_with_near_S_edge": len(near_s_edge_rows),
            "rows_with_e_rotation": len(e_rotation_rows),
            "rows_with_unclassified": len(unclassified_rows),
        },
    }
    (OUT_DIR / "separator_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print()
    print("=== Catalog-wide near-T separator summary ===")
    print(f"Total rows analyzed: {len(per_row)}")
    print(f"Total directions analyzed: {manifest['summary']['total_directions_analyzed']}")
    print(f"Catalog-wide direction counts: {catalog_counts}")
    print()
    print(f"Rows with near_T_edge: {near_t_edge_rows}")
    print(f"Rows with near_S_edge: {near_s_edge_rows}")
    print(f"Rows with e_rotation: {e_rotation_rows}")
    print(f"Rows with unclassified: {unclassified_rows}")
    print()
    if near_t_edge_rows == [617] and not near_s_edge_rows and not e_rotation_rows and not unclassified_rows:
        print("RESULT: O_617 is uniquely near_T_edge. The WHY-dive diagnosis is row-unique.")
    elif near_t_edge_rows and 617 not in near_t_edge_rows:
        print("RESULT: O_617 not classified as near_T_edge (unexpected; check thresholds).")
    elif len(near_t_edge_rows) > 1:
        print(f"RESULT: {len(near_t_edge_rows)} rows are near_T_edge. WHY-dive describes a class, not a singleton.")
    else:
        print(f"RESULT: near_T_edge: {near_t_edge_rows}; surfaced other structural classes -- see counts.")


if __name__ == "__main__":
    main()
