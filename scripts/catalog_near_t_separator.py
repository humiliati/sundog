"""Catalog-wide isotypic-edge separator: extends the O_617 WHY-dive classifier across
the 21 strict G.2 rows. Asks one question: is O_617's near-S bridge direction
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
The v1 near-T reading was corrected by the v2 signed-isotypic pass. O_617's
bridge is near-S/sign-isotypic: F_beta acts as -I, sigma_3 acts almost as +I,
and sigma_3 drift keeps the direction above the clean relation floor. The
projector also catches a small `edge_E_other` contamination because the
underlying group action on the bridge-admitted kernel is only approximate.

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


NEAR_T_VERSION = "v0.3h-isotypic-edge-separator"
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


CLASSIFICATIONS = (
    "clean_T", "clean_S", "clean_E",
    "edge_T", "edge_S",
    "edge_E_as_T", "edge_E_as_S", "edge_E_other",
)


def vector_subspace_basis_local(projector, floor):
    """Orthonormal column basis for the range of `projector` via SVD truncation.

    Local copy of the workbench helper, included here to keep this script
    self-contained (so it does not silently depend on workbench internals
    beyond the documented public surface).
    """
    if projector.size == 0:
        return np.zeros((projector.shape[0], 0), dtype=float)
    u, sv, _vh = np.linalg.svd(projector, full_matrices=False)
    rank = int(np.sum(sv > floor))
    return u[:, :rank]


def isotypic_decomposition(K, sigma3, f_beta, projector_floor):
    """Decompose kernel basis K (18 x k) into T, S, E isotypic 18-dim bases."""
    k_dim = K.shape[1]
    if k_dim == 0:
        return (
            np.zeros((K.shape[0], 0), dtype=float),
            np.zeros((K.shape[0], 0), dtype=float),
            np.zeros((K.shape[0], 0), dtype=float),
        )
    s3_K = K.T @ sigma3 @ K
    fb_K = K.T @ f_beta @ K
    I_K = np.eye(k_dim)
    s3_sq_K = s3_K @ s3_K
    fb_s3_K = fb_K @ s3_K
    fb_s3_sq_K = fb_K @ s3_sq_K
    P_T = (I_K + s3_K + s3_sq_K + fb_K + fb_s3_K + fb_s3_sq_K) / 6.0
    P_S = (I_K + s3_K + s3_sq_K - fb_K - fb_s3_K - fb_s3_sq_K) / 6.0
    P_E = I_K - P_T - P_S
    T_basis_K = vector_subspace_basis_local(P_T, projector_floor)
    S_basis_K = vector_subspace_basis_local(P_S, projector_floor)
    E_basis_K = vector_subspace_basis_local(P_E, projector_floor)
    return K @ T_basis_K, K @ S_basis_K, K @ E_basis_K


def classify_isotypic_direction(assigned_isotype, align_s3, align_fb, closure_s3, closure_fb):
    """Pre-registered classification given the projector-assigned isotype."""
    s3_clean = closure_s3 < NEAR_T_CLOSURE_CLEAN
    fb_clean = closure_fb < NEAR_T_CLOSURE_CLEAN
    if assigned_isotype == "T":
        if (
            s3_clean and fb_clean
            and align_s3 >= NEAR_T_CLEAN_ALIGNMENT_FLOOR
            and align_fb >= NEAR_T_CLEAN_ALIGNMENT_FLOOR
        ):
            return "clean_T"
        return "edge_T"
    if assigned_isotype == "S":
        if (
            s3_clean and fb_clean
            and align_s3 >= NEAR_T_CLEAN_ALIGNMENT_FLOOR
            and align_fb <= -NEAR_T_CLEAN_ALIGNMENT_FLOOR
        ):
            return "clean_S"
        return "edge_S"
    # assigned E
    if (
        s3_clean and fb_clean
        and abs(align_s3) < NEAR_T_ALIGNMENT_FLOOR
    ):
        return "clean_E"
    if align_s3 >= NEAR_T_ALIGNMENT_FLOOR and align_fb >= NEAR_T_ALIGNMENT_FLOOR:
        return "edge_E_as_T"
    if align_s3 >= NEAR_T_ALIGNMENT_FLOOR and align_fb <= -NEAR_T_ALIGNMENT_FLOOR:
        return "edge_E_as_S"
    return "edge_E_other"


def analyze_row(row_index, adaptive_floor):
    """Per-row table using the isotypic-projector classifier (v2)."""
    row_dir = SENTINEL_ROOT / f"O{row_index}"
    M_i = np.load(row_dir / "M_i.npy")
    sigma3 = np.load(row_dir / "D3_sigma3.npy")
    f_beta = np.load(row_dir / "D3_F_beta.npy")
    identity = np.eye(M_i.shape[0])
    _u, svs, vh = np.linalg.svd(M_i - identity)
    # Kernel basis at the bridge-admit upper bound; this admits any bridge
    # SV in (adaptive_floor, 1e-3) so the projector sees it.
    mask = svs < NEAR_T_BRIDGE_BAND_UPPER
    K = vh.T[:, mask]
    kernel_dim = int(K.shape[1])
    floor = float(adaptive_floor) if adaptive_floor is not None else NEAR_T_BRIDGE_BAND_LOWER

    T_basis, S_basis, E_basis = isotypic_decomposition(
        K, sigma3, f_beta, projector_floor=NEAR_T_CLOSURE_CLEAN
    )
    sigma3_cubed = sigma3 @ sigma3 @ sigma3
    f_beta_squared = f_beta @ f_beta

    def evaluate(basis, label):
        out = []
        for j in range(basis.shape[1]):
            v = basis[:, j]
            v_norm = np.linalg.norm(v)
            if v_norm <= 1e-300:
                continue
            v = v / v_norm
            align_s3 = float(np.dot(v, sigma3 @ v))
            align_fb = float(np.dot(v, f_beta @ v))
            closure_s3 = float(np.linalg.norm(sigma3_cubed @ v - v))
            closure_fb = float(np.linalg.norm(f_beta_squared @ v - v))
            residual_M = float(np.linalg.norm(M_i @ v - v))
            classification = classify_isotypic_direction(
                label, align_s3, align_fb, closure_s3, closure_fb
            )
            out.append({
                "assigned_isotype": label,
                "isotypic_index": int(j),
                "alignment_sigma3": align_s3,
                "alignment_F_beta": align_fb,
                "closure_sigma3": closure_s3,
                "closure_F_beta": closure_fb,
                "residual_M_minus_I": residual_M,
                "classification": classification,
            })
        return out

    directions = (
        evaluate(T_basis, "T") + evaluate(S_basis, "S") + evaluate(E_basis, "E")
    )
    counts = {cls: 0 for cls in CLASSIFICATIONS}
    for direction in directions:
        counts[direction["classification"]] = counts.get(direction["classification"], 0) + 1
    isotypic_dims = {
        "T": int(T_basis.shape[1]),
        "S": int(S_basis.shape[1]),
        "E": int(E_basis.shape[1]),
    }

    has_edge_E_as_T = counts.get("edge_E_as_T", 0) > 0
    has_edge_E_as_S = counts.get("edge_E_as_S", 0) > 0
    has_edge_E_other = counts.get("edge_E_other", 0) > 0
    has_edge_T = counts.get("edge_T", 0) > 0
    has_edge_S = counts.get("edge_S", 0) > 0
    has_clean_E = counts.get("clean_E", 0) > 0

    return {
        "row_index": row_index,
        "adaptive_floor": floor,
        "kernel_dim_at_bridge_admit_floor": kernel_dim,
        "isotypic_dims": isotypic_dims,
        "M_i_norm_inf": float(np.linalg.norm(M_i, ord=np.inf)),
        "singular_values": [float(s) for s in svs],
        "directions": directions,
        "counts_by_classification": counts,
        "has_edge_E_as_T": has_edge_E_as_T,
        "has_edge_E_as_S": has_edge_E_as_S,
        "has_edge_E_other": has_edge_E_other,
        "has_edge_T": has_edge_T,
        "has_edge_S": has_edge_S,
        "has_clean_E": has_clean_E,
    }


def main():
    floors = load_adaptive_floors()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_row = []
    catalog_counts = {cls: 0 for cls in CLASSIFICATIONS}
    for idx in W.DEFAULT_KFACET_STRICT_INDICES:
        adaptive = floors.get(idx)
        row_result = analyze_row(idx, adaptive)
        per_row.append(row_result)
        for cls, n in row_result["counts_by_classification"].items():
            catalog_counts[cls] = catalog_counts.get(cls, 0) + n
        anomalies = [d["classification"] for d in row_result["directions"]
                     if d["classification"].startswith("edge_") or d["classification"] == "clean_E"]
        anomaly_str = f"; anomalies={anomalies}" if anomalies else ""
        print(
            f"[isotypic-edge-separator] O_{idx}: "
            f"isotypic_dims={row_result['isotypic_dims']} "
            f"counts={row_result['counts_by_classification']}"
            f"{anomaly_str}"
        )
    edge_E_as_T_rows = [r["row_index"] for r in per_row if r["has_edge_E_as_T"]]
    edge_E_as_S_rows = [r["row_index"] for r in per_row if r["has_edge_E_as_S"]]
    edge_E_other_rows = [r["row_index"] for r in per_row if r["has_edge_E_other"]]
    edge_T_rows = [r["row_index"] for r in per_row if r["has_edge_T"]]
    edge_S_rows = [r["row_index"] for r in per_row if r["has_edge_S"]]
    clean_E_rows = [r["row_index"] for r in per_row if r["has_clean_E"]]
    manifest = {
        "mode": "kfacet_isotypic_edge_separator",
        "version": NEAR_T_VERSION,
        "classifier": "isotypic_projector_v2",
        "thresholds": {
            "alignment_floor": NEAR_T_ALIGNMENT_FLOOR,
            "clean_alignment_floor": NEAR_T_CLEAN_ALIGNMENT_FLOOR,
            "closure_clean": NEAR_T_CLOSURE_CLEAN,
            "bridge_band_lower": NEAR_T_BRIDGE_BAND_LOWER,
            "bridge_band_upper": NEAR_T_BRIDGE_BAND_UPPER,
        },
        "rows": per_row,
        "catalog_counts_by_classification": catalog_counts,
        "edge_E_as_T_rows": edge_E_as_T_rows,
        "edge_E_as_S_rows": edge_E_as_S_rows,
        "edge_E_other_rows": edge_E_other_rows,
        "edge_T_rows": edge_T_rows,
        "edge_S_rows": edge_S_rows,
        "clean_E_rows": clean_E_rows,
        "summary": {
            "total_rows": len(per_row),
            "total_directions_analyzed": sum(len(r["directions"]) for r in per_row),
            "rows_with_edge_E_as_T": len(edge_E_as_T_rows),
            "rows_with_edge_E_as_S": len(edge_E_as_S_rows),
            "rows_with_edge_E_other": len(edge_E_other_rows),
            "rows_with_edge_T": len(edge_T_rows),
            "rows_with_edge_S": len(edge_S_rows),
            "rows_with_clean_E": len(clean_E_rows),
        },
    }
    (OUT_DIR / "separator_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print()
    print("=== Catalog-wide isotypic-edge separator (v2 isotypic projector) summary ===")
    print(f"Total rows analyzed: {len(per_row)}")
    print(f"Total directions analyzed: {manifest['summary']['total_directions_analyzed']}")
    print(f"Catalog-wide direction counts: {catalog_counts}")
    print()
    print(f"Rows with edge_E_as_T:                 {edge_E_as_T_rows}")
    print(f"Rows with edge_E_as_S:                 {edge_E_as_S_rows}")
    print(f"Rows with edge_E_other:                {edge_E_other_rows}")
    print(f"Rows with edge_T:                      {edge_T_rows}")
    print(f"Rows with edge_S:                      {edge_S_rows}")
    print(f"Rows with clean_E:                     {clean_E_rows}")
    print()
    only_o617_edge = (
        edge_S_rows == [617]
        and edge_E_other_rows == [617]
        and not edge_E_as_T_rows
        and not edge_E_as_S_rows
        and not edge_T_rows
        and not clean_E_rows
    )
    if only_o617_edge:
        print("RESULT: O_617 is the only structural-edge row across the catalog.")
        print("        The bridge is near-S/sign-isotypic with a small edge_E_other")
        print("        projector-contamination residue; every other row is clean T/S only.")
    elif edge_S_rows and 617 not in edge_S_rows:
        print("RESULT: O_617 not classified as edge_S (unexpected; check thresholds).")
    elif len(set(edge_S_rows + edge_E_other_rows + edge_T_rows + edge_E_as_T_rows + edge_E_as_S_rows)) > 1:
        rows = sorted(set(edge_S_rows + edge_E_other_rows + edge_T_rows + edge_E_as_T_rows + edge_E_as_S_rows))
        print(f"RESULT: multiple structural-edge rows detected: {rows}. WHY-dive describes a class.")
    else:
        print(
            "RESULT: edge summary: "
            f"edge_S={edge_S_rows}, edge_E_other={edge_E_other_rows}, "
            f"edge_T={edge_T_rows}, edge_E_as_T={edge_E_as_T_rows}, "
            f"edge_E_as_S={edge_E_as_S_rows}."
        )


if __name__ == "__main__":
    main()
