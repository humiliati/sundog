"""O_617 deep-dive: six no-integration probes consolidated into one synth table.

This is a one-shot diagnostic, not a registered subcommand. It pulls existing
matrices, the bridge audit receipt, and the sigma3-scan receipts. If any of
the six probes converge on a clear answer, the diagnostic can later be
promoted to a `kfacet-row-anatomy` subcommand of the workbench.

The probes triangulate the question: does the defective D3 at the bridge come
from (a) a localized geometric anomaly, (b) the bridge direction carrying a
non-D3 representation, (c) a near-bifurcation Floquet structure, or
(d) none of the above?
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent))
import isotrophy_workbench as W


ROOT = Path(__file__).resolve().parent.parent
SENTINEL_DIR = ROOT / "results/isotrophy/k-facet-v03-sentinel-sweep-calibrated/O617"
AUDIT_RECEIPT = ROOT / "results/isotrophy/k-facet-v03-bridge-audit-21/O617/bridge_audit_receipt.json"
SIGMA3_CSV = ROOT / "results/isotrophy/m3eq1-sigma3-precondition-fixed-inverse-orientation-25/residuals.csv"
CATALOG_PATH = ROOT / "docs/isotrophy/supplementary-A_periodic-3d_mirror.txt"


def load_inputs():
    M_i = np.load(SENTINEL_DIR / "M_i.npy")
    d3_names = ("e", "sigma3", "sigma3_sq", "F_beta", "F_beta_sigma3", "F_beta_sigma3_sq")
    d3_ops = {n: np.load(SENTINEL_DIR / f"D3_{n}.npy") for n in d3_names}
    audit = json.loads(AUDIT_RECEIPT.read_text(encoding="utf-8"))
    text = W.read_text(str(CATALOG_PATH))
    rows = W.parse_rows(text, "A")
    row_617 = next(r for r in rows if r.index == 617 and abs(r.m3 - 1.0) < 1e-12)
    return M_i, d3_ops, audit, row_617, rows


def probe_1_geometric_anatomy(audit):
    """Decompose v_bridge into position/velocity per body and check (12) symmetry."""
    bridge = audit["bridge_vectors"][0]
    v = np.array(bridge["right_singular_vector"], dtype=float)
    v_pos = v[:9].reshape(3, 3)  # 3 bodies x 3 axes (positions)
    v_vel = v[9:].reshape(3, 3)  # 3 bodies x 3 axes (velocities)
    pos_norms = np.linalg.norm(v_pos, axis=1)
    vel_norms = np.linalg.norm(v_vel, axis=1)
    # (12) symmetry: swap body 1 and body 2 in both position and velocity blocks.
    perm12 = np.array([1, 0, 2])
    v_pos_swapped = v_pos[perm12, :]
    v_vel_swapped = v_vel[perm12, :]
    sym_pos = (v_pos + v_pos_swapped) / 2.0
    asym_pos = (v_pos - v_pos_swapped) / 2.0
    sym_vel = (v_vel + v_vel_swapped) / 2.0
    asym_vel = (v_vel - v_vel_swapped) / 2.0
    return {
        "per_body_position_norms": [float(x) for x in pos_norms],
        "per_body_velocity_norms": [float(x) for x in vel_norms],
        "per_axis_position_norms": [float(x) for x in np.linalg.norm(v_pos, axis=0)],
        "per_axis_velocity_norms": [float(x) for x in np.linalg.norm(v_vel, axis=0)],
        "norm_12_symmetric_position": float(np.linalg.norm(sym_pos)),
        "norm_12_antisymmetric_position": float(np.linalg.norm(asym_pos)),
        "norm_12_symmetric_velocity": float(np.linalg.norm(sym_vel)),
        "norm_12_antisymmetric_velocity": float(np.linalg.norm(asym_vel)),
        "total_position_norm": float(np.linalg.norm(v_pos)),
        "total_velocity_norm": float(np.linalg.norm(v_vel)),
    }


def probe_2_sigma3_closure(M_i, d3_ops):
    """Compute sigma_3^3 - I on ambient and per-kernel-floor."""
    sigma3 = d3_ops["sigma3"]
    identity = np.eye(M_i.shape[0])
    s3_cubed_minus_I = sigma3 @ sigma3 @ sigma3 - identity
    ambient_residual = float(np.linalg.norm(s3_cubed_minus_I, ord=np.inf))

    # Decompose by kernel floor
    _u, svs, vh = np.linalg.svd(M_i - identity)
    floors = [1e-7, 3e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    per_floor = []
    for floor in floors:
        mask = svs < floor
        if mask.sum() == 0:
            per_floor.append({"floor": floor, "k_dim": 0, "residual": None})
            continue
        K = vh.T[:, mask]
        complement = vh.T[:, ~mask]
        # σ_3³ - I restricted to kernel basis and complement.
        in_kernel = float(np.linalg.norm(K.T @ s3_cubed_minus_I @ K, ord=np.inf))
        in_complement = float(np.linalg.norm(complement.T @ s3_cubed_minus_I @ complement, ord=np.inf))
        cross = float(np.linalg.norm(K.T @ s3_cubed_minus_I @ complement, ord=np.inf))
        per_floor.append(
            {
                "floor": floor,
                "k_dim": int(mask.sum()),
                "residual_in_kernel": in_kernel,
                "residual_in_complement": in_complement,
                "residual_cross_term": cross,
            }
        )
    return {
        "ambient_residual_inf": ambient_residual,
        "per_floor": per_floor,
    }


def probe_3_monodromy_spectrum(M_i):
    """Eigenvalue / Schur spectrum of M_i. Eig is diagnostic (per signed tweak)."""
    eigenvalues = np.linalg.eigvals(M_i)
    eigvals_abs = np.abs(eigenvalues)
    abs_minus_one = np.abs(eigenvalues - 1.0)
    near_unit_circle = [
        {
            "value_real": float(np.real(lam)),
            "value_imag": float(np.imag(lam)),
            "abs": float(abs(lam)),
            "abs_minus_one": float(abs(lam - 1.0)),
        }
        for lam in eigenvalues
        if abs(abs(lam) - 1.0) < 1e-2
    ]
    near_one = [e for e in near_unit_circle if e["abs_minus_one"] < 1e-2]
    cond = float(np.linalg.cond(M_i))
    # Real Schur decomposition: T is block-upper-triangular with 1x1 and 2x2 blocks.
    from scipy.linalg import schur
    T_real, Z_real = schur(M_i, output="real")
    # Identify 2x2 blocks (complex eigenvalue pairs).
    n = T_real.shape[0]
    blocks = []
    i = 0
    while i < n:
        if i + 1 < n and abs(T_real[i + 1, i]) > 1e-12:
            sub = T_real[i : i + 2, i : i + 2]
            trace = float(np.trace(sub))
            det = float(np.linalg.det(sub))
            blocks.append({"type": "2x2", "start": i, "trace": trace, "det": det})
            i += 2
        else:
            blocks.append({"type": "1x1", "start": i, "value": float(T_real[i, i])})
            i += 1
    return {
        "eigenvalue_abs_range": [float(eigvals_abs.min()), float(eigvals_abs.max())],
        "eigenvalues_near_unit_circle_count": len(near_unit_circle),
        "eigenvalues_near_one_count": len(near_one),
        "eigenvalues_near_unit_circle": near_unit_circle,
        "matrix_condition_number": cond,
        "schur_blocks_count_1x1": sum(1 for b in blocks if b["type"] == "1x1"),
        "schur_blocks_count_2x2": sum(1 for b in blocks if b["type"] == "2x2"),
        "schur_blocks_summary": blocks,
    }


def probe_4_invariant_sensitivity(M_i, audit, row):
    """Project v_bridge onto ∇H and ∇|L| at y0."""
    masses = row.masses
    _, x0_pos, v0 = W.expand_initial_state(row, center_com=True)
    y0 = W.pack_state(x0_pos, v0)
    rhs = W.rhs_factory(masses)  # rhs returns dy/dt = (v, a)

    def hamiltonian(y):
        x = y[:9].reshape(3, 3)
        v = y[9:].reshape(3, 3)
        kinetic = 0.5 * float(np.sum(masses[:, None] * v * v))
        potential = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                r = np.linalg.norm(x[i] - x[j])
                potential -= masses[i] * masses[j] / max(r, 1e-300)
        return kinetic + potential

    def angular_momentum(y):
        x = y[:9].reshape(3, 3)
        v = y[9:].reshape(3, 3)
        L = np.zeros(3)
        for i in range(3):
            L += masses[i] * np.cross(x[i], v[i])
        return L

    h = 1e-6
    grad_H = np.zeros(18)
    L0 = angular_momentum(y0)
    L0_mag = float(np.linalg.norm(L0))
    grad_L_mag = np.zeros(18)
    for i in range(18):
        e_i = np.zeros(18)
        e_i[i] = h
        grad_H[i] = (hamiltonian(y0 + e_i) - hamiltonian(y0 - e_i)) / (2 * h)
        L_plus = angular_momentum(y0 + e_i)
        L_minus = angular_momentum(y0 - e_i)
        grad_L_mag[i] = (np.linalg.norm(L_plus) - np.linalg.norm(L_minus)) / (2 * h)

    bridge = audit["bridge_vectors"][0]
    v_bridge = np.array(bridge["right_singular_vector"], dtype=float)
    v_bridge_unit = v_bridge / max(np.linalg.norm(v_bridge), 1e-300)

    projection_H = float(np.dot(grad_H, v_bridge_unit))
    grad_H_norm = float(np.linalg.norm(grad_H))
    cosangle_H = projection_H / max(grad_H_norm, 1e-300)

    projection_L = float(np.dot(grad_L_mag, v_bridge_unit))
    grad_L_norm = float(np.linalg.norm(grad_L_mag))
    cosangle_L = projection_L / max(grad_L_norm, 1e-300)

    return {
        "H_at_y0": float(hamiltonian(y0)),
        "L_at_y0": [float(x) for x in L0],
        "L_magnitude_at_y0": L0_mag,
        "grad_H_norm": grad_H_norm,
        "grad_L_magnitude_norm": grad_L_norm,
        "v_bridge_dot_grad_H": projection_H,
        "v_bridge_cos_angle_to_grad_H": cosangle_H,
        "v_bridge_dot_grad_L_magnitude": projection_L,
        "v_bridge_cos_angle_to_grad_L_magnitude": cosangle_L,
        "interpretation": (
            "Cosines should be near zero if v_bridge is tangent to the (E,|L|) "
            "level set. Nonzero values would indicate the bridge breaks "
            "conservation, which would be unphysical for a true Hamiltonian "
            "perturbation."
        ),
    }


def _csv_bool(value):
    return str(value).strip().lower() == "true"


def probe_5_cross_row_sigma3_residuals():
    """Pull orientation-aware sigma_3 admission residuals for the 21 strict rows."""
    strict_indices = set(W.DEFAULT_KFACET_STRICT_INDICES)
    rows_data = []
    with SIGMA3_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index"])
            if idx not in strict_indices:
                continue
            canonical_strict = _csv_bool(row["sigma_strict_single_curve_candidate"])
            opposite_strict = _csv_bool(row["sigma_opposite_strict_single_curve_candidate"])
            if canonical_strict and not opposite_strict:
                admission_orientation = "canonical"
                admission_residual = float(row["sigma_group_residual_inf"])
                admission_to_closure = float(row["sigma_group_to_closure"])
            elif opposite_strict and not canonical_strict:
                admission_orientation = "opposite"
                admission_residual = float(row["sigma_opposite_group_residual_inf"])
                admission_to_closure = float(row["sigma_opposite_group_to_closure"])
            elif canonical_strict and opposite_strict:
                canonical_residual = float(row["sigma_group_residual_inf"])
                opposite_residual = float(row["sigma_opposite_group_residual_inf"])
                if canonical_residual <= opposite_residual:
                    admission_orientation = "canonical"
                    admission_residual = canonical_residual
                    admission_to_closure = float(row["sigma_group_to_closure"])
                else:
                    admission_orientation = "opposite"
                    admission_residual = opposite_residual
                    admission_to_closure = float(row["sigma_opposite_group_to_closure"])
            else:
                admission_orientation = "none"
                admission_residual = float("nan")
                admission_to_closure = float("nan")
            rows_data.append(
                {
                    "index": idx,
                    "period": float(row["period"]),
                    "stability": row["stability"],
                    "sigma_group_residual_inf": float(row["sigma_group_residual_inf"]),
                    "sigma_group_to_closure": float(row["sigma_group_to_closure"]),
                    "sigma_group_rotation_angle_rad": float(row["sigma_group_rotation_angle_rad"]),
                    "sigma_strict_single_curve_candidate": canonical_strict,
                    "sigma_opposite_group_residual_inf": float(row["sigma_opposite_group_residual_inf"]),
                    "sigma_opposite_group_to_closure": float(row["sigma_opposite_group_to_closure"]),
                    "sigma_opposite_strict_single_curve_candidate": opposite_strict,
                    "admission_orientation": admission_orientation,
                    "admission_residual_inf": admission_residual,
                    "admission_to_closure": admission_to_closure,
                    "F_beta_residual_inf": float(row["F_beta_residual_inf"]),
                    "F_beta_to_closure": float(row["F_beta_to_closure"]),
                }
            )
    rows_data.sort(key=lambda r: r["admission_residual_inf"])
    o617 = next(r for r in rows_data if r["index"] == 617)
    admission_residuals = sorted(r["admission_residual_inf"] for r in rows_data)
    rank = admission_residuals.index(o617["admission_residual_inf"])
    fbeta_residuals = sorted(r["F_beta_residual_inf"] for r in rows_data)
    rank_fbeta = fbeta_residuals.index(o617["F_beta_residual_inf"])
    canonical_count = sum(1 for r in rows_data if r["admission_orientation"] == "canonical")
    opposite_count = sum(1 for r in rows_data if r["admission_orientation"] == "opposite")
    return {
        "rows_count": len(rows_data),
        "O_617": o617,
        "canonical_strict_count": canonical_count,
        "opposite_strict_count": opposite_count,
        "rank_in_admission_residual_sorted_asc": rank,
        "rank_in_F_beta_residual_sorted_asc": rank_fbeta,
        "min_admission_residual": float(admission_residuals[0]),
        "max_admission_residual": float(admission_residuals[-1]),
        "median_admission_residual": float(admission_residuals[len(admission_residuals) // 2]),
        "min_F_beta_residual": float(fbeta_residuals[0]),
        "max_F_beta_residual": float(fbeta_residuals[-1]),
        "median_F_beta_residual": float(fbeta_residuals[len(fbeta_residuals) // 2]),
        "full_distribution": rows_data,
    }


def probe_6_catalog_metadata(row):
    """Physical context for O_617."""
    return {
        "index": row.index,
        "m3": row.m3,
        "z0": row.z0,
        "vx": row.vx,
        "vy": row.vy,
        "vz": row.vz,
        "period": row.period,
        "stability": row.stability,
        "label": row.label,
        "source": row.source,
        "line_no": row.line_no,
        "inertia_degenerate_candidate": abs(row.m3 * row.z0 * row.z0 - 2.0) < 1e-3,
    }


def main():
    M_i, d3_ops, audit, row_617, _all_rows = load_inputs()
    results = {
        "row": row_617.label,
        "probe_1_geometric_anatomy": probe_1_geometric_anatomy(audit),
        "probe_2_sigma3_closure": probe_2_sigma3_closure(M_i, d3_ops),
        "probe_3_monodromy_spectrum": probe_3_monodromy_spectrum(M_i),
        "probe_4_invariant_sensitivity": probe_4_invariant_sensitivity(M_i, audit, row_617),
        "probe_5_cross_row_sigma3_residuals": probe_5_cross_row_sigma3_residuals(),
        "probe_6_catalog_metadata": probe_6_catalog_metadata(row_617),
    }
    out_dir = ROOT / "results/isotrophy/k-facet-v03-O617-deep-dive"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "deep_dive_receipt.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(f"O_617 deep dive complete. Receipt: {out_dir / 'deep_dive_receipt.json'}")
    # Print a compact synthesis table.
    p1 = results["probe_1_geometric_anatomy"]
    print("\n--- Probe 1: Geometric anatomy of v_bridge ---")
    print(f"  Per-body position norms: {p1['per_body_position_norms']}")
    print(f"  Per-body velocity norms: {p1['per_body_velocity_norms']}")
    print(f"  Per-axis position norms: {p1['per_axis_position_norms']}")
    print(f"  Per-axis velocity norms: {p1['per_axis_velocity_norms']}")
    print(f"  (12)-symmetric position norm:    {p1['norm_12_symmetric_position']:.4e}")
    print(f"  (12)-antisymmetric position norm: {p1['norm_12_antisymmetric_position']:.4e}")
    print(f"  (12)-symmetric velocity norm:    {p1['norm_12_symmetric_velocity']:.4e}")
    print(f"  (12)-antisymmetric velocity norm: {p1['norm_12_antisymmetric_velocity']:.4e}")
    print(f"  Total position norm: {p1['total_position_norm']:.4e}; total velocity norm: {p1['total_velocity_norm']:.4e}")

    p2 = results["probe_2_sigma3_closure"]
    print("\n--- Probe 2: sigma3^3 - I closure decomposition ---")
    print(f"  Ambient ||sigma3^3 - I||_inf: {p2['ambient_residual_inf']:.4e}")
    print(f"  {'floor':>10} {'k_dim':>6} {'in_kernel':>12} {'in_complement':>14} {'cross':>12}")
    for entry in p2["per_floor"]:
        if entry.get("residual_in_kernel") is None:
            continue
        print(
            f"  {entry['floor']:>10.0e} {entry['k_dim']:>6d} "
            f"{entry['residual_in_kernel']:>12.4e} "
            f"{entry['residual_in_complement']:>14.4e} "
            f"{entry['residual_cross_term']:>12.4e}"
        )

    p3 = results["probe_3_monodromy_spectrum"]
    print("\n--- Probe 3: Monodromy spectrum ---")
    print(f"  Eigenvalue |.| range: {p3['eigenvalue_abs_range']}")
    print(f"  On unit circle (|.|-1| < 1e-2): {p3['eigenvalues_near_unit_circle_count']}")
    print(f"  Near +1 (|.-1| < 1e-2): {p3['eigenvalues_near_one_count']}")
    print(f"  Schur 1x1 blocks: {p3['schur_blocks_count_1x1']}; 2x2 blocks: {p3['schur_blocks_count_2x2']}")
    print(f"  Condition number: {p3['matrix_condition_number']:.3e}")

    p4 = results["probe_4_invariant_sensitivity"]
    print("\n--- Probe 4: (E, |L|) sensitivity along v_bridge ---")
    print(f"  H(y0) = {p4['H_at_y0']:.6f};  |L|(y0) = {p4['L_magnitude_at_y0']:.6e}")
    print(f"  cos(v_bridge, grad H) = {p4['v_bridge_cos_angle_to_grad_H']:.4e}")
    print(f"  cos(v_bridge, grad |L|) = {p4['v_bridge_cos_angle_to_grad_L_magnitude']:.4e}")

    p5 = results["probe_5_cross_row_sigma3_residuals"]
    o617 = p5["O_617"]
    print("\n--- Probe 5: Cross-row orientation-aware sigma_3 and F_beta residuals (21 strict rows) ---")
    print(f"  Strict split: canonical={p5['canonical_strict_count']}; opposite={p5['opposite_strict_count']}")
    print(f"  O_617 admission orientation: {o617['admission_orientation']}")
    print(f"  O_617 admission residual:   {o617['admission_residual_inf']:.4e}")
    print(f"  O_617 admission to closure: {o617['admission_to_closure']:.4e}  (smaller=tighter)")
    print(f"  O_617 canonical residual:   {o617['sigma_group_residual_inf']:.4e}  (diagnostic only for opposite rows)")
    print(f"  Catalog range admission residual: [{p5['min_admission_residual']:.2e}, {p5['max_admission_residual']:.2e}]; median {p5['median_admission_residual']:.2e}")
    print(f"  O_617 rank in admission residual (asc): {p5['rank_in_admission_residual_sorted_asc']} of {p5['rows_count']}")
    print(f"  O_617 F_beta_residual:   {o617['F_beta_residual_inf']:.4e}")
    print(f"  Catalog range F_beta residual: [{p5['min_F_beta_residual']:.2e}, {p5['max_F_beta_residual']:.2e}]; median {p5['median_F_beta_residual']:.2e}")
    print(f"  O_617 rank in F_beta residual (asc): {p5['rank_in_F_beta_residual_sorted_asc']} of {p5['rows_count']}")

    p6 = results["probe_6_catalog_metadata"]
    print("\n--- Probe 6: Catalog metadata ---")
    print(f"  Label: {p6['label']}; stability: {p6['stability']}; period: {p6['period']:.6f}")
    print(f"  z0={p6['z0']:.6f}; vx={p6['vx']:.6f}; vy={p6['vy']:.6f}; vz={p6['vz']:.6f}")
    print(f"  inertia_degenerate_candidate: {p6['inertia_degenerate_candidate']}")


if __name__ == "__main__":
    main()
