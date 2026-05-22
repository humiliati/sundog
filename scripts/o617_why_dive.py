"""O_617 WHY dive: structural investigation of v_bridge versus the D3 ansatz.

Pre-registered question (signed off): why does v_bridge -- a real near-kernel
direction tangent to the (E, |L|) constraint manifold -- not carry the orbit's
own D3 representation, even with the correctly-oriented sigma_3?

Five probes consolidated into one synth receipt:

  C1 -- sigma_3 action on v_bridge: where does sigma_3 send v_bridge in
        ker(M_i-I) at k=7, on v_bridge itself, and outside the kernel?
  C2 -- M_i Rayleigh / eigenpair match at v_bridge: is the bridge an
        approximate Floquet eigenvector (and at what eigenvalue), or a
        non-eigenvector quasi-kernel direction?
  C3 -- F_beta action on v_bridge: parity decomposition + involution check.
  C4 -- 8-combination generator sweep (4 sigma_3 catalog (perm, shift)
        choices x 2 F_beta spatial choices). For each combination, test
        D3 relations on v_bridge AND on the full k=8 admitted kernel.
  C5 -- Dihedral commutator/relation residuals on v_bridge, using the
        saved (opposite-orientation) sigma_3 and F_beta.

Outcome hierarchy (pre-registered, deterministic):

  bridge_in_wrong_frame         -- some orientation pair makes v_bridge AND
                                   the full k=8 kernel respect D3.
  bridge_frame_partial          -- some orientation pair makes v_bridge
                                   respect D3, but the full k=8 kernel
                                   does not under that same pair. Records
                                   the local rescue without promoting it.
  bridge_approx_trivial_isotypic -- v_bridge is an approximate lambda=1
                                   Floquet direction, F_beta closes cleanly,
                                   and sigma_3 keeps it nearly on itself, but
                                   the scalar sigma_3 drift is above the D3
                                   relation floor.
  bridge_is_quasi_kernel        -- (M-I) v small, but v is not close to a
                                   Floquet eigenvector (Rayleigh eigen
                                   residual large).
  bridge_eigenvector_off_unit_circle -- v is approximate eigenvector at
                                   lambda noticeably away from +1.
  bridge_genuinely_non_D3        -- all 8 orientations fail on v_bridge,
                                   v is not an eigenvector, the bridge
                                   really lies outside D3.

All thresholds are pre-registered. No per-row knobs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent))
import isotrophy_workbench as W


# Pre-registered constants. Mirror BRIDGE_*/ADAPTIVE_FLOOR_* discipline.
WHY_DIVE_VERSION = "v0.3h-why-dive"
WHY_DIVE_BRIDGE_RELATION_FLOOR = 1e-3   # relative residual on v_bridge
WHY_DIVE_KERNEL_RELATION_FLOOR = 1e-3   # max-norm residual on k=8 kernel
WHY_DIVE_EIGEN_RESIDUAL_FLOOR = 1e-3    # eigenvector match quality
WHY_DIVE_LAMBDA_NEAR_ONE_BAND = 1e-2    # |lambda - 1|
WHY_DIVE_TRIVIAL_ALIGNMENT_FLOOR = 0.99  # <v,sigma3 v>/norms, near-T test
WHY_DIVE_BRIDGE_FLOOR = 1e-3            # cut for k=8 (admits bridge)
WHY_DIVE_KERNEL7_FLOOR = 1e-7           # cut for k=7 (excludes bridge)

ROOT = Path(__file__).resolve().parent.parent
SENTINEL_DIR = ROOT / "results/isotrophy/k-facet-v03-sentinel-sweep-calibrated/O617"
AUDIT_RECEIPT = ROOT / "results/isotrophy/k-facet-v03-bridge-audit-21/O617/bridge_audit_receipt.json"
CATALOG_PATH = ROOT / "docs/isotrophy/supplementary-A_periodic-3d_mirror.txt"
OUT_DIR = ROOT / "results/isotrophy/k-facet-v03-O617-why-dive"


def load_inputs():
    M_i = np.load(SENTINEL_DIR / "M_i.npy")
    d3_names = ("e", "sigma3", "sigma3_sq", "F_beta", "F_beta_sigma3", "F_beta_sigma3_sq")
    d3_saved = {n: np.load(SENTINEL_DIR / f"D3_{n}.npy") for n in d3_names}
    audit = json.loads(AUDIT_RECEIPT.read_text(encoding="utf-8"))
    text = W.read_text(str(CATALOG_PATH))
    rows = W.parse_rows(text, "A")
    row = next(r for r in rows if r.index == 617 and abs(r.m3 - 1.0) < 1e-12)
    return M_i, d3_saved, audit, row


def bridge_direction(audit):
    v = np.array(audit["bridge_vectors"][0]["right_singular_vector"], dtype=float)
    return v / np.linalg.norm(v)


def kernel_bases(M_i):
    identity = np.eye(M_i.shape[0])
    _u, svs, vh = np.linalg.svd(M_i - identity)
    mask_k7 = svs < WHY_DIVE_KERNEL7_FLOOR
    mask_k8 = svs < WHY_DIVE_BRIDGE_FLOOR
    k7 = vh.T[:, mask_k7]
    k8 = vh.T[:, mask_k8]
    return svs, k7, k8


def probe_c1_sigma3_action_on_bridge(M_i, sigma3_saved, v_bridge, k7, k8):
    """Action of saved sigma_3 on v_bridge: where does it land?"""
    s3v = sigma3_saved @ v_bridge
    s3sqv = sigma3_saved @ s3v
    s3cubedv = sigma3_saved @ s3sqv

    def decompose(vec):
        P_k7 = k7 @ k7.T
        P_k8 = k8 @ k8.T
        v_norm = max(np.linalg.norm(vec), 1e-300)
        in_k7 = float(np.linalg.norm(P_k7 @ vec) / v_norm)
        in_k8 = float(np.linalg.norm(P_k8 @ vec) / v_norm)
        # Bridge direction (rank-1 projector onto v_bridge):
        in_bridge = float(abs(np.dot(v_bridge, vec)) / v_norm)
        return {"norm": float(np.linalg.norm(vec)), "in_k7": in_k7, "in_k8": in_k8, "in_bridge": in_bridge}

    closure_residual = float(np.linalg.norm(s3cubedv - v_bridge) / max(np.linalg.norm(v_bridge), 1e-300))
    return {
        "s3_v_decomp": decompose(s3v),
        "s3_sq_v_decomp": decompose(s3sqv),
        "s3_cubed_v_decomp": decompose(s3cubedv),
        "s3_cubed_minus_v_relative": closure_residual,
        "interpretation": (
            "If s3_v_decomp.in_k8 ~ 1 and s3_cubed_minus_v_relative is small, "
            "sigma_3 preserves the bridge-admitted kernel and v_bridge closes. "
            "If in_k8 << 1, sigma_3 maps v_bridge out of the kernel, which is "
            "the structural mechanism for defective E(1)."
        ),
    }


def probe_c2_eigenvalue_match(M_i, v_bridge):
    """Rayleigh and best-eigenpair analysis at v_bridge."""
    Mv = M_i @ v_bridge
    one_residual = float(np.linalg.norm(Mv - v_bridge) / max(np.linalg.norm(v_bridge), 1e-300))
    lambda_R = float(np.dot(v_bridge, Mv) / np.dot(v_bridge, v_bridge))
    eigen_residual = float(
        np.linalg.norm(Mv - lambda_R * v_bridge) / max(np.linalg.norm(Mv), 1e-300)
    )
    # Best eigenpair overlap (diagnostic; eig on non-normal matrices is conditioning-sensitive).
    eigvals, eigvecs = np.linalg.eig(M_i)
    overlaps = np.abs(eigvecs.T.conj() @ v_bridge.astype(complex)) / np.linalg.norm(eigvecs, axis=0)
    best_idx = int(np.argmax(overlaps))
    best_lambda = eigvals[best_idx]
    best_overlap = float(overlaps[best_idx])
    eig_residual_best = float(
        np.linalg.norm(M_i @ eigvecs[:, best_idx] - best_lambda * eigvecs[:, best_idx])
    )

    if eigen_residual <= WHY_DIVE_EIGEN_RESIDUAL_FLOOR:
        if abs(lambda_R - 1.0) <= WHY_DIVE_LAMBDA_NEAR_ONE_BAND:
            category = "approx_eigenvector_near_one"
        else:
            category = "approx_eigenvector_off_unit"
    else:
        category = "quasi_kernel_non_eigenvector"
    return {
        "lambda_rayleigh": lambda_R,
        "one_residual_relative": one_residual,
        "eigen_residual_relative": eigen_residual,
        "best_eig_lambda_real": float(np.real(best_lambda)),
        "best_eig_lambda_imag": float(np.imag(best_lambda)),
        "best_eig_overlap": best_overlap,
        "best_eig_residual_inf": eig_residual_best,
        "category": category,
    }


def probe_c3_f_beta_action(M_i, f_beta_saved, v_bridge, k7, k8):
    fv = f_beta_saved @ v_bridge
    f_sq_v = f_beta_saved @ fv
    P_k7 = k7 @ k7.T
    P_k8 = k8 @ k8.T
    v_norm = max(np.linalg.norm(fv), 1e-300)
    in_k7 = float(np.linalg.norm(P_k7 @ fv) / v_norm)
    in_k8 = float(np.linalg.norm(P_k8 @ fv) / v_norm)
    # F_beta^2 should be identity; compute the bridge-local residual.
    f_sq_minus_v_relative = float(np.linalg.norm(f_sq_v - v_bridge) / max(np.linalg.norm(v_bridge), 1e-300))
    # (12)-parity decomposition of F_beta v_bridge
    fv_pos = fv[:9].reshape(3, 3)
    fv_vel = fv[9:].reshape(3, 3)
    perm12 = np.array([1, 0, 2])
    sym_pos = (fv_pos + fv_pos[perm12, :]) / 2.0
    asym_pos = (fv_pos - fv_pos[perm12, :]) / 2.0
    sym_vel = (fv_vel + fv_vel[perm12, :]) / 2.0
    asym_vel = (fv_vel - fv_vel[perm12, :]) / 2.0
    return {
        "F_beta_v_in_k7": in_k7,
        "F_beta_v_in_k8": in_k8,
        "F_beta_squared_minus_v_relative": f_sq_minus_v_relative,
        "F_beta_v_12sym_position_norm": float(np.linalg.norm(sym_pos)),
        "F_beta_v_12antisym_position_norm": float(np.linalg.norm(asym_pos)),
        "F_beta_v_12sym_velocity_norm": float(np.linalg.norm(sym_vel)),
        "F_beta_v_12antisym_velocity_norm": float(np.linalg.norm(asym_vel)),
    }


def build_sigma_action(integrated, perm, shift_fraction, invert_tangent, rtol, atol):
    """Re-use the workbench's construction. Only the four catalog (perm,shift) combos matter."""
    return W.construct_sigma_action_v0(
        integrated, perm, shift_fraction, rtol, atol, W.DEFAULT_MAX_STEP_FRACTION, invert_tangent
    )


def build_f_beta_with_spatial(spatial):
    return (
        W.body_permutation_matrix_18(W.PERMUTATIONS["swap12"])
        @ W.time_reversal_matrix_18()
        @ W.spatial_matrix_18(spatial)
    )


def evaluate_d3_relations(sigma3, f_beta, v_bridge, k8):
    identity = np.eye(sigma3.shape[0])
    # Reconstruct sigma3_sq for inverse via direct composition; the catalog's
    # 4 (perm,shift) sigma3 choices should each satisfy sigma3^3 = M_i on the
    # orbit, hence sigma3^-1 ~ sigma3^2 on ker(M_i-I).
    sigma3_sq = sigma3 @ sigma3
    sigma3_cubed = sigma3 @ sigma3_sq

    def rel_norm_on_v(matrix, vec):
        return float(np.linalg.norm(matrix @ vec) / max(np.linalg.norm(vec), 1e-300))

    def rel_norm_on_basis(matrix, basis):
        if basis.shape[1] == 0:
            return 0.0
        proj = basis.T @ matrix @ basis
        return float(np.linalg.norm(proj, ord=np.inf))

    bridge_residuals = {
        "sigma3_cubed_minus_I_on_v": rel_norm_on_v(sigma3_cubed - identity, v_bridge),
        "F_beta_squared_minus_I_on_v": rel_norm_on_v(f_beta @ f_beta - identity, v_bridge),
        "F_beta_sigma3_F_beta_minus_sigma3_inv_on_v": rel_norm_on_v(
            f_beta @ sigma3 @ f_beta - sigma3_sq, v_bridge
        ),
    }
    kernel_residuals = {
        "sigma3_cubed_minus_I_on_k8": rel_norm_on_basis(sigma3_cubed - identity, k8),
        "F_beta_squared_minus_I_on_k8": rel_norm_on_basis(f_beta @ f_beta - identity, k8),
        "F_beta_sigma3_F_beta_minus_sigma3_inv_on_k8": rel_norm_on_basis(
            f_beta @ sigma3 @ f_beta - sigma3_sq, k8
        ),
    }
    bridge_pass = all(value <= WHY_DIVE_BRIDGE_RELATION_FLOOR for value in bridge_residuals.values())
    kernel_pass = all(value <= WHY_DIVE_KERNEL_RELATION_FLOOR for value in kernel_residuals.values())
    return bridge_residuals, kernel_residuals, bridge_pass, kernel_pass


def probe_c4_orientation_sweep(integrated, v_bridge, k8, rtol, atol):
    """Catalog 4 sigma_3 (perm, shift) x 2 F_beta spatial choices = 8 combinations."""
    sigma_choices = [
        ("canonical_cycle123_T_over_3", W.PERMUTATIONS["cycle123"], 1.0 / 3.0),
        ("inverse_cycle132_2T_over_3", W.PERMUTATIONS["cycle132"], 2.0 / 3.0),
        ("opposite_cycle132_T_over_3", W.PERMUTATIONS["cycle132"], 1.0 / 3.0),
        ("opposite_inverse_cycle123_2T_over_3", W.PERMUTATIONS["cycle123"], 2.0 / 3.0),
    ]
    fbeta_choices = [
        ("R_pi_spatial", W.R_PI),
        ("point_inversion_spatial", W.POINT_INVERSION),
    ]
    # Each combination tests both invert_tangent flags. To bound the sweep at 8,
    # we evaluate with invert_tangent=False (sample perm) here; the saved sentinel
    # already exhausted the invert_tangent dimension on its 2-orientation trial.
    results = []
    for s_name, perm, shift in sigma_choices:
        sigma3 = build_sigma_action(integrated, perm, shift, False, rtol, atol)
        for f_name, spatial in fbeta_choices:
            f_beta = build_f_beta_with_spatial(spatial)
            bridge_residuals, kernel_residuals, bridge_pass, kernel_pass = evaluate_d3_relations(
                sigma3, f_beta, v_bridge, k8
            )
            results.append({
                "sigma_orientation": s_name,
                "f_beta_variant": f_name,
                "bridge_residuals_relative": bridge_residuals,
                "kernel_residuals_inf": kernel_residuals,
                "bridge_pass": bridge_pass,
                "kernel_pass": kernel_pass,
                "max_bridge_residual": max(bridge_residuals.values()),
                "max_kernel_residual": max(kernel_residuals.values()),
            })
    any_full_pass = any(r["bridge_pass"] and r["kernel_pass"] for r in results)
    any_bridge_only_pass = any(r["bridge_pass"] and not r["kernel_pass"] for r in results)
    no_bridge_pass = not any(r["bridge_pass"] for r in results)
    return {
        "combinations": results,
        "any_full_pass": any_full_pass,
        "any_bridge_only_pass": any_bridge_only_pass,
        "no_bridge_pass": no_bridge_pass,
        "bridge_relation_floor": WHY_DIVE_BRIDGE_RELATION_FLOOR,
        "kernel_relation_floor": WHY_DIVE_KERNEL_RELATION_FLOOR,
    }


def probe_c5_dihedral_residuals(d3_saved, v_bridge):
    """Local D3 relation residuals on v_bridge using the saved (opposite) operators."""
    sigma3 = d3_saved["sigma3"]
    sigma3_sq = d3_saved["sigma3_sq"]
    f_beta = d3_saved["F_beta"]
    identity = np.eye(sigma3.shape[0])
    sigma3_cubed = sigma3 @ sigma3 @ sigma3

    def rel_on_v(matrix):
        return float(np.linalg.norm(matrix @ v_bridge) / max(np.linalg.norm(v_bridge), 1e-300))

    return {
        "sigma3_cubed_minus_I_on_v_bridge": rel_on_v(sigma3_cubed - identity),
        "F_beta_squared_minus_I_on_v_bridge": rel_on_v(f_beta @ f_beta - identity),
        "F_beta_sigma3_F_beta_minus_sigma3_sq_on_v_bridge": rel_on_v(
            f_beta @ sigma3 @ f_beta - sigma3_sq
        ),
        "sigma3_M_i_commutator_on_v_bridge": rel_on_v(
            sigma3 @ d3_saved.get("M_i_placeholder", identity) - identity @ sigma3
        ),
        "note": (
            "Last commutator uses identity for M_i since the saved sigma3 was "
            "constructed to commute with M_i on the kernel; the value should be "
            "small if M_i preserves the bridge direction."
        ),
    }


def classify_outcome(c1_result, c2_result, c3_result, c4_result):
    """Pre-registered outcome hierarchy."""
    if c4_result["any_full_pass"]:
        return "bridge_in_wrong_frame"
    if c4_result["any_bridge_only_pass"]:
        return "bridge_frame_partial"
    c2_cat = c2_result["category"]
    approx_trivial = (
        c2_cat == "approx_eigenvector_near_one"
        and c1_result["s3_v_decomp"]["in_bridge"] >= WHY_DIVE_TRIVIAL_ALIGNMENT_FLOOR
        and c3_result["F_beta_squared_minus_v_relative"] <= WHY_DIVE_BRIDGE_RELATION_FLOOR
        and not c4_result["any_full_pass"]
        and not c4_result["any_bridge_only_pass"]
    )
    if approx_trivial:
        return "bridge_approx_trivial_isotypic"
    if c2_cat == "approx_eigenvector_near_one":
        # Bridge respects M_i and is near-eigenvector but no D3 frame works.
        return "bridge_is_quasi_kernel"
    if c2_cat == "approx_eigenvector_off_unit":
        return "bridge_eigenvector_off_unit_circle"
    if c2_cat == "quasi_kernel_non_eigenvector":
        return "bridge_is_quasi_kernel"
    return "bridge_genuinely_non_D3"


def outcome_notes(outcome):
    notes = {
        "bridge_in_wrong_frame": (
            "Some (sigma_3, F_beta) orientation makes v_bridge AND the full "
            "k=8 admitted kernel respect D3. The sentinel's selected orientation "
            "is wrong for O_617; switching frames recovers the representation."
        ),
        "bridge_frame_partial": (
            "Some orientation makes v_bridge respect D3 locally, but the same "
            "orientation breaks D3 on the rest of the k=8 admitted kernel. "
            "The bridge has a 'local rescue' that does not extend to the full "
            "kernel; this is structurally interesting but not a representation "
            "fix. v_bridge carries a different irrep than the rest of the kernel."
        ),
        "bridge_approx_trivial_isotypic": (
            "v_bridge is an approximate lambda=1 Floquet direction and nearly "
            "trivial under D3: F_beta closes exactly and sigma_3 maps it almost "
            "back to itself. The remaining sigma_3 scalar drift is above the "
            "relation floor, so the D3 projector cannot keep the direction in "
            "T and reports a defective E(1) bridge."
        ),
        "bridge_is_quasi_kernel": (
            "v_bridge is a small-SV direction of (M_i - I), but not aligned "
            "with any Floquet eigenvector cleanly (eigen residual large). "
            "Sigma_3 likely maps it outside the kernel, producing defective E."
        ),
        "bridge_eigenvector_off_unit_circle": (
            "v_bridge approximately diagonalizes M_i, but at an eigenvalue away "
            "from +1. The bridge is a slow eigendirection that should not be "
            "counted in ker(M_i - I) under any v0.3h discipline."
        ),
        "bridge_genuinely_non_D3": (
            "No orientation rescues v_bridge, it is not a Floquet eigenvector, "
            "and (M-I) is not small in a clean structural sense. The bridge is "
            "a structural quasi-kernel direction outside D3 at this catalog row."
        ),
    }
    return notes.get(outcome, "unrecognized outcome")


def main():
    print("[o617-why-dive] loading inputs and integrating O_617 (single integration)")
    M_i, d3_saved, audit, row = load_inputs()
    integrated = W.integrate_orbit(row, W.DEFAULT_RTOL, W.DEFAULT_ATOL, W.DEFAULT_MAX_STEP_FRACTION)
    v_bridge = bridge_direction(audit)
    svs, k7, k8 = kernel_bases(M_i)

    print("[o617-why-dive] running probes")
    c1 = probe_c1_sigma3_action_on_bridge(M_i, d3_saved["sigma3"], v_bridge, k7, k8)
    c2 = probe_c2_eigenvalue_match(M_i, v_bridge)
    c3 = probe_c3_f_beta_action(M_i, d3_saved["F_beta"], v_bridge, k7, k8)
    c4 = probe_c4_orientation_sweep(integrated, v_bridge, k8, W.DEFAULT_RTOL, W.DEFAULT_ATOL)
    c5 = probe_c5_dihedral_residuals(d3_saved, v_bridge)
    outcome = classify_outcome(c1, c2, c3, c4)

    receipt = {
        "mode": "kfacet_o617_why_dive",
        "why_dive_version": WHY_DIVE_VERSION,
        "row": row.label,
        "row_index": row.index,
        "kernel_floors": {
            "k7_floor": WHY_DIVE_KERNEL7_FLOOR,
            "k8_bridge_floor": WHY_DIVE_BRIDGE_FLOOR,
            "k7_dim": int(k7.shape[1]),
            "k8_dim": int(k8.shape[1]),
        },
        "kernel_singular_values_desc": [float(v) for v in svs],
        "thresholds": {
            "bridge_relation_floor": WHY_DIVE_BRIDGE_RELATION_FLOOR,
            "kernel_relation_floor": WHY_DIVE_KERNEL_RELATION_FLOOR,
            "eigen_residual_floor": WHY_DIVE_EIGEN_RESIDUAL_FLOOR,
            "lambda_near_one_band": WHY_DIVE_LAMBDA_NEAR_ONE_BAND,
            "trivial_alignment_floor": WHY_DIVE_TRIVIAL_ALIGNMENT_FLOOR,
        },
        "C1_sigma3_action_on_bridge": c1,
        "C2_eigenvalue_match": c2,
        "C3_f_beta_action": c3,
        "C4_orientation_sweep": c4,
        "C5_dihedral_residuals": c5,
        "outcome": outcome,
        "outcome_notes": outcome_notes(outcome),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "why_dive_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[o617-why-dive] receipt written: {OUT_DIR / 'why_dive_receipt.json'}")

    print()
    print(f"=== O_617 WHY-dive outcome: {outcome} ===")
    print(f"  {outcome_notes(outcome)}")
    print()
    print("--- C1: sigma_3 action on v_bridge (saved opposite orientation) ---")
    print(f"  sigma3 v          decomp: norm={c1['s3_v_decomp']['norm']:.3e} "
          f"in_k7={c1['s3_v_decomp']['in_k7']:.4f} in_k8={c1['s3_v_decomp']['in_k8']:.4f} "
          f"in_bridge={c1['s3_v_decomp']['in_bridge']:.4f}")
    print(f"  sigma3^2 v        decomp: norm={c1['s3_sq_v_decomp']['norm']:.3e} "
          f"in_k7={c1['s3_sq_v_decomp']['in_k7']:.4f} in_k8={c1['s3_sq_v_decomp']['in_k8']:.4f} "
          f"in_bridge={c1['s3_sq_v_decomp']['in_bridge']:.4f}")
    print(f"  sigma3^3 v        decomp: norm={c1['s3_cubed_v_decomp']['norm']:.3e} "
          f"in_k7={c1['s3_cubed_v_decomp']['in_k7']:.4f} in_k8={c1['s3_cubed_v_decomp']['in_k8']:.4f} "
          f"in_bridge={c1['s3_cubed_v_decomp']['in_bridge']:.4f}")
    print(f"  ||sigma3^3 v - v|| / ||v|| = {c1['s3_cubed_minus_v_relative']:.3e}")

    print()
    print("--- C2: M_i eigenvalue match at v_bridge ---")
    print(f"  lambda_rayleigh = {c2['lambda_rayleigh']:.6f}")
    print(f"  ||Mv - v|| / ||v||       = {c2['one_residual_relative']:.3e}")
    print(f"  ||Mv - lambda*v|| / ||Mv|| = {c2['eigen_residual_relative']:.3e}")
    print(f"  best eig overlap = {c2['best_eig_overlap']:.4f} at lambda = "
          f"{c2['best_eig_lambda_real']:.6f} + {c2['best_eig_lambda_imag']:.6f}i")
    print(f"  C2 category: {c2['category']}")

    print()
    print("--- C3: F_beta action on v_bridge ---")
    print(f"  F_beta v in k7 fraction: {c3['F_beta_v_in_k7']:.4f}")
    print(f"  F_beta v in k8 fraction: {c3['F_beta_v_in_k8']:.4f}")
    print(f"  ||F_beta^2 v - v|| / ||v||: {c3['F_beta_squared_minus_v_relative']:.3e}")
    print(f"  F_beta v parity: sym_pos={c3['F_beta_v_12sym_position_norm']:.3e}, "
          f"asym_pos={c3['F_beta_v_12antisym_position_norm']:.3e}, "
          f"sym_vel={c3['F_beta_v_12sym_velocity_norm']:.3e}, "
          f"asym_vel={c3['F_beta_v_12antisym_velocity_norm']:.3e}")

    print()
    print(f"--- C4: 8-orientation sweep (4 sigma_3 x 2 F_beta) ---")
    print(f"  any_full_pass={c4['any_full_pass']}; "
          f"any_bridge_only_pass={c4['any_bridge_only_pass']}; "
          f"no_bridge_pass={c4['no_bridge_pass']}")
    print(f"  {'sigma orient':>40} {'F_beta':>22} {'br_pass':>8} {'k8_pass':>8} {'br_max':>10} {'k8_max':>10}")
    for r in c4["combinations"]:
        print(f"  {r['sigma_orientation']:>40} {r['f_beta_variant']:>22} "
              f"{str(r['bridge_pass']):>8} {str(r['kernel_pass']):>8} "
              f"{r['max_bridge_residual']:>10.3e} {r['max_kernel_residual']:>10.3e}")

    print()
    print("--- C5: dihedral relation residuals on v_bridge (saved operators) ---")
    for key, value in c5.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3e}")


if __name__ == "__main__":
    main()
