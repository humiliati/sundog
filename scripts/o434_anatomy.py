"""O_434(0.4) row anatomy: v0.4a0 structural investigation.

Pre-registered question (signed off 2026-05-22): is O_434(0.4)'s F_beta
closure residual of `0.252` a structural smaller-symmetry feature of the
orbit, or a measurement/gauge artifact?

The verdict determines whether v0.4 needs a smaller-symmetry outlier lane
from the start (the two-track asymmetric design), or whether v0.4 can
close as a clean one-track Z_2-primary chapter.

Five probes, all no-integration except Q5 (one short sigma3-scan re-run):

  Q1 -- F_beta operator self-consistency:
        Compute ||F_beta^2 - I||_inf on the saved 18D operator. Rules out
        operator-construction bug as the cause of broken closure.
  Q2 -- F_beta closure SVD spectrum on ker(M_i - I):
        SVD of the commutator F_beta * M_i - M_i * F_beta restricted to the
        kernel, plus F_beta^2 - I on the same kernel. Locates the most-broken
        direction and decomposes it by body/axis.
  Q3 -- Inertia-degenerate metadata check:
        m_3 * z_0^2 vs the critical value 2. The Sundog spec flags rows in
        this band as gauge-degenerate by default.
  Q4 -- Cross-row F_beta comparison among m_3=0.4 sentinels:
        O_50, O_62, O_67 are F_beta-clean; O_434 is broken. Does any
        catalog-metadata feature distinguish O_434?
  Q5 -- F_beta residual sensitivity to gauge minimization:
        Re-run sigma3-scan with tighter `identity_rotation_tolerance` (1e-9)
        and finer `phase_grid` (361). If the residual drops to <= 1e-2, the
        original 0.25 was a gauge-minimization artifact.

Outcome categories (pre-registered):

  gauge_artifact            -- Q5 residual drops to <= V04A0_GAUGE_ARTIFACT_THRESHOLD
  inertia_degenerate        -- Q3 finds |m_3 * z_0^2 - 2| < threshold
  operator_broken           -- Q1 fails (F_beta^2 != I on the operator itself)
  structural_anomaly_named  -- Q2 localizes the break on a specific body/axis
  robust_smaller_symmetry   -- none of the above; smaller symmetry is real
                               and structural, not a measurement artifact

The proposed_v04_classification field carries the landed outcome, marked
`pending_v04_registration` until v0.4b locks the formal classifier.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import isotrophy_workbench as W


# Pre-registered constants.
V04A0_VERSION = "v0.4a0-o434-row-anatomy"
V04A0_GAUGE_ARTIFACT_THRESHOLD = 1e-2
V04A0_INERTIA_DEGENERATE_THRESHOLD = 1e-3
V04A0_OPERATOR_INVOLUTION_FLOOR = 1e-10
V04A0_TIGHTER_IDENTITY_ROTATION_TOLERANCE = 1e-9
V04A0_TIGHTER_PHASE_GRID = 361
V04A0_BRIDGE_ADMIT_FLOOR = 1e-3  # for kernel basis at the bridge-admit edge
V04A0_BODY_DOMINANCE_RATIO = 1.8  # body norm > 1.8 * (sum of other body norms)
                                  # for the most-broken direction to count as
                                  # localized on a single body

ROOT = Path(__file__).resolve().parent.parent
SENTINEL_DIR = ROOT / "results/isotrophy/k-facet-v03-gamma-crossm3/m3eq0.4/O434"
SYMMETRY_PROBE_CSV = ROOT / "results/isotrophy/k-facet-v03-piano-symmetry-probe/m3eq0.4/residuals.csv"
CATALOG_PATH = ROOT / "docs/isotrophy/supplementary-B_piano-init-condit-3d.txt"
OUT_DIR = ROOT / "results/isotrophy/k-facet-v04a0-o434-anatomy"


def load_inputs():
    M_i = np.load(SENTINEL_DIR / "M_i.npy")
    F_beta = np.load(SENTINEL_DIR / "D3_F_beta.npy")
    text = W.read_text(str(CATALOG_PATH))
    rows = W.parse_rows(text, "B")
    row = next(r for r in rows if r.index == 434 and abs(r.m3 - 0.4) < 1e-12)
    return M_i, F_beta, row


def probe_q1_fbeta_self_consistency(F_beta: np.ndarray) -> dict:
    """Q1: Is F_beta an exact involution on the 18D tangent space?"""
    identity = np.eye(F_beta.shape[0])
    F_sq_minus_I = F_beta @ F_beta - identity
    residual_inf = float(np.linalg.norm(F_sq_minus_I, ord=np.inf))
    residual_fro = float(np.linalg.norm(F_sq_minus_I, ord="fro"))
    return {
        "F_beta_squared_minus_I_inf": residual_inf,
        "F_beta_squared_minus_I_fro": residual_fro,
        "operator_involution_floor": V04A0_OPERATOR_INVOLUTION_FLOOR,
        "involution_passed": residual_inf <= V04A0_OPERATOR_INVOLUTION_FLOOR,
        "interpretation": (
            "If involution_passed=False, the F_beta operator itself is "
            "broken (constructor bug) and the closure residual is a "
            "downstream effect, not a structural orbit feature."
        ),
    }


def probe_q2_fbeta_closure_spectrum(M_i: np.ndarray, F_beta: np.ndarray) -> dict:
    """Q2: SVD of F_beta-M_i commutator + F_beta^2-I on ker(M_i - I)."""
    identity = np.eye(M_i.shape[0])
    _u, svs_minus_I, vh = np.linalg.svd(M_i - identity)
    mask = svs_minus_I < V04A0_BRIDGE_ADMIT_FLOOR
    K = vh.T[:, mask]
    k_dim = int(K.shape[1])
    if k_dim == 0:
        return {
            "kernel_dim_at_bridge_admit_floor": 0,
            "note": "ker(M_i - I) at floor 1e-3 is empty; commutator probe undefined.",
        }
    commutator = F_beta @ M_i - M_i @ F_beta
    F_sq_minus_I = F_beta @ F_beta - identity
    comm_restricted = K.T @ commutator @ K
    f2_restricted = K.T @ F_sq_minus_I @ K
    # SVD of commutator on kernel to find the most-broken direction
    u_c, s_c, vh_c = np.linalg.svd(comm_restricted)
    s_c_list = [float(s) for s in s_c]
    # Right singular vector at largest SV gives the kernel-coords direction
    # whose commutator residual is maximal. Lift back to 18D.
    v_break_kernel = vh_c[0, :]
    v_break_full = K @ v_break_kernel
    v_break_norm = float(np.linalg.norm(v_break_full))
    if v_break_norm > 1e-300:
        v_break_full_unit = v_break_full / v_break_norm
    else:
        v_break_full_unit = v_break_full
    v_pos = v_break_full_unit[:9].reshape(3, 3)
    v_vel = v_break_full_unit[9:].reshape(3, 3)
    per_body_pos = np.linalg.norm(v_pos, axis=1)
    per_body_vel = np.linalg.norm(v_vel, axis=1)
    per_axis_pos = np.linalg.norm(v_pos, axis=0)
    per_axis_vel = np.linalg.norm(v_vel, axis=0)
    return {
        "kernel_dim_at_bridge_admit_floor": k_dim,
        "kernel_floor_used": V04A0_BRIDGE_ADMIT_FLOOR,
        "commutator_F_beta_M_i_kernel_singular_values": s_c_list,
        "max_commutator_residual_on_kernel": s_c_list[0] if s_c_list else 0.0,
        "F_beta_squared_minus_I_kernel_inf": float(np.linalg.norm(f2_restricted, ord=np.inf)),
        "most_broken_direction_per_body_pos_norms": [float(x) for x in per_body_pos],
        "most_broken_direction_per_body_vel_norms": [float(x) for x in per_body_vel],
        "most_broken_direction_per_axis_pos_norms": [float(x) for x in per_axis_pos],
        "most_broken_direction_per_axis_vel_norms": [float(x) for x in per_axis_vel],
        "most_broken_direction_18d": [float(x) for x in v_break_full_unit],
    }


def probe_q3_inertia_degenerate(row) -> dict:
    """Q3: Is O_434 in the m_3 * z_0^2 ~ 2 inertia-degenerate band?"""
    val = float(row.m3) * float(row.z0) * float(row.z0)
    distance_to_2 = abs(val - 2.0)
    return {
        "m_3": float(row.m3),
        "z_0": float(row.z0),
        "m_3_times_z_0_squared": val,
        "critical_value": 2.0,
        "distance_to_critical_value": distance_to_2,
        "threshold": V04A0_INERTIA_DEGENERATE_THRESHOLD,
        "is_inertia_degenerate": distance_to_2 < V04A0_INERTIA_DEGENERATE_THRESHOLD,
        "interpretation": (
            "If is_inertia_degenerate=True, the SO(3) gauge minimization is "
            "structurally ill-posed because the t=0 inertia tensor has a "
            "degenerate eigenvalue, and the broken closure is a known gauge "
            "artifact rather than a structural orbit feature."
        ),
    }


def probe_q4_cross_row_comparison(row) -> dict:
    """Q4: Compare O_434 against the other m_3=0.4 sentinels."""
    if not SYMMETRY_PROBE_CSV.is_file():
        return {"missing_input": str(SYMMETRY_PROBE_CSV)}
    out_rows = []
    with SYMMETRY_PROBE_CSV.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m3 = float(r["m3"])
            z0 = float(r["z0"])
            out_rows.append({
                "label": r["label"],
                "index": int(r["index"]),
                "period": float(r["period"]),
                "stability": r["stability"],
                "z_0": z0,
                "m_3_times_z_0_squared": m3 * z0 * z0,
                "sigma_group_residual_inf": float(r["sigma_group_residual_inf"]),
                "F_beta_residual_inf": float(r["F_beta_residual_inf"]),
                "F_beta_to_closure": float(r["F_beta_to_closure"]),
            })
    # Identify O_434 vs siblings; report distinguishing metadata
    target = next((r for r in out_rows if r["index"] == row.index), None)
    siblings = [r for r in out_rows if r["index"] != row.index]
    distinguishing = {}
    if target is not None and siblings:
        for key in ("period", "z_0", "m_3_times_z_0_squared", "stability"):
            target_val = target[key]
            sibling_vals = [s[key] for s in siblings]
            distinguishing[key] = {
                "target": target_val,
                "siblings": sibling_vals,
                "target_is_extremal": (
                    target_val == max(sibling_vals + [target_val])
                    or target_val == min(sibling_vals + [target_val])
                ) if isinstance(target_val, (int, float)) else None,
            }
    return {
        "rows": out_rows,
        "target_row": target,
        "distinguishing_metadata": distinguishing,
    }


def probe_q5_tighter_gauge(target_index: int, original_F_beta_residual: float) -> dict:
    """Q5: Re-run sigma3-scan on O_434 with tighter gauge tolerance + phase grid."""
    out_dir = OUT_DIR / "q5_tighter_gauge"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/isotrophy_workbench.py", "sigma3-scan",
        "--source", "B",
        "--path", str(CATALOG_PATH.relative_to(ROOT)),
        "--m3", "0.4",
        "--indices", str(target_index),
        "--n-samples", "1009",
        "--phase-grid", str(V04A0_TIGHTER_PHASE_GRID),
        "--rtol", "1e-12",
        "--atol", "1e-12",
        "--sigma-tolerance", "1e-5",
        "--sigma-closure-multiple", "3",
        "--identity-rotation-tolerance",
        f"{V04A0_TIGHTER_IDENTITY_ROTATION_TOLERANCE:.0e}",
        "--out", str(out_dir),
    ]
    started = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(ROOT)
    )
    elapsed = time.time() - started
    tighter_csv = out_dir / "residuals.csv"
    tighter_F_beta_residual = None
    tighter_F_beta_to_closure = None
    tighter_sigma_group_residual = None
    if tighter_csv.is_file():
        with tighter_csv.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if int(r["index"]) == target_index:
                    tighter_F_beta_residual = float(r["F_beta_residual_inf"])
                    tighter_F_beta_to_closure = float(r["F_beta_to_closure"])
                    tighter_sigma_group_residual = float(r["sigma_group_residual_inf"])
                    break
    ratio = None
    if tighter_F_beta_residual is not None and original_F_beta_residual > 0:
        ratio = float(tighter_F_beta_residual / original_F_beta_residual)
    return {
        "tighter_identity_rotation_tolerance": V04A0_TIGHTER_IDENTITY_ROTATION_TOLERANCE,
        "tighter_phase_grid": V04A0_TIGHTER_PHASE_GRID,
        "original_F_beta_residual": original_F_beta_residual,
        "tighter_F_beta_residual": tighter_F_beta_residual,
        "tighter_F_beta_to_closure": tighter_F_beta_to_closure,
        "tighter_sigma_group_residual": tighter_sigma_group_residual,
        "ratio_tighter_over_original": ratio,
        "gauge_artifact_threshold": V04A0_GAUGE_ARTIFACT_THRESHOLD,
        "is_gauge_artifact": (
            tighter_F_beta_residual is not None
            and tighter_F_beta_residual <= V04A0_GAUGE_ARTIFACT_THRESHOLD
        ),
        "wall_time_seconds": elapsed,
        "subprocess_returncode": result.returncode,
        "subprocess_stdout_tail": result.stdout[-400:] if result.stdout else "",
        "subprocess_stderr_tail": result.stderr[-400:] if result.stderr else "",
    }


def categorize_outcome(q1, q2, q3, q5):
    """Pre-registered outcome categorization, deterministic."""
    if q5.get("is_gauge_artifact"):
        return "gauge_artifact"
    if q3.get("is_inertia_degenerate"):
        return "inertia_degenerate"
    if not q1.get("involution_passed"):
        return "operator_broken"
    # Body dominance check on Q2's most-broken direction
    body_pos = q2.get("most_broken_direction_per_body_pos_norms") or [0, 0, 0]
    body_vel = q2.get("most_broken_direction_per_body_vel_norms") or [0, 0, 0]
    body_total = [p + v for p, v in zip(body_pos, body_vel)]
    if body_total:
        max_body = max(body_total)
        rest_sum = sum(body_total) - max_body
        if max_body > V04A0_BODY_DOMINANCE_RATIO * max(rest_sum, 1e-300):
            return "structural_anomaly_named"
    return "robust_smaller_symmetry"


def outcome_notes(outcome: str) -> str:
    notes = {
        "gauge_artifact": (
            "Tighter SO(3) gauge minimization drops the F_beta residual below "
            f"{V04A0_GAUGE_ARTIFACT_THRESHOLD}. The original 0.25 was a phase-grid + "
            "rotation-tolerance discretization artifact, NOT a structural orbit "
            "feature. v0.4 can close as a one-track Z_2-primary chapter; O_434 "
            "reclassifies as marginal_Z2 or Z2_clean under the v0.4a four-band "
            "classifier."
        ),
        "inertia_degenerate": (
            "O_434's IC sits in the inertia-degenerate band m_3 * z_0^2 ~ 2. "
            "The broken F_beta closure is the canonical gauge artifact the "
            "Sundog spec flags by default. Not a structural smaller-symmetry. "
            "v0.4 can use the inertia_degenerate flag as a separate well-known "
            "exception class."
        ),
        "operator_broken": (
            "F_beta^2 != I on the saved 18D operator. The broken closure is a "
            "constructor bug, not an orbit feature. Investigate the F_beta "
            "construction in scripts/isotrophy_workbench.py before any v0.4 "
            "claim."
        ),
        "structural_anomaly_named": (
            "Q2's most-broken direction is dominated by one body (body norm > "
            f"{V04A0_BODY_DOMINANCE_RATIO}x sum of other bodies). The smaller "
            "symmetry has a specific physical character. v0.4 needs an outlier "
            "lane with a named structural diagnosis; O_434 is the worked "
            "example."
        ),
        "robust_smaller_symmetry": (
            "Tighter gauge does not rescue, inertia is not degenerate, F_beta "
            "operator is clean, and Q2 finds no single-body dominance. The "
            "smaller symmetry is structurally distributed -- a real Z_2-or-smaller "
            "class without a named single-cause explanation. v0.4 needs an "
            "outlier lane from the start."
        ),
    }
    return notes.get(outcome, "unrecognized outcome")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[o434-anatomy] loading inputs (M_i, F_beta, catalog row)")
    M_i, F_beta, row = load_inputs()
    print(f"[o434-anatomy] {row.label} period={row.period} stability={row.stability}")

    print("[o434-anatomy] Q1: F_beta operator self-consistency")
    q1 = probe_q1_fbeta_self_consistency(F_beta)
    print(f"  ||F_beta^2 - I||_inf = {q1['F_beta_squared_minus_I_inf']:.3e}  "
          f"involution_passed={q1['involution_passed']}")

    print("[o434-anatomy] Q2: F_beta closure SVD spectrum on ker(M_i - I)")
    q2 = probe_q2_fbeta_closure_spectrum(M_i, F_beta)
    print(f"  kernel_dim_at_bridge_admit_floor = {q2.get('kernel_dim_at_bridge_admit_floor')}")
    print(f"  max commutator residual on kernel = {q2.get('max_commutator_residual_on_kernel', 'n/a')!r}")
    if "most_broken_direction_per_body_pos_norms" in q2:
        print(f"  most-broken direction per-body pos norms = {q2['most_broken_direction_per_body_pos_norms']}")
        print(f"  most-broken direction per-body vel norms = {q2['most_broken_direction_per_body_vel_norms']}")

    print("[o434-anatomy] Q3: inertia-degenerate metadata check")
    q3 = probe_q3_inertia_degenerate(row)
    print(f"  m_3 * z_0^2 = {q3['m_3_times_z_0_squared']:.4e}  distance to 2 = {q3['distance_to_critical_value']:.4e}")
    print(f"  is_inertia_degenerate = {q3['is_inertia_degenerate']}")

    print("[o434-anatomy] Q4: cross-row comparison among m_3=0.4 sentinels")
    q4 = probe_q4_cross_row_comparison(row)
    if q4.get("target_row"):
        print(f"  target F_beta_residual = {q4['target_row']['F_beta_residual_inf']:.3e}")
        sib_vals = [s['F_beta_residual_inf'] for s in q4['rows'] if s['index'] != row.index]
        print(f"  siblings F_beta_residual range = [{min(sib_vals):.3e}, {max(sib_vals):.3e}]")

    # Q5 uses the original F_beta residual from the symmetry probe
    original_F_beta = q4.get("target_row", {}).get("F_beta_residual_inf", 0.252)
    print(f"[o434-anatomy] Q5: tighter gauge re-run (phase_grid={V04A0_TIGHTER_PHASE_GRID}, "
          f"identity_rotation_tolerance={V04A0_TIGHTER_IDENTITY_ROTATION_TOLERANCE:.0e})")
    q5 = probe_q5_tighter_gauge(row.index, original_F_beta)
    print(f"  original F_beta residual: {q5['original_F_beta_residual']:.3e}")
    if q5.get("tighter_F_beta_residual") is not None:
        print(f"  tighter F_beta residual:  {q5['tighter_F_beta_residual']:.3e}")
        print(f"  ratio (tighter/original): {q5['ratio_tighter_over_original']:.3e}")
    else:
        print(f"  tighter F_beta residual: NOT AVAILABLE (subprocess returncode={q5.get('subprocess_returncode')})")
    print(f"  is_gauge_artifact = {q5['is_gauge_artifact']}")

    outcome = categorize_outcome(q1, q2, q3, q5)
    receipt = {
        "mode": "kfacet_v04a0_o434_row_anatomy",
        "version": V04A0_VERSION,
        "row": row.label,
        "row_index": row.index,
        "m3": row.m3,
        "thresholds": {
            "gauge_artifact_threshold": V04A0_GAUGE_ARTIFACT_THRESHOLD,
            "inertia_degenerate_threshold": V04A0_INERTIA_DEGENERATE_THRESHOLD,
            "operator_involution_floor": V04A0_OPERATOR_INVOLUTION_FLOOR,
            "tighter_identity_rotation_tolerance": V04A0_TIGHTER_IDENTITY_ROTATION_TOLERANCE,
            "tighter_phase_grid": V04A0_TIGHTER_PHASE_GRID,
            "body_dominance_ratio": V04A0_BODY_DOMINANCE_RATIO,
        },
        "Q1_F_beta_operator_self_consistency": q1,
        "Q2_F_beta_closure_spectrum_on_kernel": q2,
        "Q3_inertia_degenerate_metadata": q3,
        "Q4_cross_row_comparison_m3eq04_sentinels": q4,
        "Q5_tighter_gauge_minimization": q5,
        "outcome": outcome,
        "outcome_notes": outcome_notes(outcome),
        "proposed_v04_classification": {
            "label": outcome,
            "status": "pending_v04_registration",
            "note": (
                "Classification is provisional until v0.4b locks the formal "
                "four-band classifier and predictor."
            ),
        },
    }
    (OUT_DIR / "anatomy_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print()
    print(f"=== O_434(0.4) anatomy outcome: {outcome} ===")
    print(f"  {outcome_notes(outcome)}")
    print()
    print(f"Receipt: {OUT_DIR / 'anatomy_receipt.json'}")


if __name__ == "__main__":
    main()
