#!/usr/bin/env python3
"""AT-1 rung 2 -- margin-band excision curve (frozen spec:
docs/chatv2/AT1_PALINSTROPHY_BOUNDARY_LAYER_SPEC.md; scope: AT1_HARNESS_SIGNOFF.md).

Reads the schema-v1 at1-samples.npz side artifact (rung 1) and evaluates the frozen
branch table. Does NOT touch the simulator or the banked receipts. Non-promotional.
Run: python scripts/at1_boundary_layer.py --npz results/proof/at1-g200/at1-samples.npz
"""
import argparse, json, math, os, sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pde_c1_kolmogorov_cell import aggregate_knn_sweep, held_out_r2  # read-only import

BANDS = [0.0, 0.02, 0.05, 0.10, 0.20]           # frozen band grid (|margin| quantiles)
POS_LINE, NEG_A_LINE, POWER = 0.005, 0.015, (0.20, 0.40)   # inherited, frozen
# the knn-sweep cfg surface (lock_disc_g200 pinned values; aggregate_ + summarize_)
CFG = SimpleNamespace(delta_action=0.10, s_pos=0.50, delta_incompat=0.01,
                      delta_proxy_min=0.01, preset="at1_postprocess")


def a_mm_of(sigs: np.ndarray, actions: np.ndarray, eps: float) -> float:
    knn, _, _, _ = aggregate_knn_sweep(sigs, actions.astype(np.int8), eps, CFG)
    return float(knn["mean_minority_intercept"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=False)
    assert int(z["schema_version"]) == 1, "schema mismatch"
    names = [str(n) for n in z["objective_names"]]
    pal, elow = names.index("palinstrophy"), names.index("E_low")
    phi = z["phi_k"]
    print(f"AT1_BOUNDARY_LAYER  preset={z['preset']}  n_adj={phi.shape[0]}  "
          f"[NON-PROMOTIONAL]\n", flush=True)

    # frozen eps from the UNBANDED palinstrophy e_max
    e_max_pal = float(z["e_max"][pal])
    eps = 0.05 * math.sqrt(max(0.0, 2.0 * e_max_pal))

    # liveness: E_low unbanded must reproduce the banked POSITIVE
    eps_elow = 0.05 * math.sqrt(max(0.0, 2.0 * float(z["e_max"][elow])))
    a_elow = a_mm_of(phi, z["actions"][:, elow], eps_elow)
    live_ok = a_elow <= POS_LINE
    print(f"(liveness) E_low unbanded a_mm = {a_elow:.4f} (must be <= {POS_LINE}) "
          f"{'OK' if live_ok else 'FAIL'}")
    if not live_ok:
        print("\nVERDICT: VOID (liveness) - export path questioned; fix and re-run.")
        return

    # unbanded reproduction + sibling row
    act_pal = z["actions"][:, pal]
    m_pal = z["m_adj"][:, pal]
    margin = np.abs(z["margin"][:, pal])
    damp0 = float(act_pal.mean())
    a0 = a_mm_of(phi, act_pal, eps)
    r2_0, _ = held_out_r2(phi, m_pal, 17 + pal)
    print(f"(reproduce) palinstrophy unbanded: a_mm={a0:.4f} (banked 0.195), "
          f"damp={damp0:.3f}, R2={r2_0:.4f}")
    act_sib = z["actions_mean_pal"]
    damp_sib = float(act_sib.mean())
    eps_sib = 0.05 * math.sqrt(max(0.0, 2.0 * float(z["e_max_mean_pal"])))
    a_sib = a_mm_of(phi, act_sib, eps_sib)
    r2_sib, _ = held_out_r2(phi, z["m_mean_pal"], 41)
    sib_powered = POWER[0] <= damp_sib <= POWER[1]
    print(f"(sibling) lookahead-MEAN palinstrophy unbanded: a_mm={a_sib:.4f}, "
          f"damp={damp_sib:.3f} ({'powered' if sib_powered else 'UNPOWERED'}), R2={r2_sib:.4f}\n")

    # excision curve
    print(f"  {'band':>6} {'n_kept':>7} {'damp':>6} {'powered':>8} {'a_mm':>8} {'R2':>8}")
    rows, powered_rows = [], []
    order = np.argsort(margin)
    for beta in BANDS:
        cut = int(round(beta * len(margin)))
        keep = np.ones(len(margin), bool)
        keep[order[:cut]] = False
        damp = float(act_pal[keep].mean())
        powered = POWER[0] <= damp <= POWER[1]
        amm = a_mm_of(phi[keep], act_pal[keep], eps)
        r2, _ = held_out_r2(phi[keep], m_pal[keep], 17 + pal)
        rows.append({"band": beta, "n_kept": int(keep.sum()), "damp": damp,
                     "powered": powered, "a_mm": amm, "r2": r2})
        if powered:
            powered_rows.append(rows[-1])
        print(f"  {beta:>6.2f} {keep.sum():>7} {damp:>6.3f} {str(powered):>8} "
              f"{amm:>8.4f} {r2:>8.4f}", flush=True)

    # frozen branch table
    if not (POWER[0] <= damp0 <= POWER[1]) or not powered_rows or not sib_powered:
        verdict = "AT1_UNDERPOWERED"
    elif any(r["a_mm"] <= POS_LINE for r in powered_rows) and a_sib <= POS_LINE:
        verdict = ("AT1_BOUNDARY_LAYER_ARTIFACT - banded a_mm reaches POSITIVE and the "
                   "burst-robust sibling is POSITIVE unbanded: the anomaly is the "
                   "lookahead-max decision boundary layer")
    elif all(r["a_mm"] >= NEG_A_LINE for r in powered_rows) and a_sib >= NEG_A_LINE:
        verdict = ("AT1_TWO_POLE_CONFIRMED - decision-level failure persists at every "
                   "powered band AND replicates on the burst-robust sibling: the natural "
                   "cell carries a regime-2 (E_low) and a regime-3 (palinstrophy) objective "
                   "at the same shadow")
    else:
        verdict = "AT1_INCONCLUSIVE_MIXED - neither branch fires cleanly; recorded"
    out = {"spec": "AT1_PALINSTROPHY_BOUNDARY_LAYER_SPEC.md", "date": "2026-07-02",
           "npz": a.npz, "eps": eps, "liveness_E_low_a_mm": a_elow,
           "unbanded": {"a_mm": a0, "damp": damp0, "r2": r2_0},
           "sibling_mean": {"a_mm": a_sib, "damp": damp_sib, "powered": sib_powered,
                            "r2": r2_sib},
           "bands": rows, "verdict": verdict}
    path = os.path.join(os.path.dirname(a.npz), "at1_boundary_layer.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n(wrote) {path}")
    print(f"VERDICT: {verdict}")
    print("  (Non-promotional. C1 separation untouched; no infinite-dim NSE claim.)")


if __name__ == "__main__":
    main()
