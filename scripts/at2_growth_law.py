#!/usr/bin/env python3
"""AT-2 rung 2 -- growth-law post-processing (frozen spec v1.1:
docs/chatv2/AT2_GROWTH_LAW_SPEC.md; harness scope: AT2_HARNESS_SIGNOFF_REQUEST.md).

Reads the schema-v1 at2-samples.npz artifacts (one per regime) and evaluates the frozen
branch table + the v1.1 sub-reads (M1 spacing side-read; event-conditioned K_min).
Non-promotional; measured on the 32x32 truncation; no infinite-dim claim.
Run: python scripts/at2_growth_law.py --npz results/proof/at2-g200/at2-samples.npz \
        --npz300 results/proof/at2-g300/at2-samples.npz
"""
import argparse, json, math, os, sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pde_c1_kolmogorov_cell import aggregate_knn_sweep, held_out_r2  # read-only import

TAUS = [250, 500, 1000, 2000]
KS = [1, 2, 3, 4, 5, 6]
POS_LINE, POWER, ATOM_W, ATOM_MASS = 0.005, (0.20, 0.40), 1e-6, 0.05
R2_STATE_LINE, CENSOR = 0.5, 7
CFG = SimpleNamespace(delta_action=0.10, s_pos=0.50, delta_incompat=0.01,
                      delta_proxy_min=0.01, preset="at2_postprocess")


def a_mm_of(sigs, actions, eps):
    knn, _, _, _ = aggregate_knn_sweep(sigs, actions.astype(np.int8), eps, CFG)
    return float(knn["mean_minority_intercept"])


def lookmax(series, starts, tau):
    return np.array([float(np.max(series[s:s + tau + 1])) for s in starts])


def k_min(phi6, cols, actions, eps, mask=None):
    """Least K with a_mm <= POS_LINE, else CENSOR (=7, printed '>6')."""
    for k in KS:
        sl = phi6[:, cols[k]] if mask is None else phi6[mask][:, cols[k]]
        act = actions if mask is None else actions[mask]
        if a_mm_of(sl, act, eps) <= POS_LINE:
            return k
    return CENSOR


def analyze(path):
    z = np.load(path, allow_pickle=False)
    assert int(z["schema_version"]) == 1 and str(z["kind"]) == "at2"
    G = float(z["grashof"])
    e = z["e_low_k3"].astype(np.float64)
    adj, cal = z["adj_starts"], z["calib_starts"]
    phi6 = z["phi_k6"]
    cols = {k: z[f"cols_k{k}"] for k in KS}
    print(f"\n=== G={G:.0f}  ({os.path.basename(path)}; n_adj={len(adj)}) ===", flush=True)

    # value control (tau-independent): one K_min per regime
    v_adj = e[adj]
    act_val = (v_adj > np.median(e[cal])).astype(np.int8)
    eps_val = 0.05 * math.sqrt(max(0.0, 2.0 * float(np.median(e[cal]))))
    kmin_val = k_min(phi6, cols, act_val, eps_val)
    print(f"(value control) action = E_low_K3 > calib median; K_min = "
          f"{'>6' if kmin_val == CENSOR else kmin_val}")

    # state-proxy crossing (vacuity gauge, declared): R2(Phi_K -> high-mode norm)
    r2s = {}
    for k in KS:
        r2s[k], _ = held_out_r2(phi6[:, cols[k]], z["sample_high_norm"], 29 + k)
    k_state = next((k for k in KS if r2s[k] >= R2_STATE_LINE), CENSOR)
    print(f"(state proxy) R2(Phi_K->high_norm) = " +
          " ".join(f"K{k}:{r2s[k]:.3f}" for k in KS) +
          f"  -> crossing K_state = {'>6' if k_state == CENSOR else k_state}")

    # event flag (v1.1 SS2.5): backward-looking, at tau=500's calibrated threshold
    m500_cal = lookmax(e, cal, 500)
    e_max500 = float(np.quantile(m500_cal, float(z["quantile"])))
    back = np.array([float(np.max(e[s - 500:s + 1])) for s in adj])
    post_event = back > e_max500
    print(f"(event flag) post-event fraction = {post_event.mean():.3f}")

    rows = []
    print(f"\n  {'tau':>5} {'damp':>6} {'atom':>6} {'incl':>5} {'K_min(J)':>9} "
          f"{'K_ev':>5} {'K_qu':>5} {'a_mm@K1..K6'}")
    for tau in TAUS:
        m_adj = lookmax(e, adj, tau)
        e_max = float(np.quantile(lookmax(e, cal, tau), float(z["quantile"])))
        act = (m_adj > e_max).astype(np.int8)
        damp = float(act.mean())
        atom = float((np.abs(m_adj - e_max) <= ATOM_W).mean())
        powered = POWER[0] <= damp <= POWER[1]
        atom_ok = atom <= ATOM_MASS
        incl = powered and atom_ok
        eps = 0.05 * math.sqrt(max(0.0, 2.0 * e_max))
        amms = [a_mm_of(phi6[:, cols[k]], act, eps) for k in KS]
        kmin = next((k for k, a in zip(KS, amms) if a <= POS_LINE), CENSOR)
        # event sub-read (reported tier): per-slice K_min, slice inclusion checked
        kev = kqu = None
        if incl:
            for name, msk in (("ev", post_event), ("qu", ~post_event)):
                d = float(act[msk].mean()) if msk.sum() else -1.0
                ok = POWER[0] <= d <= POWER[1] and \
                    float((np.abs(m_adj[msk] - e_max) <= ATOM_W).mean()) <= ATOM_MASS
                val = k_min(phi6, cols, act, eps, mask=msk) if ok else None
                if name == "ev":
                    kev = val
                else:
                    kqu = val
        rows.append({"tau": tau, "damp": damp, "atom_mass": atom, "included": incl,
                     "k_min": kmin, "k_event": kev, "k_quiet": kqu, "a_mm": amms,
                     "e_max": e_max})
        fmt = lambda v: "--" if v is None else (">6" if v == CENSOR else str(v))
        print(f"  {tau:>5} {damp:>6.3f} {atom:>6.3f} {str(incl):>5} "
              f"{fmt(kmin):>9} {fmt(kev):>5} {fmt(kqu):>5} "
              + " ".join(f"{a:.3f}" for a in amms), flush=True)
    return {"G": G, "rows": rows, "kmin_val": kmin_val, "k_state": k_state,
            "r2_state": r2s, "event_fraction": float(post_event.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--npz300", default=None)
    a = ap.parse_args()
    print("AT2_GROWTH_LAW  [NON-PROMOTIONAL]  spec: AT2_GROWTH_LAW_SPEC.md v1.1", flush=True)
    regs = [analyze(a.npz)] + ([analyze(a.npz300)] if a.npz300 else [])

    # frozen branch table (spec section 3), evaluated across regimes
    verdicts = []
    for r in regs:
        inc = [row for row in r["rows"] if row["included"]]
        if len(inc) < 3:
            verdicts.append((r["G"], "NO_GATE_READ (<3 included tau-cells)"))
            continue
        kmins = [row["k_min"] for row in inc]
        n_cens = sum(1 for k in kmins if k == CENSOR)
        grew = max(kmins) - min(kmins) >= 2
        val_flat = r["kmin_val"] != CENSOR  # single value; flat by construction
        vac = all(row["k_min"] == r["k_state"] for row in inc)
        if n_cens >= (len(inc) + 1) // 2:
            v = "AT2_CENSORED"
        elif vac:
            v = "AT2_COLLAPSE_VACUOUS"
        elif grew and val_flat:
            v = "AT2_GROWTH_CONFIRMED (within-regime)"
        elif max(kmins) - min(kmins) <= 1:
            v = "AT2_FLAT_NULL (within-regime)"
        else:
            v = "AT2_NO_BRANCH (pattern recorded)"
        # M1 side-read: tau spacing of increments
        incs = [(inc[i]["tau"], kmins[i]) for i in range(len(inc))]
        verdicts.append((r["G"], v + f"  [K_min over tau: {incs}]"))
    # cross-regime arm (>=1 at matched tau, G200->300)
    cross = None
    if len(regs) == 2:
        gains = []
        for row2 in regs[0]["rows"]:
            row3 = next((x for x in regs[1]["rows"] if x["tau"] == row2["tau"]), None)
            if row3 and row2["included"] and row3["included"] \
                    and CENSOR not in (row2["k_min"], row3["k_min"]):
                gains.append((row2["tau"], row3["k_min"] - row2["k_min"]))
        cross = gains
    # event sub-read tokens (reported tier)
    ev_tokens = []
    for r in regs:
        gaps = [(row["tau"], row["k_event"] - row["k_quiet"])
                for row in r["rows"]
                if row["included"] and row["k_event"] not in (None, CENSOR)
                and row["k_quiet"] not in (None, CENSOR)]
        if not gaps:
            tok = "AT2_EVENT_UNPOWERED"
        elif sum(1 for _, g in gaps if g >= 2) >= 2:
            tok = "AT2_EVENT_CARRIED"
        else:
            tok = "AT2_EVENT_FLAT"
        ev_tokens.append((r["G"], tok, gaps))

    print("\n(verdicts, per frozen table)")
    for G, v in verdicts:
        print(f"  G={G:.0f}: {v}")
    if cross is not None:
        print(f"  cross-regime K_min gains at matched tau (G200->G300): {cross}")
    for G, tok, gaps in ev_tokens:
        print(f"  event sub-read G={G:.0f}: {tok}  gaps={gaps}")
    out = {"spec": "AT2_GROWTH_LAW_SPEC.md v1.1", "date": "2026-07-03",
           "regimes": regs, "verdicts": [[g, v] for g, v in verdicts],
           "cross_regime_gains": cross,
           "event_subread": [[g, t, gp] for g, t, gp in ev_tokens]}
    path = os.path.join(os.path.dirname(a.npz), "at2_growth_law.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"(wrote) {path}")
    print("  (Non-promotional. 32x32 truncation measurement; no infinite-dim NSE claim.)")


if __name__ == "__main__":
    main()
