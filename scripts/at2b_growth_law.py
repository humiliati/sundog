#!/usr/bin/env python3
"""AT-2b -- in-band growth law, mean-form primary (frozen spec:
docs/chatv2/AT2B_GROWTH_LAW_SPEC.md). Pure post-processing on the banked AT-2 exports —
no simulation, no harness change. Non-promotional; 32x32 truncation; no infinite-dim claim.
Run: python scripts/at2b_growth_law.py --npz results/proof/at2-g200/at2-samples.npz \
        --npz300 results/proof/at2-g300/at2-samples.npz
"""
import argparse, json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pde_c1_kolmogorov_cell import held_out_r2  # noqa: E402  (read-only import)
from at2_growth_law import CFG, a_mm_of, lookmax  # noqa: E402  (shared frozen machinery)

TAUS = [100, 250, 400, 500, 750]          # frozen in-band grid
TAUS_MAX_SIB = [100, 250, 400, 500]       # max-form sibling band (reported)
KS = [1, 2, 3, 4, 5, 6]
POS_LINE, POWER, ATOM_W, ATOM_MASS = 0.005, (0.20, 0.40), 1e-6, 0.05
R2_STATE_LINE, CENSOR = 0.5, 7


def lookmean(series, starts, tau):
    c = np.cumsum(np.r_[0.0, series])
    s = np.asarray(starts)
    return (c[s + tau + 1] - c[s]) / (tau + 1)


def k_min_curve(phi6, cols, actions, eps, mask=None):
    amms = []
    for k in KS:
        sl = phi6[:, cols[k]] if mask is None else phi6[mask][:, cols[k]]
        act = actions if mask is None else actions[mask]
        amms.append(a_mm_of(sl, act, eps))
    kmin = next((k for k, a in zip(KS, amms) if a <= POS_LINE), CENSOR)
    return kmin, amms


def cell(series, phi6, cols, adj, cal, tau, quantile, functional):
    f = lookmean if functional == "mean" else lookmax
    m_adj = f(series, adj, tau)
    thr = float(np.quantile(f(series, cal, tau), quantile))
    act = (m_adj > thr).astype(np.int8)
    damp = float(act.mean())
    atom = float((np.abs(m_adj - thr) <= ATOM_W).mean())
    incl = (POWER[0] <= damp <= POWER[1]) and atom <= ATOM_MASS
    eps = 0.05 * math.sqrt(max(0.0, 2.0 * thr))
    return m_adj, thr, act, damp, atom, incl, eps


def analyze(path):
    z = np.load(path, allow_pickle=False)
    G = float(z["grashof"])
    e = z["e_low_k3"].astype(np.float64)
    adj, cal = z["adj_starts"], z["calib_starts"]
    phi6 = z["phi_k6"]
    cols = {k: z[f"cols_k{k}"] for k in KS}
    q = float(z["quantile"])
    print(f"\n=== G={G:.0f} ===", flush=True)

    # state proxy + value control (unchanged machinery)
    r2s = {k: held_out_r2(phi6[:, cols[k]], z["sample_high_norm"], 29 + k)[0] for k in KS}
    k_state = next((k for k in KS if r2s[k] >= R2_STATE_LINE), CENSOR)
    act_val = (e[adj] > np.median(e[cal])).astype(np.int8)
    eps_val = 0.05 * math.sqrt(max(0.0, 2.0 * float(np.median(e[cal]))))
    kmin_val, _ = k_min_curve(phi6, cols, act_val, eps_val)
    # event flag (same registered definition: backward MAX at max-form e_max(500))
    e_max500 = float(np.quantile(lookmax(e, cal, 500), q))
    post_event = np.array([float(np.max(e[s - 500:s + 1])) for s in adj]) > e_max500
    print(f"(state) K_state={'>6' if k_state == CENSOR else k_state} | "
          f"(value) K_min={'>6' if kmin_val == CENSOR else kmin_val} | "
          f"(event) post-event frac={post_event.mean():.3f}")

    fmt = lambda v: "--" if v is None else (">6" if v == CENSOR else str(v))
    out_rows = {}
    for functional, taus, tag in (("mean", TAUS, "MEAN-form (PRIMARY)"),
                                  ("max", TAUS_MAX_SIB, "MAX-form (sibling, reported)")):
        print(f"\n  [{tag}]")
        print(f"  {'tau':>5} {'damp':>6} {'atom':>6} {'incl':>5} {'K_min':>6} {'Delta':>6} "
              f"{'K_ev':>5} {'K_qu':>5}  a_mm@K1..K6")
        rows = []
        for tau in taus:
            m_adj, thr, act, damp, atom, incl, eps = cell(e, phi6, cols, adj, cal, tau, q, functional)
            kmin, amms = k_min_curve(phi6, cols, act, eps)
            delta = (k_state - kmin) if (incl and kmin != CENSOR and k_state != CENSOR) else None
            kev = kqu = None
            if incl and functional == "mean":
                for name, msk in (("ev", post_event), ("qu", ~post_event)):
                    d = float(act[msk].mean()) if msk.sum() else -1.0
                    ok = POWER[0] <= d <= POWER[1] and \
                        float((np.abs(m_adj[msk] - thr) <= ATOM_W).mean()) <= ATOM_MASS
                    val = k_min_curve(phi6, cols, act, eps, mask=msk)[0] if ok else None
                    if name == "ev":
                        kev = val
                    else:
                        kqu = val
            rows.append({"tau": tau, "damp": damp, "atom": atom, "included": incl,
                         "k_min": kmin, "delta": delta, "k_event": kev, "k_quiet": kqu,
                         "a_mm": amms})
            print(f"  {tau:>5} {damp:>6.3f} {atom:>6.3f} {str(incl):>5} {fmt(kmin):>6} "
                  f"{fmt(delta):>6} {fmt(kev):>5} {fmt(kqu):>5}  "
                  + " ".join(f"{a:.3f}" for a in amms), flush=True)
        out_rows[functional] = rows
    return {"G": G, "rows_mean": out_rows["mean"], "rows_max": out_rows["max"],
            "kmin_val": kmin_val, "k_state": k_state, "r2_state": r2s,
            "event_fraction": float(post_event.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--npz300", required=True)
    a = ap.parse_args()
    print("AT2B_GROWTH_LAW  [NON-PROMOTIONAL]  spec: AT2B_GROWTH_LAW_SPEC.md", flush=True)
    regs = [analyze(a.npz), analyze(a.npz300)]

    verdicts = []
    for r in regs:
        inc = [row for row in r["rows_mean"] if row["included"]]
        if len(inc) < 3:
            verdicts.append((r["G"], "NO_GATE_READ (<3 included tau-cells)"))
            continue
        kmins = [row["k_min"] for row in inc]
        n_cens = sum(1 for k in kmins if k == CENSOR)
        if n_cens >= (len(inc) + 1) // 2:
            v = "AT2B_CENSORED"
        elif all(row["k_min"] == r["k_state"] for row in inc):
            v = "AT2B_COLLAPSE_VACUOUS (K_min = K_state at every included cell)"
        elif max(kmins) - min(kmins) >= 2 and r["kmin_val"] != CENSOR:
            v = "AT2B_GROWTH_CONFIRMED (within-regime)"
        elif max(kmins) - min(kmins) <= 1:
            v = "AT2B_FLAT_NULL (within-regime)"
        else:
            v = "AT2B_NO_BRANCH (pattern recorded)"
        verdicts.append((r["G"], v + f"  [K_min(mean) over tau: "
                         f"{[(row['tau'], row['k_min']) for row in inc]}]"))
    # cross-regime arm: >=3 matched included cells required (frozen)
    matched = []
    for r2 in regs[0]["rows_mean"]:
        r3 = next((x for x in regs[1]["rows_mean"] if x["tau"] == r2["tau"]), None)
        if r3 and r2["included"] and r3["included"] \
                and CENSOR not in (r2["k_min"], r3["k_min"]):
            matched.append((r2["tau"], r3["k_min"] - r2["k_min"]))
    cross = None
    if len(matched) >= 3:
        gains_ok = all(g >= 1 for _, g in matched)
        vals_flat = abs(regs[1]["kmin_val"] - regs[0]["kmin_val"]) <= 1
        cross = ("AT2B_GROWTH_CONFIRMED (cross-regime arm)"
                 if gains_ok and vals_flat else "cross-regime arm: no fire")
    ev_tokens = []
    for r in regs:
        gaps = [(row["tau"], row["k_event"] - row["k_quiet"]) for row in r["rows_mean"]
                if row["included"] and row["k_event"] not in (None, CENSOR)
                and row["k_quiet"] not in (None, CENSOR)]
        tok = ("AT2B_EVENT_UNPOWERED" if not gaps else
               "AT2B_EVENT_CARRIED" if sum(1 for _, g in gaps if g >= 2) >= 2 else
               "AT2B_EVENT_FLAT")
        ev_tokens.append((r["G"], tok, gaps))

    print("\n(verdicts, per frozen table)")
    for G, v in verdicts:
        print(f"  G={G:.0f}: {v}")
    print(f"  cross-regime: matched-included gains {matched}"
          + (f" -> {cross}" if cross else " (<3 matched: no arm read)"))
    for G, tok, gaps in ev_tokens:
        print(f"  event sub-read G={G:.0f}: {tok}  gaps={gaps}")
    for r in regs:
        deltas = [(row["tau"], row["delta"]) for row in r["rows_mean"] if row["delta"] is not None]
        print(f"  Delta(K_state - K_min) G={r['G']:.0f}: {deltas}")
    out = {"spec": "AT2B_GROWTH_LAW_SPEC.md", "date": "2026-07-03", "regimes": regs,
           "verdicts": [[g, v] for g, v in verdicts], "cross_regime": [matched, cross],
           "event_subread": [[g, t, gp] for g, t, gp in ev_tokens]}
    path = os.path.join(os.path.dirname(a.npz), "at2b_growth_law.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"(wrote) {path}")
    print("  (Non-promotional. 32x32 truncation; no infinite-dim NSE claim.)")


if __name__ == "__main__":
    main()
