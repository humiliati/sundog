#!/usr/bin/env python3
"""AT-4 -- crossover transplant: maintained ledger vs order-blind surface, one stream
(frozen spec: docs/chatv2/AT4_CROSSOVER_TRANSPLANT_SPEC.md). Read-only imports; no
harness change. Non-promotional; 32x32 truncation; licensed grammar only.
Run: python scripts/at4_crossover_transplant.py --out results/proof/at4-g200
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pde_c1_nudging_ledger import (BURNIN, CAL_COUNT, CAL_GAP, INTERVAL, LOOK, Q,
                                   WINDOW, TRANSIENT_SKIP, KolmogorovStepper,
                                   make_cfg, nudged_step, obs_mask, k3_mask)

G, K_OBS, MU = 200.0, 1, 10.0
W_GRID, GRAM_W, N_BINS, HASH_DIM = [250, 500, 1000], [1, 2, 4, 8], 8, 4096
SLICE_PCTL, N_MIN, DELTA, LIVE_MIN = 30, 800, 0.10, 0.95


def integrate_all(cfg, window):
    """Truth (with obs stream + sigs) and both ledgers (full, scrambled) in one pass."""
    st = KolmogorovStepper(cfg)
    mask = obs_mask(st, K_OBS)
    m3 = k3_mask(st)
    u = st.initial_state()
    for _ in range(BURNIN):
        u = st.step(u)
    total = CAL_COUNT + CAL_GAP + window + LOOK + 1
    off = CAL_COUNT + CAL_GAP
    e_series = np.empty(total)
    # calibration pass (truth only)
    uu = u.copy()
    u0 = None
    for t in range(total):
        sig = st.signature(uu)
        e_series[t] = float(np.dot(sig, sig))
        if t == off:
            u0 = uu.copy()
        uu = st.step(uu)
    cal_starts = np.arange(0, CAL_COUNT - LOOK - 1, INTERVAL)
    cal_m = np.array([float(np.max(e_series[s:s + LOOK + 1])) for s in cal_starts])
    e_max = float(np.quantile(cal_m, Q))
    e_win = e_series[off:]
    # calibration stats for surface quantization / liveness (frozen: calibration block)
    # re-walk calibration to collect the observed-mode stream there
    print("(truth) calibration pass done; collecting calib obs stream ...", flush=True)
    uu = u.copy()
    obs_cal = np.empty(CAL_COUNT, dtype=complex)
    obs_idx = np.argwhere(mask)[0]  # representative observed entry (single mode + conj)
    for t in range(CAL_COUNT):
        obs_cal[t] = uu[obs_idx[0], obs_idx[1]]
        uu = st.step(uu)
    re_bins = np.quantile(obs_cal.real, np.linspace(0, 1, N_BINS + 1)[1:-1])
    live_med = float(np.median(obs_cal.real))
    # window pass: truth + full ledger + scrambled ledger together
    print("(window) integrating truth + ledger + scrambled ...", flush=True)
    rng_v = np.random.default_rng(1)
    m = cfg.grid_size
    v = np.fft.fft2(rng_v.standard_normal((m, m)))
    v[~st.dealias_mask] = 0.0
    v[st.zero_mean] = 0.0
    vs = v.copy()          # scrambled ledger state
    rng_scr = np.random.default_rng(2)
    obs_buf = []
    uw = u0.copy()
    obs_stream = np.empty(window, dtype=complex)
    sig_u, sig_v, sig_s, inst = [], [], [], []
    t0 = time.time()
    for t in range(window):
        obs_stream[t] = uw[obs_idx[0], obs_idx[1]]
        obs_buf.append(uw[mask].copy())
        v = nudged_step(st, v, uw, mask, MU)
        u_scr = uw.copy()
        u_scr[mask] = obs_buf[int(rng_scr.integers(0, len(obs_buf)))]
        vs = nudged_step(st, vs, u_scr, mask, MU)
        uw = st.step(uw)
        if t % INTERVAL == 0:
            inst.append(t)
            sig_u.append(st.signature(uw))
            sig_v.append(st.signature(v))
            sig_s.append(st.signature(vs))
    print(f"(window) done in {time.time()-t0:.0f}s", flush=True)
    return (np.array(inst), np.array(sig_u), np.array(sig_v), np.array(sig_s),
            obs_stream, e_win, e_max, re_bins, live_med)


def probe_acc(X, y, tr, te):
    from sklearn.linear_model import LogisticRegression
    if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
        return float("nan")
    mu_x, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    clf = LogisticRegression(max_iter=500).fit((X[tr] - mu_x) / sd, y[tr])
    return float(clf.score((X[te] - mu_x) / sd, y[te]))


def surface_features(obs_stream, starts, w, re_bins, rng=None):
    """Order-blind features of the trailing window [s-w, s]; optional permutation
    (surface order-sanity: must be ~unchanged)."""
    momf, quaf, gramf = [], [], {gw: [] for gw in GRAM_W}
    for s in starts:
        win = obs_stream[max(0, s - w):s + 1]
        if rng is not None:
            win = win[rng.permutation(len(win))]
        re, im = win.real, win.imag
        momf.append([re.mean(), re.std(), re.min(), re.max(), np.abs(re).mean(),
                     im.mean(), im.std(), im.min(), im.max(), np.abs(im).mean()])
        quaf.append(np.concatenate([np.quantile(re, np.linspace(0.1, 0.9, 9)),
                                    np.quantile(im, np.linspace(0.1, 0.9, 9))]))
        sym = np.digitize(re, re_bins)
        for gw in GRAM_W:
            vec = np.zeros(HASH_DIM if gw > 1 else N_BINS)
            if gw == 1:
                for x in sym:
                    vec[x] += 1
            else:
                for i in range(len(sym) - gw + 1):
                    h = 0
                    for j in range(gw):
                        h = (h * 131 + int(sym[i + j])) % HASH_DIM
                    vec[h] += 1
            gramf[gw].append(vec)
    out = {"moments": np.array(momf), "quantiles": np.array(quaf)}
    for gw in GRAM_W:
        out[f"gram_w{gw}"] = np.array(gramf[gw])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(f"AT4_CROSSOVER_TRANSPLANT  G={G:.0f} carrier=(K{K_OBS}, mu{MU:g})  "
          f"[NON-PROMOTIONAL]", flush=True)
    cfg = make_cfg(G)
    inst, sig_u, sig_v, sig_s, obs, e_win, e_max, re_bins, live_med = \
        integrate_all(cfg, WINDOW)

    # labels + margins at post-transient eval instants
    start = int(len(inst) * TRANSIENT_SKIP)
    inst_pt = inst[start:]
    m_adj = np.array([float(np.max(e_win[s:s + LOOK + 1])) for s in inst_pt])
    y = (m_adj > e_max).astype(int)
    margin = np.abs(m_adj - e_max)
    band = np.percentile(margin, SLICE_PCTL)
    sl = margin <= band
    print(f"(labels) n={len(y)} damp={y.mean():.3f} | slice mass={sl.mean():.3f} "
          f"n_slice={int(sl.sum())} slice-damp={y[sl].mean():.3f}", flush=True)

    split = int(0.7 * len(y))
    gap = max(1, 2500 // INTERVAL)
    tr_all = np.arange(0, split)
    te_all = np.arange(split + gap, len(y))
    res = {"readers": {}, "surface": {}, "surface_permuted": {}}

    def sliced(idx):
        return idx[sl[idx]]

    readers = {"ledger": sig_v[start:], "scrambled": sig_s[start:], "ceiling_truth": sig_u[start:]}
    for name, X in readers.items():
        res["readers"][name] = {
            "bulk": probe_acc(X, y, tr_all, te_all),
            "slice": probe_acc(X, y, sliced(tr_all), sliced(te_all))}
        print(f"  {name:>14}: bulk={res['readers'][name]['bulk']:.3f} "
              f"slice={res['readers'][name]['slice']:.3f}", flush=True)

    rng_perm = np.random.default_rng(3)
    surf_max_slice = -1.0
    live_acc = float("nan")
    for w in W_GRID:
        feats = surface_features(obs, inst_pt, w, re_bins)
        featsP = surface_features(obs, inst_pt, w, re_bins, rng=rng_perm)
        for pname, X in feats.items():
            acc_b = probe_acc(X, y, tr_all, te_all)
            acc_s = probe_acc(X, y, sliced(tr_all), sliced(te_all))
            accP = probe_acc(featsP[pname], y, sliced(tr_all), sliced(te_all))
            res["surface"][f"W{w}_{pname}"] = {"bulk": acc_b, "slice": acc_s}
            res["surface_permuted"][f"W{w}_{pname}"] = accP
            if acc_s == acc_s:
                surf_max_slice = max(surf_max_slice, acc_s)
            print(f"  surface W={w:>4} {pname:>10}: bulk={acc_b:.3f} slice={acc_s:.3f} "
                  f"perm={accP:.3f}", flush=True)
        if w == 500:  # liveness: bag-determined axis, moment arm, on-slice
            y_live = (np.array([obs[max(0, s - w):s + 1].real.mean() for s in inst_pt])
                      > live_med).astype(int)
            live_acc = probe_acc(feats["moments"], y_live, sliced(tr_all), sliced(te_all))
            print(f"  (liveness) window-mean axis on-slice: {live_acc:.3f}", flush=True)

    # frozen branch table
    led = res["readers"]["ledger"]["slice"]
    scr = res["readers"]["scrambled"]["slice"]
    n_slice_total = int(sl.sum())
    if n_slice_total < N_MIN:
        verdict = f"AT4_SLICE_THIN (n={n_slice_total} < {N_MIN}; mass/skew reported)"
    elif not (live_acc == live_acc and live_acc >= LIVE_MIN):
        verdict = f"AT4_DEAD_APPARATUS (liveness {live_acc:.3f} < {LIVE_MIN})"
    elif led >= surf_max_slice + DELTA and led >= scr + DELTA:
        verdict = (f"AT4_CROSSOVER_CONFIRMED - ledger {led:.3f} >= surface_max "
                   f"{surf_max_slice:.3f}+{DELTA} and >= scrambled {scr:.3f}+{DELTA} on-slice")
    elif surf_max_slice >= led - 0.05:
        verdict = (f"AT4_SURFACE_SUFFICIENT - surface_max {surf_max_slice:.3f} >= ledger "
                   f"{led:.3f}-0.05: the label is window-statistic-determined on-slice")
    else:
        verdict = (f"AT4_NO_BRANCH (pattern recorded: ledger {led:.3f}, surface "
                   f"{surf_max_slice:.3f}, scrambled {scr:.3f})")
    out = {"spec": "AT4_CROSSOVER_TRANSPLANT_SPEC.md", "date": "2026-07-04",
           "carrier": {"G": G, "k_obs": K_OBS, "mu": MU}, "e_max": e_max,
           "slice_pctl": SLICE_PCTL, "n_slice": n_slice_total,
           "surface_max_slice": surf_max_slice, "liveness": live_acc,
           "results": res, "verdict": verdict}
    with open(os.path.join(a.out, "at4_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nVERDICT: {verdict}")
    print(f"(wrote) {a.out}/at4_summary.json")
    print("  (Non-promotional. Relay-form caveat carries over; no world-model language.)")


if __name__ == "__main__":
    main()
