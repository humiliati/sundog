#!/usr/bin/env python3
"""AT-4b rung 0 -- data/surface admission for the rollout crossover (frozen spec:
docs/chatv2/AT4B_ROLLOUT_DETRENDED_SPEC.md section 1). Truth-only; no ledgers; read-only
imports. Non-promotional. v1.2 command:
python scripts/at4b0_admission.py --formation-version v1.2 --out results/proof/at4b0-g300-v12
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pde_c1_nudging_ledger import BURNIN, INTERVAL, KolmogorovStepper, make_cfg, obs_mask
from at4_crossover_transplant import probe_acc, surface_features

G, EPOCH, RQ = 300.0, 50_000, 0.70
TAUS, W_GRID, BANDS = [1500, 2500, 5000], [1000, 2500, 5000], [0.30, 0.50, 1.00]
TAIL = max(TAUS) + 1
DAMP_WIN, BAL_WIN, N_TEST_MIN, SURF_STOP, LIVE_MIN = (0.20, 0.40), (0.40, 0.60), 400, 0.90, 0.95
FORMATIONS = {
    "v1.1": {
        "stream_steps": 500_000,
        "split": "contiguous_70_30",
    },
    "v1.2": {
        "stream_steps": 2_000_000,
        "split": "blocked_alternating",
        "block_steps": 50_000,
        "guard_steps": 5_000,
    },
}


def rolling_quantile(x, win, q, stride):
    """q-quantile of x over [i-win, i) evaluated at stride points (i >= win)."""
    pts = np.arange(win, len(x), stride)
    return pts, np.array([np.quantile(x[p - win:p], q) for p in pts])


def split_indices(usable, formation_version):
    formation = FORMATIONS[formation_version]
    if formation["split"] == "contiguous_70_30":
        split = int(0.7 * len(usable))
        gap = max(1, 2500 // INTERVAL)
        tr = np.arange(0, split)
        te = np.arange(split + gap, len(usable))
        return tr, te, {"split": "contiguous_70_30", "gap_steps": gap * INTERVAL}

    block = formation["block_steps"]
    guard = formation["guard_steps"]
    cycle = block + guard
    rel = usable - EPOCH
    block_idx = rel // cycle
    offset = rel % cycle
    in_block = offset < block
    tr = np.where(in_block & ((block_idx % 2) == 0))[0]
    te = np.where(in_block & ((block_idx % 2) == 1))[0]
    return tr, te, {
        "split": "blocked_alternating",
        "block_steps": block,
        "guard_steps": guard,
        "assignment": "even_blocks_train_odd_blocks_test",
    }


def mean_or_nan(x):
    return float(np.mean(x)) if len(x) else float("nan")


def quintile_means(y):
    cuts = np.linspace(0, len(y), 6, dtype=int)
    return [mean_or_nan(y[cuts[i]:cuts[i + 1]]) for i in range(5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formation-version", choices=sorted(FORMATIONS), default="v1.2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    formation = FORMATIONS[a.formation_version]
    stream = formation["stream_steps"]
    os.makedirs(a.out, exist_ok=True)
    print(
        f"AT4B0_ADMISSION  G={G:.0f}  formation={a.formation_version}  "
        "truth-only  [NON-PROMOTIONAL]",
        flush=True,
    )

    cfg = make_cfg(G)
    st = KolmogorovStepper(cfg)
    mask = obs_mask(st, 1)
    obs_idx = np.argwhere(mask)[0]
    u = st.initial_state()
    for _ in range(BURNIN):
        u = st.step(u)
    total = stream + TAIL
    e = np.empty(total)
    obs = np.empty(total, dtype=complex)
    t0 = time.time()
    for t in range(total):
        sig = st.signature(u)
        e[t] = float(np.dot(sig, sig))
        obs[t] = u[obs_idx[0], obs_idx[1]]
        u = st.step(u)
    print(f"(truth) {total} steps in {time.time()-t0:.0f}s", flush=True)

    # detrended autocorrelation time (E_low minus rolling median over the epoch)
    med_pts, med = rolling_quantile(e[:stream], EPOCH, 0.5, INTERVAL)
    det = e[med_pts] - med
    det -= det.mean()
    ac = np.correlate(det, det, "full")[len(det) - 1:] / (det @ det)
    zc = int(np.argmax(ac < 0)) * INTERVAL if (ac < 0).any() else -1
    print(f"(detrended) autocorr first zero ~ {zc} steps", flush=True)

    # v1.1 matched-functional rolling threshold: threshold_s = rolling q of the trailing
    # lookahead-maxes m_tau(p), p in [s-EPOCH, s-tau) — windows entirely in s's past.
    from numpy.lib.stride_tricks import sliding_window_view
    usable = np.arange(EPOCH, stream, INTERVAL)
    usable = usable[usable + max(TAUS) < total]
    tau_primary, y = None, None
    horizon_rows = []
    for tau in TAUS:
        mfull = sliding_window_view(e, tau).max(axis=1)  # mfull[t] = max e[t:t+tau]
        m = mfull[usable + 1]                            # m_tau(s) = max e(s, s+tau]
        thr_u = np.array([float(np.quantile(mfull[s - EPOCH + 1:s - tau + 1], RQ))
                          for s in usable])
        yy = (m > thr_u).astype(int)
        damp = float(yy.mean())
        beyond = bool(tau > zc > 0 or zc < 0)
        ok = DAMP_WIN[0] <= damp <= DAMP_WIN[1]
        horizon_rows.append({"tau": tau, "damp": damp, "beyond_autocorr": beyond, "clears": bool(ok)})
        print(f"(horizon) tau={tau}: damp={damp:.3f} beyond-autocorr={tau > zc > 0 or zc < 0}"
              f" {'<- PRIMARY' if ok and tau_primary is None else ''}", flush=True)
        if ok and tau_primary is None:
            tau_primary, y = tau, yy
            margin = np.abs(m - thr_u)
    if tau_primary is None:
        print("\nVERDICT: AT4B_UNPOWERED_INPUT (no horizon clears the damp window)")
        with open(os.path.join(a.out, "at4b0_summary.json"), "w") as f:
            json.dump(
                {
                    "verdict": "AT4B_UNPOWERED_INPUT",
                    "stage": "horizon",
                    "formation_version": a.formation_version,
                    "stream_steps": stream,
                    "split": formation["split"],
                    "horizon_rows": horizon_rows,
                    "surface_read": False,
                },
                f,
                indent=1,
            )
        return

    tr_all, te_all, split_meta = split_indices(usable, a.formation_version)
    q_damp = quintile_means(y)
    train_damp = mean_or_nan(y[tr_all])
    test_damp = mean_or_nan(y[te_all])
    print(
        f"(split) {split_meta['split']}: n_train={len(tr_all)} n_test={len(te_all)} "
        f"train_damp={train_damp:.3f} test_damp={test_damp:.3f}",
        flush=True,
    )
    print("(split) quintile damp: " + " -> ".join(f"{q:.3f}" for q in q_damp), flush=True)

    # balanced slice construction (frozen candidate ladder + one registered repair)
    slice_mask, slice_note = None, None
    candidate_rows = []
    for beta in BANDS:
        band = np.percentile(margin, beta * 100)
        cand = margin <= band
        for repaired in (False, True):
            msk = cand.copy()
            if repaired:  # majority-subsample within band to 50/50 (seed 0)
                rng = np.random.default_rng(0)
                idx1, idx0 = np.where(msk & (y == 1))[0], np.where(msk & (y == 0))[0]
                if min(len(idx1), len(idx0)) == 0:
                    continue
                keep = min(len(idx1), len(idx0))
                drop = idx1 if len(idx1) > len(idx0) else idx0
                msk[rng.permutation(drop)[keep:]] = False
            te_m = te_all[msk[te_all]]
            tr_m = tr_all[msk[tr_all]]
            te_balance = mean_or_nan(y[te_m])
            candidate_rows.append(
                {
                    "beta": beta,
                    "repaired": repaired,
                    "n_train": int(len(tr_m)),
                    "n_test": int(len(te_m)),
                    "train_balance": mean_or_nan(y[tr_m]),
                    "test_balance": te_balance,
                }
            )
            if len(te_m) >= N_TEST_MIN and BAL_WIN[0] <= te_balance <= BAL_WIN[1]:
                slice_mask, slice_note = msk, f"beta={beta}{' repaired' if repaired else ''}"
                break
        if slice_mask is not None:
            break
    if slice_mask is None:
        print("\nVERDICT: AT4B_UNPOWERED_INPUT (no balanced slice with test mass)")
        with open(os.path.join(a.out, "at4b0_summary.json"), "w") as f:
            json.dump(
                {
                    "verdict": "AT4B_UNPOWERED_INPUT",
                    "stage": "slice",
                    "formation_version": a.formation_version,
                    "stream_steps": stream,
                    "split": split_meta,
                    "tau": tau_primary,
                    "detrended_autocorr_steps": zc,
                    "horizon_rows": horizon_rows,
                    "quintile_damp": q_damp,
                    "train_damp": train_damp,
                    "test_damp": test_damp,
                    "candidate_rows": candidate_rows,
                    "surface_read": False,
                },
                f,
                indent=1,
            )
        return
    tr_s, te_s = tr_all[slice_mask[tr_all]], te_all[slice_mask[te_all]]
    print(f"(slice) {slice_note}: n_tr={len(tr_s)} n_te={len(te_s)} "
          f"te-balance={y[te_s].mean():.3f}", flush=True)

    # surface suite on the balanced slice (strongest registered shot)
    re_bins = np.quantile(obs[:EPOCH].real, np.linspace(0, 1, 9)[1:-1])
    live_med = float(np.median(obs[:EPOCH].real))
    surf_max, table = -1.0, {}
    live_acc = float("nan")
    for w in W_GRID:
        feats = surface_features(obs, usable, w, re_bins)
        for pname, X in feats.items():
            acc = probe_acc(X, y, tr_s, te_s)
            table[f"W{w}_{pname}"] = acc
            if acc == acc:
                surf_max = max(surf_max, acc)
            print(f"  surface W={w:>4} {pname:>10}: slice={acc:.3f}", flush=True)
        if w == W_GRID[0]:
            y_live = (np.array([obs[max(0, s - w):s + 1].real.mean() for s in usable])
                      > live_med).astype(int)
            live_acc = probe_acc(feats["moments"], y_live, tr_s, te_s)
            print(f"  (liveness) window-mean axis on-slice: {live_acc:.3f}", flush=True)

    if not (live_acc == live_acc and live_acc >= LIVE_MIN):
        verdict = f"AT4B_DEAD_APPARATUS (liveness {live_acc:.3f})"
    elif surf_max >= SURF_STOP:
        verdict = (f"AT4B_SURFACE_SUFFICIENT_ADMISSION - surface_max {surf_max:.3f} >= "
                   f"{SURF_STOP} on the balanced slice of the hard regime: the crossover "
                   "form closes cleanly; rung 1 does not run")
    else:
        verdict = (f"AT4B0_ROOM_EXISTS - surface_max {surf_max:.3f} < {SURF_STOP}: "
                   f"rung 1 (rollout carrier at tau={tau_primary}) is unblocked")
    out = {"spec": "AT4B_ROLLOUT_DETRENDED_SPEC.md", "date": "2026-07-05", "G": G,
           "formation_version": a.formation_version, "stream_steps": stream, "split": split_meta,
           "tau_primary": tau_primary, "detrended_autocorr_steps": zc,
           "horizon_rows": horizon_rows, "quintile_damp": q_damp,
           "train_damp": train_damp, "test_damp": test_damp,
           "slice": slice_note, "n_test_slice": int(len(te_s)),
           "test_balance": float(y[te_s].mean()), "surface_max": surf_max,
           "liveness": live_acc, "surface_table": table, "surface_read": True,
           "verdict": verdict}
    with open(os.path.join(a.out, "at4b0_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nVERDICT: {verdict}")
    print(f"(wrote) {a.out}/at4b0_summary.json")
    print("  (Non-promotional. Truth-only admission; no ledger compute spent.)")


if __name__ == "__main__":
    main()
