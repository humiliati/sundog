#!/usr/bin/env python3
"""
Riemann Q3 control — window / unfolding robustness sweep of the Probe 05
reversibility null.

Answers External Review Packet Q3 ("window / unfolding call"): re-run the
IDENTICAL S2 reversibility statistic D with MORE than the first 5,000 Odlyzko
ordinates (the local cache holds all 100,000) and under alternative unfoldings,
and show the bounded null |D| <= tau_ind = 3/sqrt(m) persists — i.e. it is a
substrate-level feature, not a 5,000-window or unfolding artifact.

Self-check: at N=5000 with the registered Riemann-von Mangoldt unfolding this
MUST reproduce the canonical runner's result bit-for-bit
(nPlus=2483, nMinus=2515, D=-0.006402561024409764). If it does not, the sweep is
not trustworthy and aborts.

Not a structural-zero probe; not evidence for or against RH.
"""
import argparse
import hashlib
import json
import os
import numpy as np

TWO_PI = 2.0 * np.pi
TIE_TOL = 1e-8
REGISTERED_SHA = "3436c916a7878261ac183fd7b9448c9a4736b8bbccf1356874a6ce1788541632"
REG_N5000 = {"N": 5000, "m": 4998, "nPlus": 2483, "nMinus": 2515,
             "D": -0.006402561024409764}


def load_zeros(path, nmax):
    vals = []
    with open(path, "r") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            vals.append(float(t))
            if len(vals) >= nmax:
                break
    return np.array(vals, dtype=np.float64)


def D_stat(s):
    """Identical S2 sign statistic and tie rule to the probe."""
    delta = s[:-1] - s[1:]
    signs = np.where(delta > TIE_TOL, 1, np.where(delta < -TIE_TOL, -1, 0))
    m = signs.size
    nPlus = int((signs > 0).sum())
    nMinus = int((signs < 0).sum())
    nTie = int((signs == 0).sum())
    D = float(signs.sum() / m)
    tau = float(3.0 / np.sqrt(m))
    return {"D": D, "m": m, "nPlus": nPlus, "nMinus": nMinus, "nTie": nTie,
            "tau_ind": tau, "abs_D_le_tau_ind": bool(abs(D) <= tau)}


def unfold_rvm(gaps, centers):
    """Registered unfolding: s_i = gap_i * log(center_i/2pi)/2pi."""
    return gaps * (np.log(centers / TWO_PI) / TWO_PI)


def unfold_raw(gaps, centers):
    """Alternative: no unfolding (raw gaps). Isolates unfolding-sensitivity of the sign."""
    return gaps.copy()


def unfold_smoothed(gaps, centers, win=101):
    """Alternative: empirical local-density unfolding (gap / windowed-mean-gap),
    trimming win//2 from each end to avoid convolution edge artifacts."""
    kernel = np.ones(win) / win
    local_mean = np.convolve(gaps, kernel, mode="same")
    s = gaps / local_mean
    h = win // 2
    return s[h:-h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",
                    default="results/riemann/probe01-isotropy-zero-pairs/source/zeros1.txt")
    ap.add_argument("--out", default="results/riemann/probe05-q3-window-sweep")
    ap.add_argument("--windows", default="5000,10000,20000,50000,100000")
    args = ap.parse_args()

    sha = hashlib.sha256(open(args.source, "rb").read()).hexdigest()
    z = load_zeros(args.source, 100000)
    gaps_all = z[1:] - z[:-1]
    centers_all = (z[1:] + z[:-1]) / 2.0

    # --- self-check at N=5000 (registered RvM unfolding) ---
    s5 = unfold_rvm(gaps_all[:4999], centers_all[:4999])
    chk = D_stat(s5)
    self_check_pass = (
        chk["nPlus"] == REG_N5000["nPlus"]
        and chk["nMinus"] == REG_N5000["nMinus"]
        and abs(chk["D"] - REG_N5000["D"]) < 1e-12
    )
    if not self_check_pass:
        print(json.dumps({"SELF_CHECK": "FAIL", "got": chk, "expected": REG_N5000}, indent=2))
        raise SystemExit("Self-check failed: reimplementation does not match the canonical runner.")

    # --- axis 1: ordinate-count sweep (registered RvM unfolding) ---
    windows = [int(x) for x in args.windows.split(",")]
    sweep = []
    for N in windows:
        s = unfold_rvm(gaps_all[:N - 1], centers_all[:N - 1])
        r = D_stat(s)
        r["N"] = N
        r["max_height"] = float(z[N - 1])
        sweep.append(r)

    # --- axis 2: alternative unfoldings (at N=5000 and the full N=100000) ---
    unfoldings = {}
    for N in (5000, 100000):
        gi, ci = gaps_all[:N - 1], centers_all[:N - 1]
        unfoldings[str(N)] = {
            "rvm_registered": D_stat(unfold_rvm(gi, ci)),
            "raw_no_unfolding": D_stat(unfold_raw(gi, ci)),
            "smoothed_local_density_win101": D_stat(unfold_smoothed(gi, ci, 101)),
        }

    all_bounded = all(r["abs_D_le_tau_ind"] for r in sweep) and all(
        u["abs_D_le_tau_ind"]
        for N in unfoldings.values() for u in N.values()
    )
    verdict = (
        "BOUNDED NULL PERSISTS across all windows (5k..100k) and unfoldings (RvM, "
        "raw, smoothed): |D| <= tau_ind everywhere. The reversibility null is a "
        "substrate-level feature, not a 5,000-window or unfolding artifact."
        if all_bounded else
        "NOT UNIFORMLY BOUNDED: at least one window/unfolding breaches the floor; "
        "investigate before claiming substrate-level robustness."
    )

    out = {
        "probe": "riemann_q3_window_unfolding_sweep",
        "purpose": "External Review Packet Q3 — is the Probe 05 reversibility null robust to window size and unfolding?",
        "source": {"path": args.source, "sha256": sha,
                   "registered_sha_match": sha == REGISTERED_SHA},
        "self_check_n5000": {"pass": self_check_pass, "observed": chk, "registered": REG_N5000},
        "window_sweep_rvm": sweep,
        "alt_unfoldings": unfoldings,
        "verdict_all_bounded": bool(all_bounded),
        "verdict": verdict,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "q3_window_sweep_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({
        "self_check_n5000_pass": self_check_pass,
        "source_sha_match": sha == REGISTERED_SHA,
        "window_sweep": [{"N": r["N"], "D": round(r["D"], 6),
                          "tau_ind": round(r["tau_ind"], 6),
                          "bounded": r["abs_D_le_tau_ind"]} for r in sweep],
        "alt_unfoldings_N100000": {k: {"D": round(v["D"], 6), "bounded": v["abs_D_le_tau_ind"]}
                                   for k, v in unfoldings["100000"].items()},
        "verdict_all_bounded": all_bounded,
    }, indent=2))


if __name__ == "__main__":
    main()
