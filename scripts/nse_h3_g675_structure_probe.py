#!/usr/bin/env python3
"""Dig-in analysis of the fenced NSE-H3-FORCING-GENERAL-RELATIVE positive: is there
a control-sufficient regime-2 NUCLEUS at G=675, or is the positive uniformly thin?

Pure post-processing on the banked samples (signatures + actions only; the 700 MB
high-mode array is NOT loaded -- control-sufficiency = action-disagree needs only
actions + the signature neighbour structure). One BallTree pass. Non-promotional.

Reads results/proof/c1-h3-kf3-g675-adaptive/samples.npz.
"""
import json, os, sys, time

import numpy as np
from sklearn.neighbors import BallTree

SAMPLES = os.path.join("results", "proof", "c1-h3-kf3-g675-adaptive", "samples.npz")
K, DENSE_KMIN = 50, 10   # A's density criterion


def frac(mask_num, mask_den):
    d = int(mask_den.sum())
    return (int(mask_num.sum()) / d) if d else float("nan"), d


def main():
    t0 = time.time()
    z = np.load(SAMPLES)
    sig = z["signatures"].astype(np.float64)
    act = z["actions"]
    eps = float(z["epsilon_k"])
    n = len(sig)
    E_low = np.sum(sig ** 2, axis=1)
    print(f"[probe] n={n}  eps_K={eps:.5f}  damp_frac={float(act.mean()):.4f}  "
          f"[load {time.time()-t0:.0f}s]", flush=True)

    tree = BallTree(sig, metric="euclidean")
    dist, idx = tree.query(sig, k=K)
    nbr_dist, nbr_idx = dist[:, 1:], idx[:, 1:]
    within = nbr_dist <= eps
    dense_count = within.sum(axis=1)
    dense = dense_count >= DENSE_KMIN
    print(f"[probe] query done [{time.time()-t0:.0f}s]", flush=True)

    cov_any = float((dense_count >= 1).mean())
    cov_dense = float(dense.mean())
    print(f"[probe] coverage: >=1-near {cov_any:.4f} (frozen candidate ~0.469)  "
          f">= {DENSE_KMIN}-near (dense) {cov_dense:.4f} (A sliver ~0.036)", flush=True)

    # coverage / density by energy tercile -- where is the witness resolvable?
    q1, q2 = np.quantile(E_low, [1 / 3, 2 / 3])
    print("[probe] by energy tercile:", flush=True)
    for lab, m in (("lowE ", E_low <= q1), ("midE ", (E_low > q1) & (E_low <= q2)), ("highE", E_low > q2)):
        print(f"    {lab}: cov_any={float((dense_count[m] >= 1).mean()):.3f}  "
              f"cov_dense={float(dense[m].mean()):.3f}  median_near_count={float(np.median(dense_count[m])):.1f}",
              flush=True)

    # control-sufficiency: action-disagree among near pairs (directed; symmetric).
    adiff = act[:, None] != act[nbr_idx]
    both_dense = dense[:, None] & dense[nbr_idx]
    d_all, n_all = frac(within & adiff, within)
    d_core, n_core = frac(within & both_dense & adiff, within & both_dense)
    d_sparse, n_sparse = frac(within & ~both_dense & adiff, within & ~both_dense)
    print(f"[probe] disagree among near pairs (directed):", flush=True)
    print(f"    all-near     : {d_all:.4f}  (n_dir {n_all})", flush=True)
    print(f"    DENSE-CORE   : {d_core:.4f}  (n_dir {n_core})  <- the nucleus", flush=True)
    print(f"    sparse-halo  : {d_sparse:.4f}  (n_dir {n_sparse})", flush=True)

    # readout
    verdict = ("NUCLEUS_CONTROL_SUFFICIENT" if (d_core == d_core and d_core <= 0.05)
               else "NO_CLEAN_NUCLEUS")
    print(f"[probe] READ: dense core disagree {d_core:.4f} -> "
          f"{'anchor-like (clean regime-2 nucleus)' if verdict=='NUCLEUS_CONTROL_SUFFICIENT' else 'not clean'}",
          flush=True)
    out = {
        "cell": "fallback_v7_g675_kf3", "n": n, "epsilon_k": eps,
        "coverage_any": cov_any, "coverage_dense": cov_dense, "dense_kmin": DENSE_KMIN,
        "disagree_all_near": d_all, "disagree_dense_core": d_core, "disagree_sparse_halo": d_sparse,
        "dense_core_directed_pairs": n_core, "dense_sample_count": int(dense.sum()),
        "nucleus_verdict": verdict,
    }
    outp = os.path.join("results", "proof", "c1-h3-kf3-g675-adaptive", "structure_probe.json")
    json.dump(out, open(outp, "w"), indent=1)
    print(f"[probe] wrote {outp}  [{time.time()-t0:.0f}s]  (non-promotional)", flush=True)


if __name__ == "__main__":
    main()
