#!/usr/bin/env python3
"""NSE-H1 rung 1 -- fiber transfer on the rung-0 v1.1 export (frozen spec:
docs/chatv2/NSE_H1_JSELECTOR_SPEC.md sections 4-5). Pure post-processing of
h1_export.npz; no integration, no harness change. Non-promotional.

OWNER-GATED: running this on a real export reads adjudication numbers.
Run:   python scripts/nse_h1_jselector_fiber.py --grashof 200 --export results/proof/nse-h1-g200-v11/h1_export.npz --out results/proof/nse-h1-g200-v11
Self-test (synthetic only): --self-test
"""
import argparse, json, os, sys

import numpy as np

# Frozen (spec section 4). Banked matched radii from
# results/proof/c1-paired-fiber-g{200,300}/manifest.json -- full precision.
EPS_K = {200: 0.06059758455293647, 300: 0.06642189017389008}
DELTA_ACTION, K_NN = 0.10, 30
MIN_PAIRS = 100
N_SHUFFLE, SHUFFLE_SEED = 20, 3
DISAGREE_MAX, REL_FACTOR, SHUFFLE_FLOOR = 0.05, 2.0, 0.25


def fiber_pairs(sig, eps):
    """Unique pairs (i<j) within eps in signature space (exact, no subsampling)."""
    from scipy.spatial.distance import cdist
    d = cdist(sig, sig)
    iu = np.triu_indices(len(sig), 1)
    within = d[iu] <= eps
    return iu[0][within], iu[1][within]


def disagree_fraction(pi, pj, labels):
    if len(pi) == 0:
        return float("nan")
    return float(np.mean(labels[pi] != labels[pj]))


def shuffle_floor(pi, pj, labels):
    rng = np.random.default_rng(SHUFFLE_SEED)
    vals = []
    for _ in range(N_SHUFFLE):
        perm = rng.permutation(labels)
        vals.append(disagree_fraction(pi, pj, perm))
    return float(np.mean(vals))


def knn_read(sig, labels, eps):
    """Mirror of the harness aggregate_knn (Fork A): k=30 incl self, minority
    fraction among fidelity-passing (r_k <= eps) samples, incompat if
    minority > delta_action."""
    from sklearn.neighbors import BallTree
    n = len(sig)
    k = min(K_NN, n)
    dist, idx = BallTree(sig, metric="euclidean").query(sig, k=k)
    r_k = dist[:, -1]
    nb = labels[idx].astype(int)
    damp_count = nb.sum(axis=1)
    majority_count = np.maximum(damp_count, k - damp_count)
    minority = 1.0 - majority_count / float(k)
    fid = r_k <= eps
    f_count = int(fid.sum())
    incompat = int((fid & (minority > DELTA_ACTION)).sum())
    return {"k": k, "fidelity_count": f_count, "fidelity_coverage": f_count / n,
            "incompat_count": incompat,
            "incompat_fraction": incompat / f_count if f_count else float("nan")}


def run(grashof, export, out):
    eps = EPS_K[int(grashof)]
    z = np.load(export)
    nc = int(z["n_cal"])
    sig, a_j, y_pi = z["sig"][nc:], z["a_j"][nc:], z["y_pi"][nc:]
    print(f"NSE_H1_FIBER rung 1  G={grashof:.0f}  eps_K={eps:.6f}  "
          f"n_eval={len(sig)}  [NON-PROMOTIONAL]", flush=True)
    pi, pj = fiber_pairs(sig, eps)
    n_pairs = len(pi)
    print(f"(pairs) {n_pairs} unique pairs within eps_K (gate >= {MIN_PAIRS})", flush=True)
    if n_pairs < MIN_PAIRS:
        result = {"grashof": grashof, "eps_k": eps, "n_pairs": n_pairs,
                  "branch": "NSE-H1-UNPOWERED(rung1-pairs)",
                  "note": "power option A (stride 50) is the one registered extension"}
    else:
        dis_aj = disagree_fraction(pi, pj, a_j)
        dis_yp = disagree_fraction(pi, pj, y_pi)
        floor = shuffle_floor(pi, pj, a_j)
        knn_aj = knn_read(sig, a_j, eps)
        knn_yp = knn_read(sig, y_pi, eps)
        transfer = bool(dis_aj <= DISAGREE_MAX and dis_aj <= REL_FACTOR * dis_yp
                        and floor >= SHUFFLE_FLOOR)
        result = {
            "spec": "NSE_H1_JSELECTOR_SPEC.md", "grashof": grashof, "eps_k": eps,
            "n_eval": int(len(sig)), "n_pairs": n_pairs,
            "fiber_disagree_a_j": dis_aj, "fiber_disagree_y_pi": dis_yp,
            "shuffle_floor_mean": floor,
            "banked_pi_hat_disagree": 0.0367 if int(grashof) == 200 else 0.0382,
            "knn_a_j": knn_aj, "knn_y_pi": knn_yp,
            "criteria": {"disagree_max": DISAGREE_MAX, "rel_factor": REL_FACTOR,
                         "shuffle_floor": SHUFFLE_FLOOR},
            "cell_transfer": transfer,
        }
        print(f"(reads) a_J disagree={dis_aj:.4f}  y_pi disagree={dis_yp:.4f}  "
              f"shuffle={floor:.3f}  knn_incompat a_J={knn_aj['incompat_fraction']:.4f} "
              f"(cov {knn_aj['fidelity_coverage']:.3f})", flush=True)
        print(f"CELL TRANSFER: {transfer}", flush=True)
    path = os.path.join(out, "h1_fiber.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"(wrote) {path}", flush=True)
    print("  (Non-promotional. The H1 verdict combines both cells per the frozen table.)",
          flush=True)


def self_test():
    print("SELF-TEST (synthetic only; non-verdict)", flush=True)
    rng = np.random.default_rng(0)
    # two clusters, labels constant per cluster -> disagree 0, shuffle high
    c = rng.integers(0, 2, 400)
    sig = c[:, None] * 10.0 + rng.normal(0, 0.01, (400, 18))
    y = c.astype(bool)
    pi, pj = fiber_pairs(sig, 0.06)
    assert len(pi) >= MIN_PAIRS, "F1 pair count"
    assert disagree_fraction(pi, pj, y) == 0.0, "F1 constancy"
    assert shuffle_floor(pi, pj, y) > 0.25, "F1 shuffle floor"
    print(f"  F1 PASS  fiber-constant labels: disagree 0.000, shuffle "
          f"{shuffle_floor(pi, pj, y):.3f}, pairs {len(pi)}", flush=True)
    # random labels on the same pairs -> disagree ~ 2p(1-p)
    yr = rng.random(400) < 0.3
    d = disagree_fraction(pi, pj, yr)
    assert abs(d - 2 * 0.3 * 0.7) < 0.05, "F2"
    print(f"  F2 PASS  random labels: disagree {d:.3f} ~ 0.42", flush=True)
    # kNN mirror: cluster-constant labels -> zero incompat at full coverage
    k = knn_read(sig, y, 0.06)
    assert k["fidelity_coverage"] > 0.9 and k["incompat_fraction"] == 0.0, "F3"
    print(f"  F3 PASS  knn mirror: coverage {k['fidelity_coverage']:.2f}, "
          f"incompat 0.000", flush=True)
    # sparse data -> pair gate would fire
    sparse = rng.normal(0, 100.0, (50, 18))
    ps, _ = fiber_pairs(sparse, 0.06)
    assert len(ps) < MIN_PAIRS, "F4"
    print("  F4 PASS  sparse data trips the pair gate", flush=True)
    print("SELF-TEST 4/4 PASS", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grashof", type=float)
    ap.add_argument("--export")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if a.grashof is None or a.export is None or a.out is None:
        ap.error("--grashof, --export, --out required unless --self-test")
    run(a.grashof, a.export, a.out)


if __name__ == "__main__":
    main()
