#!/usr/bin/env python3
"""Sundog Certificate Problem -- v1 capacity harness (Prange ISD attacker).

Frozen contract: docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V1_SLATE.md

Measures the capacity-relative one-wayness threshold C of the syndrome certificate:
the effort at which Prange information-set decoding, given ONLY the compact shadow
z = Hy, recovers a valid light deviation witness e* (He* = z, wt(e*) <= tau) -- while
the witness-verifier stays cheap. Safety predicate is the existence form
Safe(y) := exists e* : He* = Hy and wt(e*) <= tau (planted e is a label only).

Rank-valid convention (frozen): B counts rank-valid information-set trials. A random
size-(n-k) column set whose submatrix is singular is a rank-FAIL draw: charged to
measured ops, audited as rank_fail_draws, NOT counted toward B. The analytic curve
1 - (1-p)^B (p = C(n-k,w)/C(n,w)) is read against rank-valid trials.

Default regime = the FROZEN [128,64] w=12 (do NOT run without the operator go).
--smoke runs a THROWAWAY [64,32] w=6 regime (a different code seed) to validate the
attacker against the analytic prediction BEFORE the frozen run. Deterministic.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---- GF(2) helpers ------------------------------------------------------------

def gf2_matvec(M: np.ndarray, v: np.ndarray):
    out = (M & v[None, :]).sum(axis=1) & 1
    return out.astype(np.uint8), int(M.shape[0] * M.shape[1])


def gf2_inverse(M: np.ndarray):
    """Gauss-Jordan over GF(2). Return (inverse | None, ok, bit-ops). Vectorized
    row elimination so per-call cost is O(m) numpy ops."""
    m = M.shape[0]
    A = M.copy().astype(np.uint8)
    I = np.eye(m, dtype=np.uint8)
    ops = 0
    for col in range(m):
        rows = np.nonzero(A[col:, col])[0]
        ops += int(m - col)
        if rows.size == 0:
            return None, False, ops  # singular -> rank-fail
        piv = col + int(rows[0])
        if piv != col:
            A[[col, piv]] = A[[piv, col]]
            I[[col, piv]] = I[[piv, col]]
        mask = A[:, col].astype(bool).copy()
        mask[col] = False
        if mask.any():
            A[mask] ^= A[col]
            I[mask] ^= I[col]
            ops += int(mask.sum()) * m * 2
    return I, True, ops


def make_code(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    m = n - k
    P = rng.integers(0, 2, size=(k, m), dtype=np.uint8)
    G = np.concatenate([np.eye(k, dtype=np.uint8), P], axis=1)
    H = np.concatenate([P.T, np.eye(m, dtype=np.uint8)], axis=1)
    assert (((G @ H.T) & 1).sum() == 0)
    return G, H


def sample_body(n, k, w, rng):
    s = rng.integers(0, 2, size=k, dtype=np.uint8)
    e = np.zeros(n, dtype=np.uint8)
    e[rng.choice(n, size=w, replace=False)] = 1
    return s, e


def observe(G, s, e):
    return ((s @ G) & 1 ^ e).astype(np.uint8)


def verify(H, y, e_star, tau):
    """Existence-semantics witness verifier (op-counted)."""
    ops = 0
    z, o = gf2_matvec(H, y); ops += o
    Hes, o = gf2_matvec(H, e_star); ops += o
    ops += int(H.shape[0]) + int(e_star.shape[0])
    if np.array_equal(Hes, z) and int(e_star.sum()) <= tau:
        return "accept", ops
    return "quarantine", ops


# ---- Prange ISD attacker (rank-valid convention) ------------------------------

def prange_first_success(H, z, n, k, tau, max_B, rng):
    """Run Prange ISD up to max_B RANK-VALID trials. Return
    (first_success_B or None, rank_valid_trials, rank_fail_draws, ops).
    Each rank-valid trial: random size-(n-k) column set J; if H[:,J] invertible,
    candidate e_J = H_J^{-1} z supported on J; success iff wt(e_J) <= tau."""
    m = n - k
    B = 0
    rank_fail = 0
    ops = 0
    while B < max_B:
        J = rng.choice(n, size=m, replace=False)
        Hj = H[:, J]
        inv, ok, eops = gf2_inverse(Hj)
        ops += eops
        if not ok:
            rank_fail += 1            # rank-fail: charged to ops, NOT to B
            continue
        B += 1
        eJ, mvops = gf2_matvec(inv, z); ops += mvops
        if int(eJ.sum()) <= tau:
            return B, B, rank_fail, ops
    return None, B, rank_fail, ops


# ---- experiment ---------------------------------------------------------------

def analytic(n, k, w):
    p = math.comb(n - k, w) / math.comb(n, w)   # success per rank-valid trial
    N = 1.0 / p
    return p, N


def run(n, k, w, tau, code_seed, T, ladder, max_B, run_seed, label, frozen, out_dir):
    G, H = make_code(n, k, code_seed)
    rng = np.random.default_rng(run_seed)
    p, N = analytic(n, k, w)

    # one cheap verifier op-count (flat baseline): verify a genuine safe body.
    s0, e0 = sample_body(n, k, w, rng)
    y0 = observe(G, s0, e0)
    _, check_ops = verify(H, y0, e0, tau)

    # per target: run ISD once up to max_B rank-valid trials, record first success.
    first_success = []
    rank_fail_total = 0
    ops_total = 0
    isd_rng = np.random.default_rng(run_seed + 1)
    for _ in range(T):
        s, e = sample_body(n, k, w, isd_rng)
        z, _ = gf2_matvec(H, observe(G, s, e))
        fb, B_used, rf, ops = prange_first_success(H, z, n, k, tau, max_B, isd_rng)
        first_success.append(fb)
        rank_fail_total += rf
        ops_total += ops

    rows = []
    for B in ladder:
        succ = sum(1 for fb in first_success if fb is not None and fb <= B)
        rows.append({
            "budget_B_rank_valid": B,
            "measured_forge_success": succ / T,
            "predicted_forge_success": 1.0 - (1.0 - p) ** B,
        })

    results = {
        "schema": "pvnp-certificate-syndrome-v1-isd",
        "label": label,
        "frozen_regime": frozen,
        "regime": {"n": n, "k": k, "w": w, "tau": tau, "code_seed": code_seed,
                   "T": T, "max_B": max_B, "run_seed": run_seed,
                   "full_enum_C_n_w": math.comb(n, w)},
        "analytic": {"p_per_rank_valid_trial": p, "expected_iters_N": N,
                     "breakpoint_50pct_B": N * math.log(2)},
        "verifier_check_ops_flat": check_ops,
        "isd_rank_audit": {"rank_valid_convention": "B counts rank-valid trials only",
                           "rank_fail_draws_total": rank_fail_total,
                           "measured_attacker_ops_total_incl_rankfail": ops_total},
        "capacity_curve": rows,
        "deterministic": True,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = "isd_smoke.json" if not frozen else "isd_capacity_curve.json"
    (out_dir / fn).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run a THROWAWAY regime to validate the attacker (not the frozen run)")
    ap.add_argument("--out", type=str, default="results/pvnp/certificate-syndrome-v1")
    args = ap.parse_args()

    if args.smoke:
        # THROWAWAY validation regime -- different (n,k,w) and code seed; NOT frozen.
        n, k, w, tau, code_seed, T, max_B, run_seed = 64, 32, 6, 6, 999, 64, 600, 424242
        ladder = [1, 5, 20, 83, 166, 332, 600]
        label, frozen = "THROWAWAY smoke (validate ISD vs analytic; not the frozen run)", False
    else:
        # FROZEN regime -- operator-gated; do NOT run without the go.
        n, k, w, tau, code_seed, T, max_B, run_seed = 128, 64, 12, 12, 2026128, 64, 40000, 7000000
        ladder = [64, 1024, 5004, 16384, 32768]
        label, frozen = "FROZEN [128,64] w=12 capacity run", True

    res = run(n, k, w, tau, code_seed, T, ladder, max_B, run_seed, label, frozen,
              REPO_ROOT / args.out)
    a = res["analytic"]
    print(f"{res['label']}")
    print(f"regime [{n},{k}] w={w} tau={tau}; full-enum C(n,w)={res['regime']['full_enum_C_n_w']:.3e}")
    print(f"analytic: p={a['p_per_rank_valid_trial']:.3e}  N={a['expected_iters_N']:.1f}  "
          f"50%-breakpoint={a['breakpoint_50pct_B']:.1f} rank-valid trials")
    print(f"verifier check ops (flat) = {res['verifier_check_ops_flat']}")
    ra = res["isd_rank_audit"]
    print(f"rank audit: rank_fail_draws={ra['rank_fail_draws_total']}  "
          f"measured_attacker_ops={ra['measured_attacker_ops_total_incl_rankfail']:.3e}")
    print("budget B (rank-valid) -> measured forge | predicted forge:")
    ok = True
    for r in res["capacity_curve"]:
        d = abs(r["measured_forge_success"] - r["predicted_forge_success"])
        flag = "" if d < 0.12 else "  <-- DEVIATION"
        if d >= 0.12: ok = False
        print(f"  B={r['budget_B_rank_valid']:>6} -> measured={r['measured_forge_success']:.3f} "
              f"| predicted={r['predicted_forge_success']:.3f} (|d|={d:.3f}){flag}")
    if not frozen:
        print(f"\nSMOKE VERDICT: ISD {'MATCHES' if ok else 'DEVIATES FROM'} the analytic prediction "
              f"(tol 0.12). {'Validated for the frozen run.' if ok else 'Investigate before the frozen run.'}")


if __name__ == "__main__":
    main()
