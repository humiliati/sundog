#!/usr/bin/env python
"""SUNDOG_V_DECONFOUND Phase 0C - de-confound-stress boundary cell.

Implements docs/deconfound/PHASE0C_DECONFOUND_STRESS_SPEC.md (locked, post-calibration).

Sweeps an injected shared-factor correlation knob over the calibrated rungs, re-runs the
frozen Phase-0B closure read at each rung, and asks whether the double-dissociation degrades:
the headline is continuous state_det_u, the STATE-KEEPER's max selection-corrected-significant
held-out det(u) over k, baseline-subtracted from the HOLD rung. The inherited binary
k_func>=0.70 read is retained only as the HOLD retro-flag.

Imports the frozen Phase-0B runner (no edit to it); adds only the alpha-injection substrate +
the rung loop + the boundary classifier. Substrate matches deconfound_attack_b_alpha_calibration.py
exactly (same pooling, z-score, fixed g seed, median binarization).
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from numpy.random import default_rng
import torch
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from deconfound_attack_b_phase0_closure import (          # frozen 0B machinery (imported, not edited)
    strat_split, Model, train_model, body_acts, read_body, bracket, _det, _native, _design,
    S_IDX, OUT_IDX, SPLIT_SEED, D, LAM, ALPHA,
)

FACTOR_SEED = 20260604
RUNGS = [0.0, 0.75, 1.0, 2.5]                              # calibrated (receipt 2026-06-04)


def pooled_feats():
    x = load_digits().images.astype(np.float64)
    feats = np.zeros((x.shape[0], D)); f = 0
    for br in range(4):
        for bc in range(2):
            feats[:, f] = x[:, br * 2:(br + 1) * 2, bc * 4:(bc + 1) * 4].mean((1, 2)); f += 1
    return feats


def substrate(alpha, g):
    feats = pooled_feats()
    z = (feats - feats.mean(0)) / feats.std(0)
    y = z + alpha * g
    b = (y > np.median(y, 0)).astype(np.float32)
    u = (b[:, S_IDX].astype(int).sum(1) & 1).astype(np.float32)
    return b, u


def deconf_det(b, u):
    base = float(max(u.mean(), 1 - u.mean()))
    acc = float(cross_val_score(LogisticRegression(max_iter=500), b, u, cv=5).mean())
    return (acc - base) / (1 - base + 1e-9)


def learned_gates(sk, fk, b, te, u):
    with torch.no_grad():
        pu = torch.sigmoid(fk(torch.tensor(b[te]))[0]).numpy().ravel() > 0.5
        ps = torch.sigmoid(sk(torch.tensor(b[te]))[0]).numpy() > 0.5
    ua = float((pu == u[te]).mean()); ub = max(u[te].mean(), 1 - u[te].mean())
    sa = float(np.mean([(ps[:, j] == b[te, j]).mean() for j in range(D)]))
    sd = float(np.mean([_det((ps[:, j] == b[te, j]).mean(),
                             max(b[te, j].mean(), 1 - b[te, j].mean())) for j in range(D)]))
    return {"func_learned": bool(_det(ua, ub) >= 0.70 and ua >= 0.80),
            "state_learned": bool(sd >= 0.70 and sa >= 0.80),
            "func_acc": round(ua, 3), "state_acc": round(sa, 3),
            "func_det": round(_det(ua, ub), 3), "state_det": round(sd, 3)}


def state_det_u(H, u, seed, n_perm):
    """Max selection-corrected-significant det(u) over k for the state-keeper body.

    This mirrors the Phase-0B subset-selection procedure but removes the inherited
    det>=0.70 crossing bar. For each k, the subset is selected on probe-train accuracy,
    scored on probe-heldout det, and tested against a label-permutation null that repeats
    the same subset-selection step. The headline is max significant det, or 0 if no k is
    significant.
    """
    n, d = H.shape
    tr, he = strat_split(u, [0.6, 0.4], SPLIT_SEED + 911 + seed)
    y = u.astype(np.float32)
    base_he = float(max(y[he].mean(), 1 - y[he].mean()))
    rng = default_rng(SPLIT_SEED + 1701 + seed)
    perms = np.stack([rng.permutation(n) for _ in range(n_perm)])
    best_sig = {"det": 0.0, "k": None, "p": None}
    rows = []

    for k in range(1, d + 1):
        subsets = list(itertools.combinations(range(d), k))
        observed = {"train_acc": -1.0, "det": -1.0}
        for subset in subsets:
            xtr, xhe = _design(H[tr], subset), _design(H[he], subset)
            solve = np.linalg.solve(xtr.T @ xtr + LAM * np.eye(xtr.shape[1]), xtr.T)
            w = solve @ (2 * y[tr] - 1)
            train_acc = float(((xtr @ w > 0) == y[tr]).mean())
            if train_acc > observed["train_acc"]:
                held_acc = float(((xhe @ w > 0) == y[he]).mean())
                observed = {"train_acc": train_acc, "det": _det(held_acc, base_he)}

        null_train = np.empty((len(subsets), n_perm))
        null_det = np.empty((len(subsets), n_perm))
        for si, subset in enumerate(subsets):
            xtr, xhe = _design(H[tr], subset), _design(H[he], subset)
            solve = np.linalg.solve(xtr.T @ xtr + LAM * np.eye(xtr.shape[1]), xtr.T)
            yp = y[perms]
            w = solve @ (2 * yp[:, tr] - 1).T
            ptr = xtr @ w > 0
            phe = xhe @ w > 0
            null_train[si] = (ptr == yp[:, tr].T).mean(axis=0)
            held_acc = (phe == yp[:, he].T).mean(axis=0)
            base = np.maximum(yp[:, he].mean(axis=1), 1 - yp[:, he].mean(axis=1))
            null_det[si] = (held_acc - base) / np.maximum(1 - base, 1e-9)
        pick = null_train.argmax(axis=0)
        selected_null_det = null_det[pick, np.arange(n_perm)]
        p = (1 + int((selected_null_det >= observed["det"]).sum())) / (n_perm + 1)
        row = {"k": int(k), "det": round(float(observed["det"]), 4), "p": round(float(p), 4)}
        rows.append(row)
        if p <= ALPHA and observed["det"] > best_sig["det"]:
            best_sig = row

    return {
        "det": round(float(best_sig["det"]), 4),
        "k": best_sig["k"],
        "p": best_sig["p"],
        "per_k": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/deconfound/attack-b-phase0c-stress")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--rungs", type=float, nargs="+", default=RUNGS)
    ap.add_argument("--smoke", action="store_true", help="1 seed, 2 rungs, 200 perms (timing)")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.n_perm, args.rungs = [0], 200, [0.0, 2.5]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    nimg = len(load_digits().images)
    g = default_rng(FACTOR_SEED).standard_normal((nimg, 1))
    feats = pooled_feats()
    native = (feats > np.median(feats, 0)).astype(np.float32)
    assert np.array_equal(substrate(0.0, g)[0], native), "alpha=0 must reproduce the 0B substrate"

    per_rung = []
    t0 = time.time()
    for alpha in args.rungs:
        b, u = substrate(alpha, g)
        det = deconf_det(b, u)
        tr, va, te = strat_split(u, [0.6, 0.2, 0.2], SPLIT_SEED)
        seeds_out = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            init = Model(8, 1).body.state_dict()                  # matched init for the rd8 pair
            sk = train_model("state", 8, b, u, tr, va, seed, init_state=init)
            fk = train_model("func", 8, b, u, tr, va, seed, init_state=init)
            gt = learned_gates(sk, fk, b, te, u)
            Hs = body_acts(sk, b)
            rs = read_body(Hs, b, u, seed, 8, args.n_perm)
            sdu = state_det_u(Hs, u, seed, args.n_perm)
            rf = read_body(body_acts(fk, b), b, u, seed, 8, args.n_perm)
            unull_hit = (rs["k_null"]["k"] is not None) or (rf["k_null"]["k"] is not None)
            seeds_out.append({
                "seed": seed, "interpreted": gt["func_learned"] and gt["state_learned"],
                "gates": gt, "state_det_u": sdu, "state_kfunc_context": rs["k_func"]["k"],
                "state_kstate": rs["k_state"]["k"],
                "func_kfunc": rf["k_func"]["k"], "func_kstate": rf["k_state"]["k"],
                "func_bracket": bracket(rf, 8)[0], "u_null_hit": bool(unull_hit)})
        interp = [s for s in seeds_out if s["interpreted"]]
        state_det_vals = [s["state_det_u"]["det"] for s in interp]
        med_state_det_u = float(np.median(state_det_vals)) if state_det_vals else 0.0
        label = "HOLD" if det <= 0.10 else ("MARG" if det <= 0.20 else "LEAK")
        per_rung.append({"alpha": alpha, "det": round(det, 4), "label": label,
                         "n_interpreted": len(interp),
                         "state_det_u_median": round(med_state_det_u, 4),
                         "seeds": seeds_out})
        print(f"[alpha {alpha:.2f}] input_det={det:+.3f} {label} | "
              f"state_det_u={med_state_det_u:.3f} ({len(interp)} interpreted) "
              f"({round(time.time()-t0,1)}s)", flush=True)

    # ---- branch (precedence: voids -> boundary classification) ----
    def rung(a): return next((r for r in per_rung if abs(r["alpha"] - a) < 1e-9), None)
    unull = any(s["u_null_hit"] for r in per_rung for s in r["seeds"] if s["interpreted"])
    r0, rdeep = rung(0.0), rung(2.5)
    endpoints_ok = bool(r0 and rdeep and r0["n_interpreted"] >= 2 and rdeep["n_interpreted"] >= 2)
    any_leak = any(r["det"] > 0.20 for r in per_rung)
    s0 = r0["state_det_u_median"] if r0 else 0.0
    sdeep = rdeep["state_det_u_median"] if rdeep else 0.0
    rise = round(float(sdeep - s0), 4)
    hold_state_kfunc_hits = 0
    if r0:
        hold_state_kfunc_hits = sum(
            1 for s in r0["seeds"]
            if s["interpreted"] and s["state_kfunc_context"] is not None
        )
    hold_state_kfunc_majority = bool(r0 and hold_state_kfunc_hits >= 2)

    if not endpoints_ok:
        verdict = "closure_void_unlearned"
    elif unull:
        verdict = "closure_void_control"
    elif not any_leak:
        verdict = "rungs_missed_boundary"
    elif hold_state_kfunc_majority:
        verdict = "closure_confounded_throughout"
    elif rise >= 0.15:
        verdict = "deconfound_load_bearing_confirmed"
    else:
        verdict = "closure_robust_to_leak"

    summary = {"phase": "attack-b-phase0c-stress", "factor_seed": FACTOR_SEED,
               "n_seeds": len(args.seeds), "n_perm": args.n_perm, "verdict": verdict,
               "headline": {"s0": s0, "sdeep": sdeep, "rise": rise,
                            "hold_state_kfunc_hits": hold_state_kfunc_hits,
                            "hold_state_kfunc_majority": hold_state_kfunc_majority},
               "boundary_table": [{"alpha": r["alpha"], "det": r["det"], "label": r["label"],
                                   "state_det_u_median": r["state_det_u_median"],
                                   "n_interpreted": r["n_interpreted"]} for r in per_rung],
               "per_rung": per_rung}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_native))
    print("\n==== VERDICT:", verdict, "====")
    print(f"  s0={s0:.3f}  sdeep={sdeep:.3f}  rise={rise:.3f}")
    print(f"  HOLD state_kfunc hits={hold_state_kfunc_hits} majority={hold_state_kfunc_majority}")
    print("  alpha | input_det | label | state_det_u")
    for r in per_rung:
        print(f"  {r['alpha']:.2f}  | {r['det']:+.3f}    | {r['label']:<5} | "
              f"{r['state_det_u_median']:.3f} ({r['n_interpreted']} interpreted)")
    print(f"  wrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
