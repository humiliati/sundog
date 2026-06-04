#!/usr/bin/env python
"""SUNDOG_V_DECONFOUND Phase 0C - de-confound-stress boundary cell.

Implements docs/deconfound/PHASE0C_DECONFOUND_STRESS_SPEC.md (locked, post-calibration).

Sweeps an injected shared-factor correlation knob over the calibrated rungs, re-runs the
frozen Phase-0B closure read at each rung, and locates where the double-dissociation collapses:
the headline is the STATE-KEEPER's k_func(u), which should be `none` while the de-confound holds
and flip `finite` once u leaks linearly into the input the state body must carry.

Imports the frozen Phase-0B runner (no edit to it); adds only the alpha-injection substrate +
the rung loop + the boundary classifier. Substrate matches deconfound_attack_b_alpha_calibration.py
exactly (same pooling, z-score, fixed g seed, median binarization).
"""
import argparse
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
    strat_split, Model, train_model, body_acts, read_body, bracket, _det, _native,
    S_IDX, OUT_IDX, SPLIT_SEED, D,
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
            "func_acc": round(ua, 3), "state_acc": round(sa, 3)}


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
            rs = read_body(body_acts(sk, b), b, u, seed, 8, args.n_perm)
            rf = read_body(body_acts(fk, b), b, u, seed, 8, args.n_perm)
            unull_hit = (rs["k_null"]["k"] is not None) or (rf["k_null"]["k"] is not None)
            seeds_out.append({
                "seed": seed, "interpreted": gt["func_learned"] and gt["state_learned"],
                "gates": gt, "state_kfunc": rs["k_func"]["k"], "state_kstate": rs["k_state"]["k"],
                "func_kfunc": rf["k_func"]["k"], "func_kstate": rf["k_state"]["k"],
                "func_bracket": bracket(rf, 8)[0], "u_null_hit": bool(unull_hit)})
        interp = [s for s in seeds_out if s["interpreted"]]
        nfin = sum(s["state_kfunc"] is not None for s in interp)
        nnone = sum(s["state_kfunc"] is None for s in interp)
        summ = ("exposes_u" if (interp and nfin >= 2) else
                "hides_u" if (interp and nnone >= 2) else "split")
        label = "HOLD" if det <= 0.10 else ("MARG" if det <= 0.20 else "LEAK")
        per_rung.append({"alpha": alpha, "det": round(det, 4), "label": label,
                         "n_interpreted": len(interp), "state_summary": summ,
                         "state_kfunc_finite": nfin, "state_kfunc_none": nnone, "seeds": seeds_out})
        print(f"[alpha {alpha:.2f}] det={det:+.3f} {label} | state k_func -> {summ} "
              f"(finite {nfin}/{len(interp)}) ({round(time.time()-t0,1)}s)", flush=True)

    # ---- branch (precedence: voids -> boundary classification) ----
    def rung(a): return next((r for r in per_rung if abs(r["alpha"] - a) < 1e-9), None)
    unull = any(s["u_null_hit"] for r in per_rung for s in r["seeds"] if s["interpreted"])
    r0, rdeep = rung(0.0), rung(2.5)
    endpoints_ok = bool(r0 and rdeep and r0["n_interpreted"] >= 2 and rdeep["n_interpreted"] >= 2)
    any_leak = any(r["det"] > 0.20 for r in per_rung)
    hold = [r for r in per_rung if r["det"] <= 0.10]

    if not endpoints_ok:
        verdict = "closure_void_unlearned"
    elif unull:
        verdict = "closure_void_control"
    elif not any_leak:
        verdict = "rungs_missed_boundary"
    elif any(r["state_summary"] == "exposes_u" for r in hold):
        verdict = "closure_confounded_throughout"
    elif all(r["state_summary"] == "hides_u" for r in hold) and rdeep["state_summary"] == "exposes_u":
        verdict = "deconfound_load_bearing_confirmed"
    elif all(r["state_summary"] == "hides_u" for r in hold) and rdeep["state_summary"] == "hides_u":
        verdict = "closure_robust_to_leak"
    else:
        verdict = "boundary_inconclusive"      # safety net (split verdict on a required rung; not in spec's 6)

    summary = {"phase": "attack-b-phase0c-stress", "factor_seed": FACTOR_SEED,
               "n_seeds": len(args.seeds), "n_perm": args.n_perm, "verdict": verdict,
               "boundary_table": [{"alpha": r["alpha"], "det": r["det"], "label": r["label"],
                                   "state_summary": r["state_summary"],
                                   "n_interpreted": r["n_interpreted"]} for r in per_rung],
               "per_rung": per_rung}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_native))
    print("\n==== VERDICT:", verdict, "====")
    print("  alpha | det    | label | state k_func")
    for r in per_rung:
        print(f"  {r['alpha']:.2f}  | {r['det']:+.3f} | {r['label']:<5} | {r['state_summary']} "
              f"(finite {r['state_kfunc_finite']}/{r['n_interpreted']})")
    print(f"  wrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
