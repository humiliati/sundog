#!/usr/bin/env python
"""Chat-v2 Phase 1 seed-stability adjudicator (Gate E).

Reads per-seed phase0.2 manifests, maps each seed's record at the primary H to the
Phase 1 branch taxonomy, and applies Gate E (>= ceil(2N/3) seeds in the same branch).
Reuses the harness's own per-H status (which applies Gate C / the UNLEARNED guard),
then splits SHARP by the d_dec >= bar high-dimensional gate.

DRAFT adjudication helper, not a verdict freeze. Pre-reg:
docs/chatv2/PHASE1_RESIDUAL_BODY_SCALING_SPEC.md (Gates C/D/E, branch taxonomy).
"""
import argparse
import glob
import json
import math
import os
from collections import Counter


def classify(rec, bar):
    """Map a per-H record (with harness status) to a Phase 1 branch."""
    st = rec.get("status")
    if st == "UNLEARNED":
        return "phase1_learnability_block"
    if st == "MARGINAL":
        return "phase1_highH_marginal"
    if st == "SHARP":
        return "phase1_highdim_sharp" if rec["generative"]["d_dec"] >= bar else \
               "phase1_scaling_sharp_below_bar"
    return "phase1_void_runner_or_receipt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob of per-seed manifest.json files")
    ap.add_argument("--H", type=int, default=8, help="primary H to adjudicate")
    ap.add_argument("--bar", type=int, default=20, help="Gate D high-dim d_dec bar")
    ap.add_argument("--out", default=None, help="write branch_adjudication.json here")
    a = ap.parse_args()

    rows = []
    for mf in sorted(glob.glob(a.glob)):
        man = json.load(open(mf))
        rec = next((r for r in man.get("records", []) if r["H"] == a.H), None)
        if rec is None:
            continue
        g = rec["generative"]
        rows.append({
            "seed": man["cfg"]["seed"], "manifest": mf, "branch": classify(rec, a.bar),
            "status": rec.get("status"), "d_dec": g["d_dec"], "z1_acc": g["z1_acc"],
            "leak": g["cross_latent_leak"], "body_carry": g["body_carry"],
            "twin_body_carry": (rec["twin"] or {}).get("body_carry"),
            "eval_loss": round(rec["gen_train"]["eval_loss"], 4),
            "fair_readout": g.get("fair_readout"),
        })

    n = len(rows)
    cnt = Counter(r["branch"] for r in rows)
    need = math.ceil(2 * n / 3) if n else 0          # Gate E: >= 2/3 same branch
    top, topn = (cnt.most_common(1)[0] if cnt else ("none", 0))
    gateE = bool(n and topn >= need)
    verdict = top if gateE else "phase1_seed_unstable"

    print(f"H={a.H}  seeds={n}  branch_counts={dict(cnt)}  Gate-E need>={need} "
          f"-> {topn}x {top}  ==>  VERDICT: {verdict}")
    for r in rows:
        print(f"  seed {r['seed']:>5}  {r['branch']:<32} status={r['status']:<9} "
              f"d_dec={r['d_dec']:.1f} z1={r['z1_acc']:.2f} leak={r['leak']:.2f} "
              f"body_carry={r['body_carry']:.2f}(twin {r['twin_body_carry']}) "
              f"eval_loss={r['eval_loss']} fair={r['fair_readout']}")

    if a.out:
        out = {"H": a.H, "bar": a.bar, "n_seeds": n, "branch_counts": dict(cnt),
               "gateE_need": need, "gateE_pass": gateE, "verdict": verdict, "rows": rows}
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
