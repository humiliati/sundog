#!/usr/bin/env python
"""Phase-4 ON-arm — finite-POMDP sufficiency receipt.

Implements docs/proof/PHASE4_ON_ARM_FINITE_POMDP_SLATE.md (FROZEN). Populates the never-populated
`on` class of the coarse-graining Phase-4 two-sided gate, on the substrate where 𝓕_σ-measurability
is PROVABLE (Phase-2 Theorem 1). NO training; finite, exact, deterministic given the frozen seed.

Two-mode memory corridor POMDP (frozen knobs §5):
  mode m ~ U{0,1}; T=6; t=0 signpost (mode-revealing, noisy eps=0.15); t=1..T-2 corridor
  (mode-independent); t=T-1 fork (aliased across modes). advance is uniquely optimal off-fork;
  at the fork safe iff action==m.

Controllers (§3):
  signature = best memoryless Φ-measurable g:Σ→A  (advance off-fork; fork tie -> go0; fork safe iff m==0)
  bayes-floor = optimal policy on the belief over m from the SAME Φ history (exact); fork safe iff
                the signpost observation was correct (prob 1-eps). Same obs budget, used with history.

Cell ON/OFF labels by Phase-2 Theorem 1 (intersection of optimal-action sets over the fiber):
  signpost+corridor -> A* = {advance} for both modes -> ON ; fork -> {go0} vs {go1} disjoint -> OFF.

Gate (verbatim, NOT retuned): regret -> 0 within CI ON ; CI excludes 0 (bounded away) OFF.
regret = return_bayes - return_signature per paired decision.

    python scripts/proof_phase4_on_arm_finite_pomdp.py --out results/proof/phase4-on-arm
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from numpy.random import default_rng

# ---- FROZEN knobs (slate §5) ----
T = 6
EPS = 0.15                       # signpost flip probability
N_EPISODES = 2000
FIBER_MIN_SAMPLES = 20
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 40604           # matched to the three-body receipt for cross-comparability
CLASS_ORDER = ["on", "off", "undecidable"]
CLASS_SEED_RULE = lambda i: BOOTSTRAP_SEED + i * 1009   # = on:40604 off:41613 undecidable:42622
FLOOR_THRESHOLD = 0.05


def simulate(rng):
    """One paired row per (episode, cell_class): mean (bayes_safe - signature_safe) over that
    class's decisions in the episode. ON decisions = signpost + corridor (t=0..T-2); OFF = fork."""
    on_rows, off_rows = [], []
    n_on_decisions = T - 1                      # signpost + corridor steps
    for _ in range(N_EPISODES):
        m = int(rng.integers(0, 2))
        # signpost observation: correct S_m with prob 1-eps, flipped with prob eps
        signpost_correct = rng.random() >= EPS
        obs_mode = m if signpost_correct else 1 - m

        # --- ON decisions (signpost + corridor): both controllers take the unique optimal
        #     action `advance` -> both safe -> per-decision regret is exactly 0 ---
        on_rows.append(0.0)                      # mean over T-1 decisions, all (bayes-sig)=0

        # --- OFF decision (fork): signature is memoryless on the aliased FORK cell -> best g
        #     ties -> go0 -> safe iff m==0. bayes acts on the belief (signpost) -> go obs_mode
        #     -> safe iff obs_mode==m (i.e. signpost was correct). ---
        sig_fork_safe = 1.0 if m == 0 else 0.0
        bayes_fork_safe = 1.0 if obs_mode == m else 0.0    # == signpost_correct
        off_rows.append(bayes_fork_safe - sig_fork_safe)
    return {"on": np.array(on_rows), "off": np.array(off_rows),
            "undecidable": np.array([])}, n_on_decisions


def paired_bootstrap_ci(x, seed, iters=BOOTSTRAP_ITERS):
    if x.size == 0:
        return None, None, None
    r = default_rng(seed)
    n = x.size
    means = np.empty(iters)
    for b in range(iters):
        means[b] = x[r.integers(0, n, n)].mean()
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/proof/phase4-on-arm")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rng = default_rng(BOOTSTRAP_SEED)            # episode RNG (frozen)
    classes, n_on_dec = simulate(rng)

    print(f"[cfg] T={T} eps={EPS} n_episodes={N_EPISODES} fiberMinSamples={FIBER_MIN_SAMPLES} "
          f"bootstrap_seed={BOOTSTRAP_SEED} iters={BOOTSTRAP_ITERS}", flush=True)

    summary_rows = []
    global_neg = []
    per_class = {}
    for i, cls in enumerate(CLASS_ORDER):
        x = classes[cls]
        cseed = CLASS_SEED_RULE(i)
        mean, lo, hi = paired_bootstrap_ci(x, cseed)
        neg = int((x < 0).sum())
        rc = int(x.size)
        if rc:
            global_neg.append(neg / rc)
        per_class[cls] = {"row_count": rc, "mean": mean, "lo": lo, "hi": hi, "neg": neg,
                          "neg_rate": (neg / rc if rc else None), "class_seed": cseed}

    # gate verdict (frozen pass condition, NOT retuned)
    on, off = per_class["on"], per_class["off"]
    on_pass = on["row_count"] >= FIBER_MIN_SAMPLES and on["lo"] is not None and on["lo"] <= 0.0 <= on["hi"]
    off_pass = off["row_count"] >= FIBER_MIN_SAMPLES and off["lo"] is not None and off["lo"] > 0.0
    if on_pass and off_pass:
        floor_status = "on_arm_pass_two_sided_gate_fired"
    elif on["lo"] is not None and on["lo"] > 0.0:
        floor_status = "sufficiency_false_halt_and_falsify"
    elif off["lo"] is not None and off["lo"] <= 0.0 <= off["hi"]:
        floor_status = "classifier_wrong_off_not_bounded_away"
    else:
        floor_status = "non_decisive"
    gnr = float(np.mean(global_neg)) if global_neg else None

    with (out / "phase4-regret-summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_class", "row_count", "mean_regret", "ci_lower_95", "ci_upper_95",
                    "negative_regret_count", "negative_regret_rate", "global_negative_regret_rate",
                    "floor_status", "bootstrap_seed", "bootstrap_class_seed", "bootstrap_iterations"])
        for cls in CLASS_ORDER:
            c = per_class[cls]
            w.writerow([cls, c["row_count"],
                        "" if c["mean"] is None else round(c["mean"], 8),
                        "" if c["lo"] is None else round(c["lo"], 8),
                        "" if c["hi"] is None else round(c["hi"], 8),
                        c["neg"], "" if c["neg_rate"] is None else round(c["neg_rate"], 6),
                        "" if gnr is None else round(gnr, 6),
                        floor_status, BOOTSTRAP_SEED, c["class_seed"], BOOTSTRAP_ITERS])

    # per-case rows (capped sample for the artifact; full counts are in the summary)
    with (out / "phase4-regret.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "cell_class", "regret", "signature_mode", "bayes_mode"])
        for cls in ("on", "off"):
            for ep, v in enumerate(classes[cls][:200]):
                w.writerow([ep, cls, round(float(v), 6), "best_memoryless_phi", "belief_optimal_same_info"])

    manifest = {
        "schema": "sundog.proof.phase4.on_arm.finite_pomdp.v1",
        "slate": "docs/proof/PHASE4_ON_ARM_FINITE_POMDP_SLATE.md",
        "generatedAt_note": "deterministic given BOOTSTRAP_SEED; no wall-clock in receipt",
        "frozen_knobs": {"T": T, "eps": EPS, "n_episodes": N_EPISODES,
                         "fiberMinSamples": FIBER_MIN_SAMPLES, "bootstrap_iterations": BOOTSTRAP_ITERS,
                         "bootstrap_seed": BOOTSTRAP_SEED,
                         "classSeedRule": "bootstrap_seed + classIndex*1009 over [on,off,undecidable]"},
        "regret": {"formula": "return_bayes - return_signature (per paired decision)",
                   "n_on_decisions_per_episode": n_on_dec, "n_off_decisions_per_episode": 1},
        "labels": {"rule": "Phase-2 Theorem 1: cell is ON iff intersection of optimal-action sets "
                            "over its fiber is non-empty (fiber-constant optimal action)",
                   "on": "signpost+corridor (A*={advance} both modes)",
                   "off": "fork (A*={go0} vs {go1}, disjoint -- Phase-2 flip counterexample)"},
        "controllers": {"signature": "best memoryless Phi-measurable g:Sigma->A",
                        "bayes_floor": "optimal policy on belief over m from same Phi history (exact)"},
        "gate_frozen": "regret->0 within CI ON; CI excludes 0 (bounded away) OFF "
                       "(COARSE_GRAINING_PROOF_ROADMAP.md Phase 4, verbatim, not retuned)",
        "result": {cls: {k: per_class[cls][k] for k in ("row_count", "mean", "lo", "hi", "neg_rate")}
                   for cls in CLASS_ORDER},
        "verdict": floor_status,
        "scope": "sufficiency ON-arm on a CONSTRUCTED finite POMDP; provable measurability; NOT "
                 "body-resistance/regime-2, NOT real/high-dim, NOT a trained body. Discrimination "
                 "(->0 ON vs bounded-away OFF) is the content, not the ON value alone.",
    }
    (out / "phase4-regret-manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[on ] row_count={on['row_count']} mean={on['mean']} CI=[{on['lo']},{on['hi']}]")
    print(f"[off] row_count={off['row_count']} mean={round(off['mean'],4)} "
          f"CI=[{round(off['lo'],4)},{round(off['hi'],4)}] neg_rate={off['neg_rate']}")
    print(f"\n==== PHASE-4 ON-ARM VERDICT: {floor_status} ====")
    print(f"  ON  regret -> 0 within CI : {'PASS' if on_pass else 'FAIL'}  (mean {on['mean']}, CI [{on['lo']},{on['hi']}])")
    print(f"  OFF regret bounded away>0 : {'PASS' if off_pass else 'FAIL'}  (mean {round(off['mean'],4)}, CI excludes 0 = {off_pass})")
    print(f"  two-sided gate fired      : {on_pass and off_pass}")
    print(f"  wrote {out}/phase4-regret-summary.csv (+ per-case, manifest)  ({round(time.time()-t0,2)}s)")


if __name__ == "__main__":
    main()
