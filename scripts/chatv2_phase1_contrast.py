#!/usr/bin/env python
"""Chat-v2 Phase 1 contrast decomposition — shore up the at-the-bar SHARP.

The seed-stability run found the body-resistance robust (d_dec, z1, leak stable
across seeds) but the gen-vs-twin gap clustered at the 0.20 SHARP bar, because the
control-only twin is not a clean null — it *incidentally* builds the non-decision
latents (~0.6). This script replaces the binary gap>=0.20 coin-flip with a
continuous decomposition against a baseline ladder, aggregated across the frozen
seeds:

    chance (0.5)
      + architectural  = untrained backbone (gen's random init) - chance
      + incidental      = control-only twin - untrained         (training-but-not-objective)
      + objective_excess= generative - twin                     (the objective-driven part)
    = generative body_carry

Reuses the saved gen/twin bodies; the only new compute is a cheap untrained-
backbone extraction (no training). Pre-reg: PHASE1_RESIDUAL_BODY_SCALING_SPEC.md.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatv2_phase0_bodyresist as m


def _body_carry(bodies, z, cfg):
    return m.fingerprint(bodies, z, cfg)["body_carry"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob of per-seed manifest.json")
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    dev = torch.device("cpu")
    chance = 0.5
    H = args.H
    rows = []
    for mf in sorted(glob.glob(args.glob)):
        man = json.load(open(mf))
        cfg = m.Cfg(**man["cfg"])
        seed = cfg.seed
        hs = seed + 1000 * H
        bd = os.path.join(os.path.dirname(mf), "bodies")
        g = np.load(os.path.join(bd, f"H{H}_gen.npz"), allow_pickle=True)
        t = np.load(os.path.join(bd, f"H{H}_twin.npz"), allow_pickle=True)
        bc_gen = _body_carry(g["bodies"], g["z"], cfg)
        bc_twin = _body_carry(t["bodies"], t["z"], cfg)
        # untrained backbone = the gen's random-init starting point (same seed)
        torch.manual_seed(hs)
        unt = m.TinyGPT(2, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.max_len).to(dev)
        ub, uz = m.extract_body(unt, H, cfg, dev, hs)
        bc_unt = _body_carry(ub, uz, cfg)
        rows.append({
            "seed": seed,
            "untrained": round(bc_unt, 4), "twin": round(bc_twin, 4), "gen": round(bc_gen, 4),
            "architectural": round(bc_unt - chance, 4),
            "incidental": round(bc_twin - bc_unt, 4),
            "objective_excess": round(bc_gen - bc_twin, 4),
            "objective_fraction": round((bc_gen - bc_twin) / (bc_gen - chance + 1e-9), 4),
        })

    keys = ["untrained", "twin", "gen", "architectural", "incidental",
            "objective_excess", "objective_fraction"]
    summary = {k: {"mean": round(float(np.mean([r[k] for r in rows])), 4),
                   "std": round(float(np.std([r[k] for r in rows])), 4)} for k in keys}

    print(f"H={H}  seeds={len(rows)}  (ladder: chance {chance} -> untrained -> twin -> gen)")
    for r in rows:
        print(f"  seed {r['seed']:>2}: untrained={r['untrained']:.2f} twin={r['twin']:.2f} "
              f"gen={r['gen']:.2f} | arch={r['architectural']:.2f} incid={r['incidental']:.2f} "
              f"OBJ_EXCESS={r['objective_excess']:.2f} obj_frac={r['objective_fraction']:.2f}")
    print("aggregate (mean +/- std):")
    for k in keys:
        print(f"  {k:18} {summary[k]['mean']:.3f} +/- {summary[k]['std']:.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"H": H, "chance": chance, "rows": rows, "summary": summary},
                  open(args.out, "w"), indent=2)
        print("->", args.out)


if __name__ == "__main__":
    main()
