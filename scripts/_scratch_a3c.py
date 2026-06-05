"""ROLE A3 scratch part C:
1. nan-drop fragility: how often does a checkpoint tick det vanish to nan across seeds/n?
   This matters because the gate uses u_det_MAX over surviving ticks; a dropped tick is
   invisible. Quantify P(any tick nan) and which ticks at n in {1500,2000,3000}.
2. Median-over-ticks masking: the committed metric reports median & max over ALL 12 ticks,
   but the BODY only reads u at CHECKPOINTS {4,8,12}. Recompute the de-confound stat
   restricted to checkpoint ticks (the ones the spec will actually probe). Does restricting
   to checkpoints change the de-confound picture?
3. Downstream interpretability: at gate 5 the discard read trains body->z on ALL rows and
   scores on held-out FLIPS only. flip_min=250 -> but the score subset is the flips at ONE
   (ckpt,channel). With p_noise=0.10 the flip rate is ~10%, score n ~ heldout flips. Compute
   the binomial standard error of a per-cell flip-read accuracy at that support and ask if a
   GEN-vs-JEPA gap is resolvable.
"""
import numpy as np
from numpy.random import default_rng

from jepa_0d_accumulator_preflight import AccCfg, gen_accumulator, _cv_acc, _det, safe_median

EPS = 1e-9


def per_tick_dets(X, u, T):
    dets = []
    nans = []
    for t in range(T):
        acc, maj = _cv_acc(X, u[:, t], multiclass=True)
        d = _det(acc, maj)
        dets.append(d)
        if np.isnan(d):
            nans.append(t + 1)
    return dets, nans


def main():
    print("=== 1+2. nan-drop & checkpoint-restricted de-confound across n and seeds ===")
    CKPTS = [4, 8, 12]
    for n in [1500, 2000, 3000]:
        for seed in [0, 1, 2]:
            cfg = AccCfg(); cfg.n = n; cfg.seed = seed
            rng = default_rng(seed)
            data = gen_accumulator(cfg, rng)
            X = data["X"].astype(np.float64)
            u = data["u"]
            dets, nans = per_tick_dets(X, u, cfg.T)
            all_med = safe_median(dets)
            all_max = max(d for d in dets if not np.isnan(d))
            ck_dets = [dets[c-1] for c in CKPTS]
            ck_med = safe_median(ck_dets)
            ck_max = max((d for d in ck_dets if not np.isnan(d)), default=float('nan'))
            ck_nan = [c for c in CKPTS if np.isnan(dets[c-1])]
            print(f"n={n} seed={seed}: all_med={all_med:+.4f} all_max={all_max:+.4f} "
                  f"nan_ticks={nans}  CKPT_med={ck_med:+.4f} CKPT_max={ck_max:+.4f} ckpt_nan={ck_nan}")

    print("\n=== 3. flip-read support / binomial SE at gate-5 score subset ===")
    cfg = AccCfg(); cfg.n = 3000
    # flip_min from receipt = 250 flips per (ckpt,channel) over the full n.
    # gate-5 trains on all rows, SCORES on held-out flips. With a typical 80/20 or 75/25 CV
    # split the held-out flips per cell ~ 0.2..0.25 * 250 = 50..63.
    for held_frac in [0.2, 0.25, 0.5]:
        n_flip_held = 250 * held_frac
        # accuracy SE at p=0.5 (worst case) and p=0.8 (a plausible GEN read)
        se50 = np.sqrt(0.5*0.5/n_flip_held)
        se80 = np.sqrt(0.8*0.2/n_flip_held)
        print(f"held_frac={held_frac}: n_flip_held~{n_flip_held:.0f}  "
              f"SE@p=.5={se50:.3f}  SE@p=.8={se80:.3f}  "
              f"95%CI half-width@.8={1.96*se80:.3f}")
    # how big a GEN-vs-JEPA gap could this resolve? need gap > ~2*(SE_gen+SE_jepa)
    print("\nIf gate-5 pools flips across all 8 channels at a checkpoint instead of per-cell, "
          "support multiplies by 8 -> ~2000 flips -> SE shrinks ~2.8x. Check whether the spec "
          "pools or scores per-cell; flip_min is reported PER CELL.")


if __name__ == "__main__":
    main()
