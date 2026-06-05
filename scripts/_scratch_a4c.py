"""A4 part C: readout-route degradation, nan dropping in medians, _cv_acc folds, json/CSV."""
import numpy as np
from numpy.random import default_rng

from jepa_0d_accumulator_preflight import (
    AccCfg, gen_accumulator, parity_encode, parity_decode, int_to_bits, PSI,
    raw_linear_probes, oracle_recover, base_rates, _cv_acc, _det, safe_median,
)

print("=" * 70)
print("TEST 9: WHY does readout-route per-ckpt degrade 0.95 -> 0.92 -> 0.90?")
print("=" * 70)
# Two candidate readouts (rows 0 and 7) are IDENTICAL ([0,0,1]). So with n_U=8 there
# are only 7 *distinct* PSI rows; the readout is rank-3 (good) but two channels are
# redundant. The degradation likely tracks the count distribution shifting to higher u
# at later ckpts where some clean codewords collide under 10% bit noise + nearest-cw.
cfg = AccCfg(n=2000, seed=0)
data = gen_accumulator(cfg, default_rng(0))
b = cfg.count_bits
Psi = data["Psi"]
all_codes = int_to_bits(np.arange(cfg.K + 1), b)
code_clean = (all_codes @ Psi.T) % 2
# Pairwise Hamming distances between clean codewords (min margin => error-prone under noise)
D = (code_clean[:, None, :] != code_clean[None, :, :]).sum(2)
np.fill_diagonal(D, 99)
print("min pairwise Hamming distance between codewords (0..K):", D.min())
print("per-value nearest-neighbor distance:", D.min(1))
# decode error is dominated by values with min distance; report per-ckpt u distribution
u = data["u"]
for c in data["ckpts"]:
    vals, cnts = np.unique(u[:, c-1], return_counts=True)
    print(f"  ckpt {c}: u dist {dict(zip(vals.tolist(), cnts.tolist()))}")
print("NOTE: row0==row7 ([0,0,1]) -> 8 channels but only 7 distinct PSI rows (redundant readout).")

print()
print("=" * 70)
print("TEST 10: can _cv_acc silently return nan and get dropped, masking a leak?")
print("=" * 70)
# safe_median drops nans. raw_linear u_det_max uses np.max over non-None u_det.
# Construct a target where some ticks have <2 classes or min-class < 2 folds.
# At tick 1 u in {0,1}; later ticks have rare high classes. With folds=min(4,min_count)
# if min_count==1, folds<2 -> returns (nan, maj). Then u_det None -> dropped from max.
# Does any u-per-tick or emission cell go nan at n=2000/3000? Check directly.
raw = raw_linear_probes(data, cfg)
nan_u = [r["tick"] for r in raw["u_per_tick"] if r["u_det"] is None]
nan_e = [r["tick"] for r in raw["e_per_tick"] if r["e_det"] is None]
print("u-per-tick ticks dropped (nan/None):", nan_u)
print("e-per-tick ticks dropped (nan/None):", nan_e)
print("u_det_max (over surviving):", raw["u_det_max"], " median:", raw["u_det_median"])
print("emission clean/obs median:", raw["emission"]["clean_det_median"], raw["emission"]["obs_det_median"])
print("bitcount_uT_det:", raw["bitcount_uT_det"])

# Stress: force a starved high class to see if a nan would hide a leaking cell.
# Lower n hugely so rare classes have count 1 -> folds<2 -> nan dropped.
print()
print("  -- stress: tiny n=60 to force starvation/nan dropping --")
cfg_s = AccCfg(n=60, seed=3)
data_s = gen_accumulator(cfg_s, default_rng(3))
raw_s = raw_linear_probes(data_s, cfg_s)
nan_us = [r["tick"] for r in raw_s["u_per_tick"] if r["u_det"] is None]
print("  ticks dropped at n=60:", nan_us, " u_det_max:", raw_s["u_det_max"])
# If u_det_max computed only over surviving ticks, a leaking-but-starved tick is invisible.

print()
print("=" * 70)
print("TEST 11: does the per-position MAJORITY baseline fairly strip the count prior?")
print("=" * 70)
# u_det = (acc - maj)/(1-maj). maj here is the GLOBAL majority of u[:,t] (counts.max/sum),
# NOT a position-conditioned majority beyond the fact that t is fixed. The doc claims a
# 'per-position majority baseline'. Since each tick t is probed separately, maj IS the
# per-position majority. Confirm a position-only predictor scores ~0 (gets no count credit).
# Build a degenerate X = just the tick-constant (no info) and check u_det ~ 0.
X_const = np.ones((cfg.n, 2), dtype=np.float64)
X_const[:, 1] = 0.0
for t in [0, 5, 11]:
    acc, maj = _cv_acc(X_const, data["u"][:, t], multiclass=True)
    print(f"  tick {t+1}: const-X acc={round(acc,4)} maj={round(maj,4)} u_det={round(_det(acc,maj),4)}")

print()
print("=" * 70)
print("TEST 12: json_clean / NaN — does a nan gate value flip a FAIL into PASS?")
print("=" * 70)
# g1 requires emission medians <= 0.10 AND bitcount None-or-<=0.10. If safe_median over
# an all-nan list returns nan, the comparison `nan <= 0.10` is False -> g1 False (safe).
print("safe_median([nan,nan]) =", safe_median([float('nan'), float('nan')]))
print("nan <= 0.10 evaluates to:", float('nan') <= 0.10, "(False => gate would FAIL, not pass)")
print("BUT: clean_det_median is round(safe_median(...),4); round(nan)=", round(float('nan'),4))
# round(nan) is nan; comparison still False. Safe. Confirm g1 logic with a synthetic nan.
