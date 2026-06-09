#!/usr/bin/env python
"""Frozen test for H3 v2 (scripts/shadow_pooled_synthetic_v2.py) — the CORRECTED imported-wall result.
Asserts the reproducible, probe-robust finding: once raw (linear) averaging has washed the continuous c,
a body trained to KEEP c (reg_c) RECOVERS it post-pool (demodulate-then-pool) while the same architecture
trained to classify d (clf_d) SUPPRESSES c to ~0 -> the Shadow-Invertibility continuous-resist is
objective-dependent and fragile to a trained nonlinear encoder. The discrete d is determined throughout.
Retrains the bodies (~1-2 min CPU, single-thread deterministic). Run: python scripts/test_shadow_pooled_synthetic_v2.py
"""
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import shadow_pooled_synthetic_v2 as v2          # noqa: E402
import torch                                      # noqa: E402
from sklearn.linear_model import Ridge            # noqa: E402
from sklearn.neural_network import MLPRegressor   # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import KFold, cross_val_score  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


def probe(X, y, strong=False):
    Xs = StandardScaler().fit_transform(X)
    kf = KFold(5, shuffle=True, random_state=0)
    est = (MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=600, random_state=0) if strong
           else Ridge(alpha=1.0))
    return float(max(0.0, cross_val_score(est, Xs, y, cv=kf, scoring="r2").mean()))


print("H3 v2 — objective-dependent defeat of the Shadow-Invertibility continuous-resist:\n")

# train clf_d + reg_c once (deterministic: torch single-thread + fixed per-objective seeds)
u_tr, c_tr, d_tr = v2.gen(v2.N_TRAIN, v2.TRAIN_LAM, v2.SEED + 1)
clf, clf_fit = v2.train_body("clf_d", u_tr, c_tr, d_tr)
reg, reg_fit = v2.train_body("reg_c", u_tr, c_tr, d_tr)

check("reg_c learned to demodulate c per-unit (train-fit >= 0.7)", reg_fit >= 0.7, f"train-fit={reg_fit:.3f}")

# C1: c present in a single un-pooled unit at lam=0
u0, c0, d0 = v2.gen(v2.N_PROBE, 0.0, v2.SEED + 7)
check("C1: c recoverable from a single un-pooled unit (lam=0, R2 >= 0.5)",
      probe(u0[:, 0, :], c0) >= 0.5)

# washed regime: lam=2.0 (raw fully washed)
u, c, d = v2.gen(v2.N_PROBE, 2.0, v2.SEED + 7 + 2000)
raw = u.mean(axis=1)
zc, zr = v2.phi_pool(clf, u), v2.phi_pool(reg, u)
raw_c = probe(raw, c)
clf_c, reg_c = probe(zc, c), probe(zr, c)
clf_c_s, reg_c_s = probe(zc, c, strong=True), probe(zr, c, strong=True)

check("C0 anti-confound: raw (linear) averaging WASHES c at lam=2.0 (R2 <= 0.05)", raw_c <= 0.05, f"raw_c={raw_c:.3f}")
check("DEFEAT: reg_c RECOVERS c post-pool where raw washed (R2 >= 0.40)", reg_c >= 0.40, f"reg_c={reg_c:.3f}")
check("SUPPRESS: clf_d washes c post-pool (R2 <= 0.10)", clf_c <= 0.10, f"clf_d={clf_c:.3f}")
check("OBJECTIVE GAP (reg_c - clf_d, same arch+training) >= 0.30", (reg_c - clf_c) >= 0.30,
      f"gap={reg_c - clf_c:.3f}")
check("PROBE-ROBUST: reg_c recovers c under a STRONG nonlinear probe too (>= 0.35)", reg_c_s >= 0.35,
      f"reg_c(strong)={reg_c_s:.3f}")
check("PROBE-ROBUST: clf_d suppression survives a STRONG probe (c-R2 <= 0.15) -> genuine, not linear-artifact",
      clf_c_s <= 0.15, f"clf_d(strong)={clf_c_s:.3f}")

# DETERMINE: discrete d recovered post-pool through lossy averaging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def dprobe(X, y):
    Xs = StandardScaler().fit_transform(X)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(LogisticRegression(max_iter=2000), Xs, (y > 0).astype(int), cv=skf,
                                 scoring="balanced_accuracy").mean())


check("DETERMINE: d recovered post-pool (both bodies, balanced-acc >= 0.95) through lossy averaging",
      dprobe(zc, d) >= 0.95 and dprobe(zr, d) >= 0.95,
      f"clf_d d-acc={dprobe(zc, d):.3f}, reg_c d-acc={dprobe(zr, d):.3f}")

# reproducibility: re-derive reg_c@lam=2.0 once more, must match (single-thread deterministic)
reg2, _ = v2.train_body("reg_c", u_tr, c_tr, d_tr)
reg_c2 = probe(v2.phi_pool(reg2, u), c)
check("REPRODUCIBLE: reg_c c-R2 byte-stable across a retrain (|diff| < 1e-6)", abs(reg_c2 - reg_c) < 1e-6,
      f"{reg_c:.6f} vs {reg_c2:.6f}")

print(f"\n{'ALL PASS -- the resist is objective-dependent: reg_c demodulates-then-pools to defeat it (probe-robust ~0.5), clf_d suppresses c to ~0; d determined throughout. Imported wall: trained nonlinear bodies do NOT inherit the continuous-resist.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
