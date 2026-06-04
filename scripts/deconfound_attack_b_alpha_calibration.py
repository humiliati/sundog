#!/usr/bin/env python
"""Model-free alpha calibration for Deconfound Phase 0C.

Measures the realized input-deconfound leak for the shared-factor digit substrate:

    b_j = 1[zscore(feat_j) + alpha * g > median]
    u   = XOR(b_0,b_1,b_2)

No model training. This is the calibration receipt used to lock Phase 0C rungs.
"""
import argparse

import numpy as np
from numpy.random import default_rng
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


D = 8
S = [0, 1, 2]


def digit_features():
    x = load_digits().images.astype(np.float64)
    feats = np.zeros((x.shape[0], D))
    f = 0
    for br in range(4):
        for bc in range(2):
            feats[:, f] = x[:, br * 2:(br + 1) * 2, bc * 4:(bc + 1) * 4].mean((1, 2))
            f += 1
    return feats


def mean_offdiag_corr(b):
    corr = np.corrcoef(b.T)
    return float(np.abs(corr[np.triu_indices(b.shape[1], 1)]).mean())


def deconfound_det(b, u):
    base = float(max(u.mean(), 1 - u.mean()))
    acc = float(cross_val_score(LogisticRegression(max_iter=500), b, u, cv=5).mean())
    det = (acc - base) / (1 - base + 1e-9)
    return acc, base, det


def flag(det):
    if det <= 0.10:
        return "HOLD"
    if det <= 0.20:
        return "MARG"
    return "LEAK"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,4.0,5.0")
    parser.add_argument("--factor-seed", type=int, default=20260604)
    args = parser.parse_args()

    feats = digit_features()
    z = (feats - feats.mean(axis=0)) / feats.std(axis=0)
    g = default_rng(args.factor_seed).standard_normal((len(z), 1))
    native_bits = (feats > np.median(feats, axis=0)).astype(np.int64)

    print(f"[cfg] digit shared-factor calibration D={D} S={S} factor_seed={args.factor_seed}")
    print("alpha,bitcorr,acc,base,det,flag,u_rate,exact_alpha0_match")
    for alpha in [float(x) for x in args.alphas.split(",") if x.strip()]:
        y = z + alpha * g
        b = (y > np.median(y, axis=0)).astype(np.int64)
        u = (b[:, S].sum(axis=1) & 1).astype(np.int64)
        acc, base, det = deconfound_det(b, u)
        exact = int(np.array_equal(b, native_bits))
        print(
            f"{alpha:.2f},{mean_offdiag_corr(b):.3f},{acc:.3f},{base:.3f},"
            f"{det:+.3f},{flag(det)},{u.mean():.3f},{exact}"
        )


if __name__ == "__main__":
    main()
