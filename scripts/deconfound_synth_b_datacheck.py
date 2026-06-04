#!/usr/bin/env python
"""SUNDOG_V_DECONFOUND Attack-B (semi-synthetic, real-feature) — 0-pre DE-CONFOUND CORRELATION SWEEP.

Re-posed after the first 0-pre run + reading the toy (`chatv2_phase0_bodyresist._gen_coupled`,
`input_probe_precheck`): the toy's input-undecodability is MANUFACTURED by its synthetic
parity-channel encoder (the de-confound targets z_i, made nonlinear by the encoding). So
"toy + real noise" does NOT test a real de-confound. The real Attack-B (SYNLABEL) puts REAL
features in the INPUT path and makes the functional a parity over them; the de-confound is
then a genuine, MODEL-FREE question: does real feature-correlation leak the constructed parity?

CONSTRUCTION (SYNLABEL):
  input  = real feature bits b in {0,1}^D
  label  = u = XOR(b_S) for a FIXED hidden subset S (|S| = s)        [constructed functional]
De-confound = a real-linear probe on b -> u. For INDEPENDENT balanced b, parity is
input-undecodable (det ~ 0). REAL correlation among b can make u linearly leak. That is the
§7 wall as a measurable boundary — and it needs NO model (the de-confound is just a probe).

EXPERIMENT — map de-confound leak vs feature-correlation:
  - SYNTHETIC curve: 1-factor Gaussian copula, tunable equicorrelation rho -> balanced
    correlated bits; sweep rho, measure de-confound det(u | b).
  - REAL anchors: sklearn digits pooled to D bits (fine D=8 and coarse D=4 = stronger corr),
    median-binarized; realized bit-correlation + de-confound det, overlaid on the curve.

PRE-REGISTERED (frozen before running). Tie the bar to the toy's own de-confound standard
(`precheck_max = 0.60` on balanced binary == det 0.20):
  det = (acc - base)/(1 - base);   HOLD iff det <= 0.10 (clean) ;  MARG 0.10 < det <= 0.20 ;
  LEAK det > 0.20 (exceeds the toy precheck bar).
  Primary subset size s = 3 (odd; matches the toy's m=3); s = 2 reported as a more
  correlation-sensitive secondary.
  GREEN-LIGHT the (training-gated) closure read iff the REAL anchors HOLD/MARG (det <= 0.20).
  A LEAKing real anchor = an honest de-confound-wall-on-real-data result: the closure
  substrate must drop correlation (finer pooling / decorrelation) or the cell reports the wall.
"""
import numpy as np
from numpy.random import default_rng
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def parity(b, S):
    return (b[:, S].sum(1) & 1).astype(np.int64)


def deconfound_det(b, u):
    """5-fold CV linear-probe accuracy of b -> u, reported as det over base rate."""
    base = float(max(u.mean(), 1 - u.mean()))
    if len(np.unique(u)) < 2:
        return 1.0, base, 0.0                      # constant u: 'predictable' by base alone, det 0
    acc = float(cross_val_score(LogisticRegression(max_iter=300), b, u, cv=5).mean())
    return acc, base, (acc - base) / (1 - base + 1e-9)


def mean_offdiag_corr(b):
    if b.shape[1] < 2:
        return 0.0
    cc = np.corrcoef(b.T)
    return float(np.abs(cc[np.triu_indices(b.shape[1], 1)]).mean())


def synth_bits(n, D, rho, rng):
    """1-factor Gaussian copula -> balanced correlated bits (pairwise latent corr = rho)."""
    f = rng.standard_normal((n, 1))
    e = rng.standard_normal((n, D))
    g = np.sqrt(rho) * f + np.sqrt(1.0 - rho) * e
    return (g > 0).astype(np.int64)                # median threshold -> ~balanced


def digit_bits(D):
    """sklearn digits pooled to D median-binarized bits with REAL spatial correlation.
    D=8 -> 4x2 blocks ; D=4 -> 2x2 blocks (coarser = stronger correlation)."""
    X = load_digits().images.astype(np.float64)
    grid = {8: (4, 2), 4: (2, 2)}[D]
    rb, cb = 8 // grid[0], 8 // grid[1]
    n = X.shape[0]
    feats = np.zeros((n, D))
    f = 0
    for br in range(grid[0]):
        for bc in range(grid[1]):
            feats[:, f] = X[:, br * rb:(br + 1) * rb, bc * cb:(bc + 1) * cb].mean((1, 2))
            f += 1
    return (feats > np.median(feats, 0)).astype(np.int64)


def flag(det):
    return "HOLD" if det <= 0.10 else ("MARG" if det <= 0.20 else "LEAK")


if __name__ == "__main__":
    rng = default_rng(0)
    D = 8
    Ssets = {2: [0, 1], 3: [0, 1, 2]}              # fixed hidden subsets
    N = 20000
    rhos = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"[cfg] synthetic N={N} D={D} ; real = sklearn-digits ; bar tied to toy precheck "
          f"(det<=0.20). HOLD<=0.10 < MARG <=0.20 < LEAK\n")

    for s in (3, 2):
        S = Ssets[s]
        print(f"================  s = |S| = {s}  (u = XOR of {s} features)  ================")
        print("  SYNTHETIC 1-factor copula:")
        print("   rho | bitcorr | acc / base |   det    flag")
        for rho in rhos:
            b = synth_bits(N, D, rho, rng)
            u = parity(b, S)
            acc, base, det = deconfound_det(b, u)
            print(f"  {rho:.1f} |  {mean_offdiag_corr(b):.3f}  | {acc:.3f}/{base:.3f} | "
                  f"{det:+.3f}  {flag(det)}")
        print("  REAL digit anchors:")
        for Dd in (8, 4):
            Sd = S if s <= Dd else list(range(min(s, Dd)))
            b = digit_bits(Dd)
            u = parity(b, Sd)
            acc, base, det = deconfound_det(b, u)
            print(f"   digits D={Dd}: bitcorr {mean_offdiag_corr(b):.3f} | "
                  f"acc {acc:.3f}/base {base:.3f} | det {det:+.3f}  {flag(det)}")
        print()

    print("VERDICT KEY: green-light closure training iff the REAL anchors are HOLD/MARG "
          "(det<=0.20).")
    print("            a LEAKing real anchor = honest de-confound-wall-on-real-data result.")
