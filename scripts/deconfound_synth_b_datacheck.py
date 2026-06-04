#!/usr/bin/env python
"""SUNDOG_V_DECONFOUND Attack-B (semi-synthetic, real-feature) — model-free 0-pre data-check.

Pre-registered gate (frozen BEFORE running; mirrors the Phase-7 data-validation and the
Othello legal-move screen). No model. Pure numpy + sklearn.

CONSTRUCTION (minimal controlled upgrade of the frozen Phase-7b coupled-latent positive):
  u ~ Uniform({0,1}^3)                      hidden source (constructed functional; input-undecodable)
  A = _COUPLE_A (8x3 coupling graph)         same graph as Phase 7/7b/7c
  clean_i = parity(u, A_i)                    latent i's clean value
  z_i = clean_i XOR b_i                       observed latent
  b_i = REAL digit-feature bit (flip noise), P(b_i=1) ~= q (q=0.12 ~ Phase-7b's working ~0.10)
The ONLY change from Phase-7b is that the 8 noise bits b carry REAL spatial correlation
(handwritten-digit structure) instead of being i.i.d. The CONTROLLED comparison arm shuffles
each b-column independently (kills cross-i correlation, preserves the marginal flip rate q) =
the i.i.d. Phase-7b regime at matched q. So real correlation is the only changed variable.

PRE-REGISTERED 0-pre PASS CRITERIA:
  (1) DE-CONFOUND holds: real-linear probe (LogReg) on z -> each u-bit in [0.45, 0.55] (chance
      base 0.50), on BOTH real and iid arms. >0.55 on the real arm => real correlation LEAKS the
      parity (the de-confound fails — a finding, needs lower q / decorrelation).
  (2) CLOSURE bracket (real arm), data-level via conditional-entropy determination:
      k_func(u) <= 3  AND  k_state(z_j) >= k_func + 2   (a >=2-gap, Phase-7b-class signature).
  (3) CONTROLLED read: real-arm bracket vs iid-arm bracket (measured headline, not pass/fail).

Substrate: sklearn load_digits (8x8 handwritten digits = a download-free MNIST-class real
feature distribution). The eventual TRAINING spec may use full MNIST; for the model-free
data-level check, digits' real correlation structure is what matters.
"""
import numpy as np
from numpy.random import default_rng
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

# Same coupling graph as Phase 7/7b/7c (chatv2_phase0_bodyresist._COUPLE_A).
_COUPLE_A = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1],
                      [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=np.int8)
M = 3            # hidden source bits
D = 8            # observed latents (= rows of _COUPLE_A)
Q = 0.12         # target marginal flip rate P(b_i=1) ~ Phase-7b working regime


def real_feature_bits(rng):
    """8 spatially-pooled, quantile-binarized digit features -> a pool of real b-bit rows.

    Returns B_real (n_img, 8) with column-wise P(=1) ~= Q and REAL cross-column correlation.
    """
    X = load_digits().images.astype(np.float64)              # (1797, 8, 8)
    n = X.shape[0]
    feats = np.zeros((n, D), dtype=np.float64)
    f = 0
    for br in range(4):                                       # 4x2 grid of 2-row x 4-col blocks -> 8 feats
        for bc in range(2):
            feats[:, f] = X[:, br * 2:(br + 1) * 2, bc * 4:(bc + 1) * 4].mean(axis=(1, 2))
            f += 1
    thr = np.quantile(feats, 1.0 - Q, axis=0)                # per-feature (1-Q) quantile
    B = (feats > thr).astype(np.int8)                        # ~Q ones per column, real correlation
    return B


def gen(n, rng, correlated=True):
    """Draw n samples of (u, z). b sampled from the real digit-bit pool (correlated) or
    column-independently shuffled (iid control)."""
    B_pool = real_feature_bits(rng)
    npool = B_pool.shape[0]
    if correlated:
        idx = rng.integers(0, npool, size=n)
        b = B_pool[idx]                                      # rows kept intact => real cross-i correlation
    else:
        b = np.empty((n, D), dtype=np.int8)
        for j in range(D):                                   # independent per-column resample => iid, same marginal
            b[:, j] = B_pool[rng.integers(0, npool, size=n), j]
    u = rng.integers(0, 2, size=(n, M)).astype(np.int8)
    clean = (u @ _COUPLE_A.T) & 1                            # parity(u, A_i) per latent  (XOR via mod 2)
    z = (clean ^ b).astype(np.int8)
    return u, z, b


def _entropy(counts):
    p = counts[counts > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def cond_ent(y, x, ny, nx):
    """H(Y|X) in bits, plug-in from counts. y in [0,ny), x in [0,nx)."""
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(np.float64)
    px = joint.sum(axis=1)
    tot = px.sum()
    H = 0.0
    for xi in range(nx):
        if px[xi] > 0:
            H += (px[xi] / tot) * _entropy(joint[xi])
    return H


def _enc(bits):
    """Encode an (n, k) bit array as integer codes 0..2^k-1."""
    k = bits.shape[1]
    w = (1 << np.arange(k)).astype(np.int64)
    return (bits.astype(np.int64) * w).sum(axis=1)


def sweep(u, z, rng, ks, R=24):
    u_idx = _enc(u)                                          # 0..7
    Hu = _entropy(np.bincount(u_idx, minlength=2 ** M))
    out = {}
    for k in ks:
        df, ds = [], []
        for _ in range(R):
            S = rng.choice(D, k, replace=False)
            xS = _enc(z[:, S])
            df.append(1.0 - cond_ent(u_idx, xS, 2 ** M, 2 ** k) / (Hu + 1e-12))
            J = [j for j in range(D) if j not in set(S.tolist())]
            for j in rng.choice(J, min(4, len(J)), replace=False):
                zj = z[:, j].astype(np.int64)
                Hzj = _entropy(np.bincount(zj, minlength=2))
                if Hzj > 1e-9:
                    ds.append(1.0 - cond_ent(zj, xS, 2, 2 ** k) / Hzj)
        out[k] = (float(np.mean(df)), float(np.mean(ds)))
    return out


def deconfound_acc(u, z):
    """Max held-out LogReg accuracy predicting any u-bit from z (real-linear). Want ~0.50."""
    n = len(u); tr = slice(0, n // 2); he = slice(n // 2, n)
    accs = []
    for j in range(M):
        if len(np.unique(u[tr, j])) < 2:
            continue
        m = LogisticRegression(max_iter=200).fit(z[tr], u[tr, j])
        accs.append(m.score(z[he], u[he, j]))
    return float(max(accs)) if accs else float("nan")


def bracket(res, ks, thr=0.95):
    kf = next((k for k in ks if res[k][0] >= thr), None)
    ksx = next((k for k in ks if res[k][1] >= thr), None)
    ok = kf is not None and (ksx is None or ksx >= kf + 2)
    return kf, ksx, ok


if __name__ == "__main__":
    rng = default_rng(0)
    N = 30000
    ks = [1, 2, 3, 4, 5, 6, 7]
    print(f"[cfg] N={N} D={D} m={M} q(target)={Q}  substrate=sklearn-digits 8x8 -> 8 pooled bits\n")

    for arm, corr in (("REAL-correlated", True), ("iid-control", False)):
        u, z, b = gen(N, rng, correlated=corr)
        qhat = b.mean(axis=0)
        cc = np.corrcoef(b.T)
        offdiag = np.abs(cc[np.triu_indices(D, 1)])
        dc = deconfound_acc(u, z)
        res = sweep(u, z, rng, ks)
        kf, ksx, ok = bracket(res, ks)
        print(f"==== {arm} ====")
        print(f"  flip rate q_hat per latent: {np.array2string(qhat, precision=2)}")
        print(f"  mean |b cross-corr| (off-diag): {offdiag.mean():.3f}  (real should exceed iid)")
        print(f"  de-confound: max LogReg acc(z -> u-bit) = {dc:.3f}   [want in 0.45..0.55]")
        print("   k | det_func(u) | det_state(z_j)")
        for k in ks:
            print(f"  {k:>2} |   {res[k][0]:.3f}    |   {res[k][1]:.3f}")
        print(f"  -> k_func(u)={kf}  k_state(z_j)={ksx}  bracket(>=2 gap)={ok}\n")

    print("PRE-REG VERDICT KEY:")
    print("  PASS  = de-confound in 0.45..0.55 (both arms) AND real-arm bracket True")
    print("  LEAK  = real-arm de-confound > 0.55 (real correlation defeats input-undecodability)")
    print("  WASH  = de-confound holds but real-arm bracket False (real correlation washes closure)")
