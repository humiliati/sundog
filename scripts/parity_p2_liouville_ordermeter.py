#!/usr/bin/env python3
"""Parity-barrier slate P-2: the Liouville order-meter experiment.

Run the H9-strong order-k Markov ladder (own-R2 = cross-validated max of linear +
MLP, faithful copy of scripts/epsilon_machine_shadow.own_r2) on the REAL Liouville
sequence lambda(n) = (-1)^Omega(n), n <= N.

Pre-registered prediction (the parity barrier holds): own-R2(lambda, k) ~ 0 for
EVERY accessible order k - lambda is invisible to all finite-order predictors,
linear AND nonlinear. Order-meter controls (the apparatus is NOT broken): a
finite-order parity LFSR (order d) and an arithmetic residue (period q) MUST be
caught at their order. Imported wall: Chowla (consistent-with, never proved).
Kill/interest-flip: any order-k signal on lambda must first be shown to be a known
finite-n bias (Liouville summatory drift L(n), small-prime density) - almost
certainly an artifact, not a barrier breach. Cannot breach the barrier.
Run: python scripts/parity_p2_liouville_ordermeter.py
"""
import sys, time
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

N = 10**7  # Liouville range (the real arithmetic object)
NC = 200000  # control-sequence length (controls are periodic/finite-order; short is plenty)
M = 5000   # sampled (n, history) pairs for the ladder
K = 12     # max Markov order
LFSR_D = 5 # finite-order parity control order
RES_Q = 3  # arithmetic residue control period


def own_r2(X, y, both=False):
    """Faithful copy of epsilon_machine_shadow.own_r2: 4-fold CV R2, max(0, linear, MLP)."""
    X = np.asarray(X, float); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    kf = KFold(4, shuffle=True, random_state=0)
    lin = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2").mean()
    mlp = cross_val_score(MLPRegressor(hidden_layer_sizes=(24,), max_iter=300, random_state=0),
                          X, y, cv=kf, scoring="r2").mean()
    return (max(0.0, lin), max(0.0, mlp), max(0.0, lin, mlp)) if both else max(0.0, lin, mlp)


def liouville(N):
    """lambda(n) = (-1)^Omega(n) via a prime-power sieve. Returns float +/-1 array, index 0..N."""
    omega = np.zeros(N + 1, dtype=np.int8)
    comp = np.zeros(N + 1, dtype=bool)
    for p in range(2, N + 1):
        if not comp[p]:
            comp[p * p::p] = True
            pe = p
            while pe <= N:
                omega[pe::pe] += 1
                pe *= p
    return (1 - 2 * (omega & 1)).astype(np.float64)


def lfsr_parity(N, d, seed=0xACE):
    """Order-d finite-parity recurrence c(n)=c(n-1) xor c(n-d); +/-1. Caught by the ladder at k=d."""
    b = np.zeros(N + 1, dtype=np.int8)
    rng = np.random.default_rng(seed)
    b[1:d + 1] = rng.integers(0, 2, size=d)
    for n in range(d + 1, N + 1):
        b[n] = b[n - 1] ^ b[n - d]
    return (1 - 2 * b).astype(np.float64)


def residue(N, q):
    """Arithmetic period-q signal c(n)= +1 if n%q==0 else -1; caught at low order."""
    n = np.arange(N + 1)
    return np.where(n % q == 0, 1.0, -1.0)


def ladder(seq, K, M, rng):
    pos = rng.integers(K + 1, len(seq), size=M)
    y = seq[pos]
    out = {}
    for k in range(1, K + 1):
        X = np.column_stack([seq[pos - j] for j in range(1, k + 1)])
        out[k] = own_r2(X, y)
    return out


def main():
    t0 = time.time()
    print(f"PARITY_P2_LIOUVILLE_ORDERMETER  N={N}  ladder k=1..{K}  sample M={M}", flush=True)
    lam = liouville(N)
    sumL = lam[1:N + 1].sum(); meanL = lam[1:N + 1].mean()
    print(f"  sieve done ({time.time()-t0:.1f}s). Liouville summatory L(N)={int(sumL)}  mean lambda={meanL:.5f}", flush=True)
    print(f"  [known finite-n bias] L(N)<0 (Polya region) is an ORDER-0 mean drift, not finite-order predictability.", flush=True)

    seqLfsr = lfsr_parity(NC, LFSR_D)
    seqRes = residue(NC, RES_Q)
    lamL = ladder(lam, K, M, np.random.default_rng(20260629))
    lfsr = ladder(seqLfsr, K, M, np.random.default_rng(7))
    res = ladder(seqRes, K, M, np.random.default_rng(11))

    print(f"\n  k   own-R2(lambda)   own-R2(LFSR d={LFSR_D})   own-R2(residue q={RES_Q})", flush=True)
    for k in range(1, K + 1):
        print(f"  {k:>2}     {lamL[k]:.3f}            {lfsr[k]:.3f}              {res[k]:.3f}", flush=True)

    # lin-vs-MLP split at the LFSR crossing: parity is nonlinear, so MLP must carry it
    rng2 = np.random.default_rng(99); pos = rng2.integers(K + 1, NC, size=M); yL = seqLfsr[pos]
    Xk = np.column_stack([seqLfsr[pos - j] for j in range(1, LFSR_D + 1)])
    lin, mlp, _ = own_r2(Xk, yL, both=True)
    print(f"\n  LFSR at k=d={LFSR_D}: linear-only R2={lin:.3f}, MLP R2={mlp:.3f}  (finite-order PARITY needs the nonlinear model)", flush=True)

    lam_max = max(lamL.values())
    verdict = ("BARRIER HOLDS empirically: lambda is invisible to every accessible finite order (own-R2 ~ 0, "
               "linear AND nonlinear), while the controls are caught at their order -> apparatus confirmed, null = invisible."
               if lam_max < 0.05 and lfsr[LFSR_D] > 0.5 and res[2] > 0.5 else "see numbers")
    print(f"\n  max own-R2(lambda) over k = {lam_max:.3f}", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"  (Chowla imported; this confirms the barrier at accessible scale, it cannot breach it.)  [{time.time()-t0:.1f}s]", flush=True)


if __name__ == "__main__":
    main()
