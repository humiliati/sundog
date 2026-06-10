#!/usr/bin/env python
"""H9-strong — a determine latent LOAD-BEARING vs ALL finite-order surrogates (a causal-state / eps-machine).

The decisive strengthening of the H9 weak positive (docs/atlas/H9_LOADBEARING_DETERMINE_RESULT.md), which was
load-bearing only vs the time-symmetric class. Here the latent lives in a GENUINELY infinite-order statistic
with NO finite-order sufficient statistic: the running PARITY of a fair-coin driver -- a strictly-sofic process
(2 causal states, infinite Markov order, the canonical Crutchfield eps-machine example).

  b_t ~ Bernoulli(1/2) observed;  hidden causal state P_t = b_1 xor ... xor b_t (needs ALL history);
  c_t = P_t with prob (1+phi)/2 else 1-P_t.   Latent phi = parity-readout fidelity, corr(c_t,P_t)=phi.

phi is uncorrelated with EVERY finite-order function of the sequence (for any finite index set S, the
complementary parity xor_{i not in S} b_i is an independent fair coin that randomizes the relation), so it is
readable ONLY via the full-history parity (the causal state). Test: real recovers phi; an order-k Markov-resample
surrogate ladder (k=1..K) does NOT, for every k. Negative control: an order-d latent IS recovered once k>=d
(the ladder works). Pre-reg: docs/atlas/H9S_EPSILON_MACHINE_PREREG.md. NOT public-eligible.
"""
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ============================ the parity eps-machine ============================ #
def gen_parity(phi, L, rng, order=None):
    """Observable (b, c). c reads the running parity (order=None, the full causal state) OR the order-d parity
    (order=d, a FINITE-order latent for the negative control), with fidelity phi."""
    b = rng.integers(0, 2, L)
    if order is None:
        target = np.cumsum(b) % 2                       # full running parity P_t (infinite Markov order)
    else:
        pad = np.concatenate([np.zeros(order - 1, int), b])
        target = np.array([pad[i:i + order].sum() % 2 for i in range(L)]) % 2   # order-d parity (finite order)
    flip = rng.random(L) > (1 + phi) / 2                # c = target, flipped w.p. (1-phi)/2
    c = np.where(flip, 1 - target, target)
    return np.stack([b, c], 1)                          # (L, 2) in {0,1}


def parity_feat(seq, D=4):
    """Same extraction on real & surrogate: corr(c_t, xor of last d b's) for d=1..D, plus corr(c_t, full running
    parity). +-1 coding so each corr = the readout fidelity at that order. The full-parity entry is the only one
    that can carry an infinite-order latent; the d<=D entries carry finite-order latents."""
    b, c = seq[:, 0], seq[:, 1]
    cs = 2 * c - 1                                       # +-1
    feats = []
    for d in range(1, D + 1):
        pad = np.concatenate([np.zeros(d - 1, int), b])
        pard = 2 * (np.array([pad[i:i + d].sum() % 2 for i in range(len(b))]) % 2) - 1
        feats.append(np.mean(cs * pard))
    full = 2 * (np.cumsum(b) % 2) - 1
    feats.append(np.mean(cs * full))
    return np.array(feats)


# ============================ order-k Markov-resample surrogate ============================ #
def markov_k_surrogate(seq, k, rng):
    """Fit a k-th order Markov model over the joint 4-symbol alphabet (b,c) and resample. Preserves every
    (k+1)-block joint statistic in expectation; destroys all longer-range structure. As k->inf -> the real
    process. (k=0 = i.i.d. resample of the symbol marginal.)"""
    sym = seq[:, 0] * 2 + seq[:, 1]                      # 0..3
    L = len(sym)
    if k == 0:
        vals, cnts = np.unique(sym, return_counts=True)
        out = rng.choice(vals, L, p=cnts / cnts.sum())
        return np.stack([out // 2, out % 2], 1)
    from collections import defaultdict
    trans = defaultdict(lambda: np.ones(4))             # Laplace prior
    for i in range(k, L):
        trans[tuple(sym[i - k:i])][sym[i]] += 1
    cdf = {ctx: np.cumsum(v / v.sum()) for ctx, v in trans.items()}
    default = np.array([0.25, 0.5, 0.75, 1.0])
    out = list(sym[:k]); u = rng.random(L)              # inverse-CDF sampling (faster than per-step rng.choice)
    for i in range(k, L):
        c = cdf.get(tuple(out[i - k:i]), default)
        out.append(int(np.searchsorted(c, u[i])))
    out = np.array(out)
    return np.stack([out // 2, out % 2], 1)


# ============================ probe ============================ #
def own_r2(X, y):
    X = np.asarray(X); X = (X - X.mean(0)) / (X.std(0) + 1e-9); kf = KFold(4, shuffle=True, random_state=0)
    lin = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2").mean()
    mlp = cross_val_score(MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=0),
                          X, y, cv=kf, scoring="r2").mean()
    return max(0.0, lin, mlp)


def dataset(n, L, seed, order=None):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 1.0, n)
    X = np.array([parity_feat(gen_parity(phi[i], L, np.random.default_rng(seed + 100 + i), order))
                  for i in range(n)])
    return phi, X


def surrogate_ladder_r2(phi, n, L, seed, order, K):
    """For each k in 1..K: build the order-k surrogate of each sequence, extract the feature, own-R2(phi)."""
    out = {}
    for k in range(1, K + 1):
        Xk = []
        for i in range(n):
            seq = gen_parity(phi[i], L, np.random.default_rng(seed + 100 + i), order)
            sur = markov_k_surrogate(seq, k, np.random.default_rng(seed + 700 + 13 * k + i))
            Xk.append(parity_feat(sur))
        out[k] = own_r2(np.array(Xk), phi)
    return out


def main():
    frozen = "--frozen" in sys.argv
    n, L, K, seed = (250, 6000, 4, 20260609) if frozen else (160, 5000, 4, 999)
    mode = "frozen" if frozen else "calibrate"
    print(f"[H9-strong {mode}] n={n} L={L} ladder k=1..{K} seed={seed}  (parity eps-machine)", flush=True)

    # --- POSITIVE: full-history parity (infinite-order) latent ---
    phi, X = dataset(n, L, seed, order=None)
    r_real = own_r2(X, phi)
    r_perm = own_r2(X, np.random.default_rng(seed + 7).permutation(phi))
    lad = surrogate_ladder_r2(phi, n, L, seed, None, K)
    print("POSITIVE latent = full-history parity (causal state, infinite Markov order):")
    print(f"  real own-R2(phi)            = {r_real:.3f}   (>=0.70 recoverable via the causal state)")
    print(f"  trivial-FAIL (shuffled phi)  = {r_perm:.3f}   (~0)")
    for k in range(1, K + 1):
        print(f"  order-{k} surrogate own-R2(phi) = {lad[k]:.3f}   (<=0.20: NOT recoverable at order {k})")
    lb = r_real >= 0.70 and r_perm <= 0.20 and all(lad[k] <= 0.20 for k in lad)
    print(f"  ** LOAD-BEARING vs all finite order (k<=K) = {lb} **")

    # --- NEGATIVE CONTROL: order-d latent. A d-consecutive-bit parity spans a d-block, so the order-k
    # Markov surrogate preserves it iff k >= d-1 -> recovery RISES and crosses at k=d-1, the crossing TRACKING
    # the latent's order. The full-parity positive never crosses. (d=3,4 keep the crossing inside k=1..K.) ---
    print("NEGATIVE CONTROL (order-d latent -- the ladder is not broken; it detects finite order, crossing at k=d-1):")
    ctrl_ok = True
    for d in (3, 4):
        if d - 1 > K:
            continue
        phid, Xd = dataset(n, L, seed + 1000 * d, order=d)
        rd_real = own_r2(Xd, phid)
        ladd = surrogate_ladder_r2(phid, n, L, seed + 1000 * d, d, K)
        xk = d - 1                                       # crossing order
        above = ladd[xk] >= 0.40 and ladd[K] >= 0.40
        below = ladd[xk - 1] <= 0.30 if xk - 1 >= 1 else True
        print(f"  order-{d} latent (cross@k={xk}): real={rd_real:.3f}  surrogate k=" +
              " ".join(f"{k}:{ladd[k]:.2f}" for k in ladd) + f"   (low<k, high>=k: {above and below})")
        ctrl_ok = ctrl_ok and above and below
    print(f"  ** ladder valid (detects finite-order controls, crossing tracks the order) = {ctrl_ok} **")

    # --- determine concentration (the H9 lineage: this is a determine latent, not a resist) ---
    conc = [np.std([parity_feat(gen_parity(0.6, Lt, np.random.default_rng(seed + 900 + j)))[-1]
                    for j in range(40)]) for Lt in (500, 2000, 8000)]
    print(f"  determine concentration std@L=500/2000/8000 = {conc[0]:.4f}/{conc[1]:.4f}/{conc[2]:.4f} (~1/sqrt(L))")

    verdict = ("POSITIVE (strong notion): a determine latent on the full-history PARITY causal state is "
               "load-bearing vs the entire order-k surrogate ladder (k<=K) -- no finite-order sufficient "
               "statistic -- while the order-d negative controls ARE detected at k>=d (the ladder works). "
               "All-k is analytic (complementary-parity independence); the ladder confirms it empirically."
               if lb and ctrl_ok else "CHECK (a gate failed)")
    print(f"\nH9-strong {mode}: load-bearing={lb} AND ladder-valid={ctrl_ok} => {verdict}")


if __name__ == "__main__":
    main()
