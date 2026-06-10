"""Frozen test for the H9-strong probe (a determine latent load-bearing vs ALL finite-order surrogates).
Locks: the full-history PARITY latent is recoverable (causal state) but blind to every order-k Markov-resample
surrogate (no finite-order sufficient statistic); a finite order-d control IS recovered once k>=d-1 (the ladder
is a working order-meter, crossing tracks the order); determine concentration; anti-vacuity.
Run via the repo's direct-harness convention (pytest not installed):
  cd /c/Users/hughe/Dev/sundog/scripts && python -c "import test_epsilon_machine_shadow as T; ..."
"""
import numpy as np
import epsilon_machine_shadow as M


def _ladder(phi, n, L, seed, order, ks):
    out = {}
    for k in ks:
        Xk = [M.parity_feat(M.markov_k_surrogate(
                  M.gen_parity(phi[i], L, np.random.default_rng(seed + 100 + i), order),
                  k, np.random.default_rng(seed + 700 + 13 * k + i))) for i in range(n)]
        out[k] = M.own_r2(np.array(Xk), phi)
    return out


def test_full_parity_load_bearing_vs_all_finite_order():
    """Real recovers phi via the causal state; NO order-k surrogate (k=1..3) does."""
    phi, X = M.dataset(140, 5000, 1234, order=None)
    assert M.own_r2(X, phi) >= 0.70
    lad = _ladder(phi, 140, 5000, 1234, None, (1, 2, 3))
    for k in (1, 2, 3):
        assert lad[k] <= 0.20            # invisible at every finite order


def test_trivial_fail():
    phi, X = M.dataset(140, 5000, 1234, order=None)
    assert M.own_r2(X, np.random.default_rng(5).permutation(phi)) <= 0.20


def test_ladder_detects_finite_order_control():
    """An order-3 latent: surrogate blind at k=1 but recovered at k>=2 (crossing at k=d-1). Proves the ladder
    is not broken -- it CAN detect finite order, and the crossing tracks the latent's order."""
    phi, X = M.dataset(140, 5000, 9001, order=3)
    assert M.own_r2(X, phi) >= 0.70
    lad = _ladder(phi, 140, 5000, 9001, 3, (1, 2, 3))
    assert lad[1] <= 0.30                # below the crossing
    assert lad[2] >= 0.40 and lad[3] >= 0.40   # at/above the crossing (k = d-1 = 2)


def test_determine_concentration():
    """The causal-state correlation estimator concentrates std ~ 1/sqrt(L): a determine latent (H9 lineage)."""
    def std_at(L):
        return np.std([M.parity_feat(M.gen_parity(0.6, L, np.random.default_rng(900 + j)))[-1]
                       for j in range(30)])
    assert std_at(500) > std_at(8000) * 1.8
