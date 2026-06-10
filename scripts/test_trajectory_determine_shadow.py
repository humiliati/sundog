"""Frozen test for the H9 load-bearing-determine probe (arrow of time on a trajectory shadow).
Locks the weak-notion positive: arrow recoverable, time-symmetric + IAAFT foils blind (load-bearing),
determine-type concentration, and the fGn-Hurst negative control coming out NOT load-bearing (apparatus
not rigged). Run: cd /c/Users/hughe/Dev/sundog && python -m pytest scripts/test_trajectory_determine_shadow.py -q
"""
import numpy as np
import trajectory_determine_shadow as M


def _arrow_dataset(n, W, seed):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(-1.5, 1.5, n)
    Xa, Xs, Xi = [], [], []
    for i in range(n):
        g = np.random.default_rng(seed + 100 + i); tr = M.rot_ou(phi[i], W, g)
        Xa.append(M.arrow_feat(tr)); Xs.append(M.sym_o2_feat(tr))
        Xi.append(M.arrow_feat(M.iaaft_per_channel(tr, g)))
    return phi, np.array(Xa), np.array(Xs), np.array(Xi)


def test_arrow_recoverable_and_load_bearing():
    """The crux dissection: arrow recovers signed phi; both time-symmetric foils are blind."""
    phi, Xa, Xs, Xi = _arrow_dataset(160, 2000, 4321)
    assert M.own_r2(Xa, phi) >= 0.70          # recoverable from the directed trajectory
    assert M.own_r2(Xs, phi) <= 0.20          # symmetric-order-2 blind (symmetry-guaranteed fair foil)
    assert M.own_r2(Xi, phi) <= 0.20          # IAAFT matched-spectrum surrogate blind (load-bearing)


def test_trivial_fail():
    """Shuffled-phi label -> no recovery (anti-vacuity)."""
    phi, Xa, _, _ = _arrow_dataset(160, 2000, 4321)
    assert M.own_r2(Xa, np.random.default_rng(9).permutation(phi)) <= 0.20


def test_determine_concentration():
    """Determine signature: the current estimator concentrates std ~ 1/sqrt(W) (the OPPOSITE of resist)."""
    def std_at(W):
        return np.std([M.arrow_feat(M.rot_ou(1.0, W, np.random.default_rng(900 + j)))[0] for j in range(30)])
    s_small, s_big = std_at(250), std_at(4000)
    assert s_small > s_big * 1.8              # concentrates with window length


def test_ensemble_shadow_is_determine():
    """The ACTUAL H8 object: recover phi from the jitter-AVERAGED shadow (mean arrow feat over a jitter
    population). A determine latent has finite mean => the average survives the jitter (graceful decay),
    the opposite of a resist (which would wash to 0). Gate the lam<=1.0 region."""
    rng = np.random.default_rng(555)
    phi = rng.uniform(-1.5, 1.5, 120)
    for lam in (0.5, 1.0):
        Xsh = np.array([M.shadow_feat(phi[i], lam, 8, 1200, np.random.default_rng(500 + i))
                        for i in range(120)])
        assert M.own_r2(Xsh, phi) >= 0.50      # survives jitter -> DETERMINE on the shadow object


def test_negative_control_hurst_is_geometric():
    """fGn Hurst -- a spectral determine latent -- must be recovered by BOTH real and matched-spectrum
    surrogate (NOT load-bearing). Proves the apparatus can say 'geometric', i.e. is not rigged."""
    rng = np.random.default_rng(77)
    H = rng.uniform(0.2, 0.8, 120)
    Xr, Xsg = [], []
    for i in range(120):
        g = np.random.default_rng(300 + i); x = M.fgn(H[i], 2000, g)
        Xr.append(M.logspec(x)); Xsg.append(M.logspec(M.matched_spectrum(x, g)))
    assert M.own_r2(np.array(Xr), H) >= 0.60
    assert M.own_r2(np.array(Xsg), H) >= 0.60   # surrogate ALSO recovers -> geometric, not load-bearing


def test_graded_control_is_discriminating():
    """A latent in BOTH the (clean) arrow and a (confounded) variance channel: the matched-spectrum surrogate
    keeps the variance, loses the arrow -> recovers psi PARTIALLY (intermediate), strictly below real.
    Proves the apparatus outputs a discriminating intermediate, not just 0 or 1."""
    rng = np.random.default_rng(21)
    ng = 120; psi = rng.uniform(0.5, 2.0, ng); nui = rng.uniform(0.0, 1.0, ng)
    sig = 0.7 + 0.55 * psi + 0.6 * nui
    trs = M.rot_ou_batch(psi, 1500, np.random.default_rng(22), sigma=sig)
    feat = lambda tr: np.concatenate([M.arrow_feat(tr), [np.var(tr[:, 0]) + np.var(tr[:, 1])]])
    Xr = np.array([feat(trs[i]) for i in range(ng)])
    Xs = np.array([feat(M.iaaft_per_channel(trs[i], np.random.default_rng(600 + i))) for i in range(ng)])
    r_real, r_surr = M.own_r2(Xr, psi), M.own_r2(Xs, psi)
    assert r_real >= 0.60
    assert 0.20 <= r_surr <= r_real - 0.15      # intermediate, strictly below real
