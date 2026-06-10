#!/usr/bin/env python
"""H9 — a LOAD-BEARING determine-type latent on a TRAJECTORY shadow (the arrow of time).

The principled frontier opened by the H8 capstone theorem (docs/atlas/H8_SHADOW_GEOMETRICITY_THEOREM.md):
H8 proved no load-bearing RESIST on snapshot shadows (resist => phase => geometric) and flagged (R2) that the
only escape -- trajectory-irreducible invariants -- are DETERMINE-type. H9 shows one is genuinely LOAD-BEARING:
the non-equilibrium PROBABILITY CURRENT / arrow of time of a rotational Ornstein-Uhlenbeck process,
  dv = -(k I + phi J) v dt + sigma dW,   J = 90-deg rotation,   phi = signed rotational current.
phi is a DETERMINE-type latent (ergodic time-average, concentrates by LLN) recoverable from the directed
trajectory but PROVABLY invisible to every TIME-SYMMETRIC order-2 statistic (time reversal preserves all
symmetric order-2 exactly and flips only the arrow; the rotational-OU stationary covariance is phi-independent,
Risken). So phi is load-bearing vs the time-symmetric (equilibrium / IAAFT) surrogate class -- the standard
time-series null -- the WEAK/order-k notion (honest: NOT vs all finite orders; that is the eps-machine follow-on).

The own-R^2-WITHIN-distribution test (the v3 fix; never train-real/test-surrogate transfer-R^2). NEGATIVE
CONTROL: the Hurst exponent of fractional Gaussian noise -- a determine latent fully IN the power spectrum, so
a matched-spectrum surrogate recovers it => NOT load-bearing (proves the apparatus can say "geometric" too).
Pre-reg: docs/atlas/H9_LOADBEARING_DETERMINE_PREREG.md. NOT public-eligible.
"""
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

K, SIGMA, DT = 1.0, 1.0, 0.02


# ============================ rotational OU (the directed substrate) ============================ #
def rot_ou(phi, W, rng):
    v = rng.standard_normal(2); out = np.empty((W, 2)); sq = SIGMA * np.sqrt(DT)
    for t in range(W):
        drift = -np.array([K * v[0] - phi * v[1], K * v[1] + phi * v[0]])
        v = v + DT * drift + sq * rng.standard_normal(2); out[t] = v
    return out


def rot_ou_batch(phis, W, rng, sigma=SIGMA):
    """Vectorized over a population of phis -> (B, W, 2). Same Euler step as rot_ou, batched for speed so the
    lossy ENSEMBLE shadow (mean over a jitter population) is cheap. sigma may be a per-trajectory array."""
    phis = np.asarray(phis, float); B = len(phis); sig = np.broadcast_to(sigma, (B,))
    v = rng.standard_normal((B, 2)); out = np.empty((B, W, 2)); sq = (sig * np.sqrt(DT))[:, None]
    for t in range(W):
        dx = -(K * v[:, 0] - phis * v[:, 1]); dy = -(K * v[:, 1] + phis * v[:, 0])
        v = v + DT * np.stack([dx, dy], 1) + sq * rng.standard_normal((B, 2)); out[:, t] = v
    return out


def shadow_feat(phi, lam, J, W, rng):
    """The H8 lossy ENSEMBLE shadow: jitter phi -> phi + lam*xi over a J-population and AVERAGE the arrow
    feature. A DETERMINE latent has finite mean => the average survives (graceful decay in lam), the opposite
    of a resist (which would wash to 0). Returns the mean arrow feature over the jitter population."""
    xs = phi + lam * rng.standard_normal(J)
    trs = rot_ou_batch(xs, W, rng)
    return np.mean([arrow_feat(trs[j]) for j in range(J)], axis=0)


def arrow_feat(traj):
    """directed (antisymmetric two-time) feature: <v_t x v_{t+1}> at a few lags = the arrow / current."""
    vx, vy = traj[:, 0], traj[:, 1]
    return np.array([np.mean(vx[:-L] * vy[L:] - vy[:-L] * vx[L:]) for L in (1, 2, 3)])


def sym_o2_feat(traj):
    """TIME-SYMMETRIC order-2 (variance + symmetric autocov + SYMMETRIZED cross-cov + histogram). Every entry
    is time-reversal-invariant => provably blind to the arrow: the symmetry-guaranteed fair-foil witness."""
    vx, vy = traj[:, 0], traj[:, 1]; out = [np.var(vx), np.var(vy), np.mean(vx * vy)]
    for L in (1, 2, 3):
        out.append(0.5 * (np.mean(vx[:-L] * vx[L:]) + np.mean(vy[:-L] * vy[L:])))
        out.append(0.5 * (np.mean(vx[:-L] * vy[L:] + vy[:-L] * vx[L:])))
    out += list(np.histogram(vx, bins=8, range=(-4, 4), density=True)[0])
    return np.array(out)


def iaaft_per_channel(traj, rng):
    out = np.empty_like(traj)
    for c in (0, 1):
        x = traj[:, c]; f = np.fft.rfft(x - x.mean())
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(f))); ph[0] = 1
        out[:, c] = np.fft.irfft(np.abs(f) * ph, n=len(x)) + x.mean()
    return out


# ============================ fGn negative control (a SPECTRAL determine latent) ================ #
def fgn(H, W, rng):
    """fractional Gaussian noise, Hurst H, by spectral synthesis (PSD ~ f^{1-2H}). Gaussian => H is fully in
    the power spectrum => a matched-spectrum surrogate recovers it => NOT load-bearing (the negative control)."""
    f = np.fft.rfftfreq(W); f[0] = f[1]
    amp = f ** ((1 - 2 * H) / 2.0)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(f))); ph[0] = 1
    x = np.fft.irfft(amp * ph, n=W)
    return (x - x.mean()) / (x.std() + 1e-9)


def logspec(x, nb=24):
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    idx = np.linspace(1, len(p) - 1, nb + 1).astype(int)
    return np.array([np.log(p[idx[i]:idx[i + 1]].mean() + 1e-12) for i in range(nb)])


def matched_spectrum(x, rng):
    f = np.fft.rfft(x - x.mean())
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(f))); ph[0] = 1
    return np.fft.irfft(np.abs(f) * ph, n=len(x))


# ============================ probe ============================================================ #
def own_r2(X, y):
    X = (X - X.mean(0)) / (X.std(0) + 1e-9); kf = KFold(4, shuffle=True, random_state=0)
    lin = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2").mean()
    mlp = cross_val_score(MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=0),
                          X, y, cv=kf, scoring="r2").mean()
    return max(0.0, lin, mlp)


def main():
    frozen = "--frozen" in sys.argv
    n, W, seed = (400, 4000, 20260609) if frozen else (300, 2000, 999)
    mode = "frozen" if frozen else "calibrate"
    print(f"[H9 {mode}] n={n} W={W} seed={seed}  (rotational OU, k={K} sigma={SIGMA} dt={DT})", flush=True)
    rng = np.random.default_rng(seed)

    # --- features for the arrow-of-time dissection (per-trajectory, at known phi) ---
    phi = rng.uniform(-1.5, 1.5, n)
    Xa, Xs, Xi = [], [], []
    for i in range(n):
        g = np.random.default_rng(seed + 100 + i); tr = rot_ou(phi[i], W, g)
        Xa.append(arrow_feat(tr)); Xs.append(sym_o2_feat(tr)); Xi.append(arrow_feat(iaaft_per_channel(tr, g)))
    Xa, Xs, Xi = map(np.array, (Xa, Xs, Xi))
    r_arrow = own_r2(Xa, phi); r_sym = own_r2(Xs, phi); r_iaaft = own_r2(Xi, phi)
    r_perm = own_r2(Xa, np.random.default_rng(seed + 7).permutation(phi))
    corr = abs(np.corrcoef(Xa[:, 0], phi)[0, 1])
    # LEAD WITH THE DISSECTION (the red-team caveat): the foils being blind ARE the content. The arrow
    # recovery is the trivially-recoverable leg -- the arrow feature IS the current estimator (corr~0.99 w/phi),
    # so 0.97 is "a line fits a line"; the claim rests entirely on the time-symmetric foils scoring ~0.
    print("DISSECTION (the content is the foils being blind, NOT the recovery number):")
    print(f"  [foil] symmetric-order-2 own-R2 = {r_sym:.3f}   (<=0.20: blind by time-reversal symmetry, Risken)")
    print(f"  [foil] IAAFT matched-spectrum   = {r_iaaft:.3f}   (<=0.20: load-bearing vs the standard surrogate)")
    print(f"  [trivial-PASS] arrow own-R2     = {r_arrow:.3f}   (the arrow feat IS the phi estimator, corr={corr:.3f})")
    print(f"  [trivial-FAIL] shuffled phi      = {r_perm:.3f}   (~0)")

    # --- the ACTUAL H8 object: load-bearing on the LOSSY ENSEMBLE SHADOW (jitter-averaged), not per-traj ---
    ns, J, Ws = min(n, 160), 10, min(W, 1500)
    sh_phi = np.random.default_rng(seed + 11).uniform(-1.5, 1.5, ns)
    print(f"ENSEMBLE SHADOW (the H8 object: mean arrow feat over a J={J} jitter population, n={ns} W={Ws}):")
    sh_ok = True
    for lam in (0.0, 0.5, 1.0, 2.0):
        Xsh = np.array([shadow_feat(sh_phi[i], lam, J, Ws, np.random.default_rng(seed + 500 + i))
                        for i in range(ns)])
        r_sh = own_r2(Xsh, sh_phi)
        print(f"  lambda={lam:>3}: shadow own-R2(phi) = {r_sh:.3f}   "
              f"({'DETERMINE: survives jitter' if r_sh >= 0.40 else 'washed'})")
        if lam <= 1.0 and r_sh < 0.50:
            sh_ok = False
    print(f"  ** shadow graceful-decay (determine, not washed) = {sh_ok} **")

    conc = [np.std([arrow_feat(rot_ou(1.0, Wt, np.random.default_rng(seed + 900 + j)))[0] for j in range(40)])
            for Wt in (250, 1000, 4000)]
    print(f"  determine concentration std@W=250/1000/4000 = {conc[0]:.4f}/{conc[1]:.4f}/{conc[2]:.4f} (~1/sqrt(W))")
    lb = (r_arrow >= 0.70 and r_sym <= 0.20 and r_iaaft <= 0.20 and r_perm <= 0.20
          and conc[0] > conc[2] * 1.8 and sh_ok)
    print(f"  ** LOAD-BEARING DETERMINE (vs time-symmetric, on the shadow object) = {lb} **")

    # --- negative controls: the apparatus must be able to say 'geometric' AND give an INTERMEDIATE verdict ---
    # (1) trivial all-spectral extreme: fGn Hurst (H == the spectral slope) -> surrogate recovers it -> geometric
    H = rng.uniform(0.2, 0.8, n)
    Xr, Xsg = [], []
    for i in range(n):
        g = np.random.default_rng(seed + 300 + i); x = fgn(H[i], W, g)
        Xr.append(logspec(x)); Xsg.append(logspec(matched_spectrum(x, g)))
    rH_real = own_r2(np.array(Xr), H); rH_surr = own_r2(np.array(Xsg), H)
    # (2) GRADED control: a latent in BOTH the arrow (phi=psi, clean) AND the variance, but the variance channel
    # is CONFOUNDED by an independent nuisance so it only PARTIALLY determines psi. IAAFT preserves variance and
    # destroys the arrow -> surrogate recovers psi only through the confounded variance -> an INTERMEDIATE verdict
    # strictly below real (which also has the clean arrow). Earns 'not rigged' across a range, not just at an endpoint.
    ng, Wg = min(n, 160), min(W, 1500)
    rg = np.random.default_rng(seed + 21)
    psi = rg.uniform(0.5, 2.0, ng); nui = rg.uniform(0.0, 1.0, ng)
    sig = 0.7 + 0.55 * psi + 0.6 * nui                                  # variance encodes psi, confounded by nui
    trs = rot_ou_batch(psi, Wg, np.random.default_rng(seed + 22), sigma=sig)   # phi=psi (clean arrow), sigma=confounded
    Xg_real, Xg_surr = [], []
    for i in range(ng):
        g = np.random.default_rng(seed + 600 + i)
        feat = lambda tr: np.concatenate([arrow_feat(tr), [np.var(tr[:, 0]) + np.var(tr[:, 1])]])
        Xg_real.append(feat(trs[i])); Xg_surr.append(feat(iaaft_per_channel(trs[i], g)))
    rg_real = own_r2(np.array(Xg_real), psi); rg_surr = own_r2(np.array(Xg_surr), psi)
    print("NEGATIVE CONTROLS (the apparatus is not rigged to always find load-bearing):")
    print(f"  (1) fGn Hurst [all-spectral]:  real={rH_real:.3f}  surrogate={rH_surr:.3f}  "
          f"(both high => GEOMETRIC)")
    print(f"  (2) arrow+variance [graded]:   real={rg_real:.3f}  surrogate={rg_surr:.3f}  "
          f"(surrogate INTERMEDIATE: recovers the variance half, loses the arrow half)")
    neg_ok = rH_real >= 0.60 and rH_surr >= 0.60 and rg_surr >= 0.25 and (rg_real - rg_surr) >= 0.15
    print(f"  ** negative controls behave (geometric + a discriminating intermediate) = {neg_ok} **")

    verdict = ("POSITIVE (weak notion): a DETERMINE-type latent (the arrow of time / non-equilibrium current) "
               "is load-bearing on the lossy ensemble shadow, vs the time-symmetric surrogate class -- the "
               "determine-side inversion of the H8 resist no-go. Content = the foils are blind (low-order, by "
               "time-reversal symmetry); apparatus not rigged (geometric + intermediate controls behave)."
               if lb and neg_ok else "CHECK (a gate failed)")
    print(f"\nH9 {mode}: load-bearing-determine={lb} AND negative-controls-valid={neg_ok} => {verdict}")


if __name__ == "__main__":
    main()
