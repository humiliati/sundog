"""H8 v5 crux (careful, un-confounded) — is the obstacle SUBSTRATE-GENERAL? Slow-fast deterministic chaos
(logistic map; slow phase phi modulates r through the period-doubling/chaos regime). DECISIVE geometric/
non-geometric discriminator: own-R2(REAL window) vs own-R2(MATCHED-SPECTRUM phase-randomized surrogate).
GAP = the nonlinear-dynamical (non-geometric) contribution that a matched-spectrum foil cannot reproduce.
  geometric/substrate-general  <=> spectral own-R2 ~ real own-R2 (phi is in the linear/spectral structure)
  NON-GEOMETRIC escape         <=> real >= 0.70 AND spectral <= 0.30 (phi survives the nonlinear determinism)
Also dissect: histogram-only vs spectrum-only own-R2 (where does phi live?)."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold

W = 64           # window length (iterations)
r0, A = 3.66, 0.28
w_slow = 0.0     # phi ~ constant within a window (the slow drive is quasi-static per window)


def logistic_window(phi, burn, rng):
    """W iterations of the logistic map at r=r0+A*cos(phi) (the slow phase sets the nonlinear regime),
    after a burn-in from a jittered start."""
    r = r0 + A * np.cos(phi)
    x = 0.5 + 0.01 * rng.standard_normal()
    for _ in range(burn):
        x = r * x * (1 - x)
    out = np.empty(W)
    for i in range(W):
        x = r * x * (1 - x); out[i] = x
    return out


def phase_randomize(win, rng):
    """matched-spectrum surrogate: preserve the power spectrum, randomize phases (destroys nonlinear
    determinism + the amplitude distribution)."""
    f = np.fft.rfft(win - win.mean())
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(f))); ph[0] = 1
    s = np.fft.irfft(np.abs(f) * ph, n=W)
    return s + win.mean()


def feat_hist(win):
    return np.histogram(win, bins=16, range=(0, 1), density=True)[0]


def feat_spec(win):
    return np.abs(np.fft.rfft(win - win.mean()))


def own_r2(X, y):
    X = (X - X.mean(0)) / (X.std(0) + 1e-9); kf = KFold(4, shuffle=True, random_state=0)
    lin = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2").mean()
    mlp = cross_val_score(MLPRegressor(hidden_layer_sizes=(64,), max_iter=700, random_state=0),
                          X, y, cv=kf, scoring="r2").mean()
    return max(0.0, lin, mlp)


n = 240
rng = np.random.default_rng(3)
phi = rng.uniform(0.0, 1.8, n)                       # latent slow phase
NOISE = 0.02
Wr, Wspec, Whist, Wsurr = [], [], [], []
for i in range(n):
    g = np.random.default_rng(500 + i)
    win = logistic_window(phi[i], burn=300, rng=g) + NOISE * g.standard_normal(W)
    surr = phase_randomize(win, g)
    Wr.append(win); Wspec.append(feat_spec(win)); Whist.append(feat_hist(win)); Wsurr.append(surr)
Wr, Wspec, Whist, Wsurr = map(np.array, (Wr, Wspec, Whist, Wsurr))

r_real = own_r2(Wr, phi)                 # full real window (spectrum + histogram + return-map)
r_surr = own_r2(Wsurr, phi)             # matched-spectrum surrogate (linear structure only)
r_spec = own_r2(Wspec, phi)             # spectrum feature only
r_hist = own_r2(Whist, phi)             # amplitude histogram only (a nonlinear marginal)
print(f"phi own-R2:  REAL window={r_real:.3f}   MATCHED-SPECTRUM surrogate={r_surr:.3f}")
print(f"  dissection: spectrum-only={r_spec:.3f}   histogram-only={r_hist:.3f}")
gap = r_real - r_surr
print(f"  GAP (real - matched-spectrum) = {gap:.3f}")
if r_surr > 0.30:
    print("  => matched-spectrum recovers phi => phi is SPECTRAL/GEOMETRIC => obstacle SUBSTRATE-GENERAL (NULL)")
elif r_real >= 0.70 and r_surr <= 0.30:
    print("  => phi survives in the NONLINEAR determinism (real high, spectral low) => NON-GEOMETRIC => ESCAPE LEAD => full build")
else:
    print("  => weak/confounded: phi not cleanly recoverable")
