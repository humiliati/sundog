"""H8 v4 GO/NO-GO (the decisive SUP2 check). Latent Delta-theta = relative phase of two vortices.
Is Delta-theta recoverable from SO(2)-invariant registered features of (A) an INTERACTING evolved CGL pair
vs (B) SUP2 = the SAME two-vortex ansatz NOT evolved (no PDE coupling)? If SUP2 own-R2 is HIGH (~ the
interacting pair), Delta-theta is GEOMETRIC -> KILL-GEOMETRIC -> v4 NULL (obstacle confirmed 4th time). Only
if SUP2 FAILS while the interacting pair PASSES is Delta-theta load-bearing -> proceed to the full build."""
import sys
import numpy as np
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold

G = 72
b1, c1, dt, r0 = 0.0, -1.4, 0.05, 0.12
SEP = 14            # core separation (pixels)


def lap(a):
    return np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4 * a


def pair_ic(dtheta, jitter_rng):
    """two same-charge vortices at +/-SEP/2 on x-axis, relative phase Delta-theta; small core jitter."""
    g = np.arange(G) - G / 2
    X, Y = np.meshgrid(g, g)
    cx = SEP / 2 + 0.5 * jitter_rng.standard_normal()
    r1 = np.hypot(X + cx, Y); r2 = np.hypot(X - cx, Y)
    th1 = np.arctan2(Y, X + cx); th2 = np.arctan2(Y, X - cx)
    # SUPERPOSITION (sum): the relative phase Delta-theta sets the INTERFERENCE (in a product it cancels)
    A = (np.tanh(r1 / (r0 * G)) * np.exp(1j * (th1 + dtheta / 2)) +
         np.tanh(r2 / (r0 * G)) * np.exp(1j * (th2 - dtheta / 2)))
    return A.astype(np.complex128)


def evolve(A, steps):
    for _ in range(steps):
        A = A + dt * (A + (1 + 1j * b1) * lap(A) - (1 + 1j * c1) * np.abs(A) ** 2 * A)
    return A


def register(A):
    """SO(2)-invariant: find the 2 cores (|A| minima), rotate so they are horizontal, center. Returns the
    downsampled registered field (real+imag) -- depends on Delta-theta, blind to the global rotation."""
    amp = np.abs(A)
    sm = ndimage.gaussian_filter(amp, 2)
    # two lowest minima, far apart
    idx = np.argsort(sm.ravel())[:40]
    pts = np.array(np.unravel_index(idx, sm.shape)).T.astype(float)  # (y,x)
    c0 = pts[0]
    far = pts[np.argmax(((pts - c0) ** 2).sum(1))]
    mid = (c0 + far) / 2
    ang = np.degrees(np.arctan2(far[0] - c0[0], far[1] - c0[1]))
    Ar = ndimage.shift(A.real, (G / 2 - mid[0], G / 2 - mid[1]), order=1)
    Ai = ndimage.shift(A.imag, (G / 2 - mid[0], G / 2 - mid[1]), order=1)
    Ar = ndimage.rotate(Ar, ang, reshape=False, order=1); Ai = ndimage.rotate(Ai, ang, reshape=False, order=1)
    D = 12
    fr = Ar[:(G // D) * D, :(G // D) * D].reshape(D, G // D, D, G // D).mean((1, 3))
    fi = Ai[:(G // D) * D, :(G // D) * D].reshape(D, G // D, D, G // D).mean((1, 3))
    return np.concatenate([fr.ravel(), fi.ravel()])


def own_r2(X, y):
    X = (X - X.mean(0)) / (X.std(0) + 1e-9); kf = KFold(4, shuffle=True, random_state=0)
    lin = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2").mean()
    mlp = cross_val_score(MLPRegressor(hidden_layer_sizes=(64,), max_iter=600, random_state=0), X, y, cv=kf, scoring="r2").mean()
    return max(0.0, lin, mlp)


n = 120
rng = np.random.default_rng(0)
dth = rng.uniform(0.0, 2.0, n)                     # Delta-theta in [0,2] rad (bounded, no wrap)
Xint, Xsup = [], []
for i in range(n):
    jr = np.random.default_rng(100 + i)
    A0 = pair_ic(dth[i], jr)
    Xsup.append(register(A0) + 0.03 * np.random.default_rng(900 + i).standard_normal(2 * 144))      # SUP2 (un-evolved)
    Xint.append(register(evolve(A0.copy(), 1500)) + 0.03 * np.random.default_rng(900 + i).standard_normal(2 * 144))  # interacting
Xint, Xsup = np.array(Xint), np.array(Xsup)
r_int, r_sup = own_r2(Xint, dth), own_r2(Xsup, dth)
print(f"Delta-theta own-R2:  INTERACTING pair = {r_int:.3f}   SUP2 (un-evolved) = {r_sup:.3f}")
print(f"  => {'SUP2 recovers Delta-theta => GEOMETRIC => v4 NULL (obstacle confirmed)' if r_sup > 0.30 else 'SUP2 FAILS while pair recovers => LOAD-BEARING candidate => GO to full build' if r_int > 0.70 else 'both weak: registration/regime issue'}")
