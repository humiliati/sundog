"""De-risk the H8 v2 substrate: does an oscillatory FitzHugh-Nagumo medium sustain a spiral with
(a) a clean topological charge (chirality), (b) charge sign that FLIPS under mirror (=> +/-q are exact
mirror images, first-order-identical), (c) an m=1 azimuthal phase that SHIFTS under spatial rotation
(the phase-resist carrier)? If yes, the v2 design is implementable on real RD fields."""
import sys
import numpy as np
from scipy import ndimage

GRID = 96


def lap(a):
    return (np.roll(a, 1, 1) + np.roll(a, -1, 1) + np.roll(a, 1, 2) + np.roll(a, -1, 2) - 4 * a)


def fhn_spiral(B, steps, chirality, rng, b1=0.5, c1=-0.8, dt=0.05):
    """Complex Ginzburg-Landau (the canonical oscillatory-RD amplitude equation): dA/dt = A +
    (1+i b1) lap(A) - (1+i c1)|A|^2 A, A = u + i w. Seeded with a charge-`chirality` vortex
    (A0 = tanh(r/r0) e^{i*chirality*theta}); CGL relaxes it to a real spiral whose core is a genuine
    phase singularity (u,w in quadrature => nonzero topological charge). Returns (u, w)."""
    g = np.linspace(-1, 1, GRID)
    X, Y = np.meshgrid(g, g)
    r = np.sqrt(X ** 2 + Y ** 2); th = np.arctan2(Y, X)
    A = np.zeros((B, GRID, GRID), np.complex64)
    for bi in range(B):
        off = 0.3 * rng.standard_normal()
        A[bi] = (np.tanh(r / 0.15) * np.exp(1j * (chirality * th + off))).astype(np.complex64)
    for _ in range(steps):
        A += dt * (A + (1 + 1j * b1) * lap(A) - (1 + 1j * c1) * np.abs(A) ** 2 * A)
    return np.real(A).astype(np.float32), np.imag(A).astype(np.float32)


def topo_charge(u, w):
    """Chirality = boundary winding number of arg(u+iw) around a loop enclosing the core (rotation- &
    global-phase-invariant). Sample arg on a circle at r~0.5, unwrap, total change / 2pi."""
    c = GRID // 2
    rr = 0.45 * c
    ang = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    xs = (c + rr * np.cos(ang)).astype(int); ys = (c + rr * np.sin(ang)).astype(int)
    phi = np.arctan2(w[ys, xs], u[ys, xs])
    return float(np.sum(np.diff(np.unwrap(np.concatenate([phi, phi[:1]])))) / (2 * np.pi))


def m1_phase(u):
    """Argument of the m=1 azimuthal Fourier coefficient at a mid radius — the rotational phase."""
    c = GRID // 2
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    th = np.arctan2(yy - c, xx - c)
    ring = (r > 0.2 * c) & (r < 0.6 * c)
    z = np.sum(u[ring] * np.exp(1j * th[ring]))
    return float(np.angle(z))


rng = np.random.default_rng(0)
up, wp = fhn_spiral(1, 2000, +1, rng)
um, wm = fhn_spiral(1, 2000, -1, rng)
amp = np.sqrt(up[0] ** 2 + wp[0] ** 2)
print(f"    |A| range [{amp.min():.2f},{amp.max():.2f}] mean {amp.mean():.2f}  (want ~1 away from core)")
Sp, Sm = topo_charge(up[0], wp[0]), topo_charge(um[0], wm[0])
print(f"(a) spiral formed: charge chirality+ = {Sp:+.2f}, chirality- = {Sm:+.2f}  (want ~ +1 / -1)")
print(f"(b) chirality sign: +q charge = {Sp:+.1f}, -q charge = {Sm:+.1f}  (want opposite signs)")

# mirror of a +q spiral must flip the charge and preserve first-order stats
upm = np.fliplr(up[0]); wpm = np.fliplr(wp[0])
Spm = topo_charge(upm, wpm)
ps_p = np.abs(np.fft.fft2(up[0] - up[0].mean()))
ps_m = np.abs(np.fft.fft2(upm - upm.mean()))
hist_p = np.histogram(up[0], 16, (-2, 2))[0]; hist_m = np.histogram(upm, 16, (-2, 2))[0]
print(f"(b') mirror(+q): charge {Sp:+.1f} -> {Spm:+.1f} (want sign flip); "
      f"power-spectrum identical={np.allclose(np.sort(ps_p.ravel()), np.sort(ps_m.ravel()))}; "
      f"histogram identical={np.array_equal(np.sort(hist_p), np.sort(hist_m))}")

# rotation must shift the m=1 phase by ~the rotation angle
base = up[0]
for deg in [0, 45, 90, 135]:
    rot = ndimage.rotate(base, deg, reshape=False, mode="grid-wrap", order=1)
    print(f"(c) rotate {deg:>3} deg -> m1 phase {np.degrees(m1_phase(rot)):+7.1f} deg "
          f"(want ~ {np.degrees(m1_phase(base)) + (-deg):+7.1f} mod 360)")
