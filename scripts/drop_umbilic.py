#!/usr/bin/env python
"""Step 2 of "referee the rainbow's umbilic" — a REAL spheroidal-raindrop forward chart, to test the Atlas's
now-calibrated corank-2 / D4 detector against a LITERATURE-established physical D4.

PHYSICS (attribution: Marston & Trinh, Nature 312:529 (1984); Nye, Nature 312:531 (1984); Berry & Upstill):
an OBLATE water drop (symmetry axis z vertical), illuminated by a horizontal parallel beam (+x), forms a
primary-rainbow caustic that becomes a HYPERBOLIC UMBILIC (D4+, "D+4") at the critical aspect ratio
D/H = 1.305 (D = equatorial diameter = 2a, H = polar diameter = 2c; so a/c = D/H, oblate => a>c). The
mechanism is ray-optic (skew rays joining the equatorial fold), so the D4 lives in the ray-caustic skeleton.

THE TEST: forward-trace the primary rainbow (enter -> ONE internal reflection -> exit) over the 2-D entry
aperture (y0,z0), chart the scattered direction, and run the chart-based corank-2 detector
(atlas_jet_classify.corank_from_chart). It MUST fire corank-2 (s1_min_rel < 0.05) near D/H = 1.305 and stay
corank-1 away from it. This is INDEPENDENT-REFEREE / detector-validation — NOT a new-physics claim; the
raindrop D4 is Marston/Nye's. NOT public-eligible.
"""
import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_caustic_map as cm
import atlas_jet_classify as jc

N_WATER = 1.333


def _normal(P, a, c):
    n = np.stack([P[:, 0] / a ** 2, P[:, 1] / a ** 2, P[:, 2] / c ** 2], axis=-1)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-30)


def _hit(P0, d, a, c, eps=1e-9):
    """First forward intersection (t>eps) of the ray P0+t·d with the spheroid x²/a²+y²/a²+z²/c²=1."""
    inv = np.array([1.0 / a ** 2, 1.0 / a ** 2, 1.0 / c ** 2])
    A = np.sum(d * d * inv, axis=-1)
    B = 2.0 * np.sum(P0 * d * inv, axis=-1)
    C = np.sum(P0 * P0 * inv, axis=-1) - 1.0
    disc = B * B - 4 * A * C
    ok = disc >= 0
    sq = np.sqrt(np.clip(disc, 0, None))
    t1 = (-B - sq) / (2 * A)
    t2 = (-B + sq) / (2 * A)
    t = np.where(t1 > eps, t1, t2)               # smaller positive root, else the other
    return P0 + t[:, None] * d, ok & (t > eps)


def rainbow_chart(a, c, ng=320, ap=0.99, b_lo=0.62, b_hi=0.965):
    """Trace the primary rainbow of an oblate spheroid (semi-axes a,a,c) for a +x beam over the (y0,z0)
    entry aperture. Returns (X, Y, good, dy, dz): X,Y = transverse scattered direction (d3_y, d3_z). `good`
    is restricted to the RAINBOW ANNULUS (normalized impact-parameter radius b∈[b_lo,b_hi]) to EXCLUDE the
    axial backscattering-glory focus (a corank-2 at every aspect ratio) and the grazing silhouette edge —
    so the detector sees the rainbow caustic + its umbilic, not those confounds. Sphere rainbow ray b≈0.86."""
    y = np.linspace(-ap * a, ap * a, ng)
    z = np.linspace(-ap * c, ap * c, ng)
    Y0, Z0 = np.meshgrid(y, z, indexing="ij")
    Y0f, Z0f = Y0.ravel(), Z0.ravel()
    M = Y0f.shape[0]
    d0 = np.zeros((M, 3)); d0[:, 0] = 1.0                       # beam along +x
    P0 = np.stack([np.full(M, -3.0 * a), Y0f, Z0f], axis=-1)   # start far on -x
    P1, v1 = _hit(P0, d0, a, c)                                # entry (front surface)
    N1 = _normal(P1, a, c)
    d1, r1 = cm._refract_vec(d0, N1, 1.0 / N_WATER)            # air -> water
    P2, v2 = _hit(P1 + 1e-7 * d1, d1, a, c)                    # first internal surface
    N2 = _normal(P2, a, c)
    d2 = d1 - 2.0 * np.sum(d1 * N2, axis=-1, keepdims=True) * N2   # internal reflection (primary rainbow)
    P3, v3 = _hit(P2 + 1e-7 * d2, d2, a, c)                    # exit surface
    N3 = _normal(P3, a, c)
    d3, r3 = cm._refract_vec(d2, N3, N_WATER)                  # water -> air (TIR -> r3 False)
    b2 = Y0f ** 2 / a ** 2 + Z0f ** 2 / c ** 2                 # normalized elliptical impact-parameter²
    annulus = (b2 > b_lo ** 2) & (b2 < b_hi ** 2)             # the rainbow ring (no glory core, no edge)
    good = (v1 & r1 & v2 & v3 & r3 & annulus).reshape(ng, ng)
    X = d3[:, 1].reshape(ng, ng)                               # transverse scattered direction
    Y = d3[:, 2].reshape(ng, ng)
    dy = 2 * ap * a / (ng - 1); dz = 2 * ap * c / (ng - 1)
    return X, Y, good, dy, dz


def corank_at(DH, ng=320):
    """corank-2 / D4 detector on the oblate-drop primary-rainbow chart, for aspect ratio D/H = a/c."""
    a, c = DH, 1.0                                             # a/c = D/H; equatorial a, polar c
    X, Y, good, dy, dz = rainbow_chart(a, c, ng=ng)
    Xg = np.where(good, X, 0.0); Yg = np.where(good, Y, 0.0)
    return jc.corank_from_chart(Xg, Yg, dy, dz, good=good, edge=4)


def main():
    print("Step 2 — refereeing a LITERATURE D4: the oblate-raindrop primary-rainbow hyperbolic umbilic")
    print("  (Marston & Trinh 1984 / Nye 1984: D4+ at D/H = 1.305). corank-2 iff s1_min_rel < 0.05.\n")
    print(f"  {'D/H':>6}{'s1_min_rel':>14}{'corank':>8}   note")
    print("  " + "-" * 56)
    best = (1e9, None)
    for DH in (1.00, 1.10, 1.20, 1.27, 1.30, 1.305, 1.31, 1.34, 1.40, 1.50):
        r = corank_at(DH)
        if r is None:
            print(f"  {DH:>6.3f}{'(no caustic)':>14}")
            continue
        if r["s1_min_rel"] < best[0]:
            best = (r["s1_min_rel"], DH)
        flag = "<== D4 FIRES (corank-2)" if r["corank"] == 2 else ("sphere (D/H=1)" if DH == 1.0 else "")
        print(f"  {DH:>6.3f}{r['s1_min_rel']:>14.4f}{r['corank']:>8}   {flag}")
    print(f"\n  detector minimum at D/H = {best[1]} (s1_min_rel = {best[0]:.4f})"
          f" vs the Marston/Nye literature umbilic D/H = 1.305")
    return 0


if __name__ == "__main__":
    sys.exit(main())
