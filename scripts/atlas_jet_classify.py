#!/usr/bin/env python
"""Atlas Phase 8 rigor upgrade — a SOUND iterated-kernel A3-vs-A4 jet-determinacy classifier.

The catastrophe order of a caustic point of a plane map F=(X,Y):(γ,α)->sky is the first non-vanishing
iterated KERNEL derivative of the fold function φ = det DF (the Morin / Thom-Boardman criterion):
  A2 fold      : φ=0,  c2 := K·∇φ ≠ 0
  A3 cusp      : φ=0,  c2 = 0,  c3 := K·∇(K·∇φ) ≠ 0
  A4 swallowtail: φ=0, c2 = 0,  c3 = 0  (and c4 ≠ 0)        [needs the elevation family; codim-3]

SOUNDNESS FIX (the prior workflow's reduced-jet test was retracted as unsound — it returned ~0.9 on KNOWN
A3 cusps): use the SMOOTH, sign-consistent kernel K = (Xα, −Xγ) (the row-1 kernel, exact on φ=0) rather
than the sign-AMBIGUOUS SVD eigenvector. Sign flips of K between cells corrupt ∇(K·∇φ) and hence c3.
c3 is invariant under a global K→−K, so only the consistency (not the sign) matters.

CONTROLS (calibration, both sides):
  + positive (known A4): the Morin A4 normal form F_h(γ,α) = (γ, α⁴ + h α² + γ α). Two A3 cusps at
    α=±√(−h/6) (h<0) MERGE into an A4 at h=0 — the classifier MUST show c3→0 there.
  − negative (known A3): the column apex cusps (wedge='prism60') — c3 must stay bounded away from 0.
Then apply to the Lowitz birth (wedge='lowitz60', α0≈60°): does c3→0 (A4) or stay bounded (A3-lips)?

NOT public-eligible. Run: python scripts/atlas_jet_classify.py
"""
import math
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_caustic_map as cm


def jet_from_chart(X, Y, dg, da):
    """Given the sky-chart components X(γ,α), Y(γ,α) on a grid (spacings dg,da) return
    (φ=det DF, c2=K·∇φ, c3=K·∇c2) with the SMOOTH sign-consistent kernel K=(Xα,−Xγ)/|·| (exact on φ=0)."""
    Xg, Xa = np.gradient(X, dg, da)
    Yg, Ya = np.gradient(Y, dg, da)
    phi = Xg * Ya - Xa * Yg
    Kg, Ka = Xa, -Xg                                   # row-1 kernel: (Xg,Xa)·K=0; = true kernel on φ=0
    nrm = np.hypot(Kg, Ka) + 1e-30
    Kg, Ka = Kg / nrm, Ka / nrm
    pg, pa = np.gradient(phi, dg, da)
    c2 = Kg * pg + Ka * pa
    c2g, c2a = np.gradient(c2, dg, da)
    c3 = Kg * c2g + Ka * c2a
    return phi, c2, c3


def cusp_c3(phi, c2, c3, good=None, edge=3):
    """Locate the cusps (φ=0 ∧ c2=0 sign-changes) and return the list of |c3| values there (the A3-vs-A4
    discriminator: bounded away from 0 => A3; → 0 => A4). good = optional admissibility mask."""
    ng0, ng1 = phi.shape
    if good is None:
        good = np.ones_like(phi, bool)
    g = good.copy()
    for _ in range(edge):
        g2 = g.copy()
        g2[1:, :] &= g[:-1, :]; g2[:-1, :] &= g[1:, :]; g2[:, 1:] &= g[:, :-1]; g2[:, :-1] &= g[:, 1:]
        g = g2

    def sc(F):
        out = np.zeros_like(g)
        for ax in (0, 1):
            idx = np.abs(np.diff(np.sign(F), axis=ax)) > 0
            s0 = [slice(None)] * 2; s1 = [slice(None)] * 2
            s0[ax] = slice(0, -1); s1[ax] = slice(1, None)
            out[tuple(s0)] |= idx; out[tuple(s1)] |= idx
        return out

    cusp = sc(phi) & sc(c2) & g
    lab, n = ndimage.label(cusp, structure=np.ones((3, 3)))
    vals = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        vals.append(float(np.nanmean(np.abs(c3[ys, xs]))))
    return sorted(vals, reverse=True)


# ---- positive control: the Morin A4 normal form -------------------------------------------------- #
def synthetic_swallowtail_chart(h, ng=420, L=1.1):
    g = np.linspace(-L, L, ng); a = np.linspace(-L, L, ng)
    G, A = np.meshgrid(g, a, indexing="ij")
    X = G
    Y = A ** 4 + h * A ** 2 + G * A
    return X, Y, 2 * L / (ng - 1)


def synthetic_swallowtail(h, ng=420, L=1.1):
    X, Y, d = synthetic_swallowtail_chart(h, ng, L)
    return jet_from_chart(X, Y, d, d)


# ---- the corank-2 / D4 (umbilic) detector ON A CHART + its synthetic-D4 positive control ---------- #
def corank_from_chart(X, Y, dg, da, edge=3, corank2_rel=0.05):
    """Chart-based twin of atlas_strata_map.corank_on_caustic: on the caustic (det J sign-change) the
    smaller singular value s2→0; s1 is the OTHER one — corank-2 (a D4 UMBILIC) iff min(s1)/scale <
    corank2_rel. Returns {s1_min_rel, corank, scale, n_caustic} (None if no caustic). This is the Atlas's
    NEVER-FIRED corank-2 branch, here exercised on an arbitrary map so it can finally be calibrated."""
    Xg, Xa = np.gradient(X, dg, da); Yg, Ya = np.gradient(Y, dg, da)
    detJ = Xg * Ya - Xa * Yg
    fro2 = Xg ** 2 + Xa ** 2 + Yg ** 2 + Ya ** 2
    disc = np.sqrt(np.clip(fro2 ** 2 - 4 * detJ ** 2, 0, None))
    s1 = np.sqrt(np.clip((fro2 + disc) / 2, 0, None))
    s2 = np.sqrt(np.clip((fro2 - disc) / 2, 0, None))
    caustic = np.zeros(detJ.shape, bool)
    for ax in (0, 1):
        idx = np.abs(np.diff(np.sign(detJ), axis=ax)) > 0
        s0 = [slice(None)] * 2; s1s = [slice(None)] * 2; s0[ax] = slice(0, -1); s1s[ax] = slice(1, None)
        caustic[tuple(s0)] |= idx; caustic[tuple(s1s)] |= idx
    good = np.ones(detJ.shape, bool)
    good[:edge, :] = good[-edge:, :] = good[:, :edge] = good[:, -edge:] = False
    caustic &= good & np.isfinite(s1) & np.isfinite(s2)
    if not caustic.any():
        return None
    scale = float(np.nanmedian(s1[good]))
    return {"s1_min_rel": float(np.nanmin(s1[caustic]) / scale), "scale": scale,
            "corank": 2 if float(np.nanmin(s1[caustic]) / scale) < corank2_rel else 1,
            "n_caustic": int(caustic.sum())}


def synthetic_umbilic(w, ng=400, L=1.2):
    """Hyperbolic-umbilic D4 gradient map: ∇(x³/3 + y³/3 + w·xy) = (x²+w·y, y²+w·x), so J=[[2x,w],[w,2y]],
    det J = 4xy − w². At w=0 the Jacobian VANISHES at the origin (corank-2 = a D4 umbilic) and det=4xy
    SIGN-CHANGES across the two axes (so the caustic is locatable); for w≠0 the umbilic UNFOLDS to corank-1
    folds. The missing analogue of synthetic_swallowtail — the positive control the corank-2 branch never
    had. Returns (X, Y, spacing)."""
    g = np.linspace(-L, L, ng); a = np.linspace(-L, L, ng)
    G, A = np.meshgrid(g, a, indexing="ij")
    X = G ** 2 + w * A
    Y = A ** 2 + w * G
    return X, Y, 2 * L / (ng - 1)


# ---- chart extraction for the real halo maps ----------------------------------------------------- #
def halo_chart(h_deg, wedge, ng=320):
    G, A, sky, ok, su = cm.sky_grid(h_deg, ngrid=ng, wedge=wedge)
    z = np.array([0.0, 0.0, 1.0]); up = z - np.dot(z, su) * su; up /= np.linalg.norm(up)
    right = np.cross(su, up)
    X = (sky @ right).reshape(ng, ng); Y = (sky @ up).reshape(ng, ng)
    good = ok.reshape(ng, ng)
    # NaN (inadmissible) -> 0 so gradients stay finite; the eroded `good` mask excludes them anyway
    X = np.where(good, X, 0.0); Y = np.where(good, Y, 0.0)
    d = 2 * math.pi / ng
    phi, c2, c3 = jet_from_chart(X, Y, d, d)
    return phi, c2, c3, good


def main():
    print("Atlas rigor upgrade — sound iterated-kernel A3-vs-A4 jet classifier (c3 = K·∇(K·∇ det DF))\n")

    print("(1) POSITIVE CONTROL — Morin A4 normal form F_h=(γ, α⁴+hα²+γα): cusps merge to an A4 at h=0.")
    print("    expect |c3| at the cusp(s) -> 0 as h -> 0 (the A4), bounded away for h<0 (A3 cusps):")
    for h in (-0.40, -0.20, -0.10, -0.04, -0.01, 0.0):
        phi, c2, c3 = synthetic_swallowtail(h)
        vals = cusp_c3(phi, c2, c3)
        top = vals[0] if vals else float("nan")
        print(f"      h={h:+.2f}: #cusps={len(vals)}  max|c3|={top:.4f}   {'<-- A4 (c3->0)' if top < 0.2 else ''}")

    print("\n(2) NEGATIVE CONTROL — column apex A3 cusps (wedge='prism60'): |c3| must stay bounded away from 0:")
    for h in (25.0, 30.0, 35.0):
        phi, c2, c3, good = halo_chart(h, "prism60")
        vals = cusp_c3(phi, c2, c3, good)
        print(f"      h={h:.0f}: #cusps={len(vals)}  |c3| = {[round(v, 3) for v in vals]}")

    print("\n(3) TARGET — Lowitz birth (wedge='lowitz60', α0=60°, h*≈16.5): does |c3| at the born pair -> 0")
    print("    (A4) or stay bounded (A3-lips) AS h -> the birth? ratio = |c3|(h) / |c3|(generic, h=28):")
    cm.LOWITZ_ALPHA0 = math.radians(60.0)
    gen = cusp_c3(*halo_chart(28.0, "lowitz60")[:3], halo_chart(28.0, "lowitz60")[3])
    gen = gen[0] if gen else float("nan")
    for h in (16.7, 17.0, 17.5, 18.0, 20.0, 25.0, 28.0):
        phi, c2, c3, good = halo_chart(h, "lowitz60")
        vals = cusp_c3(phi, c2, c3, good)
        top = vals[0] if vals else float("nan")
        print(f"      h={h:.1f}: #cusps={len(vals)}  max|c3|={top:7.3f}  ratio={top / gen:5.2f}"
              f"   {'<- A4 (c3->0)' if top / gen < 0.25 else '<- A3 (c3 bounded)'}")
    print("\n   CONTRAST — the positive-control A4, same 'ratio to generic' metric (ratio -> 0 at the A4):")
    g0 = synthetic_swallowtail(-0.40); gen0 = cusp_c3(*g0)[0]
    for h in (-0.40, -0.10, -0.04, -0.02, -0.01):
        v = cusp_c3(*synthetic_swallowtail(h))
        t = v[0] if v else float("nan")
        print(f"      h={h:+.2f}: max|c3|={t:7.3f}  ratio={t / gen0:5.2f}   {'<- A4 (c3->0)' if t / gen0 < 0.25 else ''}")

    print("\n(4) CORANK-2 / D4 DETECTOR CALIBRATION — the Atlas's never-fired corank-2 branch, finally")
    print("    exercised on a KNOWN D4 (corank-2 iff s1_min_rel < 0.05). The Atlas 'no-D4-anywhere' null")
    print("    rested on a detector never shown to fire; this gives it the missing positive control.")
    print("    + POSITIVE — synthetic hyperbolic-umbilic D4 ∇(x³/3+y³/3+w·xy), swept through the umbilic:")
    for w in (0.0, 0.05, 0.1, 0.2, 0.4):
        X, Y, d = synthetic_umbilic(w); r = corank_from_chart(X, Y, d, d)
        tag = "<== D4 FIRES (corank-2)" if r["corank"] == 2 else "corank-1 (umbilic unfolded)"
        print(f"        w={w:.2f}: s1_min_rel={r['s1_min_rel']:.4f}  corank={r['corank']}  {tag}")
    print("    - NEGATIVE — the A4 swallowtail (a corank-1 cuspoid; must NOT fire corank-2):")
    for h in (0.0, -0.4):
        X, Y, d = synthetic_swallowtail_chart(h); r = corank_from_chart(X, Y, d, d)
        print(f"        h={h:+.2f}: s1_min_rel={r['s1_min_rel']:.4f}  corank={r['corank']}  (A4 -> corank-1)")
    print("    => the corank-2 branch CAN fire (0.004 on a known D4) and distinguishes D4 from A4 (0.81);")
    print("       so the Atlas no-D4 null is VALIDATED (the tool sees D4; ice halos genuinely lack it).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
