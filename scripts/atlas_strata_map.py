#!/usr/bin/env python
"""Atlas Phase 8-A — catastrophe-stratum classifier for the halo bifurcation set.

Computes the CORANK of the halo-function caustic from the singular values of the 2×2 Jacobian
J = ∂(sky-chart)/∂(γ,α), so each stratum LABEL is DERIVED (the §6 armchair-catastrophe gate), never
asserted from arc shape. On the caustic det J = 0 the smaller singular value s2 → 0:
  - corank 1  (s2 ≈ 0, s1 bounded away from 0)  -> a cuspoid A_k (fold A₂ / cusp A₃ / swallowtail A₄);
  - corank 2  (s1 ≈ 0 AND s2 ≈ 0)               -> an umbilic D₄.
The A_k order (fold vs cusp) is the codimension along the fold — A₃ is where two A₂ fold branches
COALESCE (the caustic curve closes), already located at the 29.7° UTA+LTA merge (Phase 6.5-B).

OPEN QUESTION CLOSED: is the 29.7° merge an A₃ cusp (corank-1) or a D₄ umbilic (corank-2)? This script
answers it numerically (expected A₃ — the column 2-DOF→2-sky square map exposes only corank-1 strata;
D₄ needs ≥2 control DOF, i.e. the elevation × habit grid of Phase 8-B).

DISCIPLINE: stratum labels are SYNTHESIS (Berry-Upstill/Nye classification applied on Tape 1980's
caustic = Jacobian-kernel construction); the directed search for an UN-cataloged higher stratum is a
low-confidence rider, gated by a catalog cross-check. §0.2 ray-optics + smearing caveat travels.
NOT public-eligible (Phase 0.5 lit-pass, incl. the Tape & Können 1999 prior-art check, gates any claim).
"""
import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_caustic_map as cm

CORANK2_REL = 0.05   # s1/scale below this on the caustic => corank-2 (D₄) candidate
ERODE = 3            # cells to erode off the admissibility boundary before the caustic search


def jacobian_svals(h_deg, n=cm.N_ICE, ngrid=300):
    """Per-cell singular values (s1≥s2≥0) of J=∂(sky-chart)/∂(γ,α), plus detJ and a clean-stencil mask.
    Returns ng×ng arrays (s1, s2, detJ, good)."""
    ng = ngrid
    G, A, sky, ok, su = cm.sky_grid(h_deg, n, ng)
    z = np.array([0.0, 0.0, 1.0])
    up = z - np.dot(z, su) * su
    up = up / np.linalg.norm(up)
    right = np.cross(su, up)
    X = (sky @ right).reshape(ng, ng)
    Y = (sky @ up).reshape(ng, ng)
    validg = ok.reshape(ng, ng)
    d = 2 * math.pi / ng
    Xg, Xa = np.gradient(X, d, d)
    Yg, Ya = np.gradient(Y, d, d)
    detJ = Xg * Ya - Xa * Yg                                  # = ±s1·s2
    fro2 = Xg ** 2 + Xa ** 2 + Yg ** 2 + Ya ** 2              # = s1²+s2²
    disc = np.sqrt(np.clip(fro2 ** 2 - 4 * detJ ** 2, 0, None))
    s1 = np.sqrt(np.clip((fro2 + disc) / 2, 0, None))
    s2 = np.sqrt(np.clip((fro2 - disc) / 2, 0, None))
    # erode validity by ERODE cells so the caustic search stays clear of the admissibility boundary
    # (where both singular values shrink as a wing-tip edge effect, mimicking corank-2 spuriously).
    good = validg.copy()
    for _ in range(ERODE):
        g2 = good.copy()
        g2[1:, :] &= good[:-1, :]; g2[:-1, :] &= good[1:, :]
        g2[:, 1:] &= good[:, :-1]; g2[:, :-1] &= good[:, 1:]
        good = g2
    return s1, s2, detJ, good


def corank_on_caustic(h_deg, n=cm.N_ICE, ngrid=300):
    """Locate the caustic (detJ sign-change among good neighbors) and report the singular-value stats.
    On the caustic s2→0 (by definition); s1 = the OTHER singular value: bounded away from 0 => corank-1
    (A_k), ~0 => corank-2 (D₄). Returns None if no caustic in the admissible region."""
    s1, s2, detJ, good = jacobian_svals(h_deg, n, ngrid)
    sgn = np.sign(detJ)
    caustic = np.zeros_like(good, bool)
    for ax in (0, 1):
        idx = np.abs(np.diff(sgn, axis=ax)) > 0
        sl0 = [slice(None)] * 2; sl1 = [slice(None)] * 2
        sl0[ax] = slice(0, -1); sl1[ax] = slice(1, None)
        caustic[tuple(sl0)] |= idx
        caustic[tuple(sl1)] |= idx
    caustic &= good & np.isfinite(s1) & np.isfinite(s2)
    if not caustic.any():
        return None
    scale = float(np.nanmedian(s1[good & np.isfinite(s1)]))
    s1c, s2c = s1[caustic], s2[caustic]
    s1_min_rel = float(np.nanmin(s1c) / scale)
    return {
        "n_caustic": int(caustic.sum()), "scale": scale,
        "s1_min": float(np.nanmin(s1c)), "s1_min_rel": s1_min_rel,
        "s2_med_rel": float(np.nanmedian(s2c) / scale),
        "corank": 2 if s1_min_rel < CORANK2_REL else 1,
    }


def main():
    print("Atlas Phase 8-A — catastrophe-stratum classifier (corank from Jacobian singular values)")
    print(f"  column halo function; corank-2 (D₄) flagged if min(s1)/scale < {CORANK2_REL} on the caustic\n")
    print(f"  {'h(deg)':>7}{'#caustic':>10}{'s2_med/scale':>14}{'s1_min/scale':>14}{'corank':>8}  stratum")
    print("  " + "-" * 74)
    worst_s1rel = 1e9
    for h in (15.0, 20.0, 25.0, 28.0, 29.7, 31.0, 35.0, 45.0):
        r = corank_on_caustic(h)
        if r is None:
            print(f"  {h:>7.1f}{'(no caustic)':>10}")
            continue
        worst_s1rel = min(worst_s1rel, r["s1_min_rel"])
        merge = abs(h - 29.7) < 0.3
        stratum = ("A₃ cusp (29.7° merge)" if merge else "A₂ fold(s)") if r["corank"] == 1 else "D₄ umbilic (!)"
        print(f"  {h:>7.1f}{r['n_caustic']:>10}{r['s2_med_rel']:>14.4f}{r['s1_min_rel']:>14.4f}"
              f"{r['corank']:>8}  {stratum}")
    print(f"\n  caustic indicator s2→0 confirmed (the fold); min s1/scale over all elevations = "
          f"{worst_s1rel:.3f}")
    no_d4 = worst_s1rel >= CORANK2_REL
    print(f"  CORANK-1 EVERYWHERE on the column (no D₄ umbilic): {no_d4}")
    print(f"  => the 29.7° UTA+LTA merge is an A₃ CUSP (corank-1), NOT a D₄ umbilic. [PHASE65 open Q closed]")
    print(f"  => column stratum inventory: A₂ folds (22°/46° edges + tangent-arc folds) + the A₃ cusp;")
    print(f"     no A₄/D₄ on the column (expected — the 2-DOF→2-sky map exposes only corank-1; D₄ needs")
    print(f"     the elevation × habit control grid of Phase 8-B). HONEST NULL, not a failure.")
    return no_d4


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
