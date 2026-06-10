#!/usr/bin/env python
"""Atlas Phase 6.5-B — the component-A cusp: the 29 deg UTA+LTA -> circumscribed-halo merge.

Implements Tape's halo-function caustic computation (Tape 1980, JOSA 70:1175) for singly-oriented
HORIZONTAL COLUMN crystals (2 orientation DOF), so the ~29 deg tangent-arc MERGE elevation falls OUT
as a DERIVED output (replacing the hardcoded TANGENT_ARC_CIRCUMSCRIBED_H=29 in parhelion-geometry.mjs
-- the §6 armchair-catastrophe gate). The merge is an A3-class caustic metamorphosis: two fold caustics
coalescing as the elevation control varies (Phase 8-B sharpens this — the point-cusps are the apexes).

Model (de-risk-confirmed minimal sufficient):
  - State: (gamma, alpha) on the 2-torus. gamma = c-axis azimuth about vertical; alpha = roll about
    the c-axis. "c-axis horizontal" removes the tilt DOF -> a square 2-DOF->2-sky-DOF map.
  - Forward map: 3-D vector Snell ray trace through the 60deg prism wedge (faces 120deg apart),
    n=1.31. sky = apparent source direction (-outgoing). NO Monte-Carlo, NO 3rd DOF.
  - Caustic = where the (gamma,alpha)->sky map folds (ray density diverges). The tangent arc is this
    caustic; the merge is where it closes into a loop around the sun (the A3-class metamorphosis).

SCOPE (atlas §0.2): ray-optics skeleton; the real arc is the smoothed image (sun 0.5deg disk + tilt +
dispersion). Physics fixed by Snell + Tape geometry; nothing tuned. NOT public-eligible.
"""
import math
import sys
import numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 console robustness
except Exception:
    pass

N_ICE = 1.31
WEDGE_DEG = 60.0          # 22deg-halo prism wedge: two prism faces 120deg apart -> 60deg apex
FACE_SEP = math.radians(180.0 - WEDGE_DEG)   # angle between the two outward face normals (120deg)
LOWITZ_ALPHA0 = math.radians(60.0)           # fixed reference roll for the Lowitz (gamma,phi) manifold
PYR_X = math.radians(61.99)                  # ice {10-11} pyramid-face normal angle from the c-axis
PYR_DPHI = math.radians(120.0)               # pyramid-facet azimuthal offset (120deg -> 23.8deg; 180deg -> 9deg)


def refract(d, n, eta):
    """3-D vector Snell refraction. d=incident propagation unit; n=unit normal; eta=n_in/n_out.
    Orients n against the ray. Returns transmitted unit vector, or None on total internal reflection."""
    d = d / np.linalg.norm(d)
    n = n / np.linalg.norm(n)
    if np.dot(d, n) > 0:          # ensure n points back toward the incident side
        n = -n
    cos_i = -np.dot(d, n)         # > 0
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    if k < 0.0:
        return None               # TIR
    t = eta * d + (eta * cos_i - math.sqrt(k)) * n
    return t / np.linalg.norm(t)


def _column_normals(gamma, alpha):
    """Outward normals of the two refracting prism side faces of a horizontal column.
    c-axis at azimuth gamma (horizontal); face normals lie in the plane perpendicular to c,
    parameterized by roll alpha around c."""
    c = np.array([math.cos(gamma), math.sin(gamma), 0.0])
    u = np.array([math.sin(gamma), -math.cos(gamma), 0.0])   # horizontal, perpendicular to c
    w = np.array([0.0, 0.0, 1.0])                            # vertical, perpendicular to c
    n1 = math.cos(alpha) * u + math.sin(alpha) * w
    n2 = math.cos(alpha + FACE_SEP) * u + math.sin(alpha + FACE_SEP) * w
    return n1, n2


def sun_dir(h_deg, az_deg=0.0):
    """Unit vector toward the sun at elevation h, azimuth az."""
    h, a = math.radians(h_deg), math.radians(az_deg)
    return np.array([math.cos(h) * math.cos(a), math.cos(h) * math.sin(a), math.sin(h)])


def halo_ray(h_deg, gamma, alpha, n=N_ICE):
    """Apparent sky direction (unit vector) of the column-refracted ray for orientation (gamma,alpha)
    at sun elevation h, via the 60deg-wedge path. None if the path is inadmissible (wrong face side
    or TIR). The apparent source = -outgoing propagation."""
    s = sun_dir(h_deg)
    d0 = -s                                # sunlight propagation direction
    n1, n2 = _column_normals(gamma, alpha)
    # entry face must face the incoming ray (ray enters through it)
    if np.dot(d0, n1) >= 0:
        return None
    d1 = refract(d0, n1, 1.0 / n)          # air -> ice
    if d1 is None:
        return None
    # exit face: ray must be leaving through it (propagating outward)
    if np.dot(d1, n2) <= 0:
        return None
    d2 = refract(d1, n2, n)                # ice -> air
    if d2 is None:
        return None                        # TIR at exit
    return -d2 / np.linalg.norm(d2)        # apparent source direction


def _refract_vec(d, n, eta):
    """Vectorized 3-D Snell refraction. d,n: (M,3) unit; eta scalar. Returns (t (M,3), valid (M,))."""
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    dn = np.sum(d * n, axis=-1)
    nf = np.where((dn > 0)[:, None], -n, n)                  # orient against the ray
    cos_i = -np.sum(d * nf, axis=-1)                         # >= 0
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    valid = k >= 0.0
    t = eta * d + (eta * cos_i - np.sqrt(np.clip(k, 0, None)))[:, None] * nf
    t = t / np.linalg.norm(t, axis=-1, keepdims=True)
    return t, valid


def sky_grid(h_deg, n=N_ICE, ngrid=300, wedge="prism60"):
    """Vectorized column halo function over the (gamma,alpha) torus. Returns G,A (M,) orientation grids;
    sky (M,3) apparent-source unit vectors (NaN where inadmissible); valid (M,) bool; su (3,) sun dir.
    M = ngrid*ngrid. (Shared kernel for the Phase-8 Jacobian/strata/cusp classifiers.)

    wedge selects the face-pair (both keep the 2 orientation DOF γ,α of the horizontal column):
      'prism60' — two prism SIDE faces 120° apart (60° refracting wedge) -> the 22° / tangent-arc family.
      'basal90' — a prism side face + a BASAL (end) face (normal = c, ⟂ the side faces -> 90° wedge) ->
                  the 46° / supralateral / infralateral family (Phase 8-B's 2-DOF swallowtail extension).
      'wegener' — entry prism side face -> internal REFLECTION off a basal end face (TIR) -> exit the
                  other prism side face (60° wedge + one basal bounce) -> the Wegener anthelic arc.
      'lowitz60' — the LOWITZ manifold: the column frame ROTATED by φ about the horizontal a-axis u
                  (so the c-axis tilts out of horizontal, c_z=sin φ — a geometrically DISTINCT 2-surface
                  in SO(3), meeting the column torus only at φ=0). The 2 DOF are (γ, φ) with the roll
                  FIXED at LOWITZ_ALPHA0; the SAME 60° prism side-face pair. -> the Lowitz arcs.
      'pyrcol' — PYRAMIDAL-CAPPED horizontal column: same (γ,α) column manifold, but the exit face is a
                  PYRAMID {10-11} cap face (normal at PYR_X=62° from c, facet offset PYR_DPHI). The odd
                  Galle wedge (dphi=120deg -> 23.8deg / dphi=180deg -> 9deg) -> the oriented odd-radius arcs."""
    su = sun_dir(h_deg)
    d0 = -su
    g = np.linspace(0.0, 2 * math.pi, ngrid, endpoint=False)
    a = np.linspace(0.0, 2 * math.pi, ngrid, endpoint=False)
    G, A = np.meshgrid(g, a, indexing="ij")
    G, A = G.ravel(), A.ravel()
    cg, sg = np.cos(G), np.sin(G)
    reflect_basal = False
    if wedge == "lowitz60":
        # A = φ (Lowitz rotation about u); roll fixed at a0. Frame: u=(sg,-cg,0) [horizontal a-axis],
        # w' = w·cosφ − c·sinφ = (−cg·sinφ, −sg·sinφ, cosφ). n_k = cos(a0+kΔ)·u + sin(a0+kΔ)·w'.
        phi = A
        sphi, cphi = np.sin(phi), np.cos(phi)
        wpx, wpy, wpz = -cg * sphi, -sg * sphi, cphi
        c0, s0 = math.cos(LOWITZ_ALPHA0), math.sin(LOWITZ_ALPHA0)
        c0b, s0b = math.cos(LOWITZ_ALPHA0 + FACE_SEP), math.sin(LOWITZ_ALPHA0 + FACE_SEP)
        n1 = np.stack([c0 * sg + s0 * wpx, -c0 * cg + s0 * wpy, s0 * wpz], axis=-1)
        n2 = np.stack([c0b * sg + s0b * wpx, -c0b * cg + s0b * wpy, s0b * wpz], axis=-1)
    else:
        cA, sA = np.cos(A), np.sin(A)
        n1 = np.stack([cA * sg, -cA * cg, sA], axis=-1)             # entry: prism side face (roll α)
        cA2, sA2 = np.cos(A + FACE_SEP), np.sin(A + FACE_SEP)
        n2_side = np.stack([cA2 * sg, -cA2 * cg, sA2], axis=-1)     # the OTHER prism side face
        nb = np.stack([cg, sg, np.zeros_like(cg)], axis=-1)        # basal end face, normal = c
        if wedge == "prism60":
            n2 = n2_side                                           # 60° wedge: two side faces
        elif wedge == "basal90":
            n2 = nb                                                # 90° wedge: side + basal
        elif wedge == "wegener":
            n2 = n2_side; reflect_basal = True                    # side -> basal reflection -> side
        elif wedge == "pyrcol":
            # exit = pyramid {10-11} cap face: n2 = cos(x)·c + sin(x)·(cos(α+dφ)·u + sin(α+dφ)·w),
            # c=(cg,sg,0), u=(sg,-cg,0), w=(0,0,1). |n2|=1 (c ⟂ inplane). Odd Galle wedge (PYR_X, PYR_DPHI).
            cx, sx = math.cos(PYR_X), math.sin(PYR_X)
            cAd, sAd = np.cos(A + PYR_DPHI), np.sin(A + PYR_DPHI)
            n2 = np.stack([cx * cg + sx * cAd * sg, cx * sg - sx * cAd * cg, sx * sAd], axis=-1)
        else:
            raise ValueError(f"unknown wedge {wedge!r}")
    M = G.shape[0]
    d0b = np.broadcast_to(d0, (M, 3))
    entry_ok = (n1 @ d0) < 0
    d1, v1 = _refract_vec(d0b, n1, 1.0 / n)
    if reflect_basal:
        dn = np.sum(d1 * nb, axis=-1, keepdims=True)
        d_int = d1 - 2.0 * dn * nb                                  # internal reflection off the basal plane
        tir_ok = np.abs(dn[:, 0]) < math.cos(math.asin(1.0 / n))   # incidence > critical angle => TIR
    else:
        d_int = d1
        tir_ok = np.ones(M, dtype=bool)
    exit_ok = np.sum(d_int * n2, axis=-1) > 0
    d2, v2 = _refract_vec(d_int, n2, n)
    ok = entry_ok & v1 & tir_ok & exit_ok & v2
    sky = -d2
    sky[~ok] = np.nan
    return G, A, sky, ok, su


def caustic_coverage(h_deg, n=N_ICE, ngrid=360, dmax=46.0):
    """Sweep (gamma,alpha) vectorized; return per-psi minimum deviation delta_min(psi) of the lit
    region near the sun. The tangent-arc caustic is delta_min(psi); a finite value means the arc
    reaches that azimuth. Returns (psi_centers_deg, delta_min_deg), NaN where no admissible ray lands."""
    su = sun_dir(h_deg)
    d0 = -su
    g = np.linspace(0.0, 2 * math.pi, ngrid, endpoint=False)
    a = np.linspace(0.0, 2 * math.pi, ngrid, endpoint=False)
    G, A = np.meshgrid(g, a, indexing="ij")
    G, A = G.ravel(), A.ravel()
    cg, sg, cA, sA = np.cos(G), np.sin(G), np.cos(A), np.sin(A)
    cA2, sA2 = np.cos(A + FACE_SEP), np.sin(A + FACE_SEP)
    # u=(sg,-cg,0), w=(0,0,1); n1=cA*u+sA*w, n2=cA2*u+sA2*w  (outward face normals)
    n1 = np.stack([cA * sg, -cA * cg, sA], axis=-1)
    n2 = np.stack([cA2 * sg, -cA2 * cg, sA2], axis=-1)
    M = G.shape[0]
    d0b = np.broadcast_to(d0, (M, 3))
    entry_ok = (n1 @ d0) < 0                                 # ray enters through face 1
    d1, v1 = _refract_vec(d0b, n1, 1.0 / n)                  # air -> ice
    exit_ok = np.sum(d1 * n2, axis=-1) > 0                   # ray leaves through face 2
    d2, v2 = _refract_vec(d1, n2, n)                         # ice -> air (TIR -> v2 False)
    sky = -d2
    # polar coords about the sun
    z = np.array([0.0, 0.0, 1.0])
    up = z - np.dot(z, su) * su
    up = up / np.linalg.norm(up)
    right = np.cross(su, up)
    delta = np.degrees(np.arccos(np.clip(sky @ su, -1.0, 1.0)))
    sp = sky - (sky @ su)[:, None] * su
    psi = np.degrees(np.arctan2(sp @ right, sp @ up))
    ok = entry_ok & v1 & exit_ok & v2 & (delta <= dmax) & np.isfinite(delta)
    nb = 180
    edges = np.linspace(0.0, 180.0, nb + 1)
    best = np.full(nb, np.nan)
    b = np.clip((np.abs(psi) / 180.0 * nb).astype(int), 0, nb - 1)
    for i in np.nonzero(ok)[0]:
        bi = b[i]
        if math.isnan(best[bi]) or delta[i] < best[bi]:
            best[bi] = delta[i]
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, best


def is_merged(h_deg, n=N_ICE, ngrid=400):
    """True if the tangent-arc caustic is a CONNECTED closed loop (the circumscribed halo) — i.e. the
    delta_min(psi) caustic has no GAP between the upper (UTA) and lower (LTA) arcs. The gap is the
    inadmissibility region between the wing-tips; it closes at the A3-cusp merge."""
    centers, best = caustic_coverage(h_deg, n, ngrid=ngrid)
    mid = (centers >= 30.0) & (centers <= 150.0)        # between the UTA top and LTA bottom
    return int(np.sum(np.isnan(best[mid]))) <= 2         # gap closed (allow 2 bins for grid noise)


def merge_elevation(n=N_ICE, lo=22.0, hi=36.0, tol=0.2, ngrid=400):
    """Bisection for the smallest sun elevation at which the UTA + LTA fold caustics coalesce into the
    connected circumscribed-halo loop — the DERIVED A3-cusp merge elevation."""
    if is_merged(lo, n, ngrid):
        return lo
    if not is_merged(hi, n, ngrid):
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if is_merged(mid, n, ngrid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main():
    print("Atlas Phase 6.5-B — horizontal-column halo function: the 29deg tangent-arc merge (A3 cusp)")
    print(f"  n={N_ICE}, 60deg wedge; 2 orientation DOF (gamma,alpha); ray-optics caustic (det J=0).\n")
    # sanity: caustic touches ~22deg at the top (psi~0) at a low sun
    centers, best = caustic_coverage(20.0)
    top = best[centers < 8.0]
    top_min = np.nanmin(top) if np.any(~np.isnan(top)) else float("nan")
    print(f"  sanity (h=20): caustic min deviation near the top (psi<8deg) = {top_min:.2f} deg "
          f"(expect ~21.8, the 22deg-halo fold)")
    # caustic topology at a few elevations: the mid-psi GAP between the UTA and LTA wing-tips
    for h in (20.0, 25.0, 28.0, 29.0, 30.0, 35.0):
        centers, best = caustic_coverage(h, ngrid=400)
        mid = (centers >= 30.0) & (centers <= 150.0)
        ngap = int(np.sum(np.isnan(best[mid])))
        merged = ngap <= 2
        print(f"  h={h:>4.1f}: mid-psi gap = {ngap:>3d} bins   "
              f"-> {'CIRCUMSCRIBED (merged loop)' if merged else 'separate UTA + LTA arcs'}")
    hm = merge_elevation()
    # chromatic band
    hm_red = merge_elevation(n=1.307)
    hm_blue = merge_elevation(n=1.317)
    print(f"\n  DERIVED merge elevation (white light n={N_ICE}): {hm:.1f} deg  (documented ~29 deg)")
    if hm_red and hm_blue:
        print(f"  chromatic band: {min(hm_red, hm_blue):.1f}-{max(hm_red, hm_blue):.1f} deg "
              f"(n {1.307}red..{1.317}blue) -> the documented '29-32' spread")
    ok = hm is not None and abs(hm - 29.0) <= 1.0
    print(f"\n  merge within ±1.0deg of documented 29deg: {ok}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
