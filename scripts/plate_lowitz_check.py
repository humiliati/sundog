#!/usr/bin/env python
"""LINE 3 — independent PLATE-LOWITZ cross-check of the column-Lowitz cusp-pair birth.

Constructs the CLASSIC plate-Lowitz orientation family from scratch (NOT reusing the repo's
'lowitz60' wedge, which is the column-Lowitz: c starts HORIZONTAL). Here:

  - A hexagonal PLATE crystal: c-axis VERTICAL at rest (beta=0). The six prism SIDE faces have
    outward normals in the horizontal plane, at azimuths 0,60,120,...; the sun-dog (parhelion)
    path refracts through an ALTERNATE side-face pair separated by 120deg (a 60deg refracting
    wedge, == cm.FACE_SEP), with n=cm.N_ICE.
  - The LOWITZ rotation: tilt the whole crystal by angle beta about a HORIZONTAL axis L lying in
    the basal plane (the plate's c=0 plane), at azimuth psi. Classic plate-Lowitz: L passes
    through opposite prism EDGES. The 2 DOF are (psi, beta).

At beta=0 this must reproduce the plate PARHELION (~22deg from the sun, in the sun's almucantar).
Then we run the SAME interior cusp-count + boundary-distance metamorphosis search across sun
elevation h that detected the column-Lowitz 2->4->2 birth, and ask whether the plate-Lowitz
reproduces it.

Reuses cm._refract_vec / cm.sun_dir / cm.FACE_SEP / cm.N_ICE. Does NOT touch existing wedges.
"""
import math
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_caustic_map as cm


def _rot_axis(axis, theta):
    """Rotation matrix for angle theta about unit axis (Rodrigues), axis shape (...,3) or (3,)."""
    axis = np.asarray(axis, float)
    ax = axis / np.linalg.norm(axis)
    x, y, z = ax
    c, s = math.cos(theta), math.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


# The two refracting prism side-face outward normals of a RESTING hex plate (c vertical).
# Side-face normals lie in the horizontal plane. A free crystal-azimuth phase 'theta_az' would
# rotate them about the vertical; for the plate parhelion the sun-relative geometry is set by the
# face pair, so we expose the azimuth as the 2nd usable DOF only via the Lowitz axis (psi). The
# resting face pair: normal A at azimuth 0, normal B at azimuth 120deg (== 180-WEDGE = FACE_SEP).
_FA0 = np.array([1.0, 0.0, 0.0])
_FB0 = np.array([math.cos(cm.FACE_SEP), math.sin(cm.FACE_SEP), 0.0])
_CVERT = np.array([0.0, 0.0, 1.0])


def plate_lowitz_normals(psi, beta):
    """Outward normals (n1,n2) of the sun-dog refracting prism-face pair for a plate-Lowitz crystal
    tilted by beta about the horizontal axis L at azimuth psi (L in the resting basal plane).
    psi,beta scalars -> (n1,n2) each (3,)."""
    L = np.array([math.cos(psi), math.sin(psi), 0.0])   # horizontal Lowitz axis in the basal plane
    R = _rot_axis(L, beta)
    return R @ _FA0, R @ _FB0


def sky_grid_platelowitz(h_deg, n=cm.N_ICE, ngrid=300, az0_deg=0.0):
    """Plate-Lowitz halo function over the (psi, beta) torus. Mirrors cm.sky_grid's contract:
    returns G(=psi), A(=beta) (M,), sky (M,3) apparent-source unit (NaN where inadmissible),
    ok (M,) bool, su (3,). 60deg side-face pair, single basal-plane Lowitz tilt axis.

    psi spans [0,2pi); beta spans [0,2pi) to match the torus convention of the column search
    (the physics is pi-periodic in beta up to face relabeling, but a full 2pi sweep keeps the
    grid topology identical to the lowitz60 comparison and never misses a cusp)."""
    su = cm.sun_dir(h_deg, az0_deg)
    d0 = -su
    psi = np.linspace(0.0, 2 * math.pi, ngrid, endpoint=False)
    beta = np.linspace(0.0, 2 * math.pi, ngrid, endpoint=False)
    P, B = np.meshgrid(psi, beta, indexing="ij")
    P, B = P.ravel(), B.ravel()
    M = P.shape[0]
    cP, sP = np.cos(P), np.sin(P)
    cB, sB = np.cos(B), np.sin(B)
    # Rodrigues rotation of the two resting face normals about L=(cP,sP,0) by B, vectorized.
    # For a vector v and axis L (unit), R v = v cosB + (L x v) sinB + L (L.v)(1-cosB).
    def rot(v):
        vx, vy, vz = v
        # L.v
        Ldv = cP * vx + sP * vy  # vz term is 0 since L_z=0
        # L x v = (sP*vz - 0*vy, 0*vx - cP*vz, cP*vy - sP*vx)
        crx = sP * vz
        cry = -cP * vz
        crz = cP * vy - sP * vx
        rx = vx * cB + crx * sB + cP * Ldv * (1 - cB)
        ry = vy * cB + cry * sB + sP * Ldv * (1 - cB)
        rz = vz * cB + crz * sB + 0.0 * Ldv * (1 - cB)
        return np.stack([rx, ry, rz], axis=-1)
    n1 = rot(_FA0)
    n2 = rot(_FB0)
    d0b = np.broadcast_to(d0, (M, 3))
    entry_ok = np.sum(n1 * d0b, axis=-1) < 0          # ray enters through face 1
    d1, v1 = cm._refract_vec(d0b, n1, 1.0 / n)        # air -> ice
    exit_ok = np.sum(d1 * n2, axis=-1) > 0            # ray leaves through face 2
    d2, v2 = cm._refract_vec(d1, n2, n)               # ice -> air (TIR -> v2 False)
    ok = entry_ok & v1 & exit_ok & v2
    sky = -d2
    sky[~ok] = np.nan
    return P, B, sky, ok, su


def cusp_field_platelowitz(h_deg, n=cm.N_ICE, ngrid=300):
    """Cusp locator for the plate-Lowitz family — same construction as atlas_strata_map.cusp_field
    (caustic detJ=0 ∩ K·∇(detJ)=0), but driven by sky_grid_platelowitz."""
    ng = ngrid
    G, A, sky, ok, su = sky_grid_platelowitz(h_deg, n, ng)
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
    detJ = Xg * Ya - Xa * Yg
    a = Xg * Xg + Yg * Yg
    b = Xg * Xa + Yg * Ya
    c = Xa * Xa + Ya * Ya
    tr = a + c
    disc = np.sqrt(np.clip(tr * tr - 4 * (a * c - b * b), 0, None))
    lam2 = (tr - disc) / 2
    k1, k2 = b, lam2 - a
    alt = np.abs(b) < 1e-12
    k1 = np.where(alt, lam2 - c, k1)
    k2 = np.where(alt, b, k2)
    norm = np.sqrt(k1 * k1 + k2 * k2) + 1e-30
    Kg, Ka = k1 / norm, k2 / norm
    dJg, dJa = np.gradient(detJ, d, d)
    g = Kg * dJg + Ka * dJa
    good = validg.copy()
    for _ in range(cm_ERODE + 1):
        g2 = good.copy()
        g2[1:, :] &= good[:-1, :]; g2[:-1, :] &= good[1:, :]
        g2[:, 1:] &= good[:, :-1]; g2[:, :-1] &= good[:, 1:]
        good = g2

    def signchange(F):
        sc = np.zeros_like(good, bool)
        for ax in (0, 1):
            idx = np.abs(np.diff(np.sign(F), axis=ax)) > 0
            sl0 = [slice(None)] * 2; sl1 = [slice(None)] * 2
            sl0[ax] = slice(0, -1); sl1[ax] = slice(1, None)
            sc[tuple(sl0)] |= idx; sc[tuple(sl1)] |= idx
        return sc

    cusp = signchange(detJ) & signchange(g) & good
    return cusp, detJ, g, good, X, Y, su


cm_ERODE = 3  # match atlas_strata_map.ERODE


def ninterior_platelowitz(h, ng=320, bd=5.0):
    """Count interior cusp clusters whose mean distance-to-boundary >= bd cells (deep-interior,
    not admissibility-wall artifacts) — IDENTICAL metric to the column-Lowitz reproducer."""
    cusp, detJ, g, good, X, Y, su = cusp_field_platelowitz(h, ngrid=ng)
    dist = ndimage.distance_transform_edt(good)
    lab, nlab = ndimage.label(cusp, structure=np.ones((3, 3)))
    return sum(1 for i in range(1, nlab + 1)
               if np.mean(dist[tuple(np.where(lab == i))]) >= bd)


def verify_parhelion(ng=600):
    """At beta=0 (plate flat) the admissible rays must land at the plate parhelion: ~22deg from the
    sun, in the almucantar (same elevation as the sun). Check across a few sun elevations."""
    print("Plate-parhelion check (beta=0 slice): apparent deviation from the sun + elevation offset")
    for h in (10.0, 20.0, 30.0):
        su = cm.sun_dir(h)
        # sample beta near 0 (use the resting face pair directly)
        best = None
        for psi in np.linspace(0, 2 * math.pi, 720, endpoint=False):
            n1, n2 = plate_lowitz_normals(psi, 0.0)
            d0 = -su
            if np.dot(n1, d0) >= 0:
                continue
            d1 = cm.refract(d0, n1, 1.0 / cm.N_ICE)
            if d1 is None or np.dot(d1, n2) <= 0:
                continue
            d2 = cm.refract(d1, n2, cm.N_ICE)
            if d2 is None:
                continue
            sky = -d2
            dev = math.degrees(math.acos(np.clip(np.dot(sky, su), -1, 1)))
            elev = math.degrees(math.asin(np.clip(sky[2], -1, 1)))
            if best is None or abs(dev - 22.0) < abs(best[0] - 22.0):
                best = (dev, elev, math.degrees(psi))
        if best:
            print(f"  h={h:>4.1f}: closest-to-22deg ray  dev={best[0]:.2f}deg  "
                  f"sky_elev={best[1]:.2f}deg (sun_elev={h:.1f})  at psi={best[2]:.1f}deg")
        else:
            print(f"  h={h:>4.1f}: NO admissible ray")


def main():
    print("LINE 3 — PLATE-LOWITZ independent cross-check\n")
    verify_parhelion()
    print("\nInterior cusp-cluster count vs sun elevation (deep-interior, dist>=5 cells), ng=320:")
    print("  (column-Lowitz baseline for comparison: 2,2,4,4,4,4,4,2 at h=14,16,17,18,20,25,30,31)")
    print(f"  {'h':>5}{'#interior_cusps':>18}")
    res = {}
    for h in (12, 14, 16, 17, 18, 20, 22, 25, 28, 30, 31, 33, 35):
        ni = ninterior_platelowitz(h)
        res[h] = ni
        print(f"  {h:>5}{ni:>18}")
    counts = list(res.values())
    born = max(counts) > min(counts) and max(counts) >= min(counts) + 2
    print(f"\n  plate-Lowitz interior cusp range: {min(counts)}..{max(counts)}")
    print(f"  reproduces a 2->4->2-style interior cusp-pair birth: {born}")
    return res


if __name__ == "__main__":
    main()
