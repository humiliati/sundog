#!/usr/bin/env python
"""LINE 3 — independent PLATE-LOWITZ cross-check of the column-Lowitz cusp-pair birth.

Constructs the CLASSIC plate-Lowitz orientation family from scratch (NOT reusing the repo's
'lowitz60' wedge, which is the column-Lowitz: c starts HORIZONTAL). Here:

  - A hexagonal PLATE crystal: c-axis VERTICAL at rest. The six prism SIDE faces have outward
    normals in the horizontal plane at azimuths 0,60,...,300; the prism EDGES bisect them at
    30,90,... The sun-dog (parhelion) path refracts through an ALTERNATE side-face pair separated
    by 120deg (a 60deg refracting wedge == cm.FACE_SEP), n=cm.N_ICE.
  - 2 DOF (theta, beta):
        theta = crystal spin about its own c-axis (the fast variable that draws the parhelion);
        beta  = LOWITZ tilt about a BODY-FIXED horizontal axis L that lies in the basal plane and
                passes through opposite prism EDGES (azimuth 30deg in the body frame). L spins with
                the crystal: the tilt is applied AFTER the spin, about R_z(theta)·L0.
    This is the classic plate-Lowitz: at beta=0 the family is the ordinary plate parhelion; beta
    tilts the crystal about an edge, lofting the image into the Lowitz arcs.

At beta=0 this reproduces the plate PARHELION (~22deg from the sun, in the almucantar) — verified.
Then we run the SAME interior cusp-count + boundary-distance metamorphosis search across sun
elevation h that detected the column-Lowitz 2->4->2 birth, and ask whether plate-Lowitz reproduces it.

Reuses cm._refract_vec / cm.refract / cm.sun_dir / cm.FACE_SEP / cm.N_ICE. Touches no existing wedge.
"""
import math
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_caustic_map as cm

ERODE = 3  # match atlas_strata_map.ERODE

# Resting hex-plate geometry (c vertical = +z):
_FA0 = np.array([1.0, 0.0, 0.0])                                  # prism side face A, azimuth 0
_FB0 = np.array([math.cos(cm.FACE_SEP), math.sin(cm.FACE_SEP), 0.0])  # side face B, azimuth 120
# Body Lowitz axis through opposite prism EDGES (edges bisect faces -> azimuth 30deg), in basal plane:
_LEDGE_AZ = math.radians(30.0)
_L0 = np.array([math.cos(_LEDGE_AZ), math.sin(_LEDGE_AZ), 0.0])


def _rot_z(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_axis(axis, theta):
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    x, y, z = ax
    c, s = math.cos(theta), math.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def plate_lowitz_normals(theta, beta, ledge_az=_LEDGE_AZ):
    """Outward normals (n1,n2) of the sun-dog face pair for a plate-Lowitz crystal spun by theta
    about c then Lowitz-tilted by beta about the body edge axis L (spun with the crystal)."""
    Rz = _rot_z(theta)
    L0 = np.array([math.cos(ledge_az), math.sin(ledge_az), 0.0])
    L = Rz @ L0
    Rl = _rot_axis(L, beta)
    R = Rl @ Rz
    return R @ _FA0, R @ _FB0


def sky_grid_platelowitz(h_deg, n=cm.N_ICE, ngrid=300, az0_deg=0.0, ledge_az=_LEDGE_AZ):
    """Plate-Lowitz halo function over the (theta, beta) torus. Mirrors cm.sky_grid's contract:
    returns G(=theta), A(=beta) (M,), sky (M,3) apparent-source unit (NaN where inadmissible),
    ok (M,) bool, su (3,). 60deg side-face pair, body-fixed Lowitz edge axis.
    theta in [0,2pi); beta in [0,2pi) (full sweep keeps grid topology identical to the lowitz60
    comparison and never misses a cusp; the physics is pi-periodic in beta up to face relabeling)."""
    su = cm.sun_dir(h_deg, az0_deg)
    d0 = -su
    th = np.linspace(0.0, 2 * math.pi, ngrid, endpoint=False)
    be = np.linspace(0.0, 2 * math.pi, ngrid, endpoint=False)
    T, B = np.meshgrid(th, be, indexing="ij")
    T, B = T.ravel(), B.ravel()
    M = T.shape[0]
    cT, sT = np.cos(T), np.sin(T)
    cB, sB = np.cos(B), np.sin(B)
    # Body edge axis spun by theta:  L = R_z(theta) L0
    cL0, sL0 = math.cos(ledge_az), math.sin(ledge_az)
    Lx = cT * cL0 - sT * sL0
    Ly = sT * cL0 + cT * sL0
    Lz = np.zeros_like(Lx)

    def spin_then_tilt(v0):
        # v_spun = R_z(theta) v0
        v0x, v0y, v0z = v0
        vx = cT * v0x - sT * v0y
        vy = sT * v0x + cT * v0y
        vz = np.full_like(cT, v0z)
        # Rodrigues tilt by beta about L=(Lx,Ly,Lz): R v = v cosB + (L x v) sinB + L (L.v)(1-cosB)
        Ldv = Lx * vx + Ly * vy + Lz * vz
        crx = Ly * vz - Lz * vy
        cry = Lz * vx - Lx * vz
        crz = Lx * vy - Ly * vx
        rx = vx * cB + crx * sB + Lx * Ldv * (1 - cB)
        ry = vy * cB + cry * sB + Ly * Ldv * (1 - cB)
        rz = vz * cB + crz * sB + Lz * Ldv * (1 - cB)
        return np.stack([rx, ry, rz], axis=-1)

    n1 = spin_then_tilt(_FA0)
    n2 = spin_then_tilt(_FB0)
    d0b = np.broadcast_to(d0, (M, 3))
    entry_ok = np.sum(n1 * d0b, axis=-1) < 0
    d1, v1 = cm._refract_vec(d0b, n1, 1.0 / n)
    exit_ok = np.sum(d1 * n2, axis=-1) > 0
    d2, v2 = cm._refract_vec(d1, n2, n)
    ok = entry_ok & v1 & exit_ok & v2
    sky = -d2
    sky[~ok] = np.nan
    return T, B, sky, ok, su


def cusp_field_platelowitz(h_deg, n=cm.N_ICE, ngrid=300, ledge_az=_LEDGE_AZ):
    """Cusp locator for plate-Lowitz — identical construction to atlas_strata_map.cusp_field
    (caustic detJ=0 ∩ K·∇(detJ)=0), driven by sky_grid_platelowitz."""
    ng = ngrid
    G, A, sky, ok, su = sky_grid_platelowitz(h_deg, n, ng, ledge_az=ledge_az)
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
    for _ in range(ERODE + 1):
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


def ninterior_platelowitz(h, ng=320, bd=5.0, ledge_az=_LEDGE_AZ, return_locs=False):
    """Count interior cusp clusters with mean distance-to-boundary >= bd cells — IDENTICAL metric to
    the column-Lowitz reproducer. If return_locs, also return their (delta_deg,psi_deg,meandist)."""
    cusp, detJ, g, good, X, Y, su = cusp_field_platelowitz(h, ngrid=ng, ledge_az=ledge_az)
    dist = ndimage.distance_transform_edt(good)
    lab, nlab = ndimage.label(cusp, structure=np.ones((3, 3)))
    n_int = 0
    locs = []
    for i in range(1, nlab + 1):
        ys, xs = np.where(lab == i)
        md = float(np.mean(dist[ys, xs]))
        if md >= bd:
            n_int += 1
            if return_locs:
                Xc = float(np.nanmean(X[ys, xs])); Yc = float(np.nanmean(Y[ys, xs]))
                delta = math.degrees(math.asin(min(1.0, math.hypot(Xc, Yc))))
                psi = math.degrees(math.atan2(Xc, Yc))
                locs.append((round(delta, 1), round(psi, 1), round(md, 1)))
    if return_locs:
        return n_int, locs
    return n_int


def verify_parhelion(ng=2000):
    """At beta=0 the family must draw the plate PARHELION: ~22deg from the sun, in the almucantar
    (sky elevation == sun elevation). Sweep theta at beta=0."""
    print("Plate-parhelion check (beta=0, sweep crystal azimuth theta):")
    for h in (5.0, 10.0, 20.0, 30.0):
        su = cm.sun_dir(h); d0 = -su
        best = None
        for th in np.linspace(0, 2 * math.pi, ng, endpoint=False):
            n1, n2 = plate_lowitz_normals(th, 0.0)
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
                best = (dev, elev)
        if best:
            print(f"  h={h:>4.1f}: parhelion dev={best[0]:.2f}deg (expect ~22)  "
                  f"sky_elev={best[1]:.2f}deg (sun_elev={h:.1f}, expect equal -> almucantar)")
        else:
            print(f"  h={h:>4.1f}: NO admissible ray (!)")


def main():
    print("LINE 3 — PLATE-LOWITZ independent cross-check\n")
    verify_parhelion()
    print("\nInterior cusp-cluster count vs sun elevation (deep-interior, dist>=5 cells), ng=320,")
    print("Lowitz edge axis at 30deg (through opposite prism edges):")
    print("  (column-Lowitz baseline: 2,2,4,4,4,4,4,2 at h=14,16,17,18,20,25,30,31)")
    print(f"  {'h':>5}{'#interior':>11}   interior cusp sky-positions (delta,psi,dist)")
    res = {}
    for h in (12, 14, 16, 17, 18, 20, 22, 25, 28, 30, 31, 33, 35):
        ni, locs = ninterior_platelowitz(h, return_locs=True)
        res[h] = ni
        print(f"  {h:>5}{ni:>11}   {locs}")
    counts = list(res.values())
    print(f"\n  plate-Lowitz interior cusp range: {min(counts)}..{max(counts)}")
    return res


if __name__ == "__main__":
    main()
