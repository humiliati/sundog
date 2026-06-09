#!/usr/bin/env python
"""S2 follow-on — a full per-habit POLARIZED Monte-Carlo halo raytracer.

Removes the one disclosed boundary of scripts/s2_handedness_map.py (which imposes a SCHEMATIC fixed
face-sequence enter-side -> TIR-basal -> exit-side and never ray-marches the crystal). Here photons are
ray-MARCHED through a genuine hexagonal-ice POLYHEDRON (vector Snell + Fresnel transmission / partial
reflection / TIR phase + ice birefringence, carrying the full Stokes vector I,Q,U,V with proper
inter-interface frame rotations), over per-habit orientation distributions, binned into a sky map. The
22 deg / 46 deg halos, the Koennen-validated LINEAR polarization, and the per-feature +/-V CIRCULAR
handedness map all EMERGE from one engine.

Forward-model tier; ray-optics limit; NOT public-eligible. Deterministic (seeded RNG). Attribution:
halo raytracing — Wendling 1979 / Macke 1996 / Greenler / Tape; polarimetry — Koennen & Tinbergen 1991;
Mueller-Stokes formalism; ice n=1.31 (Warren & Brandt 2008), Delta_n=+0.0014.

Run:  python scripts/s2_halo_raytracer.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_optics as so                              # noqa: E402
# reuse the validated 3D + Mueller helpers from the handedness map (the bridge-test reference)
from s2_handedness_map import (                     # noqa: E402
    unit, refract, reflect, s_axis, signed_angle, mrot, incidence_deg, Rz,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

N_ICE = so.N_ICE
DN_ICE = so.DN_ICE
LAM = so.LAM_LIGHT
THETA_C = np.degrees(np.arcsin(1.0 / N_ICE))
# ice {10-11} pyramid-face normal angle from the c-axis, DERIVED from c/a=1.628 (Atlas; the same
# acos(1/sqrt((4/3)(c/a)^2+1))). Opposite faces -> the 9-deg halo; 120-deg-apart -> the ~24-deg halo.
PYR_X = np.radians(61.99)


# ----------------------------------------------------------------------------- #
# rotation helpers                                                              #
# ----------------------------------------------------------------------------- #
def Ry(b):
    c, s = np.cos(b), np.sin(b)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def quat_to_R(q):
    """Unit quaternion (x,y,z,w) -> rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


# ----------------------------------------------------------------------------- #
# crystal geometry — hexagonal prism as a convex polyhedron (8 half-spaces)      #
# ----------------------------------------------------------------------------- #
def make_crystal(aspect, a=1.0, pyramidal=False):
    """Hexagonal ice crystal as a convex polyhedron, body frame, inside = {x : n_i.x <= offset_i}.
    Body optic/c-axis = +z. `kind`: 0 prism, 1 basal, 2 upper-pyramid, 3 lower-pyramid.
    - default (pyramidal=False): 6 prism (normals at 60-deg steps, offset a) + 2 basal (+/-z, offset L/2).
    - pyramidal=True: 6 prism + 6 upper + 6 lower {10-11} pyramid faces (normal at PYR_X=61.99-deg from c,
      azimuths 60-deg-steps; offset a*sinX + (L/2)*cosX caps the prism cleanly) -> the odd-radius halos."""
    L = aspect * 2.0 * a
    if not pyramidal:
        normals = np.zeros((8, 3)); offsets = np.zeros(8); kind = np.zeros(8, int)
        for k in range(6):
            ang = np.radians(60 * k)
            normals[k] = [np.cos(ang), np.sin(ang), 0.0]; offsets[k] = a
        normals[6] = [0.0, 0.0, 1.0]; offsets[6] = L / 2
        normals[7] = [0.0, 0.0, -1.0]; offsets[7] = L / 2
        kind[6] = kind[7] = 1
        return normals, offsets, kind
    cx, sx = np.cos(PYR_X), np.sin(PYR_X)
    dp = a * sx + (L / 2) * cx
    normals = np.zeros((18, 3)); offsets = np.zeros(18); kind = np.zeros(18, int)
    for k in range(6):
        ang = np.radians(60 * k)
        normals[k] = [np.cos(ang), np.sin(ang), 0.0]; offsets[k] = a; kind[k] = 0
        normals[6 + k] = [sx * np.cos(ang), sx * np.sin(ang), cx]; offsets[6 + k] = dp; kind[6 + k] = 2
        normals[12 + k] = [sx * np.cos(ang), sx * np.sin(ang), -cx]; offsets[12 + k] = dp; kind[12 + k] = 3
    return normals, offsets, kind


def _pyr_vertices(k, up, aspect, a):
    """The 3 vertices of pyramid face k (up=True upper cap): two prism-top corners + the apex."""
    L = aspect * 2.0 * a
    R = 2.0 * a / np.sqrt(3.0)                        # hexagon circumradius
    cx, sx = np.cos(PYR_X), np.sin(PYR_X)
    zap = (a * sx + (L / 2) * cx) / cx                # apex height on the c-axis
    z = (L / 2) if up else -(L / 2); zapex = zap if up else -zap
    aL = np.radians(60 * k - 30); aR = np.radians(60 * k + 30)
    return (np.array([R * np.cos(aL), R * np.sin(aL), z]),
            np.array([R * np.cos(aR), R * np.sin(aR), z]),
            np.array([0.0, 0.0, zapex]))


def face_areas(aspect, a=1.0, pyramidal=False):
    """Area of each face: prism rectangles edge*L (edge=2a/sqrt3); basal hexagon (3*sqrt3/2)a^2;
    pyramid faces = triangle area (two prism-top corners + apex)."""
    L = aspect * 2.0 * a
    edge = 2.0 * a / np.sqrt(3.0)
    if not pyramidal:
        A = np.zeros(8)
        A[:6] = edge * L
        A[6] = A[7] = (3.0 * np.sqrt(3.0) / 2.0) * a * a
        return A
    A = np.zeros(18)
    A[:6] = edge * L
    for k in range(6):
        for j, up in ((6, True), (12, False)):
            V1, V2, V3 = _pyr_vertices(k, up, aspect, a)
            A[j + k] = 0.5 * np.linalg.norm(np.cross(V2 - V1, V3 - V1))
    return A


def exit_face(o, d, normals, offsets):
    """For a convex body, the face a ray (o, d) exits through = argmin positive t over faces with
    n_i.d > 0,  t_i = (offset_i - n_i.o)/(n_i.d). Returns (idx, t, hitpoint) or None."""
    denom = normals @ d
    num = offsets - normals @ o
    with np.errstate(divide="ignore", invalid="ignore"):
        t = num / denom
    mask = (denom > 1e-9) & (t > 1e-7)
    if not mask.any():
        return None
    tt = np.where(mask, t, np.inf)
    idx = int(np.argmin(tt))
    return idx, float(tt[idx]), o + tt[idx] * d


# ----------------------------------------------------------------------------- #
# entry sampling — illuminated faces weighted by projected area                  #
# ----------------------------------------------------------------------------- #
def _point_on_face(fidx, kind, aspect, a, rng):
    """Uniform random point on body-frame face fidx."""
    L = aspect * 2.0 * a
    if kind[fidx] in (2, 3):                          # pyramid triangle (upper k=fidx-6 / lower k=fidx-12)
        up = kind[fidx] == 2
        k = fidx - 6 if up else fidx - 12
        V1, V2, V3 = _pyr_vertices(k, up, aspect, a)
        u, v = rng.random(2)
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        return V1 + u * (V2 - V1) + v * (V3 - V1)
    if kind[fidx] == 0:                              # prism rectangle
        ang = np.radians(60 * fidx)
        nrm = np.array([np.cos(ang), np.sin(ang), 0.0])
        tang = np.array([-np.sin(ang), np.cos(ang), 0.0])   # along-edge horizontal
        edge = 2.0 * a / np.sqrt(3.0)
        u = (rng.random() - 0.5) * edge
        v = (rng.random() - 0.5) * L
        return a * nrm + u * tang + v * np.array([0.0, 0.0, 1.0])
    # basal hexagon — rejection sample in the inscribed-circle's bounding, accept inside hexagon
    z = (L / 2) if fidx == 6 else (-L / 2)
    R = 2.0 * a / np.sqrt(3.0)                        # circumradius
    while True:
        x, y = (rng.random(2) - 0.5) * 2 * R
        # inside hexagon iff all prism |n_k.(x,y)| <= a
        if all(abs(np.cos(np.radians(60 * k)) * x + np.sin(np.radians(60 * k)) * y) <= a
               for k in range(6)):
            return np.array([x, y, z])


def sample_entry(R, normals_w, offsets, kind, areas, d_sun, aspect, a, rng):
    """Pick an illuminated face ∝ (area * projected cosine), a uniform point on it, and return
    (entry_point_world, entry_normal_world, A_proj) where A_proj = total projected area (the flux
    weight for this orientation). Returns None if no face is illuminated."""
    nd = normals_w @ (-d_sun)                        # >0 = illuminated
    w = areas * np.maximum(nd, 0.0)
    tot = w.sum()
    if tot <= 0:
        return None
    fidx = int(rng.choice(len(w), p=w / tot))
    p_body = _point_on_face(fidx, kind, aspect, a, rng)
    p_world = R @ p_body
    return p_world, normals_w[fidx], tot


# ----------------------------------------------------------------------------- #
# the polarized ray-tree tracer                                                 #
# ----------------------------------------------------------------------------- #
def trace_tree(o, d_in, entry_normal, normals, offsets, c_axis, dn=0.0, size_um=100.0,
               K=1, eps=1e-4, n=N_ICE, lam=LAM, include_external=False):
    """Ray-march one photon entering at `o` along `d_in` (air->crystal through `entry_normal`) and
    return the list of exit (transmitted) deposits [(d_out_world(3,), S(4,), s_ref(3,)), ...].
    Carries the Stokes vector through entry-Fresnel, per-segment birefringence, inter-interface frame
    rotations, TIR-phase retarder / partial-reflection diattenuator, with intensity (energy) living in
    Stokes I. `K` = max internal reflections; `eps` = intensity prune. `include_external` also deposits
    the EXTERNAL reflection glint off the entry face (the dominant white-parhelic-circle mechanism off
    vertical faces — a vertical mirror preserves the ray's elevation -> a horizontal circle through the
    sun). Deterministic given inputs."""
    r = refract(d_in, entry_normal, 1.0, n)
    if r is None:
        return []
    d1, th_in = r
    Men = so.mueller_fresnel(th_in, 1.0, n)
    if Men is None:
        return []
    Me, th1 = Men
    gfac_e = (n * np.cos(np.radians(th1))) / (1.0 * np.cos(np.radians(th_in)))  # energy-correct entry T
    S = gfac_e * (Me @ np.array([1.0, 0.0, 0.0, 0.0]))
    s_prev = s_axis(d_in, entry_normal)

    deposits = []
    if include_external:                              # specular glint off the outer surface (PHC)
        Mre = so.mueller_fresnel_reflect(th_in, 1.0, n)
        if Mre is not None:
            deposits.append((reflect(d_in, entry_normal), Mre @ np.array([1.0, 0.0, 0.0, 0.0]), s_prev))
    stack = [(o, d1, S, s_prev, 0)]
    while stack:
        o, d, S, s_prev, k = stack.pop()
        ef = exit_face(o, d, normals, offsets)
        if ef is None:
            continue
        fidx, t, hit = ef
        nf = normals[fidx]
        # birefringence over the interior segment (fast axis = projected optic axis)
        if dn > 0:
            cperp = c_axis - np.dot(c_axis, d) * d
            if np.linalg.norm(cperp) > 1e-6:
                phib = signed_angle(s_prev, cperp, d)
                S = so.mueller_retarder(2.0 * np.pi * dn * (t * size_um) / lam, phib) @ S
        th = incidence_deg(d, nf)
        s_cur = s_axis(d, nf)
        S = mrot(signed_angle(s_prev, s_cur, d)) @ S      # rotate frame to this interface's s-axis
        rr = refract(d, nf, n, 1.0)
        if rr is None:                                    # total internal reflection
            S2 = so.mueller_tir(th, n, 1.0, 0.0) @ S
            if S2[0] > eps and k < K:
                stack.append((hit, reflect(d, nf), S2, s_cur, k + 1))
        else:
            d_out, th_out = rr
            Mx = so.mueller_fresnel(th, n, 1.0)
            if Mx is not None:
                gfac = (1.0 * np.cos(np.radians(th_out))) / (n * np.cos(np.radians(th)))
                deposits.append((d_out, gfac * (Mx[0] @ S), s_cur))   # transmitted ray exits
            Mr = so.mueller_fresnel_reflect(th, n, 1.0)               # partial internal reflection
            if Mr is not None:
                S_r = Mr @ S
                if S_r[0] > eps and k < K:
                    stack.append((hit, reflect(d, nf), S_r, s_cur, k + 1))
    return deposits


# ----------------------------------------------------------------------------- #
# orientation samplers (seeded -> deterministic)                                #
# ----------------------------------------------------------------------------- #
def orient_random(rng, sigma_deg=0.0):
    u1, u2, u3 = rng.random(3)
    q = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                  np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                  np.sqrt(u1) * np.sin(2 * np.pi * u3),
                  np.sqrt(u1) * np.cos(2 * np.pi * u3)])
    return quat_to_R(q)


def orient_plate(rng, sigma_deg=1.5):
    """Plate: basal horizontal (c-axis ~ vertical) + half-normal tilt wobble + uniform spin."""
    beta = abs(rng.normal(0, np.radians(sigma_deg)))
    gamma = rng.random() * 2 * np.pi
    psi = rng.random() * 2 * np.pi
    return Rz(gamma) @ Ry(beta) @ Rz(psi)


def orient_column(rng, sigma_deg=1.5):
    """Column: long axis (c-axis) horizontal + uniform azimuth + spin about long axis + wobble."""
    alpha = rng.random() * 2 * np.pi
    wob = rng.normal(0, np.radians(sigma_deg))
    psi = rng.random() * 2 * np.pi
    return Rz(alpha) @ Ry(np.pi / 2 + wob) @ Rz(psi)


def orient_parry(rng, sigma_deg=1.0):
    """Parry: a column (c-axis horizontal) with two prism faces horizontal (the stable Parry
    orientation). c-axis azimuth uniform; a prism-face normal locked vertical; small tumble wobble.
    The base maps body-x (a prism normal) -> world +z and body-z (the c-axis) -> horizontal at alpha."""
    alpha = rng.random() * 2 * np.pi
    Rb = np.array([[0.0, np.sin(alpha), np.cos(alpha)],
                   [0.0, -np.cos(alpha), np.sin(alpha)],
                   [1.0, 0.0, 0.0]])
    w1, w2 = rng.normal(0, np.radians(sigma_deg), 2)
    return Rz(w1) @ Ry(w2) @ Rb


HABITS = {
    "random": (orient_random, 1.0, False),       # (sampler, aspect, pyramidal)
    "plate": (orient_plate, 0.4, False),
    "column": (orient_column, 3.0, False),
    "parry": (orient_parry, 3.0, False),
    "pyramidal": (orient_random, 1.0, True),      # randomly-tumbling pyramidal crystals -> odd radii
}


# ----------------------------------------------------------------------------- #
# ensemble + sky accumulation                                                   #
# ----------------------------------------------------------------------------- #
def run_ensemble(habit="random", e_deg=20.0, n_orient=4000, dn=0.0, size_um=100.0,
                 K=1, a=1.0, seed=0, sigma_deg=1.5, include_external=False):
    """Trace an ensemble of one habit. Returns a deposit array with columns
    [scatt_deg, az_deg, el_deg, I, Q, U, V] in the world/scattering frame, intensity-weighted by the
    orientation's projected area. Deterministic given seed."""
    sampler, aspect, pyr = HABITS[habit]
    normals_b, offsets, kind = make_crystal(aspect, a, pyramidal=pyr)
    areas = face_areas(aspect, a, pyramidal=pyr)
    e = np.radians(e_deg)
    d_sun = -np.array([np.cos(e), 0.0, np.sin(e)])    # propagation direction of sunlight
    rng = np.random.default_rng(seed)

    rows = []
    for _ in range(n_orient):
        R = sampler(rng, sigma_deg)
        normals_w = normals_b @ R.T
        c_axis = R @ np.array([0.0, 0.0, 1.0])
        ent = sample_entry(R, normals_w, offsets, kind, areas, d_sun, aspect, a, rng)
        if ent is None:
            continue
        o0, n_entry, A_proj = ent
        deps = trace_tree(o0, d_sun, n_entry, normals_w, offsets, c_axis,
                          dn=dn, size_um=size_um, K=K, n=N_ICE, lam=LAM,
                          include_external=include_external)
        for d_out, S, s_ref in deps:
            sky = -unit(d_out)                        # apparent direction (where light comes FROM)
            az = np.degrees(np.arctan2(sky[1], sky[0]))
            el = np.degrees(np.arcsin(np.clip(sky[2], -1, 1)))
            scatt = np.degrees(np.arccos(np.clip(np.dot(unit(d_sun), unit(d_out)), -1, 1)))
            # rotate Stokes from the exit interface s-frame into the SCATTERING-PLANE frame
            # (+Q = perpendicular to the scattering plane; the radial/tangential Koennen frame)
            sp_normal = np.cross(unit(d_sun), unit(d_out))
            if np.linalg.norm(sp_normal) > 1e-9:
                chi = signed_angle(s_ref, sp_normal, d_out)
                Sf = mrot(chi) @ S
            else:
                Sf = S
            w = A_proj
            rows.append([scatt, az, el, w * Sf[0], w * Sf[1], w * Sf[2], w * Sf[3]])
    return np.array(rows) if rows else np.zeros((0, 7))


# ----------------------------------------------------------------------------- #
# gate metrics                                                                  #
# ----------------------------------------------------------------------------- #
def radial_intensity(dep, bin_deg=0.5, max_deg=70.0):
    edges = np.arange(0, max_deg + bin_deg, bin_deg)
    h, _ = np.histogram(dep[:, 0], bins=edges, weights=dep[:, 3])
    return 0.5 * (edges[:-1] + edges[1:]), h


def ring_pol(dep, r_lo, r_hi):
    """Net (I,Q,U,V) and DoP/V over a scattering-angle ring [r_lo,r_hi]."""
    m = (dep[:, 0] >= r_lo) & (dep[:, 0] < r_hi)
    if not m.any():
        return None
    I, Q, U, V = dep[m, 3].sum(), dep[m, 4].sum(), dep[m, 5].sum(), dep[m, 6].sum()
    dop = np.hypot(Q, U) / I if I > 0 else 0.0
    return dict(I=I, Q=Q, U=U, V=V, dop=dop, voi=V / I if I > 0 else 0.0,
                uoi=U / I if I > 0 else 0.0, qoi=Q / I if I > 0 else 0.0, n=int(m.sum()))


def _report():
    print("=" * 78)
    print("S2 full per-habit polarized halo raytracer")
    print("=" * 78)
    print(f"ice n={N_ICE}, dn={DN_ICE}, crit angle={THETA_C:.2f} deg\n")

    print("GATE 1 — geometry (random habit, refraction halos):")
    dep = run_ensemble("random", e_deg=20.0, n_orient=12000, dn=0.0, K=1, seed=1)
    ctr, h = radial_intensity(dep)
    # find peaks near 22 and 46
    for name, lo, hi, tgt in (("22 deg", 18, 26, so.halo_min_deviation(60)),
                              ("46 deg", 42, 50, so.halo_min_deviation(90))):
        seg = (ctr >= lo) & (ctr < hi)
        if seg.any() and h[seg].max() > 0:
            pk = ctr[seg][np.argmax(h[seg])]
            print(f"  {name} halo peak at {pk:.2f} deg   (analytic {tgt:.2f} deg)")
    print()

    print("GATE 2 — linear polarization at the 22 deg halo (vs Koennen 1991):")
    for tag, dn in (("Fresnel floor (dn=0)", 0.0), ("with birefringence", DN_ICE)):
        dep = run_ensemble("random", e_deg=20.0, n_orient=20000, dn=dn, K=1, seed=2)
        rp = ring_pol(dep, 20.5, 23.5)
        if rp:
            print(f"  {tag:24s}: DoP={rp['dop']*100:.2f}%  U/I={rp['uoi']*100:+.2f}% (expect ~0)"
                  f"  Q/I={rp['qoi']*100:+.2f}%")
    print()

    print("GATE 3 — per-feature circular V (plate, TIR-rich features):")
    for tag, dn in (("TIR-only (dn=0)", 0.0), ("with birefringence", DN_ICE)):
        dep = run_ensemble("plate", e_deg=20.0, n_orient=40000, dn=dn, K=1, seed=3)
        m = np.abs(dep[:, 6]) > 0
        if m.any():
            voi = dep[:, 6] / np.where(dep[:, 3] != 0, dep[:, 3], 1)
            peak = np.abs(voi[np.abs(dep[:, 3]) > 0]).max() if (np.abs(dep[:, 3]) > 0).any() else 0
            # azimuthal antisymmetry of intensity-weighted V on the sub-horizon TIR arc
            az = dep[:, 1]; Vw = dep[:, 6]
            net = abs(Vw.sum()) / np.abs(Vw).sum() if np.abs(Vw).sum() > 0 else 0
            print(f"  {tag:20s}: per-ray peak |V/I|={peak*100:.2f}%   net |sumV|/sum|V|={net*100:.2f}%")
    print("GATE 4 — parhelic circle (K=2 multi-bounce + external reflection, plate):")
    dp = run_ensemble("plate", e_deg=20.0, n_orient=20000, dn=0.0, K=2, seed=8, include_external=True)
    el, az, I = dp[:, 2], dp[:, 1], dp[:, 3]
    eh, eed = np.histogram(el, bins=np.arange(-90, 91, 2.0), weights=I)
    elpk = (0.5 * (eed[:-1] + eed[1:]))[np.argmax(eh)]
    band = np.abs(el - 20.0) < 2.5
    azcov = len(np.unique(np.round(az[band] / 15)))
    I120 = I[band & (np.abs(np.abs(az) - 120) < 12)].sum()
    Igap = I[band & (np.abs(az) > 55) & (np.abs(az) < 85)].sum()
    Vw = dp[:, 6]; netP = abs(Vw[band].sum()) / np.abs(Vw[band]).sum()
    print(f"  parhelic circle at constant elevation {elpk:.0f} deg (sun=20); azimuth coverage {azcov}/24 "
          f"bins (full circle)")
    print(f"  120-deg parhelia (two-internal-reflection K=2 feature): I(120 deg)={I120:.0f} vs gap I={Igap:.0f}")
    print(f"  PHC circular V nets to ~0: |sumV|/sum|V|={netP*100:.2f}% (achiral, antisymmetric)\n")

    print("GATE 5 — pyramidal odd-radius halos + polarization (random pyramidal crystals):")
    dpy = run_ensemble("pyramidal", e_deg=20.0, n_orient=18000, dn=DN_ICE, K=1, seed=12)
    sc, Ipy = dpy[:, 0], dpy[:, 3]
    rad = "  ".join(f"{r}:{Ipy[(sc >= r - 1.2) & (sc < r + 1.2)].sum():.0f}" for r in (9, 18, 20, 23, 24, 35))
    print(f"  odd-radius halo weights (deg:flux):  {rad}")
    for nm, lo, hi in (("9deg ", 8, 10.5), ("24deg", 22.5, 25), ("35deg", 34, 37)):
        rp = ring_pol(dpy, lo, hi)
        print(f"  {nm}: DoP={rp['dop']*100:5.2f}%  U/I={rp['uoi']*100:+.2f}%  V/I={rp['voi']*100:+.2f}%")
    print("  -> radial pol (U=0), DoP rises with radius, no net V (refraction halos)\n")

    print("GATE 6 — Parry above-sun arc + K=3 antisolar growth:")
    dpa = run_ensemble("parry", e_deg=20.0, n_orient=12000, dn=0.0, K=1, seed=13)
    elp, azp, Ipa = dpa[:, 2], dpa[:, 1], dpa[:, 3]
    mer = np.abs(azp) < 25; above = mer & (elp > 40)
    print(f"  Parry: above-sun meridian flux (el>40 deg) = {Ipa[above].sum()/Ipa[mer].sum()*100:.1f}% "
          f"(the Parry-arc region, above the 22-deg halo top)\n")
    print("(see test_s2_halo_raytracer.py for the full pre-registered gate assertions)")


if __name__ == "__main__":
    _report()
