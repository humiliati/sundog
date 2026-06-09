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
def make_crystal(aspect, a=1.0):
    """Hexagonal ice prism (inradius a, length L = aspect*2a). Returns body-frame
    (normals (8,3), offsets (8,), face_kind (8,))  with inside = {x : n_i.x <= offset_i}.
    Faces 0..5 = prism (normals at 60-deg steps, offset a); 6,7 = basal (+/-z, offset L/2).
    Body optic/c-axis = +z. aspect<1 = plate, aspect>1 = column."""
    L = aspect * 2.0 * a
    normals = np.zeros((8, 3))
    offsets = np.zeros(8)
    kind = np.zeros(8, int)                          # 0 prism, 1 basal
    for k in range(6):
        ang = np.radians(60 * k)
        normals[k] = [np.cos(ang), np.sin(ang), 0.0]
        offsets[k] = a
    normals[6] = [0.0, 0.0, 1.0]; offsets[6] = L / 2
    normals[7] = [0.0, 0.0, -1.0]; offsets[7] = L / 2
    kind[6] = kind[7] = 1
    return normals, offsets, kind


def face_areas(aspect, a=1.0):
    """Area of each face (prism rectangles a*L... edge=2a/sqrt3; basal hexagon (3*sqrt3/2)a²)."""
    L = aspect * 2.0 * a
    edge = 2.0 * a / np.sqrt(3.0)
    A = np.zeros(8)
    A[:6] = edge * L
    A[6] = A[7] = (3.0 * np.sqrt(3.0) / 2.0) * a * a
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
    fidx = int(rng.choice(8, p=w / tot))
    p_body = _point_on_face(fidx, kind, aspect, a, rng)
    p_world = R @ p_body
    return p_world, normals_w[fidx], tot


# ----------------------------------------------------------------------------- #
# the polarized ray-tree tracer                                                 #
# ----------------------------------------------------------------------------- #
def trace_tree(o, d_in, entry_normal, normals, offsets, c_axis, dn=0.0, size_um=100.0,
               K=1, eps=1e-4, n=N_ICE, lam=LAM):
    """Ray-march one photon entering at `o` along `d_in` (air->crystal through `entry_normal`) and
    return the list of exit (transmitted) deposits [(d_out_world(3,), S(4,), s_ref(3,)), ...].
    Carries the Stokes vector through entry-Fresnel, per-segment birefringence, inter-interface frame
    rotations, TIR-phase retarder / partial-reflection diattenuator, with intensity (energy) living in
    Stokes I. `K` = max internal reflections; `eps` = intensity prune. Deterministic given inputs."""
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
def orient_random(rng):
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


HABITS = {
    "random": (orient_random, 1.0),       # (sampler, aspect)
    "plate": (orient_plate, 0.4),
    "column": (orient_column, 3.0),
}


# ----------------------------------------------------------------------------- #
# ensemble + sky accumulation                                                   #
# ----------------------------------------------------------------------------- #
def run_ensemble(habit="random", e_deg=20.0, n_orient=4000, dn=0.0, size_um=100.0,
                 K=1, a=1.0, seed=0, sigma_deg=1.5):
    """Trace an ensemble of one habit. Returns a deposit array with columns
    [scatt_deg, az_deg, el_deg, I, Q, U, V] in the world/scattering frame, intensity-weighted by the
    orientation's projected area. Deterministic given seed."""
    sampler, aspect = HABITS[habit]
    normals_b, offsets, kind = make_crystal(aspect, a)
    areas = face_areas(aspect, a)
    e = np.radians(e_deg)
    d_sun = -np.array([np.cos(e), 0.0, np.sin(e)])    # propagation direction of sunlight
    rng = np.random.default_rng(seed)

    rows = []
    for _ in range(n_orient):
        R = sampler(rng, sigma_deg) if habit != "random" else sampler(rng)
        normals_w = normals_b @ R.T
        c_axis = R @ np.array([0.0, 0.0, 1.0])
        ent = sample_entry(R, normals_w, offsets, kind, areas, d_sun, aspect, a, rng)
        if ent is None:
            continue
        o0, n_entry, A_proj = ent
        deps = trace_tree(o0, d_sun, n_entry, normals_w, offsets, c_axis,
                          dn=dn, size_um=size_um, K=K, n=N_ICE, lam=LAM)
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
    print("\n(see test_s2_halo_raytracer.py for the full pre-registered gate assertions)")


if __name__ == "__main__":
    _report()
