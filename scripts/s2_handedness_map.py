#!/usr/bin/env python
"""S2 deepening — the per-feature +/-V(phi) handedness map (the owed in-house Stage-C deliverable).

Closes the pre-registered prediction in docs/atlas/S2_MEASURED_SKY_SCOPE.md ("per-feature +/-V ~1%
antisymmetric around the ring, integrating to ~0") that the scope NAMES but never computed.

What was missing (now fixed in s2_optics): the transmission-only Mueller chain returned ZERO for any
total-internal-reflection ray, so it was blind to the TIR PHASE retardance (the Fresnel-rhomb s-p phase)
— the PRIMARY linear->circular mechanism on the TIR-rich features. tir_retardance() adds it.

This script forward-models V(phi)/I around a TIR-rich path (entry-refract -> TIR bounce -> exit-refract)
through a hexagonal ice plate spinning about the vertical, with a GENUINE 3D polarization ray trace
(vector Snell + the validated Fresnel / TIR / birefringence Mueller matrices + proper inter-interface
frame rotations). Hexagonal ice is ACHIRAL (sigma_v mirror planes), so the crystal supports a ray path
AND its mirror partner with equal weight; including both (as physics requires) makes the total map
V_tot(phi) = V_P(phi) - V_P(-phi) emerge ANTISYMMETRIC -> net integral 0. We report BOTH the single
chiral sub-path (nonzero net) and the achiral total (zero net), so the cancellation is shown to be the
V-analog of Koennen & Tinbergen 1991's measured U=0, not an imposed assumption.

Forward-model tier only. NOT a per-habit raytracer, NOT a measured-sky detection. NOT public-eligible.
Run:  python scripts/s2_handedness_map.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_optics as so  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

N_ICE = so.N_ICE
DN_ICE = so.DN_ICE
LAM = so.LAM_LIGHT
THETA_C = np.degrees(np.arcsin(1.0 / N_ICE))   # ice critical angle ~49.76 deg


# ----------------------------------------------------------------------------- #
# 3D geometry helpers                                                           #
# ----------------------------------------------------------------------------- #
def unit(v):
    v = np.asarray(v, float)
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def Rz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def refract(d, nrm, n1, n2):
    """Vector Snell. d = incident propagation unit; nrm = surface normal (auto-oriented against d).
    Returns (refracted_unit, incidence_angle_deg) or None on total internal reflection."""
    d = unit(d); nrm = unit(nrm)
    cosi = -np.dot(d, nrm)
    if cosi < 0:                       # normal on the same side as d -> flip to face the ray
        nrm = -nrm; cosi = -np.dot(d, nrm)
    eta = n1 / n2
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    if k < 0.0:
        return None                    # TIR
    t = eta * d + (eta * cosi - np.sqrt(k)) * nrm
    return unit(t), np.degrees(np.arccos(np.clip(cosi, -1, 1)))


def reflect(d, nrm):
    d = unit(d); nrm = unit(nrm)
    return unit(d - 2.0 * np.dot(d, nrm) * nrm)


def incidence_deg(d, nrm):
    return np.degrees(np.arccos(np.clip(abs(np.dot(unit(d), unit(nrm))), -1, 1)))


def s_axis(d, nrm):
    """s-polarization axis (perpendicular to the plane of incidence) = unit(d x nrm)."""
    s = np.cross(unit(d), unit(nrm))
    nn = np.linalg.norm(s)
    if nn < 1e-9:                      # near-normal incidence: plane of incidence undefined
        # pick any axis perpendicular to d
        a = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, unit(d))) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        s = np.cross(unit(d), a)
    return unit(s)


def signed_angle(a, b, axis):
    """Signed angle from a to b about axis (right-hand), with a,b projected into the plane ⟂ axis."""
    u = unit(axis)
    a = unit(a - np.dot(a, u) * u)
    b = unit(b - np.dot(b, u) * u)
    x = np.clip(np.dot(a, b), -1, 1)
    y = np.dot(np.cross(a, b), u)
    return np.arctan2(y, x)


def mrot(chi):
    """Mueller reference-frame rotation by angle chi (rotates the Stokes Q/U basis)."""
    c, s = np.cos(2 * chi), np.sin(2 * chi)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c,   s,   0.0],
        [0.0, -s,  c,   0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


# ----------------------------------------------------------------------------- #
# the single-ray polarization trace: entry-refract -> biref -> TIR -> exit-refract
# ----------------------------------------------------------------------------- #
def trace_ray(psi, e_deg, entry_face=0, exit_face=2, basal=-1,
              n=N_ICE, dn=DN_ICE, L_um=120.0, lam=LAM):
    """Trace one ray through a hexagonal ice plate (basal horizontal, c-axis vertical) spun by psi
    about the vertical. Path: enter side face `entry_face`, TIR off the `basal` basal face, exit side
    face `exit_face`. Returns dict(valid, sky_az_deg, sky_el_deg, I, V, VoverI, th_tir) or
    dict(valid=False, ...). `dn=0` isolates the pure-TIR mechanism (birefringence off)."""
    R = Rz(psi)
    m_entry = R @ np.array([np.cos(np.radians(60 * entry_face)),
                            np.sin(np.radians(60 * entry_face)), 0.0])
    m_exit = R @ np.array([np.cos(np.radians(60 * exit_face)),
                           np.sin(np.radians(60 * exit_face)), 0.0])
    m_basal = np.array([0.0, 0.0, float(basal)])      # vertical, unchanged by spin about z
    c_axis = np.array([0.0, 0.0, 1.0])                # ice optic axis = plate normal (vertical)

    # sunlight propagation direction (photons travel away from the sun)
    e = np.radians(e_deg)
    d0 = -np.array([np.cos(e), 0.0, np.sin(e)])

    bad = {"valid": False, "sky_az_deg": np.nan, "sky_el_deg": np.nan,
           "I": 0.0, "V": 0.0, "VoverI": 0.0, "th_tir": np.nan}

    # entry face must face the sun (outward normal opposes incoming propagation)
    if np.dot(d0, m_entry) >= 0:
        return bad
    r1 = refract(d0, m_entry, 1.0, n)
    if r1 is None:
        return bad
    d1, th1 = r1
    s1 = s_axis(d0, m_entry)
    Men = so.mueller_fresnel(th1, 1.0, n)
    if Men is None:
        return bad
    S = Men[0] @ np.array([1.0, 0.0, 0.0, 0.0])       # unpolarized in (frame-independent)

    # birefringence over the internal path (fast axis = projected optic axis)
    if dn > 0:
        cperp = c_axis - np.dot(c_axis, d1) * d1
        if np.linalg.norm(cperp) > 1e-6:
            phib = signed_angle(s1, cperp, d1)
            delta_b = 2.0 * np.pi * dn * L_um / lam
            S = so.mueller_retarder(delta_b, phib) @ S

    # TIR off the basal face
    th2 = incidence_deg(d1, m_basal)
    if th2 <= THETA_C:                                # not a TIR ray for this path
        return bad
    s2 = s_axis(d1, m_basal)
    S = mrot(signed_angle(s1, s2, d1)) @ S
    S = so.mueller_tir(th2, n, 1.0, 0.0) @ S          # fast axis = s2 (phi=0 in this frame)
    d2 = reflect(d1, m_basal)

    # exit side face (must refract OUT, i.e. not a second TIR)
    r3 = refract(d2, m_exit, n, 1.0)
    if r3 is None:
        return bad
    d3, th3 = r3
    s3 = s_axis(d2, m_exit)
    S = mrot(signed_angle(s2, s3, d2)) @ S
    Mex = so.mueller_fresnel(th3, n, 1.0)
    if Mex is None:
        return bad
    S = Mex[0] @ S

    sky = -unit(d3)                                   # apparent direction = where light comes FROM
    az = np.degrees(np.arctan2(sky[1], sky[0]))       # azimuth relative to the sun (principal plane=0)
    el = np.degrees(np.arcsin(np.clip(sky[2], -1, 1)))
    I = float(S[0]); V = float(S[3])
    return {"valid": True, "sky_az_deg": az, "sky_el_deg": el,
            "I": I, "V": V, "VoverI": (V / I if I > 0 else 0.0), "th_tir": th2}


# ----------------------------------------------------------------------------- #
# the sweep: V(phi) around the feature for a chiral sub-path and the achiral total
# ----------------------------------------------------------------------------- #
def sweep(e_deg=20.0, exit_face=2, dn=DN_ICE, L_um=120.0, n_psi=2880,
          az_bins=72, el_window=12.0):
    """Spin the plate over psi in [0,2pi); collect (sky_az, V*I_weight) for the chiral path P
    (entry 0 -> TIR basal -> exit `exit_face`) and its MIRROR partner P' (exit -`exit_face`), which the
    achiral hexagonal crystal supports with equal weight. Bin intensity-weighted V by azimuth.
    Returns the az grid + V profiles (chiral P, mirror P', achiral total) and summary metrics."""
    psis = np.linspace(0.0, 2.0 * np.pi, n_psi, endpoint=False)
    edges = np.linspace(-180.0, 180.0, az_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # auto-center the elevation band on the feature's own elevation (the basal-TIR arc sits at
    # el = -e_deg, a sub-horizon arc, not near 0) so the +/-V map is read on the actual feature.
    all_el = [trace_ray(p, e_deg, exit_face=exit_face, dn=dn, L_um=L_um)["sky_el_deg"]
              for p in psis]
    all_el = [x for x in all_el if np.isfinite(x)]
    el_center = float(np.median(all_el)) if all_el else 0.0

    def collect(xfaces):
        """Sum intensity-weighted V and I over a list of exit faces (the physical exit set).
        Also returns the max single-ray |V/I| on the feature (the per-ray mechanism strength)."""
        Vnum = np.zeros(az_bins)        # sum of V (intensity-weighted)
        Iden = np.zeros(az_bins)        # sum of I
        peak_ray = 0.0
        for xface in xfaces:
            for psi in psis:
                r = trace_ray(psi, e_deg, exit_face=xface, dn=dn, L_um=L_um)
                if not r["valid"] or abs(r["sky_el_deg"] - el_center) > el_window:
                    continue
                b = int(np.clip(np.digitize(r["sky_az_deg"], edges) - 1, 0, az_bins - 1))
                Vnum[b] += r["V"]
                Iden[b] += r["I"]
                peak_ray = max(peak_ray, abs(r["VoverI"]))
        return Vnum, Iden, peak_ray

    # chiral SUB-path: a single exit face (one-handed sub-ensemble) vs the achiral FULL crystal
    # (sum over ALL physically-valid exit faces 1..5 of the hexagonal plate — a mirror-closed set,
    # so its antisymmetry EMERGES from the physical exit enumeration, not a hand-picked mirror pair).
    Vp, Ip, peak_ray_P = collect([exit_face])
    Vt, It, peak_ray_tot = collect([1, 2, 3, 4, 5])

    with np.errstate(divide="ignore", invalid="ignore"):
        prof_P = np.where(Ip > 0, Vp / Ip, 0.0)              # V/I for the chiral sub-path alone
        prof_M = prof_P[::-1]                                # its mirror (for reference)
        Itot = It
        prof_tot = np.where(Itot > 0, Vt / Itot, 0.0)        # achiral full-crystal V/I

    # metrics — all INTENSITY-WEIGHTED (the physical flux map), so near-empty edge bins with a large
    # but weightless V/I do not contaminate them.
    Vt_binned = prof_tot * Itot              # intensity-weighted V per azimuth bin (= summed V)
    Vp_binned = prof_P * Ip

    def net_ratio(Vbinned):                  # |integral V| / integral |V| over the feature
        den = np.sum(np.abs(Vbinned))
        return float(np.abs(np.sum(Vbinned)) / den) if den > 0 else 0.0

    def antisym_residual(Vbinned):           # symmetric-part energy / total, on the flux-weighted map
        rev = Vbinned[::-1]                  # az -> -az (bin centers are symmetric about 0)
        a = np.linalg.norm(Vbinned) + 1e-30
        return float(np.linalg.norm(Vbinned + rev) / (2 * a))

    # peak |V/I| only over bins carrying real flux (>1% of the max bin intensity)
    def peak_voi(prof, Iw):
        m = Iw > 0.01 * (Iw.max() if Iw.max() > 0 else 1.0)
        return float(np.max(np.abs(prof[m]))) if m.any() else 0.0

    peak_P = peak_voi(prof_P, Ip)
    peak_tot = peak_voi(prof_tot, Itot)
    return {
        "az": centers, "prof_P": prof_P, "prof_M": prof_M, "prof_tot": prof_tot,
        "I_P": Ip, "I_tot": Itot, "el_center": el_center,
        "peak_VoverI_chiral": peak_P,
        "peak_VoverI_total": peak_tot,
        "peak_ray_VoverI": peak_ray_tot,
        "net_ratio_chiral": net_ratio(Vp_binned),
        "net_ratio_total": net_ratio(Vt_binned),
        "antisym_residual_total": antisym_residual(Vt_binned),
        "n_az_populated": int(np.sum(Itot > 0)),
    }


def _print_report():
    print("=" * 78)
    print("S2 deepening — per-feature +/-V(phi) handedness map")
    print("=" * 78)
    print(f"ice n={N_ICE}, dn={DN_ICE}, critical angle theta_c={THETA_C:.2f} deg, lambda={LAM} um\n")

    # Stage-1 sanity: the TIR retardance at its analytic max
    th_max = np.degrees(np.arcsin(np.sqrt(2 * (1 / N_ICE) ** 2 / (1 + (1 / N_ICE) ** 2))))
    d_max = np.degrees(so.tir_retardance(th_max, N_ICE, 1.0))
    print(f"[mechanism] ice TIR retardance peak  delta_max = {d_max:.2f} deg @ theta = {th_max:.2f} deg"
          f"  (analytic 30.56 deg @ 59.1 deg)\n")

    for tag, dn in (("TIR-only (dn=0)", 0.0), ("TIR + birefringence", DN_ICE)):
        r = sweep(dn=dn)
        print(f"--- {tag} ---")
        print(f"  feature elevation (basal-TIR arc)     : {r['el_center']:.1f} deg")
        print(f"  populated azimuth bins on the feature : {r['n_az_populated']}")
        print(f"  Claim A  per-ray peak |V/I| (mechanism) : {r['peak_ray_VoverI']*100:.2f}%"
              f"   (a single contributing ray)")
        print(f"  Claim A  flux-avg feature |V/I| (observable): {r['peak_VoverI_total']*100:.2f}%"
              f"   (what a polarimeter sees)")
        print(f"  chiral sub-path net |∮V|/∮|V|          : {r['net_ratio_chiral']*100:.1f}%"
              f"   (a real one-handed sub-ensemble)")
        print(f"  Claim B  achiral TOTAL net |∮V|/∮|V|   : {r['net_ratio_total']*100:.2f}%"
              f"   (-> 0 = net handedness cancels)")
        print(f"  Claim B  antisymmetry residual (total) : {r['antisym_residual_total']*100:.2f}%"
              f"   (-> 0 = V(phi) odd about principal plane)\n")


if __name__ == "__main__":
    _print_report()
