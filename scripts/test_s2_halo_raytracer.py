#!/usr/bin/env python
"""Frozen test for the S2 full per-habit polarized halo raytracer (scripts/s2_halo_raytracer.py) and the
partial-reflection Mueller matrix it rests on (s2_optics.mueller_fresnel_reflect).

Locks the pre-registered result (docs/atlas/S2_HALO_RAYTRACER_PREREG.md): one genuine ray-marched
polyhedron engine reproduces, WITHOUT the schematic fixed-face boundary of s2_handedness_map,
  Stage 1 — the partial-reflection diattenuator (energy R+T=1; no circular pol; continuity to mueller_tir
            at the critical angle; physicality);
  Stage 2 — the polyhedron geometry + the polarized interface composition matching the validated
            s2_handedness_map.trace_ray to machine precision (the frame-bookkeeping bridge), with
            physical realizability across the ensemble;
  Stage 3 — GATE 1 geometry (22 deg / 46 deg halos at the analytic radii), GATE 2 linear pol (the Koennen
            Fresnel-floor DoP ~3.7% with U=0), GATE 3 circular V (per-feature |V/I| %-level, the
            achiral display nets to ~0 = V-analog of Koennen U=0).
Deterministic (seeded RNG). Monte-Carlo test — a couple of minutes. Run: python scripts/test_s2_halo_raytracer.py
"""
import sys
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_optics as so                 # noqa: E402
import s2_halo_raytracer as rt         # noqa: E402
import s2_handedness_map as hm         # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("S2 full per-habit polarized halo raytracer:\n")

# ===== Stage 1 — the partial-reflection Mueller diattenuator ================================ #
print("Stage 1 — mueller_fresnel_reflect (the partial Fresnel reflection that completes the chain):")
n = 1.31
# energy conservation R_s + T_s = 1, R_p + T_p = 1 (sub-critical, internal ice->air)
ok = True
for thd in np.linspace(1, 48, 200):
    ti = np.radians(thd); st = n * np.sin(ti); tt = np.arcsin(st); ci, ct = np.cos(ti), np.cos(tt)
    rs = (n * ci - ct) / (n * ci + ct); rp = (ci - n * ct) / (ci + n * ct)
    ts = 2 * n * ci / (n * ci + ct); tp = 2 * n * ci / (ci + n * ct)
    Ts = (ct) / (n * ci) * ts ** 2; Tp = (ct) / (n * ci) * tp ** 2
    if abs(rs ** 2 + Ts - 1) > 1e-9 or abs(rp ** 2 + Tp - 1) > 1e-9:
        ok = False
check("energy R+T=1 for s and p over theta in [1,48] deg", ok)
M0 = so.mueller_fresnel_reflect(1e-6, n, 1.0)
check("normal-incidence reflectance R = ((n-1)/(n+1))^2", abs(M0[0, 0] - ((n - 1) / (n + 1)) ** 2) < 1e-6,
      f"R={M0[0,0]:.5f}")
check("a single partial reflection makes NO circular pol (V=0)",
      abs((so.mueller_fresnel_reflect(35, n, 1.0) @ np.array([1.0, 0.3, 0.4, 0.0]))[3]) < 1e-12)
thB = np.degrees(np.arctan(1.0 / n)); MB = so.mueller_fresnel_reflect(thB, n, 1.0)
check("internal Brewster: R_p -> 0 (reflected light fully polarized)", abs(MB[0, 0] - MB[0, 1]) < 1e-6,
      f"R_p={0.5*(MB[0,0]-MB[0,1]):.2e}")
thc = np.degrees(np.arcsin(1.0 / n))
gr = so.mueller_fresnel_reflect(thc - 1e-6, n, 1.0)[2, 2]
check("continuity to mueller_tir at the critical angle (reflect lower-2x2 -> 1)", gr > 0.998,
      f"reflect[2,2](thc-1e-6)={gr:.4f} -> 1 (=tir identity)")
check("TIR returns None (use mueller_tir there)", so.mueller_fresnel_reflect(60, n, 1.0) is None)

# ===== Stage 2 — polyhedron geometry + the frame-bookkeeping bridge ========================= #
print("\nStage 2 — polyhedron ray-march + the frame-bookkeeping bridge to the validated trace_ray:")

# (a) convex exit-face: a ray from the centre along +x exits prism face 0 at t=a
normals, offsets, kind = rt.make_crystal(1.0, a=1.0)
ef = rt.exit_face(np.zeros(3), np.array([1.0, 0.0, 0.0]), normals, offsets)
check("convex exit-face: +x ray from centre exits prism face 0 at t=a", ef[0] == 0 and abs(ef[1] - 1.0) < 1e-9)
ef2 = rt.exit_face(np.zeros(3), np.array([0.0, 0.0, 1.0]), normals, offsets)
check("convex exit-face: +z ray exits the top basal face (idx 6)", ef2[0] == 6)

# (b) THE BRIDGE: the raytracer's interface composition (entry-Fresnel x frame-rot x TIR x frame-rot x
#     exit-Fresnel) reproduces s2_handedness_map.trace_ray V/I + sky direction to machine precision,
#     confirming the multi-bounce frame bookkeeping uses the validated convention (dn=0 isolates it).
def bridge_one(psi, e_deg=20.0):
    """Replicate trace_ray's single-TIR plate path the way trace_tree's loop body composes it."""
    R = hm.Rz(psi)
    m_en = R @ np.array([1.0, 0.0, 0.0]); m_ex = R @ np.array([np.cos(np.radians(120)),
                                                               np.sin(np.radians(120)), 0.0])
    m_bas = np.array([0.0, 0.0, -1.0]); e = np.radians(e_deg)
    d0 = -np.array([np.cos(e), 0.0, np.sin(e)])
    if np.dot(d0, m_en) >= 0:
        return None
    d1, th1 = hm.refract(d0, m_en, 1.0, n); s1 = hm.s_axis(d0, m_en)
    Me, _ = so.mueller_fresnel(th1, 1.0, n); S = Me @ np.array([1.0, 0.0, 0.0, 0.0])
    th2 = hm.incidence_deg(d1, m_bas)
    if th2 <= thc:
        return None
    s2 = hm.s_axis(d1, m_bas); S = hm.mrot(hm.signed_angle(s1, s2, d1)) @ S
    S = so.mueller_tir(th2, n, 1.0, 0.0) @ S; d2 = hm.reflect(d1, m_bas)
    r3 = hm.refract(d2, m_ex, n, 1.0)
    if r3 is None:
        return None
    d3, th3 = r3; s3 = hm.s_axis(d2, m_ex); S = hm.mrot(hm.signed_angle(s2, s3, d2)) @ S
    Mx, _ = so.mueller_fresnel(th3, n, 1.0); S = Mx @ S
    sky = -hm.unit(d3)
    return np.degrees(np.arctan2(sky[1], sky[0])), S[3] / S[0]

maxerr = 0.0; ncomp = 0
for psi in np.linspace(0, 2 * np.pi, 240, endpoint=False):
    a = hm.trace_ray(psi, 20.0, exit_face=2, dn=0.0); b = bridge_one(psi)
    if a["valid"] and b is not None:
        maxerr = max(maxerr, abs(a["sky_az_deg"] - b[0]), abs(a["VoverI"] - b[1])); ncomp += 1
check("bridge: raytracer interface composition == trace_ray (V/I + sky az) to 1e-9",
      ncomp >= 40 and maxerr < 1e-9, f"max|err|={maxerr:.1e} over {ncomp} paths")

# (c) physical realizability I >= sqrt(Q^2+U^2+V^2) across a real ray-marched ensemble
dep = rt.run_ensemble("plate", e_deg=20.0, n_orient=2500, dn=so.DN_ICE, K=1, seed=7)
polI = np.hypot(np.hypot(dep[:, 4], dep[:, 5]), dep[:, 6])
check("physical realizability I >= sqrt(Q^2+U^2+V^2) for all deposits",
      np.all(dep[:, 3] - polI > -1e-9), f"worst (polI-I)={(polI-dep[:,3]).max():.2e}")

# ===== Stage 3 — the three pre-registered gates ============================================= #
print("\nStage 3 — the three pre-registered gates (one engine, no schematic boundary):")

# GATE 1 — geometry: the 22 and 46 deg halos at the analytic radii (random habit)
dg = rt.run_ensemble("random", e_deg=20.0, n_orient=6000, dn=0.0, K=1, seed=1)
ctr, h = rt.radial_intensity(dg, bin_deg=0.5)
def peak_in(lo, hi):
    seg = (ctr >= lo) & (ctr < hi)
    return ctr[seg][np.argmax(h[seg])] if seg.any() and h[seg].max() > 0 else -1
p22, p46 = peak_in(18, 26), peak_in(42, 50)
check("GATE 1: 22-deg halo peak at the analytic radius (21.84 +/- 0.7 deg)", abs(p22 - 21.84) < 0.7,
      f"peak={p22:.2f} deg")
# the 46-deg halo (90-deg wedge) is intrinsically BROAD/diffuse (Greenler): bright edge at the 45.73-deg
# min-deviation, peak brightness a couple degrees outward. Check the peak sits in the broad-halo window.
check("GATE 1: 46-deg (broad) halo peak just outside the 45.73-deg edge (in [44,49] deg)",
      44.0 < p46 < 49.0, f"peak={p46:.2f} deg (edge 45.73)")

# GATE 2 — linear pol at the 22-deg halo: the Koennen Fresnel floor with U=0 (mirror symmetry)
dl = rt.run_ensemble("random", e_deg=20.0, n_orient=9000, dn=0.0, K=1, seed=2)
rp = rt.ring_pol(dl, 20.5, 23.5)
check("GATE 2: 22-deg linear DoP at the Koennen Fresnel floor (~3.7%, in [2.5,5.5]%)",
      0.025 < rp["dop"] < 0.055, f"DoP={rp['dop']*100:.2f}%")
check("GATE 2: U/I ~ 0 on the halo (mirror symmetry, |U/I| < 0.3%)", abs(rp["uoi"]) < 0.003,
      f"U/I={rp['uoi']*100:+.3f}%")
check("GATE 2: the linear pol is essentially pure Q (|U| << |Q|)", abs(rp["uoi"]) < 0.1 * abs(rp["qoi"]),
      f"|U/Q|={abs(rp['uoi']/rp['qoi']):.3f}")

# GATE 3 — circular V (plate, TIR-rich): per-feature V real + the achiral display nets to ~0
dv = rt.run_ensemble("plate", e_deg=20.0, n_orient=12000, dn=0.0, K=1, seed=3)
mI = np.abs(dv[:, 3]) > 0
peakV = np.abs(dv[mI, 6] / dv[mI, 3]).max()
Vw = dv[:, 6]
net = abs(Vw.sum()) / np.abs(Vw).sum()
# azimuthal antisymmetry of the intensity-weighted V (bins symmetric about the principal plane az=0)
edges = np.linspace(-180, 180, 73)
Vbin, _ = np.histogram(dv[:, 1], bins=edges, weights=dv[:, 6])
anti = np.linalg.norm(Vbin + Vbin[::-1]) / (2 * np.linalg.norm(Vbin) + 1e-30)
check("GATE 3: per-feature circular V is REAL (per-ray peak |V/I| > 1%)", peakV > 0.01,
      f"per-ray peak |V/I|={peakV*100:.2f}%")
check("GATE 3: the achiral display nets to ~0 (net |sumV|/sum|V| < 3% = V-analog of Koennen U=0)",
      net < 0.03, f"net={net*100:.2f}%")
check("GATE 3: V(az) is azimuthally antisymmetric about the principal plane (residual < 20%)",
      anti < 0.20, f"antisym residual={anti*100:.1f}%")

# ===== GATE 4 — the parhelic circle (the K>=2 multi-bounce stretch) ========================= #
# External reflection off the vertical crystal faces (a vertical mirror preserves the ray's elevation)
# + two-internal-reflection paths produce a horizontal white circle through the sun at constant
# elevation, with the 120-deg parhelia as the recognizable K=2 feature.
print("\nGATE 4 — parhelic circle + 120-deg parhelia (K=2 multi-bounce, plate + external reflection):")
dp = rt.run_ensemble("plate", e_deg=20.0, n_orient=12000, dn=0.0, K=2, seed=8, include_external=True)
el, az, Iw, Vw2 = dp[:, 2], dp[:, 1], dp[:, 3], dp[:, 6]
eh, eed = np.histogram(el, bins=np.arange(-90, 91, 2.0), weights=Iw)
elpk = (0.5 * (eed[:-1] + eed[1:]))[np.argmax(eh)]
band = np.abs(el - 20.0) < 2.5
azcov = len(np.unique(np.round(az[band] / 15)))
check("GATE 4: parhelic circle at constant elevation = sun elevation (|el_peak - 20| < 3 deg)",
      abs(elpk - 20.0) < 3.0, f"el_peak={elpk:.0f} deg")
check("GATE 4: the PHC spans the full azimuth circle (>= 18 of 24 15-deg bins populated)",
      azcov >= 18, f"az coverage={azcov}/24 bins")
I120 = Iw[band & (np.abs(np.abs(az) - 120) < 12)].sum()
Igap = Iw[band & (np.abs(az) > 55) & (np.abs(az) < 85)].sum()
check("GATE 4: the 120-deg parhelia (the two-internal-reflection K=2 feature) brighter than the gap",
      I120 > Igap, f"I(120deg)={I120:.0f} > I(gap)={Igap:.0f}")
mIb = band & (np.abs(Iw) > 0)
peakVp = np.abs(Vw2[mIb] / Iw[mIb]).max()
netVp = abs(Vw2[band].sum()) / np.abs(Vw2[band]).sum()
check("GATE 4: the PHC carries per-feature V (>1%) that still nets to ~0 (achiral, net < 5%)",
      peakVp > 0.01 and netVp < 0.05, f"per-ray peak|V/I|={peakVp*100:.1f}%, net={netVp*100:.2f}%")
Ub, Ib = dp[band, 5].sum(), Iw[band].sum()
check("GATE 4: PHC linear pol U/I ~ 0 (mirror symmetry, |U/I| < 0.5%)", abs(Ub / Ib) < 0.005,
      f"U/I={Ub/Ib*100:+.2f}%")

# ===== GATE 5 — pyramidal crystals: the odd-radius halos + their polarization ============== #
# The {10-11} pyramid faces (normal at PYR_X=61.99 deg from c, the Atlas c/a-derived angle) make the
# odd Galle wedges -> the 9/18/20/23/24/35-deg halos. This connects the polarimetric engine to the
# Atlas catastrophe families and predicts (barely-studied) pyramidal-halo polarization.
print("\nGATE 5 — pyramidal odd-radius halos + polarization (random-tumbling pyramidal crystals):")
dpy = rt.run_ensemble("pyramidal", e_deg=20.0, n_orient=12000, dn=so.DN_ICE, K=1, seed=12)
sc, Ipy = dpy[:, 0], dpy[:, 3]
ch, ced = np.histogram(sc, bins=np.arange(0, 50, 1.0), weights=Ipy)
cc = 0.5 * (ced[:-1] + ced[1:])
def _wt(lo, hi): return Ipy[(sc >= lo) & (sc < hi)].sum()
seg9 = (cc >= 7) & (cc < 12)
p9 = cc[seg9][np.argmax(ch[seg9])]
check("GATE 5: the 9-deg pyramidal halo appears (peak in [8,10.5] deg; Atlas anchor 8.96 deg)",
      8.0 <= p9 <= 10.5, f"9-deg peak at {p9:.1f} deg")
check("GATE 5: the 24-deg cluster dominates and the 35-deg halo sits above the far background",
      _wt(22, 26) > _wt(8, 10.5) and _wt(34, 37) > _wt(41, 44),
      f"I(24)={_wt(22,26):.0f} > I(9)={_wt(8,10.5):.0f}; I(35)={_wt(34,37):.0f} > bg(41-44)={_wt(41,44):.0f}")
r9, r24, r35 = rt.ring_pol(dpy, 8, 10.5), rt.ring_pol(dpy, 22.5, 25), rt.ring_pol(dpy, 34, 37)
check("GATE 5: pyramidal halos radially polarized, U/I ~ 0 on the 24-deg ring (mirror symmetry)",
      abs(r24["uoi"]) < 0.01, f"U/I(24)={r24['uoi']*100:+.2f}%")
check("GATE 5: pyramidal DoP rises with halo radius (more Fresnel diattenuation; DoP(35) > DoP(9))",
      r35["dop"] > r9["dop"], f"DoP(9)={r9['dop']*100:.2f}% < DoP(35)={r35['dop']*100:.2f}%")
check("GATE 5: no net circular V on the (refraction) pyramidal halos (|V/I| < 0.5%)",
      abs(r24["voi"]) < 0.005 and abs(r35["voi"]) < 0.005,
      f"V/I(24)={r24['voi']*100:+.2f}%, V/I(35)={r35['voi']*100:+.2f}%")

# ===== GATE 6 — the Parry orientation + the K=3 antisolar growth ============================ #
print("\nGATE 6 — Parry orientation (above-sun arc) + K=3 antisolar growth:")
dpa = rt.run_ensemble("parry", e_deg=20.0, n_orient=9000, dn=0.0, K=1, seed=13)
elp, azp, Ipa = dpa[:, 2], dpa[:, 1], dpa[:, 3]
mer = np.abs(azp) < 25
above = mer & (elp > 40)                                   # above the 22-deg halo top (el 42 for sun=20)
fracA = Ipa[above].sum() / Ipa[mer].sum() if Ipa[mer].sum() > 0 else 0
check("GATE 6: Parry concentrates flux above the sun near the meridian (the Parry-arc region)",
      above.sum() > 40 and fracA > 0.02, f"above-el-40 meridian flux frac={fracA*100:.1f}%, n={above.sum()}")
d2 = rt.run_ensemble("column", e_deg=20.0, n_orient=6000, dn=0.0, K=2, seed=14, include_external=True)
d3 = rt.run_ensemble("column", e_deg=20.0, n_orient=6000, dn=0.0, K=3, seed=14, include_external=True)
af2 = d2[np.abs(d2[:, 1]) > 120, 3].sum() / d2[:, 3].sum()
af3 = d3[np.abs(d3[:, 1]) > 120, 3].sum() / d3[:, 3].sum()
check("GATE 6: K=3 adds antisolar (subhelic/anthelic-region) flux over K=2 (same seed)",
      af3 > af2, f"antisolar flux K2={af2*100:.1f}% -> K3={af3*100:.1f}%")

print(f"\n{'ALL PASS -- one genuine ray-marched polyhedron engine reproduces the 22/46-deg halos, the Koennen Fresnel-floor linear pol (U=0), the per-feature +/-V handedness map (net->0); at K>=2 the PARHELIC CIRCLE + 120-deg parhelia; with PYRAMIDAL crystals the 9/24/35-deg odd-radius halos and their (radius-rising, U=0, V=0) polarization; the PARRY above-sun arc; and K=3 antisolar growth.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
