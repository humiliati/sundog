#!/usr/bin/env python
"""S3-A3 — Moilanen-arc mechanism-class polarization discriminator (internal leg).

PREREG (binding, frozen 2026-06-11 BEFORE this file first ran):
  docs/atlas/MOILANEN_POL_DISCRIMINATOR_PREREG.md
Slate entry: S3-A3 of internal/slates/HYP_SLATE_3_EXTERNAL_ANCHORS_2026-06-11.md.

Apparatus: the FROZEN scripts/s2_optics.py Mueller chain ONLY (untouched here), composed into
  - wedge_chain(apex, theta_i1, n): signed two-refraction prism chain (off min-deviation capable);
  - the LEF (Lefaudeux 2010) composite path: apex=30, entry face VERTICAL, theta_i1 = -e (signed);
  - REFL-P(b)/REFL-TIR(b) envelope columns on the W-34 min-deviation backbone.

Stages (prereg 4): 0 gate (Konnen anchors at apex=60)  1 floor law  2 class table (LEF FIRST)
  3 kill-gate smear sweep (solar disk 0.53 deg FWHM on EVERY cell)  4 track co-predictions.
Pure deterministic numpy/scipy — no RNG anywhere; byte-identical reruns required by the frozen
test (scripts/test_moilanen_pol_discriminator.py).

Sign convention (prereg section 2): signed DoLP q = -Q/I in the chain s/p frame;
q > 0 = radial (p-dominant), q < 0 = tangential (s-dominant).
"""
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 console robustness

sys.path.insert(0, "scripts")
import s2_optics as so

# ---- pinned constants (prereg section 3) ----------------------------------- #
N = so.N_ICE                      # 1.31
N_O, N_E = 1.3090, 1.3104         # ice o/e indices @ ~590 nm (s2_konnen_validate constants)
APEX_W = 34.0                     # the atoptics/Tape simulation wedge
APEX_LEF = 30.0                   # Lefaudeux composite effective prism (Ursa post, source-pinned)
FLOOR = 0.005                     # 0.5% DoLP-equivalent separation floor (campaign floors doc)
B_INERT = 5.0                     # deg; a-priori optically-inert bounce bound (prereg 3.4)
DISK_FWHM = 0.53                  # deg; solar-disk convolution on EVERY cell (audit fix)
W0 = 0.10                         # deg; pinned intrinsic arc-ridge FWHM
SIGMA_REL = 0.01                  # pinned background photometric RMS
C0_SPAN = (0.05, 0.10, 0.30)      # conservative unsmeared-contrast span (explicit grid axis)
K_SPAN = (2.0, 3.0, 5.0)          # visibility threshold multipliers (3 primary)
WOBBLE_IN = (0.1, 0.3, 0.5, 1.0)  # deg sigma, in-box
PSF_IN = (0.02, 0.05, 0.10)       # deg FWHM, in-box
WOBBLE_OUT = (2.0, 3.0)           # out-of-box diagnostics (validity limits, never verdicts)
PSF_OUT = (0.30,)
B_GRID = np.concatenate([np.arange(0.5, 5.01, 0.5), np.arange(6.0, 49.51, 1.0)])  # REFL-P bounce grid
B_TIR = np.arange(50.0, 88.01, 2.0)
PHI_TIR = np.linspace(0.0, np.pi, 64, endpoint=False)
UNPOL = np.array([1.0, 0.0, 0.0, 0.0])
LEF_E = (0.0, 10.0, 15.0, 20.0, 25.0)
LEF_PUB = {0.0: 11.0, 10.0: 13.0, 15.0: 15.0, 20.0: 18.0}     # published track (Ursa post)
TRACK_D = (11.04, 13.0, 15.0, 18.0)

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


# ---- the apex-parameterized two-refraction chain (signed; prereg 2.1) ------- #
def wedge_chain(apex_deg, ti1_deg, n):
    """Two-refraction prism Mueller chain at SIGNED entry incidence ti1 [deg].
    Prism constraint (signed): theta_i2 = apex - theta_t1. Returns dict or None past TIR.
    Fresnel matrices are even in incidence sign (same plane of incidence throughout -> s/p
    frames aligned, no rotation; U stays identically 0 -> control C2)."""
    ent = so.mueller_fresnel(abs(ti1_deg), 1.0, n)
    if ent is None:
        return None
    M1, tt1_abs = ent
    tt1 = float(np.sign(ti1_deg) * tt1_abs) if ti1_deg != 0 else 0.0
    ti2 = apex_deg - tt1                              # signed internal incidence, exit face
    ext = so.mueller_fresnel(abs(ti2), n, 1.0)
    if ext is None:
        return None                                   # exit-face TIR -> path dark
    M2, tt2_abs = ext
    tt2 = float(np.sign(ti2) * tt2_abs)
    D = (ti1_deg - tt1) + (tt2 - ti2)                 # total deviation, signed chain
    M = M2 @ M1
    S = M @ UNPOL
    q = -S[1] / S[0]                                  # signed DoLP; >0 radial
    return {"M": M, "D": float(D), "q": float(q), "u": float(S[2] / S[0]),
            "v": float(S[3] / S[0]), "ti2": float(ti2), "tt2": float(tt2)}


def min_dev_entry(apex_deg, n):
    return float(np.degrees(np.arcsin(n * np.sin(np.radians(apex_deg / 2.0)))))


# ---- REFL envelope columns on the W-34 min-deviation backbone --------------- #
def _backbone(n=N):
    ti1 = min_dev_entry(APEX_W, n)
    Me, tt1 = so.mueller_fresnel(ti1, 1.0, n)
    Mx, _ = so.mueller_fresnel(APEX_W - tt1, n, 1.0)
    return Me, Mx


def refl_partial_q(b_deg, n=N):
    """Signed q of entry-Fresnel x internal partial bounce(b) x exit-Fresnel (prereg 2, REFL-P)."""
    Me, Mx = _backbone(n)
    R = so.mueller_fresnel_reflect(b_deg, n, 1.0)
    if R is None:
        return np.nan
    S = Mx @ R @ Me @ UNPOL
    return float(-S[1] / S[0])


def refl_tir_envelope(n=N):
    """Max-over-(b, phi) |V/I| and q range for one TIR bounce on the backbone (ENVELOPE, labeled)."""
    Me, Mx = _backbone(n)
    vmax, qlo, qhi = 0.0, np.inf, -np.inf
    for b in B_TIR:
        for phi in PHI_TIR:
            S = Mx @ so.mueller_tir(b, n, 1.0, phi) @ Me @ UNPOL
            v = abs(S[3] / S[0]); q = -S[1] / S[0]
            vmax = max(vmax, v); qlo = min(qlo, q); qhi = max(qhi, q)
    return float(vmax), float(qlo), float(qhi)


# ---- smear / visibility machinery (prereg 3.1-3.2) -------------------------- #
def total_fwhm(sigma_wobble, psf_fwhm):
    return float(np.sqrt(W0 ** 2 + DISK_FWHM ** 2 + (2.355 * sigma_wobble) ** 2 + psf_fwhm ** 2))


def dilution(sigma_wobble, psf_fwhm):
    return W0 / total_fwhm(sigma_wobble, psf_fwhm)


def visible(sigma_wobble, psf_fwhm, c0, k):
    return c0 * dilution(sigma_wobble, psf_fwhm) >= k * SIGMA_REL


def ledge_pair_separation(sigma_wobble, psf_fwhm, n_o=N_O, n_e=N_E, imask=0.1):
    """W-Ih vs W-iso smeared ledge-peak DoP excess (prereg 2.3 + AMENDMENT A1; konnen_validate
    profile recipe, eigenimage weights (1 +/- q_floor)/2 so the far-field DoP is exactly the
    Fresnel floor). AMENDMENT A1 (pre-run): the peak is taken over Ic >= imask * plateau — the
    dark-side Gaussian-tail ratio is a zero-flux artifact (P -> 1 at e^-8 intensity), not a
    measurable feature; imask=0.1 primary, 0.3 robustness column."""
    th_o = float(so.halo_min_deviation(APEX_W, n_o))
    th_e = float(so.halo_min_deviation(APEX_W, n_e))
    q_floor = float(so.halo_pol_dop((th_o + th_e) / 2.0))
    step = 0.0015                                     # deg; 34 samples across the 0.0508 split
    th = np.arange(th_o - 2.5, th_o + 3.5 + step / 2, step)
    Io = (th >= th_o).astype(float)
    Ie = (th >= th_e).astype(float)
    wo, we = 0.5 * (1 + q_floor), 0.5 * (1 - q_floor)
    I = wo * Io + we * Ie
    Q = wo * Io - we * Ie                              # o radial (+), e tangential (-)
    sig_samp = (total_fwhm(sigma_wobble, psf_fwhm) / 2.355) / step
    Ic = gaussian_filter1d(I, sig_samp, mode="nearest")
    Qc = gaussian_filter1d(Q, sig_samp, mode="nearest")
    P = np.divide(Qc, Ic, out=np.zeros_like(Qc), where=Ic > 1e-12)
    win = (th >= th_o - 1.0) & (th <= th_e + 1.5)      # adjudication window near the edge
    win &= Ic >= imask * Ic.max()                      # A1 intensity mask (measurable flux only)
    return float(P[win].max() - q_floor), q_floor


def lef_tir_cutoff(n=N):
    lo, hi = 20.0, 35.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if wedge_chain(APEX_LEF, -mid, n) is None:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def w34_track_solutions(D_target, n=N):
    """All (ti1, q) crossings of D(ti1) = D_target on a fine grid (both branches)."""
    sols, prev = [], None
    for ti in np.arange(-20.5, 89.5, 0.02):
        r = wedge_chain(APEX_W, float(ti), n)
        if r is None:
            prev = None
            continue
        if prev is not None:
            d0, d1 = prev[1] - D_target, r["D"] - D_target
            if d0 * d1 <= 0 and abs(d1 - d0) < 1.0:
                frac = abs(d0) / (abs(d0) + abs(d1) + 1e-300)
                ti_x = prev[0] + frac * (ti - prev[0])
                rx = wedge_chain(APEX_W, float(ti_x), n)
                if rx is not None:
                    sols.append((float(ti_x), rx["q"]))
        prev = (ti, r["D"])
    return sols


# ============================ registered run ================================= #
def main():
    print("=" * 78)
    print("S3-A3 Moilanen polarization discriminator — registered run (prereg frozen 2026-06-11)")
    print("=" * 78)

    # ---- STAGE 0 — gate ---- #
    print("\nSTAGE 0 — apex-60 Konnen calibration gate (CALIBRATION KILL if any fails):")
    r60 = wedge_chain(60.0, min_dev_entry(60.0, N), N)
    dmin60 = float(so.halo_min_deviation(60.0, N))
    check("min-deviation radius 21.84 +/- 0.01 deg", abs(dmin60 - 21.84) < 0.01, f"got {dmin60:.4f}")
    check("chain deviation == module min-deviation", abs(r60["D"] - dmin60) < 1e-9, f"D={r60['D']:.6f}")
    floor60 = float(so.halo_pol_dop(dmin60))
    check("Fresnel floor 3.71 +/- 0.10 abs%", abs(100 * r60["q"] - 3.71) < 0.10,
          f"chain q={100*r60['q']:.3f}%  halo_pol_dop={100*floor60:.3f}%")
    split60 = float(so.halo_min_deviation(60.0, N_E) - so.halo_min_deviation(60.0, N_O))
    check("o/e split 0.106 +/- 0.005 deg", abs(split60 - 0.106) < 0.005, f"got {split60:.4f}")
    check("U == 0 in-plane (control C2)", abs(r60["u"]) < 1e-12, f"|U/I|={abs(r60['u']):.1e}")
    check("V == 0 pure refraction", abs(r60["v"]) < 1e-12, f"|V/I|={abs(r60['v']):.1e}")
    R5 = so.mueller_fresnel_reflect(B_INERT, N, 1.0)
    db5 = float(R5[0, 1] / R5[0, 0])
    print(f"  [info] bounce-alone diattenuation at b_inert={B_INERT:.0f} deg: {100*abs(db5):.2f}% "
          f"(recorded per prereg 3.4, no tuning)")
    if fail:
        print("\nCALIBRATION KILL — gate failed; nothing downstream is read. Fix or withdraw.")
        return 1

    # ---- STAGE 1 — floor law ---- #
    print("\nSTAGE 1 — floor law q(theta_i1) >= DoP(D) at apex=34 (FLOOR-LAW KILL if violated):")
    dmin34 = float(so.halo_min_deviation(APEX_W, N))
    viol, n_ok, worst = 0, 0, np.inf
    for ti in np.concatenate([np.arange(-20.5, 0.0, 0.25), np.arange(0.0, 89.5, 0.25)]):
        r = wedge_chain(APEX_W, float(ti), N)
        if r is None:
            continue
        n_ok += 1
        margin = r["q"] - float(so.halo_pol_dop(r["D"]))
        worst = min(worst, margin)
        if margin < -1e-9:
            viol += 1
    check(f"floor law holds at all {n_ok} valid incidences", viol == 0,
          f"violations={viol}, worst margin={worst:.2e}")
    check("min-deviation anchor dmin(34) = 11.040 deg", abs(dmin34 - 11.040) < 0.005,
          f"got {dmin34:.4f}")
    if fail:
        print("\nFLOOR-LAW KILL — analytic floor claim retracted; rebuild refraction row numerically.")
        return 1

    # ---- STAGE 2 — class table, LEF FIRST ---- #
    print("\nSTAGE 2 — class table. LEF computed FIRST (binding order; prereg 2.2):")
    print("  LEF (Lefaudeux 2010 composite, apex=30, vertical entry face, theta_i1 = -e):")
    lef_rows = {}
    for e in LEF_E:
        r = wedge_chain(APEX_LEF, -e, N)
        lef_rows[e] = r
        pub = LEF_PUB.get(e)
        tag = f" (published ~{pub:.0f})" if pub is not None else ""
        print(f"    e={e:4.1f}:  D={r['D']:7.3f} deg{tag}   q={100*r['q']:+.3f}%  (radial)"
              f"   |V/I|={abs(r['v']):.1e}")
    for e, pub in LEF_PUB.items():
        check(f"LEF track consistency e={e:.0f} -> {pub:.0f} deg (+/-0.3)",
              abs(lef_rows[e]["D"] - pub) < 0.3, f"D={lef_rows[e]['D']:.3f}")
    e_cut = lef_tir_cutoff()
    check("LEF TIR cutoff ~26.3 deg (published 'higher than 26')", abs(e_cut - 26.3) < 0.3,
          f"e_cut={e_cut:.3f}")

    print("  W-34 refraction rows (min-deviation edge):")
    rW = wedge_chain(APEX_W, min_dev_entry(APEX_W, N), N)
    qW = rW["q"]
    split34 = float(so.halo_min_deviation(APEX_W, N_E) - so.halo_min_deviation(APEX_W, N_O))
    print(f"    W-Ih : D={rW['D']:.3f}  q={100*qW:+.3f}% radial  doublet={split34:.4f} deg  V=0")
    print(f"    W-iso: D={rW['D']:.3f}  q={100*qW:+.3f}% radial  doublet=NONE             V=0")
    check("Ih o/e doublet at apex 34 = 0.0508 +/- 0.0005 deg", abs(split34 - 0.0508) < 0.0005,
          f"got {split34:.5f}")
    check("U == 0 on W-34 chain", abs(rW["u"]) < 1e-12)

    print("  REFL-P(b) envelope (internal partial bounce on the W-34 backbone; b explicit axis):")
    q_refl = np.array([refl_partial_q(float(b)) for b in B_GRID])
    conv = abs(refl_partial_q(1e-6) - qW)
    check("convergence anchor: b -> 0 reproduces W-34 row (<0.05%)", conv < 5e-4, f"|dq|={conv:.2e}")
    sep = np.abs(q_refl - qW)
    for b_show in (2.0, 5.0, 10.0, 20.0, 30.0, 45.0):
        i = int(np.argmin(np.abs(B_GRID - b_show)))
        print(f"    b={B_GRID[i]:4.1f}: q={100*q_refl[i]:+7.3f}%   |dq vs W-34|={100*sep[i]:6.3f}%")
    ok_from = np.array([np.all(sep[i:] >= FLOOR) for i in range(len(B_GRID))])
    b_star = float(B_GRID[int(np.argmax(ok_from))]) if ok_from.any() else np.inf
    sep_active_min = float(sep[B_GRID >= B_INERT].min())
    sep_max = float(sep.max())
    print(f"    b* (separation >= 0.5% for all b' >= b): {b_star:.1f} deg"
          f"   min|dq| over active b >= {B_INERT:.0f}: {100*sep_active_min:.3f}%"
          f"   max|dq|: {100*sep_max:.2f}%")

    vmax_tir, qlo_tir, qhi_tir = refl_tir_envelope()
    print("  REFL-TIR envelope (max over b in TIR x 64 azimuths, backbone-pinned, LABELED ENVELOPE):")
    print(f"    max|V/I| = {100*vmax_tir:.3f}%   q range [{100*qlo_tir:+.2f}%, {100*qhi_tir:+.2f}%]")
    print("    NOTE: V is EXCLUDED from the separation matrix (prereg 2.3); refraction-class V==0 is")
    print("    the forbidden signature (live-leg detection kills them).")

    # ---- STAGE 3 — kill-gate smear sweep ---- #
    print("\nSTAGE 3 — kill-gate smear sweep (solar disk 0.53 deg FWHM on EVERY cell; low-sun config):")
    print("  flat-DoP rows (W-34, LEF, REFL-P) are smear-invariant in intrinsic DoP =>")
    print(f"  b*(cell) is cell-uniform = {b_star:.1f} deg; the cell grid adjudicates visibility +")
    print("  the Ih/iso ledge pair. Per-slice verdicts (prereg section 5):")
    cells_in = [(w, p) for w in WOBBLE_IN for p in PSF_IN]
    cells_out = [(w, p) for w in WOBBLE_OUT for p in list(PSF_IN) + list(PSF_OUT)] + \
                [(w, 0.30) for w in WOBBLE_IN]
    ledge_cache = {c: ledge_pair_separation(*c) for c in cells_in + cells_out}

    slice_verdicts = {}
    for c0 in C0_SPAN:
        for k in K_SPAN:
            vis = [c for c in cells_in if visible(c[0], c[1], c0, k)]
            if not vis:
                slice_verdicts[(c0, k)] = "NO-VISIBLE-CELLS"
            elif sep_max < FLOOR:
                slice_verdicts[(c0, k)] = "PRIMARY-KILL"
            elif b_star <= B_INERT:
                slice_verdicts[(c0, k)] = "PASS" if sep_active_min >= 2 * FLOOR else "MARGINAL"
            else:
                slice_verdicts[(c0, k)] = "PARTIAL-DOMAIN"
    for c0 in C0_SPAN:
        for k in K_SPAN:
            nvis = sum(1 for c in cells_in if visible(c[0], c[1], c0, k))
            print(f"    C0={c0:.2f} k={k:.0f}: visible in-box cells {nvis:2d}/12  ->  "
                  f"{slice_verdicts[(c0, k)]}")

    print("  W-Ih vs W-iso ledge pair (PARTIAL tier; smeared ledge-peak DoP excess vs 0.5% floor;")
    print("  A1 intensity mask Ic >= 0.1 plateau primary, >= 0.3 robustness column):")
    for c in cells_in:
        exc, _ = ledge_cache[c]
        exc30, _ = ledge_pair_separation(*c, imask=0.3)
        mark = "SEP" if exc >= FLOOR else "dead"
        print(f"    wobble={c[0]:.1f} psf={c[1]:.2f}:  excess={100*exc:6.3f}%"
              f"  (mask0.3: {100*exc30:6.3f}%)  [{mark}]")
    ih_iso_alive = [c for c in cells_in if ledge_cache[c][0] >= FLOOR]
    print(f"    -> Ih/iso separable on {len(ih_iso_alive)}/12 in-box cells (PARTIAL-tier report)")
    print("  out-of-box diagnostics (validity limits, never verdicts):")
    for c in cells_out:
        exc, _ = ledge_cache[c]
        print(f"    wobble={c[0]:.1f} psf={c[1]:.2f}:  ledge excess={100*exc:6.3f}%"
              f"   dilution={dilution(*c):.3f}")

    # ---- STAGE 4 — track co-predictions ---- #
    print("\nSTAGE 4 — track co-predictions (each class on its OWN track; forward points only):")
    print("  W-34 class at observed distances (two-branch band; floor inequality):")
    for Dt in TRACK_D:
        sols = w34_track_solutions(Dt)
        qs = sorted(q for _, q in sols)
        flo = float(so.halo_pol_dop(Dt))
        band = (f"[{100*qs[0]:.3f}%, {100*qs[-1]:.3f}%]" if qs else "no solution")
        print(f"    D={Dt:5.2f}: q band {band}   floor DoP(D)={100*flo:.3f}%   branches={len(sols)}")
        if qs:
            check(f"floor inequality at D={Dt:.2f}", qs[0] >= flo - 1e-9,
                  f"q_low={100*qs[0]:.3f}% floor={100*flo:.3f}%")
    print("  LEF class on its own track (single-valued; rises toward TIR cutoff):")
    for e in LEF_E:
        r = lef_rows[e]
        flo = float(so.halo_pol_dop(r["D"]))
        print(f"    e={e:4.1f}: D={r['D']:6.3f}  q={100*r['q']:+.3f}%   floor={100*flo:.3f}%"
              f"   q/floor={r['q']/flo:5.2f}")
        check(f"LEF floor inequality at e={e:.0f}", r["q"] >= flo - 1e-9)
    print(f"    LEF TIR cutoff e_cut={e_cut:.2f} deg — the class's own falsifiable endpoint")
    print("    (2.66x equality-level ratio test DEFERRED to the polyhedron ticket per prereg 3.3)")

    # ---- VERDICT ---- #
    print("\n" + "=" * 78)
    print("VERDICT (prereg section 5, per contrast-model slice):")
    primary = slice_verdicts[(0.30, 3.0)]
    for (c0, k), v in sorted(slice_verdicts.items()):
        star = "  <- primary slice (C0=0.30, k=3)" if (c0, k) == (0.30, 3.0) else ""
        print(f"  C0={c0:.2f} k={k:.0f}: {v}{star}")
    print(f"\n  refr-vs-REFL: b* = {b_star:.1f} deg (claim binds b* <= {B_INERT:.0f});"
          f" active-b margin = {100*sep_active_min:.3f}% vs floor 0.5% / marginal band [0.5,1.0)%")
    print(f"  Ih/iso PARTIAL tier: separable on {len(ih_iso_alive)}/12 in-box cells")
    print("  LEF resolved (source-pinned): REFRACTION class, own track + TIR-cutoff signature;")
    print(f"  named bonus pair LEF-vs-W34 at low sun: |dq| = {100*abs(lef_rows[0.0]['q']-qW):.3f}%")
    print(f"\n  OVERALL (primary slice): {primary}")
    print(f"  check failures: {fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
