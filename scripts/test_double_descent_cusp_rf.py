#!/usr/bin/env python
"""Frozen test for H8-RF (scripts/double_descent_cusp_rf.py) — locks the registered K2 verdict and every
load-bearing number of the 2026-06-12 registered run (prereg docs/atlas/H8_RF_CUSP_PREREG.md; result
docs/atlas/H8_RF_CUSP_RESULT.md): the Mei-Montanari closed form (MC-validated), the quartic tol_cal
control, the finite-interior fold-pair annihilation (K1 does NOT fire), the corank-1 cusp on every locus
slice, and the |c3| locus band hit (min shared-normalization ratio 0.475 in [0.25,0.5) -> K2,
germ-indeterminate). Also regression-locks the post-verdict DESCRIPTIVE diagnostics that motivate the
named follow-up (slope 0.526; self-normalized trend reversal). Run: python scripts/test_double_descent_cusp_rf.py
"""
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import double_descent_cusp_rf as dd   # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("H8-RF — frozen test (registered verdict: K2, finite merge of unresolved germ class):\n")

# (1) pinned activation constants + closed form (byte-stable cells from the registered run)
check("zeta^2 (Eq. 9 pinned)", abs(dd.Z2 - 2.751938393884) < 1e-9, f"{dd.Z2:.12f}")
for (p1, p2, lb, t2, want) in ((3.0, 10.0, 0.06, 0.8, 0.417479388727),
                               (25.0, 10.0, 0.06, 0.8, 0.217228169324),
                               (6.0, 3.0, 0.11, 0.5, 0.437687358828),
                               (0.5, 3.0, 1.1, 0.5, 0.836862509354)):
    r, resid = dd.risk_rf(np.array([p1]), p2, lb, t2)
    check(f"R_RF({p1},{p2},{lb},{t2}) pinned", abs(float(r[0]) - want) < 1e-9 and resid < dd.RESID_TOL,
          f"{float(r[0]):.12f}")
chi, _ = dd.nu_chi(np.array([3.0]), 10.0, 0.06)
check("chi(3,10,0.06) pinned (C+ branch, real negative)", abs(float(chi[0]) + 1.748214263613) < 1e-9)

# (2) MC gate cell (Eq. 2 estimator, seeded): closed form within 12%, value regression-locked
rc = float(dd.risk_rf(np.array([1.0]), 3.0, 0.55, 0.5)[0][0])
rm = dd.risk_mc(1.0, 3.0, 0.55, 0.5, 100)
check("MC cell (psi2=3,d=100,psi1=1,lbar=0.55): closed vs MC rel err < 12%", abs(rc - rm) / rm < 0.12,
      f"closed={rc:.4f} MC={rm:.4f} ({abs(rc-rm)/rm:.2%})")
check("MC cell value regression-locked", abs(rm - 0.6963093333) / 0.6963093333 < 1e-3, f"{rm:.10f}")

# (3) tol_cal scaling control (quartic exact 1/2 law, SAME code path; the bisection sign-bug regression)
h_c, slope_ctl, r2_ctl, np_ctl = dd.quartic_control()
check("quartic control h_c pinned (negative-control bisection terminates)", abs(h_c + 0.423466) < 1e-4,
      f"h_c={h_c:.6f}")
check("quartic control slope = 1/2 within 0.05, R2 >= 0.999", abs(slope_ctl - 0.5) <= 0.05 and r2_ctl >= 0.999,
      f"slope={slope_ctl:.4f} R2={r2_ctl:.5f} ({np_ctl} pts)")

# (4) the K1 outcome: finite-interior fold-pair annihilation on the primary slice (the structural headline)
w = dd.slice_windows(dd.PRIMARY)
check("primary slice (tau2=0.8): pair-at-floor exists, lbar_c pinned",
      w is not None and abs(w["lbar_c"] - 0.064727) < 1e-4, f"lbar_c={w['lbar_c']:.6f}")
check("primary merge psi1* pinned, interior", abs(w["psi1c"] - 3.8328) < 5e-3
      and dd.CEN_LO + 1 < w["psi1c"] < dd.CEN_HI - 1, f"psi1*={w['psi1c']:.4f}")

# (5) charts on the five registered windows (rebuilt from pinned lbar_c/psi1* exactly as main() does)
REG = {0.4: (0.038507, 4.5723, 59.7157), 0.8: (0.064727, 3.8328, 95.3131),
       1.2: (0.087629, 3.3896, 125.6974), 1.6: (0.108810, 3.0709, 153.0509),
       2.4: (0.148681, 2.6153, 197.9384)}
f0, f1, f2, f3 = dd.CHART_FACTORS
lcp, pcp, _ = REG[dd.PRIMARY]
prim_span_p = (f0 * pcp, f1 * pcp)
prim_span_l = (max(dd.LBAR_FLOOR, f2 * lcp), f3 * lcp)
base = dd.rf_chart(dd.PSI2, dd.PRIMARY, prim_span_p, prim_span_l, (0.0, 1.0, prim_span_l[0], prim_span_l[1]))
check("shared normalization constants pinned",
      abs(base["R_mean"] - 0.402573) < 1e-3 and abs(base["R_std"] - 0.063700) < 1e-3,
      f"m_ref={base['R_mean']:.6f} s_ref={base['R_std']:.6f}")
norm = (base["R_mean"], base["R_std"], prim_span_l[0], prim_span_l[1])
c3s = {}
for t2, (lc, pc, want) in REG.items():
    r = dd.rf_chart(dd.PSI2, t2, (f0 * pc, f1 * pc), (max(dd.LBAR_FLOOR, f2 * lc), f3 * lc), norm)
    c3s[t2] = r["c3"][0] if r["c3"] else float("nan")
    check(f"tau2={t2}: exactly 1 corank-1 cusp, shared-norm |c3| pinned (+-2%)",
          r["n_cusps"] == 1 and r["corank"] == 1 and abs(c3s[t2] - want) / want < 0.02,
          f"|c3|={c3s[t2]:.4f} (registered {want})")

# (6) THE REGISTERED VERDICT: |c3| locus min ratio in the [0.25, 0.5) band -> K2 fires
med = float(np.median(list(c3s.values())))
rmin = min(v / med for v in c3s.values())
check("locus min shared-norm ratio = 0.475 (+-0.01), inside the [0.25,0.5) band",
      abs(rmin - 0.475) < 0.01 and 0.25 <= rmin < 0.5, f"min ratio={rmin:.4f} median={med:.4f}")
print("\n  => REGISTERED VERDICT REPRODUCED: K2 — finite merge of unresolved germ class (band rule).")

# (7) cheap in-context controls (the full set lives in test_atlas_jet_classify.py)
import atlas_jet_classify as jc   # noqa: E402
a4r = jc.cusp_c3(*jc.synthetic_swallowtail(-0.02))[0] / jc.cusp_c3(*jc.synthetic_swallowtail(-0.40))[0]
check("Morin A4 dive control (< 0.25)", a4r < 0.25, f"ratio={a4r:.3f}")
Xu, Yu, du = jc.synthetic_umbilic(0.0)
check("synthetic D4 fires corank-2", jc.corank_from_chart(Xu, Yu, du, du)["corank"] == 2)
import double_descent_cusp as ddi  # noqa: E402
iso = ddi.cusp_chart((1.05, 4.0), (0.05, 0.6))
check("banked isotropic chart through this pipeline: caustic, 0 cusps", iso["caustic"] and iso["n_cusps"] == 0)

# (8) DESCRIPTIVE diagnostics regression-locks (motivate the named follow-up; verdict unaffected)
def crit_at(lb):
    return dd.census(dd.PSI2, dd.PRIMARY, lb)[0]
slope, r2, npts = dd.scaling_fit(crit_at, w["lbar_c"], 2)
check("descriptive K3 slope locked (0.5258 +-0.01, R2>=0.999, 12 pts; NEVER adjudicated)",
      abs(slope - 0.5258) < 0.01 and r2 >= 0.999 and npts == 12, f"slope={slope:.4f} R2={r2:.5f}")
sn = {}
for t2 in (0.4, 2.4):
    lc, pc, _ = REG[t2]
    spans = ((f0 * pc, f1 * pc), (max(dd.LBAR_FLOOR, f2 * lc), f3 * lc))
    pre = dd.rf_chart(dd.PSI2, t2, *spans, (0.0, 1.0, spans[1][0], spans[1][1]))
    r = dd.rf_chart(dd.PSI2, t2, *spans, (pre["R_mean"], pre["R_std"], spans[1][0], spans[1][1]))
    sn[t2] = r["c3"][0] if r["c3"] else float("nan")
check("descriptive self-norm trend REVERSAL locked (c3(0.4) > c3(2.4) self-normalized; 133.65/57.79 +-2%)",
      sn[0.4] > sn[2.4] and abs(sn[0.4] - 133.65) / 133.65 < 0.02 and abs(sn[2.4] - 57.794) / 57.794 < 0.02,
      f"self-norm c3: tau2=0.4 -> {sn[0.4]:.2f}, tau2=2.4 -> {sn[2.4]:.2f}")

print(f"\n{'ALL PASS — the registered K2 verdict and every load-bearing number of the 2026-06-12 run are locked: the RF closed form has a GENUINE finite-interior fold-pair annihilation (unlike the isotropic peak-escape), read as a grid-stable corank-1 cusp on all five locus slices, with the A3 germ label INDETERMINATE under the pre-registered shared-normalization |c3| band rule.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
