#!/usr/bin/env python
"""Frozen test for scripts/moilanen_pol_discriminator.py (S3-A3 internal leg).

PINNED BEFORE THE REGISTERED RUN (prereg frozen-test contract,
docs/atlas/MOILANEN_POL_DISCRIMINATOR_PREREG.md): the apex-60 Konnen gate anchors, the apex-34
wedge anchors (dmin, o/e split), the DoP(D) floor track, the LEF published-track consistency +
TIR cutoff, the REFL-P b->0 convergence anchor, floor-law spot checks, U==0, and determinism
(byte-identical recomputation). Post-run verdict pins are appended AFTER the registered run
(standard lane pattern; marked below).

Run: python scripts/test_moilanen_pol_discriminator.py
"""
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 console robustness

sys.path.insert(0, "scripts")
import s2_optics as so
import moilanen_pol_discriminator as mp

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("S3-A3 frozen test — pre-run apparatus pins:")

print("apex-60 Konnen gate anchors (CALIBRATION KILL tolerances):")
dmin60 = float(so.halo_min_deviation(60.0, mp.N))
check("dmin(60) = 21.84 +/- 0.01", abs(dmin60 - 21.84) < 0.01, f"{dmin60:.4f}")
r60 = mp.wedge_chain(60.0, mp.min_dev_entry(60.0, mp.N), mp.N)
check("chain D == module dmin (1e-9)", abs(r60["D"] - dmin60) < 1e-9)
check("Fresnel floor 3.71 +/- 0.10 abs%", abs(100 * r60["q"] - 3.71) < 0.10, f"{100*r60['q']:.3f}%")
check("chain q == halo_pol_dop(R) at min-dev (<1e-6)",
      abs(r60["q"] - float(so.halo_pol_dop(dmin60))) < 1e-6)
split60 = float(so.halo_min_deviation(60.0, mp.N_E) - so.halo_min_deviation(60.0, mp.N_O))
check("o/e split(60) = 0.106 +/- 0.005 deg", abs(split60 - 0.106) < 0.005, f"{split60:.4f}")
check("U == 0 (in-plane chain)", abs(r60["u"]) < 1e-12)
check("V == 0 (pure refraction)", abs(r60["v"]) < 1e-12)

print("apex-34 wedge anchors:")
dmin34 = float(so.halo_min_deviation(34.0, mp.N))
check("dmin(34) = 11.040 +/- 0.005", abs(dmin34 - 11.040) < 0.005, f"{dmin34:.4f}")
split34 = float(so.halo_min_deviation(34.0, mp.N_E) - so.halo_min_deviation(34.0, mp.N_O))
check("o/e split(34) = 0.0508 +/- 0.0005 deg", abs(split34 - 0.0508) < 0.0005, f"{split34:.5f}")

print("DoP(D) floor track (the refuter-verified numbers):")
for D, val in ((11.04, 0.930), (13.0, 1.290), (15.0, 1.718), (18.0, 2.477)):
    got = 100 * float(so.halo_pol_dop(D))
    check(f"DoP({D}) = {val:.3f} +/- 0.005 abs%", abs(got - val) < 0.005, f"{got:.4f}%")

print("LEF source-pinned chain (apex 30, vertical entry, theta_i1 = -e):")
for e, pub in ((0.0, 11.0), (10.0, 13.0), (15.0, 15.0), (20.0, 18.0)):
    r = mp.wedge_chain(mp.APEX_LEF, -e, mp.N)
    check(f"LEF D({e:.0f}) within 0.3 of published {pub:.0f}", abs(r["D"] - pub) < 0.3,
          f"D={r['D']:.3f}")
e_cut = mp.lef_tir_cutoff()
check("LEF TIR cutoff = 26.30 +/- 0.05 deg (published '>26')", abs(e_cut - 26.30) < 0.05,
      f"{e_cut:.4f}")
check("LEF chain dark above cutoff", mp.wedge_chain(mp.APEX_LEF, -27.0, mp.N) is None)

print("REFL-P envelope anchors:")
qW = mp.wedge_chain(34.0, mp.min_dev_entry(34.0, mp.N), mp.N)["q"]
conv = abs(mp.refl_partial_q(1e-6) - qW)
check("b -> 0 convergence to W-34 row (<0.05%)", conv < 5e-4, f"|dq|={conv:.2e}")
check("partial bounce is s-dominant (tangential, q < 0) at b=20",
      mp.refl_partial_q(20.0) < 0, f"q={100*mp.refl_partial_q(20.0):+.3f}%")
check("sub-critical bounce returns None past critical (TIR handoff)",
      so.mueller_fresnel_reflect(50.0, mp.N, 1.0) is None)

print("floor-law spot checks (q >= DoP(D)):")
for ti in (5.0, -15.0, 60.0):
    r = mp.wedge_chain(34.0, ti, mp.N)
    check(f"floor law at ti1={ti:+.0f}", r["q"] >= float(so.halo_pol_dop(r["D"])) - 1e-9,
          f"q={100*r['q']:.3f}% floor={100*float(so.halo_pol_dop(r['D'])):.3f}%")

print("smear machinery sanity:")
d_sharp = mp.dilution(0.1, 0.02)
d_blur = mp.dilution(1.0, 0.10)
check("dilution monotone in smear", d_sharp > d_blur, f"{d_sharp:.3f} > {d_blur:.3f}")
exc_sharp, qf = mp.ledge_pair_separation(0.1, 0.02)
exc_blur, _ = mp.ledge_pair_separation(1.0, 0.10)
check("ledge excess >= 0 and monotone-decreasing under smear",
      exc_sharp >= exc_blur >= 0.0, f"sharp={100*exc_sharp:.3f}% blur={100*exc_blur:.3f}%")
check("ledge far-field floor equals halo_pol_dop", abs(qf - float(so.halo_pol_dop(dmin34))) < 1e-4)

print("determinism (byte-identical recomputation):")
def fingerprint():
    return repr((mp.refl_partial_q(20.0), mp.wedge_chain(34.0, 30.0, mp.N)["q"],
                 mp.ledge_pair_separation(0.3, 0.05)[0], mp.lef_tir_cutoff()))
check("fingerprint repr byte-identical across recomputation", fingerprint() == fingerprint())

# ---- POST-RUN VERDICT PINS (appended after the registered run 2026-06-11; ---- #
# ---- docs/atlas/MOILANEN_POL_DISCRIMINATOR_RESULT.md) ------------------------- #
print("post-run verdict pins (registered run 2026-06-11):")
q_refl_grid = np.array([mp.refl_partial_q(float(b)) for b in mp.B_GRID])
sep = np.abs(q_refl_grid - qW)
ok_from = np.array([np.all(sep[i:] >= mp.FLOOR) for i in range(len(mp.B_GRID))])
b_star = float(mp.B_GRID[int(np.argmax(ok_from))]) if ok_from.any() else np.inf
check("b* = 3.0 deg (refr-vs-REFL separation threshold)", abs(b_star - 3.0) < 1e-9, f"{b_star}")
margin = 100 * float(sep[mp.B_GRID >= mp.B_INERT].min())
check("active-b margin = 2.011% +/- 0.01 (PASS band, >= 2x floor)", abs(margin - 2.011) < 0.01,
      f"{margin:.4f}%")
vmax_tir, _, _ = mp.refl_tir_envelope()
check("backbone TIR V envelope = 0.236% +/- 0.01 (below 0.5% live-leg floor; scope note)",
      abs(100 * vmax_tir - 0.236) < 0.01, f"{100*vmax_tir:.4f}%")
exc_a, _ = mp.ledge_pair_separation(0.1, 0.02)
exc_b, _ = mp.ledge_pair_separation(1.0, 0.10)
check("ledge excess (0.1, 0.02) = 17.728% +/- 0.05", abs(100 * exc_a - 17.728) < 0.05,
      f"{100*exc_a:.4f}%")
check("ledge excess (1.0, 0.10) = 3.782% +/- 0.05", abs(100 * exc_b - 3.782) < 0.05,
      f"{100*exc_b:.4f}%")
r20 = mp.wedge_chain(mp.APEX_LEF, -20.0, mp.N)
check("LEF q(e=20) = 8.671% +/- 0.01 (elevation-rising signature)",
      abs(100 * r20["q"] - 8.671) < 0.01, f"{100*r20['q']:.4f}%")
lef0 = mp.wedge_chain(mp.APEX_LEF, 0.0, mp.N)
bonus = abs(lef0["q"] - qW)
check("LEF-vs-W34 low-sun bonus pair in marginal band [0.5, 1.0)%",
      0.005 <= bonus < 0.010, f"{100*bonus:.4f}%")

print(f"\n{'ALL PASS' if fail == 0 else str(fail) + ' FAILED'} — S3-A3 frozen test")
sys.exit(1 if fail else 0)
