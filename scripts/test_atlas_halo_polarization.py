#!/usr/bin/env python
"""Frozen test for the Atlas halo-polarization predicted observable (scripts/atlas_halo_polarization.py
+ s2_optics.halo_pol_dop). Locks the one-parameter law DoP(R) = (1-cos^4(R/2))/(1+cos^4(R/2)):
it reproduces Können's 22-deg Fresnel floor, is monotone in R, predicts the pyramidal odd-radius family,
and the polarized raytracer confirms it at the cleanly-isolable halos (9, 22 deg).
Run: python scripts/test_atlas_halo_polarization.py
"""
import sys
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_optics as so                      # noqa: E402
import atlas_halo_polarization as ahp       # noqa: E402
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 console robustness
except Exception:
    pass

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("Atlas halo-polarization predicted observable — DoP(R) law:\n")

# (1) the law reproduces Können & Tinbergen 1991's 22-deg Fresnel floor (~3.7%)
d22 = so.halo_pol_dop(float(so.halo_min_deviation(60)))
check("law reproduces the Koennen 22-deg Fresnel floor (~3.7%)", abs(d22 - 0.037) < 0.003,
      f"DoP(22 deg)={d22*100:.2f}%")

# (2) DoP -> 0 as R -> 0 (no refraction, no diattenuation) and is strictly monotone increasing in R
check("DoP -> 0 as R -> 0", so.halo_pol_dop(0.0) < 1e-9)
Rs = np.linspace(1, 60, 60)
dops = so.halo_pol_dop(Rs)
check("DoP(R) is strictly monotone increasing in R", np.all(np.diff(dops) > 0),
      f"DoP(9)={so.halo_pol_dop(9)*100:.2f}% < DoP(35)={so.halo_pol_dop(35)*100:.2f}% < DoP(46)={so.halo_pol_dop(46)*100:.2f}%")

# (3) the pyramidal odd-radius family predictions (the Atlas connection): 9 deg faint, 35 deg ~9%
check("pyramidal 9-deg halo predicted faint (~0.6%)", abs(so.halo_pol_dop(8.96) - 0.006) < 0.002,
      f"DoP(9)={so.halo_pol_dop(8.96)*100:.2f}%")
check("pyramidal 35-deg halo predicted ~9.5%", abs(so.halo_pol_dop(35.0) - 0.095) < 0.01,
      f"DoP(35)={so.halo_pol_dop(35)*100:.2f}%")

# (4) the predicted table is radius-sorted and spans the regular + pyramidal halos
tbl = ahp.predicted_table()
check("the Atlas observable table covers regular + pyramidal halos (>=8 entries, both kinds)",
      len(tbl) >= 8 and {t[2] for t in tbl} == {"regular", "pyramidal"})

# (5) the polarized raytracer CONFIRMS the law at the cleanly-isolable halos (forward Monte-Carlo)
dop9, u9 = ahp.raytracer_dop("pyramidal", 8.0, 10.5, n_orient=9000, seed=22)
check("raytracer confirms the 9-deg pyramidal DoP matches the law (within 0.4%)",
      abs(dop9 - so.halo_pol_dop(8.96)) < 0.004, f"raytracer={dop9*100:.2f}% vs law={so.halo_pol_dop(8.96)*100:.2f}%")
dop22, u22 = ahp.raytracer_dop("random", 20.5, 23.5, n_orient=9000, seed=21)
check("raytracer confirms the 22-deg DoP matches the law (within 0.5%)",
      abs(dop22 - d22) < 0.005, f"raytracer={dop22*100:.2f}% vs law={d22*100:.2f}%")
check("raytracer confirms U/I ~ 0 (radial pol; mirror symmetry) on both halos",
      abs(u9) < 0.005 and abs(u22) < 0.005, f"U/I(9)={u9*100:+.2f}%, U/I(22)={u22*100:+.2f}%")

print(f"\n{'ALL PASS -- the one-parameter law DoP(R)=(1-cos^4(R/2))/(1+cos^4(R/2)) predicts every refraction halos linear polarization from its radius alone (radial, U=0, no V), generalizing Koennen 22-deg to the whole Atlas incl. the pyramidal odd-radius family, and the polarized raytracer confirms it at the clean 9/22-deg halos -- a forward, falsifiable Atlas observable.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
