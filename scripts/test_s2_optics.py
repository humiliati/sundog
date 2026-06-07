#!/usr/bin/env python
"""Physics unit tests for scripts/s2_optics.py (PHASE5 §3.12 forward model).

Anchors the cited physics before the model is used in the frozen S2 run:
  - min-deviation halo radii hit 21.8 deg (apex 60) and 45.6 deg (apex 90);
  - corona Airy ring angle scales as 1/a (size encoding) and sits at the J1-zero;
  - Mueller Stokes-V sign-flips with parity, vanishes at phi=0/90, is extremal at 45 deg;
  - every Stokes vector is physical: I >= sqrt(Q^2+U^2+V^2).
Run: python scripts/test_s2_optics.py
"""
import sys
import numpy as np
sys.path.insert(0, "scripts")
import s2_optics as so

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("min-deviation halo radii:")
r22 = float(so.halo_min_deviation(60))
r46 = float(so.halo_min_deviation(90))
check("apex 60 -> 22 deg halo", abs(r22 - 21.84) < 0.1, f"got {r22:.3f}")
check("apex 90 -> 46 deg halo", abs(r46 - 45.6) < 0.3, f"got {r46:.3f}")
check("halo radius is size-independent", so.halo_min_deviation(60) == r22)

print("corona Airy size encoding:")
d15 = so.airy_first_dark_deg(15.0)
d30 = so.airy_first_dark_deg(30.0)
check("first dark ring halves when size doubles (theta ~ 1/a)",
      abs(d15 / d30 - 2.0) < 0.02, f"d(15)={d15:.3f} d(30)={d30:.3f}")
# corona_intensity should dip to ~0 at the first dark ring angle
th = np.radians(so.airy_first_dark_deg(15.0))
I_dark = float(so.corona_intensity(th, 15.0))
I_center = float(so.corona_intensity(np.array([0.0]), 15.0)[0])
check("intensity ~0 at first dark ring", I_dark < 1e-3, f"I_dark={I_dark:.2e}")
check("intensity = 1 at center", abs(I_center - 1.0) < 1e-9, f"I0={I_center}")
# population averaging smears ring contrast (washout direction)
thetas = np.radians(np.linspace(0.2, 8.0, 400))
mono = so.corona_profile(thetas, np.full(64, 15.0))
poly = so.corona_profile(thetas, 15.0 * (1 + 0.6 * np.random.default_rng(0).standard_normal(64)))
check("size-spread reduces ring contrast (washout)",
      poly.std() < mono.std(), f"std mono={mono.std():.4f} poly={poly.std():.4f}")

print("Mueller Stokes-V handedness:")
Vp = so.stokes_V_over_I(40.0, np.pi / 4, 100.0, parity=+1)
Vm = so.stokes_V_over_I(40.0, np.pi / 4, 100.0, parity=-1)
check("V nonzero for oblique birefringent path", abs(Vp) > 1e-4, f"V+={Vp:.5f}")
check("V sign-flips with parity", np.sign(Vp) == -np.sign(Vm) and abs(Vp + Vm) < 1e-9,
      f"V+={Vp:.5f} V-={Vm:.5f}")
V0 = so.stokes_V_over_I(40.0, 0.0, 100.0, parity=+1)
V90 = so.stokes_V_over_I(40.0, np.pi / 2, 100.0, parity=+1)
V45 = abs(so.stokes_V_over_I(40.0, np.pi / 4, 100.0, parity=+1))
check("V ~ 0 at fast-axis phi=0", abs(V0) < 1e-6, f"V(0)={V0:.2e}")
check("V ~ 0 at fast-axis phi=90", abs(V90) < 1e-6, f"V(90)={V90:.2e}")
check("V extremal at phi=45", V45 > abs(V0) and V45 > abs(V90), f"|V(45)|={V45:.5f}")

print("Stokes physicality:")
worst = 0.0
rng = np.random.default_rng(1)
for _ in range(200):
    ti = rng.uniform(5, 70)
    phi = rng.uniform(0, np.pi)
    L = rng.uniform(10, 400)
    par = rng.choice([-1, 1])
    S = so.ray_stokes(ti, phi, L, parity=par)
    pol = np.sqrt(S[1] ** 2 + S[2] ** 2 + S[3] ** 2)
    worst = max(worst, pol - S[0])
check("I >= sqrt(Q^2+U^2+V^2) for all rays", worst < 1e-9, f"worst (pol-I)={worst:.2e}")

print(f"\n{'ALL PASS' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
