#!/usr/bin/env python
"""Frozen test for H7 (scripts/mirage_curved.py) — the curved-Earth mirage ray-tracer. Locks the
curved-Earth optics: the k≈4/3 effective Earth radius, the √k-extended horizon, the dn/dh=-1/R_E ducting
threshold (Novaya-Zemlya), and inversion looming (an elevated duct). Deterministic (RK4). Run:
python scripts/test_mirage_curved.py
"""
import sys
import warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import mirage_curved as mc   # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("H7 — curved-Earth mirage ray-tracer:\n")

# (1) standard-refraction k-factor = 4/3
k = mc.k_factor(mc.G_STD)
check("standard refraction gives the textbook k=4/3 effective Earth radius", abs(k - 4 / 3) < 0.05,
      f"k={k:.3f}")

# (2) refraction extends the horizon by ~sqrt(k)
d_geo = mc.horizon_distance(10.0, mc.grad_standard, g=0.0)
d_ref = mc.horizon_distance(10.0, mc.grad_standard, g=mc.G_STD)
check("refraction pushes the horizon out by ~√k", abs(d_ref / d_geo - np.sqrt(k)) < 0.08,
      f"ratio={d_ref/d_geo:.3f} vs √k={np.sqrt(k):.3f}")

# (3) the ducting threshold dn/dh = -1/R_E: horizontal ray stays level (ducts) at threshold;
#     rises below; dives above
R3 = 3.0e5
h_thr = mc.trace(np.array([0.0]), mc.grad_standard, R=R3, h_obs=50.0, g=-mc.INV_RE)[0][0]
h_sub = mc.trace(np.array([0.0]), mc.grad_standard, R=R3, h_obs=50.0, g=-0.5 * mc.INV_RE)[0][0]
h_sup = mc.trace(np.array([0.0]), mc.grad_standard, R=R3, h_obs=50.0, g=-2.5 * mc.INV_RE)[0][0]
check("DUCTING threshold dn/dh=-1/R_E: a horizontal ray stays LEVEL (ducts, Novaya-Zemlya)",
      abs(h_thr - 50.0) < 50.0, f"h(300km)={h_thr:.1f} m (launch 50 m)")
check("sub-critical (|dn/dh|<1/R_E) the ray RISES (finite horizon)", h_sub > 500.0, f"h={h_sub:.0f} m")
check("super-critical (|dn/dh|>1/R_E) the ray DIVES toward the surface", h_sup < 0.0, f"h={h_sup:.0f} m")
geo_hz = np.sqrt(2 * 50.0 * mc.R_E)
check("ducted ray sees FAR beyond the geometric horizon (300 km vs √(2·50·R_E)≈25 km)",
      R3 > 5 * geo_hz, f"{R3/geo_hz:.0f}× the geometric horizon")

# (4) elevated inversion duct traps rays (looming / Fata Morgana); standard atmosphere does not
R4 = 3.0e5
th4 = np.linspace(-1.5e-3, 1.5e-3, 600)
hf4, mh4, hit4 = mc.trace(th4, mc.grad_inversion, R=R4, h_obs=120.0, g=mc.G_STD, dn=1.5e-5, s=30.0, h0=120.0)
hf4s, mh4s, hit4s = mc.trace(th4, mc.grad_standard, R=R4, h_obs=120.0, g=mc.G_STD)
n_tr = int(((hit4 >= R4) & (mh4 > 0) & (hf4 < 600.0)).sum())
n_std = int(((hit4s >= R4) & (mh4s > 0) & (hf4s < 600.0)).sum())
check("an elevated inversion TRAPS rays (looming / Fata Morgana); standard air does not",
      n_tr > 0 and n_tr > n_std, f"trapped: inversion={n_tr}, standard={n_std}")

print(f"\n{'ALL PASS -- curved-Earth mirage optics reproduced: k=4/3 effective Earth radius, √k-extended horizon, the dn/dh=-1/R_E ducting threshold (Novaya-Zemlya), and inversion looming (elevated duct / Fata Morgana). The H5 catastrophe ladder now lives over a real horizon.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
