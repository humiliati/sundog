#!/usr/bin/env python
"""Frozen test for the charFun-spectrum determine/resist law (scripts/shadow_charfun_populations.py),
Hypothesis 2's empirical leg. Asserts the pre-registered dichotomy: resistance tracks the averaging
population's CHARACTERISTIC FUNCTION decay (not its variance), and determination tracks its finite
centered mean -- two DIFFERENT conditions, with Cauchy as the separator (resists but breaks determine).
Run: python scripts/test_shadow_charfun_populations.py
"""
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import shadow_charfun_populations as scp   # noqa: E402

fail = 0
# focused lambda grid: 0.0 base, 0.5 lattice-envelope null (dip), 1.0 recurrence, 2.0 endpoint
LAMS = [0.0, 0.5, 1.0, 2.0]


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("charFun-spectrum determine/resist law -- swap the S0 averaging population:\n")

# ---- RESIST half: cont at the lambda endpoints (n reduced for test speed; split is robust) ---- #
cont = {}
for pop in scp.POPS:
    c, _ = scp.sweep_pop(pop, n=400, lams=LAMS)
    cont[pop] = dict(zip(LAMS, c))
    print(f"  {pop:<8} cont @ lam{LAMS} = {c}")
print()

for pop in ["gaussian", "uniform", "cauchy"]:
    check(f"{pop} (AC) RESISTS: phi_mu -> 0 washes the continuous fringe (cont(lam=2) <= 0.10)",
          cont[pop][2.0] <= 0.10, f"cont(2.0)={cont[pop][2.0]:.3f}")

check("CAUCHY resists DESPITE infinite variance (charFun exp(-|s|) governs, not Var)",
      cont["cauchy"][2.0] <= 0.10 and cont["cauchy"][0.0] > 0.5,
      f"base={cont['cauchy'][0.0]:.2f} -> cont(2.0)={cont['cauchy'][2.0]:.3f}")

check("LATTICE SURVIVES: phi_mu=cos(s) recurs to 1 -> fringe never washes (cont(lam=2) > 0.30)",
      cont["lattice"][2.0] > 0.30, f"cont(2.0)={cont['lattice'][2.0]:.3f}")

check("lattice shows the RESONANT recurrence: cont recovers from the lam=0.5 envelope-null by lam=1.0",
      cont["lattice"][1.0] > cont["lattice"][0.5],
      f"cont(0.5)={cont['lattice'][0.5]:.2f} -> cont(1.0)={cont['lattice'][1.0]:.2f}")

# ---- DETERMINE half: concentration (LLN holds for finite mean, fails for Cauchy) ---- #
print()
det = {pop: scp.determination_probe(pop) for pop in scp.POPS}
for pop in scp.POPS:
    ratio = det[pop][0] / max(det[pop][-1], 1e-9)
    print(f"  {pop:<8} median|avg-d| over K{scp.DET_KS} = "
          f"{[round(v, 3) for v in det[pop]]}  ({ratio:.1f}x)")
print()

for pop in ["gaussian", "uniform", "lattice"]:
    ratio = det[pop][0] / max(det[pop][-1], 1e-9)
    check(f"{pop} DETERMINES: finite centered mean -> average concentrates (ratio > 3x over K)",
          ratio > 3.0, f"ratio={ratio:.1f}x")

cauchy_ratio = det["cauchy"][0] / max(det["cauchy"][-1], 1e-9)
check("CAUCHY BREAKS determine: no finite mean -> K-average never concentrates (ratio < 1.5x)",
      cauchy_ratio < 1.5, f"ratio={cauchy_ratio:.1f}x (sample mean of Cauchy is itself Cauchy)")

# ---- the analytic charFun envelope corroborates the mechanism ---- #
print()
s2 = 2 * np.pi * 2.0 * scp.T_REF                      # charFun arg at lam=2, fringe-peak probe
check("analytic envelope: gaussian |phi(2pi*2*t0_f)| ~ 0 (washed) but lattice |phi| = 1 (cos(2pi))",
      abs(scp.charfun_re("gaussian", s2)) < 0.01 and abs(abs(scp.charfun_re("lattice", s2)) - 1.0) < 1e-6,
      f"|phi_gauss|={abs(scp.charfun_re('gaussian', s2)):.4f}, |phi_lat|={abs(scp.charfun_re('lattice', s2)):.4f}")

print(f"\n{'ALL PASS -- resistance = charFun decay (Cauchy resists despite infinite variance); determine = finite mean (Cauchy breaks it). Two conditions, one separator.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
