#!/usr/bin/env python
"""Frozen test for Atlas Phase 6.5-A (scripts/atlas_bifurcation_set.py).

Asserts the component-B walls + A₂ fold primitives are DERIVED from {n, geometry} and reproduce the
documented transitions within the pre-registered ±1.0° tolerance — and, critically (the §6 "armchair
catastrophe" gate), that the numbers are COMPUTED from n, not hardcoded constants.
Run: python scripts/test_atlas_bifurcation_set.py
"""
import sys
import numpy as np
sys.path.insert(0, "scripts")
import atlas_bifurcation_set as bs
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


print("derived transitions within pre-registered tolerance (±1.0 deg):")
for r in bs.results():
    check(f"{r['label']} derived {r['derived_deg']:.3f} vs documented {r['documented_deg']:.0f}",
          r["pass"], f"residual {r['residual_deg']:.3f} deg")

print("the numbers are DERIVED from n, not hardcoded (the §6 armchair-catastrophe gate):")
# changing n must change every wall/fold output — proves computation, not a constant
cza_131, cza_140 = bs.cza_disappearance_deg(1.31), bs.cza_disappearance_deg(1.40)
cha_131, cha_140 = bs.cha_appearance_deg(1.31), bs.cha_appearance_deg(1.40)
f22_131, f22_140 = bs.fold_radius_deg(60, 1.31), bs.fold_radius_deg(60, 1.40)
check("CZA wall recomputes from n (32@1.31 != value@1.40)", abs(cza_131 - cza_140) > 1.0,
      f"{cza_131:.2f} vs {cza_140:.2f}")
check("CHA wall recomputes from n", abs(cha_131 - cha_140) > 1.0, f"{cha_131:.2f} vs {cha_140:.2f}")
check("22deg fold recomputes from n", abs(f22_131 - f22_140) > 1.0, f"{f22_131:.2f} vs {f22_140:.2f}")

print("the CZA/CHA complement identity (one 90deg plate prism, opposite faces):")
check("CHA appearance = 90 - CZA disappearance (cos h <-> sin h face-swap)",
      abs(cha_131 - (90.0 - cza_131)) < 1e-9, f"CHA={cha_131:.3f}, 90-CZA={90 - cza_131:.3f}")
check("TIR critical angle = arcsin(1/n) ~ 49.76 deg", abs(bs.tir_critical_angle_deg() - 49.76) < 0.05,
      f"{bs.tir_critical_angle_deg():.2f}")

print("the §0.2 smearing band is reported (chromatic spread, not a sharp point):")
for r in bs.results():
    lo, hi = r["chromatic_band_deg"]
    check(f"{r['label']} reports a non-trivial chromatic band", hi - lo > 0.05, f"band {lo:.2f}-{hi:.2f} deg")

print(f"\n{'ALL PASS — component-B walls derived, not asserted' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
