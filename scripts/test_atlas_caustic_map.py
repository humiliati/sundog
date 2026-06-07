#!/usr/bin/env python
"""Frozen test for Atlas Phase 6.5-B (scripts/atlas_caustic_map.py).

Asserts the ~29° UTA+LTA→circumscribed-halo merge is a DERIVED output of the horizontal-column halo-
function caustic (Tape 1980), within the pre-registered ±1.0°, and is computed from {n, geometry}
(not the hardcoded TANGENT_ARC_CIRCUMSCRIBED_H=29 — the §6 armchair-catastrophe gate).
Run: python scripts/test_atlas_caustic_map.py   (~20-30 s; vectorized orientation sweeps + bisection)
"""
import sys
import numpy as np
sys.path.insert(0, "scripts")
import atlas_caustic_map as m

NG = 300
fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("sanity — the halo function reproduces the 22deg fold:")
c, b = m.caustic_coverage(20.0, ngrid=NG)
top = float(np.nanmin(b[c < 8.0]))
check("caustic touches the 22deg fold at the top (psi~0)", abs(top - 21.84) < 0.3, f"{top:.2f} deg")

print("caustic topology — separate arcs below the merge, connected loop above:")
check("h=28: separate UTA + LTA (gap open)", not m.is_merged(28.0, ngrid=NG))
check("h=31: circumscribed (gap closed)", m.is_merged(31.0, ngrid=NG))

print("the 29deg merge is DERIVED within the pre-registered ±1.0deg:")
hm = m.merge_elevation(ngrid=NG)
check("merge elevation within ±1.0 of documented 29deg", hm is not None and abs(hm - 29.0) <= 1.0,
      f"derived {hm:.2f} deg" if hm else "None")

print("the merge is COMPUTED from n, not hardcoded (the §6 gate); chromatic band:")
hm_red = m.merge_elevation(n=1.307, ngrid=NG)
hm_blue = m.merge_elevation(n=1.317, ngrid=NG)
check("merge recomputes from n (red != blue -> non-trivial chromatic band)",
      hm_red is not None and hm_blue is not None and abs(hm_red - hm_blue) > 0.1,
      f"red(n=1.307)={hm_red:.2f}  blue(n=1.317)={hm_blue:.2f}")

print(f"\n{'ALL PASS — 29deg cusp DERIVED, not asserted' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
