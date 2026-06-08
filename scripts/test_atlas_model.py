#!/usr/bin/env python
"""Frozen test for Atlas Phase 11 — the capstone small-parameter model (scripts/atlas_model.py).

Asserts the structural-model claim is real: the atlas SKELETON is generated from n (+ the fixed ice
geometry), it RECOMPUTES from n (derived, not a list of constants), and the parameter set is the claimed
small one. Run: python scripts/test_atlas_model.py  (~2 s).
"""
import sys
sys.path.insert(0, "scripts")
import atlas_model as am

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


a = am.generate_atlas(am.N_VIS)
b = am.generate_atlas(1.40)

print("the atlas skeleton is GENERATED from n (+ the fixed ice geometry):")
check("the {10-11} pyramid angle is DERIVED from the ice c/a ratio (not free)", abs(am.pyramid_angle() - 61.99) < 0.1,
      f"{am.pyramid_angle():.2f}° from c/a={am.ICE_CA:.3f}")
check("fold radii from n: 22° halo = 21.84°, 46° halo = 45.73°",
      abs(a["fold_radii_deg"]["22 halo"] - 21.839) < 0.02 and abs(a["fold_radii_deg"]["46 halo"] - 45.733) < 0.02)
check("walls from n: CZA off = 32.196°, CHA on = 57.804°, and CZA+CHA = 90° (complement identity)",
      abs(a["walls_deg"]["CZA_off"] - 32.196) < 0.02 and abs(a["walls_deg"]["CHA_on"] - 57.804) < 0.02
      and abs(a["walls_deg"]["CZA_off"] + a["walls_deg"]["CHA_on"] - 90.0) < 1e-9)

print("RECOMPUTE-FROM-n: vary n -> every number shifts (the derived-not-hardcoded demonstration):")
moved_radii = all(abs(b["fold_radii_deg"][k] - a["fold_radii_deg"][k]) > 1.0 for k in a["fold_radii_deg"])
moved_walls = all(abs(b["walls_deg"][k] - a["walls_deg"][k]) > 1.0 for k in ("CZA_off", "CHA_on"))
check("every fold radius AND every wall moves when n: 1.31 -> 1.40 (generated, not constant)",
      moved_radii and moved_walls,
      f"22°: {a['fold_radii_deg']['22 halo']:.2f}->{b['fold_radii_deg']['22 halo']:.2f}, "
      f"CZA: {a['walls_deg']['CZA_off']:.2f}->{b['walls_deg']['CZA_off']:.2f}")
check("the CZA+CHA complement identity holds at the OTHER n too (structural, not coincidental)",
      abs(b["walls_deg"]["CZA_off"] + b["walls_deg"]["CHA_on"] - 90.0) < 1e-9)

print("the parameter set is the claimed SMALL one:")
check("1 continuous physical parameter (n) + a discrete habit enumeration (~7 'platonic solids')",
      isinstance(am.N_VIS, float) and 5 <= len(am.HABITS) <= 9, f"{len(am.HABITS)} habits: {list(am.HABITS)}")

print(f"\n{'ALL PASS — the atlas is GENERATED from ~1 physical parameter (n) + fixed ice geometry + a discrete habit list; recomputes from n' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
