#!/usr/bin/env python
"""Frozen test for Atlas Phase 8-A (scripts/atlas_strata_map.py).

Asserts the catastrophe-stratum corank is COMPUTED from the Jacobian (not asserted from arc shape — the
§6 armchair gate), closing the PHASE65 open question: the 29.7° UTA+LTA merge is an A₃ cusp (corank-1),
NOT a D₄ umbilic, and the column exposes only corank-1 strata (the honest null; D₄ needs the elevation ×
habit grid of Phase 8-B). Run: python scripts/test_atlas_strata_map.py  (~10-20 s).
"""
import sys
sys.path.insert(0, "scripts")
import atlas_strata_map as sm

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("the caustic is located and corank is COMPUTED from the Jacobian singular values:")
r = sm.corank_on_caustic(29.7)
check("29.7° merge: caustic exists (s2/scale → 0, the fold)", r is not None and r["s2_med_rel"] < 0.1,
      f"s2_med/scale={r['s2_med_rel']:.4f}" if r else "None")
check("29.7° merge is corank-1 (A₃ cusp) — NOT a D₄ umbilic (corank-2)",
      r["corank"] == 1 and r["s1_min_rel"] > 0.10, f"s1_min/scale={r['s1_min_rel']:.3f}")

print("generic fold + no-D₄-on-the-column (the honest null):")
r20 = sm.corank_on_caustic(20.0)
check("h=20 caustic is corank-1 (A₂ folds)", r20["corank"] == 1, f"s1_min/scale={r20['s1_min_rel']:.3f}")
worst = min(sm.corank_on_caustic(h)["s1_min_rel"] for h in (15.0, 20.0, 25.0, 29.7, 35.0, 45.0))
check("corank-1 everywhere on the column (no D₄ umbilic)", worst >= sm.CORANK2_REL,
      f"min s1/scale over sweep = {worst:.3f} (threshold {sm.CORANK2_REL})")

print("the corank is COMPUTED from the geometry, not hardcoded (the §6 armchair gate):")
rn = sm.corank_on_caustic(29.7, n=1.40)
check("Jacobian recomputes from n (absolute scale shifts); corank label stays 1 (structural)",
      rn is not None and abs(r["scale"] - rn["scale"]) > 1e-3 and rn["corank"] == 1,
      f"scale: n=1.31 -> {r['scale']:.4f}, n=1.40 -> {rn['scale']:.4f}")

print(f"\n{'ALL PASS — strata corank DERIVED; 29.7° merge = A₃ cusp, no D₄' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
