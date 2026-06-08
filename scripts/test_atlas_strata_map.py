#!/usr/bin/env python
"""Frozen test for Atlas Phase 8-A + 8-B (scripts/atlas_strata_map.py).

8-A: the catastrophe-stratum corank is COMPUTED from the Jacobian (not asserted from arc shape — the §6
armchair gate), closing the PHASE65 open question: the 29.7° UTA+LTA merge is corank-1 (A₃-class), NOT a
D₄ umbilic, and the column exposes only corank-1 strata (the honest null; D₄ needs the elevation × habit
grid of Phase 8-B).
8-B: the CUSP LOCATOR (K·∇(det J)=0) finds exactly 2 A₃ point-cusps — the UTA/LTA apexes — stable across
the robust regime (h≥22°) → NO A₄ swallowtail (confirms Berry 1994); the cusp δ recomputes from n.
Run: python scripts/test_atlas_strata_map.py  (~20-40 s).
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
check("29.7° merge is corank-1 (A₃-class metamorphosis) — NOT a D₄ umbilic (corank-2)",
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

print("\nPhase 8-B — cusp locator (A₃ point-cusps) + swallowtail (A₄) search:")
nc25, locs25 = sm.cusp_count(25.0)
nc35, locs35 = sm.cusp_count(35.0)
check("exactly 2 A₃ point-cusps on the column (the UTA/LTA apexes), h=25° and h=35°",
      nc25 == 2 and nc35 == 2, f"#cusps: h25={nc25}, h35={nc35}")
apex_ok = (all(abs(d - 21.3) < 1.5 for d, _ in locs25) and
           {round(abs(p) / 180) for _, p in locs25} == {0, 1})   # ψ at 0 (top) and ±180 (bottom)
check("the cusps sit at the apexes (δ≈21.8° on the 22° fold; ψ=0 top, 180 bottom)", apex_ok,
      f"locs={locs25}")
counts = [sm.cusp_count(h)[0] for h in (22.0, 25.0, 28.0, 29.7, 31.0, 35.0, 45.0)]
check("NO A₄ swallowtail: cusp count stable at 2 across the robust regime (no pair born/annihilated)",
      set(counts) == {2}, f"counts over h∈[22,45]: {counts}  (confirms Berry 1994)")
ncn, locsn = sm.cusp_count(25.0, n=1.40)        # the 22° fold (hence the apex δ) moves with n
delta_n131 = locs25[0][0]; delta_n140 = locsn[0][0]
check("the cusp δ recomputes from n (apex tracks the n-dependent 22° fold), count stays 2 (structural)",
      ncn == 2 and abs(delta_n131 - delta_n140) > 3.0,
      f"apex δ: n=1.31 -> {delta_n131}°, n=1.40 -> {delta_n140}°")

print("\nPhase 8-B — 90°-wedge family (46° / supralateral / infralateral arcs), the 2nd 2-DOF caustic:")
rb = sm.corank_on_caustic(20.0, wedge="basal90")
check("the 90°-wedge caustic exists and is the 46° family (cusps at δ≈47–58°, the supralateral arc)",
      rb is not None and rb["corank"] == 1, f"corank={rb['corank'] if rb else None}")
worst_b = min(sm.corank_on_caustic(h, wedge="basal90")["s1_min_rel"] for h in (10.0, 15.0, 20.0, 25.0))
check("corank-1 everywhere on the 90°-wedge family (no D₄ umbilic)", worst_b >= sm.CORANK2_REL,
      f"min s1/scale = {worst_b:.3f} (≫ {sm.CORANK2_REL})")
ncb, locsb = sm.cusp_count(20.0, wedge="basal90")
sym = ncb == 2 and len({round(abs(p)) for _, p in locsb}) == 1 and all(40 < d < 60 for d, _ in locsb)
check("2 cusps — a ψ-symmetric pair at the sides (the lateral-arc cusps), not the apexes", sym,
      f"locs={locsb}")
counts_b = [sm.cusp_count(h, wedge="basal90")[0] for h in (12.0, 18.0, 22.0, 25.0, 28.0)]
check("NO A₄ swallowtail on the 46° family: cusp count stable at 2 (confirms Berry on the 2nd family)",
      set(counts_b) == {2}, f"counts over h≲28: {counts_b}")

print(f"\n{'ALL PASS — corank + cusps DERIVED on BOTH 2-DOF families; 29.7° = A₃-class metamorphosis, apex/lateral point-cusps, no A₄/D₄' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
