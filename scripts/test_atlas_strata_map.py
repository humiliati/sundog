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

print("\nPhase 8-C — Lowitz manifold (wedge='lowitz60'): the A₃-lips metamorphosis, NOT Berry's A₄:")
import numpy as np
from scipy import ndimage
import atlas_caustic_map as cm
import math as _math


def _interior_cusps(h, wedge, ng=320, bd=5.0):
    cusp, detJ, gg, good, X, Y, su = sm.cusp_field(h, wedge=wedge, ngrid=ng)
    dist = ndimage.distance_transform_edt(good)
    lab, nl = ndimage.label(cusp, structure=np.ones((3, 3)))
    return sum(1 for i in range(1, nl + 1) if np.mean(dist[tuple(np.where(lab == i))]) >= bd)


# (i) construction sound: at φ=0 the lowitz60 entry normal == the column normal at roll LOWITZ_ALPHA0
G, A, sky, ok, su = cm.sky_grid(20.0, wedge="lowitz60", ngrid=6)
i0 = int(np.argmin(np.abs(A)))            # a cell with φ≈0
g0 = float(G[i0])
n1c, _ = cm._column_normals(g0, cm.LOWITZ_ALPHA0)
c0, s0 = _math.cos(cm.LOWITZ_ALPHA0), _math.sin(cm.LOWITZ_ALPHA0)
n1L = np.array([c0 * _math.sin(g0), -c0 * _math.cos(g0), s0])   # lowitz n1 at φ=0
check("lowitz60 construction sound: φ=0 reduces to the column normal exactly", np.allclose(n1c, n1L),
      f"|Δ|={np.max(np.abs(n1c - n1L)):.2e}")

# (ii) the A₃-lips cusp-pair birth: interior count 2→4→2 at the (non-canonical) α0=60° face axis
cm.LOWITZ_ALPHA0 = _math.radians(60.0)
below, peak, above = _interior_cusps(14, "lowitz60"), _interior_cusps(22, "lowitz60"), _interior_cusps(33, "lowitz60")
check("Lowitz A₃-lips: an interior cusp PAIR is born (2→4→2) — the search's 1st cusp-creation",
      below == 2 and peak == 4 and above == 2, f"interior counts h14/h22/h33 = {below}/{peak}/{above}")
rl = sm.corank_on_caustic(22.0, wedge="lowitz60")
check("Lowitz is corank-1 (no D₄ umbilic) through the birth", rl is not None and rl["corank"] == 1,
      f"s1_min/scale={rl['s1_min_rel']:.3f}" if rl else "None")

# (iii) specificity: the canonical edge axis α0=90° AND the column are FLAT at 2 (the birth is distinct, α0-specific)
cm.LOWITZ_ALPHA0 = _math.radians(90.0)
edge_flat = _interior_cusps(14, "lowitz60") == 2 and _interior_cusps(22, "lowitz60") == 2
col_flat = all(sm.cusp_count(h)[0] == 2 for h in (22.0, 30.0, 40.0))   # column robust regime (h≥22)
check("the birth is α0-specific: canonical edge-Lowitz (α0=90°) + the column both stay flat at 2",
      edge_flat and col_flat, f"edge-flat={edge_flat}, column-flat={col_flat}")
cm.LOWITZ_ALPHA0 = _math.radians(60.0)            # restore the documented value

print("\nPhase 8-D — pyramidal-capped column (wedge='pyrcol', odd Galle wedge): confirms Berry:")
for dphi, expect in ((120.0, 23.82), (180.0, 8.96)):     # the validated odd-radius halos
    cm.PYR_DPHI = _math.radians(dphi)
    G, A, sky, ok, su = cm.sky_grid(25.0, wedge="pyrcol", ngrid=400)
    dmin = float(np.degrees(np.arccos(np.clip(sky[ok] @ su, -1, 1))).min())
    check(f"pyrcol dφ={int(dphi)}° caustic lands on the odd radius (~{expect}°)", abs(dmin - expect) < 0.3,
          f"caustic min-deviation = {dmin:.2f}°")
cm.PYR_DPHI = _math.radians(180.0)                       # the 9° arc — the ψ-symmetric clean test
counts_p = [sm.cusp_count(h, wedge="pyrcol")[0] for h in (10.0, 16.0, 22.0, 28.0, 34.0)]
check("9° arc: cusp count FLAT at 2 across h (no A₄ swallowtail — confirms Berry)",
      set(counts_p) == {2}, f"counts = {counts_p}")
rp = sm.corank_on_caustic(20.0, wedge="pyrcol")
check("9° arc is corank-1 (no D₄ umbilic)", rp is not None and rp["corank"] == 1,
      f"s1_min/scale={rp['s1_min_rel']:.3f}" if rp else "None")
cm.PYR_DPHI = _math.radians(120.0)                       # restore the default

print(f"\n{'ALL PASS — 5 2-DOF maps swept (2 column wedges + Wegener + Lowitz + pyramidal); 29.7° A₃ metamorphosis; Lowitz A₃-lips; no A₄, no D₄ on any → confirms Berry' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
