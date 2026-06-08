#!/usr/bin/env python
"""Frozen test for Atlas Phase 7 (scripts/atlas_forward_sweep.py).

Asserts the master bifurcation set of the halo phase diagram REPRODUCES from the repo machinery (the §6
armchair gate), and that the TWO transitions the Phase-7 verify pass refuted STAY refuted (negative
receipts, so the collapsed cells are auditable). Run: python scripts/test_atlas_forward_sweep.py (~30-60 s).
"""
import sys
sys.path.insert(0, "scripts")
import atlas_forward_sweep as fs
from s2_optics import halo_min_deviation

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("the master bifurcation set reproduces (DERIVED from the repo machinery, not asserted):")
check("22° / 46° A₂ fold radii", abs(halo_min_deviation(60, fs.N) - 21.839) < 0.02
      and abs(halo_min_deviation(90, fs.N) - 45.733) < 0.02)
check("column UTA+LTA → circumscribed merge ≈ 29.7° (A-catastrophe)",
      abs(fs.cm.merge_elevation() - 29.7) < 0.5, f"{fs.cm.merge_elevation():.2f}°")
check("CZA off = 32.196° and CHA on = 57.804° (B; exact complements, sum 90°)",
      abs(fs.cza_wall() - 32.196) < 0.02 and abs(fs.cha_wall() - 57.804) < 0.02
      and abs(fs.cza_wall() + fs.cha_wall() - 90.0) < 1e-6)
plate = fs.plate_parhelia_limit()
check("plate parhelia disappear ≈ 60.7° (B; the plate forward model)",
      plate is not None and abs(plate - 60.7) < 1.5, f"{plate:.2f}°" if plate else "None")
lips = fs.lowitz_lips_birth()
check("Lowitz A₃-lips cusp-pair birth ≈ 16.1° (A; the one higher catastrophe)",
      lips is not None and abs(lips - 16.1) < 0.8, f"{lips:.2f}°" if lips else "None")

print("the two REFUTED transitions stay refuted (negative receipts — the verify pass caught fabrications):")
wmin = fs.wegener_reaches_anthelion_FALSE()
check("Wegener does NOT reach the anthelion (min ray-to-antisolar angle ≫ 0, flat ~44°)",
      wmin > 35.0, f"min angle = {wmin:.1f}° (the '22.13°' was fabricated)")
plo, phi = fs.pyramidal_lower_branch_flat()
check("pyramidal 23.8° arc has NO h≈20° lower-branch event (min-dev flat over h)",
      (phi - plo) < 0.5, f"min-dev range over h = {plo:.2f}–{phi:.2f}° (the '20°' was fabricated)")

print(f"\n{'ALL PASS — Phase-7 master bifurcation set reproduced; both fabricated transitions stay refuted' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
