#!/usr/bin/env python
"""Frozen test for Atlas Phase 12 (scripts/atlas_phase12_polarized_phasediagram.py): the raytracer
closes the Phase-7 forward-model gaps and adds the polarization layer. Locks: every classified feature
carries a polarization signature (refraction folds -> the radial DoP(R) ladder; TIR features -> the
antisymmetric ±V net->0), and the raytracer reproduces the random-ring HORIZON-CLIP occlusion (the
flagged gap). Run: python scripts/test_atlas_phase12_polarized_phasediagram.py
"""
import sys
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_optics as so                              # noqa: E402
import atlas_phase12_polarized_phasediagram as p12  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("Atlas Phase 12 — forward-generated polarization phase diagram:\n")

layer = p12.polarization_layer()
refr = [r for r in layer if r["kind"] == "refraction"]
tir = [r for r in layer if r["kind"] == "tir"]

# (1) the polarization layer covers BOTH refraction folds and TIR features
check("the polarization layer covers refraction folds AND TIR features",
      len(refr) >= 6 and len(tir) >= 3, f"{len(refr)} refraction, {len(tir)} TIR")

# (2) refraction folds carry the radial DoP(R) law (matches halo_pol_dop, U=0, no V)
ok_dop = all(abs(r["dop"] - so.halo_pol_dop(r["R"])) < 1e-12 and r["circular"] == "V=0" for r in refr)
check("refraction folds carry the radial DoP(R) law (U=0, no circular V)", ok_dop)

# (3) the refraction DoP ladder is monotone in radius (9° faint -> 46° strong)
dops = [r["dop"] for r in sorted(refr, key=lambda x: x["R"])]
check("the refraction DoP ladder is monotone in radius", all(np.diff(dops) > 0),
      f"DoP {dops[0]*100:.2f}% (9°) -> {dops[-1]*100:.2f}% (46°)")

# (4) TIR features carry the antisymmetric ±V (net->0) signature
check("TIR features carry the antisymmetric ±V (net->0) signature",
      all("antisymmetric" in r["circular"] for r in tir))

# (5) horizon-clip elevation = R (a ring clears the horizon when the sun rises to its radius)
check("horizon-clip elevation equals the ring radius for every fold",
      all(abs(p12.horizon_clip_elevation(R) - R) < 1e-9 for _, R in p12.REFRACTION_FOLDS))

# (6) the raytracer REPRODUCES the horizon-clip: 22° ring below-horizon flux drops from substantial
#     (sun below the ring radius) to ~0 (sun above it) — the flagged random-orientation gap, closed
lo, hi = p12.validate_horizon_clip(n_orient=5000, seed=32)
check("raytracer reproduces the horizon-clip occlusion (below-horizon flux: high below R -> ~0 above R)",
      lo > 0.15 and hi < 0.03, f"below-horizon flux {lo*100:.0f}% (e<R) -> {hi*100:.0f}% (e>R)")

print(f"\n{'ALL PASS -- the polarized raytracer closes the Phase-7 forward-model gaps (it IS the random/plate/Parry generator, and it reproduces the random-ring horizon-clip occlusion) and adds the POLARIZATION LAYER: every refraction fold carries the radial DoP(R) ladder (U=0, no V), every TIR feature the antisymmetric ±V (net->0). Orientation-driven boundary elevations stay with the Phase-7 machinery (honest boundary).' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
