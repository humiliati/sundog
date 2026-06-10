#!/usr/bin/env python
"""Frozen test for the Atlas rigor upgrade (scripts/atlas_jet_classify.py).

Asserts the SOUND iterated-kernel A3-vs-A4 jet classifier (c3 = K·∇(K·∇ det DF), smooth row-kernel K)
is correctly CALIBRATED on both controls and that the Lowitz birth is A3, not A4:
  + positive control (Morin A4 normal form): |c3| -> 0 at the A4 (registers the swallowtail);
  − negative control (column apex cusps): |c3| bounded away from 0 (registers A3);
  → Lowitz birth: |c3| stays bounded (ratio ~0.8) right at the birth -> A3-lips, NOT Berry's A4.
Run: python scripts/test_atlas_jet_classify.py  (~30-60 s).
"""
import math
import sys
sys.path.insert(0, "scripts")
import atlas_jet_classify as jc
import atlas_caustic_map as cm
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


def top_c3(vals):
    return vals[0] if vals else float("nan")


print("positive control — the Morin A4 normal form (the classifier MUST register c3->0 at the A4):")
a4_far = top_c3(jc.cusp_c3(*jc.synthetic_swallowtail(-0.40)))     # two A3 cusps, far from the merge
a4_near = top_c3(jc.cusp_c3(*jc.synthetic_swallowtail(-0.01)))    # cusps about to merge into the A4
check("synthetic A4: |c3| DIVES toward 0 as the cusps merge (ratio < 0.25)",
      a4_near / a4_far < 0.25, f"|c3|: far={a4_far:.3f}, near-A4={a4_near:.3f}, ratio={a4_near / a4_far:.2f}")

print("negative control — column apex A3 cusps (|c3| must be bounded away from 0 = a real A3 cusp):")
col = jc.cusp_c3(*jc.halo_chart(25.0, "prism60")[:3], jc.halo_chart(25.0, "prism60")[3])
check("column A3 cusps: |c3| bounded away from 0 (registers A3, not A4)", top_c3(col) > 2.0,
      f"|c3| = {top_c3(col):.3f}")

print("target — Lowitz birth (α0=60°): |c3| must STAY bounded at the birth (A3-lips), not dive (A4):")
cm.LOWITZ_ALPHA0 = math.radians(60.0)
low_birth = top_c3(jc.cusp_c3(*jc.halo_chart(17.0, "lowitz60")[:3], jc.halo_chart(17.0, "lowitz60")[3]))
low_gen = top_c3(jc.cusp_c3(*jc.halo_chart(28.0, "lowitz60")[:3], jc.halo_chart(28.0, "lowitz60")[3]))
ratio = low_birth / low_gen
check("Lowitz birth is A3-lips: |c3| at the birth stays bounded (ratio > 0.5, vs A4's < 0.25)",
      ratio > 0.5, f"|c3|: birth(h17)={low_birth:.3f}, generic(h28)={low_gen:.3f}, ratio={ratio:.2f}")
check("Lowitz |c3| is in the A3 range (comparable to the column A3 control, not ~0)",
      low_birth > 2.0, f"Lowitz |c3|(birth)={low_birth:.3f} vs column A3 |c3|={top_c3(col):.3f}")

print("corank-2 / D4 detector calibration — the Atlas's never-fired corank-2 branch, exercised on a known D4:")


def corank(chart):
    X, Y, d = chart
    return jc.corank_from_chart(X, Y, d, d)


d4 = corank(jc.synthetic_umbilic(0.0))               # the hyperbolic-umbilic D4 at the umbilic point
d4_unfolded = corank(jc.synthetic_umbilic(0.2))      # the SAME family, umbilic unfolded -> corank-1
a4 = corank(jc.synthetic_swallowtail_chart(0.0))     # the A4 swallowtail -> a corank-1 cuspoid
check("POSITIVE control: the corank-2 branch FIRES on a synthetic D4 umbilic (s1_min_rel < 0.05)",
      d4["corank"] == 2 and d4["s1_min_rel"] < 0.05, f"s1_min_rel={d4['s1_min_rel']:.4f}")
check("the D4 UNFOLDS to corank-1 as w grows (catastrophe-theory expectation; threshold not trivially low)",
      d4_unfolded["corank"] == 1 and d4_unfolded["s1_min_rel"] > 0.05,
      f"w=0.2: s1_min_rel={d4_unfolded['s1_min_rel']:.4f}")
check("NEGATIVE control: the A4 swallowtail does NOT fire corank-2 (detector distinguishes D4 from A4)",
      a4["corank"] == 1 and a4["s1_min_rel"] > 0.5, f"A4 s1_min_rel={a4['s1_min_rel']:.4f}")
check("the 0.05 threshold cleanly separates D4 (≪0.05) from A4 (≫0.05) — validates the Atlas no-D4 null",
      d4["s1_min_rel"] < 0.05 < a4["s1_min_rel"], f"D4={d4['s1_min_rel']:.4f} < 0.05 < A4={a4['s1_min_rel']:.4f}")

print(f"\n{'ALL PASS — jet classifier calibrated (A4 dives, A3 bounded), Lowitz = A3-lips; AND the corank-2/D4 branch FIRES on a known D4 (validates the Atlas no-D4 null)' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
