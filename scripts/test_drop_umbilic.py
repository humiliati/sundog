#!/usr/bin/env python
"""Frozen test for the raindrop-umbilic referee (scripts/drop_umbilic.py) — Step 2 of "referee the rainbow's
umbilic". Asserts the Atlas's corank-2 / D4 detector, fed a real oblate-drop primary-rainbow forward chart,
fires corank-2 ONLY in the Marston & Trinh / Nye 1984 hyperbolic-umbilic band (D/H = 1.305 ± 0.016) and
stays corank-1 for the sphere and the over-flattened drop. INDEPENDENT-REFEREE validation of a published
physical D4 — NOT a new-physics claim (the raindrop D4 is Marston/Nye's). Run: python scripts/test_drop_umbilic.py
"""
import sys
sys.path.insert(0, "scripts")
import drop_umbilic as du

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("the corank-2 / D4 detector on the oblate-drop primary-rainbow chart (Marston/Nye umbilic D/H=1.305):")
sphere = du.corank_at(1.00)
check("sphere (D/H=1): corank-1 — no D4 umbilic (the axisymmetric rainbow fold)",
      sphere["corank"] == 1 and sphere["s1_min_rel"] > 0.5, f"s1_min_rel={sphere['s1_min_rel']:.3f}")

umbilic = du.corank_at(1.305)
check("the literature D4 (D/H=1.305): detector FIRES corank-2 (s1_min_rel < 0.05)",
      umbilic["corank"] == 2 and umbilic["s1_min_rel"] < 0.05, f"s1_min_rel={umbilic['s1_min_rel']:.4f}")

far = du.corank_at(1.50)
check("over-flattened (D/H=1.5): the umbilic has unfolded → corank-1",
      far["corank"] == 1 and far["s1_min_rel"] > 0.2, f"s1_min_rel={far['s1_min_rel']:.3f}")

below = du.corank_at(1.28)["corank"]
above = du.corank_at(1.33)["corank"]
check("the firing band is LOCALIZED on the literature value (corank-1 at 1.28 and 1.33, just outside 1.305±0.016)",
      below == 1 and above == 1, f"corank at D/H=1.28 -> {below}, D/H=1.33 -> {above}")

print(f"\n{'ALL PASS — the corank-2/D4 detector reproduces the Marston/Nye raindrop hyperbolic umbilic at D/H≈1.305 from ray optics; no-D4 null validated on a REAL physical D4' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
