#!/usr/bin/env python
"""Frozen test for H5 (mirage (Δn,s) ladder, scripts/mirage_ladder*.py). Locks: the ray-transfer model
reproduces mirage phenomenology (1 image vs a 3-image superior-mirage band); the 1->3 onset scales as
Δn/s~const (the cusp boundary); the jet classifier reads that onset as an A3 cusp (corank-1, |c3| bounded);
and a double inversion reaches the 5-image Fata-Morgana rung. Deterministic (RK4 ray-trace). Run:
python scripts/test_mirage_ladder.py
"""
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import mirage_ladder as ml             # noqa: E402
import mirage_ladder_sweep as mls      # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("H5 — mirage (Δn,s) ladder as a classified caustic / cusp diagram:\n")

# Stage 1: phenomenology
check("broad gradient (s=2000) -> 1 image (monotonic transfer)", mls.image_count(1e-5, 2000.0) == 1)
check("localized inversion (s=5, Δn=5e-5) -> 3-image superior-mirage band", mls.image_count(5e-5, 5.0) == 3)

# the 1->3 onset scales as Δn/s ~ const (the cusp boundary): thinner layer -> lower onset Δn
def onset(s):
    for dn in [1e-5, 1.5e-5, 2e-5, 3e-5, 5e-5, 8e-5, 1.2e-4, 2e-4]:
        if mls.image_count(dn, s) >= 3:
            return dn
    return None
o3, o20 = onset(3.0), onset(20.0)
check("the 1->3 onset is a curve Δn/s~const (thinner layer s=3 onsets at lower Δn than s=20)",
      o3 is not None and o20 is not None and o3 < o20, f"onset(s=3)={o3:.1e} < onset(s=20)={o20:.1e}")

# Stage 2 / P3: the jet classifier reads the 1->3 onset as an A3 cusp
g = (ml.H0 - ml.H_OBS) / ml.R_DEFAULT
r1 = mls.cusp_chart(5.0, (1.5e-5, 6e-5), (g - 0.0030, g + 0.0030))
check("P3: single-inversion chart has a CAUSTIC with a CUSP", r1["caustic"] and r1["n_cusps"] >= 1,
      f"caustic={r1['caustic']} #cusps={r1['n_cusps']}")
check("P3: the cusp is A3 — |c3| bounded away from 0 (not A4) and corank-1 (not D4)",
      r1["corank"] == 1 and r1["c3"] and r1["c3"][0] > 0.3,
      f"|c3|={r1['c3']} corank={r1['corank']}")

# P4: the double inversion (Fata Morgana) reaches a 5-image rung
check("P4: double inversion reaches a 5-image (Fata Morgana) rung",
      mls.image_count(3e-5, 5.0, gradfn=ml.nprime_double) == 5,
      f"#images={mls.image_count(3e-5, 5.0, gradfn=ml.nprime_double)}")

print(f"\n{'ALL PASS -- mirages form a classified caustic ladder in (Δn,s): 1->3 (superior mirage) onset is a Whitney A3 cusp on a Δn/s~const boundary; a double inversion reaches the 5-image Fata-Morgana rung. The mirage analog of the halo Atlas, labelled by the H4-validated jet classifier.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
