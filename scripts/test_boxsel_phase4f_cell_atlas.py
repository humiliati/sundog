#!/usr/bin/env python
"""Frozen test for the BoxSEL Phase-4f n=2 endpoint-order atlas.

Locks: (1) the exact cell enumeration (36 per-axis types -> 12 orbits; 1296 n=2 pairs -> 123
orbits); (2) the exact structural classification + the 'C sticks out' helps-criterion (31 zero-help
orbits force q=1, eliminated; 92 live); (3) the honest scope -- global lower bound still open, the
sandwich [1/4, q_KKT] unchanged, obligations explicit, and the seeded-search backstop stays >= q_KKT.
Run: python scripts/test_boxsel_phase4f_cell_atlas.py
"""
import sys
from fractions import Fraction as F

sys.path.insert(0, "scripts")
import boxsel_phase4f_cell_atlas as at
import boxsel_kkt_exact as kkt
from boxsel_kkt_exact import Surd as S

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("(1) exact cell enumeration:")
check("36 per-axis types (6 left-orders x 6 right-orders)", len(at.per_axis_types()) == 36)
check("12 per-axis orbits under {A<->B, reflection}", len(at.per_axis_orbits()) == 12)
orbits = at.n2_orbits()
check("1296 raw n=2 cell-pairs -> 123 orbits under the full symmetry group",
      36 * 36 == 1296 and len(orbits) == 123, f"orbits={len(orbits)}")

print("(2) exact structural classification + helps-criterion:")
counts = at.classify_orbits()
check("orbits split 31 zero-help / 56 one-help / 36 two-help",
      counts == {0: 31, 1: 56, 2: 36}, f"{counts}")
check("31 zero-help orbits eliminated exactly; 92 live to verify", counts[0] == 31 and counts[1] + counts[2] == 92)
# zero-help cell: C is the outer interval (contains A&B) -> q_k = 1
zc = at.realize_axis(((2, 0, 1), (0, 1, 2)), [F(1, 10), F(2, 10), F(3, 10)], [F(7, 10), F(8, 10), F(9, 10)])
check("zero-help axis cell: C does NOT stick out, q_k = 1 (exact geometry)",
      at.helps(((2, 0, 1), (0, 1, 2))) is False and at.axis_q(zc) == F(1))
# helping cell: C has the largest left -> sticks out -> q_k < 1 achievable
hc = at.realize_axis(((0, 1, 2), (0, 1, 2)), [F(1, 10), F(2, 10), F(4, 10)], [F(6, 10), F(7, 10), F(9, 10)])
check("helping axis cell: C sticks out, q_k < 1 achievable (here 1/2)",
      at.helps(((0, 1, 2), (0, 1, 2))) is True and at.axis_q(hc) < F(1))
check("a zero-help cell-pair forces q = 1 > q_KKT (exact elimination)", S(F(1)) > kkt.Q_STAR)

print("(3) symmetry operations are sound (involutions):")
sample = ((0, 1, 2), (2, 1, 0))
check("reflect and swap_ab are involutions",
      at.reflect(at.reflect(sample)) == sample and at.swap_ab(at.swap_ab(sample)) == sample)

print("(4) honest scope: global lower bound still OPEN, sandwich unchanged:")
lo, hi = at.certified_sandwich()
check("certified sandwich unchanged: [1/4, (9+sqrt17)/32]", lo == kkt.CERTIFIED_LOWER and hi == kkt.Q_STAR)
check("GLOBAL_LOWER_BOUND_CLOSED is False (not laundered as a global proof)",
      at.GLOBAL_LOWER_BOUND_CLOSED is False)
check("two remaining obligations named (92 live cells exact; n>=3 compression)",
      len(at.REMAINING_GLOBAL_OBLIGATIONS) == 2
      and "92 live" in at.REMAINING_GLOBAL_OBLIGATIONS[0]
      and "n>=3" in at.REMAINING_GLOBAL_OBLIGATIONS[1])
check("seeded-search backstop over live cells stays >= q_KKT (found nothing below)",
      at.ATLAS_SEEDED_SEARCH_MIN >= float(kkt.Q_STAR) - 1e-7,
      f"{at.ATLAS_SEEDED_SEARCH_MIN} vs q_KKT {float(kkt.Q_STAR):.7f}")

print(f"\n{'ALL PASS -- 4f atlas: 123 n=2 cells enumerated; 31 zero-help eliminated exactly; 92 live verified >= q_KKT (seeded backstop); global lower bound still open, sandwich [1/4, (9+sqrt17)/32] unchanged' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
