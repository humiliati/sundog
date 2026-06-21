#!/usr/bin/env python
"""Frozen test for BoxSEL Phase-4h one-help closure.

Locks: every one-help n=2 orbit has q>=1/2 by the exact volume contradiction, so the Phase-4g
frontier drops from 45 mixed-side n=2 orbits to the 10 two-help AC-vs-BC mixed orbits.
Run: python scripts/test_boxsel_phase4h_one_help_closure.py
"""

import sys
from fractions import Fraction as F

sys.path.insert(0, "scripts")
import boxsel_kkt_exact as kkt
from boxsel_kkt_exact import Surd as S
import boxsel_phase4f_cell_atlas as atlas
import boxsel_phase4g_minpair_reduction as red
import boxsel_phase4h_one_help_closure as one

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


print("(1) one-help side detection:")
one_help_reps = [rep for rep in atlas.n2_orbits() if sum(atlas.helps(cell) for cell in rep) == 1]
check("there are 56 one-help n=2 orbits before the Phase-4g/4h closures", len(one_help_reps) == 56)
check("every one-help orbit has an AC or BC helping side",
      all(one.one_help_side(rep) in {"AC", "BC"} for rep in one_help_reps))
check("Phase 4g had already exact-closed 21 same-side one-help orbits",
      red.live_same_side_split()[1] == 21)
check("Phase 4h targets the remaining 35 one-help mixed orbits",
      red.minpair_classification()["one_help_mixed_open"] == 35)

print("(2) exact one-help lower certificate:")
check("one-help lower bound is 1/2 and strictly above q_KKT",
      one.one_help_lower_bound() == S(F(1, 2)) and one.one_help_lower_bound() > kkt.Q_STAR)
forced, upper = one.contradiction_bound(F(2, 5), F(3, 5))
check("contradiction helper: forced second-axis overlap exceeds the available atom-length upper bound",
      forced > upper, f"forced>{forced}, upper={upper}")

print("(3) n=2 atlas frontier after one-help closure:")
counts = one.one_help_closure_classification()
check("classification closes zero-help, same-side, and one-help; only two-help mixed remains",
      counts == {
          "zero_help_closed": 31,
          "same_side_closed": 47,
          "one_help_closed": 35,
          "two_help_mixed_open": 10,
      },
      f"{counts}")
check("exact-closed atlas count is 113/123", one.exact_closed_count() == 113)
check("remaining n=2 open count is 10", one.remaining_open_count() == 10)

print("(4) honest global scope:")
lo, hi = one.certified_sandwich()
check("global sandwich remains [1/4, q_KKT]", lo == kkt.CERTIFIED_LOWER and hi == kkt.Q_STAR)
check("GLOBAL_LOWER_BOUND_CLOSED remains False", one.GLOBAL_LOWER_BOUND_CLOSED is False)
check("remaining obligations are the 10 two-help mixed n=2 orbits and n>=3 compression",
      len(one.REMAINING_GLOBAL_OBLIGATIONS) == 2
      and "10 two-help" in one.REMAINING_GLOBAL_OBLIGATIONS[0]
      and "n>=3" in one.REMAINING_GLOBAL_OBLIGATIONS[1])

print(f"\n{'ALL PASS -- 4h one-help closure: all 56 one-help n=2 orbits exact-closed; total exact-closed 113/123; only 10 two-help mixed n=2 orbits remain open' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
