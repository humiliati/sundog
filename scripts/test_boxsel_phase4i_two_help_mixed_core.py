#!/usr/bin/env python
"""Frozen test for BoxSEL Phase-4i two-help mixed-core reduction.

Locks the reduction of the 10 remaining n=2 mixed cells to the two-parameter envelope P(r,t).
Phase 4j closes the envelope maximum; GLOBAL_LOWER_BOUND_CLOSED must still remain False until
n>=3 compression is proved.
Run: python scripts/test_boxsel_phase4i_two_help_mixed_core.py
"""

import sys
from fractions import Fraction as F

sys.path.insert(0, "scripts")
import boxsel_kkt_exact as kkt
import boxsel_phase4i_two_help_mixed_core as core

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


print("(1) the remaining n=2 frontier is exactly the 10 two-help mixed orbits:")
reps = core.open_two_help_mixed_reps()
check("10 two-help mixed reps remain", len(reps) == 10)
check("all remaining reps are AC-vs-BC mixed patterns",
      core.open_mixed_pattern_counts() == {("AC", "BC"): 10},
      f"{core.open_mixed_pattern_counts()}")

print("(2) exact endpoint identities for the mixed-core envelope:")
check("P(2,1) = X_OPT = (9 - sqrt17)/8 exactly",
      core.balance_product_at_r2_t1() == kkt.X_OPT)
check("1/(4*P(2,1)) = q_KKT exactly",
      core.candidate_query_from_product(core.balance_product_at_r2_t1()) == kkt.Q_STAR)
t1 = core.balance_product_at_t1(F(2, 1))
check("P(r,1) data at r=2 has radicand 17 and sqrt coefficient 1",
      t1["radicand"] == 17 and t1["sqrt_coefficient"] == 1)
check("P(r,1) monotonic certificate has positive squared margin 16r^3 on sample rationals",
      all(core.derivative_positive_margin_at_t1(r) > 0 for r in (F(1), F(3, 2), F(2))))

print("(3) two-parameter rational grid guard (not a proof):")
tested, violations, best = core.rational_envelope_grid_report(80)
check("grid guard is non-vacuous", tested > 3000, f"tested={tested}")
check("grid guard finds no P(r,t) above X_OPT", violations == 0, f"best={best:.9f}, target={float(kkt.X_OPT):.9f}")
check("grid guard approaches the target at (r,t)=(2,1)", abs(best - float(kkt.X_OPT)) < 1e-12)

print("(4) honest scope:")
lo, hi = core.certified_sandwich()
check("global sandwich remains [1/4, q_KKT]", lo == kkt.CERTIFIED_LOWER and hi == kkt.Q_STAR)
check("GLOBAL_LOWER_BOUND_CLOSED remains False", core.GLOBAL_LOWER_BOUND_CLOSED is False)
check("Phase 4j is recorded; remaining global obligation is n>=3 compression",
      core.MIXED_ENVELOPE_CLOSED_BY_PHASE4J is True
      and len(core.REMAINING_GLOBAL_OBLIGATIONS) == 1
      and "n>=3" in core.REMAINING_GLOBAL_OBLIGATIONS[0])

print(f"\n{'ALL PASS -- 4i mixed-core reduction: 10 n=2 orbits reduce to P(r,t); endpoint q_KKT exact; envelope closed by 4j; n>=3 remains open' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
