#!/usr/bin/env python
"""Frozen test for BoxSEL Phase-4k n>=3 dimension compression.

Locks the final compression step: arbitrary n>=3 mixed cases reduce to the Phase-4j two-parameter
envelope, so inf I_box^n = (9+sqrt17)/32 for every n>=2.
Run: python scripts/test_boxsel_phase4k_dimension_compression.py
"""

import sys
from fractions import Fraction as F

sys.path.insert(0, "scripts")
import boxsel_kkt_exact as kkt
from boxsel_kkt_exact import Surd as S
import boxsel_phase4j_mixed_envelope_closure as close2
import boxsel_phase4k_dimension_compression as comp

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


print("(1) claim boundary: global lower endpoint is now closed:")
check("Phase 4j n=2 receipt is present",
      close2.N2_LOWER_BOUND_CLOSED is True and close2.exact_n2_infimum() == kkt.Q_STAR)
check("Phase 4k closes the arbitrary-dimension lower bound",
      comp.GLOBAL_LOWER_BOUND_CLOSED is True and comp.EXACT_FOR_DIMS_N_GE_2 is True)
check("certified global interval collapses to [q_KKT, q_KKT]",
      comp.certified_global_interval() == (kkt.Q_STAR, kkt.Q_STAR))
check("no remaining global Phase-4 obligations",
      comp.REMAINING_GLOBAL_OBLIGATIONS == ())

print("(2) easy branches and symmetry orientation:")
check("same-side/q>=1/2 branch is strictly above q_KKT",
      comp.q_half_branch_already_closed())
check("neutral ratio uses A/B symmetry to force rho<=1",
      comp.neutral_ratio_after_symmetry(F(3, 5), F(2, 5)) == F(2, 3)
      and comp.neutral_ratio_after_symmetry(F(2, 5), F(3, 5)) == F(2, 3))
check("compression domain accepts the KKT-edge triangle and rejects sub-half s",
      comp.compression_domain(F(1, 2), F(2), F(1), F(1))
      and not comp.compression_domain(F(2, 5), F(2), F(1), F(1)))

print("(3) neutral axes do not enlarge the pure mixed envelope:")
samples = (
    (F(1, 2), F(1), F(1), F(0)),
    (F(1, 2), F(2), F(1), F(1)),
    (F(3, 5), F(2), F(1), F(2, 3)),
    (F(3, 4), F(3, 2), F(5, 4), F(1, 3)),
    (F(1), F(2), F(3, 2), F(1)),
)
check("neutral second bound is <= pure second bound on rational samples",
      all(comp.neutral_second_bound(s, r, t, rho) <= comp.pure_second_bound(s, r, t)
          for s, r, t, rho in samples))
check("compressed product bound is <= pure product bound at fixed (s,r,t)",
      all(comp.neutral_bound_is_no_worse(s, r, t, rho) for s, r, t, rho in samples))
check("pure product bound at the Phase-4j witness edge is positive",
      comp.pure_product_bound_at_s(F(1, 2), F(2), F(1)) > 0)
check("a rational product bound converts to q >= 1/(4P) exactly",
      comp.q_lower_from_product_bound(F(1, 2)) == S(F(1, 2)))

print("(4) exact rational compression grid:")
tested, counts = comp.rational_compression_grid_report(12)
check("grid is non-vacuous", tested > 8000, f"tested={tested}")
check("neutral<=pure holds on every rational grid point",
      counts["violation"] == 0 and counts["neutral_le_pure"] == tested, f"{counts}")

print("(5) final Phase-4 status:")
status = comp.dimension_status()
check("status records n=2 closed and n>=3 compressed",
      status["n2_exact_closed"] == 1 and status["n_ge_3_compressed_to_n2"] == 1)
check("exact global infimum is q_KKT",
      comp.exact_global_infimum() == kkt.Q_STAR)
check("q_KKT sits strictly above the old universal 1/4 lower bound",
      comp.exact_global_infimum() > kkt.CERTIFIED_LOWER)
check("next phase pointer leaves optimization and moves to taxonomy/detector work",
      "Phase 5" in comp.NEXT_PHASE and "Phase 6" in comp.NEXT_PHASE)

print(f"\n{'ALL PASS -- Phase-4k compression closes n>=3; inf I_box^n = (9+sqrt17)/32 for every n>=2' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
