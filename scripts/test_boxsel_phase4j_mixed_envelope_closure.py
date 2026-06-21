#!/usr/bin/env python
"""Frozen test for BoxSEL Phase-4j mixed-envelope closure.

Locks the exact maximum certificate for the Phase-4i two-help mixed envelope. This closes all
123 n=2 endpoint-order orbits, but deliberately leaves the arbitrary-dimension global lower bound
open until the n>=3 compression proof is banked.
Run: python scripts/test_boxsel_phase4j_mixed_envelope_closure.py
"""

import sys
from fractions import Fraction as F

sys.path.insert(0, "scripts")
import boxsel_kkt_exact as kkt
from boxsel_kkt_exact import Surd as S
import boxsel_phase4i_two_help_mixed_core as core
import boxsel_phase4j_mixed_envelope_closure as close

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


print("(1) frontier accounting: all n=2 endpoint-order orbits are closed:")
counts = close.n2_closure_counts()
check("Phase 4i frontier still has exactly 10 two-help mixed reps",
      len(core.open_two_help_mixed_reps()) == 10)
check("4j assigns those 10 to two_help_mixed_closed",
      counts["two_help_mixed_closed"] == 10)
check("123/123 n=2 orbits exact-closed",
      sum(v for k, v in counts.items() if k.endswith("_closed")) == 123 and counts["open_n2"] == 0,
      f"{counts}")
check("n=2 lower bound is closed but global arbitrary-dim lower bound is not",
      close.N2_LOWER_BOUND_CLOSED is True and close.GLOBAL_LOWER_BOUND_CLOSED is False)
check("only remaining global obligation is n>=3 compression",
      len(close.REMAINING_GLOBAL_OBLIGATIONS) == 1
      and "n>=3" in close.REMAINING_GLOBAL_OBLIGATIONS[0])

print("(2) exact endpoint value:")
check("P(2,1) = X_OPT exactly",
      core.balance_product_at_r2_t1() == kkt.X_OPT)
check("1/(4*P(2,1)) = q_KKT exactly",
      core.candidate_query_from_product(core.balance_product_at_r2_t1()) == kkt.Q_STAR)
check("exact n=2 infimum reports q_KKT",
      close.exact_n2_infimum() == kkt.Q_STAR)

print("(3) M>0 => F>=0 proof ingredients:")
sample_as = (F(0), F(1, 4), F(1, 2), F(3, 4), F(1))
check("M is decreasing in b on the shifted triangle",
      all(close.m_derivative_upper_bound(a) < S(0) for a in sample_as))
check("M(a,a) is negative on the right boundary",
      all(close.m_at_b_equals_a(a) < S(0) for a in sample_as))
check("M(a,0) is increasing in a",
      all(close.m_at_b0_derivative(a) > S(0) for a in sample_as))
check("F is concave in b",
      all(close.f_b2_coefficient(a) < S(0) for a in sample_as))
check("F(a,0)'s positive threshold is sqrt17 - 4",
      close.f0_positive_root() == S(-4, 1)
      and close.m0_at_f0_positive_root() == -S(12) * S(-4, 1)
      and close.m0_at_f0_positive_root() < S(0))
check("F(a,0) is nonnegative once M(a,0)>0 on representative samples",
      all(close.f_at_b0(a) >= S(0) for a in (F(1, 4), F(1, 2), F(3, 4), F(1))))
check("Res_b(F,M) is strictly negative for live a>0 samples",
      all(close.common_zero_resultant(a) < S(0) for a in (F(1, 4), F(1, 2), F(3, 4), F(1))))

print("(4) squared residual factorization:")
factor_samples = ((F(1, 4), F(0)), (F(1, 2), F(1, 4)), (F(3, 4), F(1, 2)), (F(1), F(0)), (F(1), F(1, 2)))
check("D-L^2 factorization matches exactly on rational samples",
      all(close.squared_residual_direct(a, b) == close.squared_residual_factored(a, b)
          for a, b in factor_samples))
check("known witness has zero squared residual factor",
      close.squared_residual_factored(F(1), F(0)) == S(0))
check("non-witness M>0 samples have nonnegative F",
      all(close.f_polynomial(a, b) >= S(0)
          for a, b in factor_samples
          if close.m_polynomial(a, b) > S(0)))

print("(5) exact rational envelope guard:")
tested, branch_counts = close.exact_envelope_grid_report(80)
check("grid guard is non-vacuous", tested > 3000, f"tested={tested}")
check("both proof branches are exercised",
      branch_counts["automatic"] > 0 and branch_counts["squared"] > 0, f"{branch_counts}")
check("no rational grid point violates the exact branch proof",
      branch_counts["violation"] == 0, f"{branch_counts}")

print("(6) global claim boundary:")
lo, hi = close.certified_global_sandwich()
check("global sandwich remains [1/4, q_KKT] until n>=3 compression",
      lo == kkt.CERTIFIED_LOWER and hi == kkt.Q_STAR and lo < hi)

print(f"\n{'ALL PASS -- Phase-4j closes the 10 two-help mixed n=2 orbits; n=2 infimum exact; global n>=3 compression remains open' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
