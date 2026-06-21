#!/usr/bin/env python
"""Frozen test for BoxSEL Phase-4e lower-bound closure start.

Locks: (1) the global lower bound is still open, with certified sandwich
[1/4, (9+sqrt17)/32]; (2) the Phase-4D KKT point is a true minimum inside its structured 2-D
normal form; (3) the remaining proof obligations are explicit rather than laundered as equality.
Run: python scripts/test_boxsel_phase4e_lower_bound.py
"""

import sys
from fractions import Fraction as F

sys.path.insert(0, "scripts")
import boxsel_kkt_exact as kkt
from boxsel_kkt_exact import Surd as S
import boxsel_phase4e_lower_bound as lb

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


print("(1) global claim boundary: lower-bound closure is still open:")
lo, upper = lb.candidate_sandwich()
check("global sandwich is [1/4, q_KKT], not an equality claim",
      lo == S(F(1, 4)) and upper == kkt.Q_STAR and lo < upper and lb.LOWER_BOUND_CLOSED is False)
check("remaining obligations name both missing pieces",
      len(lb.REMAINING_GLOBAL_OBLIGATIONS) == 2
      and "2D endpoint-order" in lb.REMAINING_GLOBAL_OBLIGATIONS[0]
      and "n>=3" in lb.REMAINING_GLOBAL_OBLIGATIONS[1])

print("(2) exact structured-family certificate at the KKT point:")
cert = lb.restricted_family_certificate()
for key, value in cert.items():
    check(f"{key}", value is True, f"value={value}")
check("x* is the lower root: 4x^2 - 9x + 4 = 0 exactly",
      lb.root_polynomial(kkt.X_OPT) == S(0))
z_star = S(2) * (S(1) - kkt.X_OPT)
check("structured formula matches exact box geometry at the candidate",
      lb.as_surd(lb.structured_query_formula(kkt.X_OPT, z_star)) == lb.structured_query_geometry(kkt.X_OPT, z_star))
check("candidate query equals (9+sqrt17)/32 exactly",
      lb.structured_query_geometry(kkt.X_OPT, z_star) == kkt.Q_STAR)

print("(3) reduced one-variable boundary behavior:")
check("boundary q0(1) = 1/2 > q_KKT",
      lb.as_surd(lb.structured_boundary_query(F(1))) == S(F(1, 2))
      and lb.as_surd(lb.structured_boundary_query(F(1))) > kkt.Q_STAR)
check("q0'(x) positive before 4/5, zero at 4/5, negative after",
      lb.structured_boundary_derivative(F(3, 5)) > 0
      and lb.structured_boundary_derivative(F(4, 5)) == 0
      and lb.structured_boundary_derivative(F(9, 10)) < 0)
check("q(x,z) is concave in z inside the structured family",
      lb.structured_q_second_derivative_z(F(3, 5), F(4, 5)) < 0
      and lb.structured_q_second_derivative_z(F(9, 10), F(1, 1)) < 0)
check("structured feasible x interval starts above 3/5 and is nonempty by 2/3",
      lb.structured_z_interval(F(3, 5)) is None and lb.structured_z_interval(F(2, 3)) is not None)

print("(4) exact rational structured-family grid smoke:")
tested, violations, best = lb.rational_structured_grid_report(120)
check("grid smoke is non-vacuous", tested > 50, f"tested={tested}")
check("no rational structured-family sample beats q_KKT", violations == 0, f"best={best}")
check("best sampled value stays >= q_KKT",
      best is not None and best >= kkt.Q_STAR, f"best={best}, q_KKT={kkt.Q_STAR}")

print(f"\n{'ALL PASS -- Phase-4e started: structured-family lower bound certified; global lower bound still open with obligations explicit' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
