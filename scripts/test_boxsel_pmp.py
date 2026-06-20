#!/usr/bin/env python
"""Frozen test for the BoxSEL Phase-1 PMP replication gate (scripts/boxsel_pmp.py).

Locks: (1) the four hand cases + the paper's toy exact interval on the body Proposition 2 form;
(2) the printed Algorithm 2 slack typo and its q1=1 vacuity; (3) sharp-vs-paper (coincide at
points, paper loose on intervals); (4) the premise-shape audit -- a finite-model counterexample
where dropping 'and Q1' drives the bound below the truth.
Run: python scripts/test_boxsel_pmp.py
"""
import sys
from fractions import Fraction
sys.path.insert(0, "scripts")
import boxsel_pmp as pmp
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


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


print("(1) body Proposition 2 (point form) -- the four LOCKED hand cases:")
for q2 in (0.0, 0.3, 0.8, 1.0):
    lo, hi = pmp.pmp_point(1.0, q2)
    check(f"q1=1, q2={q2} -> [{q2}, {q2}]", approx(lo, q2) and approx(hi, q2), f"got [{lo:.3f}, {hi:.3f}]")
for q1 in (0.0, 0.3, 0.8, 1.0):
    lo, hi = pmp.pmp_point(q1, 1.0)
    check(f"q2=1, q1={q1} -> [{q1}, 1]", approx(lo, q1) and approx(hi, 1.0), f"got [{lo:.3f}, {hi:.3f}]")
for q2 in (0.0, 0.5, 1.0):
    lo, hi = pmp.pmp_point(0.0, q2)
    check(f"q1=0, q2={q2} -> [0, 1]", approx(lo, 0.0) and approx(hi, 1.0), f"got [{lo:.3f}, {hi:.3f}]")
lo, hi = pmp.pmp_point(0.2, 0.8)
check("q1=0.2, q2=0.8 -> [0.16, 0.96]", approx(lo, 0.16) and approx(hi, 0.96), f"got [{lo:.3f}, {hi:.3f}]")

print("(2) the paper's toy example exact interval IS the q1=0.2, q2=0.8 point bound:")
lo, hi = pmp.pmp_point(0.2, 0.8)
check("exact interval [0.16, 0.96] reproduced", approx(lo, 0.16) and approx(hi, 0.96), f"[{lo:.2f}, {hi:.2f}]")

print("(3) printed Algorithm 2 slack typo diverges (confirmed line 16: min(1, q1*q2 + 1 - q2)):")
body = pmp.pmp_point(0.2, 0.8)[1]
alg2 = pmp.pmp_upper_alg2(0.2, 0.8)
check("toy: body upper 0.96 vs Alg2 upper 0.36 (gap 0.60)",
      approx(body, 0.96) and approx(alg2, 0.36), f"body={body:.2f}, alg2={alg2:.2f}")
check("Alg2 FAILS the q1=1 sanity check: returns vacuous 1.0, not q2",
      approx(pmp.pmp_upper_alg2(1.0, 0.7), 1.0), f"alg2(q1=1,q2=0.7)={pmp.pmp_upper_alg2(1.0, 0.7):.2f}")
check("body upper PASSES the q1=1 sanity check (= q2)", approx(pmp.pmp_point(1.0, 0.7)[1], 0.7))

print("(4) sharp vs paper Prop 2 upper -- coincide at points, paper loose on intervals:")
check("coincide at point premises (l=u)",
      approx(pmp.pmp_interval(0.2, 0.2, 0.8, 0.8)[1], pmp.pmp_upper_sharp(0.2, 0.2, 0.8, 0.8)))
paper_iv = pmp.pmp_interval(0.5, 1.0, 0.0, 0.5)[1]
sharp_iv = pmp.pmp_upper_sharp(0.5, 1.0, 0.0, 0.5)
check("interval premises: sharp (0.75) < paper (1.0), both sound (>= true max 0.75)",
      approx(sharp_iv, 0.75) and approx(paper_iv, 1.0) and sharp_iv <= paper_iv,
      f"sharp={sharp_iv:.2f}, paper={paper_iv:.2f}")

print("(5) premise-shape audit -- dropping 'and Q1' is NOT harmless in general (counterexample):")
Q1, A, Q2 = pmp.premise_shape_counterexample()
check("model has A not subset Q1, so the drift is live (not harmless)",
      not pmp.premise_shape_harmless(Q1, A, Q2))
q1, q2m, up_m, true_v = pmp.pmp_point_from_model(Q1, A, Q2, marginal_premise=True)
check("plugging P(Q2|A) (Alg 2) gives upper 0.52 BELOW the true 0.80 -> SOUNDNESS BREAK",
      up_m == Fraction(13, 25) and true_v == Fraction(4, 5) and up_m < true_v,
      f"q1={q1}, P(Q2|A)={q2m}, upper={float(up_m):.2f} < true={float(true_v):.2f}")
q1c, q2c, up_c, true_c = pmp.pmp_point_from_model(Q1, A, Q2, marginal_premise=False)
check("the CORRECT premise P(Q2 | A and Q1)=1 restores a sound upper (>= true)",
      q2c == 1 and up_c >= true_c, f"P(Q2|A and Q1)={q2c}, upper={float(up_c):.2f} >= true={float(true_c):.2f}")
check("control: when A subset Q1 the drift IS harmless",
      pmp.premise_shape_harmless({1, 2, 3, 4, 5}, {1, 2, 3, 4}, {1, 2, 3}))

print(f"\n{'ALL PASS -- body PMP locked (4 hand cases + toy [0.16,0.96]); Alg2 slack typo + premise drift characterized; premise drift breaks soundness when A not-subset Q1' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
