#!/usr/bin/env python
"""Frozen smoke test for the BoxSEL Phase-2 exact micro-SEL oracle.

Run: python scripts/test_boxsel_exact_oracle.py
"""
import sys
from fractions import Fraction

sys.path.insert(0, "scripts")

import boxsel_exact_oracle as oracle
import boxsel_pmp as pmp

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

fail = 0


def check(name, condition, detail=""):
    global fail
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not condition:
        fail += 1


def approx(a, b, tol=1e-8):
    return abs(a - b) <= tol


print("(1) exact oracle agrees with Phase-1 PMP hand cases:")
for q2 in (0.0, 0.3, 0.8, 1.0):
    got = oracle.exact_pmp_interval(Fraction(1), Fraction(str(q2))).interval()
    want = pmp.pmp_point(1.0, q2)
    check(f"q1=1, q2={q2} -> [{q2}, {q2}]",
          approx(got[0], want[0]) and approx(got[1], want[1]),
          f"got [{got[0]:.3f}, {got[1]:.3f}]")

for q1 in (0.0, 0.3, 0.8, 1.0):
    got = oracle.exact_pmp_interval(Fraction(str(q1)), Fraction(1)).interval()
    want = pmp.pmp_point(q1, 1.0)
    check(f"q2=1, q1={q1} -> [{q1}, 1]",
          approx(got[0], want[0]) and approx(got[1], want[1]),
          f"got [{got[0]:.3f}, {got[1]:.3f}]")

for q2 in (0.0, 0.5, 1.0):
    got = oracle.exact_pmp_interval(Fraction(0), Fraction(str(q2))).interval()
    want = pmp.pmp_point(0.0, q2)
    check(f"q1=0, q2={q2} -> [0, 1]",
          approx(got[0], want[0]) and approx(got[1], want[1]),
          f"got [{got[0]:.3f}, {got[1]:.3f}]")

toy = oracle.exact_pmp_interval(Fraction(1, 5), Fraction(4, 5)).interval()
check("paper toy exact interval q1=0.2, q2=0.8 -> [0.16, 0.96]",
      approx(toy[0], 0.16) and approx(toy[1], 0.96),
      f"got [{toy[0]:.2f}, {toy[1]:.2f}]")

print("(2) interval-premise oracle prefers the sharp upper over the loose paper Prop-2 upper:")
C = oracle.atom("C")
D = oracle.atom("D")
E = oracle.atom("E")
ontology = [
    oracle.conditional(D, C, Fraction(1, 2), Fraction(1)),
    oracle.conditional(E, C & D, Fraction(0), Fraction(1, 2)),
]
exact = oracle.exact_interval(ontology, E, C)
paper_upper = pmp.pmp_interval(0.5, 1.0, 0.0, 0.5)[1]
sharp_upper = pmp.pmp_upper_sharp(0.5, 1.0, 0.0, 0.5)
check("exact upper is sharp 0.75, while paper interval upper is loose 1.0",
      exact.feasible and approx(exact.lower, 0.0) and approx(exact.upper, sharp_upper)
      and approx(paper_upper, 1.0),
      f"exact=[{exact.lower:.2f}, {exact.upper:.2f}], paper_upper={paper_upper:.2f}")

print("(3) bounded search finds an as-printed Algorithm 2 shipped soundness counterexample:")
ce = oracle.find_alg2_shipped_counterexample(max_n=4)
check("counterexample found with live premise drift and shipped upper below true query", ce is not None)
if ce is not None:
    check("drift is live: P(Q2|A) != P(Q2|A and Q1)", ce.q2_marginal != ce.q2_correct,
          f"{ce.q2_marginal} vs {ce.q2_correct}")
    check("as-printed shipped upper is below true P(Q2|Q1)",
          ce.shipped_upper < ce.true_query,
          f"upper={ce.shipped_upper} < true={ce.true_query}")
    check("small explicit witness is the expected 3-point pattern",
          ce.universe == frozenset({1, 2, 3})
          and ce.q1_set == frozenset({1, 2})
          and ce.a_set == frozenset({1, 3})
          and ce.q2_set == frozenset({1, 2}),
          f"U={sorted(ce.universe)} Q1={sorted(ce.q1_set)} A={sorted(ce.a_set)} Q2={sorted(ce.q2_set)}")

print("(4) tiny ontology/corpus generator is deterministic and oracle-feasible:")
corpus_a = oracle.generate_tiny_corpus(case_count=6, seed=240711821)
corpus_b = oracle.generate_tiny_corpus(case_count=6, seed=240711821)
check("generator returns the requested case count", len(corpus_a) == 6)
check("generator is deterministic under the same seed",
      [case.summary_key() for case in corpus_a] == [case.summary_key() for case in corpus_b])
for case in corpus_a:
    exact_case = case.exact()
    check(f"{case.case_id}: source model satisfies generated ontology",
          all(case.source_model.satisfies(ax) for ax in case.ontology))
    check(f"{case.case_id}: exact interval feasible and contains source query value",
          exact_case.feasible
          and exact_case.lower <= float(case.source_query_value) + 1e-8
          and exact_case.upper + 1e-8 >= float(case.source_query_value),
          f"exact=[{exact_case.lower:.3f}, {exact_case.upper:.3f}], source={float(case.source_query_value):.3f}")

widths = [case.exact().upper - case.exact().lower for case in corpus_a]
check("generated smoke corpus includes at least one non-vacuous interval",
      any(width < 0.999 for width in widths),
      f"widths={[round(width, 3) for width in widths]}")

print("(5) finite-counting vs type-volume semantic-alignment smoke (scale-invariance):")
for case in corpus_a:
    check(f"{case.case_id}: counts == normalized volumes == rescaled (diff weights, same constraints)",
          oracle.finite_type_volume_alignment(case))
nontrivial = [c for c in corpus_a
              if oracle.normalized_weights(c.source_model.weights) != c.source_model.weights]
check("alignment is NON-TRIVIAL: >=1 case has normalized volumes != integer counts, yet constraints agree",
      len(nontrivial) >= 1 and all(oracle.finite_type_volume_alignment(c) for c in nontrivial),
      f"{len(nontrivial)}/{len(corpus_a)} cases have normalized volumes != integer counts")

print("(6) the oracle is exact-rational (certified ground truth, no floats):")
toy_exact = oracle.exact_pmp_interval(Fraction(1, 5), Fraction(4, 5)).interval_exact()
check("toy exact endpoints are exact Fractions [4/25, 24/25]",
      toy_exact == (Fraction(4, 25), Fraction(24, 25))
      and all(isinstance(v, Fraction) for v in toy_exact),
      f"{toy_exact}")
thirds = oracle.exact_pmp_interval(Fraction(1, 3), Fraction(2, 3)).interval_exact()
check("thirds case stays exact (a float trap): q1=1/3, q2=2/3 -> [2/9, 8/9]",
      thirds == (Fraction(2, 9), Fraction(8, 9)), f"{thirds}")

print(f"\n{'ALL PASS -- Phase-2 exact micro-SEL oracle (rational simplex) + tiny corpus/alignment smoke green' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
