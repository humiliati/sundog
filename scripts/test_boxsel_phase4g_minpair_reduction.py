#!/usr/bin/env python
"""Frozen test for BoxSEL Phase-4g min-pair reduction.

Locks: (1) triple-overlap equals the minimum pairwise overlap on every axis cell; (2) the exact
same-side product certificate closes 47/92 live n=2 orbits; (3) total exact-closed atlas count is
78/123, leaving 45 mixed-side n=2 orbits plus n>=3 compression.
Run: python scripts/test_boxsel_phase4g_minpair_reduction.py
"""

import sys
from fractions import Fraction as F

sys.path.insert(0, "scripts")
import boxsel_kkt_exact as kkt
from boxsel_kkt_exact import Surd as S
import boxsel_phase4f_cell_atlas as atlas
import boxsel_phase4g_minpair_reduction as red

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


print("(1) per-axis min-pair identity on all 36 endpoint-order cells:")
pattern_counts = {}
for cell in atlas.per_axis_types():
    intervals = atlas.realize_axis(cell, [F(1, 10), F(2, 10), F(3, 10)], [F(7, 10), F(8, 10), F(9, 10)])
    pair_values = {label: red.pair_overlap(intervals, label) for label in ("AB", "AC", "BC")}
    triple = red.triple_overlap(intervals)
    pattern = red.min_pair_pattern(cell)
    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    if triple != min(pair_values.values()):
        check(f"triple=min pairwise for {cell}", False, f"triple={triple}, pairs={pair_values}")
    if "=" not in pattern and triple != pair_values[pattern]:
        check(f"pattern {pattern} names the minimum pair for {cell}", False, f"triple={triple}, pairs={pair_values}")
check("all 36 cells accounted for by the six min-pair patterns",
      pattern_counts == {"AB": 8, "AC": 8, "BC": 8, "AB=AC": 4, "AB=BC": 4, "AC=BC": 4},
      f"{pattern_counts}")

print("(2) exact side-factor rule:")
check("AB strict has no AC/BC side certificate", red.exact_factor_sides("AB") == frozenset())
check("AB=AC certifies AC side; AB=BC certifies BC side",
      red.exact_factor_sides("AB=AC") == frozenset({"AC"})
      and red.exact_factor_sides("AB=BC") == frozenset({"BC"}))
check("AC=BC can certify either side",
      red.exact_factor_sides("AC=BC") == frozenset({"AC", "BC"}))
check("AC and BC strict certify their own side",
      red.exact_factor_sides("AC") == frozenset({"AC"})
      and red.exact_factor_sides("BC") == frozenset({"BC"}))

print("(3) n=2 orbit classification after min-pair reduction:")
counts = red.minpair_classification()
check("classification closes 31 zero-help + 47 same-side live, leaves 45 mixed-side open",
      counts == {
          "zero_help_closed": 31,
          "same_side_closed": 47,
          "one_help_mixed_open": 35,
          "two_help_mixed_open": 10,
      },
      f"{counts}")
check("same-side live split is 21 one-help + 26 two-help",
      red.live_same_side_split() == {1: 21, 2: 26}, f"{red.live_same_side_split()}")
check("exact-closed atlas count is 78/123", red.exact_closed_count() == 78)
check("remaining n=2 open count is 45", red.remaining_open_count() == 45)

print("(4) exact lower certificates and honest global scope:")
check("same-side closed orbits have q >= 1/2 > q_KKT",
      red.same_side_lower_bound() == S(F(1, 2)) and red.same_side_lower_bound() > kkt.Q_STAR)
lo, hi = red.certified_sandwich()
check("global sandwich remains [1/4, q_KKT]", lo == kkt.CERTIFIED_LOWER and hi == kkt.Q_STAR)
check("GLOBAL_LOWER_BOUND_CLOSED remains False", red.GLOBAL_LOWER_BOUND_CLOSED is False)
check("remaining obligations include 35 one-help, 10 two-help, and n>=3 compression",
      len(red.REMAINING_GLOBAL_OBLIGATIONS) == 3
      and "35 one-help" in red.REMAINING_GLOBAL_OBLIGATIONS[0]
      and "10 two-help" in red.REMAINING_GLOBAL_OBLIGATIONS[1]
      and "n>=3" in red.REMAINING_GLOBAL_OBLIGATIONS[2])

print(f"\n{'ALL PASS -- 4g min-pair reduction: 47/92 live n=2 orbits exact-closed; total exact-closed 78/123; 45 mixed-side n=2 orbits remain open' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
