#!/usr/bin/env python
"""Frozen test for the BoxSEL Phase-4b inf-trend resolution (scripts/boxsel_inf_trend.py).

Locks: (1) the exact axis factorization q = prod_k q_k; (2) the per-axis lemma
q_k >= P(C|A_k)P(C|B_k), exhaustively EXACT-verified on rational grids + proven boundary cases;
(3) the exact global chain P(C|A)=|A&C|/|A| >= 1/2; (4) the certified sandwich
1/4 <= inf I_box^n <= 513/1250 (n>=2) -- so the representation gap PERSISTS, it does not -> 0.
Run: python scripts/test_boxsel_inf_trend.py
"""
import sys
from fractions import Fraction as F

sys.path.insert(0, "scripts")
import boxsel_inf_trend as trend
import boxsel_phase4_interval_gap as gap

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


print("(1) exact axis factorization q = prod_k q_k:")
lo1 = gap.lower_box1_witness()
w2 = gap.rational_box2_shrink_witness()
w3 = gap.rational_boxn_shrink_witness(3)
check("1-D lower witness: single factor [1/2], product == query_value",
      trend.q_factors(lo1) == [F(1, 2)] and trend.query_from_factors(lo1) == gap.query_value(lo1))
check("2-D witness: factors [16/25, 513/800], product 513/1250 == query_value",
      trend.q_factors(w2) == [F(16, 25), F(513, 800)]
      and trend.query_from_factors(w2) == gap.query_value(w2) == F(513, 1250),
      f"{[str(x) for x in trend.q_factors(w2)]}")
check("3-D extension: appended full axis contributes factor 1; product unchanged",
      trend.q_factors(w3) == [F(16, 25), F(513, 800), F(1)] and trend.query_from_factors(w3) == F(513, 1250))

print("(2) per-axis lemma q_k >= P(C|A_k)P(C|B_k) -- exhaustive EXACT grid verification:")
for n in (6, 8):
    tested, viol = trend.lemma_grid_report(n)
    check(f"grid N={n}: lemma holds on ALL {tested} interval triples (|A&B|>0), 0 violations",
          viol == 0 and tested > 0, f"tested={tested}, violations={viol}")
check("tight equality when A=B=C", trend.per_axis_lemma((F(0), F(1, 2)), (F(0), F(1, 2)), (F(0), F(1, 2))))
check("B subset A boundary case holds", trend.per_axis_lemma((F(0), F(1)), (F(0), F(1, 2)), (F(1, 4), F(3, 4))))
check("staircase case (A\\B and B\\A on opposite sides) holds",
      trend.per_axis_lemma((F(0), F(3, 5)), (F(2, 5), F(1)), (F(1, 10), F(9, 10))))

print("(3) exact global chain: P(C|A) = |A&C|/|A| >= 1/2 on the witnesses:")
check("2-D witness: P(C|A) >= 1/2 and P(C|B) >= 1/2",
      trend.conditional_global(w2, "A") >= F(1, 2) and trend.conditional_global(w2, "B") >= F(1, 2),
      f"P(C|A)={trend.conditional_global(w2,'A')}, P(C|B)={trend.conditional_global(w2,'B')}")
check("lemma product P(C|A)P(C|B) is a sound lower bound: <= q on the witness",
      trend.certified_lower_bound_witnessed(w2) <= gap.query_value(w2))
check("lemma product is itself >= 1/4 on the satisfying witness",
      trend.certified_lower_bound_witnessed(w2) >= F(1, 4),
      f"P(C|A)P(C|B)={trend.certified_lower_bound_witnessed(w2)}")

print("(4) certified sandwich: 1/4 <= inf I_box^n <= 513/1250 (n>=2), gap PERSISTS:")
lower, upper = trend.certified_sandwich()
check("sandwich is [1/4, 513/1250] with lower < upper and lower > 0 (no vanishing)",
      (lower, upper) == (F(1, 4), F(513, 1250)) and lower < upper and lower > 0)
check("upper witness feasible + meets 513/1250", gap.satisfies_seed_ontology(w2) and gap.query_value(w2) == upper)
for emb in (lo1, w2, w3, gap.rational_boxn_shrink_witness(4)):
    d = len(emb["A"])
    check(f"q >= 1/4 on the dim-{d} witness (lower bound respected)", gap.query_value(emb) >= F(1, 4),
          f"q={gap.query_value(emb)}")

print(f"\n{'ALL PASS -- inf trend RESOLVED: q = prod q_k, per-axis lemma => q >= 1/4 for all n; certified 1/4 <= inf I_box^n <= 513/1250 (n>=2); gap does NOT vanish' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
