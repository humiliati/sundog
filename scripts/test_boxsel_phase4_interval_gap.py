#!/usr/bin/env python
"""Frozen test for the BoxSEL Phase-4 Helly interval-gap seed.

Run: python scripts/test_boxsel_phase4_interval_gap.py
"""
import sys
from fractions import Fraction as F

sys.path.insert(0, "scripts")

import boxsel_phase4_interval_gap as gap
import boxsel_single_box as sb

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


print("(1) exact oracle interval for the Helly-seeded ontology:")
exact = gap.exact_interval_gap()
check("I* query interval is exactly [0, 1]", exact == (F(0), F(1)), f"{exact}")

lo_model = gap.exact_lower_type_model()
hi_model = gap.exact_upper_type_model()
ontology, consequent, condition = gap.helly_interval_gap_ontology()
check("exact lower witness satisfies ontology", all(lo_model.satisfies(ax) for ax in ontology))
check("exact lower witness has q=0", lo_model.conditional(consequent, condition) == F(0))
check("exact upper witness satisfies ontology", all(hi_model.satisfies(ax) for ax in ontology))
check("exact upper witness has q=1", hi_model.conditional(consequent, condition) == F(1))

print("(2) one-dimensional single-box interval is [1/2, 1]:")
box_interval = gap.box1_interval_gap()
check("analytic I_box^1 interval is [1/2, 1]", box_interval == (F(1, 2), F(1)), f"{box_interval}")

lo_emb = gap.lower_box1_witness()
hi_emb = gap.upper_box1_witness()
check("lower box witness satisfies seed ontology", gap.satisfies_seed_ontology_1d(lo_emb))
check("lower box witness achieves q=1/2", gap.query_value_1d(lo_emb) == F(1, 2),
      f"q={gap.query_value_1d(lo_emb)}")
check("upper box witness satisfies seed ontology", gap.satisfies_seed_ontology_1d(hi_emb))
check("upper box witness achieves q=1", gap.query_value_1d(hi_emb) == F(1),
      f"q={gap.query_value_1d(hi_emb)}")

print("(3) measured query-interval gap:")
measured = gap.measured_gap()
check("box lower is strictly above exact lower", measured.box1_lower > measured.exact_lower,
      f"{measured.box1_lower} > {measured.exact_lower}")
check("gap size is exactly 1/2", measured.lower_gap == F(1, 2), f"gap={measured.lower_gap}")
check("I_box^1 is strictly inside I* on the lower endpoint and matches the upper endpoint",
      (measured.exact_lower, measured.exact_upper) == (F(0), F(1))
      and (measured.box1_lower, measured.box1_upper) == (F(1, 2), F(1)))

print("(4) analytic lower-bound proof fires on the lower witness:")
check("center-diameter proof certifies q >= 1/2 for satisfying 1-D boxes",
      gap.analytic_box1_lower_bound(lo_emb) == F(1, 2))

print("(5) battery: across ALL satisfying 1-D embeddings, q >= 1/2 and the analytic bound is sound:")
grid = [F(k, 8) for k in range(0, 5)]  # interval left endpoints 0, 1/8, 1/4, 3/8, 1/2 (boxes within [0,1])
tested = 0
min_q = None
bound_sound = True
all_ge_half = True
for la in grid:
    for lb in grid:
        for lc in grid:
            emb = {"A": sb.make_box((la, la + F(1, 2))),
                   "B": sb.make_box((lb, lb + F(1, 2))),
                   "C": sb.make_box((lc, lc + F(1, 2)))}
            if not gap.satisfies_seed_ontology_1d(emb):
                continue
            tested += 1
            q = gap.query_value_1d(emb)
            if gap.analytic_box1_lower_bound(emb) > q:  # the bound must never exceed the true query
                bound_sound = False
            if q < F(1, 2):
                all_ge_half = False
            min_q = q if min_q is None else min(min_q, q)
check("battery is non-vacuous (many satisfying 1-D configs)", tested >= 10, f"tested={tested}")
check("analytic bound soundly lower-bounds the true query on EVERY satisfying config", bound_sound)
check("every satisfying config has q >= 1/2 and the battery min is exactly 1/2 (infimum corroborated + tight)",
      all_ge_half and min_q == F(1, 2), f"min q over battery = {min_q}")

print("(6) higher-dimensional certified witnesses shrink the 1-D lower endpoint:")
box2 = gap.rational_box2_shrink_witness()
check("2-D rational witness satisfies the seed ontology", gap.satisfies_seed_ontology(box2))
check("2-D rational witness has q = 513/1250 < 1/2",
      gap.query_value(box2) == F(513, 1250) and gap.query_value(box2) < F(1, 2),
      f"q={gap.query_value(box2)}")
check("2-D witness pairwise overlaps meet the required thresholds",
      sb.meet_volume([box2["A"], box2["B"]]) == F(25, 82)
      and sb.meet_volume([box2["A"], box2["C"]]) == F(1, 4)
      and sb.meet_volume([box2["B"], box2["C"]]) == F(513, 2050)
      and sb.meet_volume([box2["A"], box2["B"], box2["C"]]) == F(513, 4100))

rows = gap.dimension_sweep_candidates(4)
check("dimension sweep returns dims 1..4", [row.dim for row in rows] == [1, 2, 3, 4])
check("dim=1 row is the tight analytic lower q=1/2",
      rows[0].certified_box_q == F(1, 2) and rows[0].status == "tight_analytic")
check("dims 2..4 reuse the certified q=513/1250 upper-bound witness",
      all(row.certified_box_q == F(513, 1250) and row.status == "certified_upper_bound" for row in rows[1:]),
      f"{[(row.dim, row.certified_box_q) for row in rows]}")
for dim in (3, 4):
    emb = gap.rational_boxn_shrink_witness(dim)
    check(f"dim={dim}: full-axis extension satisfies ontology and preserves q",
          gap.satisfies_seed_ontology(emb) and gap.query_value(emb) == F(513, 1250))

print(f"\n{'ALL PASS -- Phase-4 seed: exact I*=[0,1], I_box^1=[1/2,1], dims>=2 certified q<=513/1250' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
