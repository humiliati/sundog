#!/usr/bin/env python
"""Frozen test for the BoxSEL Phase-2 single-box realizability probes (scripts/boxsel_single_box.py).

Locks: (1) exact type-volume computation on a hand example; (2) a single-box config realizes a
non-trivial Venn target (positive control); (3) axis-parallel Helly-2 holds on an explicit triple
and a random battery in 1-D and 2-D; (4) the forbidden 'pairwise cells positive / triple cell zero'
model is a valid SEL type-volume model the oracle admits, yet unrealizable by any single-box
embedding (the I_box ( I* representation-gap seed).
Run: python scripts/test_boxsel_single_box.py
"""
import sys
from fractions import Fraction

sys.path.insert(0, "scripts")
import boxsel_single_box as sb
import boxsel_exact_oracle as oracle

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


F = Fraction
AB, BC, AC, ABC = (frozenset({"A", "B"}), frozenset({"B", "C"}),
                   frozenset({"A", "C"}), frozenset({"A", "B", "C"}))

print("(1) exact type-volume computation (1-D chain A=[0,2], B=[1,3], C=[2,4] in [0,4]):")
chain = {"A": sb.make_box((0, 2)), "B": sb.make_box((1, 3)), "C": sb.make_box((2, 4))}
ambient1 = sb.make_box((0, 4))
vols = sb.type_volumes(chain, ("A", "B", "C"), ambient1)
expected = {frozenset({"A"}): F(1), AB: F(1), BC: F(1), frozenset({"C"}): F(1)}
ok = all(vols[k] == v for k, v in expected.items()) and all(vols[k] == 0 for k in vols if k not in expected)
check("Venn cells exact: {A}={A,B}={B,C}={C}=1, all others 0", ok,
      f"A&C={vols[AC]}, A&B&C={vols[ABC]}")
check("total volume == ambient length 4", sum(vols.values()) == F(4))

print("(2) positive control: a single-box config realizes a non-trivial Venn (chain, no A&C):")
check("chain realizes {A,B}>0 and {B,C}>0 but {A,C}=0 and triple=0 (a legitimate box model)",
      vols[AB] > 0 and vols[BC] > 0 and vols[AC] == 0 and vols[ABC] == 0)

print("(3) axis-parallel Helly-2: pairwise-positive => triple-positive (explicit + random battery):")
ax, bx, cx = sb.make_box((0, 2), (0, 2)), sb.make_box((1, 3), (0, 2)), sb.make_box((1, 3), (1, 3))
check("explicit 2-D triple: all three pairwise meets positive",
      sb.meet_volume([ax, bx]) > 0 and sb.meet_volume([bx, cx]) > 0 and sb.meet_volume([ax, cx]) > 0)
check("=> triple meet positive (Helly-2 holds)", sb.triple_overlap_forced([ax, bx, cx]) is True,
      f"triple meet vol = {sb.meet_volume([ax, bx, cx])}")
for d in (1, 2):
    tested, viol = sb.helly_battery(dim=d)
    check(f"dim={d} meet battery: 0 Helly-2 violations over sampled pairwise-positive triples",
          viol == 0 and tested > 0, f"tested={tested}, violations={viol}")

print("(4) forbidden Venn is oracle-valid but NOT single-box realizable (I_box ( I*):")
weights = sb.forbidden_pattern_weights()
atoms = ("A", "B", "C")
types = oracle.enumerate_types(atoms)
weight_vec = tuple(weights.get(t, F(0)) for t in types)
model = oracle.WeightedTypeModel(atoms, weight_vec)  # constructs iff nonneg + positive total
check("forbidden weights form a VALID oracle model (nonneg, positive total)",
      sum(weight_vec) > 0 and all(w >= 0 for w in weight_vec)
      and model.measure(oracle.atom("A") & oracle.atom("B")) == weights[AB])
check("the model has positive pairwise mass but ZERO triple mass (the forbidden Venn)",
      weights[AB] > 0 and weights[BC] > 0 and weights[AC] > 0 and weights[ABC] == 0)
t2, v2 = sb.forbidden_pattern_battery(dim=2)
check("dim=2: some embeddings stage all 3 pairwise cells positive, but NONE has a zero triple cell (Helly-2)",
      t2 > 0 and v2 == 0, f"tested={t2}, violations={v2}")
t1, v1 = sb.forbidden_pattern_battery(dim=1)
check("dim=1: STRONGER -- no interval embedding even achieves all 3 pairwise cells positive",
      t1 == 0 and v1 == 0, f"tested={t1}, violations={v1}")

print(f"\n{'ALL PASS -- single-box realizability: Helly-2 holds; the forbidden Venn is an oracle-valid model that no single-box embedding can realize (seeds I_box ( I*)' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
