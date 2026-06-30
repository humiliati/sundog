#!/usr/bin/env python3
"""FC-1 sanity check: an independent Python mirror of the Lean AbstractionCert verifier.
Confirms the three facts the Lean module proves (sundogcert/Sundogcert/AbstractionCert.lean):
  (1) the CHECK accepts the program that generated a task (verify_planted),
  (2) the CHECK rejects an off-by-one (inconsistent) program,
  (3) train_underdetermines: id and recolor(1,2) BOTH pass a task with no color-1, yet disagree
      on a held-out grid that has color 1 -> training evidence does not pin the program.
The Lean file is the proof; this is a 20-line illustration. Run: python scripts/findcheck_fc1_sanity.py
"""

def ev(p, g):
    """Eval the tiny DSL on a grid (list of rows of int colors). Mirrors AbstractionCert.eval."""
    k = p[0]
    if k == "id":      return [row[:] for row in g]
    if k == "recolor": return [[p[2] if c == p[1] else c for c in row] for row in g]
    if k == "flipH":   return [list(reversed(row)) for row in g]
    if k == "flipV":   return list(reversed(g))
    if k == "comp":    return ev(p[1], ev(p[2], g))
    raise ValueError(k)

def verify(p, task):  # the CHECK: program reproduces every (input, output) training pair
    return all(ev(p, i) == o for i, o in task)

ID, REC12 = ("id",), ("recolor", 1, 2)
task = [([[0]], [[0]])]            # one training pair; input has NO color 1
print("(1) verify_planted   :", verify(REC12, [(i, ev(REC12, i)) for i in [[[1, 0]], [[2]]]]))
print("(2) reject off-by-one:", not verify(("recolor", 0, 9), task))   # maps 0->9, breaks the pair
held = [[1]]                       # held-out grid that DOES have color 1
print("(3) underdetermines  :",
      ID != REC12 and verify(ID, task) and verify(REC12, task) and ev(ID, held) != ev(REC12, held),
      f"(id->{ev(ID, held)}  recolor(1,2)->{ev(REC12, held)})")
