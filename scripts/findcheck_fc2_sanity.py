#!/usr/bin/env python3
"""FC-2 sanity check: a Python mirror of the Lean AbstractionQueryGap needle family.
Illustrates the facts the Lean module proves (sundogcert/Sundogcert/AbstractionQueryGap.lean):
  - needle structure: rule j changes ONLY its own probe (eval_ruleOf_probe_self/_other),
  - cbit = Verify on one example (the CHECK is one probe; cbit_eq_verify),
  - the adversary's instances are real behaviors: identity -> all-false (cvec_id),
    rule m -> one-hot at m (cvec_rule),
  - the gap: CHECK a claimed candidate = 1 probe; a finder that probes < D positions on the
    all-false (identity) oracle can be fooled -> FIND needs >= D probes (search_needs_n_queries).
The Lean file is the proof; this is an illustration. Run: python scripts/findcheck_fc2_sanity.py
"""

def ev(p, g):  # the FC-1 DSL eval (recolor only, the rules used here)
    return [[p[2] if c == p[1] else c for c in row] for row in g] if p[0] == "recolor" else \
           [row[:] for row in g]

def ruleOf(j): return ("recolor", j, j + 1)        # candidate j: recolor color j -> j+1
def probe(j):  return [[j]]                         # candidate j's distinguishing 1x1 grid
def cbit(beh, j): return ev(ruleOf(j), probe(j)) == beh(probe(j))   # the CHECK of candidate j

D = 6
# needle structure: rule j changes its own probe; every other rule fixes it
self_changes = all(ev(ruleOf(j), probe(j)) != probe(j) for j in range(D))
other_fixes  = all(ev(ruleOf(k), probe(j)) == probe(j) for j in range(D) for k in range(D) if k != j)
# cvec_id: identity behavior -> all-false ; cvec_rule: rule m -> one-hot at m
cvec_id   = [cbit(lambda g: [r[:] for r in g], j) for j in range(D)]
cvec_rule = lambda m: [cbit(lambda g: ev(ruleOf(m), g), j) for j in range(D)]
# the adversary: a finder that probes only positions in Q, on the all-false (identity) oracle, learns
# nothing distinguishing it from "rule k fits" for any unprobed k -> must probe all D to be correct
def fooled_if_under_D(Q):           # Q = set of probed candidate indices on the identity oracle
    return any(k not in Q for k in range(D))   # an unprobed k could be the needle

print("(needle) self changes / others fix :", self_changes, "/", other_fixes)
print("(cvec_id)   identity -> all-false  :", cvec_id == [False] * D)
print("(cvec_rule) rule 3   -> one-hot@3  :", cvec_rule(3) == [i == 3 for i in range(D)])
print("(cbit=Verify) one-probe CHECK works:", cbit(lambda g: ev(ruleOf(2), g), 2) is True)
print(f"(gap) any finder probing < D={D} on the all-false oracle is foolable:",
      all(fooled_if_under_D(set(range(q))) for q in range(D)))
