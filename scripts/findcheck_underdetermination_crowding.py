#!/usr/bin/env python3
"""FC-3 addendum: a toy model of the Branch E v2 "top-2 crowding" mechanism.

Branch E v2 (docs/prereg/arc/PHASE3_BRANCH_E_V2_PROGRAM_SEARCH_SPEC.md, Amendment B) GREW the
deterministic primitive library (intricate masks + morphology + depth-3) and got ZERO new held-out
solves -- and slightly HURT validation, because a larger train-consistent candidate set crowds the
correct program out of the deterministic top-2. So the ARC deterministic-search ceiling is NOT a
library-COVERAGE limit; it is the UNDER-DETERMINATION / SELECTION wall (FC-1 `train_underdetermines`):
the cheap CHECK (train-pair consistency) is satisfiable by many programs that disagree off-train.

This instrument reproduces that mechanism in a controlled toy: recolor programs over C colors. A
program is train-consistent iff it agrees with the hidden truth on the colors that APPEAR in the
train inputs; colors NOT seen in train are FREE -> many train-consistent completions that differ on a
held-out query. As coverage drops (free colors grow):
  - the train-consistent candidate count EXPLODES,
  - a deterministic simplicity top-2 selector's held-out hit-rate FALLS (crowding),
  - but the ORACLE ceiling stays 1.0 (the truth is always in the pool) -> it is SELECTION, not
    coverage. Growing the library can only make it worse.
Mirrors E3's `oracle_candidate_ceiling` vs `v2_deterministic_selector` controls. Toy, illustrative;
the Lean proof of the under-determination is FC-1 `AbstractionCert.train_underdetermines`.
Run: python scripts/findcheck_underdetermination_crowding.py
"""
import itertools, random

C = 6          # number of colors (the "library" = all color maps on C colors)
GRID = 9       # cells per grid
TRIALS = 4000
random.seed(20260630)

def apply(sigma, g):            # a recolor program: relabel each cell
    return tuple(sigma[c] for c in g)

def deterministic_top2(free, truth):
    """The v2-style deterministic selector: on FREE colors prefer identity, then the +1 shift.
    Returns up to 2 completions of the truth on covered colors. Crowding = neither matches truth."""
    base = dict(truth)
    a = dict(base); b = dict(base)
    for c in free:
        a[c] = c                       # attempt 1: identity on free colors
        b[c] = (c + 1) % C             # attempt 2: +1 shift on free colors
    return [a, b]

def trial(cov_frac):
    truth = {c: (c * 2 + 1) % C for c in range(C)}        # the hidden recolor program
    ncov = max(1, round(cov_frac * C))
    covered = set(random.sample(range(C), ncov))
    free = [c for c in range(C) if c not in covered]
    # train inputs only use covered colors -> a program is train-consistent iff it = truth on `covered`
    n_consistent = C ** len(free)                          # free colors are unconstrained
    # held-out query uses ALL colors (so free colors matter)
    query = tuple(random.randrange(C) for _ in range(GRID))
    target = apply(truth, query)
    top2 = deterministic_top2(free, truth)
    hit = any(apply(s, query) == target for s in top2)
    oracle = True                                          # truth is always an admitted candidate
    return n_consistent, hit, oracle

print(f"toy: C={C} colors, library = all {C**C} color maps; hidden truth fixed")
print(f"{'coverage':>9} {'avg #train-consistent':>22} {'det top-2 hit-rate':>20} {'oracle ceiling':>15}")
for cov in (1.0, 0.83, 0.66, 0.5, 0.33, 0.17):
    rows = [trial(cov) for _ in range(TRIALS)]
    nc = sum(r[0] for r in rows) / len(rows)
    hr = sum(r[1] for r in rows) / len(rows)
    orc = sum(r[2] for r in rows) / len(rows)
    print(f"{cov:>9.2f} {nc:>22.1f} {hr:>20.3f} {orc:>15.3f}")
print("\nReading: as coverage falls, train-consistent count explodes and the deterministic top-2")
print("hit-rate collapses, but the oracle ceiling stays 1.0 -> the wall is SELECTION (under-")
print("determination), NOT coverage. Growing the library only enlarges the train-consistent set.")
