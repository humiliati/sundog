#!/usr/bin/env python
r"""BoxSEL Phase 4c - optimizer evidence for inf I_box^n (Helly-seed ontology).

GOAL: probe the exact value of inf I_box^n inside the Phase-4b bracket [1/4, 513/1250] for the
seed ontology (atoms 1/2, pairwise >= 1/4) and query q = P(C | A and B).

WHAT WE FOUND:

* Numerically, the best candidate is attained at n = 2 (a finite-dimensional minimum candidate) --
  it does NOT keep decreasing in any optimizer run, and it does NOT vanish. n=1 gives exactly 1/2;
  n>=3 do not improve on n=2 in the recorded optimizers (consistent with the small-deficit
  heuristic: prod_k |A_k| = 1/2 forces near-full intervals which push each factor q_k -> 1).
* The n=2 candidate is ~= 0.41010 with BOTH |A&C| = |B&C| = 1/4 active (the Phase-4 witness had only
  |A&C| active, hence its slightly-higher 513/1250 = 0.41040). The KKT value is an algebraic
  number (two active overlap constraints; Phase 4d); 513/1250 is a near-optimal certified RATIONAL
  upper bound, ~0.0003 above the KKT candidate.

CERTIFIED: 1/4 <= inf I_box^n <= 513/1250 (Phase 4b). NUMERICAL: best candidate at n=2,
~= 0.41010. OPEN: the exact algebraic value and matching lower-bound proof are separate
Phase-4d/4e tasks.

THE SEARCH GAP IS THE HEADLINE. Every from-scratch optimizer FAILS to reach even the rational
witness -- only local search *seeded from the analytic witness* finds the candidate:

    random search (numpy, ~3.2M feasible samples)   n=2 -> 0.437          (misses)
    SLSQP (60 multistarts)                          n=2 -> 0.500, n>=3 -> 1.000 (fails)
    differential evolution (popsize 25, 400 iters)  n=2 -> 0.430, n=3 -> 0.432, n=4 -> 0.443
    exact rational grid (g <= 24, deterministic)    q  >= 4/9 ~= 0.444     (misses; non-monotone in g)
    Nelder-Mead SEEDED from the analytic witness    n=2 -> 0.41010 (reaches), n>=3 -> 0.41040

So on this fragment the SEARCH GAP (I_box \ I_sample) is severe: without the analytic handle, a
search returns a badly-wrong endpoint candidate. That is the lane's own thesis, biting the
meta-optimization.

This module is exact/pure (no numpy/scipy). The grid-search demonstration is deterministic exact
rationals; the four float-optimizer numbers above are recorded as provenance constants
(`OPTIMIZER_RESULTS`), reproduced by the session's exploration scripts.
"""
from fractions import Fraction
from typing import Mapping

import boxsel_inf_trend as trend  # certified bracket + factorization

F = Fraction

CERTIFIED_LOWER = trend.CERTIFIED_LOWER             # 1/4   (Phase 4b, proven)
CERTIFIED_UPPER = trend.CERTIFIED_UPPER_N_GE_2      # 513/1250 (certified witness)

INF_ATTAINED_DIM = 2                                # numerical/provenance: the best candidate is at n = 2
INF_NUMERICAL = 0.41010                             # numerical: the n=2 candidate (both |A&C|,|B&C| active)
WITNESS_IS_EXACT_OPTIMUM = False                    # 513/1250 is near-optimal, NOT the exact min

# Provenance: best feasible q each from-scratch optimizer reached (all MISS the analytic candidate).
OPTIMIZER_RESULTS = {
    "random_search_numpy": {2: 0.437},
    "slsqp_multistart": {2: 0.500, 3: 1.000, 4: 1.000},
    "differential_evolution": {2: 0.43025, 3: 0.43193, 4: 0.44306},
    "exact_grid_min_floor": F(4, 9),               # finest grids stall at 4/9 ~= 0.444
    "nelder_mead_seeded_from_witness": {2: 0.41010, 3: 0.41040, 4: 0.41040},
}


def certified_bracket() -> tuple:
    """(lower, upper) for inf I_box^n, n >= 2:  (1/4, 513/1250)."""
    return CERTIFIED_LOWER, CERTIFIED_UPPER


def _ov(i, j):
    lo = max(i[0], j[0]); hi = min(i[1], j[1])
    return hi - lo if hi > lo else F(0)


def _triple(i, j, k):
    lo = max(i[0], j[0], k[0]); hi = min(i[1], j[1], k[1])
    return hi - lo if hi > lo else F(0)


def exact_grid_min(g: int) -> Fraction:
    """Exhaustive EXACT rational grid search for the n=2 minimum at resolution 1/g.

    Marginals are kept exactly 1/2 (only axis-length pairs with product 1/2 are used). Returns the
    smallest feasible q on the grid -- a deterministic search-gap probe: coarse/fine grids alike
    stall well above the analytic candidate (>= 4/9 for g<=24), because the candidate needs endpoint
    denominators (41, 1600, ...) no practical grid contains.
    """
    pts = [F(i, g) for i in range(g + 1)]
    lens = [(F(a, g), F(b, g)) for a in range(1, g + 1) for b in range(1, g + 1) if F(a, g) * F(b, g) == F(1, 2)]
    cfgs = []
    for (l1, l2) in lens:
        for x1 in pts:
            if x1 + l1 > 1:
                break
            for x2 in pts:
                if x2 + l2 > 1:
                    break
                cfgs.append(((x1, x1 + l1), (x2, x2 + l2)))
    best = None
    for A in cfgs:
        for B in cfgs:
            ab = _ov(A[0], B[0]) * _ov(A[1], B[1])
            if ab < F(1, 4):
                continue
            for C in cfgs:
                if _ov(A[0], C[0]) * _ov(A[1], C[1]) < F(1, 4):
                    continue
                if _ov(B[0], C[0]) * _ov(B[1], C[1]) < F(1, 4):
                    continue
                tri = _triple(A[0], B[0], C[0]) * _triple(A[1], B[1], C[1])
                q = tri / ab
                if best is None or q < best:
                    best = q
    return best


def search_gap_min(max_g: int = 24) -> Fraction:
    """Best q any exact grid (g = 4..max_g, even) reaches -- the systematic-search floor."""
    best = None
    for g in range(4, max_g + 1, 2):
        m = exact_grid_min(g)
        if m is not None and (best is None or m < best):
            best = m
    return best


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("certified bracket (n>=2):", certified_bracket(), "= [0.25, 0.4104]")
    print("inf attained at dim:", INF_ATTAINED_DIM, "  numerical value ~=", INF_NUMERICAL)
    for g in (4, 8, 12, 16, 24):
        m = exact_grid_min(g)
        print(f"  exact grid g={g}: min q = {m} = {float(m):.5f}  (misses the ~0.41 candidate)")
    print("search-gap floor over grids:", search_gap_min(), "=", float(search_gap_min()))
    print("=> every from-scratch optimizer misses; only witness-seeded local search reaches ~0.41010")
