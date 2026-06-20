#!/usr/bin/env python
r"""BoxSEL Phase 4b - resolve the inf I_box^n trend (does the box lower endpoint -> 0?).

ANSWER: NO. For the Helly-seed ontology (atoms sized 1/2, pairwise co-occurrence >= 1/4) and the
query q = P(C | A and B), the box-attainable lower endpoint is sandwiched:

    1/4  <=  inf I_box^n  <=  513/1250          for every n >= 2

so the representation gap (I* lower = 0  vs  I_box^n lower) PERSISTS at >= 1/4; it does NOT vanish
as embedding dimension grows.

Three certified ingredients:

1. FACTORIZATION (exact). Axis-parallel box volumes and overlaps factor over axes, so
       q = |A&B&C| / |A&B| = prod_k |A_k&B_k&C_k| / prod_k |A_k&B_k| = prod_k q_k,
   with each q_k = P(C_k | A_k & B_k) in (0, 1]. (This is why 1-D is stuck at one factor and
   higher dimensions can be smaller -- a product of sub-unit factors.)

2. PER-AXIS LEMMA (proven; exhaustively grid-verified). For three intervals with |A&B| > 0,
       |A&B&C| * |A| * |B|  >=  |A&B| * |A&C| * |B&C|      i.e.   q_k >= P(C|A_k) * P(C|B_k).
   Proof: write A&B = J, with the tails A\B and B\A on OPPOSITE sides of J (one of A,B starts
   first, the other ends last; the degenerate "B subset A" case reduces to |A| >= |A&C|). C is an
   interval, so if it reaches both opposite tails it must cover all of J, giving |C&J| = |J|; then
   (|A\B|+|J|)(|J|+|B\A|) >= (|C&(A\B)|+|J|)(|J|+|C&(B\A)|) since each tail term only shrinks.
   If C misses a tail, that tail's contribution is 0 and the bound is immediate.

3. GLOBAL CHAIN (exact). prod_k P(C|A_k) = (prod_k |A_k&C_k|)/(prod_k |A_k|) = |A&C|/|A| >=
   (1/4)/(1/2) = 1/2, and likewise |B&C|/|B| >= 1/2. Multiplying the per-axis lemma over all axes,
       q = prod_k q_k >= prod_k P(C|A_k) P(C|B_k) = (|A&C|/|A|)(|B&C|/|B|) >= 1/4.

A naive random search OVER-estimates the infimum in higher dimensions (it could not even reach the
certified 513/1250 witness at n=2) -- a live instance of the search gap. The analytic bound, not
the search, resolves the trend. Exact rational throughout.
"""
from fractions import Fraction
from itertools import combinations
from typing import Mapping

import boxsel_single_box as sb

F = Fraction
CERTIFIED_LOWER = F(1, 4)            # q >= 1/4 for every n (proven below)
CERTIFIED_UPPER_N_GE_2 = F(513, 1250)  # certified witness (Phase 4)


def _iv_box(interval) -> tuple:
    """Wrap a 1-D interval (lo, hi) as a 1-D box."""
    return (interval,)


def q_factors(embedding: Mapping[str, tuple]) -> list:
    """Per-axis factors q_k = |A_k&B_k&C_k| / |A_k&B_k| (exact); their product is query_value."""
    A, B, C = embedding["A"], embedding["B"], embedding["C"]
    factors = []
    for k in range(len(A)):
        ab = sb.meet_volume([_iv_box(A[k]), _iv_box(B[k])])
        if ab == 0:
            raise ValueError(f"axis {k}: |A&B| = 0 (query undefined)")
        factors.append(sb.meet_volume([_iv_box(A[k]), _iv_box(B[k]), _iv_box(C[k])]) / ab)
    return factors


def query_from_factors(embedding: Mapping[str, tuple]) -> Fraction:
    prod = F(1)
    for qk in q_factors(embedding):
        prod *= qk
    return prod


def per_axis_lemma(A: tuple, B: tuple, C: tuple) -> bool:
    """|A&B&C|*|A|*|B| >= |A&B|*|A&C|*|B&C| for three 1-D intervals (exact)."""
    Ai, Bi, Ci = _iv_box(A), _iv_box(B), _iv_box(C)
    lhs = sb.meet_volume([Ai, Bi, Ci]) * sb.box_volume(Ai) * sb.box_volume(Bi)
    rhs = sb.meet_volume([Ai, Bi]) * sb.meet_volume([Ai, Ci]) * sb.meet_volume([Bi, Ci])
    return lhs >= rhs


def grid_intervals(n: int) -> list:
    pts = [F(i, n) for i in range(n + 1)]
    return [(p, q) for p, q in combinations(pts, 2)]


def lemma_grid_report(n: int) -> tuple:
    """Exhaustive EXACT check of the per-axis lemma over all interval triples on an n-grid."""
    ivs = grid_intervals(n)
    tested = violations = 0
    for a in ivs:
        for b in ivs:
            if sb.meet_volume([_iv_box(a), _iv_box(b)]) == 0:
                continue
            for c in ivs:
                tested += 1
                if not per_axis_lemma(a, b, c):
                    violations += 1
    return tested, violations


def conditional_global(embedding: Mapping[str, tuple], given: str, target: str = "C") -> Fraction:
    """P(target | given) = |given & target| / |given| (exact); factors over axes."""
    g, t = embedding[given], embedding[target]
    return sb.meet_volume([g, t]) / sb.box_volume(g)


def certified_lower_bound_witnessed(embedding: Mapping[str, tuple]) -> Fraction:
    """The proven per-axis-lemma product bound q >= P(C|A)*P(C|B) (>= 1/4 on satisfying configs)."""
    return conditional_global(embedding, "A") * conditional_global(embedding, "B")


def certified_sandwich() -> tuple:
    """(lower, upper) for inf I_box^n, n >= 2: ([1/4, 513/1250])."""
    return CERTIFIED_LOWER, CERTIFIED_UPPER_N_GE_2


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import boxsel_phase4_interval_gap as gap
    for dim in (1, 2, 3):
        emb = gap.rational_boxn_shrink_witness(dim)
        print(f"dim={dim}: factors={[str(x) for x in q_factors(emb)]} product={query_from_factors(emb)} "
              f"lower-bound P(C|A)P(C|B)={certified_lower_bound_witnessed(emb)}")
    for n in (6, 8):
        print(f"per-axis lemma grid N={n}:", lemma_grid_report(n))
    print("certified sandwich for n>=2:", certified_sandwich(), "=> inf does NOT tend to 0")
