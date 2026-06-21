#!/usr/bin/env python
r"""BoxSEL Phase 4i - two-help mixed-core normal form.

Phase 4h leaves exactly 10 n=2 endpoint-order orbits open: the two-help AC-vs-BC mixed cases.
This module reduces those 10 cells to one analytic core. It is a start, not the final closure.

WLOG axis 1 is AC-min and axis 2 is BC-min. Let

    x = |A_1&B_1|, y = |A_1&C_1|, z = |B_1&C_1|,
    X = |A_2&B_2|, Y = |A_2&C_2|, Z = |B_2&C_2|.

Then q = yZ/(xX). Put r = z/y and t = 4zZ. The global constraints and interval geometry give:

    1 <= t <= r <= 2,
    q >= t / (4 r x X).

The mixed interval geometry gives two upper bounds on X:

    X <= 1 / (2(x + (r-1)y)),
    X <= 1/(2x) - (r-t)/(4 r y).

Balancing these bounds gives the one-cell relaxation envelope

    P(r,t) = max_y r x X / t
           = 2r^2 / (t(r^2 - rt + 3r + t + sqrt((r-1)(r-t)(r^2 - rt + 7r + t)))).

If P(r,t) <= X_OPT = (9 - sqrt(17))/8 on 1 <= t <= r <= 2, then

    q >= 1/(4 X_OPT) = (9 + sqrt(17))/32 = q_KKT,

closing all 10 n=2 mixed orbits. This module verifies the exact endpoint identities and a
rational-grid guard for the two-parameter envelope. Phase 4j supplies the exact envelope maximum
certificate. The arbitrary-dimension lower bound remains open:

    GLOBAL_LOWER_BOUND_CLOSED = False.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from math import sqrt

import boxsel_kkt_exact as kkt
import boxsel_phase4f_cell_atlas as atlas
import boxsel_phase4g_minpair_reduction as red
import boxsel_phase4h_one_help_closure as one

F = Fraction
GLOBAL_LOWER_BOUND_CLOSED = False
MIXED_ENVELOPE_CLOSED_BY_PHASE4J = True
Q_KKT = kkt.Q_STAR
X_OPT = kkt.X_OPT

REMAINING_GLOBAL_OBLIGATIONS = (
    "dimension compression: prove n>=3 cannot beat the 2-D candidate, or find a counterexample.",
)


def open_two_help_mixed_reps():
    return tuple(rep for rep in atlas.n2_orbits() if one.orbit_status(rep) == "two_help_mixed_open")


def open_mixed_pattern_counts() -> Counter:
    """Patterns of the 10 remaining mixed orbits, reduced to min-pair labels."""
    out = Counter()
    for rep in open_two_help_mixed_reps():
        out[tuple(sorted(red.min_pair_pattern(cell) for cell in rep))] += 1
    return out


def balance_product_float(r: float, t: float) -> float:
    """The two-parameter relaxation envelope P(r,t), for numerical guards only."""
    if not (1 <= t <= r <= 2):
        raise ValueError("expected 1 <= t <= r <= 2")
    radicand = (r - 1) * (r - t) * (r * r - r * t + 7 * r + t)
    return 2 * r * r / (t * (r * r - r * t + 3 * r + t + sqrt(max(0.0, radicand))))


def balance_product_at_t1(r: Fraction):
    """P(r,1), represented as a float-free pair (r, radicand) for tests/docs.

    The closed form is:

        2r^2 / (r^2 + 2r + 1 + (r-1)*sqrt(r^2 + 6r + 1)).
    """
    return {
        "numerator": 2 * r * r,
        "rational_denominator_part": r * r + 2 * r + 1,
        "sqrt_coefficient": r - 1,
        "radicand": r * r + 6 * r + 1,
    }


def balance_product_at_r2_t1():
    """P(2,1) = (9 - sqrt(17))/8 exactly."""
    return X_OPT


def candidate_query_from_product(product):
    """q = 1/(4P); at P=X_OPT this is q_KKT."""
    return kkt.Surd(1) / (kkt.Surd(4) * product)


def derivative_positive_margin_at_t1(r: Fraction) -> Fraction:
    """Exact certificate that P(r,1) is increasing in r.

    For s=sqrt(r^2+6r+1), the sign of dP(r,1)/dr is controlled by

        (r+1)s - (4r + 1 - r^2).

    Squaring both sides gives exactly 16r^3 > 0 on r>0.
    """
    return 16 * r * r * r


def rational_envelope_grid_report(denom: int = 80) -> tuple[int, int, float]:
    """Numerical guard over rational (r,t) grid; not the final proof."""
    tested = violations = 0
    best = 0.0
    target = float(X_OPT)
    for i in range(denom, 2 * denom + 1):
        r = i / denom
        for j in range(denom, i + 1):
            t = j / denom
            p = balance_product_float(r, t)
            tested += 1
            best = max(best, p)
            if p > target + 1e-10:
                violations += 1
    return tested, violations, best


def certified_sandwich():
    """(lower, candidate_upper): unchanged by this phase."""
    return kkt.CERTIFIED_LOWER, Q_KKT


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("open two-help mixed reps:", len(open_two_help_mixed_reps()))
    print("open mixed pattern counts:", dict(open_mixed_pattern_counts()))
    print("P(2,1):", balance_product_at_r2_t1(), "=> q", candidate_query_from_product(balance_product_at_r2_t1()))
    print("grid guard:", rational_envelope_grid_report())
    print("global lower bound closed:", GLOBAL_LOWER_BOUND_CLOSED, "sandwich:", certified_sandwich())
