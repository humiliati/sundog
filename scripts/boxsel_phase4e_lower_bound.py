#!/usr/bin/env python
r"""BoxSEL Phase 4e - lower-bound closure start.

This is a proof-workbench, not a global closure claim.

Phase 4d produced an exact n=2 KKT witness with

    q_KKT = (9 + sqrt(17)) / 32 ~= 0.4100970.

What is globally certified before 4e:

    1/4 <= inf I_box^n <= q_KKT       for n >= 2.

The matching lower bound, inf I_box^n >= q_KKT, is still open. This module starts the closure by
certifying the KKT value inside the structured 2-D normal form used to generate the witness. That is
useful but deliberately narrower than the full theorem: it does not rule out all 2-D endpoint-order
cells, and it does not compress arbitrary n >= 3 boxes down to the 2-D form.

Structured family:

    A = [1-x, 1] x [1 - 1/(2x), 1]
    B = [0, 1]   x [1/2, 1]
    C = [0, z]   x [1 - 1/(2x), 1 - 1/(2x) + 1/(2z)]

with |A|=|B|=|C|=1/2. The seed constraints reduce to

    z >= 2(1-x)          (AC >= 1/4)
    z <= x/(2(1-x))     (BC >= 1/4)

plus endpoint-validity bounds. Thus the feasible z interval is nonempty only once

    2(1-x) <= x/(2(1-x))
    4x^2 - 9x + 4 <= 0,

whose lower root is x* = (9 - sqrt(17))/8. The query in this family is

    q(x,z) = 2(x+z-1)(1/2 - 1/(2x) + 1/(2z)) / x.

For fixed x, q(x,z) is concave in z, so its minimum on the feasible z interval is attained at an
endpoint. Both active-constraint endpoints give the same one-variable boundary value

    q0(x) = (-2x^2 + 5x - 2) / (2x^2).

On the feasible x interval, q0 has no interior minimum below its endpoints: q0'(x) changes sign
from positive to negative at x=4/5. Since q0(x*) = q_KKT and q0(1) = 1/2, this proves the
restricted-family lower bound q >= q_KKT, with equality at the KKT witness.
"""

from __future__ import annotations

from fractions import Fraction

import boxsel_kkt_exact as kkt

F = Fraction
S = kkt.Surd

PROVEN_GLOBAL_LOWER = kkt.CERTIFIED_LOWER
CANDIDATE_UPPER = kkt.Q_STAR
LOWER_BOUND_CLOSED = False

REMAINING_GLOBAL_OBLIGATIONS = (
    "2D endpoint-order atlas: rule out every non-structured interval cell order.",
    "dimension compression: prove n>=3 cannot beat the 2D candidate, or find a counterexample.",
)


def as_surd(x) -> S:
    return S.coerce(x)


def candidate_sandwich() -> tuple[S, S]:
    """Current certified global sandwich: 1/4 <= inf I_box^n <= q_KKT."""
    return PROVEN_GLOBAL_LOWER, CANDIDATE_UPPER


def root_polynomial(x) -> S:
    """Return 4x^2 - 9x + 4 in Q(sqrt 17)."""
    x = as_surd(x)
    return S(4) * x * x - S(9) * x + S(4)


def structured_boxes(x, z) -> dict[str, tuple]:
    """Build the structured-family 2-D boxes for exact arithmetic."""
    x, z = as_surd(x), as_surd(z)
    y0 = S(1) - S(1) / (S(2) * x)
    return {
        "A": [(S(1) - x, S(1)), (y0, S(1))],
        "B": [(S(0), S(1)), (S(F(1, 2)), S(1))],
        "C": [(S(0), z), (y0, y0 + S(1) / (S(2) * z))],
    }


def structured_query_formula(x, z):
    """Closed-form q(x,z) for the structured family.

    For rational inputs this returns a Fraction; for Surd inputs it returns a Surd.
    """
    half = F(1, 2)
    return F(2) * (x + z - 1) * (half - half / x + half / z) / x


def structured_query_geometry(x, z) -> S:
    """Geometric q computed from exact box intersections, used as a formula cross-check."""
    e = structured_boxes(x, z)
    return kkt.meet_volume([e["A"], e["B"], e["C"]]) / kkt.meet_volume([e["A"], e["B"]])


def structured_family_feasible(x, z) -> bool:
    """Exact feasibility check for the seed ontology inside the structured family."""
    e = structured_boxes(x, z)
    A, B, C = e["A"], e["B"], e["C"]
    zero, one, half, quarter = S(0), S(1), S(F(1, 2)), S(F(1, 4))
    boxes = (A, B, C)
    endpoints_ok = all(lo >= zero and hi <= one and hi > lo for box in boxes for lo, hi in box)
    if not endpoints_ok:
        return False
    marginals_ok = all(kkt.box_volume(box) == half for box in boxes)
    pairwise_ok = (
        kkt.meet_volume([A, B]) >= quarter
        and kkt.meet_volume([A, C]) >= quarter
        and kkt.meet_volume([B, C]) >= quarter
    )
    return marginals_ok and pairwise_ok


def structured_z_interval(x: Fraction) -> tuple[Fraction, Fraction] | None:
    """The exact rational z interval induced by the structured-family constraints.

    The interval encodes endpoint validity plus AB/AC/BC constraints. A None result means this
    rational x has no feasible z in the structured family.
    """
    if x < F(1, 2) or x > F(1):
        return None
    lower = max(x, F(2) * (F(1) - x))
    upper = min(F(1), x / (F(2) * (F(1) - x))) if x < 1 else F(1)
    if lower > upper:
        return None
    return lower, upper


def structured_boundary_query(x):
    """q0(x), the query on either active-constraint boundary z=2(1-x) or z=x/(2(1-x))."""
    return (-F(2) * x * x + F(5) * x - F(2)) / (F(2) * x * x)


def structured_q_second_derivative_z(x, z):
    """d^2 q(x,z) / dz^2 = -2(1-x)/(x z^3), negative for 0<x<1."""
    return -F(2) * (F(1) - x) / (x * z * z * z)


def structured_boundary_derivative(x):
    """q0'(x) = (4 - 5x)/(2x^3)."""
    return (F(4) - F(5) * x) / (F(2) * x * x * x)


def restricted_family_certificate() -> dict[str, S | bool]:
    """Exact checkpoint facts for the structured-family lower-bound certificate."""
    x = kkt.X_OPT
    z = S(2) * (S(1) - x)
    return {
        "root_equation_zero": root_polynomial(x) == S(0),
        "candidate_feasible": structured_family_feasible(x, z),
        "formula_matches_geometry": as_surd(structured_query_formula(x, z)) == structured_query_geometry(x, z),
        "query_equals_candidate": structured_query_geometry(x, z) == CANDIDATE_UPPER,
        "right_endpoint_above_candidate": as_surd(structured_boundary_query(F(1))) > CANDIDATE_UPPER,
    }


def rational_structured_grid_report(denom: int = 120) -> tuple[int, int, S | None]:
    """Exact rational smoke over the structured family.

    For each rational x on the grid with a feasible z interval, test both z endpoints and the
    midpoint. The continuum proof is the concavity + boundary derivative argument above; this
    grid smoke protects the implementation from algebra slips.
    """
    tested = violations = 0
    best = None
    for i in range(1, denom + 1):
        x = F(i, denom)
        interval = structured_z_interval(x)
        if interval is None:
            continue
        lo, hi = interval
        for z in (lo, (lo + hi) / 2, hi):
            if not structured_family_feasible(x, z):
                continue
            tested += 1
            q = structured_query_geometry(x, z)
            best = q if best is None or q < best else best
            if q < CANDIDATE_UPPER:
                violations += 1
    return tested, violations, best


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("global status: lower-bound closure open =", not LOWER_BOUND_CLOSED)
    print("certified sandwich:", candidate_sandwich())
    print("structured-family certificate:", restricted_family_certificate())
    print("structured rational grid:", rational_structured_grid_report())
    print("remaining obligations:")
    for item in REMAINING_GLOBAL_OBLIGATIONS:
        print(" -", item)
