#!/usr/bin/env python
r"""BoxSEL Phase 4j - exact closure of the two-help mixed n=2 envelope.

Phase 4i reduced the final 10 n=2 endpoint-order orbits to the two-parameter envelope

    P(r,t) = 2r^2 / (t(B + sqrt(D))),

on the triangle 1 <= t <= r <= 2, where

    B = r^2 - rt + 3r + t,
    D = (r-1)(r-t)(r^2 - rt + 7r + t).

This module proves the missing envelope maximum:

    P(r,t) <= P(2,1) = X_OPT = (9 - sqrt(17))/8.

The proof uses shifted variables a=r-1, b=t-1 with 0 <= b <= a <= 1. Let alpha=X_OPT and

    L = 2r^2/(alpha*t) - B.

If L <= 0 the comparison is immediate. Since 2/alpha = (9+sqrt(17))/4, the sign of L is the
sign of the Q(sqrt17) polynomial M(a,b) below. If M > 0, squaring is legal and

    D - L^2 = (a+1)^2 F(a,b) / (8(b+1)^2).

So the only nontrivial obligation is M>0 => F>=0. For fixed a, M is strictly decreasing in b
on [0,a], while F is concave in b. The M>0 region is therefore an interval [0,beta). F is
nonnegative at b=0 whenever M>0, and is positive on the M=0 endpoint because Res_b(F,M) has a
fixed nonzero sign on a>0. Concavity then gives F>=0 on the whole M>0 interval.

Combined with Phases 4f-4h, this closes all 123 n=2 endpoint-order orbits:

    inf I_box^2 = (9 + sqrt(17))/32.

The global lower bound over arbitrary n is still open until the n>=3 compression proof is banked.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

import boxsel_kkt_exact as kkt
import boxsel_phase4i_two_help_mixed_core as core

F = Fraction
S = kkt.Surd
SQRT17 = S(0, 1)

N2_LOWER_BOUND_CLOSED = True
GLOBAL_LOWER_BOUND_CLOSED = False
Q_KKT = kkt.Q_STAR
X_OPT = kkt.X_OPT

REMAINING_GLOBAL_OBLIGATIONS = (
    "dimension compression: prove n>=3 cannot beat the exact n=2 candidate, or find a counterexample.",
)


def as_surd(x) -> S:
    return S.coerce(x)


def shifted_domain(a, b) -> bool:
    """The Phase-4i envelope triangle in shifted variables: 0 <= b <= a <= 1."""
    return F(0) <= a <= F(1) and F(0) <= b <= a


def m_polynomial(a, b) -> S:
    """M(a,b) = 4tL; M<=0 makes P<=X_OPT immediate."""
    a, b = as_surd(a), as_surd(b)
    return (
        S(4) * a * b * b
        - S(4) * a * a * b
        - S(12) * a * b
        - S(16) * b
        + S(5) * a * a
        + S(2) * a
        - S(7)
        + SQRT17 * (a + S(1)) * (a + S(1))
    )


def f_polynomial(a, b) -> S:
    """F(a,b), the exact squared residual factor when M>0."""
    a, b = as_surd(a), as_surd(b)
    rational_part = (
        S(36) * a * a * b
        - S(13) * a * a
        - S(36) * a * b * b
        + S(108) * a * b
        + S(46) * a
        - S(128) * b * b
        - S(112) * b
        - S(33)
    )
    sqrt_part = (
        S(4) * a * a * b
        - S(5) * a * a
        - S(4) * a * b * b
        + S(12) * a * b
        - S(2) * a
        + S(16) * b
        + S(7)
    )
    return rational_part + SQRT17 * sqrt_part


def m_derivative_upper_bound(a) -> S:
    """Upper bound for dM/db on b in [0,a]: 8ab-4a^2-12a-16 <= 4a^2-12a-16 < 0."""
    a = as_surd(a)
    return S(4) * a * a - S(12) * a - S(16)


def m_at_b_equals_a(a) -> S:
    """M(a,a) = (a+1)^2(sqrt17-7) < 0."""
    a = as_surd(a)
    return (a + S(1)) * (a + S(1)) * (SQRT17 - S(7))


def m_at_b0(a) -> S:
    """M(a,0)."""
    return m_polynomial(a, F(0))


def m_at_b0_derivative(a) -> S:
    """d M(a,0)/da = (10a+2) + (2a+2)sqrt17 > 0 for a>=0."""
    a = as_surd(a)
    return S(10) * a + S(2) + SQRT17 * (S(2) * a + S(2))


def f_b2_coefficient(a) -> S:
    """The b^2 coefficient of F: -128 - 36a - 4a sqrt17 < 0, so F is concave in b."""
    a = as_surd(a)
    return -S(128) - S(36) * a - SQRT17 * S(4) * a


def f_at_b0(a) -> S:
    """F(a,0) = (1-a)((13+5sqrt17)a + 7sqrt17 - 33)."""
    a = as_surd(a)
    return (S(1) - a) * ((S(13) + S(5) * SQRT17) * a + S(7) * SQRT17 - S(33))


def f0_positive_root() -> S:
    """Positive root of the second factor in F(a,0): sqrt17 - 4."""
    return SQRT17 - S(4)


def m0_at_f0_positive_root() -> S:
    """M(sqrt17-4,0) = -12(sqrt17-4) < 0; hence M(a,0)>0 implies F(a,0)>=0."""
    return m_at_b0(f0_positive_root())


def common_zero_resultant(a) -> S:
    """Resultant Res_b(F,M) after reducing sqrt17^2=17.

    Res_b(F,M) = -32 a (a+1)^4 (135a sqrt17 + 1247a + 448sqrt17 + 6080).
    It is strictly negative for 0<a<=1, so F and M have no common b-root on the live branch.
    """
    a = as_surd(a)
    positive_factor = S(1247) * a + S(6080) + SQRT17 * (S(135) * a + S(448))
    return -S(32) * a * (a + S(1)) * (a + S(1)) * (a + S(1)) * (a + S(1)) * positive_factor


def squared_residual_direct(a: Fraction, b: Fraction) -> S:
    """D-L^2 in shifted variables, computed directly in Q(sqrt17)."""
    if not shifted_domain(a, b):
        raise ValueError("expected 0 <= b <= a <= 1")
    r, t = F(1) + a, F(1) + b
    d = S(a * (a - b) * (a * a - a * b + 8 * a + 8))
    l = m_polynomial(a, b) / S(4 * t)
    return d - l * l


def squared_residual_factored(a: Fraction, b: Fraction) -> S:
    """The factorization D-L^2 = (a+1)^2 F(a,b)/(8(b+1)^2)."""
    if not shifted_domain(a, b):
        raise ValueError("expected 0 <= b <= a <= 1")
    return S((a + 1) * (a + 1)) * f_polynomial(a, b) / S(8 * (b + 1) * (b + 1))


def envelope_comparison_status(a: Fraction, b: Fraction) -> str:
    """Classify the exact proof branch for a rational shifted point."""
    if not shifted_domain(a, b):
        raise ValueError("expected 0 <= b <= a <= 1")
    m = m_polynomial(a, b)
    if m <= S(0):
        return "automatic"
    return "squared" if f_polynomial(a, b) >= S(0) else "violation"


def exact_envelope_grid_report(denom: int = 80) -> tuple[int, Counter]:
    """Exact rational smoke over the envelope triangle using the proof branches."""
    counts: Counter = Counter()
    tested = 0
    for i in range(0, denom + 1):
        a = F(i, denom)
        for j in range(0, i + 1):
            b = F(j, denom)
            counts[envelope_comparison_status(a, b)] += 1
            tested += 1
    return tested, counts


def n2_closure_counts() -> Counter:
    """All 123 n=2 endpoint-order orbits are now exact-closed."""
    return Counter(
        {
            "zero_help_closed": 31,
            "same_side_closed": 47,
            "one_help_closed": 35,
            "two_help_mixed_closed": len(core.open_two_help_mixed_reps()),
            "open_n2": 0,
        }
    )


def exact_n2_infimum() -> S:
    """The exact n=2 box infimum for the Helly-seed query."""
    return Q_KKT


def certified_global_sandwich() -> tuple[S, S]:
    """The global arbitrary-dimension sandwich is unchanged until n>=3 compression closes."""
    return kkt.CERTIFIED_LOWER, Q_KKT


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("n=2 lower bound closed:", N2_LOWER_BOUND_CLOSED)
    print("n=2 closure counts:", dict(n2_closure_counts()))
    print("exact n=2 infimum:", exact_n2_infimum(), "~=", float(exact_n2_infimum()))
    print("global lower bound closed:", GLOBAL_LOWER_BOUND_CLOSED)
    print("global sandwich:", certified_global_sandwich())
    print("grid certificate:", exact_envelope_grid_report())
