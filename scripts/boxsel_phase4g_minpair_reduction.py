#!/usr/bin/env python
r"""BoxSEL Phase 4g - min-pair reduction for the n=2 endpoint-order atlas.

Phase 4f enumerated 123 n=2 endpoint-order orbits and exactly eliminated the 31 zero-help orbits.
This module closes the next exact chunk of the 92 live orbits.

Key interval fact:

    For three pairwise-overlapping intervals on one axis,
        |A & B & C| = min(|A & B|, |A & C|, |B & C|).

So each axis contributes one of:

    q_k = 1              if |A&B| is a minimum pairwise overlap;
    q_k = |A&C|/|A&B|   if |A&C| is the minimum;
    q_k = |B&C|/|A&B|   if |B&C| is the minimum;

with the obvious exact choices when minima are tied.

Exact closure rule:

    If every axis can be assigned to the SAME side S in {AC, BC} so that
        q_k = |S_k| / |A_k&B_k|
    then
        q = prod_k q_k = |S| / |A&B| >= (1/4) / (1/2) = 1/2 > q_KKT.

This closes 47 of the 92 live n=2 orbits exactly (21 one-help + 26 two-help). Together with the
31 zero-help orbits from Phase 4f, 78/123 n=2 orbits are now exact-closed. The remaining 45 are
precisely the mixed-side cases:

    35 one-help open orbits,
    10 two-help AC-vs-BC mixed orbits (the KKT candidate lives here).

The global sandwich is still unchanged:

    1/4 <= inf I_box^n <= (9 + sqrt(17)) / 32.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

import boxsel_kkt_exact as kkt
import boxsel_phase4f_cell_atlas as atlas

F = Fraction
A, B, C = atlas.A, atlas.B, atlas.C

GLOBAL_LOWER_BOUND_CLOSED = False
Q_KKT = kkt.Q_STAR

PAIR_LABEL = {
    frozenset((A, B)): "AB",
    frozenset((A, C)): "AC",
    frozenset((B, C)): "BC",
}

REMAINING_GLOBAL_OBLIGATIONS = (
    "exact lower certificate for the 35 one-help mixed-side n=2 orbits.",
    "exact lower certificate for the 10 two-help AC-vs-BC mixed n=2 orbits.",
    "dimension compression: prove n>=3 cannot beat the 2-D candidate, or find a counterexample.",
)


def min_pair_pattern(axis_cell) -> str:
    """Which pairwise overlap equals the triple on this endpoint-order cell.

    If the same interval has the largest left endpoint and smallest right endpoint, that interval is
    contained in the other two and two pairwise overlaps tie for the minimum.
    """
    sigma, tau = axis_cell
    largest_left = sigma[2]
    smallest_right = tau[0]
    if largest_left != smallest_right:
        return PAIR_LABEL[frozenset((largest_left, smallest_right))]
    return {
        A: "AB=AC",
        B: "AB=BC",
        C: "AC=BC",
    }[largest_left]


def exact_factor_sides(pattern: str) -> frozenset[str]:
    """Sides S for which q_k is exactly |S_k|/|AB_k| on this axis."""
    return frozenset({
        "AB": (),
        "AB=AC": ("AC",),
        "AB=BC": ("BC",),
        "AC": ("AC",),
        "BC": ("BC",),
        "AC=BC": ("AC", "BC"),
    }[pattern])


def pair_overlap(intervals, label: str) -> Fraction:
    pair = {
        "AB": (A, B),
        "AC": (A, C),
        "BC": (B, C),
    }[label]
    lo = max(intervals[pair[0]][0], intervals[pair[1]][0])
    hi = min(intervals[pair[0]][1], intervals[pair[1]][1])
    return hi - lo if hi > lo else F(0)


def triple_overlap(intervals) -> Fraction:
    lo = max(intervals[A][0], intervals[B][0], intervals[C][0])
    hi = min(intervals[A][1], intervals[B][1], intervals[C][1])
    return hi - lo if hi > lo else F(0)


def same_side_certificate(pair_rep) -> str | None:
    """Return AC or BC if this n=2 orbit is exact-closed by the same-side product rule."""
    possible = {"AC", "BC"}
    for axis_cell in pair_rep:
        possible &= set(exact_factor_sides(min_pair_pattern(axis_cell)))
    return sorted(possible)[0] if possible else None


def orbit_status(pair_rep) -> str:
    helping_axes = sum(atlas.helps(axis_cell) for axis_cell in pair_rep)
    if helping_axes == 0:
        return "zero_help_closed"
    if same_side_certificate(pair_rep) is not None:
        return "same_side_closed"
    if helping_axes == 1:
        return "one_help_mixed_open"
    return "two_help_mixed_open"


def minpair_classification() -> Counter:
    return Counter(orbit_status(rep) for rep in atlas.n2_orbits())


def live_same_side_split() -> Counter:
    counts = Counter()
    for rep in atlas.n2_orbits():
        if orbit_status(rep) == "same_side_closed":
            counts[sum(atlas.helps(axis_cell) for axis_cell in rep)] += 1
    return counts


def exact_closed_count() -> int:
    counts = minpair_classification()
    return counts["zero_help_closed"] + counts["same_side_closed"]


def remaining_open_count() -> int:
    counts = minpair_classification()
    return counts["one_help_mixed_open"] + counts["two_help_mixed_open"]


def same_side_lower_bound():
    """The exact lower bound for every same-side closed orbit: |S|/|AB| >= (1/4)/(1/2)."""
    return kkt.Surd(F(1, 2))


def certified_sandwich():
    """(lower, candidate_upper): unchanged by this phase."""
    return kkt.CERTIFIED_LOWER, Q_KKT


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("min-pair classification:", dict(minpair_classification()))
    print("same-side live split:", dict(live_same_side_split()))
    print("exact-closed n=2 orbits:", exact_closed_count(), "/ 123")
    print("remaining open n=2 orbits:", remaining_open_count())
    print("same-side lower bound:", same_side_lower_bound(), "> q_KKT =", Q_KKT)
    print("global lower bound closed:", GLOBAL_LOWER_BOUND_CLOSED, "sandwich:", certified_sandwich())
