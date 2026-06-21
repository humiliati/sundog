#!/usr/bin/env python
r"""BoxSEL Phase 4h - exact closure of all one-help n=2 endpoint-order orbits.

Phase 4g exact-closed the same-side live orbits and left 45 mixed-side n=2 orbits:

    35 one-help mixed,
    10 two-help AC-vs-BC mixed.

This module closes the 35 one-help mixed orbits exactly. In fact, it proves every one-help orbit
has q >= 1/2.

Proof (AC case; BC is symmetric). Suppose axis 1 is the only helping axis and

    q = |A_1&C_1| / |A_1&B_1| < 1/2.

Then |A_1&C_1| < |A_1&B_1|/2. Since the ontology requires |A&C| >= 1/4,

    |A_2&C_2| > 1 / (2 |A_1&B_1|).

But |A_2&C_2| <= |A_2|, while |A_1&B_1| <= |A_1| and |A_1||A_2| = 1/2, so

    |A_2| = 1/(2|A_1|) <= 1/(2|A_1&B_1|),

contradiction. Therefore q >= 1/2 > q_KKT.

After Phase 4h:

    113 / 123 n=2 orbits exact-closed,
    10 / 123 n=2 orbits remain open.

The remaining n=2 cases are exactly the two-help AC-vs-BC mixed orbits, where the KKT candidate
lives. The global sandwich is still unchanged until those 10 and the n>=3 compression are closed.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

import boxsel_kkt_exact as kkt
import boxsel_phase4f_cell_atlas as atlas
import boxsel_phase4g_minpair_reduction as red

F = Fraction
GLOBAL_LOWER_BOUND_CLOSED = False
Q_KKT = kkt.Q_STAR

REMAINING_GLOBAL_OBLIGATIONS = (
    "exact lower certificate for the 10 two-help AC-vs-BC mixed n=2 orbits.",
    "dimension compression: prove n>=3 cannot beat the 2-D candidate, or find a counterexample.",
)


def one_help_side(pair_rep) -> str | None:
    """Return AC or BC if the representative is a one-help orbit, else None.

    AC=BC on the helping axis can be certified by either side; choose AC deterministically.
    """
    helping = [cell for cell in pair_rep if atlas.helps(cell)]
    if len(helping) != 1:
        return None
    sides = red.exact_factor_sides(red.min_pair_pattern(helping[0]))
    if "AC" in sides:
        return "AC"
    if "BC" in sides:
        return "BC"
    return None


def orbit_status(pair_rep) -> str:
    previous = red.orbit_status(pair_rep)
    if previous == "one_help_mixed_open":
        return "one_help_closed"
    return previous


def one_help_closure_classification() -> Counter:
    return Counter(orbit_status(rep) for rep in atlas.n2_orbits())


def exact_closed_count() -> int:
    counts = one_help_closure_classification()
    return counts["zero_help_closed"] + counts["same_side_closed"] + counts["one_help_closed"]


def remaining_open_count() -> int:
    return one_help_closure_classification()["two_help_mixed_open"]


def one_help_lower_bound():
    """Every one-help n=2 orbit has q >= 1/2 by the volume contradiction above."""
    return kkt.Surd(F(1, 2))


def contradiction_bound(pair_overlap_axis1: Fraction, atom_length_axis1: Fraction) -> tuple[Fraction, Fraction]:
    """Return the forced lower and available upper in the one-help contradiction.

    If q < 1/2, the second-axis side overlap is forced above
    `1/(2*pair_overlap_axis1)`. But the relevant atom length on axis 2 is
    `1/(2*atom_length_axis1)`, and `pair_overlap_axis1 <= atom_length_axis1` makes this upper bound
    no larger than the forced lower bound.
    """
    if pair_overlap_axis1 <= 0 or atom_length_axis1 <= 0:
        raise ValueError("lengths must be positive")
    if pair_overlap_axis1 > atom_length_axis1:
        raise ValueError("pair overlap must be bounded by the corresponding atom length")
    forced_lower = F(1, 2) / pair_overlap_axis1
    available_upper = F(1, 2) / atom_length_axis1
    return forced_lower, available_upper


def certified_sandwich():
    """(lower, candidate_upper): unchanged by this phase."""
    return kkt.CERTIFIED_LOWER, Q_KKT


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("one-help closure classification:", dict(one_help_closure_classification()))
    print("exact-closed n=2 orbits:", exact_closed_count(), "/ 123")
    print("remaining open n=2 orbits:", remaining_open_count())
    print("one-help lower bound:", one_help_lower_bound(), "> q_KKT =", Q_KKT)
    print("global lower bound closed:", GLOBAL_LOWER_BOUND_CLOSED, "sandwich:", certified_sandwich())
