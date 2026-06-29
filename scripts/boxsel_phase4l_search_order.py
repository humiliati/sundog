#!/usr/bin/env python
r"""BoxSEL Phase 4l - the SEARCH-ORDER filtration: a measured determine/resist split on REACHABILITY.

C1 of the cross-lane conjecture slate (docs/boxsel/BOXSEL_CONJECTURE_SLATE.md). It turns the Phase-4c
qualitative finding -- "every from-scratch search misses the optimum" -- into a measured ORDER
PARAMETER, and shows search-reachability is a determine/resist filtration of the SAME shape the
sufficient-statistic-order schema found for determination (but on a genuinely different axis: what a
bounded PROCESS reaches, not what a statistic DETERMINES).

    search filtration  F_g  = configs reachable by exact rational search at resolution 1/g,
                              nested along a divisibility chain (F_g subset F_{2g} subset ...),
                              so grid_min(g) is MONOTONE non-increasing -> a proper filtration.
    search order        sigma_search(target, eps) = least g in the chain with grid_min(g) <= target+eps
                              (the order at which bounded exact search first reaches within eps of target).

  DETERMINE side (finite order, reachable): any target ABOVE the grid's reachable floor -- e.g. 1/2.
  RESIST   side (infinite order over the feasible chain): the n=2 closed-form optimum (9+sqrt 17)/32,
           which sits BELOW the floor. Bounded exact search returns ~4/9 and never approaches the truth;
           the measured RESIST MARGIN = floor - optimum ~= 0.034.

PRE-REGISTERED VACUITY GUARD (else this is a pretty essay, not a result -- the Phase-7 lesson):
  "irrational, ergo not on any finite grid" is trivially true and proves nothing. C1 is SUPPORTED only
  if all three hold:
    (a) the order-meter is LIVE -- it REACHES a determine-side target (1/2) at finite order, so the
        optimum's unreachability is a property of the TARGET, not a dead probe (cf. the H9-strong
        order-meter control);
    (b) a MEASURED plateau -- grid_min(g) stays >= optimum + MARGIN for every g in the chain (the search
        does not even approach the truth), not merely "the exact value is irrational";
    (c) MONOTONE along the nested chain (the filtration is proper).
  If grid_min descends within MARGIN of the optimum at any feasible g, C1 is NULL (search-reachability
  is not a resist filtration here) -- and the probe says so.

Exact/pure (Fraction + the Q(sqrt 17) Surd for the optimum); deterministic; reuses Phase-4c's
exact_grid_min and Phase-4d's closed form.
"""
from fractions import Fraction

import boxsel_exact_inf_optimizer as opt   # exact_grid_min(g): exhaustive exact rational grid search
import boxsel_kkt_exact as kkt             # Q_STAR = (9+sqrt 17)/32, the Surd field

F = Fraction
S = kkt.Surd

# Nested divisibility chain 6 | 12 | 24 -> F_6 subset F_12 subset F_24 (every i/6 = 4i/24, lengths scale)
# -> grid_min monotone non-increasing -> a proper filtration. (Extendable; 48 is feasible but slow.)
NESTED_CHAIN = (6, 12, 24)

OPTIMUM = kkt.Q_STAR                 # (9 + sqrt 17)/32 ~= 0.4100970 -- the resist-side ground truth
DETERMINE_TARGET = S(F(1, 2))        # 1/2 -- a determine-side target above the reachable floor
RESIST_MARGIN = F(2, 100)            # 0.02: the optimum must stay this far below unreached (measured plateau)


def grid_floors(chain=NESTED_CHAIN):
    """exact_grid_min(g) for each g in the nested chain (exact Fractions)."""
    return {g: opt.exact_grid_min(g) for g in chain}


def _as_surd(x):
    return x if isinstance(x, S) else S(x)


def search_order(target, eps, floors):
    """Least g in the chain with grid_min(g) <= target + eps (the search order to reach within eps).

    Returns None if no rung reaches it (infinite order over the feasible chain). target may be a
    Fraction or a Surd; eps is a Fraction; comparison is exact in Q(sqrt 17).
    """
    bound = _as_surd(target) + _as_surd(eps)
    for g in sorted(floors):
        if _as_surd(floors[g]) <= bound:
            return g
    return None


def is_monotone(floors):
    """grid_min non-increasing along the nested chain (proper filtration)."""
    vals = [floors[g] for g in sorted(floors)]
    return all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def reachable_floor(floors):
    """The best (smallest) value bounded exact search reaches over the chain (an exact Fraction)."""
    return min(floors.values())


def determine_resist_report(chain=NESTED_CHAIN):
    """The measured determine/resist split on the search filtration + the pre-registered verdict."""
    floors = grid_floors(chain)
    floor = reachable_floor(floors)
    margin = _as_surd(floor) - OPTIMUM                 # floor - optimum (exact Surd) = resist margin

    meter_live = search_order(DETERMINE_TARGET, F(0), floors) is not None
    plateau_holds = all(_as_surd(floors[g]) >= OPTIMUM + RESIST_MARGIN for g in floors)
    monotone = is_monotone(floors)
    supported = meter_live and plateau_holds and monotone

    # the optimum's reachability: smallest eps at which it is reached (the resist margin), and that no
    # finite rung reaches it any closer
    opt_order_exact = search_order(OPTIMUM, F(0), floors)            # exact value: None (irrational)
    opt_order_within_margin = search_order(OPTIMUM, RESIST_MARGIN, floors)  # opt+0.02: None (plateau)
    opt_order_at_floor = search_order(OPTIMUM, margin if isinstance(margin, S) else _as_surd(margin), floors)

    return {
        "chain": chain,
        "floors": floors,
        "reachableFloor": floor,
        "optimum": OPTIMUM,
        "resistMarginExact": margin,
        "monotone": monotone,
        "meterLive": meter_live,
        "plateauHolds": plateau_holds,
        # the split, as search orders:
        "determineTargetOrder": search_order(DETERMINE_TARGET, F(0), floors),   # finite
        "optimumOrderExact": opt_order_exact,                                   # None (resist)
        "optimumOrderWithinMargin": opt_order_within_margin,                    # None (resist)
        "optimumOrderAtFloor": opt_order_at_floor,                              # finite (only at the floor)
        "c1Supported": supported,
        "verdict": "search_resist_confirmed" if supported else "search_resist_null",
    }


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    r = determine_resist_report()
    print("BoxSEL Phase 4l - search-order filtration (nested chain", r["chain"], ")")
    print()
    print("  grid_min(g) along the nested filtration:")
    for g in r["chain"]:
        print(f"    g={g:>3}:  grid_min = {r['floors'][g]}  = {float(r['floors'][g]):.6f}")
    print(f"  reachable floor (best bounded exact search) = {r['reachableFloor']} = {float(r['reachableFloor']):.6f}")
    print(f"  optimum (9+sqrt17)/32                        = {float(r['optimum']):.6f}")
    print(f"  RESIST MARGIN (floor - optimum)              = {float(r['resistMarginExact']):.6f}")
    print()
    print("  determine/resist split on the search filtration:")
    print(f"    DETERMINE  sigma_search(1/2, 0)            = {r['determineTargetOrder']}  (finite -> reachable)")
    print(f"    RESIST     sigma_search(optimum, 0)        = {r['optimumOrderExact']}  (None -> infinite order)")
    print(f"    RESIST     sigma_search(optimum, +0.02)    = {r['optimumOrderWithinMargin']}  (None -> not even approached)")
    print()
    print(f"  meter live (1/2 reached): {r['meterLive']}   plateau holds: {r['plateauHolds']}   monotone: {r['monotone']}")
    print(f"  VERDICT: {r['verdict']}  (C1 search-reachability filtration {'SUPPORTED' if r['c1Supported'] else 'NULL'})")
