#!/usr/bin/env python
"""BoxSEL Phase 7 - preregistered falsifier protocol.

This file is a prereg lock, not a result harness. It freezes the held-out protocol before any
Phase-7 held-out detector runs are executed.
"""

from __future__ import annotations

from dataclasses import dataclass

import boxsel_phase6_trace_detector as phase6


PHASE7_PREREG_LOCKED = True
RESULTS_STATUS = "NOT_RUN"
RESULT_ROWS: tuple = ()

DETECTOR_VERSION = "phase6_trace_detector_start"
BASELINE_VERSION = "restart_variance_only_v0"

TRACE_ONLY_FEATURES = (
    "sample_lower",
    "sample_upper",
    "sample_width",
    "early_lower_drop",
    "late_lower_drop",
    "max_loss",
    "min_slack",
    "seed_low_range",
    "dimension_low_spread",
)

FORBIDDEN_FEATURE_TOKENS = ("exact", "oracle", "ibox", "i_box", "istar", "i_star")

TRAINING_OR_SEED_TRAP_SEEDS = (314159, 271828, 11, 23, 37, 53, 101)
HELDOUT_SEEDS = (7001, 7003, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7121)


@dataclass(frozen=True)
class HeldoutFamily:
    name: str
    count: int
    role: str
    description: str


HELDOUT_FAMILIES = (
    HeldoutFamily(
        name="helly_threshold_variants",
        count=6,
        role="false_closure_trap",
        description=(
            "Role-free three-atom Helly variants with atom marginals near 1/2 and pairwise lower "
            "constraints chosen before running; exact oracle must show a substantially wider lower "
            "endpoint than ordinary restart sampling."
        ),
    ),
    HeldoutFamily(
        name="pmp_interval_chain_variants",
        count=4,
        role="false_closure_trap",
        description=(
            "Small PMP-shaped role-free ontologies generated from frozen rational premise intervals; "
            "exact oracle labels interval width before detector scoring."
        ),
    ),
    HeldoutFamily(
        name="true_narrow_controls",
        count=4,
        role="acceptance_control",
        description=(
            "Tiny ontologies whose exact query interval is already narrow, used to prevent the guard "
            "from degenerating into always-widen or always-abstain."
        ),
    ),
    HeldoutFamily(
        name="loss_escape_controls",
        count=2,
        role="loss_control",
        description="Near-feasible runs with nonzero loss; guard must abstain rather than accept.",
    ),
)


FALSE_CLOSURE_GAP_THRESHOLD = phase6.FALSE_CLOSURE_GAP_THRESHOLD
SUBSTANTIAL_EXACT_WIDENING = 0.10
MAX_ACCEPTED_FALSE_CLOSURE_RATE = 0.10
KILL_ACCEPTED_FALSE_CLOSURE_RATE = 0.25
MIN_BASELINE_IMPROVEMENT = 0.20
MIN_TRUE_NARROW_ACCEPT_RATE = 0.50
MAX_LOSS_ESCAPE_ACCEPTS = 0


PREDICTIONS = (
    "P7-P1: the trace guard will widen or abstain on at least 90% of held-out false-closure traps.",
    "P7-P2: accepted false-closure rate will be at least 20 percentage points lower than the restart-variance-only baseline.",
    "P7-P3: stable true-narrow controls will be accepted at least 50% of the time.",
    "P7-P4: no loss-escape control may be accepted.",
)

KILL_CRITERIA = (
    "KILL7-1: accepted false-closure rate is 25% or higher on held-out traps.",
    "KILL7-2: the guard fails to beat the restart-variance-only baseline by 20 percentage points.",
    "KILL7-3: any loss-escape control is accepted.",
    "KILL7-4: thresholds or feature definitions are changed after held-out results are generated.",
)


def detector_thresholds() -> dict[str, float]:
    """The Phase-6 guard thresholds frozen for Phase 7."""
    return {
        "LOSS_TOLERANCE": phase6.LOSS_TOLERANCE,
        "EARLY_DROP_THRESHOLD": phase6.EARLY_DROP_THRESHOLD,
        "LATE_DROP_THRESHOLD": phase6.LATE_DROP_THRESHOLD,
        "ACTIVE_SLACK_THRESHOLD": phase6.ACTIVE_SLACK_THRESHOLD,
        "SEED_RANGE_THRESHOLD": phase6.SEED_RANGE_THRESHOLD,
        "DIMENSION_SPREAD_THRESHOLD": phase6.DIMENSION_SPREAD_THRESHOLD,
        "FALSE_CLOSURE_GAP_THRESHOLD": phase6.FALSE_CLOSURE_GAP_THRESHOLD,
    }


def heldout_case_count(role: str | None = None) -> int:
    families = HELDOUT_FAMILIES if role is None else tuple(f for f in HELDOUT_FAMILIES if f.role == role)
    return sum(f.count for f in families)


def seeds_are_held_out() -> bool:
    return set(HELDOUT_SEEDS).isdisjoint(TRAINING_OR_SEED_TRAP_SEEDS)


def feature_list_is_oracle_free() -> bool:
    lowered = tuple(name.lower() for name in TRACE_ONLY_FEATURES)
    return all(not any(token in name for token in FORBIDDEN_FEATURE_TOKENS) for name in lowered)


def prereg_summary() -> dict[str, object]:
    return {
        "locked": PHASE7_PREREG_LOCKED,
        "results_status": RESULTS_STATUS,
        "detector_version": DETECTOR_VERSION,
        "baseline_version": BASELINE_VERSION,
        "heldout_cases": heldout_case_count(),
        "false_closure_traps": heldout_case_count("false_closure_trap"),
        "thresholds": detector_thresholds(),
    }


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("Phase 7 prereg:", prereg_summary())
    print("families:")
    for family in HELDOUT_FAMILIES:
        print(f" - {family.name}: {family.count} ({family.role})")
