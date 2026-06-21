#!/usr/bin/env python
"""BoxSEL Phase 7b - preregistration start after the Phase-7 bounded null.

This file is intentionally *not* a locked prereg.  It freezes the first Phase-7b boundary
conditions and records the blockers that must be cleared before a held-out v2 detector run can
begin.
"""

from __future__ import annotations

from dataclasses import dataclass

import boxsel_phase6b_trace_schema as schema
import boxsel_phase7_prereg as phase7
import boxsel_phase7b_v2_detector as v2


PHASE7B_PREREG_STATUS = "LOCKED_NOT_RUN"
PHASE7B_PREREG_LOCKED = True
RESULTS_STATUS = "NOT_RUN"
RESULT_ROWS: tuple = ()

SCHEMA_VERSION = schema.SCHEMA_VERSION
DETECTOR_VERSION = v2.DETECTOR_VERSION
DETECTOR_STATUS = v2.DETECTOR_STATUS
THRESHOLD_VERSION = v2.THRESHOLD_VERSION
THRESHOLD_STATUS = v2.THRESHOLD_STATUS
FROZEN_THRESHOLDS = v2.frozen_thresholds()
CORPUS_GENERATOR_STATUS = "BUILT"
EVALUATOR_STATUS = "BUILT"
HELDOUT_RUN_STATUS = "READY_NOT_RUN"

PRIMARY_BASELINE_VERSION = "restart_variance_only_v0"
DIAGNOSTIC_BASELINE_VERSION = "phase6_trace_detector_start"
BASELINES = (PRIMARY_BASELINE_VERSION, DIAGNOSTIC_BASELINE_VERSION)

FROZEN_TRACE_FEATURES = schema.feature_names()
REQUIRED_V2_FEATURES = (
    "pressure_low_shift",
    "optimizer_low_spread",
    "support_floor",
    "max_constraint_violation",
    "min_constraint_slack",
)
FORBIDDEN_FEATURE_TOKENS = schema.FORBIDDEN_FEATURE_TOKENS

SEEN_CASE_EXCLUSIONS = schema.PHASE7_SEEN_CASE_IDS
TRAINING_OR_DIAGNOSTIC_SEEDS = (
    *phase7.TRAINING_OR_SEED_TRAP_SEEDS,
    *phase7.HELDOUT_SEEDS,
    0,
    1,
    2,
)
RESERVED_HELDOUT_SEEDS = (
    9001,
    9011,
    9029,
    9041,
    9059,
    9067,
    9091,
    9103,
    9127,
    9137,
    9151,
    9173,
    9187,
    9203,
    9221,
    9239,
)


@dataclass(frozen=True)
class HeldoutFamily:
    name: str
    count: int
    role: str
    description: str


HELDOUT_FAMILIES = (
    HeldoutFamily(
        name="stable_pmp_pressure_variants",
        count=8,
        role="false_closure_trap",
        description=(
            "Stable low-loss PMP-shaped false closures where ordinary restarts are quiet but "
            "query-pressure traces should move the lower endpoint."
        ),
    ),
    HeldoutFamily(
        name="helly_threshold_variants_v2",
        count=4,
        role="false_closure_trap",
        description=(
            "Fresh Helly threshold cases generated from new seeds/thresholds, not Phase-7 seed "
            "variants."
        ),
    ),
    HeldoutFamily(
        name="support_floor_variants",
        count=4,
        role="false_closure_trap",
        description=(
            "Tiny role-free cases designed to test denominator/numerator support-floor signals "
            "under query pressure."
        ),
    ),
    HeldoutFamily(
        name="true_narrow_controls_v2",
        count=6,
        role="acceptance_control",
        description="Tiny cases whose exact query interval is narrow; guards against always-widen.",
    ),
    HeldoutFamily(
        name="pressure_noop_controls",
        count=3,
        role="acceptance_control",
        description="Controls where query pressure should not materially move the endpoint.",
    ),
    HeldoutFamily(
        name="loss_escape_controls_v2",
        count=3,
        role="loss_control",
        description="Nonzero-loss controls; v2 must abstain rather than accept.",
    ),
)

SUBSTANTIAL_EXACT_WIDENING = 0.10
MAX_ACCEPTED_FALSE_CLOSURE_RATE = 0.10
KILL_ACCEPTED_FALSE_CLOSURE_RATE = 0.25
MIN_BASELINE_IMPROVEMENT = 0.20
MIN_TRUE_NARROW_ACCEPT_RATE = 0.50
MAX_LOSS_ESCAPE_ACCEPTS = 0

LOCK_BLOCKERS = ()

PREDICTIONS = (
    "P7B-P1: once locked, v2 will widen or abstain on at least 90% of held-out false-closure traps.",
    "P7B-P2: accepted false-closure rate will beat restart_variance_only_v0 by at least 20 percentage points.",
    "P7B-P3: stable true-narrow controls will be accepted at least 50% of the time.",
    "P7B-P4: no loss-escape control may be accepted.",
    "P7B-P5: stable PMP pressure variants will trigger pressure or optimizer-spread warnings more often than restart-variance warnings.",
)

KILL_CRITERIA = (
    "KILL7B-1: accepted false-closure rate is 25% or higher on held-out traps.",
    "KILL7B-2: v2 fails to beat restart_variance_only_v0 by 20 percentage points.",
    "KILL7B-3: any loss-escape control is accepted.",
    "KILL7B-4: Phase-7 seen cases are reused as held-out validation rows.",
    "KILL7B-5: thresholds or feature definitions are changed after the locked prereg.",
)


def heldout_case_count(role: str | None = None) -> int:
    families = HELDOUT_FAMILIES if role is None else tuple(f for f in HELDOUT_FAMILIES if f.role == role)
    return sum(f.count for f in families)


def reserved_seeds_are_clean() -> bool:
    return set(RESERVED_HELDOUT_SEEDS).isdisjoint(TRAINING_OR_DIAGNOSTIC_SEEDS)


def seen_cases_are_excluded(case_ids: tuple[str, ...]) -> bool:
    return set(case_ids).isdisjoint(SEEN_CASE_EXCLUSIONS)


def feature_list_is_oracle_free() -> bool:
    lowered = tuple(name.lower() for name in FROZEN_TRACE_FEATURES)
    return all(not any(token in name for token in FORBIDDEN_FEATURE_TOKENS) for name in lowered)


def required_v2_features_present() -> bool:
    return set(REQUIRED_V2_FEATURES).issubset(set(FROZEN_TRACE_FEATURES))


def prereg_can_lock() -> bool:
    return PHASE7B_PREREG_LOCKED and not LOCK_BLOCKERS


def prereg_summary() -> dict[str, object]:
    return {
        "status": PHASE7B_PREREG_STATUS,
        "locked": PHASE7B_PREREG_LOCKED,
        "results_status": RESULTS_STATUS,
        "schema_version": SCHEMA_VERSION,
        "detector_version": DETECTOR_VERSION,
        "detector_status": DETECTOR_STATUS,
        "threshold_version": THRESHOLD_VERSION,
        "threshold_status": THRESHOLD_STATUS,
        "heldout_cases": heldout_case_count(),
        "false_closure_traps": heldout_case_count("false_closure_trap"),
        "lock_blockers": LOCK_BLOCKERS,
    }


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("Phase 7b prereg start:", prereg_summary())
    print("families:")
    for family in HELDOUT_FAMILIES:
        print(f" - {family.name}: {family.count} ({family.role})")
