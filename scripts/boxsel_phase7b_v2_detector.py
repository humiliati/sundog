#!/usr/bin/env python
"""BoxSEL Phase 7b - frozen v2 trace detector.

The detector is trace-only.  It consumes Phase-6b ``GeneralTraceFeatures`` and returns
``accept / widen / abstain`` without reading exact endpoints, oracle labels, or evaluator output.
"""

from __future__ import annotations

from dataclasses import dataclass

import boxsel_phase6b_trace_schema as schema


DETECTOR_VERSION = "phase7b_v2_trace_detector_v1"
DETECTOR_STATUS = "FROZEN"
THRESHOLD_VERSION = "phase7b_v2_thresholds_v1"
THRESHOLD_STATUS = "FROZEN"
RESULTS_STATUS = "NOT_RUN"
FEATURE_SCHEMA_VERSION = schema.SCHEMA_VERSION

ACTION_ACCEPT = "accept"
ACTION_WIDEN = "widen"
ACTION_ABSTAIN = "abstain"
VALID_ACTIONS = (ACTION_ACCEPT, ACTION_WIDEN, ACTION_ABSTAIN)

LOSS_TOLERANCE = 1e-9
CONSTRAINT_VIOLATION_THRESHOLD = 1e-9
PRESSURE_LOW_SHIFT_THRESHOLD = 0.05
OPTIMIZER_LOW_SPREAD_THRESHOLD = 0.05
SUPPORT_FLOOR_THRESHOLD = 0.08
EARLY_LOWER_DROP_THRESHOLD = 0.03
LATE_LOWER_DROP_THRESHOLD = 0.015
SEED_LOW_RANGE_THRESHOLD = 0.02
DIMENSION_LOW_SPREAD_THRESHOLD = 0.02

FROZEN_FEATURES = schema.feature_names()
PRESSURE_RESPONSE_FLAGS = ("pressure_low_shift", "optimizer_low_spread")
IMMEDIATE_ABSTAIN_FLAGS = ("max_loss", "max_constraint_violation", *PRESSURE_RESPONSE_FLAGS)
WIDEN_ONLY_FLAGS = (
    "support_floor",
    "early_lower_drop",
    "late_lower_drop",
    "seed_low_range",
    "dimension_low_spread",
)


@dataclass(frozen=True)
class V2Decision:
    action: str
    flags: tuple[str, ...]
    features: schema.GeneralTraceFeatures
    detector_version: str = DETECTOR_VERSION
    threshold_version: str = THRESHOLD_VERSION

    @property
    def should_accept(self) -> bool:
        return self.action == ACTION_ACCEPT


def frozen_thresholds() -> dict[str, float]:
    return {
        "max_loss": LOSS_TOLERANCE,
        "max_constraint_violation": CONSTRAINT_VIOLATION_THRESHOLD,
        "pressure_low_shift": PRESSURE_LOW_SHIFT_THRESHOLD,
        "optimizer_low_spread": OPTIMIZER_LOW_SPREAD_THRESHOLD,
        "support_floor": SUPPORT_FLOOR_THRESHOLD,
        "early_lower_drop": EARLY_LOWER_DROP_THRESHOLD,
        "late_lower_drop": LATE_LOWER_DROP_THRESHOLD,
        "seed_low_range": SEED_LOW_RANGE_THRESHOLD,
        "dimension_low_spread": DIMENSION_LOW_SPREAD_THRESHOLD,
    }


def feature_flags(features: schema.GeneralTraceFeatures) -> tuple[str, ...]:
    flags = []
    if features.max_loss > LOSS_TOLERANCE:
        flags.append("max_loss")
    if features.max_constraint_violation > CONSTRAINT_VIOLATION_THRESHOLD:
        flags.append("max_constraint_violation")
    if features.pressure_low_shift >= PRESSURE_LOW_SHIFT_THRESHOLD:
        flags.append("pressure_low_shift")
    if features.optimizer_low_spread >= OPTIMIZER_LOW_SPREAD_THRESHOLD:
        flags.append("optimizer_low_spread")
    if features.support_floor <= SUPPORT_FLOOR_THRESHOLD:
        flags.append("support_floor")
    if features.early_lower_drop >= EARLY_LOWER_DROP_THRESHOLD:
        flags.append("early_lower_drop")
    if features.late_lower_drop >= LATE_LOWER_DROP_THRESHOLD:
        flags.append("late_lower_drop")
    if features.seed_low_range >= SEED_LOW_RANGE_THRESHOLD:
        flags.append("seed_low_range")
    if features.dimension_low_spread >= DIMENSION_LOW_SPREAD_THRESHOLD:
        flags.append("dimension_low_spread")
    return tuple(flags)


def guarded_decision(features: schema.GeneralTraceFeatures) -> V2Decision:
    flags = feature_flags(features)
    if any(flag in IMMEDIATE_ABSTAIN_FLAGS for flag in flags):
        action = ACTION_ABSTAIN
    elif flags:
        action = ACTION_WIDEN
    else:
        action = ACTION_ACCEPT
    return V2Decision(action=action, flags=flags, features=features)


def detector_decision(
    trace: schema.GeneralTrace,
    *,
    seed_traces: tuple[schema.GeneralTrace, ...] = (),
    dimension_traces: tuple[schema.GeneralTrace, ...] = (),
    optimizer_traces: tuple[schema.GeneralTrace, ...] = (),
    pressure_traces: tuple[schema.GeneralTrace, ...] = (),
) -> V2Decision:
    return guarded_decision(
        schema.trace_features(
            trace,
            seed_traces=seed_traces,
            dimension_traces=dimension_traces,
            optimizer_traces=optimizer_traces,
            pressure_traces=pressure_traces,
        )
    )


def detector_summary() -> dict[str, object]:
    return {
        "detector_version": DETECTOR_VERSION,
        "detector_status": DETECTOR_STATUS,
        "threshold_version": THRESHOLD_VERSION,
        "threshold_status": THRESHOLD_STATUS,
        "results_status": RESULTS_STATUS,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "frozen_features": FROZEN_FEATURES,
        "frozen_thresholds": frozen_thresholds(),
    }


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("Phase 7b v2 detector:", detector_summary())
