#!/usr/bin/env python
"""BoxSEL Phase 6 - trace detector start.

Phase 6 asks whether observable embedding traces can flag false closure without using the oracle.
This first slice turns Phase-3 restart traces into a guarded decision:

    accept / widen / abstain.

Important boundary: the detector features below are trace-only. They use sampled endpoints, endpoint
movement, zero-loss status, constraint slack, seed variance, and dimension sensitivity. They do NOT
use I*, the exact I_box endpoint, or the Phase-4 closed form. The exact endpoint is used only by the
separate evaluator that scores whether the detector accepted a known false-closure case.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import boxsel_phase3_restart_sampler as sampler


ACTION_ACCEPT = "accept"
ACTION_WIDEN = "widen"
ACTION_ABSTAIN = "abstain"

LOSS_TOLERANCE = 1e-9
EARLY_DROP_THRESHOLD = 0.05
LATE_DROP_THRESHOLD = 0.01
ACTIVE_SLACK_THRESHOLD = 0.005
SEED_RANGE_THRESHOLD = 0.02
DIMENSION_SPREAD_THRESHOLD = 0.02
FALSE_CLOSURE_GAP_THRESHOLD = 0.05


@dataclass(frozen=True)
class TraceFeatures:
    restarts: int
    sample_lower: float
    sample_upper: float
    sample_width: float
    early_lower_drop: float
    late_lower_drop: float
    max_loss: float
    min_slack: float
    seed_low_range: float
    dimension_low_spread: float


@dataclass(frozen=True)
class GuardDecision:
    action: str
    flags: tuple[str, ...]
    features: TraceFeatures

    @property
    def should_accept(self) -> bool:
        return self.action == ACTION_ACCEPT


@dataclass(frozen=True)
class OracleEvaluation:
    false_closed: bool
    accepted_false_closure: bool
    lower_search_gap: float
    decision: GuardDecision


def _lower_drop(report: sampler.SampleReport, fraction: float) -> float:
    movement = sampler.cumulative_endpoint_trace(report)
    index = max(1, min(len(movement), math.ceil(len(movement) * fraction))) - 1
    return movement[index][1] - movement[-1][1]


def seed_low_range(seed_reports: tuple[sampler.SampleReport, ...] = ()) -> float:
    if not seed_reports:
        return 0.0
    lows = [report.sample_interval[0] for report in seed_reports]
    return max(lows) - min(lows)


def dimension_low_spread(dimension_reports: tuple[sampler.SampleReport, ...] = ()) -> float:
    if not dimension_reports:
        return 0.0
    lows = [report.sample_interval[0] for report in dimension_reports]
    return max(lows) - min(lows)


def trace_features(
    report: sampler.SampleReport,
    seed_reports: tuple[sampler.SampleReport, ...] = (),
    dimension_reports: tuple[sampler.SampleReport, ...] = (),
) -> TraceFeatures:
    lo, hi = report.sample_interval
    return TraceFeatures(
        restarts=report.restarts,
        sample_lower=lo,
        sample_upper=hi,
        sample_width=hi - lo,
        early_lower_drop=_lower_drop(report, 0.125),
        late_lower_drop=_lower_drop(report, 0.75),
        max_loss=report.max_loss,
        min_slack=report.min_slack,
        seed_low_range=seed_low_range(seed_reports),
        dimension_low_spread=dimension_low_spread(dimension_reports),
    )


def feature_flags(features: TraceFeatures) -> tuple[str, ...]:
    flags = []
    if features.max_loss > LOSS_TOLERANCE:
        flags.append("loss_escape")
    if features.early_lower_drop > EARLY_DROP_THRESHOLD:
        flags.append("endpoint_drift")
    if features.late_lower_drop > LATE_DROP_THRESHOLD:
        flags.append("late_endpoint_drift")
    if features.min_slack < ACTIVE_SLACK_THRESHOLD:
        flags.append("active_constraint_slack")
    if features.seed_low_range > SEED_RANGE_THRESHOLD:
        flags.append("seed_variance")
    if features.dimension_low_spread > DIMENSION_SPREAD_THRESHOLD:
        flags.append("dimension_sensitivity")
    return tuple(flags)


def guarded_decision(features: TraceFeatures) -> GuardDecision:
    flags = feature_flags(features)
    if "loss_escape" in flags or len(flags) >= 3:
        action = ACTION_ABSTAIN
    elif flags:
        action = ACTION_WIDEN
    else:
        action = ACTION_ACCEPT
    return GuardDecision(action=action, flags=flags, features=features)


def detector_decision(
    report: sampler.SampleReport,
    seed_reports: tuple[sampler.SampleReport, ...] = (),
    dimension_reports: tuple[sampler.SampleReport, ...] = (),
) -> GuardDecision:
    return guarded_decision(trace_features(report, seed_reports, dimension_reports))


def evaluate_with_oracle(report: sampler.SampleReport, decision: GuardDecision) -> OracleEvaluation:
    """Evaluation-only: uses the exact endpoint to label false closure after the decision is made."""
    false_closed = report.lower_search_gap > FALSE_CLOSURE_GAP_THRESHOLD
    return OracleEvaluation(
        false_closed=false_closed,
        accepted_false_closure=false_closed and decision.should_accept,
        lower_search_gap=report.lower_search_gap,
        decision=decision,
    )


def helly_seed_detector_receipt() -> tuple[sampler.SampleReport, GuardDecision, OracleEvaluation]:
    """Canonical Phase-6 seed receipt over the Phase-3 Helly trace."""
    report = sampler.ordinary_restart_report(dim=2, restarts=128, seed=314159)
    seed_reports = sampler.seed_variance_report(dim=2, restarts=64, seeds=(11, 23, 37, 53))
    dim3 = sampler.ordinary_restart_report(dim=3, restarts=96, seed=271828)
    decision = detector_decision(report, seed_reports=seed_reports, dimension_reports=(report, dim3))
    return report, decision, evaluate_with_oracle(report, decision)


def synthetic_report(
    q_values: tuple[float, ...],
    *,
    min_pair_overlap: float = 0.35,
    max_loss: float = 0.0,
) -> sampler.SampleReport:
    """Small deterministic control report for detector unit tests."""
    traces = []
    for index, q in enumerate(q_values):
        traces.append(
            sampler.RestartTrace(
                restart=index,
                attempts=1,
                loss=max_loss,
                q=q,
                atom_volumes=(0.5, 0.5, 0.5),
                pair_overlaps=(min_pair_overlap, min_pair_overlap, min_pair_overlap),
            )
        )
    return sampler.SampleReport(dim=2, restarts=len(traces), seed=0, tolerance=sampler.DEFAULT_TOLERANCE, traces=tuple(traces))


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    report, decision, evaluation = helly_seed_detector_receipt()
    print("I_sample:", report.sample_interval)
    print("features:", decision.features)
    print("decision:", decision.action, decision.flags)
    print("oracle evaluation:", evaluation)
