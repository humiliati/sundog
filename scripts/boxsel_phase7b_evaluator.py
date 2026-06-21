#!/usr/bin/env python
"""BoxSEL Phase 7b - evaluator infrastructure.

The evaluator computes exact labels and scores future detector decisions.  It does not contain a
v2 detector and does not execute a held-out detector run.
"""

from __future__ import annotations

from dataclasses import dataclass

import boxsel_phase6_trace_detector as phase6
import boxsel_phase7b_corpus as corpus
import boxsel_phase7b_prereg as prereg


EVALUATOR_VERSION = "phase7b_evaluator_v0"
EVALUATOR_STATUS = "BUILT_NOT_RUN"

ACTION_ACCEPT = "accept"
ACTION_WIDEN = "widen"
ACTION_ABSTAIN = "abstain"
VALID_ACTIONS = (ACTION_ACCEPT, ACTION_WIDEN, ACTION_ABSTAIN)


@dataclass(frozen=True)
class CaseLabel:
    case_id: str
    family: str
    role: str
    sample_lower: float
    sample_upper: float
    exact_lower: float
    exact_upper: float
    lower_exact_widening: float
    low_loss: bool
    false_closed: bool
    pressure_low_shift: float
    optimizer_low_spread: float
    seed_low_range: float
    support_floor: float
    max_loss: float


@dataclass(frozen=True)
class DecisionEvaluation:
    case_id: str
    family: str
    role: str
    action: str
    flags: tuple[str, ...]
    false_closed: bool
    accepted_false_closure: bool
    loss_escape_accepted: bool
    label: CaseLabel


@dataclass(frozen=True)
class DecisionSummary:
    case_count: int
    false_closure_traps: int
    accepted_false_closures: int
    accepted_false_closure_rate: float
    true_narrow_accept_rate: float
    loss_escape_accept_count: int
    pressure_warning_rate_on_stable_pmp: float
    kill_criteria_triggered: tuple[str, ...]


def label_case(case: corpus.Phase7bCase) -> CaseLabel:
    exact = case.exact()
    if not exact.feasible:
        raise RuntimeError(f"case {case.case_id} exact oracle infeasible")
    features = case.features()
    sample_lower, sample_upper = case.ordinary_trace.sample_interval
    exact_lower, exact_upper = exact.interval()
    widening = sample_lower - exact_lower
    low_loss = features.max_loss <= phase6.LOSS_TOLERANCE
    false_closed = widening >= prereg.SUBSTANTIAL_EXACT_WIDENING and low_loss
    return CaseLabel(
        case_id=case.case_id,
        family=case.family,
        role=case.role,
        sample_lower=sample_lower,
        sample_upper=sample_upper,
        exact_lower=exact_lower,
        exact_upper=exact_upper,
        lower_exact_widening=widening,
        low_loss=low_loss,
        false_closed=false_closed,
        pressure_low_shift=features.pressure_low_shift,
        optimizer_low_spread=features.optimizer_low_spread,
        seed_low_range=features.seed_low_range,
        support_floor=features.support_floor,
        max_loss=features.max_loss,
    )


def restart_variance_baseline_action(case: corpus.Phase7bCase) -> tuple[str, tuple[str, ...]]:
    features = case.features()
    if features.seed_low_range > phase6.SEED_RANGE_THRESHOLD:
        return ACTION_WIDEN, ("seed_variance",)
    return ACTION_ACCEPT, ()


def evaluate_decision(
    case: corpus.Phase7bCase,
    action: str,
    flags: tuple[str, ...] = (),
) -> DecisionEvaluation:
    if action not in VALID_ACTIONS:
        raise ValueError(f"invalid action {action!r}")
    label = label_case(case)
    accepted = action == ACTION_ACCEPT
    return DecisionEvaluation(
        case_id=case.case_id,
        family=case.family,
        role=case.role,
        action=action,
        flags=flags,
        false_closed=label.false_closed,
        accepted_false_closure=label.false_closed and accepted,
        loss_escape_accepted=case.role == "loss_control" and accepted,
        label=label,
    )


def evaluate_restart_variance_baseline(cases: tuple[corpus.Phase7bCase, ...]) -> tuple[DecisionEvaluation, ...]:
    evaluations = []
    for case in cases:
        action, flags = restart_variance_baseline_action(case)
        evaluations.append(evaluate_decision(case, action, flags))
    return tuple(evaluations)


def summarize_decisions(evaluations: tuple[DecisionEvaluation, ...]) -> DecisionSummary:
    traps = tuple(ev for ev in evaluations if ev.label.false_closed and ev.role == "false_closure_trap")
    controls = tuple(ev for ev in evaluations if ev.role == "acceptance_control")
    loss_controls = tuple(ev for ev in evaluations if ev.role == "loss_control")
    stable_pmp = tuple(ev for ev in evaluations if ev.family == "stable_pmp_pressure_variants")

    trap_count = len(traps)
    accepted_false_closures = sum(1 for ev in traps if ev.accepted_false_closure)
    accepted_rate = accepted_false_closures / trap_count if trap_count else 0.0
    true_narrow_accept_rate = (
        sum(1 for ev in controls if ev.action == ACTION_ACCEPT) / len(controls) if controls else 0.0
    )
    loss_escape_accept_count = sum(1 for ev in loss_controls if ev.loss_escape_accepted)
    pressure_warning_rate = (
        sum(
            1
            for ev in stable_pmp
            if ("pressure_low_shift" in ev.flags or "optimizer_low_spread" in ev.flags)
        )
        / len(stable_pmp)
        if stable_pmp
        else 0.0
    )

    kills = []
    if accepted_rate >= prereg.KILL_ACCEPTED_FALSE_CLOSURE_RATE:
        kills.append("KILL7B-1")
    if loss_escape_accept_count > prereg.MAX_LOSS_ESCAPE_ACCEPTS:
        kills.append("KILL7B-3")

    return DecisionSummary(
        case_count=len(evaluations),
        false_closure_traps=trap_count,
        accepted_false_closures=accepted_false_closures,
        accepted_false_closure_rate=accepted_rate,
        true_narrow_accept_rate=true_narrow_accept_rate,
        loss_escape_accept_count=loss_escape_accept_count,
        pressure_warning_rate_on_stable_pmp=pressure_warning_rate,
        kill_criteria_triggered=tuple(kills),
    )


def label_corpus(cases: tuple[corpus.Phase7bCase, ...] | None = None) -> tuple[CaseLabel, ...]:
    cases = corpus.generate_phase7b_corpus() if cases is None else cases
    return tuple(label_case(case) for case in cases)


def evaluator_summary(cases: tuple[corpus.Phase7bCase, ...] | None = None) -> dict[str, object]:
    cases = corpus.generate_phase7b_corpus() if cases is None else cases
    labels = label_corpus(cases)
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_status": EVALUATOR_STATUS,
        "case_count": len(cases),
        "false_closed_labels": sum(1 for label in labels if label.false_closed),
        "lossy_labels": sum(1 for label in labels if not label.low_loss),
        "baseline_version": prereg.PRIMARY_BASELINE_VERSION,
    }


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("Phase 7b evaluator:", evaluator_summary())
