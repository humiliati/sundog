#!/usr/bin/env python
"""BoxSEL Phase 7b - execute the locked v2 false-closure run.

This is deliberately separate from ``boxsel_phase7b_prereg``.  The prereg file remains the
pre-result lock; this module builds the held-out run ledger and writes result artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import boxsel_phase7b_corpus as corpus
import boxsel_phase7b_evaluator as evaluator
import boxsel_phase7b_prereg as prereg
import boxsel_phase7b_v2_detector as detector


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "boxsel" / "phase7b_false_closure_run"

RUN_STATUS_PASS = "PASS_PREREG_GATE"
RUN_STATUS_FAIL = "FAIL_PREREG_GATE"


@dataclass(frozen=True)
class CaseRun:
    case_id: str
    family: str
    role: str
    seed: int
    exact_lower: float
    exact_upper: float
    sample_lower: float
    sample_upper: float
    lower_exact_widening: float
    false_closed: bool
    detector_action: str
    detector_flags: tuple[str, ...]
    baseline_action: str
    baseline_flags: tuple[str, ...]
    accepted_false_closure: bool
    baseline_accepted_false_closure: bool
    max_loss: float
    pressure_low_shift: float
    optimizer_low_spread: float
    support_floor: float
    seed_low_range: float
    dimension_low_spread: float
    early_lower_drop: float
    late_lower_drop: float
    note: str


@dataclass(frozen=True)
class RunSummary:
    status: str
    case_count: int
    false_closure_traps: int
    detector_accepted_false_closures: int
    baseline_accepted_false_closures: int
    accepted_false_closure_rate: float
    baseline_accepted_false_closure_rate: float
    baseline_improvement: float
    true_narrow_accept_rate: float
    loss_escape_accept_count: int
    pressure_warning_rate_on_stable_pmp: float
    baseline_pressure_warning_rate_on_stable_pmp: float
    kill_criteria_triggered: tuple[str, ...]
    predictions_supported: tuple[str, ...]
    limitations: tuple[str, ...]


def _detector_decision(case: corpus.Phase7bCase) -> detector.V2Decision:
    return detector.detector_decision(
        case.ordinary_trace,
        seed_traces=case.seed_traces,
        dimension_traces=case.dimension_traces,
        optimizer_traces=case.optimizer_traces,
        pressure_traces=case.pressure_traces,
    )


def _case_note(case: corpus.Phase7bCase, decision: detector.V2Decision) -> str:
    if case.family == "stable_pmp_pressure_variants":
        return "Stable PMP pressure trap; v2 should abstain from query-pressure/optimizer spread."
    if case.family == "helly_threshold_variants_v2":
        return "Fresh Helly threshold trap; v2 should widen from ordinary endpoint movement."
    if case.family == "support_floor_variants":
        return "Low-support pressure trap; v2 should abstain or widen from pressure/support warnings."
    if case.role == "loss_control":
        return "Nonzero-loss control; v2 must abstain."
    if decision.action == detector.ACTION_ACCEPT:
        return "Acceptance control; v2 should accept when no frozen warning flags fire."
    return case.description


def _case_run(case: corpus.Phase7bCase) -> CaseRun:
    decision = _detector_decision(case)
    baseline_action, baseline_flags = evaluator.restart_variance_baseline_action(case)
    label = evaluator.label_case(case)
    features = decision.features
    accepted = decision.action == detector.ACTION_ACCEPT
    baseline_accepted = baseline_action == evaluator.ACTION_ACCEPT
    return CaseRun(
        case_id=case.case_id,
        family=case.family,
        role=case.role,
        seed=case.seed,
        exact_lower=label.exact_lower,
        exact_upper=label.exact_upper,
        sample_lower=label.sample_lower,
        sample_upper=label.sample_upper,
        lower_exact_widening=label.lower_exact_widening,
        false_closed=label.false_closed,
        detector_action=decision.action,
        detector_flags=decision.flags,
        baseline_action=baseline_action,
        baseline_flags=baseline_flags,
        accepted_false_closure=label.false_closed and accepted,
        baseline_accepted_false_closure=label.false_closed and baseline_accepted,
        max_loss=features.max_loss,
        pressure_low_shift=features.pressure_low_shift,
        optimizer_low_spread=features.optimizer_low_spread,
        support_floor=features.support_floor,
        seed_low_range=features.seed_low_range,
        dimension_low_spread=features.dimension_low_spread,
        early_lower_drop=features.early_lower_drop,
        late_lower_drop=features.late_lower_drop,
        note=_case_note(case, decision),
    )


def run_cases() -> tuple[CaseRun, ...]:
    if not prereg.PHASE7B_PREREG_LOCKED or prereg.RESULTS_STATUS != "NOT_RUN":
        raise RuntimeError("Phase-7b prereg must be locked and result-free before running")
    if prereg.LOCK_BLOCKERS:
        raise RuntimeError(f"Phase-7b lock blockers remain: {prereg.LOCK_BLOCKERS}")
    if prereg.HELDOUT_RUN_STATUS != "READY_NOT_RUN":
        raise RuntimeError("Phase-7b held-out run is not marked READY_NOT_RUN")
    cases = corpus.generate_phase7b_corpus()
    if not prereg.seen_cases_are_excluded(tuple(case.case_id for case in cases)):
        raise RuntimeError("Phase-7 seen case id reused in Phase-7b corpus")
    return tuple(_case_run(case) for case in cases)


def summarize(cases: tuple[CaseRun, ...]) -> RunSummary:
    traps = tuple(case for case in cases if case.role == "false_closure_trap" and case.false_closed)
    controls = tuple(case for case in cases if case.role == "acceptance_control")
    loss_controls = tuple(case for case in cases if case.role == "loss_control")
    stable_pmp = tuple(case for case in cases if case.family == "stable_pmp_pressure_variants")

    trap_count = len(traps)
    detector_accepted = sum(1 for case in traps if case.accepted_false_closure)
    baseline_accepted = sum(1 for case in traps if case.baseline_accepted_false_closure)
    accepted_rate = detector_accepted / trap_count if trap_count else 0.0
    baseline_rate = baseline_accepted / trap_count if trap_count else 0.0
    baseline_improvement = baseline_rate - accepted_rate
    true_narrow_accept_rate = (
        sum(1 for case in controls if case.detector_action == detector.ACTION_ACCEPT) / len(controls)
        if controls
        else 0.0
    )
    loss_escape_accept_count = sum(1 for case in loss_controls if case.detector_action == detector.ACTION_ACCEPT)
    pressure_warning_rate = (
        sum(
            1
            for case in stable_pmp
            if "pressure_low_shift" in case.detector_flags or "optimizer_low_spread" in case.detector_flags
        )
        / len(stable_pmp)
        if stable_pmp
        else 0.0
    )
    baseline_pressure_warning_rate = (
        sum(
            1
            for case in stable_pmp
            if "pressure_low_shift" in case.baseline_flags or "optimizer_low_spread" in case.baseline_flags
        )
        / len(stable_pmp)
        if stable_pmp
        else 0.0
    )

    kills = []
    if accepted_rate >= prereg.KILL_ACCEPTED_FALSE_CLOSURE_RATE:
        kills.append("KILL7B-1")
    if baseline_improvement < prereg.MIN_BASELINE_IMPROVEMENT:
        kills.append("KILL7B-2")
    if loss_escape_accept_count > prereg.MAX_LOSS_ESCAPE_ACCEPTS:
        kills.append("KILL7B-3")

    predictions = []
    if accepted_rate <= prereg.MAX_ACCEPTED_FALSE_CLOSURE_RATE:
        predictions.append("P7B-P1")
    if baseline_improvement >= prereg.MIN_BASELINE_IMPROVEMENT:
        predictions.append("P7B-P2")
    if true_narrow_accept_rate >= prereg.MIN_TRUE_NARROW_ACCEPT_RATE:
        predictions.append("P7B-P3")
    if loss_escape_accept_count == prereg.MAX_LOSS_ESCAPE_ACCEPTS:
        predictions.append("P7B-P4")
    if pressure_warning_rate > baseline_pressure_warning_rate:
        predictions.append("P7B-P5")

    limitations = (
        "The corpus is still tiny role-free micro-SEL infrastructure, not a real-KG claim.",
        "The v2 pressure traces are deterministic query-pressure probes, not exact inference.",
        "The prereg lock remains the frozen boundary; this run writes separate result artifacts.",
    )
    return RunSummary(
        status=RUN_STATUS_FAIL if kills else RUN_STATUS_PASS,
        case_count=len(cases),
        false_closure_traps=trap_count,
        detector_accepted_false_closures=detector_accepted,
        baseline_accepted_false_closures=baseline_accepted,
        accepted_false_closure_rate=accepted_rate,
        baseline_accepted_false_closure_rate=baseline_rate,
        baseline_improvement=baseline_improvement,
        true_narrow_accept_rate=true_narrow_accept_rate,
        loss_escape_accept_count=loss_escape_accept_count,
        pressure_warning_rate_on_stable_pmp=pressure_warning_rate,
        baseline_pressure_warning_rate_on_stable_pmp=baseline_pressure_warning_rate,
        kill_criteria_triggered=tuple(kills),
        predictions_supported=tuple(predictions),
        limitations=limitations,
    )


def result_payload() -> dict[str, object]:
    cases = run_cases()
    summary = summarize(cases)
    return {
        "prereg": prereg.prereg_summary(),
        "detector": detector.detector_summary(),
        "corpus": corpus.corpus_summary(),
        "run_config": {
            "results_status_in_prereg": prereg.RESULTS_STATUS,
            "heldout_run_status_at_execution": prereg.HELDOUT_RUN_STATUS,
            "primary_baseline": prereg.PRIMARY_BASELINE_VERSION,
        },
        "summary": asdict(summary),
        "cases": [asdict(case) for case in cases],
    }


def write_results(out_dir: Path = DEFAULT_RESULTS_DIR) -> dict[str, object]:
    payload = result_payload()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    payload = write_results()
    summary = payload["summary"]
    print("BoxSEL Phase 7b run:", summary["status"])
    print(
        "false-closure accepted:",
        f"{summary['detector_accepted_false_closures']}/{summary['false_closure_traps']}",
        "baseline:",
        f"{summary['baseline_accepted_false_closures']}/{summary['false_closure_traps']}",
    )
    print("baseline improvement:", summary["baseline_improvement"])
    print("kill criteria:", ", ".join(summary["kill_criteria_triggered"]) or "none")
    print("results:", DEFAULT_RESULTS_DIR / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
