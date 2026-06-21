#!/usr/bin/env python
"""BoxSEL Phase 7 - execute the preregistered false-closure run.

This is deliberately separate from ``boxsel_phase7_prereg``.  The prereg file remains the
pre-result lock; this module builds the held-out run ledger and writes result artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path

import boxsel_exact_oracle as oracle
import boxsel_phase3_restart_sampler as sampler
import boxsel_phase6_trace_detector as detector
import boxsel_phase7_prereg as prereg


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "boxsel" / "phase7_false_closure_run"

RUN_STATUS_PASS = "PASS_PREREG_GATE"
RUN_STATUS_FAIL = "FAIL_PREREG_GATE"

MAIN_RESTARTS = 128
SEED_RESTARTS = 64
DIMENSION_RESTARTS = 96


@dataclass(frozen=True)
class CaseRun:
    case_id: str
    family: str
    role: str
    seed: int
    trace_source: str
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
    min_slack: float
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
    kill_criteria_triggered: tuple[str, ...]
    predictions_supported: tuple[str, ...]
    limitations: tuple[str, ...]


def _float_fraction(value: Fraction) -> float:
    return float(value.numerator) / float(value.denominator)


def _helly_exact_interval() -> tuple[float, float]:
    a = oracle.atom("A")
    b = oracle.atom("B")
    c = oracle.atom("C")
    ontology = (
        oracle.conditional(a, oracle.TOP, Fraction(1, 2)),
        oracle.conditional(b, oracle.TOP, Fraction(1, 2)),
        oracle.conditional(c, oracle.TOP, Fraction(1, 2)),
        oracle.conditional(b, a, Fraction(1, 2), Fraction(1)),
        oracle.conditional(c, a, Fraction(1, 2), Fraction(1)),
        oracle.conditional(c, b, Fraction(1, 2), Fraction(1)),
    )
    result = oracle.exact_interval(ontology, c, a & b, atoms=("A", "B", "C"))
    if not result.feasible:
        raise RuntimeError("Helly exact oracle case is infeasible")
    lo, hi = result.interval_exact()
    return _float_fraction(lo), _float_fraction(hi)


def _baseline_decision(features: detector.TraceFeatures) -> tuple[str, tuple[str, ...]]:
    """restart_variance_only_v0: widen iff lower endpoints vary across seeds."""
    if features.seed_low_range > prereg.detector_thresholds()["SEED_RANGE_THRESHOLD"]:
        return detector.ACTION_WIDEN, ("seed_variance",)
    return detector.ACTION_ACCEPT, ()


def _features_from_decision(decision: detector.GuardDecision) -> detector.TraceFeatures:
    return decision.features


def _case_from_report(
    *,
    case_id: str,
    family: str,
    role: str,
    seed: int,
    trace_source: str,
    report: sampler.SampleReport,
    exact_interval: tuple[float, float],
    decision: detector.GuardDecision,
    note: str,
) -> CaseRun:
    features = _features_from_decision(decision)
    baseline_action, baseline_flags = _baseline_decision(features)
    sample_lower, sample_upper = report.sample_interval
    exact_lower, exact_upper = exact_interval
    widening = sample_lower - exact_lower
    false_closed = widening >= prereg.SUBSTANTIAL_EXACT_WIDENING and report.max_loss <= detector.LOSS_TOLERANCE
    return CaseRun(
        case_id=case_id,
        family=family,
        role=role,
        seed=seed,
        trace_source=trace_source,
        exact_lower=exact_lower,
        exact_upper=exact_upper,
        sample_lower=sample_lower,
        sample_upper=sample_upper,
        lower_exact_widening=widening,
        false_closed=false_closed,
        detector_action=decision.action,
        detector_flags=decision.flags,
        baseline_action=baseline_action,
        baseline_flags=baseline_flags,
        accepted_false_closure=false_closed and decision.action == detector.ACTION_ACCEPT,
        baseline_accepted_false_closure=false_closed and baseline_action == detector.ACTION_ACCEPT,
        max_loss=report.max_loss,
        min_slack=features.min_slack,
        seed_low_range=features.seed_low_range,
        dimension_low_spread=features.dimension_low_spread,
        early_lower_drop=features.early_lower_drop,
        late_lower_drop=features.late_lower_drop,
        note=note,
    )


def _synthetic_report(
    q_values: tuple[float, ...],
    *,
    min_slack: float = 0.10,
    max_loss: float = 0.0,
) -> sampler.SampleReport:
    return detector.synthetic_report(
        q_values,
        min_pair_overlap=sampler.PAIR_TARGET + min_slack,
        max_loss=max_loss,
    )


def _stable_values(center: float, count: int = 48, width: float = 0.006) -> tuple[float, ...]:
    values = []
    for index in range(count):
        offset = ((index % 7) - 3) * width / 6.0
        values.append(max(0.0, min(1.0, center + offset)))
    return tuple(values)


def _pmp_case(index: int, seed: int, q1: Fraction, q2: Fraction, sample_center: float) -> CaseRun:
    exact = oracle.exact_pmp_interval(q1, q2)
    exact_interval = (_float_fraction(exact.lower_exact), _float_fraction(exact.upper_exact))
    report = _synthetic_report(_stable_values(sample_center), min_slack=0.12)
    decision = detector.detector_decision(report)
    return _case_from_report(
        case_id=f"pmp-{index:02d}",
        family="pmp_interval_chain_variants",
        role="false_closure_trap",
        seed=seed,
        trace_source="deterministic PMP-shaped low-loss synthetic trace",
        report=report,
        exact_interval=exact_interval,
        decision=decision,
        note=(
            f"PMP point premises q1={q1}, q2={q2}; exact lower is q1*q2. "
            "The stable sampled trace is a preregistered false-closure stressor."
        ),
    )


def _true_narrow_case(index: int, seed: int, value: Fraction) -> CaseRun:
    x = oracle.atom("X")
    y = oracle.atom("Y")
    exact = oracle.exact_interval((oracle.conditional(y, x, value),), y, x, atoms=("X", "Y"))
    exact_interval = (_float_fraction(exact.lower_exact), _float_fraction(exact.upper_exact))
    report = _synthetic_report(_stable_values(float(value), width=0.001), min_slack=0.20)
    decision = detector.detector_decision(report)
    return _case_from_report(
        case_id=f"narrow-{index:02d}",
        family="true_narrow_controls",
        role="acceptance_control",
        seed=seed,
        trace_source="deterministic true-narrow synthetic trace",
        report=report,
        exact_interval=exact_interval,
        decision=decision,
        note=f"Point conditional control with exact interval [{value}, {value}].",
    )


def _loss_escape_case(index: int, seed: int, value: Fraction) -> CaseRun:
    x = oracle.atom("X")
    y = oracle.atom("Y")
    exact = oracle.exact_interval((oracle.conditional(y, x, value),), y, x, atoms=("X", "Y"))
    exact_interval = (_float_fraction(exact.lower_exact), _float_fraction(exact.upper_exact))
    report = _synthetic_report(_stable_values(float(value), count=16, width=0.001), min_slack=0.20, max_loss=1e-6)
    decision = detector.detector_decision(report)
    return _case_from_report(
        case_id=f"loss-{index:02d}",
        family="loss_escape_controls",
        role="loss_control",
        seed=seed,
        trace_source="deterministic nonzero-loss synthetic trace",
        report=report,
        exact_interval=exact_interval,
        decision=decision,
        note="Nonzero-loss control: the guard must abstain immediately.",
    )


def run_cases() -> tuple[CaseRun, ...]:
    if not prereg.PHASE7_PREREG_LOCKED or prereg.RESULTS_STATUS != "NOT_RUN":
        raise RuntimeError("Phase-7 prereg must be locked and result-free before running")
    if not prereg.seeds_are_held_out():
        raise RuntimeError("held-out seeds overlap seed-trap seeds")

    seeds = prereg.HELDOUT_SEEDS
    helly_exact = _helly_exact_interval()
    shared_seed_reports = tuple(
        sampler.ordinary_restart_report(dim=2, restarts=SEED_RESTARTS, seed=seed)
        for seed in seeds[6:10]
    )
    shared_dim3 = sampler.ordinary_restart_report(dim=3, restarts=DIMENSION_RESTARTS, seed=seeds[10])

    cases: list[CaseRun] = []
    for index, seed in enumerate(seeds[:6]):
        report = sampler.ordinary_restart_report(dim=2, restarts=MAIN_RESTARTS, seed=seed)
        decision = detector.detector_decision(
            report,
            seed_reports=shared_seed_reports,
            dimension_reports=(report, shared_dim3),
        )
        cases.append(
            _case_from_report(
                case_id=f"helly-{index:02d}",
                family="helly_threshold_variants",
                role="false_closure_trap",
                seed=seed,
                trace_source="Phase-3 Helly box restart sampler",
                report=report,
                exact_interval=helly_exact,
                decision=decision,
                note=(
                    "Held-out seed variant of the Phase-3 Helly threshold case "
                    "(atom marginals 1/2, pairwise lower threshold 1/4)."
                ),
            )
        )

    pmp_specs = (
        (Fraction(1, 3), Fraction(2, 3), 0.38),
        (Fraction(2, 5), Fraction(3, 5), 0.39),
        (Fraction(1, 2), Fraction(1, 2), 0.40),
        (Fraction(3, 5), Fraction(2, 5), 0.38),
    )
    for index, (q1, q2, sample_center) in enumerate(pmp_specs):
        cases.append(_pmp_case(index, seeds[index], q1, q2, sample_center))

    for index, value in enumerate((Fraction(13, 20), Fraction(7, 10), Fraction(3, 4), Fraction(4, 5))):
        cases.append(_true_narrow_case(index, seeds[6 + index], value))

    for index, value in enumerate((Fraction(3, 5), Fraction(2, 3))):
        cases.append(_loss_escape_case(index, seeds[10 + index], value))

    return tuple(cases)


def summarize(cases: tuple[CaseRun, ...]) -> RunSummary:
    trap_cases = tuple(case for case in cases if case.role == "false_closure_trap" and case.false_closed)
    control_cases = tuple(case for case in cases if case.role == "acceptance_control")
    loss_cases = tuple(case for case in cases if case.role == "loss_control")

    detector_accepted = sum(1 for case in trap_cases if case.accepted_false_closure)
    baseline_accepted = sum(1 for case in trap_cases if case.baseline_accepted_false_closure)
    trap_count = len(trap_cases)
    accepted_rate = detector_accepted / trap_count if trap_count else 0.0
    baseline_rate = baseline_accepted / trap_count if trap_count else 0.0
    baseline_improvement = baseline_rate - accepted_rate
    true_narrow_accept_rate = (
        sum(1 for case in control_cases if case.detector_action == detector.ACTION_ACCEPT) / len(control_cases)
        if control_cases
        else 0.0
    )
    loss_escape_accept_count = sum(1 for case in loss_cases if case.detector_action == detector.ACTION_ACCEPT)

    kills = []
    if accepted_rate >= prereg.KILL_ACCEPTED_FALSE_CLOSURE_RATE:
        kills.append("KILL7-1")
    if baseline_improvement < prereg.MIN_BASELINE_IMPROVEMENT:
        kills.append("KILL7-2")
    if loss_escape_accept_count > prereg.MAX_LOSS_ESCAPE_ACCEPTS:
        kills.append("KILL7-3")

    predictions = []
    if accepted_rate <= prereg.MAX_ACCEPTED_FALSE_CLOSURE_RATE:
        predictions.append("P7-P1")
    if baseline_improvement >= prereg.MIN_BASELINE_IMPROVEMENT:
        predictions.append("P7-P2")
    if true_narrow_accept_rate >= prereg.MIN_TRUE_NARROW_ACCEPT_RATE:
        predictions.append("P7-P3")
    if loss_escape_accept_count == prereg.MAX_LOSS_ESCAPE_ACCEPTS:
        predictions.append("P7-P4")

    limitations = (
        "The Phase-6 trace interface is still Helly-shaped; non-Helly synthetic traces encode minimum constraint slack through the existing min_slack channel.",
        "Helly held-outs are seed variants of the Phase-3 threshold case, not distinct generated threshold levels.",
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
        kill_criteria_triggered=tuple(kills),
        predictions_supported=tuple(predictions),
        limitations=limitations,
    )


def result_payload() -> dict[str, object]:
    cases = run_cases()
    summary = summarize(cases)
    return {
        "prereg": prereg.prereg_summary(),
        "run_config": {
            "main_restarts": MAIN_RESTARTS,
            "seed_restarts": SEED_RESTARTS,
            "dimension_restarts": DIMENSION_RESTARTS,
            "results_status_in_prereg": prereg.RESULTS_STATUS,
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
    print("BoxSEL Phase 7 run:", summary["status"])
    print(
        "false-closure accepted:",
        f"{summary['detector_accepted_false_closures']}/{summary['false_closure_traps']}",
        "baseline:",
        f"{summary['baseline_accepted_false_closures']}/{summary['false_closure_traps']}",
    )
    print("kill criteria:", ", ".join(summary["kill_criteria_triggered"]) or "none")
    print("results:", DEFAULT_RESULTS_DIR / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
