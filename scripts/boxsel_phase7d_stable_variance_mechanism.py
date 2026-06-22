#!/usr/bin/env python
"""BoxSEL Phase 7d - stable false-closure vs restart-variance mechanism.

This is the mechanism receipt behind the Phase-7b bake-off.  It proves the
restart-variance baseline is structurally blind to the stable PMP failure class:
if the only observed signal is seed-low range, then a stable false closure is
accepted by definition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import boxsel_phase6_trace_detector as phase6
import boxsel_phase7b_run as run
import boxsel_phase7b_v2_detector as detector


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "results" / "boxsel" / "phase7d_stable_variance_mechanism"
OUTPUT_PATH = OUTPUT_DIR / "manifest.json"

MECHANISM_VERSION = "phase7d_stable_variance_mechanism_v0"
MECHANISM_STATUS = "MECHANISM_RECEIPT"

PRIMARY_MECHANISM_CLAIM = (
    "Restart-variance-only detection is structurally blind to stable false closure: "
    "when seed_low_range is at or below the frozen variance threshold, the baseline "
    "accepts regardless of whether the exact interval is wider. Query-pressure traces "
    "are a distinct observable and separate the stable PMP traps from pressure-noop controls."
)

BOUNDARY = (
    "This is a toy micro-SEL mechanism receipt. It proves a signal-access asymmetry on the "
    "Phase-7b fragment; it is not a real-KG, calibration, or product claim."
)

VARIANCE_OBSERVABLES = ("seed_low_range",)
PRESSURE_OBSERVABLES = ("pressure_low_shift", "optimizer_low_spread")


@dataclass(frozen=True)
class MechanismRow:
    case_id: str
    family: str
    role: str
    false_closed: bool
    seed_low_range: float
    stable_under_variance: bool
    pressure_low_shift: float
    optimizer_low_spread: float
    pressure_signal_present: bool
    baseline_action: str
    detector_action: str
    detector_flags: tuple[str, ...]

    @property
    def baseline_blind_accept(self) -> bool:
        return self.false_closed and self.stable_under_variance and self.baseline_action == detector.ACTION_ACCEPT

    @property
    def detector_separates(self) -> bool:
        return self.false_closed and self.stable_under_variance and self.pressure_signal_present and self.detector_action != detector.ACTION_ACCEPT


@dataclass(frozen=True)
class EquivalencePair:
    trap_case_id: str
    control_case_id: str
    shared_seed_low_range: float
    baseline_action: str
    trap_false_closed: bool
    control_false_closed: bool
    trap_detector_action: str
    control_detector_action: str
    trap_pressure_low_shift: float
    control_pressure_low_shift: float

    @property
    def proves_variance_non_separation(self) -> bool:
        return (
            self.trap_false_closed
            and not self.control_false_closed
            and self.trap_detector_action != self.control_detector_action
            and self.trap_pressure_low_shift > self.control_pressure_low_shift
        )


def variance_baseline_action(seed_low_range: float) -> str:
    """The locked restart_variance_only_v0 decision rule."""

    return detector.ACTION_WIDEN if seed_low_range > phase6.SEED_RANGE_THRESHOLD else detector.ACTION_ACCEPT


def variance_blindness_theorem() -> dict[str, object]:
    return {
        "baselineVersion": "restart_variance_only_v0",
        "observable": VARIANCE_OBSERVABLES,
        "threshold": phase6.SEED_RANGE_THRESHOLD,
        "decisionRule": "widen iff seed_low_range > threshold; otherwise accept",
        "stableClassPredicate": "seed_low_range <= threshold",
        "consequence": (
            "Every false-closure case satisfying the stable predicate is accepted by this "
            "baseline, because the baseline has no other observable."
        ),
    }


def _row(case: run.CaseRun) -> MechanismRow:
    stable = case.seed_low_range <= phase6.SEED_RANGE_THRESHOLD
    pressure_signal = (
        case.pressure_low_shift >= detector.PRESSURE_LOW_SHIFT_THRESHOLD
        or case.optimizer_low_spread >= detector.OPTIMIZER_LOW_SPREAD_THRESHOLD
    )
    return MechanismRow(
        case_id=case.case_id,
        family=case.family,
        role=case.role,
        false_closed=case.false_closed,
        seed_low_range=case.seed_low_range,
        stable_under_variance=stable,
        pressure_low_shift=case.pressure_low_shift,
        optimizer_low_spread=case.optimizer_low_spread,
        pressure_signal_present=pressure_signal,
        baseline_action=case.baseline_action,
        detector_action=case.detector_action,
        detector_flags=case.detector_flags,
    )


def stable_pmp_rows(cases: tuple[run.CaseRun, ...] | None = None) -> tuple[MechanismRow, ...]:
    cases = run.run_cases() if cases is None else cases
    return tuple(_row(case) for case in cases if case.family == "stable_pmp_pressure_variants")


def pressure_noop_rows(cases: tuple[run.CaseRun, ...] | None = None) -> tuple[MechanismRow, ...]:
    cases = run.run_cases() if cases is None else cases
    return tuple(_row(case) for case in cases if case.family == "pressure_noop_controls")


def equivalence_pairs(
    traps: tuple[MechanismRow, ...] | None = None,
    controls: tuple[MechanismRow, ...] | None = None,
) -> tuple[EquivalencePair, ...]:
    traps = stable_pmp_rows() if traps is None else traps
    controls = pressure_noop_rows() if controls is None else controls
    out = []
    for trap in traps:
        for control in controls:
            if trap.seed_low_range == control.seed_low_range and trap.baseline_action == control.baseline_action:
                out.append(
                    EquivalencePair(
                        trap_case_id=trap.case_id,
                        control_case_id=control.case_id,
                        shared_seed_low_range=trap.seed_low_range,
                        baseline_action=trap.baseline_action,
                        trap_false_closed=trap.false_closed,
                        control_false_closed=control.false_closed,
                        trap_detector_action=trap.detector_action,
                        control_detector_action=control.detector_action,
                        trap_pressure_low_shift=trap.pressure_low_shift,
                        control_pressure_low_shift=control.pressure_low_shift,
                    )
                )
    return tuple(out)


def mechanism_summary() -> dict[str, object]:
    cases = run.run_cases()
    traps = stable_pmp_rows(cases)
    controls = pressure_noop_rows(cases)
    pairs = equivalence_pairs(traps, controls)
    baseline_blind = tuple(row for row in traps if row.baseline_blind_accept)
    detector_separated = tuple(row for row in traps if row.detector_separates)
    pressure_noop_clear = tuple(
        row
        for row in controls
        if row.stable_under_variance
        and not row.false_closed
        and not row.pressure_signal_present
        and row.baseline_action == detector.ACTION_ACCEPT
        and row.detector_action == detector.ACTION_ACCEPT
    )
    return {
        "mechanismVersion": MECHANISM_VERSION,
        "status": MECHANISM_STATUS,
        "primaryMechanismClaim": PRIMARY_MECHANISM_CLAIM,
        "boundary": BOUNDARY,
        "theorem": variance_blindness_theorem(),
        "stablePmpTrapCount": len(traps),
        "stablePmpBaselineBlindAccepts": len(baseline_blind),
        "stablePmpDetectorSeparations": len(detector_separated),
        "pressureNoopControlCount": len(controls),
        "pressureNoopControlsClear": len(pressure_noop_clear),
        "baselineObservableEquivalencePairs": len(pairs),
        "allEquivalencePairsProveNonSeparation": all(pair.proves_variance_non_separation for pair in pairs),
        "varianceObservables": VARIANCE_OBSERVABLES,
        "pressureObservables": PRESSURE_OBSERVABLES,
        "stablePmpRows": [asdict(row) for row in traps],
        "pressureNoopRows": [asdict(row) for row in controls],
        "equivalencePairs": [asdict(pair) for pair in pairs],
    }


def write_results(path: Path = OUTPUT_PATH) -> dict[str, object]:
    data = mechanism_summary()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def main() -> int:
    data = write_results()
    print(f"BoxSEL Phase 7d mechanism: {data['status']}")
    print("stable PMP baseline-blind accepts:", f"{data['stablePmpBaselineBlindAccepts']}/{data['stablePmpTrapCount']}")
    print("stable PMP detector separations:", f"{data['stablePmpDetectorSeparations']}/{data['stablePmpTrapCount']}")
    print("baseline-equivalence pairs:", data["baselineObservableEquivalencePairs"])
    print("manifest:", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
