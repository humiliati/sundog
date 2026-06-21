#!/usr/bin/env python
"""BoxSEL Phase 8 - build static workbench data from the Phase-7b receipt."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import boxsel_phase7b_run as phase7b
import boxsel_phase7b_v2_detector as detector


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "public" / "data" / "boxsel-phase8-workbench.json"
WORKBENCH_DATA_VERSION = "phase8_boxsel_workbench_v0"
QKKT = (9.0 + sqrt(17.0)) / 32.0


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _interval(lower: float, upper: float) -> dict[str, float]:
    return {"lower": _round(lower), "upper": _round(upper)}


def _pressure_interval(case: phase7b.CaseRun) -> dict[str, float] | None:
    if case.pressure_low_shift <= 0.0:
        return None
    return _interval(case.sample_lower - case.pressure_low_shift, case.sample_upper)


def _box_interval(case: phase7b.CaseRun) -> dict[str, float] | None:
    if case.family != "helly_threshold_variants_v2":
        return None
    return _interval(QKKT, case.exact_upper)


def _dimension_profile(case: phase7b.CaseRun) -> dict[str, object] | None:
    if case.family != "helly_threshold_variants_v2":
        return None
    return {
        "dimensionControlled": True,
        "dim1": _interval(0.5, case.exact_upper),
        "dim2Plus": _interval(QKKT, case.exact_upper),
        "exact": _interval(case.exact_lower, case.exact_upper),
    }


def _case_record(case: phase7b.CaseRun) -> dict[str, object]:
    pressure = _pressure_interval(case)
    box = _box_interval(case)
    middle_kind = "box_attainable" if box else ("pressure_probe" if pressure else "none")
    detector_flags = list(case.detector_flags)
    return {
        "caseId": case.case_id,
        "family": case.family,
        "role": case.role,
        "seed": case.seed,
        "decision": case.detector_action,
        "detectorFlags": detector_flags,
        "baselineAction": case.baseline_action,
        "baselineAcceptedFalseClosure": case.baseline_accepted_false_closure,
        "falseClosed": case.false_closed,
        "gap": _round(case.lower_exact_widening),
        "sample": _interval(case.sample_lower, case.sample_upper),
        "exact": _interval(case.exact_lower, case.exact_upper),
        "box": box,
        "pressure": pressure,
        "middleKind": middle_kind,
        "dimensionProfile": _dimension_profile(case),
        "trace": {
            "maxLoss": case.max_loss,
            "pressureLowShift": _round(case.pressure_low_shift),
            "optimizerLowSpread": _round(case.optimizer_low_spread),
            "supportFloor": _round(case.support_floor),
            "seedLowRange": _round(case.seed_low_range),
            "dimensionLowSpread": _round(case.dimension_low_spread),
            "earlyLowerDrop": _round(case.early_lower_drop),
            "lateLowerDrop": _round(case.late_lower_drop),
        },
        "note": case.note,
    }


def build_data() -> dict[str, object]:
    cases = phase7b.run_cases()
    summary = phase7b.summarize(cases)
    return {
        "schemaVersion": 1,
        "workbenchDataVersion": WORKBENCH_DATA_VERSION,
        "sourceManifest": "results/boxsel/phase7b_false_closure_run/manifest.json",
        "sourceResultNote": "docs/boxsel/PHASE7B_FALSE_CLOSURE_RUN.md",
        "phase8Note": "docs/boxsel/PHASE8_WORKBENCH_START.md",
        "boundary": (
            "Toy role-free micro-SEL workbench. Not a real-KG result, not a calibration guarantee, "
            "and not an Ask Sundog product claim."
        ),
        "summary": {
            "status": summary.status,
            "caseCount": summary.case_count,
            "falseClosureTraps": summary.false_closure_traps,
            "detectorAcceptedFalseClosures": summary.detector_accepted_false_closures,
            "acceptedFalseClosureRate": summary.accepted_false_closure_rate,
            "baselineAcceptedFalseClosures": summary.baseline_accepted_false_closures,
            "baselineAcceptedFalseClosureRate": summary.baseline_accepted_false_closure_rate,
            "baselineImprovement": summary.baseline_improvement,
            "trueNarrowAcceptRate": summary.true_narrow_accept_rate,
            "lossEscapeAcceptCount": summary.loss_escape_accept_count,
            "pressureWarningRateOnStablePmp": summary.pressure_warning_rate_on_stable_pmp,
            "baselinePressureWarningRateOnStablePmp": summary.baseline_pressure_warning_rate_on_stable_pmp,
            "predictionsSupported": list(summary.predictions_supported),
            "killCriteriaTriggered": list(summary.kill_criteria_triggered),
        },
        "thresholds": detector.frozen_thresholds(),
        "intervalReference": {
            "hellyBoxLowerDim1": 0.5,
            "hellyBoxLowerDim2Plus": _round(QKKT),
            "hellyExact": _interval(0.0, 1.0),
        },
        "cases": [_case_record(case) for case in cases],
    }


def write_data(path: Path = OUTPUT_PATH) -> dict[str, object]:
    data = build_data()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def main() -> int:
    data = write_data()
    print(f"BoxSEL Phase 8 workbench data written: {OUTPUT_PATH}")
    print(
        "cases:",
        data["summary"]["caseCount"],
        "status:",
        data["summary"]["status"],
        "accepted false closures:",
        data["summary"]["detectorAcceptedFalseClosures"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
