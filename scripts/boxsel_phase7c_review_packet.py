#!/usr/bin/env python
"""BoxSEL Phase 7c - external-review packet for the Phase-7b toy result.

Phase 7c does not add a new detector result. It packages the Phase-7 failure, the
locked Phase-7b pass, and the evidence boundary into a reviewable artifact.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import boxsel_phase6b_trace_schema as schema
import boxsel_phase7_run as phase7
import boxsel_phase7b_prereg as prereg
import boxsel_phase7b_run as run
import boxsel_phase7b_v2_detector as detector
import boxsel_phase7d_stable_variance_mechanism as phase7d
import boxsel_phase7e_oracle_free_recovery as phase7e
import boxsel_phase7f_active_set_discovery as phase7f


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "results" / "boxsel" / "phase7c_external_review_packet"
OUTPUT_PATH = OUTPUT_DIR / "manifest.json"

REVIEW_PACKET_VERSION = "phase7c_external_review_packet_v0"
REVIEW_PACKET_STATUS = "READY_FOR_EXTERNAL_REVIEW"
REVIEW_OUTCOME_STATUS = "NOT_REVIEWED"

PRIMARY_REVIEW_CLAIM = (
    "On the locked tiny role-free micro-SEL fragment, restart-variance-only detection is "
    "structurally blind to stable false closure because it observes only seed_low_range; "
    "the stable PMP traps and pressure-noop controls share seed_low_range=0 while "
    "pressure-response traces separate the labels."
)

BOUNDARY = (
    "Toy micro-SEL only: no real-KG claim, no calibration guarantee, no Ask Sundog "
    "product behavior claim, and no claim that query pressure is exact inference."
)

NON_CLAIMS = (
    "The packet does not claim the detector works on real KGs.",
    "The packet does not claim calibrated uncertainty or conformal coverage.",
    "The packet does not claim Ask Sundog abstains correctly in production.",
    "The packet does not claim query-pressure traces are exact inference.",
    "The packet does not erase the failed Phase-7 preregistered run.",
)

REVIEW_ARTIFACTS = (
    "docs/SUNDOG_V_BOXSEL.md",
    "docs/boxsel/BOXSEL_LITPASS_MEMO.md",
    "docs/boxsel/PHASE7_FALSE_CLOSURE_PREREG.md",
    "docs/boxsel/PHASE7_FALSE_CLOSURE_RUN.md",
    "docs/boxsel/PHASE7_FAILURE_ANALYSIS_AND_V2_SPEC.md",
    "docs/boxsel/PHASE7B_FALSE_CLOSURE_PREREG_START.md",
    "docs/boxsel/PHASE7B_CORPUS_EVALUATOR_START.md",
    "docs/boxsel/PHASE7B_V2_FREEZE_LOCK.md",
    "docs/boxsel/PHASE7B_FALSE_CLOSURE_RUN.md",
    "docs/boxsel/PHASE7C_EXTERNAL_REVIEW_PACKET.md",
    "docs/boxsel/PHASE7D_STABLE_VARIANCE_MECHANISM.md",
    "docs/boxsel/PHASE7E_ORACLE_FREE_RECOVERY.md",
    "docs/boxsel/PHASE7F_ACTIVE_SET_DISCOVERY.md",
    "docs/boxsel/PHASE8_WORKBENCH_START.md",
    "scripts/boxsel_phase6b_trace_schema.py",
    "scripts/boxsel_phase7_run.py",
    "scripts/boxsel_phase7b_prereg.py",
    "scripts/boxsel_phase7b_v2_detector.py",
    "scripts/boxsel_phase7b_corpus.py",
    "scripts/boxsel_phase7b_evaluator.py",
    "scripts/boxsel_phase7b_run.py",
    "scripts/boxsel_phase7c_review_packet.py",
    "scripts/boxsel_phase7d_stable_variance_mechanism.py",
    "scripts/boxsel_phase7e_oracle_free_recovery.py",
    "scripts/boxsel_phase7f_active_set_discovery.py",
    "scripts/test_boxsel_phase7b_run.py",
    "scripts/test_boxsel_phase7c_review_packet.py",
    "scripts/test_boxsel_phase7d_stable_variance_mechanism.py",
    "scripts/test_boxsel_phase7e_oracle_free_recovery.py",
    "scripts/test_boxsel_phase7f_active_set_discovery.py",
    "results/boxsel/phase7_false_closure_run/manifest.json",
    "results/boxsel/phase7b_false_closure_run/manifest.json",
    "results/boxsel/phase7d_stable_variance_mechanism/manifest.json",
    "results/boxsel/phase7e_oracle_free_recovery/manifest.json",
    "results/boxsel/phase7f_active_set_discovery/manifest.json",
    "boxsel.html",
    "public/data/boxsel-phase8-workbench.json",
)

FORBIDDEN_DETECTOR_IMPORT_TOKENS = (
    "boxsel_exact_oracle",
    "boxsel_phase7b_evaluator",
    "boxsel_phase7b_corpus",
    "boxsel_phase7b_run",
    "phase4",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_hashes(paths: tuple[str, ...] = REVIEW_ARTIFACTS) -> tuple[dict[str, object], ...]:
    rows = []
    for rel in paths:
        path = REPO_ROOT / rel
        rows.append(
            {
                "path": rel,
                "exists": path.exists(),
                "sizeBytes": path.stat().st_size if path.exists() else None,
                "sha256": _sha256(path) if path.exists() else None,
            }
        )
    return tuple(rows)


def review_questions() -> tuple[dict[str, str], ...]:
    return (
        {
            "id": "P7C-Q1",
            "category": "leakage",
            "question": "Does any detector feature, threshold, or action read exact endpoints, oracle labels, Phase-4 closed forms, or evaluator-only fields?",
            "breaksClaimIf": "A detector decision depends on evaluator-only truth rather than GeneralTrace observations.",
        },
        {
            "id": "P7C-Q2",
            "category": "heldout",
            "question": "Are Phase-7 diagnostic rows and seeds excluded from Phase-7b held-out validation?",
            "breaksClaimIf": "A Phase-7 seen case ID or training/diagnostic seed is reused as Phase-7b held-out evidence.",
        },
        {
            "id": "P7C-Q3",
            "category": "baseline",
            "question": "Is the Phase-7d stable/variance dichotomy correct: does restart_variance_only_v0 truly observe only seed_low_range, making stable false closure a blind spot by construction?",
            "breaksClaimIf": "The baseline has another observable that can separate the stable PMP traps from pressure-noop controls.",
        },
        {
            "id": "P7C-Q4",
            "category": "pressure",
            "question": "Is query pressure a legitimate observable trace, or is it too close to extremal inference for the intended abstention claim?",
            "breaksClaimIf": "Pressure traces smuggle in the exact endpoint or require access equivalent to the evaluator.",
        },
        {
            "id": "P7C-Q5",
            "category": "semantic_alignment",
            "question": "Do the exact finite-counting labels, PMP cases, Helly cases, and support-floor cases align with the BoxSEL volume semantics they are meant to stress?",
            "breaksClaimIf": "A case is labeled false-closed only because the finite oracle and geometric-volume semantics are mismatched.",
        },
        {
            "id": "P7C-Q6",
            "category": "controls",
            "question": "Are the true-narrow, pressure-noop, and loss controls enough to rule out an always-abstain or pressure-hypersensitive detector?",
            "breaksClaimIf": "A trivial abstention rule or pressure-only threshold can match the result without respecting controls.",
        },
        {
            "id": "P7C-Q7",
            "category": "scope",
            "question": "Does every outward sentence preserve the toy micro-SEL boundary and the failed Phase-7 history?",
            "breaksClaimIf": "The packet implies real-KG transfer, calibration, product behavior, or a retroactive Phase-7 pass.",
        },
        {
            "id": "P7C-Q8",
            "category": "recovery_conditionality",
            "question": "Do Phases 7e/7f anywhere imply BLIND recovery? Confirm both presuppose a held config (7e: a named active set; 7f: a given config's raw intervals), and that neither claims ordinary restarts find it.",
            "breaksClaimIf": "Phase 7e/7f are presented as recovering the endpoint without already holding the optimal config -- i.e. the open search gap is smuggled shut.",
        },
    )


def review_outcomes() -> tuple[dict[str, str], ...]:
    return (
        {
            "id": "P7C-O1",
            "name": "toy_claim_review_pass",
            "meaning": "Reviewer accepts only the bounded toy micro-SEL claim and identifies no leakage or semantic-label break.",
        },
        {
            "id": "P7C-O2",
            "name": "followup_gate_required",
            "meaning": "Reviewer accepts the packet shape but requires a stricter baseline, recovery test, more controls, or a larger registered corpus before any stronger claim.",
        },
        {
            "id": "P7C-O3",
            "name": "claim_withdrawn_or_redesigned",
            "meaning": "Reviewer finds leakage, semantic mismatch, or overclaim sufficient to withdraw the Phase-7b detector claim.",
        },
    )


def _role_counts(cases: tuple[run.CaseRun, ...]) -> dict[str, int]:
    return dict(sorted(Counter(case.role for case in cases).items()))


def _family_counts(cases: tuple[run.CaseRun, ...]) -> dict[str, int]:
    return dict(sorted(Counter(case.family for case in cases).items()))


def _detector_source_audit() -> dict[str, object]:
    source = (REPO_ROOT / "scripts" / "boxsel_phase7b_v2_detector.py").read_text(encoding="utf-8")
    forbidden_hits = tuple(token for token in FORBIDDEN_DETECTOR_IMPORT_TOKENS if token in source)
    return {
        "forbiddenImportTokens": FORBIDDEN_DETECTOR_IMPORT_TOKENS,
        "forbiddenImportHits": forbidden_hits,
        "passes": forbidden_hits == (),
    }


def leakage_and_boundary_audit(cases: tuple[run.CaseRun, ...], phase7_summary, phase7b_summary) -> dict[str, object]:
    case_ids = tuple(case.case_id for case in cases)
    features = schema.feature_names()
    forbidden_feature_hits = tuple(
        name
        for name in features
        if any(token in name.lower() for token in schema.FORBIDDEN_FEATURE_TOKENS)
    )
    return {
        "phase7FailurePreserved": (
            phase7_summary.status == phase7.RUN_STATUS_FAIL
            and phase7_summary.detector_accepted_false_closures == 4
            and phase7_summary.baseline_accepted_false_closures == 4
        ),
        "phase7bPassPreserved": (
            phase7b_summary.status == run.RUN_STATUS_PASS
            and phase7b_summary.detector_accepted_false_closures == 0
            and phase7b_summary.baseline_accepted_false_closures == 16
        ),
        "preregResultFree": prereg.RESULTS_STATUS == "NOT_RUN" and prereg.RESULT_ROWS == (),
        "detectorFrozen": detector.DETECTOR_STATUS == "FROZEN" and detector.THRESHOLD_STATUS == "FROZEN",
        "featureListOracleFree": prereg.feature_list_is_oracle_free() and forbidden_feature_hits == (),
        "forbiddenFeatureHits": forbidden_feature_hits,
        "requiredV2FeaturesPresent": prereg.required_v2_features_present(),
        "reservedSeedsClean": prereg.reserved_seeds_are_clean(),
        "phase7SeenCasesExcluded": prereg.seen_cases_are_excluded(case_ids),
        "caseIdsUnique": len(case_ids) == len(set(case_ids)),
        "detectorSourceAudit": _detector_source_audit(),
    }


def packet_payload() -> dict[str, object]:
    phase7_cases = phase7.run_cases()
    phase7_summary = phase7.summarize(phase7_cases)
    phase7b_cases = run.run_cases()
    phase7b_summary = run.summarize(phase7b_cases)

    return {
        "schemaVersion": 1,
        "packetVersion": REVIEW_PACKET_VERSION,
        "status": REVIEW_PACKET_STATUS,
        "reviewOutcomeStatus": REVIEW_OUTCOME_STATUS,
        "primaryReviewClaim": PRIMARY_REVIEW_CLAIM,
        "boundary": BOUNDARY,
        "nonClaims": NON_CLAIMS,
        "phase7Failure": {
            "status": phase7_summary.status,
            "acceptedFalseClosures": phase7_summary.detector_accepted_false_closures,
            "falseClosureTraps": phase7_summary.false_closure_traps,
            "baselineAcceptedFalseClosures": phase7_summary.baseline_accepted_false_closures,
            "killCriteriaTriggered": phase7_summary.kill_criteria_triggered,
        },
        "phase7bResult": {
            "status": phase7b_summary.status,
            "caseCount": phase7b_summary.case_count,
            "roleCounts": _role_counts(phase7b_cases),
            "familyCounts": _family_counts(phase7b_cases),
            "falseClosureTraps": phase7b_summary.false_closure_traps,
            "detectorAcceptedFalseClosures": phase7b_summary.detector_accepted_false_closures,
            "baselineAcceptedFalseClosures": phase7b_summary.baseline_accepted_false_closures,
            "acceptedFalseClosureRate": phase7b_summary.accepted_false_closure_rate,
            "baselineAcceptedFalseClosureRate": phase7b_summary.baseline_accepted_false_closure_rate,
            "baselineImprovement": phase7b_summary.baseline_improvement,
            "trueNarrowAcceptRate": phase7b_summary.true_narrow_accept_rate,
            "lossEscapeAcceptCount": phase7b_summary.loss_escape_accept_count,
            "pressureWarningRateOnStablePmp": phase7b_summary.pressure_warning_rate_on_stable_pmp,
            "baselinePressureWarningRateOnStablePmp": phase7b_summary.baseline_pressure_warning_rate_on_stable_pmp,
            "killCriteriaTriggered": phase7b_summary.kill_criteria_triggered,
            "predictionsSupported": phase7b_summary.predictions_supported,
        },
        "phase7dMechanism": {
            key: value
            for key, value in phase7d.mechanism_summary().items()
            if key
            in {
                "mechanismVersion",
                "status",
                "stablePmpTrapCount",
                "stablePmpBaselineBlindAccepts",
                "stablePmpDetectorSeparations",
                "pressureNoopControlCount",
                "pressureNoopControlsClear",
                "baselineObservableEquivalencePairs",
                "allEquivalencePairsProveNonSeparation",
                "varianceObservables",
                "pressureObservables",
            }
        },
        "phase7eRecovery": {
            key: value
            for key, value in phase7e.recovery_summary().items()
            if key
            in {
                "recoveryVersion",
                "status",
                "primaryRecoveryClaim",
                "boundary",
                "traceInputOracleFree",
                "recoveryRule",
                "recoveredEndpointPayload",
                "closedFormPayload",
            }
        },
        "phase7fDiscovery": {
            key: value
            for key, value in phase7f.discovery_summary().items()
            if key
            in {
                "discoveryVersion",
                "status",
                "primaryDiscoveryClaim",
                "boundary",
                "rawTraceOracleFree",
                "discoveryRule",
                "discovered",
                "negativeControl",
            }
        },
        "frozenDetector": detector.detector_summary(),
        "leakageAndBoundaryAudit": leakage_and_boundary_audit(phase7b_cases, phase7_summary, phase7b_summary),
        "reviewQuestions": review_questions(),
        "reviewOutcomes": review_outcomes(),
        "artifactHashes": artifact_hashes(),
    }


def write_packet(path: Path = OUTPUT_PATH) -> dict[str, object]:
    data = packet_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def main() -> int:
    data = write_packet()
    result = data["phase7bResult"]
    print(f"BoxSEL Phase 7c review packet: {data['status']}")
    print("claim:", data["primaryReviewClaim"])
    print(
        "phase7b accepted false closures:",
        f"{result['detectorAcceptedFalseClosures']}/{result['falseClosureTraps']}",
        "baseline:",
        f"{result['baselineAcceptedFalseClosures']}/{result['falseClosureTraps']}",
    )
    print("review questions:", len(data["reviewQuestions"]))
    print("manifest:", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
