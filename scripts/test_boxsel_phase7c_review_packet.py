#!/usr/bin/env python
"""Frozen test for the BoxSEL Phase-7c external-review packet.

Run: python scripts/test_boxsel_phase7c_review_packet.py
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, "scripts")
import boxsel_phase7c_review_packet as packet

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("(1) packet status and claim boundary:")
data = packet.packet_payload()
check("packet is ready for review, not reviewed",
      data["packetVersion"] == "phase7c_external_review_packet_v0"
      and data["status"] == "READY_FOR_EXTERNAL_REVIEW"
      and data["reviewOutcomeStatus"] == "NOT_REVIEWED")
check("primary claim is the stable/variance mechanism, not just the bake-off",
      "structurally blind" in data["primaryReviewClaim"]
      and "seed_low_range" in data["primaryReviewClaim"]
      and "pressure-response" in data["primaryReviewClaim"])
check("boundary blocks real-KG, calibration, product, and exact-pressure overclaims",
      "no real-KG claim" in data["boundary"]
      and "no calibration guarantee" in data["boundary"]
      and "no Ask Sundog product behavior claim" in data["boundary"]
      and "no claim that query pressure is exact inference" in data["boundary"])
check("non-claims preserve Phase-7 failure and no public/product claim",
      any("failed Phase-7" in item for item in data["nonClaims"])
      and any("real KGs" in item for item in data["nonClaims"])
      and any("Ask Sundog" in item for item in data["nonClaims"]))

print("(2) Phase-7 failure and Phase-7b result are both preserved:")
phase7 = data["phase7Failure"]
phase7b = data["phase7bResult"]
check("Phase-7 failed in packet",
      phase7["status"] == "FAIL_PREREG_GATE"
      and phase7["acceptedFalseClosures"] == 4
      and phase7["baselineAcceptedFalseClosures"] == 4
      and set(phase7["killCriteriaTriggered"]) == {"KILL7-1", "KILL7-2"})
check("Phase-7b pass metrics match the locked run",
      phase7b["status"] == "PASS_PREREG_GATE"
      and phase7b["caseCount"] == 28
      and phase7b["falseClosureTraps"] == 16
      and phase7b["detectorAcceptedFalseClosures"] == 0
      and phase7b["baselineAcceptedFalseClosures"] == 16)
check("Phase-7b controls and pressure contrast are carried through",
      phase7b["trueNarrowAcceptRate"] == 1.0
      and phase7b["lossEscapeAcceptCount"] == 0
      and phase7b["pressureWarningRateOnStablePmp"] == 1.0
      and phase7b["baselinePressureWarningRateOnStablePmp"] == 0.0)
check("family and role counts match the corpus",
      phase7b["roleCounts"] == {"acceptance_control": 9, "false_closure_trap": 16, "loss_control": 3}
      and phase7b["familyCounts"]["stable_pmp_pressure_variants"] == 8
      and phase7b["familyCounts"]["support_floor_variants"] == 4)

print("(3) leakage and boundary audit is explicit:")
audit = data["leakageAndBoundaryAudit"]
check("audit says both historical facts are preserved",
      audit["phase7FailurePreserved"] and audit["phase7bPassPreserved"])
check("prereg remains result-free and detector remains frozen",
      audit["preregResultFree"] and audit["detectorFrozen"])
check("feature list is oracle-free and required v2 features are present",
      audit["featureListOracleFree"]
      and not audit["forbiddenFeatureHits"]
      and audit["requiredV2FeaturesPresent"])
check("held-out hygiene checks pass",
      audit["reservedSeedsClean"] and audit["phase7SeenCasesExcluded"] and audit["caseIdsUnique"])
check("detector source does not import evaluator/oracle/corpus/run modules",
      audit["detectorSourceAudit"]["passes"]
      and not audit["detectorSourceAudit"]["forbiddenImportHits"])

print("(4) mechanism receipt is included in the review packet:")
mechanism = data["phase7dMechanism"]
check("Phase-7d mechanism summary is present",
      mechanism["mechanismVersion"] == "phase7d_stable_variance_mechanism_v0"
      and mechanism["status"] == "MECHANISM_RECEIPT")
check("stable PMP rows instantiate variance blindness",
      mechanism["stablePmpTrapCount"] == 8
      and mechanism["stablePmpBaselineBlindAccepts"] == 8
      and mechanism["stablePmpDetectorSeparations"] == 8)
check("pressure-noop controls and equivalence pairs are carried through",
      mechanism["pressureNoopControlCount"] == 3
      and mechanism["pressureNoopControlsClear"] == 3
      and mechanism["baselineObservableEquivalencePairs"] == 24
      and mechanism["allEquivalencePairsProveNonSeparation"])

print("(5) recovery receipt is included in the review packet:")
recovery = data["phase7eRecovery"]
check("Phase-7e recovery summary is present",
      recovery["recoveryVersion"] == "phase7e_oracle_free_recovery_v0"
      and recovery["status"] == "RECOVERY_RECEIPT")
check("recovery summary preserves oracle-free input and validation split",
      recovery["traceInputOracleFree"]
      and recovery["recoveryRule"]["usesExactOracle"] is False
      and recovery["recoveryRule"]["usesClosedFormForRecovery"] is False
      and recovery["recoveryRule"]["usesClosedFormForValidation"] is True)
check("recovered endpoint matches the closed-form payload",
      recovery["recoveredEndpointPayload"]["repr"] == recovery["closedFormPayload"]["repr"]
      and recovery["recoveredEndpointPayload"]["repr"] == "(9/32 + 1/32*sqrt17)")

print("(6) active-set discovery receipt is included in the review packet:")
discovery = data["phase7fDiscovery"]
check("Phase-7f discovery summary is present",
      discovery["discoveryVersion"] == "phase7f_active_set_discovery_v0"
      and discovery["status"] == "DISCOVERY_RECEIPT")
check("discovery summary preserves raw oracle-free input and no active-label input",
      discovery["rawTraceOracleFree"]
      and discovery["discoveryRule"]["input"] == "raw box intervals"
      and discovery["discoveryRule"]["usesActiveLabelsAsInput"] is False
      and discovery["discoveryRule"]["usesClosedFormForDiscovery"] is False)
check("discovered active set and equation feed the recovered endpoint",
      set(discovery["discovered"]["active_pairs"]) == {"AC", "BC"}
      and discovery["discovered"]["slack_pairs"] == ("AB",)
      and discovery["discovered"]["active_equation"] == (4, -9, 4)
      and discovery["discovered"]["recovered_endpoint"] == "(9/32 + 1/32*sqrt17)")
check("negative control is not applicable to Phase-7e recovery",
      discovery["negativeControl"]["active_pairs"] == ("AC",)
      and discovery["negativeControl"]["active_equation"] is None
      and discovery["negativeControl"]["applicable"] is False)

print("(7) review questions are concrete breakpoints:")
questions = data["reviewQuestions"]
categories = {item["category"] for item in questions}
check("eight named review questions are present",
      [item["id"] for item in questions] == [f"P7C-Q{i}" for i in range(1, 9)])
check("question categories cover the actual weak points",
      {"leakage", "heldout", "baseline", "pressure", "semantic_alignment", "controls", "scope",
       "recovery_conditionality"}.issubset(categories))
check("baseline question names the stable/variance dichotomy",
      "stable/variance dichotomy" in questions[2]["question"]
      and "seed_low_range" in questions[2]["question"])
check("recovery-conditionality question guards 7e/7f against blind-recovery overclaim",
      questions[7]["category"] == "recovery_conditionality"
      and "BLIND recovery" in questions[7]["question"]
      and "search gap is smuggled shut" in questions[7]["breaksClaimIf"])
check("each question has a break condition",
      all(item["breaksClaimIf"] for item in questions))
outcomes = data["reviewOutcomes"]
check("review outcomes include pass, follow-up gate, and redesign paths",
      {item["name"] for item in outcomes}
      == {"toy_claim_review_pass", "followup_gate_required", "claim_withdrawn_or_redesigned"})

print("(8) artifacts and note are packetized:")
hashes = data["artifactHashes"]
by_path = {item["path"]: item for item in hashes}
required_paths = {
    "docs/boxsel/PHASE7C_EXTERNAL_REVIEW_PACKET.md",
    "docs/boxsel/PHASE7D_STABLE_VARIANCE_MECHANISM.md",
    "docs/boxsel/PHASE7E_ORACLE_FREE_RECOVERY.md",
    "docs/boxsel/PHASE7F_ACTIVE_SET_DISCOVERY.md",
    "scripts/boxsel_phase7c_review_packet.py",
    "scripts/boxsel_phase7d_stable_variance_mechanism.py",
    "scripts/boxsel_phase7e_oracle_free_recovery.py",
    "scripts/boxsel_phase7f_active_set_discovery.py",
    "scripts/test_boxsel_phase7c_review_packet.py",
    "scripts/test_boxsel_phase7d_stable_variance_mechanism.py",
    "scripts/test_boxsel_phase7e_oracle_free_recovery.py",
    "scripts/test_boxsel_phase7f_active_set_discovery.py",
    "results/boxsel/phase7d_stable_variance_mechanism/manifest.json",
    "results/boxsel/phase7e_oracle_free_recovery/manifest.json",
    "results/boxsel/phase7f_active_set_discovery/manifest.json",
    "results/boxsel/phase7b_false_closure_run/manifest.json",
    "boxsel.html",
}
check("required review artifacts are hashed",
      required_paths.issubset(by_path.keys()) and all(by_path[path]["exists"] for path in required_paths))
check("hash rows have SHA-256 digests and sizes",
      all(item["exists"] and len(item["sha256"]) == 64 and item["sizeBytes"] > 0 for item in hashes))
written = packet.write_packet()
manifest_path = ROOT / "results" / "boxsel" / "phase7c_external_review_packet" / "manifest.json"
loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
check("manifest writes and round-trips",
      loaded["packetVersion"] == written["packetVersion"]
      and loaded["phase7bResult"]["detectorAcceptedFalseClosures"] == 0
      and loaded["phase7dMechanism"]["baselineObservableEquivalencePairs"] == 24
      and loaded["phase7eRecovery"]["recoveredEndpointPayload"]["repr"] == "(9/32 + 1/32*sqrt17)"
      and loaded["phase7fDiscovery"]["discovered"]["active_equation"] == [4, -9, 4])
note = (ROOT / "docs" / "boxsel" / "PHASE7C_EXTERNAL_REVIEW_PACKET.md").read_text(encoding="utf-8")
check("review note carries status, questions, and exact metrics",
      "READY_FOR_EXTERNAL_REVIEW" in note
      and "P7C-Q1" in note
      and "0 / 16" in note
      and "16 / 16" in note
      and "stable false closure" in note
      and "Phase 7 failed" in note
      and "active-set discovery" in note)

print(f"\n{'ALL PASS -- Phase-7c external-review packet is review-ready and bounded' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
