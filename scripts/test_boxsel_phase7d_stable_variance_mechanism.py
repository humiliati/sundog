#!/usr/bin/env python
"""Frozen test for the BoxSEL Phase-7d stable/variance mechanism receipt.

Run: python scripts/test_boxsel_phase7d_stable_variance_mechanism.py
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, "scripts")
import boxsel_phase6_trace_detector as phase6
import boxsel_phase7d_stable_variance_mechanism as mech
import boxsel_phase7b_v2_detector as detector

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


print("(1) the mechanism claim is structural, not a bigger benchmark:")
summary = mech.mechanism_summary()
check("version and status are explicit",
      summary["mechanismVersion"] == "phase7d_stable_variance_mechanism_v0"
      and summary["status"] == "MECHANISM_RECEIPT")
check("claim names structural blindness and query pressure",
      "structurally blind" in summary["primaryMechanismClaim"]
      and "Query-pressure" in summary["primaryMechanismClaim"])
check("boundary remains toy micro-SEL only",
      "toy micro-SEL" in summary["boundary"]
      and "not a real-KG" in summary["boundary"])

print("(2) restart_variance_only_v0 has no stable false-closure channel:")
theorem = summary["theorem"]
check("baseline observable is only seed_low_range",
      theorem["observable"] == ("seed_low_range",)
      and summary["varianceObservables"] == ("seed_low_range",))
check("threshold matches the locked Phase-6/7 baseline threshold",
      theorem["threshold"] == phase6.SEED_RANGE_THRESHOLD == 0.02)
check("baseline accepts every stable seed range by rule",
      mech.variance_baseline_action(0.0) == detector.ACTION_ACCEPT
      and mech.variance_baseline_action(phase6.SEED_RANGE_THRESHOLD) == detector.ACTION_ACCEPT)
check("baseline widens only above threshold",
      mech.variance_baseline_action(phase6.SEED_RANGE_THRESHOLD + 1e-12) == detector.ACTION_WIDEN)
check("theorem consequence states acceptance for stable false closure",
      "Every false-closure case" in theorem["consequence"]
      and "accepted" in theorem["consequence"])

print("(3) stable PMP traps instantiate the blind class:")
stable = summary["stablePmpRows"]
check("there are eight stable PMP traps",
      summary["stablePmpTrapCount"] == len(stable) == 8)
check("all stable PMP rows are false-closed and variance-stable",
      all(row["false_closed"] and row["stable_under_variance"] for row in stable))
check("all stable PMP rows have zero seed range",
      all(row["seed_low_range"] == 0.0 for row in stable))
check("restart variance accepts all stable PMP false closures",
      summary["stablePmpBaselineBlindAccepts"] == 8
      and all(row["baseline_action"] == detector.ACTION_ACCEPT for row in stable))
check("pressure/optimizer signals are present on every stable PMP trap",
      all(row["pressure_signal_present"] for row in stable)
      and all({"pressure_low_shift", "optimizer_low_spread"}.issubset(set(row["detector_flags"])) for row in stable))
check("v2 detector separates all stable PMP traps",
      summary["stablePmpDetectorSeparations"] == 8
      and all(row["detector_action"] == detector.ACTION_ABSTAIN for row in stable))

print("(4) pressure-noop controls share the variance observable but not the pressure signal:")
controls = summary["pressureNoopRows"]
check("there are three pressure-noop controls",
      summary["pressureNoopControlCount"] == len(controls) == 3)
check("pressure-noop controls are stable, accepted, and not false-closed",
      summary["pressureNoopControlsClear"] == 3
      and all(row["stable_under_variance"] and not row["false_closed"] for row in controls)
      and all(row["baseline_action"] == detector.ACTION_ACCEPT for row in controls)
      and all(row["detector_action"] == detector.ACTION_ACCEPT for row in controls))
check("pressure-noop controls have no pressure signal",
      all(not row["pressure_signal_present"] for row in controls)
      and all(row["pressure_low_shift"] == 0.0 and row["optimizer_low_spread"] == 0.0 for row in controls))

print("(5) equivalence pairs prove non-separation for the variance baseline:")
pairs = summary["equivalencePairs"]
check("all stable traps pair with all pressure-noop controls",
      summary["baselineObservableEquivalencePairs"] == len(pairs) == 8 * 3)
check("each pair has identical baseline observable and action",
      all(pair["shared_seed_low_range"] == 0.0 and pair["baseline_action"] == detector.ACTION_ACCEPT for pair in pairs))
check("each pair has opposite false-closure labels",
      all(pair["trap_false_closed"] and not pair["control_false_closed"] for pair in pairs))
check("pressure signal, not variance, separates each pair",
      summary["allEquivalencePairsProveNonSeparation"]
      and all(pair["trap_pressure_low_shift"] > pair["control_pressure_low_shift"] for pair in pairs)
      and all(pair["trap_detector_action"] != pair["control_detector_action"] for pair in pairs))
check("pressure observables are named separately",
      summary["pressureObservables"] == ("pressure_low_shift", "optimizer_low_spread"))

print("(6) manifest and note round-trip:")
written = mech.write_results()
manifest_path = ROOT / "results" / "boxsel" / "phase7d_stable_variance_mechanism" / "manifest.json"
loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
check("manifest writes the mechanism summary",
      loaded["mechanismVersion"] == written["mechanismVersion"]
      and loaded["stablePmpBaselineBlindAccepts"] == 8
      and loaded["baselineObservableEquivalencePairs"] == 24)
note = ROOT / "docs" / "boxsel" / "PHASE7D_STABLE_VARIANCE_MECHANISM.md"
check("Phase-7d note exists and names the structural dichotomy",
      note.exists()
      and "stable false closure" in note.read_text(encoding="utf-8")
      and "variance" in note.read_text(encoding="utf-8"))

print(f"\n{'ALL PASS -- Phase-7d mechanism receipt proves stable false-closure variance blindness' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
