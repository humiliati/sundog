#!/usr/bin/env python
"""Frozen test for the BoxSEL Phase-7e oracle-free recovery receipt.

Run: python scripts/test_boxsel_phase7e_oracle_free_recovery.py
"""

import json
from fractions import Fraction
from pathlib import Path
import sys

sys.path.insert(0, "scripts")
import boxsel_kkt_exact as kkt
import boxsel_phase4k_dimension_compression as closed
import boxsel_phase7e_oracle_free_recovery as rec

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


print("(1) recovery boundary and claim:")
summary = rec.recovery_summary()
check("version and status are explicit",
      summary["recoveryVersion"] == "phase7e_oracle_free_recovery_v0"
      and summary["status"] == "RECOVERY_RECEIPT")
check("claim names oracle-free active-set recovery and closed-form validation",
      "oracle-free active-set pressure trace" in summary["primaryRecoveryClaim"]
      and "4x^2 - 9x + 4" in summary["primaryRecoveryClaim"]
      and "(9+sqrt17)/32" in summary["primaryRecoveryClaim"])
check("boundary blocks broad claims",
      "Toy micro-SEL" in summary["boundary"]
      and "not a general optimizer" in summary["boundary"]
      and "not a real-KG" in summary["boundary"]
      and "not an Ask Sundog product claim" in summary["boundary"])

print("(2) trace input is oracle-free and recovery-applicable:")
trace = rec.recovery_trace()
check("trace carries no oracle/evaluator fields",
      summary["traceInputOracleFree"] and not trace.oracle_fields_present)
check("trace is pressure/extremal, dimension >= 2, with active AC/BC and slack AB",
      rec.recovery_applicable(trace)
      and trace.dimension == 2
      and trace.optimizer_mode == "query_pressure_extremal"
      and set(trace.active_pairs) == {"AC", "BC"}
      and trace.slack_pairs == ("AB",))
check("active equation is exactly the KKT equation",
      trace.active_equation == (4, -9, 4))
bad_trace = rec.ActiveSetTrace(
    trace_id="bad",
    family=trace.family,
    dimension=2,
    optimizer_mode=trace.optimizer_mode,
    atom_volumes=trace.atom_volumes,
    pair_volumes=trace.pair_volumes,
    triple_volume=trace.triple_volume,
    query_denominator=trace.query_denominator,
    query_value=trace.query_value,
    active_pairs=("AC",),
    slack_pairs=("AB",),
    active_equation=trace.active_equation,
)
check("recovery refuses a missing-active-pair trace",
      not rec.recovery_applicable(bad_trace))

print("(3) active-set recovery solves the algebra exactly:")
x = rec.recover_x_from_active_equation(trace)
q = rec.recover_lower_endpoint(trace)
check("recovered x is the feasible root in (1/2,1)",
      kkt.Surd(Fraction(1, 2)) < x < kkt.Surd(1)
      and x == kkt.Surd(9, -1) / kkt.Surd(8))
check("recovered x solves 4x^2 - 9x + 4 = 0",
      kkt.Surd(4) * x * x - kkt.Surd(9) * x + kkt.Surd(4) == kkt.Surd(0))
check("endpoint formula is q = 1/(4x)",
      q == kkt.Surd(1) / (kkt.Surd(4) * x))
check("recovered endpoint is the closed-form surd (9+sqrt17)/32",
      q == kkt.Surd(9, 1) / kkt.Surd(32))

print("(4) recovered trace agrees with observed geometry:")
result = rec.recovery_result(trace)
check("observed query value equals recovered endpoint",
      result.observed_matches_recovered
      and result.observed_query_value == result.recovered_endpoint)
check("closed-form validation is after-the-fact and exact",
      result.validation_matches_closed_form
      and q == closed.exact_global_infimum())
check("recovery rule records validation as separate from recovery",
      summary["recoveryRule"]["usesExactOracle"] is False
      and summary["recoveryRule"]["usesClosedFormForRecovery"] is False
      and summary["recoveryRule"]["usesClosedFormForValidation"] is True)

print("(5) recovery beats ordinary search and the earlier rational witness:")
check("ordinary restart lower endpoint sits above the recovered endpoint",
      result.ordinary_restart_gap_above_recovered > 0.0,
      f"gap={result.ordinary_restart_gap_above_recovered}")
check("old rational witness is also above the recovered endpoint",
      result.rational_witness_gap_above_recovered > 0.0,
      f"gap={result.rational_witness_gap_above_recovered}")
check("recovered endpoint is numerically the KKT value",
      abs(result.recovered_endpoint_float - float(kkt.Q_STAR)) < 1e-15)

print("(6) manifest and note round-trip:")
written = rec.write_results()
manifest_path = ROOT / "results" / "boxsel" / "phase7e_oracle_free_recovery" / "manifest.json"
loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
check("manifest writes the recovery summary",
      loaded["recoveryVersion"] == written["recoveryVersion"]
      and loaded["recovered"]["validation_matches_closed_form"] is True)
note = ROOT / "docs" / "boxsel" / "PHASE7E_ORACLE_FREE_RECOVERY.md"
check("Phase-7e note exists and names recovery plus closed-form validation",
      note.exists()
      and "oracle-free recovery" in note.read_text(encoding="utf-8")
      and "(9+sqrt17)/32" in note.read_text(encoding="utf-8"))

print(f"\n{'ALL PASS -- Phase-7e recovers the Helly box endpoint from oracle-free active-set traces' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
