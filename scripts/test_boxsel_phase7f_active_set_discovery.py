#!/usr/bin/env python
"""Frozen test for BoxSEL Phase-7f active-set discovery.

Run: python scripts/test_boxsel_phase7f_active_set_discovery.py
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, "scripts")
import boxsel_kkt_exact as kkt
import boxsel_phase7e_oracle_free_recovery as recovery
import boxsel_phase7f_active_set_discovery as disc

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


print("(1) discovery boundary and claim:")
summary = disc.discovery_summary()
check("version and status are explicit",
      summary["discoveryVersion"] == "phase7f_active_set_discovery_v0"
      and summary["status"] == "DISCOVERY_RECEIPT")
check("claim names raw geometry, residuals, and active equation",
      "raw oracle-free box geometry" in summary["primaryDiscoveryClaim"]
      and "exact residuals" in summary["primaryDiscoveryClaim"]
      and "4x^2 - 9x + 4" in summary["primaryDiscoveryClaim"])
check("boundary blocks general optimizer and real-KG claims",
      "Toy micro-SEL" in summary["boundary"]
      and "not a general active-set learner" in summary["boundary"]
      and "not a global optimizer" in summary["boundary"]
      and "not a real-KG" in summary["boundary"])

print("(2) raw trace has no active-set labels as input:")
raw = disc.raw_kkt_box_trace()
check("raw trace is oracle-free",
      summary["rawTraceOracleFree"] and not raw.oracle_fields_present)
check("raw trace object has boxes but no active pair/equation fields",
      hasattr(raw, "boxes")
      and not hasattr(raw, "active_pairs")
      and not hasattr(raw, "active_equation"))
check("discovery rule records no exact oracle, no active-label input, and no closed-form discovery",
      summary["discoveryRule"]["usesExactOracle"] is False
      and summary["discoveryRule"]["usesActiveLabelsAsInput"] is False
      and summary["discoveryRule"]["usesClosedFormForDiscovery"] is False)

print("(3) exact residuals discover the active set:")
atoms = disc.atom_volumes(raw)
pairs = disc.pair_volumes(raw)
residuals = disc.pair_residuals(raw)
active, slack = disc.discover_pairs(raw)
check("atom volumes are exactly 1/2",
      all(value == disc.ATOM_TARGET for value in atoms.values()))
check("AC and BC residuals are exactly zero",
      residuals["AC"].sign() == 0 and residuals["BC"].sign() == 0)
check("AB residual is positive slack",
      residuals["AB"].sign() > 0)
check("discovered active/slack pairs are AC/BC active and AB slack",
      set(active) == {"AC", "BC"} and slack == ("AB",))

print("(4) structured geometry derives the KKT equation:")
check("raw trace matches the structured KKT geometry predicate",
      disc.structured_kkt_geometry(raw))
x, z = disc.structured_parameters(raw)
z_from_ac = kkt.Surd(2) * (kkt.Surd(1) - x)
z_from_bc = x / (kkt.Surd(2) * (kkt.Surd(1) - x))
check("discovered x and z are the KKT parameters",
      x == kkt.Surd(9, -1) / kkt.Surd(8)
      and z == z_from_ac == z_from_bc)
check("derived active equation is 4x^2 - 9x + 4 = 0",
      disc.derive_active_equation(raw) == (4, -9, 4)
      and kkt.Surd(4) * x * x - kkt.Surd(9) * x + kkt.Surd(4) == kkt.Surd(0))

print("(5) discovered active trace feeds Phase-7e recovery:")
active_trace = disc.discovered_active_trace(raw)
recovered = recovery.recovery_result(active_trace)
check("discovered trace is accepted by the Phase-7e applicability predicate",
      recovery.recovery_applicable(active_trace))
check("discovered trace carries active labels produced by discovery",
      set(active_trace.active_pairs) == {"AC", "BC"}
      and active_trace.slack_pairs == ("AB",)
      and active_trace.active_equation == (4, -9, 4))
check("recovery from discovered trace equals the KKT closed form",
      recovered.validation_matches_closed_form
      and recovered.recovered_endpoint == "(9/32 + 1/32*sqrt17)")
check("summary carries the recovered endpoint",
      summary["phase7eRecovered"]["recoveredEndpoint"] == "(9/32 + 1/32*sqrt17)"
      and summary["phase7eRecovered"]["validationMatchesClosedForm"] is True)

print("(6) negative control rejects the old rational witness:")
negative_raw = disc.raw_rational_witness_trace()
neg_active, neg_slack = disc.discover_pairs(negative_raw)
neg_result = disc.discovery_result(negative_raw)
check("rational witness has only AC active",
      neg_active == ("AC",) and set(neg_slack) == {"AB", "BC"})
check("rational witness does not derive the KKT active equation",
      disc.derive_active_equation(negative_raw) is None
      and neg_result.active_equation is None
      and not neg_result.applicable)
try:
    disc.discovered_active_trace(negative_raw)
    negative_rejected = False
except ValueError:
    negative_rejected = True
check("rational witness cannot be converted into a Phase-7e recovery trace",
      negative_rejected)

print("(7) manifest and note round-trip:")
written = disc.write_results()
manifest_path = ROOT / "results" / "boxsel" / "phase7f_active_set_discovery" / "manifest.json"
loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
check("manifest writes discovery summary",
      loaded["discoveryVersion"] == written["discoveryVersion"]
      and loaded["discovered"]["active_equation"] == [4, -9, 4])
note = ROOT / "docs" / "boxsel" / "PHASE7F_ACTIVE_SET_DISCOVERY.md"
check("Phase-7f note exists and names active-set discovery",
      note.exists()
      and "active-set discovery" in note.read_text(encoding="utf-8")
      and "raw box intervals" in note.read_text(encoding="utf-8"))

print(f"\n{'ALL PASS -- Phase-7f discovers the active set from raw box traces before recovery' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
