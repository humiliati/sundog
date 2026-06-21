#!/usr/bin/env python
"""Frozen test for the BoxSEL Phase-7b held-out v2 run result.

Run: python scripts/test_boxsel_phase7b_run.py
"""

import sys

sys.path.insert(0, "scripts")
import boxsel_phase7b_evaluator as evaluator
import boxsel_phase7b_prereg as prereg
import boxsel_phase7b_run as run
import boxsel_phase7b_v2_detector as detector

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("(1) locked prereg boundary is preserved:")
cases = run.run_cases()
summary = run.summarize(cases)
check("prereg remains locked and result-free",
      prereg.PHASE7B_PREREG_LOCKED is True
      and prereg.RESULTS_STATUS == "NOT_RUN"
      and prereg.RESULT_ROWS == ())
check("held-out run was ready at execution",
      prereg.HELDOUT_RUN_STATUS == "READY_NOT_RUN" and prereg.LOCK_BLOCKERS == ())
check("run has the preregistered 28 cases",
      len(cases) == prereg.heldout_case_count() == 28)
check("run has the preregistered 16 false-closure traps",
      sum(1 for case in cases if case.role == "false_closure_trap") == 16)

print("(2) labels are computed after trace decisions:")
traps = tuple(case for case in cases if case.role == "false_closure_trap")
stable_pmp = tuple(case for case in traps if case.family == "stable_pmp_pressure_variants")
helly = tuple(case for case in traps if case.family == "helly_threshold_variants_v2")
support = tuple(case for case in traps if case.family == "support_floor_variants")
check("all trap-family cases are actually false-closed",
      all(case.false_closed for case in traps))
check("stable PMP traps have substantial pressure movement",
      len(stable_pmp) == 8
      and all(case.pressure_low_shift >= detector.PRESSURE_LOW_SHIFT_THRESHOLD for case in stable_pmp))
check("Helly traps are fresh v2 family rows",
      len(helly) == 4 and all(case.case_id.startswith("p7b-helly-") for case in helly))
check("support traps have low support and pressure movement",
      len(support) == 4
      and all(case.support_floor <= detector.SUPPORT_FLOOR_THRESHOLD for case in support))

print("(3) v2 decisions pass the preregistered falsifier:")
check("stable PMP false closures are not accepted",
      all(case.detector_action == detector.ACTION_ABSTAIN and not case.accepted_false_closure for case in stable_pmp))
check("stable PMP flags are pressure/optimizer warnings",
      all({"pressure_low_shift", "optimizer_low_spread"}.issubset(set(case.detector_flags)) for case in stable_pmp))
check("Helly false closures widen rather than accept",
      all(case.detector_action == detector.ACTION_WIDEN and not case.accepted_false_closure for case in helly))
check("support false closures abstain rather than accept",
      all(case.detector_action == detector.ACTION_ABSTAIN and not case.accepted_false_closure for case in support))
check("detector accepted false-closure count is 0/16",
      summary.detector_accepted_false_closures == 0 and summary.false_closure_traps == 16)

print("(4) baseline comparison and predictions:")
check("restart-variance baseline accepts all 16 traps",
      summary.baseline_accepted_false_closures == 16
      and summary.baseline_accepted_false_closure_rate == 1.0)
check("baseline improvement is 1.0",
      summary.baseline_improvement == 1.0)
check("pressure warning rate beats restart variance on stable PMP",
      summary.pressure_warning_rate_on_stable_pmp == 1.0
      and summary.baseline_pressure_warning_rate_on_stable_pmp == 0.0)
check("no kill criteria trigger and all predictions are supported",
      summary.status == run.RUN_STATUS_PASS
      and summary.kill_criteria_triggered == ()
      and summary.predictions_supported == ("P7B-P1", "P7B-P2", "P7B-P3", "P7B-P4", "P7B-P5"))

print("(5) controls behave as preregistered:")
controls = tuple(case for case in cases if case.role == "acceptance_control")
loss_controls = tuple(case for case in cases if case.role == "loss_control")
check("acceptance controls are accepted",
      len(controls) == prereg.heldout_case_count("acceptance_control") == 9
      and all(case.detector_action == detector.ACTION_ACCEPT for case in controls))
check("true-narrow accept rate is 1.0",
      summary.true_narrow_accept_rate == 1.0)
check("loss controls abstain and none are accepted",
      len(loss_controls) == prereg.heldout_case_count("loss_control") == 3
      and all(case.detector_action == detector.ACTION_ABSTAIN for case in loss_controls)
      and summary.loss_escape_accept_count == prereg.MAX_LOSS_ESCAPE_ACCEPTS == 0)

print("(6) payload exposes run results separately from prereg lock:")
payload = run.result_payload()
check("payload summary matches direct summary",
      payload["summary"]["status"] == summary.status
      and payload["summary"]["detector_accepted_false_closures"] == summary.detector_accepted_false_closures)
check("payload records frozen detector metadata",
      payload["detector"]["detector_version"] == detector.DETECTOR_VERSION
      and payload["detector"]["threshold_status"] == "FROZEN")
check("payload records prereg result status as NOT_RUN",
      payload["run_config"]["results_status_in_prereg"] == "NOT_RUN")
baseline_summary = evaluator.summarize_decisions(evaluator.evaluate_restart_variance_baseline(()))
check("empty baseline summary remains well-defined",
      baseline_summary.case_count == 0 and baseline_summary.accepted_false_closure_rate == 0.0)

print(f"\n{'ALL PASS -- Phase-7b run passed prereg gate: v2 catches stable pressure traps and beats restart variance' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
