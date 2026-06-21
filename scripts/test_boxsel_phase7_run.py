#!/usr/bin/env python
"""Frozen test for the BoxSEL Phase-7 held-out run result.

Run: python scripts/test_boxsel_phase7_run.py
"""

import sys

sys.path.insert(0, "scripts")
import boxsel_phase6_trace_detector as detector
import boxsel_phase7_prereg as prereg
import boxsel_phase7_run as run

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


print("(1) prereg boundary is preserved:")
cases = run.run_cases()
summary = run.summarize(cases)
check("prereg remains locked and result-free",
      prereg.PHASE7_PREREG_LOCKED is True and prereg.RESULTS_STATUS == "NOT_RUN" and prereg.RESULT_ROWS == ())
check("run has the preregistered 16 cases",
      len(cases) == prereg.heldout_case_count() == 16)
check("run has the preregistered 10 false-closure-family cases",
      sum(1 for case in cases if case.role == "false_closure_trap") == 10)
check("held-out seeds stay disjoint from seed-trap seeds",
      prereg.seeds_are_held_out())

print("(2) oracle labels are computed after trace decisions:")
traps = tuple(case for case in cases if case.role == "false_closure_trap")
helly = tuple(case for case in traps if case.family == "helly_threshold_variants")
pmp = tuple(case for case in traps if case.family == "pmp_interval_chain_variants")
check("all 10 trap-family cases are actually false-closed under Phase-7 label",
      all(case.false_closed for case in traps))
check("Helly cases have exact lower zero and visible exact widening",
      len(helly) == 6 and all(case.exact_lower == 0.0 and case.lower_exact_widening > 0.5 for case in helly))
check("PMP cases have substantial but smaller exact widening",
      len(pmp) == 4 and all(0.10 <= case.lower_exact_widening < 0.20 for case in pmp))
check("all trap cases are zero-loss for scoring",
      all(case.max_loss <= detector.LOSS_TOLERANCE for case in traps))

print("(3) detector decisions expose the failure mode:")
check("Helly false closures are caught by abstention",
      all(case.detector_action == detector.ACTION_ABSTAIN and not case.accepted_false_closure for case in helly))
check("PMP-shaped stable false closures are all accepted",
      all(case.detector_action == detector.ACTION_ACCEPT and case.accepted_false_closure for case in pmp))
check("PMP accepted cases have no trace flags",
      all(case.detector_flags == () for case in pmp))
check("detector accepted false-closure count is 4/10",
      summary.detector_accepted_false_closures == 4 and summary.false_closure_traps == 10)

print("(4) preregistered metrics and kill criteria:")
check("accepted false-closure rate is 0.4",
      abs(summary.accepted_false_closure_rate - 0.4) < 1e-15)
check("restart-variance baseline also accepts 4/10",
      summary.baseline_accepted_false_closures == 4
      and abs(summary.baseline_accepted_false_closure_rate - 0.4) < 1e-15)
check("baseline improvement is zero",
      summary.baseline_improvement == 0.0)
check("kill criteria KILL7-1 and KILL7-2 trigger",
      summary.status == run.RUN_STATUS_FAIL and summary.kill_criteria_triggered == ("KILL7-1", "KILL7-2"))
check("only P7-P3 and P7-P4 are supported",
      summary.predictions_supported == ("P7-P3", "P7-P4"))

print("(5) controls behave as intended:")
narrow = tuple(case for case in cases if case.role == "acceptance_control")
loss = tuple(case for case in cases if case.role == "loss_control")
check("true-narrow controls are accepted",
      len(narrow) == 4 and all(case.detector_action == detector.ACTION_ACCEPT for case in narrow))
check("true-narrow accept rate is 1.0",
      summary.true_narrow_accept_rate == 1.0)
check("loss controls abstain",
      len(loss) == 2 and all(case.detector_action == detector.ACTION_ABSTAIN for case in loss))
check("loss controls have zero accepted allowance and zero accepted count",
      summary.loss_escape_accept_count == prereg.MAX_LOSS_ESCAPE_ACCEPTS == 0)

print("(6) payload and limitations are explicit:")
payload = run.result_payload()
check("payload summary matches direct summary",
      payload["summary"]["status"] == summary.status
      and payload["summary"]["detector_accepted_false_closures"] == summary.detector_accepted_false_closures)
check("payload records prereg result status as NOT_RUN",
      payload["run_config"]["results_status_in_prereg"] == "NOT_RUN")
check("limitations record the non-general trace interface boundary",
      any("Helly-shaped" in item for item in summary.limitations))

print(f"\n{'ALL PASS -- Phase-7 run reproduced: guard fails prereg gate on stable PMP false closures' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
