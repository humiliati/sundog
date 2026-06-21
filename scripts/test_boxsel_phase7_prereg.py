#!/usr/bin/env python
"""Frozen test for BoxSEL Phase-7 prereg lock.

This test verifies the prereg protocol is locked before held-out runs exist. It does not run held-out
cases and must fail if result rows are added to the prereg artifact.
Run: python scripts/test_boxsel_phase7_prereg.py
"""

import sys

sys.path.insert(0, "scripts")
import boxsel_phase6_trace_detector as phase6
import boxsel_phase7_prereg as prereg

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


print("(1) prereg is locked and contains no held-out results:")
check("prereg lock is explicit", prereg.PHASE7_PREREG_LOCKED is True)
check("results status is NOT_RUN", prereg.RESULTS_STATUS == "NOT_RUN")
check("no result rows are present in the prereg", prereg.RESULT_ROWS == ())

print("(2) detector thresholds are frozen from Phase 6:")
thresholds = prereg.detector_thresholds()
check("threshold keys are complete",
      set(thresholds) == {
          "LOSS_TOLERANCE",
          "EARLY_DROP_THRESHOLD",
          "LATE_DROP_THRESHOLD",
          "ACTIVE_SLACK_THRESHOLD",
          "SEED_RANGE_THRESHOLD",
          "DIMENSION_SPREAD_THRESHOLD",
          "FALSE_CLOSURE_GAP_THRESHOLD",
      })
check("threshold values match Phase 6 constants",
      thresholds["LOSS_TOLERANCE"] == phase6.LOSS_TOLERANCE
      and thresholds["EARLY_DROP_THRESHOLD"] == phase6.EARLY_DROP_THRESHOLD
      and thresholds["LATE_DROP_THRESHOLD"] == phase6.LATE_DROP_THRESHOLD
      and thresholds["ACTIVE_SLACK_THRESHOLD"] == phase6.ACTIVE_SLACK_THRESHOLD
      and thresholds["SEED_RANGE_THRESHOLD"] == phase6.SEED_RANGE_THRESHOLD
      and thresholds["DIMENSION_SPREAD_THRESHOLD"] == phase6.DIMENSION_SPREAD_THRESHOLD
      and thresholds["FALSE_CLOSURE_GAP_THRESHOLD"] == phase6.FALSE_CLOSURE_GAP_THRESHOLD)

print("(3) held-out corpus plan is non-vacuous and seed-separated:")
check("held-out seeds do not reuse Phase-3/Phase-6 seed-trap seeds",
      prereg.seeds_are_held_out())
check("planned held-out corpus has at least 16 cases",
      prereg.heldout_case_count() >= 16, f"count={prereg.heldout_case_count()}")
check("planned false-closure trap count is at least 10",
      prereg.heldout_case_count("false_closure_trap") >= 10,
      f"traps={prereg.heldout_case_count('false_closure_trap')}")
check("control families are present",
      prereg.heldout_case_count("acceptance_control") > 0
      and prereg.heldout_case_count("loss_control") > 0)

print("(4) features and scoring boundary are oracle-clean:")
check("trace feature list is oracle-free",
      prereg.feature_list_is_oracle_free(), f"{prereg.TRACE_ONLY_FEATURES}")
check("detector and baseline versions are named",
      prereg.DETECTOR_VERSION and prereg.BASELINE_VERSION)
check("substantial widening threshold is stricter than the detector false-closure threshold",
      prereg.SUBSTANTIAL_EXACT_WIDENING > prereg.FALSE_CLOSURE_GAP_THRESHOLD)

print("(5) pass/kill criteria are explicit and ordered:")
check("prediction list is populated",
      len(prereg.PREDICTIONS) == 4 and all(item.startswith("P7-") for item in prereg.PREDICTIONS))
check("kill list is populated",
      len(prereg.KILL_CRITERIA) == 4 and all(item.startswith("KILL7-") for item in prereg.KILL_CRITERIA))
check("success rate is stricter than kill rate",
      prereg.MAX_ACCEPTED_FALSE_CLOSURE_RATE < prereg.KILL_ACCEPTED_FALSE_CLOSURE_RATE)
check("baseline improvement margin is positive",
      prereg.MIN_BASELINE_IMPROVEMENT > 0)
check("loss escapes have zero accepted allowance",
      prereg.MAX_LOSS_ESCAPE_ACCEPTS == 0)

print("(6) summary exposes prereg metadata only:")
summary = prereg.prereg_summary()
check("summary reports NOT_RUN and case counts, not results",
      summary["results_status"] == "NOT_RUN"
      and summary["heldout_cases"] == prereg.heldout_case_count()
      and "result" not in summary)

print(f"\n{'ALL PASS -- Phase-7 prereg locked; thresholds/corpus/kill criteria frozen; no held-out results run' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
