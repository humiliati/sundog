#!/usr/bin/env python
"""Frozen test for the BoxSEL Phase-7b locked preregistration.

This verifies that Phase 7b has a locked detector/threshold boundary without pretending held-out
results exist.
Run: python scripts/test_boxsel_phase7b_prereg.py
"""

import sys

sys.path.insert(0, "scripts")
import boxsel_phase6b_trace_schema as schema
import boxsel_phase7b_prereg as prereg
import boxsel_phase7b_v2_detector as v2

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


print("(1) prereg is locked but not run:")
check("status is LOCKED_NOT_RUN",
      prereg.PHASE7B_PREREG_STATUS == "LOCKED_NOT_RUN")
check("locked flag is true",
      prereg.PHASE7B_PREREG_LOCKED is True)
check("results status remains NOT_RUN with no rows",
      prereg.RESULTS_STATUS == "NOT_RUN" and prereg.RESULT_ROWS == ())
check("held-out run is ready but not run",
      prereg.HELDOUT_RUN_STATUS == "READY_NOT_RUN")
check("lock blockers are cleared",
      prereg.prereg_can_lock() and prereg.LOCK_BLOCKERS == (),
      f"blockers={prereg.LOCK_BLOCKERS}")

print("(2) v2 detector and thresholds are frozen:")
check("detector version is frozen from the v2 module",
      prereg.DETECTOR_VERSION == v2.DETECTOR_VERSION and prereg.DETECTOR_STATUS == "FROZEN")
check("thresholds are frozen from the v2 module",
      prereg.THRESHOLD_VERSION == v2.THRESHOLD_VERSION
      and prereg.THRESHOLD_STATUS == "FROZEN"
      and prereg.FROZEN_THRESHOLDS == v2.frozen_thresholds())
check("corpus generator and evaluator are built",
      prereg.CORPUS_GENERATOR_STATUS == "BUILT" and prereg.EVALUATOR_STATUS == "BUILT")

print("(3) schema and feature boundary are clean:")
check("Phase-7b uses the current Phase-6b schema version",
      prereg.SCHEMA_VERSION == schema.SCHEMA_VERSION)
check("frozen trace features match schema fields",
      prereg.FROZEN_TRACE_FEATURES == schema.feature_names())
check("feature list is oracle-free",
      prereg.feature_list_is_oracle_free(), f"{prereg.FROZEN_TRACE_FEATURES}")
check("required v2 pressure/support features are present",
      prereg.required_v2_features_present(), f"{prereg.REQUIRED_V2_FEATURES}")

print("(4) seen cases and seeds are fenced off:")
check("all Phase-7 seen rows are exclusion rows",
      prereg.SEEN_CASE_EXCLUSIONS == schema.PHASE7_SEEN_CASE_IDS)
check("reserved held-out seeds are disjoint from Phase-3/6/7 and diagnostic seeds",
      prereg.reserved_seeds_are_clean())
check("a new candidate id set passes exclusion",
      prereg.seen_cases_are_excluded(("p7b-stable-pmp-00", "p7b-control-00")))
check("reusing a Phase-7 row fails exclusion",
      not prereg.seen_cases_are_excluded(("p7b-stable-pmp-00", "pmp-00")))

print("(5) held-out corpus plan is non-vacuous and includes the new failure class:")
check("planned corpus has at least 24 cases",
      prereg.heldout_case_count() >= 24, f"count={prereg.heldout_case_count()}")
check("planned traps have at least 16 cases",
      prereg.heldout_case_count("false_closure_trap") >= 16,
      f"traps={prereg.heldout_case_count('false_closure_trap')}")
check("stable PMP pressure variants are a named trap family",
      any(f.name == "stable_pmp_pressure_variants" and f.role == "false_closure_trap" for f in prereg.HELDOUT_FAMILIES))
check("controls include acceptance and loss-control families",
      prereg.heldout_case_count("acceptance_control") >= 6
      and prereg.heldout_case_count("loss_control") >= 3)

print("(6) metrics, predictions, and kill criteria are explicit:")
check("primary baseline remains restart variance",
      prereg.PRIMARY_BASELINE_VERSION == "restart_variance_only_v0"
      and prereg.PRIMARY_BASELINE_VERSION in prereg.BASELINES)
check("substantial exact widening threshold remains 0.10",
      prereg.SUBSTANTIAL_EXACT_WIDENING == 0.10)
check("success threshold is stricter than kill threshold",
      prereg.MAX_ACCEPTED_FALSE_CLOSURE_RATE < prereg.KILL_ACCEPTED_FALSE_CLOSURE_RATE)
check("baseline improvement margin is positive",
      prereg.MIN_BASELINE_IMPROVEMENT > 0)
check("loss escapes have zero accepted allowance",
      prereg.MAX_LOSS_ESCAPE_ACCEPTS == 0)
check("prediction list is populated and Phase-7b named",
      len(prereg.PREDICTIONS) == 5 and all(item.startswith("P7B-") for item in prereg.PREDICTIONS))
check("kill list is populated and includes seen-case reuse",
      len(prereg.KILL_CRITERIA) == 5
      and all(item.startswith("KILL7B-") for item in prereg.KILL_CRITERIA)
      and any("seen cases" in item for item in prereg.KILL_CRITERIA))

print("(7) summary exposes prereg metadata only:")
summary = prereg.prereg_summary()
check("summary reports locked and NOT_RUN",
      summary["locked"] is True and summary["results_status"] == "NOT_RUN")
check("summary reports frozen detector metadata",
      summary["detector_version"] == v2.DETECTOR_VERSION
      and summary["detector_status"] == "FROZEN"
      and summary["threshold_version"] == v2.THRESHOLD_VERSION
      and summary["threshold_status"] == "FROZEN")
check("summary reports counts and cleared blockers, not outcomes",
      summary["heldout_cases"] == prereg.heldout_case_count()
      and summary["false_closure_traps"] == prereg.heldout_case_count("false_closure_trap")
      and summary["lock_blockers"] == prereg.LOCK_BLOCKERS)

print(f"\n{'ALL PASS -- Phase-7b prereg locked: v2 detector/thresholds frozen, results still NOT_RUN' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
