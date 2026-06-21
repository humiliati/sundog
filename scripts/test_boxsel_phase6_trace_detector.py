#!/usr/bin/env python
"""Frozen test for BoxSEL Phase-6 trace detector start.

Locks the first accept/widen/abstain guard over observable restart traces. The guard is evaluated
against the exact endpoint, but the decision features themselves are oracle-free.
Run: python scripts/test_boxsel_phase6_trace_detector.py
"""

import sys

sys.path.insert(0, "scripts")
import boxsel_phase6_trace_detector as det

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


print("(1) Helly-seed trace receipt triggers a guard, not accept:")
report, decision, evaluation = det.helly_seed_detector_receipt()
check("canonical report is the Phase-3 deterministic sample",
      report.dim == 2 and report.restarts == 128 and report.seed == 314159)
check("sampled lower endpoint is the locked Phase-3 value",
      abs(report.sample_interval[0] - 0.5336525204919725) < 1e-15)
check("detector abstains on the Helly false-closure trace",
      decision.action == det.ACTION_ABSTAIN, f"action={decision.action}, flags={decision.flags}")
check("flags include endpoint drift, active slack, seed variance, and dimension sensitivity",
      {"endpoint_drift", "active_constraint_slack", "seed_variance", "dimension_sensitivity"}.issubset(set(decision.flags)),
      f"flags={decision.flags}")
check("oracle evaluation labels the case false-closed but not accepted",
      evaluation.false_closed is True and evaluation.accepted_false_closure is False,
      f"gap={evaluation.lower_search_gap:.6f}, action={decision.action}")

print("(2) feature values are trace-only and shaped for Phase 6:")
features = decision.features
check("feature extraction records sampled interval and endpoint movement",
      features.sample_lower == report.sample_interval[0]
      and features.sample_upper == report.sample_interval[1]
      and features.early_lower_drop > det.EARLY_DROP_THRESHOLD)
check("late movement can be stable while early movement still warns",
      features.late_lower_drop <= det.LATE_DROP_THRESHOLD
      and "late_endpoint_drift" not in decision.flags)
check("slack, seed variance, and dimension spread cross their thresholds",
      features.min_slack < det.ACTIVE_SLACK_THRESHOLD
      and features.seed_low_range > det.SEED_RANGE_THRESHOLD
      and features.dimension_low_spread > det.DIMENSION_SPREAD_THRESHOLD)
check("feature dataclass has no exact/oracle endpoint fields",
      all("exact" not in name and "oracle" not in name for name in features.__dataclass_fields__))

print("(3) accept/widen/abstain controls:")
stable = det.synthetic_report(tuple([0.8] * 32), min_pair_overlap=0.4)
stable_decision = det.detector_decision(stable)
check("stable high-slack synthetic trace is accepted",
      stable_decision.action == det.ACTION_ACCEPT and stable_decision.flags == (),
      f"{stable_decision}")
one_flag = det.synthetic_report(tuple([0.8] * 32), min_pair_overlap=0.2501)
one_flag_decision = det.detector_decision(one_flag)
check("single warning flag widens rather than accepts",
      one_flag_decision.action == det.ACTION_WIDEN
      and one_flag_decision.flags == ("active_constraint_slack",),
      f"{one_flag_decision}")
lossy = det.synthetic_report(tuple([0.8] * 8), min_pair_overlap=0.4, max_loss=1e-6)
lossy_decision = det.detector_decision(lossy)
check("loss escape abstains immediately",
      lossy_decision.action == det.ACTION_ABSTAIN and "loss_escape" in lossy_decision.flags,
      f"{lossy_decision}")

print("(4) evaluator stays separate from the detector:")
stable_eval = det.evaluate_with_oracle(stable, stable_decision)
check("oracle evaluator can label after the decision without changing it",
      stable_eval.decision is stable_decision and stable_decision.action == det.ACTION_ACCEPT)
check("false-closure threshold is positive and explicit",
      det.FALSE_CLOSURE_GAP_THRESHOLD > 0)
check("Helly lower search gap comfortably exceeds evaluator threshold",
      evaluation.lower_search_gap > det.FALSE_CLOSURE_GAP_THRESHOLD)

print(f"\n{'ALL PASS -- Phase-6 trace detector start: Helly false closure abstains; accept/widen controls locked; decision features are oracle-free' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
