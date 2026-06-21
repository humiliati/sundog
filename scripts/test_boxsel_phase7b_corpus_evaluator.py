#!/usr/bin/env python
"""Frozen test for BoxSEL Phase-7b corpus generator and evaluator infrastructure.

Run: python scripts/test_boxsel_phase7b_corpus_evaluator.py
"""

import sys

sys.path.insert(0, "scripts")
import boxsel_phase7b_corpus as corpus
import boxsel_phase7b_evaluator as evaluator
import boxsel_phase7b_prereg as prereg

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


print("(1) corpus generator matches the prereg-start plan:")
cases = corpus.generate_phase7b_corpus()
summary = corpus.corpus_summary(cases)
check("corpus generator status is built-not-run",
      summary["corpus_generator_version"] == "phase7b_corpus_generator_v0"
      and summary["corpus_status"] == "BUILT_NOT_RUN")
check("case count matches prereg plan",
      len(cases) == prereg.heldout_case_count() == 28)
check("trap/control counts match prereg plan",
      summary["false_closure_traps"] == prereg.heldout_case_count("false_closure_trap") == 16
      and summary["acceptance_controls"] == prereg.heldout_case_count("acceptance_control") == 9
      and summary["loss_controls"] == prereg.heldout_case_count("loss_control") == 3)
check("family counts match every planned family",
      summary["families"] == {family.name: family.count for family in prereg.HELDOUT_FAMILIES})

print("(2) corpus respects seed and seen-case exclusions:")
case_ids = tuple(case.case_id for case in cases)
check("case ids are unique",
      len(case_ids) == len(set(case_ids)))
check("no Phase-7 seen case ids are reused",
      prereg.seen_cases_are_excluded(case_ids))
check("reserved seeds remain clean",
      prereg.reserved_seeds_are_clean())
check("all case seeds come from the reserved pool",
      set(case.seed for case in cases).issubset(set(prereg.RESERVED_HELDOUT_SEEDS)))

print("(3) stable PMP and support families expose pressure movement:")
stable = tuple(case for case in cases if case.family == "stable_pmp_pressure_variants")
support = tuple(case for case in cases if case.family == "support_floor_variants")
check("stable PMP pressure family has 8 cases",
      len(stable) == 8)
check("stable PMP traces are quiet under ordinary restarts but pressure moves them",
      all(case.features().early_lower_drop <= 0.01 and case.features().pressure_low_shift > 0.10 for case in stable))
check("stable PMP optimizer spread equals pressure shift",
      all(abs(case.features().optimizer_low_spread - case.features().pressure_low_shift) < 1e-15 for case in stable))
check("support-floor family has low support and pressure movement",
      len(support) == 4
      and all(case.features().support_floor <= 0.125 and case.features().pressure_low_shift > 0.15 for case in support))

print("(4) evaluator labels corpus without detector decisions:")
labels = evaluator.label_corpus(cases)
label_by_id = {label.case_id: label for label in labels}
check("one exact label per case",
      len(labels) == len(cases) and set(label_by_id) == set(case_ids))
check("all false-closure trap families label as false-closed",
      all(label_by_id[case.case_id].false_closed for case in cases if case.role == "false_closure_trap"))
check("acceptance controls are not false-closed",
      all(not label_by_id[case.case_id].false_closed for case in cases if case.role == "acceptance_control"))
check("loss controls are not low-loss and not false-closed",
      all((not label_by_id[case.case_id].low_loss) and (not label_by_id[case.case_id].false_closed)
          for case in cases if case.role == "loss_control"))
check("evaluator summary reports labels and no detector run",
      evaluator.evaluator_summary(cases)["evaluator_status"] == "BUILT_NOT_RUN"
      and evaluator.evaluator_summary(cases)["false_closed_labels"] == 16)

print("(5) restart-variance baseline scoring hook is present:")
baseline = evaluator.evaluate_restart_variance_baseline(cases)
baseline_summary = evaluator.summarize_decisions(baseline)
check("baseline returns one decision per case",
      len(baseline) == len(cases))
check("stable PMP pressure variants are accepted by restart-variance baseline",
      all(ev.action == evaluator.ACTION_ACCEPT for ev in baseline if ev.family == "stable_pmp_pressure_variants"))
check("baseline accepts at least the stable quiet false closures",
      baseline_summary.accepted_false_closures >= len(stable))
check("baseline summary is a scoring hook, not a v2 detector claim",
      baseline_summary.case_count == len(cases)
      and "KILL7B-1" in baseline_summary.kill_criteria_triggered)

print("(6) future decision evaluator handles actions and controls:")
all_abstain = tuple(evaluator.evaluate_decision(case, evaluator.ACTION_ABSTAIN, ("test_abstain",)) for case in cases)
all_abstain_summary = evaluator.summarize_decisions(all_abstain)
check("all-abstain accepts no false closures but fails true-narrow acceptance",
      all_abstain_summary.accepted_false_closures == 0
      and all_abstain_summary.true_narrow_accept_rate == 0.0)
all_accept = tuple(evaluator.evaluate_decision(case, evaluator.ACTION_ACCEPT, ()) for case in cases)
all_accept_summary = evaluator.summarize_decisions(all_accept)
check("all-accept triggers false-closure and loss-control failures",
      all_accept_summary.accepted_false_closure_rate == 1.0
      and all_accept_summary.loss_escape_accept_count == 3
      and {"KILL7B-1", "KILL7B-3"}.issubset(set(all_accept_summary.kill_criteria_triggered)))
try:
    evaluator.evaluate_decision(cases[0], "maybe")
    invalid_action_failed = False
except ValueError:
    invalid_action_failed = True
check("invalid detector action is rejected",
      invalid_action_failed)

print(f"\n{'ALL PASS -- Phase-7b corpus generator and evaluator built; no v2 detector or held-out run executed' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
