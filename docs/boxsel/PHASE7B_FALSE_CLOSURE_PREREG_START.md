# BoxSEL Phase 7b - False-Closure Preregistration Start

**Date:** 2026-06-21  
**Status:** `STARTED_NOT_LOCKED`

This opens the Phase-7b preregistration lane after the Phase-7 bounded null. It is not a locked
preregistration and it authorizes no held-out run.

## Boundary

Phase 7 failed:

```text
FAIL_PREREG_GATE
accepted false closures : 4 / 10 = 0.40
baseline accepted       : 4 / 10 = 0.40
triggered kills         : KILL7-1, KILL7-2
```

The failure class is now named:

```text
stable_low_loss_false_closure
```

All Phase-7 rows are diagnostic/training rows only. They cannot be used as Phase-7b held-out
validation rows:

```text
helly-00 ... helly-05
pmp-00 ... pmp-03
narrow-00 ... narrow-03
loss-00 ... loss-01
```

## Lock State

```text
PHASE7B_PREREG_STATUS = STARTED_NOT_LOCKED
PHASE7B_PREREG_LOCKED = False
RESULTS_STATUS        = NOT_RUN
HELDOUT_RUN_STATUS    = BLOCKED_UNTIL_LOCK
```

Current blockers:

```text
V2_DETECTOR_RULE_NOT_FROZEN
V2_THRESHOLDS_NOT_FROZEN
```

Until those clear, this file is a preregistration start, not a preregistration lock.

Infrastructure now built, still not a held-out run:

```text
CORPUS_GENERATOR_STATUS = BUILT
EVALUATOR_STATUS        = BUILT
```

## Frozen So Far

Schema:

```text
phase6b_general_trace_schema_v1
```

Trace features:

```text
sample_lower
sample_upper
sample_width
early_lower_drop
late_lower_drop
max_loss
max_constraint_violation
min_constraint_slack
condition_mass_floor
numerator_mass_floor
support_floor
pressure_low_shift
seed_low_range
dimension_low_spread
optimizer_low_spread
```

Required v2 signals:

```text
pressure_low_shift
optimizer_low_spread
support_floor
max_constraint_violation
min_constraint_slack
```

Baselines:

```text
primary     : restart_variance_only_v0
diagnostic  : phase6_trace_detector_start
```

Reserved seed pool:

```text
9001, 9011, 9029, 9041, 9059, 9067, 9091, 9103,
9127, 9137, 9151, 9173, 9187, 9203, 9221, 9239
```

These are disjoint from Phase-3/Phase-6 seed traps, Phase-7 held-out seeds, and Phase-6b diagnostic
seeds.

## Planned Corpus

```text
8  stable_pmp_pressure_variants   false_closure_trap
4  helly_threshold_variants_v2    false_closure_trap
4  support_floor_variants         false_closure_trap
6  true_narrow_controls_v2        acceptance_control
3  pressure_noop_controls         acceptance_control
3  loss_escape_controls_v2        loss_control
---
28 planned cases
16 false-closure traps
```

The stable PMP pressure variants are the key new family. They directly target the Phase-7 miss:
ordinary restarts can be stable and low-loss while query-pressure traces expose lower-endpoint
movement.

## Scoring Plan

Primary metric:

```text
accepted_false_closure_rate
```

Secondary metrics:

```text
baseline improvement over restart_variance_only_v0
true_narrow_accept_rate
loss_escape_accept_count
pressure_warning_rate_on_stable_pmp
```

Label boundary:

```text
false closure iff exact lower endpoint is at least 0.10 below sampled lower endpoint
```

The exact oracle label is evaluator-only and must be computed after the trace-only decision.

## Predictions

```text
P7B-P1: once locked, v2 will widen or abstain on at least 90% of held-out false-closure traps.
P7B-P2: accepted false-closure rate will beat restart_variance_only_v0 by at least 20 percentage points.
P7B-P3: stable true-narrow controls will be accepted at least 50% of the time.
P7B-P4: no loss-escape control may be accepted.
P7B-P5: stable PMP pressure variants will trigger pressure or optimizer-spread warnings more often than restart-variance warnings.
```

## Kill Criteria

```text
KILL7B-1: accepted false-closure rate is 25% or higher on held-out traps.
KILL7B-2: v2 fails to beat restart_variance_only_v0 by 20 percentage points.
KILL7B-3: any loss-escape control is accepted.
KILL7B-4: Phase-7 seen cases are reused as held-out validation rows.
KILL7B-5: thresholds or feature definitions are changed after the locked prereg.
```

## What Must Happen Before Lock

1. Build and freeze the v2 detector rule.
2. Freeze v2 thresholds.
3. Confirm no Phase-7 case IDs or seeds are reused as held-out validation.
4. Replace this prereg-start note with a locked prereg or add a locked addendum.

Built before lock, but not yet sufficient to lock:

```text
Phase-7b corpus generator
Phase-7b evaluator
```

## Results

```text
NOT_RUN
```

This prereg start contains no held-out outcomes.

## Artifacts

- `scripts/boxsel_phase7b_prereg.py`
- `scripts/boxsel_phase7b_corpus.py`
- `scripts/boxsel_phase7b_evaluator.py`
- `scripts/test_boxsel_phase7b_prereg.py`
- `scripts/test_boxsel_phase7b_corpus_evaluator.py`
- `docs/boxsel/PHASE7B_CORPUS_EVALUATOR_START.md`

Verification:

```text
python scripts/test_boxsel_phase7b_prereg.py
python scripts/test_boxsel_phase7b_corpus_evaluator.py
```

Result:

```text
29/29 checks pass, exit 0.
24/24 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-7b preregistration start. Internal; not locked, no held-out run.*
