# BoxSEL Phase 7b - V2 Freeze Lock

**Date:** 2026-06-21  
**Status:** `LOCKED_NOT_RUN`

This freezes the Phase-7b v2 detector and thresholds. It is a lock receipt, not a held-out run.

## Frozen Detector

```text
DETECTOR_VERSION  = phase7b_v2_trace_detector_v1
DETECTOR_STATUS   = FROZEN
THRESHOLD_VERSION = phase7b_v2_thresholds_v1
THRESHOLD_STATUS  = FROZEN
RESULTS_STATUS    = NOT_RUN
```

Artifact:

```text
scripts/boxsel_phase7b_v2_detector.py
```

The detector consumes only `GeneralTraceFeatures` from the Phase-6b schema. It does not import the
exact oracle, evaluator labels, or Phase-7b corpus generator.

## Frozen Rule

Immediate abstain:

```text
max_loss > 1e-9
max_constraint_violation > 1e-9
pressure_low_shift >= 0.05
optimizer_low_spread >= 0.05
```

Widen:

```text
support_floor <= 0.08
early_lower_drop >= 0.03
late_lower_drop >= 0.015
seed_low_range >= 0.02
dimension_low_spread >= 0.02
```

Accept:

```text
no frozen warning flags
```

`min_constraint_slack` is retained in the frozen feature schema but is not a standalone v2 trigger.
That keeps exact-equality point controls from being widened solely because their slack is zero.

## Lock State

```text
PHASE7B_PREREG_STATUS = LOCKED_NOT_RUN
PHASE7B_PREREG_LOCKED = True
LOCK_BLOCKERS         = ()
HELDOUT_RUN_STATUS    = READY_NOT_RUN
RESULTS_STATUS        = NOT_RUN
```

The corpus generator and evaluator remain built, not run as a detector outcome:

```text
scripts/boxsel_phase7b_corpus.py
scripts/boxsel_phase7b_evaluator.py
```

## Diagnostic Receipt

The unit receipt checks only synthetic and seen-diagnostic trace shapes:

```text
stable PMP pressure response -> abstain
pressure-noop point control  -> accept
low-support trace            -> widen
ordinary endpoint drift      -> widen
loss escape                  -> abstain
constraint violation         -> abstain
seed/dimension disagreement  -> widen
```

It does not score the Phase-7b held-out corpus. That run remains the next separate step.

## Verification

```text
python scripts/test_boxsel_phase7b_v2_detector.py
python scripts/test_boxsel_phase7b_prereg.py
python scripts/test_boxsel_phase7b_corpus_evaluator.py
```

Results:

```text
16/16 checks pass, exit 0.
30/30 checks pass, exit 0.
24/24 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-7b v2 freeze. Internal; detector locked, held-out run not executed.*
