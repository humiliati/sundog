# BoxSEL Phase 7b - Corpus Generator And Evaluator Start

**Date:** 2026-06-21  
**Status:** Corpus generator and evaluator built; v2 detector subsequently frozen.

This clears two Phase-7b prereg-start blockers:

```text
PHASE7B_CORPUS_GENERATOR_NOT_BUILT  -> cleared
PHASE7B_EVALUATOR_NOT_BUILT         -> cleared
```

At this infrastructure step, the detector rule and thresholds were not yet frozen.

```text
detector rule pending
thresholds pending
```

Those blockers were later cleared by `docs/boxsel/PHASE7B_V2_FREEZE_LOCK.md`. No held-out result
claim is made in this infrastructure note.

## Corpus Generator

Artifact:

```text
scripts/boxsel_phase7b_corpus.py
```

The generator builds the prereg-start corpus:

```text
8  stable_pmp_pressure_variants   false_closure_trap
4  helly_threshold_variants_v2    false_closure_trap
4  support_floor_variants         false_closure_trap
6  true_narrow_controls_v2        acceptance_control
3  pressure_noop_controls         acceptance_control
3  loss_escape_controls_v2        loss_control
---
28 cases
16 false-closure traps
9 acceptance controls
3 loss controls
```

It enforces:

- no Phase-7 case IDs reused;
- all seeds drawn from the reserved Phase-7b pool;
- stable PMP traces are quiet under ordinary restarts but move under query pressure;
- support-floor traces have low support and pressure movement;
- loss controls carry nonzero loss.

## Evaluator

Artifact:

```text
scripts/boxsel_phase7b_evaluator.py
```

The evaluator can:

- compute exact labels from the tiny exact oracle;
- compute `false_closed` from exact lower endpoint widening and low-loss status;
- score future detector actions `accept / widen / abstain`;
- score the `restart_variance_only_v0` baseline;
- summarize accepted false closures, true-narrow accept rate, loss-escape accepts, and pressure
  warning rate on stable PMP cases.

The evaluator does not contain a v2 detector rule.

## Receipt

The current generated corpus labels as intended:

```text
false-closure traps       : 16 / 16 false_closed
acceptance controls       : 0 / 9 false_closed
loss controls             : 0 / 3 false_closed, all non-low-loss
```

The restart-variance baseline accepts the stable PMP pressure variants, which preserves the Phase-7
failure target for the future v2 detector.

## Verification

```text
python scripts/test_boxsel_phase7b_corpus_evaluator.py
python scripts/test_boxsel_phase7b_prereg.py
```

Results:

```text
24/24 checks pass, exit 0.
30/30 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-7b corpus/evaluator infrastructure. Internal; not a held-out detector run.*
