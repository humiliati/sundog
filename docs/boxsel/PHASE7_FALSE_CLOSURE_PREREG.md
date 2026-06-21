# BoxSEL Phase 7 - False-Closure Preregistration

**Date:** 2026-06-21  
**Status:** Prereg locked. Held-out runs **not run**.

## Purpose

Phase 7 is the kill-gate for the trace detector:

```text
Can trace-only signals flag false closure on held-out tiny ontologies?
```

The detector must make its decision without `I*`, without the exact `I_box` endpoint, and without
the Phase-4 closed form. Exact inference is used only after the decision, to score whether the case
was truly false-closed.

## Frozen Detector

Detector version:

```text
phase6_trace_detector_start
```

Frozen trace features:

```text
sample_lower
sample_upper
sample_width
early_lower_drop
late_lower_drop
max_loss
min_slack
seed_low_range
dimension_low_spread
```

Frozen thresholds:

```text
loss_escape              : max_loss > 1e-9
endpoint_drift           : early lower-endpoint drop > 0.05
late_endpoint_drift      : late lower-endpoint drop > 0.01
active_constraint_slack  : min pairwise slack < 0.005
seed_variance            : sampled lower range across seeds > 0.02
dimension_sensitivity    : sampled lower spread across dimensions > 0.02
false_closure_gap        : lower search gap > 0.05
```

Decision rule:

```text
loss_escape or >=3 flags  -> abstain
1-2 flags                 -> widen
0 flags                   -> accept
```

No threshold, feature, or rule changes are allowed after held-out results are generated.

## Held-Out Corpus Plan

Reserved held-out seeds:

```text
7001, 7003, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7121
```

These are disjoint from the Phase-3/Phase-6 seed-trap seeds:

```text
314159, 271828, 11, 23, 37, 53, 101
```

Planned families:

```text
6  helly_threshold_variants      false_closure_trap
4  pmp_interval_chain_variants   false_closure_trap
4  true_narrow_controls          acceptance_control
2  loss_escape_controls          loss_control
---
16 held-out cases total
10 false-closure traps
```

Family definitions:

- `helly_threshold_variants`: role-free three-atom Helly variants with atom marginals near `1/2`
  and pairwise lower constraints chosen before running. The exact oracle must show a substantially
  wider lower endpoint than ordinary restart sampling.
- `pmp_interval_chain_variants`: small PMP-shaped role-free ontologies generated from frozen
  rational premise intervals. The exact oracle labels interval width before detector scoring.
- `true_narrow_controls`: tiny ontologies whose exact query interval is already narrow, used to
  prevent an always-widen or always-abstain detector.
- `loss_escape_controls`: near-feasible runs with nonzero loss; the guard must abstain rather than
  accept.

## Baseline

Baseline version:

```text
restart_variance_only_v0
```

This baseline may use only restart lower-endpoint variation across seeds. It may not use:

- dimension sensitivity,
- constraint slack,
- endpoint movement over restarts,
- loss/query-gradient features,
- `I*`,
- exact `I_box`.

The Phase-6 guard must beat this baseline by the preregistered margin below.

## Labels And Scoring

A held-out case is a false-closure trap iff:

```text
exact lower endpoint is at least 0.10 below the sampled lower endpoint
```

and the ordinary sampled runs are low-loss/nested. The exact oracle label is computed after the
trace-only decision is made.

Primary metric:

```text
accepted_false_closure_rate
```

where an accepted false closure is a case labeled false-closed by the oracle evaluator while the
guard action is `accept`.

Secondary metrics:

```text
baseline improvement over restart_variance_only_v0
true_narrow_accept_rate
loss_escape_accept_count
```

## Predictions

```text
P7-P1: the trace guard will widen or abstain on at least 90% of held-out false-closure traps.
P7-P2: accepted false-closure rate will be at least 20 percentage points lower than the restart-variance-only baseline.
P7-P3: stable true-narrow controls will be accepted at least 50% of the time.
P7-P4: no loss-escape control may be accepted.
```

## Kill Criteria

```text
KILL7-1: accepted false-closure rate is 25% or higher on held-out traps.
KILL7-2: the guard fails to beat the restart-variance-only baseline by 20 percentage points.
KILL7-3: any loss-escape control is accepted.
KILL7-4: thresholds or feature definitions are changed after held-out results are generated.
```

Passing this phase would support only the bounded claim in the roadmap: on small exact-oracle
fragments, trace signals flagged false closure better than restart variance alone. It would not be
a calibration guarantee, a real-KG claim, or an Ask Sundog product claim.

## Results

```text
NOT_RUN
```

This prereg contains no held-out outcomes.

## Artifacts

- `scripts/boxsel_phase7_prereg.py`
- `scripts/test_boxsel_phase7_prereg.py`

Verification:

```text
python scripts/test_boxsel_phase7_prereg.py
```

Result:

```text
18/18 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-7 preregistration. Internal; protocol locked before held-out
runs.*
