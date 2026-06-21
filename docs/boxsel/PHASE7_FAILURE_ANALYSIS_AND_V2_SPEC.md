# BoxSEL Phase 7 Failure Analysis And V2 Spec

**Date:** 2026-06-21  
**Status:** Phase-7 bounded null analyzed; Phase-6b trace schema/adapters started; Phase-7b v2 detector frozen; Phase-7b run passed on the toy micro-corpus.

## Boundary

This note records the redesign after the failed Phase-7 run and the subsequent Phase-7b pass. It
does not change the failed Phase-7 detector or convert the toy micro-corpus result into a real-KG,
calibration, or product claim.

The Phase-7 result remains:

```text
FAIL_PREREG_GATE
```

with:

```text
accepted false closures : 4 / 10 = 0.40
baseline accepted       : 4 / 10 = 0.40
baseline improvement    : 0.00
triggered kills         : KILL7-1, KILL7-2
```

The failed run is documented in:

```text
docs/boxsel/PHASE7_FALSE_CLOSURE_RUN.md
```

## What Failed

The first guard was good at detecting visible boundary symptoms:

```text
endpoint movement
active constraint slack
seed variance
dimension sensitivity
loss escape
```

It was bad at detecting stable false closure:

```text
low loss
high slack
stable endpoints
narrow seed disagreement
ordinary restarts all agree
exact lower endpoint still substantially lower
```

The run split:

```text
6 / 6 Helly seed-variant false closures  -> abstain
4 / 4 stable PMP-shaped false closures   -> accept
4 / 4 true-narrow controls               -> accept
2 / 2 loss-escape controls               -> abstain
```

So the first rule detected **turbulent false closure**, not **quiet false closure**.

## Seen Cases

All Phase-7 cases are now diagnostic/training cases only:

```text
helly-00 ... helly-05
pmp-00 ... pmp-03
narrow-00 ... narrow-03
loss-00 ... loss-01
```

They may be used to design and debug v2 features. They must not be reused as held-out validation
for Phase 7b.

## Missing Signal

The missing signal is not restart variance. The stable PMP failures had no warning flags under the
Phase-6 feature set. A v2 guard needs a way to ask whether the sampled lower endpoint is stable
because the model space is genuinely concentrated, or merely because ordinary restarts never
searched the endpoint-sensitive direction.

Candidate trace-only signals:

```text
per-axiom slack, not just one Helly pairwise slack
condition denominator mass
query numerator mass
support floors for atoms and meets
ordinary-vs-query-pressure endpoint movement
ordinary-vs-extremal optimizer disagreement
optimizer-mode lower spread
constraint violation under pressure
support collapse under pressure
```

The critical addition is **pressure response**:

```text
ordinary restart endpoint is stable
but query-conditioned lower pressure moves it
```

This is not exact inference. It is still trace-only: the pressure run does not read `I*`, the exact
`I_box` endpoint, or the Phase-4 closed form. It only asks whether the sampled embedding basin has
a hidden downhill direction for the query.

## Phase 6b Deliverable

Build a general trace schema before writing another detector:

```text
scripts/boxsel_phase6b_trace_schema.py
```

The schema must separate:

```text
endpoint observations
per-constraint observations
support observations
seed/dimension/optimizer comparisons
query-pressure observations
```

It must not include evaluator-only fields such as exact endpoints, oracle labels, or Phase-4 closed
forms.

Current scaffold:

```text
EndpointObservation
ConstraintTrace
SupportTrace
GeneralTrace
GeneralTraceFeatures
```

Current adapters/producers:

```text
phase3_report_to_general_trace
phase7_case_to_general_trace
phase7_general_traces
pmp_query_pressure_trace
phase7_pmp_pressure_traces
```

The adapters now convert:

```text
Phase-3 Helly sampler reports -> GeneralTrace
Phase-7 result rows           -> GeneralTrace diagnostics
stable PMP failures           -> query-pressure traces
```

The pressure producer is diagnostic only. It exposes ordinary-vs-query-pressure movement for the
seen stable PMP failure class; it is not a detector threshold and not a Phase-7b held-out result.

Feature names are intentionally oracle-free:

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

## V2 Detector Rule Shape

No thresholds are frozen yet. The candidate rule should be drafted only after the schema can ingest
both Helly-style and PMP-style traces.

A plausible v2 guard:

```text
loss escape                         -> abstain
constraint violation under pressure -> abstain
large pressure_low_shift            -> widen / abstain
optimizer_low_spread                -> widen / abstain
support_floor collapse              -> widen / abstain
ordinary-only turbulence            -> widen / abstain
no warnings                         -> accept
```

The point is not to make the detector more nervous everywhere. The true-narrow controls in Phase 7
were useful: an always-widen detector is not acceptable.

## Phase 7b Prereg Requirements

Frozen before any v2 held-out run:

- freeze the v2 detector and thresholds;
- keep the new held-out seed list disjoint from Phase 3, Phase 6, and Phase 7;
- exclude all Phase-7 cases from validation;
- include stable false-closure traps as a named family;
- keep true-narrow and loss-escape controls;
- keep `restart_variance_only_v0` or name a stricter baseline;
- preserve the same primary kill metric: accepted false-closure rate.

Phase 7b now has a locked preregistration receipt:

```text
docs/boxsel/PHASE7B_FALSE_CLOSURE_PREREG_START.md
```

The corpus generator and evaluator now have an infrastructure receipt:

```text
docs/boxsel/PHASE7B_CORPUS_EVALUATOR_START.md
```

That clears the `PHASE7B_CORPUS_GENERATOR_NOT_BUILT` and `PHASE7B_EVALUATOR_NOT_BUILT` blockers.

The v2 freeze receipt is:

```text
docs/boxsel/PHASE7B_V2_FREEZE_LOCK.md
```

It clears the v2 detector and threshold blockers:

```text
LOCK_BLOCKERS = ()
PHASE7B_PREREG_STATUS = LOCKED_NOT_RUN
HELDOUT_RUN_STATUS = READY_NOT_RUN
```

Phase 7b has a result note:

```text
docs/boxsel/PHASE7B_FALSE_CLOSURE_RUN.md
```

The result is:

```text
PASS_PREREG_GATE
accepted false closures : 0 / 16 = 0.00
baseline accepted       : 16 / 16 = 1.00
baseline improvement    : 1.00
```

This unblocks Phase 8 only inside the toy micro-SEL workbench boundary. It is not a real-KG,
calibration, or Ask Sundog product claim.

## Claim Language

Allowed now:

> The first trace-only guard caught Helly-style false closures but failed the preregistered stable
> false-closure traps. The next design target is a general trace schema plus pressure-response
> signals that remain oracle-free.

Allowed after Phase 7b:

> On the locked tiny Phase-7b micro-corpus, the frozen v2 trace detector accepted 0/16
> false-closure traps while the restart-variance baseline accepted 16/16.

Forbidden:

- "Phase 7 almost passed."
- "The detector works on real KGs."
- "Phase 8 product/public claims are unblocked."
- "Pressure response is exact inference."

## Artifacts

- `scripts/boxsel_phase6b_trace_schema.py`
- `scripts/test_boxsel_phase6b_trace_schema.py`
- `scripts/boxsel_phase7b_v2_detector.py`
- `scripts/test_boxsel_phase7b_v2_detector.py`
- `scripts/boxsel_phase7b_corpus.py`
- `scripts/boxsel_phase7b_evaluator.py`
- `scripts/test_boxsel_phase7b_corpus_evaluator.py`
- `scripts/boxsel_phase7b_run.py`
- `scripts/test_boxsel_phase7b_run.py`

Verification:

```text
python scripts/test_boxsel_phase6b_trace_schema.py
python scripts/test_boxsel_phase7b_v2_detector.py
python scripts/test_boxsel_phase7b_corpus_evaluator.py
python scripts/test_boxsel_phase7b_run.py
```

Result:

```text
39/39 checks pass, exit 0.
16/16 checks pass, exit 0.
24/24 checks pass, exit 0.
24/24 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-7 failure analysis and Phase-6b v2 schema start.*
