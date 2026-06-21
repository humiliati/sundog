# BoxSEL Phase 7b - False-Closure Run

**Date:** 2026-06-21  
**Status:** `PASS_PREREG_GATE`

This is the result receipt for the locked Phase-7b preregistration:

```text
docs/boxsel/PHASE7B_FALSE_CLOSURE_PREREG_START.md
docs/boxsel/PHASE7B_V2_FREEZE_LOCK.md
```

The prereg file remains a pre-result lock: `RESULTS_STATUS = NOT_RUN`. Outcomes live only in the
run harness and result manifest.

## Verdict

```text
PASS_PREREG_GATE
```

The frozen v2 trace detector caught all held-out false closures and beat the
restart-variance-only baseline.

```text
detector accepted false closures : 0 / 16 = 0.00
baseline accepted false closures : 16 / 16 = 1.00
baseline improvement             : 1.00
true-narrow accept rate           : 9 / 9 = 1.00
loss-escape accepted              : 0 / 3
stable-PMP pressure warning rate  : 8 / 8 = 1.00
baseline pressure warning rate    : 0 / 8 = 0.00
```

Triggered kill criteria:

```text
none
```

Supported predictions:

```text
P7B-P1  v2 widens or abstains on at least 90% of held-out false-closure traps
P7B-P2  accepted false-closure rate beats restart_variance_only_v0 by at least 20 percentage points
P7B-P3  stable true-narrow controls are accepted at least 50% of the time
P7B-P4  no loss-escape control is accepted
P7B-P5  stable PMP pressure variants trigger pressure/optimizer warnings more often than restart variance
```

## Run Shape

```text
8 / 8 stable PMP pressure false closures  -> abstain
4 / 4 Helly threshold false closures      -> widen
4 / 4 support-floor false closures        -> abstain
9 / 9 acceptance controls                 -> accept
3 / 3 loss-escape controls                -> abstain
```

The Phase-7 failure class was the stable PMP-shaped false closure. In Phase 7b, all eight stable
PMP pressure variants fired `pressure_low_shift` and `optimizer_low_spread`; restart variance fired
on none of them.

## Case Ledger

```text
case                    family                         action    false_closed  gap       flags
p7b-stable-pmp-00       stable_pmp_pressure_variants    abstain   yes           0.154778  pressure_low_shift,optimizer_low_spread
p7b-stable-pmp-01       stable_pmp_pressure_variants    abstain   yes           0.147000  pressure_low_shift,optimizer_low_spread
p7b-stable-pmp-02       stable_pmp_pressure_variants    abstain   yes           0.147000  pressure_low_shift,optimizer_low_spread
p7b-stable-pmp-03       stable_pmp_pressure_variants    abstain   yes           0.137000  pressure_low_shift,optimizer_low_spread
p7b-stable-pmp-04       stable_pmp_pressure_variants    abstain   yes           0.145750  pressure_low_shift,optimizer_low_spread
p7b-stable-pmp-05       stable_pmp_pressure_variants    abstain   yes           0.151286  pressure_low_shift,optimizer_low_spread
p7b-stable-pmp-06       stable_pmp_pressure_variants    abstain   yes           0.157000  pressure_low_shift,optimizer_low_spread
p7b-stable-pmp-07       stable_pmp_pressure_variants    abstain   yes           0.144778  pressure_low_shift,optimizer_low_spread
p7b-helly-00            helly_threshold_variants_v2     widen     yes           0.546299  early_lower_drop
p7b-helly-01            helly_threshold_variants_v2     widen     yes           0.568699  early_lower_drop
p7b-helly-02            helly_threshold_variants_v2     widen     yes           0.545777  early_lower_drop,late_lower_drop
p7b-helly-03            helly_threshold_variants_v2     widen     yes           0.530664  late_lower_drop
p7b-support-00          support_floor_variants          abstain   yes           0.437000  pressure_low_shift,optimizer_low_spread,support_floor
p7b-support-01          support_floor_variants          abstain   yes           0.457000  pressure_low_shift,optimizer_low_spread,support_floor
p7b-support-02          support_floor_variants          abstain   yes           0.477000  pressure_low_shift,optimizer_low_spread,support_floor
p7b-support-03          support_floor_variants          abstain   yes           0.497000  pressure_low_shift,optimizer_low_spread,support_floor
p7b-narrow-00           true_narrow_controls_v2         accept    no           -0.000500  -
p7b-narrow-01           true_narrow_controls_v2         accept    no           -0.000500  -
p7b-narrow-02           true_narrow_controls_v2         accept    no           -0.000500  -
p7b-narrow-03           true_narrow_controls_v2         accept    no           -0.000500  -
p7b-narrow-04           true_narrow_controls_v2         accept    no           -0.000500  -
p7b-narrow-05           true_narrow_controls_v2         accept    no           -0.000500  -
p7b-pressure-noop-00    pressure_noop_controls          accept    no           -0.000500  -
p7b-pressure-noop-01    pressure_noop_controls          accept    no           -0.000500  -
p7b-pressure-noop-02    pressure_noop_controls          accept    no           -0.000500  -
p7b-loss-00             loss_escape_controls_v2         abstain   no           -0.000500  max_loss
p7b-loss-01             loss_escape_controls_v2         abstain   no           -0.000500  max_loss
p7b-loss-02             loss_escape_controls_v2         abstain   no           -0.000500  max_loss
```

## Boundary Notes

This is a pass on the locked tiny role-free micro-SEL corpus. It is not a real-KG claim, not an Ask
Sundog product claim, and not a calibration guarantee.

The v2 pressure traces are deterministic query-pressure probes, not exact inference. The exact
oracle labels are evaluator-only and are computed after the detector decisions.

## Artifacts

- `scripts/boxsel_phase7b_run.py`
- `scripts/test_boxsel_phase7b_run.py`
- `results/boxsel/phase7b_false_closure_run/manifest.json`

Verification:

```text
python scripts/boxsel_phase7b_run.py
python scripts/test_boxsel_phase7b_run.py
```

Result:

```text
24/24 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-7b result. Internal pass receipt; toy micro-SEL boundary still applies.*
