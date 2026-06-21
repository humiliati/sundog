# BoxSEL Phase 7 - False-Closure Run

**Date:** 2026-06-21  
**Status:** `FAIL_PREREG_GATE`

This is the result receipt for the locked preregistration:

```text
docs/boxsel/PHASE7_FALSE_CLOSURE_PREREG.md
```

The prereg file remains a pre-result artifact: `RESULTS_STATUS = NOT_RUN`. Outcomes live only in
the run harness and result manifest.

## Verdict

```text
FAIL_PREREG_GATE
```

The Phase-6 trace guard accepted too many held-out false closures and did not beat the
restart-variance-only baseline.

```text
detector accepted false closures : 4 / 10 = 0.40
baseline accepted false closures : 4 / 10 = 0.40
baseline improvement             : 0.00
true-narrow accept rate           : 4 / 4 = 1.00
loss-escape accepted              : 0 / 2
```

Triggered kill criteria:

```text
KILL7-1  accepted false-closure rate is 25% or higher on held-out traps
KILL7-2  guard fails to beat restart_variance_only_v0 by 20 percentage points
```

Supported predictions:

```text
P7-P3  stable true-narrow controls are accepted at least 50% of the time
P7-P4  no loss-escape control is accepted
```

Unsupported predictions:

```text
P7-P1  guard widens/abstains on at least 90% of held-out false-closure traps
P7-P2  accepted false-closure rate beats baseline by 20 percentage points
```

## Failure Shape

The split is crisp:

```text
6 / 6 Helly seed-variant false closures       -> abstain
4 / 4 stable PMP-shaped false closures        -> accept
4 / 4 true-narrow controls                    -> accept
2 / 2 loss-escape controls                    -> abstain
```

The guard detects false closure when the trace has visible boundary symptoms: endpoint movement,
active constraint slack, seed variance, or dimension sensitivity. It fails on stable, low-loss,
high-slack PMP-shaped false closures where the sampled interval is narrow but the exact lower
endpoint remains substantially lower.

That is exactly the preregistered falsifier class.

## Case Ledger

```text
case       family                        action    false_closed  gap
helly-00   helly_threshold_variants      abstain   yes           0.530829
helly-01   helly_threshold_variants      abstain   yes           0.589853
helly-02   helly_threshold_variants      abstain   yes           0.513243
helly-03   helly_threshold_variants      abstain   yes           0.543862
helly-04   helly_threshold_variants      abstain   yes           0.529412
helly-05   helly_threshold_variants      abstain   yes           0.530366
pmp-00     pmp_interval_chain_variants   accept    yes           0.154778
pmp-01     pmp_interval_chain_variants   accept    yes           0.147000
pmp-02     pmp_interval_chain_variants   accept    yes           0.147000
pmp-03     pmp_interval_chain_variants   accept    yes           0.137000
narrow-00  true_narrow_controls          accept    no           -0.000500
narrow-01  true_narrow_controls          accept    no           -0.000500
narrow-02  true_narrow_controls          accept    no           -0.000500
narrow-03  true_narrow_controls          accept    no           -0.000500
loss-00    loss_escape_controls          abstain   no           -0.000500
loss-01    loss_escape_controls          abstain   no           -0.000500
```

## Boundary Notes

Two limitations are recorded in the manifest and matter for the next design:

- The Phase-6 trace interface is still Helly-shaped; non-Helly synthetic traces encode minimum
  constraint slack through the existing `min_slack` channel.
- The Helly held-outs are seed variants of the Phase-3 threshold case, not distinct generated
  threshold levels.

These limitations do not rescue the detector result. The preregistered stable false-closure class
still did the thing it was supposed to do: it found a trace shape the first guard accepts.

## Artifacts

- `scripts/boxsel_phase7_run.py`
- `scripts/test_boxsel_phase7_run.py`
- `results/boxsel/phase7_false_closure_run/manifest.json`

Verification:

```text
python scripts/boxsel_phase7_run.py
python scripts/test_boxsel_phase7_run.py
```

Result:

```text
24/24 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-7 result. Internal bounded-null receipt; Phase 8 remains gated.*
