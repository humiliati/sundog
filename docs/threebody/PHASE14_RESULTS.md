# Three-Body Phase 14 - Mechanism Decomposition Result Note

This document records Phase 14 results for
[`PHASE14_SPEC.md`](PHASE14_SPEC.md). Phase 14 decomposes the Phase 13
guarded-TRACK survival win into warning quality, action coupling, and outcome
effect.

Status: full lock complete. The spec's exact Phase 13 regression gate
(`npm run threebody:phase13`, unchanged command) was rerun this session and
reproduced the Phase 13 lock bit-for-bit (3,456 trials; 88 / 324 candidate
envelope rows; 81 promising best cells; terminal outcomes 1,154 bounded /
2,030 escape / 272 close approach, matching `PHASE13_RESULTS.md`), so the
shared-harness edit is verified non-perturbing. Remaining caveats are
scientific, not gate-related: see Sections 5 and 8.

Cross-doc verification (2026-05-15): Phase 14 numbers confirmed consistent
across `PHASE14_RESULTS.md`, `SUNDOG_V_THREEBODY.md`, and `threebody-writeup.md`
and matching the lock result files (130 / 648 envelope rows; guarded TRACK
77/108 · 71/81; signal delay 48/108 · 36/81; action shuffle 3/108 · 0/81;
signal shuffle 2/108 · 0/81; sign flip 0/108 · 0/81). The favorable-pocket
passive tidal AUROC ≈ 0.006 is a genuine decidable failure (81/81 favorable
cells have defined AUROC; the high 0.38–0.47 values sit in the
`velocityScale=0.95` boundary column, not the favorable pocket), not a
coverage/undecidable artifact.

## 1. Full Lock

Command:

```bash
npm run threebody:phase14
```

Output:

- `results/threebody/phase14-mechanism-decomposition-lock/`
- 6,048 trials
- 5,184 paired non-passive rows
- Terminal outcomes from the runner: 1,269 bounded, 4,616 escape, 163 close
  approach
- Candidate envelope rows: 130 / 648

The run used seven executed modes: `off`, `naive`,
`track_sensor_accel_guarded`, `track_sensor_accel_signal_shuffle`,
`track_sensor_accel_action_shuffle`, `track_sensor_accel_signal_delay`, and
`track_sensor_accel_sign_flip`.

## 2. Regression Check

The spec's exact regression gate (`npm run threebody:phase13`, unchanged
command) was rerun after the shared-harness edit and reproduced the Phase 13
lock bit-for-bit:

- 3,456 trials
- 88 / 324 candidate envelope rows
- 81 promising best cells
- terminal outcomes 1,154 bounded / 2,030 escape / 272 close approach

These match `PHASE13_RESULTS.md` exactly, so the additive ablation modes and the
flag-gated action-coupling instrumentation did not perturb the frozen guard or
the Phase 13 code path. Log: `results/phase13-regression.log`.

The shared-mode aggregate subset is also internally consistent between the two
runs:

| mode | Phase 13 candidates | Phase 14 candidates | class balance | mean survival delta | mean time delta |
| --- | ---: | ---: | --- | ---: | ---: |
| naive | 0 | 0 | 57 neutral / 24 risky / 27 negative | -0.166667 | -2.786674 |
| guarded TRACK | 77 | 77 | 77 promising / 19 mixed / 7 risky / 5 negative | 0.700231 | 8.998706 |

## 3. Per-Arm Outcome Table

| mode | candidates | promising | mixed | neutral | risky | negative | mean survival delta | mean time delta | mean delta-v |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 0 / 108 | 0 | 0 | 57 | 24 | 27 | -0.167 | -2.787 | 0.872 |
| guarded TRACK | 77 / 108 | 77 | 19 | 0 | 7 | 5 | 0.700 | 8.999 | 3.435 |
| signal delay | 48 / 108 | 48 | 16 | 30 | 3 | 11 | 0.167 | 5.472 | 3.292 |
| action shuffle | 3 / 108 | 3 | 0 | 57 | 20 | 28 | -0.123 | 0.147 | 0.445 |
| signal shuffle | 2 / 108 | 2 | 0 | 62 | 20 | 24 | -0.116 | 0.106 | 0.425 |
| sign flip | 0 / 108 | 0 | 0 | 57 | 21 | 30 | -0.160 | -2.763 | 0.794 |

Best-cell read across all Phase 14 arms:

- Guarded TRACK is best in 84 / 108 cells.
- Signal delay is best in 24 / 108 cells.
- Best-cell class balance is 92 promising, 13 mixed, and 3 negative.

## 4. Favorable Pocket

Using the high-velocity pocket (`velocityScale >= 1.05`) as the favorable
region:

| mode | candidates | promising | mixed | risky | negative | mean survival delta | action agreement | signed effect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 0 / 81 | 0 | 0 | 12 | 27 | -0.130 | 0.059 | -0.794 |
| guarded TRACK | 71 / 81 | 71 | 10 | 0 | 0 | 0.790 | 1.000 | 1.000 |
| signal delay | 36 / 81 | 36 | 10 | 3 | 5 | 0.153 | 1.000 | 0.969 |
| action shuffle | 0 / 81 | 0 | 0 | 9 | 27 | -0.102 | 0.942 | 0.818 |
| signal shuffle | 0 / 81 | 0 | 0 | 9 | 23 | -0.096 | 0.942 | 0.814 |
| sign flip | 0 / 81 | 0 | 0 | 12 | 27 | -0.120 | 0.000 | -1.000 |

Read: the intact guarded arm still dominates the favorable pocket, and the
action-shuffled, signal-shuffled, and sign-flipped arms do not reproduce that
benefit. This weakens the guard-only explanation.

The delay arm is the important caveat. A 0.5-second signal delay retains 36
candidate rows in the favorable pocket and 48 overall. That means the mechanism
is not a brittle instantaneous-signal handle; it has meaningful temporal
tolerance or hysteresis.

## 5. Warning Quality

The pre-registered high-warning-quality bar does not pass.

- Favorable high-velocity aggregate coverage: 81 / 81 cells have defined
  passive tidal-magnitude AUROC.
- Favorable high-velocity mean passive tidal-magnitude AUROC: 0.006.
- Favorable high-velocity paired-row coverage: 564 / 648 rows have defined
  passive tidal-magnitude AUROC.
- Favorable high-velocity paired-row mean passive tidal-magnitude AUROC: 0.005.
- The pre-registered pass bar was `>= 0.70`.

This blocks a clean Phase 14 pass. It may mean the passive tidal AUROC metric is
the wrong warning-quality readout for the Phase 13 pocket, or that the useful
control handle is not captured by that scalar warning framing. Either way, the
Phase 14 mechanism claim cannot rest on "high passive warning quality" as
specified.

## 6. Action Coupling

The action-coupling metric strongly separates sign-flip from intact control, but
it does not cleanly separate intact control from shuffled historical actions:

- Guarded TRACK: mean agreement 1.000, signed effect 1.000.
- Sign flip: mean agreement 0.000, signed effect -1.000.
- Action shuffle: mean agreement 0.915 overall and 0.942 in the favorable
  pocket, despite losing the candidate envelope.
- Signal shuffle: mean agreement 0.915 overall and 0.942 in the favorable
  pocket, despite losing the candidate envelope.

Interpretation: outcome, not agreement alone, is decisive here. Historical or
shuffled thrust can still align with the smooth local gradient often enough to
score well on the agreement metric, but it does not preserve the survival
benefit. Future mechanism work should add a timing-sensitive coupling metric or
a per-step counterfactual effect score.

## 7. Boundary Read

In the best-cell table, the only negative cells are equal-mass
`velocityScale=0.95` cases at `radiusScale=1.05`, across timesteps `0.008`,
`0.01`, and `0.012`; the signal-delay arm is the best available arm in those
negative cells. This does not erase the Phase 13 boundary. It sharpens the
diagnosis: adding ablations can find better boundary behavior in some cells,
but the low-velocity equal-mass region remains outside the clean positive
pocket.

## 8. Branch

Pre-registered branch: **provisional partial / mechanism narrowed**.

Phase 14 does **not** support the clean pass branch because:

- passive tidal AUROC fails the high-warning-quality bar;
- signal delay retains a substantial share of the candidate envelope;
- action-coupling agreement alone does not drop toward chance for shuffled arms.

Phase 14 also does **not** support the guard-only fail branch because:

- action shuffle and signal shuffle lose nearly all candidate rows;
- sign flip destroys the candidate envelope and inverts signed coupling;
- intact guarded TRACK remains the only strong arm in the favorable pocket.

Preserved claim:

> In the tested planar restricted setup, a guarded accelerometer-proxy TRACK
> controller improves survival over passive and naive local baselines across a
> mapped high-velocity near-escape pocket through a 16-second tested horizon.

Narrowed mechanism read:

> The Phase 13 win depends on signal-directed thrust timing and sign, not merely
> on the guard suppressing bad thrust. However, the useful handle is not
> established as a clean instantaneous passive-warning signal: the
> pre-registered passive AUROC bar fails, and a 0.5-second delayed signal keeps
> a substantial fraction of the benefit.
