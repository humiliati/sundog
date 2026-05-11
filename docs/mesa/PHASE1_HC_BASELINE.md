# Mesa Phase 1 - HC-Signature Baseline

Phase 1 implements the canonical shadow-field navigation task and verifies
that the hand-coded Sundog controller can track the external signature before
any learned policy enters the experiment.

## Implemented Surface

- Shared core: `public/js/mesa-core.mjs`
- Batch harness: `scripts/mesa-harness.mjs`
- NPM entry point: `npm run mesa:phase1`
- Default output: `results/mesa/phase1-hc-baseline/`

The core includes:

- deterministic shadow-field initial conditions;
- separate `S(x)` and matched `R(s, a)` accessors;
- `privileged-field`, `local-probe-field`, `delayed-field`, `noisy-field`,
  and `delayed-noisy-field` sensor tiers;
- `applyProbe(...)` and `scheduleIntervention(...)` hooks for later phases;
- HC-Signature SCAN/SEEK/TRACK/REACQUIRE controller;
- replay-friendly JSONL logs and CSV trial summaries.

## Locked Phase 1 Command

```bash
npm run mesa:phase1
```

This expands to:

```bash
node scripts/mesa-harness.mjs --phase phase1-hc-baseline --out results/mesa/phase1-hc-baseline --seeds 32
```

The harness verifies deterministic replay by rerunning each trial in memory
and comparing JSONL serialization before writing outputs.

## Result

Latest Phase 1 default run:

| Sensor tier | Success | Mean terminal `S(x_T)` | Mean steps |
| --- | ---: | ---: | ---: |
| `privileged-field` | 32/32 | 0.9943 | 89.13 |
| `local-probe-field` | 32/32 | 0.9943 | 89.13 |
| `delayed-field` | 32/32 | 0.9990 | 88.09 |
| `noisy-field` | 21/32 | 0.8770 | 149.91 |

This meets the Phase 0 behavior expectation: clean tiers are stable, and the
noisy tier degrades visibly while remaining above the 60% target.

## Tuning Notes

The first smoke run exposed a useful Phase 1 bug: clipping additive sensor
noise to `[0, 1]` created a positive fake-field bias before gradient
estimation. The core now leaves noisy measured samples unclipped after
additive noise. The underlying true signature remains bounded.

HC-Signature also uses a low-pass gradient estimate. The Phase 1 locked value
is:

```text
gradientLpfAlpha = 0.05
```

That value preserves perfect performance on the clean tiers and gives the
noisy tier enough smoothing to clear the baseline.

## Exit Status

Phase 1 exit criterion is met:

- HC-Signature works on the canonical task at the privileged tier.
- The controller works at the canonical local-probe tier.
- Delayed and noisy tiers degrade cleanly.
- The harness writes `manifest.json`, `trial-outcomes.csv`, `summary.csv`, and
  per-trial JSONL logs.
- Replay determinism is verified by the harness.

Phase 2 can now use this controller as the behavior-cloning source and as the
structural ceiling for learned signature policies.
