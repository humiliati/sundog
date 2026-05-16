# Copilot Cloud-Agent Test-Run Brief — threebody:phaseNN

Purpose: measure whether a GitHub cloud sandbox can run the long Threebody
experiment harness materially faster (or just usefully offloaded) versus the
local 2–5 h (now measured: up to ~75 h) waits, so more elaborate
claims/ratchets do not bottleneck on the project machine.

This is an **experiment about runtime**, not a license to interpret results.
Read "Discipline boundary" before running anything.

## Measured local baselines (this machine, single-thread Node)

From the run `manifest.json` `startedAt`/`completedAt` (exact wall-clock, not
estimates):

| run | trials | wall-clock | s/trial | dominant cost |
| --- | ---: | ---: | ---: | --- |
| `npm run threebody:phase13` | 3,456 | 48.7 min | 0.845 | `oracle` mode (9 cand × 16-step lookahead) |
| `npm run threebody:phase14` | 6,048 | 4.9 min | 0.049 | no oracle; cheap ablations |
| `npm run threebody:phase15:smoke` | 144 | 57.3 min | 23.86 | `forward_oracle_strict` (9 × 32 × 8 substeps) + per-step counterfactual strict-oracle call + dt=0.004 (4,000 steps/trial) |

Takeaway: cost is dominated by oracle-class lookahead, not trial count
(phase14 has 1.75× the trials of phase13 but ran in 1/10 the time because it
has no oracle).

## Local estimate for the Phase 15 full lock (NOT yet run)

`npm run threebody:phase15` = 12,960 trials over the 5-step ladder
{0.004,0.006,0.008,0.01,0.012}, 9 modes, 8 seeds, duration 16,
`--track-action-coupling 1 --precision-receipts 1`.

Extrapolation from the smoke (the representative heavy path: same 9 modes,
precision receipts on), scaled by the average steps/trial ratio
(`avg(1/dt)` full 145.0 vs smoke 166.7 ⇒ ×0.87):

- ≈ 23.86 s/trial × 0.87 × 12,960 ≈ **~75 hours** (range ~65–90 h given the
  mode/dt cost spread). Naive un-adjusted upper bound ≈ 86 h.
- Method and rates recorded here so a later agent can re-extrapolate without
  re-measuring (per `AGENTS.md` workflow rule).

Dominant drivers (each recurs every step, so cost is ~linear in steps/trial):
`forward_oracle_strict` does 9 × 32 × 8 = 2,304 `integrateStep` evals per
controller step; the per-step counterfactual injects a full
`computeStrictOracleThrust` (same 2,304 evals) on every eligible thrusting step
of the other thrusting arms; dt=0.004 is 4,000 steps/trial.

## What the cloud agent should run, in order

```bash
npm ci   # or npm install; pure Node ESM harness, no native build step
npm run threebody:phase13        # hard-void gate A  (local ~49 min)
npm run threebody:phase14        # hard-void gate B  (local ~5 min)
npm run threebody:phase15:smoke  # amended Richardson smoke (local ~57 min)
```

## What to report back (append to this file under "Cloud sandbox results")

1. **Wall-clock per command** (use the `manifest.json` `startedAt`/`completedAt`
   in each `results/threebody/<phase>*/manifest.json`, not a stopwatch).
2. **Gate reproduction numbers**, verbatim from each run's console
   `[threebody] …` lines + promising-best-cell count:
   - phase13 must be: 3,456 trials; candidate envelope rows 88/324; 81
     promising best cells; outcomes 1,154 bounded / 2,030 escape / 272 close.
   - phase14 must be: 6,048 trials; 5,184 paired; candidate envelope rows
     130/648; outcomes 1,269 bounded / 4,616 escape / 163 close.
3. **Environment**: Node version (`node -v`), OS, CPU model, core count.
4. Whether `richardson-order-map.csv` was emitted by the smoke and the
   smoke's `earlyTrajectoryPointCount` on `off` trials (non-zero).

Do NOT run `npm run threebody:phase15` (the ~75 h full lock) — see boundary.

## Two caveats that decide whether this is worth it

1. **Cross-platform bit-for-bit risk (the gates).** The hard-void gates demand
   *bit-identical* reproduction (3,456 / 88·324 / 81 / 1154·2030·272 and
   6,048 / 5,184 / 130·648 / 1269·4616·163). The harness PRNG (`makeRng`,
   integer ops) is deterministic, but `Math.sqrt/sin/cos/log` (used in
   `computeAcceleration`, `seededInitialParticle`, etc.) can differ by ~1 ULP
   across libm / CPU / Node version, and RK4 amplifies tiny differences in the
   chaotic regime. **A gate "deviation" on a different machine may be a
   platform-fp difference, not a code regression.** If a cloud gate deviates,
   do NOT declare Phase 15 void — report the exact numbers and environment so
   it can be triaged against the committed local
   `results/threebody/<phase>*/` artifacts first.
2. **Single-thread: more cores ≠ faster.** The `envelopeCases` loop in
   `scripts/threebody-operating-envelope.mjs` is strictly sequential; the
   scripts spawn no worker threads. A many-core sandbox does **not** speed a
   single run — only a faster single core or pure offload helps. The realistic
   win is *offload* (it runs off the project machine, unattended), not a
   speedup, unless the harness is parallelized first (a separate task: it is
   embarrassingly parallel across `cases` if sharded by `--mass-ratios` /
   `--timesteps` and the per-cell outputs are merged).

## Discipline boundary (pre-registration is binding)

`PHASE15_SPEC.md` is locked. The full lock (`npm run threebody:phase15`) stays
**operator-gated**: it may only run after both hard-void gates reproduce
bit-for-bit, the smoke runs, and `T_window` + Richardson-order evidence are
recorded in `PHASE15_RESULTS.md` and the operator signs off (see
`PHASE15_RESULTS.md` "pending" list). This cloud experiment measures
*feasibility and speed of the gates + smoke only*. It does not run, interpret,
or unblock the full lock, and it does not alter any pinned spec parameter.

## Cloud sandbox results

_(cloud agent: append measured wall-clock, gate numbers, and environment here)_
