# Three-Body Phase 15 - Forward-Oracle / Precision Lock Spec

This document is the implementation-grade spec for Phase 15 of
[`../SUNDOG_V_THREEBODY.md`](../SUNDOG_V_THREEBODY.md). Phase 13 showed a guarded
accelerometer-proxy TRACK survival pocket through a 16-second horizon. Phase 14
closed provisional partial / mechanism narrowed: ablations collapse and sign-flip
inverts, but the passive tidal-magnitude AUROC warning readout failed badly and the
action-coupling agreement metric did not separate intact from shuffled control.
Phase 15 asks whether the pocket survives stricter numerical and privileged forward
checks, and whether a timing-sensitive per-step counterfactual and a replacement
warning-quality readout can resolve the Phase 14 mechanism question. The binding
scope sources are the "Phase 15 - Forward-Oracle / Precision Lock" section of
[`CROSS_SUBSTRATE_NOTES.md`](CROSS_SUBSTRATE_NOTES.md) and the §5/§6 gaps recorded in
[`PHASE14_RESULTS.md`](PHASE14_RESULTS.md); this spec must not exceed the claims
permitted there.

Where this spec and the roadmap disagree, the roadmap wins. Where both are silent,
this spec is authoritative for Phase 15.

## 1. Decision Lock

Phase 15 starts with eight pinned calls:

- **The Phase 13 guard freeze is preserved unchanged across all comparisons.** Every
  non-passive arm reuses the Phase 13 `hazard_quantile` `0.75` passive-derived guard
  thresholds; the guard predicate is byte-identical to Phase 13/14. Two exact
  unchanged regression commands (Section 3) must reproduce their locked numbers
  bit-for-bit or the Phase 15 run is void.
- **Same Phase 13/14 cell grid, stricter timestep ladder.** Mass ratios, radii, and
  velocities are exactly the Phase 13/14 grid (including the `velocityScale=0.95`
  boundary). Timesteps are `0.004, 0.006, 0.008, 0.01, 0.012`: the Phase 13/14
  steps plus two finer steps for the precision lock.
- **Privileged forward oracle is explicitly non-deployable.** `forward_oracle_strict`
  chooses from the same 9-direction candidate set as the deployable `oracle`, but
  scores each candidate over a 32-coarse-step horizon, with each coarse step
  resolved by eight `dt/8` RK4 substeps and the same scoreState weighting shape as
  the deployable `oracle`. It is a reference yardstick: its candidate rows and
  best-cell wins are reported but never used to support or refute the deployable
  claim. The existing `oracle` stays in the slate as a continuity anchor.
- **Per-step counterfactual is measured against privileged read-only truth.** The
  timing-sensitive coupling metric uses an energy-like hazard score
  `H = computeSignatures(state).energy`, where lower is safer for the escape-risk
  component, evaluated on one-step `integrateStep` rollouts (actual action vs
  no-op vs the first action chosen by `forward_oracle_strict`), never the
  controller's possibly-ablated signal.
- **Warning quality is re-grounded.** The pre-registered warning-quality readout is
  `oracleHazardAuroc` (energy-precursor score, privileged forward-oracle hazard
  label), computed on passive trials only. The Phase 14 passive tidal-magnitude
  AUROC is still reported but is explicitly non-gating.
- **Ablation arms carry forward unchanged.** The four Phase 14 ablation modes run
  with their Phase 14-pinned definitions so the per-step counterfactual can be shown
  to separate them where the Phase 14 agreement metric failed.
- **Boundary negatives stay visible.** `velocityScale=0.95` and equal-mass cells are
  reported as their own rows in every column and never averaged into the pocket
  verdict.
- **Phase 15 cannot upgrade scope.** It may confirm-and-strengthen, hold, or retract
  the Phase 13/14 mechanism read under stricter checks. It does not extend horizon,
  sensor model, or move to 3D (that is Phase 16).

## 2. Scope

Phase 15 owns:

- The `forward_oracle_strict` controller mode and the per-step counterfactual,
  Richardson sampler, energy-drift diagnostics, and oracle-warning instrumentation,
  added additively to the shared
  `scripts/threebody-operating-envelope.mjs` and `public/js/threebody-core.mjs`
  behind a new default-off `--precision-receipts` flag and the existing
  `--track-action-coupling` flag
- `npm run threebody:phase15:smoke`
- `npm run threebody:phase15`
- Outputs under `results/threebody/phase15-forward-oracle-precision-smoke/`
- Outputs under `results/threebody/phase15-forward-oracle-precision-lock/`
- `cell-precision-map.csv` and `richardson-order-map.csv` (new additive
  precision-receipt outputs)
- Result note [`PHASE15_RESULTS.md`](PHASE15_RESULTS.md)
- Roadmap and writeup receipt bullets after the lock lands

Phase 15 does **not** own:

- Controller redesign or guard retuning
- Spatial/3D extension (Phase 16)
- New sensor models
- Any earned-claim scope upgrade beyond the Phase 13/14 pocket
- Retuning or arm/threshold changes after observing the result

## 3. Commands

Smoke:

```bash
npm run threebody:phase15:smoke
```

Full lock:

```bash
npm run threebody:phase15
```

The smoke covers mass ratio `1`, timesteps `0.004` and `0.012`, radii `1.025` and
`1.075`, velocities `0.95` and `1.1`, nine modes (`off`, `naive`,
`track_sensor_accel_guarded`, `track_sensor_accel_signal_shuffle`,
`track_sensor_accel_action_shuffle`, `track_sensor_accel_signal_delay`,
`track_sensor_accel_sign_flip`, `oracle`, `forward_oracle_strict`), and two seeds at
`duration=16`. That is 8 cases and 144 trials. At `dt=0.004`, each actual trial has
4,000 simulation steps; the strict oracle is expensive because each oracle decision
evaluates 9 candidates over 32 coarse steps x 8 substeps. The smoke spans the
ladder extremes so the Richardson sampler, precision-map proxy, and demoted
energy-drift diagnostics are exercised.

The full lock covers:

- mass ratios: `0.01`, `0.3`, `1`
- timesteps: `0.004`, `0.006`, `0.008`, `0.01`, `0.012`
- radii: `1.025`, `1.05`, `1.075`
- velocities: `0.95`, `1.05`, `1.1`, `1.15`
- modes: the nine listed above
- seeds: `8`
- duration: `16`

That is 180 cases, 12,960 trials, 11,520 paired non-passive rows, and 1,440 envelope
rows; candidate envelope rows are reported as `N / 1,440`. The finer timesteps and
the strict oracle make this materially heavier per trial than Phase 13/14; treat
the full lock as a multi-hour staged operator run under the repository's long-run
rule (a linear-from-Phase-13 estimate is a lower bound, not a promise).
Implementation must run the smoke, derive and record `T_window` plus supporting
Richardson-order evidence, and stop for readback before the full lock is started.

Two exact unchanged regression commands are rerun and reported pass/fail **before
any Phase 15 column is interpreted**; both are hard-void gates:

```bash
npm run threebody:phase13
npm run threebody:phase14
```

- `npm run threebody:phase13` must reproduce 3,456 trials; 88 / 324 candidate
  envelope rows; 81 promising best cells; terminal outcomes 1,154 bounded / 2,030
  escape / 272 close approach.
- `npm run threebody:phase14` must reproduce 6,048 trials; 5,184 paired non-passive
  rows; 130 / 648 candidate envelope rows; terminal outcomes 1,269 bounded / 4,616
  escape / 163 close approach.

If either deviates, the shared-harness edit perturbed a frozen path and the Phase 15
run is void. New ablation/instrumentation paths are additive and gated behind new
mode names or default-off flags so these commands hit byte-identical code paths.

## 4. Metrics

Read these files first:

- `aggregate-envelope.csv`
- `candidate-envelope.csv`
- `best-by-cell.csv`
- `cell-class-map.csv`
- `cell-warning-quality-map.csv`
- `cell-precision-map.csv`
- `richardson-order-map.csv`
- `trial-outcomes.csv`

Primary metrics, reported per arm and per timestep:

- Outcome effect: candidate envelope rows out of total, best-cell class balance,
  survival delta versus passive, worsened rate, mean time delta, mean delta-v and
  per-second, dominant failure mechanism — at every timestep in the ladder
- Per-step counterfactual: `counterfactualMeanEffect` (primary),
  `counterfactualPositiveRate`, `meanGapToOracle`, `counterfactualEligibleSteps`
- Warning quality: passive mean `oracleHazardAuroc` (primary, decidable),
  `localAccelerationMagnitudeAuroc` (secondary), `tidalMagnitudeAuroc` (reported,
  non-gating, continuity with Phase 14)
- Precision receipt (primary, gating): passive early-window Richardson
  cross-timestep trajectory order. Per `off` cell-seed,
  `D(dt) = max over common-grid times t <= T_window of
  || pos_test(dt, t) - pos_test(0.004, t) ||_2` (position-only L2 on the test
  particle; common physical-time grid step `Δ = 0.12`; reference `dt = 0.004`);
  fitted order `p = OLS slope(log D vs log dt)` over
  `dt in {0.006, 0.008, 0.01, 0.012}`. Reported per cell in
  `richardson-order-map.csv` with the favorable-pocket median order and coverage.
- Precision diagnostics (demoted, reported, non-gating): passive
  `finalRelEnergyDrift` / `maxAbsEnergyDrift` per timestep and
  `integrationErrorProxy` in `cell-precision-map.csv` are retained as diagnostics
  only and are explicitly **restricted-model non-conservation + close-encounter
  softening, not integration order** (in the restricted problem the primaries
  ignore the test particle, total 3-body energy is not a conserved invariant of
  the integrated equations, and the `r < 0.01` / `0.01` softening makes the
  drift `dt`-insensitive — flat across the ladder in the smoke). They carry no
  pass/fail weight.

Operational definitions:

- Per-step counterfactual is evaluated only on non-warmup steps with
  `|thrust| > 1e-6`. Hazard score `H(state) = computeSignatures(state).energy`;
  lower `H` is safer for the escape-risk component of this test.
  At each eligible step, one-step `integrateStep` rollouts from the realized state
  give `H(noop)`, `H(actual)`, `H(oracleStrict)`; `effectVsNoop = H(noop) −
  H(actual)`; `normalizer = max(|H(noop) − H(oracleStrict)|, 1e-9)`;
  `counterfactualScore = clamp(effectVsNoop / normalizer, -1, 1)`. Per-trial
  `counterfactualMeanEffect` = mean over eligible steps. The oracle/no-op rollouts
  use the privileged read-only `forward_oracle_strict` and `[0,0]`, never the
  controller's signal.
  `H(oracleStrict)` means the one-step state reached by applying the first action
  selected by `forward_oracle_strict` and then integrating one normal `dt` step;
  it is not the endpoint of the strict oracle's internal 32-step scoring rollout.
- `oracleHazardAuroc` is computed on passive/off trials only, score channel
  `energy`, label positive iff `forward_oracle_strict` rollouts from the sample
  state reach `r3 > escapeRadius` or `minPrimaryDistance < closeApproachRadius`.
  AUROC-null cells are reported as coverage, not successes. A
  pass on warning quality requires mean defined `oracleHazardAuroc >= 0.70` and at
  least two thirds of favorable high-velocity cells with defined passive
  `oracleHazardAuroc`; otherwise warning quality is partial/undecidable.
- The precision receipt is the early-window Richardson cross-timestep trajectory
  order, set by procedure, not pinned blind. Common physical-time grid step
  `Δ = 0.12` divides every ladder timestep into an integer step count
  (0.004→30, 0.006→20, 0.008→15, 0.01→12, 0.012→10). Under `--precision-receipts`
  an in-`runTrial` sampler records the **unrounded** test-particle state
  `(x3, y3, vx3, vy3)` at every grid time up to `EARLY_GRID_MAX_T = 4.8` on
  passive (`off`) trials. Harness-side, per `off` cell-seed,
  `D(dt) = max over grid times t <= T_window of
  || pos_test(dt, t) - pos_test(0.004, t) ||_2` and `p` = OLS slope of
  `log D(dt)` vs `log dt` over the four non-reference ladder steps. A cell-seed
  order is **defined** iff every ladder timestep has a non-terminated trajectory
  with `>= MIN_INWINDOW_GRID_POINTS = 12` in-window grid points and finite
  positive `D(dt)`; otherwise it is coverage-null, not a failure. The
  favorable-pocket order is the **median** of defined per-seed `p` over `off`
  cells with `velocityScale >= 1.05`; equal-mass and `velocityScale=0.95`
  boundary cells are their own rows and are never averaged into the pocket. The
  receipt is **decidable** iff `>= 2/3` favorable `off` cells have a defined
  order, else it is undecidable.
- `T_window` is set by procedure and locked from the re-run smoke before the
  full lock is interpreted (the endorsed pin-procedure-derive-number pattern):
  `T_window` = the largest common-grid time `t* <= 4.8` such that, on every
  smoke passive (`off`) cell-seed, (a) the fitted order using the max over grid
  times `<= t*` lies in `[3.0, 5.0]`, (b) `max D(0.012) within [0, t*] <
  EARLY_DIV_ABS_CAP = 1e-6`, and (c) no smoke `off` cell-seed terminates at or
  before `t*` on any ladder timestep — capped at `T_WINDOW_CAP = 2.4`. The
  derived `T_window`, `MIN_INWINDOW_GRID_POINTS`, and the supporting smoke
  per-cell fitted-order evidence are recorded in `PHASE15_RESULTS.md` before the
  full lock is interpreted. The former finer-not-worse monotonicity check is
  subsumed: a clean `O(dt^4)` Richardson slope (`p ≈ 4`, in `[3, 5]`) is exactly
  the statement that trajectory error shrinks at the RK4 rate as `dt` refines.

Pre-registered numeric bars (favorable high-velocity pocket, `velocityScale >= 1.05`):

- Clean counterfactual separation: intact `track_sensor_accel_guarded`
  `counterfactualMeanEffect >= +0.20`; `signal_shuffle`, `action_shuffle`,
  `signal_delay` each `<= +0.05`; `sign_flip` `<= -0.10`; intact-vs-each-shuffled
  gap `>= 0.15`.
- Warning quality decidable + passing: passive mean `oracleHazardAuroc >= 0.70`
  with `>= 2/3` favorable cells defined.
- Precision stability: (i) the passive early-window Richardson receipt is
  decidable (`>= 2/3` favorable `off` cells with a defined order) and the
  favorable-pocket **median fitted order `p >= 3.0`** (the RK4 `O(dt^4)`
  signature; well below the empirically observed ≈4.3 with margin); AND (ii) the
  favorable-pocket candidate-row fraction at `dt=0.004` is within absolute
  `0.10` of the fraction at `dt=0.01`. The demoted energy-drift diagnostics
  carry no pass/fail weight.

## 5. Pre-Registered Branches

**Pass:** in the favorable high-velocity pocket the frozen guarded accelerometer
TRACK arm keeps a positive candidate envelope against passive and naive baselines at
every timestep including `0.004` and `0.006`, the passive early-window
Richardson receipt is decidable and the favorable-pocket median fitted order is
`p >= 3.0` (a clean RK4 `O(dt^4)` cross-timestep trajectory signature), and the
favorable-pocket candidate fraction at `dt=0.004` is within `0.10` of
`dt=0.01`; the per-step
counterfactual cleanly separates intact control from mistimed control at the pinned
thresholds (closing the Phase 14 gap); and the replacement warning-quality readout
is decidable and passes (`oracleHazardAuroc >= 0.70`, `>= 2/3` cells defined). The
`forward_oracle_strict` reference is reported but not used to support the claim.
This supports upgrading the Phase 14 narrowed-mechanism read to "the Phase 13 win
survives a stricter numerical and privileged forward check, and the operative handle
is signal-directed, timing-sensitive thrust."

**Partial:** the frozen arm keeps a positive candidate envelope and the per-step
counterfactual still separates it from the shuffled arms, but at least one precision
condition is missed (candidate fraction moves more than `0.10` between `dt=0.01` and
`dt=0.004`, or the passive early-window Richardson receipt is undecidable —
`< 2/3` favorable `off` cells with a defined order), or the replacement
warning-quality readout is undecidable. The earned
claim stays "guarded TRACK improves survival in the mapped high-velocity pocket
through the 16-second tested horizon," annotated "mechanism is signal-directed and
timing-sensitive, but numerical robustness (early-window RK4 trajectory order)
or passive warning quality at finer timesteps is not cleanly established." No
stronger precision-locked claim is made.

**Fail:** the favorable-pocket candidate envelope collapses or becomes
oracle/naive-dominated at finer timesteps (a coarse-`dt` discretization artifact),
and/or the passive early-window Richardson receipt is decidable but the
favorable-pocket median fitted order is `p < 3.0` (the integrator is not
exhibiting its `O(dt^4)` rate in the pre-Lyapunov window — a genuine
numerical-order failure, distinct from chaotic divergence which is reported as
undecidable, not Fail), and/or the per-step counterfactual fails to separate
intact control from the shuffled/mistimed arms by the pinned margin. Phase 15 is then a useful negative
result: the preserved claim is narrowed to the original `0.008–0.012`
coarse-timestep regime only and the next step is control-handle or controller
redesign, not bigger sweeps.

**Pre-registered negative (explicit):** if the frozen guarded arm retains its
candidate envelope at finer timesteps but `counterfactualMeanEffect <= 0` for intact
control in the favorable pocket (the privileged one-step counterfactual shows the
controller's actions do not improve the hazard score versus no-op), this is
registered as a mechanism negative even with a positive outcome envelope: outcome
survival without privileged-counterfactual hazard improvement does not support a
causal-control claim. The survival wording is preserved; any "signal-directed
control" mechanism wording is retracted.

## 6. Readback Template

After the full lock finishes, update [`PHASE15_RESULTS.md`](PHASE15_RESULTS.md) with:

- command and wall-clock runtime
- total trial count and terminal outcome counts
- both regression-gate results (Phase 13 and Phase 14), pass/fail with numbers
- the smoke-derived `T_window`, supporting smoke Richardson-order evidence, and
  favorable-pocket decidability coverage
- per-arm × per-timestep outcome table
- per-step counterfactual table (intact vs each shuffled arm vs sign-flip)
- warning-quality table (`oracleHazardAuroc` primary; tidal AUROC non-gating)
- precision read: early-window Richardson favorable-pocket median fitted order
  with decidability/coverage and the locked `T_window`; candidate-fraction
  stability across the ladder; demoted energy-drift diagnostic (restricted-model
  non-conservation + softening, non-gating)
- favorable-pocket read and boundary read
- pre-registered branch taken
- claim wording to preserve, strengthen, narrow, or retract
