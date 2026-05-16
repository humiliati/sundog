# Phase 4 Bayesian-Floor Buildout

> Buildout roadmap for the Bayesian-floor controller required by
> [`PHASE4_THREEBODY.md`](PHASE4_THREEBODY.md).
> Status: staged/open, 2026-05-16. This document is a build contract, not an
> empirical result. No Phase 4 proof run is admitted until the floor, regret
> reducer, and capped-probe receipt land.

## Purpose

Phase 4 needs a Bayesian floor inside the three-body workbench. Existing
`oracle` and `forward_oracle_strict` modes are privileged references: they read
simulator state or privileged forward rollouts, and therefore cannot serve as
`π*_Bayes` under the signature information regime.

The controller built here must estimate the best action using the same admitted
signature history available to the signature-only policy:

```text
π*_Bayes(h_t^σ) ∈ argmax E[T_safe / T_max | h_t^σ]
```

It must not open a separate `bayes_v_sundog` track. It lives in the existing
Phase 4 workbench and writes receipts under `results/proof/phase4/`.

Non-negotiables:

- no raw-state action selection;
- deterministic seeds and tie handling;
- explicit unknown-mode validation before any proof command;
- paired regret against the signature controller on the same cell-seed slate;
- capped probe first, full lock staged only after the measured rate is known.

## Information Regime

The Bayes floor receives the admitted signature history `h_t^σ`, not
`state_t`. The history may include:

- `|T_hat_t|`;
- `∂|T_hat_t|/∂x` and `∂|T_hat_t|/∂y`;
- `guard_t`;
- sensor tier, sensor noise standard deviation, and probe delta;
- the envelope-cell configuration known before the rollout;
- past admitted signature observations;
- past actions selected by the Bayes-floor controller.

Forbidden for action selection:

- raw simulator positions or velocities;
- direct access to `x3,y3,vx3,vy3`;
- hidden terminal outcome lookahead except through simulated particles;
- the privileged `oracle` or `forward_oracle_strict` action.

Full state may be logged only as truth for audit, regret readout, or test
fixtures. Any code path that uses full state to choose the Bayes action voids
the floor as a Phase 4 baseline.

## Recommended Algorithm

Use a particle-belief MPC floor unless an exact finite belief-grid dynamic
program is implemented and reviewed first. Exact DP is acceptable in principle,
but the current continuous three-body signature and envelope grid make a
particle filter the conservative first build.

### Belief State

For each envelope cell, initialize particles from the same near-escape initial
distribution used by the envelope harness. Each particle carries a hypothesized
raw simulator state and a deterministic RNG stream. The particle set is an
internal belief approximation, not extra information given to the controller.

After each admitted observation, update particle weights by comparing each
particle's predicted signature observation to the actual `Φ_t`:

```text
w_i ∝ p(Φ_t | particle_i, c)
```

For `sensorNoiseStd = 0`, use a deterministic tolerance fixed in the manifest.
For noisy cells, use the same accelerometer-array noise model as
`track_sensor_accel_guarded`. Record effective sample size and resampling
events in `belief-diagnostics.csv`.

### Action Lattice

Use a fixed thrust lattice derived from the current oracle candidate directions,
scaled by `thrustLimit`, including zero thrust. The action order is part of the
contract and must be recorded in the manifest. Ties resolve in this order:

1. maximize expected `T_safe / T_max`;
2. if tied within one integrator step, minimize expected `totalDeltaV`;
3. if still tied, choose the first action in the pre-registered order.

The initial candidate order should put zero thrust first, then the existing
axis and diagonal directions from `oracleCandidateThrusts`. If implementation
discovers a mismatch with the existing helper, update this document before the
proof run rather than silently changing the order.

### Planning Objective

For each candidate action, propagate particles forward with the existing
integrator and estimate expected safe time under the finite planning horizon.
The final horizon, particle count, and resampling threshold are locked only
after the capped probe records a rate.

Start with this smoke profile:

```text
particle-count: 256
planning-horizon-steps: 16
resample-threshold: 0.5
duration: 16
```

If that profile is too slow for the ~10-minute rule, reduce the smoke profile
only for rate measurement and stage the proof-capable settings to a long-budget
runner. Do not reinterpret a reduced smoke as the proof floor.

## Build Phases

### BF-0: Harness Safety

Add explicit validation so an unknown controller or evaluator name cannot
silently behave like zero thrust. This is required because the current
`computeControlThrust` fallback can mask a missing Bayes mode.

Required receipts:

- bogus mode fails loudly in the harness;
- accepted mode/evaluator names are listed in the manifest;
- the validation test is run before any Phase 4 proof command is staged.

### BF-1: Signature Observation Contract

Expose or add a pure observation function for the guarded accelerometer-proxy
signature used by `track_sensor_accel_guarded`. It must return the exact fields
admitted in [`PHASE4_THREEBODY.md`](PHASE4_THREEBODY.md), including guard state
and sensor-noise conventions.

Required receipts:

- recorded `Φ_t` matches the signature controller's own observation on a fixed
  fixture;
- noise-free observation is deterministic;
- noisy observation records the seed and noise parameters.

### BF-2: Particle-Belief Evaluator

Implement the Bayes floor as a separate proof evaluator or as an explicitly
validated controller mode named only after it works, for example
`bayes_floor_particle_mpc`.

Preferred entrypoint:

```text
node scripts/threebody-phase4-bayes-floor.mjs
```

The script should write:

- `manifest.json`;
- `signature-observations.jsonl`;
- `belief-diagnostics.csv`;
- `bayes-actions.csv`;
- `bayes-trial-outcomes.csv`.

### BF-3: Regret Reducer

Add a reducer that joins Bayes-floor rows to signature-controller rows by
cell-seed and computes the Phase 4 regret readout:

```text
regret_i = (T_safe(π*_Bayes, i) - T_safe(π_signature, i)) / T_max
```

The reducer also writes the empirical fiber classification. It must use the
fiber-classification procedure pinned in
[`PHASE4_THREEBODY.md`](PHASE4_THREEBODY.md) §5 exactly as written — the same
`Σ` partition keys, the ≥20-Bayes-reached-samples bin threshold, the
exact-match common-action rule, and the `undecidable` handling — not a
reinvented binning. That section is the single authoritative definition; this
reducer is its implementation.

Outputs:

- `phase4-regret.csv`;
- `phase4-regret-summary.csv`;
- `cell-fibers.json`.

Negative-regret sanity rule: if the Bayes floor produces negative regret on
more than 5% of rows, the run is non-decisive and the floor must be repaired
before Phase 4 can interpret the gate.

### BF-4: Capped Probe

Run a capped probe that obeys the ~10-minute rule. Record:

- wall-clock from `manifest.json` `startedAt` / `completedAt`;
- trials completed;
- seconds per trial;
- seconds per envelope cell;
- estimated wall-clock for the full proof lock;
- whether the full lock must move to a long-budget runner.

Scratch output should live under `results/proof/phase4/_probe-*` and be cleaned
after the measured rate is copied into the spec or result note.

### BF-5: Full Lock Handoff

Only after BF-0 through BF-4 pass, update
[`PHASE4_THREEBODY.md`](PHASE4_THREEBODY.md) with the exact runnable PowerShell
for the proof lock, resume rules, read-back paths, and branch selected by each
gate outcome. The full lock stays operator/runner-gated.

## Validation Gates

- **Unknown-mode gate:** a bogus controller/evaluator name fails before rollout.
- **No-state-leak gate:** the Bayes action code path is audited so raw simulator
  truth is used only for particle simulation, test fixtures, and readout, never
  as the observed state.
- **Observation-parity gate:** recorded `Φ_t` equals the signature controller's
  observation contract on fixed fixtures.
- **Degenerate-full-observation gate:** in a smoke-only fixture where `Φ` is
  deliberately enriched to full state and the action lattice/horizon match the
  strict oracle, the Bayes evaluator should match the strict oracle's chosen
  action except for documented tie cases. This fixture is a planner-correctness
  check only; it does **not** alter the admitted Phase 4 signature defined in
  [`PHASE4_THREEBODY.md`](PHASE4_THREEBODY.md) §2, and no proof run may use the
  enriched `Φ`.
- **Floor-sanity gate:** negative regret above 5% voids the run as a floor.
- **Runtime gate:** capped probe completes within ~10 minutes or the remaining
  work is staged to a long-budget runner.

## Target Output Shape

The buildout should produce this directory shape before Phase 4 is unblocked:

```text
results/proof/phase4/
  bayes-floor-probe/
    manifest.json
    signature-observations.jsonl
    belief-diagnostics.csv
    bayes-actions.csv
    bayes-trial-outcomes.csv
  phase4-regret.csv
  phase4-regret-summary.csv
  cell-fibers.json
```

The full-lock result may use a different final directory name, but it must
preserve the same read-back fields.

## Target Smoke Command

This is a target contract for the scripts and flags to build. It is **not
runnable yet** and must not be used as a proof command until BF-0 through BF-3
exist.

```powershell
Set-Location C:\Users\hughe\Dev\sundog
$outRoot = "results\proof\phase4"
New-Item -ItemType Directory -Force "$outRoot\bayes-floor-smoke" | Out-Null

node scripts/threebody-phase4-bayes-floor.mjs `
  --phase phase4-bayes-floor-smoke `
  --out "$outRoot\bayes-floor-smoke" `
  --regimes near_escape `
  --mass-ratios 1 `
  --timesteps 0.01 `
  --radius-scales 1.075 `
  --velocity-scales 1.1 `
  --thrust-limits 0.4 `
  --sensor-noise-sweep 0 `
  --track-guard-mode hazard_quantile `
  --track-guard-quantile 0.75 `
  --track-guard-min-radius-sweep 1.15 `
  --track-guard-max-local-acceleration-sweep 2.5 `
  --track-guard-max-tidal-magnitude-sweep 35 `
  --seeds 2 `
  --duration 16 `
  --particle-count 256 `
  --planning-horizon-steps 16 `
  --resample-threshold 0.5 `
  --bootstrap-seed 40604
```

## Decision Points

1. **Floor form.** Default to particle-belief MPC. Switch to exact DP only if a
   finite belief grid can be pinned without changing the Phase 4 substrate.
2. **Integration form.** Prefer a separate proof evaluator until the no-leak
   and runtime gates pass. A harness controller mode is acceptable afterward if
   it is validated explicitly.
3. **Final approximation settings.** Lock particle count, horizon, and
   resampling threshold after the capped probe, not before rate measurement.

## Exit Criteria

This buildout exits when:

- unknown-mode validation exists and is tested;
- the signature observation contract is implemented and audited;
- the particle-belief Bayes evaluator writes the target receipts;
- the regret reducer and cell-fiber classifier write Phase 4 readbacks;
- a capped probe records measured rate and floor sanity;
- [`PHASE4_THREEBODY.md`](PHASE4_THREEBODY.md) is updated with the exact proof
  lock command and outcome branches.

Only then does Phase 4 empirical entry open.
