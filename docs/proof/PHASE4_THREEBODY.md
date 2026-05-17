# Phase 4 Three-Body Spec

> Phase 4 artifact for
> [`COARSE_GRAINING_PROOF_ROADMAP.md`](../COARSE_GRAINING_PROOF_ROADMAP.md).
> Status: spec drafted, empirical entry blocked, 2026-05-16. Phase 3 closed
> positive in [`PHASE3_BOUNDARY.md`](PHASE3_BOUNDARY.md). This document maps the
> planar restricted three-body workbench onto the Phase 0 substrate-admission
> checklist before any proof-track empirical run is admitted.

## Entry & Gate

Phase 4 has one satisfied entry condition and one unsatisfied entry condition:

- Phase 3 exit is satisfied.
- The required Bayesian-floor baseline is **not** found in the current
  three-body workbench. Existing `oracle` and `forward_oracle_strict` modes are
  privileged heuristic / forward-lookahead references; they are not a
  Bayes-optimal controller under the signature information regime.

Therefore the substrate is **not yet admitted** to the proof track. This spec
pins the intended objects and the blockers, but no Phase 4 proof run is
authorized until the Bayesian floor and regret reducer land.
The build path for that blocker is staged in
[`PHASE4_BAYESIAN_FLOOR_BUILDOUT.md`](PHASE4_BAYESIAN_FLOOR_BUILDOUT.md).

Roadmap gate, quoted unchanged:

> Gate, fixed now: signature-only regret vs Bayes must → 0 (within bootstrap CI)
> **on** the `𝓕_σ`-measurable cell set and stay bounded away from 0 (CI excludes
> 0) **off** it. If regret is bounded-away *on* the measurable set,
> sufficiency-for-control is empirically false on a real substrate → halt and
> falsify. If regret → 0 *off* the measurable set, the boundary (Phase 3) is
> wrong → reopen Phase 3.

This exact gate is the one the eventual result must evaluate. It is not softened
to oracle comparison, candidate-envelope count, or survival-vs-passive.

## Admission Checklist Map

| Phase 0 item | Phase 4 status |
| --- | --- |
| `X` | Pinned below as the raw planar restricted decision state plus time/config. |
| `Φ, Σ` | Pinned below as the guarded accelerometer-proxy signature, including noise and guard handling. |
| `J, μ, regret` | Pinned below as safe-time regret against Bayes, with bootstrap readout. |
| `π*` | **Blocked:** no Bayesian-floor policy exists yet. Existing oracles are reference yardsticks only; buildout staged in [`PHASE4_BAYESIAN_FLOOR_BUILDOUT.md`](PHASE4_BAYESIAN_FLOOR_BUILDOUT.md). |
| Measurable/off-cell split | Procedure pinned below, but final labels depend on the missing Bayes floor. |

## 1. `X`

The admitted world state for this phase is the current-time raw state of the
planar restricted simulator:

```text
X_t = (state_t, t, c)
state_t = [x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3]
```

where `c` is the envelope-cell configuration: regime, mass ratio, timestep,
radius scale, velocity scale, thrust limit, sensor-noise standard deviation,
guard mode, guard quantile, and duration.

This is a **raw-state** decision domain for the workbench truth. It is not a
history or belief state. The signature-only controller receives only `Φ_t`; the
Bayesian floor, once implemented, must optimize over the posterior induced by
the admitted signature history `h_t^σ`, not over privileged full state.

The current harness entrypoint is:

```text
node scripts/threebody-operating-envelope.mjs
```

The underlying dynamics and signature code live in
[`../../public/js/threebody-core.mjs`](../../public/js/threebody-core.mjs).

## 2. `Φ, Σ`

The signature is the guarded accelerometer-proxy used by
`track_sensor_accel_guarded`.

Operationally, the controller estimates the local tidal tensor from acceleration
samples at the controlled particle and two nearby probe offsets. The active
signature readout is:

```text
Φ_t = (
  |T_hat_t|,
  ∂|T_hat_t|/∂x,
  ∂|T_hat_t|/∂y,
  guard_t,
  sensor_tier,
  sensor_noise_std,
  probe_delta
)
```

where `guard_t` is the predicate:

```text
radius >= trackGuardMinRadius
and localAccelerationMagnitude <= trackGuardMaxLocalAcceleration
and |T_hat_t| <= trackGuardMaxTidalMagnitude
```

For `trackGuardMode = hazard_quantile`, the guard thresholds are calibrated
from matched passive (`off`) trials before the non-passive modes run. This is
allowed because the calibration is cell-level and pre-action; it must be logged
in `manifest.json` through `trackGuardQuantile`,
`trackGuardMinRadius`, `trackGuardMaxLocalAcceleration`,
`trackGuardMaxTidalMagnitude`, and `guardCalibrationSampleCount`.

Noise handling is part of `Φ`, not an afterthought. The admitted sensor variant
is `accelerometer_array_noisy`, with `sensorNoiseStd` supplied by the envelope
cell. Delayed and micro-maneuver variants are outside this Phase 4 proof run
unless the spec is amended before execution.

`Σ` is the measurable space of these finite logged fields plus real-valued
signature coordinates. For cell classification, continuous coordinates are
partitioned by the pre-registered measurable-set procedure below; the raw state
is never used as a signature coordinate.

## 3. `J, μ, Regret Readout`

The proof-track objective is safe time in the near-escape pocket, with fuel only
as a tie-breaker:

```text
J(π) = E_μ[T_safe(π) / T_max]
```

`T_safe` is the simulated time until `terminalOutcome != bounded`, capped at
`T_max = duration`. If two policies are tied within one integrator step on
`T_safe`, lower `totalDeltaV` wins the tie. This keeps the main regret readout
aligned with the existing operating-envelope surface while avoiding arbitrary
fuel weights.

The evaluation measure `μ` is uniform over the pre-registered finite envelope
grid and seed set. The default proof grid, matching the settled Phase 13/14
near-escape lock shape, is:

```text
regimes: near_escape
mass-ratios: 0.01,0.3,1
timesteps: 0.008,0.01,0.012
radius-scales: 1.025,1.05,1.075
velocity-scales: 0.95,1.05,1.1,1.15
thrust-limits: 0.4
sensor-noise-sweep: 0,0.01
track-guard-mode: hazard_quantile
track-guard-quantile: 0.75
track-guard-min-radius-sweep: 1.15
track-guard-max-local-acceleration-sweep: 2.5
track-guard-max-tidal-magnitude-sweep: 35
duration: 16
```

For each matched cell-seed:

```text
regret_i = (T_safe(π*_Bayes, i) - T_safe(π_signature, i)) / T_max
```

The primary Phase 4 readout is the mean signed regret by cell class, with a
95% paired bootstrap CI over cell-seed rows. Use a deterministic bootstrap seed
recorded in the result manifest. If a Bayes proxy produces negative regret on
more than 5% of rows, the proxy is not acting as a floor and the run is
non-decisive until the floor is repaired.

The gate interpretation below is the **pre-registered operationalization of the
verbatim roadmap gate**, not a new or softened gate. The roadmap condition is
"signature-only regret vs Bayes → 0 (within bootstrap CI) on the measurable set
and bounded away from 0 (CI excludes 0) off it"; the numeric realizations of
"→ 0" and "bounded away" are fixed here, before any run, and only tighten — they
never relax — that condition:

- **On measurable cells:** "→ 0" is realized as: the 95% CI includes `0` **and**
  the point estimate is no larger than one timestep divided by `T_max`. The
  magnitude bound is what makes "→ 0" testable (a CI that includes 0 but is wide
  could still hide a large regret); it is a strengthening of the roadmap
  condition, fixed at spec time.
- **Off measurable cells:** "bounded away from 0" is realized as: the 95% CI
  lower bound is strictly above `0` (equivalently, the CI excludes `0` from
  below).

Fuel tie-break readout:

```text
fuel_excess_i = totalDeltaV(π_signature, i) - totalDeltaV(π*_Bayes, i)
```

Report it only for rows where `|T_safe(π*_Bayes)-T_safe(π_signature)| <= dt`.
Fuel never rescues a failed safe-time gate.

## 4. `π*`

For Phase 4, `π*` means the Bayesian-floor policy under the signature
information regime:

```text
π*_Bayes(h_t^σ) ∈ argmax E[T_safe / T_max | h_t^σ]
```

It is **not** the existing `oracle` mode, and it is **not**
`forward_oracle_strict`. Those modes inspect simulator state or privileged
forward rollouts and remain useful for debugging and upper-bound context only.

The Bayesian floor must be implemented and documented before any Phase 4 proof
run. The build contract is
[`PHASE4_BAYESIAN_FLOOR_BUILDOUT.md`](PHASE4_BAYESIAN_FLOOR_BUILDOUT.md).
Acceptable implementations:

1. exact dynamic programming on a pre-registered finite belief grid;
2. particle-belief MPC with fixed particle count, action lattice, resampling
   rule, horizon, random seed, and approximation-error audit.

Tie handling is fixed (reconciled 2026-05-16 with the BF-2 shaped objective;
authoritative definition in
[`PHASE4_BAYESIAN_FLOOR_BUILDOUT.md`](PHASE4_BAYESIAN_FLOOR_BUILDOUT.md) ▸
Planning Objective / Action Lattice):

1. maximize the shaped planning score (expected within-horizon survival time
   `+ shapeFraction · dt ·` expected terminal safety margin);
2. if within `1e-9` of the best shaped score, minimize expected `totalDeltaV`;
3. if still tied, choose the first action in the pre-registered action order.

The shaping term is a tractable internal surrogate for `π*_Bayes =
argmax E[T_safe/T_max | h]` under a finite horizon. With `shapeFraction ∈
[0,1)` it is strictly less than one `dt`, so the floor never prefers an
earlier-escape action (floor-validity). This changes only the floor's internal
action selection — **`X`, `Φ`, `Σ`, `J`, `μ`, the regret readout, and the gate
above are unchanged.** The earlier "maximize expected `T_safe/T_max`; tied
within one integrator step → min `totalDeltaV`" wording is superseded for the
reason recorded in the BF-4 Probe Receipt (it degenerated the floor to passive).

Resolved blocker: the harness previously did not validate unknown controller
mode names inside `computeControlThrust` (an unimplemented mode silently acted
like zero thrust). BF-0 landed `KNOWN_CONTROLLER_MODES` + a loud throw and a
validation receipt; the Bayes floor is a separate evaluator that adds no core
controller mode, so this blocker is cleared.

## 5. Measurable Cell Set / Off-Cell Boundary

The Phase 4 cell split uses the Phase 3 fiber-conflict vocabulary:

- an **on-set** cell is one where every positive-mass empirical `Φ`-fiber admits
  a common Bayes-optimal action;
- an **off-set** cell is one where at least one positive-mass empirical
  `Φ`-fiber has disjoint Bayes-optimal action sets.

The empirical fiber classifier is pre-registered as follows, but remains
blocked until `π*_Bayes` exists:

1. Use the Bayes-floor calibration rollouts and the signature-controller
   rollouts on the same `μ` slate.
2. Partition `Σ` within each envelope cell by:
   `guard_t`, log-binned `|T_hat_t|`, gradient-angle sector, gradient-magnitude
   bin, and `sensorNoiseStd`. Bin edges must be written to
   `results/proof/phase4/cell-fibers.json` before the lock is interpreted.
3. For every bin with at least 20 Bayes-reached samples, compute the Bayes
   optimal-action correspondence under the admitted action lattice. Actions are
   common if their thrust vectors match exactly on the lattice, or if both are
   the zero action.
4. A cell is **on** if all positive-mass bins have a nonempty common-action
   intersection. A cell is **off** if any positive-mass bin has an empty
   intersection. Cells without enough positive-mass bins are `undecidable` and
   do not count toward either side of the gate.

This deliberately reuses Phase 3's condition:

```text
Φ(x0) = Φ(x1)  and  A*(x0) ∩ A*(x1) = ∅
```

The off-cell boundary is therefore not "where the controller happened to fail";
it is where the admitted signature collapses states that require incompatible
Bayes-optimal actions.

## 6. Capped Probe + Full Lock

No proof-track empirical command is authorized yet, because the Bayes floor is
missing. The existing envelope harness and manifest shape are nevertheless
pinned here so the eventual command is not invented later.

### Existing Harness Rate Probe

This is a current-harness rate probe only. It does not decide Phase 4 and does
not satisfy the Bayesian-floor entry condition. Run only by an operator/runner
when rate measurement is needed.

```powershell
Set-Location C:\Users\hughe\Dev\sundog
$outRoot = "results\proof\phase4"
New-Item -ItemType Directory -Force "$outRoot\_probe-envelope" | Out-Null

node scripts/threebody-operating-envelope.mjs `
  --phase phase4-proof-probe-envelope `
  --out "$outRoot\_probe-envelope" `
  --regimes near_escape `
  --modes off,track_sensor_accel_guarded,forward_oracle_strict `
  --mass-ratios 1 `
  --timesteps 0.01 `
  --radius-scales 1.025,1.075 `
  --velocity-scales 1.05,1.1 `
  --thrust-limits 0.4 `
  --sensor-noise-sweep 0 `
  --track-guard-mode hazard_quantile `
  --track-guard-quantile 0.75 `
  --track-guard-min-radius-sweep 1.15 `
  --track-guard-max-local-acceleration-sweep 2.5 `
  --track-guard-max-tidal-magnitude-sweep 35 `
  --seeds 2 `
  --duration 16 `
  --sensor-audit-every 240 `
  --precision-receipts 1
```

Read:

- `results/proof/phase4/_probe-envelope/manifest.json`
- `startedAt`, `completedAt`, and trial count
- seconds per trial and seconds per envelope cell

If this probe exceeds ~10 minutes, stop and stage all further work to a
long-budget runner.

### Blocked Proof Lock

The full proof lock is **not pinned as runnable** until the Bayes-floor decision
is made. Do not use a placeholder controller mode. The post-blocker command must
be written here with:

- the accepted Bayes-floor mode or evaluator name;
- its action lattice / belief approximation settings;
- the regret reducer command;
- measured probe rate and extrapolated full wall-clock;
- resume-safety rules;
- read-back path `results/proof/phase4/manifest.json`.

Current lower-bound scale, before Bayes overhead: the settled Phase 13 lock ran
3,456 trials in roughly one hour; Phase 15's strict-oracle smoke measured about
23.86 seconds per trial for the expensive precision/strict-oracle mix, making
the full precision lock a multi-day operator run. The Bayes floor may be closer
to Phase 15 than Phase 13, so no inline run is admissible.

## 7. Read-Back

The eventual Phase 4 result must land under `results/proof/phase4/` and include:

- `manifest.json` with `startedAt`, `completedAt`, exact args, git SHA if
  available, bootstrap seed, and Bayes-floor implementation metadata;
- `trial-outcomes.csv` from the envelope harness;
- `phase4-regret.csv`, one row per matched cell-seed with `T_safe_signature`,
  `T_safe_bayes`, `regret`, `fuel_excess`, and cell class;
- `phase4-regret-summary.csv`, grouped by `on`, `off`, and `undecidable`;
- `cell-fibers.json`, recording signature-bin edges and on/off classification
  evidence;
- `PHASE4_THREEBODY_RESULTS.md` or an appended Exit Status section here.

## 8. Outcome Branches

1. **On-set fail.** If the on-set regret CI excludes `0` above the threshold,
   halt and falsify Postulate 1 on a real substrate. Do not rescue by redefining
   `Φ`, moving cells, or switching to oracle comparison.
2. **Off-set success.** If the off-set regret CI includes `0`, reopen Phase 3:
   the measured off-cell did not instantiate the predicted fiber conflict, or
   the boundary theorem's empirical mapping is wrong.
3. **Clean pass.** If on-set regret is zero within CI and off-set regret is
   bounded away from zero, Phase 4 closes positive and Phase 5 entry opens.
4. **Near-threshold ambiguity.** If a CI touches `0` on the wrong side or the
   on/off classifier is `undecidable` for more than one third of candidate cells,
   add exactly one midpoint envelope slice before changing status. Choose the
   nearest implicated boundary dimension in this order: velocity scale, radius
   scale, then sensor noise. Record the midpoint command before running it.

## Exit Status

Phase 4 result: **blocked / open.**

Spec status: **drafted, 2026-05-16.** The user-proposed structure is accepted
with one load-bearing correction: the current three-body workbench does not yet
meet the roadmap Entry condition because the Bayesian floor is absent. The
existing harness, signature definition, regret readout, measurable-cell
vocabulary, and long-run discipline are pinned here so the missing floor can be
implemented without moving the gate. The implementation roadmap for that missing
floor is now staged in
[`PHASE4_BAYESIAN_FLOOR_BUILDOUT.md`](PHASE4_BAYESIAN_FLOOR_BUILDOUT.md).
