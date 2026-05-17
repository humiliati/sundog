# Phase 4 Bayesian-Floor Buildout

> Buildout roadmap for the Bayesian-floor controller required by
> [`PHASE4_THREEBODY.md`](PHASE4_THREEBODY.md).
> Status: staged/open, 2026-05-16. BF-4b off-set calibration has a first
> negative receipt: the pre-registered cell classified `off`, but the off-set
> regret CI was `[0, 0]`, so BF-5 remains blocked. No Phase 4 proof run is
> admitted until the floor, regret reducer, capped-probe receipt, and a passing
> BF-4b receipt land.

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
contract and must be recorded in the manifest. Ties resolve in this order
(reconciled 2026-05-16 after the BF-4 probe; see Planning Objective):

1. include the guarded-signature policy itself as candidate `0`;
2. deviate from that same-information baseline only if the best predicted
   shaped score exceeds it by at least `signatureAdvantageDtMultiplier * dt`;
3. inside that guard, maximize the **shaped planning score** (Planning
   Objective below);
4. if within `1e-9` of the best shaped score, minimize expected `totalDeltaV`;
5. if still tied, choose the first action in the pre-registered order.

The pre-BF-4 wording "maximize expected `T_safe / T_max`; if tied within one
integrator step, minimize `totalDeltaV`" is superseded: the dt-wide tie band
combined with a horizon-flat survival estimate is exactly what degenerated the
floor to passive. The shaped score (which already folds a sub-dt safety-margin
discriminator) replaces it; the `1e-9` band only collapses genuinely identical
trajectories.

The candidate order should put the guarded-signature policy first, then the
existing zero/axis/diagonal lattice from `oracleCandidateThrusts`. Lattice
candidates are scored as one-step deviations followed by the guarded-signature
policy on the rollout particles. If implementation discovers a mismatch with
the existing helper, update this document before the proof run rather than
silently changing the order.

### Planning Objective

For each candidate action, propagate particles forward with the existing
integrator. The score is **not** raw within-horizon survival time (the BF-4
probe proved that is non-discriminative — see BF-4 Probe Receipt). The pinned
objective is the **shaped planning score**:

```text
score(action) = E[ survivalTime_within_horizon
                    + shapeFraction * dt * terminalSafetyMargin ]
terminalSafetyMargin = min( escape-radius margin, close-approach margin )
                       of the horizon-end state, in [0,1];
                       0 if a terminal hazard was reached during the rollout.
```

**Floor-validity invariant (load-bearing):** `shapeFraction ∈ [0, 1)`, so the
shaping term is strictly less than one `dt`. A candidate whose true survival is
longer by at least one `dt` therefore can never be overtaken on margin alone —
the floor never prefers an earlier-escape action, so it remains a valid lower
bound. `shapeFraction = 0` recovers the old pure-survival objective; the
intended `π*_Bayes` (`argmax E[T_safe/T_max | h]`) is recovered as
`shapeFraction → 0` with horizon → escape timescale. Shaping is an internal
tractable surrogate for that argmax under a finite horizon; it does **not**
change `J`, `Φ`, `μ`, the regret readout, or the Phase 4 gate.

The second BF-4 shaped re-probe showed that shaped-margin differences alone can
still authorize destructive max-thrust actions when within-horizon survival is
flat. Therefore the evaluator now uses a **signature-baseline guard**: the
guarded-signature policy is an explicit candidate, and the Bayes floor only
deviates from it when predicted shaped-score advantage is at least
`signatureAdvantageDtMultiplier * dt` (default `1`). This is conservative but
valid for BF-4: the signature policy is admissible under the same `Φ` history,
so the floor approximation must not be worse than it before Phase 4 can
interpret regret.

The final horizon, particle count, resampling threshold, `shapeFraction`, and
`signatureAdvantageDtMultiplier` are locked only after the (re-)capped probe
records a rate and a passing floor-sanity result.

Start with this smoke profile:

```text
particle-count: 256
planning-horizon-steps: 16
resample-threshold: 0.5
shape-fraction: 0.5
signature-advantage-dt-multiplier: 1
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

#### BF-4 Probe Receipt (2026-05-16, `bf4-probe-20260516-173223`)

First capped probe, 1 envelope cell (`mu_1 dt_0.01 r_1.075 v_1.1 thrust_0.4
noise_0`, near_escape, 2 seeds). Recorded:

- **Rate / runtime gate: EXCEEDED.** Bayes-floor step
  `00:32:25 → 00:46:55` ≈ 14.5 min for 2 trials + per-cell passive guard
  calibration ⇒ **~7 min per Bayes trial** at smoke settings (256 particles,
  horizon 16). The PHASE4_THREEBODY §3 proof grid (~216 cells × seeds + per-cell
  calibration) extrapolates to a multi-day run. **Disposition: BF-5 / full lock
  must move to a long-budget runner; no inline or local expansion.**
- **Join / caseId-drift guard: PASS.** `joinedRowCount = 2`; the pre-run
  zero-join assertion held (finding-3 trap did not bite this run). Keep the
  assertion before any future probe.
- **Floor-sanity gate: NON-DECISIVE — root cause found.** Negative regret on
  both rows (`regret ≈ −0.60`). Diagnosis from `bayes-actions.csv`: the floor
  chose zero thrust on 643/649 steps and `expected_safe_time` was identically
  `0.16` (= horizon 16 × dt 0.01) across **all nine** candidates. Escape in the
  near-escape pocket occurs at ~6 time units (~600 steps), so within any
  16-step rollout no particle reaches a terminal event — every candidate
  accrues the full horizon, the dt-wide tie band flattens them, and the ΔV
  tie-break forces zero thrust. The floor degenerated to the passive `off`
  controller (ΔV ≈ 0.024 vs the signature controller's ≈ 1.99), so it is not a
  valid lower bound and the gate is correctly held un-evaluated.
- **Repair applied (BF-2 design change, 2026-05-16):** the shaped planning
  objective + reconciled tie order above. A tiny sanity smoke confirmed the
  shaped score now discriminates candidates (terminal-margin spread) and the
  floor selects steering actions instead of passive zero thrust. Floor-validity
  invariant preserved by construction.
- **Required before BF-5:** a fresh capped re-probe on the long-budget runner
  with the shaped objective, confirming floor-sanity PASS (negative-regret rate
  ≤ 5%) and recording the new per-trial rate; **and** a passing BF-4b off-set
  guard-calibration receipt (see below) — floor-sanity alone is no longer
  sufficient to stage the full lock.

#### BF-4 Shaped Re-Probe Receipt (2026-05-16, `bf4-shaped-reprobe-20260516-180425`)

Second capped probe, same 1-cell / 2-seed slate. Recorded:

- **Rate / runtime gate: EXCEEDED.** Elapsed wall-clock was 26.82 min; Bayes
  `01:04:27 -> 01:31:14` UTC. This confirms BF-5 / full lock must remain on a
  long-budget runner.
- **Join / caseId-drift guard: PASS.** `joinedRowCount = 2`.
- **Floor-sanity gate: NON-DECISIVE.** Negative regret on both rows
  (`T_safe_bayes = 7.73, 15.42` vs signature `16, 16`; global negative-regret
  rate `1`). Nonzero action rows were `1492 / 2317`, so the passive-degeneracy
  repair worked, but the floor became an active worse-than-signature controller.
- **Root cause:** expected survival remained flat at the short horizon; tiny
  shaped-margin differences chose full-thrust lattice actions. The lattice did
  not include the guarded-signature policy itself, even though that policy is
  admissible under the same signature history and survived both rows.
- **Repair applied (BF-2 design change):** add the guarded-signature policy as
  candidate `0`; score lattice candidates as one-step deviations followed by
  the guarded-signature rollout policy; require predicted advantage of at least
  `signatureAdvantageDtMultiplier * dt` before deviating from the signature
  baseline. Re-probe required again before BF-5.

#### BF-4 Signature-Guard Re-Probe Receipt (2026-05-16, `bf4-shaped-reprobe-20260516-191423`)

Third capped probe, same 1-cell / 2-seed slate, with
`signatureAdvantageDtMultiplier = 1`. Recorded:

- **Join / caseId-drift guard: PASS.** `joinedRowCount = 2`.
- **Floor-sanity gate: PASS.** `globalNegativeRegretRate = 0`,
  `floorSanity.status = floor_sanity_pass`; both rows had
  `T_safe_bayes = T_safe_signature = 16`, `regret = 0`, and
  `fuel_excess = 0`.
- **Signature-baseline guard receipt: PASS.** `3202 / 3202` Bayes action rows
  used `action_family = signature_policy`; `974 / 3202` rows were nonzero
  thrust, matching the guarded signature controller rather than passive zero
  thrust.
- **Runtime gate: EXCEEDED.** Bayes manifest records
  `2026-05-17T02:14:25.659Z -> 2026-05-17T03:39:01.236Z`, about 84.6 minutes
  for 2 Bayes trials, or about **42.3 minutes per Bayes trial** at the probe
  settings. The serial full proof grid (216 cells x 8 seeds = 1728 Bayes
  trials, before signature/passive harness cost) extrapolates to roughly
  **51 days**. BF-5 must therefore be sharded/resumable on a long-budget runner;
  a single monolithic local or interactive-agent command is not admissible.
- **Operational repair applied:** the Bayes evaluator wrote complete artifacts
  but left an idle Node process alive after `completedAt`; the script now exits
  explicitly after successful writes. Re-run the small receipts before any BF-5
  shard.

### BF-4b: Off-Set Guard Calibration (hard pre-BF-5 gate)

**Why this gate exists.** The signature-baseline guard makes the floor
`max(signature_policy, confident-deviation) ≥ signature` *by construction*.
Two consequences, pinned here so they are not over-read later:

1. `floor_sanity_pass` (negative-regret ≤ 5%) is now **near-tautological**: a
   completely inert planner that always falls back to the signature policy also
   passes it. Floor-sanity therefore certifies only *floor ≥ signature*
   (floor-validity), **not** that the floor is a good Bayes proxy. It is no
   longer a floor-quality gate.
2. All gate-discriminating power now rests on the **off-set arm** of the Phase 4
   gate (regret bounded away from 0 where signature is insufficient). Its
   sensitivity is governed entirely by `signatureAdvantageDtMultiplier`
   (default `1`), a value chosen for floor-*validity* and never calibrated for
   off-set *detection power*. Too high ⇒ a real Bayes advantage is suppressed,
   the floor collapses to signature off-set, regret → 0 off-set, and the gate
   misreads it as "boundary wrong → reopen the closed-positive Phase 3" or
   masks a true separation. Too low ⇒ the floor deviates on noise and can drop
   below signature, breaking floor-validity. Discovering either *after* the
   ~51-day full lock is the expensive failure mode this gate prevents.

**Pre-registered off-set calibration cell.** The cell must satisfy a
*demonstrated-headroom* criterion, not just a physics hunch: a valid
pre-registered off-set cell is one where (i) §5 classifies it `off` **and**
(ii) the privileged oracle yardstick beats the signature policy on it with
regret-headroom 95% CI lower bound `> 0` (the Satisfiability Probe). Criterion
(ii) is what makes the off-set arm physically able to fire for *some* floor;
without it an `off` label is necessary but not sufficient. The current
candidate cell (below) was chosen by physics hunch and **must pass the
Satisfiability Probe before it is treated as pre-registered**; if it fails,
re-pick by scanning the §3 grid for a cell maximizing oracle-minus-signature
headroom subject to a `§5 off` label, then re-register here:

```text
regime: near_escape
mass-ratio: 1
timestep: 0.01
radius-scale: 1.075
velocity-scale: 1.15      # high-velocity near-escape stress (vs 1.1 on-set probe)
sensor-noise-sweep: 0.01  # noisy Φ: history/belief should beat reactive
thrust-limit: 0.4
track-guard-mode: hazard_quantile
track-guard-quantile: 0.75
track-guard-min-radius-sweep: 1.15
track-guard-max-local-acceleration-sweep: 2.5
track-guard-max-tidal-magnitude-sweep: 35
duration: 16
seeds: ≥ 8   # enough for the §5 fiber classifier to decide on/off, not undecidable
```

**Acceptance criteria (same readout, off-set definition, and CI as the real
gate — only the slate is one pre-registered cell at smoke scale).** All must
hold before BF-5 may stage the full lock:

0. **The cell is satisfiable (cell-validity gate; runs first).** A `§5`-`off`
   fiber label does **not** imply regret headroom: if the *signature* policy
   already reaches the `T_safe = duration` cap (or no admissible controller can
   beat it on this cell), then regret is structurally `0` for *every* floor and
   the off-set arm cannot fire there — the cell, not the floor, is the problem.
   Before any floor calibration on the cell, the **BF-4b Satisfiability Probe**
   below must show privileged-yardstick headroom over signature (oracle beats
   signature, 95% CI lower bound `> 0`). If it does not, the pre-registered
   cell is vacuous: re-pick it by the strengthened criterion below and
   re-register — do **not** build or tune any more floor on a vacuous cell.
1. **The cell is empirically off-set.** The PHASE4_THREEBODY §5 fiber
   classifier labels it `off` on the smoke slate (a positive-mass fiber with a
   disjoint Bayes-optimal action set). If it comes back `undecidable`, raise
   only this cell's seed count until it decides — do not reinterpret an
   undecidable cell as a pass.
2. **The off-set arm fires.** On that off-set cell, the floor with the chosen
   `signatureAdvantageDtMultiplier` yields regret whose 95% paired-bootstrap CI
   lower bound is strictly `> 0` (the exact §3 off-set pass condition). This
   demonstrates the floor *can and does* beat the signature policy where
   signature is insufficient — i.e. the guard is not so conservative that it
   suppresses the gate.
3. **Floor-validity still holds.** On the existing sufficient/on-set probe cell
   the negative-regret rate stays ≤ 5% (already shown), so the multiplier is
   not so low that the floor drops below signature.

If (2) fails while (1) holds, the multiplier is too conservative *or* the
particle-MPC floor is too weak off-set: retune `signatureAdvantageDtMultiplier`
(or repair the floor) and re-run BF-4b — **do not stage the 51-day BF-5 lock
on an off-set arm that has never been shown to fire.** The chosen multiplier is
pre-registered only once (1) ∧ (2) ∧ (3) hold simultaneously; that triple is
the recorded justification, replacing the floor-validity-only rationale in the
Planning Objective.

This is a hard gate: **BF-5 staging is blocked until a BF-4b receipt records
(0) ∧ (1) ∧ (2) ∧ (3) PASS.** It is cheap (one extra cell at smoke scale)
relative to the full lock and converts the off-set arm's sensitivity from an
untested default into a pre-registered, validated quantity.

#### BF-4b Satisfiability Probe (criterion 0; no new code; operator-gated)

Purpose: decide whether the off-set arm is even *satisfiable* on the candidate
cell before any further floor building, using the existing envelope harness and
the privileged oracle **as a yardstick only** (it is not the floor and is never
admitted as `π*_Bayes`; this is the sanctioned reference use). This runs before
criteria (1)–(3).

Compute discipline: `forward_oracle_strict` is the expensive mode (~24 s/trial
in prior receipts); 1 cell × 8 seeds × 3 modes may exceed the ~10-min inline
budget. **Do not run inline** — stage to the operator/long-budget runner; the
agent only pins the command and the decision.

```powershell
$cell = "results\proof\phase4\_bf4b-satisfiability"
node scripts/threebody-operating-envelope.mjs `
  --phase phase4-bf4b-satisfiability `
  --out $cell `
  --regimes near_escape --modes off,track_sensor_accel_guarded,forward_oracle_strict `
  --mass-ratios 1 --timesteps 0.01 --radius-scales 1.075 `
  --velocity-scales 1.15 --thrust-limits 0.4 --sensor-noise-sweep 0.01 `
  --track-guard-mode hazard_quantile --track-guard-quantile 0.75 `
  --track-guard-min-radius-sweep 1.15 `
  --track-guard-max-local-acceleration-sweep 2.5 `
  --track-guard-max-tidal-magnitude-sweep 35 `
  --seeds 8 --duration 16
```

Headroom readout (same `T_safe`/`T_max`/bootstrap-CI machinery as the gate):
`headroom_i = (T_safe(forward_oracle_strict, i) − T_safe(track_sensor_accel_guarded, i)) / T_max`,
mean with 95% paired-bootstrap CI over the 8 seeds.

Decision branches (pre-registered):

- **Vacuous cell** — signature already at the `T_safe = duration` cap, or the
  oracle-minus-signature headroom CI includes `0`: the off-set arm cannot fire
  here for *any* floor. The cell choice was wrong, not the floor. **Re-pick**:
  scan the §3 grid for the cell maximizing oracle-minus-signature headroom
  subject to a `§5 off` label, re-register it above, and only then resume floor
  work. No particle-MPC / exact-DP building until a non-vacuous cell exists.
- **Satisfiable, privileged-only** — headroom CI lower bound `> 0` but the
  advantage comes from forward lookahead / full state: the arm is non-vacuous
  but may need a stronger same-`Φ` floor than particle-MPC. Escalate to the
  (B) decision (exact belief-grid DP vs. a structural floor redesign) with the
  knowledge that headroom genuinely exists.
- **Satisfiable** — headroom CI lower bound `> 0`: proceed to criteria (1)–(3)
  with the (still open) floor-objective work, now justified because the arm
  provably can fire.

#### BF-4b Receipt (2026-05-16, `bf4b-offset-guard-20260516-215056`)

Operator-gated smoke ran the pre-registered cell with
`signatureAdvantageDtMultiplier = 1`, 8 seeds, `particleCount = 256`,
`planningHorizonSteps = 16`, `shapeFraction = 0.5`, and
`sensorNoiseStd = 0.01`. Artifacts:
`results/proof/phase4/bf4b-offset-guard-20260516-215056/`.

Acceptance result: **FAIL; BF-5 remains blocked.**

- Criterion (1), classifier: **PASS.** The reducer joined 8 rows and every row
  classified as `off`.
- Criterion (2), off-set arm: **FAIL.** Off-set mean regret was `0`, with 95%
  CI `[0, 0]`; the floor exactly matched the signature policy on every joined
  seed.
- Criterion (3), floor-validity: **PASS.** `floor_sanity_pass` with global
  negative-regret rate `0`.

Diagnostics: all 4,175 Bayes action rows selected `signature_policy`; 806 of
those rows were nonzero only because the signature policy itself thrusts. The
largest recorded pre-guard score advantage over the signature baseline was
`0.000001`, i.e. `0.0001 dt`, far below the configured threshold
`signatureAdvantageDtMultiplier * dt = 0.01`. This is the failure mode the
BF-4b gate was meant to catch: the floor is valid but the off-set arm does not
fire. Retune `signatureAdvantageDtMultiplier` or repair the floor/objective and
re-run BF-4b before staging BF-5.

#### BF-4b Follow-up (2026-05-17): energy-trend objective tried, then bounded out

Per the pre-registered "escalate, don't keep turning knobs" bound, two repair
attempts were made and capped:

- **Multiplier retune rejected by diagnostic.** The largest pre-guard
  advantage (`0.000001`) is numerically ~0, not a real advantage clipped by a
  too-high threshold. Lowering `signatureAdvantageDtMultiplier` would
  manufacture noise-driven, non-real deviations and break floor-validity — so
  the multiplier is not the lever.
- **Energy-trend terminal value implemented, smoke still inert.** The bounded
  radius/close-approach margin was replaced with a self-scaled energy-trend
  margin (`clamp(0.5 + 0.5·(E_start − E_end)/max(|E_start|,1e-9), 0, 1)`,
  belief-only). It is the correct terminal value and is **retained**, but a
  tiny sanity smoke still showed max pre-guard advantage `~1e-6`. Root cause is
  structural and upstream of the terminal-value form: candidates differ by
  exactly one thrust step then follow the signature policy over a horizon ~40×
  shorter than the escape timescale, so all rollout end-states — and any
  function of them — are near-identical. No terminal-value form escapes a
  one-step perturbation that has not propagated.

Decision (honoring the pre-registered bound): **stop tuning the particle-MPC
floor** and run the **BF-4b Satisfiability Probe** first. If the candidate cell
is vacuous (signature already optimal / no privileged headroom), the cell — not
the floor — is the problem and must be re-picked before any further floor work.
This is the next operator-gated action; no more inline floor building until the
probe decides.

### BF-5: Full Lock Handoff

BF-4 floor-sanity is passed for the smoke slate, **but BF-5 is additionally
blocked on a passing BF-4b off-set guard-calibration receipt** (floor-sanity
alone is near-tautological under the signature-baseline guard and does not
certify the off-set arm can fire). The first BF-4b receipt failed criterion
(2), so BF-5 is not yet stageable. Assuming a later BF-4b passes, the measured
rate changes the BF-5 handoff: the full proof lock must be staged as
sharded/resumable PowerShell (or a long-budget workflow) with per-shard
manifests and merge readbacks. A single full-grid invocation is intentionally
not written here, because the measured serial cost is multi-week and violates
the compute discipline.

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
  Under the signature-baseline guard this certifies only *floor ≥ signature*
  (floor-validity), **not** floor quality — it is near-tautological on its own.
- **Off-set guard-calibration gate (BF-4b, hard pre-BF-5):** on the
  pre-registered off-set cell the §5 classifier returns `off` and the floor's
  regret 95% CI lower bound is strictly `> 0`, while the sufficient cell keeps
  negative-regret ≤ 5%. This is the actual floor-quality / off-set-sensitivity
  gate; BF-5 staging is blocked until it passes.
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
3. **Final approximation settings.** Lock particle count, horizon, resampling
   threshold, and `shapeFraction` after the capped re-probe, not before rate
   measurement. `shapeFraction` default `0.5`; must stay in `[0, 1)` to keep
   the floor-validity invariant.
4. **Signature-advantage multiplier.** `signatureAdvantageDtMultiplier`
   default `1` is *pinned* only by a passing BF-4b receipt (off-set cell regret
   CI lower > 0 ∧ sufficient cell negative-regret ≤ 5%), not by the
   floor-validity argument alone. The first BF-4b receipt with multiplier `1`
   failed the off-set CI condition, so the multiplier remains unpinned. If
   BF-4b forces a retune, the new value and its BF-4b receipt are the recorded
   pre-registration; BF-5 uses that value unchanged.

## Exit Criteria

This buildout exits when:

- unknown-mode validation exists and is tested;
- the signature observation contract is implemented and audited;
- the particle-belief Bayes evaluator writes the target receipts;
- the regret reducer and cell-fiber classifier write Phase 4 readbacks;
- a capped probe records measured rate and floor sanity;
- a BF-4b off-set guard-calibration receipt records (1) ∧ (2) ∧ (3) PASS, with
  the pinned `signatureAdvantageDtMultiplier`;
- [`PHASE4_THREEBODY.md`](PHASE4_THREEBODY.md) is updated with the exact proof
  lock command and outcome branches.

Only then does Phase 4 empirical entry open.
