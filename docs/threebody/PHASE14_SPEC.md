# Three-Body Phase 14 - Mechanism Decomposition and Action Coupling Spec

This document is the implementation-grade spec for Phase 14 of
[`../SUNDOG_V_THREEBODY.md`](../SUNDOG_V_THREEBODY.md). Phase 13 showed that
guarded accelerometer TRACK improves survival over passive and naive baselines
across a mapped high-velocity near-escape pocket through a 16-second tested
horizon, with the low-velocity `velocityScale=0.95` boundary preserved. Phase 14
asks whether that win is because the accelerometer/tidal signal is the operative
causal handle, or because the frozen guard mostly suppresses bad thrust in the
tested pocket. The binding scope source is the "Phase 14 - Mechanism
Decomposition and Action Coupling" section of
[`CROSS_SUBSTRATE_NOTES.md`](CROSS_SUBSTRATE_NOTES.md); this spec must not exceed
the claims permitted there.

Where this spec and the roadmap disagree, the roadmap wins. Where both are
silent, this spec is authoritative for Phase 14.

## 1. Decision Lock

Phase 14 starts with seven pinned calls:

- **The Phase 13 guard is frozen, not re-derived.** Every arm reuses the
  Phase 13 `hazard_quantile` `0.75` freeze and the identical constant fallback
  sweeps; the guard predicate is byte-identical across all arms. A run
  restricted to `off` and `track_sensor_accel_guarded` over the lock slate must
  reproduce the Phase 13 lock numbers bit-for-bit or the run is void.
- **Same Phase 13 locked slate.** Mass ratios, timesteps, radii, velocities,
  seeds, and duration are exactly the `phase13-long-horizon-lock` slate,
  including the `velocityScale=0.95` boundary cells; only the controller arm
  varies.
- **Five controller arms, pre-registered and fixed.** Frozen guarded TRACK plus
  signal-shuffled, action-shuffled, signal-delayed, and sign-flipped. Passive,
  naive, and oracle stay on the slate so outcome effect is measured against
  passive and naive. No arm is added or dropped after observing results.
- **Three mechanisms are separate columns, never collapsed.** Warning quality,
  action coupling, and outcome effect are reported as distinct columns; a
  survival gain that does not require intact action coupling is not a controller
  claim.
- **Ablation parameters are pinned before the run.** Signal and action shuffles
  are within-trial temporal permutations with a fixed mode-hashed per-trial seed
  and a causal earlier-or-equal clamp; the signal delay is
  `round(0.5 simulated seconds / dt)` steps with a neutral warmup; sign-flip
  negates the post-guard thrust vector. Cross-trial shuffles and signal-level
  sign-flip are rejected.
- **Boundary negatives stay visible.** `velocityScale=0.95` and equal-mass best
  cells are reported as their own rows in every column and are never averaged
  into the pocket verdict.
- **Phase 14 cannot upgrade the claim.** It may only confirm, narrow, or
  retract the mechanism behind the Phase 13 earned wording. It does not extend
  scope, horizon, or sensor model.

## 2. Scope

Phase 14 owns:

- The four ablation controller modes and the action-coupling instrumentation,
  added additively to the shared `scripts/threebody-operating-envelope.mjs` and
  `public/js/threebody-core.mjs` behind new mode names
- `npm run threebody:phase14:smoke`
- `npm run threebody:phase14`
- Outputs under `results/threebody/phase14-mechanism-decomposition-smoke/`
- Outputs under `results/threebody/phase14-mechanism-decomposition-lock/`
- Result note [`PHASE14_RESULTS.md`](PHASE14_RESULTS.md)
- Roadmap and writeup receipt bullets after the lock lands

Phase 14 does **not** own:

- Controller redesign or guard retuning
- Forward-oracle precision lock (Phase 15)
- Spatial/3D extension (Phase 16)
- New sensor models
- Any earned-claim upgrade
- Retuning or arm changes after observing the result

## 3. Commands

Smoke:

```bash
npm run threebody:phase14:smoke
```

Full lock:

```bash
npm run threebody:phase14
```

The smoke covers mass ratio `1`, timestep `0.01`, radii `1.025` and `1.075`,
velocities `0.95` and `1.1`, seven controller modes
(`off`, `naive`, `track_sensor_accel_guarded`,
`track_sensor_accel_signal_shuffle`, `track_sensor_accel_action_shuffle`,
`track_sensor_accel_signal_delay`, `track_sensor_accel_sign_flip`, `oracle`),
and two seeds at `duration=16`. That is 56 trials, expected to take about a
minute.

The full lock covers:

- mass ratios: `0.01`, `0.3`, `1`
- timesteps: `0.008`, `0.01`, `0.012`
- radii: `1.025`, `1.05`, `1.075`
- velocities: `0.95`, `1.05`, `1.1`, `1.15`
- modes: `off`, `naive`, `track_sensor_accel_guarded`,
  `track_sensor_accel_signal_shuffle`, `track_sensor_accel_action_shuffle`,
  `track_sensor_accel_signal_delay`, `track_sensor_accel_sign_flip`, `oracle`
- seeds: `8`
- duration: `16`

That is 6,048 trials and 5,184 paired non-passive rows. The Phase 13 lock of
3,456 trials took about an hour, so the linear estimate is about one hour and
forty-five minutes. Treat the full lock as a staged operator run under the
repository's long-run rule.

The ablation modes and the action-coupling metric are added additively to the
shared envelope harness; existing Phase 9, 11, and 13 scripts pass fixed mode
lists that do not include the new names and must remain byte-identical in
behavior. Before the ablation arms are trusted, a Phase 14 run restricted to
`off` and `track_sensor_accel_guarded` over the full lock slate must reproduce
the Phase 13 lock result exactly: 3,456 trials, 88 / 324 candidate envelope
rows, and 81 promising best cells. If it does not, the shared-harness edit broke
the guard freeze and the run is void.

## 4. Metrics

Read these files first:

- `aggregate-envelope.csv`
- `candidate-envelope.csv`
- `best-by-cell.csv`
- `cell-class-map.csv`
- `cell-warning-quality-map.csv`
- `trial-outcomes.csv`

Primary metrics, reported per arm as separate columns:

- Warning quality: passive mean tidal-magnitude AUROC (primary) and passive
  mean tidal warning lead-time (secondary)
- Action coupling: thrusting-step intended-direction agreement rate (primary)
  and signed effect size (secondary), with the action-shuffled arm as the
  contrast baseline
- Outcome effect: candidate envelope rows out of total, best-cell class
  balance, survival delta versus passive, worsened rate versus passive, mean
  simulated-time delta versus passive, mean delta-v, mean delta-v per simulated
  second, dominant failure mechanism

Pre-registered numeric bars:

- "High warning quality" means passive mean tidal-magnitude AUROC `>= 0.70` in
  the favorable high-velocity pocket.
- "Weak action coupling" means mean intended-direction agreement `<= 0.55`, or
  agreement less than `0.05` above the action-shuffled arm.

Phase 14-specific checks:

- The frozen guarded-TRACK arm should show high warning quality and strong
  action coupling; the signal-shuffled, signal-delayed, and action-shuffled arms
  should each lose the candidate envelope and drop action coupling toward
  chance; the sign-flipped arm should invert the action-coupling sign.
- The action-shuffled arm is the decisive contrast: if it reproduces the frozen
  arm's outcome effect within noise, the Phase 13 win is guard-mediated thrust
  suppression rather than signal-directed action.
- Low-velocity boundary cells stay explicit per arm even when negative.
- The Phase 13-equivalence regression check (Section 3) is reported as a
  pass/fail line before any ablation column is interpreted.

## 5. Pre-Registered Branches

**Pass:** in the favorable high-velocity pocket the frozen guarded accelerometer
TRACK arm shows high warning quality and strong action coupling, and its outcome
effect remains a positive candidate envelope against passive and naive
baselines, while the signal-shuffled, signal-delayed, and action-shuffled arms
each lose the candidate envelope and drop action coupling toward chance and the
sign-flipped arm inverts the action-coupling sign. This supports the claim that
the accelerometer/tidal signal is the operative causal handle, not merely a
guard that suppresses bad thrust in the tested pocket.

**Partial:** the frozen arm keeps high warning quality and a positive outcome
effect, but action coupling is only moderately above the action-shuffled arm, or
the action-shuffled arm retains a non-trivial share of the survival benefit. The
earned claim stays "guarded TRACK improves survival in the mapped pocket" with
an explicit caveat that part of the benefit is guard-mediated thrust suppression
rather than signal-directed action, and no stronger causal-handle claim is made.

**Fail:** the frozen arm shows high warning quality but weak action coupling,
and/or the action-shuffled arm reproduces the frozen arm's outcome effect within
noise. High warning quality with weak action coupling does not support a
controller claim: the signal forecasts hazards but thrust does not move it in
the intended direction, so the Phase 13 win is attributable to the guard
suppressing bad thrust in the tested pocket. Phase 14 is then a useful negative
result and the next project step is control-handle or controller redesign, not
bigger sweeps.

## 6. Readback Template

After the full lock finishes, update [`PHASE14_RESULTS.md`](PHASE14_RESULTS.md)
with:

- command and wall-clock runtime
- total trial count and terminal outcome counts
- the Phase 13-equivalence regression check result
- warning quality, action coupling, and outcome effect as a per-arm table
- candidate-envelope count and best-cell class balance per arm
- the action-shuffled contrast read
- favorable-pocket read
- boundary-cell read
- pre-registered branch taken
- claim wording to preserve, narrow, or retract
