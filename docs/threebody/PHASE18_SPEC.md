# Three-Body Phase 18 - Radius-Only Controller Control Spec

The Phase 15→17 chain established that guarded TRACK's survival edge in the
favorable near-escape pocket is **real and precision-stable**, but is **not** a
per-step first-action counterfactual effect: 15C rejected energy steering, and
17 rejected the hazard-margin counterfactual even with the geometrically-correct
observable (the margin effect is a generic "thrust pushes inward" property —
`escapePositiveRate = 1.000` for every arm — and the delay/shuffle ablations
match or exceed guarded). The mechanism is therefore holistic, and Phase 17's
pre-registered next move asks the simplest possible reduction:

**Does a controller gated solely on `radius` reproduce guarded TRACK's survival
envelope?** If yes, the survival mechanism reduces to radius-gated thrusting and
the tidal/sensor intelligence is not load-bearing. If no, the survival edge is a
genuine multi-factor policy effect.

Phase 18 is a controller **probe** (adds ablated modes for comparison); it is not
a redesign. It does **not** retune guarded TRACK, change the guard thresholds, the
slate, or the hazard label, and it does **not** revise the locked Phase 15
Fail-Magnitude verdict or broaden the gravity claim under any outcome — it is a
mechanism diagnostic only.

## 1. Decision Lock

Operator lock: **2026-05-29**, after review of the two-rung design, matched-DeltaV
calibration rule, magnitude grid `{0.1,0.2,0.3,0.4}`, and the A/B/C comparison
bars. No Phase 18 code has been written and no Phase 18 command has been run at
lock time.

- **Frozen reference + thresholds.** `track_sensor_accel_guarded`, the Phase 13
  passive-derived per-cell guard thresholds (`deriveGuardThresholds`:
  `trackGuardMinRadius = quantile(non-hazard passive radii, 1−q)`, q=0.75), the
  thrust limit `0.4`, sensor settings, and the hazard geometry
  (`r3 > 4 OR minPrimaryDistance < 0.08`) are all unchanged. The radius-only modes
  **inherit** the same per-cell `trackGuardMinRadius` — no new threshold parameter.
- **Frozen slate (dt=0.004).** The Phase 15B/15C/17 lock geometry: `near_escape`,
  mass ratios `0.01,0.3,1`, radius scales `1.025,1.05,1.075`, velocity scales
  `0.95,1.05,1.1,1.15`, thrust limit `0.4`, sensor noise `0`, hazard-quantile guard
  `0.75`, eight seeds, duration `16`. Favorable pocket = `velocityScale ≥ 1.05`;
  `0.95` is a boundary control.
- **Two additive modes only.** Register `track_radius_guard` and
  `track_radius_inward` (§4). They are **not** in `PHASE14_ABLATION_MODES`. No
  existing mode, `shouldRunGuardedTrack`, or `deriveGuardThresholds` is changed →
  Phase 13/14 paths are byte-identical.
- **Matched-duty calibrated on ΔV, blind to survival.** The rung-2 inward magnitude
  is selected from a pre-registered grid by closest mean favorable-pocket
  `totalDeltaV` to guarded TRACK, never by survival (§4). matched-m + ΔV ratio are
  recorded in `PHASE18_RESULTS.md` **before** the measurement lock.
- **No peeking.** The comparison metric, bars, magnitude grid, and selection rule
  are locked **before** the measurement run. Any change requires a filed amendment
  + a fresh run.
- **No claim upgrade.** All branches preserve the locked Phase 15 Fail-Magnitude
  verdict. A "reduces to radius" read *explains* the mechanism; it does not convert
  Phase 15 to Pass or broaden the gravity claim. Any verdict revision would require
  a fresh pre-registered lock with the metrics fixed in advance.

## 2. Scope

Phase 18 owns:

- two additive controller modes in `public/js/threebody-core.mjs`
  (`track_radius_guard`, `track_radius_inward`) + a `radiusInwardMagnitude` config;
- the `--radius-inward-magnitude` flag in `scripts/threebody-operating-envelope.mjs`;
- npm scripts: `threebody:phase18:smoke`, `threebody:phase18:calibrate`,
  `threebody:phase18:control`;
- outputs under `results/threebody/phase18-radius-control-*`;
- this spec and a future `PHASE18_RESULTS.md`.

Phase 18 does **not** own a new guard threshold, a new survival metric, sensor
redesign, alternate hazard labels, audit instrumentation, 3D/spatial extension, or
any public-copy change. The comparison reuses the existing `candidateEnvelope` +
`survivalDeltaVsPassive` machinery (harness `summarizeRows`).

## 3. Command Shape

Reserved until the implementation commit adds the modes, flag, and npm scripts.

**Smoke** (column/sign sanity, capped under the ten-minute rule): the favorable
smoke cell (`massRatio=1`, `dt=0.004`, radius `1.025`, velocity `1.1`, one seed,
duration `4`) over `off,track_sensor_accel_guarded,track_radius_guard,track_radius_inward`.

**Calibration** (favorable pocket only; selects matched-m): for each
`m ∈ {0.1,0.2,0.3,0.4}`, run `track_sensor_accel_guarded,track_radius_inward` with
`--radius-inward-magnitude m`; compare mean favorable-pocket `totalDeltaV`. guarded's
ΔV is the reference (magnitude-independent); matched-m = `argmin_m |meanΔV(inward,m)
− meanΔV(guarded)|`. Record matched-m + the ΔV ratio in `PHASE18_RESULTS.md`.

**Measurement lock** (full grid, plain controller run — no audits, ~1,152 trials):
`off,track_sensor_accel_guarded,track_radius_guard,track_radius_inward` with
`--radius-inward-magnitude <matched-m>`. The implementation commit records the smoke
rate probe; the lock is cheap (no oracle/audit overhead) but operator-staged.

Hard-void gates (`npm run threebody:phase13`, `:phase14`) must be byte-identical
before lock interpretation (the modes are additive). Either deviation voids Phase 18.

## 4. Mode Definitions

The test-particle radius is `radius = sqrt(state[4]² + state[5]²)`. Both modes gate
on the **inherited frozen** per-cell `trackGuardMinRadius`.

- **Rung 1 — `track_radius_guard` (radius-only guard).** Identical to
  `track_sensor_accel_guarded` — same noisy tidal-gradient direction and same
  magnitude rule `min(|0.5·(tidalMagnitude − targetTidal)|, thrustLimit)` — **except**
  the guard predicate is `radius ≥ trackGuardMinRadius` **only**, dropping the
  `localAcceleration ≤ trackGuardMaxLocalAcceleration` and
  `tidalMagnitude ≤ trackGuardMaxTidalMagnitude` conditions of `shouldRunGuardedTrack`.
  Question: *is the guard just its radius condition?* (Duty naturally comparable —
  same thrust rule, fires somewhat more often.)
- **Rung 2 — `track_radius_inward` (full radius-only).** Gate on
  `radius ≥ trackGuardMinRadius`; when open, thrust **radially inward**:
  `thrust = −m · (state[4], state[5]) / radius`, fixed magnitude `m =
  radiusInwardMagnitude` (the calibrated matched-m). **No tidal sensing, no gradient,
  no error term.** Question: *does survival reduce to radius-gated inward thrust?*

**Matched-duty (rung 2).** The magnitude grid `{0.1,0.2,0.3,0.4}` and the
`argmin |ΔV| ` selection rule are locked here; matched-m is computed in the
calibration pass on `totalDeltaV` alone, blind to survival. The full m-sweep survival
is reported as a sensitivity curve in the readback for transparency.

## 5. Comparison Metric

Favorable pocket (`velocityScale ≥ 1.05`), each radius-only mode vs
`track_sensor_accel_guarded`, using the existing per-cell reducer
(`candidateEnvelope = survivalDeltaVsPassive ≥ 0.001 AND worsenedRate ≤ 0.1`):

1. **Candidate-cell overlap** — Jaccard of the favorable candidate-envelope cell
   sets: `|C_radius ∩ C_guarded| / |C_radius ∪ C_guarded|`.
2. **Survival-delta agreement** — over guarded's favorable candidate cells, the
   fraction with `|survivalDeltaVsPassive(radius-only) − survivalDeltaVsPassive(guarded)|
   ≤ 0.10`.

Secondary: mean favorable survival-delta per mode; per-mass-ratio split; the ΔV
ratio at matched-m; the full inward m-sweep survival sensitivity; the boundary
control (`v=0.95`).

## 6. Pre-Registered Branches

**(A) Reduces to radius (mechanism explained).** The matched-duty rung-2
(`track_radius_inward`) **and/or** rung-1 (`track_radius_guard`) reaches
**Jaccard ≥ 0.60 AND ≥ 2/3** of guarded's favorable candidate cells within the
`0.10` survival band. Interpretation: guarded TRACK's survival reduces to
radius-gated thrusting; the tidal/sensor intelligence and the extra guard conditions
are not load-bearing for survival. (Closes Phase 17's open mechanism question.)

**(B) Does not reduce (multi-factor).** Both radius-only modes have
**Jaccard ≤ 0.30 OR** mean favorable survival-delta below **½** of guarded's.
Interpretation: the tidal sensing / precise guard / steering is load-bearing; the
survival edge is a genuine multi-factor policy effect that radius alone cannot
reproduce.

**(C) Partial.** Anything between: rung-1 reproduces but rung-2 does not (the guard
reduces to radius but the steering does not), or vice versa; reproduction only in
some mass ratios; or only outside the matched-duty point. Interpretation: Phase 18
localizes which controller component carries the survival edge.

All branches preserve the Phase 15 formal verdict. No branch promotes Phase 15 or
broadens the gravity claim.

## 7. Readback

After the smoke: command + wall-clock; both modes thrust; sign sanity (rung-2
thrusts inward when radius is high; rung-1 fires more often than guarded); rate probe.

After calibration: mean ΔV per mode/magnitude; matched-m + ΔV ratio (recorded
before the measurement lock).

After the measurement lock: hard-void gate status; branch (A/B/C); candidate-cell
Jaccard + survival-band fraction for each radius-only mode; per-mass-ratio split;
the inward m-sweep sensitivity; and whether the next registered move is a refined
controller decomposition, a claim-language update, or closure of the mechanism line.
