# Three-Body Phase 15C - Multi-Step Counterfactual Horizon Audit Spec

Phase 15 ended as **Fail-Magnitude**: the guarded TRACK survival envelope was
stable across the precision ladder, and ablations collapsed or inverted, but
the privileged one-step counterfactual missed the pre-registered magnitude bar.
Phase 15B then showed that the `1e-9` denominator floor is real and
near-universal, but it is **not** the full explanation: floor-hit steps carry no
TRACK-specific hidden positive one-step signal, and shuffled arms can score
higher on floor-hit positive rate while failing to produce candidate cells.

Phase 15C is the pre-registered follow-on named by Phase 15B: a diagnostic
multi-step counterfactual horizon audit. It asks whether TRACK's favorable
pocket is explained by cumulative trajectory steering over short horizons
rather than by one-step energy reduction.

Phase 15C does **not** revise the Phase 15 verdict, retune the controller,
change guard thresholds, or upgrade the earned claim. It is a mechanism
diagnostic only.

## 1. Decision Lock

- **Frozen controller.** Reuse the Phase 13/15 guarded TRACK controller, Phase
  13 passive-derived guard thresholds, thrust limit, sensor settings, and
  action coupling. No guard retuning, thrust retuning, sensor retuning, or new
  controller mode is admitted.
- **Same cell slate.** Reuse the Phase 15B lock slate: `near_escape`,
  `dt=0.004`, mass ratios `0.01,0.3,1`, radius scales
  `1.025,1.05,1.075`, velocity scales `0.95,1.05,1.1,1.15`, thrust limit
  `0.4`, sensor noise `0`, hazard quantile guard `0.75`, eight seeds, duration
  `16`.
- **Same mode slate.** Reuse the Phase 15B audit modes:
  `off`, `track_sensor_accel_guarded`, `track_sensor_accel_signal_delay`,
  `track_sensor_accel_signal_shuffle`, `track_sensor_accel_action_shuffle`,
  and `track_sensor_accel_sign_flip`.
- **Audit only.** The audit is additive. Default Phase 13/14/15 paths remain
  unchanged. New receipts must be written only behind a new explicit
  multi-step audit flag.
- **Horizon set.** The frozen horizons are simulation-step counts
  `N in {4, 8, 16, 32}`. With `dt=0.004`, these correspond to
  `0.016, 0.032, 0.064, 0.128` seconds of lookahead.
- **Primary stratification.** Read results by
  `candidateEnvelope=true/false`, with `signal_delay` candidate vs
  non-candidate rows elevated as a primary diagnostic because Phase 15B found
  their floor-step sign behavior diverged.
- **No claim upgrade.** A positive Phase 15C read can support a mechanism
  explanation for Phase 15's survival pocket; it cannot convert Phase 15 to
  Pass or broaden the gravity claim.

## 2. Scope

Phase 15C owns:

- additive multi-step counterfactual audit receipts in
  `public/js/threebody-core.mjs`;
- additive CSV columns in `paired.csv`, `trial-outcomes.csv`, and
  `aggregate-envelope.csv`;
- future npm scripts:
  - `npm run threebody:phase15c:multistep-smoke`
  - `npm run threebody:phase15c:multistep`
- outputs under:
  - `results/threebody/phase15c-multistep-counterfactual-smoke/`
  - `results/threebody/phase15c-multistep-counterfactual-lock/`
- this spec and a future `PHASE15C_RESULTS.md`.

Phase 15C does not own alternate hazard scores, warning-quality reruns,
spatial/3D extension, spacecraft-domain extension, controller redesign, or any
public-copy upgrade.

## 3. Command Shape

The following command names are reserved but not runnable until an
implementation commit adds the flag, CSV columns, and npm scripts.

Smoke:

```bash
npm run threebody:phase15c:multistep-smoke
```

The smoke must be capped under the repository's ten-minute rule. Its intended
shape is the Phase 15B smoke cell with horizons `4,8,16,32`: `massRatio=1`,
`dt=0.004`, radius `1.025`, velocity `1.1`, the six Phase 15B modes, one seed,
duration `4`. It is a column-flow and sanity check only.

Lock:

```bash
npm run threebody:phase15c:multistep
```

The lock uses the Phase 15B lock slate and all four horizons. It is expected to
exceed the inline ten-minute rule, so the implementation commit must record a
capped rate probe, wall-clock estimate, resume/readback path, and operator-run
or long-runner plan before the lock is started.

## 4. Multi-Step Counterfactual Definition

For each eligible simulation step in a trial, compute a matched set of
counterfactual rollouts from the same current state:

- `actual`: the controller-selected action for the current mode at the current
  step, followed by the mode's frozen policy for the remaining `N - 1` steps;
- `noop`: zero thrust at the current step, followed by the same frozen policy
  for the remaining `N - 1` steps;
- `oracleStrict`: the Phase 15 forward-oracle strict action at the current
  step, followed by the same frozen policy for the remaining `N - 1` steps.

For every horizon `N`, report:

- raw terminal score difference `score(actual_N) - score(noop_N)`;
- raw oracle gap `score(oracleStrict_N) - score(noop_N)`;
- normalized score
  `clamp((score(actual_N) - score(noop_N)) /
  max(abs(score(oracleStrict_N) - score(noop_N)), 1e-9), -1, 1)`;
- positive-rate indicator for the raw terminal difference;
- floor-hit indicator for the same `1e-9` normalizer floor used in Phase 15B.

The scoring function must be the same terminal-state score family used by the
Phase 15 counterfactual receipt unless the implementation amendment discovers
that the Phase 15 function is not separable from one-step-only fields; any such
change must be filed before running the smoke.

## 5. Metrics

Trial-level fields, per horizon `N`:

- `counterfactualH{N}EligibleSteps`
- `counterfactualH{N}MeanEffectVsNoop`
- `counterfactualH{N}MeanAbsEffectVsNoop`
- `counterfactualH{N}PositiveRate`
- `counterfactualH{N}MeanRawNormalizer`
- `counterfactualH{N}NormalizerFloorRate`
- `counterfactualH{N}MeanScore`
- `counterfactualH{N}FloorPositiveRate`
- `counterfactualH{N}NonFloorMeanScore`

Aggregate rows report corresponding means/totals by
`mode x candidateEnvelope x horizon`.

Primary readouts:

- **Horizon lift:** TRACK candidate rows show increasing or materially higher
  normalized score / positive rate at `N in {8,16,32}` than at `N=4`.
- **Ablation separation:** TRACK horizon-lift exceeds delayed/shuffled/sign-flip
  arms on candidate rows.
- **Non-candidate control:** non-candidate TRACK rows do not show the same
  horizon-lift pattern, or are explicitly interpreted as a broad mode effect
  rather than a candidate-envelope mechanism.
- **Signal-delay asymmetry:** `signal_delay` candidate and non-candidate splits
  are reported separately at every horizon, with no pooling before the primary
  read.

## 6. Pre-Registered Branches

**Multi-step steering supported:** candidate TRACK rows show a monotone or
material horizon lift from `N=4` to at least one of `N=16` or `N=32`, and at
least two of `signal_delay`, `signal_shuffle`, `action_shuffle`, and
`sign_flip` trail TRACK by `>= 0.10` on positive rate or normalized score in
the same candidate split. Interpretation: Phase 15's survival pocket is
consistent with cumulative trajectory steering not visible to the one-step
yardstick.

**Multi-step steering rejected:** TRACK candidate rows remain near chance or
do not improve with horizon, or shuffled/mistimed arms match or exceed TRACK at
all horizons. Interpretation: the Phase 15 survival pocket is not explained by
this multi-step counterfactual; look next at the hazard score, event-warning
quality, or a controller-design limitation.

**Mixed / partial diagnostic:** horizon lift appears only in non-candidate
rows, only in `signal_delay`, or only at one horizon without separation from
ablation arms. Interpretation: Phase 15C localizes a new diagnostic but does
not explain the mechanism.

All branches preserve the Phase 15 formal verdict.

## 7. Readback

After the smoke, record:

- command and wall-clock;
- trial count;
- presence of all new horizon columns in `paired.csv`,
  `trial-outcomes.csv`, and `aggregate-envelope.csv`;
- per-horizon TRACK candidate and non-candidate read for the smoke cell;
- whether the lock command is worth staging unchanged;
- capped per-trial or per-cell rate if the smoke is representative enough to
  estimate lock runtime.

After the lock, record:

- branch;
- candidate/non-candidate split by mode and horizon;
- the signal-delay candidate/non-candidate asymmetry;
- whether the next registered move should be hazard-score audit,
  event-warning-quality rerun, controller redesign, or Phase 16/3D deferral.
