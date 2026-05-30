# Three-Body Phase 17 - Hazard-Aligned Counterfactual Spec

Phase 15 ended **Fail-Magnitude** because the guarded TRACK survival envelope
was real and precision-stable, but the privileged counterfactual was scored with
energy and missed its magnitude bar. Phase 15B and Phase 15C then showed the
energy counterfactual is the wrong instrument: the normalizer floor dominates,
multi-step energy steering is rejected, and the signal-delay arm can improve
survival while scoring negative.

Phase 16 and 16B repaired the warning side. The frozen strict-oracle hazard
label is warnable, and the Phase-15-shaped warning verdict flips cleanly when
the score is `radius` rather than energy. Phase 17 is the corresponding
mechanism audit: **when the guarded TRACK arm acts, does its action move the
state away from the hazard boundary that the label actually uses?**

This phase is not a controller redesign. It keeps the controller, guard,
initial-condition slate, and hazard label frozen. It only changes the
counterfactual score from energy to a hazard-aligned geometric margin.

## 1. Decision Lock

Status: **operator lock 2026-05-29.** The adjustment pass closed the
post-hoc-promotion door (Phase 17 cannot promote the locked Phase 15 verdict;
§1, §6) and sharpened the pass interpretation (raw margin positivity is
near-expected because `hazardMargin <= 0` *is* the label, so the binding
evidence is the energy-contrast + ablation separation; §5, §6). No Phase 17
harness code has been written or run at lock time.

- **Frozen controller.** Reuse the Phase 13/15 guarded TRACK controller, Phase
  13 passive-derived guard thresholds, thrust limit, sensor settings, and action
  coupling. No guard retuning, thrust retuning, sensor retuning, or new
  controller policy is admitted.
- **Frozen hazard label.** Reuse the Phase 15/16 terminal hazard geometry:
  `r3 > escapeRadius (4)` OR `minPrimaryDistance < closeApproachRadius (0.08)`.
  No subtype relabeling and no alternate oracle label.
- **Same cell slate.** Reuse the Phase 15B/15C lock slate at `dt=0.004`:
  `near_escape`, mass ratios `0.01,0.3,1`, radius scales
  `1.025,1.05,1.075`, velocity scales `0.95,1.05,1.1,1.15`, thrust limit
  `0.4`, sensor noise `0`, hazard-quantile guard `0.75`, eight seeds, duration
  `16`.
- **Same mode slate.** Reuse:
  `off`, `track_sensor_accel_guarded`, `track_sensor_accel_signal_delay`,
  `track_sensor_accel_signal_shuffle`, `track_sensor_accel_action_shuffle`, and
  `track_sensor_accel_sign_flip`.
- **Same candidate split.** Candidate-envelope classification is read exactly as
  in Phase 15B/15C from the standard envelope reducer. The favorable pocket is
  `velocityScale >= 1.05`; `velocityScale = 0.95` is a boundary control.
- **Audit only.** The audit is additive and default-off behind a new explicit
  flag. Frozen Phase 13/14/15/15B/15C/16/16B paths must remain unchanged.
- **No claim upgrade; the Phase 15 verdict is immutable.** A positive Phase 17
  can update the mechanism *explanation* from "energy mechanism rejected" to
  "hazard-boundary action supported." It does **not** retune the controller,
  broaden the three-body claim outside the mapped near-escape pocket, or revise
  the locked Phase 15 Fail-Magnitude verdict. That verdict is a historical fact
  about the pre-registered *energy* metrics; Phases 16/16B/17 reframe the
  instruments but cannot reverse it. Any actual Pass must be earned by a **fresh
  pre-registered lock** with the geometric metrics (radius warning + hazardMargin
  counterfactual) fixed in advance — never by this post-hoc readback.

## 2. Scope

Phase 17 owns:

- additive hazard-aligned counterfactual receipts in
  `public/js/threebody-core.mjs`, behind a new flag such as
  `--hazard-counterfactual-audit`;
- additive CSV columns in `paired.csv`, `trial-outcomes.csv`, and
  `aggregate-envelope.csv`;
- future npm scripts:
  - `npm run threebody:phase17:hazard-cf-smoke`
  - `npm run threebody:phase17:hazard-cf`
- outputs under:
  - `results/threebody/phase17-hazard-counterfactual-smoke/`
  - `results/threebody/phase17-hazard-counterfactual-lock/`
- this spec and a future `PHASE17_RESULTS.md`.

Phase 17 does **not** own new sensors, new guard thresholds, a radius-only
controller, fitted warning models, alternate hazard labels, 3D/isotrophy work,
or public-copy upgrade.

## 3. Command Shape

Reserved but not runnable until an implementation commit adds the flag, columns,
and npm scripts.

Smoke:

```bash
npm run threebody:phase17:hazard-cf-smoke
```

Intended smoke shape: the Phase 15B/15C smoke cell (`massRatio=1`, `dt=0.004`,
radius `1.025`, velocity `1.1`, six modes, one seed, duration `4`) with the new
hazard-counterfactual flag enabled. The smoke is a column-flow and sign sanity
check only and must stay under the repository's ten-minute rule.

Lock:

```bash
npm run threebody:phase17:hazard-cf
```

The lock uses the full Phase 15B/15C slate: 288 cases x 6 modes = **1,728
trials**. If the smoke rate implies a run longer than ten minutes, stage the
exact operator command and wall-clock estimate in `PHASE17_RESULTS.md` before
the lock starts; the expected execution is the Phase 15C 12-shard pattern
(mass-ratio x velocity), though Phase 17 should run cheaper than 15C (two-arm
rollout, no per-step oracle thrust). The `sign_flip` candidate-row fallback
(>=10-trial rule in §6) is expected to be the standard path, not an exception,
since `sign_flip` produced zero candidate envelopes in 15C.

Because the implementation touches shared core/harness code, the implementation
commit must run cheap syntax checks immediately and stage the long hard-void
regression gates before lock interpretation:

```bash
node --check public/js/threebody-core.mjs
node --check scripts/threebody-operating-envelope.mjs
npm run threebody:phase13
npm run threebody:phase14
```

The Phase 13/14 commands exceed the inline-agent rule; they are operator-staged
unless the operator explicitly authorizes the wall time. Either gate deviation
voids Phase 17 interpretation.

## 4. Hazard-Aligned Counterfactual Definition

At each eligible controlled step (`!stepWarmup` and `|thrust| > 1e-6`), compute a
matched pair of rollouts from the same current state:

- `actual`: apply the controller-selected thrust at step 1, then continue with
  that same thrust for the remaining horizon steps;
- `noop`: apply `[0,0]` at step 1, then continue with the same controller-selected
  thrust used by `actual` for the remaining horizon steps.

This mirrors the Phase 15C first-action intervention pattern while dropping the
energy oracle normalizer. The comparison isolates the causal effect of the
realized first action under a matched continuation.

Frozen horizons: `N in {1,4,8,16,32}` simulation steps. At `dt=0.004`, these are
`0.004`, `0.016`, `0.032`, `0.064`, and `0.128` seconds.

For every rollout state, compute:

```text
radius = sqrt(x3^2 + y3^2)
minPrimaryDistance = min(distance(testParticle, primary_0), distance(testParticle, primary_1))
escapeMargin = escapeRadius - radius
closeMargin = minPrimaryDistance - closeApproachRadius
hazardMargin = min(escapeMargin, closeMargin)
```

Interpretation: `hazardMargin > 0` is inside the non-terminal region, and larger
is safer. `hazardMargin <= 0` is on or beyond one of the frozen hazard
boundaries.

For each horizon `N`, report:

- `hazardMarginEffectH{N} = hazardMargin(actual_N) - hazardMargin(noop_N)`
  (positive means the action moved the state farther from the nearest hazard
  boundary);
- `hazardMarginPositiveH{N}` = `hazardMarginEffectH{N} > 0`;
- `escapeMarginEffectH{N} = escapeMargin(actual_N) - escapeMargin(noop_N)`
  (positive means the action reduced escape risk);
- `closeMarginEffectH{N} = closeMargin(actual_N) - closeMargin(noop_N)`
  (positive means the action reduced close-approach risk);
- `hazardAvoidedH{N}` = `1` if the `noop` path reaches terminal hazard by or
  before `N` and the `actual` path does not; `-1` if actual reaches hazard and
  noop does not; `0` otherwise.

Primary score is **raw hazard-margin effect**, not normalized by an oracle gap.
The Phase 15/15C normalizer-floor failure is therefore not imported into Phase
17. Escape and close margins are diagnostics for the OR-label subtype mix; they
cannot replace the primary hazard-margin branch after lock.

## 5. Metrics

Trial-level fields, per horizon `N`:

- `hazardCfH{N}EligibleSteps`
- `hazardCfH{N}MeanMarginEffect`
- `hazardCfH{N}MeanAbsMarginEffect`
- `hazardCfH{N}PositiveRate`
- `hazardCfH{N}MeanEscapeMarginEffect`
- `hazardCfH{N}EscapePositiveRate`
- `hazardCfH{N}MeanCloseMarginEffect`
- `hazardCfH{N}ClosePositiveRate`
- `hazardCfH{N}MeanHazardAvoided`
- `hazardCfH{N}HazardAvoidedRate`
- `hazardCfH{N}HazardCausedRate`

Aggregate rows report corresponding means/totals by
`mode x candidateEnvelope x horizon`, plus favorable-pocket-only rows
(`velocityScale >= 1.05`).

Primary readouts:

- **Guarded candidate direction:** guarded TRACK candidate rows have positive
  mean `hazardMarginEffect` and positive rate above chance at the hazard-relevant
  horizons.
- **Ablation separation:** guarded TRACK candidate rows beat the mistimed or
  inverted arms under the same candidate split.
- **Delay asymmetry:** `signal_delay` candidate and non-candidate splits remain
  separate. If delay improves survival while its hazard-margin effect is weak or
  negative, record that as mechanism narrowing, not as a guarded-TRACK pass.
- **Subtype honesty:** if the effect is carried almost entirely by
  `escapeMarginEffect`, say so. That is expected after Phase 16's escape-dominant
  label mix and is not a defect, but it constrains the claim to near-escape.

## 6. Pre-Registered Branches

The primary horizons are `N in {8,16,32}`. `N=1` and `N=4` are onset diagnostics.

**Hazard-directed mechanism supported.** On guarded TRACK candidate rows in the
favorable pocket:

- at least two of the three primary horizons have
  `hazardCfH{N}MeanMarginEffect > 0`;
- at least two of the three primary horizons have
  `hazardCfH{N}PositiveRate >= 0.60`;
- guarded TRACK exceeds `signal_delay` candidate rows by at least `0.10` positive
  rate at two of the three primary horizons;
- `sign_flip` is non-positive on mean margin effect or at least `0.10` below
  guarded TRACK positive rate at two of the three primary horizons. If
  `sign_flip` has fewer than 10 candidate-row trials, use all favorable-pocket
  `sign_flip` controlled rows as the inverted-action negative-control comparison
  and report the candidate-row absence explicitly.

Interpretation: the Phase 13/15 survival pocket is not energy-steering, but it
is consistent with first actions that move the state away from the frozen hazard
boundary. Note the bar: because `hazardMargin <= 0` *is* the hazard label, raw
margin positivity is near-expected for a guard controller and is **not** itself
the finding — a pass does not re-discover survival (already established in
Phase 13/15). The binding evidence is the **contrast with Phase 15C's
energy-null** (the same matched rollout shows a per-step signal the energy score
could not see) and the **ablation separation**; a pass localizes the survival
mechanism to per-step hazard-directed action.

**Hazard-directed mechanism rejected.** Guarded TRACK candidate rows are
decidable but fail the direction and positive-rate bars, or `signal_delay` /
shuffle arms match or exceed guarded TRACK at the primary horizons.
Interpretation: the survival edge remains real and warning is repaired, but this
hazard-margin counterfactual still does not explain the action mechanism.

**Mixed / partial diagnostic.** Any of:

- guarded TRACK is positive but misses the `0.60` positive-rate bar;
- the signal appears only in `escapeMarginEffect` while `hazardMarginEffect` is
  suppressed by close-boundary cells;
- candidate rows are too sparse for `signal_delay`, or sparse shuffle/sign-flip
  rows make the ablation comparison depend entirely on the favorable-pocket
  fallback;
- the result appears only in non-candidate rows or only in `signal_delay`.

Interpretation: Phase 17 localizes the action signal but does not fully upgrade
the mechanism claim. The next move should be a radius-only matched-duty
controller control, not another energy counterfactual.

All branches preserve the Phase 15 formal verdict — full stop. Phase 17 cannot
promote Phase 15; any verdict revision must be a fresh pre-registered lock with
the geometric metrics fixed in advance (see §1).

## 7. Readback

After the smoke, record:

- command and wall-clock;
- trial count and outcome mix;
- presence of all new hazard-counterfactual columns in `paired.csv`,
  `trial-outcomes.csv`, and `aggregate-envelope.csv`;
- one sign-sanity table for the smoke cell across modes and horizons;
- whether the lock command is worth staging unchanged;
- capped per-trial rate and extrapolated lock wall time.

After the lock, record:

- branch;
- candidate/non-candidate split by mode and horizon;
- favorable-pocket guarded TRACK candidate read at `N=8,16,32`;
- guarded-vs-delay and guarded-vs-sign-flip separation;
- escape-vs-close subtype diagnostic;
- whether the next registered move is the radius-only / matched-duty guard
  ablation, a controller redesign, or a claim-language update.
