# Sundog Pressure Mines Workbench

Working hook:

> The mine is hidden. The field still bends around it.

Sundog Pressure Mines is the proposed first game-native workbench for the
Sundog application family. It should not be framed as "better puzzle AI" and
not as a perfect-information solver wearing a noisier skin. It is a bounded,
browser-native partial-observation experiment asking whether useful decisions
can be made when the decisive state is hidden but nearby structure leaks
through a distorted field.

The public question is small enough to defend:

> Can an agent make useful safe-move decisions when mine locations are hidden
> but nearby tiles emit only a noisy, lossy pressure field instead of exact
> adjacency counts?

The workbench should live beside `balance.html` and `threebody.html` as a new
public tab, eventually backed by a matching writeup, seeded harness, and
failure-boundary panel. The goal is not to claim that Sundog solves
Minesweeper. The goal is to show a legible, familiar environment in which
hidden danger leaves enough indirect structure for bounded action inside a
mapped observability envelope.

## Sundog Expression *(canonical for the APPLICATIONS.md row)*

- **Hidden target:** local mine occupancy and downstream board risk.
- **Indirect signal:** noisy pressure values, local field gradients, and bounded
  scan returns rather than exact adjacency counts.
- **Transformation:** SCAN/SEEK/TRACK hazard-reading with confidence gating,
  bounded action ledger (which tiles were scanned vs revealed, no privileged
  adjacency counts retained between turns), and conservative fallback when
  observability drops.
- **Actionable output:** bounded reveal / flag / scan / abstain decisions that
  improve survival and board progress over naive local heuristics inside a
  mapped observability envelope, where the survival/progress lead must remain
  net-positive after deducting scan-budget consumed, and which fail cleanly
  outside the envelope.

This block is here so the eventual gallery card, `APPLICATIONS.md` row, and
public writeup do not drift from the same hidden / indirect / transformation /
output shape already used for Balance and the other workbenches.

## Why This Workbench

Balance asks whether the body can be controlled through its cast shadow.
Three-body asks whether useful action can be taken from compressed signatures
around instability rather than full state reconstruction. Pressure Mines asks
the same family question in a game that nearly everyone already understands:

1. the danger is hidden;
2. nearby tiles carry traces;
3. those traces are partial, noisy, and sometimes misleading;
4. the player must still act.

This makes Pressure Mines a strong first game-native Sundog surface because the
core theoremic move is legible without any physics background. The viewer does
not need to accept orbital mechanics, optics, or robotics conventions. The
board itself says what the experiment is about: the decisive thing is not
visible, but the environment is not silent.

It is also a cleaner public metaphor than casino-adjacent card games. A mines
field communicates hidden risk, imperfect local evidence, and bounded caution
without inviting "advantage play" or gambling discourse. It is closer to the
repo's current scientific posture: not domination through compute, but useful
action from incomplete traces.

## Claim Boundary

The claim is not that hidden hazards can always be inferred indirectly. The
claim is narrower:

> In some minefield regimes, a lossy pressure field preserves enough
> control-relevant structure for bounded reveal/flag decisions to outperform
> naive local heuristics, while explicit noise, delay, and ambiguity boundaries
> mark where that structure stops being usable.

That is the whole point of the workbench. The field is not mystical. It is a
projection of hidden structure. In some board geometries and sensor regimes,
that projection is informative enough to support action. In others, it should
collapse and the controller should lose cleanly.

The negative region ships with the positive region. The workbench does not
earn its claim by publishing a favorable pocket without publishing matched-seed
cells where the same controller harms outcomes. This is a hard binding on the
public artifact, not a stylistic preference.

Avoid broader formulations such as:

- "Sundog solves Minesweeper."
- "The pressure field reveals the hidden board in general."
- "Indirect inference beats logical deduction."
- "This proves uncommon intuition."
- "This is better game AI."
- "The controller reads the field for free." (Scans cost budget. Reveals risk
  termination. Every input the controller harvests is paid for in the same
  ledger the comparison statistic deducts from.)
- "Memory alone explains the lead." (No privileged adjacency counts are
  retained between turns. The action ledger is bounded and public; if a
  baseline can read the same ledger and the same field, parity should hold.)

## Actionability Audit

The workbench must separate hidden state, indirect signals, and privileged
diagnostics from the start.

| Signal | Tier | Use |
| --- | --- | --- |
| true mine occupancy grid | Privileged | board generation, audit, oracle baseline |
| exact adjacent mine count | Privileged or disabled | optional ablation only; not available in primary Sundog mode |
| pressure value at tile | Sensor-available | Sundog controller input |
| local pressure gradient | Sensor-available, derived | controller input |
| pressure confidence / variance | Sensor-available | confidence gating |
| scan result | Bounded sensor action | optional active probe; debit from scan budget on each use |
| revealed safe/unsafe tile outcomes | Public action ledger | available to all non-passive controllers identically; not a Sundog-only input |
| flag history | Public action ledger | available to all non-passive controllers identically |
| scan-budget remaining | Public state | required input to controllers that spend scans; also a comparison-statistic deduction |
| board seed / mine density / noise params | Calibration parameter | harness manifest, not hidden target |

The central typing should be explicit:

```text
G_t = latent local hazard / expected risk residual
S_t = [pressure, gradient, confidence, bounded scan]
u_t = reveal / flag / scan / abstain
H_t = local coupling between recent actions and changes in exposed field structure
```

The "pressure field" must be defined carefully enough that it is not merely a
reskinned exact count. If it is a monotone transform of normal Minesweeper
numbers with no meaningful ambiguity, the workbench is not earning its claim.
The field should preserve local structure while introducing controlled
uncertainty through blur, overlap, attenuation, delay, dropout, terrain, or
noise.

## Ratified Hook Language

Safe hook:

> Sundog Pressure Mines asks whether an agent can make safer progress when the
> mines themselves stay hidden and nearby tiles offer only a fuzzy pressure
> signal instead of exact counts.

Short version:

> The mine is hidden. The field still bends around it.

Alternative short version:

> The danger is buried. The board still deforms around it.

Avoid:

- "Sundog reads the minefield perfectly."
- "This is a superhuman Minesweeper player."
- "The pressure field is all you need."
- "The controller infers the board."
- "A visual demo is evidence."

## Visual Direction

The page should feel like a laboratory field board, not a toy puzzle clone.
The first screen should open directly into the board and telemetry.

Target composition:

- central board: concealed tiles, revealed pressure texture, flags, scan pulses,
  and recent action highlights;
- right rail: controller mode, pressure-field model, noise/delay settings, scan
  budget, and board preset;
- lower strip: survival probability estimate, revealed-safe count, false-flag
  count, confidence trace, and local field panel;
- optional side-by-side mode: Sundog, naive local heuristic, and privileged
  oracle on the same seed.

The visual language should make the distinction between direct knowledge and
field-reading unmistakable. The board is hidden. The field is visible. The
field must look like a projection, not like a disguised number label.

## Roadmap

### Phase 0 - Claim Boundary And Benchmark Choice

Goal: define the exact puzzle/task before writing the page.

Deliverables:

- Fix the first board family: square grid with seeded random mine placement,
  finite dimensions, and declared mine densities.
- Declare the primary hidden residual: local hazard / expected safe progress.
- Declare what the Sundog controller cannot read: exact mine occupancy and exact
  adjacency counts.
- Define the evidence tier: planned workbench, not yet a research result.
- Define page shape: `mines.html`, linked from the gallery and nav as a
  Planned Workbench until implementation earns a higher tier.

Exit criterion: the workbench has one sentence that can be attacked and one
task definition that can be reproduced.

### Phase 1 - Board Core

Goal: implement a deterministic board generator and action loop independent of
rendering.

Deliverables:

- Seeded board generation with declared width, height, and mine count.
- State transitions for `reveal`, `flag`, `scan`, and optional `abstain`.
- Terminal events: mine triggered, full clear, scan budget exhausted, or
  timeout / turn cap.
- Presets: easy sparse field, clustered field, ambiguous overlap field,
  noisy-sensor field, delayed-sensor field.
- Shared board module usable by browser and headless harness.

Exit criterion: the same board logic drives the browser page and batch runs.

### Phase 2 - Pressure Sensor Model

Goal: turn hidden mine geometry into an indirect local field.

Deliverables:

- A pressure-field definition pinned in the doc before implementation. For
  example: each mine contributes a decaying kernel over nearby tiles, summed
  into a scalar pressure surface, then perturbed by configurable noise,
  quantization, dropout, and delay.
- Candidate sensor outputs (consumed by controllers, not kernel families):
  - tile pressure;
  - local finite-difference gradient;
  - confidence / observability score;
  - bounded scan pulse result if active probing is enabled.
- Privileged audit comparing pressure-derived risk estimates against true mine
  occupancy without exposing the occupancy to the controller.
- Irreducibility falsifier: with noise=0, dropout=0, delay=0, and a narrow
  kernel, the pressure field becomes invertible to exact adjacency counts.
  Pre-commit the minimum kernel-width, noise floor, and dropout rate below
  which the workbench refuses to claim its result — that point is the trivial
  degenerate case the workbench must not promote evidence from.

### Pinned Pressure Field Definition

This block is the doc-canonical Phase 2 specification. The implementation in
`public/js/mines-sensor.mjs` must match this; if the implementation diverges,
this section is updated *and* the prior text is preserved as an addendum so
the pre-registration record remains auditable.

**Primary kernel: isotropic Gaussian.** For tile `t` at integer coordinates
`(x_t, y_t)` and a mine `m` at `(x_m, y_m)`, the per-mine contribution is

```text
K(t, m) = exp(-d(t, m)^2 / (2 * sigma^2))
```

where `d` is Euclidean distance in tile units. The true scalar pressure
surface is

```text
p_true(t) = sum over mines m of K(t, m)
```

Pressure is computed over the full board. The controller does not get to see
`p_true`; it sees `p_obs`, defined below.

**Observation pipeline (applied in order):**

1. Additive Gaussian noise: `p_obs(t) = p_true(t) + epsilon(t)`,
   `epsilon ~ N(0, sigma_noise^2)`.
2. Per-tile Bernoulli dropout at rate `delta`: dropped tiles emit no value
   and read confidence `0`. Non-dropped tiles read confidence `1` (smoothable
   into a local trust score downstream).
3. Optional quantization to `N` levels (default disabled).
4. Optional delay of `k` turns (default `0`; the observed field at turn `T` is
   the noisy field computed on the state at turn `T - k`).

**Derived sensor outputs:**

- *tile pressure:* `p_obs(t)`.
- *local finite-difference gradient:*
  `grad_p(t) = ((p_obs(t + e_x) - p_obs(t - e_x)) / 2,
                 (p_obs(t + e_y) - p_obs(t - e_y)) / 2)`,
  with edge tiles using one-sided differences.
- *confidence / observability:* per-tile `c(t)` in `{0, 1}` after dropout,
  optionally rolled into a smoothed local trust score.
- *bounded scan pulse:* a single targeted probe at tile `t` returning a
  reading drawn from `N(p_true(t), sigma_scan^2)` with
  `sigma_scan << sigma_noise`. Each scan debits the scan budget. The scan
  budget is a comparison-statistic deduction (see Phase 4 budget-tax rule).

**Default operating point pinned to this doc:**

```text
sigma          = 1.0
sigma_noise    = 0.10
delta          = 0.10
quantization   = disabled
delay          = 0
sigma_scan     = 0.02
```

**Irreducibility floor — below which Pressure Mines does not publish:**

```text
sigma_min       = 0.7
sigma_noise_min = 0.05   (on the unit-normalized pressure scale)
delta_min       = 0.05
```

Rationale: at `sigma >= 0.7` the kernel necessarily spreads across the
8-neighborhood of every mine (a distance-1 tile gets contribution
`exp(-1 / (2 * 0.49)) ≈ 0.36`, a distance-sqrt(2) tile gets ≈ 0.13, a
distance-2 tile gets ≈ 0.018), so one observed tile pressure is a weighted
sum over a non-trivial spatial support and cannot be inverted to a single
integer adjacency count. At `sigma_noise >= 0.05` the noise envelope is
wide enough that distinct multi-mine geometries with the same scalar pressure
sit inside overlapping observation distributions. At `delta >= 0.05` at
least one tile in twenty is unreadable each turn, so reconstructing a fully-
consistent local count requires inference, not lookup.

Below *any* of these three thresholds, the workbench treats the result as a
degenerate exact-count restatement, not a Sundog claim. Sub-floor cells may
still appear in the harness for sanity checks, but they cannot be cited as
evidence in the public artifact.

Exit criterion: the board can show when the field is informative and when it
collapses into ambiguity, and the kernel/noise floor below which Sundog Mines
does not earn its claim is named in the doc before any controller runs.

### Phase 3 - Diagnostic Benchmark

Goal: test whether the indirect field forecasts useful risk at all.

Questions:

- Does tile pressure predict mine adjacency or mine occupancy better than a
  naive prior?
- Does local pressure gradient predict safer frontier expansion?
- How much lead time exists before a forced-risk region?
- Which mine densities and field kernels produce ambiguity cliffs?

Metrics:

- AUROC / precision-recall for hazardous-tile detection;
- correlation between pressure and true local risk;
- frontier-safe-move precision;
- degradation under noise, delay, dropout, and clustering.

Pre-commit before Phase 3 runs: the primary diagnostic statistic is hazardous-
tile AUROC against the privileged adjacency-or-occupancy ground truth, and the
informativeness threshold is AUROC ≥ 0.65 on at least one named envelope cell
(mine density, kernel width, noise level, delay) that is not the irreducibility
degenerate case from Phase 2. The named envelope cell is fixed in the doc
before logs land.

Named Phase 3 envelope cell before first logs land:

```text
preset        = easy_sparse
sensor_cell   = doc_default
sigma         = 1.0
sigma_noise   = 0.10
dropout       = 0.10
delay         = 0
threshold     = mean mine-occupancy AUROC >= 0.65
script        = npm run mines:phase3
```

Exit criterion: at least one field signature remains informative under the
pre-committed AUROC threshold inside a named operating envelope, and the
envelope cells where AUROC collapses to chance are recorded alongside the
positive cell.

### Phase 4 - Baseline Set

Goal: establish fair comparison lanes.

Required modes:

- **Passive / random reveal:** random legal reveal subject to minimal validity.
- **Naive local heuristic:** reveal lowest-pressure frontier tile without memory
  or confidence modeling.
- **Naive threshold flagger:** flag tiles above a fixed pressure threshold.
- **Sundog controller:** uses pressure residuals, gradients, reveal history,
  action history, and confidence gating.
- **Privileged oracle:** reads true occupancy or exact counts for an upper-bound
  baseline.
- **Ablations:** shuffled pressure, delayed pressure, no gradient, no scan, no
  action history, no confidence gate.

Budget-tax discipline: every lane that spends scan-budget reports its raw
safe-tile / survival metrics and the budget-adjusted version that deducts the
scan cost. If Sundog's lead over the naive local heuristic disappears once
scan-budget is netted out, the workbench reports the lead as un-earned for
that envelope cell. This is the falsifier that prevents "I scanned more, so I
revealed more" from disguising bad field inference.

Exit criterion: every public lane has a stated information budget, and every
public comparison reports both raw and budget-adjusted metrics on matched
seeds.

Phase 4 scaffold status after the Phase 5 merge:

- `public/js/mines-controllers.mjs` declares implemented baseline/oracle lanes
  plus the Phase 5 Sundog and ablation information budgets.
- `scripts/mines-phase4-baselines.mjs` / `npm run mines:phase4` writes local,
  ignored matched-seed outputs under `results/mines/phase4-baselines`.
- Implemented baseline/oracle lanes: `random_reveal`, `naive_pressure`,
  `threshold_flagger`, `naive_pressure_shuffled`, `naive_pressure_delayed`,
  and `oracle_safe`.
- Implemented Phase 5 lanes: `sundog_controller`, `sundog_no_gradient`,
  `sundog_no_scan`, `sundog_no_action_history`, and
  `sundog_no_confidence_gate`, plus the lean recompositions `sundog_lean`
  and `sundog_minimal`.
- Budget-adjusted safe tiles are currently `rawSafeTiles - scanCount`; this
  is equal to raw safe tiles for no-scan lanes and applies a direct scan tax
  to Sundog lanes that spend bounded probes.

### Phase 5 - Sundog Controller Prototype

Goal: build the first controller that acts from field traces rather than hidden
occupancy.

Candidate structure:

- `SCAN`: spend a bounded scan action or low-commitment reveal on tiles that
  maximize expected information gain under the field model.
- `SEEK`: move toward frontier regions where pressure and gradient imply a safe
  expansion corridor.
- `TRACK`: continue harvesting low-risk territory while confidence remains high.
- `REACQUIRE`: when the field becomes ambiguous, step back into probing,
  conservative flagging, or abstention.

This does not need to be learned initially. A hand-authored controller is
acceptable because the first claim is about observability and bounded action,
not about training a general puzzle agent.

Exit criterion: the controller survives longer, clears more safe tiles, or
improves expected return versus passive and naive local baselines on a small
seeded slate inside the diagnostic-positive envelope.

Phase 5 prototype status:

- `public/js/mines-controllers.mjs` now contains the first hand-authored
  Sundog controller. It combines observed pressure, confidence/dropout gating,
  gradient magnitude, revealed-neighbor history, scan history, and conservative
  pressure-threshold flagging without reading true occupancy or exact counts.
- `npm run mines:phase5` runs the controller and ablations on the initial
  matched 16-seed slate, writing ignored local artifacts under
  `results/mines/phase5-controller`, including `phase5-controller.md`,
  `summary-rows.csv`, and `matched-comparison-summary.csv`.
- `npm run mines:phase5:reroll` repeats the same lane set on a fresh 64-seed
  slate beginning at seed 1000, writing ignored local artifacts under
  `results/mines/phase5-controller-reroll64`.
- Initial 16-seed result: the named `sundog_controller` improves
  budget-adjusted safe tiles versus `naive_pressure` in publishable
  `doc_default` clustered boards (+3.000) and dense boards (+1.875), and in
  noisy/dropout dense (+3.3125) and easy-sparse (+2.1875). It loses
  noisy/dropout clustered (-6.6875) and `doc_default` easy-sparse (-0.8125).
- Fresh 64-seed reroll: the named controller's 16-seed gains do not hold. It
  loses to `naive_pressure` on all three publishable `doc_default` boards:
  clustered (-2.0625), dense (-6.03125), and easy-sparse (-0.890625). It is
  effectively flat only on noisy/dropout dense (+0.015625 after scan tax), and
  loses noisy/dropout clustered (-5.890625) and easy-sparse (-4.09375).
- Scan-tax check: the reroll weakness is not caused by scan spending. The
  `doc_default` reroll rows use zero mean scans, so the controller is losing
  from policy choices: conservative false flags plus pressure/gradient/history
  tie-breaking that perturbs the naive pressure ordering.
- Interpretation: Phase 5 is a useful falsifiable workbench lane, but the
  current hand-authored controller is not robust enough to support an
  operating-envelope claim. The ablations remain useful diagnostics: the
  no-gradient and no-scan variants expose when added channels are helping
  versus overfitting, while the threshold flagger shows survival can be bought
  with many false flags and must not be mistaken for clean field inference.

Phase 5 lean reroll gate, pre-registered before running the new variant:

- `sundog_lean` is a recomposition of existing channels: naive pressure
  ordering, confidence gating, conservative threshold flagging, and
  flag-neighbor risk penalty. It disables gradient, scan policy, scan bonuses,
  and corridor/action-history bonuses. `sundog_minimal` is the same composition
  without the flag action policy, included only to isolate whether threshold
  flagging helps or hurts.
- Primary acceptance: on the fresh 64-seed reroll, `sundog_lean` must have
  budget-adjusted delta >= 0 versus `naive_pressure` on at least four of the
  six publishable cells (`doc_default` and `blur_noise_cliff` crossed with
  clustered, dense, and easy-sparse), with no cell worse than -1.0 tiles.
- Secondary check: `sundog_lean` must beat the full `sundog_controller` by at
  least +3 budget-adjusted tiles on `blur_noise_cliff` clustered and at least
  +5 on `blur_noise_cliff` easy-sparse. If it misses this check, the ablation
  stack is not reproducing the failure diagnosis.
- Failure interpretation: if `sundog_lean` misses the primary gate, Phase 5's
  lesson remains structural: hand-authored channel composition does not yet
  survive reroll on this preset family. Move to Phase 6 with the public page
  exposing side-by-side `sundog_lean` versus `naive_pressure`, so the loss is
  visible instead of hidden.

Phase 5 lean reroll outcome:

- Structural smoke test: on a no-noise `easy_sparse` board with seed 42,
  `sundog_lean` matched `naive_pressure` reveal choices until the conservative
  threshold flagger fired on turn 2. This confirms the lean composition is not
  leaking gradient, scan, or action-history bonuses before its flag policy
  engages.
- Primary gate failed. On the fresh 64-seed reroll, `sundog_lean` cleared
  budget-adjusted delta >= 0 against `naive_pressure` in only one of six
  publishable cells: `blur_noise_cliff` easy-sparse (+2.78125). It missed
  `doc_default` clustered (-2.390625), `doc_default` dense (-0.171875),
  `doc_default` easy-sparse (-1.9375), `blur_noise_cliff` clustered
  (-4.5625), and `blur_noise_cliff` dense (-0.25).
- Secondary check was mixed. `sundog_lean` beat full `sundog_controller` by
  +6.875 budget-adjusted tiles on `blur_noise_cliff` easy-sparse, clearing the
  +5 check, but beat it by only +1.328125 on `blur_noise_cliff` clustered,
  missing the +3 check.
- `sundog_minimal` was useful but did not rescue the claim. It beat
  `sundog_lean` on noisy easy-sparse (+3.34375 versus +2.78125) and avoided
  lean's false flags, but still failed the primary gate with only one positive
  cell and multiple cells below -1.0.
- Resulting claim: the lean surgery validated one diagnosis, that scan,
  gradient, and action-history bonuses hurt the noisy easy-sparse pocket, but
  it did not produce a robust controller. Phase 5 should remain framed as a
  falsifiable workbench and failure map until a later controller redesign
  survives the reroll gate.

### Phase 6 - Real-Time Web Projection

Goal: create the public browser workbench shell early.

Deliverables:

- `mines.html` with responsive layout matching the workbench family.
- Browser renderer showing:
  - concealed board;
  - revealed pressure field;
  - scan pulses;
  - flags;
  - privileged overlay only when diagnostics are enabled.
- Telemetry panels for confidence, frontier size, safe reveals, false flags,
  and survival status.
- Mode controls for passive, naive, Sundog, oracle, and ablations.
- Seed controls and board presets.
- Side-by-side comparison mode on matched seeds.

Exit criterion: the page demonstrates the qualitative phenomenon before large
batch studies are complete. The page is allowed to show a qualitative pocket
*provisionally*: any framing that asserts a confirmed envelope, named
operating cell, or improvement-over-baseline must be replaced by Phase 7+
matched-seed evidence before promotion past Planned Workbench. Until then the
page copy uses provisional language and the evidence-tier badge stays at the
lowest applicable tier.

*Status addendum (post Phase 10 / Phase 11):* this rule has now fired. The
Phase 10 CONFIRM produced a single matched-seed candidate cell with a
bootstrap CI excluding zero and a paired published failure cell, satisfying
the pre-registration. The Phase 11 page polish promoted the tier badge from
"Planned Workbench" to "Operating-Envelope Study" and replaced the "Open
Question" claim box with the "Confirmed pocket / Failure region" framing.
The provisional-copy clause above is preserved as the historical
pre-registration record and is no longer load-bearing.

Phase 6 browser status:

- `mines.html` now runs the shared Phase 1-5 modules directly in the browser:
  `public/js/mines-core.mjs`, `public/js/mines-sensor.mjs`, and
  `public/js/mines-controllers.mjs`. The page no longer has a placeholder
  renderer or a separate toy simulation.
- `public/js/mines-browser.mjs` creates matched-seed lanes, opens the center
  tile, asks the selected controller for actions, applies reveal/flag/scan
  transitions, and records public scan readings in the same shape used by the
  headless harness.
- The first viewport includes live canvas rendering, seed/preset/sensor
  controls, mode selection, pause/step/reset, matched side-by-side comparison,
  and opt-in privileged diagnostics. The default view compares `sundog_lean`
  against `naive_pressure` on the noisy easy-sparse pocket; the controls can
  switch immediately to the doc-default and dense failure cells.
- Telemetry reports safe reveals, flags, false flags, scan spend, frontier
  size, and mean frontier confidence. False-flag and hidden-mine diagnostics
  are audit surfaces; they are not controller-visible inputs.

### Phase 7 - Reproducible Harness

Goal: move from visual toy to repeatable evidence.

Deliverables:

- Headless JS runner sharing the same board core and controller modules as the
  browser page.
- Seeded trial manifests recording controller mode, board dimensions, mine
  density, field kernel, noise, delay, dropout, scan budget, and initial seed.
- Per-trial JSONL or CSV logs for exposed field values, controller actions,
  confidence, frontier statistics, and terminal outcome.
- Browser replay URLs and replay verification support.
- NPM scripts mirroring the Balance / Three-body pattern, for example:

```bash
npm run mines:phase7
npm run mines:phase8
npm run mines:phase9
```

Exit criterion: a browser seed can be replayed exactly in the harness.

Phase 7 harness status:

- `scripts/mines-phase4-baselines.mjs` now exports `parseArgs` and `runSuite`
  so replay verification can reuse the exact harness path instead of spawning
  a parallel implementation.
- `trial-rows.csv` now includes replay-oriented manifest columns:
  `trialId`, `browserReplayUrl`, `harnessReplayCommand`, `actionTraceHash`,
  `scanBudget`, and `kernelFamily`, alongside the existing board, sensor, and
  terminal metrics.
- `--trace-jsonl` writes `step-traces.jsonl` with one public per-action row
  for Phase 8 event metrics. Rows include attempted/applied action, fallback
  status, frontier size, mean frontier confidence, action-tile pressure,
  action-tile confidence, scan readings, safe-tile counts, false flags, scan
  count, and terminal state.
- Every batch writes `replay-index.json`, a compact fixture list of browser
  replay URLs plus stable action-trace hashes.
- `scripts/mines-verify-replays.mjs` reads the replay index, reruns each
  replay URL through the harness, and fails if the terminal, turn count, or
  action-trace hash changes.
- `npm run mines:phase7` runs the canonical smoke batch and verification:
  `npm run mines:phase7:batch && npm run mines:phase7:verify`. The current
  local run produced 336 trials, 5,765 step-trace rows, and 336/336 replay
  verifications passing.

### Phase 8 - Recovery And Event Metrics

Goal: make the claim about useful decision-making, not just flashy clearing.

Metrics:

- survival rate;
- safe tiles revealed;
- false flag count;
- frontier collapse count;
- scan budget efficiency;
- hazardous reveal rate;
- confidence-loss count;
- lead time before entering forced-risk regions.

Deliverables:

- Event labels: safe reveal, risky reveal, mine trigger, false flag, scan
  success, frontier collapse, recovery.
- Threshold sweeps for risk warnings from pressure and gradient signals.
- Comparison tables against passive and naive baselines on matched seeds.

Exit criterion: the page and docs distinguish useful field-reading from lucky
tile clicking.

Phase 8 event-metrics status:

- `scripts/mines-phase8-metrics.mjs` consumes a replayable Phase 7 output
  directory, defaulting to `results/mines/phase7-smoke`, and writes ignored
  local Phase 8 artifacts under `results/mines/phase8-events`.
- `npm run mines:phase8` runs the Phase 7 smoke batch plus replay verification,
  then analyzes `step-traces.jsonl` into event metrics.
- Event labels currently emitted: `safe_reveal`, `risky_reveal`,
  `mine_trigger`, `false_flag`, `scan_success`, `frontier_collapse`,
  `confidence_loss`, and `recovery`.
- The `risky_reveal` label is a public-sensor warning label, not privileged
  mine knowledge. By default it fires for reveal actions under pressure
  warning >= 1.2, action confidence < 0.5, frontier confidence < 0.5, or
  gradient warning >= 0.5. The `mine_trigger` label is the hazardous reveal
  audit event.
- Outputs include `event-rows.csv`, `trial-event-summary.csv`,
  `mode-event-summary.csv`, `matched-event-comparisons.csv`,
  `matched-event-comparison-summary.csv`, `warning-threshold-sweeps.csv`,
  `summary.json`, and `phase8-events.md`.
- The current local Phase 8 run reads 336 Phase 7 trials and 5,765 replayable
  trace rows, emits 7,306 event labels, and preserves `sourcePhase` so the
  Phase 8 tables remain tied to the Phase 7 trace batch that produced them.
- The first Phase 8 summary already separates the shape that Phase 5 blurred:
  `sundog_lean` gains safe tiles over `naive_pressure` in the noisy
  easy-sparse pocket while adding risky-reveal pressure, and it still shows
  doc-default false-flag costs. This is instrumentation, not a promoted
  envelope claim.

### Phase 9 - Sensor Degradation And Observability Boundary

Goal: turn the failure boundary into first-class content.

Sweeps:

- mine density;
- clustering strength;
- pressure noise;
- sensor delay;
- scan dropout;
- kernel sharpness / blur;
- board size;
- scan budget.

Expected boundary:

- sparse fields may leave readable local gradients;
- clustered or overlapping fields may create deceptive plateaus;
- delay should harm frontier decisions more than static flagging;
- high blur or high dropout should collapse local distinction between safe and
  unsafe frontier tiles.

Exit criterion: the workbench has a visible "where Sundog should not be used"
panel.

Phase 9 first-pass status:

- `npm run mines:phase9` now runs `scripts/mines-phase9-boundary.mjs`, a
  first-pass degradation sweep over preset, mine density, clustering, pressure
  noise, sensor delay, scan dropout, kernel blur, board size, and scan budget.
- The run writes ignored local artifacts under
  `results/mines/phase9-boundary/`: `manifest.json`,
  `boundary-panel.json`, `cell-manifest.csv`, `trial-outcomes.csv`,
  `boundary-summary.csv`, `matched-comparison.csv`,
  `matched-comparison-summary.csv`, `unsafe-cells.csv`, `summary.json`, and
  `phase9-boundary.md`.
- The browser workbench now has a live **Use Boundary** panel driven by
  `public/js/mines-boundary.mjs`, so the page and the headless sweep share
  labels for `field_uninformative`, `frontier_ambiguity`, `delay_misread`,
  `overflagged`, `probe_budget_exhausted`, and
  `controller_overcommitted`.
- Current local smoke run: 34 boundary cells x 7 modes x 8 seeds = 1,904
  trials. It marks 77 of 102 Sundog-variant groups as unsafe/boundary material.
  The sharpest cells are high clustering, high blur, high dropout, and long
  delay. `sundog_lean` also shows false-flag costs in several otherwise
  non-terminal comparisons.
- This is deliberately not the Phase 10 operating-envelope verdict. The axis
  spokes expose failure mechanisms and odd pockets; Phase 10 still needs the
  locked grid map before any public "inside this envelope" claim can graduate.

### Phase 10 - Operating Envelope Map

Goal: lock the earned claim.

Deliverables:

- Grid map over mine density, field blur, noise, delay, clustering, and scan
  budget.
- Candidate-envelope CSV where Sundog improves over passive and naive baselines
  while staying below false-flag / hazard-trigger thresholds.
- Best-cell and worst-cell replay links for the browser.
- Failure-mechanism labels such as:
  - `field_uninformative`
  - `frontier_ambiguity`
  - `delay_misread`
  - `overflagged`
  - `probe_budget_exhausted`
  - `controller_overcommitted`

Exit criterion: the public copy can say "inside this envelope" and point to a
map rather than hand-waving.

#### Phase 10 Pre-Registration (locked before grid run)

This block must land before any Phase 10 grid data is generated. The commits
below are not negotiable post-hoc — if a real reason emerges to change any of
them, the doc updates with an addendum block preserving the original
pre-registration text so the audit record stays intact.

**Candidates.** `sundog_lean` and `sundog_minimal` are evaluated as parallel
candidates against identical gates. If their envelopes differ, both ship with
their own positive and negative cells. If `sundog_lean` dominates
`sundog_minimal` on every published cell, the minimal lane is retired.
`sundog_controller` remains in the matrix as a lineage row but is not graded
against Phase 10 candidate gates — its failed status was settled in Phase 5.

**Baselines for matched-seed deltas.** `naive_pressure` (the primary
comparator) and `random_reveal` (the floor). `threshold_flagger` and
`oracle_safe` remain in the matrix for context but do not gate candidacy.

**Grid shape.** A 3D primary factorial plus 1D spokes.

- Primary 3D factorial: `mine_density` × `pressure_noise` × `dropout_rate`
  - `mine_density`: 0.10, 0.16, 0.22 (drop 0.28 — inside the `do_not_use` regime)
  - `pressure_noise`: 0.1, 0.5, 1.0, 2.0 (drop 5/10 — inside the `do_not_use` regime)
  - `dropout_rate`: 0.05, 0.20, 0.35 (drop 0.65/0.85 — inside the `do_not_use` regime)
  - = 36 primary cells
- 1D spokes (all other axes at doc-default settings):
  - `sensor_delay`: 0, 1, 2 (drop 4 — inside the `do_not_use` regime) = 3 cells
  - `clustering_strength`: 0, 0.35, 0.65 (drop 0.9) = 3 cells (skip 0 if it
    duplicates the primary grid origin)
  - `kernel_blur`: 1.0, 2.0, 3.0 (drop 5/8) = 3 cells
  - `scan_budget`: 0, 1, 3, 6 = 4 cells
  - = ~13 spoke cells (de-duped)
- Total cells in published verdict: ~49
- Seeds per cell: 64 primary acceptance, 16 for the `phase10:smoke` variant

**Candidate-cell gates (all must hold for a positive verdict).**

A cell is `candidate=true` for a given Sundog variant if all of the following
hold on the 64-seed matched-pair sample:

1. Budget-adjusted matched-seed mean delta vs `naive_pressure` ≥ **+1.0** safe
   tiles.
2. Budget-adjusted matched-seed mean delta vs `random_reveal` ≥ **0**.
3. The cell is *not* in the `do_not_use` static boundary regime per
   `assessStaticBoundary`.
4. Mean false-flag count delta vs `naive_pressure` ≤ **+1.0**.
5. Mean mine-trigger rate delta vs `naive_pressure` ≤ **+0.10** (Sundog cannot
   buy safe tiles by triggering more mines than naive does).
6. 95% bootstrap CI on the budget-adjusted matched-seed delta vs
   `naive_pressure` excludes zero. The bootstrap is 1,000 resamples over the
   matched-pair list.

A cell is `failure_regime=true` if the budget-adjusted matched-seed mean delta
vs `naive_pressure` ≤ **−1.0** safe tiles. Mechanism labels are pulled from
the static boundary mechanisms and merged with the empirical observation
(e.g., a `do_not_use` cell that also shows controller harm relative to naive
gets both labels in `mechanism_codes`).

**Worst-cell publication rule.** Every published positive cell ships
alongside at least one matched-seed `failure_regime` cell of equal
documentation weight. The Phase 11 public artifact's default replay link
shows the worst cell first, the best cell second. If Phase 10 produces zero
candidate cells, that is itself a valid verdict — the doc and the public page
say so directly and the workbench stays at Planned Workbench tier.

**Promotion to Operating-Envelope Study tier requires** all of: (a) at least
one cell with `candidate=true` for at least one Sundog variant, (b) at least
one paired `failure_regime` cell published with the candidate, (c) the
budget-adjusted lead survives net of scan-budget tax (already encoded in the
primary statistic), (d) the static boundary mechanisms and empirical verdict
labels do not contradict each other in a way that suggests measurement error.

Phase 10 verdict status: `CONFIRM` on the locked 64-seed run.

- `npm run mines:phase10` now runs
  `scripts/mines-phase10-envelope.mjs` and writes ignored local verdict
  artifacts under `results/mines/phase10-envelope/`: `manifest.json`,
  `summary.json`, `verdict.json`, `verdict.md`, `cell-manifest.csv`,
  `trial-outcomes.csv`, `matched-comparison.csv`, `envelope.csv`,
  `cell-class-map.csv`, and `best-worst-cells.csv`.
- Locked run shape: 46 de-duplicated envelope cells x 7 modes x 64 seeds =
  20,608 trials. The script emits 138 envelope rows: `sundog_lean`,
  `sundog_minimal`, and lineage-only `sundog_controller` over each cell.
- Result: 2 candidate rows passed all pre-registered gates and 34
  `failure_regime` rows mapped the paired negative boundary. Candidates split
  evenly: one `sundog_lean` row and one `sundog_minimal` row.
- Confirmed candidate cell: density 0.16 / pressure noise 2.0 / dropout 0.2.
  `sundog_minimal` reached +7.21875 budget-adjusted safe tiles versus
  `naive_pressure` with 95% bootstrap CI [3.375, 11.078516]; `sundog_lean`
  reached +6.3125 with CI [1.921094, 11.25]. Both stayed within false-flag
  and mine-trigger gates.
- Worst publication cell: `sundog_lean` on density 0.22 / pressure noise 1.0 /
  dropout 0.35, with -4.71875 budget-adjusted safe tiles versus
  `naive_pressure` and CI [-8.375781, -1.496484]. Mechanisms:
  `frontier_ambiguity|field_uninformative|controller_overcommitted`.
- Best/worst replay URLs round-trip through `npm run mines:phase7:replay` with
  stable action-trace hashes. Best seed: 47, `sundog_minimal` versus
  `naive_pressure`. Worst seed: 39, `sundog_lean` versus `naive_pressure`.
- Disposition: promote Pressure Mines to Operating-Envelope Study tier after
  Phase 11 public polish, with the claim restricted to the named density /
  noise / dropout pocket and published beside the worst-cell failure replay.

**Prerequisites settled before grid run:**

- Boundary vocabulary frozen in `public/js/mines-boundary.mjs`, now split into
  `assessStaticBoundary` (config-only; consumed by Phase 10 envelope verdict)
  and `assessLiveBoundary` (per-seed overlay; consumed by the browser panel
  only). The compat wrapper `assessMinesBoundary` preserves the original API.
- Replay URL grammar extended in the harness (`scripts/mines-phase4-baselines.mjs`)
  to accept opt-in overrides for `mine_count`, `width`, `height`, `scan_budget`,
  `cluster_strength`, `sigma`, `sigma_noise`, `dropout`, `delay`. Every Phase 10
  cell is reproducible via `--replay-url` even when its config falls outside
  any named preset/sensor pair. Browser-side hydration now accepts the same
  override grammar, and `mines.html` defaults to the Phase 10 best replay while
  exposing one-click best/worst replay buttons for public inspection.
- Phase 9 boundary classifier thresholds verified coherent with Phase 9 sweep
  values: density (0.18/0.24), clustering (0.45/0.75), blur/noise/dropout
  multi-knob (σ≥3 OR noise≥1 OR dropout≥0.35 caution; σ≥6 OR noise≥5 OR
  dropout≥0.7 unsafe), delay (1/3), board size ≥256 tiles, scan budget ≤1/0.
  Phase 10 inherits these without redefinition.

### Phase 11 - Public Artifact And Promotion Pass

Goal: make Pressure Mines promotion-ready without overclaiming.

Deliverables:

- Final `mines.html` tab linked from the public nav.
- Short writeup titled around "Reading the Minefield."
- Applications gallery card with evidence tier tied to the runnable artifacts.
- `docs/APPLICATIONS.md` row carrying the bounded claim after the harness and
  operating-envelope result land.
- Short motion capture or replay clip of a good-cell and bad-cell seed.
- Caption language that names both success and failure boundary.

Exit criterion: a first-time visitor can see the board, understand the theoremic
move in under a minute, and also see that the field is not always enough.

Phase 11 status:

- Tier badge on `mines.html` promoted from "Planned Workbench" to
  "Operating-Envelope Study" in the stage eyebrow and the footer note. The
  promotion is grounded in the Phase 10 CONFIRM, not in visual impression.
- The "Open Question" claim box on the body copy replaced with a
  "Confirmed pocket / Failure region" pair that names the specific cell
  (`density 0.16 / pressure noise 2.0 / dropout 0.2`), the
  budget-adjusted matched-seed delta (`+7.21875` for `sundog_minimal`,
  `+6.3125` for `sundog_lean`), the 95% bootstrap CI bounds, and the
  matched failure cell. Negative region ships at equal prominence per the
  pre-registration rule.
- One-click best/worst replay shortcuts added in two places: the right
  rail (`mines-load-best`, `mines-load-worst`) and the body copy
  (`mines-body-load-best`, `mines-body-load-worst`). Both call into a
  `loadCellParams` helper that assembles a URL from frozen
  `BEST_CELL_PARAMS` / `WORST_CELL_PARAMS` objects baked into
  `public/js/mines-browser.mjs` from the Phase 10 verdict's `bestCell` and
  `worstCell` entries.
- Browser-side replay URL plumbing landed for real this pass:
  `buildReplayURL`, `copyReplayURL`, `hydrateFromURL`, and
  `loadCellParams` all wired. The page hydrates URL params on boot,
  including the opt-in board/sensor overrides (`mine_count`, `width`,
  `height`, `scan_budget`, `cluster_strength`, `sigma`, `sigma_noise`,
  `dropout`, `delay`). Round-trip determinism matches the harness's
  `--replay-url` contract.
- `docs/APPLICATIONS.md` row updated to Operating-Envelope Study tier and
  cites the Phase 10 verdict cell. The applications gallery card and landing
  page card now point at the runnable Mines workbench with the promoted tier.
- Deferred to a later pass (not blocking Phase 11 exit): retiring the
  `sundog_no_*` ablation modes from the live UI mode dropdown (they remain
  in the harness for diagnostic use); adding a row to
  `docs/presentation/applications-index.md`; producing a short public replay
  capture of the confirmed pocket and paired failure region.

## Pre-Registered Comparison Shape

The public evidence should be about matched-seed, bounded comparisons, not
free-floating highlight runs.

Minimum comparison shape:

- Sundog versus passive/random;
- Sundog versus naive local pressure heuristic;
- Sundog versus threshold flagger;
- Sundog versus privileged oracle;
- Sundog ablations under shuffled, delayed, or gradient-removed field.

### Pre-Registration Discipline

The following commitments must land in this doc before Phase 7 logs are
generated, not after:

1. **Primary outcome statistic.** A single comparison number per envelope cell
   (for example: budget-adjusted matched-seed delta in safe-tiles-revealed
   versus the naive local pressure heuristic, with the threshold flagger
   reported as a second statistic, not a substitute).
2. **Envelope axes.** The exact axes that define the operating envelope grid
   (mine density, clustering strength, kernel sharpness/blur, pressure noise,
   sensor delay, scan dropout, scan budget) and the values swept on each axis.
3. **Effect-size or significance threshold.** What counts as a positive cell
   (for example: matched-seed bootstrap CI excludes zero at a stated level, or
   effect size exceeds a stated floor). Reporting both raw and budget-adjusted
   versions is required.
4. **Failure-mechanism palette.** The named failure labels Phase 10 will use
   (`field_uninformative`, `frontier_ambiguity`, `delay_misread`, etc.) are
   committed in the doc so post-hoc relabeling is not available.
5. **Negative-region publication rule.** Every published positive cell ships
   alongside at least one matched-seed negative cell. The public artifact in
   Phase 11 carries this binding: no positive copy without negative copy.

### Expected Result Shape

- Sundog improves over passive and naive heuristics inside a named envelope;
- oracle remains the ceiling;
- outside the envelope, the controller degrades or fails cleanly;
- the negative region ships with the positive region.

## Current State

**Evidence tier:** Operating-Envelope Study

The Sundog Mines workbench has completed the Phase 1-10 evidence spine and is
now in Phase 11 public polish. The promoted claim is narrow: in the Phase 10
density 0.16 / pressure noise 2.0 / dropout 0.2 pocket, `sundog_minimal` and
`sundog_lean` reveal more budget-adjusted safe tiles than `naive_pressure`
before mine trigger. It is not a claim that Sundog clears the minefield.

Core shipped pieces:

- `public/js/mines-core.mjs`: deterministic board generation and action loop.
- `public/js/mines-sensor.mjs`: pressure field, gradient, scan pulse, floor
  enforcement, and privileged audit helpers.
- `scripts/mines-phase3-diagnostics.mjs` / `npm run mines:phase3`: local,
  ignored diagnostic outputs under `results/mines/phase3-diagnostic`.
- `public/js/mines-controllers.mjs`: Phase 4 baseline lane definitions and
  information budgets.
- `scripts/mines-phase4-baselines.mjs` / `npm run mines:phase4`: local,
  ignored matched-seed baseline outputs under `results/mines/phase4-baselines`.
- `scripts/mines-phase10-envelope.mjs` / `npm run mines:phase10`: locked
  operating-envelope grid and verdict artifacts under
  `results/mines/phase10-envelope`.
- `mines.html`: Phase 11 public surface, defaulting to the confirmed replay and
  publishing the matched failure replay beside it.

The promoted page must keep the caveats visible: both Sundog and naive trigger
mines on every seed in the best cell; `threshold_flagger` has higher survival
there at the cost of false flags; the candidate cell is `watch_boundary`, not a
clean readability region; and neither `sundog_lean` nor `sundog_minimal` retires
the other.

## Recommendation

Proceed with Pressure Mines before Cards if the goal is to make Sundog legible
as a broad vocabulary for occluded systems rather than as gambling or
game-optimization technology.

Pressure Mines is strong because it expresses the theorem in ordinary language:

- the danger is hidden;
- the environment leaks structure;
- action is still required;
- the leak is useful only inside a boundary.

That is the right kind of game-native workbench for `sundog.cc`.
