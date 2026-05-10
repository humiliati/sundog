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
  memory over prior reveals, and conservative fallback when observability
  drops.
- **Actionable output:** bounded reveal / flag / scan / abstain decisions that
  improve survival and board progress over naive local heuristics inside a
  mapped observability envelope, failing cleanly outside it.

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

Avoid broader formulations such as:

- "Sundog solves Minesweeper."
- "The pressure field reveals the hidden board in general."
- "Indirect inference beats logical deduction."
- "This proves uncommon intuition."
- "This is better game AI."

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
| scan result | Bounded sensor action | optional active probe action |
| revealed safe/unsafe tile history | Public memory | available to non-passive controllers |
| flag history | Public memory | available to non-passive controllers |
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

> Sundog Pressure Mines asks whether an agent can clear a minefield when the
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
- Define page shape: `mines.html`, linked later from the gallery and nav if it
  earns the slot.

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
- Candidate kernel families:
  - tile pressure;
  - local finite-difference gradient;
  - confidence / observability score;
  - bounded scan pulse result if active probing is enabled.
- Privileged audit comparing pressure-derived risk estimates against true mine
  occupancy without exposing the occupancy to the controller.

Exit criterion: the board can show when the field is informative and when it
collapses into ambiguity.

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

Exit criterion: at least one field signature remains informative in a named
operating envelope.

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

Exit criterion: every public lane has a stated information budget.

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
batch studies are complete.

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

## Pre-Registered Comparison Shape

The public evidence should be about matched-seed, bounded comparisons, not
free-floating highlight runs.

Minimum comparison shape:

- Sundog versus passive/random;
- Sundog versus naive local pressure heuristic;
- Sundog versus threshold flagger;
- Sundog versus privileged oracle;
- Sundog ablations under shuffled, delayed, or gradient-removed field.

The expected result shape should be:

- Sundog improves over passive and naive heuristics inside a named envelope;
- oracle remains the ceiling;
- outside the envelope, the controller degrades or fails cleanly;
- the negative region ships with the positive region.

## Current State

**Evidence tier:** Planned Workbench

The Sundog Mines workbench is currently in Phase 0. This roadmap document and
initial stub implementation establish the claim boundary and task definition.

Promotion to **Instrumented Prototype** requires Phase 1-6 deliverables landing
in the public tree.

Promotion to **Operating-Envelope Study** requires Phase 7-10 artifacts and a
confirmed improvement pocket over passive and naive baselines.

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
