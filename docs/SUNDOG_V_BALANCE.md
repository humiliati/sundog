# Sundog Balance Workbench

Working hook:

> It balances the body by reading the shadow.

Sundog Balance is the proposed last pre-promotion application workbench. It is
not another optics extension and not another large physics claim. It is a
classic control task made visually obvious: a cart-pole / inverted-pendulum
system must keep a pole upright while the Sundog controller is denied the pole
angle. It can read only the pole's cast shadow, cart proprioception, and its own
action history.

The public question is small enough to defend:

> Can an agent maintain upright balance from an indirect projection when the
> true alignment variable is hidden?

The workbench should live beside the three-body tab as `balance.html`: a public,
interactive surface with browser simulation, baselines, telemetry panels,
failure-boundary presets, and a matching writeup. The final promotional pass can
capture a short `.gif` of the arm/pole recovering from a disturbance and place
it after the hero on `index.html`.

*[nice-to-have #12]*

## Sundog Expression *(pre-staged for the eventual APPLICATIONS.md row)*

- **Hidden target:** upright pole angle (`theta`).
- **Indirect signal:** cast-shadow centroid, length, and velocity on the floor
  plane, sampled with bounded noise and delay.
- **Transformation:** SCAN/SEEK/TRACK shadow-residual control with
  cart-proprioceptive history and observability gating.
- **Actionable output:** bounded cart force command maintaining upright pole
  inside a mapped lighting and delay envelope, failing cleanly outside it.

This block is here so the cross-application comparison row in
`APPLICATIONS.md` and the gallery card's indirect-signal/transformation/output
triplet do not drift between the roadmap and the eventual writeup. When Phase
11 lands, both surfaces should quote this block verbatim.

*[/nice-to-have #12]*

***[must-fix #3]***

## Current State

First executable scaffold started. The Sundog tree now includes:

- `balance.html`: an unpromoted browser workbench tab;
- `public/js/balance-core.mjs`: deterministic cart-pole dynamics, shadow
  projection, sensor delay/noise scaffolding, and baseline controller modes;
- `public/js/balance-browser.mjs`: canvas renderer, controls, telemetry strip,
  and Lounas-derived robotics balance motif;
- `scripts/balance-harness.mjs`: a Phase 7 smoke/replay runner sharing the
  same dynamics, shadow sensor, and controller modules as the browser page.

No committed Balance results directory, no Phase 8 metric tables, and no Phase
10 operating-envelope verdict ships in the Sundog tree today. Local Phase 7
smoke and replay outputs are written under ignored `results/balance/` paths
and are calibration artifacts, not public evidence. The current artifact is a
Phase 1-7 scaffold, not evidence.

This section is here so the broadcast surfaces (gallery card, APPLICATIONS.md,
claims-and-scope.md) do not get a row until the runnable-artifact gates below
are met. Until then, Balance sits at **Forward-Looking Application Design /
Conceptual Lineage** tier in the application family — sibling to LAGM in the
EyesOnly section, not sibling to the EyesOnly headless runner.

Promotion to **Planned Workbench** tier requires Phase 0–4 deliverables
landing +++[forward-plan-A]+++ *(achieved 2026-05-08; phases 5–6 also landed,
Phase 7 smoke in flight)* +++[/forward-plan-A]+++. Promotion to
**Operating-Envelope Study** tier requires the Pre-Registered Verdict Template
below to return a CONFIRM verdict.

+++[forward-plan-A]+++

### Phase 7 Smoke Read (2026-05-08)

A 12-seed smoke run of the Phase 7 harness produced a result shape that is
encouraging signal for the workbench's central claim:

- Sundog **survives all 12 seeds** on the `easy`, `recoverable`, and
  `near_fall` lighting/disturbance presets.
- Sundog **degrades** on the `noisy_shadow` preset (intermittent survival).
- Sundog **fails** on the `delayed_shadow` preset (does not recover from
  the standard impulse).

The shape matches the Ratified Hook Language line — *"the shadow is enough,
until it is not"* — at smoke scale. The overhead-light/high-delay regime
collapses observability and the controller loses cleanly rather than
thrashing.

**Discipline guard.** This is signal, not verdict. The Pre-Registered Verdict
Template below remains gated on Phase 10 operating-envelope data on a
≥ 100-seed slate, not on the Phase 7 smoke result. A clean 12-seed smoke
that fails to reproduce under the Phase 10 slate would still REFUTE; a
smoke result that misses a lighting boundary the Phase 10 data recovers
would still CONFIRM. The verdict gate does not move forward to Phase 7.
The smoke result is here to inform the Phase 12–14 forward planning below
under the assumption that Phase 10 returns CONFIRM, not to short-circuit
Phase 10's reads.

+++[/forward-plan-A]+++

***[/must-fix #3]***

## Why This Workbench

Three-body gave Sundog a high-recognition physics hook, but the result is hard
to read without accepting the apparatus. Balance is the opposite: viewers
understand success or failure instantly. The pole is upright or it falls.

This also returns the theorem to its native vocabulary without repeating the
core mirror experiment:

1. hidden target: upright pole angle;
2. indirect signal: shadow centroid, length, and motion;
3. action coupling: cart force changes pole motion, which changes the shadow;
4. control: stabilize from the shadow-derived signature;
5. boundary: when the light geometry destroys observability, the controller
   should fail cleanly.

## Visual Direction

The page should feel like a workbench, not a marketing splash. The first screen
should open directly into the simulation.

Target composition:

- foreground: cart, pole, cast shadow, target upright band, disturbance pulses;
- background: subtle spherical-robot / gyroscopic-balance motif, visually rich
  but not allowed to obscure telemetry;
- right rail: controller mode, sensor tier, disturbances, light geometry;
- lower strip: angle error, shadow residual, control force, survival timeline;
- side-by-side comparison mode: direct oracle, Sundog shadow controller, and
  naive baseline on the same seed.

The referenced `Gontary101/lounas-portfolio` repo is an approved design and
implementation reference for this workbench. Per the project-owner note from the
LinkedIn agreement, Sundog may borrow heavily from Lounas for the Balance visual
background and for a BoxForge/CSS arm template derived from that robotics style.
Keep borrowed pieces traceable and credited in
[`THIRD_PARTY_REUSE.md`](THIRD_PARTY_REUSE.md).

Implementation preference:

- keep simulation equations renderer-independent;
- start with Canvas 2D if speed matters;
- borrow the Lounas robotics visual language for the background/arm template
  when it improves the first-read experience;
- add Three.js only if the background or robot arm needs true 3D motion;
- if Three.js is added, keep the cart-pole foreground readable and verify
  desktop/mobile screenshots before promotion.

## Actionability Audit

The workbench must separate observability tiers from day one.

| Signal | Tier | Use |
| --- | --- | --- |
| true pole angle `theta` | Privileged | oracle baseline, metric, hidden residual |
| true angular velocity `theta_dot` | Privileged | oracle baseline, metric |
| pole energy | Privileged diagnostic | analysis only |
| shadow endpoint / centroid | Sensor-available | Sundog controller input |
| shadow length | Sensor-available | Sundog controller input and observability gate |
| shadow velocity | Sensor-available, derived | controller input after finite differencing |
| cart position / velocity | Proprioceptive | available to all non-passive controllers |
| force command history | Proprioceptive | available to Sundog controller |
| light position / angle | Calibration parameter | known environment setting, not a hidden target |

The central typing should be explicit:

```text
G_t = theta_t                         hidden goal residual
S_t = [shadow_x, shadow_length, dS/dt] indirect projection
u_t = cart force command
H_t = local coupling between recent u and recent S
```

The claim is not that the shadow is mystical. The claim is that in some light
geometries the shadow is an observable and controllable projection of the hidden
balance residual. In other geometries, such as near-overhead lighting or heavy
sensor delay, the projection should become uninformative and the controller
should lose.

## Ratified Hook Language

Safe hook:

> Sundog Balance asks whether an agent can keep a pole upright without seeing
> the pole angle, using only the shadow the pole casts as it moves.

Short version:

> The body is hidden. The shadow is enough, until it is not.

Avoid:

- "Sundog solves balance control."
- "The controller infers physical truth from shadows in general."
- "This proves embodied alignment."
- "The shadow controller beats the oracle."
- "A gif is evidence."

+++[forward-plan-C]+++

**Forward-planning Avoid additions** (gated to land alongside Phases 12–14
when those phases ship; pre-registered here so the broadcast pass cannot
drift):

- "Sundog Balance is robotics-ready." Phase 13 produces a controller that
  respects hardware-realistic constraints in sim. Hardware validity is a
  separate tier (Phase 13.5). The safe phrasing is: *Sundog Balance's
  controller respects hardware-realistic constraints in simulation;
  hardware deployment is a separate engineering pass.*
- "The Sundog controller can be deployed to a robot." Same reason as
  above. The ROS-shaped state interface is documented, not deployed.
- "Sundog Balance generalises to all robotic balance problems." Phase 14
  tests a three-plant slate. Generalisation is bounded to the documented
  family of plants tested. The safe phrasing is: *The pattern transfers
  to N of M tested plants under documented operating envelopes; falsified
  plants are reported alongside confirmed ones.*
- "Beating a human player proves the controller is correct." Phase 12's
  human-vs-agent mode is a falsification surface, not a benchmark.
  Human-survival-margin distributions are reported as evidence about the
  controller's failure modes, not as proof of correctness. A controller
  that beats untrained humans on the easy preset and loses to skilled
  humans on the noisy preset is teaching observers about its own limits;
  that is the point.
- "Losing to a human player refutes the theorem." Symmetric guard. Human
  performance is a falsification *surface*, not a falsification *gate*.
  The Phase 10 verdict template, not the human leaderboard, decides
  CONFIRM / REFUTE / AMBIGUOUS.

+++[/forward-plan-C]+++

## Roadmap

### Phase 0 - Claim Boundary And Benchmark Choice

Goal: define the exact control task before building the page.

Deliverables:

- Choose the first plant: classic cart-pole with one pole, one cart, finite
  rail, bounded force, and deterministic timestep.
- Declare the hidden residual: pole angle from upright.
- Declare what the Sundog controller cannot read: true angle, true angular
  velocity, and full simulator state.
- Declare the evidence tier: planned workbench, not yet a research result.
- Define target page shape: `balance.html`, linked later from the gallery and
  index nav if it earns the slot.

Exit criterion: the workbench has one sentence that can be attacked and one
task definition that can be reproduced.

### Phase 1 - Physics Core

Goal: implement or specify a deterministic cart-pole model independent of
rendering.

Deliverables:

- State vector: `x`, `x_dot`, `theta`, `theta_dot`.
- Inputs: bounded cart force, optional disturbance impulse.
- Integrator: fixed-step RK4 or semi-implicit Euler with documented timestep.
- Reset presets: easy balance, recoverable lean, near-fall, rail-edge, noisy
  shadow, delayed shadow.
- Terminal events: pole fallen, rail hit, timeout success.

Exit criterion: the same model can drive browser animation and headless batch
runs.

### Phase 2 - Shadow Sensor Model

Goal: turn the hidden pole state into a lower-dimensional indirect signal.

Deliverables:

- Light geometry controls: azimuth, elevation, source distance, softness.
  **[should-add #6]** *Pin the projection model in the doc before the
  implementer writes it: directional light source treated as point-at-infinity
  for the default case (parallel rays); shadow endpoint computed by ray
  intersection from each pole endpoint along the light direction with the
  floor plane `z = 0`; soft shadow approximated as a penumbra width
  proportional to (source softness × pole height / source distance). The
  default sweep uses the directional model; the perspective-light model is
  Phase 9 sweep variant only.*
- Cast-shadow projection from pole endpoints to floor plane. **[should-add #6]**
  *For pole base at `(x_b, 0)` and pole tip at `(x_t, h_t)` with light
  direction unit vector `(l_x, l_z)` (z-up, l_z < 0 means the light points
  downward), the floor-plane shadow tip is at `x_b + (x_t − x_b) − h_t · (l_x / l_z)`.
  Shadow length is the distance from base-shadow to tip-shadow. Both fields
  are passed to the sensor model as floats with the noise/quantisation
  applied per the noise model below.*
- Sensor outputs: shadow endpoint, centroid, length, velocity estimate, and
  confidence / observability score.
- Noise model: pixel jitter, frame delay, quantization, dropout.
- Privileged audit: compare shadow-derived estimates against true angle without
  giving estimates to the controller unless explicitly in an ablation.

Exit criterion: the workbench can show when the shadow is informative and when
the lighting collapses the signal.

### Phase 3 - Diagnostic Benchmark

Goal: verify that the indirect signal forecasts balance-relevant events before
using it for control.

Questions:

- Does shadow endpoint or length predict pole angle sign?
- Does shadow velocity predict angular velocity sign?
- How much lead time exists before fall?
- Which light elevations make the signal ambiguous?

Metrics:

- sign accuracy for angle and angular velocity;
- AUROC or precision/recall for near-fall warnings;
- correlation between shadow residual and hidden angle residual;
- degradation under noise, delay, and overhead light.

Exit criterion: at least one shadow signature remains informative in a named
operating envelope.

### Phase 4 - Baseline Set

Goal: establish fair comparison lanes.

Required modes:

- **Passive:** no force.
- **Naive cart centering:** centers the cart without pole-angle information.
- **Naive shadow centering:** tries to keep shadow endpoint centered without
  dynamics or history.
- **Sundog shadow controller:** uses shadow residual, shadow velocity,
  cart proprioception, and action history.
- **Privileged oracle:** LQR/PID over true `theta` and `theta_dot`.
- **Ablations:** shuffled shadow, delayed shadow, no shadow velocity, no action
  history.

Exit criterion: every public comparison lane has a stated information budget.

### Phase 5 - Sundog Controller Prototype

Goal: build the first controller that acts from the shadow rather than true
angle.

**[should-add #7]** *Ground the SCAN/SEEK/TRACK pattern in canonical text
before implementing. The pattern is borrowed from the photometric mirror
experiment; cite `<sundog>/notebooks/sundog alignment theorem.txt` and
`<sundog>/docs/PAPER_v1_draft.md` so the implementer matches the algorithmic
shape that already exists in the program rather than re-deriving it.*

Candidate structure:

- `SCAN`: apply tiny bounded force probes and record how the shadow residual
  changes.
- `SEEK`: choose the force direction that reduces shadow residual and increases
  upright confidence.
- `TRACK`: maintain an upright-shadow signature with a small history-dependent
  controller.
- `REACQUIRE`: when confidence drops or delay grows, switch back to probing or
  controlled slowdown.

This does not need to be learned initially. A hand-authored controller is
acceptable because the goal is to test the observability channel, not to claim
learning.

Exit criterion: the controller keeps the pole alive longer than passive and
naive baselines on a small seeded slate inside the diagnostic-positive envelope.
**[should-add #5]** *Pin the minimum required behaviour: the controller
maintains `|theta| < 0.3 rad` for ≥ 10 simulated seconds inside the Phase 3
diagnostic-positive envelope on at least 80% of the seeded slate. Below that
floor, the controller is not yet functional and Phase 6 web shell work is
gated behind further controller iteration. Above that floor, Phase 5 is
considered complete and Phase 6 unblocks.*

### Phase 6 - Real-Time Web Projection

Goal: create the public browser workbench shell early.

**[should-add #8]** *Pre-condition before any borrowed asset lands: confirm the
Lounas credit clause in `<sundog>/docs/THIRD_PARTY_REUSE.md` is current and
matches the LinkedIn agreement scope. Borrowing visual assets without that
ledger updated is the kind of audit-trail gap the program has committed to
avoiding. Update the ledger first, borrow second.*

Deliverables:

- `balance.html` with responsive layout matching `threebody.html`.
- Canvas or Three.js foreground showing:
  - cart, pole, floor shadow, light vector, disturbance cue;
  - hidden angle ghost only when "show privileged diagnostics" is enabled;
  - controller force arrows and phase labels.
- Telemetry charts for shadow residual, hidden angle metric, force, survival,
  and observability score.
- Mode controls for passive, naive, Sundog, oracle, and ablations.
- Seed controls and presets for lighting / disturbance / rail constraint.

Exit criterion: the page demonstrates the qualitative phenomenon even before
large batch studies are complete. ~~if Three.js is added, keep the cart-pole
foreground readable and verify desktop/mobile screenshots before promotion~~
**[should-add #9]** *Hard exit: desktop **and** mobile screenshot acceptance
pass before the page links from any nav surface. The workbench will be linked
from `index.html` and visited on phones; mobile readability is not optional
and not gated on Three.js. The acceptance pass lives in
`results/balance/screenshots_<datetime>/` with one desktop and one mobile
capture per controller mode.*

### Phase 7 - Reproducible Harness

Goal: move from a visual toy to repeatable evidence.

Deliverables:

- Headless JS runner sharing the same dynamics and controller modules as the
  browser page.
- Seeded trial manifests recording controller mode, light geometry, force
  limit, disturbance schedule, noise, delay, timestep, and initial state.
- Per-trial JSONL or CSV logs for state, sensor outputs, controller action,
  observability, and terminal outcome.
- Browser replay URLs and `npm run balance:phase7:replay` support so a workbench
  configuration can round-trip through the headless runner.
- Replay verification with `npm run balance:phase7:verify`, checking emitted
  replay links against their recorded terminal outcome and survival time.
- NPM scripts mirroring the three-body pattern, for example:

```bash
npm run balance:phase7
npm run balance:phase8
npm run balance:phase9
```

Exit criterion: a browser seed can be replayed exactly in the harness.

+++[forward-plan-A]+++ *Phase 7 continuation landed: the workbench now exposes
seeded replay URLs, the harness accepts `--replay-url`, and local harness runs
write `replay-index.json` entries with browser URLs plus replay commands. The
next Phase 7 hardening pass added replay verification, writing
`replay-verification.csv` and failing the command on outcome/time drift. This is
still replay infrastructure, not a verdict artifact.* +++[/forward-plan-A]+++

### Phase 8 - Recovery And Event Metrics

Goal: make the claim about useful control, not just pretty balancing.

Metrics:

- survival time;
- RMS hidden angle error;
- max angle after disturbance;
- recovery time after impulse;
- rail-hit rate;
- force budget / saturation count;
- shadow-confidence loss count;
- fall-warning lead time from shadow metrics.

Deliverables:

- Event labels: disturbance, near-fall, recovery, rail hit, fallen.
- Threshold sweeps for fall-warning from shadow residual and shadow velocity.
- Comparison tables against passive and naive modes on matched seeds.

Exit criterion: the page can show a recovery curve, and the docs can distinguish
successful balance from merely delaying a fall.

### Phase 9 - Sensor Degradation And Observability Boundary

Goal: turn the failure boundary into first-class content.

Sweeps:

- light elevation from low/long shadow to overhead/short shadow;
- shadow sensor delay;
- pixel jitter / noise;
- dropped frames;
- rail length;
- force limit;
- disturbance magnitude.

Expected boundary:

- low-angle light may produce long, readable shadows but noisy scaling;
- overhead light should collapse shadow length and remove angle information;
- delay should harm recovery faster than steady-state balance;
- rail and force saturation should expose limits similar to the photometric
  joint-limit cliff.

Exit criterion: the workbench has a visible "where Sundog should not be used"
panel.

### Phase 10 - Operating Envelope Map

Goal: lock the earned claim.

Deliverables:

- Grid map over light elevation, delay, noise, force limit, and disturbance.
- Candidate-envelope CSV where Sundog improves over passive and naive baselines
  while staying below force/saturation thresholds.
- Best-cell and worst-cell replay links for the browser.
- Failure-mechanism labels such as `shadow_unobservable`, `delay_destabilized`,
  `force_saturated`, `rail_limited`, and `controller_overcorrected`.

Exit criterion: the public copy can say "inside this envelope" and point to the
map rather than hand-waving.

***[must-fix #1]***

### Pre-Registered Verdict Template

Phase 10 produces an Operating Envelope Map and a candidate-envelope CSV. The
verdict template below pre-commits to *what those artifacts have to say* before
the workbench can promote tiers or broadcast. This is the same Money-Bags-style
discipline that ratifies Stage 1 verdicts before captures, restated for the
Balance workbench. Verdict is data-driven; disposition is locked.

**Pre-registered predictions.**

*P1 — central effect.* Inside the diagnostic-positive envelope from Phase 9,
the Sundog shadow controller's mean survival time exceeds naive shadow-centering
by at least 1.5× on at least 30% of operating-envelope cells, on a matched-seed
slate of ≥ 100 seeds. Failure here REFUTES the workbench's central claim.

*P2 — failure boundary realness.* On overhead-light cells (light elevation
≥ 80°) and on high-delay cells (sensor delay ≥ 200 ms), Sundog should not
exceed naive shadow-centering. Sundog beating naive in the failure regimes
indicates the baseline is undertuned; triggers re-run after baseline
re-tuning, not a CONFIRM.

*P3 — recovery monotonicity.* Recovery time after the Phase 8 standard impulse
is monotonically worse as sensor delay increases across the slate. Non-monotonic
curves AMBIGUATE the slate (sensor model has a confound; light-geometry sweep
not orthogonalised against delay sweep).

*P4 — privileged-oracle ceiling.* The privileged LQR/PID oracle exceeds Sundog
by a measurable margin on at least 80% of slate cells. Sundog matching or
beating the oracle indicates the oracle is mis-implemented (gain tuning,
integration-step mismatch, or `theta_dot` sign error). Triggers oracle audit
before any verdict.

**Verdict template.**

After the Phase 10 candidate-envelope CSV and Phase 8 metric tables land,
the writeup asserts *one* of three verdicts:

*CONFIRM.* P1 holds with seed-paired bootstrap CI excluding zero. P2 holds.
P3 holds. P4 holds. The workbench promotes to Operating-Envelope Study tier.
The gallery card upgrades the indirect-signal/transformation/output triplet
from forward-looking language to past-tense achieved language. The
applications-gallery row, the APPLICATIONS.md row, and the
claims-and-scope.md row all land in the same pass.

*REFUTE.* P1 fails — Sundog does not beat naive shadow-centering by the
pre-registered margin. The workbench stays at Planned Workbench tier with
a negative-finding banner: "Sundog Balance is the application that
attempted shadow-derived control of an inverted pendulum and did not
beat naive shadow-centering on the tested slate." The negative finding
is itself reportable — it is the same kind of evidence Money Bags would
have produced if Stage 1 captures had returned an architecture-mode
kill-switch read. The Avoid list in this doc is amended to include
"Sundog Balance demonstrates shadow-based stabilisation," and the
broadcast surfaces report the negative finding rather than omitting it.

*AMBIGUOUS.* P3 or P4 fails. Slate is invalidated. Sensor-model audit
(P3 failure path) or oracle audit (P4 failure path) lands before re-run.
No tier promotion until a clean CONFIRM or REFUTE.

**Disposition is locked.** The verdict is filed in the same directory as the
data: `results/balance/envelope_<datetime>/verdict.md`. The choice of verdict
is from the predictions, not author-discretionary. REFUTE means the
broadcast says REFUTE; the workbench does not get a "yes but" footnote
that softens the read.

***[/must-fix #1]***

### Phase 11 - Public Artifact, Gallery, And Promo GIF

Goal: make Sundog Balance promotion-ready without overclaiming.

Deliverables:

- Final `balance.html` tab linked from nav beside `threebody.html`.
- Short writeup titled around "Balancing By The Shadow."
- Applications gallery card with evidence tier set to "Planned Workbench" until
  Phase 7-10 artifacts exist, then ~~"Operating-Envelope Study" if the metrics
  earn it.~~ ***[must-fix #2]*** "Operating-Envelope Study" if and only if the
  Pre-Registered Verdict Template above returns a CONFIRM verdict. REFUTE
  keeps the card at "Planned Workbench" with a negative-finding banner.
  AMBIGUOUS holds the card unchanged until a re-run produces CONFIRM or
  REFUTE. ***[/must-fix #2]***
- `docs/APPLICATIONS.md` row only after the workbench has runnable artifacts
  ***[must-fix #2]*** *and* a CONFIRM verdict; on REFUTE, the row lands
  carrying the negative finding rather than being omitted ***[/must-fix #2]***.
- A short `.gif` or lightweight video of the arm/pole recovering from a
  disturbance. ***[must-fix #4]*** **The gif is decoration, not evidence.**
  The evidence is the operating-envelope CSV at
  `results/balance/envelope_<datetime>/envelope.csv`, the Phase 8 matched-seed
  metric tables, and the Phase 10 best-cell / worst-cell replay links. The
  gif's caption must link to one of those, not stand alone. ***[/must-fix #4]***
- Final index pass: place that motion clip after the hero on `index.html`,
  with a concise caption and a link to `balance.html`. ***[must-fix #4]*** The
  caption must include a phrase that points at the failure boundary, not just
  the success — e.g., "balance recovered inside the lighting envelope; fails
  cleanly outside it." A success-only caption violates the Avoid list. ***[/must-fix #4]***

Exit criterion: a first-time visitor can watch the hero, see the balance clip,
click into the workbench, and understand the theoremic move in under one
minute: the hidden body is controlled through the visible shadow ***[must-fix #4]*** —
*and* sees, in the same minute, that the shadow stops being enough at some
boundary, with a one-click path to the operating-envelope map ***[/must-fix #4]***.

+++[forward-plan-B]+++

## Post-Verdict / Conditional Roadmap

The phases below are pre-registered against the assumption that Phase 10
returns a CONFIRM verdict. They are *not* pre-emptive of that verdict. The
Phase 7 smoke result is encouraging signal that informs forward planning;
it does not move the verdict gate.

If Phase 10 returns REFUTE or AMBIGUOUS, this section is reshaped or
scrapped before the broadcast pass. Specifically: Phase 12 (UI feature) can
ship regardless of verdict because it does not extend any research claim.
Phase 13 and Phase 14 are gated on CONFIRM and do not run on a REFUTE
verdict — they would extend a claim the Phase 10 data has refuted.

This section is here so the *shape* of the broader-claim attack is
pre-registered with the same discipline as the Phase 10 verdict template,
rather than being authored after a CONFIRM verdict has changed what counts
as a load-bearing claim.

### Phase 12 - Human-vs-Agent Competition Mode

Goal: install an input layer so observers can drive the cart against the
Sundog controller on the same seed and disturbance schedule. The workbench's
value is partly in *being attackable by viewers*; Phase 12 ships the UI
that lets observers attack the claim themselves.

**Gating:** parallel with Phase 11 acceptable. Does not require CONFIRM
verdict because it is a UI feature, not a research claim.

Deliverables:

- Input map: WASD or arrow keys for desktop; on-screen left/right buttons
  for touch. Both bind to bounded cart force commands with **identical
  authority** to all controllers — no human-only force advantage.
- Side-by-side mode in `balance.html` rendering two cart-pole simulations
  on the same seed: one human-controlled, one Sundog-controlled. The
  disturbance schedule fires identically on both runs at the same
  simulated time.
- Telemetry overlay: who survived longer, who recovered faster after the
  impulse, hidden-angle integral comparison. The hidden-angle ghost
  appears on both runs only when "show privileged diagnostics" is enabled,
  matching Phase 6's privileged-audit discipline.
- Optional: local-storage leaderboard scoped to seed + lighting preset.
  No remote leaderboard — keeps the workbench inspectable without a
  backend dependency.

Exit criterion: a first-time visitor plays a head-to-head run against
Sundog within two clicks of `balance.html`, and reads who survived
longer without translating any units.

**Falsification angle.** If humans consistently outperform Sundog inside
the diagnostic-positive envelope, the workbench is teaching observers
about a controller deficiency. This is itself reportable. The
human-survival-margin distribution is filed as a Phase 8 metric category
(see *Avoid list updates* below for the misread that needs guarding
against — beating humans is not evidence of correctness).

### Phase 13 - Robotics-Implementable Controller

Goal: lift the SCAN/SEEK/TRACK controller from a sim-friendly form into a
form that respects hardware-realistic constraints. Demonstrates that the
indirect-signal-to-action discipline is not a sim artifact.

**Gating:** requires Phase 10 CONFIRM verdict. A controller built against
hardware-realistic constraints on a base claim that has been refuted is
extending a refuted claim.

Deliverables:

- **Bounded actuator latency.** Control commands take effect after a
  configurable delay (default 30 ms). The controller must reason about
  command-vs-effect rather than assuming instantaneous force.
- **Discrete sampling decoupling.** Sensor reads at a configurable fps
  (default 30); control loop runs at a configurable rate (default 100 Hz);
  the two are independent and the controller respects both.
- **Force quantization.** Force command rounded to a configurable
  resolution (default 0.05 N). Resists "infinitely smooth" sim
  assumptions that would not survive on real actuators.
- **Energy budget.** Cumulative `|force × velocity|` over a run is capped.
  The controller must spend force only when shadow residual exceeds its
  confidence band, not as continuous fine-tuning.
- **ROS-friendly state shape.** Serialise state into a
  `geometry_msgs/Pose2D` + `sensor_msgs/Image` analogue (or a documented
  JSON shape) so the controller can be lifted into a ROS node without
  re-architecting. The shape is documented; the ROS node itself is a
  Phase 13.5 follow-up.
- **Hardware-realistic disturbance.** Replace the Phase 8 impulse model
  with a wind-gust-style stochastic forcing whose frequency content
  matches real-world disturbances (low-frequency drift plus
  high-frequency jitter).

Exit criterion: SCAN/SEEK/TRACK survives the diagnostic-positive envelope
while respecting all six constraints above, on a slate of ≥ 50 seeds.
Below that floor, the workbench's "robotics implementable" claim is
suspended pending controller iteration.

**Phase 13.5 follow-up (separate evidence tier).** A physical cart-pole
arm with a webcam reading the shadow, running the same controller. The
boundary matters because hardware demos drift the claim from "shadow
control in sim under hardware-realistic constraints" to "shadow control
on hardware" — a separate evidence tier with its own audit surface
(camera calibration, motor friction, real-world lighting, on-hardware
seed equivalent). 13.5 is named here so it does not silently expand
Phase 13's scope.

### Phase 14 - Broader Claim: Sibling Plant Replication

Goal: test whether the SCAN/SEEK/TRACK shadow-control pattern transfers
to plants other than cart-pole, or whether the workbench's claim is
plant-specific.

**Gating:** requires Phase 10 CONFIRM verdict. Phase 13 should ideally
land first so the constraint set is consistent across plants. If Phase
13 is delayed, Phase 14 can run on the Phase 10 sim controller, but the
broader claim it earns is "shadow control transfers in sim across
plants" rather than "shadow control transfers under hardware-realistic
constraints across plants."

Sibling plants (initial slate of three):

- **Furuta pendulum.** Rotational arm with a hanging pole; the shadow
  projects from the rotational tip onto a horizontal floor. Different
  geometry than cart-pole; same observability question.
- **Reaction wheel pendulum.** Single-axis pendulum stabilised by a
  flywheel; the shadow projects from the pendulum body. Different
  actuation modality than cart-pole.
- **Acrobot or pendubot.** Under-actuated two-link arm. Hardest case;
  expected to either confirm the pattern's generality or expose a clean
  failure mode.

Per-plant deliverables:

- Plant-specific physics module sharing the renderer-independent
  constraint from Phase 1.
- Plant-specific shadow projection adapted from the directional-light
  model in Phase 2.
- Plant-specific Phase 3 diagnostic benchmark — does shadow forecast
  useful events for *this* plant?
- Plant-specific operating-envelope map and a per-plant verdict
  (CONFIRM / REFUTE / AMBIGUOUS) using the same template structure as
  the cart-pole verdict.

**Pre-registered prediction for the broader claim.** The failure-boundary
*shape* transfers across plants (overhead light collapses signal,
high delay destabilises recovery), but the operating envelope's *area*
differs by plant. A plant whose failure boundary does *not* transfer is
itself reportable — it identifies which dimensions of cart-pole were
doing the structural work for the original claim.

Exit criterion: at least one sibling plant returns either a CONFIRM or a
REFUTE verdict with the per-plant template. Not all sibling plants need
to CONFIRM. The pattern's robustness across plants is the read; the
broadcast uses whichever verdicts land, and an even split (some CONFIRM,
some REFUTE) is the most epistemically informative outcome — better than
a uniform CONFIRM that would understate the pattern's brittleness.

**Promotion to pattern-claim.** Success across at least two of three
sibling plants lifts the workbench from a one-plant claim into a
"shadow-control-for-balance" pattern claim. The pattern claim then
enters `docs/APPLICATIONS.md § Cross-Application Comparison` as a
*meta-row* describing the pattern itself, with the cart-pole and
sibling-plant rows as evidence under that meta-row. The Pre-Committed
Cross-Application Comparison Row below is updated at that point to
distinguish the per-plant rows from the pattern meta-row.

+++[/forward-plan-B]+++

## Claim Boundary

Safe claim after implementation and positive envelope results:

> In the tested cart-pole workbench, a controller denied the true pole angle can
> maintain and recover balance from shadow-derived projection signals inside a
> mapped lighting and delay envelope, while failing when the projection becomes
> unobservable or control authority saturates.

What this would strengthen:

- the theorem's observability framing;
- the public intuition that shadows can be sensors;
- the application portfolio's balance between famous physics, game systems,
  and instantly legible embodied control.

What it would not prove:

- general robot balance;
- hardware validity;
- learned inference;
- universal embodied alignment;
- superiority over a privileged controller.

*[nice-to-have #10]*

## Pre-Committed Cross-Application Comparison Row

When Phase 11 lands and Balance enters
`docs/APPLICATIONS.md § Cross-Application Comparison`, the row should read:

| Application | Domain | Indirect signal | Transformation | Actionable output |
| --- | --- | --- | --- | --- |
| Sundog Balance | Embodied control | Cast-shadow centroid, length, and velocity on the floor plane under controlled light geometry | SCAN/SEEK/TRACK shadow-residual control with cart-proprioceptive history, action-history coupling, and observability gating | Bounded cart force command maintaining upright pole inside a mapped lighting and delay envelope, failing cleanly outside it |

Pre-staging the row here means the writeup pass cannot drift the language
between the roadmap and the broadcast. CONFIRM verdict promotes the row from
"Forward-Looking Application Design" to a live row in the table; REFUTE
verdict keeps the row out of the table and lands the negative finding in a
sibling row of the Cross-Application Comparison's "Refuted Applications"
column (currently absent — Balance under REFUTE would be the first occupant
and would establish the pattern for the rest of the program).

*[/nice-to-have #10]*

## Suggested First Build Slice

The first implementation pass should be deliberately small:

1. Implement deterministic cart-pole dynamics in a shared JS module.
2. Build `balance.html` with one canvas and three controller buttons.
3. Add the shadow projection and true-angle diagnostic overlay.
4. Implement passive, naive shadow-centering, and privileged oracle.
5. Add the first Sundog shadow controller only after the shadow diagnostic shows
   useful coupling.

That slice is enough to decide whether the workbench has the clean, almost
annoyingly obvious hook we want before promotions.

*[nice-to-have #11]*

**Exit criterion for the first slice:** the shadow-projection diagnostic
overlay shows non-trivial coupling between cart force and shadow residual on
at least one lighting preset (low-elevation directional light is the
candidate). Below this floor — shadow residual is uncorrelated with cart
force across all presets — do not commit to the SCAN/SEEK/TRACK structure;
re-scope the workbench against an alternative indirect signal (e.g.,
shadow-edge contrast over time) before any Phase 5 controller work.

*[/nice-to-have #11]*
