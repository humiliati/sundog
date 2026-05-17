# Sundog Balance Workbench

Working hook:

> It balances the body by reading the shadow.

Sundog Balance is the final pre-promotion application workbench. It is
not another optics extension and not another large physics claim. It is a
classic control task made visually obvious: a cart-pole / inverted-pendulum
system must keep a pole upright while the Sundog controller is denied the pole
angle. It can read only the pole's cast shadow, cart proprioception, and its own
action history.

The public question is small enough to defend:

> Can an agent maintain upright balance from an indirect projection when the
> true alignment variable is hidden?

The workbench lives beside the three-body tab as `balance.html`: a public,
interactive surface with browser simulation, baselines, telemetry panels,
failure-boundary presets, and a matching writeup. The final promotional pass
should reserve the post-hero slot on `index.html` for a Balance-first motion
gallery rather than a one-off clip: cheap looping media from multiple
applications, each with a punchy title and description overlay.

## Sundog Expression *(canonical for the APPLICATIONS.md row)*

- **Hidden target:** upright pole angle (`theta`).
- **Indirect signal:** cast-shadow centroid, length, and velocity on the floor
  plane, sampled with bounded noise and delay.
- **Transformation:** SCAN/SEEK/TRACK shadow-residual control with
  cart-proprioceptive history and observability gating.
- **Actionable output:** bounded cart force command maintaining upright pole
  inside a mapped lighting and delay envelope, failing cleanly outside it.

This block is here so the cross-application comparison row in
`APPLICATIONS.md` and the gallery card's indirect-signal/transformation/output
triplet do not drift between the roadmap and the public writeup. Phase 11 keeps
the broadcast surfaces aligned to this claim shape.

## Current State

The executable workbench and verdict harness have landed. The Sundog tree now
includes:

- `balance.html`: the public browser workbench tab with a live Phase 8
  recovery curve, impulse/recovery status panel, and Phase 9 use-boundary
  warning panel;
- `public/js/balance-core.mjs`: deterministic cart-pole dynamics, shadow
  projection, sensor delay/noise/dropout scaffolding, shared boundary
  classifier, and baseline controller modes;
- `public/js/balance-browser.mjs`: canvas renderer, controls, telemetry strip,
  and Lounas-derived robotics balance motif;
- `scripts/balance-harness.mjs`: a Phase 7 smoke/replay runner sharing the
  same dynamics, shadow sensor, and controller modules as the browser page;
- `scripts/balance-phase8-metrics.mjs`: a Phase 8 recovery/event metrics runner
  for local metric tables, recovery curves, and fall-warning threshold sweeps.
- `scripts/balance-phase9-boundary.mjs`: a Phase 9 degradation sweep over light,
  delay, noise, dropped frames, rail length, force limit, and disturbance
  magnitude, writing local boundary summaries and unsafe-cell tables.
- `scripts/balance-phase10-envelope.mjs`: the Phase 10 operating-envelope slate
  and repaired verdict runner, writing local envelope, comparison, best/worst
  cell, and verdict artifacts.

No committed Balance results directory is expected: Phase 7-10 outputs are
written under ignored `results/balance/` paths so the public site does not ship
bulk slate data. The broadcastable result is the repaired Phase 10 CONFIRM
summary documented below and reproducible with `npm run balance:phase10`.

This section records the promotion gates that keep the broadcast surfaces
(gallery card, APPLICATIONS.md, claims-and-scope.md) tied to runnable artifacts
and a fixed verdict. After the repaired Phase 10 CONFIRM rerun, Balance sits at
**Operating-Envelope Study** tier in the application family. The Phase 11
broadcast pass (nav link, application-map copy, applications-gallery card,
APPLICATIONS.md row, claims-and-scope row) and the Phase 11.1 motion-rail
scaffold on `index.html` have both landed. Phase 11.2 filled the Balance rail
poster with a compressed best/worst Phase 10 replay composite at
`public/media/balance-phase10-rail-poster.jpg`; animated loop export remains a
later polish task, not a Phase 11 blocker.

Promotion to **Planned Workbench** tier required Phase 0–4 deliverables
landing — achieved 2026-05-08, with phases 5–6 and the Phase 7 smoke also
completed in the same pass. Promotion to **Operating-Envelope Study** tier
required the Pre-Registered Verdict Template below to return a CONFIRM
verdict — achieved by the repaired Phase 10 rerun.

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

Cross-document visual target: the evidence workbench stays a readable
cart-pole simulation first. The future expressive skin is governed by
[`UI_UX_THEME_FOUNDATION.md`](UI_UX_THEME_FOUNDATION.md) section 4c: the
"toyful" Paper Mario / 2.5D indie-game balance canvas. That visual pass may add
layered parallax paper planes, pixel-art cart and pole sprites, soft shadows,
motion lines, sparkles, squash/anticipation, and a warmer palette. It must not
change dynamics, sensor math, controller code, metrics, replay URLs, verdict
gates, or the ability to switch back to raw diagnostic rendering.

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

## Bayesian Floor Profile

Profile id: `balance-bayesian-floor-v1`

Status: staged profile, not yet an earned result.

Purpose: add a same-observation Bayesian floor for Balance so the shadow
controller is compared against the best partial-observation controller we can
reasonably afford, not only against passive, naive, and privileged-oracle
baselines. The floor is a claim-hygiene instrument: it can strengthen the
Balance row only after it passes the gates below, and it can also narrow the
claim if the same-shadow floor materially dominates SCAN/SEEK/TRACK inside the
confirmed Phase 10 envelope.

Truth state and hidden variables:

- `X_t = (x, x_dot, theta, theta_dot, t, c)`, where `c` records the preset,
  light elevation, rail and force limits, noise, delay, dropout, disturbance
  schedule, seed, and envelope cell.
- `theta`, `theta_dot`, pole energy, and terminal verdict labels remain hidden
  from the Bayesian-floor controller during a run.
- Truth-state logging is allowed for oracle comparison, metrics, fixtures, and
  post-run audits.
- The first population `mu` is the repaired Phase 10 envelope slate. Any
  capped probe must use the same cell and seed definitions, with fewer cells or
  seeds only when the runtime budget requires it.

Admitted observation:

```text
Phi_t = [
  shadow_endpoint_x,
  shadow_centroid_x,
  shadow_length,
  shadow_residual,
  shadow_confidence,
  finite_difference_shadow_velocity,
  cart_x,
  cart_x_dot,
  recent_force_commands,
  light_elevation_deg,
  sensor_noise,
  sensor_delay,
  sensor_dropout,
  dt,
  preset_id,
  envelope_cell
]
```

The floor may use the full history of prior `Phi_t` values and its own prior
actions. It may not read true pole angle, true angular velocity, pole energy,
oracle actions, or post-hoc verdict labels. The observation source should be
the same `sampleShadowSensor` path that feeds `computeBalanceControl`, with a
parity test proving that serialized floor observations match the
`sundog_shadow` controller input on the same replay.

Objective and regret:

```text
J(pi) = E_mu[survival_time / T_max]
regret_cell = (T_safe_bayes_floor - T_safe_sundog_shadow) / T_max
```

Recovery time, upright-angle margin, shadow confidence, and action effort are
diagnostics. Action effort is a tie-breaker only when survival performance is
otherwise tied. A floor that performs worse than the existing shadow controller
on easy observable cells is not evidence for the shadow controller; it is a
failed floor and must be repaired before claim language is promoted.

Floor policy:

- First implementation: a low-dimensional particle filter over `(theta,
  theta_dot)` using cart state as admitted proprioception.
- Likelihood: compare predicted shadow endpoint, centroid, length, residual,
  and confidence against `sampleShadowSensor` under the configured noise,
  delay, and dropout model.
- Candidate actions: include the live `sundog_shadow` action as candidate 0,
  plus a fixed bounded force lattice such as `[-F, -0.5F, 0, 0.5F, F]`.
- Planning horizon: modest, pre-locked after the capped probe, and scored by
  expected survival plus a bounded terminal-upright diagnostic. Do not tune the
  horizon per failed seed.
- Optional follow-up: an EKF floor may be added later if the particle floor is
  too slow, but the first public profile should prefer auditability over speed.

Required comparators:

- `sundog_shadow`
- `bayes_floor_shadow_particle`
- `oracle`
- `naive_shadow`
- `naive_cart`
- `passive`

Receipts should live under `results/balance/phase15-bayesian-floor/` and be
reduced into public data only after the gates pass:

- `manifest.json`
- `signature-observations.jsonl`
- `belief-diagnostics.csv`
- `bayes-actions.csv`
- `trial-outcomes.csv`
- `bayes-regret.csv`
- `bayes-regret-summary.csv`
- `observability-fibers.json`

Gates:

- unknown mode is rejected by the harness;
- no-state-leak audit proves `theta`, `theta_dot`, pole energy, oracle action,
  and verdict labels are unavailable to the floor at decision time;
- observation parity proves serialized `Phi_t` equals the shadow-controller
  observation on the same run;
- easy-cell sanity proves the floor can match or exceed naive baselines on
  diagnostic-positive observable cells;
- runtime probe records particles, horizon, cells, seeds, trials/sec, and the
  estimated full-slate wall clock before any full run is staged;
- claim gate blocks public language until the regret summary has been reduced
  and linked from the Balance data surface.

Outcome branches:

- If the floor fails the no-leak, parity, or easy-cell sanity gates, Phase 15 is
  invalid and earns no claim.
- If the floor materially dominates `sundog_shadow` inside confirmed Phase 10
  cells, keep the Phase 10 operating-envelope claim but narrow any language
  about near-optimality or recovered hidden-state structure.
- If `sundog_shadow` stays near the floor in diagnostic-positive cells and both
  fail in the predicted overhead/delay cells, the claim can strengthen to:
  *the hand-built shadow controller recovers most of the same-observation
  Bayesian floor inside the mapped envelope while failing at the same
  observability boundary.*
- If `sundog_shadow` appears to beat a weak floor, do not promote the result
  until the floor has passed an adversarial repair pass.

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
- "The toyful sprite skin is evidence."

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
- "The toyful sprite skin proves the theorem is intuitive." The Phase 12.5
  skin is a comprehension layer and promotion surface. It can make the
  recovery/failure story easier to read, but it cannot substitute for the
  Phase 8 tables, Phase 10 verdict, or replayable raw diagnostic view.

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
  *Projection model, pinned before implementation: directional light source
  treated as point-at-infinity for the default case (parallel rays); shadow
  endpoint computed by ray intersection from each pole endpoint along the
  light direction with the floor plane `z = 0`; soft shadow approximated as
  a penumbra width proportional to (source softness × pole height / source
  distance). The default sweep uses the directional model; the
  perspective-light model is Phase 9 sweep variant only.*
- Cast-shadow projection from pole endpoints to floor plane.
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

*Ground the SCAN/SEEK/TRACK pattern in canonical text before implementing.
The pattern is borrowed from the photometric mirror experiment; cite
`<sundog>/notebooks/sundog alignment theorem.txt` and
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
naive baselines on a small seeded slate inside the diagnostic-positive
envelope. Minimum required behaviour: the controller maintains
`|theta| < 0.3 rad` for ≥ 10 simulated seconds inside the Phase 3
diagnostic-positive envelope on at least 80% of the seeded slate. Below
that floor, the controller is not yet functional and Phase 6 web shell
work is gated behind further controller iteration. Above that floor,
Phase 5 is considered complete and Phase 6 unblocks.

### Phase 6 - Real-Time Web Projection

Goal: create the public browser workbench shell early.

*Pre-condition before any borrowed asset lands: confirm the Lounas credit
clause in `<sundog>/docs/THIRD_PARTY_REUSE.md` is current and matches the
LinkedIn agreement scope. Borrowing visual assets without that ledger
updated is the kind of audit-trail gap the program has committed to
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
large batch studies are complete. Hard exit: desktop **and** mobile
screenshot acceptance pass before the page links from any nav surface. The
workbench will be linked from `index.html` and visited on phones; mobile
readability is not optional and not gated on Three.js. The acceptance pass
lives in `results/balance/screenshots_<datetime>/` with one desktop and one
mobile capture per controller mode.

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

*Status — Phase 7 continuation landed:* the workbench now exposes seeded
replay URLs, the harness accepts `--replay-url`, and local harness runs
write `replay-index.json` entries with browser URLs plus replay commands.
The next Phase 7 hardening pass added replay verification, writing
`replay-verification.csv` and failing the command on outcome/time drift.
This is still replay infrastructure, not a verdict artifact.

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

*Status — Phase 8 first pass landed:* `npm run balance:phase8` writes
local ignored `trial-metrics.csv`, `event-log.csv`, `metric-summary.csv`,
`warning-thresholds.csv`, `matched-comparison.csv`, `recovery-curves.csv`,
and `samples.jsonl`. The browser workbench now also shows a live recovery
curve after the standard disturbance, including peak hidden angle, peak
time, recovery time, and terminal failure status. These are event/recovery
diagnostics, not Phase 10 operating-envelope evidence.

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

*Status — Phase 9 first pass landed:* `npm run balance:phase9` writes
ignored `manifest.json`, `boundary-panel.json`, `trial-outcomes.csv`,
`boundary-summary.csv`, `matched-comparison.csv`, and `unsafe-cells.csv`
under `results/balance/phase9-boundary/`. The browser workbench now shows
a live Use Boundary panel driven by the shared boundary classifier in
`public/js/balance-core.mjs`, with controls for every Phase 9 sweep axis:
light elevation, delay, noise, dropped frames, rail length, force limit,
and disturbance magnitude. This is a degradation diagnostic and
where-not-to-use surface, not the Phase 10 operating-envelope verdict.

### Phase 10 - Operating Envelope Map

Goal: lock the earned claim.

Deliverables:

- Grid map over light elevation, delay, noise, dropped-frame rate, rail
  length, force limit, and disturbance.
- Phase 10 grid axis values match the Phase 9 sweep values exactly. Cells are
  joinable after normalizing the Phase 9 column names to `(light_elev_deg,
  delay_ms, noise_sigma, dropout_rate, rail_limit, force_limit,
  disturbance_mag)` keys; verdict checks do not interpolate across mismatched
  bins.
- Candidate-envelope CSV (`envelope.csv`) where Sundog improves over passive
  and naive baselines while staying below force/saturation thresholds.
- Locked `envelope.csv` schema, in order: `cell_id`, `case_id`, `axis`,
  `axis_value`, `preset`, `light_elev_deg`, `delay_ms`, `delay_steps`,
  `noise_sigma`, `dropout_rate`, `rail_limit`, `force_limit`,
  `disturbance_mag`, `cell_class`, `static_boundary_mechanisms`,
  `survival_passive_mean`, `survival_sundog_mean`, `survival_naive_mean`,
  `survival_oracle_mean`, `rms_theta_sundog_mean`, `rms_theta_naive_mean`,
  `rms_theta_oracle_mean`, `recovery_time_after_impulse`,
  `sundog_naive_paired_margin_mean`, `sundog_naive_survival_ratio`,
  `sundog_beats_naive_1p5x`, `oracle_sundog_paired_margin_mean`,
  `seed_count`, `paired_margin_bootstrap_low`,
  `paired_margin_bootstrap_high`, `sundog_saturation_rate_mean`,
  `sundog_force_budget_mean`, `replay_url`, `naive_replay_url`, and
  `oracle_replay_url`.
- Best-cell and worst-cell replay links for the browser.
- Failure-mechanism labels such as `shadow_unobservable`, `delay_destabilized`,
  `force_saturated`, `rail_limited`, and `controller_overcorrected`.

Exit criterion: the public copy can say "inside this envelope" and point to the
map rather than hand-waving.

### Pre-Registered Verdict Template

Phase 10 produces an Operating Envelope Map and a candidate-envelope CSV. The
verdict template below pre-commits to *what those artifacts have to say* before
the workbench can promote tiers or broadcast. This is the same Money-Bags-style
discipline that ratifies Stage 1 verdicts before captures, restated for the
Balance workbench. Verdict is data-driven; disposition is locked.

**Cell classes and P1 denominator.**

`cell_class` is assigned before the Phase 10 slate verdict:
`diagnostic_positive` for Phase 9 boundary-classifier cells where the central
shadow-derived-control claim is alive, `failure_regime` for overhead,
high-delay, force/rail-limited, or otherwise intentionally unsafe cells, and
`borderline` for cells whose Phase 9 label is unstable across seeds or adjacent
axis bins. The P1 denominator is `diagnostic_positive` cells only. It is not
all grid cells, and it is not cells where any controller survives.

**Pre-registered predictions.**

*P1 - central effect.* Inside the Phase 9 `diagnostic_positive` cells, each
cell computes the Sundog-minus-naive survival margin from seed-paired trial
pairs. A cell counts as a yes-cell only if Sundog beats naive shadow-centering
by at least 1.5x on the matched-seed slate of at least 100 seeds. P1 holds
only if the aggregate yes-cell fraction over the `diagnostic_positive`
denominator has a bootstrap lower bound greater than 0.30. Failure
here REFUTES the workbench's central claim. `verdict.md` reports
`diagnostic_positive_cell_count`, `yes_cell_count`, `yes_cell_fraction`,
`yes_fraction_bootstrap_low`, and `yes_fraction_bootstrap_high` so the
aggregate read is mechanical.

*P2 - failure boundary realness.* P2 splits failure-regime behavior into a
gated hard check and a reported soft margin. P2a holds iff the hard violation
count is zero: no cell where Sundog reaches terminal success while naive fails
inside an overhead-light or high-delay `failure_regime` cell. Transient
recovery markers inside trials that later fall remain telemetry, not P2a
success. P2b reports all-fail survival margins where both controllers fail in
every seed pair but Sundog lasts longer than naive under confidence gating.
P2b is degradation behavior, not evidence of usable overhead-light control, and
does not contribute to CONFIRM, REFUTE, or AMBIGUOUS.

*P3 — recovery monotonicity.* Recovery time after the Phase 8 standard impulse
is monotonically worse as sensor delay increases across the slate. Non-monotonic
curves AMBIGUATE the slate (sensor model has a confound; light-geometry sweep
not orthogonalised against delay sweep).

*P4 - privileged-oracle ceiling.* P4 uses a dual-ceiling rule. P4a reads
uncapped cells: where at least one of oracle or Sundog falls below the duration
cap on at least one seed, oracle survival must exceed Sundog survival on at
least 80% of uncapped cells, or the uncapped-cell count must be zero. P4b reads
capped cells: where oracle and Sundog both reach the duration cap on every
seed, oracle must have lower hidden-angle RMS than Sundog on at least 80% of
capped cells, or the capped-cell count must be zero. P4 holds iff P4a and P4b
both hold. The audit response chooses this capped-cell quality rule over
lengthening the simulation duration.

**Verdict template.**

After the Phase 10 candidate-envelope CSV and Phase 8 metric tables land,
the writeup asserts *one* of three verdicts:

*CONFIRM.* P1 holds under the `diagnostic_positive` denominator and aggregate
yes-cell bootstrap rule above. P2a holds; P2b margins are reported only. P3
holds. P4 holds under the dual-ceiling rule. The workbench promotes to
Operating-Envelope Study tier.
The gallery card upgrades the indirect-signal/transformation/output triplet
from forward-looking language to past-tense achieved language. The
applications-gallery row, the APPLICATIONS.md row, and the
claims-and-scope.md row all land in the same pass.

*REFUTE.* P1 fails — Sundog does not beat naive shadow-centering by the
pre-registered margin — or P2a hard violation count is greater than zero. The
workbench stays at Planned Workbench tier with
a negative-finding banner: "Sundog Balance attempted shadow-derived
stabilisation of an inverted pendulum and did not beat naive shadow-centering
on the tested slate; the workbench remains as a Planned Workbench." The
negative finding is itself reportable — it is the same kind of evidence Money
Bags would
have produced if Stage 1 captures had returned an architecture-mode
kill-switch read. The Avoid list in this doc is amended to include
"Sundog Balance demonstrates shadow-based stabilisation," and the
broadcast surfaces report the negative finding rather than omitting it.

Pre-staged REFUTE hook language: "The hidden body was controlled through its
shadow only where the lighting geometry allowed it; on this slate, that was
not enough to beat naive shadow-centering. The failure boundary is the
finding."

*AMBIGUOUS.* P3 fails or P4 fails under the repaired dual-ceiling rule. Slate
is invalidated. Sensor-model audit (P3 failure path) or oracle/metric audit
(P4 failure path) lands before re-run. No tier promotion until a clean CONFIRM
or REFUTE.

**Disposition is locked.** The verdict is filed in the same directory as the
data: the default runner writes `results/balance/phase10-envelope/verdict.md`
and timestamped reruns may use `results/balance/envelope_<datetime>/verdict.md`.
The choice of verdict is from the predictions, not author-discretionary. REFUTE
means the broadcast says REFUTE; the workbench does not get a "yes but"
footnote that softens the read.

**Status timeline — Phase 10 verdict path.** Preserved inline because the
disposition discipline (initial AMBIGUOUS read, audit, pre-registered repairs,
repaired-rerun CONFIRM) is itself load-bearing for the workbench's claim.

*1. Initial Phase 10 run — AMBIGUOUS.* `npm run balance:phase10` writes
ignored `manifest.json`, `trial-outcomes.csv`, `matched-comparison.csv`,
`envelope.csv`, `cell-class-map.csv`, `best-worst-cells.csv`, `verdict.json`,
and `verdict.md` under `results/balance/phase10-envelope/`. The first 100-seed
slate emitted 27,200 trials over 68 operating-envelope cells and returned
AMBIGUOUS: P1 held (`28/28` diagnostic-positive yes-cells; lower bootstrap CI
`1`), P2 found two failure-regime violations that triggered baseline audit, P3
held, and P4 failed the survival-ceiling check (`16/68` cells) while the
auxiliary RMS audit showed oracle lower than Sundog on `68/68` cells. Not a
promotion result; next natural work was oracle/baseline audit before rerun.

*2. Phase 10.5 audit landed.* `npm run balance:phase10:audit` reads
`results/balance/phase10-envelope/` and writes ignored `manifest.json`,
`oracle-ceiling-audit.csv`, `p2-failure-audit.csv`, and `audit.md` under
`results/balance/phase10-audit/`. Finding: P4 failed because survival was
capped at the fixed `8s` duration. Oracle lower-RMS held on `68/68` cells, and
`50/68` cells had both oracle and Sundog at the duration cap, so the
survival-only ceiling masked oracle quality rather than exposing an oracle
sign bug. P2's two overhead-light violations were soft all-fail margins:
Sundog and naive both fail in every seed pair, with Sundog merely lasting
longer under confidence gating. High-delay failure cells held the intended
boundary shape. Phase 10 verdict remained AMBIGUOUS; no promotion or REFUTE
rewrite was allowed from the audit alone.

**Phase 10 rerun repair list.** Before rerunning the verdict slate, pre-register
two metric repairs. First, split P2 into hard success violations (Sundog
timeouts or recovers where naive fails inside a failure regime) and all-fail
survival margins (both controllers fail, but one lasts longer). The former
invalidates the boundary; the latter is reported as degradation behavior, not
as evidence of usable overhead-light control. Second, supplement P4's survival
ceiling with a capped-cell quality check: when both oracle and Sundog reach the
duration cap, oracle must have lower hidden-angle RMS or hidden-angle integral
on at least 80% of capped cells. Alternatively, lengthen or stress the slate
until oracle and Sundog survival separate without a secondary quality metric.

*3. Audit response and pre-registered repairs.* The repaired P2 and P4
criteria above are formally pre-registered in
[`results/balance/phase10-audit/audit-response.md`](../results/balance/phase10-audit/audit-response.md),
the sister document to `audit.md`. The response document locks the
dual-ceiling P4 rule (capped-cell quality check chosen over duration
extension), splits P2 into gated P2a hard violations and reported-only P2b
all-fail margins, names what the rerun must NOT change (no baseline re-tune,
no oracle re-tune, no P1 denominator drift, no third repair authored after
the audit), and pre-registers the implied rerun verdict of CONFIRM given the
deterministic slate. The roadmap-side Pre-Registered Verdict Template above
mirrors the repaired criteria, so the canonical pre-registration lives here
rather than only in the sibling.

*4. Repaired Phase 10 rerun — CONFIRM.* `npm run balance:phase10` regenerated
`results/balance/phase10-envelope/verdict.md` under the repaired criteria and
returned CONFIRM. Mechanical read: P1 `28/28` diagnostic-positive yes-cells
with bootstrap lower CI `1`; P2a hard violations `0`; P2b all-fail
survival-margin cells `2` reported only; P3 monotonic recovery holds; P4a
oracle-survival ceiling holds on uncapped cells (`16/18`, `0.888889`); P4b
oracle lower-RMS ceiling holds on capped cells (`50/50`, `1`). This is the
Phase 10 broadcast input; P2b overhead-light margins ship only as degradation
behavior, not usable overhead-light control.

### Phase 11 - Public Artifact, Gallery, And Motion Rail

Goal: make Sundog Balance promotion-ready without overclaiming.

Deliverables:

- Final `balance.html` tab linked from nav beside `threebody.html`. *(landed)*
- Short writeup titled around "Balancing By The Shadow." *(landed as bounded
  Balance page and application-map copy)*
- Applications gallery card with evidence tier set to "Operating-Envelope
  Study" after the repaired Phase 10 CONFIRM verdict. *(landed)*
- `docs/APPLICATIONS.md` row carrying the Balance bounded claim after runnable
  artifacts and a CONFIRM verdict. On REFUTE, this row would have carried the
  negative finding rather than being omitted. *(landed after CONFIRM)*
- A post-hero application motion rail on `index.html`, initially seeded with
  Balance recovery/failure media and later expanded to short loops from every
  public application. Use the cheapest acceptable format after size testing
  (`.webm`, `.ogv`/`.ogg`, MP4, or `.gif` only if it wins on compatibility and
  cost). **The media rail is decoration and orientation, not evidence.** The
  evidence is the operating-envelope CSV at
  `results/balance/envelope_<datetime>/envelope.csv`, the Phase 8 matched-seed
  metric tables, and the Phase 10 best-cell / worst-cell replay links. The
  first Balance card's caption must link to one of those, not stand alone.
- Motion-rail interaction model: start on Balance, let the clip play once, then
  ease/peek or scroll to the next application card in the manner of a Netflix
  rail. Users can swipe, drag, use arrow buttons, or keyboard-step the rail.
  Each application card gets a short title plus a description overlay that
  reads like a punchy Steam-homepage tag over a game clip, not like a long
  documentation paragraph.
- Locked Phase 11.1 card contract: each rail item carries `title`,
  `description`, `href`, `evidenceHref`, `poster`, `media`, `mediaFormat`, and
  `status`. The current index implementation may leave `poster`, `media`, and
  `mediaFormat` empty while rendering CSS/poster placeholders, but it must keep
  those slots present so later `.webm`/MP4/`.ogv`/`.gif` exports do not require
  a markup rewrite.
- Cross-reference
  [`UI_UX_THEME_FOUNDATION.md`](UI_UX_THEME_FOUNDATION.md) section 4c before
  recording the Balance media and section 4d before implementing the rail. If
  Phase 12.5 has landed, the Balance card should use the toyful sprite skin
  while linking to raw replay evidence. If the skin has not landed, the Balance
  card remains diagnostic and the caption says so plainly.
- Final index pass: place the motion rail after the hero on `index.html`, with
  a concise Balance opening card and a link to `balance.html`. The Balance card
  caption must include a phrase that points at the failure boundary, not just
  the success — e.g., "balance recovered inside the lighting envelope; fails
  cleanly outside it." A success-only caption violates the Avoid list.

Exit criterion: a first-time visitor can watch the hero, see the Balance-first
motion rail, click into the workbench, and understand the theoremic move in
under one minute: the hidden body is controlled through the visible shadow —
*and* sees, in the same minute, that the shadow stops being enough at some
boundary, with a one-click path to the operating-envelope map. The same rail
also makes room for later application clips without giving any one clip
evidentiary weight.

*Status — Phase 11 first broadcast pass landed (after Phase 10 CONFIRM):*
`balance.html` is linked from the public nav, its page copy now names the
Operating-Envelope Study tier and P2b degradation caveat, the applications
gallery and landing-page application grid include a Balance card, and
`docs/APPLICATIONS.md`, `docs/README.md`, `README.md`, and
`docs/presentation/claims-and-scope.md` carry the bounded Balance claim. The
Phase 11 visual work then moved to the Balance-first post-hero motion rail,
with slots for later application clips while keeping the Balance caption tied
to both recovery and the failure boundary.

*Status — Phase 11.1 motion-rail scaffold landed:* `index.html` now has a
post-hero application rail seeded with Balance, Three-Body, Photometric
Alignment, Pressure Mines, EyesOnly, Dungeon Gleaner, and Money Bags. Each
card carries the locked `title` / `description` / `href` / `evidenceHref` /
`poster` / `media` / `mediaFormat` / `status` contract, with empty media
fields allowed until capture/export work begins. The rail is static
CSS/poster art for now, with arrow, swipe, scroll, and keyboard focus
affordances; no gif/video assets were added in this slice.

*Status - Phase 11.2 poster capture landed:* the Balance opening card now uses
`public/media/balance-phase10-rail-poster.jpg`, a 77 KB composite captured from
the repaired Phase 10 best-cell and worst-cell replay URLs. The poster is
orientation media, not evidence; the evidentiary link remains the Phase 10
verdict and replay artifacts. `data-media` remains empty for a later loop export
without changing the rail contract.

## Post-Verdict Roadmap

The phases below were pre-registered against the assumption that Phase 10 would
return a CONFIRM verdict, and the repaired Phase 10 rerun has now returned
CONFIRM, so this section is the active post-verdict roadmap.

The pre-registration is preserved: had Phase 10 returned REFUTE or AMBIGUOUS,
Phase 12 (UI feature) could still have shipped because it does not extend any
research claim, Phase 12.5 (visual skin) could also have shipped as long as it
preserved the negative-finding and raw-diagnostic paths, and Phase 13 and
Phase 14 were gated on CONFIRM and are now eligible to run.

The *shape* of the broader-claim attack is pre-registered here with the same
discipline as the Phase 10 verdict template, rather than being authored after
a CONFIRM verdict has changed what counts as a load-bearing claim.

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
- Toyful interaction affordances, shared with Phase 12.5: the human and
  Sundog lanes use the same animated state vocabulary (idle, push-left,
  push-right, recover, fall). The sprites may make input and recovery timing
  legible, but neither lane gets visual treatment that implies hidden
  controller advantage.
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

### Phase 12.5 - Toyful Animated Sprite Skin

Goal: implement the section-4c visual promise from
[`UI_UX_THEME_FOUNDATION.md`](UI_UX_THEME_FOUNDATION.md) without changing the
research surface. This is the last-layer polish pass that makes the workbench
feel like a living toy while keeping the theoremic read intact: hidden body,
visible shadow, recoverable envelope, clean failure boundary.

**Gating:** can land after Phase 11 or alongside Phase 12. Does not require
CONFIRM verdict because it is a visualization layer, not a research claim.
On REFUTE, the same skin may still ship if the negative-finding banner,
failure boundary, and raw diagnostic replay toggle remain first-class.

Deliverables:

- Sprite atlas or CSS/Canvas sprite templates for cart, pole, floor shadow,
  wall projection, impulse marker, recovery sparkle, and failure pose.
- State machine shared across diagnostic, human-vs-agent, and promo contexts:
  idle, push-left, push-right, wobble, recover, stabilize, fall.
- Layered paper/parallax background inspired by the approved Lounas reuse
  direction, documented in [`THIRD_PARTY_REUSE.md`](THIRD_PARTY_REUSE.md)
  when concrete borrowed motifs land.
- `prefers-reduced-motion` fallback and a raw diagnostic skin toggle. Reduced
  motion preserves state changes through color, pose, and line weight rather
  than continuous animation.
- Replay parity check: the same Phase 8 replay URL renders in diagnostic and
  toyful skins with identical metrics, controller state, survival time, and
  verdict-relevant telemetry.
- Promo export recipe: best-cell recovery and worst-cell failure clips are
  captured from replay URLs, not hand-driven sessions, so the Balance media card
  remains tied to audited runs even when it sits inside the broader application
  motion rail.

Exit criterion: a first-time visitor can identify push, wobble, recovery, and
fall states from the animation alone, then toggle raw diagnostics and see the
same run with the same metrics.

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

### Phase 15 - Bayesian Floor Profile And Same-Shadow Baseline

Goal: turn the staged Bayesian Floor Profile into an executable, reusable
Balance baseline that uses the same shadow observations as SCAN/SEEK/TRACK.
This is the highest-value next research add because it tests whether the
confirmed Balance envelope is merely beating weak baselines or is recovering
most of what a same-information Bayesian controller can extract.

**Gating:** Phase 10 CONFIRM is sufficient to start. Phase 13 and Phase 14 are
not prerequisites because the floor is a claim-hygiene layer on the existing
cart-pole workbench. If Phase 13 lands first, run the floor against both the
Phase 10 sim controller and the hardware-realistic constrained variant, but do
not mix those tiers in one summary table.

Deliverables:

- `scripts/balance-bayes-floor.mjs`, sharing Balance core dynamics and the
  existing `sampleShadowSensor` observation path.
- A mode registry entry for `bayes_floor_shadow_particle` that rejects unknown
  modes and cannot silently fall back to `sundog_shadow`.
- Observation-parity and no-state-leak tests proving the floor receives only
  the admitted `Phi_t` profile.
- A capped runtime probe that records particles, horizon, cells, seeds,
  trials/sec, and estimated full-slate wall clock. If the full slate exceeds
  the repo's inline runtime rule, stage the exact PowerShell commands instead
  of running it in-session.
- A regret reducer writing `bayes-regret.csv` and
  `bayes-regret-summary.csv` under
  `results/balance/phase15-bayesian-floor/`.
- A short negative-result branch documenting what changes if the floor
  dominates, ties, or fails the shadow controller.

Public data products, only after the gates pass:

- `public/data/balance-bayesian-floor-profile.json`
- `public/data/balance-bayesian-floor-summary.json`
- `public/data/balance-observability-fibers.json`

Exit criterion: a complete regret summary across the repaired Phase 10 cell
slate, or a documented runtime-gated staged-command package with enough capped
measurements to estimate the full run. The public claim is promoted only if the
floor itself passes sanity and no-leak gates.

Initial implementation slice (2026-05-17):

- `public/js/balance-core.mjs` now declares a shared Balance controller registry
  and adds `bayes_floor_shadow_particle` as a runnable same-shadow,
  particle-belief baseline. The mode keeps a posterior over `(theta,
  thetaDot)` from legal shadow observations and cart proprioception. The first
  repair adds an analytic inverse from shadow residual to angle proposal, then
  uses a same-information guard: the floor defaults to the live
  `sundog_shadow` candidate unless the posterior proposal clears a predicted
  advantage threshold. This is now a guarded floor scaffold, not yet a
  claim-ready floor.
- `scripts/balance-phase15-bayes-floor.mjs` writes the Phase 15 receipt shape:
  `manifest.json`, `profile.json`, `signature-observations.jsonl`,
  `observation-parity.jsonl`, `belief-diagnostics.csv`, `bayes-actions.csv`,
  `trial-outcomes.csv`, `bayes-regret.csv`, `bayes-regret-summary.csv`, and
  `observability-fibers.json`.
- `npm run balance:phase15:smoke` is the capped probe. The first repaired smoke
  ran 128 trials in 21.443 s (5.97 trials/s) with 61 particles and a 0.05 s
  horizon. Observation parity, no-state-leak, and unknown-mode rejection passed.
- Smoke interpretation after the analytic-inverse / guard repair: the baseline
  passes the easy-cell sanity check versus `naive_shadow`, exactly matches
  `sundog_shadow` in the two readable 28-degree smoke cells, and improves over
  Sundog in the two 84-degree overhead-light smoke cells. This still does not
  promote claim language; it marks Phase 15 as executable with a sane
  same-information guard.
- Phase 10 cell-slate loader (2026-05-17): the Phase 15 harness now accepts
  `--cell-slate phase10-output` to read Phase 10 `envelope.csv` or
  `cell-class-map.csv`, preserving per-cell preset, axis, light, delay, noise,
  dropout, force limit, rail limit, disturbance magnitude, cell class, and
  static-boundary mechanisms. It also accepts `--cell-slate phase10-default`
  when receipts are not present locally.
- `npm run balance:phase15:phase10-slate:smoke` is the capped loader check. It
  read four cells from `results/balance/phase10-smoke`, ran 32 trials in 3.889 s
  (8.23 trials/s), and passed observation parity, no-state-leak, and unknown
  mode rejection. The loaded cells all passed Bayes-vs-naive sanity but still
  showed negative regret versus `sundog_shadow`, so this remains an executable
  floor scaffold rather than a promoted claim.
- Pre-repair stratified Phase 10 loader probe (2026-05-17): the operator ran
  `phase15-phase10-stratified-probe` against 12 loaded Phase 10 cells, 4 seeds,
  and the four-mode slate (`naive_shadow`, `sundog_shadow`,
  `bayes_floor_shadow_particle`, `oracle`). It completed 192 trials in
  21.166 s (9.07 trials/s), with observation parity, no-state-leak, and
  unknown-mode rejection all passing. All 12 cells passed Bayes-vs-naive
  sanity, but every cell still showed negative mean regret versus
  `sundog_shadow`: mean -0.687 normalized survival, range -0.782 to -0.350,
  mean negative-regret rate 0.958. Interpretation: the floor is now measurable
  across loaded Phase 10 cells, but it remains a weak-floor artifact rather than
  a same-floor claim.
- Post-repair stratified Phase 10 loader probe (2026-05-17): rerunning the same
  12-cell / 4-seed probe after the analytic-inverse proposal and
  same-information guard completed 192 trials in 26.08 s (7.36 trials/s), with
  all three audits passing. All 12 cells passed Bayes-vs-naive sanity. Mean
  regret versus `sundog_shadow` was +0.0336 normalized survival, range 0 to
  +0.403; mean negative-regret rate was 0. The candidate-selection diagnostic
  shows the guard doing most of the work (`sundog_guard`: 1531 logged rows,
  `bayes_proposal`: 5 logged rows). Interpretation: the floor is no longer
  weaker than the same-information controller on this stratified probe, but it
  is still mostly a guarded parity floor rather than evidence that explicit
  belief dominates Balance.
- Larger capped loader probe (2026-05-17): `phase15-phase10-stratified-probe-24`
  expanded the loaded Phase 10 smoke slate to 24 cells and 4 seeds. It ran
  384 trials in 50.515 s (7.6 trials/s), with observation parity,
  no-state-leak, and unknown-mode rejection still passing. Aggregate mean
  regret versus `sundog_shadow` was +0.0165 normalized survival, but one cell
  was negative and failed Bayes-vs-naive sanity: recoverable
  `sensor_noise__0p03` had mean regret -0.0112 and sanity false. The diagnostic
  remained mostly guarded (`sundog_guard`: 2636 rows, `bayes_proposal`: 8).
- Sensor-noise admission probe (2026-05-17): a focused 10-cell sensor-noise
  slate with 8 seeds ran 320 trials in 22.57 s (14.18 trials/s), with all three
  audits passing. It confirmed the block is on the sensor-noise axis rather than
  a cell-ordering fluke: 6/10 cells passed sanity, two cells had negative mean
  regret versus `sundog_shadow` (recoverable `noise_0p03`: -0.0383;
  recoverable `noise_0p08`: -0.0008), and the high-noise failure-regime cells
  need an explicit admission rule before they can be treated as claim gates.
- Full Phase 10-equivalent estimate: 68 loaded Phase 10 cells x 100 seeds x 4
  modes = 27,200 trials. At the post-repair measured 7.36 trials/s, the full
  run is about 62 minutes, which exceeds the repo's inline runtime rule. Do not
  run it as a claim lock until the sensor-noise admission/guard issue above is
  resolved; the command below is staged only as a diagnostic lock:

```powershell
node scripts/balance-phase15-bayes-floor.mjs --phase phase15-phase10-full-lock --out results/balance/phase15-phase10-full-lock --cell-slate phase10-output --phase10-out results/balance/phase10-envelope --limit-cells all --modes naive_shadow,sundog_shadow,bayes_floor_shadow_particle,oracle --seeds 100 --duration 8 --particle-count 61 --horizon-seconds 0.05
```

- Next implementation target: repair the sensor-noise admission path before the
  full claim lock. The minimum fix is to pre-register whether Phase 10
  failure-regime noise cells are reported-only or hard Bayes-sanity gates, then
  either tighten the noisy-cell guard or tune the posterior proposal until the
  recoverable `noise_0p03` borderline cell is non-negative versus
  `sundog_shadow` on a capped rerun.

### Phase 16 - Balance Data Surfaces And Claim Ratchet

Goal: convert the Balance evidence into richer public surfaces so the site can
show not just that Balance confirmed, but where, why, against what baselines,
and under which claim boundary.

**Gating:** Phase 10 CONFIRM is already enough for the first data-surface pass.
Bayesian-floor fields remain hidden or marked `pending` until Phase 15 earns
receipts. Phase 13 and Phase 14 add optional columns rather than blocking the
core Balance surface.

Deliverables:

- A public Balance data bundle that reduces Phase 10, Phase 11, and Phase 15
  receipts into cell-level JSON: preset, light geometry, delay/noise/dropout,
  controller, survival, verdict, boundary reason, and artifact links.
- A claim-card data shape with explicit tiers: current Phase 10 claim, optional
  same-shadow Bayesian-floor claim, optional hardware-realistic sim claim, and
  optional sibling-plant pattern claim.
- A `balance.html` evidence panel that can render the operating-envelope map,
  best/worst replay selectors, raw artifact links, and a Bayesian-floor regret
  strip when Phase 15 data exists.
- A compact `docs/APPLICATIONS.md` refresh that links the richer Balance data
  instead of relying only on prose.
- A gallery/poster refresh that can use the richer claim-card data without
  expanding the claim language by hand.

Claim ladder:

- Baseline live claim: Balance maintains and recovers from shadow-derived
  projection signals inside a mapped lighting and delay envelope, and fails
  cleanly outside it.
- If Phase 15 passes and `sundog_shadow` tracks the same-shadow floor: Balance
  recovers most of the actionable information available from the admitted
  shadow observation inside the mapped envelope.
- If Phase 15 shows the floor dominates: Balance remains a confirmed
  operating-envelope workbench, but the site must avoid any near-optimality or
  floor-adjacent language.
- If Phase 13 passes: add "hardware-realistic constraints in simulation" as a
  separate tier.
- If Phase 14 passes on sibling plants: add the pattern claim as a meta-row,
  not as a replacement for the cart-pole row.

Exit criterion: a public Balance evidence surface where each visible claim is
backed by a machine-readable receipt path, and missing future tiers are visibly
absent rather than implied.

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

## Pre-Committed Cross-Application Comparison Row

Phase 11 has landed and Balance now appears in
`docs/APPLICATIONS.md` under the cross-application comparison. The row should
continue to read:

| Application | Domain | Indirect signal | Transformation | Actionable output |
| --- | --- | --- | --- | --- |
| Sundog Balance | Embodied control | Cast-shadow centroid, length, and velocity on the floor plane under controlled light geometry | SCAN/SEEK/TRACK shadow-residual control with cart-proprioceptive history, action-history coupling, and observability gating | Bounded cart force command maintaining upright pole inside a mapped lighting and delay envelope, failing cleanly outside it |

Keeping the row here means later writeup passes cannot drift the language
between the roadmap and the broadcast. The repaired Phase 10 CONFIRM verdict
promoted the row from forward-looking design to a live row in the table; a
future contradictory rerun would require an explicit negative-finding update
rather than quiet omission.

## Initial Build Slice

The first implementation pass was deliberately small:

1. Implement deterministic cart-pole dynamics in a shared JS module.
2. Build `balance.html` with one canvas and three controller buttons.
3. Add the shadow projection and true-angle diagnostic overlay.
4. Implement passive, naive shadow-centering, and privileged oracle.
5. Add the first Sundog shadow controller only after the shadow diagnostic shows
   useful coupling.

That slice is enough to decide whether the workbench has the clean, almost
annoyingly obvious hook we want before promotions.

**Exit criterion for the first slice:** the shadow-projection diagnostic
overlay shows non-trivial coupling between cart force and shadow residual on
at least one lighting preset (low-elevation directional light is the
candidate). Below this floor — shadow residual is uncorrelated with cart
force across all presets — do not commit to the SCAN/SEEK/TRACK structure;
re-scope the workbench against an alternative indirect signal (e.g.,
shadow-edge contrast over time) before any Phase 5 controller work.
