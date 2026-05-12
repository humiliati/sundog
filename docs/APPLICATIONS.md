# Sundog Application Map

This document links the Sundog research repo to product systems that express
the same alignment program in practical software.

The root scientific claim still belongs to this repository: the current
controlled result is the photometric mirror-alignment experiment. The
applications below are not substitutes for that result. They are evidence that
the same pattern can be turned into usable systems and test surfaces across
different domains: operating-envelope workbenches, procedural agents,
physical-feeling game simulation, and softbody telemetry.

## Shared Pattern

Sundog applications tend to have the same shape:

1. A system cannot, should not, or does not need to observe the whole truth.
2. It receives an indirect signal: shadow, intensity, occluded game state,
   graph deformation, contact, torque, flow, or player-visible disturbance.
3. That signal is transformed into a lower-cost, control-relevant form.
4. The system acts from the transformed signal rather than from full
   world-state symmetry.

The older theorem language named this family of transformations with
`H(x) = dS/dtau`: a halo signature, or a relationship between environmental
projection and applied force. In the application repos, the literal symbols
may not appear. The relevant question is whether indirect structure becomes
actionable information.

## Evidence Tiers

Use these tiers when discussing application evidence:

| Tier | Meaning | Current examples |
| --- | --- | --- |
| Research result | Controlled task, metrics, baselines, reproducible artifacts. | Photometric mirror alignment in this repo. |
| Operating-envelope study | Experiment workbench with bounded sweeps, baselines, and mapped failure regions, but not a global domain solution. | Three-body dynamics Phase 11; Sundog Balance Phase 10; Pressure Mines Phase 11. |
| Instrumented prototype | Product system with telemetry and repeatable harnesses, but not yet a paper-style study. | Money Bags playtest bundles, Sundog Gone Rogue runner. |
| Product expression | Player-facing mechanic or agent system that embodies the idea, but needs formal measurement. | Dungeon Gleaner verb-field NPC behavior. |
| Conceptual lineage | Historical or design-language connection without enough evidence yet. | Older theorem docs, broad broadcast language. |

The public-facing story can mention all five. Academic writing should keep
them separate.

## Three-Body Dynamics Workbench

Local page: [`threebody.html`](../threebody.html)

Phase 11 summary:
[`docs/THREEBODY_PHASE11_SUMMARY.md`](THREEBODY_PHASE11_SUMMARY.md)

### Sundog Expression

The three-body workbench asks whether a controller can act on an indirect,
sensor-motivated instability signature rather than full privileged state. It
keeps local proxy signals, privileged diagnostics, and evaluation metrics
separate so the claim can stay audit-friendly.

The useful current claim is bounded:

> In the tested planar restricted setup, the guarded accelerometer-proxy TRACK
> controller improves survival over passive and naive local baselines in a
> robust high-velocity near-escape pocket. The result is not global: lower
> velocity and equal-mass boundary cells still expose controller harms, mostly
> through controller-shortened passive survival and control effort/saturation.

### Evidence Shape

Phase 11 is not a claim that Sundog solves arbitrary three-body control,
long-term chaos prediction, or physical sensor validation. It is an
operating-envelope result:

- guard-quantile sweeps show the positive pocket survives `0.5`, `0.75`, and
  `0.9` gates, with quantile controlling the survival/effort trade;
- outside-pocket expansion maps the boundary: velocity is the clearest axis,
  and equal-mass cells carry the negative best cases;
- matched comparison shows `track_sensor_accel_guarded` positive in all 81
  high-velocity near-escape comparison rows, while the naive local baseline has
  no candidate rows in that slate.

This makes the tab experiment a public workbench surface: it can demonstrate
how Sundog-style proxy control is tested, where it works, and where it fails,
without borrowing certainty from the photometric research result.

## Sundog Balance Workbench

Local page: [`balance.html`](../balance.html)

Phase 10 verdict artifact:
`results/balance/phase10-envelope/verdict.md` (ignored local result; summarized
in the roadmap for public docs)

Roadmap and audit trail:
[`docs/sundog_v_balance.md`](sundog_v_balance.md)

### Sundog Expression

The Balance workbench asks whether a cart-pole controller can stabilize a
hidden pole angle from the visible cast shadow rather than privileged body
state. It keeps the shadow controller, naive shadow-centering baseline, passive
baseline, and privileged oracle separate so success and failure boundaries are
inspectable.

The useful current claim is bounded:

> In the tested browser cart-pole setup, the Sundog shadow controller beats
> naive shadow-centering inside the diagnostic-positive operating envelope.
> The result is not global: overhead-light and high-delay cells remain
> degradation boundaries, and two overhead-light all-fail margins are reported
> only as degradation behavior, not as usable overhead-light control.

### Evidence Shape

Phase 10 is an operating-envelope result:

- P1 held on `28/28` diagnostic-positive cells, with the yes-cell fraction
  bootstrap lower bound at `1`;
- P2a had `0` hard failure-boundary violations; P2b reported `2` all-fail
  overhead-light survival margins where both Sundog and naive failed;
- P3's recovery curve was monotonic in delay across the slate;
- P4 passed the repaired dual-ceiling oracle check: oracle survival exceeded
  Sundog on `16/18` uncapped cells, and oracle hidden-angle RMS was lower than
  Sundog on `50/50` capped cells.

This makes Balance a second public operating-envelope workbench: it shows the
theoremic move in an embodied-control toy, while preserving where the shadow
signal stops being enough.

## Sundog Pressure Mines Workbench

Local page: [`mines.html`](../mines.html)

Roadmap: [`docs/sundog_v_minesweeper.md`](sundog_v_minesweeper.md)

### Sundog Expression

The Pressure Mines workbench asks whether an agent can make safer progress when
mine locations are hidden and nearby tiles offer only a noisy, lossy pressure
field instead of exact adjacency counts. It keeps the pressure-field sensor,
naive local-pressure heuristic, passive baseline, and privileged oracle
separate so success and failure boundaries are inspectable.

The promoted claim is narrow:

> In some minefield regimes, a lossy pressure field preserves enough
> control-relevant structure for bounded reveal/flag decisions to outperform
> naive local heuristics, while explicit noise, delay, and ambiguity boundaries
> mark where that structure stops being usable.

### Evidence Shape

**Evidence tier:** Operating-Envelope Study

Phase 10 confirms one narrow publishable pocket: density 0.16 / pressure noise
2.0 / dropout 0.2. In that cell, `sundog_minimal` beat `naive_pressure` by
+7.21875 budget-adjusted safe tiles with 95% bootstrap CI [3.375, 11.078516],
and `sundog_lean` also passed at +6.3125 with CI [1.921094, 11.25].

The publication pair is intentionally two-sided: `sundog_lean` loses -4.71875
budget-adjusted safe tiles versus `naive_pressure` at density 0.22 / pressure
noise 1.0 / dropout 0.35, CI [-8.375781, -1.496484]. In the best cell, both
Sundog and naive still trigger mines on every seed; the confirmed result is
safe-tile progress before failure, not field clearance.

### Sundog Signal Separation

- **Hidden target:** local mine occupancy and downstream board risk.
- **Indirect signal:** noisy pressure values, local field gradients, and bounded
  scan returns rather than exact adjacency counts.
- **Transformation:** confidence-gated pressure ordering with conservative
  threshold flagging. The original scan, gradient, and action-history channels
  remain visible as lineage, but Phase 5-10 evidence marks them as harmful in
  this controller family.
- **Actionable output:** bounded reveal / flag decisions that improve
  budget-adjusted safe-tile progress over naive pressure in the named pocket,
  failing cleanly outside it.

### Claim Boundary

The workbench should avoid broader formulations. It does not claim that:

- "Sundog solves Minesweeper."
- "The pressure field reveals the hidden board in general."
- "Indirect inference beats logical deduction."
- "This is better game AI."

The claim is about bounded usefulness inside a named operating envelope, not
global mine inference.

## EyesOnly / Gone Rogue

Repository: [humiliati/EyesOnly](https://github.com/humiliati/EyesOnly)

Local sibling path on the maintainer machine:
`C:\Users\hughe\Dev\EyesOnly`

Detailed scaffold:
[`docs/applications/EYES_ONLY_DETAILED.md`](applications/EYES_ONLY_DETAILED.md)

### Sundog Expression

The load-bearing Sundog application in EyesOnly is the headless Gone Rogue
turn-envelope runner. Gone Rogue is a procedural roguelike embedded in the
larger spy-game platform. The Sundog runner in this repo interacts with the
real JavaScript engine through Playwright and the `GoneRogue.headless` API.

### Three Surfaces

EyesOnly / Gone Rogue contains three Sundog-adjacent surfaces at different
evidence tiers. The card-level tier reflects the strongest one; the other two
are named explicitly so they do not borrow credibility from the runner.

| Surface | Where | Tier | What it is |
| --- | --- | --- | --- |
| Headless turn-envelope runner | `<sundog>/runners/` and `<EyesOnly>/public/js/gone-rogue.js` (`GoneRogue.headless`) | Instrumented prototype | A Sundog `PERCEIVE / PLAN / EXECUTE_BATCH` runner driving the real Gone Rogue JavaScript engine through Playwright. |
| UI-bound playtest agent | `<EyesOnly>/public/js/playtest-agent.js` | Product expression: automation neighbor, not Sundog evidence | A regression bot for UX edge cases under human UI constraints. Useful automation, but not an indirect-signal-to-action policy. |
| Live Agentic Game Moderation (LAGM) | `<EyesOnly>/docs/MANIPULATION_LAYER_AGENT_MODERATION.md` | Conceptual lineage / forward-looking application design | A design for an above-game agent that reads compressed player-competence telemetry and emits next-floor intent. Not built. |

The central pattern is intentionally incomplete symmetry:

- the runner compresses the full game state into a smaller perception payload
  such as floor, biome, HP ratio, alert level, visible enemies, combat state,
  inventory, and visible gates;
- the policy chooses an axis such as progression, survival, or resource;
- the turn envelope batches actions until a stop condition indicates the world
  has become volatile again;
- the agent does not need authorial knowledge of the full procedural system to
  produce coherent play.

That maps cleanly to the theorem program: alignment is not direct target
inspection, but adaptive action from compressed environmental feedback.

### Inspectable Surfaces

Sundog-side runner:

- `docs/runners.md`: formal turn-envelope architecture, CLI flags, and policy
  contract.
- `runners/adapters/gone_rogue.py`: Playwright adapter, perception compression,
  and turn-envelope execution.
- `runners/gone_rogue_headless.py`: headless batch CLI.
- `runners/gone_rogue_ui.py`: visible-browser CLI for inspection and captures.
- `runners/policies/gone_rogue_greedy.py`: built-in greedy policy with axis
  selection and volatility handling.
- `runners/adapters/gone_rogue_harness.html`: self-contained browser harness.
- `tests/runners/test_gone_rogue.py`: unit and integration tests against the
  JavaScript API.

EyesOnly-side runner target:

- `public/js/gone-rogue.js`: exposes `GoneRogue.headless`.
- `public/js/agent-api-system.js`: headless action API.
- `public/js/agent-integration.js`: agent testing bridge with UI modes.
- `public/js/agent-command-system.js`: AGENT command surface.
- `public/js/kernel-manager.js`: external agent integration layer.

EyesOnly-side automation neighbor:

- `public/js/playtest-agent.js`: UI-bound regression bot for UX edge cases such
  as touch targets, z-index, clipping, and animation loss. It ships alongside
  the runner, but it is not the load-bearing Sundog policy surface.

Forward-looking design:

- `docs/MANIPULATION_LAYER_AGENT_MODERATION.md`: Live Agentic Game Moderation
  design: Player Competence Model, Live Agentic Moderator, Floor Synthesis
  Engine, and Human-in-the-Loop Console. This is inspectable design lineage,
  not shipping code.

### What It Demonstrates

EyesOnly demonstrates three bounded things:

- The Sundog turn envelope can drive a real product engine through a stable
  headless API. `GoneRogue.headless` is an EyesOnly engine surface; the Sundog
  Python adapter calls it through Playwright; `PERCEIVE / PLAN / EXECUTE_BATCH`
  runs end-to-end with a greedy policy. The apparatus exists and runs.
- The runner architecture is policy-pluggable and seedable. New policies can
  implement the policy contract, and `GoneRogue.setSeed(seed)` supports
  deterministic matched-seed studies once the policy set is filled out.
- A UI-bound automation surface exists alongside the runner, but its scope is
  UX regression. That is valuable product automation. It should not be counted
  as evidence that a Sundog policy operates under visible UI constraints.

### What To Measure Next

To turn this from apparatus into a controlled product study:

- Matched-seed multi-policy study: define success metrics such as floor reached,
  survival rate, resources collected, run volatility, cards wasted, combat
  losses, completion time, and total steps. Compare the existing greedy policy
  against `random_legal` and a target-aware/debug-state upper-bound policy on
  identical seeds.
- Compressed-perception ablation: run the same policy with full perception and
  reduced perception payloads, then report where compression helps and where it
  loses information.
- Volatility-threshold sweep: vary the stop-condition threshold and measure
  survival rate, steps per floor, and batch length.
- Behavior clips: use the visible runner to capture identical seeds under
  different policies for the applications gallery.
- LAGM feeder data: store matched-seed JSONL bundles in a shape that could later
  feed the above-game moderator's competence model.

### Forward-Looking Application Design

The current Sundog evidence in EyesOnly is the under-game runner. A separate,
currently unbuilt above-game application is designed in EyesOnly as Live Agentic
Game Moderation (LAGM).

LAGM's Sundog shape, on paper, applies the indirect-signal-to-action pattern
above the game:

- indirect signal: per-floor player telemetry such as completion time, damage
  taken, ability usage, puzzle solve time, backtracking, secret discovery,
  exploit attempts, and hoarding behavior;
- transformation: a competence model compresses telemetry into competence,
  confidence, frustration, and per-system mastery indices;
- output: the moderator emits next-floor intent for a floor synthesis engine.

This is conceptual lineage and forward-looking application design. It is not
shipping code, and it should not be presented as current evidence.

### Claim Boundary

Safe claims:

> The under-game headless turn-envelope runner drives the real Gone Rogue
> JavaScript engine through a Playwright bridge against the `GoneRogue.headless`
> API. `PERCEIVE` compresses game state into a typed payload; `PLAN` selects an
> axis and stop-conditioned action batch; `EXECUTE_BATCH` applies actions until a
> stop condition fires. The apparatus is policy-pluggable and seedable.
>
> The UI-bound playtest agent ships alongside the runner, but it is scoped to UX
> regression. It is sibling automation, not a Sundog policy expression.
>
> LAGM is a design doc for a future above-game Sundog application. It is not
> built.

Avoid:

> EyesOnly proves the theorem for procedural games.
>
> The playtest agent demonstrates Sundog under tighter UI constraints.
>
> EyesOnly performs Sundog-driven level manipulation.

The runner is apparatus. The matched-seed study and policy comparison have not
run yet; the playtest agent is UX automation; LAGM is forward-looking design.

## Dungeon Gleaner / DCgamejam2026

Repository: [humiliati/DCgamejam2026](https://github.com/humiliati/DCgamejam2026)

Local sibling path on the maintainer machine:
`C:\Users\hughe\Dev\Dungeon Gleaner Main`

### Sundog Expression

Dungeon Gleaner applies Sundog to NPC idle behavior in a raycast dungeon
crawler. The product question is practical: how do you make a dungeon town feel
inhabited without paying the compute and authoring cost of GOAP, hierarchical
task networks, or hand-authored behavior trees?

The shipped answer is verb-field diffusion. Each NPC carries a small set of
prioritizing verbs such as `duty`, `social`, `errands`, and `rest`. Each verb
has a need value that changes over time and a set of satisfier nodes in the
world. A bonfire might satisfy `social`; a faction post might satisfy `duty` for
matching-faction NPCs.

Each tick, reachable nodes are scored from the NPC's unmet verb needs, satisfier
matches, inverse distance, linger gates, and small noise. The NPC takes a
greedy step toward the strongest pull. Arrival drops the relevant need, the NPC
lingers, and other verbs begin to reassert. The orbit emerges from the field.

This is the same family as the photometric experiment in a non-optical domain:
the agent does not need an authored plan for where it should go next. It reads
an indirect signal, the gradient of unmet need over satisfier proximity, and
acts from the transformed signal.

### Inspectable Surfaces

- `docs/VERB_FIELD_NPC_ROADMAP.md`: design doc for the verb-field NPC system;
  algorithm, encounter taxonomy, archetype presets, and implementation phases.
- `docs/NPC_SYSTEM_ROADMAP.md`: legacy patrol substrate the verb-field replaces
  and coexists with.
- `engine/verb-field.js`: per-tick verb-field resolution: decay, linger gate,
  scoring, step, arrival, and satisfaction drop.
- `engine/verb-nodes.js`: spatial node registry for satisfier sources.
- `engine/dungeon-verb-nodes.js`, `engine/verb-node-seed.js`, and
  `engine/verb-node-overrides-seed.js`: node populations and authored overrides
  for shipping floors.
- `engine/npc-system.js`: integration point wiring verb-field behavior beside
  legacy patrol behavior.
- `engine/reanimated-behavior.js`: second consumer of the same field tick for
  reanimated friendly actors.

### What It Demonstrates

Dungeon Gleaner demonstrates three bounded things:

- The Sundog indirect-signal pattern can transpose from optics to game-agent
  behavior. Detector intensity becomes unmet-verb gradient; scan/seek/track
  becomes need decay, node scoring, and greedy movement.
- Verb diffusion can serve as a lightweight substitute for hand-authored idle
  planners in a town-simulation context. The current evidence tier is Product
  Expression because it is shipped behavior, not a controlled benchmark.
- Personality can be represented as field weighting. Different archetypes share
  a verb vocabulary, but their need rates and satisfier weights produce
  different rhythms.

### What To Measure Next

- Orbit telemetry: per-NPC traces of current verb, need values, current node,
  target node, and dominant pull at a fixed cadence.
- GOAP-substitution comparison: the same town fixture implemented with a
  minimal GOAP or behavior-tree baseline, compared on authoring time, code size,
  runtime cost, and observable variability.
- Tuning sensitivity: sweeps over distance weight, noise, linger time, and
  satisfaction drop to locate flicker, permanent attachment, and robotic
  convergence regimes.
- Archetype distinguishability: node residency time and orbit period across
  archetypes on the same map.
- Encounter rate and classification: how often NPCs converge on the same node,
  and whether the encounter labels match outside observer reads.

### Claim Boundary

Safe claim:

> Dungeon Gleaner uses a verb-field diffusion model for NPC idle behavior in
> lieu of GOAP, hierarchical task networks, or hand-authored behavior trees.
> Each NPC's prioritizing verbs change over time; a per-tick score combines
> unmet need with inverse distance to satisfier nodes; the NPC takes a greedy
> step toward the strongest pull. Encounters and personality emerge from the
> verb vocabulary plus per-archetype weighting. This is a product expression of
> the same indirect-signal-to-action shape as the photometric mirror experiment,
> applied to lightweight game agency.

Avoid:

> Dungeon Gleaner proves verb-field diffusion outperforms GOAP for town
> simulation.

The substitution claim is qualitative until the comparison and telemetry work
above exists.

## Money Bags

Repository: [humiliati/Money-Bags](https://github.com/humiliati/Money-Bags)

Local sibling path on the maintainer machine:
`C:\Users\hughe\Dev\Money Bags\Money Bags`

### Sundog Expression

Money Bags applies Sundog to softbody terrain systems. The project is a Godot
4 prototype built around an N-body spring rig moving over complex terrain. The
current product direction emphasizes passive or semi-passive observation:
deposit the rig onto terrain, apply repeatable disturbances, and interpret the
resulting motion.

Here the indirect signal is not light. It is deformation:

- rig topology;
- spring graph state;
- per-node contact;
- centroid travel;
- radial and tangential deformation;
- torsion;
- torque from contact tangents;
- center-of-gravity behavior;
- recovery after disturbance.

The Sundog move is to treat frame-by-frame physics data as an analyzable graph
signature rather than raw motion noise.

The Money Bags adaptation lifts the abstraction layer from this repository
rather than transpiling Python to GDScript. Specifically: BloomTracker's
positional spread shape became the per-tick shape-deviation field; the
torque-stability variance shape became torsion stability; the
TorqueShadowAgent phase-machine pattern became the POP_EVENT extractor; the
alignment theorem's `radial × (1 - torsion/TORSION_REF) × symmetry`
formulation became the alignment scalar. Each adaptation is banner-marked in
the collector source for traceability.

Money Bags' distinctive shape among Sundog applications is that its strongest
work to date is **structural**: a pre-registered falsification apparatus
authored before measurement runs. Vocabulary, slate composition, prediction
direction, verdict template, and disposition rule are committed before the
experiment dispatches. The project's value to the Sundog research program is
currently in showing what it looks like to make the see-saw musing
falsifiable, not in confirming or refuting it.

### Inspectable Surfaces

Core rig:

- `README.md`: project map, passive direction, harness hotkeys, playtest
  bundle format.
- `alpha/Alpha_PlayerController.gd`: N-body spring-rig controller.
- `alpha/Alpha_PlayerTuning.gd`: exported tuning surface for topology,
  springs, FrameForce shape restoration, gravity, contact torque, rotation
  knobs, and related fields.
- `alpha/Alpha_RigNodeBody.gd`: per-node contact tangent torque behavior.
- `alpha/aREADME.md`: Alpha lane overview and controller responsibilities.

Instrumentation:

- `echo/Echo_Harness.tscn`: development harness.
- `echo/Echo_RigMetricsCollector.gd`: the Sundog-adapted collector. Per-tick
  alignment / torsion / shape-deviation / energy-partitioning / contact /
  bias readouts. Banner-marked imports from the Sundog abstraction layer.
- `echo/Echo_RigMetricsCollector.md`: calibration sidecar documenting
  scoring constants, thresholds, and the Stage 1 unit-weight calibration
  discipline.
- `echo/Echo_RigHealthMonitor.gd`: catastrophe watchdog (NaN / escape /
  overstretch). Sibling to the metrics collector — health watches for
  catastrophe; metrics watches for coherence.
- `echo/Echo_PerNodeStateInspector.gd`: per-node state inspection.
- `echo/Echo_NodeCountWidget.gd`: topology controls.
- `echo/Echo_JostleScenario.gd`: repeatable disturbance definitions.
- `echo/Echo_JostlePlayer.gd`: scripted impulse playback.
- `playtest/<capture>/metrics.json`: alignment, torsion, speed, deformation,
  symmetry, recovery, and shape-coherence-bias metrics.
- `playtest/<capture>/metrics.csv`: full-resolution per-tick telemetry.
- `playtest/CROSS_PROFILE_AGGREGATE_<datetime>.md`: auto-emitted comparison
  table across same-session bundles, with the bias readout per profile.

Design and discipline:

- `docs/design/Sundog_toolset_PROPOSAL.md`: the contractor proposal that
  adapted Sundog's abstraction layer into Money Bags' instrumentation
  pattern. Includes the asset-mapping table from `<sundog>/utils`,
  `<sundog>/agents`, and the alignment theorem text.
- `docs/design/Shape_Coherence_Architecture.md`: the ratified vocabulary
  doc. Names FFSR as rotation-locking, adjacent springs as
  rotation-permissive, and the see-saw bias axis as the rotation-permissiveness
  knob in disguise. Sign convention and stage gating committed.
- `docs/design/Shape_Coherence_Stage_1_Rubric.md`: the falsification
  apparatus. Slate composition (4 fixtures × 4 profiles × 3 captures),
  predicted signatures, verdict template, four-mode kill-switch taxonomy,
  bias-band threshold extraction methodology, and the canonical five-factor
  rotation-suppressor reference.
- `docs/design/Shape_Coherence_Pass_1_5_Candidates.md`: pre-Stage-1 tooling
  candidates filed against actual exploratory data; later ratified into the
  collector Pass 1.5(α/β/γ/δ/ε) ship sequence.
- `docs/CANONICAL_RIG_EXPERIMENT_2026-04-26.md`: Alpha's parallel
  canonical-rig experiment, sibling to the falsification rubric. The two
  experiments are orthogonal (different slates, different verdict
  semantics); the rubric's GATE 4 reads Stage 1's bias-band output.
- `docs/handoffs/Echo_HANDOFF.md`: addendum-appended audit trail of the
  c.Echo work shipping into Echo's lane file.

Terrain:

- `foxtrot/fREADME.md`: terrain fixture descriptions.
- `foxtrot/Foxtrot_*.tscn`: calibration ramps, Rube-Goldberg fixtures,
  trip-hazard scenes, and SlopeFunnel variants.

### Metric Glossary

The collector publishes the following per-tick and per-window quantities.
Each is operationally defined; the formulas are sourced from
`echo/Echo_RigMetricsCollector.gd`.

**Alignment family.**

- `alignment_score` — per-tick scalar. Computed as
  `radial_frac × (1 - clamp(torsion_index / torsion_ref, 0, 1)) × symmetry_factor`.
  Returns NaN when `grand_total_ke < idle_epsilon` (the idle sentinel
  introduced for noise-grade settled rigs).
- `alignment_mean` / `alignment_min` / `alignment_max` — per-window
  aggregates. NaN-safe: idle samples skipped. The max field is the Stage 1
  Prediction 2 data source (peak alignment in the rolling buffer surfaces the
  recovery dynamic that window-mean dilutes).

**Torsion family.**

- `torsion_index` — fraction of internal kinetic energy in tangential
  motion. Per-tick scalar, range [0, 1]. Tangential energy is rotational
  energy in a deformable rig — perpendicular-to-radial component of
  velocity-relative-to-centroid summed across nodes.
- `torsion_mean` / `torsion_max` — per-window aggregates with timestamp.

**Shape-deviation family.**

- `radius_stddev` — standard deviation of per-node radii from centroid
  per-tick. Captures shape distortion magnitude.
- `peak_deformation.radius_stddev` — max radius_stddev in window with
  timestamp.
- `radius_stddev_mean` / `radius_stddev_max` — per-window aggregates.

**Energy-partitioning family.**

- `radial_frac` / `tangential_frac` / `translational_frac` — fractions of
  total kinetic energy in each component. Per-tick they sum to 1.0 when the
  rig has motion. Window-means computed over active-energy ticks only.
- `energy_split_at_peak` / `energy_split_at_settle` — `(c, r, t)` triplets
  sampled at named-phase ticks (peak velocity moment for `_at_peak`,
  post-impact-settle marker for `_at_settle`). The `_at_settle.t` field is
  the rotation-rate canonical signal for the Stage 1 verdict template.

**Contact-topology family.**

- `grounded_count` / `grounded_clusters` / `support_offset` — per-tick
  contact aggregates. Adjacency clusters computed via DFS on the spring
  subgraph induced by grounded nodes.

**Shape-coherence-bias family.**

- `shape_coherence_bias_actual` — `(k_ffsr × ffsr_unit_weight) /
  (k_ffsr × ffsr_unit_weight + k_spring × spring_unit_weight)`, normalized
  to [-1, +1]. Sign convention: positive = toward FFSR (centripetal,
  rotation-locking, marble direction); negative = toward adjacent
  (rotation-permissive, rolling direction).
- `k_ffsr_raw` / `k_spring_raw` — raw stiffness values published alongside
  bias for unit-weight observability.

**Symmetry family.**

- `symmetry_factor` — per-tick `1 - clamp(radius_stddev / avg_radius)`.
  Captures uniformity of node-radius distribution around centroid.
- `symmetry_score.mean` / `.max` / `.at_settle` — per-window aggregates,
  with the `at_settle` reading at the post-impact-settle phase.

**Motion + recovery + topology + diagnostic.**

- `centroid_travel_in_buffer_px` — total centroid translation in window.
- `max_centroid_speed_in_buffer` / `_at_session` — peak speed reached.
- `time_to_settle_ms` — milliseconds from velocity peak to sustained
  centroid speed below 5 px/s for 30 ticks. The post-impact-settle marker
  used for recovery-time reads.
- `pop_completed` / `pop_recovered` / `pop_timed_out` — POP_EVENT lifecycle
  counts. Each event is a structured record in the bundle's `[pop_events]`
  block.
- `rig_n` — node count. Spring count is `N(N-1)/2` for the complete-graph
  topology.
- `compute_ms_last` / `hot_disabled` — collector cost-budget diagnostics.

The same field names appear in the per-bundle `metrics.json` digest, the
per-tick `metrics.csv` rows, and the cross-profile aggregate table. Naming
is stable across surfaces; field semantics are stable across MB releases
that share the collector source.

### What It Demonstrates

Money Bags demonstrates three things, each evidence-bounded:

**1. Sundog's abstraction layer transposes from optics to softbody graph
telemetry.** BloomTracker's positional-spread shape became per-tick
shape-deviation. The torque-stability variance shape became torsion stability.
The alignment-theorem formula became the alignment scalar. Banner-marked
adaptations in collector source allow line-by-line traceability from the
theorem text to the running code. Shadow is replaced by shape projection;
torque is replaced or supplemented by contact tangent torque and spring
deformation; the indirect-signal pattern persists across the optics-to-
softbody substitution.

**2. A pre-registered falsification apparatus can be built before
measurement runs.** The Money Bags work has produced a ratified vocabulary
doc (sign-convention and stage gating committed), a four-fixture × four-
profile slate with predicted per-profile signatures, a verdict template with
explicit disposition rules, a four-mode kill-switch taxonomy distinguishing
fixture-mode / architecture-mode / bias-mode / suppressor-stack-incomplete
failures, and a bias-band threshold extraction methodology that feeds into
the parallel canonical-rig experiment's GATE 4. The structure exists; the
captures it interprets do not yet. The ordering is intentional.

**3. External audit catches architectural blind spots that internal teams
compose around.** A freelancer audit late in the project surfaced a
hardcoded `body.lock_rotation = true` constraint that the contractor, the
lane engineer, and Command had all missed. The team had been compensating
around the suppressor without naming it. The finding extended the project's
suppressor model from FFSR-only to a five-factor canonical reference and
reframed the falsification rubric's verdict-authoring rule. The lesson is
codified in the project's memory and process discipline: audit existing
controller hardcodes before chartering new knobs.

What Money Bags is **not** demonstrating: that the see-saw musing
empirically holds, that the bias-axis vocabulary is validated by experiment,
or that alignment readings correlate with subjective feel. Those readings
require structured Stage 1 captures the project hasn't run yet.

### What To Measure Next

The earlier version of this section listed: define alignment formally,
freeze terrain fixtures, run matched disturbance scripts, define failure
regimes. Most of those items are now closed by the apparatus described
above. What remains is the Stage 1 capture run itself, gated on a
short list of architecture-side and tooling-side prerequisites.

**Stage 1 capture prerequisites in flight:**

- Capture-crash fix for the high-gain rotation-permissive primitive
  (the `_v_tumble_unlocked` profile at gain=1.0 currently surfaces a
  positive-feedback runaway through the synchronous capture path; the
  diagnostic harness has shipped, mitigation pending data).
- Architecture-knob exposure: the project's hardcoded `body.lock_rotation`
  is now exposed as `node_lock_rotation` per the freelancer audit; the
  rotation-aware FFSR knob (`frame_force_rotation_aware`) is shipped; full
  capture validation pending.
- The verification profile (`_v_tumble_*`) authored by the contractor
  partner side, demonstrating non-zero tumble on the trip-hazard fixture
  with both architecture knobs enabled.

**Stage 1 reads that become available once captures land:**

- Apply the Stage 1 verdict template across the 4-fixture × 4-profile
  slate; produce CONFIRM / REFUTE / AMBIGUOUS verdict on the see-saw
  musing.
- If CONFIRM: extract the bias-band thresholds (`Z_lo, Z_hi`) per the
  rubric's bias-band methodology; route the result into the parallel
  canonical-rig experiment's GATE 4.
- Validate or refine the unit-weight calibration (`ffsr_unit_weight = 1.0,
  spring_unit_weight = 0.5` per Path A): pre-flight reads of the anchor
  profile should land at `-0.85 ± 0.03`. Pre-Stage-1 captures already match
  this prediction across 4 anchor reads; structured captures should
  reproduce.

**Independent of Stage 1 (gameplay-design surface):**

- Validate the gameplay-primitive corollary discovered during the
  apparatus build: each suppressor-lift × rotation-aware-knob-gain
  combination produces a distinct gameplay primitive (Tumble at gain=0.5
  locked, Pressure-release jump at gain=1.0 locked, Centrifugal spill at
  gain=1.0 unlocked). Filed as forward-looking design surface for any
  Stage 2 work.

**Cross-program comparison the captures will support:**

- Graph-enriched interpretation versus raw center-of-mass telemetry on
  the same captures: does the graph layer predict recovery behavior or
  player-readable feel better than the centroid trace alone? Apparatus
  is in place to read both side by side.

### Pre-Stage-1 Findings

The apparatus build produced three findings before any structured capture
run, each documented in the Money Bags repo and ratified by Command. They
are reported here because they are the kind of detail a skeptical
researcher would want before granting that the Stage 1 captures, when they
land, will be measuring something real.

**Path A unit-weight calibration finding.** The bias readout's two
unit-weight constants (`ffsr_unit_weight` and `spring_unit_weight`) ship at
defaults derived from a Stage 1 finding rather than a Stage 0 assumption.
The `_v_bias_zero` profile authored at `(k_ffsr = 18, k_spring = 36)`
measured at `-0.33` rather than `0.00` under the initial `1.0 / 1.0`
defaults — non-symmetric across the slate. The corrected weights
(`spring_unit_weight = 0.5`) place `_v_bias_zero` at exactly `0.00` and
the anchor profile at `-0.846` across all post-correction reads. The
calibration also exposes that adjacent stiffness contributes
approximately twice per unit relative to FFSR in the bias-axis arithmetic,
which is itself useful structural information for any future Stage 2
meta-knob design.

**Bounded inflation versus socket failure discriminator.** The collector
publishes `radius_stddev_mean` and `radius_stddev_max` per window. Bounded
skin-readable inflation (rig sits in a slightly-deformed equilibrium)
produces `radius_stddev_mean ≈ radius_stddev_max`; socket failure (rig
oscillates through a limit cycle of FFSR-vs-adjacent contention) produces
`radius_stddev_max` significantly above `radius_stddev_mean`. The
distinction is a clean data-side discriminator between two interpretations
that could otherwise be confused at the same nominal deformation
magnitude. Useful for any rig-experiment dispatch that needs to admit a
non-zero idle-shape signature without conflating bounded inflation with
broken sockets.

**Freelancer audit precedent.** The project's third significant
architectural finding came from an external freelancer's CCC audit pass,
not from internal review. The freelancer surfaced a single-line hardcoded
`body.lock_rotation = true` constraint at the rig controller that the
contractor partner, the lane engineer, and Command had each missed while
designing and ratifying three new rotation-aware tuning knobs. The team
had been compensating around the suppressor without naming it. The
finding extended the suppressor model from FFSR-only to a five-factor
canonical reference, reframed the falsification rubric's verdict-authoring
rule with a `suppressor-stack-incomplete` mode, and produced a durable
process lesson now codified in the project's memory: *audit existing
controller hardcodes before chartering new knobs — they may already exist
as toggle-switches awaiting exposure.* The lesson is itself reportable
content for the Sundog research program: internal teams develop blind
spots around constraints they have been compensating around, and periodic
external audits surface those blind spots more reliably than longer
internal review cycles.

### Claim Boundary

Safe claim:

> Money Bags built a pre-registered falsification apparatus for the Sundog
> see-saw musing applied to softbody rigs. The apparatus includes a
> ratified vocabulary doc with sign-convention and stage-gating, a
> four-fixture by four-profile by three-capture-per-cell falsification
> slate, a pre-registered verdict template with explicit disposition rule,
> a four-mode kill-switch taxonomy distinguishing fixture-mode /
> architecture-mode / bias-mode / suppressor-stack-incomplete failures, and
> a five-factor rotation-suppressor canonical reference produced via
> external audit. Per-bundle telemetry already captures alignment, torsion,
> deformation, symmetry, recovery, and shape-coherence-bias signals.

Avoid:

> Money Bags proves Sundog applies to softbody rigs.

The Stage 1 verdict is currently UNTESTABLE: structured captures pending
architecture-side knob exposure and capture-crash fix. The project's
strongest evidence to date is the apparatus itself and the pre-Stage-1
findings produced during the build. The captures the apparatus is
designed to interpret have not yet run.

## Cross-Application Comparison

| Application | Domain | Indirect signal | Transformation | Actionable output |
| --- | --- | --- | --- | --- |
| Sundog core | Photometric control | Detector intensity and proprioception | Scan, seek, extremum tracking | Mirror alignment without target position |
| Three-body dynamics | Planar restricted dynamics | Local tidal-tensor proxy, acceleration magnitude, and separated privileged diagnostics | Guarded TRACK control over a sensor-motivated instability signature | High-velocity near-escape operating pocket with mapped low-velocity and equal-mass harms |
| Sundog Balance | Embodied control | Cast-shadow centroid, length, residual, and velocity under controlled light geometry | SCAN/SEEK/TRACK shadow-residual control with confidence gating and separated privileged diagnostics | Bounded cart force maintaining upright balance inside a mapped lighting and delay envelope, with overhead-light and high-delay degradation boundaries |
| EyesOnly / Gone Rogue | Procedural agent play | Floor, biome, HP, alert, inventory, gate, and combat state from `GoneRogue.headless.getState` | Perception compression plus axis-selected stop-conditioned batches through Playwright | Seedable, policy-pluggable JSONL runs ready for matched-seed comparison studies and future LAGM feeder data |
| Dungeon Gleaner | Procedural NPC behavior | Unmet-verb gradient over satisfier nodes | Need decay, inverse-distance scoring, linger gates, and noise | Emergent NPC idle orbits without scripted plans |
| Money Bags | Softbody terrain physics | Spring graph, contact, deformation, torque, centroid motion | Graph metrics and playtest telemetry; pre-registered falsification apparatus around the rotation-permissiveness bias axis | Interpretable rig state, recovery analysis, and a falsifiable hypothesis with verdict template committed before captures |

## Broadcast-Aligned Summary

For public communication, the applications can be summarized this way:

> Since the initial theorem release, Sundog has moved from a single
> mirror-alignment experiment into workbench and product systems. The
> three-body dynamics tab tests guarded proxy control in a bounded
> high-velocity near-escape pocket while preserving the failure map. Balance
> tests shadow-derived cart-pole control inside a mapped lighting and delay
> envelope while reporting overhead-light all-fail margins as degradation, not
> usable control. EyesOnly applies the idea to a headless turn-envelope runner
> against a real procedural roguelike engine,
> while keeping sibling UX automation and unbuilt LAGM design in separate
> evidence tiers. Dungeon Gleaner applies it to
> verb-field NPC behavior, where unmet needs diffuse across satisfier nodes to
> produce idle orbits without scripted planners. Money Bags applies it to
> softbody rigs, where graph telemetry makes torsion, torque,
> center-of-gravity behavior, and recovery legible frame by frame — and where
> the project's strongest current contribution is a pre-registered
> falsification apparatus authored before measurement runs, including a
> ratified vocabulary, a four-fixture slate, a verdict template, and a
> five-factor rotation-suppressor canonical reference produced through
> external audit.

For academic communication, add the boundary:

> These applications motivate the research program. The controlled claim still
> rests on the photometric mirror-alignment experiment. Three-body and Balance
> now have bounded operating-envelope studies; the product applications still
> require their own baselines, metrics, and reproducible studies.

## Suggested Inspection Order

1. Read the root `README.md` in this repo.
2. Read `docs/RESEARCHER_GUIDE.md`.
3. Inspect `docs/PAPER_v1_draft.md` and `results/analysis/analysis_summary.json`.
4. Read `docs/THREEBODY_PHASE11_SUMMARY.md` and launch `threebody.html`.
5. Read `docs/sundog_v_balance.md`, then launch `balance.html` and inspect
   `results/balance/phase10-envelope/verdict.md`.
6. Read `docs/runners.md` for the EyesOnly bridge.
7. Inspect EyesOnly's headless runner, UI automation neighbor, and LAGM design
   surface as separate evidence tiers.
8. Inspect Dungeon Gleaner's verb-field NPC docs and runtime modules.
9. Inspect Money Bags' playtest bundles and graph/rig telemetry.

## Next Documentation Tasks

- Add an EyesOnly matched-seed multi-policy study, compressed-perception
  ablation, volatility sweep, and behavior clips.
- Promote the three-body workbench into the public gallery with Phase 11 charts,
  the comparison slate, and an explicit failure-boundary view.
- Replace the Balance-first rail's Phase 10 best/worst poster with a cheap
  exported loop when `.webm`/MP4/`.ogv` size testing justifies it; keep the
  poster fallback and keep both recovery and failure boundary named.
- Add Dungeon Gleaner orbit telemetry for verb, need, node, and dominant-pull
  traces.
- Add a Dungeon Gleaner GOAP-substitution comparison and tuning-sensitivity
  sweep.
- Add a Money Bags metric glossary with formulas for alignment, torsion,
  symmetry, and recovery.
- Add an "Applications as Experiments" appendix once each product has at least
  one reproducible benchmark.
