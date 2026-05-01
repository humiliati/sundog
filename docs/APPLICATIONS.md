# Sundog Application Map

This document links the Sundog research repo to product systems that express
the same alignment program in practical software.

The root scientific claim still belongs to this repository: the current
controlled result is the photometric mirror-alignment experiment. The
applications below are not substitutes for that result. They are evidence that
the same pattern can be turned into usable systems across different domains:
procedural agents, physical-feeling game simulation, and softbody telemetry.

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
| Instrumented prototype | Product system with telemetry and repeatable harnesses, but not yet a paper-style study. | Money Bags playtest bundles, Sundog Gone Rogue runner. |
| Product expression | Player-facing mechanic or agent system that embodies the idea, but needs formal measurement. | Dungeon Gleaner pressure washing and glass/window rendering. |
| Conceptual lineage | Historical or design-language connection without enough evidence yet. | Older theorem docs, broad broadcast language. |

The public-facing story can mention all four. Academic writing should keep
them separate.

## EyesOnly / Gone Rogue

Repository: [humiliati/EyesOnly](https://github.com/humiliati/EyesOnly)

Local sibling path on the maintainer machine:
`C:\Users\hughe\Dev\EyesOnly`

### Sundog Expression

EyesOnly applies Sundog as procedural control under partial observation.
Gone Rogue is a procedural roguelike embedded in the larger spy-game platform.
The Sundog runner in this repo interacts with the real JavaScript engine through
Playwright and the `GoneRogue.headless` API.

The central pattern is intentionally incomplete symmetry:

- the runner compresses the full game state into a smaller perception payload;
- the policy chooses an axis such as progression, survival, or resource;
- the turn envelope batches actions until a stop condition indicates the world
  has become volatile again;
- the agent does not need authorial knowledge of the full procedural system to
  produce coherent play.

That maps cleanly to the theorem program: alignment is not direct target
inspection, but adaptive action from compressed environmental feedback.

### Inspectable Surfaces

In this repo:

- `docs/runners.md`
- `runners/adapters/gone_rogue.py`
- `runners/policies/gone_rogue_greedy.py`
- `tests/runners/test_gone_rogue.py`

In EyesOnly:

- `public/js/gone-rogue.js`: exposes `GoneRogue.headless`.
- `public/js/agent-api-system.js`: headless action API.
- `public/js/agent-integration.js`: agent testing bridge with UI modes.
- `public/js/agent-command-system.js`: AGENT command surface.
- `public/js/playtest-agent.js`: human-UI-bound playtest agent.
- `public/js/kernel-manager.js`: external agent integration layer.

### What It Demonstrates

EyesOnly demonstrates that Sundog-style action envelopes can operate against a
real product engine rather than a toy simulator. The agent bridge can run
through the actual game API, and the human-UI-bound playtest agent provides a
separate constraint: an agent can be forced to respect visible UI affordances
instead of bypassing the rendering pipeline.

This is important because it moves the framework from "controller in a lab
task" toward "agent in a live procedural system."

### What To Measure Next

To turn this from product expression into a controlled result:

- define success metrics: floor reached, survival rate, resources collected,
  run volatility, cards wasted, combat losses, or completion time;
- compare the Sundog turn-envelope runner against a flat greedy policy, random
  legal actions, and a target-aware/debug-state policy;
- run matched seeds across all policies;
- report where compressed perception helps and where it loses information;
- separate headless-agent performance from human-UI-bound playtest-agent
  performance.

### Claim Boundary

Safe claim:

> EyesOnly shows Sundog-derived turn envelopes operating against a real
> procedural roguelike through compressed perception and stop-conditioned
> action batches.

Avoid:

> EyesOnly proves the theorem for procedural games.

It does not yet, because the repo needs a study with metrics and baselines.

## Dungeon Gleaner / DCgamejam2026

Repository: [humiliati/DCgamejam2026](https://github.com/humiliati/DCgamejam2026)

Local sibling path on the maintainer machine:
`C:\Users\hughe\Dev\Dungeon Gleaner Main`

### Sundog Expression

Dungeon Gleaner applies Sundog to physical-feeling simulation in a raycast
dungeon crawler. The broadcast identifies two major expressions:

- approximate light reflection across glass surfaces at roughly one-twelfth
  conventional computational cost;
- a pressure-washing mechanic that produces coherent helical or arcing water
  behavior.

The useful research framing is not "this is physically exact." It is:

> Selected physical phenomena can be compressed into structured approximations
> that retain player-facing coherence at lower computational cost.

That is the same family as the photometric experiment: take an observable
projection of physical structure, transform it cheaply, and use the result for
software behavior.

### Inspectable Surfaces

Pressure washing:

- `README.md`: cleaning and pressure-washing gameplay loops.
- `docs/ADR-001-PRESSURE_WASHER_FPS_WEAPON.md`: pressure washer as
  first-person physical tool, projectile model, bounce/arc model, kink readout,
  and energy model.
- `engine/hose-state.js`: hose energy, pressure, path length, and kink state.
- `engine/spray-system.js`: spray activation and delivery.
- `engine/spray-viewport-fx.js`: viewport water effects.
- `engine/torch-hit-resolver.js`: pressure-wash hit resolution for torches.
- `engine/torch-state.js`: torch state mutation under pressure washing.

Glass and reflection:

- `engine/window-sprites.js`: facade-facing glass panes, interior billboard
  sprites, glint animation, exterior-face detection, and gap-filler logic.
- `engine/texture-atlas.js`: procedural glass and window material generation.
- `engine/tiles.js`: window tile contracts and transparency/collision rules.

### What It Demonstrates

Dungeon Gleaner demonstrates product-level compression of physical effects:

- water behavior is routed through stateful hose pressure, kink, energy, spray,
  projectile, decal, and torch-interaction systems rather than a full fluid
  simulation;
- window/glass rendering uses face-aware heuristics, glints, transparent
  cavities, and sprite billboards rather than expensive scene reflection;
- the resulting player read is physical enough to support gameplay loops.

This is a strong product expression of the theorem program because the player
experiences physical coherence while the system avoids full simulation.

### What To Measure Next

For the one-twelfth cost claim, add an inspection harness with:

- the exact conventional baseline being compared;
- the optimized Sundog-derived implementation path;
- fixed scenes with representative glass/window cases;
- per-frame timing under the same hardware and browser;
- visual error or acceptability criteria;
- a cold-start/warm-cache distinction.

For pressure washing, add:

- a deterministic spray scene;
- frame cost with and without spray active;
- hose path length, kink count, pressure, and energy traces;
- coverage/decal metrics;
- a qualitative or quantitative measure of helical/arc coherence;
- side-by-side comparison against simpler hitscan spray.

### Claim Boundary

Safe claim:

> Dungeon Gleaner shows Sundog-shaped compression of physical-feeling effects:
> glass/window rendering and pressure-washing behavior are made coherent enough
> for gameplay without full physical simulation.

Avoid:

> Dungeon Gleaner scientifically proves a general one-twelfth-cost physics
> theorem.

That claim requires the measurement harness above.

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

### Inspectable Surfaces

Core rig:

- `README.md`: project map, passive direction, harness hotkeys, playtest
  bundle format.
- `alpha/Alpha_PlayerController.gd`: N-body spring-rig controller.
- `alpha/Alpha_PlayerTuning.gd`: exported tuning surface for topology,
  springs, FrameForce shape restoration, gravity, contact torque, and related
  knobs.
- `alpha/Alpha_RigNodeBody.gd`: per-node contact tangent torque behavior.
- `alpha/aREADME.md`: Alpha lane overview and controller responsibilities.

Instrumentation:

- `echo/Echo_Harness.tscn`: development harness.
- `echo/Echo_PerNodeStateInspector.gd`: per-node state inspection.
- `echo/Echo_NodeCountWidget.gd`: topology controls.
- `echo/Echo_JostleScenario.gd`: repeatable disturbance definitions.
- `echo/Echo_JostlePlayer.gd`: scripted impulse playback.
- `playtest/<capture>/metrics.json`: alignment, torsion, speed, deformation,
  symmetry, and recovery metrics.
- `playtest/<capture>/metrics.csv`: full-resolution telemetry.

Terrain:

- `foxtrot/fREADME.md`: terrain fixture descriptions.
- `foxtrot/Foxtrot_*.tscn`: calibration ramps, Rube-Goldberg fixtures, and
  jostle/trip-hazard scenes.

### Current Metric Vocabulary

Money Bags already records the kind of quantities that make a future study
plausible:

| Metric family | Example fields |
| --- | --- |
| Alignment | `alignment_mean`, `alignment_max`, `alignment_sample_count` |
| Torsion | `torsion_index` in playtest markdown summaries |
| Shape | `peak_deformation.radius_stddev`, FrameForce settings |
| Motion | `centroid_travel_in_buffer_px`, `max_centroid_speed_in_buffer` |
| Recovery | `time_to_settle_ms`, pop recovered/timed-out counts |
| Graph/topology | `rig_n`, complete graph spring count, node count controls |
| Energy split | radial, tangential, and centroid components |
| Symmetry | `symmetry_score.mean`, `symmetry_score.max`, `symmetry_score.at_settle` |

This is closer to an instrumented prototype than a pure product expression.

### What It Demonstrates

Money Bags demonstrates a concrete path from raw physics to structured
interpretability. A softbody rig can be observed through graph-aware telemetry:
which nodes touch terrain, how deformation splits between radial/tangential
components, whether the rig recovers after impulse, and whether alignment
improves or collapses under tuning changes.

That makes it the clearest bridge from `H(x)` to non-optical systems.
Shadow is replaced by shape projection; torque is replaced or supplemented by
contact tangent torque and spring deformation.

### What To Measure Next

To turn Money Bags into a controlled Sundog study:

- define an alignment score formally in terms of graph state, centroid,
  terrain normals, and recovery behavior;
- freeze a set of terrain fixtures;
- run matched disturbance scripts across tuning profiles;
- compare graph-enriched interpretation against raw center-of-mass telemetry;
- report whether graph metrics predict recovery, controllability, or player
  readability better than raw physics traces;
- define failure regimes: excessive torsion, low symmetry, unrecoverable pop,
  drift, or draped/passive settling.

### Claim Boundary

Safe claim:

> Money Bags extends Sundog from optical alignment into graph interpretation of
> softbody motion, with playtest telemetry already capturing alignment,
> torsion, deformation, symmetry, and recovery signals.

Avoid:

> Money Bags proves softbody alignment is solved.

It is a promising instrumented prototype. It still needs formal metrics and
matched experimental runs.

## Cross-Application Comparison

| Application | Domain | Indirect signal | Transformation | Actionable output |
| --- | --- | --- | --- | --- |
| Sundog core | Photometric control | Detector intensity and proprioception | Scan, seek, extremum tracking | Mirror alignment without target position |
| EyesOnly / Gone Rogue | Procedural agent play | Compressed and volatile game state | Turn envelope and stop conditions | Coherent action batches |
| Dungeon Gleaner | Physical-feeling rendering and tools | Glass faces, hose path, pressure, kinks, impacts | Face heuristics, glints, spray/projectile/decal state | Cheap coherent reflection and water behavior |
| Money Bags | Softbody terrain physics | Spring graph, contact, deformation, torque, centroid motion | Graph metrics and playtest telemetry | Interpretable rig state and recovery analysis |

## Broadcast-Aligned Summary

For public communication, the applications can be summarized this way:

> Since the initial theorem release, Sundog has moved from a single
> mirror-alignment experiment into working systems. EyesOnly applies the idea
> to procedural agent play under occluded state. Dungeon Gleaner applies it to
> physical-feeling simulation, including glass/window reflection approximations
> and pressure-washing behavior. Money Bags applies it to softbody rigs, where
> graph telemetry makes torsion, torque, center-of-gravity behavior, and
> recovery legible frame by frame.

For academic communication, add the boundary:

> These applications motivate the research program. The controlled claim still
> rests on the photometric mirror-alignment experiment until each application
> receives its own baselines, metrics, and reproducible study.

## Suggested Inspection Order

1. Read the root `README.md` in this repo.
2. Read `docs/RESEARCHER_GUIDE.md`.
3. Inspect `docs/PAPER_v1_draft.md` and `results/analysis/analysis_summary.json`.
4. Read `docs/runners.md` for the EyesOnly bridge.
5. Inspect EyesOnly's headless and UI-bound agent surfaces.
6. Inspect Dungeon Gleaner's pressure-washer ADR and window/glass modules.
7. Inspect Money Bags' playtest bundles and graph/rig telemetry.

## Next Documentation Tasks

- Add an EyesOnly runner result table with matched-seed policy comparisons.
- Add a Dungeon Gleaner performance note for the glass/window path and its
  baseline.
- Add a Dungeon Gleaner spray-system note with deterministic scene captures.
- Add a Money Bags metric glossary with formulas for alignment, torsion,
  symmetry, and recovery.
- Add an "Applications as Experiments" appendix once each product has at least
  one reproducible benchmark.
