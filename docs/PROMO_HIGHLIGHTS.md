# Sundog Promo Highlights

This document is for broadcasts, landing pages, pitch decks, interviews,
release notes, and short-form public writing. It is intentionally more
provocative than the research docs.

Use it as a message bank. Do not treat every sentence here as a paper claim.
The academic claim remains narrower: photometric mirror alignment without
target-position access, tested with matched MuJoCo experiments and baselines.

## Core Hook

Sundog turns indirect signals into usable control.

Where conventional systems ask for full world state, Sundog asks a stranger
question: what if the shadow, the torque, the occlusion, the deformation, or
the disturbance already contains enough structure to act?

## Short Positioning Lines

- Alignment does not always require direct sight.
- A system can learn where to aim by reading what the world does back to it.
- Sundog is a theory of useful partial information.
- We are building software that treats indirect signal as first-class input.
- The future of simulation is not always more fidelity. Sometimes it is better
  structure.
- H(x) is the promise that the halo around a process can be more useful than a
  direct measurement of the thing itself.
- We are not trying to render the whole world. We are trying to find the part
  of the world that matters.
- Sundog compresses certain physical behaviors into forms that agents can use.
- The theorem begins where ordinary perception gives up.
- The most useful signal is not always the most obvious one.

## One-Paragraph Promo

Sundog is a research and product program for transforming indirect observable
phenomena into usable software control. In the core experiment, a controller
aligns a reflected beam without target-position access, using only sparse
photometric feedback and joint state. In product systems, the same idea appears
as roguelike agents acting under occluded state, verb-field NPC behavior, graph
telemetry that makes softbody motion analyzable frame by frame, and a
three-body dynamics workbench where a sensor-limited controller uses only local
gravitational field readings to improve survival in a regime where a naive
controller fails. The claim is simple and dangerous:
some systems do not need to see the target directly. They can align by reading
the structure of the disturbance.

## Broadcast-Ready Version

Last year, Sundog was a theorem-shaped provocation: alignment may emerge from
the relationship between action, projection, and environmental response.

This year, it is a working program.

The core repo now contains a defensible mirror-alignment experiment: a
controller with no Cartesian target access reaches terminal accuracy comparable
to a target-aware analytic baseline in the tested MuJoCo setting. EyesOnly
extends the idea into procedural agent play under occluded state. Dungeon
Gleaner pushes it into verb-field NPC behavior, where unmet needs diffuse across
satisfier nodes instead of relying on scripted idle planners. Money Bags brings it
into softbody terrain systems, where torsion, torque, center of gravity,
deformation, and recovery become graph-readable telemetry. The three-body
workbench pushes the pattern into chaotic dynamics: a sensor-limited controller
that reads only local gravitational field gradients — not the positions or
masses of the primaries — improves survival in a bounded near-escape operating
pocket where a naive local baseline fails. The full state is 18-dimensional.
The controller uses three.

The direction is bigger than any one demo: software systems that do not need
perfect state to behave intelligently. Systems that can act from the halo.

## Hooks By Audience

### For Researchers

- What if indirect feedback can match direct-state control in constrained
  alignment tasks?
- Sundog reframes alignment as a measurable relationship between action and
  environmental response.
- The interesting result is not magic. It is the trade: less privileged state,
  slower acquisition, comparable terminal accuracy.
- The theorem is broad, but the current paper claim is narrow enough to attack.

### For Game Developers

- Sundog is a way to make systems feel reactive without giving them impossible
  information.
- It helps turn cheap approximations into coherent player-facing behavior.
- It gives agents a way to act from partial, compressed game state.
- It gives physics-heavy prototypes a telemetry language for motion that would
  otherwise look like noise.

### For Simulation Engineers

- More fidelity is not always the winning move.
- Sundog asks which projection of a physical process is control-relevant.
- If the approximation preserves the actionable structure, the system can be
  cheaper, more legible, and easier to tune.
- The future stack may combine high-fidelity simulation where needed with
  Sundog-style transforms where full fidelity is waste.

### For AI / Agent Builders

- An agent does not always need the full state. It needs the right compressed
  signal.
- Occlusion is not only a limitation. It can be a design surface.
- Stop-conditioned action batches are a practical bridge between reactive
  agents and unstable environments.
- Alignment may be less about telling an agent where the target is and more
  about giving it a world whose responses can be read.

## Provocative Statements

Use these when the context rewards sharpness. Pair them with evidence if the
audience is technical.

- Direct perception is overrated.
- Full state is a luxury, not a requirement.
- The shadow can be a sensor.
- The disturbance is the data.
- The halo is cheaper than the object.
- The system does not need the truth. It needs a transform that preserves the
  part of truth that matters.
- Physics can be compressed without becoming fake.
- Some behaviors are expensive only because we insist on simulating the wrong
  layer.
- A good approximation is not a shortcut. It is an argument about what matters.
- Occlusion is not failure. It is an interface.
- Alignment is not always a coordinate. Sometimes it is a resonance.
- We are building systems that can infer from the wake they leave behind.
- The future of game AI is not omniscient agents. It is agents that know how
  little they know.
- A beam can be aligned without seeing the target. That should bother people.
- The next generation of simulation tools will not just compute physics. They
  will decide which physical signatures are worth keeping.

- The shadow of chaos can be a compass.
- Three-body dynamics are hard because no closed-form solution exists. Sundog asks whether you need one.
- Full state is 18-dimensional. You need three.
- The most chaotic systems still cast shadows before they break.
- A tidal sensor in a gravitational field is already enough to tell you something useful.

## Product Highlights

### Sundog Core

Highlight:

> A sparse photometric controller aligns a reflected beam without direct target
> coordinates, matching the terminal accuracy of a target-aware analytic
> baseline in the tested MuJoCo setup.

Why it matters:

- proves the program has a measurable core;
- gives researchers a compact artifact to inspect;
- shows the trade clearly: less privileged state, more acquisition time;
- anchors the broader theorem in a real experiment.

### EyesOnly / Gone Rogue

Highlight:

> Gone Rogue shows Sundog as procedural control under occlusion: an agent can
> act from compressed state, plan in batches, and stop when volatility returns.

Why it matters:

- moves the idea from lab controller to live game engine;
- treats partial information as a design feature;
- makes the agent feel reactive without giving it authorial omniscience;
- creates a path toward matched-seed agent studies.

### Dungeon Gleaner

Highlight:

> Dungeon Gleaner shows Sundog as physical compression: glass, reflection, hose
> pressure, kinks, spray, splatter, and torch response become coherent gameplay
> without full physical simulation.

Why it matters:

- converts expensive physical motifs into cheap player-readable systems;
- treats visual and mechanical coherence as the target, not exhaustive fidelity;
- gives the pressure washer a physics signature players can feel;
- opens a route to performance claims once benchmarked.

### Money Bags

Highlight:

> Money Bags shows Sundog as softbody interpretation: rig topology, torsion,
> torque, deformation, symmetry, and recovery become telemetry instead of
> chaos.

Why it matters:

- extends the theorem beyond optics;
- makes frame-by-frame simulation data inspectable;
- gives softbody design a graph vocabulary;
- creates a path toward predicting recovery and controllability from motion
  signatures.

### Three-Body Dynamics Workbench

Highlight:

> In the tested planar restricted setup, a sensor-limited controller using only
> local gravitational field readings improves survival over passive and naive
> local baselines in a robust high-velocity near-escape pocket. Full state is
> 18-dimensional. The controller reads three.

Why it matters:

- pushes the Sundog pattern into a classically hard dynamical regime;
- demonstrates that indirect dynamical signatures (tidal tensor, local
  acceleration) can support useful control in three-body dynamics without
  primary-position or mass access;
- maps the operating envelope and failure boundary explicitly: low-velocity and
  equal-mass cells are known harms, not hidden caveats;
- shows the workbench as a living demonstration that claims discipline and
  honest failure maps are part of the result, not footnotes.

### Pushable Occluder (falsification slate)

Highlight:

> Pushable Occluder is the program's first deliberate falsification surface:
> a pushable block that occludes the beam until moved. A flat photometric
> controller is predicted to fail because the indirect signal does not
> expose the preparatory push as a usable gradient. The verdict surface is
> `BOUNDARY FOUND`, not `BUSTED` — the theorem stands; the method's honest
> upper limit is reached.

Why it matters:

- demonstrates the program is willing to publish a measured negative
  result, not only positive demos;
- clarifies the theorem boundary: indirect signal is not enough when the
  useful gradient appears only after a preparatory action;
- gives the highlights rail its first stamp-interrupting card and earns
  the rail's broader MythBusters rhythm;
- roadmap lives at [`PUSHABLE_OCCLUDER_ROADMAP.md`](PUSHABLE_OCCLUDER_ROADMAP.md);
  rail integration in [`HIGHLIGHTS_RAIL_ROADMAP.md`](HIGHLIGHTS_RAIL_ROADMAP.md).

## Future Bets

These are forward-looking statements. They are useful for pitch language, but
they should be marked as roadmap or research direction.

### Bet 1: Agents Will Need Less World State

Future agents will not always be fed complete world models. They will operate
through compressed, uncertain, and partial views. Sundog is positioned for that
world because it treats indirect feedback as the primary signal, not a degraded
backup.

### Bet 2: Cheap Physics Will Become More Valuable

High-fidelity simulation will remain essential, but many product systems do not
need truth at every layer. They need coherence, controllability, and legibility.
Sundog-style transforms can identify the physical signature worth preserving
and discard the rest.

### Bet 3: Game AI Will Move Away From Omniscience

The most interesting game agents will not be those that know everything. They
will be those that act convincingly from limited knowledge. Occlusion, partial
state, and stop-conditioned planning can make agents feel more alive because
they are not cheating with perfect information.

### Bet 4: Telemetry Will Become A Design Medium

In softbody systems, physics traces are often treated as debugging exhaust.
Money Bags points toward a different future: telemetry as a design surface.
Torsion, deformation, symmetry, and recovery can become the language designers
use to tune feel.

### Bet 5: The Most Interesting Signal Is Often A Derivative

The direct object may be expensive or impossible to observe. The change in its
projection may be cheaper and more useful. This is the spirit of `H(x)`: look
for the relationship between action and environmental response.

## Big Future Claims To Explore

These are intentionally bold. They are not yet proven.

- Sundog could become a general method for designing agents that act under
  occlusion.
- Sundog-style transforms could reduce the cost of selected simulation effects
  by preserving only control-relevant signatures.
- Softbody systems may become easier to tune when interpreted through graph
  alignment rather than raw physics state.
- Procedural games may feel more authored when their agents know less, not
  more.
- The right indirect signal may outperform a noisy direct measurement.
- Future engines may expose "signature layers" alongside physics, rendering,
  and AI.
- The most powerful design move may be choosing what not to simulate.

## The Gravity Claim

This is the most outlandish public framing in the program. It is intentionally
hard to falsify cheaply, and it points at a horizon experiment that would cost
real resources to run. Use the language here only in venues that can also
carry the boundary text below. The full staging ledger is
[`docs/SUNDOG_V_GRAVITY.md`](SUNDOG_V_GRAVITY.md).

### The claim, named tightly

> Sundog is gravity for agents under partial observability: a geometry-derived
> background field that deflects policy without ever being a target.

Sister formulations:

> Reinforcement learning gave agents a reward to optimize. Sundog gives them a
> field to fall through. The first invites Goodhart. The second has no metric
> to corrupt.

> An agent that reads tidal gradients cannot reward-hack gravity. The mass is
> where the mass is.

> Optimization made the agent face the target. Sundog makes the target part of
> the geometry the agent moves in.

### The Goodhart sidestep

Reward optimization places the objective inside the agent — a learned value
function, a critic, a numerical target the agent can corrupt. Signature
control places the objective inside the environment's geometry. The tidal
tensor at a point is a function of the masses, not the agent's policy. You
cannot gradient-hack a tidal field. You can only move through it.

This is the original unhinged intuition of the Sundog program, restated as a
structural argument about threat models rather than a slogan. It is not a
proof of adversarial robustness. It is a falsifiable claim that
signature-driven control and reward-driven control should behave differently
under hostile conditions, with the experiments to test that difference
deliberately listed and deliberately expensive.

### The three-body wedge

The three-body workbench is the audience-conceptualizable entry point because
the gravity is literal. A controller reads the tidal tensor of a real
gravitational potential and maintains regime in a near-escape pocket without
being told where the masses are. The audience does not need to grant a
metaphor — they watch a controller fall through real gravity from indirect
state, and they understand on first contact what is meant by "the field is
the objective."

The three-body controller is the wedge that lets every other Sundog
application — shadow geometry, pressure fields, wake structure, scent
gradients — be read as "gravity for agents in that domain."

### The horizon experiments

Three candidate experiments are staged in `SUNDOG_V_GRAVITY.md`:

- **Spacecraft trajectory under unmodeled perturbation.** A signature
  controller using only local accelerometry holds an orbit family in a
  high-fidelity simulator (and eventually on a cubesat-class deployment)
  against adversarial jamming, ephemeris spoofing, or unmodeled
  solar-radiation events. Matched against a privileged-ephemeris baseline.
  First-priority because it inherits the most three-body discipline.
- **Adversarial signature benchmark.** Two agent families — signature-tracking
  and reward-optimizing — face a named red team across a shared partially-
  observed environment. Primary metric: rate of catastrophic policy failure
  under matched adversary budgets. The cleanest attack on the "your signature
  is just a reward in costume" objection.
- **Side-channel defense (stretch).** A signature-driven monitor reads syscall
  residue, timing geometry, or flow autocorrelation rather than labeled attack
  samples. Active red team. The most outlandish framing because the industry
  is already shipping reward-trained classifiers that are observably being
  Goodharted in production.

### Stir lines that stack into this frame

These already live elsewhere in this document; the gravity framing turns them
from loose provocations into a coherent stack:

- Direct perception is overrated.
- Full state is a luxury, not a requirement.
- The disturbance is the data.
- Occlusion is not failure. It is an interface.
- Alignment is not always a coordinate. Sometimes it is a resonance.
- The future of game AI is not omniscient agents. It is agents that know how
  little they know.
- We are building systems that can infer from the wake they leave behind.

### Boundary language for the gravity frame

Always pair the gravity claim, in any public venue that may reach researchers,
with text like:

> The gravity claim is the program's speculative public frame. The controlled
> evidence is narrower: photometric mirror alignment without target-position
> access in a MuJoCo experiment, and a bounded near-escape operating pocket in
> the planar restricted three-body workbench. The experiments that would
> ratchet the gravity claim into earned language are staged in
> `docs/SUNDOG_V_GRAVITY.md` and have not been run.

## Headlines

- The Shadow Can Be A Sensor
- Alignment Without Sight
- Useful Physics, Not Full Physics
- The Halo Is The Interface
- When Occlusion Becomes Control
- Reading The Wake Of A System
- Less State, More Structure
- The Future Is Not Omniscient
- A Theory Of Useful Partial Information
- From Shadow Geometry To Software Control
- The Disturbance Is The Data
- H(x): The Signal Around The Signal
- Steering By The Shadow Of Chaos
- 18 Dimensions. Three Signals.
- The Three-Body Problem Has A Shadow Too
- Gravity For Agents
- You Cannot Reward-Hack A Tidal Tensor
- The Field Is The Objective
- A Threat Model Without A Target

## Taglines

- Turn indirect signal into control.
- Build agents that can act from the halo.
- Compress physics into usable signatures.
- Make partial information productive.
- Find the structure hiding in disturbance.
- Align without omniscience.
- Simulate what matters.
- Let the world answer back.

## Social Posts

### Short

Sundog asks a simple question with strange consequences:

What if the indirect signal is enough?

The shadow. The torque. The occlusion. The deformation. The wake.

Not full state. Usable structure.

### Medium

We rebuilt Sundog as a research-facing artifact.

The core result is narrow and measurable: mirror alignment without direct target
coordinates, using sparse photometric feedback. The broader program now shows
up in EyesOnly, Dungeon Gleaner, Money Bags, and the three-body dynamics
workbench: procedural agents under occlusion, verb-field NPC behavior,
graph-readable softbody telemetry, and sensor-limited survival in a chaotic
gravitational setting where the controller reads tidal field gradients instead
of full system state.

The future we are chasing: systems that do not need perfect information to
behave intelligently.

### Technical

The Sundog pattern:

1. deny full world-state symmetry;
2. observe an indirect signal;
3. transform that signal into a control-relevant signature;
4. act from the signature;
5. measure the failure boundary honestly.

That pattern now spans photometric alignment, procedural roguelike agents,
verb-field NPC behavior, softbody graph telemetry, and three-body dynamics
control. In the three-body workbench, the agent reads local tidal gradients —
not primary positions, not masses — and improves survival in a near-escape
operating pocket where naive local control fails. The failure boundary is
reported, not hidden.

## Pitch Deck Outline

1. The problem: direct state is expensive, unavailable, or unrealistic.
2. The insight: indirect environmental response can carry actionable structure.
3. The theorem: `H(x)` as the relationship between projection and applied force.
4. The core experiment: photometric mirror alignment without target-position
   access.
5. The trade: comparable terminal accuracy, slower acquisition.
6. The applications: EyesOnly, Dungeon Gleaner, Money Bags.
7. The three-body result: sensor-limited survival in a chaotic dynamical setting,
   with an honest operating envelope and failure map.
8. The future: agents and simulations built around useful partial information.
9. The ask: collaborators, replication, benchmarks, hardware validation, and
   application-specific studies.

## Boundary Language

Use this whenever the promo is likely to reach researchers:

> The broad theorem is a research program. The controlled result currently
> defended in the Sundog repo is narrower: photometric mirror alignment without
> target-position access in a MuJoCo experiment. The three-body workbench is a
> bounded operating-envelope study: a sensor-limited controller using local tidal
> field readings improves survival over passive and naive baselines in a tested
> near-escape pocket; lower-velocity and equal-mass cells remain known harm
> boundaries. EyesOnly, Dungeon Gleaner, and Money Bags are application
> expressions that motivate the next round of controlled studies.

## Closing Lines

- Sundog is not about seeing everything. It is about seeing enough.
- We are building toward software that understands the shape of its own
  disturbance.
- The future belongs to systems that can act intelligently from incomplete
  truth.
- The halo was never decoration. It was the signal.
