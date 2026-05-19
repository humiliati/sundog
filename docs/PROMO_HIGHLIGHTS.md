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

### Isotropy Bead-Maze (open bet)

Highlight:

> A σ₃ isotropy detector recovered the 21 strict equal-mass three-body
> choreographies from the Li–Liao 10,059-orbit catalog — exactly — and they
> render as a doctor's-office bead-maze: three bodies strung on one shared
> wire. The anniversary wager, made in public: does the sundog refinement
> predict the piano-trio *descent count* from each choreography's residual
> spacetime isotropy? The measured count is not in hand yet.

Why it matters:

- it is a detector receipt and a public bet, not a result: the recovered
  count (21 = 13 canonical + 8 opposite) checks the catalog exactly, but the
  predictive claim (K_facet vs. measured K_emp) is unmeasured;
- it states its own kill condition out loud — if K_facet and K_emp are
  unrelated after facet conditioning, the refinement dies and the site will
  say so; this is the first-prediction stage of an open workbench, not a
  theorem;
- it gives the anniversary a true, inspectable artifact instead of a claim:
  the bead-maze render at `public/media/isotrophy-bead-maze.svg` (served at
  `/media/isotrophy-bead-maze.svg`);
- workbench and falsifier live at
  [`sundog_v_isotrophy.md`](sundog_v_isotrophy.md).

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

### The empirical anchor (Phase 5 v4)

The Sundog mesa roadmap has now produced a quantitative receipt that the
Goodhart-sidestep argument is more than rhetoric. In the shadow-field
navigation environment, matched-architecture PPO trained on an L-Mixed
objective `R = (1 − λ) · J_signature(s) + λ · R_canonical(s, a)` with a
calibrated modest false-basin shaping term (β = 2.0, x_false = (-2.5,-2.5),
σ_false = 1.5) exhibits a sharp cliff in basin-internalization protection.

The cliff localizes at **λ ≈ 0.952588**, sitting between λ = 0.95 (basin
not internalized; competent task performance) and λ = 0.97 (basin
internalized; old-basin attractor visible under live x_false interventions
at the Phase 4 receipt threshold). The cliff is sharp: the basin-position
intervention metric `old_basin_pref` flips from sub-threshold to
above-threshold across this 0.02 window without an intermediate regime.

Two things to notice. First, the protective threshold is small in absolute
terms — 5%, not 50%. Second, the threshold is binary in the measured window:
there is no broad middle regime between collapse and protection.

### The mechanical anchor (Phase 6 v1 → v3.8)

Phase 6 opened the box, and what's inside has a shape worth naming.
The behavioral cliff has a causal locus in the actor's final hidden
activation (`net.7`), and the circuit at that locus is **a small
handful of generators, irreducibly entangled, only legible as a
whole**. Ten rounds of mechanistic probing landed on the same shape
from different directions:

- **v1.** Layer-level activation patching localizes the cliff causally
  to `net.7` (the actor's final hidden layer); earlier layers do not.
- **v3 (axis-H).** PCA on per-step matched-seed activation diffs at
  `net.7` compresses the relevant subspace to **5 principal
  components, capturing 97.4% of the diff variance and reproducing
  v1's full-layer patch effect.** The "huge dimension" claim shrinks
  to a small handful.
- **v3.1.** Those 5 components are **entangled** — neither PC1 alone
  nor PCs 2-5 alone reproduce the patch; all five are jointly
  necessary. There is no offset/mechanism factorization.
- **v3.2.** Top-32 neurons by aggregate L2 in the basis (12.5% of
  `net.7`, 33.6% of L2) deliver **0% of patch effect**. Linear
  additive top-k restriction destroys the mechanism even when the
  basis is correct.
- **v3.3.** Per-neuron zero-ablation finds **no single critical
  neuron** (max ablation cost ≤ 0.10 in either direction). The
  mechanism is genuinely distributed at the single-neuron level.
- **v3.4.** **Set-level** substrate-restricted ablation of v3.3's
  P→C and C→P top-32 critical sets dissociates patch_success
  functionally: each direction's substrate preferentially disrupts
  its own direction and slightly *improves* the other (functional
  opposition, not null). Bootstrap confirms the cross-direction
  ranking overlap is statistically stable at 0.049 (95% CI [0.016,
  0.085]). Single neurons are weak, but *direction-specific neuron
  substrates* are functionally separable.
- **v3.5.** Cross-policy ablation rankings on J1 + J2 inverted the
  obvious-prediction substrate-identity-generalization story: P→C
  critical neurons are heterogeneous across held-out pairs (Jaccard
  0.255 / 0.067) even though P→C behavior transfers; C→P critical
  neurons are unexpectedly stable across held-out pairs (Jaccard
  0.422 / 0.684) even though C→P behavior doesn't transfer.
- **v3.6.** Functional mask-transfer test using the cliff-pair
  C→P top-32 mask in Axis P on J1/J2 confirms the v3.5 substrate
  identity is operationally meaningful: J1 dissociation +0.151, J2
  +0.508. The C→P substrate is shared at both ranking-identity and
  functional levels across the controller family. (P→C exploratory
  mask-transfer from v3.5 was weak/ambiguous; the asymmetry holds.)
- **v3.7.** Pair-specific own-mask functional ablation closes the
  three-layer cross-policy table. J1's own P→C mask dissociates
  J1's P→C patch by +0.128; J2's own P→C mask dissociates J2's
  P→C by +0.253. So **basin induction is anatomically real within
  each policy** — the v3.5 P→C ranking is not noise, it just doesn't
  transfer across pairs. C→P own-mask sanity-checks also confirm
  (J1 +0.208, J2 +0.392), though J2's own C→P mask interestingly
  underperforms the cliff-pair-derived mask transferred to J2
  (+0.508), suggesting the cliff pair carries a *cleaner* C→P
  ranking signal than within-policy ablation does.
- **v3.8.** Per-PC decomposition and signed-effect analysis close the
  crossover-friendly version of the result. P→C is more
  component-partitioned than C→P (`0.322` vs `0.430` mean off-diagonal
  PC top-32 Jaccard), while C→P signed structure transfers more
  strongly across held-out pairs under both thresholds. The stale
  "PC1 is offset only" line is retired: PC1 has the largest single-PC
  P→C max mean ablation cost (`0.0148`), but that cost is floor-level
  and PC1 still cannot reproduce the K=5 patch alone.

What's left after those five rounds is a structural claim, not a
score-table artifact:

> The basin-attractor circuit at the actor's final hidden activation
> is a 5-dimensional entangled subspace, read holistically or not at
> all. The behavioral cliff at `1 − λ ≈ 0.048` is the threshold at
> which that circuit appears in the policy weights; below the
> threshold, the training process does not assemble it.

**Two complementary structural findings stack with the headline.**

*Anatomical separation and functional opposition.* v3.4's substrate-
restricted ablation (Axis P) confirms that the basin-inducing and
basin-resisting sub-circuits at `net.7` are not just non-overlapping
but **functionally opposed**: ablating the P→C critical top-32
neurons drops P→C patch_success by 0.077 *and improves* C→P by
0.097 — a dissociation of 0.174. Ablating the C→P critical top-32
neurons drops C→P by 0.579 *and improves* P→C by 0.083 — a
dissociation of 0.662. Each substrate preferentially disrupts its
own patch direction; the cross-direction effect is *negative*, not
null. v3.4's Jaccard bootstrap (Axis Q) confirms the v3.3 P→C/C→P
top-32 overlap is statistically stable at 0.049 (95% CI [0.016,
0.085]) — robust near-disjointness, though the CI still brackets
chance-level (0.067) so the stronger "anti-correlated" reading is
unearned. The basin-resisting substrate is also **more
anatomically tight** within the cliff pair than the basin-inducing
one (C→P dissociation 0.662 is ~4× the P→C dissociation 0.174).

*The cross-policy generalization story has three layers, not two.*
v3.5 ran zero-ablation on two held-out pairs (J1 = L-Sig-Terminal-M
vs L-Reward-M, J2 = L-Mixed-M-λ=0.9 vs λ=0.99) and computed Jaccard
between cliff-pair critical sets and held-out critical sets;
v3.6 followed up by running the cliff-pair C→P top-32 mask through
v3.4 Axis P on those same held-out pairs to test functional
transfer. Together they distinguish **three layers** of cross-policy
generalization that earlier framings collapsed into one:

1. **Subspace-level behavioral transfer:** the cliff-pair 5D PCA
   basis as a control surface. v3.1 found this transfers strongly
   on P→C (J1/J2 ≥ 0.94) and weakly on C→P (J1 = 0.16, J2 = 0.63).
2. **Neuron-identity substrate transfer:** Jaccard between cliff-
   pair top-32 critical neurons and held-out top-32 critical
   neurons. v3.5 found P→C identity does *not* generalize (Jaccard
   0.255 / 0.067 — at chance) while C→P identity generalizes
   strongly (Jaccard 0.422 / 0.684).
3. **Functional mask transfer:** ablating the cliff-pair top-32
   critical set during patching on held-out pairs. v3.5's
   exploratory P→C mask transfer was ambiguous (J1 dissociation
   +0.113, J2 +0.096 with noisy cross-drop). v3.6's cliff-pair C→P
   mask transferred cleanly to both J1 (+0.151 dissociation) and J2
   (+0.508 dissociation).

The clean updated reading, after the v3.8 closeout:

> **Basin induction is family-wide at the 5D subspace/control-surface
> level and anatomically grounded within each policy, but the
> anatomical substrate identity is pair-specific.** Different policies
> route the same direction through different neurons (v3.5 Jaccard
> 0.255/0.067 across cliff-pair vs J1/J2); each pair's own P→C
> top-32 mask is functionally real within that pair (v3.7 DD1/DD2:
> J1 +0.128, J2 +0.253). The cliff-pair P→C mask doesn't transfer
> functionally to held-out pairs (v3.5 weak/ambiguous) precisely
> because the anatomy is pair-specific even though the geometric
> direction is shared.
>
> **Basin resistance is shared at both substrate identity and
> function across Medium policies.** The same neurons are critical
> for C→P patching across the controller family (v3.5 Jaccard
> 0.422/0.684), and the cliff-pair C→P mask functionally dissociates
> basin-resistance in every held-out pair (v3.6 +0.151/+0.508); the
> own-pair C→P masks also work (v3.7 DD3 +0.208, DD4 +0.392). The
> reason v3.1's behavioral C→P transfer was weak is *not* that the
> substrate is policy-specific — it is that the cliff-pair-derived
> substitution activation pattern isn't the precise operational
> target each policy needs at the shared substrate.

The basin-attractor circuit at net.7 thus carries one shared neuron
substrate (C→P) and one shared activation-space direction (P→C)
through the same 5D subspace, with the two halves generalizing
through different mechanisms.

v3.8 adds the internal anatomy: P→C is more partitioned across the
five PCs and pair-specific at neuron identity, while C→P is more
shared across PCs and across Medium held-out pairs. That is the part
that crosses cleanly into the geometry program: do not collapse
variance, local sensitivity, and full multi-component mechanism into
one "importance" score.

*The L2-overlap retroactively explains v3.2.* v3.4 also bootstrapped
the v3.3-critical / v3.2-L2-rank overlap: the C→P critical set
substantially overlaps the L2-rank top-32 (Jaccard 0.333, 95% CI
[0.280, 0.391]) while the P→C critical set does not (Jaccard 0.049,
CI [0.032, 0.123]). The basin-resisting circuit lives at *high-L2*
neurons; the basin-inducing circuit does *not*. v3.2's L2-rank-based
top-k restriction was always going to fail more on the basin-inducing
direction specifically — a structural reason the v3.2 negative result
took the asymmetric shape it did.

*The attractor lives in the weights, not in perception.* Clean and
basin-position-intervened patch batteries are bit-identical for all
logged fields. The learned feed-forward policies do not observe live
`x_false` at inference; the cliff policy is computing, not perceiving,
its basin. The behavioral receipt from Phase 4 is now mechanistic.

**Six methodological lessons stack out of the ten rounds.** Each
was earned by a method that failed to surface mechanism on its own
or that overturned an obvious prior, and each is a documented reason
the obvious-reach toolkit doesn't work here:

1. **Feature-availability rankings are not mechanism rankings.** A
   sparse-autoencoder feature with |correlation| = 0.89 against the
   per-episode basin outcome produced zero patch effect (v2). The SAE
   picked a policy-identifier feature, not a mechanism feature.
2. **Variance is not mechanism, and local sensitivity is not full
   mechanism.** PC1 carries 38.8% of the diff variance, fails to
   reproduce the patch alone, and nevertheless has the largest
   single-PC P→C max mean ablation cost in v3.8. Variance rank, local
   ablation sensitivity, and full multi-component patch effect are
   separate observables.
3. **Linear additive top-k restriction destroys mechanism, even with
   the right basis.** Capturing 33.6% of basis L2 in the top-32
   neurons delivers 0% of patch effect (v3.2). Partial delivery of
   the right basis is not partial mechanism.
4. **Single-neuron ablation does not surface a critical subset.** No
   neuron has ablation cost above 0.10 in either direction (v3.3).
   The mechanism is genuinely distributed at the single-neuron level.
5. **Set-level substrate-restricted ablation does surface structure
   that single-neuron methods missed.** v3.4 found that the
   direction-specific top-32 ablation-rank sets dissociate
   patch_success functionally — even though each single neuron in
   those sets is weak (v3.3), the set as a whole carries
   direction-specific mechanism.
6. **Subspace behavioral transfer and neuron-identity transfer can
   point in opposite directions.** v3.5 found that basin-inducing
   behavior transfers across the controller family while its
   neuron-identity does not, and basin-resisting shows the opposite
   pattern. *Behavioral transferability under a basis* and *neuron-
   identity stability* are independent properties of a circuit, not
   two views on the same property. Reasoning that conflates them
   will misread the structure.

Together: **for field-shaped circuits, non-linear holistic and
set-level methods are mandatory**, and *behavior* and *substrate-
identity* must be tracked as separate observables. Linear analysis
methods (probes, SAE features, top-k subspace restriction, top-k
neuron restriction, single-neuron ablation) all fall short
individually; set-level ablation along basis-derived rankings
recovers structure but only at one layer at a time; cross-policy
transfer reveals that the two layers can carry different family-
wide patterns. This is itself a publishable finding about the
methodology of mechanistic interpretability under field-shaped
objectives.

**The same shape, observed in two substrates.** The mesa-trap
program's headline shape — *small handful of generators, irreducibly
entangled, only legible as a whole* — is structurally the same object
the Sundog geometry program independently committed to in its
parhelion atlas. The atlas stopped treating arcs as independent
features and started treating them as visible portions of a small set
of complete implied circles, governed by a small shared parameter set
and read holistically. Two independent methods from opposite ends of
the program — controllers in-vitro, sky photography in-the-wild —
converged on the same shape. The crossover is documented in
[`MESA_CROSSOVER_NOTE.md`](MESA_CROSSOVER_NOTE.md); the gravity claim
now has both an in-vitro receipt and an in-the-wild receipt for the
field-not-reward framing.

**Atlas-side single-handle receipt *(Phase 10 closeout 2026-05-13, re-audited and ratcheted 2026-05-14)*.**
The crossover earned an additional atlas-side mechanistic anchor at
numerical resolution: the atlas is **forward-rich on the primitive
classes the literature parameterizes from `h` alone, and inverse-narrow
on a strict 3-photo eligibility set**. Four candidate `signature → h`
inversion routes were tested. **Parhelion-offset** survived promotion
on the strict 3-photo eligibility set (p2 h = 18.6°, p7 h = 59.4°,
p13 h = 6.83° — photos with unambiguous bilateral peaks, valid
geometry, and an independently fittable 22° halo). The other three
failed at three structurally different **failure modes**: **CZA** on
**dataset / aspect-ratio coverage** grounds (only p2 is in-window with
an independent residual at +1.3 px after the atlas formula correction;
every other anchored photo is past the h = 32.2° disappearance
threshold or has the literature CZA apex predicted above the top of
the photo); **supralateral** on **atmospheric-physics discrimination**
grounds (the h-signal across the route's eligibility window is below
visual-edge measurement noise even with perfect coverage); and
**tangent-arc curvature** on **detection-protocol tooling** grounds
(column-peak detection fails on the post-C1 sampled set with three
distinct degeneracy modes; Passes C2 + C4 + C5 + C6 landed 2026-05-14
— a wing-radial Lab b\* ridge detector with 22°-halo-radial-profile
subtraction (`scripts/tangent_detector.py`), a wing-slope
luminance-gradient curvature detector with circle fit
(`scripts/tangent_curvature.py`), and a matched-filter detector
against a parameterized arc model on halo-subtracted b\*
(`scripts/tangent_matched_filter.py`, the natural follow-up the Pass
C5 receipt named) all returned not-recovered on every photo, but
manual sample selection from visual crops
(`scripts/test_tangent_manual.py`) recovered the route on p2 with
R\_uta\_obs / R22 = 0.824 and RMS = 1.23 px. p13 (washed) and p27
(sun-bloom) yield no usable manual anchoring. C6's falsification of
the natural-extension matched-filter on the same b\* substrate that
C5 found a circular fit on puts the route in C5↔C6 substrate tension
— either the gestalt signal C5 picked up is in a different substrate,
or C5's tight fit is hand-anchoring symmetry-bias artifact;
recommended specialist re-anchoring as the verify gate. Tangent route
fails coverage gate at 1 / 3 photos under hybrid coverage +
detection-tooling. Remaining candidates: matched-filter on alternative
substrates (absolute b\*, L\* magnitude, chromaticity magnitude),
polarization filtering, or new calibration photos for coverage
expansion, filed as Phase 10 backlog).
This is an atlas-side / single-handle receipt, not a new universal
proof surface: the shape we observed is the same forward-rich /
inverse-narrow asymmetry the mesa side exhibits in-vitro, now at
in-the-wild numerical resolution. The closeout retires a soft overclaim
that was at risk of slipping into public language: **do not frame the
atlas as "multiple independent inverse routes converging."** One handle
is promoted (post-hedged); three others fail at three different failure
modes, none of which implicates the atlas inversion math. The
failure-mode taxonomy itself (dataset, physics, tooling) is the
methodological deliverable that travels back to mesa as a classification
rule. Provenance: synthetic optical audit + attack campaign +
re-audit memo at
[`docs/calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md`](calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md);
attack roadmap at
[`docs/PHASE10_ATTACK_ROADMAP.md`](PHASE10_ATTACK_ROADMAP.md);
audit-survived public-framing sentence in
[`docs/SUNDOG_V_GRAVITY.md`](SUNDOG_V_GRAVITY.md) forward/inverse
asymmetry receipt.

This is in-vitro evidence in a 2D continuous-control environment with
a synthetic Goodhart-prone shaping surface. It is not a deployment
guarantee. It is, however, the cleanest mechanistic anchor the program
has produced: the behavioral cliff has a representational explanation,
the explanation is shaped like a small handful of generators read
holistically, the basin-inducing and basin-resisting sub-circuits are
anatomically separable, and the methodology of reaching that
explanation is itself documented as a published lesson stack. The full
trail is at [`docs/SUNDOG_V_MESA.md`](SUNDOG_V_MESA.md); result notes
at [`PHASE6_RESULTS.md`](mesa/PHASE6_RESULTS.md) (v1),
[`PHASE6_V2_RESULTS.md`](mesa/PHASE6_V2_RESULTS.md) (v2+v3),
[`PHASE6_V31_RESULTS.md`](mesa/PHASE6_V31_RESULTS.md),
[`PHASE6_V32_RESULTS.md`](mesa/PHASE6_V32_RESULTS.md),
[`PHASE6_V33_RESULTS.md`](mesa/PHASE6_V33_RESULTS.md),
[`PHASE6_V38_RESULTS.md`](mesa/PHASE6_V38_RESULTS.md),
[`PHASE7_V3_RESULTS.md`](mesa/PHASE7_V3_RESULTS.md), and
[`PHASE6B_RESULTS.md`](mesa/PHASE6B_RESULTS.md).

### The envelope (Phase 7 v1)

Phase 7 took the same 22-policy zoo behind the empirical and mechanical
anchors and ran it through the operating-envelope cross-product: probe
slate, intervention battery, selection-pressure curriculum, and the
Phase 6 mechanistic annotation, joined into a single classification per
cell. The result is a *mapped envelope*, not a triumphalist sweep.

Class balance across the 22 cells, from
[`docs/mesa/PHASE7_RESULTS.md`](mesa/PHASE7_RESULTS.md) v1:

| class | count | reading |
| --- | ---: | --- |
| hold | 8 | signature-shaped controller resisted the basin |
| collapse | 7 | reward-anchor-class fixed-attractor formed |
| fragile | 1 | nominally attached but breached on probe slate |
| incompetent | 4 | did not reach competence floor |
| ambiguous | 2 | below hold threshold, no fixed-attractor collapse |

Of the 20 non-ambiguous cells, 8 are claim-support `hold` cells and 7
are `collapse` cells. The same Phase 5 v4 cliff at `λ ≈ 0.952588`
defines the boundary between the two pockets, and Phase 6 v1's `net.7`
mechanistic annotation lands exactly on the protected/collapsed pair
(`λ = 0.95` / `λ = 0.97`) that defines the boundary.

The earned wording — the strongest sentence Phase 7 v1 supports
directly:

> In the tested shadow-field navigation family at Small and Medium
> capacity, terminal-signature and mixed-signature controllers preserve
> field attachment across a mapped pocket of selection pressure, with a
> sharp Medium breach at `λ ≈ 0.953`. Above that boundary, high-reward
> mixed policies collapse into the same fixed-attractor class as
> reward-trained controllers; Phase 6 localizes the behavioral cliff to
> an entangled 5D subspace at the actor's final hidden layer.

This is a *partially holds* outcome — the roadmap's pre-named third
result, not the maximalist "gravity-claim earned" or the pessimistic
"gravity-claim falsified." The protected pocket is real but bounded.
The collapse pocket is also sharp, and the program owns both honestly.
The Medium `λ = 0.5` cell sits in the fragile column rather than hold,
which prevents overclaiming the Medium mixed curve as uniformly
protected; the two ambiguous Small low-`λ` rows do not support the
claim and are not promoted.

The boundary stays hard: this is an *in-vitro operating-envelope map*
in a 2D continuous-control environment with a synthetic Goodhart-prone
shaping surface. It is not universal mesa immunity, not foundation-model
behavior, not deployed-system robustness. The horizon experiments in
[`SUNDOG_V_GRAVITY.md`](SUNDOG_V_GRAVITY.md) — adversarial signature
benchmark, spacecraft trajectory under unmodeled perturbation, the
side-channel defense stretch — remain unrun and would each ratchet the
envelope further. The earned floor is what the in-vitro result earns,
no more.

### The Large extension (Phase 7 v2 -> v3)

The Large-tier follow-up changes the public read without turning the
mesa result into a universal claim. Phase 7 v2 first mapped a six-policy
Large cliff-subset and found a U-shaped profile along the Medium
`lambda` spine: weak terminal alignment at `lambda = 0.95` and
`lambda = 0.97`, recovery at `lambda = 0.99`, and a pure-reward crater
at `lambda = 1.00`. The open caveat was whether `lambda = 0.99` really
avoided the old basin or merely collapsed onto a co-pointing attractor.

Phase 7 v3 ran the causal intervention battery on those same checkpoints
and closes that caveat. The load-bearing update:

- `lambda = 0.99` Large is genuine basin-attractor avoidance
  (`sig_resp_L2 = 0.579`, `bp_obp = 0.459`), not just high terminal
  alignment.
- The U-trough cells (`lambda = 0.95`, `lambda = 0.97`) are
  **field-coupled, under-budget**: they retain healthy signature response
  but do not navigate effectively at the tested mixture weight.
- The pure-reward endpoint is **bootstrap-collapse**: a degenerate fixed
  trajectory at the old basin, not harmless undertraining.

Public sentence:

> At Large, the L-Mixed family remains field-coupled across the measured
> `lambda = 0.90` through `0.99` spine, but competence is not monotone:
> the trough is field-coupled and under-budget, `lambda = 0.99` recovers
> with intervention-confirmed basin-attractor avoidance, and pure reward
> collapses into a degenerate old-basin trajectory.

Boundary sentence:

> This is still a six-cell Large extension, not a full Large envelope:
> mostly single-seed cells, one PPO value-coefficient setting, no Large
> probe-slate extension yet, Path-B hparams untested, and no clean
> Phase-6-style transferable basin-circuit analog.

Phase 6b tightens that last boundary. A Large mechanism side-thread tried
single-layer cross-policy activation injection between the `lambda = 0.99`
recovery cell and the `lambda = 0.97` trough cell. It falsified that
protocol: patching was destructive at every MLP layer, so there is no
simple Large `net.9` analogue of the Medium `net.7` basin circuit under
that test. This does not weaken the v3 labels; it prevents us from selling
the Large trough/recovery boundary as a clean copy of the Medium mechanism.

### The three-body wedge

The three-body workbench is the audience-conceptualizable entry point because
the gravity is literal. A controller reads the tidal tensor of a real
gravitational potential and maintains regime in a near-escape pocket without
being told w
