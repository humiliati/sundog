# Claims and Scope

This document defines what the Sundog Project can safely claim and what it should avoid claiming.

This project lives at the edge between invention, research, and mythic framing. A claims policy document is essential for maintaining credibility while preserving ambition.

## The Core Principle

**You can keep the fire in the design language and manifesto.**

**You must be precise in the research claims.**

## Safe Claims

These claims are defensible with current evidence:

### Core Experiment
- We developed a framework we call the Sundog Theorem / Sundog framework.
- We have been using it in software applications for a year.
- A photometric controller can align a mirrored end-effector without direct target coordinates, using only sparse photometric feedback and proprioception.
- The controller reaches comparable terminal accuracy to a target-aware analytic baseline in the tested MuJoCo setting; no terminal-intensity difference was detected at `n=30` (`U=526`, `p=0.264`).
- The cost of indirect feedback is convergence time, not terminal accuracy, inside the tested operating envelope; acquisition is roughly 16x slower in the reported core run.
- The known failure boundary is tight joint limits.

### Framework Characteristics
- The framework appears useful in partially observed systems.
- It has informed specific implementations in game AI, procedural generation, simulation analysis, and rendering/physics-adjacent workflows.
- The pattern involves denying full world-state access, observing indirect signals, transforming them into control-relevant signatures, and acting from those signatures.

### Applications
- EyesOnly/Gone Rogue demonstrates a headless turn-envelope runner operating
  against a real procedural roguelike through compressed perception,
  stop-conditioned action batches, and a seedable Playwright bridge.
- Dungeon Gleaner demonstrates verb-field NPC behavior: unmet needs diffuse across satisfier nodes to produce lightweight idle orbits without scripted planners.
- Money Bags demonstrates graph-based interpretation of softbody motion, making torsion, deformation, symmetry, and recovery metrics legible.
- The three-body workbench demonstrates a bounded operating-envelope result: in the tested planar restricted setup, the high-velocity near-escape survival pocket is real, but Phase 18 reduces its mechanism to radius-gated inward thrust at matched duty rather than sophisticated tidal steering, while lower-velocity and equal-mass cells remain known harm boundaries.
- The Balance workbench demonstrates a bounded operating-envelope result: in the tested browser cart-pole setup, a shadow-derived controller beats naive shadow-centering inside the diagnostic-positive envelope, while overhead-light and high-delay cells remain degradation boundaries.
- The Pressure Mines workbench demonstrates a bounded operating-envelope result: in the named density 0.16 / pressure-noise 2.0 / dropout 0.2 pocket, pressure-derived Sundog variants improve budget-adjusted safe-tile progress over naive pressure before mine trigger, while a paired density 0.22 / pressure-noise 1.0 / dropout 0.35 region is published as a failure case.
- The K_facet v0.3h verdict resolves 20 strict catalog rows as structural zeros on the strict G.2 single-curve choreographies at m₃ = 1. One row (O_617) is held back as a named quarantine — a clean opposite-strict row whose bridge direction sits outside the valid D₃ representation. Audit chain intact; theorem-facing result is not closed.
- The geometry workbench shelf hosts the cap-set primer (1980s–2016 polynomial-method precedent) and the unit-distance overlay (a plain-English read of OpenAI's 2026 disproof). These are outside-result explainers, not Sundog evidence claims.

### Research Status
- We are continuing to formalize the math and collect examples.
- The theorem is broad, but the current paper claim is narrow enough to attack.
- The paper-grade defensible result is the photometric alignment experiment; the broader theorem remains a research program now supported by three audit-chain receipts (structural-failure boundary map, mesa-trap operating envelope, K_facet v0.3h verdict), bounded operating-envelope studies, and application prototypes.

## The Audit-Chain Discipline

Sundog publishes a named class of artifact called the **structural-zero receipt** (and its siblings: named quarantine, closed-form separability, bounded operating envelope). Each receipt comes from a pre-registered audit chain — constants and outcome categories registered in code and spec before any row is interpreted — so the failure modes are visible before the result is.

There are three artifacts in the public ledger:

- **Structural-failure boundary map** (`/structural-failure`) — pre-registered five-locus falsifier for closed-form traceability. P0/P1 passed; Cut 2 closed-form separability held; Cut 3 still open.
- **Mesa-trap operating envelope** (`/mesa`) — in-vitro cliff at λ ≈ 0.953 on 22 audited policy cells; mechanistic 5D locus at `net.7`. Phase 7 v3 extends the map to Large; the Large subset is still bounded.
- **K_facet v0.3h verdict** (`/isotrophy`) — 20 of 21 strict G.2 single-curve choreographies returned structural-zero receipts at m₃ = 1; one row (O_617) was held back as a named quarantine for a bridge direction outside the valid D₃ representation. Audit chain intact; theorem-facing result is not closed.

Safe public language for any of these artifacts:

- Name the artifact class (structural-zero receipt, named quarantine).
- Name the falsification surface (the pre-registered guard set or boundary map).
- Name the quarantine separately from the prediction it would have affected.
- Refuse to absorb out-of-scope rows into a single number; refuse to collapse 20/21 into "21/21."

The discipline is the load-bearing claim. The K_facet v0.3h ledger (`docs/isotrophy/SUNDOG_V_ISOTROPHY_KFACET.md`) lists six standing rules; the most load-bearing two for public copy are Standing Rule 5 (no "21/21" framing) and Standing Rule 6 (no weak-admission framing for O_617).

## Risky Claims

**Avoid these claims** — they are not yet supported:

### Unsupported Universal Claims
- ❌ "We discovered a new law of nature."
- ❌ "This proves X universally."
- ❌ "This solves alignment." (too broad)
- ❌ "This replaces classical AI."
- ❌ "This is a revolutionary new branch of mathematics."
- ❌ "We have proven the general theorem."
- ❌ "This works in all domains."
- ❌ "Sundog eliminates the need for direct state access."
- ❌ "Sundog sidesteps Goodhart's law." (universal-form not earned; in-vitro mesa
  operating-envelope evidence is partial — see §The Gravity Frame → Earned
  envelope language for the bounded form that is defensible)
- ❌ "Sundog is gravity for agent optimization." (speculative public frame; see Gravity Frame section)
- ❌ "Signature controllers cannot be reward-hacked." (universal-form not earned;
  in-vitro mesa evidence shows the controller-family *can* be reward-hacked
  above a measured selection-pressure threshold, while remaining protected
  below it — see §The Gravity Frame → Earned envelope language)
- ❌ "Sundog is robust under hostile environments." (no adversarial benchmark has run)
- ❌ "K_facet v0.3h closes the isotropy theorem." (the audit chain is intact; the theorem-facing result is not closed — Standing Rule 5: never collapse the 20/21 verdict to "21/21")
- ❌ "O_617 was a weak-admission failure." (Standing Rule 6: O_617 is a clean opposite-strict row with admission residual 1.01e-8; the defect lives in the bridge representation outside the valid D₃ representation, not in admission and not in the Γᵢ audit chain)
- ❌ "v0.3h proves choreography isotropy." (it resolves 20 strict G.2 rows as structural zeros at m₃ = 1 — a specific, defensible, narrow finding, not a universal theorem)

### Overpromising on Applications
- ❌ "EyesOnly proves the theorem for procedural games." (needs formal study)
- ❌ "The EyesOnly playtest agent demonstrates Sundog under tighter UI constraints." (its scope is UX regression automation)
- ❌ "EyesOnly performs Sundog-driven level manipulation." (LAGM is a design doc, not shipping code)
- ❌ "Dungeon Gleaner proves verb-field diffusion outperforms GOAP for town simulation." (needs telemetry and a comparison harness)
- ❌ "Dungeon Gleaner proves a one-twelfth-cost light or physics theorem." (old framing; do not repeat as Sundog evidence)
- ❌ "Money Bags proves softbody alignment is solved." (needs controlled experiments)
- ❌ "Sundog solves the three-body problem." (Phase 11 is a bounded control-workbench result, not a global dynamics solution)
- ❌ "The three-body controller predicts chaos." (the result concerns survival improvement in a tested operating pocket)
- ❌ "Sundog Balance proves shadow-based robotics control." (Phase 10 is a browser operating-envelope study, not hardware validation)
- ❌ "Sundog Balance works under overhead light." (the overhead-light cells are all-fail degradation margins, not usable control)
- ❌ "Sundog solves Minesweeper." (Pressure Mines reports safer progress before failure in one named pocket, not board solution or logical deduction)
- ❌ "Pressure Mines clears the field." (both Sundog and naive still trigger mines on every seed in the confirmed pocket)
- ❌ "Our applications demonstrate universal applicability."

### Name and Source Confusion
- Avoid implying that this repository is a port, fork, preservation layer, or p-system continuation of *SunDog: Frozen Legacy*.
- Avoid treating Atari-era constraints, UCSD p-system details, or `laanwj/sundog` as architecture for this repository.
- Avoid describing Sundog as a crypto project or as related to SUNDOG token ecosystems.
- Avoid reading the `humiliati` GitHub owner name into a medieval, religious, or moral genealogy for the public brand.
- Safe correction: "This repository's active project is Alignment Without Sight: hidden-state control through indirect signatures."

### Premature Certainty
- ❌ "The mathematics are complete."
- ❌ "We have definitively shown..."
- ❌ "This is the final answer to..."
- ❌ "No other approach can..."

## Better Replacements

Replace risky language with measured alternatives:

| Instead of | Say |
|------------|-----|
| "revolutionary" | promising, novel, emergent, useful |
| "proves" | suggests, demonstrates, indicates |
| "solves" | addresses, informs, helps with |
| "always works" | works in tested contexts, appears effective in |
| "new law of nature" | new framework, novel pattern, emergent approach |
| "definitely" | experimentally validated in our applications, early but compelling |
| "all systems" | certain systems, systems with partial observability |
| "eliminates need for" | reduces reliance on, can operate with less |
| "Sundog solves alignment" | "the gravity frame is the program's speculative public framing; see [`SUNDOG_V_GRAVITY.md`](../SUNDOG_V_GRAVITY.md)" |
| "Sundog sidesteps Goodhart" | "a structural argument that signature-driven control inhabits a different threat model than reward optimization; the adversarial experiments that would test it are listed and have not been run" |
| "agent optimization in hostile environments" | "signature-driven control in partially-observed and potentially adversarial settings; current evidence is bounded operating-envelope work in three-body, Balance, Pressure Mines, and photometric experiments" |

## Claim Boundaries by Application

### Sundog Core Repo
**Safe:**
"A controller with no Cartesian access to a target can align a mirrored end-effector using only sparse photometric feedback. In the tested MuJoCo setting, no terminal-intensity difference was detected against a target-aware analytic baseline at `n=30`; the cost was roughly 16x slower acquisition."

**Boundary:**
"The defensible scientific claim is the photometric mirror-alignment experiment. The broader theorem language represents a research program."

### EyesOnly / Gone Rogue
**Safe:**
"EyesOnly's headless Gone Rogue runner is an instrumented prototype: it drives the real JavaScript game engine through `GoneRogue.headless`, compresses game state into a typed perception payload, selects policy axes, and executes stop-conditioned action batches. The apparatus is seedable and policy-pluggable, but the matched-seed comparison study has not run."

**Avoid:**
"EyesOnly proves the theorem for procedural games."

Also avoid treating `playtest-agent.js` as Sundog evidence; it is UI regression automation. Treat Live Agentic Game Moderation as conceptual lineage / forward-looking design until code ships.

**What's needed:**
Matched-seed multi-policy study, compressed-perception ablation, volatility-threshold sweep, behavior clips, and JSONL telemetry bundles that can later feed the LAGM design.

### Dungeon Gleaner
**Safe:**
"Dungeon Gleaner uses verb-field diffusion for NPC idle behavior: unmet needs are scored against nearby satisfier nodes, and NPCs take local steps toward the strongest pull. This is a product expression of the Sundog indirect-signal-to-action pattern, not yet a controlled benchmark."

**Avoid:**
"Dungeon Gleaner proves verb-field diffusion outperforms GOAP for town simulation."

Also avoid repeating the old glass/window one-twelfth-cost or pressure-washing framing as Sundog evidence.

**What's needed:**
Orbit telemetry, GOAP-substitution comparison, tuning-sensitivity sweeps, and archetype distinguishability metrics.

### Money Bags
**Safe:**
"Money Bags extends Sundog from optical alignment into graph interpretation of softbody motion, with playtest telemetry already capturing alignment, torsion, deformation, symmetry, and recovery signals."

**Avoid:**
"Money Bags proves softbody alignment is solved."

**What's needed:**
Formal alignment score definition, frozen terrain fixtures, matched disturbance scripts, comparison against raw telemetry.

### Three-Body Dynamics Workbench
**Safe:**
"In the tested planar restricted setup, the high-velocity near-escape survival pocket is real, but Phase 18 shows the mechanism is a radius-gated inward reflex at matched duty, not sophisticated tidal steering. The result is not global: lower velocity and equal-mass boundary cells still expose controller harms."

**Avoid:**
"Sundog solves the three-body problem."

Also avoid presenting the privileged heuristic oracle as an optimal controller. It is a comparison reference, not a ground-truth optimum.

**What's needed:**
Public Phase 11 charts, an explicit failure-boundary view, longer-horizon stress tests, and a physical-sensor validation plan before making broader control claims.

### Sundog Balance Workbench
**Safe:**
"In the tested browser cart-pole setup, the Sundog shadow controller beats naive shadow-centering inside the diagnostic-positive operating envelope. Phase 10 confirmed the claim with 28/28 diagnostic-positive yes-cells, zero hard failure-boundary violations, and a repaired oracle ceiling that reads survival on uncapped cells and hidden-angle RMS on capped cells."

**Avoid:**
"Sundog Balance proves shadow-based stabilisation works for robotics."

Also avoid presenting P2b overhead-light all-fail survival margins as usable overhead-light control. They are degradation behavior: both controllers fail, with Sundog merely lasting longer under confidence gating.

**What's needed:**
Public Phase 11 figures, a best-cell/worst-cell replay pair, the toyful visual skin or a clearly diagnostic promo clip, human-vs-agent competition mode, and later physical-sensor validation before robotics claims.

### Sundog Pressure Mines Workbench
**Safe:**
"In the named density 0.16 / pressure-noise 2.0 / dropout 0.2 pocket, pressure-derived Sundog variants improve budget-adjusted safe-tile progress over naive pressure before mine trigger. The public artifact pairs that confirmed pocket with a failure region where naive pressure wins."

**Avoid:**
"Sundog solves Minesweeper."

Also avoid implying field clearance. In the confirmed pocket, both Sundog and
naive still trigger mines on every seed; the measured result is safer progress
before failure under the pre-registered budget-adjusted metric.

**What's needed:**
Compact public charts, replay captures for the failure-first / confirmed-second
pair, and a later controller redesign before claiming a broader Mines result.

## Audience-Specific Guidelines

### When Writing for Researchers
- Lead with the narrow claim (photometric experiment)
- Acknowledge the gap between core experiment and applications
- Mark future work explicitly
- Invite scrutiny and replication
- Be precise about what has and has not been measured
- Include failure boundaries

**Template:**
"The controlled claim rests on the photometric mirror-alignment experiment [citation]. Applications in [X, Y, Z] demonstrate practical utility but require domain-specific formal studies."

### When Writing for Developers
- Focus on practical utility and concrete examples
- Show code, demos, and performance characteristics
- Be honest about where it works well and where it doesn't
- Provide clear integration guidance
- Can be more forward-looking, but mark speculation

**Template:**
"In our [Game/System], Sundog-style [technique] produces [observable benefit]. We're seeing [metric] improvements in [specific contexts]."

### When Writing Broadcast/Promo Content
- You can use provocative language in hooks
- But always anchor provocation in evidence within 2-3 paragraphs
- Include boundary language somewhere visible
- Link to technical documentation
- Make the evidence accessible (videos, graphs, demos)

**Template:**
"[Provocative hook]. [Accessible explanation]. [Concrete example with evidence]. [Link to detailed docs]."

## The Protection Clause

**Always have this language visible in broad public claims:**

"We are not claiming a completed universal theory. We are presenting an evolving mathematical and computational framework with promising practical results."

Or:

"The broad theorem is a research program. The paper-grade result currently defended in the Sundog repo is narrower: photometric mirror alignment without target-position access in a MuJoCo experiment. The Three-Body, Balance, and Pressure Mines workbenches are bounded operating-envelope studies; EyesOnly, Dungeon Gleaner, and Money Bags are application expressions that motivate the next round of controlled studies."

## The Gravity Frame

The Sundog program's most outlandish public framing — that signature-driven
control is structurally different from reward-driven control under partial
observability, and therefore sidesteps Goodhart's law in a way reward
optimization cannot — is **the gravity frame**. It is named, staged, and
deliberately separated from the safe and risky claims above because it
behaves differently than either.

It is not a safe claim. There is no controlled experiment that ratchets it
into earned language yet. The closest anchors are the photometric mirror-
alignment result and the three-body operating-envelope result, and both fall
short of the adversarial test the frame would need.

It is also not a risky claim in the ordinary sense. The argument behind it is
structural: a metric `R(s, a)` is participated in by the agent, while a
signature `S(x)` is a function of environmental state alone. That distinction
is mathematically clean and is exactly the kind of claim that can be tested
expensively rather than dismissed cheaply.

### When the gravity frame may be used

- Manifesto sections, broadcast hooks, pitch decks, and conference openings
  may use the gravity language *if* they also carry the boundary text and
  link to [`SUNDOG_V_GRAVITY.md`](../SUNDOG_V_GRAVITY.md) and
  [`PROMO_HIGHLIGHTS.md`](../promo/PROMO_HIGHLIGHTS.md) §The Gravity Claim.
- The three-body workbench is the audience-conceptualizable wedge for the
  frame. Public communication should lead with the three-body controller
  watching real gravity from indirect state, then expand the metaphor to
  other partially-observed environments.
- The frame should always be accompanied by a named horizon experiment so a
  reviewer can see the falsification path. Naming the experiment is part of
  the discipline; the gravity claim without a named horizon experiment is a
  slogan.

### When the gravity frame may not be used

- Paper abstracts, peer-reviewed venues, and grant deliverables — these
  require Layer B language and currently-defensible Layer C evidence. The
  gravity frame is Layer A and must stay there until an experiment from
  `SUNDOG_V_GRAVITY.md` runs.
- Any context where the boundary text and the link to the gravity ledger
  cannot accompany the claim.
- Any context where "Sundog solves alignment" is the reading the audience is
  primed to take. The gravity frame is structurally different from solving
  alignment, and that difference must survive the misread.

### Earned envelope language (Phase 7 v1)

The mesa-trap front, staged in [`docs/SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md)
and shipped through Phase 7 v1, has now produced controlled in-vitro evidence
that ratchets the gravity frame partway into earned language. The
defensible *bounded* form is:

> In the tested shadow-field navigation family at Small and Medium capacity,
> terminal-signature and mixed-signature controllers preserve field attachment
> across a mapped pocket of selection pressure, with a sharp Medium breach at
> `λ ≈ 0.953`. Above that boundary, high-reward mixed policies collapse into
> the same fixed-attractor class as reward-trained controllers. Phase 6
> localizes the behavioral cliff to an entangled 5D subspace at the actor's
> final hidden layer.

Phase 7 v2 / Large caveat: the Large cliff-subset now complicates the simple
monotone-cliff story. The value-coef-stabilized Large `lambda=0.99` checkpoint
recovers by the current eval metric (`mean_terminal_alignment ~= 0.885`), while
`lambda=0.95` and `lambda=0.97` remain weak. That U-shape may be more important
than the Medium cliff, but it is not yet a basin-attractor-avoidance receipt:
the eval summary does not compute `old_basin_pref`, so it cannot separate
basin-reaching from a fixed attractor that happens to co-point with the basin
direction. Public material may call it an eval-metric recovery or a Phase 7 v3
prompt, not a confirmed mesa-trap escape.

The earned language has three load-bearing parts. Use them as a set, not
piecemeal, because the parts qualify each other:

- **Behavioral anchor:** a 22-policy operating envelope at Small and Medium
  capacity classifies 8 cells as `hold` (gravity-claim earned), 7 as
  `collapse` (gravity-claim falsified at that selection pressure), 1 as
  `fragile`, 4 as `incompetent`, 2 as `ambiguous`. The hold:collapse split on
  the 20 non-ambiguous cells is 8:7 — a mapped envelope, not a sweep.
- **Quantitative anchor:** the Medium breach threshold sits at `λ ≈ 0.952588`
  with signature weight `1 − λ ≈ 0.047412`. Roughly five percent signature
  anchor is sufficient and necessary to prevent basin internalization in this
  toy family up to the measured boundary.
- **Mechanistic anchor:** the basin attractor at the actor's final hidden
  activation (`net.7`) is **a small handful of generators, irreducibly
  entangled, only legible as a whole**. Ten rounds of mechanistic probing
  (Phase 6 v1 -> v3.8) converged on this shape: (v1) layer patching
  localizes the cliff to `net.7`. (v3) PCA on per-step matched-seed
  diffs compresses the relevant subspace to 5 principal components
  capturing 97.4% of variance and reproducing v1 full-layer patch effect
  (51x compression). (v3.1) those 5 components are jointly necessary; no
  proper subset reproduces the patch -- entangled, not factorable.
  (v3.2) linear additive top-k neuron restriction by L2 destroys the
  mechanism (top-32 captures 33.6% of L2, delivers 0% of patch effect).
  (v3.3) no single critical neuron (max ablation cost <= 0.10 in either
  direction). (v3.4) but **set-level** substrate-restricted ablation of
  v3.3's P->C and C->P top-32 critical sets dissociates patch_success
  functionally and statistically: each direction's substrate
  preferentially disrupts its own direction (P->C drop 0.077, C->P drop
  0.579) and slightly *improves* the other (functional opposition, not
  null); bootstrap 95% CI on the cross-direction ranking overlap is
  [0.016, 0.085] -- robust near-disjointness. The basin-resisting
  substrate is also more anatomically tight (4x dissociation magnitude)
  and overlaps the v3.2 L2-rank top-32 substantially (Jaccard 0.333),
  while the basin-inducing substrate does not (Jaccard 0.049) -- which
  retroactively explains why v3.2's L2-rank-based top-k restriction
  failed harder on the basin-inducing direction. (v3.5+v3.6)
  Cross-policy ablation rankings on J1/J2 (v3.5) plus the cliff-pair
  C->P mask functional-transfer follow-up (v3.6) distinguish three
  layers of cross-policy generalization that earlier framings
  collapsed: (i) subspace-level behavioral transfer, (ii) neuron-
  identity substrate transfer, (iii) functional mask transfer. The
  picture, extended by v3.8's per-PC and signed-effect maps:
  **basin induction is family-wide at the 5D subspace/control-surface
  level, more partitioned across PCs, and anatomically grounded within each policy, but the
  anatomical substrate identity is pair-specific** (P->C Jaccard
  0.255/0.067 across pairs, AND each pair's own P->C mask
  functionally dissociates within that pair: J1 +0.128, J2 +0.253;
  v3.8 P->C per-PC mean off-diagonal Jaccard 0.322);
  **basin resistance is shared at both substrate identity and
  function across Medium policies and more shared across PCs** (C->P Jaccard 0.422/0.684 +
  v3.6 cliff-mask functional dissociation +0.151/+0.508 + v3.7
  own-mask dissociation +0.208/+0.392; v3.8 C->P per-PC mean
  off-diagonal Jaccard 0.430 — same neurons operationally
  necessary for C->P across the family). The reason v3.1's behavioral
  C->P transfer was weak is not that the substrate is policy-specific;
  it is that the cliff-pair-derived substitution activation pattern
  isn't the precise operational target each policy needs at that
  shared substrate. Six methodological lessons stack out of the ten
  rounds: (1) feature-availability rankings are not
  mechanism rankings (SAE at |corr|=0.89 -> 0% patch); (2) variance and
  local sensitivity are not full mechanism (PC1: 38.8% variance, largest
  single-PC P->C max mean ablation cost in v3.8, still no solo K=5 patch);
  (3) linear additive
  top-k subspace restriction destroys mechanism even with the correct
  basis; (4) single-neuron ablation does not surface a critical
  subset; (5) set-level ablation along basis-derived rankings does
  surface direction-specific structure; (6) behavioral transferability
  under a basis and neuron-identity stability are independent
  properties of a circuit -- reasoning that conflates them will
  misread the structure. The same shape appears in a second program
  substrate -- the Sundog geometry program's parhelion atlas
  independently arrived at "small set of complete implied circles, read
  holistically" -- reinforcing the field-not-reward framing across
  in-vitro and in-the-wild receipts. See
  [`docs/MESA_CROSSOVER_NOTE.md`](../MESA_CROSSOVER_NOTE.md).

What is **still not earned**:

- Universal mesa immunity.
- Foundation-model behavior.
- Deployed-system robustness.
- Adversarial robustness under named red-team budgets.
- Large-tier basin-attractor behavior. A provisional Large cliff-subset eval
  exists, but it does not yet include Phase 4-style `old_basin_pref`
  intervention.
- Cross-architecture behavior (only the matched MLP family tested).
- Any claim that signature controllers cannot be reward-hacked. They can —
  the program now has the receipt for *where* they can.

When the partially-earned gravity language is used in public material,
the bounded form above must appear in the same artifact, and the
unearned items must remain off the page.

### Traceability claim taxonomy (Mesa v2)

The Mesa v2 spine — staged in
[`docs/SUNDOG_V_MESAV2.md`](../SUNDOG_V_MESAV2.md) and ratified
2026-05-13 as a sister doc to v1 — names where the gravity frame's
traceability advantage actually lives. The taxonomy is mirrored here
because the §Earned envelope language above earns a partial gravity
claim, and the most common public misread of that claim is "Sundog
solves interpretability." It does not.

The advantage is **strongest at the signature and harness layer, and
weakest inside learned policies.** Four layers, four distinct failure
surfaces:

| Layer | Traceability claim | What can fail |
| --- | --- | --- |
| External signature `S(x)` | Auditable if derived from environment geometry | Sensor spoofing, bad geometry, cheap environment edits |
| Hand-coded controller | Auditable because `SCAN`/`SEEK`/`TRACK` is explicit | Bad thresholds, brittle coupling, hidden implementation leakage |
| Learned policy | Not automatically auditable | Internal proxy formation, shortcut learning, reward-like representation |
| Evaluation harness | Auditable if logs, seeds, probes, and interventions are reproducible | Metric overfitting, probe incompleteness, logging leakage |

Use the taxonomy as a check before reaching for traceability language
in public material. If the claim is about `S(x)` provenance or harness
reproducibility, the v1 + v2 evidence supports it. If the claim is
about the *learned policy's* interior, the v1 Phase 6 v3.x mechanistic
work earned a 5D-subspace localization in one matched MLP family and
not a general interpretability claim — and the Mesa v2 Phase 6.5
counterexample slate is specifically designed to find where the
learned-policy layer collapses entirely.

### Required boundary text

When the gravity frame appears in public material, this language (or a close
paraphrase) must appear visibly in the same artifact:

> The gravity claim is the program's speculative public frame. The controlled
> evidence is narrower: photometric mirror alignment without target-position
> access in a MuJoCo experiment, plus bounded operating-envelope studies in
> Three-Body, Balance, Pressure Mines, and the Mesa-Trap empirical front
> (Phases 0-7 of [`docs/SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md), which
> ratchets a bounded in-vitro form of the gravity frame; see §The Gravity
> Frame → Earned envelope language). The remaining experiments that would
> ratchet the gravity claim further into earned language are staged in
> [`docs/SUNDOG_V_GRAVITY.md`](../SUNDOG_V_GRAVITY.md) and have not been run.

## Red Flags

If you find yourself about to write any of these phrases, stop and revise:

- "proves beyond doubt"
- "always works"
- "revolutionary breakthrough"
- "replaces existing methods"
- "solves the problem"
- "in all cases"
- "definitively shows"
- "eliminates the need for"
- "universal law"
- "final answer"

## Green Lights

These phrases are safe and credible:

- "suggests"
- "demonstrates in tested contexts"
- "appears promising for"
- "early results indicate"
- "experimentally validated in"
- "works in systems with [specific characteristic]"
- "reduces reliance on"
- "informs"
- "shows potential for"
- "motivates further study of"

## Escalation Ladder

As evidence accumulates, claims can strengthen:

**Current (Photometric Experiment Completed):**
"demonstrates in controlled setting"

**After Application Benchmarks:**
"validated across multiple domains"

**After External Replication:**
"reproducible results confirmed by independent groups"

**After Peer Review:**
"peer-reviewed findings in [venue]"

**After Multiple Studies:**
"consistent pattern across [N] studies"

**Long-term (Maybe):**
"established framework with broad applicability"

## When in Doubt

1. Lead with evidence
2. Be specific about context
3. Acknowledge limitations
4. Invite scrutiny
5. Mark speculation clearly
6. Prefer "suggests" over "proves"

## The Sundog Credibility Mantra

**"Show, then interpret. Evidence, then ambition. Narrow claim, broad vision."**
