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
- The controller reaches terminal accuracy statistically indistinguishable from a target-aware analytic baseline in the tested MuJoCo setting (U=526, p=0.264 on terminal target intensity).
- The cost of indirect feedback is convergence time, not terminal accuracy, inside the tested operating envelope.
- The known failure boundary is tight joint limits.

### Framework Characteristics
- The framework appears useful in partially observed systems.
- It has informed specific implementations in game AI, procedural generation, simulation analysis, and rendering/physics-adjacent workflows.
- The pattern involves denying full world-state access, observing indirect signals, transforming them into control-relevant signatures, and acting from those signatures.

### Applications
- EyesOnly/Gone Rogue demonstrates procedural agent control using compressed perception and stop-conditioned action batches.
- Dungeon Gleaner demonstrates verb-field NPC behavior: unmet needs diffuse across satisfier nodes to produce lightweight idle orbits without scripted planners.
- Money Bags demonstrates graph-based interpretation of softbody motion, making torsion, deformation, symmetry, and recovery metrics legible.

### Research Status
- We are continuing to formalize the math and collect examples.
- The theorem is broad, but the current paper claim is narrow enough to attack.
- The defensible result is the photometric alignment experiment; the broader theorem remains a research program supported by application prototypes.

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

### Overpromising on Applications
- ❌ "EyesOnly proves the theorem for procedural games." (needs formal study)
- ❌ "Dungeon Gleaner proves verb-field diffusion outperforms GOAP for town simulation." (needs telemetry and a comparison harness)
- ❌ "Dungeon Gleaner proves a one-twelfth-cost light or physics theorem." (old framing; do not repeat as Sundog evidence)
- ❌ "Money Bags proves softbody alignment is solved." (needs controlled experiments)
- ❌ "Our applications demonstrate universal applicability."

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

## Claim Boundaries by Application

### Sundog Core Repo
**Safe:**
"A controller with no Cartesian access to a target can align a mirrored end-effector using only sparse photometric feedback, reaching terminal accuracy statistically indistinguishable from a target-aware analytic baseline in the tested MuJoCo setting."

**Boundary:**
"The defensible scientific claim is the photometric mirror-alignment experiment. The broader theorem language represents a research program."

### EyesOnly / Gone Rogue
**Safe:**
"EyesOnly shows Sundog-derived turn envelopes operating against a real procedural roguelike through compressed perception and stop-conditioned action batches."

**Avoid:**
"EyesOnly proves the theorem for procedural games."

**What's needed:**
Define success metrics, compare against baselines with matched seeds, report where compressed perception helps and where it loses information.

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

"The broad theorem is a research program. The controlled result currently defended in the Sundog repo is narrower: photometric mirror alignment without target-position access in a MuJoCo experiment. EyesOnly, Dungeon Gleaner, and Money Bags are application expressions that motivate the next round of controlled studies."

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
