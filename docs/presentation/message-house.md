# Message House

This document defines the core messaging hierarchy for the Sundog Project across all public communication channels.

## Three-Layer Structure

All Sundog communication should balance three layers:

### Layer A — The Bold Idea
The emotionally charged version. This is the "spark."

**The claim:**
We discovered a new alignment framework. It changes how agents interpret occluded or indirect information. We've been applying it in software and it works in real applications.

**Purpose:**
- Provokes curiosity
- Establishes ambition
- Creates memorable hooks
- Draws people in

**Outlets:**
- Manifesto sections
- Opening hooks
- T-shirt designs
- Social media
- Conference talks

### Layer B — The Legible Claim
The academically safer version. This is the "credible public claim."

**The claim:**
The Sundog Project is a framework for indirect measurement, alignment inference, and agent reasoning under partial observability. We are developing mathematical language, implementation examples, and applied demonstrations. Our current results suggest practical utility in game AI, procedural systems, simulation interpretation, and physics-adjacent software tooling.

**Purpose:**
- Establishes legitimacy
- Provides academic framing
- Enables collaboration
- Invites scrutiny

**Outlets:**
- Landing page
- Documentation homepage
- Paper abstracts
- Researcher communications
- Grant applications

### Layer C — The Proof
The evidence layer. This is the "show me."

**The claim:**
Concrete demonstrations, benchmarks, graphs, before/after comparisons, repo examples, videos of agent behavior, code paths, and measurable outputs.

**Components:**
- Photometric mirror alignment experiment with baselines
- EyesOnly/Gone Rogue agent demonstrations
- Dungeon Gleaner performance comparisons
- Money Bags softbody telemetry
- Stress test results
- Application gallery

**Purpose:**
- Prevents dismissal
- Provides inspection points
- Enables replication
- Demonstrates utility

**Outlets:**
- Application pages
- Benchmark sections
- Video demonstrations
- Results directories
- Comparison tables

## The Balance Rule

**Last year's failure:** Layer A was too far ahead of Layer C.

**This year's fix:** Anchor A in B and C.

Never use Layer A language without pointing to Layer B framing and Layer C evidence.

## Message Versions

### The One-Sentence Version

The Sundog Project is a framework for indirect measurement and agent reasoning under partial observability, with working applications in procedural systems, simulation analysis, and game AI.

### The Short Version (2-3 paragraphs)

The Sundog Project began as a mathematical intuition about alignment and indirect signals. Over the past year, we have applied that framework in software and found it useful in real systems — especially where behavior must be inferred from incomplete or occluded information.

The core experiment demonstrates photometric mirror alignment without direct target access, reaching terminal accuracy comparable to oracle baselines in a controlled MuJoCo setting. Beyond the lab task, the same framework informs procedural agent design in EyesOnly, verb-field NPC behavior in Dungeon Gleaner, and softbody interpretation in Money Bags.

We are continuing to formalize the mathematics, strengthen the experimental evidence, and explore new application domains.

### The Ambitious Version

We believe the Sundog framework offers a new way to think about alignment in complex systems: not by staring directly at the object, but by reading the signatures cast around it.

Where conventional approaches demand complete world state, Sundog asks whether the shadow, the torque, the occlusion, the deformation — the indirect signal — already contains enough structure to act.

The past year has turned this intuition into working systems: controllers that align without target coordinates, agents that operate from compressed perception, simulations that preserve physical coherence at lower cost, and softbodies whose motion becomes interpretable through graph signatures.

The future we're building: software that understands it doesn't need perfect information to behave intelligently.

### The Guarded Version

We are not claiming a completed universal theory. We are presenting an evolving mathematical and computational framework with promising practical results.

The defensible scientific claim is narrow: photometric mirror alignment without target-position access in a controlled MuJoCo experiment. The broader applications demonstrate that similar patterns can inform practical software systems, but each domain requires its own formal study.

This is a research program, not a finished product.

## Message Hierarchy by Audience

### For Researchers
**Lead with:** Layer B (legible claim) + Layer C (evidence)
**Support with:** Narrow defensible claims, baselines, failure boundaries
**Avoid:** Overpromising, universal claims, revolutionary language

### For Developers
**Lead with:** Layer C (demos) + practical utility
**Support with:** Code examples, performance comparisons, integration patterns
**Include:** Layer A hooks for motivation

### For Game Designers
**Lead with:** Layer C (before/after) + creative possibilities
**Support with:** Concrete mechanics, player-facing effects, design affordances
**Include:** Layer A to inspire experimentation

### For General/Curious Audience
**Lead with:** Layer A (bold idea) + accessible Layer B framing
**Support with:** Visual Layer C (videos, graphs, screenshots)
**Route to:** Deeper technical documentation

## Closing Lines by Context

### Academic Context
"The broad theorem is a research program. The controlled result currently defended in the Sundog repo is narrower: photometric mirror alignment without target-position access in a MuJoCo experiment. EyesOnly, Dungeon Gleaner, and Money Bags are application expressions that motivate the next round of controlled studies."

### Broadcast Context
"Sundog is not about seeing everything. It is about seeing enough."

### Product Context
"We are building toward software that understands the shape of its own disturbance."

### Future-Facing Context
"The future belongs to systems that can act intelligently from incomplete truth."

## Integration Examples

### Landing Page Hero
**Layer A hook:** "Alignment without sight"
**Layer B framing:** "A framework for indirect measurement and agent reasoning under partial observability"
**Layer C anchor:** Animated demo + "View the experiment" CTA button

### Paper Abstract
**Layer B lead:** "We present a framework for alignment inference using indirect environmental signals..."
**Layer C evidence:** "...demonstrated through photometric mirror alignment achieving terminal accuracy statistically indistinguishable from oracle baselines (U=526, p=0.264)..."
**Layer B scope:** "...with applications in procedural agent control and physical simulation."

### Conference Talk Opening
**Layer A spark:** "Most systems do not reveal their truth directly. You infer them from signatures — shadows, feedback, distortions, response curves."
**Layer B credibility:** "The Sundog Project is our attempt to formalize and apply that idea."
**Layer C proof:** "Let me show you what happens when a controller tries to align a mirror without knowing where the target is..."

## Mantras

- "Not the sun. The signature."
- "Anchor the provocation in evidence."
- "Spark, claim, proof."
- "Less state. More structure."
- "Show, then interpret."
