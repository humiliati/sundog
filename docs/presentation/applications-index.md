# Applications Index

This document structures the proof layer for the Sundog Project: how applications, demonstrations, benchmarks, and comparisons are organized and presented.

## Purpose

The applications gallery is Layer C — the "show me" that prevents dismissal. It provides concrete evidence that the Sundog framework informs real working systems.

Each application demonstrates the pattern: deny full state, observe indirect signal, transform to control-relevant signature, act from signature, measure honestly.

## Organization Principle

**Evidence Tiers:**

Applications are categorized by strength of evidence:

1. **Research Result** — Controlled task, metrics, baselines, reproducible artifacts
2. **Instrumented Prototype** — Product system with telemetry and repeatable harnesses
3. **Product Expression** — Player-facing mechanic that embodies the idea
4. **Conceptual Lineage** — Historical or design-language connection

Public-facing materials can mention all tiers. Academic writing must distinguish them.

---

## Core Applications

### 1. Photometric Mirror Alignment

**Tier:** Research Result

**Repository:** [humiliati/sundog](https://github.com/humiliati/sundog)

**Summary:**
A controller aligns a mirrored end-effector without direct target coordinates, using only sparse photometric feedback and proprioception. Terminal accuracy is statistically indistinguishable from a target-aware oracle baseline (U=526, p=0.264) in the tested MuJoCo setting.

**What It Demonstrates:**
- Indirect measurement can match direct-state control in constrained alignment tasks
- The cost is convergence time, not accuracy
- Failure boundary is known: tight joint limits
- The pattern is measurable and reproducible

**Key Metrics:**
- Terminal target intensity: 0.945 (photometric) vs 0.936 (oracle)
- Median time-to-0.9: 188 steps (photometric) vs 11.5 steps (oracle)
- Statistical test: Mann-Whitney U=526, p=0.264

**Assets Available:**
- Experiment scripts
- Baseline comparison code
- Analysis summaries
- Result graphs
- Stress test data

**Assets Needed:**
- Detailed experiment page
- Animated demo video
- Before/after comparison
- Graph package (terminal intensity, convergence, stress curves)

**Links:**
- [Repo README](../../README.md)
- [Researcher Guide](../RESEARCHER_GUIDE.md)
- [Scientific Criteria](../SCIENTIFIC_CRITERIA.md)
- [Paper Draft](../PAPER_v1_draft.md)

---

### 2. EyesOnly / Gone Rogue

**Tier:** Instrumented Prototype

**Repository:** [humiliati/EyesOnly](https://github.com/humiliati/EyesOnly)

**Summary:**
Procedural roguelike agents act from compressed perception using stop-conditioned action batches. The runner compresses game state, selects a policy axis (progression, survival, resource), batches actions until volatility returns, without authorial omniscience.

**What It Demonstrates:**
- Sundog-style turn envelopes operate against real game engines
- Compressed perception enables coherent behavior
- Agents acting from partial information feel more alive
- The pattern scales beyond lab tasks

**Key Concepts:**
- Intentionally incomplete symmetry
- Turn envelope / stop conditions
- Compressed state payload
- Policy selection (progression, survival, resource)

**Assets Available:**
- Runner implementation (`runners/adapters/gone_rogue.py`)
- Policy examples (`runners/policies/gone_rogue_greedy.py`)
- Integration tests
- Headless API documentation

**Assets Needed:**
- Application detail page
- Demo video showing agent behavior
- Comparison: compressed vs omniscient agent
- Matched-seed study with metrics (future)
- Before/after behavior graphs

**What's Needed for Research Tier:**
- Define success metrics (floor reached, survival rate, resources)
- Compare against baselines: flat greedy, random, oracle
- Run matched seeds
- Report information trade-offs

**Links:**
- [Application Map](../APPLICATIONS.md#eyesonly--gone-rogue)
- [Runners Documentation](../runners.md)
- EyesOnly repository

---

### 3. Dungeon Gleaner / DCgamejam2026

**Tier:** Product Expression

**Repository:** [humiliati/DCgamejam2026](https://github.com/humiliati/DCgamejam2026)

**Summary:**
Physical-feeling simulation in a raycast dungeon crawler. Two major expressions: glass/window reflection at claimed one-twelfth conventional cost, and pressure-washing mechanics with coherent helical/arcing water behavior.

**What It Demonstrates:**
- Physical phenomena can be compressed into structured approximations
- Player-facing coherence retained at lower computational cost
- The same indirect-measurement pattern applies to visual simulation

**Key Mechanics:**
- **Glass Reflection:** Face-aware heuristics, glints, transparent cavities, sprite billboards
- **Pressure Washing:** Stateful hose pressure, kink, energy, spray, projectile, decal, torch interaction

**Assets Available:**
- Pressure washer ADR
- Hose state and spray system code
- Window/glass rendering code
- Glass tile contracts

**Assets Needed:**
- Application detail page
- Demo video (pressure washing in action)
- Demo video (glass reflection comparison)
- Performance comparison graph (claimed 1/12 cost)
- Side-by-side: full physics vs Sundog approximation

**What's Needed for Research Tier:**
- Measurement harness with conventional baseline
- Fixed test scenes
- Per-frame timing data
- Visual error/acceptability criteria
- Spray coherence metrics

**Links:**
- [Application Map](../APPLICATIONS.md#dungeon-gleaner--dcgamejam2026)
- Dungeon Gleaner repository

---

### 4. Money Bags

**Tier:** Instrumented Prototype

**Repository:** [humiliati/Money-Bags](https://github.com/humiliati/Money-Bags)

**Summary:**
Softbody terrain system with graph-based telemetry. N-body spring rig moves over complex terrain; frame-by-frame physics data becomes analyzable through metrics: alignment, torsion, deformation, symmetry, recovery.

**What It Demonstrates:**
- Sundog extends beyond optics into graph interpretation
- Softbody motion becomes legible through structured metrics
- Raw physics traces transform into design-relevant signals
- The pattern applies to non-optical systems

**Key Metrics:**
- Alignment (mean, max, sample count)
- Torsion index
- Peak deformation (radius stddev)
- Centroid travel and speed
- Time to settle
- Symmetry score (mean, max, at settle)
- Recovery (success rate, timeout)

**Assets Available:**
- Rig controller code
- Tuning surface (topology, springs, forces)
- Playtest bundles with metrics.json
- Telemetry CSV files
- Disturbance scenarios

**Assets Needed:**
- Application detail page
- Demo video (rig behavior on terrain)
- Telemetry visualization (graphs of key metrics)
- Before/after: raw traces vs graph-enriched interpretation
- Metric glossary with formulas

**What's Needed for Research Tier:**
- Formal alignment score definition
- Frozen terrain fixtures
- Matched disturbance scripts across tuning profiles
- Comparison: graph metrics vs raw telemetry
- Recovery prediction study

**Links:**
- [Application Map](../APPLICATIONS.md#money-bags)
- Money Bags repository

---

## Cross-Application Comparison

This table provides a unified view of how the pattern manifests:

| Application | Domain | Indirect Signal | Transformation | Actionable Output | Evidence Tier |
|-------------|--------|----------------|----------------|-------------------|---------------|
| Photometric Alignment | Optical control | Detector intensity, proprioception | Scan, seek, extremum tracking | Mirror alignment without target | Research Result |
| EyesOnly / Gone Rogue | Procedural agents | Compressed game state | Turn envelope, stop conditions | Coherent action batches | Instrumented Prototype |
| Dungeon Gleaner | Physical simulation | Glass faces, hose path, pressure | Face heuristics, spray state | Coherent rendering/tools | Product Expression |
| Money Bags | Softbody physics | Spring graph, contact, deformation | Graph metrics, telemetry | Interpretable rig state | Instrumented Prototype |

---

## Gallery Structure

The applications gallery should be accessible from:
- Landing page Section 3 (Application cards)
- Documentation index
- Repo README

**Recommended URL Structure:**
```
/docs/applications/
  index.md (this document)
  photometric-alignment.md
  eyes-only.md
  dungeon-gleaner.md
  money-bags.md
  comparisons.md
  benchmarks.md
```

---

## Application Page Template

Each application should follow this structure:

### Application Name

**Evidence Tier:** [Research Result / Instrumented Prototype / Product Expression]

**Repository:** [Link]

**Summary:** [1-2 sentences]

**What It Demonstrates:**
- [Key point 1]
- [Key point 2]
- [Key point 3]

**The Sundog Pattern:**
1. **Constraint:** What information is denied?
2. **Indirect Signal:** What is observed instead?
3. **Transformation:** How is the signal made actionable?
4. **Action:** What does the system do?
5. **Measurement:** How is success/failure measured?

**Key Metrics / Mechanics:**
[Bullets or table]

**Media:**
- [Screenshots]
- [Demo videos]
- [Graphs]
- [Code snippets]

**Inspectable Surfaces:**
- [Links to code files]
- [Links to documentation]
- [Links to data]

**What's Next:**
[What's needed to strengthen evidence]

**Links:**
- Repository
- Related documentation
- Comparison pages

---

## Comparisons Page

**Purpose:** Show concrete differences between baseline and Sundog approaches

**URL:** `/docs/applications/comparisons.md`

**Structure:**

For each comparison:

#### Comparison Name

**System:** [Game/simulation/task]

**Baseline Method:** [Description]

**Sundog Method:** [Description]

**What Changed:**
- [Change 1]
- [Change 2]
- [Change 3]

**Evidence:**
- Video/GIF side-by-side
- Performance graph
- Behavior metrics

**Interpretation:**
[What the difference means]

**Caveats:**
[Limitations, context, disclaimers]

**Examples:**
- Photometric vs oracle alignment
- Compressed vs omniscient agent
- Full physics vs Sundog approximation
- Raw telemetry vs graph-enriched metrics

---

## Benchmarks Page

**Purpose:** Consolidated performance and metric data

**URL:** `/docs/applications/benchmarks.md`

**Structure:**

### Photometric Alignment Benchmarks

[Table of baseline comparison results]

**Stress Tests:**
[Failure boundary data]

### EyesOnly Agent Benchmarks

[When available: matched-seed comparisons]

### Dungeon Gleaner Performance

[When available: cost comparisons]

### Money Bags Telemetry

[Metric ranges and distributions]

---

## Proof Layer Principles

When presenting applications as evidence:

### 1. Be Honest About Evidence Tier
Don't claim product expressions are research results.

### 2. Separate Demonstration from Proof
Showing it works is not the same as proving it works better.

### 3. Show the Trade
If Sundog is slower but accurate, say so. If it's cheaper but less precise, say so.

### 4. Provide Inspection Points
Link to code, data, and reproducible harnesses.

### 5. Acknowledge What's Missing
If baselines are needed, say so. If metrics are preliminary, say so.

### 6. Mark Future Work
What would move this from product expression to research result?

---

## Media Assets Priority

For each application, create in this order:

**High Priority:**
1. Application detail page
2. Screenshot or hero image
3. Short demo video (1-2 min)
4. Key metric graphs

**Medium Priority:**
5. Before/after comparison
6. Code walkthrough
7. Extended demo video (5+ min)

**Low Priority:**
8. Animated diagrams
9. Interactive demos
10. Tutorial videos

---

## Integration with Landing Page

**Landing Page Section 3** should display application cards:

**Card Structure:**
- Application name
- One-sentence description
- Screenshot or preview
- Evidence tier badge
- "Learn More" link to detail page

**Example Card:**

---
**Photometric Mirror Alignment**
*Research Result*

A controller aligns a reflected beam without target coordinates, matching oracle baseline accuracy in controlled experiments.

[Screenshot of MuJoCo setup]

[Learn More →]

---

---

## Routing Strategy

Visitors should be able to:

1. **Browse all applications** — Gallery index page
2. **Dive into specifics** — Individual application pages
3. **Compare approaches** — Comparisons page
4. **See consolidated data** — Benchmarks page
5. **Understand the pattern** — Cross-application comparison table
6. **Access code** — Direct links to repositories and files

**Navigation Flow:**
```
Landing Page → Applications Section
    ↓
Applications Index (this page)
    ↓
Individual Application Page
    ↓
Comparisons / Benchmarks / Code
```

---

## Content Checklist

For each application, verify:

- [ ] Evidence tier clearly stated
- [ ] Summary is concise and accurate
- [ ] Sundog pattern is explicit
- [ ] Key metrics or mechanics listed
- [ ] Media assets included (screenshots, videos, graphs)
- [ ] Links to code and data provided
- [ ] Honest about what's demonstrated vs what's proven
- [ ] Future work identified
- [ ] Safe claims language used (see claims-and-scope.md)

---

## Future Applications

As new Sundog-informed systems are built:

1. Add to this index
2. Create application detail page
3. Update cross-application comparison table
4. Add to landing page gallery
5. Update benchmarks page if metrics available
6. Consider comparison page entries

**Candidate Future Applications:**
- Additional MuJoCo experiments
- New game AI integrations
- Physics-adjacent tooling
- Simulation interpretation systems
- Graph-based telemetry applications

---

## Success Metrics

The applications gallery is successful if:

- ✓ Visitors understand what Sundog does in practice
- ✓ Evidence is legible and accessible
- ✓ Claims are credible and appropriate
- ✓ Code and data are inspectable
- ✓ Pattern is clear across domains
- ✓ Trust is established through transparency

---

## Next Steps

1. **Write individual application pages** (4 pages)
2. **Create comparisons page** with side-by-side examples
3. **Create benchmarks page** with consolidated data
4. **Gather/create media assets** (screenshots, videos, graphs)
5. **Update landing page** with application cards
6. **Link from documentation index**
7. **Review for claims discipline** (use claims-and-scope.md)

---

## Related Documents

- [Message House](message-house.md) — Layer C proof strategy
- [Claims and Scope](claims-and-scope.md) — Safe vs risky claims
- [Landing Page Outline](landing-page-outline.md) — Section 3 integration
- [Asset Tracker](asset-tracker.md) — Media deliverables status
- [APPLICATIONS.md](../APPLICATIONS.md) — Technical application map
