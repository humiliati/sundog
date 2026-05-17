# Applications Index

This document structures the proof layer for the Sundog Project: how applications, demonstrations, benchmarks, and comparisons are organized and presented.

## Purpose

The applications gallery is Layer C — the "show me" that prevents dismissal. It provides concrete evidence that the Sundog framework informs real working systems.

Each application demonstrates the pattern: deny full state, observe indirect signal, transform to control-relevant signature, act from signature, measure honestly.

## Organization Principle

**Evidence Tiers:**

Applications are categorized by strength of evidence:

1. **Research Result** — Controlled task, metrics, baselines, reproducible artifacts
2. **Operating-Envelope Study** — Experiment workbench with bounded sweeps, baselines, and mapped failure regions
3. **Instrumented Prototype** — Product system with telemetry and repeatable harnesses
4. **Product Expression** — Player-facing mechanic that embodies the idea
5. **Conceptual Lineage** — Historical or design-language connection

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

### 2. Three-Body Dynamics Workbench

**Tier:** Operating-Envelope Study

**Repository:** [humiliati/sundog](https://github.com/humiliati/sundog)

**Summary:**
The three-body tab tests whether guarded proxy control can improve survival from partial local signals in a planar restricted setup. Phase 11 supports a bounded positive claim in a high-velocity near-escape pocket while preserving the low-velocity and equal-mass harm boundaries.

**What It Demonstrates:**
- Guarded accelerometer-proxy TRACK control can beat passive and naive local baselines in the tested favorable pocket
- Guard quantile changes preserve the pocket while changing the survival/effort trade
- Outside-pocket expansion exposes where the controller harms outcomes
- This is a workbench result, not a solution to arbitrary three-body control

**Key Metrics:**
- Guard-quantile sweep: 7,056 trials, 300 candidate envelope rows out of 588
- Outside-pocket sweep: 6,912 trials, 96 promising best cells out of 144
- Comparison slate: `track_sensor_accel_guarded` positive in 81/81 high-velocity near-escape rows; naive local baseline 0/81 candidate rows

**Assets Available:**
- Interactive experiment page (`/threebody`)
- Phase 11 summary
- Roadmap and detailed writeup
- Result directories for guard quantiles, outside-pocket expansion, and comparison

**Assets Needed:**
- Public Phase 11 charts
- Failure-boundary visualization
- Longer-horizon stress tests
- Physical-sensor validation plan

**Links:**
- [Phase 11 Summary](../THREEBODY_PHASE11_SUMMARY.md)
- [Research Roadmap](../SUNDOG_V_THREEBODY.md)

---

### 3. Sundog Pressure Mines Workbench

**Tier:** Operating-Envelope Study

**Repository:** [humiliati/sundog](https://github.com/humiliati/sundog)

**Summary:**
The Pressure Mines workbench tests whether a controller can make safer progress from a noisy, lossy pressure field when mine locations and exact adjacency counts are hidden. Phase 10 confirms one narrow noisy/dropout pocket and publishes it beside a matched failure region where naive pressure wins.

**What It Demonstrates:**
- Indirect pressure can preserve enough structure for bounded reveal decisions in a named pocket
- The promoted outcome is budget-adjusted safe-tile progress before mine trigger, not field clearance
- The public artifact is failure-first: the mapped negative region is the default replay, with the confirmed pocket one click away
- `threshold_flagger` remains a visible comparator because it buys survival with false flags rather than carrying the promoted claim

**Key Metrics:**
- Locked Phase 10 run: 46 envelope cells, 7 modes, 64 matched seeds, 20,608 trials
- Candidate rows: 2; failure-regime rows: 34; contradiction rows: 0
- Confirmed pocket: density 0.16 / pressure noise 2.0 / dropout 0.2; `sundog_minimal` +7.21875 budget-adjusted safe tiles versus `naive_pressure`, 95% CI [3.375, 11.078516]
- Paired failure: density 0.22 / pressure noise 1.0 / dropout 0.35; `sundog_lean` -4.71875 versus `naive_pressure`, 95% CI [-8.375781, -1.496484]

**Assets Available:**
- Interactive workbench (`/mines`)
- Roadmap and audit trail
- Phase 10 verdict artifacts under `results/mines/phase10-envelope`
- Browser replay URLs for the failure and confirmed cells

**Assets Needed:**
- Short public replay capture of the failure-first / confirmed-second pair
- Compact chart or thumbnail for the envelope map
- Later controller-redesign phase if a broader controller claim is desired

**Links:**
- [Workbench](../../mines)
- [Roadmap](../sundog_v_minesweeper.md)
- [Application Map](../APPLICATIONS.md#sundog-pressure-mines-workbench)

---

### 4. EyesOnly / Gone Rogue

**Tier:** Instrumented Prototype

**Repository:** [humiliati/EyesOnly](https://github.com/humiliati/EyesOnly)

**Summary:**
The headless Gone Rogue runner drives the real JavaScript game engine through `GoneRogue.headless` and Playwright. It compresses floor, biome, HP, alert, inventory, gate, and combat state; selects a policy axis; and executes stop-conditioned action batches. The apparatus is seedable and policy-pluggable, but the matched-seed study has not run.

**What It Demonstrates:**
- Sundog-style turn envelopes operate against a real product engine
- The runner is an instrumented apparatus, not yet a policy-comparison result
- The UI-bound playtest agent is sibling UX automation, not Sundog evidence
- LAGM is forward-looking design lineage, not shipping code

**Key Concepts:**
- Intentionally incomplete symmetry
- Turn envelope / stop conditions
- Typed compressed state payload
- Policy selection (progression, survival, resource)
- Tier-per-surface discipline: runner, sibling automation, forward-looking design

**Assets Available:**
- Runner implementation (`runners/adapters/gone_rogue.py`)
- Policy examples (`runners/policies/gone_rogue_greedy.py`)
- Integration tests
- Headless API documentation
- LAGM design document (Conceptual Lineage only)

**Assets Needed:**
- Application detail page
- Demo video showing agent behavior
- Matched-seed multi-policy study
- Compressed-perception ablation
- Volatility-threshold sweep
- Behavior clips from identical seeds

**What's Needed for Research Tier:**
- Define success metrics (floor reached, survival rate, resources, volatility, steps)
- Compare greedy, random legal, and target-aware/debug-state policies
- Run matched seeds and publish JSONL bundles
- Report information trade-offs and volatility-threshold sensitivity

**Links:**
- [Application Map](../APPLICATIONS.md#eyesonly--gone-rogue)
- [Runners Documentation](../runners.md)
- EyesOnly repository

---

### 5. Dungeon Gleaner / DCgamejam2026

**Tier:** Product Expression

**Repository:** [humiliati/DCgamejam2026](https://github.com/humiliati/DCgamejam2026)

**Summary:**
Verb-field NPC behavior in a raycast dungeon crawler. Unmet needs diffuse across satisfier nodes, producing lightweight idle orbits without GOAP, hierarchical task networks, or hand-authored behavior trees.

**What It Demonstrates:**
- The Sundog indirect-signal pattern can transpose from optics to NPC behavior
- Lightweight verb diffusion can substitute for authored idle planners in a shipped game context
- Personality can be represented as field weighting rather than bespoke behavior scripts

**Key Mechanics:**
- **Verb Field:** Need decay, satisfier proximity scoring, linger gates, and greedy local movement
- **Node Sources:** Work, social, errand, rest, and faction anchors act as satisfiers
- **Archetypes:** Shared verbs with different weights produce different rhythms

**Assets Available:**
- Verb-field NPC roadmap
- NPC system roadmap
- Verb-field and verb-node runtime modules
- Reanimated-behavior consumer of the same field tick

**Assets Needed:**
- Application detail page
- Demo video showing NPC idle orbits
- Orbit telemetry visualization
- GOAP-substitution comparison
- Tuning-sensitivity graphs

**What's Needed for Research Tier:**
- Deterministic NPC/town fixture
- Per-NPC verb, need, node, and dominant-pull traces
- GOAP or behavior-tree baseline
- Tuning sweeps for flicker, attachment, and convergence failures

**Links:**
- [Application Map](../APPLICATIONS.md#dungeon-gleaner--dcgamejam2026)
- Dungeon Gleaner repository

---

### 6. Money Bags

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
| Three-Body Dynamics | Planar restricted dynamics | Local tidal-tensor proxy, acceleration magnitude, separated privileged diagnostics | Guarded TRACK control over sensor-motivated instability signature | High-velocity near-escape pocket with mapped failure boundaries | Operating-Envelope Study |
| Pressure Mines | Hidden minefield workbench | Noisy pressure field, local gradients, bounded scan returns | Confidence-gated pressure ordering with conservative comparator lanes | Safe-tile progress in a named pocket, paired with a mapped failure region | Operating-Envelope Study |
| EyesOnly / Gone Rogue | Procedural agents | Floor/biome/HP/alert/inventory/gate/combat state | Perception compression, axis selection, stop-conditioned batches | Seedable JSONL runs for matched-seed studies | Instrumented Prototype |
| Dungeon Gleaner | Procedural NPC behavior | Unmet-verb gradient over satisfier nodes | Need decay, inverse-distance scoring, linger gates | Emergent idle orbits without scripted plans | Product Expression |
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
