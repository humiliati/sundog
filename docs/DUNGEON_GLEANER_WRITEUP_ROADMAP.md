# Dungeon Gleaner Writeup Roadmap

**Filed by:** Dungeon Gleaner contributor for Sundog applications correction (2026-05-07)
**Audience:** Sundog research project — the maintainer of sundog.cc; future-Claude executing this roadmap; skeptic-observers reading the eventual writeup
**Status:** Roadmap pass. This doc plans the correction; the next 2-3 passes execute against it.
**Sources:**
- `applications-gallery.html` § Dungeon Gleaner card (current state — flagged "Product Expression" with `Needs: performance harness, fixed scenes, and visual acceptability criteria`)
- `docs/APPLICATIONS.md` § Dungeon Gleaner / DCgamejam2026 (current state — built around glass/window reflection and pressure-washing)
- `docs/MONEY_BAGS_WRITEUP_ROADMAP.md` (template — voice, structure, evidence-tier discipline)
- `<DG>/docs/VERB_FIELD_NPC_ROADMAP.md` (DOC-83, 2026-04-08) — the actual Sundog application
- `<DG>/docs/NPC_SYSTEM_ROADMAP.md` (DOC-9) — patrol substrate the verb-field replaces
- `<DG>/engine/verb-field.js`, `<DG>/engine/verb-nodes.js`, `<DG>/engine/dungeon-verb-nodes.js`, `<DG>/engine/verb-node-seed.js` — runtime implementation
- `<DG>/engine/reanimated-behavior.js` — second consumer of the same field tick

---

## 0. The story shape — why this writeup is a correction

The Money Bags writeup is an epistemics story: "we built apparatus, we refused to verdict pre-data." **The Dungeon Gleaner writeup is a different kind of honesty story: we mis-broadcast which system in DG was the Sundog application, and the correction reframes a quieter, better-grounded use of the theorem.**

Current `APPLICATIONS.md` § Dungeon Gleaner names two expressions:

1. *"approximate light reflection across glass surfaces at roughly one-twelfth conventional computational cost"*
2. *"a pressure-washing mechanic that produces coherent helical or arcing water behavior"*

Inspected against the actual DG codebase:

- The glass/window path (`engine/window-sprites.js`, `engine/texture-atlas.js`) is a **face-aware billboard sprite system with a glint animation**. It is cheap because it is approximate, but it does not invoke a Sundog-shaped indirect-signal-to-action transformation. There is no detector reading a projection of structure to drive controller action. It is decoration, well-tuned. The "one-twelfth cost" number has no harness behind it.
- The pressure-washing system (`engine/hose-state.js`, `engine/spray-system.js`, ADR-001) is a **stateful FPS-weapon model with hose energy, pressure, kink, and spray-projectile state**. It is a competent toy physics approximation in service of gameplay, but again does not match the Sundog pattern. The spray output is not derived from indirect signal — it is direct simulation of a simplified model.

Both systems are good gameplay engineering. **Neither is a Sundog application.** The current writeup is theorem-dressing on shipping features.

What *is* a Sundog application in DG, and was overlooked in the broadcast, is the **verb-field NPC system** (DOC-83, shipped as `engine/verb-field.js` + `engine/verb-nodes.js` + downstream consumers). NPCs do not run scripted GOAP, behaviour trees, or hierarchical planners. They drift along a gradient field whose value at any tile is the sum of `verb.need × satisfier_proximity_score` across each NPC's prioritising verbs (`duty`, `social`, `errands`, `rest`). Bloom-collapse arrival drops the satisfied verb's need; other verbs reassert; the orbit emerges. This is exactly the Sundog pattern in a non-optical domain — the indirect signal is the spatial gradient of unmet needs over satisfier nodes, and the transformation is need-decay × inverse-distance scoring.

**Tactical implication for voice:** the writeup needs to do two things in the same pass — (a) retract the glass/pressure-washing theorem framing in plain language, without scolding the past broadcast, and (b) install the verb-field framing with the same structural discipline as the photometric and Money Bags writeups. The retraction is load-bearing for skeptic trust; without it, the new claim inherits the credibility deficit of the old one.

---

## 1. Audience + voice calibration

**Three audiences, one voice (parallel to MB roadmap §1):**

1. **sundog.cc skeptics** — will leave on first marketing-ish line, will reward an explicit "we mis-broadcast this previously, here is the corrected reading."
2. **Sundog research community** — wants the abstraction-layer mapping (what is the indirect signal, what is τ, what is the transformation), and the cross-application comparison row reflecting the new claim.
3. **Game-dev / interested observers** — wants to see the verb-field as a *cheaper alternative to GOAP* with concrete code pointers and a tuning table. This is the framing that makes the writeup useful to others working on agent behaviour without a planner budget.

**Voice continues to match the existing `APPLICATIONS.md` and Money Bags roadmap:** past-tense for shipped, present-tense for pending, explicit "Safe claim" / "Avoid", evidence-tier badges. Add one new structural element specific to this writeup: a **"Retraction"** subsection inside § Dungeon Gleaner that names the prior framing and bounds it. Skipping the retraction reads as silently rewriting history; including it reads as research discipline.

**Tier reframing proposal (defer to research-team ratification):** the current tier is "Product Expression." With verb-field + verb-node + reanimated-behaviour shipped and running every game tick, and with the algorithm formally documented in DOC-83 §6, DG sits at "Product Expression" still — but a different one than the broadcast claimed. There is no instrumented harness yet (no telemetry capturing emergent orbit metrics, encounter rates, or the GOAP-substitution comparison). I do not propose introducing a new tier; "Product Expression" is correct for the corrected claim until a measurement pass lands.

---

## 2. Doc structure proposal

Three docs touched, one optional new doc — same shape as MB:

```
sundog/
├── applications-gallery.html
│   └── § Dungeon Gleaner card
│       ├── Tier badge (stays "Product Expression")
│       ├── Indirect signal / Transformation / Output triplet — REPLACE
│       │   (currently glass faces / face-aware rendering / physical-feeling glass+water;
│       │    new: unmet-verb gradient over satisfier nodes / need-decay × inverse-distance
│       │    scoring / emergent NPC orbits without scripted plans)
│       ├── stub-note "Needs:" — refresh
│       └── App-links — point at refreshed APPLICATIONS.md + optional new detailed doc
├── docs/
│   ├── APPLICATIONS.md
│   │   └── § Dungeon Gleaner / DCgamejam2026 — major rewrite
│   │       ├── NEW SUBSECTION: Retraction of prior framing
│   │       ├── Sundog Expression — REWRITE around verb-field diffusion
│   │       ├── Inspectable Surfaces — REPLACE (drop window-sprites/spray; add verb-field/verb-nodes/reanimated-behavior/DOC-83)
│   │       ├── What It Demonstrates — REWRITE around the GOAP-substitution claim
│   │       ├── What To Measure Next — REWRITE around verb-field telemetry the team has not yet built
│   │       └── Claim Boundary — TIGHTEN
│   └── applications/                                    [NEW DIRECTORY, OPTIONAL]
│       └── DUNGEON_GLEANER_DETAILED.md                   [NEW, OPTIONAL]
│           └── Algorithm walk-through, tuning table, archetype presets, encounter taxonomy,
│              and the "why not GOAP / why not behaviour trees / why not utility AI"
│              comparison
```

**Decision point for Pass 2:** does the verb-field algorithm walk-through go in `APPLICATIONS.md` (making § Dungeon Gleaner ~600 lines) or in a sibling `applications/DUNGEON_GLEANER_DETAILED.md`? Default recommendation: **sibling file**, mirroring the MB recommendation. The cross-application value of `APPLICATIONS.md` is partially in its readable-in-one-sitting length.

---

## 3. Content outline for the refreshed `APPLICATIONS.md` § Dungeon Gleaner

Each subsection's intent + content delta:

### 3.1 NEW SUBSECTION — Retraction of prior framing

One short paragraph, named explicitly. Sample text:

> **Retraction (2026-05-07).** Prior versions of this section claimed two Sundog
> expressions in Dungeon Gleaner: a glass/window reflection approximation at
> roughly one-twelfth conventional cost, and a pressure-washing mechanic
> producing helical/arcing water behaviour. On inspection neither matches the
> Sundog indirect-signal-to-action pattern. Both are competent gameplay-physics
> approximations whose value is artistic, not theorem-derived. The "one-twelfth
> cost" figure had no measurement harness behind it. The corrected Sundog
> expression in Dungeon Gleaner is the verb-field NPC system described below.

Naming the retraction is the load-bearing skeptic-trust signal. Without it, the new claim arrives in a credibility-debt context.

### 3.2 Sundog Expression *(REWRITE)*

Replace the glass/water framing with the verb-field framing. The structure parallels the Money Bags Sundog Expression subsection:

> Dungeon Gleaner applies Sundog to NPC idle behaviour in a raycast dungeon
> crawler. The product question is: how do you make a town feel inhabited
> without paying the compute and authoring cost of GOAP, hierarchical task
> planning, or behaviour trees?
>
> The answer used in the shipping game is *verb diffusion*. Each NPC carries a
> small set of prioritising verbs (`duty`, `social`, `errands`, `rest`). Each
> verb has a `need` that increases over time and a list of `satisfier` node
> types in the world (a `bonfire` satisfies `social`, a `faction_post`
> satisfies `duty` for matching-faction NPCs, etc.). Each tick, every reachable
> spatial node is scored by the sum of `verb.need` across the verbs it
> satisfies, divided by distance, with a small noise term. The NPC takes one
> greedy step toward the strongest-pull node. Arrival drops the relevant verb's
> need by `SATISFACTION_DROP` and triggers a 3-7s linger; other verbs continue
> to decay, reasserting their pull as the lingering verb's need is satisfied.
> The orbit emerges.
>
> This is the same family as the photometric experiment and the Money Bags
> rig: the NPC does not need authorial knowledge of where it should go. It
> reads an indirect signal (the gradient of unmet need over satisfier
> proximity) and acts from the transformed signal rather than a planned path.
> The "alignment" is the recurring orbit between work, social, and errand
> nodes; the failure regime is constant flicker (mis-tuned `NOISE_FACTOR`) or
> permanent attachment (mis-tuned `DISTANCE_WEIGHT`).

### 3.3 Inspectable Surfaces *(REPLACE)*

Drop the prior list (window-sprites, hose-state, spray-system, etc.). Replace with:

- `<DG>/docs/VERB_FIELD_NPC_ROADMAP.md` (DOC-83) — the design doc; algorithm in §6, encounter taxonomy in §7, archetype presets in §5.3, full implementation phases in §10
- `<DG>/docs/NPC_SYSTEM_ROADMAP.md` (DOC-9) — the bounce-patrol substrate the verb-field replaces and coexists with
- `<DG>/engine/verb-field.js` — actor-agnostic per-tick verb-field resolution: decay → linger gate → score → step → arrival
- `<DG>/engine/verb-nodes.js` — spatial node registry per floor (the "field sources")
- `<DG>/engine/dungeon-verb-nodes.js`, `<DG>/engine/verb-node-seed.js`, `<DG>/engine/verb-node-overrides-seed.js` — node populations for shipping floors and authored overrides
- `<DG>/engine/npc-system.js` — wires verb-field tick alongside legacy 2-point patrol; `verbSet` opt-in per NPC entity
- `<DG>/engine/reanimated-behavior.js` — the second consumer of the same tick (defeated enemies reanimated friendly, given a verb-set, slot into the same orbit machinery)

### 3.4 What It Demonstrates *(REWRITE)*

Three evidence-bounded demonstration claims (parallel structure to MB §3.4):

1. **The Sundog indirect-signal pattern transposes from optics to AI behaviour.** The detector→intensity→scan analogue becomes need-decay→satisfier-gradient→greedy-step. The same shape (cheap-to-evaluate field, action from local gradient) produces useful behaviour without solving the planning problem. *Evidence tier:* Product Expression.
2. **Verb diffusion is a viable substitute for GOAP / behaviour trees / utility AI in a town-simulation context.** The shipping NPC population orbits between work, social, and errand nodes with no authored transition logic. Encounter classification (camaraderie / tension / passing / gossip) emerges from verb-pair × faction-match without scripted scenes. *Evidence tier:* Product Expression — the substitution claim is honest at the qualitative level; quantitative GOAP-vs-verb-field comparison has not run.
3. **Personality is field weighting, not authored variation.** Six archetypes (`scholar`, `worker`, `citizen`, `drunk`, `guard`, `granny` — DOC-83 §5.3) share one verb vocabulary; differences in spawn need + decay rates produce six distinguishable rhythms. This is the same "same equations, different boundary conditions" pattern the Sundog photometric experiment uses across mirror geometries. *Evidence tier:* Product Expression with anecdotal player-readability; not yet measured.

NOT a fourth claim about emergent narrative or "the system simulates social dynamics." Those would be marketing reach; the verb-field is doing simpler work and that simpler claim is the defensible one.

### 3.5 What To Measure Next *(REWRITE)*

The prior list (performance harness for glass, deterministic spray scene, kink count traces) is moot under the corrected claim. Replace with the actual measurement surface for verb-field:

- **Orbit telemetry.** Per-NPC time-series of (verb, need, current node, dominant pull) sampled at fixed cadence, exported to a deterministic capture format. Without this, claims about "organic orbits" are anecdotal.
- **GOAP-substitution comparison.** Same 4-NPC town, same satisfier nodes, same observable trace. Hand-author a minimal GOAP plan that produces equivalent behaviour. Compare authoring time, code size, runtime cost, and observable variability. The substitution claim becomes defensible only with this comparison; without it, "verb-field replaces GOAP" is rhetorical.
- **Tuning sensitivity.** Sweep `DISTANCE_WEIGHT` ∈ [0.05, 0.30] × `NOISE_FACTOR` ∈ [0.0, 0.15] on a fixed scenario. Report failure regimes (flicker / permanent-attachment / robotic-convergence). This makes the system's stability claim falsifiable.
- **Archetype distinguishability.** Run six archetypes through the same map, measure orbit-period and per-node residency time. Are the six distinguishable in telemetry, or only in narrative framing?
- **Encounter rate & classification.** How often do two NPCs converge on the same node? Does the encounter classification matrix (DOC-83 §7.2) match an outside observer's read of "camaraderie" vs "tension"?

The first item (orbit telemetry) is the gate; everything else depends on it landing.

### 3.6 Claim Boundary *(TIGHTEN)*

Replace existing safe-claim line with:

> Safe claim: Dungeon Gleaner uses a verb-field diffusion model for NPC idle
> behaviour in lieu of GOAP, hierarchical task networks, or behaviour trees.
> Each NPC's prioritising verbs decay over time; a per-tick score combines
> verb need with inverse distance to satisfier nodes; the NPC takes a greedy
> step toward the strongest pull. Encounters and personality emerge from the
> verb vocabulary plus per-archetype weighting. The pattern is the same
> indirect-signal-to-action shape as the photometric mirror experiment and
> the Money Bags softbody rig, applied to AI agency rather than optics or
> physics.
>
> Avoid: Dungeon Gleaner proves verb-field diffusion outperforms GOAP for
> town simulation. The substitution claim is qualitative; quantitative
> comparison has not run. Also avoid the prior framing that named glass
> reflection at one-twelfth cost or pressure-washing helical water as
> Sundog expressions — neither is the indirect-signal-to-action pattern, and
> the cost figures had no harness.

---

## 4. Cross-reference shape

```
applications-gallery.html § Dungeon Gleaner card
  ├─ links to → docs/APPLICATIONS.md § Dungeon Gleaner / DCgamejam2026
  └─ stub-note "Needs:" → REPLACE with:
       "Needs: orbit telemetry capture, GOAP-substitution comparison,
        tuning-sensitivity sweep."

docs/APPLICATIONS.md § Dungeon Gleaner / DCgamejam2026
  ├─ § Inspectable Surfaces → cross-references
  │      <DG>/docs/VERB_FIELD_NPC_ROADMAP.md (DOC-83)
  │      <DG>/docs/NPC_SYSTEM_ROADMAP.md (DOC-9)
  │      <DG>/engine/verb-field.js
  │      <DG>/engine/verb-nodes.js
  │      <DG>/engine/reanimated-behavior.js
  └─ § Retraction → optional deeper link to:
         docs/applications/DUNGEON_GLEANER_DETAILED.md (if we make it)

docs/applications/DUNGEON_GLEANER_DETAILED.md (NEW, OPTIONAL)
  ├─ Algorithm walk-through (DOC-83 §6 inlined with annotations)
  ├─ Tuning constants table + failure regimes
  ├─ Archetype preset table (six archetypes, decay rates, orbit signatures)
  ├─ Encounter classification matrix (DOC-83 §7.2)
  ├─ "Why not GOAP / behaviour trees / utility AI" comparison
  └─ Reanimated-behaviour case study — same tick, second consumer

docs/APPLICATIONS.md § Cross-Application Comparison
  └─ Refresh Dungeon Gleaner row to:
       Domain: Procedural NPC behaviour (town sim)
       Indirect signal: Unmet-verb gradient over satisfier nodes
       Transformation: Need-decay × inverse-distance scoring with noise
       Actionable output: Emergent NPC orbits without scripted plans
```

---

## 5. Skeptic-friendly evidence pattern

What to **show** (concrete, inspectable, dated):

- The DOC-83 design doc, dated 2026-04-08, with the algorithm spelled out in §6 before any captures or marketing
- `engine/verb-field.js` source — readers can verify the implementation matches DOC-83 §6 line by line
- The `DISTANCE_WEIGHT = 0.15`, `NOISE_FACTOR = 0.05`, `SATISFACTION_DROP = 0.50`, `LINGER_MIN = 3000`, `LINGER_MAX = 7000` constants in source — readers can verify nothing is hidden behind tuning magic
- The reanimated-behaviour module as a second-consumer test of the abstraction — the verb-field tick is actor-agnostic, so reanimated friendly enemies slot into the same orbit machinery without forking the algorithm

What to NOT **claim**:

- "Verb-field outperforms GOAP" — not measured
- "NPCs feel like real people" — marketing reach
- "The encounter classification matches human observation" — not yet checked
- "We replaced AI middleware" — we replaced *one* AI use case (idle behaviour) with field diffusion; combat AI in DG still uses a separate awareness/pathfind system

What to **claim**:

- "We chose verb-field diffusion over GOAP for town-NPC idle behaviour, and this is what we shipped"
- "The pattern is the same indirect-signal-to-action shape as the photometric experiment, applied to agency"
- "Personality is field weighting, not authored variation — six archetypes share one vocabulary"
- "The retraction matters: prior glass/water framing was theorem-dressing on systems that exist for other reasons"

The retraction line is the load-bearing skeptic-trust signal for *this* writeup, the way "Stage 1 verdict UNTESTABLE" was for Money Bags. An application that visibly corrects its own broadcast is harder to dismiss than an application that quietly upgrades a stale claim.

---

## 6. Future-pass execution outline

This pass = roadmap (THIS DOC). Future passes:

**Pass 2 (estimated 2-3 hours): refresh `APPLICATIONS.md` § Dungeon Gleaner.**
- Insert Retraction subsection (3.1)
- Rewrite Sundog Expression around verb diffusion (3.2)
- Replace Inspectable Surfaces (3.3)
- Rewrite What It Demonstrates (3.4)
- Rewrite What To Measure Next (3.5)
- Tighten Claim Boundary (3.6)
- Update § Cross-Application Comparison row

**Pass 3 (estimated 1 hour): refresh `applications-gallery.html` § Dungeon Gleaner card.**
- Replace indirect-signal / transformation / output triplet
- Refresh stub-note "Needs:" line
- Update app-links to point at refreshed APPLICATIONS.md sections + optional DUNGEON_GLEANER_DETAILED.md

**Pass 4 (optional, 2-3 hours): write `docs/applications/DUNGEON_GLEANER_DETAILED.md`.**
- Algorithm walk-through with annotations
- Tuning table with operational definitions and failure regimes
- Archetype preset comparison
- Encounter classification matrix
- Why-not-GOAP comparison
- Reanimated-behaviour second-consumer case study

**Pass 5 (optional, 2-3 hours): land orbit telemetry.**
- This is the gate item from §3.5. Builds the harness that turns the qualitative claim into a measurable one. Not strictly part of the writeup, but enables the next round of writeup honesty.

**Pass 6 (optional, deferred): writeup goes live on sundog.cc.**
- Coordinate with whoever maintains sundog.cc deploy
- Pre-deploy: read for marketing-ish drift; check the retraction is still front-loaded after any editorial pass
- Post-deploy: track whether the retraction lands as discipline-signal or as own-goal in skeptic feedback

**Routing:** Pass 2 + Pass 3 are the minimum-viable correction. Pass 4 is the useful-to-other-game-devs deep-dive. Passes 5-6 are measurement + deploy.

---

## 7. Open questions for the research team

1. **Retraction placement.** Lead the section with the Retraction subsection (load-bearing skeptic signal), or fold it into a footnote at the end (less alarming, less honest)? Recommend: lead with it. The discomfort is the point.
2. **Glass/water framing — fully retract or downgrade?** Option A: full retraction, both systems are explicitly *not* Sundog applications. Option B: downgrade to "Conceptual Lineage" tier ("share the project's design language but do not embody the indirect-signal-to-action pattern"). Recommend Option A — the conceptual-lineage hedge is exactly the kind of soft framing the retraction is meant to clean up.
3. **Detailed-doc location.** Sibling to `APPLICATIONS.md` (recommended, `docs/applications/DUNGEON_GLEANER_DETAILED.md`, parallel to MB) or expand `APPLICATIONS.md` itself? Same tradeoff as MB — recommend sibling.
4. **GOAP-substitution comparison framing.** Is "we chose verb-field over GOAP" too combative for a research doc that doesn't yet have the comparison study? My read: no, *if* paired with the explicit "quantitative comparison has not run" boundary in §3.6. The substitution choice is real and shipped; the strength claim is bounded.
5. **Cross-application table column.** Money Bags' roadmap §7 floated adding an "Evidence tier currently held" column to the cross-app table. The DG correction is a natural moment to introduce it — it would make the "Product Expression" tier visible alongside the corrected indirect-signal/transformation/output. Recommend adopting the column with the MB pass.

---

## 8. Closing note on this roadmap's epistemics

The Money Bags roadmap closes by noting that documentation passes that ratify within their own cycle have a higher confidence floor than passes that pile up across rounds. The Dungeon Gleaner correction has a related but distinct discipline: **a writeup correcting itself in public is more credible than a writeup that ages quietly.** The first version of `APPLICATIONS.md` § Dungeon Gleaner was filed in good faith on a misread of which DG system carried the Sundog pattern. Naming the misread, then installing the corrected pattern in the same edit, is a stronger signal than silently upgrading the prose.

If during execution we find the verb-field framing also doesn't quite fit (e.g. the indirect-signal mapping is more strained than this roadmap claims, or the pattern is more accurately a utility-AI variant than a Sundog application), update this roadmap before changing course. Defends against post-hoc reshape — the same defence the Money Bags roadmap installed.

— Dungeon Gleaner contributor, 2026-05-07
