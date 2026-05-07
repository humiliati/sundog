# Money Bags Writeup Roadmap

**Filed by:** c.Echo (Sundog contractor for Money Bags softbody simulation, 2026-04-26 → 2026-04-30 active)
**Audience:** Sundog research project — the maintainer for sundog.cc updates; future-Claude executing this roadmap; skeptic-observers reading the eventual writeup
**Status:** Roadmap pass. This doc plans the writeup; the next 3-4 passes execute against it.
**Sources:**
- `applications-gallery.html` § Money Bags card (current state — flagged as "Instrumented Prototype" with `Needs: frozen terrain fixtures, disturbance scripts, and metric glossary`)
- `docs/APPLICATIONS.md` § Money Bags (current state — has Sundog Expression, Inspectable Surfaces, Current Metric Vocabulary table, Claim Boundary)
- The c.Echo work surfaces in `<Money Bags>/docs/design/` (Sundog_toolset_PROPOSAL.md, Shape_Coherence_Architecture.md, Shape_Coherence_Stage_1_Rubric.md, Shape_Coherence_Pass_1_5_Candidates.md)
- The c.Echo lane handoff in `<Money Bags>/docs/handoffs/Echo_HANDOFF.md` (Pass 1 addendum carrying through to current state)

---

## 0. The story shape — why this writeup is unusual

Most Sundog application writeups tell a "we applied Sundog and it worked" story. **Money Bags can't tell that story honestly yet.** The actual project state is:

- Sundog's abstraction layer (BloomTracker shape, torque-stability, alignment-score formula, phase-space leverage point) was lifted into a Godot collector that produces per-tick derived metrics.
- A four-fixture × four-profile × ~3-capture Stage 1 falsification slate was scaffolded against an explicit pre-registered rubric (`Shape_Coherence_Stage_1_Rubric.md`).
- The vocabulary the metrics measure was ratified by Command (`Shape_Coherence_Architecture.md`).
- A pre-registered verdict template was committed before captures.
- An external freelancer audit surfaced architectural blind spots (`f78cb35`) the team had been compensating around without naming. Five-factor rotation-suppressor model emerged.
- **And then structured Stage 1 captures haven't run yet.** The capture infrastructure has a known crash mode at high gain × unlocked rotation; the architecture's hardcoded suppressors are being unblocked one at a time; the falsification apparatus is built but the verdict is pending.

So this is **an epistemics story, not a results story**. The writeup's value to sundog.cc skeptics is exactly *that* it isn't a confirmation. It demonstrates research discipline: building falsification infrastructure honestly, refusing to verdict pre-data, and surfacing one's own architectural blind spots through external audit rather than dismissing them.

**Tactical implication for voice:** The writeup should *resist* the temptation to claim Sundog-on-softbody-works. The honest claim is bounded:

> Money Bags built a pre-registered falsification apparatus for the Sundog see-saw musing applied to softbody rigs. The apparatus produced four ratified vocabulary cycles, a four-mode falsification taxonomy, and an empirically-supported architecture-mode finding before structured captures ran. Stage 1 verdict is currently UNTESTABLE pending architecture-side knob exposure (R3 `node_lock_rotation`, Knob 1 `frame_force_rotation_aware`, capture-crash fix).

That's a more interesting story for skeptics than a "yes it confirmed" headline.

---

## 1. Audience + voice calibration

**Three audiences, one voice:**

1. **sundog.cc skeptics** — want bounded claims, audit trails, evidence tiers, willingness to say "this isn't measured yet." They will leave on first marketing-ish line.
2. **Sundog research community** — want the abstraction-layer mapping (what BloomTracker became, how the alignment-score formula transposes), the rigorous falsification structure, and the cross-application comparison with EyesOnly / Dungeon Gleaner.
3. **Interested observers (future contractors / future Claudes / future game devs)** — want concrete inspectable surfaces, code pointers, the Cycle 1-5 process-discipline pattern, and the gameplay-primitive corollary (because that's where useful design surface emerges).

**One voice serves all three:** the existing `APPLICATIONS.md` voice. Past-tense for what's done, present-tense for what's pending, explicit "Safe claim" / "Avoid" sections, evidence-tier badges. **Don't break voice for emphasis.** If the writeup feels boring it's working — boring is the skeptic-trust signal.

**Tier reframing proposal (defer to research-team ratification):** Money Bags' current tier is "Instrumented Prototype." With the falsification apparatus now scaffolded, it sits in a sub-tier between Instrumented Prototype and Research Result that I'd call **"Pre-Registered Falsification Apparatus."** Different from Instrumented Prototype because the slate, predictions, verdict template, and disposition rule are all committed *before* the experiment runs. Different from Research Result because the experiment hasn't run.

The research team can decide whether that sub-tier is worth introducing or whether "Instrumented Prototype" is fine with a `Falsification Apparatus Pre-Registered` flag in the card body. Cheaper to skip the new tier; more honest to introduce it.

---

## 2. Doc structure proposal

Three docs touched, one optional new doc:

```
sundog/
├── applications-gallery.html
│   └── § Money Bags card
│       ├── Tier badge (current: "Instrumented Prototype")
│       ├── Indirect signal / Transformation / Output triplet (already good; minor polish)
│       ├── stub-note "Needs:" line — refresh to current state (most "needs" closed)
│       └── App-links — point at refreshed APPLICATIONS.md § Money Bags + optional new detailed doc
├── docs/
│   ├── APPLICATIONS.md
│   │   └── § Money Bags — major refresh (this is where the substance lands)
│   │       ├── Sundog Expression — keep, light edits
│   │       ├── Inspectable Surfaces — add new files (RigMetricsCollector, Architecture doc, Rubric)
│   │       ├── Current Metric Vocabulary — promote to a real glossary with formulas
│   │       ├── What It Demonstrates — REWRITE around the falsification-apparatus framing
│   │       ├── What To Measure Next — REWRITE; most prior items now closed; new items reflect Stage 1 prereqs + capture-crash fix
│   │       ├── NEW SUBSECTION: "Pre-Stage-1 findings" — Path A unit weights + restless_idle_active discriminator + freelancer audit precedent
│   │       └── Claim Boundary — TIGHTEN
│   └── applications/                                    [NEW DIRECTORY, OPTIONAL]
│       └── MONEY_BAGS_DETAILED.md                       [NEW, OPTIONAL]
│           └── The deep-dive (Sundog adaptation map, Cycle 1-5 process pattern, four-mode taxonomy, gameplay-primitive corollary)
```

**Decision point for Pass 2:** does the deep-dive go in `APPLICATIONS.md` (making § Money Bags ~800 lines, which dominates the doc) or in a sibling `applications/MONEY_BAGS_DETAILED.md`? Default recommendation: **sibling file**, because the cross-application-comparison value of `APPLICATIONS.md` is partially in its readable-in-one-sitting length. Money Bags' deep-dive deserves its own document; `APPLICATIONS.md` § Money Bags links to it for "extended context."

---

## 3. Content outline for the refreshed `APPLICATIONS.md` § Money Bags

The structure stays parallel to EyesOnly / Dungeon Gleaner sections (research-doc voice consistency). Each subsection's intent + content delta:

### 3.1 Sundog Expression *(light edits)*

Existing text already frames the indirect signal correctly (deformation as the optical-shadow analogue). Add one-paragraph note that the Sundog *abstraction* layer landed in idiomatic Godot — explicit citation that the alignment-score formula `radial × (1 - torsion/torsion_ref) × symmetry` came from `<sundog>/notebooks/sundog alignment theorem.txt`, not from a Python-to-GDScript transpile. Honors the §24 cross-cutting decision in the proposal that lifted the abstraction layer per the proposal's banner-discipline.

### 3.2 Inspectable Surfaces *(add new files)*

Existing list covers core rig + harness + jostle scenarios + foxtrot fixtures. Add:

- `<Money Bags>/docs/design/Sundog_toolset_PROPOSAL.md` — the contractor proposal that adapted Sundog into MB's architecture
- `<Money Bags>/docs/design/Shape_Coherence_Architecture.md` — the ratified vocabulary doc (rotation-locking vs rotation-permissive, see-saw bias)
- `<Money Bags>/docs/design/Shape_Coherence_Stage_1_Rubric.md` — the falsification rubric (4-mode taxonomy, verdict template, kill-switches)
- `<Money Bags>/echo/Echo_RigMetricsCollector.gd` — the collector implementing the Sundog adaptation
- `<Money Bags>/echo/Echo_RigMetricsCollector.md` — the collector's calibration sidecar
- `<Money Bags>/docs/handoffs/Echo_HANDOFF.md` — c.Echo's Pass 1 addendum + the addendum chain showing how the work landed
- `<Money Bags>/playtest/CROSS_PROFILE_AGGREGATE_*.md` — the aggregate output where the bias-axis readout surfaces

### 3.3 Current Metric Vocabulary *(promote to a real glossary)*

The existing table lists field names. Replace with a glossary that gives **the formula or operational definition** for each metric, sourced from the collector code. Sample entries:

```
alignment_score        radial_frac × (1 - clamp(torsion/torsion_ref, 0, 1)) × symmetry_factor
                       (NaN when grand_total_ke < idle_epsilon — Pass 2.5.1 idle sentinel)

torsion_index          tangential_frac (the fraction of internal KE in tangential motion;
                       per-tick scalar, 0-1)

shape_coherence_bias_  (k_ffsr × ffsr_unit_weight) / (k_ffsr × ffsr_unit_weight + k_spring × spring_unit_weight)
actual                  normalized to [-1, +1]; sign convention positive = FFSR/centripetal/marble

energy_split_at_settle Three-component (radial / tangential / translational) sampled at the
                       Pass 2.8(g) post-impact-settle marker tick

[etc.]
```

This closes the gallery card's `metric glossary` Need.

### 3.4 What It Demonstrates *(REWRITE)*

Existing text is good for Phase 1 of the work but predates the falsification-apparatus state. Rewrite around three demonstration claims, evidence-bounded:

1. **Sundog's abstraction layer transposes from optics to softbody graph telemetry.** BloomTracker → shape deviation field; torque-stability variance → torsion stability; alignment-score formula → idiomatic GDScript. Each adaptation banner-marked in collector source for traceability. *Evidence tier:* Instrumented Prototype.
2. **A pre-registered falsification apparatus can be authored before captures run.** Vocabulary ratified, predictions filed, verdict template committed, disposition rule explicit. The rubric defends against post-hoc story-fitting. *Evidence tier:* Pre-Registered Falsification Apparatus.
3. **External audit catches architectural blind spots that internal teams compose around.** The freelancer's `body.lock_rotation = true` finding (`f78cb35`) led to the five-factor rotation-suppressor model. The team had been compensating around the hardcode without naming it. *Evidence tier:* Process discipline pattern.

NOT a fourth demonstration claim about the bias-axis being mechanically valid. That's pending Stage 1 captures.

### 3.5 What To Measure Next *(REWRITE)*

Existing list said: define alignment formally, freeze fixtures, run matched disturbance scripts, etc. **Most of those are done.** New "What To Measure Next" reflects the actual remaining surface:

- Land Pass 1.5(ε) capture-crash fix (currently the only blocker for full slate captures including the centrifugal-spill primitive at gain=1.0)
- Run structured Stage 1 captures with `_v_tumble_unlocked` (Knob 1 enabled + R3 disabled) on the 4-fixture × 4-profile slate
- Apply the rubric §5 verdict template; produce CONFIRM / REFUTE / AMBIGUOUS verdict per `Shape_Coherence_Architecture.md` § "Stage 1 results"
- If CONFIRM: extract bias-band thresholds (`Z_lo, Z_hi`) per rubric §8 and route to Alpha's GATE 4 in the canonical-rig experiment
- Independent of Stage 1: validate the gameplay-primitive corollary (Tumble / Pressure-release jump / Centrifugal spill) as a Stage-2 design surface

### 3.6 NEW SUBSECTION: Pre-Stage-1 findings

This is the part that tells the story future-Claude / skeptic-observers want. Three subsections:

- **3.6.1 Path A unit-weight calibration finding.** The `_v_bias_zero` profile measured at -0.33 (not 0.0) under default `1.0/1.0` weights. Adopting Path A (`spring_unit_weight = 0.5`) per Command `99444a5` produced anchor reads at -0.846 ± 0.003 across all post-Path-A captures, validating the calibration discipline and exposing that **adjacent stiffness contributes ~2× per unit relative to FFSR in the bias-axis arithmetic** — itself a useful structural finding for Stage 2 meta-knob design.
- **3.6.2 Bounded inflation vs socket failure discriminator.** The `restless_idle_active` flag fires across many bundles even on profiles that look stable in aggregate. Bias-axis vocabulary translates: bounded skin-readable inflation = stable non-zero deformation (`radius_stddev_mean` ≈ `radius_stddev_max`); socket failure = oscillating shape (`radius_stddev_max >> radius_stddev_mean`). Same metric magnitude; different time-series signature. Documented in c.Echo's input on `RIG_EXPERIMENT_PREPASS_MATRIX_2026-04-30.md`.
- **3.6.3 Freelancer audit precedent (`f78cb35`).** The hardcoded `body.lock_rotation = true` at line 279 was missed by the contractor, the lane engineer, and Command. The freelancer found it on first audit pass. Pattern lesson now codified in MB's memory system as `feedback_audit_existing_hardcodes_before_chartering_knobs.md`: audit existing controller hardcodes before chartering new knobs — they may already exist as toggle-switches awaiting exposure.

### 3.7 Claim Boundary *(TIGHTEN)*

Existing safe-claim line: *"Money Bags extends Sundog from optical alignment into graph interpretation of softbody motion, with playtest telemetry already capturing alignment, torsion, deformation, symmetry, and recovery signals."*

Replace with:

> Safe claim: Money Bags built a pre-registered falsification apparatus for the Sundog see-saw musing applied to softbody rigs. The apparatus includes (a) a ratified vocabulary doc with sign-convention and stage-gating, (b) a four-fixture / four-profile / three-capture-per-cell falsification slate, (c) a pre-registered verdict template with explicit disposition rule, (d) a four-mode kill-switch taxonomy distinguishing fixture-mode / architecture-mode / bias-mode / suppressor-stack-incomplete failures, and (e) a five-factor rotation-suppressor canonical reference produced via external audit. Per-bundle telemetry already captures alignment, torsion, deformation, symmetry, recovery, and shape-coherence-bias signals.
>
> Avoid: Money Bags proves Sundog-on-softbody. Stage 1 captures pending architecture-side knob exposure (R3 `node_lock_rotation` + Knob 1 `frame_force_rotation_aware`) and capture-crash fix. Verdict currently UNTESTABLE for the architecture-mode + bias-mode kill-switch reads.

That's the bounded claim that survives skeptic audit.

---

## 4. Cross-reference shape

```
applications-gallery.html § Money Bags card
  ├─ links to → docs/APPLICATIONS.md § Money Bags
  └─ stub-note "Needs:" → most current Needs are closed; refresh to:
       "Needs: structured Stage 1 captures (pending capture-crash fix)
        + Stage 2 meta-knob design surface validation."

docs/APPLICATIONS.md § Money Bags
  ├─ § Inspectable Surfaces → cross-references
  │      <Money Bags>/docs/design/Sundog_toolset_PROPOSAL.md
  │      <Money Bags>/docs/design/Shape_Coherence_Architecture.md
  │      <Money Bags>/docs/design/Shape_Coherence_Stage_1_Rubric.md
  │      <Money Bags>/echo/Echo_RigMetricsCollector.gd
  └─ § Pre-Stage-1 findings → optional deeper link to:
         docs/applications/MONEY_BAGS_DETAILED.md (if we make it)

docs/applications/MONEY_BAGS_DETAILED.md (NEW, OPTIONAL)
  ├─ Cycle 1-5 process discipline pattern (full)
  ├─ Four-mode taxonomy (full, with sequencing diagram)
  ├─ Gameplay-primitive corollary (Tumble / Pressure-release jump / Centrifugal spill matrix)
  ├─ Architecture-knob inventory (Knob 1 / Knob 2 / Knob 3 / Knob 4 / R3) — the post-`f78cb35`
  │      / post-`4db91c9` state
  └─ Audit-trail timeline (2026-04-26 charter → 2026-04-30 prepass) showing how
       the falsification scope evolved without verdict-fitting
```

---

## 5. Skeptic-friendly evidence pattern

What to **show** (concrete, inspectable, dated):

- The Echo collector source — readers can verify the alignment-score formula matches what the theorem text says
- The rubric's verdict template — readers can verify it pre-commits to thresholds before captures
- The pre-flight gate validation table — anchor reads -0.846 across 4 captures matches predicted -0.85
- The `Cycle 1-5` process-discipline progression in §11.4 of the rubric — readers can trace how discipline emerged
- The `f78cb35` audit trail — freelancer found a hardcode the team had compensated around; documented in MB memory + rubric §11.4 + ratified by Command
- The four-mode taxonomy — readers can verify Mode B / Mode D were ratified within their charter cycle by data, not by argument

What to NOT **claim**:

- "Sundog confirmed for softbody"
- "Bias-axis vocabulary is validated by experiment"
- "Alignment score correlates with feel"
- "The five-factor model is complete" — actually we explicitly *don't* claim this; we say "complete enough for Stage 1 reading; sixth-factor watch active"
- "The contractor process worked perfectly" — actually we explicitly call out that the team missed `lock_rotation` and the freelancer caught it

What to claim:

- "We built a pre-registered falsification apparatus"
- "We surfaced our own architectural blind spots through external audit and ratified the lesson"
- "We refused to verdict pre-data"
- "Multiple ratified cycles of vocabulary refinement preceded any test run"
- "Stage 1 verdict is currently UNTESTABLE; here's why and what would resolve it"

The **last claim is the load-bearing skeptic-trust signal.** Sundog applications that ratify themselves are easy to dismiss. An application that explicitly says "we don't have the verdict yet, here's the apparatus, here's why it's pending" is much harder to dismiss because it's modeling research discipline, not result-fitting.

---

## 6. Future-pass execution outline

This pass = roadmap (THIS DOC). Future passes:

**Pass 2 (estimated 1-2 hours): refresh `APPLICATIONS.md` § Money Bags.**
- Refresh Inspectable Surfaces with new files
- Promote Current Metric Vocabulary to a real glossary with formulas (close gallery card's metric-glossary Need)
- Rewrite What It Demonstrates around the three evidence-bounded claims (Sundog adaptation / falsification apparatus / external audit precedent)
- Rewrite What To Measure Next reflecting current Stage 1 prereq state
- Add Pre-Stage-1 findings subsection (3.6.1 / 3.6.2 / 3.6.3 above)
- Tighten Claim Boundary

**Pass 3 (estimated 1-2 hours): refresh `applications-gallery.html` § Money Bags card.**
- Tier badge decision (Instrumented Prototype with refresh, OR Pre-Registered Falsification Apparatus sub-tier)
- Refresh stub-note `Needs:` line
- Update app-links to point at refreshed APPLICATIONS.md sections + optional MONEY_BAGS_DETAILED.md
- Optional polish on indirect-signal / transformation / output triplet

**Pass 4 (optional, 2-3 hours): write `docs/applications/MONEY_BAGS_DETAILED.md`.**
- Full Cycle 1-5 progression
- Four-mode taxonomy with operational definitions
- Gameplay-primitive corollary matrix (post-Stage-1-confirmation Stage 2 design surface preview)
- Architecture-knob inventory current state
- Audit-trail timeline

**Pass 5 (optional, 1 hour): cross-application comparison table refresh.**
- The existing table in APPLICATIONS.md § Cross-Application Comparison has Money Bags as one row
- Refresh that row to reflect current state (graph metrics → graph metrics + falsification apparatus + ratified vocabulary)
- Optionally add a new column to the cross-app table: "Evidence tier currently held"

**Pass 6 (optional, deferred): Money Bags writeup goes live on sundog.cc.**
- Coordinate with whoever maintains sundog.cc deploy
- Pre-deploy: read for marketing-ish drift; remove anything that survives this roadmap's voice discipline
- Post-deploy: track whether skeptic-observer feedback materially differs from prior application writeups (it should, because the apparatus story is structurally different from a confirmation story)

**Routing:** Pass 2 + Pass 3 are the minimum-viable refresh. Pass 4 is the deep-dive that turns the writeup into something genuinely useful for skeptic-observers + future-Claude. Passes 5-6 are polish + deploy.

---

## 7. Open questions for the research team

1. **Tier sub-classification.** Adopt "Pre-Registered Falsification Apparatus" as a new evidence tier between Instrumented Prototype and Research Result, or keep Instrumented Prototype with a note? Recommend the new tier; defer to research-team voice.
2. **Detailed doc location.** Sibling to `APPLICATIONS.md` (recommended, `docs/applications/MONEY_BAGS_DETAILED.md`) or expand `APPLICATIONS.md` itself? Tradeoff: APPLICATIONS.md's value is partially in cross-app readability; Money Bags now has enough specific content to overrun that.
3. **Voice on the falsification framing.** Is "we built apparatus to refute Sundog and refused to declare a verdict" too defensive, or appropriately disciplined? My reading: appropriately disciplined; matches the project's actual epistemic posture. But worth checking before final pass.
4. **Cross-app vocabulary alignment.** Money Bags now has a ratified vocabulary (rotation-locking / rotation-permissive / bias-axis). Should that vocabulary surface in the Cross-Application Comparison table (§ APPLICATIONS.md), or stay scoped to the Money Bags section? Recommend scoping it to Money Bags; cross-app should stay vocabulary-agnostic.
5. **Citation discipline.** When citing commits like `f78cb35` / `99444a5` / `4db91c9` / `7cf1087` — these are all in MB's repo, not Sundog's. Do we preserve the citation format or substitute repo-relative paths? Recommend preserving the commit hashes as audit-trail anchors; readers who want to verify can clone and inspect.

---

## 8. Closing note on this roadmap's epistemics

This roadmap is itself doing the thing it advocates: **pre-registering structure before execution.** The next 3-4 passes execute against this plan. If during execution we find the structure doesn't fit (e.g., the deep-dive doc grows differently than planned, or the tier sub-classification is rejected), update the roadmap before changing course. Defends against post-hoc reshape.

The Cycle 5 closing-footer note in `Shape_Coherence_Stage_1_Rubric.md` named the discipline pattern explicitly: *reframes that pass empirical validation within their own charter cycle have a much higher confidence floor than reframes that take multiple cycles to validate.* The same logic applies to documentation passes. If the writeup ratifies during its own roadmap cycle, that's a higher confidence floor than rewrites pile-up across many rounds.

— c.Echo, 2026-04-30 (Sundog contractor for Money Bags softbody simulation)
