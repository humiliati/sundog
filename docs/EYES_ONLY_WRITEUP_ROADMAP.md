# Eyes Only / Gone Rogue Writeup Roadmap

**Filed by:** EyesOnly maintainer for Sundog applications correction (2026-05-08)
**Audience:** Sundog research project — the maintainer of sundog.cc; future-Claude executing this roadmap; skeptic-observers reading the eventual writeup
**Status:** Roadmap pass. This doc plans the correction; the next 2-3 passes execute against it.
**Sources:**
- `applications-gallery.html` § EyesOnly / Gone Rogue card (current state — flagged "Instrumented Prototype" with `Needs: matched-seed study, behavior clips, and baseline comparison`)
- `docs/APPLICATIONS.md` § EyesOnly / Gone Rogue (current state — frames the runner as "Sundog as procedural control under partial observation," lists JS-side surfaces and Python-side adapter)
- `docs/runners.md` (current state — fully documented Python+Playwright runner harness against `GoneRogue.headless`)
- `docs/MONEY_BAGS_WRITEUP_ROADMAP.md` and `docs/DUNGEON_GLEANER_WRITEUP_ROADMAP.md` (templates — voice, structure, evidence-tier discipline, retraction pattern)
- `<EyesOnly>/public/js/gone-rogue.js` — `GoneRogue.headless` API surface
- `<EyesOnly>/public/js/agent-api-system.js`, `agent-integration.js`, `agent-command-system.js` — agent surfaces
- `<EyesOnly>/public/js/playtest-agent.js` — UI-bound playtest agent (scope confirmed: UX-edge-case regression bot, not a Sundog policy)
- `<EyesOnly>/public/js/kernel-manager.js` — external agent integration layer
- `<EyesOnly>/docs/MANIPULATION_LAYER_AGENT_MODERATION.md` — Live Agentic Game Moderation (LAGM) design doc
- `<sundog>/runners/adapters/gone_rogue.py`, `runners/gone_rogue_headless.py`, `runners/gone_rogue_ui.py`, `runners/policies/gone_rogue_greedy.py`, `tests/runners/test_gone_rogue.py` — Sundog-side Python runner

---

## 0. The story shape — why this writeup is a partial correction plus a forward extension

The Money Bags writeup is an epistemics story (apparatus pre-registered, verdict refused).
The Dungeon Gleaner writeup is a mis-broadcast correction (glass/pressure-washing retracted, verb-field installed).
**The EyesOnly writeup is a third shape: tighten one over-claim, re-scope one mis-described claim, and disclose one forward-looking application that has been gestured at without grounding.**

The current public broadcast (gallery card + APPLICATIONS.md § EyesOnly / Gone Rogue) is mostly correct, but three lines need work:

1. **"Headless running test harnesses" framing is real but unmeasured.** The Python runner exists, the `GoneRogue.headless` API exists, the Playwright bridge runs. None of the matched-seed studies, baseline comparisons, or behavior clips listed as Needs in the gallery card have shipped. The current broadcast reads as if the harness *is* the result; it's only the apparatus.
2. **"Human-UI-bound playtest agent provides a separate constraint" is a misread of `playtest-agent.js`.** Inspected against the actual file, the agent is a UI-edge-case regression bot (`fan-breakout-clip`, `card-deploy-lost`, `touch-dead-zone`, etc.), not a policy-bound Sundog runner restricted to UI affordances. It's a useful piece of automation, but its scope is UX QA, not "Sundog under tighter constraints."
3. **"Level manipulation" is forward-looking design that hasn't been distinguished from shipped work.** The Live Agentic Game Moderation system (LAGM, in `<EyesOnly>/docs/MANIPULATION_LAYER_AGENT_MODERATION.md`) is a real design surface — competence model → moderator → next-floor intent → floor synthesis — and it is exactly the Sundog indirect-signal pattern applied above the game rather than below it. **It is not built.** Treating it as evidence flattens the apparatus/result/expression distinction the Sundog research program has spent effort establishing.

The honest broadcast is a stratified one:

- The **headless runner** is an Instrumented Prototype with a real Sundog turn-envelope architecture, real Playwright bridge, and a known measurement gap.
- The **UI-bound playtest agent** is a Product Expression of automation discipline, but **not a Sundog application** in the indirect-signal-to-action sense; it should be cited as a sibling automation surface, not as evidence for the theorem.
- The **LAGM design** is Conceptual Lineage / forward-looking design — the *intended* Sundog application above the game, with the design doc as inspectable surface, no shipping code yet.

**Tactical implication for voice:** the writeup needs to do three things in the same pass — (a) tighten the existing runner section to emphasise apparatus over result; (b) re-scope the playtest-agent line so it doesn't borrow Sundog credibility for UX automation; (c) install the LAGM forward-looking section with explicit Conceptual Lineage tier and no implication of evidence. None of this requires a hard retraction in the Dungeon Gleaner sense — the runner framing is honest, just incomplete. The playtest-agent line gets a one-paragraph re-scope. LAGM gets a clearly tiered "Forward-Looking Application Design" subsection.

---

## 1. Audience + voice calibration

**Three audiences, one voice (parallel to MB §1 and DG §1):**

1. **sundog.cc skeptics** — will reward "the harness exists, the matched-seed study does not yet, here's why" framing. Will leave on the first sentence that conflates harness with study.
2. **Sundog research community** — wants the architecture pointer (PERCEIVE / PLAN / EXECUTE_BATCH against a real product engine), the policy comparison plan, and the cross-application table refresh that distinguishes EyesOnly's "agent under the game" from the LAGM "agent above the game" and from Money Bags' apparatus and Dungeon Gleaner's verb-field.
3. **Game-dev / live-ops / interested observers** — wants to see how a procedural roguelike can be driven by a Sundog turn-envelope runner (for balance testing, regression, and matched-seed studies) and what an above-game adaptive moderator would look like if built. This is the framing that makes the writeup useful to others working on procedural content with a small live-ops budget.

**Voice continues to match the existing `APPLICATIONS.md`, MB roadmap, and DG roadmap:** past-tense for shipped, present-tense for pending, explicit "Safe claim" / "Avoid", evidence-tier badges. New structural element specific to this writeup: a **"Three Surfaces"** subsection that names each surface and tags its tier independently, so the runner's Instrumented Prototype credit can't slosh over onto LAGM's Conceptual Lineage.

**Tier classification (defer to research-team ratification):**

| Surface | Current tier | Proposed tier | Rationale |
| --- | --- | --- | --- |
| Headless turn-envelope runner | Instrumented Prototype (whole card) | Instrumented Prototype | Real PERCEIVE/PLAN/EXECUTE_BATCH harness; real Playwright bridge; real `GoneRogue.headless` API; matched-seed study not yet run. |
| UI-bound playtest agent | (folded into Instrumented Prototype) | Product Expression — scoped sibling, **not Sundog evidence** | Real shipped automation; scope is UX-edge-case regression. Cite as automation neighbour, not theorem expression. |
| Live Agentic Game Moderation | (gestured at as "level manipulation") | Conceptual Lineage / Forward-Looking Application Design | Design doc only; no shipping code; clear Sundog-shape on paper. |

The card-level tier badge stays "Instrumented Prototype" because the runner is the load-bearing Sundog evidence in this product. The body of the section names the three surfaces separately so the reader can see which tier each piece holds.

---

## 2. Doc structure proposal

Three docs touched, one optional new doc — same shape as MB and DG:

```
sundog/
├── applications-gallery.html
│   └── § EyesOnly / Gone Rogue card
│       ├── Tier badge — stays "Instrumented Prototype"
│       ├── Indirect signal / Transformation / Output triplet — REPLACE with concrete Gone Rogue language
│       │   (currently generic: compressed game state / turn envelopes / coherent action batches;
│       │    new: floor/biome/HP/alert/inventory state + STR-combat flag / axis selection
│       │    and stop-conditioned batches against GoneRogue.headless / matched-seed runs producing JSONL telemetry)
│       ├── stub-note "Needs:" — refresh to enumerate the three open studies
│       └── App-links — point at refreshed APPLICATIONS.md + runners.md + optional new detailed doc
├── docs/
│   ├── APPLICATIONS.md
│   │   └── § EyesOnly / Gone Rogue — major refresh
│   │       ├── NEW SUBSECTION: Three Surfaces — explicit tier-per-surface table
│   │       ├── Sundog Expression — keep, light edits to anchor on the runner specifically
│   │       ├── Inspectable Surfaces — REPLACE (split into Sundog-side runner / EyesOnly-side runner-target / EyesOnly-side adjacent-but-not-evidence / EyesOnly-side forward-looking-design)
│   │       ├── What It Demonstrates — REWRITE around the apparatus-vs-result discipline
│   │       ├── What To Measure Next — REWRITE; current items are right but priority-ordered around the matched-seed study
│   │       ├── NEW SUBSECTION: Forward-Looking Application Design — LAGM as the planned above-game Sundog expression
│   │       └── Claim Boundary — TIGHTEN around the three surfaces
│   └── applications/                                    [NEW DIRECTORY, OPTIONAL — same call as MB/DG]
│       └── EYES_ONLY_DETAILED.md                        [NEW, OPTIONAL]
│           └── Runner architecture deep-dive, policy taxonomy, matched-seed study design,
│              UI-bound playtest agent scope clarification, LAGM intent-spec walk-through,
│              and the "agent above vs agent below the game" framing
```

**Decision point for Pass 2:** does the LAGM walk-through and the matched-seed-study design go in `APPLICATIONS.md` (making § EyesOnly grow significantly) or in a sibling `applications/EYES_ONLY_DETAILED.md`? Default recommendation: **sibling file**, mirroring the MB/DG recommendation. The cross-application value of `APPLICATIONS.md` is partially in its readable-in-one-sitting length, and EyesOnly already has the largest cross-link surface (Sundog runners.md, runner Python, JS-side API, JS-side playtest agent, JS-side LAGM design doc).

---

## 3. Content outline for the refreshed `APPLICATIONS.md` § EyesOnly / Gone Rogue

The structure stays parallel to other application sections (research-doc voice consistency). Each subsection's intent + content delta:

### 3.1 NEW SUBSECTION — Three Surfaces

Short table-plus-paragraph at the top of § EyesOnly / Gone Rogue, before Sundog Expression. Sample shape:

> EyesOnly / Gone Rogue contains three Sundog-adjacent surfaces, each at a different evidence tier. The card-level tier reflects the strongest one. The other two are named explicitly so they don't borrow credibility from each other.
>
> | Surface | Where | Tier | What it is |
> | --- | --- | --- | --- |
> | Headless turn-envelope runner | `<sundog>/runners/` ↔ `<EyesOnly>/public/js/gone-rogue.js` (`GoneRogue.headless`) | Instrumented Prototype | A Sundog PERCEIVE/PLAN/EXECUTE_BATCH runner driving the real Gone Rogue JS engine through Playwright. |
> | UI-bound playtest agent | `<EyesOnly>/public/js/playtest-agent.js` | Product Expression — automation neighbour, **not Sundog evidence** | A regression bot that hunts UX edge cases (fan-breakout-clip, card-deploy-lost, etc.) under human UI constraints. Sibling automation, not an indirect-signal-to-action transformation. |
> | Live Agentic Game Moderation (LAGM) | `<EyesOnly>/docs/MANIPULATION_LAYER_AGENT_MODERATION.md` | Conceptual Lineage / Forward-Looking Application Design | A design for an above-game agent that reads compressed player-competence telemetry and emits next-floor intent. Not built. |

This table is load-bearing for skeptic trust. It explicitly says *one* of the three surfaces is the Sundog evidence in this card; *one* is sibling automation that should not be cited as theorem evidence; *one* is a planned future application with no code yet.

### 3.2 Sundog Expression *(light edits)*

Existing text already frames the indirect signal correctly (compressed perception → axis selection → stop-conditioned batched action). Edits:

- Anchor on the headless runner specifically rather than "EyesOnly applies Sundog" generically. The headless runner is the Sundog application in this product.
- Add explicit citation to `<sundog>/docs/runners.md` for the `TURN_ENVELOPE { PERCEIVE / PLAN / EXECUTE_BATCH }` block — the runner doc already publishes the formal architecture, so APPLICATIONS.md should reference rather than restate.
- Replace the fuzzy "compressed game state" with the actual perception payload fields (`floor_index`, `biome`, `hp_ratio`, `alert_level`, `enemies_visible`, `in_combat`, `keys_inventory`, `gates_visible`, etc., per `runners.md` § PERCEIVE). Concrete field names are skeptic-trust signal; "compressed game state" reads as marketing.

### 3.3 Inspectable Surfaces *(REPLACE — split by surface)*

Existing list mixes Sundog-side and EyesOnly-side files without distinguishing the runner from the playtest agent from the LAGM design. Replace with a structured list:

**Sundog-side runner (Python):**
- `runners/adapters/gone_rogue.py` — Playwright adapter; `compress_perception`, `run_turn_envelope`.
- `runners/gone_rogue_headless.py` — headless batch CLI.
- `runners/gone_rogue_ui.py` — visible-browser CLI for inspection / screenshots.
- `runners/policies/gone_rogue_greedy.py` — built-in greedy policy with axis-selection + volatility.
- `runners/adapters/gone_rogue_harness.html` — self-contained browser harness page.
- `tests/runners/test_gone_rogue.py` — unit + integration tests against the JS API.
- `docs/runners.md` — formal turn-envelope architecture, CLI flags, policy contract.

**EyesOnly-side runner target (the engine the runner drives):**
- `public/js/gone-rogue.js` — exposes `GoneRogue.headless.{getState, getLegalActions, applyAction, getGrid, resetToState}` and `GoneRogue.setSeed`.
- `public/js/agent-api-system.js` — the headless action API surface.
- `public/js/agent-integration.js` — agent testing bridge with UI modes.
- `public/js/agent-command-system.js` — AGENT command surface for in-game agent control.
- `public/js/kernel-manager.js` — external agent integration layer.

**EyesOnly-side automation neighbour (sibling, not Sundog evidence):**
- `public/js/playtest-agent.js` — UI-bound regression bot for UX edge cases (fan-breakout-clip, card-deploy-lost, touch-dead-zone, z-index-inversion, orientation-layout-break, etc.). Cited here so readers know what it is and why it is *not* the load-bearing Sundog surface.

**EyesOnly-side forward-looking design (Conceptual Lineage):**
- `docs/MANIPULATION_LAYER_AGENT_MODERATION.md` — Live Agentic Game Moderation system: Player Competence Model → Live Agentic Moderator → Floor Synthesis Engine → Human-in-the-Loop Console. No shipping code. Inspectable as design surface.

### 3.4 What It Demonstrates *(REWRITE)*

Existing text is fine for the runner, but it implies the playtest agent demonstrates "Sundog under tighter constraints" which the playtest-agent source does not support. Rewrite around three demonstration claims, each evidence-bounded:

1. **The Sundog turn envelope can drive a real product engine through a stable headless API.** `GoneRogue.headless` is a public engine surface in EyesOnly; the Sundog Python adapter calls it through Playwright; PERCEIVE/PLAN/EXECUTE_BATCH runs end-to-end with a greedy policy. This is the Instrumented Prototype claim — the apparatus exists and runs. *Evidence tier:* Instrumented Prototype.
2. **The runner architecture is policy-pluggable and seedable.** `GoneRogueGreedyPolicy` is one implementation of the runner's policy contract; new policies subclass it and register in the CLI. `GoneRogue.setSeed(seed)` makes runs deterministic. The harness can therefore support matched-seed multi-policy comparison studies whenever the policy set is filled out. *Evidence tier:* Instrumented Prototype (apparatus complete; matched-seed study pending).
3. **A UI-bound automation surface exists alongside the headless runner, scoped to UX regression.** `playtest-agent.js` runs under human UI constraints, but its job is UX-edge-case detection (touch targets, z-index, viewport clipping, animation loss), not policy-bounded play. It is a useful sibling automation surface; it is not a Sundog policy. *Evidence tier:* Product Expression — scoped sibling automation, not theorem evidence.

NOT a fourth demonstration claim about LAGM moderating live floors. LAGM is a design doc; it goes in § Forward-Looking Application Design with Conceptual Lineage tier.

### 3.5 What To Measure Next *(REWRITE — priority-ordered)*

Existing list is honest (success metrics, comparisons, matched seeds, etc.). Rewrite as a priority-ordered list with each item tied to a concrete next study:

**Priority 1 — Matched-seed study (the gallery card's load-bearing Need).**

- Define success metrics: floor reached, survival rate (`outcome == "died"` rate by floor), resources collected, run volatility, cards wasted, combat losses, completion time, total steps.
- Implement at least three policies for comparison: the existing `greedy`, plus `random_legal` (uniform over `getLegalActions`), plus `target_aware` (uses authorial knowledge — exit position, gate solutions — to upper-bound performance).
- Run matched seeds across all policies via `GoneRogue.setSeed` + run-index offset (the runner already supports this through `--seed` and `--runs`).
- Report per-policy outcome distributions on the same seed set.
- Filed deliverable: `<sundog>/results/eyesonly/matched_seed_<datetime>/` containing per-policy JSONL bundles + an aggregate summary table.

**Priority 2 — Compressed-perception ablation.**

- Run the greedy policy with the full perception payload vs. a reduced payload (e.g., HP/alert only, no inventory; or inventory only, no enemies).
- Report where compressed perception helps and where it loses information; this is the most direct empirical handle on the Sundog "indirect signal" claim in this domain.

**Priority 3 — Volatility-threshold sensitivity sweep.**

- The runner accepts `--volatility F`, default `0.7`. Sweep `F ∈ {0.3, 0.5, 0.7, 0.9}` and read survival rate + steps-per-floor.
- Report whether shorter batches under high volatility actually help; this is the structural claim of the turn envelope shape.

**Priority 4 — Behavior clips for the gallery.**

- The UI-bound runner (`gone_rogue_ui.py`) supports `--screenshot-dir`. Capture short loops of `greedy` succeeding, `greedy` failing, and `random_legal` failing on identical seeds. These are the "behavior clips" the gallery card lists as a Need.

**Priority 5 — Cross-policy data for the LAGM training set.**

- Even though LAGM is not built, matched-seed JSONL bundles from the studies above are exactly the kind of synthetic player telemetry a future LAGM moderator would train against. The runner's output is therefore *also* infrastructure for the planned above-game Sundog application. This is the bridge between Priority 1 and the forward-looking section below.

### 3.6 NEW SUBSECTION — Forward-Looking Application Design (LAGM)

A clearly tiered subsection that says the planned thing is planned. Sample shape:

> **Forward-looking application design.** The current Sundog evidence in EyesOnly is the under-game runner. There is a separate, currently unbuilt, above-game application designed in `<EyesOnly>/docs/MANIPULATION_LAYER_AGENT_MODERATION.md`: the Live Agentic Game Moderation (LAGM) system.
>
> LAGM's Sundog shape, on paper, is exactly the indirect-signal-to-action pattern applied above the game rather than below it:
>
> - **Indirect signal:** per-floor player telemetry (completion time, damage taken, ability usage, rope sophistication, puzzle solve time, backtracking, secret discovery, exploit pattern attempts, hoarding behaviour).
> - **Transformation:** the Player Competence Model compresses telemetry into `competence`, `confidence`, `frustration`, and per-system `mastery` indices — operating on indirect projections of skill rather than ground-truth player intent.
> - **Output:** the Live Agentic Moderator emits a structured `next-floor intent` (goal, difficulty delta, synergy introduction, exploit counter, narrative tone) that the Floor Synthesis Engine consumes to author the next floor.
>
> This is the same indirect-signal-to-action shape as the photometric experiment, the verb-field NPC system, and the softbody graph telemetry, applied to live game pacing. **It is not built.** The design doc is the inspectable surface; no LAGM code ships in EyesOnly today. It is named here as Conceptual Lineage / Forward-Looking Application Design so that the broadcast can mention "level manipulation" honestly without implying it is currently demonstrated.
>
> The under-game runner's matched-seed study output (Priority 1 above) is the natural feeder for any eventual LAGM training or evaluation set — the same telemetry shape the moderator would consume in production. That sequencing is intentional: build the under-game runner, generate matched-seed data, then use it as the empirical floor for the above-game moderator.

### 3.7 Claim Boundary *(TIGHTEN around three surfaces)*

Existing safe-claim line: *"EyesOnly shows Sundog-derived turn envelopes operating against a real procedural roguelike through compressed perception and stop-conditioned action batches."*

That line is correct for the runner. Replace with a three-part bounded claim that surfaces the tier discipline:

> **Safe claims:**
>
> - The under-game **headless turn-envelope runner** (`<sundog>/runners/`) drives the real Gone Rogue JS engine through a Playwright bridge against the `GoneRogue.headless` API. PERCEIVE compresses floor / biome / HP / alert / inventory / combat state into a typed perception payload; PLAN selects an axis (progression / resource / survival) and emits a stop-conditioned action batch; EXECUTE_BATCH applies actions until a stop condition fires. The architecture is policy-pluggable and seedable.
> - A **UI-bound playtest agent** (`<EyesOnly>/public/js/playtest-agent.js`) ships alongside the runner, scoped to UX-edge-case regression hunting. It is a sibling automation surface, not a Sundog policy expression.
> - A **Live Agentic Game Moderation** system (`<EyesOnly>/docs/MANIPULATION_LAYER_AGENT_MODERATION.md`) is designed but not built. It is named here as the planned above-game Sundog application, with the design doc as inspectable surface and no shipping code.
>
> **Avoid:**
>
> - "EyesOnly proves the theorem for procedural games." The runner is apparatus; the matched-seed study and policy comparison have not run.
> - "The playtest agent demonstrates Sundog under tighter UI constraints." The playtest agent's scope is UX regression, not policy comparison; folding it into the Sundog evidence column conflates QA automation with theorem expression.
> - "EyesOnly performs Sundog-driven level manipulation." LAGM is currently a design doc. Treating it as evidence flattens the apparatus / result / expression / lineage tier discipline that the rest of the Sundog research program holds.

That is the bounded claim that survives skeptic audit and matches the actual state of the three surfaces.

---

## 4. Cross-reference shape

```
applications-gallery.html § EyesOnly / Gone Rogue card
  ├─ links to → docs/APPLICATIONS.md § EyesOnly / Gone Rogue
  ├─ links to → docs/runners.md (the formal harness doc)
  └─ stub-note "Needs:" → refresh to:
       "Needs: matched-seed multi-policy study (greedy / random_legal / target_aware),
        compressed-perception ablation, volatility-sweep sensitivity reads,
        and behavior clips."

docs/APPLICATIONS.md § EyesOnly / Gone Rogue
  ├─ § Three Surfaces → tier-per-surface table (load-bearing for skeptic trust)
  ├─ § Inspectable Surfaces → split by surface, cross-references
  │      <sundog>/docs/runners.md
  │      <sundog>/runners/adapters/gone_rogue.py
  │      <EyesOnly>/public/js/gone-rogue.js
  │      <EyesOnly>/public/js/playtest-agent.js (cited as sibling, not evidence)
  │      <EyesOnly>/docs/MANIPULATION_LAYER_AGENT_MODERATION.md (cited as design surface)
  ├─ § Forward-Looking Application Design → LAGM
  └─ optional deeper link to:
         docs/applications/EYES_ONLY_DETAILED.md (if we make it)

docs/applications/EYES_ONLY_DETAILED.md (NEW, OPTIONAL)
  ├─ Runner architecture deep-dive (formalises and extends runners.md for the research-program voice)
  ├─ Policy taxonomy + comparison study design (greedy / random_legal / target_aware / volatility-aware)
  ├─ Matched-seed study protocol (seeds, run counts, metric definitions, reporting template)
  ├─ Compressed-perception ablation design (which fields, which subsets, expected signatures)
  ├─ UI-bound playtest agent scope clarification (what it is, what it isn't, why it ships alongside)
  ├─ LAGM intent-spec walk-through (telemetry → indices → next-floor intent JSON, with the
  │      explicit "no code yet" framing)
  └─ Above-vs-below the game framing (the conceptual contribution this application makes
       to the Sundog research program: indirect-signal pattern composes vertically across
       runner and moderator at the same product engine)
```

---

## 5. Skeptic-friendly evidence pattern

What to **show** (concrete, inspectable, dated):

- The `runners.md` harness doc — readers can verify the PERCEIVE/PLAN/EXECUTE_BATCH architecture matches the Sundog turn-envelope formulation in the theorem text.
- The `gone_rogue.py` adapter source — readers can verify the perception compression is real and that `compress_perception` produces typed fields, not free-form blobs.
- The `gone_rogue_greedy.py` policy — readers can verify the axis selection and volatility logic are concrete and pluggable.
- The CLI flags table in `runners.md` — readers can verify the harness is configurable, seedable, and documented at the operator-flag level.
- The `playtest-agent.js` `EDGE_CASES` constant — readers can directly verify the agent's scope is UX regression, not Sundog policy. Citing the constant by name is a load-bearing skeptic-trust signal because it is checkable in one click.
- The LAGM design doc's competence-model JSON shape and `next-floor intent` JSON shape — readers can verify the indirect-signal-to-action pattern in the design without us hand-waving "Sundog applies here."

What to NOT **claim**:

- "Sundog confirmed for procedural games."
- "Headless runner produces evidence the turn envelope outperforms baseline policies." (No matched-seed study has run.)
- "The playtest agent is a UI-bounded Sundog policy."
- "EyesOnly performs adaptive level moderation through Sundog telemetry." (LAGM is unbuilt.)
- "The runner can already drive matched-seed comparisons." (It can — the apparatus supports it; the studies have not been run.)

What to claim:

- "We built a Sundog turn-envelope runner against a real product game engine through a stable headless API."
- "The harness is policy-pluggable and seedable; matched-seed comparison studies are in priority order in the next-pass plan."
- "The playtest agent is sibling automation scoped to UX regression and is not part of the Sundog evidence column."
- "The Live Agentic Game Moderation system is the planned above-game Sundog application; it is currently a design doc, not shipping code."
- "The under-game runner's matched-seed output is the natural feeder for the above-game moderator's training and evaluation; that sequencing is intentional."

The **last two claims are the load-bearing skeptic-trust signals.** They model research discipline: name what is built, name what is designed, name what is intentionally sequenced after measurement. An application writeup that does this is much harder to dismiss than one that conflates apparatus, sibling automation, and forward-looking design into a single broadcast.

---

## 6. Future-pass execution outline

This pass = roadmap (THIS DOC). Future passes:

**Pass 2 (estimated 1-2 hours): refresh `APPLICATIONS.md` § EyesOnly / Gone Rogue.**
- Add Three Surfaces table at the top of the section.
- Edit Sundog Expression to anchor on the runner specifically with concrete perception payload fields.
- Replace Inspectable Surfaces with the four-bucket split (Sundog-side runner / EyesOnly-side runner target / sibling automation / forward-looking design).
- Rewrite What It Demonstrates around three evidence-bounded claims (apparatus / policy-pluggable harness / scoped sibling automation).
- Rewrite What To Measure Next as priority-ordered (matched-seed study → ablation → volatility sweep → clips → LAGM feeder).
- Add Forward-Looking Application Design subsection covering LAGM with explicit Conceptual Lineage tier.
- Tighten Claim Boundary into three-surface safe-claim block + explicit avoid list.

**Pass 3 (estimated 30-60 minutes): refresh `applications-gallery.html` § EyesOnly / Gone Rogue card.**
- Tier badge stays "Instrumented Prototype" (the runner is the load-bearing surface).
- Replace the indirect-signal / transformation / output triplet with concrete Gone Rogue language (floor/biome/HP/alert payload; axis-selected stop-conditioned batches; matched-seed JSONL telemetry).
- Refresh stub-note `Needs:` to the four-item priority list (matched-seed study, ablation, volatility sweep, clips).
- Update app-links to point at refreshed APPLICATIONS.md sections + `runners.md` + optional `applications/EYES_ONLY_DETAILED.md`.

**Pass 4 (optional, 2-3 hours): write `docs/applications/EYES_ONLY_DETAILED.md`.**
- Runner architecture deep-dive (formalising runners.md for the research-program voice).
- Policy taxonomy + comparison study design.
- Matched-seed study protocol (filed deliverable shape, seeds, metrics, reporting template).
- Compressed-perception ablation design.
- UI-bound playtest agent scope clarification.
- LAGM intent-spec walk-through with explicit "no code yet" framing.
- Above-vs-below-the-game framing as conceptual contribution to the Sundog research program.

**Pass 5 (optional, 1 hour): cross-application comparison table refresh.**
- The existing table in APPLICATIONS.md § Cross-Application Comparison has EyesOnly as one row with `compressed and volatile game state` / `turn envelope and stop conditions` / `coherent action batches`. That row is OK but vague.
- Refresh that row to read: indirect signal = `floor/biome/HP/alert/inventory/combat state from GoneRogue.headless.getState`; transformation = `perception compression + axis-selected stop-conditioned action batches against a real product engine through Playwright`; actionable output = `seedable, policy-pluggable runs with structured JSONL telemetry suitable for matched-seed comparison studies and as feeder data for an above-game moderator`.
- Optionally split EyesOnly into two rows (under-game runner / above-game LAGM) when LAGM ships, to surface the vertical-composition contribution. Until then, keep one row.

**Pass 6 (optional, deferred): EyesOnly writeup goes live on sundog.cc.**
- Coordinate with whoever maintains sundog.cc deploy.
- Pre-deploy: read for marketing-ish drift; the most likely drift is recombining the three surfaces back into one card-level claim. Voice discipline says don't.
- Post-deploy: track whether skeptic-observer feedback materially differs from prior application writeups (it should, because the explicit tier-per-surface table is structurally different from a single-tier card).

**Routing:** Pass 2 + Pass 3 are the minimum-viable refresh. Pass 4 is the deep-dive that turns the writeup into something genuinely useful for skeptic-observers + future-Claude + game devs designing similar runner-plus-moderator architectures. Passes 5-6 are polish + deploy.

---

## 7. Open questions for the research team

1. **Tier-per-surface presentation.** Adopt the Three Surfaces table at the top of § EyesOnly, or bury the surface-distinguishing language inside paragraphs? Recommend the table; it is the load-bearing skeptic-trust signal and the cheapest place to put it. Defer to research-team voice.
2. **Re-scope of the playtest-agent line.** Current text in APPLICATIONS.md says the playtest agent "provides a separate constraint" implying Sundog-evidence sibling status. Re-scope to "automation neighbour, not theorem evidence" as proposed, or keep the constraint framing? Recommend re-scope; the agent's actual file scope is UX regression and the broadcast should match the source.
3. **LAGM treatment.** Cite LAGM as Conceptual Lineage / Forward-Looking Application Design (recommended), or omit until built? Recommend cite-with-tier-discipline; omitting it leaves the existing "level manipulation" placeholder in broadcasts and the user-visible language doesn't match the doc-level language. Citing-with-tier-discipline closes the gap honestly.
4. **Detailed doc location.** Sibling to `APPLICATIONS.md` (recommended, `docs/applications/EYES_ONLY_DETAILED.md`) or expand `APPLICATIONS.md` itself? Tradeoff matches MB and DG: APPLICATIONS.md's value is partially in cross-app readability; EyesOnly already has the largest cross-link surface and would likely overrun.
5. **Above-vs-below-the-game framing as a research-program contribution.** Once both surfaces exist (runner shipped, LAGM shipped), the EyesOnly application would be the only one in the program demonstrating that the Sundog indirect-signal pattern composes vertically at the same product. Worth surfacing as an explicit conceptual contribution in § Cross-Application Comparison, or scope it to § EyesOnly until LAGM lands? Recommend scope it to § EyesOnly until LAGM lands, then promote.
6. **Citation format for the runner CLI flags and policy class names.** When citing `--max-batch`, `GoneRogueGreedyPolicy`, `GoneRogue.headless.applyAction`, etc., do we keep the inline-code formatting per the existing voice, or substitute prose? Recommend keep inline-code; concrete identifiers are skeptic-trust signal in a way that prose paraphrase isn't.

---

## 8. Closing note on this roadmap's epistemics

This roadmap is doing the same thing the Money Bags and Dungeon Gleaner roadmaps did: **pre-registering the structure of a writeup before writing it**. The next 2-3 passes execute against this plan. If during execution a structural decision turns out wrong (e.g., the Three Surfaces table reads as bureaucratic rather than disciplined, or LAGM gets de-scoped, or a fourth Sundog surface in EyesOnly surfaces), update the roadmap before changing course. Defends against post-hoc reshape.

The contribution this writeup makes to the Sundog research program is specifically the **tier-per-surface discipline**. Money Bags' contribution was apparatus pre-registered before captures. Dungeon Gleaner's contribution was naming the actual Sundog application after retracting the broadcast that named the wrong one. EyesOnly's contribution is showing that a single product can host multiple Sundog-shaped surfaces at different evidence tiers without conflating them — and that the under-game runner's matched-seed output is the natural empirical floor for an above-game moderator that hasn't been built yet. That sequencing — apparatus, then study, then up-tier the moderator from design to instrumented prototype — is the discipline this writeup pre-registers.

— EyesOnly maintainer, 2026-05-08 (Sundog application correction pass for flapsandseals.com / Gone Rogue)
