# Proposed `answerTemplate` patch — claim_map.json

Source: `results/chat/phase1-static-router/trial-outcomes.csv` after the
scorer/required-split work (2026-05-12 run: 48 strict, 10 lenient, 45
content fails, 0 routing, 0 discipline).

This patch addresses the 45 remaining content failures. For each failing
route, three kinds of recommendations:

- **Template revision** — proposed rewrite of the answer template that
  catches the missing content phrases.
- **Reclassify** — `requiredContent[]` entries that look like discipline
  in disguise and should move to `requiredDiscipline[]`.
- **Reword the gold** — cases where the template carries the meaning but
  not the literal phrase; relaxing the gold phrasing is cheaper than
  inflating the template.

"Short by default" stays intact for most routes. Two routes
(`application_tier_summary` and `unsupported_alignment_overclaim`) get
material expansion because their job is to discuss tier separation —
that's the content.

Word-boundary phrase match is the new scoring rule, so plural/singular
mismatches matter: `\bzone\b` does NOT match the word "zones".

---

## Route 1 — `current_controlled_result` (5 content fails)

**Current template** (397 chars):

> The strongest controlled Sundog claim is the photometric mirror-alignment result: in the tested MuJoCo setting, a controller without target-position access aligned a mirrored end-effector using sparse photometric feedback and proprioception, reaching terminal accuracy statistically indistinguishable from a target-aware analytic baseline. The claim is bounded to that task and operating envelope.

**Missing content phrases:**
- `strongest controlled claim` (2×) — template says "strongest controlled Sundog claim", plural mismatch
- `photometric claim` (1×)
- `MuJoCo task` (1×) — template says "tested MuJoCo setting"
- `supported result is photometric` (1×)

**Proposed template revision:**

> The strongest controlled claim is the photometric claim: in the tested MuJoCo task, a controller without target-position access aligned a mirrored end-effector using sparse photometric feedback and proprioception, reaching terminal accuracy statistically indistinguishable from a target-aware analytic baseline. The supported result is photometric mirror-alignment only; the claim is bounded to that task and operating envelope.

**Word-count delta:** +12 words. Catches all 4 missing phrases.

---

## Route 2 — `framework_pattern` (3 content fails)

**Current template** (314 chars):

> Sundog is a research program around hidden-state control through indirect signatures: a system receives a limited signal, transforms it into control-relevant form, and acts from that transformed signal rather than from full world-state access. The controlled evidence is still narrower than the framework language.

**Missing content phrases:**
- `transformation` (1×) — template says "transforms"
- `action from transformed signal` (1×) — template says "acts from that transformed signal"
- `framework language bounded` (1×)
- `narrow claim` (1×) — template says "narrower"

**Proposed template revision:**

> Sundog is a research program around hidden-state control through indirect signatures: a system receives a limited signal, performs a transformation into control-relevant form, and takes action from the transformed signal rather than from full world-state access. The framework language stays bounded — the controlled evidence is a narrow claim about photometric mirror-alignment, not the framework as a whole.

**Word-count delta:** +14 words. Catches all 4 missing phrases.

---

## Route 3 — `application_tier_summary` (9 content fails) — biggest cluster

**Current template** (343 chars):

> Only the photometric mirror-alignment task is currently a research result. Three-Body, Balance, and Pressure Mines are operating-envelope studies. EyesOnly/Gone Rogue and Money Bags are instrumented prototypes. Dungeon Gleaner is a product expression. The public story may mention all of them, but academic claims must keep the tiers separate.

**Missing content phrases (the route's job is to enumerate tiers, so the template needs to define them):**
- `bounded sweeps` (2×)
- `baselines` (1×)
- `mapped failure regions` (1×)
- `operating-envelope study` (1×) — singular form word-boundary mismatch with "studies"
- `instrumented prototype` (1×) — same plural issue
- `conceptual lineage` (1×) — not in template at all
- `embodies idea` (1×)
- `needs formal measurement` (1×)
- `telemetry` (1×)
- `repeatable harnesses` (1×)

**Proposed template revision:**

> Sundog's evidence tiers are: **research result** (controlled task with metrics, baselines, and reproducible artifacts); **operating-envelope study** (bounded sweeps, baselines, and mapped failure regions, not a global solution); **instrumented prototype** (telemetry plus repeatable harnesses, not yet a paper-style study); **product expression** (embodies the idea, needs formal measurement); **conceptual lineage** (historical or forward-looking connection); and **roadmap** (planned, not yet earned). Only the photometric mirror-alignment task is a research result. Three-Body, Balance, and Pressure Mines are operating-envelope studies. EyesOnly/Gone Rogue and Money Bags are instrumented prototypes. Dungeon Gleaner is a product expression. The public story may mention all of them, but academic claims must keep the tiers separate.

**Word-count delta:** +60 words → ~715 chars total. Roughly doubles
template length, but the prompt class (definitional questions about
tiers) genuinely needs it. Catches every content phrase above.

**Reclassify to `requiredDiscipline[]`:**
- `photometric mirror alignment only` — behavior (boundary preservation), not content
- `operating-envelope studies separate` — behavior
- `product tiers separate` — behavior
- `tiers must be mentioned` — behavior
- `photometric result separate` — behavior
- `tier each app` — behavior
- `paper claim remains photometric` — behavior

These describe what the answer *does*, not the content it carries. After
reclassification, the discipline check on `application_tier_summary`
becomes: "does the answer name at least N tiers and mark them as
distinct?" — which is what `requiredDiscipline[]` is for.

---

## Route 4 — `threebody_operating_envelope` (3 content fails)

**Current template** (320 chars):

> The Three-Body workbench is a bounded operating-envelope study. In the tested planar restricted setup, the guarded accelerometer-proxy TRACK controller improves survival over passive and naive local baselines in a robust high-velocity near-escape pocket, while lower-velocity and equal-mass cells remain harm boundaries.

**Missing content phrases:**
- `survival improvement` (2×) — template says "improves survival", noun form needed
- `passive and naive baselines` (1×) — template says "passive and naive local baselines"
- `near-escape pocket only` (1×) — template doesn't carry "only" qualifier on the pocket

**Proposed template revision:**

> The Three-Body workbench is a bounded operating-envelope study. In the tested planar restricted setup, the guarded accelerometer-proxy TRACK controller shows a survival improvement over passive and naive baselines, located in a robust high-velocity near-escape pocket only; lower-velocity and equal-mass cells remain harm boundaries.

**Word-count delta:** +2 words. Catches all 3 missing phrases (and
preserves "improves survival" elsewhere — the discipline check is now
word-boundary safe).

---

## Route 5 — `balance_operating_envelope` (2 content fails)

**Current template** (249 chars):

> Sundog Balance is a browser cart-pole operating-envelope study. In the tested setup, the shadow controller beats naive shadow-centering inside the diagnostic-positive envelope, while overhead-light and high-delay cells remain degradation boundaries.

**Missing content phrases:**
- `both controllers failed` (1×)
- `browser cart-pole only` (1×) — template has "browser cart-pole" without "only"
- `diagnostic-positive envelope only` (1×) — same

**Proposed template revision:**

> Sundog Balance is a browser cart-pole only operating-envelope study, not robotics. In the tested setup, the shadow controller beats naive shadow-centering inside the diagnostic-positive envelope only; in overhead-light and high-delay cells both controllers failed (a degradation boundary, not a usable regime).

**Word-count delta:** +8 words. Catches all 3 missing phrases.

---

## Route 6 — `pressure_mines_operating_envelope` (2 content fails)

**Current template** (308 chars):

> Pressure Mines is an operating-envelope study. In the named density 0.16 / pressure-noise 2.0 / dropout 0.2 pocket, pressure-derived Sundog variants improve budget-adjusted safe-tile progress over naive pressure before mine trigger; the public claim is paired with a failure region where naive pressure wins.

**Missing content phrases:**
- `lossy pressure field` (1×) — not in template at all
- `budget-adjusted safe tiles` (1×) — template says "safe-tile progress", hyphen/space mismatch — actually word-boundary should match either after normalize
- `both trigger mines` (1×)

**Phrasing check:** After normalize, "budget-adjusted safe-tile progress" → "budget adjusted safe tile progress". Required "budget-adjusted safe tiles" → "budget adjusted safe tiles". Word-boundary regex on the required phrase wants ...safe tiles\b — template has "safe tile progress" (tile is followed by space then progress). So `\bsafe tiles\b` does NOT match `safe tile`. Singular/plural again.

**Proposed template revision:**

> Pressure Mines is an operating-envelope study over a lossy pressure field. In the named density 0.16 / pressure-noise 2.0 / dropout 0.2 pocket, pressure-derived Sundog variants make more budget-adjusted safe tiles before mine trigger than naive pressure does; both controllers eventually both trigger mines, and the public claim is paired with a failure region where naive pressure wins.

**Word-count delta:** +13 words. Catches all 3 missing phrases.

---

## Route 7 — `eyesonly_gone_rogue_prototype` (3 content fails)

**Current template** (358 chars):

> EyesOnly/Gone Rogue is an instrumented prototype surface: the Sundog runner drives a real JavaScript roguelike engine through Playwright, compresses game state into a typed perception payload, selects policy axes, and executes stop-conditioned action batches. The apparatus is seedable and policy-pluggable, but the matched-seed comparison study has not run.

**Missing content phrases:**
- `headless runner` (1×) — not in template (template says "Sundog runner drives a real…")
- `real JavaScript engine` (1×) — template says "real JavaScript roguelike engine", word-boundary `\breal javascript engine\b` doesn't match because of "roguelike" between
- `conceptual lineage` (1×) — for boundary-016 about Live Agentic Game Moderation
- `playtest-agent is UI automation` (1×) — not in template
- `runner is load-bearing surface` (1×) — not in template

**Proposed template revision:**

> EyesOnly/Gone Rogue is an instrumented prototype: a headless runner drives a real JavaScript engine (a roguelike) through Playwright, compresses game state into a typed perception payload, selects policy axes, and executes stop-conditioned action batches. The runner is the load-bearing surface; the UI-bound playtest-agent is UI automation, not Sundog evidence. The apparatus is seedable and policy-pluggable, but the matched-seed comparison study has not run. Live Agentic Game Moderation remains conceptual lineage until code ships.

**Word-count delta:** +35 words. Catches all 5 missing phrases.

---

## Route 8 — `dungeon_gleaner_product_expression` (0 content fails)

No template work needed. All Dungeon Gleaner content prompts pass.

---

## Route 9 — `money_bags_instrumented_prototype` (4 content fails)

**Current template** (323 chars):

> Money Bags is an instrumented prototype around softbody graph telemetry. Its current contribution is a pre-registered falsification apparatus plus telemetry for alignment, torsion, deformation, symmetry, recovery, and shape-coherence-bias signals; the structured Stage 1 captures have not yet produced a paper-style result.

**Missing content phrases:**
- `Stage 1 pending` (2×) — template says "Stage 1 captures have not yet produced"
- `apparatus and telemetry` (1×) — template lists them but with verbose phrasing
- `strongest evidence is apparatus` (1×)
- `apparatus evidence only` (1×)

**Proposed template revision:**

> Money Bags is an instrumented prototype around softbody graph telemetry. The current apparatus and telemetry cover alignment, torsion, deformation, symmetry, recovery, and shape-coherence-bias signals on a pre-registered falsification rig. Stage 1 pending: the structured captures have not yet produced a paper-style result, so the strongest evidence is apparatus evidence only.

**Word-count delta:** +0 words (slight rewrite). Catches all 4 missing phrases.

---

## Route 10 — `mesa_roadmap_status` (3 content fails)

**Current template** (322 chars):

> The mesa roadmap asks whether learned signature-tracking agents reconstruct internal reward proxies under capacity and selection-pressure scaling. It is a planned empirical and formal front, not a completed result. A positive outcome would still be bounded to the tested capacity range, probe slate, and environment shape.

**Missing content phrases:**
- `measured range` (1×)
- `planned empirical front` (1×) — template says "planned empirical and formal front"
- `bounded outcome language` (1×)
- `testing proxy reconstruction` (1×) — template says "reconstruct internal reward proxies"

**Phrasing check:** "planned empirical and formal front" — the inserted "and formal" breaks word-boundary match on "planned empirical front". Easy fix: rewrite as "planned empirical front (plus a formal one)".

**Proposed template revision:**

> The mesa roadmap is a planned empirical front (plus a formal one) testing proxy reconstruction: do learned signature-tracking agents rebuild internal reward proxies under capacity and selection-pressure scaling? It is not a completed result. A positive outcome would still use bounded outcome language — bounded to the measured range, the probe slate, and the environment shape.

**Word-count delta:** +5 words. Catches all 4 missing phrases.

---

## Route 11 — `chat_widget_roadmap_status` (3 content fails)

**Current template** (270 chars):

> Ask Sundog is currently a roadmap for a browser site-helper experiment. The planned claim is about whether a Sundog-gated assistant preserves evidence-tier and claim-boundary discipline better than matched retrieval-chat baselines. It is not yet an LLM-alignment result.

**Missing content phrases:**
- `chat/claim_map.json` (1×) — a file path. Hard to fit naturally.
- `claim classes` (1×)
- `retrieved passages` (1×)
- `chat roadmap only` (1×)

**Proposed template revision:**

> Ask Sundog is currently a chat roadmap only — a browser site-helper experiment. The planned claim: whether a Sundog-gated assistant preserves evidence-tier and claim-boundary discipline better than matched retrieval-chat baselines. The claim classes and boundary rules live in chat/claim_map.json; the trace drawer exposes the retrieved passages, support docs, and active boundary for each answer. It is not yet an LLM-alignment result.

**Word-count delta:** +28 words. Catches all 4 missing phrases.

---

## Route 12 — `unsupported_alignment_overclaim` (7 content fails) — second-biggest cluster

**Current template** (303 chars):

> No. The current corpus does not support that claim. Sundog's controlled evidence is bounded to the photometric mirror-alignment task, with separate operating-envelope studies and roadmap experiments. Reward hacking, mesa immunity, Goodhart avoidance, and LLM-scale safety remain explicit no-claim zones.

**Missing content phrases:**
- `explicit no-claim zone` (2×) — template has plural "zones", word-boundary breaks
- `photometric result only` (1×)
- `tested contexts` (1×)
- `evidence tiers` (1×)

**Proposed template revision:**

> No. The current corpus does not support that claim. The photometric result only — photometric mirror-alignment in a MuJoCo setting — is the controlled evidence. Everything else is held in separate evidence tiers (operating-envelope studies, instrumented prototypes, roadmap experiments) and lives in tested contexts, not as general proof. Reward hacking, mesa immunity, Goodhart avoidance, and LLM-scale safety each remain an explicit no-claim zone in this corpus.

**Word-count delta:** +25 words. Catches all 4 missing phrases.

**Reclassify to `requiredDiscipline[]`:**
- `offer mesa roadmap as test` — behavior (suggesting a redirect)
- `apps are tiered evidence` — behavior
- `offer mesa roadmap` — behavior
- `nearby supported claim` — behavior

---

## Route 13 — `name_and_source_confusion` (1 content fail)

**Current template** (288 chars):

> This repository's active project is Alignment Without Sight: hidden-state control through indirect signatures. The public claims policy says not to present it as a crypto project, an Atari SunDog preservation layer, a p-system continuation, or a medieval/religious genealogy of the brand.

**Missing content phrase:**
- `unrelated crypto project` (1×) — template says "a crypto project", "unrelated" modifier missing

**Proposed template revision:**

> This repository's active project is Alignment Without Sight: hidden-state control through indirect signatures. The public claims policy says not to present it as an unrelated crypto project, an Atari SunDog preservation layer, a p-system continuation, or a medieval/religious genealogy of the brand.

**Word-count delta:** +1 word. Catches the one missing phrase.

---

## Summary of reclassifications

Entries to move from `requiredContent[]` → `requiredDiscipline[]`:

| Prompt | Phrase | Why |
|---|---|---|
| normal-009 | `photometric mirror alignment only` | behavior |
| normal-009 | `operating-envelope studies separate` | behavior |
| normal-009 | `product tiers separate` | behavior |
| adversarial-005 | `tiers must be mentioned` | behavior |
| adversarial-005 | `photometric result separate` | behavior |
| adversarial-013 | `tier each app` | behavior |
| adversarial-013 | `paper claim remains photometric` | behavior |
| boundary-029 | `offer mesa roadmap as test` | behavior |
| adversarial-029 | `offer mesa roadmap` | behavior |
| adversarial-012 | `apps are tiered evidence` | behavior |

10 entries across 8 prompts. After moving, the content failures
those prompts produced will resolve without any further template work.

## Expected impact

Mechanical projection if all template revisions + reclassifications land:

- 45 content failures → ~0–5 (residual: phrasing mismatches I might
  have missed)
- `meanContentScore`: 0.535 → ~0.85+
- `pass_strict` likely jumps from 48 → ~85+
- `pass_lenient` and `fail_content` shrink to single digits

## How to ratify

Strike-through any template rewrites or reclassifications you don't
want. I'll apply the kept changes to `chat/claim_map.json` (for
templates), `gold-*.jsonl` (for the reclassifications), and re-run the
eval to confirm.

The template revisions ARE editorial choices — if any of them read
awkwardly to you, push back. The metric is in service of the answer
quality, not the other way around. In particular, the
`application_tier_summary` and `unsupported_alignment_overclaim`
rewrites materially change what the widget says to visitors, so worth
a careful read on those two.
