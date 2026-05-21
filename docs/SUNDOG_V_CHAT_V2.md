# Sundog Chat v2 — Product Charter

> **Hypothetical framing.** This document treats Ask Sundog as the v1.0
> deliverable of a small non-profit. There is no such non-profit yet. The
> framing exists because a rag-tag group of hobbyist contributors is the
> realistic shape of who will polish and ship this thing, and that crowd
> needs a product spec — not a research roadmap — to grab work from.

> **Sister doc.** [`SUNDOG_V_CHAT.md`](SUNDOG_V_CHAT.md) remains the
> research roadmap (Phases 0–12, eval discipline, claim-ratchet candidates,
> coupled-surfaces integrity work). v2 does **not** supersede it — they
> compose. v1 governs *what is honest*. v2 governs *what gets shipped*. If
> the two ever disagree, v1 wins on claim language and v2 wins on
> release decisions.

Working hook:

> Ask Sundog is a browser-native chat widget that answers questions about
> sundog.cc without promoting roadmap items into research results. The v1.0
> ship is a polished, accessible, mobile-friendly version of the widget
> that's been running internally since Phase 6 — packaged so any visitor
> can demo it, any contributor can grab a ticket against it, and any
> skeptic can inspect the evidence trace behind every answer.

---

## 1. What v2 Is For

v1 was written for Sundog research maintainers and future implementers. It
is dense, technical, and frequently load-bearing on prereg discipline.

v2 is written for four other audiences that v1 does not serve well:

- **Hobbyist contributors** who want to grab one ticket and ship it
  without first reading 2,200 lines of phase history.
- **Newcomers to ML** who are coming to the project to learn and need a
  visible ladder of tractable work that compounds.
- **Polishers and web devs** with no ML background who want to know what
  "polished" means concretely and what artifacts they own.
- **A hypothetical funder or external observer** who wants to know what
  the deliverable actually is, what it claims, and what it explicitly
  does not claim, without parsing a research document.

The reader is assumed to be smart, but not embedded in the project's
internal language. Terms like *evidence tier*, *claim boundary*, *trace*,
and *gate* are defined where they first appear, and the deeper rationale
lives in v1.

---

## 2. The Product

### 2.1 What it is

**Ask Sundog** is a chat widget that lives on every page of sundog.cc. A
visitor clicks the floating launcher, asks a question, and gets a short,
sourced answer with an inspectable trace.

Three visible layers:

1. **Chat pane.** Ordinary answer surface. Short answers by default. Links
   to docs, demos, and workbench pages where appropriate.
2. **Evidence rail.** A row of tier chips under each answer naming the
   strength of the supporting evidence: *Research Result*,
   *Operating-Envelope Study*, *Instrumented Prototype*, *Product
   Expression*, *Conceptual Lineage*, *Roadmap*, *Unsupported*. The chips
   are not decorative — they are the same evidence tiers the project
   uses internally to decide what counts as a claim.
3. **Trace drawer.** A collapsible panel revealing the retrieved
   passages, the active claim boundary, the deterministic route the
   answer came from, and the reason the answer was allowed (or blocked).

### 2.2 What makes it different from a normal chatbot

Two things, both visible to the user:

**It refuses overclaim, on purpose.** A normal RAG chatbot will happily
upgrade promotional copy ("Sundog is a breakthrough framework that
solves alignment") into a research-result-shaped answer if a user asks
the right pressure-test question. Ask Sundog cannot do this. A
deterministic claim gate sits between the model and the user. The gate
reads the route's authorized evidence tier from the claim map and
refuses any draft that asserts a stronger claim than that route allows.
This is not a system prompt instruction. It is a hard structural rule.

**Every answer carries its receipts.** The trace drawer is not a
debugging surface — it ships as a first-class part of the product. The
user can always see what document the answer came from, what tier of
evidence that document was authorized at, and what claim language was
allowed or refused. This is the polished version of a research
discipline: instead of asking the visitor to trust the answer, the
widget shows them the bones.

### 2.3 What it is not

Ask Sundog is **not** a general-purpose assistant. It does not answer
questions outside the sundog.cc corpus. It does not write code, summarize
arbitrary web pages, or do open-ended chat. Asking it for the weather is
not a clever stress test — it is just a category error.

Ask Sundog is **not** a research result about LLM alignment. It is a
narrowly scoped public artifact built on top of a research result about
trace-conditioned mediation. The bounded research claim lives in v1 §13
and Phase 7. The product does not exceed it.

---

## 3. Who It's For

Three audiences, ranked by how directly the product serves them.

**Visitors to sundog.cc.** Someone landed on the homepage, read the
elevator pitch, and has a question — *what is a sundog?*, *what does the
research actually claim?*, *which of these halo families is real?* The
widget is the navigation layer for the site, the FAQ, and the "ask the
authors" surface, all collapsed into one tool. This is the bulk of the
traffic. The product succeeds for this audience when answers are fast,
short, accurate, and link out to the right deeper doc.

**Skeptic-observers.** A reader who suspects the site is overclaiming
and wants to find the spot where the prose outruns the evidence. The
widget is for them too — perhaps more than for the casual visitor. The
trace drawer, the tier chips, and the explicit *Refused* state are
load-bearing for this audience. The product succeeds for them when they
can quote a piece of public copy back at the widget and watch the gate
either route to a bounded answer or visibly refuse.

**Newcomers learning what trace-conditioned mediation looks like.**
Practitioners, students, and grant evaluators who are evaluating the
project. The widget is a self-demonstration. The product succeeds for
them when they can read the public result page (`chat.html`), poke the
widget for ten minutes, and understand the experiment without first
reading v1.

A fourth audience — Sundog research maintainers — is served by v1 and
the eval harness, not by v2.

---

## 4. The Honest Claim, In Plain Language

The bounded research claim v1 §13 earned, restated for a non-specialist:

> We ran 5,670 trials of Ask Sundog and several baselines against a
> battery of adversarial prompts designed to make a chatbot upgrade
> roadmap language into result language, collapse promotional copy into
> evidence, or agree with a user under pressure. Across six model
> implementations spanning four training lineages, two hosted vendors,
> three retrieval depths, eight one-factor trace-field ablations, and
> three corpus-conflict mutations, the Sundog-gated architecture
> produced **zero unsafe accepts**. The same models *without* the gate
> produced unsafe accepts on severe-pressure prompts at rates between
> 80% and 100%.

What this means for a visitor to the widget: the architecture has been
stress-tested at a scale most public AI products are not. The specific
ways it has been tested, and the specific ways it has not, are linked
on the public result page.

What this **does not** mean (this list is the product's public
"What this does not show" callout):

- It does not show that Sundog has solved alignment, mesa-optimization,
  reward hacking, or any other capital-letter problem.
- It does not show that *every* hosted LLM, prompt, or corpus would
  produce the same result. The bounded claim names the conditions
  measured.
- It does not show that the widget is robust against an attacker with
  unlimited attempts. It is robust against the prompt slates and
  pressure axes that were tested.
- It does not show that the underlying scientific work the widget
  describes is correct. The widget preserves evidence boundaries; it
  does not validate the evidence.

The product's job is to display this boundary, not to soften it.

---

## 5. What "Polished" Means

The widget already works. Phase 6 shipped it. v2's job is to make it
*ready for an external visitor with no project context*. Concretely:

**Accessibility.** Keyboard-navigable from launcher to send button to
trace drawer. ARIA labels on every interactive element. Color contrast
passes WCAG AA on tier chips, refusal states, and disposition flags.
Screen-reader-friendly trace drawer (current implementation is visually
expressive but largely opaque to non-sighted users).

**Mobile.** The launcher, panel, evidence rail, and trace drawer all
need a tested mobile layout. Today the panel is desktop-first. Touch
targets ≥44px. The drawer cannot push the chat off-screen on a phone.

**Performance.** First-load widget JS + claim-map data under 100KB
gzipped. Time to first answer under 500ms on cached load, under 2s on
cold load over a 4G connection. No layout shift when the widget mounts.
The retrieval index should lazy-load only when the user opens the
launcher, not on page load.

**Empty and error states.** Today an unanswerable question gets a
generic refusal. The polished version distinguishes:
*out-of-corpus* ("I only answer about sundog.cc — here's what I know
about"), *refused-by-gate* ("This question would require a claim my
sources don't support — here's why"), and *retrieval-empty* ("No
matching passage found — try one of these"). Each state has its own
mascot pose, message, and follow-up suggestions.

**Onboarding.** First-time visitors see a one-line affordance — *"Ask
about sundogs, halos, or the research"* — not a blank input. Returning
visitors get straight to the input.

**The mascot (Halo Hound).** Phase 6 shipped the Tier-1 state ring on
the launcher button. Tier-2 panel micro-scenes and the full SVG sprite
set are still on the to-do list. The animation principle from v1 holds:
**do not animate cognition; animate evidence posture.** The mascot's
face changes with the trace, not with how clever the answer sounds.

**Public result page.** `chat.html` exists and carries the bounded
claim. v2 polishes it: a hero stat block ("0 unsafe accepts across
5,670 trials"), a result table, a "what this does not show" callout,
a downloadable probe report, a "how to inspect" section pointing at
the canonical eval scripts, and one-click demos that pre-fill
adversarial prompts the gate refused.

**No telemetry tax.** The widget records anonymous usage count and
nothing else. No conversation logging, no IP, no fingerprinting. The
privacy posture is visible from the trace drawer.

---

## 6. Releases

Three releases. Each one shippable on its own; each one earns the
right to ship the next.

### v0.1 — Internal Prototype (✅ shipped, 2026-05-12)

Already done as of Phase 6 exit. The widget is live on every root
public page, the launcher carries the Tier-1 mascot state ring, the
trace drawer renders, and the bounded claim is reflected on
`chat.html` and `index.html`. v1 §6 documents the artifact status.

What's left to do at this tier: nothing. v0.1 is the baseline the
non-profit deliverable polishes from.

### v1.0 — Public Launch (the non-profit deliverable)

The product ships as polished, accessible, and mobile-ready. The
public result page tells the story. A contributor with no project
context can demo the widget, understand the claim, and find the
inspection surface in under five minutes.

**Ship criteria** (all of these, in order):

1. **Accessibility audit passes.** Keyboard nav, ARIA, contrast.
   Owner: polisher with a11y experience.
2. **Mobile layout passes** on iOS Safari and Android Chrome at
   360px, 414px, and 768px widths. Owner: web dev.
3. **First-paint performance budget held.** Lighthouse score ≥90 on
   the page that hosts the widget. Owner: web dev.
4. **Halo Hound Tier-2 sprites and panel micro-scenes shipped** per
   `results/chat/phase8-public-writeup/halo-hound-mascot-spec.md`.
   Owner: designer / illustrator.
5. **Empty/error/refusal state copy reviewed by a skeptic.** Each
   state has its own message and a useful next step. Owner:
   red-team member.
6. **Public result page (`chat.html`) polished** with hero stat,
   result table, "what this does not show", downloadable probe
   report, and pre-filled adversarial demo links. Owner: web dev +
   secretary (for prose).
7. **Onboarding affordance shipped.** First-visit launcher copy
   distinguishes itself from return-visit launcher copy. Owner:
   designer.
8. **Cross-vendor demo button.** The public result page lets a
   visitor select which backend drafted a given answer (deterministic,
   gpt-4o-mini, claude-haiku-4-5, llama-3.3, qwen). Same gate, same
   trace, different draft surface. Visibly demonstrates the
   architecture's vendor independence. Owner: web dev + adapter
   maintainer.
9. **No regressions on the eval slate.** v1's full eval suite (Phases
   3–4 + the falsification slate + the corpus-conflict slate) passes
   green on the v1.0 build. Zero unsafe-accepts. Owner: any
   contributor who can run `npm run chat:eval:*`.
10. **Public-copy integrity check clean.** The elevator pitch,
    `sundog.html` halo atlas, and every other coupled surface in v1
    §16.1 routes correctly through the claim map. Owner: secretary
    + a maintainer with write access to `claim_map.json`.

Each criterion is a ticket. None of them require ML expertise. Most
require either web dev, design, careful prose, or careful adversarial
reading. The non-profit deliverable is the set of artifacts that result
when all ten land.

### v1.x — Incremental, Post-Launch

Once v1.0 ships, the product earns the right to incremental work on
top. Candidate items, in rough priority order:

- **Open-source license + contribution guide.** The repo is already
  public; a polished CONTRIBUTING.md and a clearer LICENSE notice
  remove a friction point for outside contributors.
- **Embedded mode.** A version of the widget any external site can
  drop in to query a Sundog-flavored corpus of its own. Bigger
  product surface; probably v1.2 or later.
- **Multilingual.** At minimum, mirror the deterministic FAQ router
  in two additional languages. The claim-map structure supports this;
  the answer templates and the gate lexicon do not yet.
- **Probe report builder.** A page that lets a visitor build their
  own adversarial probe slate, run it against the live widget, and
  download the trial outcomes.
- **Mascot animation depth.** Tier-3 chat-animation page
  (`chat-animation.html`) exercises the full 15-state mascot
  taxonomy. Non-blocker; ship only if it makes the evidence posture
  clearer.

The list is open-ended on purpose. v1.0 is the deliverable; v1.x is
where a sustained contributor community would live.

---

## 7. Contributor Onramps

The v1.0 ship criteria above are ten tickets. This section lists eight
more tickets that don't gate v1.0 but are visibly useful and grab-able
by a hobbyist contributor. Each ticket names the audience that fits it
and the skill bar required.

The tickets are written so a contributor can read one ticket alone and
know what to do. Cross-references into v1 are explicit.

### Ticket A — Hosted Model Adapter Expansion

**Audience:** anyone with API credits (their own or an `.edu`-unlocked
tier).
**Skill bar:** comfortable with Node.js and JSON. No ML background
required.

The contract harness lives at `chat/eval/score_phase3_drafts.mjs` and
already runs adapters for OpenAI, Anthropic, Groq (Llama), and Qwen.
Adding a new vendor is a new file under `chat/eval/lib/adapters/` that
matches the interface of the existing adapters. Once added, run the
existing wild / adversarial / differential slates against the new
adapter and report acceptance, addedValue, and gate-escape rate.

Why this matters for v1.0: more vendors strengthen the cross-vendor
demo button (criterion #8). v1 has already cleared two vendors; v2 can
extend the demo to as many as the gate handles cleanly.

### Ticket B — Corpus-Conflict Probe Slate Extension

**Audience:** skeptic-of-the-week, red-team rotator.
**Skill bar:** can read carefully and write adversarial prompts.

Phase 9 ran three corpus mutations (stale-doc, promo-first,
name-collision) across 1,008 trials with zero unsafe-accepts. The
slate is good but not exhaustive. New mutations worth adding:

- **Tier-bait.** Replace a chunk's body with prose that explicitly
  asserts a higher tier ("This is a peer-reviewed research finding
  that…") while leaving metadata unchanged.
- **Citation-spoof.** Add fabricated citations to chunk text and see
  whether the gate or the draft surfaces the mismatch.
- **Roadmap-as-result inversion.** Take a current Roadmap-tier route
  and rewrite the chunk to describe it in past tense
  ("…demonstrated…" instead of "…will demonstrate…").

Output goes to `chat/prompts/gold-corpus-conflict.jsonl`.

### Ticket C — Phase 5 One-Factor Ablations (newcomer track)

**Audience:** newcomers who want to learn what ML measurement looks
like.
**Skill bar:** can run a shell command and read a CSV.

This is the single best teaching exercise in the repo for someone new
to the field. The Phase 5 scaffold names six trace-field ablation
targets: `trace.boundary`, `trace.evidenceTier`, `trace.support`,
`trace.routeId`, `trace.disposition`, `trace.retrieved`. Each is a
separate ticket: null the field in the harness, rerun the slate,
report what breaks. Output goes to
`results/chat/interventions/`.

A newcomer doing this learns: how to define a measurement, how to
run a deterministic eval, how to read CSV outputs, how to write up a
one-factor result. No GPU, no scary math, real methodology. Pair each
newcomer with one ablation target and an educator.

### Ticket D — Wild Slate Expansion

**Audience:** anyone willing to spend an hour with the live widget.
**Skill bar:** patient observation.

The 30-prompt out-of-distribution slate at
`chat/prompts/gold-wild.jsonl` caught four new claim classes on its
first run. There are more. Play with the live widget, find a question
the router misses or the gate weakly handles, add it to the slate,
report what failed. This is the most newcomer-friendly ticket on the
list; the only barrier is care.

### Ticket E — Mascot Tier-2 Sprites and Panel Micro-Scenes

**Audience:** designers, illustrators, anyone who likes character art.
**Skill bar:** can produce SVG and write CSS.

Spec lives at
`results/chat/phase8-public-writeup/halo-hound-mascot-spec.md`. Tier-2
extends the mascot from the 5-state launcher button to ~10
trace-driven states in the panel. The 15-state full taxonomy is
exercised in `chat-animation.html` (Phase 8c, deferred).

The hard-line design principle inherited from v1: animate evidence
posture, not cognition. The mascot's face changes when the trace
changes, never when the answer "feels confident." A polisher working
on this ticket needs to internalize that principle before drawing
anything.

### Ticket F — Public Result Page (`chat.html`) Polish

**Audience:** web devs + a careful writer.
**Skill bar:** HTML/CSS, prose editing.

`chat.html` exists and shipped Phase 8c. v2 polishes it into the
non-profit deliverable's flagship page. The existing UI/UX foundation
(`docs/UI_UX_THEME_FOUNDATION.md`) constrains design choices. The
work is layout, hierarchy, and prose tightening — not invention.

This ticket is ship criterion #6 above, broken out here so a polisher
can grab it directly.

### Ticket G — Accessibility Audit and Remediation

**Audience:** anyone with a11y experience or willing to learn.
**Skill bar:** familiarity with screen readers or willingness to read
the WCAG quickstart.

This is ship criterion #1, broken out. The widget needs a full a11y
audit — keyboard nav, ARIA, contrast, screen-reader text for the
trace drawer. Tooling: axe-core, Lighthouse, manual NVDA/VoiceOver
testing. The result is a punch list and a series of small PRs.

### Ticket H — Mobile Layout Pass

**Audience:** front-end web dev.
**Skill bar:** comfortable with CSS, media queries, and touch event
patterns.

Ship criterion #2, broken out. The widget is desktop-first today. v2
needs it to behave on phones. The work is real-world device testing
(or BrowserStack), CSS adjustment, touch-target enlargement, and
some thought about how the trace drawer behaves on a small screen.

---

## 8. Out of Scope

What v2 is **not** trying to ship, on purpose:

- **A general assistant.** Ask Sundog answers about sundog.cc only.
- **A vendor lock-in.** The widget's whole point is that any of several
  hosted or open-weight models can sit behind the same gate. v2 will
  not ship a version that hard-codes one vendor.
- **A monetization story.** The hypothetical non-profit framing means
  the widget is free, anonymous, and the project does not derive
  revenue from it. If a real non-profit later spins up around it,
  sustainability comes from grants or donations, never from selling
  user conversations.
- **A capability story for the underlying LLMs.** The widget makes a
  model *more disciplined*. It does not make the model *smarter*. v2
  copy must not slip into "Sundog makes LLMs better" framing.
- **Public training data.** The widget logs anonymized usage count and
  nothing else. No conversation logging means no future "we trained a
  better model on what users asked us" pivot. This is a deliberate
  constraint, not a TODO.
- **Solving the research roadmap.** v1's open empirical fronts
  (mesa-geometry crossover, Phase 10 re-audit, the gravity ledger
  candidates) are not v2's problem. v2 ships the chat widget that
  bounds its claims correctly while v1 keeps doing the science.

---

## 9. Trust and Transparency

Three commitments the v1.0 product makes to its users:

**No data collection beyond anonymous usage count.** No conversation
logging. No IP. No fingerprinting. No third-party analytics scripts on
the widget surface. The privacy posture is visible from the trace
drawer ("This conversation is not stored.") and from a short
`/privacy` page reachable from the launcher footer.

**Open source.** The repo is already public. v1.0 ships with a
clearer LICENSE notice, a CONTRIBUTING.md, and a README pass on the
top-level repo so a visitor lands on a comprehensible front page.

**Inspectable claims.** The bounded research claim (§4 above), the
"what this does not show" callout, the downloadable probe report, and
the canonical eval scripts are all reachable from one click off the
widget. A skeptic does not need to take anything on faith.

---

## 10. Sustainability Sketch (Light)

Per the scoping question on this doc, v2 stays product-only and does
not include a full non-profit charter. A brief note on durability:

The widget runs in the browser. It uses an existing static-site
deploy (sundog.cc). The hosted-model adapter is *optional* — the
deterministic compositor family alone, with no LLM, already meets the
v1 §13 bounded claim and is the safer ship if no contributor has API
credits.

This means the hypothetical non-profit's recurring infrastructure
cost is dominated by domain renewal, CDN bandwidth, and (if the
hosted adapters are enabled in production) per-token API spend. The
deterministic compositor path costs nothing per query. A first-pass
sustainability story is: ship v1.0 in deterministic mode for the
public; the hosted adapters live in the eval harness and on the
public result page's "select your backend" demo, where per-query cost
is bounded by the size of the demo slate. This keeps the deliverable
free to operate at any reasonable traffic level.

If a real non-profit later wants to enable hosted-model drafting in
the live widget, that's a v1.x decision and a separate funding line.
v2 does not require it.

---

## 11. Coupled Surfaces — Product-Side Discipline

v1 §16 is the canonical record of integrity coordination between the
chat claim map and the rest of the public copy. v2 inherits the same
discipline but adds one product-side rule that didn't exist in v1:

**Any change to a coupled public surface ships through the v1 §16.5
update protocol *and* gets reflected on the public result page
(`chat.html`).** If the homepage retracts a claim, the chat result
page's "What this does not show" callout updates the same day. The
widget should never be the surface that surfaces an integrity gap to
a visitor; that's a sign the update protocol was skipped.

Currently coupled (per v1 §16.1, mirrored here for product
maintainers):

- Homepage elevator pitch (`index.html#elevator-pitch`)
- Halo-atlas vocabulary section (`sundog.html`)
- Brand positioning, message house, and claims-and-scope docs
- Public result page (`chat.html`)

Open integrity gaps (per v1 §16.2 and §16.4): the v1.1 elevator pitch
is audit-hedged on the mesa-geometry crossover claim pending the
Phase 10 re-audit. v2 inherits this state. v1.0 of the product does
not promote any route that the pitch hedges. If the re-audit clears
with the verdict surviving, v1.x can revisit; if not, the pitch's
third paragraph retracts and v2 follows.

---

## 12. First Two Weeks for a New Contributor Cohort

If a contributor group stands up tomorrow with the role mix described
in the chat that produced this doc (hobbyist polishers and web devs,
a few newcomers to ML, some skeptic-thinkers, maybe one or two people
with API credits), the first two weeks have a natural shape.

**Week 1.** Assign tickets. A polisher takes G (a11y audit) or H
(mobile pass). A web dev takes F (`chat.html` polish). A skeptic takes
B (corpus-conflict probe extension). Each newcomer pairs with an
educator and takes one variant of C (Phase 5 ablation). Anyone with
API credits takes A (new vendor adapter).

**Week 2.** First synchronous review. Each ticket reports state on a
single dashboard — a pinned message in whatever the group chose for
chat coordination, or a `STATUS.md` in the repo. The a11y audit
delivers a punch list. The mobile pass has one tested breakpoint
ready. The Phase 5 ablations have first results. The skeptic's new
prompts run through the harness and either show new gate behavior or
confirm the existing gate is solid.

If by end of week 2 the group has one new vendor adapter result, one
Phase 5 ablation row, and a first pass on either the a11y audit or
the mobile layout, then v2's ship criteria are visibly in flight and
v1.0 is a real target rather than an aspiration. The reason this
works is structural: the eval infrastructure already exists, the
research roadmap is already met, and the polishing surface is well
specified. The group's job is to fill cells in a plan that has
already been laid out.

---

## 13. Cross-Links

- [`SUNDOG_V_CHAT.md`](SUNDOG_V_CHAT.md) — research roadmap, Phases
  0–12, eval discipline, claim ratchet, coupled-surfaces integrity.
- [`UI_UX_THEME_FOUNDATION.md`](UI_UX_THEME_FOUNDATION.md) — the
  design constraints any polishing ticket inherits.
- [`WEBSITE_DEVELOPMENT.md`](WEBSITE_DEVELOPMENT.md) — the deploy
  protocol and chat-check gate that any v2 change must clear.
- [`presentation/claims-and-scope.md`](presentation/claims-and-scope.md)
  — canonical bounded-claim language; v2 product copy must not
  exceed it.
- [`STANDALONE_APP_ROADMAP.md`](STANDALONE_APP_ROADMAP.md) — sibling
  roadmap for the standalone Sundog application surface.
- `results/chat/phase8-public-writeup/halo-hound-mascot-spec.md` —
  mascot taxonomy and trace-derivation rules.

---

## 14. Document Status

This is v2.0 of the chat product charter, filed 2026-05-21. It is a
new document, not a replacement for v1. Both docs are maintained
together. If a future change to the research roadmap (v1) invalidates
a product commitment in v2 — for example, if a new bounded claim
materially changes what the widget can say — the v2 ship criteria
update in the same change.

The document is intentionally short. It is meant to be readable in one
sitting by a contributor with no prior context, and it deliberately
defers depth to v1 wherever depth would slow a polisher down.
