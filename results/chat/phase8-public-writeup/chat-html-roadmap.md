# `chat.html` page + animation roadmap

Source: the editorial-ratified `draft.md` (Phase 8 prose) + the existing
site page conventions surveyed across `about.html`, `balance.html`,
`threebody.html`, and `mines.html`. Goal of this doc: roadmap how
`chat.html` lands and whether/when a `chat-animation.html` sidecar joins it.

The page itself is the lower-risk artifact. The animation is the
higher-risk creative decision and needs more thought before commitment.

---

## 1. What site convention looks like

Surveyed across the dedicated pages already on sundog.cc:

**Common skeleton (every page):**
- `<head>`: standard favicons, `theme-color="#1A3A52"`, page-specific
  meta description and `<title>`, optional page-local `:root` token
  overrides.
- `<header class="site-header">` → `<nav class="site-nav">` with
  `.site-brand` (link home) and `.site-links` (About / Applications /
  Three-Body / Balance / Mines / Mesa / Docs / GitHub).
- `<main>` with the page's content.
- `<footer>` with the Stellar Aqua LLC copyright line and
  page-specific link row.
- `<script type="module" src="/js/sundog-chat-widget.mjs">` before
  `</body>` on every public page.

**Two distinct page patterns within that skeleton:**

**Workbench pattern** (`balance.html`, `threebody.html`, `mines.html`):
- Full-bleed `.workbench` grid: `<canvas>` left/center + side
  `.workbench-panel` for controls/copy
- Page-specific JS controls the canvas (`/js/balance-browser.mjs`,
  `/js/threebody-browser.mjs`, etc.)
- The canvas IS the page; prose is secondary, embedded in side panel
- Dense, near-100vh layout, dark stage background

**Document pattern** (`about.html`, `origin.html`):
- Hero section with eyebrow + headline + lede + signal-stack aside
- Content sections of prose
- No central canvas; pages are read top-to-bottom
- Lighter background, generous typography, editorial feel

`chat.html` has to pick one or hybridize. My recommendation in §2.

---

## 2. `chat.html` structure proposal

**Recommended pattern: document, not workbench** (for v1).

Reasons:

- The experiment's primary contribution is the result table + the
  non-claims box + the trace inspection surface — content, not a
  live simulation. The chat widget itself is already on every page;
  having it as the page's central canvas would feel redundant.
- The document pattern matches what the prose draft is shaped for
  (six sections, prose-heavy, with one table and one verbatim
  example).
- Lighter implementation cost. No new canvas/controller code beyond
  what the widget already provides.
- The animation (if it lands) doesn't have to be the page's central
  artifact — it can be a sidecar visited from a teaser.

**Section-by-section mapping** from `draft.md` to `chat.html`:

| draft.md section | chat.html block | Pattern reference |
|---|---|---|
| §1 What Ask Sundog is | hero (eyebrow + headline + lede + sidebar with "central question") | about.html's hero |
| §2 The strongest result | `.lead-framing` callout + `.result-table` block + verbatim severe prompt + trace-conditioned answer | new component; result-table is a small CSS grid |
| §3 What this does not show | `.non-claims` callout box, visually distinct, on a tinted background, full-bleed-ish | new component; needs to read as "load-bearing caveat" not "boilerplate disclaimer" |
| §4 What it does show | content section with three short subsections (tier, boundary array, channel separation) | about.html's content-section pattern |
| §5 How to inspect | content section with bullet list pointing at prompt slates, harness, results dir, GitHub, CITATION.cff | about.html's content-section pattern |
| §6 What's next | smaller closing section with two named follow-ups + falsification invitation | about.html's closing-note pattern |

**Two unique-to-chat.html components** that don't exist on other pages:

1. **`.result-table`** — the 3×4 severity × family grid. CSS-only. Bottom
   row gets visual emphasis (border, background highlight) since
   that's where the result lives. Cells use the same chip styling
   as the widget's evidence rail for visual continuity.

2. **`.non-claims`** — the five non-claims rendered as a callout block.
   Border-left accent like the about.html `.signal-stack`, but
   tinted to read as caveat (probably a slate-warm tone rather than
   the gold accent). Each non-claim is a numbered item with
   one-line title + one-line gloss.

**Widget integration:**

The existing `Ask Sundog` widget continues to live on chat.html as
on every page. On chat.html specifically, the §5 "How to inspect"
section explicitly invites visitors to open the widget and inspect
the trace — making the widget the page's interactive surface
without animation.

**File budget:** ~600 lines of HTML (mostly content from draft.md) +
~80 lines of page-local CSS. No new JS beyond the widget. Estimate:
half a day to land.

---

## 3. The animation question

The user's reference is github.com/pablodelucca/pixel-agents. I'm
going to be upfront: I can't reliably describe that project from
memory, and I haven't loaded it. Before I roadmap an animation
seriously I need you to either describe what about it you're drawn
to, or paste a screenshot/description here. The roadmap below is
**conditional on that aesthetic resolution**.

### 3.1 What an animation could *show*

Three candidate roles, in increasing experimental load:

**Role A — Decorative trace-fill.** Visitor sees a typewritten prompt
appear in a stylized chat window. The trace populates in real time:
route name, tier chip, boundary entries, support documents, answer
prose. No comparison, no failure mode. Just the inspection surface
made visual.

Load-bearing? Mostly no. The existing widget already shows the
trace; an animation of the same thing is decorative unless the
visual treatment makes the trace *legible* in a way the widget
doesn't (e.g., spatial separation of the trace fields rather than
collapsed in a drawer).

**Role B — Family comparison race.** Same prompt fed to four
parallel chat windows (B0, B1, B2, S1). Visitor watches each
family produce a draft, then watches the gate accept/reject each
draft. End state: the table from §2 made temporal.

Load-bearing? Yes. This dramatizes the result. The 100-point gap
at severe pressure is abstract on the page; here it's a visible
event — three families' drafts get rejected, one gets through.
The cost is significant engineering (four chat-window state
machines, gate animation, careful pacing).

**Role C — Severe-pressure narrative.** A single chat window. The
user types a severe-pressure prompt (the verbatim one from §2).
The page shows two parallel response panels: B2 (prompt-engineered
baseline) and S1 (trace-conditioned). B2 starts drafting a
boundary-violating answer; the gate redacts it visibly; the
fallback answer appears. S1 drafts a clean answer that passes.

Load-bearing? Most of any option. Makes the entire experiment
legible in 20 seconds of viewing. Hardest to build correctly —
the boundary-violation has to look like a *natural* response (so
visitors believe a real model would write it), and the gate's
redaction has to feel like enforcement, not censorship.

### 3.2 Where the animation lives

Two structural options:

**Inline on chat.html.** Animation lives above the fold, replaces
the static §2 table or sits next to it. Cost: chat.html gains a
canvas controller; the page becomes a workbench-pattern hybrid.
Reward: the result becomes visceral on first visit.

**Sidecar `chat-animation.html`.** Animation gets its own page,
linked from chat.html with a static preview image. Cost: another
file to maintain; visitors may not click through. Reward:
chat.html stays fast/readable; the animation can be a longer,
richer experience without the cost burden on every visit.

**My recommendation: sidecar, with a static preview image inline
on chat.html.** Three reasons:

1. The chat.html result is content-first. A heavy animation above
   the fold competes with the §2 table and §3 non-claims for the
   visitor's attention. The non-claims box is the experiment's
   most important honesty surface; nothing should outshine it
   visually.
2. The animation is creative-engineering work. Doing it well takes
   time that doesn't have to delay the page's launch.
3. A sidecar lets the animation be a separate creative artifact —
   linkable, embeddable, sharable on its own — without bundling
   its complexity into the main result page.

### 3.3 Visual language: pixel-agents fit

Honest pushback before committing: the existing sundog.cc visual
language is clean editorial typography, a paper/ink palette, and
technical-illustration aesthetics. Pixel art is charming but
aesthetically different — adopting it for a chat-window animation
would break the page's visual coherence with the rest of the
site.

That break can be **deliberate** (one playful page within an
otherwise serious site, signaling the experiment is approachable),
or it can be **accidental** (the animation looks like a different
project bolted on). Worth being intentional about the call.

A non-pixel alternative that might fit better: **a stylized
terminal-style chat window** with the existing Sundog typography
(sans for chrome, serif for prose, the gold accent for callouts),
animated by simple typewriter-fade transitions. Less charming,
more on-brand.

Or: **keep the pixel aesthetic for the animation specifically as
a "behind the scenes" treatment** — the chat-animation.html page
is explicitly the "look inside the machinery" view, and pixel art
signals that it's an inspection/diagnostic surface, distinct from
the main editorial page.

Three resolution-paths:

1. **Stay on-brand (no pixel art).** Animation uses Sundog's
   existing visual tokens. Safer; less novel.
2. **Adopt pixel-art deliberately for the sidecar only.** Sidecar
   becomes the "look inside" surface; chat.html stays editorial.
   Justifies the aesthetic break.
3. **Adopt pixel-art across chat.html and sidecar.** Most novel;
   biggest visual coherence break. Probably only justifiable if
   the experiment's overall positioning is shifting toward
   playful/approachable.

I don't have a recommendation here without knowing what about
pixel-agents you're drawn to. If it's the charm and approachability,
option 2 or 3 may be right. If it's specifically the way they show
*agent behavior* visually (which I'm guessing at), maybe what you
actually want is option 1 with the agent-behavior insight ported
into the on-brand visual language.

---

## 4. Suggested phasing

**Phase 8a — `chat.html` v1, no animation.** Land the document-pattern
page with the draft.md content, the result table, the non-claims
callout, the inspection-section pointing at the existing widget.
Index.html gets the teaser link. This ships the result publicly
without dependency on the animation.

Effort: ~half a day. Unblocks Phase 8 exit criterion ("public copy
does not exceed the measured result").

**Phase 8b — Animation prototype.** Build a small (~20-second)
animation prototype in a chat-animation-prototype.html scratch
file. Decide which role (A/B/C) and which aesthetic resolution
(1/2/3) before committing to a final implementation. The prototype
exists to make the creative decision concrete, not to ship.

Effort: 1-2 days for a credible prototype. Output: a clear answer
to "does this make the result more legible, and does it fit the
site visually?"

**Phase 8c — `chat-animation.html` sidecar** (only if Phase 8b
prototype lands well). Polished version of the prototype as a
linked-from-chat.html artifact. Static preview image embedded in
chat.html.

Effort: 2-3 days depending on role chosen (Role C is the
heaviest; Role A is lightest).

**Phase 8d — Index.html teaser.** Short block on the homepage with
the headline number ("0 escapes, 100-point gap") and link to
chat.html. One sentence; ships when chat.html lands.

Total: ~1 week to full Phase 8 if all four steps land. ~half a
day to just unblock Phase 8 exit criterion via 8a.

---

## 5. Open questions before committing

Three things I need from you before drafting actual code:

1. **The pixel-agents inspiration.** What about the project are
   you drawn to? Charm? Specific visual treatment? Showing agent
   behavior in a particular way? A description or screenshot would
   let me roadmap the animation seriously rather than speculate.

2. **Aesthetic resolution.** Stay-on-brand vs. deliberate-pixel-art
   for the sidecar vs. pixel-art-across-the-board. This is a
   product-positioning call as much as a design call.

3. **Animation role.** A (decorative trace-fill), B (family
   comparison race), or C (severe-pressure narrative). Role C is
   the most pedagogically valuable; Role A is the lightest;
   Role B is the most direct dramatization of the actual result.

If you want to push forward without resolving 1–3, I'd recommend
**Phase 8a now (ship the page), skip 8b/8c for now (defer the
animation), and 8d when the page lands.** The §13 ratchet lands
publicly without animation. The animation becomes a follow-on
creative decision rather than a critical-path Phase 8 dependency.

---

## 6. What I can do in this session

Given the open questions, the highest-leverage moves I can make
without further input:

- **Write `chat.html` v1.** Document pattern, draft.md content,
  result table, non-claims callout. Maps cleanly to existing
  site convention. Ratifies Phase 8 exit criterion. Estimate:
  ~half a day of work; ships immediately.
- **Write the index.html teaser block.** One line + one paragraph
  + link. Estimate: ~15 minutes.
- **Draft a Phase 8a/b/c/d scaffold** if you want explicit
  artifact-status blocks in `docs/SUNDOG_V_CHAT.md` for each
  step.

What I'd hold off on:

- Drafting `chat-animation.html` until you describe the
  pixel-agents inspiration and pick an aesthetic resolution.
- Building the animation prototype until the role (A/B/C) is
  chosen.

Recommended next action: answer questions 1–3 above; if any
remain open, ship 8a (chat.html v1) and 8d (index.html teaser)
this session and defer the animation work.
