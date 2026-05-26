# Highlights Slide Rail Roadmap

Date: 2026-05-12
Status: scoping doc for the post-hero application motion rail's next pass
Owner cross-references:
- [`UI_UX_THEME_FOUNDATION.md`](UI_UX_THEME_FOUNDATION.md) — Step 4d defines
  the rail as a structural feature of the index page; the paper-theme
  extension (Phases 1-2 of the §"Paper-Inspired Theme Extension") owns the
  visual primitives this roadmap composes.
- [`debunked.md`](../debunked.md) — origin brief for the verdict-stamp
  vocabulary and the case for an interrupt card on the rail.
- [`PROMO_HIGHLIGHTS.md`](../promo/PROMO_HIGHLIGHTS.md) — message bank for card
  copy; per-product headlines and boundary language live there.
- [`APPLICATIONS.md`](../APPLICATIONS.md) — current Evidence Tiers; this
  roadmap maps those tiers onto the new stamp vocabulary.
- [`PUSHABLE_OCCLUDER_ROADMAP.md`](../PUSHABLE_OCCLUDER_ROADMAP.md) — the
  falsification application that produces the rail's first BOUNDARY FOUND
  card. The rail card cannot be promoted from UNTESTED to BOUNDARY FOUND
  without the experiment evidence that roadmap delivers.

## Purpose

`index.html` already ships a working application rail (`.motion-rail-*`,
lines 697-883). It currently uses ad-hoc per-card status strings
("Operating envelope", "Research result", "Instrumented prototype",
"Product expression") drawn from the Evidence Tiers in
`APPLICATIONS.md`. Those tiers are honest, but they read as taxonomy, not
as verdicts. They cannot communicate a deliberate negative result, and
they invite the rail to feel like a tour of wins.

This roadmap rebuilds the rail around a verdict-stamp vocabulary that
includes serious failures. The rhythm should read like a MythBusters
end-card sequence — confirmed, plausible, bounded, **then a deliberate
interruption: BOUNDARY FOUND** — and only then continue. The goal is a
rail that earns humility without weakening the program.

This document does not replace the rail. It scopes its next pass.

## Scope

In scope:

- The verdict-stamp vocabulary (six values) and the rule for when each
  applies.
- **Layout rebuild from horizontal-scroll-of-equals to center-focus
  carousel**: one card centered, a glimpse of the next and previous on
  either side, ~1.5 cards in view at any time.
- **Auto-cycle on stamp landing**: when the centered card finishes its
  clip and its stamp lands, the rail advances to the next card after a
  short hold. The stamp itself is the advance cue.
- Visual treatment: rubber-stamp paper ink-bleed aesthetic that composes
  with the paper-theme primitives already landed in
  `public/css/sundog-theme.css`.
- Stamp timing: the stamp arrives **after** the clip's last beat, not at
  page load. The transition itself is the experimental verdict and is
  also the cue to advance.
- The card data contract (`data-*` attributes already present on
  `.motion-card` plus the new fields needed to drive stamp timing and
  meaning).
- Per-card content plan for the seven current cards plus the new
  Pushable Occluder interrupt.
- Accessibility, reduced-motion, and screen-reader behaviour for the
  stamp animation and the auto-cycle.
- Build phases that incrementally migrate the live rail without breaking
  it.

Out of scope:

- The canvas content of the experiments. Workbench rendering belongs to
  each workbench's own roadmap.
- The Pushable Occluder experiment itself. That lives in
  [`PUSHABLE_OCCLUDER_ROADMAP.md`](../PUSHABLE_OCCLUDER_ROADMAP.md).
- Renaming the Evidence Tiers in `APPLICATIONS.md`. The stamps are a
  public, rail-facing verdict layer that *cites* the tiers; the tiers
  remain the canonical scientific bookkeeping.

## Stamp Vocabulary

Six verdicts, ordered as they should appear in the rail rhythm:

| Stamp | When it applies | Honest claim ceiling |
| --- | --- | --- |
| `CONFIRMED` | Controlled experiment, matched baselines, reproducible artefacts. The result has survived peer-style scrutiny inside the repo. | "We measured this. The metric and the failure boundary are both stated." |
| `OPERATING ENVELOPE` | Bounded wins inside a *mapped* failure region. The win and the fail are reported together. | "It works in this pocket. Here is where it breaks." |
| `PLAUSIBLE` | Prototype with strong shape but incomplete measurement. Telemetry exists; rigorous baselines do not. | "We can run it. We have not yet earned a number." |
| `BOUNDARY FOUND` | A serious *failure* that clarifies the theorem. The method's honest upper limit is reached without invoking a different controller class. | "The theorem still stands; the method does not reach here." |
| `STALLED` | A failure where the current controller class is insufficient. A new method is needed before any verdict can be re-attempted. | "Not falsified, but not solvable by what we have." |
| `UNTESTED` | Concept surface that should not borrow evidence from sibling cards. Intentionally on the rail to set expectations, not to claim. | "We named it. We have not run it." |

Stamps are public-facing verdicts. They map onto the canonical
`APPLICATIONS.md` Evidence Tiers as follows; this mapping is the
authoritative reconciliation:

| `APPLICATIONS.md` tier | Default stamp | Promotion rule |
| --- | --- | --- |
| Research result | `CONFIRMED` | Stays `CONFIRMED` only while the artefact reproduces. |
| Operating-envelope study | `OPERATING ENVELOPE` | Demoted to `PLAUSIBLE` if the failure region has not been mapped in the current sprint. |
| Instrumented prototype | `PLAUSIBLE` | Promotes to `OPERATING ENVELOPE` only when matched baselines and a failure region exist. |
| Product expression | `PLAUSIBLE` or `UNTESTED` | `UNTESTED` is the default for product expressions that have no run harness yet. |
| Conceptual lineage | `UNTESTED` | Never auto-promotes. |
| n/a — falsification slate | `BOUNDARY FOUND` or `STALLED` | Set by the falsification roadmap that owns the experiment (e.g. `PUSHABLE_OCCLUDER_ROADMAP.md`). |

The mapping is intentionally one-way: changes to a card's tier in
`APPLICATIONS.md` can demote its stamp, but stamp promotions require
written evidence in the owning roadmap.

### Rules for new cards

1. **No card ships at `CONFIRMED` without a linked artefact**
   (`data-evidence-href` must resolve to an in-repo file the user can
   click through to).
2. **No card ships at `BOUNDARY FOUND` without an owning falsification
   roadmap.** The roadmap must name the hypothesis, the experiment, and
   what would have to happen for the stamp to come off.
3. **`UNTESTED` is fine. It is honest.** Cards may live at `UNTESTED`
   indefinitely. They must not borrow visuals or copy that implies
   measured results.

## Layout: Center-Focus Carousel

The current rail (`index.html` line 236, `.motion-rail-track`) is a
horizontal scroll of equal-weight cards at `minmax(280px, 36%)`. On
desktop that puts roughly three cards in view simultaneously, with no
visual distinction between the "current" card and its neighbours. That
layout reads as a tour. We do not want a tour. We want a verdict
sequence.

The new layout puts **one card at the centre, slightly larger, fully
legible, with its clip playing**; partial slivers of the previous and
next cards peek in from either edge as quiet promises of what comes
next. Target proportions at desktop widths:

- Centre card: ~58-64% of the rail track width.
- Each peek (left and right): ~14-18% of the rail track width.
- Gutters between cards: 1rem.
- Cards in view at once: ~1.5 (centre + two ~0.25 slivers).

The peeked neighbours are **dimmed and de-saturated** (e.g.
`filter: brightness(0.55) saturate(0.6); opacity: 0.7;`) so the eye
does not split attention. They are interactive while peeked: clicking a
peeked card **navigates directly to that card's `data-href`** (decided
2026-05-12). The rationale is that a peek is a real preview; clicking
it implies the user wants the destination, not a slide-over.

Always-overlaid skip arrows sit on the rail itself (not in the
header), absolutely positioned over the left and right edges of the
track. They support click ("skip") and on touch surfaces are paired
with swipe gestures. Clicking the next/previous arrow **immediately
lands the current card's stamp before scrolling away**. That rule is
load-bearing: verdicts are never skipped, even when the user is
power-browsing past a card. A skipped clip drops its remaining beat,
but the stamp still arrives.

Mobile (≤ 520px viewport):

- Centre card: ~80-85% of the track.
- One peek visible (the next card), ~12-15%.
- The previous card is off-screen entirely on narrow widths; an
  accessible "previous" affordance remains via the existing
  `data-rail-prev` button. This is a deliberate asymmetry — at phone
  width the forward sequence is the read; reverse is the exception.

### Implementation sketch

The current grid-based track stays but the column sizing changes:

```css
.motion-rail-track {
  display: grid;
  grid-auto-flow: column;
  grid-auto-columns: minmax(56%, 60%);
  gap: 1rem;
  padding-inline: 18%;          /* reserves the peek strips on both sides */
  scroll-snap-type: inline mandatory;
  scroll-snap-stop: always;
  scroll-padding-inline: 18%;
  overflow-x: hidden;            /* auto-cycle drives the scroll; hide native scrollbar */
}

@media (max-width: 520px) {
  .motion-rail-track {
    grid-auto-columns: minmax(82%, 84%);
    padding-inline: 8% 16%;       /* asymmetric: bigger peek on the next-card side */
    scroll-padding-inline: 8% 16%;
  }
}

.motion-card {
  scroll-snap-align: center;
  transition:
    filter 320ms ease,
    opacity 320ms ease,
    transform 320ms ease;
}

.motion-card:not([data-rail-active]) {
  filter: brightness(0.55) saturate(0.6);
  opacity: 0.7;
  transform: scale(0.94);
  /* peek stays interactive: click navigates to its data-href */
}

.motion-card[data-rail-active] {
  filter: none;
  opacity: 1;
  transform: scale(1);
  pointer-events: auto;
}
```

The rail-behaviour script tracks the active index. Advancing
`scrollLeft` (or programmatic `card.scrollIntoView({behavior: "smooth",
inline: "center"})`) is the canonical way to centre a card; the script
toggles `data-rail-active` to match.

## Auto-Cycle Sequence

The rail cycles automatically. The trigger is **stamp landing**, not a
timer. Each card runs its own life-cycle and hands off when its verdict
is on paper:

1. Card centres. `data-rail-active` is set. The previous card releases
   its active flag and begins fading back to peek state.
2. Poster fades to media if `data-media` is set; otherwise stays.
3. Media plays once. For static-poster cards, an implied beat of 1.6s
   passes.
4. `data-stamp-armed` is set; the stamp transitions in over ~320ms.
5. **Hold pause.** The card holds in armed-and-active state for a
   per-card dwell (default 1800ms). The dwell is long enough for a
   reader to register the verdict before the rail moves on.
6. Advance: the next card centres. `data-rail-active` is moved.

The hold pause is per-card and is configured by a new
`data-dwell-ms` attribute on the card. Defaults:

| Card | Default dwell |
| --- | --- |
| Standard cards | 1800ms |
| `BOUNDARY FOUND` interrupt | 3600ms |
| `STALLED` (when one exists) | 3000ms |
| `CONFIRMED` | 2200ms (slightly longer so the win is felt) |

The failure interrupt's longer dwell is the rhythm move: the audience
should sit with the failure, not roll past it.

### Pause and resume rules

- **Hover** anywhere on the centre card pauses the dwell timer.
  Resuming hover-out resumes the dwell from where it left off.
- **Keyboard focus** on the centre card pauses dwell. Tab-away resumes.
- **Manual next/prev** (the existing `data-rail-prev` /
  `data-rail-next` buttons, or arrow keys, or swipe) pauses auto-cycle
  for the rest of the page session. The rail becomes user-driven once
  the user has expressed an intent to drive. There is no resume; the
  user is in control until reload.
- **`prefers-reduced-motion: reduce`** disables auto-cycle entirely.
  All stamps render at page load. The rail becomes a static set of
  cards the user navigates manually.
- **Drag/swipe** on touch devices pauses auto-cycle (same as a manual
  next).
- **Tab to a non-rail element** on the page does not pause; the rail
  keeps cycling in the background.

### Once around, then settle

After the last card in the sequence (Money Bags, position 8) finishes
its dwell, the rail does **not** loop back to position 1 automatically.
It settles on the last card. This is deliberate: looping forever turns
the rail back into a tour. Ending on a card the user can sit with
respects that the sequence has a thesis.

The replay action is a **persistent third arrow button** in
`.motion-rail-actions` (decided 2026-05-12). It is visible throughout
the sequence, not only at the end-state. The persistent button reads as
a first-class action: at any moment a user may say "I want to see that
again from the top" and reach for it. Visual hierarchy: smaller than
prev/next, paired with the next arrow on the right edge of the track,
glyph is a refresh/loop character (↻). Clicking it clears all timers,
removes `data-stamp-armed` from every card, scrolls to card 1, and
restarts the auto-cycle.

## Visual Treatment

The stamp is a rubber-stamp paper ink-bleed, not a chip badge and not an
engraved plaque. It composes with the existing paper primitives:

- The card surface is `.sd-paper-card` or `.motion-card` (the existing
  rail card; see "Migration" below for how these reconcile). Ruled-line
  background and red margin rule come from
  `public/css/sundog-theme.css`.
- The stamp itself is a new component class `.sd-verdict-stamp`,
  positioned absolutely over the card's lower-right corner, rotated
  off-axis (between -4° and -12°), with imperfect edges and a soft
  ink-saturation gradient.
- Colour per verdict (proposed tokens, to be added to
  `public/css/sundog-theme.css` alongside `--sd-sticky-*`):
  - `CONFIRMED` → `--sd-stamp-confirmed: var(--sd-signal)` (gold-amber).
  - `OPERATING ENVELOPE` → `--sd-stamp-envelope: var(--sd-copper)`.
  - `PLAUSIBLE` → `--sd-stamp-plausible: var(--sd-violet)`.
  - `BOUNDARY FOUND` → `--sd-stamp-boundary: #b04535` (oxblood ink — the
    interruption colour; should *not* be the same red used for the paper
    margin rule).
  - `STALLED` → `--sd-stamp-stalled: var(--sd-ink-strong)` (matte black,
    no shine — deliberately heavy).
  - `UNTESTED` → `--sd-stamp-untested: rgba(20, 43, 58, 0.35)` (faded;
    reads as "not yet a verdict").

The stamp shape is a thick, slightly broken rectangle with the verdict
text typeset in `--sd-font-display` at all caps, letter-spaced wide.
Inside the rectangle, an additional thin line follows the rectangle's
inner edge to suggest a real stamp die. Both lines use the same ink
colour with `mix-blend-mode: multiply` so the underlying ruled-paper
texture shows through.

Implementation sketch (to be added to the paper-theme extension section
of `UI_UX_THEME_FOUNDATION.md` when the component lands):

```css
.sd-verdict-stamp {
  position: absolute;
  bottom: 1.25rem;
  right: 1.25rem;
  padding: 0.35rem 0.9rem;
  font-family: var(--sd-font-display);
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--sd-stamp-ink, var(--sd-ink-strong));
  border: 2px solid currentColor;
  outline: 1px solid currentColor;
  outline-offset: 2px;
  transform: rotate(-7deg) scale(0.9);
  opacity: 0;
  mix-blend-mode: multiply;
  transition:
    opacity 220ms ease-out,
    transform 320ms cubic-bezier(0.34, 1.56, 0.64, 1);
}

.motion-card[data-stamp-armed] .sd-verdict-stamp {
  opacity: 0.88;
  transform: rotate(-7deg) scale(1);
}
```

Edge imperfection is achieved with a CSS mask using a small SVG noise
filter; this avoids requiring per-stamp PNG assets. The same noise mask
covers both the rectangle and the text so the ink-bleed reads as one
contact event.

## Stamp Timing Within a Card

Within a single card's life cycle (the choreography above lays out how
the rail advances *between* cards; this section is what happens *during*
one centred card):

1. Card becomes the active centre card (`data-rail-active` set).
2. Poster fades to media if `data-media` is present; otherwise stays.
3. Media plays once. For static posters, an implied "beat" of 1.6s
   passes. Cards may override the beat length with `data-clip-ms`.
4. `data-stamp-armed` attribute is set on the `.motion-card` element.
5. The stamp transitions in over ~320ms, accompanied by a 60ms inner-edge
   tick (the "die contact") and a 180ms ink-bloom fade on the rectangle
   border.
6. The card enters its **dwell** state (see "Pause and resume rules").
7. Dwell elapses (default 1800ms, per-card override via `data-dwell-ms`).
8. The rail advances. The card retains `data-stamp-armed` so that when
   the user manually returns to it, the stamp is **already arrived** —
   no re-animation. The rail does not feel slot-machine-like on repeat
   views.

Reduced-motion users see the stamp at rest at page load; steps 4 and 5
fire instantly with no transition.

### The Pushable Occluder interrupt

The boundary card is choreographed differently from the rest of the
rail. It is the only card whose visual content depicts the failure
itself, and the only card whose dwell is doubled:

1. Beam visible, mirror searching, block in path.
2. Detector ring never peaks. The signal dithers around a local
   maximum.
3. Last beat: the controller gives up. The mirror returns to neutral.
4. Stamp hits: `BOUNDARY FOUND` in oxblood, rotated -10° (more than the
   default), with a 40ms shake on the card itself synchronous with the
   stamp contact. The shake amplitude is bounded (~2px) and is disabled
   under `prefers-reduced-motion`.
5. **Long dwell** (`data-dwell-ms="3600"`): the card holds for roughly
   twice the standard dwell before the rail advances. The reader must
   have time to register that this card is different in kind from its
   neighbours.

The interrupt is the only place we spend animation budget on the stamp
contact event itself, and the only place we double the dwell. Every
other card lets the stamp arrive and the rail move on at the standard
pace.

## Card Data Contract

The current rail already exposes a clean data contract on `.motion-card`
elements (see `index.html` lines 714-880). This roadmap extends it with
three fields and tightens one:

| Field | Status | Purpose |
| --- | --- | --- |
| `data-title` | existing | Card title. Required. |
| `data-description` | existing | Short tagline. Required. |
| `data-href` | existing | Primary CTA destination. Required. |
| `data-evidence-href` | existing | In-repo artefact. Required if `data-stamp` is `CONFIRMED` or `OPERATING ENVELOPE` or `BOUNDARY FOUND`. |
| `data-poster` | existing | Still-image fallback. Required if `data-media` is set. |
| `data-media` | existing | Loop media. Optional. |
| `data-media-format` | existing | Format hint. Optional. |
| `data-status` | existing → **deprecated** | Free-text status string. Will be removed once `data-stamp` ships. |
| `data-stamp` | **new** | One of the six verdicts, all caps. Required. Drives the stamp colour, ink, and copy. |
| `data-stamp-meaning` | **new** | One-line verdict gloss (the "Theorem meaning:" line from `debunked.md`). Surfaces as the stamp's `aria-description` for screen readers. |
| `data-stamp-source` | **new** | Repo path to the roadmap that owns the verdict. For `CONFIRMED` and `OPERATING ENVELOPE` cards this is the experiment's main doc. For `BOUNDARY FOUND` and `STALLED` cards this is the falsification roadmap. |
| `data-clip-ms` | **new, optional** | Length of the card's clip in milliseconds. For cards with `data-media`, the script reads media duration directly and this attribute is ignored. For static-poster cards, this overrides the default 1600ms "implied beat" before the stamp arms. |
| `data-dwell-ms` | **new, optional** | Length of the post-stamp hold before the rail auto-advances. Defaults per verdict (1800 standard / 2200 CONFIRMED / 3000 STALLED / 3600 BOUNDARY FOUND); a card may override its own. |

`data-status` is retained during the migration so the existing JS in
`public/js/motion-rail.mjs` (or wherever the rail behaviour lives;
verified in Phase 1 below) keeps working. After Phase 3 it is removed.

## Per-Card Content Plan

Eight cards. Pushable Occluder is the new interrupt; Pressure Mines is
preserved from the current rail.

The rail order is curated for rhythm, not for chronology of completion:

| # | Card | Stamp | One-line gloss |
| --- | --- | --- | --- |
| 1 | Sundog Balance | `OPERATING ENVELOPE` | Recovers inside the lighting envelope; fails cleanly at the shadow boundary. |
| 2 | Three-Body Dynamics | `OPERATING ENVELOPE` | Guarded local signals survive the near-escape pocket, not the whole cosmos. |
| 3 | Photometric Alignment | `CONFIRMED` | No target coordinates; just detector response, motion, and a closed loop. |
| 4 | **Pushable Occluder** | **`BOUNDARY FOUND`** | The beam was visible. The path was not. |
| 5 | Pressure Mines | `OPERATING ENVELOPE` | A noisy pressure field buys more safe progress before failure inside a narrow mapped pocket. |
| 6 | EyesOnly | `PLAUSIBLE` | Roguelike agents acting from compressed perception instead of full sight. |
| 7 | Dungeon Gleaner | `PLAUSIBLE` | NPC motion pulled by verbs and needs, not just shortest-path errands. |
| 8 | Money Bags | `PLAUSIBLE` | Softbody terrain reads through graph signatures, recovery, and strain. |

The interrupt sits in position 4 deliberately. Positions 1-3 are
positive results of increasing rigor (envelope, envelope, confirmed).
Position 4 is the deliberate cut. Positions 5-8 then resume the
positive cadence but in a chastened register. This is the
MythBusters rhythm debunked.md describes.

### Card 4: Pushable Occluder (detail)

The rail card's full data contract:

```html
<article
  class="motion-card"
  data-title="Pushable Occluder"
  data-description="The beam was visible. The path was not."
  data-href="applications-gallery.html#pushable-occluder"
  data-evidence-href="docs/PUSHABLE_OCCLUDER_ROADMAP.md"
  data-poster="/media/pushable-occluder-poster.jpg"
  data-media=""
  data-media-format=""
  data-stamp="BOUNDARY FOUND"
  data-stamp-meaning="Indirect signal is not enough when the useful gradient appears only after a preparatory action."
  data-stamp-source="docs/PUSHABLE_OCCLUDER_ROADMAP.md"
  data-clip-ms="6800"
  data-dwell-ms="3600"
>
  <!-- card body identical to the other cards; verdict stamp is rendered by
       the rail JS based on data-stamp -->
</article>
```

The card cannot ship until `PUSHABLE_OCCLUDER_ROADMAP.md` has shipped at
least its Phase 1 evidence. Until then the card is held back; it is not
faked at `UNTESTED` because the rail's interrupt rhythm only earns its
weight when the failure is real.

## Accessibility

The verdict stamp is decorative *and* informational. The implementation
must respect both readings:

- The visible stamp has `aria-hidden="true"`. The verdict is also
  exposed via a visually-hidden `<span class="sd-sr-only">` inside the
  card copy, e.g. "Verdict: Boundary Found. Indirect signal is not
  enough when the useful gradient appears only after a preparatory
  action."
- `prefers-reduced-motion: reduce` disables the stamp arrival animation,
  the card shake on the interrupt card, and the loop auto-advance. The
  stamp is present at page load instead.
- The stamp text uses `--sd-font-display` at a minimum of 14px on
  desktop and 12px on mobile narrow viewports; contrast ratios are
  preserved via the `mix-blend-mode: multiply` against the paper
  background (the multiply does *not* reduce computed contrast for the
  AT layer because the readable text comes from the SR span).
- Keyboard navigation: each card retains its existing focus-visible
  outline. The stamp does not become an interactive element.

## Build Phases

### Phase 1 — Layout rebuild + stamp seam (target: 2 sittings)

This phase changes the rail's visual model from
horizontal-scroll-of-equals to center-focus carousel, *without*
introducing stamps. Splitting layout from stamp landing keeps each
change reviewable.

1. Locate the rail behaviour script (likely `public/js/motion-rail.mjs`;
   verify by `grep -R "motion-card" public/js`). Document its current
   responsibility: card cycling, auto-advance, reduced-motion
   handling, media swap.
2. Update `.motion-rail-track` and `.motion-card` CSS in
   `index.html` (or migrated into `public/css/sundog-theme.css` if
   that promotion is also queued) to the center-focus sizing from
   "Layout: Center-Focus Carousel" above.
3. Add `data-rail-active` toggling to the rail script. The active card
   is the one whose centre is closest to the track centre on intersect
   (or, in the auto-cycle path, the one the script just programmatically
   scrolled to). Peeked cards get `filter` + `opacity` + scale-down.
4. Add the auto-cycle controller (timer + IntersectionObserver),
   *driven by a placeholder "card-ready" event* — for Phase 1 the
   event fires after a fixed 3.4s per card so the layout can be
   demoed without the stamp wiring landing yet.
5. Implement pause-on-hover / pause-on-focus / pause-on-manual-nav /
   reduced-motion-disable from "Pause and resume rules" above.
6. Land the "Replay sequence" affordance for the end-of-sequence
   settle state.
7. Land the stamp tokens (`--sd-stamp-*`) and the `.sd-verdict-stamp`
   base class in `public/css/sundog-theme.css` under the `components`
   layer marker. Stamps are not yet rendered.
8. Add a no-op `applyVerdictStamps()` step at end-of-init that walks
   `.motion-card[data-stamp]` and adds a stub `<span
   class="sd-verdict-stamp" data-pending>`. No styling, no animation.

Acceptance:

- Centre card is the visually dominant card; peeked neighbours read as
  promises, not as alternatives.
- Auto-cycle advances every 3.4s in this phase. Hover, focus, manual
  nav, and reduced-motion all behave per the rules above.
- The sequence settles on the last card; the "Replay sequence"
  affordance restores the first card to centre when clicked.
- Mobile narrow viewport (390px and 520px screenshots) confirms the
  asymmetric peek and single-card-on-screen layout reads correctly.
- DOM inspection shows pending stamp spans on each card; no stamp
  visuals on screen.

### Phase 2 — Stamp landing + auto-cycle handoff (target: 1 sitting)

This phase replaces the Phase 1 placeholder "fixed 3.4s timer" with
the real cue-from-stamp choreography.

1. Switch every existing card from `data-status="..."` to `data-stamp`
   per the mapping table. Keep `data-status` populated for one release
   as a safety net.
2. Light up the rubber-stamp aesthetic. Per-verdict colours, ink-bleed
   mask, rotation, position.
3. Replace the fixed-timer auto-cycle from Phase 1 with the
   stamp-driven sequence: media played → stamp armed → dwell → advance.
   `data-clip-ms` and `data-dwell-ms` overrides are honoured.
4. Verdict default dwell table (1800 / 2200 / 3000 / 3600) is encoded
   in the rail script so cards without explicit overrides still get
   the right pacing per verdict.

Acceptance:

- Every existing card has a verdict stamp that arrives after the clip's
  last beat. The stamp landing is the cue to advance.
- The dwell defaults produce a perceptibly different rhythm for
  `CONFIRMED` (slightly longer) and the future `BOUNDARY FOUND` card
  (much longer) than for `OPERATING ENVELOPE` / `PLAUSIBLE`.
- Reduced-motion users see all stamps at page load and the rail does
  not auto-cycle.

### Phase 3 — Pushable Occluder interrupt (blocked by Pushable Occluder roadmap Phase 1)

1. Once `PUSHABLE_OCCLUDER_ROADMAP.md` Phase 1 ships the failure clip
   and the still poster, add the new card markup at rail position 4.
2. Implement the interrupt-specific choreography: -10° rotation, oxblood
   ink, optional 40ms card shake, deeper stamp delay (the clip ends with
   the controller giving up — the stamp lands on the silence after).
3. Update the rail-header eyebrow copy from "Application Rail" to
   "Application Rail — verdicts since last sprint" to set the
   expectation.

Acceptance: the interrupt card breaks the positive cadence in a way
that earns the failure. Three reviewers (any audience) should be able
to describe the card's claim after one pass.

### Phase 4 — Verdict ledger and CI lint

1. Add `docs/_verdict_ledger.md` (a one-screen index): card name →
   stamp → owning roadmap → last-verified date. The ledger is the
   single source of truth for what the rail says.
2. Add a lint step (`npm run check:verdicts` or similar) that fails the
   build if a card's `data-stamp` doesn't match the ledger, or if a
   `CONFIRMED`/`OPERATING ENVELOPE`/`BOUNDARY FOUND` card has no
   resolvable `data-evidence-href`.
3. Remove `data-status` from all cards. Done.

Acceptance: the rail can be re-stamped only by editing the ledger plus
the card markup. The build refuses to ship inconsistent verdicts.

### Phase 5 — Second-wave stamps (deferred)

Reserved for later moves:

- A `STALLED` card when a real failure of that kind occurs. Most likely
  candidate is Sundog vs Mesa-Optimization proxy collapse if the
  experiment in `SUNDOG_V_MESA.md` lands on a falsifying side.
- A second `CONFIRMED` if the photometric result is replicated in a
  matched-baseline study other than the current MuJoCo setup.
- Promotion of the EyesOnly / Dungeon Gleaner / Money Bags cards to
  `OPERATING ENVELOPE` if their workbenches produce matched baselines.

Phase 5 has no fixed date. It is the place future verdict changes
should land cleanly.

## Open Questions

1. **Stamp text wrapping.** "OPERATING ENVELOPE" and "BOUNDARY FOUND"
   are long. Single-line vs two-line in mobile narrow viewports — first
   pass keeps single-line at the cost of smaller font; needs a
   screenshot pass at 390px.
2. **Verdict for Pressure Mines.** `OPERATING ENVELOPE` is the current
   read of `sundog_v_minesweeper.md`. If that workbench is still
   pre-baseline at rail Phase 2 time, demote to `PLAUSIBLE` and re-evaluate.
3. **Rail-header copy.** "Application Rail — verdicts since last sprint"
   is a placeholder. The real eyebrow should reflect the rhythm rather
   than the sprint cadence. Candidates: "Verdicts In Progress",
   "Stamps", "What We Are Willing To Defend Today". Final wording
   lands in Phase 3.
4. ~~**Stamp on hover vs stamp on enter.**~~ Resolved 2026-05-12: stamp
   lands when the centred card's clip ends; hover and focus *pause*
   the post-stamp dwell rather than gating the stamp itself. The
   stamp-on-enter / stamp-on-hover dichotomy goes away in the
   center-focus layout because only one card is ever in the centre at
   a time.
5. ~~**Peeked-card click behaviour.**~~ Resolved 2026-05-12: clicking a
   peeked card **navigates** to that card's `data-href`. A peek is a
   real preview; the user clicked it because they wanted the
   destination. The always-overlaid skip arrows (see "Layout") provide
   the slide-to-centre affordance for users who want to browse rather
   than commit.
6. ~~**End-of-sequence behaviour.**~~ Resolved 2026-05-12: settle on
   the last card; restart action is the persistent replay arrow (above)
   visible throughout the sequence. The loop-forever alternative is
   off the table — it would turn the rail back into a tour and would
   compete with the persistent replay action.

## References

- `debunked.md` — origin brief for the stamp vocabulary and the case
  for an interrupt card.
- `UI_UX_THEME_FOUNDATION.md` §"Step 4d" — the rail's structural role
  in the index page; this roadmap implements that step's data contract.
- `UI_UX_THEME_FOUNDATION.md` §"Paper-Inspired Theme Extension" — the
  paper primitives the stamp composes with. Adding the `.sd-verdict-stamp`
  component class here is a Phase-3-of-paper-extension migration: it
  should land in `public/css/sundog-theme.css` and be cross-referenced
  from `paper-theme-demo.html`.
- `PROMO_HIGHLIGHTS.md` §"Product Highlights" and §"Provocative Statements" —
  message bank for card descriptions and the "Theorem meaning" gloss.
- `APPLICATIONS.md` §"Evidence Tiers" — the canonical tier table the
  stamps map onto.
- `PUSHABLE_OCCLUDER_ROADMAP.md` — owner of the rail's first
  `BOUNDARY FOUND` card.
- `PHASE2_BLOCKS_DESIGN.md` — the upstream experimental design the
  Pushable Occluder roadmap is a productisation of.
- `index.html` lines 697-883 — the live rail markup this roadmap
  migrates.
- `public/css/sundog-theme.css` — the shared stylesheet where stamp
  tokens, the `.sd-verdict-stamp` class, and the `--sd-stamp-*`
  palette will land.
