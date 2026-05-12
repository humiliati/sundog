# Highlights Slide Rail Roadmap

Date: 2026-05-12
Status: scoping doc for the post-hero application motion rail's next pass
Owner cross-references:
- [`UI_UX_THEME_FOUNDATION.md`](UI_UX_THEME_FOUNDATION.md) — Step 4d defines
  the rail as a structural feature of the index page; the paper-theme
  extension (Phases 1-2 of the §"Paper-Inspired Theme Extension") owns the
  visual primitives this roadmap composes.
- [`debunked.md`](debunked.md) — origin brief for the verdict-stamp
  vocabulary and the case for an interrupt card on the rail.
- [`PROMO_HIGHLIGHTS.md`](PROMO_HIGHLIGHTS.md) — message bank for card
  copy; per-product headlines and boundary language live there.
- [`APPLICATIONS.md`](APPLICATIONS.md) — current Evidence Tiers; this
  roadmap maps those tiers onto the new stamp vocabulary.
- [`PUSHABLE_OCCLUDER_ROADMAP.md`](PUSHABLE_OCCLUDER_ROADMAP.md) — the
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
- Visual treatment: rubber-stamp paper ink-bleed aesthetic that composes
  with the paper-theme primitives already landed in
  `public/css/sundog-theme.css`.
- Stamp timing: the stamp arrives **after** the clip's last beat, not at
  page load. The transition itself is the experimental verdict.
- The card data contract (`data-*` attributes already present on
  `.motion-card` plus the new fields needed to drive stamp timing and
  meaning).
- Per-card content plan for the seven current cards plus the new
  Pushable Occluder interrupt.
- Accessibility, reduced-motion, and screen-reader behaviour for the
  stamp animation.
- Build phases that incrementally migrate the live rail without breaking
  it.

Out of scope:

- The canvas content of the experiments. Workbench rendering belongs to
  each workbench's own roadmap.
- The Pushable Occluder experiment itself. That lives in
  [`PUSHABLE_OCCLUDER_ROADMAP.md`](PUSHABLE_OCCLUDER_ROADMAP.md).
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

## Stamp Timing

The stamp must arrive **after** the card's clip plays its last beat.
This is the core experimental-verdict feel. Cards with no animated
content still get an arrival animation, but it fires at end-of-poster
fade rather than at page load.

Choreography per card:

1. Card enters the viewport (IntersectionObserver, `rootMargin: -10%`).
2. Poster fades to media if `data-media` is present; otherwise stays.
3. Media plays once. For static posters, an implied "beat" of 1.6s
   passes.
4. `data-stamp-armed` attribute is set on the `.motion-card` element.
5. The stamp transitions in over ~320ms, accompanied by a 60ms inner-edge
   tick (the "die contact") and a 180ms ink-bloom fade on the rectangle
   border.
6. The card remains in the armed state for the rest of the page session.

For the rail's auto-advance loop:

- When a card returns to the front of the rail after the user has cycled
  past it, the stamp is **already arrived** (no re-animation). This
  prevents the rail from feeling slot-machine-like on repeat views.
- Reduced-motion users see the stamp arrive at page load with no
  animation; the verdict is still legible.

### The Pushable Occluder interrupt

The boundary card is choreographed differently from the rest of the
rail. It is the only card whose visual content depicts the failure
itself:

1. Beam visible, mirror searching, block in path.
2. Detector ring never peaks. The signal dithers around a local
   maximum.
3. Last beat: the controller gives up. The mirror returns to neutral.
4. Stamp hits: `BOUNDARY FOUND` in oxblood, rotated -10° (more than the
   default), with a 40ms shake on the card itself synchronous with the
   stamp contact. The shake amplitude is bounded (~2px) and is disabled
   under `prefers-reduced-motion`.

The interrupt is the only place we spend animation budget on the stamp
contact event itself. Every other card lets the stamp arrive without
shaking the surface.

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

### Phase 1 — Inventory and seam (target: 1 sitting)

1. Locate the rail behaviour script (likely `public/js/motion-rail.mjs`;
   verify by `grep -R "motion-card" public/js`). Document its current
   responsibility: card cycling, auto-advance, reduced-motion
   handling, media swap.
2. Add a no-op `applyVerdictStamps()` step at end-of-init that walks
   `.motion-card[data-stamp]` and adds a stub `<span
   class="sd-verdict-stamp" data-pending>`. No styling, no animation.
3. Land the stamp tokens (`--sd-stamp-*`) and the `.sd-verdict-stamp`
   base class in `public/css/sundog-theme.css` under the `components`
   layer marker.

Acceptance: rail still renders identically. Inspecting the DOM shows
pending stamp spans on each card.

### Phase 2 — Visual landing (target: 1 sitting)

1. Switch every existing card from `data-status="..."` to `data-stamp`
   per the mapping table. Keep `data-status` populated for one release
   as a safety net.
2. Light up the rubber-stamp aesthetic. Per-verdict colours, ink-bleed
   mask, rotation, position.
3. Wire up the IntersectionObserver-driven `data-stamp-armed` toggle
   and the timing choreography. Per-card `data-stamp-delay` override is
   optional and unused at this phase.

Acceptance: every existing card has a verdict stamp that arrives after
poster/clip last beat. Cards on screen at page load animate their
stamps in. Reduced-motion users see the stamp at rest.

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
4. **Stamp on hover vs stamp on enter.** Current plan stamps on
   intersection entry. Alternative: hold the stamp until the user
   focuses or hovers the card, then arrive. This is more interactive
   but loses the auto-rhythm. Park unless usability testing demands it.

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
