# Sundog UI/UX Theme Foundation

Date: 2026-05-07
Status: living roadmap; refreshed 2026-05-18 after the homepage information-architecture review

## Purpose

This document captures the first shared UI direction for `sundog.cc` before the
full page rewrite. The immediate goal is to stop treating each page's inline CSS
as the final theme source. The site now has a shared stylesheet at:

```text
public/css/sundog-theme.css
```

That stylesheet is loaded after the page-local CSS so it can override current
tokens and components without forcing a high-risk rewrite of every page at once.

## Source Material Pulled

### EyesOnly / flapsandseals.com

Local repo:

```text
C:\Users\hughe\Dev\EyesOnly
```

Relevant sources:

- `ROADMAP.md`: portfolio salvage frame, especially the shift from abandoned
  stakeholder product to studio systems case study.
- `docs/PORTFOLIO_TRANSITION_ROADMAP.md`: dossier/CRT portfolio direction and
  page-level reframing.
- `public/css/themes.css`: four-theme custom-property architecture using scoped
  variables and persistent theme attributes.
- `docs/CSS_THEME_CUSTOMIZATION_ROADMAP.md`: variable categories for color,
  typography, atmosphere, borders, buttons, fields, and paper surfaces.
- `docs/HYBRID-LAYOUT-SPEC.md`: paper dossier plus terminal/monitor layout as a
  physical interface metaphor.

Takeaways for Sundog:

- Use a token layer first, not one-off color edits.
- Keep technical credibility by separating display, body, and data type.
- Let surfaces feel authored: paper, instrument panels, signal traces, measured
  glow.
- Avoid pretending portfolio artifacts are finished products. Use evidence and
  case-study language.

### Dungeon Gleaner / BoxForge

Local repo:

```text
C:\Users\hughe\Dev\Dungeon Gleaner Main
```

Relevant sources:

- `docs/BOXFORGE_README.md`: CSS 3D box geometry, pure CSS transforms, glows,
  orb/pyramid composition, and phase animation model.
- `docs/BOXFORGE_TOOLS_ROADMAP.md`: P1/P2/P3/P4 phase model, export pipeline,
  and planned sequence playback.
- `tools/cli/bf-css-emit.js`: standalone CSS export path for `.boxforge.json`
  templates.

Takeaways for Sundog:

- Use BoxForge-style CSS primitives for hero graph and button motion.
- Keep runtime dependencies low. CSS transforms and custom properties are enough
  for many visual pieces.
- Treat animation as explicit phases: idle, hover, active, handoff/settle.
- Professional Sundog use should be restrained: signal diagrams, graph orbits,
  calibrated button depth, not toy-box UI.

### Existing Sundog Direction

Relevant sources:

- `docs/presentation/logo-brief.md`
- `docs/presentation/landing-page-outline.md`
- Current `index.html`, `about.html`, `origin.html`,
  `applications-gallery.html`, `threebody.html`, `balance.html`,
  `mines.html`, `sundog-workbench.html`, `docs/index.html`, and
  `paper-theme-demo.html`

Takeaways:

- Primary identity stays optical/theoremic: halo, indirect signal, parhelion,
  graph geometry.
- Palette should keep cool scientific credibility with gold signal accents, but
  add enough copper, green, violet, paper, and ink variation to avoid a single
  dark-blue theme.
- Typography should support three voices:
  - display: serious theoremic/public headings
  - body: readable public explanation
  - data: equations, metrics, labels, graph instrumentation

## Current Token Direction

The shared stylesheet defines:

```css
--sd-ink
--sd-ink-strong
--sd-sky
--sd-ice
--sd-cloud
--sd-paper
--sd-brass
--sd-amber
--sd-copper
--sd-signal
--sd-violet
--sd-line
--sd-shadow
--sd-font-display
--sd-font-body
--sd-font-data
--sd-phase-duration
```

It also aliases the old page tokens:

```css
--primary-dark
--primary-mid
--accent-gold
--bg-light
--text-dark
--text-light
--muted
--line
```

This lets current pages keep working while we migrate away from hardcoded inline
styles.

## Component Seeds

The shared stylesheet currently normalizes:

- sticky header/nav
- hero background treatment
- buttons and CTA links
- cards and evidence panels
- theorem snippets and data text
- app-card visual bands
- footer/CTA dark surfaces

It also introduces unused starter primitives for the next pass:

- `.sd-boxforge-surface`
- `.sd-phase-button`
- `.sd-signal-graph`

These are deliberately generic. The next UI session can bind them to actual
hero graph markup, application cards, or generated BoxForge exports.

## Next Migration Steps

Refresh note, 2026-05-11: this section is now a mix of historical roadmap and
current backlog. Treat items marked "Landed" as implementation notes, not open
asks.

1. Move repeated header/footer CSS out of each HTML file into
   `public/css/sundog-theme.css`.
   **Landed 2026-05-08:** Header chrome (`.site-header`, `.site-nav`,
   `.site-brand`, `.site-links`, `.site-links a`, hovers, mobile @media)
   and footer chrome (`footer`, `footer a`, `footer a:hover`) now live
   exclusively in the shared stylesheet. Every production page
   (index/origin/applications-gallery/threebody/balance) has had its
   inline chrome blocks deleted; balance retains a trimmed `:root` for
   workbench-specific `--ink`/`--paper` tokens. threebody picked up the
   shared sheet link in this pass too — previously stranded.
   **Refreshed 2026-05-11:** production chrome now also covers About, Docs,
   Mines, Workbench, and the paper demo. Primary nav treats the brand mark as
   the Home link, omits Origin from top-level chrome, and uses About as the
   identity doorway. Footer copyright styling moved into shared
   `.footer-copyright`.
2. Split the shared stylesheet into layers when it grows:
   `tokens`, `base`, `components`, `pages`, `experiments`.
   **Landed 2026-05-08:** `public/css/sundog-theme.css` reorganized into the
   five named sections, with banner-comment markers labeling each. Implementation
   uses comment markers rather than CSS `@layer` at-rules so the load-order
   cascade with inline page `<style>` blocks is preserved unchanged — moving
   shared rules into layers would have made them lose to unlayered inline
   page rules and regressed `index.html .hero`, button hovers, and
   `.cta-link:hover`. To promote to formal `@layer` later, also wrap each
   page's inline `<style>` contents in `@layer overrides { ... }` and declare
   `@layer overrides, tokens, base, components, pages, experiments;` at the
   top of the shared sheet.
3. Replace inline card/button styles with shared component classes.
   **Partially landed 2026-05-08:** Lifted card-hover transform,
   card transitions, default card padding, `.app-card-content` padding,
   `.app-card h3/p/a` typography, generic `section h2`/`section p`
   defaults into the shared sheet. Pulled the matching dead inline
   blocks from `index.html` (sections, benefits, app-cards,
   comparison/proof/research-section bgs, cta-section, button bases).
   Hovers (`.btn-primary:hover`, `.btn-secondary:hover`, `.cta-link:hover`)
   retained inline since they carry alive properties shared doesn't
   replicate. Other pages keep their page-specific shapes
   (`.signature-card`, `.theorem-card`, `.result-item`, `.timeline`,
   `.era-list`, `.source-card`, `.trail-list`, `.story-panel`,
   `.button` standalone) — those are not duplicates of shared, they're
   genuinely page-local content shapes.
4. Create one canonical hero graph component and coordinated motion surfaces
   (split into 4a/4b/4c, with 4d reserved for the post-hero application rail -
   see below - because BoxForge applies cleanly to one canvas and not uniformly
   to all motion surfaces).

   **4a — Index hero (the "sundog eye"; math-driven workbench).**

   Vision driven by halo-display photographs of triple-sun parhelion
   events with circumzenithal arcs. The natural phenomenon reads as an
   eye: the primary sun (a **compass rose / diamond**, formed by sun
   pillar × parhelic-circle crossing — not a fire orb) is the pupil;
   the 22° halo plus stacked secondary halos form the iris layers; the
   prismatic peripheral arcs at the top of frame are the eyelids. Each
   peripheral arc tip implies a virtual sun radiating its own halo
   system — three suns total, nine intersecting halos ("the 9-halo
   eye"). Two parhelia flank the central sun as **outward-facing
   daggers** along the parhelic arc, carrying the physical dispersion
   (red-inside, blue-outside).

   Sundog (the project) is the first mathematical description of the
   parhelion phenomenon. The hero is therefore not a painting but a
   parametric optical render — a workbench visualisation that doubles
   as a brand statement. The math draws the eye.

   **Detailed roadmap, parameter taxonomy, and calibration debt:** see
   [`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md). That document owns
   the slider tier table (math-derived / free / composition fiction),
   the animation phase model (idle / active / selective 3D handoff),
   the calibration items (parhelic-curvature locked at 0.66; CZA
   bell-blur tuning open), and the path-to-promotion phases that end
   with the workbench feeding `index.html`.

   **Programmatic face of the same hero:** see
   [`SUNDOG_GENERATOR_SPEC.md`](SUNDOG_GENERATOR_SPEC.md). The hero is
   no longer just a workbench — it is the human-tunable face of a
   broader Sundog Generator (Prompt → Parser → Pose JSON → Geometry
   Solver → SVG → optional AI) with three modes: render from math,
   compare to sky, make beautiful. Appendant GPT-app tool schema
   (`renderSundog`) lives there. The Generator Spec also owns the
   project's external public framing ("deterministic browser-native
   parhelion renderer, AI optional"), which any hero copy authored in
   this document should keep consistent with.

   **Proof methodology:** see
   [`SUNDOG_OVERLAY_PROTOCOL.md`](SUNDOG_OVERLAY_PROTOCOL.md). The
   overlay procedure is the calibration gate referenced from
   `SUNDOG_V_GEOMETRY.md` Phase 2 and from Generator Mode 2; this
   document does not need to re-encode those thresholds.

   This Step 4a entry retains coordination authority over cross-cutting
   choices that touch other sections of the site:
   - alignment with **4b** (threebody-canvas framing) and **4c**
     (balance-canvas exuberance) so the three pages read as one voice;
   - shared-stylesheet token reservation (`--sd-*`) — any new theme
     tokens the workbench needs go via `sundog-theme.css` per Migration
     Step 6, not the workbench's page-local CSS;
   - BoxForge primitive selection — the `Dev\Dungeon Gleaner Main\tools`
     library is the source for any Phase 6 selective 3D motion in the
     geometry workbench, but the BoxForge orb is parked (the central
     sun is a compass rose, not a sphere).

   **4b — Threebody canvas (formal, deferred 3D).**

   Keep the 2D physics renderer. Upgrade only the *frame* and the
   *palette*: explore a CRT bezel, oscilloscope screen, or paper
   cutout treatment for the canvas border; align line weights and
   trail colors with the index hero's halo language (cool with
   prismatic accents) where it doesn't compromise scientific
   legibility. WebGL / three.js / actual 3D rendering is a separate
   decision, deferred indefinitely — 2D is the canonical view for
   planar three-body dynamics and the cost-to-payoff doesn't pencil
   out for aesthetic-only 3D.

   **4c — Balance canvas (exuberant; Paper Mario / 2.5D indie game).**

   Deliberate aesthetic departure from the scientific-minimal rest of
   the site — balance is the workbench, it gets to have its own
   personality.

   - **Background**: layered parallax. Sky / mid-ground / foreground
     planes with subtle scroll offsets. Could lean into the paper
     tokens we already have (`--sd-paper-warm`, `--sd-paper-ruled`)
     for textured cardstock layers — physical-feeling depth without
     real 3D.
   - **Cart**: pixel-art sprite, rendered at integer scale with
     `image-rendering: pixelated` for crispness. Polished palette —
     not retro-trash, indie-game-quality.
   - **Pole**: pixel-art with a soft drop shadow on the
     ground/wall plane.
   - **Floor / wall**: paper-like textured planes; the
     "shadow-derived cart-pole balancing under partial observation"
     framing in the page description gives a natural hook for the
     wall to be where the cart's projection lives.
   - **Feedback**: small motion lines, sparkles, or anticipation
     squash on cart movement. Mobile/indie polish — feel
     before fidelity.
   - **Palette**: warmer and more saturated than the Sundog cool-blue
     identity. This page can run hot.

   This is the only canvas where we go full visual-design rather
   than scientific-illustration mode.

   Cross-reference / owner split: `docs/SUNDOG_V_BALANCE.md` owns the
   evidence gates and future roadmap phases that decide when this skin is
   allowed to enter promotion. This section owns the "toyful" target:
   layered, tactile, animated sprites that make recovery and failure
   readable at a glance. The visual pass must preserve replay determinism,
   raw diagnostic mode, and Phase 8/10 metric parity.

   Roadmap hook: fold this into `docs/SUNDOG_V_BALANCE.md` Phase 11
   (Balance-first motion rail opening card), Phase 12 (human-vs-agent
   affordances), and Phase 12.5 (dedicated sprite skin). Do not pull it into
   Phase 9 or Phase 10 evidence work.

   **4d - Post-hero application motion rail (broadcast gallery).**

   Verdict-stamp pass for the rail lives in
   [`HIGHLIGHTS_RAIL_ROADMAP.md`](HIGHLIGHTS_RAIL_ROADMAP.md). That doc
   owns the six-value stamp vocabulary
   (`CONFIRMED`/`OPERATING ENVELOPE`/`PLAUSIBLE`/`BOUNDARY FOUND`/
   `STALLED`/`UNTESTED`), the rubber-stamp paper ink-bleed treatment, the
   stamp-arrives-after-clip-last-beat choreography, and the rebuilt card
   data contract. This section retains structural authority over the rail's
   role in the index page; the rail roadmap retains authority over what
   each card defends.

   The index slot after the hero should become a cross-application motion
   gallery rather than a single Balance promo clip. It starts on Sundog Balance
   because Balance is the newest operating-envelope workbench, then makes room
   for Three-Body, photometric alignment, EyesOnly, Dungeon Gleaner, Money Bags,
   and later applications as short loops become available.

   Interaction model: horizontal rail with swipe/drag, arrow buttons, keyboard
   stepping, and a reduced-motion fallback. The Balance loop may play once, then
   ease/peek or scroll to the next application card in the way a streaming-menu
   rail advertises the next title. Auto-advance must pause on hover, focus, drag,
   or `prefers-reduced-motion`.

   Media model: decide format by byte cost and browser support after export
   testing. Prefer tiny `.webm`/MP4/`.ogv` loops before `.gif` unless GIF wins
   on total compatibility and acceptable weight. Every card needs a still-image
   poster and a no-autoplay text fallback.

   Data contract: each card exposes `title`, `description`, `href`,
   `evidenceHref`, `poster`, `media`, `mediaFormat`, and `status`. Empty media
   fields are valid in the scaffold phase; later capture passes fill those
   slots without changing the rail markup or copy contract.

   Phase 11.2 note: the Balance opening card's `poster` slot is filled by
   `public/media/balance-phase10-rail-poster.jpg`, a best/worst Phase 10 replay
   composite. Its `media` slot remains empty until a loop format is worth the
   byte cost.

   Copy model: each card carries a short title and one punchy description line
   over the clip, closer to a Steam-homepage game tag or price-card tease than a
   documentation paragraph. The text should sell the application move without
   expanding the scientific claim. For Balance, the card must name both recovery
   and failure boundary, and link onward to `balance.html` plus the relevant
   evidence artifact.

   **4e - Post-rail evidence interpretation panels.**

   The Working Systems grid that follows the motion rail should stop using
   generic labelled slabs as a long-term visual strategy. On mobile those
   `.app-card-img` placeholders stack into large empty-feeling blocks and repeat
   the card heading. Treat the slot as an evidence interpretation panel instead:
   a compact chart, boundary map, proof ladder, telemetry thumbnail, or
   workbench screenshot that clarifies the claim boundary named by the card.

   Priority panels for the next pass are Mesa Optimization (`SUNDOG_V_MESA.md`
   / `mesa.html` lambda cliff and class balance), the structural-failure
   boundary map (`prereg/structural-failure-coincidence/BOUNDARY_MAP.md`,
   `PUBLICATION_PLAN.md`, and
   `public/data/structural-failure-boundary-map.json`), and the
   coarse-graining proof trunk (`COARSE_GRAINING_PROOF_ROADMAP.md` plus
   `docs/proof/*`). These belong after the rail because they are interpretation
   surfaces rather than motion hooks: they help readers understand why the
   result is bounded.

   Visual rule: no panel may consist only of the project title. If an image or
   chart is not ready, use a visibly intentional "chart pending" state with
   centered text, responsive height, and a source trail. The current placeholder
   polish is a holding state, not the finished design.

   **4f - Homepage information-architecture cleanup.**

   Status: proposed 2026-05-18. This pass supersedes the older assumption that
   the index page needs both a motion rail and a full static Working Systems
   grid. The cleaner public shape is:

   ```text
   index.html = hero, comprehension pitch, load-bearing evidence, application rail,
   compact project map.

   applications-gallery.html = full inventory of working systems, prototypes,
   product expressions, and lower-tier application notes.

   alignment.html / mesa.html / structural-failure.html / chat.html = experiment
   and evidence pages.
   ```

   The goal is to remove reductionist duplication from `index.html`. The home
   page should answer three questions only: what is Sundog, what currently bears
   evidence weight, and where can I inspect the working surfaces.

   **Pass 0 - inventory before moving anything.**

   Status: completed 2026-05-18. The `index.html` reading order was classified
   into 13 sections; every section flagged for removal now has a destination or
   an explicit duplicate-resolution note. Clear for Pass 1.

   - Classify every `index.html` section as one of: home spine, evidence pillar,
     experiment teaser, application preview, glossary/support, or duplicate.
   - Record the intended destination for any removed content:
     `applications-gallery.html`, `about.html`, `legend.html`, `h-of-x.html`,
     `alignment.html`, or delete.
   - Check before proceeding: no section is removed without a destination or an
     explicit "duplicate, delete" note.

   **Pass 1 - make the index contract explicit.**

   Status: completed 2026-05-18. Hero-reading cards still match the current
   animated hero (`home-hero.mjs`: parhelion offset, CZA cutoff, tangent merge).
   `index.html` now labels the pillar section as the evidence surface and the
   rail as application previews.

   - Keep the hero, the comprehension pitch, the hero-reading cards if they still
     describe the current animated hero, Load-Bearing Evidence, the motion rail,
     Ask Sundog only as an evidence teaser, and the compact project map.
   - Add or adjust one sentence above the rail so readers understand the rail is
     application preview, not the evidence ledger.
   - Check before proceeding: a new reader can distinguish "evidence pillar" from
     "application surface" without reading card internals.

   **Pass 2 - move static application inventory out of index.**

   - Remove the static `Working Systems` app-card grid from `index.html`.
   - Confirm every removed card has a richer anchor on `applications-gallery.html`
     with the same or stronger evidence-tier language.
   - Add a clear rail/footer link to the full applications gallery if needed.
   - Checks: anchor sweep for every rail `data-href`; browser smoke at desktop
     and 390px mobile; no large empty `.app-card-img` slabs remain on the
     homepage.

   **Pass 3 - collapse support prose into owning pages.**

   - Move or delete `Core Vocabulary` from the homepage. Vocabulary belongs in
     `legend.html`, `h-of-x.html`, and `about.html`.
   - Move or delete `Why Indirect Signals Matter` if it repeats About-page
     positioning.
   - Shrink `Alignment And Comparators` to a compact pointer unless it is carrying
     data not visible on `alignment.html`.
   - Shrink `Ongoing Research` if About or Repo Map already carries the posture.
   - Check before proceeding: every remaining homepage section has a unique job
     label: hero, pitch, evidence, application rail, experiment teaser, or map.

   **Pass 4 - verification gate.**

   - Run:

     ```powershell
     npm exec -- vite build
     npm run postbuild
     npm run site:routes -- --base http://127.0.0.1:<preview-port>
     ```

   - Browser-smoke `index.html`, `/applications-gallery`, `/alignment`, `/sundog`,
     and `/h-of-x`.
   - Visual checks: desktop, 900px tablet, and 390px mobile. Confirm no horizontal
     overflow, no CTA overlap, no rail controls covering card text, and no section
     that repeats the same app inventory in a different costume.
   - Human-read check: a proofreader should be able to summarize the homepage as
     "claim, pillars, applications" rather than "many cards saying the same
     thing."

5. Decide whether BoxForge exports live as hand-curated CSS snippets or as
   generated files from `.boxforge.json` templates.
6. When the canonical topography/typography direction lands, map it onto the
   existing `--sd-font-*` tokens first, then page sections second.
   **Landed 2026-05-08:** Option C three-voice roster confirmed —
   `--sd-font-display` (Georgia serif), `--sd-font-body` (system sans),
   `--sd-font-data` (system mono). Handwritten token removed; production pages
   migrated off `'Courier New'` literals onto `var(--sd-font-data)` and off
   inline body-font duplicates onto the shared stylesheet.

## Guardrails

- Keep the site readable and professional before making it expressive.
- Use motion to clarify signal, phase, or interaction state.
- Avoid decorative effects that do not carry the Sundog metaphor.
- Do not publish repo root content. Public site output remains `dist/`.
- Keep EyesOnly and Dungeon Gleaner as portfolio/application references, not as
  claims that broaden the core scientific result.

## Easy Actionables as of 2026-05-11

Landed in the current cleanup pass:

1. Remove redundant `Home` from experiment/workbench primary nav where the
   `Sundog` brand link already returns home.
2. Move repeated `.footer-copyright` inline styles into
   `public/css/sundog-theme.css`.
3. Link `paper-theme-demo.html` from the Docs technical-spec section so the
   implemented paper primitives are discoverable.
4. Add the shared controls vocabulary across Three-Body, Balance, and Mines:
   control panels, control groups/rows, action grids, toggle rows, status
   strips, status chips, metrics, boundary panels, notes, and replay tokens.
5. Fix narrow-viewport overflow in the workbench hero/stage areas, and stack
   Mines comparison lanes vertically on phone-width canvas layouts.
6. Delete duplicate page-local control CSS now covered by the shared
   `sd-*` vocabulary, keeping only workbench layout shells and genuinely
   page-specific panels local.

Good next bites:

1. Normalize footer link sets by page role: broad pages can keep About/Docs,
   while experiment pages can keep deeper writeup links.
2. Promote `.sd-paper-card` only onto static explanation/resource cards first;
   avoid moving live controls into the paper metaphor until screenshots prove
   the result stays legible.
3. Add a short nav policy note to `docs/WEBSITE_DEVELOPMENT.md`: brand is Home,
   About is identity, Origin is contextual/deep provenance.
4. Run a recurring mobile screenshot pass after nav changes, especially at
   390px and 520px widths.

---

## Paper-Inspired Theme Extension

Date: 2026-05-08
Status: implemented foundation; selective production adoption pending

2026-05-11 refresh: the paper tokens and component classes exist in
`public/css/sundog-theme.css`, and `paper-theme-demo.html` is the live reference
page. Do not treat Phase 1, Phase 2, or the demo-page task below as still open;
the remaining work is selective promotion onto production pages.

### Inspiration Sources

This extension draws from successful implementations of paper-inspired web design:

- **flapsandseals.com**: Physical paper aesthetic with booking and partner pages
- **EyesOnly repo**: Typography patterns and paper-surface treatments
- **Analog workspace metaphor**: Ruled paper, sticky notes, scotch tape, wood desk

The goal is to layer these tactile, analog visual cues onto Sundog's existing
theoretical/optical identity without losing scientific credibility.

### Visual Component Vocabulary

#### 1. Cards as Ruled/Lined Paper

Cards (evidence panels, research highlights, origin cards, app cards) should
adopt a notebook-paper appearance:

**Visual characteristics:**
- White or cream paper background (`--sd-paper`, `--sd-paper-warm`)
- Horizontal ruled lines (light blue or gray, subtle)
- Optional left margin line (red or pink vertical rule)
- Slight paper texture or grain
- Realistic box shadow for depth
- Optional torn or perforated edge effect

**CSS pattern:**
```css
.sd-paper-card {
  background:
    linear-gradient(transparent 95%, rgba(99, 163, 220, 0.15) 95%),
    var(--sd-paper);
  background-size: 100% 1.5rem;
  box-shadow:
    0 1px 2px rgba(0, 0, 0, 0.12),
    0 4px 12px rgba(0, 0, 0, 0.08);
  border-left: 2px solid rgba(255, 99, 99, 0.3);
  padding-left: calc(2rem + 2px);
}
```

#### 2. Sticky Note Sections

Sections within cards or standalone elements can be styled as post-it notes:

**Visual characteristics:**
- Soft pastel backgrounds (yellow, pink, blue, green)
- Subtle tilt at rest (2-4deg rotation)
- Enhanced tilt on hover (4-8deg rotation)
- Paper-style drop shadow
- Optional corner fold or curl
- Hand-drawn border or slight imperfection

**CSS pattern:**
```css
.sd-sticky-note {
  background: linear-gradient(135deg,
    var(--sd-sticky-yellow) 0%,
    color-mix(in srgb, var(--sd-sticky-yellow) 95%, transparent) 100%
  );
  transform: rotate(-2deg);
  transition: transform var(--sd-phase-duration) ease;
  box-shadow:
    2px 4px 8px rgba(0, 0, 0, 0.15),
    inset 0 -2px 4px rgba(0, 0, 0, 0.05);
  border-radius: 2px;
  padding: 1rem 1.25rem;
}

.sd-sticky-note:hover {
  transform: rotate(-4deg) translateY(-4px);
  box-shadow:
    4px 8px 16px rgba(0, 0, 0, 0.2),
    inset 0 -2px 4px rgba(0, 0, 0, 0.05);
}

/* Sticky note color variants */
.sd-sticky-yellow { --sd-sticky-color: #fef9c3; }
.sd-sticky-pink { --sd-sticky-color: #fce7f3; }
.sd-sticky-blue { --sd-sticky-color: #dbeafe; }
.sd-sticky-green { --sd-sticky-color: #dcfce7; }
```

#### 3. Tape Overlays for Frames

Decorative tape "fastening" elements to the page:

**Visual characteristics:**
- Semi-transparent white or frosted appearance
- Subtle texture or scratches
- Soft shadow beneath tape
- Optional rotation for realism
- Can cross borders or overlap content

**CSS pattern:**
```css
.sd-tape-overlay {
  position: absolute;
  top: -12px;
  left: 50%;
  transform: translateX(-50%) rotate(-2deg);
  width: 120px;
  height: 32px;
  background:
    linear-gradient(135deg,
      rgba(255, 255, 255, 0.85) 0%,
      rgba(255, 255, 255, 0.75) 100%
    );
  box-shadow:
    0 2px 4px rgba(0, 0, 0, 0.1),
    inset 0 1px 0 rgba(255, 255, 255, 0.9),
    inset 0 -1px 0 rgba(0, 0, 0, 0.05);
  border-radius: 2px;
  pointer-events: none;
  opacity: 0.9;
}

/* Optional tape texture */
.sd-tape-overlay::before {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    repeating-linear-gradient(
      45deg,
      transparent,
      transparent 2px,
      rgba(255, 255, 255, 0.1) 2px,
      rgba(255, 255, 255, 0.1) 4px
    );
  border-radius: 2px;
}
```

#### 4. Wooden Desk Background

Body background should evoke a workspace surface:

**Visual characteristics:**
- Dark wood grain texture
- Warm brown tones (walnut, mahogany, oak)
- Subtle grain direction and variation
- Can use CSS gradients or actual texture image
- Should not overwhelm content

**CSS pattern:**
```css
body[data-theme="desk"] {
  background:
    radial-gradient(circle at 20% 30%, rgba(139, 90, 43, 0.15), transparent 40%),
    radial-gradient(circle at 80% 70%, rgba(101, 67, 33, 0.15), transparent 40%),
    repeating-linear-gradient(
      90deg,
      #3e2723 0px,
      #4e342e 2px,
      #3e2723 4px
    ),
    linear-gradient(180deg, #4e342e 0%, #3e2723 100%);
  background-blend-mode: multiply, multiply, overlay, normal;
}
```

#### 5. Manila Folder / Clipboard Containers

Workspace containers (nav, sidebars, main content areas) as manila folders:

**Visual characteristics:**
- Tan or cream background (#f4e4c1, #f5deb3, #e8d5b7)
- Optional tab or folder flap at top
- Slightly textured appearance
- Subtle edge shadow
- Can include metal clip or clasp detail

**CSS pattern:**
```css
.sd-manila-container {
  background:
    linear-gradient(180deg,
      rgba(255, 255, 255, 0.3) 0%,
      transparent 20%
    ),
    #f4e4c1;
  border: 1px solid rgba(139, 90, 43, 0.2);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.5),
    0 4px 12px rgba(0, 0, 0, 0.15);
}

/* Optional folder tab */
.sd-manila-container::before {
  content: "";
  position: absolute;
  top: -16px;
  right: 20%;
  width: 120px;
  height: 16px;
  background: linear-gradient(180deg, #e8d5b7 0%, #f4e4c1 100%);
  border: 1px solid rgba(139, 90, 43, 0.2);
  border-bottom: none;
  border-radius: 4px 4px 0 0;
}
```

### Typography Hierarchy

#### Hand-Written Fonts (Sticky Notes & Paper) — DEFERRED

Status: **Not in Option C.** Retained here as reference material in case the
direction shifts toward Option A or D in a future pass.

The 2026-05-08 typography audit removed the `--sd-font-handwritten` token, the
`.sd-handwritten` class, and the sticky-note `font-family` rule from
`public/css/sundog-theme.css`, and dropped the Google Fonts `<link>` tags from
`paper-theme-demo.html`. Sticky-note color tokens remain available as accents,
but their text inherits the body sans stack.

If reactivating later:

- `'Indie Flower'` — casual handwriting, high readability
- `'Patrick Hand'` — clean hand-drawn style
- `'Caveat'` — natural handwriting with personality
- `'Shadows Into Light'` — friendly and legible

**Best practices when reactivated:**
- Use sparingly for emphasis and informal sections
- Ensure adequate size (minimum 16px) for readability
- Avoid for long-form text or critical data
- Pair with increased line-height for comfort

#### Console/Professional Fonts (Body & UI)

For technical content, navigation, buttons, and general interface:

**Recommended fonts:**
- `'Roboto Mono'` — clean, professional monospace
- `'Fira Mono'` — Mozilla's monospace, great for code
- `'Menlo', 'Monaco', 'Consolas'` — system monospace fallbacks
- `'Roboto', 'Inter', 'System UI'` — clean sans-serif for UI

**Current tokens:**
```css
--sd-font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
--sd-font-data: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
```

**Usage remains unchanged for:**
- Navigation links
- Body paragraphs (non-paper contexts)
- Buttons and form controls
- Data labels and metrics

#### Classic Serif (Major Titles)

For primary headings, hero titles, and prestigious declarations:

**Recommended fonts:**
- `Georgia` — web-safe serif, excellent readability
- `'Times New Roman'` — classic, authoritative
- `'Playfair Display'` — elegant display serif (Google Fonts)
- `'Merriweather'` — modern serif optimized for screens

**Current token:**
```css
--sd-font-display: Georgia, "Times New Roman", serif;
```

**Usage:**
```css
.hero h1,
.page-hero h1,
section > h2 {
  font-family: var(--sd-font-display);
  font-weight: 700;
  letter-spacing: -0.02em;
}
```

### Theme Integration Strategy

Implementation status:

- Phase 1 tokens: landed in `public/css/sundog-theme.css`.
- Phase 2 component classes: landed for `.sd-paper-card`, `.sd-sticky-note`,
  `.sd-tape-overlay`, `.sd-desk-bg`, and `.sd-manila-container`.
- Demo/reference page: landed as `paper-theme-demo.html` and linked from Docs.
- Phase 3 production mapping: partial. Some existing cards already receive
  ruled-paper treatment through shared selectors, but the explicit `.sd-*`
  classes remain mostly demo/reference primitives.

#### Phase 1: Extend CSS Custom Properties

Add new tokens to `public/css/sundog-theme.css`:

```css
:root {
  /* Existing tokens remain */

  /* Paper surfaces */
  --sd-paper-ruled: #fefdfb;
  --sd-paper-line: rgba(99, 163, 220, 0.15);
  --sd-paper-margin: rgba(255, 99, 99, 0.3);

  /* Sticky note colors */
  --sd-sticky-yellow: #fef9c3;
  --sd-sticky-pink: #fce7f3;
  --sd-sticky-blue: #dbeafe;
  --sd-sticky-green: #dcfce7;

  /* Tape and fastening */
  --sd-tape-base: rgba(255, 255, 255, 0.85);
  --sd-tape-shadow: rgba(0, 0, 0, 0.1);

  /* Wooden desk */
  --sd-wood-dark: #3e2723;
  --sd-wood-mid: #4e342e;
  --sd-wood-light: #5d4037;

  /* Manila/folder */
  --sd-manila-base: #f4e4c1;
  --sd-manila-dark: #e8d5b7;
  --sd-manila-border: rgba(139, 90, 43, 0.2);

  /* Typography additions */
  --sd-font-handwritten: 'Indie Flower', 'Patrick Hand', cursive;
}
```

#### Phase 2: Component Class Library

Create reusable component classes:

- `.sd-paper-card` — ruled notebook paper cards
- `.sd-sticky-note` — post-it style sections
- `.sd-sticky-yellow`, `.sd-sticky-pink`, etc. — color variants
- `.sd-tape-overlay` — decorative tape fastening
- `.sd-manila-container` — folder/clipboard containers
- `.sd-desk-bg` — wooden desk background
- `.sd-handwritten` — handwriting font application

#### Phase 3: Apply to Existing Components

Map new classes to current elements:

- `.origin-card`, `.research-highlight`, `.benefit-card` → `.sd-paper-card`
- Subsections within cards → `.sd-sticky-note`
- `.hero`, `.page-hero` → optional `.sd-desk-bg` variant
- Nav/sidebar containers → `.sd-manila-container`

#### Phase 4: Testing and Refinement

- Verify readability across devices and screen sizes
- Test hover states and animations
- Ensure accessibility (contrast ratios, focus states)
- Validate against scientific credibility guardrail

### Code Snippets Reference

#### Complete Ruled Paper Card

```css
.sd-paper-card {
  position: relative;
  background:
    /* Horizontal ruled lines */
    linear-gradient(transparent 95%, var(--sd-paper-line) 95%),
    var(--sd-paper-ruled);
  background-size: 100% 1.5rem;
  box-shadow:
    0 1px 2px rgba(0, 0, 0, 0.12),
    0 4px 12px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(20, 43, 58, 0.08);
  border-left: 2px solid var(--sd-paper-margin);
  border-radius: var(--sd-radius-md);
  padding: 1.5rem 1.5rem 1.5rem calc(2rem + 2px);
  transition: box-shadow var(--sd-phase-duration) ease;
}

.sd-paper-card:hover {
  box-shadow:
    0 2px 4px rgba(0, 0, 0, 0.15),
    0 8px 20px rgba(0, 0, 0, 0.12);
}
```

#### Complete Sticky Note with Variants

```css
.sd-sticky-note {
  position: relative;
  background: linear-gradient(135deg,
    var(--sd-sticky-yellow) 0%,
    color-mix(in srgb, var(--sd-sticky-yellow) 95%, #000) 100%
  );
  transform: rotate(-2deg);
  transition: transform var(--sd-phase-duration) ease,
              box-shadow var(--sd-phase-duration) ease;
  box-shadow:
    2px 4px 8px rgba(0, 0, 0, 0.15),
    inset 0 -2px 4px rgba(0, 0, 0, 0.05);
  border-radius: 2px;
  padding: 1rem 1.25rem;
  margin: 1rem;
}

.sd-sticky-note:hover {
  transform: rotate(-4deg) translateY(-4px);
  box-shadow:
    4px 8px 16px rgba(0, 0, 0, 0.2),
    inset 0 -2px 4px rgba(0, 0, 0, 0.05);
}

.sd-sticky-note.sd-sticky-pink {
  background: linear-gradient(135deg,
    var(--sd-sticky-pink) 0%,
    color-mix(in srgb, var(--sd-sticky-pink) 95%, #000) 100%
  );
}

.sd-sticky-note.sd-sticky-blue {
  background: linear-gradient(135deg,
    var(--sd-sticky-blue) 0%,
    color-mix(in srgb, var(--sd-sticky-blue) 95%, #000) 100%
  );
}

.sd-sticky-note.sd-sticky-green {
  background: linear-gradient(135deg,
    var(--sd-sticky-green) 0%,
    color-mix(in srgb, var(--sd-sticky-green) 95%, #000) 100%
  );
}

/* Content within sticky notes should use handwritten font */
.sd-sticky-note p,
.sd-sticky-note li,
.sd-sticky-note span {
  font-family: var(--sd-font-handwritten);
  font-size: 1.05rem;
  line-height: 1.7;
}
```

#### Complete Tape Overlay

```css
.sd-tape-overlay {
  position: absolute;
  top: -12px;
  left: 50%;
  transform: translateX(-50%) rotate(-2deg);
  width: 120px;
  height: 32px;
  background:
    linear-gradient(135deg,
      rgba(255, 255, 255, 0.88) 0%,
      rgba(255, 255, 255, 0.78) 100%
    );
  box-shadow:
    0 2px 4px var(--sd-tape-shadow),
    inset 0 1px 0 rgba(255, 255, 255, 0.95),
    inset 0 -1px 0 rgba(0, 0, 0, 0.05);
  border-radius: 2px;
  pointer-events: none;
  opacity: 0.92;
  z-index: 10;
}

/* Subtle tape texture */
.sd-tape-overlay::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    repeating-linear-gradient(
      90deg,
      transparent,
      transparent 3px,
      rgba(255, 255, 255, 0.15) 3px,
      rgba(255, 255, 255, 0.15) 6px
    );
  border-radius: 2px;
  mix-blend-mode: overlay;
}

/* Positioning variants */
.sd-tape-overlay.sd-tape-top-left {
  left: 15%;
  transform: translateX(0) rotate(8deg);
}

.sd-tape-overlay.sd-tape-top-right {
  left: 85%;
  transform: translateX(-100%) rotate(-8deg);
}
```

#### Wooden Desk Background

```css
body[data-theme="desk"],
.sd-desk-bg {
  background:
    /* Subtle lighting variations */
    radial-gradient(circle at 20% 30%, rgba(139, 90, 43, 0.15), transparent 40%),
    radial-gradient(circle at 80% 70%, rgba(101, 67, 33, 0.15), transparent 40%),
    /* Wood grain texture */
    repeating-linear-gradient(
      90deg,
      var(--sd-wood-dark) 0px,
      var(--sd-wood-mid) 2px,
      var(--sd-wood-dark) 4px
    ),
    /* Base wood color */
    linear-gradient(180deg, var(--sd-wood-mid) 0%, var(--sd-wood-dark) 100%);
  background-blend-mode: multiply, multiply, overlay, normal;
  min-height: 100vh;
}
```

#### Manila Folder Container

```css
.sd-manila-container {
  position: relative;
  background:
    /* Paper highlight */
    linear-gradient(180deg,
      rgba(255, 255, 255, 0.3) 0%,
      transparent 20%
    ),
    var(--sd-manila-base);
  border: 1px solid var(--sd-manila-border);
  border-radius: var(--sd-radius-md);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.5),
    0 4px 12px rgba(0, 0, 0, 0.15);
  padding: 2rem;
  overflow: hidden;
}

/* Optional folder tab */
.sd-manila-container.with-tab::before {
  content: "";
  position: absolute;
  top: -16px;
  right: 20%;
  width: 120px;
  height: 16px;
  background: linear-gradient(180deg, var(--sd-manila-dark) 0%, var(--sd-manila-base) 100%);
  border: 1px solid var(--sd-manila-border);
  border-bottom: none;
  border-radius: 4px 4px 0 0;
  z-index: -1;
}

/* Subtle paper texture */
.sd-manila-container::after {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0, 0, 0, 0.02) 2px,
      rgba(0, 0, 0, 0.02) 4px
    );
  pointer-events: none;
  mix-blend-mode: multiply;
}
```

### Implementation Notes

1. **Google Fonts Integration**: Add to HTML `<head>`:
   ```html
   <link rel="preconnect" href="https://fonts.googleapis.com">
   <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
   <link href="https://fonts.googleapis.com/css2?family=Indie+Flower&family=Patrick+Hand&display=swap" rel="stylesheet">
   ```

2. **Progressive Enhancement**: All paper effects should degrade gracefully on
   older browsers. Core content must remain readable without CSS transforms or
   blend modes.

3. **Performance**: Limit use of box-shadows and transforms to hover states
   where possible. Use `will-change` sparingly and only on interactive elements.

4. **Accessibility**:
   - Maintain WCAG AA contrast ratios (4.5:1 for body text)
   - Ensure handwritten fonts are large enough (minimum 16px)
   - Provide `prefers-reduced-motion` alternatives for tilt animations
   - Keep focus indicators visible on all interactive elements

5. **Theme Consistency**: Paper effects should complement, not replace, the
   existing optical/theoretical identity. Use them to add warmth and tactility
   without abandoning the scientific credibility foundation.

### Next Steps for Implementation

1. Promote paper primitives selectively to static explanation/resource cards,
   starting with one page at a time.
2. Keep `paper-theme-demo.html` as the visual reference instead of duplicating
   large snippets in production pages.
3. Move remaining repeated inline footer/chrome styles into shared classes when
   they are exact duplicates.
4. Keep handwritten fonts deferred unless the brand deliberately moves toward a
   more informal annotation layer.
5. Test promoted components across desktop and mobile screenshots before
   deploying.

### References

- EyesOnly CSS themes: `C:\Users\hughe\Dev\EyesOnly\public\css\themes.css`
- EyesOnly portfolio direction: `C:\Users\hughe\Dev\EyesOnly\docs\PORTFOLIO_TRANSITION_ROADMAP.md`
- EyesOnly hybrid layout spec: `C:\Users\hughe\Dev\EyesOnly\docs\HYBRID-LAYOUT-SPEC.md`
- Inspiration: flapsandseals.com (booking/partners pages)
- BoxForge primitives: Dungeon Gleaner `docs/BOXFORGE_README.md`

---

## Interactive Components Roadmap

Date: 2026-05-08
Status: shared controls pass landed; remaining work is selective refinement

### Overview

While the paper-inspired theme provides a strong foundation for static content
(cards, containers, typography), the Sundog site includes several interactive
components that require thoughtful styling to maintain cohesion with the paper
aesthetic while ensuring usability and scientific credibility.

2026-05-11 refresh: the effective direction is Option D with Option C restraint.
Navigation and public copy stay professionally minimal, static explanation cards
can borrow the paper system, and live workbenches should keep a restrained
instrument-panel treatment until a dedicated controls pass lands.

Shared controls pass, 2026-05-11: Three-Body, Balance, and Mines now share
explicit `sd-control-*`, `sd-status-*`, `sd-boundary-*`, `sd-note`, and
`sd-replay-token` classes while preserving page-local IDs and JS hooks.
The follow-up duplicate-control cleanup also landed: common panel, form,
button, status-strip, boundary, note, and replay-token styling now lives in the
shared sheet, while page-local CSS retains layout positioning and unique
workbench content treatments.

### Current Interactive Elements Inventory

Based on analysis of `index.html`, `threebody.html`, and other pages:

#### 1. Canvas Rendering Surfaces

**Current usage:**
- `#parhelion-canvas` in `index.html` — optical phenomenon visualization
- `#threebody-canvas` in `threebody.html` — real-time physics simulation
- Future: BoxForge-generated experiment visualizations

**Characteristics:**
- Full-screen or hero-section backgrounds
- Real-time interactive rendering
- Scientific data visualization
- Currently use dark gradient backgrounds

**Styling considerations:**
- Canvas contents are slated for BoxForge upgrades (separate deliverable)
- Frame/border treatment needs definition
- Background context (desk, paper, or standalone)
- Overlay controls and labels positioning

#### 2. Buttons and CTAs

**Current usage:**
- Play/Pause controls (`#btn-play-pause`)
- Reset system button (`#btn-reset`)
- CTA buttons in hero sections (`.cta-buttons`)
- Navigation interactions

**Characteristics:**
- Primary actions (start/stop, reset)
- Secondary navigation (links, tabs)
- State changes (active, disabled, hover)

**Styling considerations:**
- Should buttons feel like physical controls?
- Paper-themed vs. monitor/screen-themed
- Accessibility (focus states, contrast)

#### 3. Form Controls

**Current form elements:**
- Range sliders (mass ratio, time speed, thrust limit, target tidal, initial conditions)
- Checkboxes (sensor mode, display toggles, trail visibility)
- Dropdown/select menus (controller mode selector)
- Text inputs (currently "aloof" per user feedback)

**Characteristics:**
- Real-time parameter adjustment
- Toggle states for visualization options
- Numeric and categorical inputs

**Styling considerations:**
- Tactile feedback for sliders
- Checkbox presentation (paper clips? stamps? checkmarks?)
- Dropdown styling consistency with theme
- Input field treatment (forms on paper? digital displays?)

#### 4. Experimental/Special UI

**Inspiration from flapsandseals.com:**
- CRT monitor framing for digital content
- Porthole-style buttons revealing starfield backgrounds
- Hybrid analog/digital aesthetic

**Potential applications:**
- Canvas screens could be "monitors" on a paper desk
- Buttons as physical switches, dials, or portholes
- Dropdowns as file folder tabs or paper accordion menus

### Styling Philosophy Options

We have several non-exclusive approaches to explore:

#### Option A: Pure Paper Metaphor

Extend paper-inspired theme to all interactive elements:
- **Buttons**: Raised paper rectangles with fold/crease effects, pressed state
- **Sliders**: Paper strips sliding along ruled guides, or ribbon bookmarks
- **Checkboxes**: Handwritten checkmarks or rubber stamp effects
- **Dropdowns**: Accordion-folded paper or tabbed index cards
- **Canvas frames**: Clipboards, picture frames, or paper cutouts

**Pros**: Maximum thematic coherence, distinctive identity
**Cons**: May feel overly whimsical for scientific content, usability questions

#### Option B: Hybrid Analog/Digital (flapsandseals.com approach)

Mix paper surfaces with retro-tech UI elements:
- **Buttons**: CRT-style porthole buttons, toggle switches, or analog dials
- **Sliders**: Vintage radio tuning knobs or oscilloscope controls
- **Canvas frames**: Monitor bezels, oscilloscope screens, or radar displays
- **Paper elements**: Static content, notes, documentation
- **Tech elements**: Interactive controls, live data, simulations

**Pros**: Justifies digital interactions, nostalgic tech aesthetic, clear separation
**Cons**: More complex to implement, requires careful balance

#### Option C: Minimal Professional

Maintain clean, modern interface with subtle paper touches:
- **Buttons**: Standard modern buttons with paper textures or shadows
- **Forms**: Clean inputs with optional notebook-line backgrounds
- **Canvas**: Simple frames or borderless integration
- **Paper accent**: Limited to card backgrounds and decorative elements

**Pros**: Safest for scientific credibility, broad accessibility
**Cons**: Less distinctive, may not fully leverage paper theme potential

#### Option D: Context-Adaptive

Different UI styles for different content contexts:
- **Documentation/static content**: Full paper theme
- **Interactive experiments**: Retro-tech/monitor theme
- **Navigation/chrome**: Professional minimal theme
- **Annotations/notes**: Hand-drawn/sticky note theme

**Pros**: Best of all worlds, semantic clarity, flexible
**Cons**: Most complex to design and maintain, requires clear guidelines

### Roadmap Phases

#### Phase 1: Design Exploration (No Implementation Yet)

**Goal**: Explore styling directions without committing to specific designs

**Tasks**:
1. **Canvas Frame Treatment**
   - Experiment with monitor bezel styles (CRT, oscilloscope, radar)
   - Explore clipboard/picture frame alternatives
   - Consider borderless integration with desk background
   - Define BoxForge canvas content upgrade handoff point

2. **Button Style Experiments**
   - Mock up porthole/reveal-style buttons (flapsandseals.com inspiration)
   - Test raised paper button concepts
   - Explore toggle switch and dial alternatives
   - Prototype primary vs. secondary button treatments

3. **Form Control Concepts**
   - Slider styling directions (ribbon, tuning knob, paper strip)
   - Checkbox alternatives (checkmark, stamp, clip, punch hole)
   - Dropdown/select menu approaches (folder tabs, accordion, standard)
   - Text input field treatments (ruled lines, typewriter, minimal)

4. **Widget Patterns**
   - Control panel grouping styles
   - Legend/key presentation
   - Real-time data display formats
   - Status indicator patterns

**Deliverables**:
- Design sketches or wireframes (not CSS implementation)
- Pros/cons analysis for each approach
- Accessibility and usability notes
- Recommendation for styling philosophy (Options A-D)

#### Phase 2: Prototype & Test (Future)

**Tasks**:
1. Implement selected button and form styles in isolated demo
2. Create interactive canvas frame prototypes
3. User testing for usability and accessibility
4. Performance testing (especially for canvas interactions)
5. Cross-browser compatibility checks

#### Phase 3: Integration (Future)

**Tasks**:
1. Add new CSS component classes to `sundog-theme.css`
2. Update existing pages (`index.html`, `threebody.html`)
3. Document new component usage guidelines
4. Create component library or style guide

#### Phase 4: BoxForge Canvas Content (Separate Deliverable)

**Scope**: Canvas *content* rendering upgrades (separate from frame styling)

**Considerations**:
- BoxForge will handle internal visualization improvements
- Frame/border styling from Phases 1-3 should accommodate BoxForge output
- Coordinate handoff: frame styling vs. content rendering

### Current Interactive UI Inventory

#### Sundog Index Page (`index.html`)

**Canvas elements:**
- `#parhelion-canvas` — Hero section optical visualization

**Buttons:**
- CTA buttons (`.cta-buttons`) — Styled through the shared theme; remaining work
  is content priority and density, not base styling

**Interactive needs:**
- Hero CTA hierarchy and label prioritization
- Smooth scroll navigation triggers

#### Three-Body Experiment Page (`threebody.html`)

**Canvas elements:**
- `#threebody-canvas` — Real-time physics simulation

**Form controls:**
- Play/Pause button (`#btn-play-pause`)
- Reset button (`#btn-reset`)
- Mass ratio slider (`#mass-ratio`)
- Time speed slider (`#time-speed`)
- Sensor mode checkbox (`#sensor-mode`)
- Controller mode dropdown (`#controller-mode`)
- Thrust limit slider (`#thrust-limit`)
- Target tidal slider (`#target-tidal`)
- Display toggle checkboxes (thrust, virial, inertia, energy, tidal, trails)
- Initial condition sliders (x, y, vx, vy)

**Interactive needs:**
- Control panel container styling (manila folder? tech panel?)
- Button primary/secondary hierarchy
- Slider track and thumb styling
- Checkbox visual treatment
- Dropdown menu styling
- Real-time value displays

#### Balance and Mines Workbenches (`balance.html`, `mines.html`)

**Canvas / board elements:**
- Balance uses a canvas-style physics workbench and diagnostics panels.
- Mines uses a board/projection surface plus telemetry and comparison controls.

**Form controls:**
- Buttons, selects, range sliders, checkboxes, seed inputs, replay/copy actions,
  and mode selectors now repeat across both pages.

**Interactive needs:**
- Shared class names for repeated control groups.
- Unified focus, hover, active, and disabled states.
- Mobile density review for stacked controls.
- Reduced-motion behavior for replay and board/projection animations.

### Design Questions to Resolve

Before implementation, we need to decide:

1. **Canvas framing philosophy**:
   - Are experiment canvases "screens/monitors" or "paper cutouts/frames"?
   - Should different canvas types use different frames?

2. **Button aesthetic**:
   - Paper-style raised buttons, or retro-tech switches/portholes?
   - How do disabled/active states manifest?

3. **Form control coherence**:
   - Should all form elements use the same metaphor (all paper OR all tech)?
   - Or can sliders be tech while checkboxes are paper?

4. **Typography for interactives**:
   - Which font family for button labels? (Currently use body font)
   - Do form labels use handwritten font or stay professional?

5. **Color and contrast**:
   - Can pastel sticky-note colors work for active button states?
   - What background works best for form control contrast?

6. **Animation and feedback**:
   - Should buttons have paper-crumple effects or tech-glow effects?
   - How do we indicate hover/focus states accessibly?

### Next Steps

1. Add one or two shared status-chip variants if future evidence pages need
   warning/success/error states outside boundary panels.
2. Screenshot-test desktop and 390px mobile after each workbench change.
3. Keep BoxForge as canvas-content work, not a blocker for control-frame polish.

### References for Interactive Styling

- flapsandseals.com: CRT monitor frames, porthole buttons, starfield reveals
- EyesOnly booking page: Interactive form styling examples
- Sundog existing controls: `threebody.html` lines 190-220 (control styles)
- Material Design: Accessibility standards for form controls
- Paper UI inspirations: Origami buttons, folded tabs, paper sliders

## Sundog Mines Workbench Animation Roadmap

This section outlines the animation and visual polish plan for the Pressure
Mines workbench (`mines.html`), following the phased implementation structure
from the main roadmap in `docs/sundog_v_minesweeper.md`.

### Animation Goals

The Mines workbench should feel like a **laboratory field board**, not a toy
puzzle clone. The visual language must make the distinction between direct
knowledge and field-reading unmistakable:

- The board is hidden (mines are not visible)
- The field is visible (pressure distortions are rendered)
- The field must look like a projection, not like disguised number labels

### Core Animation Phases

#### Phase 6 Animation: Real-Time Web Projection (Initial Polish)

This phase runs parallel to Phase 6 core implementation and establishes the
base visual language.

**Deliverables:**

1. **Tile State Transitions**
   - Concealed → Revealing (scan pulse animation)
   - Revealing → Revealed (pressure field fade-in)
   - Safe tile → Flagged (flag placement animation)
   - Flagged → Unflagged (flag removal)
   - Tile hover states (pressure value preview)

2. **Pressure Field Rendering**
   - Gradient-based pressure visualization (not discrete numbers)
   - Heatmap or contour-style rendering showing field distortion
   - Animated pressure wave propagation on reveal
   - Confidence/variance overlay (shimmer or blur for uncertain regions)

3. **Scan Pulse Effects**
   - Radial pulse animation originating from scanned tile
   - Information-gathering visual (ripple, sonar-ping style)
   - Feedback showing scan result (safe corridor vs. hazard proximity)

4. **Action Feedback**
   - Recent action highlights (last 3-5 tiles glow briefly)
   - Controller decision breadcrumbs
   - Safe reveal success animation (subtle positive feedback)
   - Mine trigger animation (clear failure state)

5. **Telemetry Panel Animation**
   - Live-updating confidence trace (line chart animation)
   - Frontier size indicator (animated bar/number)
   - Survival probability meter (smooth transitions)
   - False-flag counter (increment animation)

**Visual Style Constraints:**

- Match existing Sundog workbench aesthetic (Balance/Three-body)
- Dark workbench background (#0D2030 or similar)
- Gold/amber accents for interactive elements
- Monospace font for data displays
- Smooth, measured transitions (not flashy game animations)

#### Post-Phase 6: Polish and Juice Pass

After core functionality lands, a dedicated polish pass adds secondary
animations and refinements.

**Additional Animations:**

1. **Board Load Animation**
   - Tiles fade in sequentially (scan-line effect)
   - Pressure field gradually resolves
   - Board seed/preset announcement

2. **Controller State Changes**
   - Mode transition animations (Passive → Sundog, etc.)
   - Strategy shift indicators (SCAN → SEEK → TRACK)
   - Confidence gate activation/deactivation

3. **Failure Boundary Warnings**
   - Visual warning when approaching observability cliff
   - Pressure field degradation animation (signal loss)
   - Controller fallback indicators

4. **Comparison Mode UI**
   - Side-by-side board sync animations
   - Matched action replay highlights
   - Performance differential visualization

5. **Replay and Seed Controls**
   - Replay speed indicator
   - Step-through scrubbing feedback
   - Seed string copy confirmation

**Performance Considerations:**

- Throttle animations during high-speed replay
- Respect reduced-motion accessibility settings
- Optimize canvas redraws (only changed regions)
- Consider requestAnimationFrame budgets

#### Phase 8+ Animation: Metrics Visualization

When event metrics and recovery analysis land, add supporting visualizations.

**Deliverables:**

1. **Event Timeline**
   - Horizontal timeline showing safe reveals, risky reveals, flags, scans
   - Color-coded event markers
   - Scrubbing/seeking through timeline

2. **Confidence Trace Chart**
   - Real-time line chart of controller confidence
   - Threshold markers for confidence gates
   - Historical confidence decay visualization

3. **Frontier Collapse Indicator**
   - Visual warning before forced-risk regions
   - Frontier quality score meter
   - Lead-time countdown

4. **Comparative Replay Animation**
   - Dual-pane synchronized replay
   - Outcome divergence highlighting
   - Victory/failure path comparison

#### Phase 9+ Animation: Boundary Visualization

When degradation sweeps and boundary mapping are complete, add failure-mode
animations.

**Deliverables:**

1. **Sensor Degradation Effects**
   - Increasing noise visualization (field static/grain)
   - Delay lag animation (ghosting, trailing indicators)
   - Dropout animation (tiles losing signal)

2. **Operating Envelope Map UI**
   - Interactive 2D parameter grid
   - Cell color-coding (positive/negative/boundary)
   - Hover expansion showing cell details
   - Navigation to best/worst-case replay URLs

3. **Failure-Mechanism Labels**
   - Animated failure annotations
   - Diagnostic overlay showing why controller failed
   - Field ambiguity visualization (overlapping pressure plateaus)

### Implementation Strategy

**DO:**
- Use CSS transitions for simple state changes (opacity, color, scale)
- Use canvas animation for complex field effects (pressure gradients, pulses)
- Implement animation timing that matches workbench pace (deliberate, not frenetic)
- Add accessibility controls (reduced motion, animation speed)

**DON'T:**
- Add gratuitous particle effects or game-style explosions
- Use animations that obscure telemetry or occlude controls
- Implement animations that block interaction (no forced waits)
- Prioritize animation over functional clarity

### Accessibility Requirements

All animations must respect:
- `prefers-reduced-motion` media query (disable decorative animations)
- WCAG contrast requirements (animated elements remain legible)
- Keyboard navigation (animations don't interfere with focus)
- Screen reader compatibility (animations are non-essential)

### Technical Notes

**Animation Libraries:**
- Prefer vanilla CSS/Canvas over heavy animation libraries
- Consider GSAP only if complex timeline choreography is needed
- Match Balance/Three-body implementation patterns for consistency

**Performance Budget:**
- Target 60fps on reference hardware
- Degrade gracefully on low-end devices
- Profile canvas redraws during development
- Use layer promotion sparingly (will-change, transform3d)

### Timeline Integration with Main Roadmap

| Main Phase | Animation Work | Priority |
|------------|----------------|----------|
| Phase 0-1 | None (core logic only) | N/A |
| Phase 2 | Prototype pressure field rendering | Medium |
| Phase 6 | Full animation suite (base visual language) | High |
| Phase 7-8 | Metrics visualization | Medium |
| Phase 9-10 | Boundary visualization | Low-Medium |
| Phase 11 | Final polish pass | High |

### Reference Implementations

Existing workbenches with animation patterns to follow:

- **balance.html**: Shadow projection rendering, recovery curve animation, status panel updates
- **threebody.html**: Orbital trajectory rendering, signature display, instability warnings
- **public/js/balance-browser.mjs**: Canvas animation loop structure
- **public/js/threebody-browser.mjs**: Real-time physics rendering

### Open Questions for Animation Design

1. Should pressure field use heatmap (red-yellow), topographic (contour lines),
   or vector field (arrows/gradients) style?
2. How to visually distinguish noisy/uncertain pressure from confident readings?
3. Should mine trigger animations be dramatic (emphasize failure) or understated
   (maintain lab aesthetic)?
4. What level of animation juiciness matches the scientific tone without feeling
   sterile?

### Success Criteria

The animation roadmap succeeds when:

1. A first-time viewer can distinguish hidden state from indirect signal within
   10 seconds of page load
2. The pressure field looks like a projection/distortion, not a number reskin
3. Controller actions and boundary failures are legible in replay
4. Animations enhance understanding without distracting from telemetry
5. The workbench feels cohesive with Balance and Three-body aesthetics
