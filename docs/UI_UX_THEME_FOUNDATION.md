# Sundog UI/UX Theme Foundation

Date: 2026-05-07
Status: starter foundation for the coming visual overhaul

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
- Current `index.html`, `origin.html`, and `applications-gallery.html`

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

1. Move repeated header/footer CSS out of each HTML file into
   `public/css/sundog-theme.css`.
2. Split the shared stylesheet into layers when it grows:
   `tokens`, `base`, `components`, `pages`, `experiments`.
3. Replace inline card/button styles with shared component classes.
4. Create one canonical hero graph component:
   optical halo plus theorem graph plus BoxForge phase animation discipline.
5. Decide whether BoxForge exports live as hand-curated CSS snippets or as
   generated files from `.boxforge.json` templates.
6. When the canonical topography/typography direction lands, map it onto the
   existing `--sd-font-*` tokens first, then page sections second.

## Guardrails

- Keep the site readable and professional before making it expressive.
- Use motion to clarify signal, phase, or interaction state.
- Avoid decorative effects that do not carry the Sundog metaphor.
- Do not publish repo root content. Public site output remains `dist/`.
- Keep EyesOnly and Dungeon Gleaner as portfolio/application references, not as
  claims that broaden the core scientific result.
