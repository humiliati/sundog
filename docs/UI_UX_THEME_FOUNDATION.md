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

---

## Paper-Inspired Theme Extension

Date: 2026-05-08
Status: expansion of typography overhaul to include physical paper metaphors

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

#### Hand-Written Fonts (Sticky Notes & Paper)

For informal, note-style content on sticky notes and paper cards:

**Recommended Google Fonts:**
- `'Indie Flower'` — casual handwriting, high readability
- `'Patrick Hand'` — clean hand-drawn style
- `'Caveat'` — natural handwriting with personality
- `'Shadows Into Light'` — friendly and legible

**Usage:**
```css
--sd-font-handwritten: 'Indie Flower', 'Patrick Hand', cursive;

.sd-sticky-note,
.sd-paper-annotation,
.sd-handwritten {
  font-family: var(--sd-font-handwritten);
  font-size: 1.1rem;
  line-height: 1.6;
}
```

**Best practices:**
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

1. Add new CSS custom properties to `public/css/sundog-theme.css`
2. Implement component classes (`.sd-paper-card`, `.sd-sticky-note`, etc.)
3. Create a demo page or section showcasing each effect
4. Apply selectively to existing components in `index.html`
5. Test across browsers and devices
6. Document usage guidelines for future page development
7. Consider creating a visual component library or style guide page

### References

- EyesOnly CSS themes: `C:\Users\hughe\Dev\EyesOnly\public\css\themes.css`
- EyesOnly portfolio direction: `C:\Users\hughe\Dev\EyesOnly\docs\PORTFOLIO_TRANSITION_ROADMAP.md`
- EyesOnly hybrid layout spec: `C:\Users\hughe\Dev\EyesOnly\docs\HYBRID-LAYOUT-SPEC.md`
- Inspiration: flapsandseals.com (booking/partners pages)
- BoxForge primitives: Dungeon Gleaner `docs/BOXFORGE_README.md`
