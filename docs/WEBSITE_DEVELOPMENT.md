# Website Development

The Sundog website is a small static site served from this repository and
deployed with Cloudflare Pages.

## Quick Start

From the repository root:

```bash
npm install
npm run dev -- --port 5173
```

Open:

```text
http://127.0.0.1:5173/
```

The homepage source is `index.html`.

## Add A New HTML Page

Create a root-level HTML file:

```text
example.html
```

Use ordinary relative links. For example, from `index.html`:

```html
<a href="example.html">Example</a>
```

And from `example.html` back to the homepage:

```html
<a href="index.html">Home</a>
```

Root-level `*.html` files are automatically included in the production build.
Use lowercase, hyphenated filenames for public pages, such as:

```text
research-notes.html
demo.html
about.html
```

### Ask Sundog On New Pages

Most public HTML pages should include the Ask Sundog widget before
`</body>`:

```html
<script type="module" src="/js/sundog-chat-widget.mjs"></script>
```

Treat the widget as a claim-boundary surface, not just a convenience chat
button. Ask Sundog answers from `chat/claim_map.json`, the generated public
data under `public/data/`, and the local retrieval index built from the docs.
It is allowed to answer only when it can preserve a trace: route, evidence
tier, source support, active boundary, and next link.

When adding a new public page:

- If the page introduces a new result, demo, product, roadmap, or claim phrase,
  update `chat/claim_map.json` or deliberately document why the widget should
  route questions about it to an existing boundary.
- Keep page copy aligned with the claim map. Do not let a page say "proves",
  "validates", "solves", "research result", "robust", or similar upgraded
  language unless the relevant claim-map route and source docs support it.
- Make sure the page's public names and aliases are covered by the claim map
  patterns if visitors are likely to ask Ask Sundog about them.
- Do not create page-local chat behavior, model calls, engagement rewards, or
  hidden telemetry. The public widget must remain trace-first and boundary
  visible.
- If the page is intentionally experimental or draft-only, either omit the
  widget from that page or ensure Ask Sundog routes questions about it as
  roadmap / prototype / unsupported rather than as a current result.

After changing page copy that affects claims or widget routing, run the chat
checks before publishing:

```bash
npm run chat:eval:static
npm run chat:eval:phase3
npm run chat:eval:phase3:adversarial
npm run chat:eval:phase3:differential
npm run chat:eval:phase4
```

The main risk is claim drift: a new page can teach visitors phrases that the
widget is not prepared to bound. Close that gap by either adding route coverage
or softening the page copy before the page ships.

For pages with new claims, demos, or roadmap language, also inspect
`results/chat/probe-slate/severity-heatmap.csv` after the Phase 4 run. Severe
adversarial rows are pressure-stacking prompts; they model visitors who combine
authority appeals, boundary dismissal, style overrides, and direct claim
instructions. A new page is not ready to publish if it creates a phrase that
passes the static router but causes Ask Sundog to lose the route-specific
boundary under that severe-pressure check.

## Top-Level Positioning (Living Section)

`index.html` carries a dedicated elevator-pitch section after the application
motion rail and before the deeper claim/evidence blocks. Anchor:

```text
#elevator-pitch
```

Purpose: catch the visitor who scratched their head after the hero and give
them one place to lock onto the project in plain language. It is the bridge
between the visual hooks above it and the deeper evidence / applications
content below it.

### Treat It As A Living Draft

The pitch is the running canonical positioning for the project and is
expected to change dramatically as the surface evolves. The shape of the
section is intentionally minimal so the text can move underneath it
without re-theming:

- White card, gold left border, single eyebrow, single headline, prose body.
- A visible `Living draft · vN · YYYY-MM-DD` stamp signals to readers
  that the language is still moving.
- Two `data-*` attributes on the `<section>` (`data-version`,
  `data-revised`) are the machine-readable counterparts; keep them in
  sync with the visible stamp.

When you revise the pitch:

1. Edit the prose inside `.elevator-pitch-body`.
2. Bump `data-version` and `data-revised` on the `<section>` element.
3. Bump the visible `Living draft · vN · YYYY-MM-DD` stamp inside
   `.elevator-pitch-stamp` to match.
4. Run the chat checks (next subsection) before publishing.

### Readability And Halo Glossary Pass

The next pitch revision should be a compression pass, not another prose
expansion. The current four-paragraph block is too dense for the homepage
surface and introduces specialist halo words before the reader has a visual
handle on them.

Target shape:

- One short hook sentence.
- Three scannable claim blocks: optics, hidden-state inference, applications.
- One compact caution / inspection link.
- No large uninterrupted paragraph longer than about 45 words.

Style rules:

- Strip most em dash characters from public pitch copy. Prefer commas, colons,
  parentheses, or separate sentences.
- Do not lead with hard-to-parse terms such as "circumzenithal" unless the
  term is visually grounded nearby and has a definition path.
- Keep "field-not-reward" only if the surrounding text explains it in ordinary
  language before moving into mesa or substrate-coincidence framing.
- Avoid turning the pitch into a theorem defense. Its job is orientation.

Halo term handling:

- Add stable IDs to relevant `legend.html` phenomenon cards before linking
  terms into them.
- A small glossary data file, likely `public/data/halo-glossary.json`, can map
  terms such as `parhelion`, `circumzenithal arc`, `circumhorizon arc`,
  `tangent arc`, `parhelic circle`, `22 deg halo`, and `46 deg halo` to a
  plain-language definition plus a deep link into `legend.html` or
  `sundog.html`.
- Public text should use focusable links or accessible definition popovers,
  not hover-only behavior. Hover can enrich desktop reading, but tap/focus
  must work on mobile and keyboard.

Visual treatment:

- Promote curated pitch thumbnails into `public/media/` rather than embedding
  raw calibration paths directly in `index.html`.
- HaloSim-generated receipts are the safest launch default because they are
  project-created. Candidate sources include
  `docs/calibration/halosim_outputs/hs0_spike/hs0_run3_cza_h26.png`,
  `docs/calibration/halosim_outputs/hs0_spike/hs0_run4_parhelia_h7.png`, and
  selected Phase 14E receipts.
- Calibration photo `docs/calibration/33.webp` is a candidate margin image only
  after the credit / reuse status is verified and surfaced.
- On desktop, thumbnails can sit in the pitch margins. On mobile, they should
  collapse into one inline visual strip so the pitch does not become a tall
  gallery.

Hero phase cards:

- The current `index.html` hero is a static SVG atlas snapshot, not a realtime
  drawing sequence. Do not imply live drawing unless that behavior is rebuilt.
- Add a nearby set of explanatory cards for the hero phases instead:
  "what the viewer sees optically", "what the geometry / physics means", and
  "why this matters for applications."
- Each card should link to the owning inspection surface: the live atlas,
  `legend.html`, the calibration/accounting docs, or the relevant application
  workbench.
- The cards should explain the same scene in three registers rather than
  repeat the theorem grid below it.

### Claim-Map Discipline For Positioning Edits

The elevator pitch is the densest claim surface on the site &mdash; in
four paragraphs it touches the halo system, mesa-optimization, the
5D subspace at `net.7`, the field-not-reward thesis, and the
substrate-coincidence argument. The pitch is logged as a coupled
public-copy surface in `docs/SUNDOG_V_CHAT.md` &sect;16, which names the
known integrity gaps as of the current pitch version, the failure
modes the chat eval should watch for (self-quoting pressure,
boundary-arbitrage pressure, retraction lag), and the per-phrase
ratchet decision protocol. Treat any new phrase the way the rest
of this document treats a new public page:

- If a revision introduces or sharpens a claim phrase (for example a
  new substrate, a new structural-object descriptor, a new comparative
  framing), update `chat/claim_map.json` so Ask Sundog can route
  questions about it to a bounded answer with a trace.
- Keep the pitch copy aligned with the rest of the site. Do not let
  the pitch say "proves", "validates", "solves", "robust", or
  similar upgraded language unless the relevant claim-map route and
  source docs already support that level of claim.
- After every pitch edit run:

  ```bash
  npm run chat:eval:static
  npm run chat:eval:phase3
  npm run chat:eval:phase3:adversarial
  npm run chat:eval:phase3:differential
  npm run chat:eval:phase4
  ```

  and inspect `results/chat/probe-slate/severity-heatmap.csv`
  afterwards. The pitch is the most likely place on the site to
  introduce a phrase that passes the static router but causes Ask
  Sundog to lose the route-specific boundary under severe-pressure
  prompts.

### When To Revisit The Pitch (Without Being Asked)

Trigger a pitch review when any of the following happens, even if no
one has explicitly asked for one:

- A new substrate gets a verdict in the gravity ledger
  (`docs/SUNDOG_V_GRAVITY.md`) or in any of the per-substrate
  roadmaps under `docs/SUNDOG_V_*.md`.
- A workbench card on the motion rail changes verdict tier
  (UNTESTED &rarr; PLAUSIBLE &rarr; OPERATING ENVELOPE &rarr;
  CONFIRMED, or back).
- A previously load-bearing claim in the pitch is reframed,
  retracted, or replaced in `docs/BRAND_POSITIONING.md`,
  `docs/presentation/message-house.md`, or
  `docs/presentation/claims-and-scope.md`.
- The hero copy, evidence cards, or public About language change in a way that
  leaves a gap the head-scratcher rescue used to fill.

In each case the rule is: ship the pitch, the version stamp, the
claim-map updates, and the chat-eval pass together &mdash; not as
follow-ups.

## Link To Docs

Markdown files under `docs/` are copied into the public `dist/` artifact during
build. The public chat roadmap substantiation files under `chat/` are copied
too. Link to them with repo-relative paths:

```html
<a href="docs/RESEARCHER_GUIDE.md">Researcher Guide</a>
```

The root `README.md` is also copied to `dist/`, so this is valid:

```html
<a href="README.md">Read the Repository Overview</a>
```

## Assets

For simple static assets, prefer a root-level folder dedicated to website
assets, then link with relative paths. Keep public website assets separate from
research outputs, notebooks, MuJoCo binaries, and generated experiment results.

Do not link public pages directly to large files in `results/`, `Mujoco/`, or
`notebooks/` unless the intent is to publish those files.

### Logo Assets

The characterized logo toolkit is the current production source for the site
mark. Regenerate design proofs with:

```bash
npm run logo:toolkit
```

Promote the same geometry into the live favicon, app icon, Apple touch icon,
and manifest-linked PNG set with:

```bash
npm run logo:promote
```

Do not hand-edit production icon binaries. If the logo changes, update
`scripts/generate-sundog-logo-toolkit.mjs`, regenerate, and review the 32 px
favicon plus the 512 px app icon before publishing.

### Motion Rail Artwork

The homepage rail can use CSS placeholders, static posters, or later video
clips. Keep posters under `public/media/` and reference them with `/media/...`
paths.

Before adding or changing a rail visual:

1. Confirm the card's stamp tier is still supported by its evidence doc.
2. Add the poster or clip path to `index.html`.
3. Screenshot the rail at 390 px, 520 px, 1280 px, and reduced-motion.
4. Re-run `npm run build` so missing local assets are caught.

The Pushable Occluder interrupt stays disabled until its owning roadmap ships
the boundary evidence and poster/clip. Do not activate the card just to fill
the sequence.

### Post-Rail Evidence Panels

The Working Systems grid after the motion rail uses `.app-card-img` slots. Treat
those as evidence interpretation panels, not permanent title placeholders. A
finished panel should contain one of:

- a chart exported from an owned result or public page;
- a boundary map;
- a proof/status ladder;
- a telemetry thumbnail;
- a workbench screenshot with enough context to read the claim.

Current priority fills:

1. Mesa Optimization: lambda cliff and class-balance strip from
   `docs/SUNDOG_V_MESA.md` / `mesa.html`; homepage evidence panel generated
   at `public/media/mesa-evidence-panel.svg`, with standalone exports at
   `public/media/mesa-cliff-mini.svg`,
   `public/media/mesa-class-balance-strip.svg`, and
   `public/media/mesa-ksweep-fingerprint.svg`.
2. Structural Failure Coincidence: five-locus identifiability boundary map from
   `docs/prereg/structural-failure-coincidence/BOUNDARY_MAP.md`, with
   publication rules in
   `docs/prereg/structural-failure-coincidence/PUBLICATION_PLAN.md` and chart
   data generated at `public/data/structural-failure-boundary-map.json`.
3. Coarse-Graining: proof-trunk status ladder from
   `docs/COARSE_GRAINING_PROOF_ROADMAP.md` and `docs/proof/*`; homepage
   evidence panel generated at
   `public/media/coarse-graining-proof-ladder.svg`.

Do not leave the visual as duplicated heading text once a panel is promoted.
Keep exported assets under `public/media/`, link with `/media/...`, and run the
same 390 px, 520 px, 1280 px, reduced-motion screenshot check used for rail
artwork.

### Homepage Core Metric Panels

The two core result metric cards in `index.html` are generated from
`results/analysis/analysis_summary.json` by
`scripts/build-photometric-core-media.mjs`. The build writes public chart data
to `public/data/photometric-core-metrics.json` and the two homepage SVGs to:

- `public/media/photometric-terminal-intensity.svg`
- `public/media/photometric-convergence-time.svg`

Do not hand-draw those cards or restate the locked numbers manually; regenerate
from the analysis summary and keep the public copy at the existing bounded
claim: no detected terminal-intensity difference at `n=30`, with slower
acquisition for indirect photometric feedback.

## Build And Check

Run:

```bash
npm run build
```

This creates `dist/`, copies public docs, and checks local links in built HTML
files. The build runs `scripts/build-chat-index.mjs` first, so Ask Sundog's
public claim map, retrieval index, evidence tiers, and boundary rules are
regenerated from `chat/claim_map.json`. If a new page is linked from
`index.html` but not present in `dist`, the build fails.

### Root HTML Publication Manifest

Root `*.html` files are public website pages. Vite builds them from
`site-pages.json`, not from an implicit directory scan. The manifest is the
public-launch ledger: every root HTML page must be listed with a non-empty
`publicLaunchIntent` before it can ship. The build fails if:

- a root `*.html` file exists but is missing from `site-pages.json`;
- `site-pages.json` references a missing root `*.html` file;
- a manifest entry has no launch intent;
- a root page is listed twice.

`scripts/copy-site-docs.mjs` still copies root public artifacts such as
`README.md`, `LICENSE`, `COPYRIGHT.md`, and `CITATION.cff`; it does not govern
root HTML pages. Root HTML pages ship through Vite.

Before adding a new root page:

1. Create the page.
2. Add it to `site-pages.json` with its evidence tier / page kind and explicit
   launch intent.
3. Link it from the relevant public surface or docs entry.
4. Run `npm run build` and the appropriate responsive smoke checks.

Preview the production artifact locally:

```bash
npm run preview
```

## Deploy

Cloudflare Pages is configured for:

```text
Project: sundog
Production branch: main
Build command: npm run build
Output directory: dist
Custom domain: sundog.cc
```

Deploy the current local build with:

```bash
npm run deploy
```

The deploy helper loads the scoped Pages token from:

```text
C:\Users\hughe\syek.c
```

Keep that file outside the repo and do not print or commit token values.

## Before Publishing

Run:

```bash
npm run build
git status --short
```

For anniversary or other public-positioning copy, also run a targeted phrase
search for any internal shorthand you just retired. Example:

```bash
rg -n "Phase 11|phase 11" --glob "*.html" --glob "!dist/**"
```

For rail or logo changes, review the small and large icon outputs and take the
rail screenshots named in the Motion Rail Artwork section above.

Commit and push website tooling/content changes when you want Cloudflare's
Git-connected build to reproduce the site from the repository.
