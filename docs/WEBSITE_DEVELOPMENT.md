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
