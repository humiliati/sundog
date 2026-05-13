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

Commit and push website tooling/content changes when you want Cloudflare's
Git-connected build to reproduce the site from the repository.
