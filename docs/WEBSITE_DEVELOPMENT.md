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
