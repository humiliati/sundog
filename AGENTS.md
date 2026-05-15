# Sundog Agent Notes

This repository is both a research artifact and the source for the public
Sundog website at `sundog.cc`.

## Website Tooling

- Use Node/npm from the repo root for website work.
- The root `index.html` is the website entry point.
- Vite is installed locally as a dev dependency. Do not rely on a global Vite
  install.
- Wrangler is installed locally as a dev dependency for Cloudflare Pages deploys.
  Do not rely on a global Wrangler install.
- The accidental scaffold lives at `C:\Users\hughe\Dev\sundog-vite`; the real
  website source is this repo root, `C:\Users\hughe\Dev\sundog`.

## Common Commands

```bash
npm install
npm run dev -- --port 5173
npm run build
npm run preview
npm run deploy
npm run cf:auth
npm run cf:inspect
npm run cf:pages
npm run cf:tokens
npm run cf:domain
```

`npm run dev` serves the source site locally. `npm run build` regenerates Ask
Sundog's public data from `chat/claim_map.json`, creates `dist/` from the root
`index.html`, then copies `README.md`, `docs/**`, and the public chat
substantiation artifacts into `dist` so public internal documentation links do
not break. The build also checks that local links from `dist/index.html`
resolve inside `dist`.

`npm run deploy` rebuilds first, then runs:

```bash
node scripts/deploy-pages.mjs
```

The deploy helper loads `CLOUDFLARE_TOKEN_SUNDOG_PAGES_DEPLOY` from
`C:\Users\hughe\syek.c` and passes it to Wrangler as `CLOUDFLARE_API_TOKEN`.
If `CLOUDFLARE_API_TOKEN` is already set in the environment, that wins.

## HaloSim Halo Rendering (cinematic + geometry confirmation)

HaloSim3 is a Monte Carlo halo ray-tracer used two ways in this repo:
**validation** (is the atlas geometry right?) and **generation**
(labelled, sun-altitude-swept renders for hero / press / logo).

- Binary: `C:\Users\hughe\HalSim361.exe` — a 2004 GUI app with **no CLI
  and no headless mode**. The `.sim` / `.xsh` / `.xng` / `.xmt` asset
  library and the `h*.txt` help corpus are in `C:\Users\hughe\`.
- Canonical docs (read before touching HaloSim):
  - `docs/SUNDOG_HALOSIM_CINEMATIC_SIDECAR.md` — the HS-0…HS-7 pipeline.
    **HS-0 is proven**; its mechanism, timing table, and receipts
    (`docs/calibration/halosim_outputs/hs0_spike/`) live there.
  - `docs/calibration/HALOSIM_VALIDATION_PROTOCOL.md` — the
    validation-direction procedure (22°-halo scale-lock, feature-locus
    search, auto-zoom gotcha).

### Proven zero-click mechanism (HS-0)

No CLI exists. Drive it file-first, GUI-minimal:

1. Copy the target `.sim` over `C:\Users\hughe\Startup.sim` (HaloSim's
   auto-loaded preference file). **Back it up first and restore it
   afterward** — it is the user's real preference file.
2. Launch HaloSim once (auto-loads `Startup.sim`); for each subsequent
   frame click **Reset** to reload it.
3. Click **Start** — two fixed-position clicks per frame, no dialogs.
4. Harvest `C:\Users\hughe\autosave.bmp` (auto-written on completion;
   needs Tools→Options ▸ *Autosave simulation* checked). It is one file
   overwritten every run — rename it before the next Start.
5. **Completion is poll-based, never a fixed sleep.** `autosave.bmp`
   mtime advances only after the on-screen Run Status reads `100%`.

GUI driving uses the computer-use MCP (HaloSim resolves as app name
`HaloSim`, full tier). The window is unavoidably on-screen for the whole
batch — unattended (no human clicks) but not background.

### Ray-count guidance

Pick ray count by purpose. Colour costs roughly **10× the rays of B&W**
for the same signal-to-noise:

| purpose | mode | rays |
| --- | --- | --- |
| Geometry confirmation | **B&W** (black dots / grey-on-white) | **~300k** — fast (seconds); cleanest loci for overlay & scale-lock |
| Geometry confirmation | **Colour** | **~3M** — colour equivalent of the 300k B&W pass |
| Spectacular thumbnails / logo candidates | Colour | **~10M** — press-grade; tune up/down by the contrast wanted (more rays = smoother gradients, fewer = punchier highlights) |

Below ~1M rays colour renders show Monte-Carlo asymmetry — never measure
geometry off a noisy colour render. Use the **B&W ~300k** pass for
geometry, **~3M colour** to confirm, and reserve **~10M colour** for
beauty / logo exploration.

## Cloudflare Credentials

There is a local legacy Cloudflare Global API Key file at:

```text
C:\Users\hughe\yek eralfduolc.txt
```

Treat that file as a break-glass owner credential. Do not print it, commit it,
copy it into repo files, or pass it to ordinary session agents. Legacy global
key authentication also requires the Cloudflare account email. The local auth
checker can read both values from the file, or from `CLOUDFLARE_EMAIL` plus
`CLOUDFLARE_API_KEY`.

For normal work, use scoped API tokens:

- `sundog-pages-deploy`: Cloudflare Pages write access for deploying `dist/`.
- `sundog-workers-edit`: Workers script edit access for Worker projects.
- `sundog-dns-edit`: DNS write access for the `sundog.cc` zone only.
- `sundog-session-readonly`: read-only zone/account visibility for exploratory
  agent sessions.
- `sundog-token-factory`: short-lived token created from Cloudflare's "Create
  additional tokens" template. Use it only to mint or rotate scoped tokens, and
  prefer IP and TTL restrictions.

Check the current local Cloudflare auth shape with:

```bash
npm run cf:auth
npm run cf:inspect
```

That command verifies either `CLOUDFLARE_API_TOKEN`, or the legacy global
credential pair, without printing secrets.

`npm run cf:inspect` shows non-secret Cloudflare account shape. As of the last
check, the local legacy credential verifies, `sundog.cc` is visible as a zone,
token permission groups are readable, and there are no existing Pages projects
in the visible account.

`npm run cf:pages` ensures the `sundog` Pages project exists and is connected to
`humiliati/sundog` on `main` with build command `npm run build` and output
directory `dist`. It writes non-repo Cloudflare IDs/project metadata to:

```text
C:\Users\hughe\syek.c
```

Keep `C:\Users\hughe\syek.c` outside the repo. It may later hold scoped token
values, which are shown only once by Cloudflare.

`npm run cf:tokens` creates scoped Cloudflare API tokens for Pages deploy,
Workers editing, DNS editing, and readonly session-agent inspection, then writes
their token IDs and one-time token values only to `C:\Users\hughe\syek.c`.
Prefer those scoped tokens over the legacy global key for ordinary automation.

`npm run cf:domain` ensures `sundog.cc` is attached as the apex custom domain
for the `sundog` Pages project. It also creates the proxied apex CNAME to the
project's `*.pages.dev` subdomain if Cloudflare has not already created it.

## Publishing Shape

Publish only `dist/`. Do not deploy the repository root: it contains research
code, notebooks, results, MuJoCo binaries, and other files that should not be
part of the public static site artifact.

Cloudflare DNS for `sundog.cc` currently points at Cloudflare nameservers, so
Cloudflare Pages plus Wrangler is the preferred deploy path unless the owner
chooses a different host.
