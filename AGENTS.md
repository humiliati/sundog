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

`npm run dev` serves the source site locally. `npm run build` creates `dist/`
from the root `index.html`, then copies `README.md` and `docs/**` into `dist`
so public internal documentation links do not break. The build also checks that
local links from `dist/index.html` resolve inside `dist`.

`npm run deploy` rebuilds first, then runs:

```bash
node scripts/deploy-pages.mjs
```

The deploy helper loads `CLOUDFLARE_TOKEN_SUNDOG_PAGES_DEPLOY` from
`C:\Users\hughe\syek.c` and passes it to Wrangler as `CLOUDFLARE_API_TOKEN`.
If `CLOUDFLARE_API_TOKEN` is already set in the environment, that wins.

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
