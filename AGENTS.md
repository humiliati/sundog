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

## Website SEO and Social Readiness

- New entries in `site-pages.json` must add or update the matching row in
  `docs/site/SEO_AND_SOCIAL_READINESS_ROADMAP.md` and clear Bucket 1 before the
  entry's `publicLaunchIntent` is treated as satisfied. For Class A/B
  public-share pages, that means OG/Twitter metadata, a designed 1200x630
  `og:image`, JSON-LD, tuned title/description, an internal link path, sitemap
  coverage, and a post-deploy LinkedIn/Twitter validator pass before external
  sharing.

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

## Workflow — Running Experiments, Smokes, and Measurements

**The ~10-minute rule.** Run an experiment, smoke test, or timing
measurement inline only if it completes in **under ~10 minutes**
wall-clock. For anything expected to exceed that, **do not run it** —
instead **stage the exact PowerShell command(s)** for the operator,
with a wall-clock estimate and the decision/output it feeds.

Why: the project machine is CPU-only (no GPU; `torch` is
`*+cpu`), and several pipelines (mesa Large-tier PPO, HaloSim long
sweeps, full probe×intervention cross-products) run hours-to-days. An
agent should *measure* with cheap capped probes, *extrapolate*, and
hand the operator the long runs rather than block on them.

How to apply:

- **Prefer a capped probe over the full run.** Measure a per-unit rate
  (s/update, env-steps/sec, frames/sec) from a short capped run, then
  extrapolate the full cost. Capped probes themselves must obey the
  ~10-min rule.
- **Staged commands are a deliverable, not a deferral.** Write them
  into the relevant spec/results doc as runnable PowerShell with: the
  exact invocation (module form where needed, e.g. `python -m
  training.mesa.train_ppo …`), a wall-clock estimate, resume-safety
  notes, and the read-back path + the branch each outcome selects.
  Canonical example: `docs/mesa/PHASE7_SPEC.md` §14.6.
- **Record measured rates in the doc**, not just the conclusion, so a
  later agent can re-extrapolate without re-measuring.
- **Pre-register the negative.** When a measurement gates a go/no-go,
  state the threshold and the branch each result selects *before*
  running it (mesa pre-registered-negative discipline).
- Domain specs may pin their own threshold (e.g.
  `PHASE7_SPEC.md` §11 used 30 min for cheap missing-cell fills); the
  general default when a doc is silent is **~10 minutes**.
- Clean up throwaway probe output (`results/**/_*` scratch dirs) once
  the rate is recorded.

### Cloud-agent run offload (experimental) — threebody:phaseNN

A second lane of the same long-run principle: instead of (or alongside)
staging for the local operator, hand the gates + smoke to a GitHub cloud
agent and have it report wall-clock from each run's
`manifest.json` `startedAt`/`completedAt`. Brief + measured local baselines
+ the extrapolated full-lock cost live in **`copilot_test_readme.md`** (repo
root); the cloud agent appends its sandbox timings + environment there.

Two caveats are load-bearing and documented in that file: (1) the hard-void
gates demand *bit-for-bit* reproduction, but `Math.sqrt/sin/cos/log` can
differ ~1 ULP across CPU/libm/Node and RK4 amplifies it in the chaotic
regime — a cloud gate "deviation" may be platform-fp, not a code regression,
so triage against the committed local `results/` artifacts before declaring
any phase void; (2) the harness is strictly single-threaded
(`envelopeCases` is a sequential loop, no workers), so a many-core sandbox
gives *offload*, not speedup, unless the harness is sharded first. The full
lock stays operator-gated by the locked spec — the cloud experiment measures
the gates + smoke only and never unblocks or interprets the full lock.

First run (2026-05-16, Linux x86_64 / AMD EPYC 7763 4-core / Node v20.20.2):
**feasible** — both hard-void gates reproduced **bit-for-bit** (caveat 1 did
not bite on this stack; triage rule retained for any future deviation), and
the harness is pure Node (no build step). But raw compute was **≈ parity with
high variance** (phase13 ~45 min vs 48.7 local; phase14 ~10 min vs 4.9 local —
*slower*) on a contended 4-core host: the win is offload + cross-platform
reproducibility, not speed. The chat-pool agent hit a **~1 h session cap** and
timed out after phase14 *before reporting* — so the operative rule is:
**direct the cloud agent to persist/report results incrementally per phase,
before expiry, never batched to the end**; budget ~1 h; prefer an
issue-assigned Copilot coding agent (possibly a longer-lived pool) but still
report incrementally. Full detail + per-run log: `copilot_test_readme.md`.

Run 2 (2026-05-16, issue-assigned agent) validated the model: it worked
silently ~1 h then delivered a **PR** (durable — dodges the chat-pool
timeout entirely; prefer this over the chat pool). Gates reproduced
bit-for-bit a third time (caveat 1 retired in practice). But it
runtime-gate-skipped phase14 + the smoke on a ~30-min threshold — so a
measurement issue MUST explicitly authorize the specific phases
(`phase13` + `phase14` + `phase15:smoke`), overriding the generic ~10/30-min
rule for that issue, and instruct per-phase incremental commits to the PR
branch. Full lock stays out of scope and operator-gated.

Run 3 (2026-05-16, issue-assigned, **Intel Xeon 8370C** — a different ISA from
Runs 1/2 on AMD EPYC) reproduced phase13 bit-for-bit a 4th time: caveat 1 is
retired across ISAs. Structural limit confirmed: phase13 alone (~46 min)
consumes a ~1 h session, so Runs 2 and 3 each landed phase13 only even when
the issue authorized all three phases. Authorizing phases cannot create
session time. **Cloud measurement issues must therefore drop phase13
(measured 4×) and run `phase14` and `phase15:smoke` as separate, each-under-1h
issues with per-phase commits** — see `copilot_test_readme.md` ▸ "Revised
next-issue guidance".

Runs 4–5 (2026-05-16, EPYC 9V74) closed the gate question and escalated the
blocker: phase14 reproduced bit-for-bit on the assigned pool in 5m40s (gates
are now **done** on the agent path — phase13 5×, phase14 3×, stop issuing
them). But Run 5 proved `phase15:smoke` **does not fit a ~1 h interactive
session even as the sole command** (expired at 87/144 trials, no manifest, no
`richardson-order-map.csv`; ~70–80 min on the contended 4-core box).
Conclusion: interactive Copilot-agent offload is structurally unfit for the
smoke and the ~75 h full lock — **move those to a long-budget runner** (GitHub
Actions `workflow_dispatch`, ~6 h job limit; or self-hosted) rather than an
agent session. Full lock + `T_window` stay operator-gated by the locked spec
regardless of where the compute runs. See `copilot_test_readme.md` ▸
"Synthesis (current — after Run 5) + revised plan".

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
| Geometry confirmation / reproduction receipt | **B&W** ("Grey shades on white") | **~1M** — clean, reliable loci. **~300k b&w is NOT reliable** (Monte-Carlo noise; empirically found in the Phase 14E receipt pass — sparse, asymmetric features). Do not drop below ~1M for any geometry/receipt work. |
| Geometry confirmation | **Colour** | **~3M** — colour equivalent of the ~1M B&W pass |
| Spectacular thumbnails / logo candidates | Colour | **~10M** — press-grade; tune up/down by the contrast wanted (more rays = smoother gradients, fewer = punchier highlights) |

Below ~1M rays **both** B&W and colour renders show Monte-Carlo
asymmetry/sparseness — never measure geometry or take a receipt off a
sub-1M render. Use the **B&W ~1M** pass for geometry/receipts, **~3M
colour** to confirm, and reserve **~10M colour** for beauty / logo
exploration.

**Isolating one feature.** When a multi-block recipe renders unrelated
halos that drown the target, disable the non-target crystal blocks
("Clr" them or pre-edit the `.sim` block to the `no selection` pattern,
as `halosim_p2_h18.6_columnonly.sim` keeps only column+random) and keep
the target block (+ optionally `random.xng` for the 22° scale
reference). See `docs/calibration/HALOSIM_VALIDATION_PROTOCOL.md`.

### Long-run sweeps (HS-1 / HS-2)

- Generate per-frame sims: `python scripts/halosim_gen_frames.py
  --template <sim> --out docs/calibration/halosim_outputs/hs_frames
  --start 0 --stop 60 --step 2 --rays <N>`. Fisheye templates are
  preferred (only sun altitude changes; sidesteps the HS-3 auto-zoom).
- Drive the render: `python scripts/halosim_run_sweep.py calibrate`
  once (reads true-1920×1080 Reset/Start coords into
  `scripts/halosim_sweep_config.json`), then `… run --frames-dir
  docs/calibration/halosim_outputs/hs_frames --resume`. **HaloSim must
  be foreground.** This is a **standalone local controller** (pyautogui
  + `autosave.bmp` mtime-poll) — it is NOT run through the agent/MCP,
  so a multi-hour sweep is fine. It backs up/restores `Startup.sim`,
  is resumable, and stall-aborts if the window moved (recalibrate).
- Output goes to the **gitignored**
  `docs/calibration/halosim_outputs/_staging/<runtag>/` (BMP+PNG+TSV
  log; pass `--no-bmp` to keep only ~100 KB PNGs over a 1000-frame
  run). Never commit staging — only the final HS-5 clip + curated
  stills are tracked. A `*.bmp` gitignore rule blocks stray BMP bloat.
- PyMacroRecord (`C:\Users\hughe\PyMacroRecord`) was evaluated as the
  driver and rejected (fixed-delay replay desyncs on variable render
  time); it is only a no-dependency fallback for a strictly uniform
  ray-count sweep.

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
