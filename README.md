# Sundog

**Copyright © 2026 Stellar Aqua LLC. Licensed under the MIT License.**

Sundog Research Lab is an independent applied research program sponsored by Stellar Aqua LLC.

Sundog-authored source and associated documentation are published under the
[MIT License](LICENSE). For citation metadata, see [CITATION.cff](CITATION.cff).

### Pending ownership transfer to a public-interest entity

The lab is in process of forming a US public-interest research entity
(501(c)(3) target). The working plan is for copyright and stewardship of
Sundog-authored material to transfer from Stellar Aqua LLC to the new
nonprofit once that entity is incorporated, a board with an independent
majority is seated, and a fiscal-sponsor or direct-incorporation path
has been chosen. The MIT licence on existing material is not affected by
the transfer.

The formation plan, governance policy, fiscal-sponsor evaluation,
contributor licence template, and the *Conscium Initiative* framework
the lab develops and publishes are working drafts in
[`docs/501c3/`](docs/501c3/) — start with the
[Initiative Index](docs/501c3/INITIATIVE_INDEX.md). **Nothing in that
folder is yet legally operative.** The transfer is the *goal* of the
plan, not a current fact, and outside contributors should treat the
governance and claims-review policy as a description of the regime the
founders are committed to building, not a regime that already binds the
lab today.

If you are a potential collaborator, sponsor, fiscal-sponsor candidate,
or contributor who needs the entity's legal status today,
[`docs/LEGAL_STANDING.md`](docs/LEGAL_STANDING.md) is the current
factual ground; [`docs/501c3/SUNDOG_FOUNDING_PLAN_v0.1.md`](docs/501c3/SUNDOG_FOUNDING_PLAN_v0.1.md)
is the bridge document between current state and target state.

---

Sundog is a research program for *traceable* agent policy under partial
observability. A geometrically-constituted policy is accountable in ways a
reward-maximizing policy may not be: each decision carries a cheap ledger
packet, and out-of-bounds escapes are observable and refused. The repo should
be read as a research artifact, not a product monorepo.

## What This Is (for reviewers landing from a GitHub link)

The most-visible artifact in this repo is the optics tooling — the
parhelion atlas, the cap-set workbench, the `h(x)` parhelion-offset
inverse — because that's where the geometry was first found, in a
photometric mirror-alignment MuJoCo experiment. The bigger picture is
that the same discipline (indirect signal → trace → transformation →
bounded action with a named failure boundary) is now carried into:

- **Control workbenches**: Balance (shadow cart-pole), Three-Body (local
  pocket survival), Pressure Mines (hidden-hazard navigation) — see
  [Evidence Snapshot](#current-evidence-snapshot).
- **A measured chat experiment**: [sundog.cc/chat](https://sundog.cc/chat) —
  a trace-conditioned browser assistant tested for claim-boundary
  discipline. Latest result: **0 unsafe-accepts across 5,670 trials**
  spanning six model implementations across four training lineages
  (deterministic compositor + OpenAI + Anthropic + Meta Llama at two sizes
  + Alibaba Qwen). Rudimentary, deliberately so. See
  [`docs/SUNDOG_V_CHAT.md`](docs/SUNDOG_V_CHAT.md).
- **Discrete-geometry workbenches**: an [interactive cap-set primer](https://sundog.cc/capset)
  staged as a hands-on lens for the 2026 OpenAI unit-distance disproof.
  See [`docs/SUNDOG_V_CAPSET.md`](docs/SUNDOG_V_CAPSET.md).
- **Ledger documents** under `docs/SUNDOG_V_*.md` that stage every public
  claim with an explicit falsifier, evidence tier, and named failure
  surface.

## What This Is Not

- **Not an inference API or chat-completions service.** The chat widget
  at [sundog.cc/chat](https://sundog.cc/chat) is a measured *experiment*
  in claim-boundary discipline, not a product offering.
- **Not a universal alignment theorem.** The Sundog Alignment Theorem is
  staged in research-grade pieces with explicit failure boundaries. The
  current paper-grade claim is narrow (photometric mirror-alignment in a
  tested MuJoCo setting); see [Research Claim](#research-claim) below.
- **Not an optics paper despite the optics-heavy surface area.** Optics
  is the worked example; agent traceability is the through-line. If you
  came in via the cap-set link, start with
  [`docs/SUNDOG_V_CAPSET.md`](docs/SUNDOG_V_CAPSET.md) for the
  geometry-side reading.

## Research Claim

The current narrow claim is:

> A controller with no Cartesian access to a target can align a mirrored
> end-effector using only sparse photometric feedback, reaching terminal
> accuracy statistically indistinguishable from a target-aware analytic
> baseline in the tested MuJoCo setting.

This is intentionally smaller than the original theorem language. The older
Sundog framing proposed that alignment can emerge from indirect interaction
among light, shadow, torque, and environmental structure. The present research
layer asks what can be defended under scientific review: measurable tasks,
matched baselines, reproducible results, stress tests, and explicit failure
boundaries.

The evidence posture has strengthened beyond a single demo. The core
paper-grade claim remains the photometric mirror-alignment result, while the
Three-Body, Balance, and Pressure Mines workbenches now sit in a separate
**Operating-Envelope Study** tier: bounded sweeps with baselines, confirmed
positive pockets, and named regions where the method should not be used.

## Where To Start

Sorted by reviewer landing intent.

**If you're a technical reviewer trying to figure out what this is:**

- [Researcher guide](docs/RESEARCHER_GUIDE.md) — the shortest path
  through the repo for a skeptical reader.
- [Scientific criteria](docs/SCIENTIFIC_CRITERIA.md) — what's testable
  vs. what isn't.
- [Paper draft](docs/PAPER_v1_draft.md) — the photometric
  mirror-alignment writeup.

**If you landed via the cap-set / unit-distance link:**

- [`docs/SUNDOG_V_CAPSET.md`](docs/SUNDOG_V_CAPSET.md) — the cap-set
  ledger; couples Sundog's apparatus to the 2026 OpenAI unit-distance
  disproof.
- [sundog.cc/capset](https://sundog.cc/capset) — the interactive cap-set
  workbench.
- [sundog.cc/geometry](https://sundog.cc/geometry) — geometry hub
  (cap-set, halo, h(x), and bounded-apparatus notes).

**If you want to evaluate the agent-traceability claim directly:**

- [sundog.cc/chat](https://sundog.cc/chat) — measured chat experiment;
  0 unsafe-accepts across 5,670 trials. The result write-up lives in
  [`docs/SUNDOG_V_CHAT.md`](docs/SUNDOG_V_CHAT.md).
- [`docs/SUNDOG_V_MESA.md`](docs/SUNDOG_V_MESA.md) — mesa-optimization
  trap: in-vitro operating envelope with a sharp cliff at λ ≈ 0.953.
- [sundog.cc/structural-failure](https://sundog.cc/structural-failure) —
  pre-registered five-locus boundary map; Cut 2 separability held, Cut 3
  still open.

**If you want the working surfaces:**

- [Applications map](docs/APPLICATIONS.md) — how Three-Body, Balance,
  Pressure Mines, EyesOnly, Dungeon Gleaner, and Money Bags express
  Sundog ideas in test surfaces and product systems.
- [sundog.cc/h-of-x](https://sundog.cc/h-of-x) — math workbench for the
  parhelion-offset inverse `cos(h) = R₂₂ / α₀`.
- [Standalone app roadmap](docs/STANDALONE_APP_ROADMAP.md) — plan for a
  double-click observer app.

**Project housekeeping:**

- [Documentation index](docs/README.md) — local map of all research-facing
  docs.
- [Website development guide](docs/WEBSITE_DEVELOPMENT.md) — editing the
  public site, deploying through Cloudflare Pages.
- [`docs/SEO_AND_SOCIAL_READINESS_ROADMAP.md`](docs/SEO_AND_SOCIAL_READINESS_ROADMAP.md)
  — per-page OG/Twitter/JSON-LD matrix and Bucket 2 follow-ups.
- [Paper outline and review notes](docs/PAPER_OUTLINE_v0.md) — venue
  framing, reviewer risks, stress-test interpretation.

## Repository Map

| Path | Purpose |
| --- | --- |
| `agents/` | Photometric controller and baselines. |
| `env_v2.py`, `optics.py` | MuJoCo environment wrapper and geometric optics model. |
| `experiments/` | Reproducible experiment, analysis, and stress-test scripts. |
| `results/` | Saved run artifacts, statistical summaries, and generated plots. |
| `docs/` | Research landing docs, paper drafts, runners, and Phase-2 design notes. |
| `index.html`, root `*.html` | Public website pages built by Vite and deployed through Cloudflare Pages. |
| `scripts/` | Website build/deploy helpers, Cloudflare inspection scripts, and link checks. |
| `runners/` | Turn-envelope runner framework used with EyesOnly / Gone Rogue. |
| `SundogMujoco2.0/` | Earlier leisure-environment artifact and historical context. |
| `sundog_alignment_theorem_final_fixed/`, `notebooks/` | Original theorem text and legacy paper materials. |

## Current Evidence Snapshot

Sundog currently has two evidence layers:

- **Research result:** photometric mirror alignment in MuJoCo, with matched
  baselines and stress tests.
- **Operating-envelope studies:** Three-Body, Balance, and Pressure Mines,
  each reporting a bounded positive pocket and an explicit failure or
  degradation boundary.

The headline experiment uses 30 matched MuJoCo scenes. The photometric
controller observes detector intensities and proprioception only; baselines
use either target-aware oracle geometry, noisy oracle geometry, or random
actions.

| Condition | Terminal target intensity | Median time-to-0.9 |
| --- | ---: | ---: |
| photometric | 0.945 | 188 steps |
| doa_direct | 0.936 | 11.5 steps |
| doa_noisy | 0.911 | 14 steps |
| random | approximately 0 | censored at 500 |

The photometric-vs-oracle comparison reports `U = 526`, `p = 0.264` on
terminal target intensity. The cost of indirect feedback is convergence time,
not terminal accuracy, inside the tested operating envelope.

The known failure boundary is tight joint limits. At a 1.0 rad symmetric joint
limit, the photometric controller collapses while the oracle degrades more
gracefully because it can project an externally known optimum into the
reachable workspace.

Current operating-envelope studies:

| Workbench | Current bounded result | Boundary |
| --- | --- | --- |
| Three-Body Dynamics | Guarded accelerometer-proxy TRACK improves survival over passive and naive local baselines across a mapped high-velocity near-escape pocket through a 16-second tested horizon. | The low-velocity `velocityScale=0.95` boundary, especially equal-mass cells, still exposes controller harms. |
| Sundog Balance | Shadow-derived cart-pole control beats naive shadow-centering on 28/28 diagnostic-positive cells in the repaired Phase 10 verdict. | Overhead-light and high-delay cells remain degradation boundaries. |
| Pressure Mines | In the density 0.16 / pressure-noise 2.0 / dropout 0.2 pocket, pressure-derived Sundog variants improve budget-adjusted safe-tile progress before mine trigger. | The paired density 0.22 / pressure-noise 1.0 / dropout 0.35 region is a published failure case. |

## Reproducing The Experiment

From the repository parent directory, make sure this repo is importable as the
`sundog` package. Then run:

```bash
python -m sundog.experiments.run_baseline_comparison
python -m sundog.experiments.analysis
```

The expected artifacts are written under `results/`, including
`results/run_summary.json`, `results/analysis/analysis_summary.json`, and
plots under `results/analysis/`.

Stress tests are summarized in:

```text
results/stress_tests/stress_summary.csv
results/stress_tests/<stressor>/sweep_summary.json
results/stress_tests/<stressor>/stress_curve.png
```

## Website And Deployment

The public site is served from `index.html` and any other root-level `*.html`
pages. It builds with Vite and deploys to Cloudflare Pages.

```bash
npm install
npm run dev -- --port 5173
npm run build
npm run deploy
```

Cloudflare Pages is configured for the `sundog` project, production branch
`main`, build command `npm run build`, output directory `dist`, and custom
domain `sundog.cc`. The deploy helper reads local scoped Cloudflare material
from `C:\Users\hughe\syek.c`; keep that file outside the repo.

See [Website development](docs/WEBSITE_DEVELOPMENT.md) before adding public
pages or changing deployment behavior.

## Related Sundog Applications

**Discrete-geometry / number-theory surface**

- [Cap-set workbench](capset.html): hands-on primer for the 2016
  Croot–Lev–Pach / Ellenberg–Gijswijt polynomial-method bound, staged
  as the historical precedent for the 2026 OpenAI unit-distance
  disproof of an Erdős conjecture. The companion ledger
  ([`docs/SUNDOG_V_CAPSET.md`](docs/SUNDOG_V_CAPSET.md)) couples
  Sundog's evaluator apparatus to AI-produced mathematics — the
  evaluator front is defensible now; the substrate-analogue horizon
  awaits external review.

**Agent-traceability surface**

- [Ask Sundog](chat.html): trace-conditioned browser chat assistant
  tested against adversarial pressure. Current state:
  [zero unsafe-accepts across 5,670 trials](docs/SUNDOG_V_CHAT.md)
  spanning six model implementations across four training lineages.
  The cheap ledger packet attached to each completion is the
  load-bearing artifact, not the chat UI.

**Control workbenches**

- [Three-body dynamics workbench](threebody.html): local experiment
  surface whose Phase 13 result maps a bounded high-velocity
  near-escape pocket where guarded accelerometer-proxy TRACK control
  improves survival through a 16-second tested horizon over passive
  and naive local baselines, while the low-velocity/equal-mass
  boundary remains explicit.
- [Sundog Balance workbench](balance.html): cart-pole surface where
  the controller is denied pole angle and acts from cast-shadow
  geometry. Phase 10 confirmed the diagnostic-positive envelope while
  preserving overhead-light and high-delay degradation boundaries.
- [Sundog Pressure Mines](mines.html): game-native hidden-hazard
  workbench where noisy pressure values support safer
  budget-adjusted progress in one named pocket, published beside a
  matched failure region.

**Product / instrumentation surfaces**

- [EyesOnly](https://github.com/humiliati/EyesOnly): live spy-game
  platform and Gone Rogue procedural roguelike integration.
- [Dungeon Gleaner / DCgamejam2026](https://github.com/humiliati/DCgamejam2026):
  Dungeon Crawler Game Jam 2026 entry whose corrected Sundog
  expression is verb-field NPC behavior — unmet needs diffuse across
  satisfier nodes to produce lightweight idle orbits without scripted
  planners.
- [Money Bags](https://github.com/humiliati/Money-Bags): upcoming
  Godot softbody/terrain prototype exploring torsion, torque,
  center-of-gravity, and graph-structured interpretation of
  frame-by-frame simulation data.

These applications are not replacements for the core photometric
research claim. The control workbenches are bounded operating-envelope
studies with baselines and failure maps; the chat experiment is the
clearest current evidence that trace-conditioned policy reduces
out-of-bounds escapes under pressure; the cap-set workbench is a
primer that uses an external mathematical event as the worked example
of Sundog's evaluator discipline.

## Status

This repository is the inspection point for Sundog research work.
Present it carefully:

- **Paper-grade result** is still the photometric mirror-alignment
  experiment in MuJoCo.
- **Bounded operating-envelope studies** add Three-Body, Balance, and
  Pressure Mines — each with baselines, named positive pockets, and
  named failure regions.
- **Load-bearing pillars** on the public site stage five claim supports
  beneath the apparatus: the first equation `cos(h) = R₂₂ / α₀`, the
  structural-failure boundary map (Cut 2 separability held; Cut 3
  open), the mesa-optimization operating envelope (sharp cliff at
  λ ≈ 0.953 localized to a 5D basin at `net.7`), the Isotrophy K_facet
  v0.3h verdict (20 of 21 strict G.2 single-curve choreographies
  returned structural-zero receipts; O_617 held back as a named
  quarantine for a bridge direction outside the valid D₃
  representation — audit chain intact, theorem-facing result not
  closed), and the coarse-graining proof trunk (Phases 0–3 closed
  positive, Phase 4 open at the Bayesian-floor gate, Phase 6 staged).
- **Trace-conditioned chat experiment** at
  [sundog.cc/chat](https://sundog.cc/chat) — 0 unsafe-accepts across
  5,670 trials, scoped to claim-boundary discipline under adversarial
  pressure; not a product offering, not an inference API.
- **Cap-set / unit-distance staging ledger** at
  [`docs/SUNDOG_V_CAPSET.md`](docs/SUNDOG_V_CAPSET.md) — Sundog's
  apparatus as an evaluator for AI-produced mathematics, with the
  cap-set workbench as the one current anchor.
- **Social-readiness state** tracked in
  [`docs/SEO_AND_SOCIAL_READINESS_ROADMAP.md`](docs/SEO_AND_SOCIAL_READINESS_ROADMAP.md)
  — Phase 1 cleared 2026-05-21 across thirteen Class A pages; Phase 2
  staged.
The broader theorem remains a research program rather than a universal
claim. Where this repo says "result," it means a specific bounded
artifact; where it says "pillar," it means a load-bearing support for
the apparatus, not a universal theorem.
