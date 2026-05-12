# Sundog

**Copyright © 2026 Stellar Aqua LLC. All rights reserved.**

Sundog Research Lab is an independent applied research program sponsored by Stellar Aqua LLC.

---

Sundog is the research spine for the Sundog Alignment Theorem: a practical
program for turning indirect, observable structure into controllable software
systems.

The current repository should be read as a research artifact, not as a product
monorepo. It contains the theorem history, the rebuilt mirror-alignment
experiment, stress-test results, three operating-envelope workbenches, and
runner infrastructure that connects the alignment framing to other Sundog
applications.

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

- [Researcher guide](docs/RESEARCHER_GUIDE.md): the shortest path through the
  repo for a reviewer, collaborator, or technically skeptical reader.
- [Documentation index](docs/README.md): local map of the research-facing docs.
- [Website development guide](docs/WEBSITE_DEVELOPMENT.md): minimal directions
  for editing the public site and deploying it through Cloudflare Pages.
- [Scientific criteria](docs/SCIENTIFIC_CRITERIA.md): what has been made
  testable, what has not, and what would strengthen the paper.
- [Applications map](docs/APPLICATIONS.md): how the Three-Body, Balance, and
  Pressure Mines workbenches, EyesOnly, Dungeon Gleaner, and Money Bags express
  Sundog-derived alignment ideas in test surfaces and product systems.
- [Promo highlights](docs/PROMO_HIGHLIGHTS.md): hooks, headlines, and
  future-facing language for public communication.
- [Standalone app roadmap](docs/STANDALONE_APP_ROADMAP.md): plan for a
  double-click observer app with minimal dependencies.
- [Paper draft](docs/PAPER_v1_draft.md): current paper-shaped writeup for the
  photometric mirror-alignment experiment.
- [Paper outline and review notes](docs/PAPER_OUTLINE_v0.md): venue framing,
  reviewer risks, and stress-test interpretation.

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
| Three-Body Dynamics | Guarded accelerometer-proxy TRACK improves survival over passive and naive local baselines in a high-velocity near-escape pocket. | Lower velocity and equal-mass cells expose controller harms. |
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

- [Three-body dynamics workbench](threebody.html): local experiment surface
  whose Phase 11 result maps a bounded high-velocity near-escape pocket where
  guarded accelerometer-proxy TRACK control improves survival over passive and
  naive local baselines, while lower-velocity and equal-mass cells remain
  explicit harm boundaries.
- [Sundog Balance workbench](balance.html): cart-pole surface where the
  controller is denied pole angle and acts from cast-shadow geometry. Phase 10
  confirmed the diagnostic-positive envelope while preserving overhead-light
  and high-delay degradation boundaries.
- [Sundog Pressure Mines](mines.html): game-native hidden-hazard workbench
  where noisy pressure values support safer budget-adjusted progress in one
  named pocket, published beside a matched failure region.
- [EyesOnly](https://github.com/humiliati/EyesOnly): live spy-game platform
  and Gone Rogue procedural roguelike integration.
- [Dungeon Gleaner / DCgamejam2026](https://github.com/humiliati/DCgamejam2026):
  Dungeon Crawler Game Jam 2026 entry whose corrected Sundog expression is
  verb-field NPC behavior: unmet needs diffuse across satisfier nodes to
  produce lightweight idle orbits without scripted planners.
- [Money Bags](https://github.com/humiliati/Money-Bags): upcoming Godot
  softbody/terrain prototype exploring torsion, torque, center-of-gravity, and
  graph-structured interpretation of frame-by-frame simulation data.

Those applications are not replacements for the core photometric research
claim. They are now stronger than simple demos, though: three of them are
bounded operating-envelope studies with baselines and failure maps, and the
remaining systems show where the same vocabulary is becoming product and
instrumentation.

## Status

This repository is the inspection point for Sundog research work. Present it
carefully: the paper-grade result is still photometric alignment, the
application evidence now includes multiple bounded operating-envelope studies,
and the broader theorem remains a research program rather than a universal
claim.
