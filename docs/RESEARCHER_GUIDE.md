# Researcher Guide

This guide is for readers who want to inspect Sundog as a research
artifact without first reconstructing the project history. It is
designed to be readable end-to-end in 10–15 minutes.

## One-Sentence Summary

Sundog is a research program for *traceable* agent policy under
partial observability — every claim it ships carries a named falsifier,
a bounded operating envelope, and an explicit out-of-scope statement.

## The Paper-Grade Scientific Unit

The paper-shaped contribution is the photometric mirror-alignment
experiment in MuJoCo:

- A 2-DoF pole has a mirrored end-effector.
- A ceiling laser reflects from the mirror to a floor detector ring.
- The agent must align the reflected beam to detector 0.
- The agent sees detector intensities, joint angles, joint velocities,
  and torque proxies.
- The agent does not see the laser position, mirror hit point, target
  Cartesian position, or floor-hit geometry.

The controller uses a three-stage loop:

1. `SCAN`: sweep the bounded joint space with a Lissajous trajectory.
2. `SEEK`: move to the best intensity observed during the scan.
3. `TRACK`: refine using perturb-and-observe extremum seeking.

This is the measurable form of the older Sundog idea that shadow,
torque, and light can become an indirect alignment signal. The paper
draft and outline live at [`PAPER_v1_draft.md`](PAPER_v1_draft.md) and
[`PAPER_OUTLINE_v0.md`](PAPER_OUTLINE_v0.md).

## The Evidence Stack Beyond the Paper Unit

Since the photometric experiment was first written up, the lab has
added several additional evidence surfaces. **None of them replace or
upgrade the photometric claim**; each lives in its own bounded scope.

### Operating-envelope studies

Each is a bounded sweep with baselines, a named positive pocket, and
a named failure region:

- *Three-Body Dynamics* — guarded accelerometer-proxy TRACK improves
  survival inside a mapped high-velocity near-escape pocket through a
  16-second tested horizon. See
  [`SUNDOG_V_THREEBODY.md`](SUNDOG_V_THREEBODY.md).
- *Sundog Balance* — shadow-derived cart-pole control beats naive
  shadow-centering on 28/28 diagnostic-positive cells. See
  [`sundog_v_balance.md`](sundog_v_balance.md).
- *Sundog Pressure Mines* — pressure-derived Sundog variants improve
  budget-adjusted safe-tile progress inside a named pocket; a paired
  failure region is published beside it. See
  [`sundog_v_minesweeper.md`](sundog_v_minesweeper.md).

### Load-bearing pillars

Five claim supports underneath the apparatus, each with an inline
workbench or evidence page on the public site:

- *h(x)* — the parhelion-offset inverse `cos(h) = R₂₂ / α₀`, the
  first equation the project trusts on a strict 3-photo subset.
  Workbench at [sundog.cc/h-of-x](https://sundog.cc/h-of-x); roadmap
  [`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md).
- *Structural Failure Boundary Map* — pre-registered five-locus
  falsifier; P0 and P1 passed, Cut 2 separability held, Cut 3 still
  open. Page at
  [sundog.cc/structural-failure](https://sundog.cc/structural-failure);
  pre-registration in
  [`prereg/structural-failure-coincidence/`](prereg/structural-failure-coincidence/).
- *Mesa Optimization Envelope* — in-vitro operating envelope with a
  sharp cliff at λ ≈ 0.953 localized to a 5D basin at the actor's
  final hidden layer. See [`SUNDOG_V_MESA.md`](SUNDOG_V_MESA.md).
- *Isotrophy K_facet v0.3h verdict* — 20 of 21 strict G.2 single-curve
  choreographies returned structural-zero receipts at m₃ = 1; the 21st
  (O_617) is held back as a named quarantine for a bridge direction
  outside the valid D₃ representation. Audit chain intact;
  theorem-facing result is not closed. Page at
  [sundog.cc/isotrophy](https://sundog.cc/isotrophy); ledger
  [`SUNDOG_V_ISOTROPHY_KFACET.md`](SUNDOG_V_ISOTROPHY_KFACET.md).
- *Coarse-Graining Proof Trunk* — Phases 0–3 closed positive; Phase 4
  open at the Bayesian-floor gate; Phase 6 staged. See
  [`COARSE_GRAINING_PROOF_ROADMAP.md`](COARSE_GRAINING_PROOF_ROADMAP.md).

### Trace-conditioned chat experiment

At [sundog.cc/chat](https://sundog.cc/chat). Most recent result: zero
unsafe-accepts across 5,670 trials spanning six model implementations
across four training lineages (deterministic compositor + OpenAI +
Anthropic + Meta Llama at two sizes + Alibaba Qwen), plus three
prompt-type slates and a falsification slate. The experiment is scoped
to claim-boundary discipline under adversarial pressure — it is **not
an inference API** and **not a product offering**. See
[`SUNDOG_V_CHAT.md`](SUNDOG_V_CHAT.md).

### Cap-set / unit-distance staging ledger

At [`SUNDOG_V_CAPSET.md`](SUNDOG_V_CAPSET.md). Couples Sundog's
apparatus to AI-produced mathematics. The cap-set workbench
([sundog.cc/capset](https://sundog.cc/capset)) is the one current
anchor, staged as a primer for the 2016 polynomial-method bound that
the 2026 OpenAI unit-distance disproof rhymes with. **Not a
Sundog-original mathematical claim** — see
[`APPLICATIONS.md`](APPLICATIONS.md) for the explicit out-of-scope
statement.

## What The Evidence Shows

At the baseline operating point, the photometric controller reaches terminal
target intensity statistically indistinguishable from a target-aware analytic
baseline:

| Condition | n | Terminal intensity mean | 95% CI | Median time-to-0.9 |
| --- | ---: | ---: | --- | ---: |
| photometric | 30 | 0.945 | [0.936, 0.954] | 188 |
| doa_direct | 30 | 0.936 | [0.925, 0.947] | 11.5 |
| doa_noisy | 30 | 0.911 | [0.894, 0.927] | 14 |
| random | 30 | approximately 0 | approximately [0, 0] | 500 censored |

Pairwise Mann-Whitney tests on terminal target intensity:

- photometric vs. doa_direct: `U = 526`, `p = 0.264`
- photometric vs. doa_noisy: `U = 649`, `p = 0.003`
- photometric vs. random: `p = 1.96e-11`

The interpretation is not "photometric beats the oracle." It is that indirect
feedback can match terminal oracle accuracy in this setting, while paying a
large acquisition-time cost.

## What The Evidence Does Not Show

This repository does not yet demonstrate:

- a general theorem over arbitrary embodied agents;
- a hardware result;
- scalability to high-DoF robots;
- robustness to all occlusion, diffraction, polarization, or multi-bounce
  optical effects;
- a learned controller that discovers the scan/seek/track structure from
  scratch;
- a claim that every application repo is independently validated science.

Those are future research directions. They should not be implied in an
academic abstract.

## Known Boundary

The joint-limit sweep is the important negative result. With symmetric joint
limits below roughly 1.2 rad, the photometric agent can lose badly to the
oracle because the reachable workspace no longer contains the optimum. The
oracle can still clip an externally known optimum toward the boundary; the
photometric controller only sees a low local signal and cannot infer that the
true optimum lies outside reach.

The paper should own this boundary. It is the difference between a scientific
claim and a pitch.

## Primary Files To Inspect

**Paper-grade scientific unit (the photometric experiment):**

| File | Why it matters |
| --- | --- |
| `docs/PAPER_v1_draft.md` | Full paper-style narrative and results. |
| `docs/PAPER_OUTLINE_v0.md` | Venue framing, reviewer risks, and stress-test notes. |
| `experiments/run_baseline_comparison.py` | Matched-seed experimental design and run loop. |
| `experiments/analysis.py` | Metrics, bootstrap CIs, and statistical tests. |
| `experiments/stress_tests.py` | Stressor sweep implementation. |
| `optics.py` | Analytic reflection model and oracle solve. |
| `agents/photometric.py` | Scan/seek/track controller. |
| `agents/baselines.py` | DOA-direct, DOA-noisy, and random baselines. |
| `results/analysis/analysis_summary.json` | Headline computed statistics. |
| `results/stress_tests/stress_summary.csv` | Stress-test summary table. |

**Evidence stack beyond the paper unit (ledgers and roadmaps):**

| File | Why it matters |
| --- | --- |
| `docs/SUNDOG_V_GEOMETRY.md` | Geometry roadmap, h(x) workbench, calibration boundaries. |
| `docs/SUNDOG_V_MESA.md` | Mesa-optimization trap; in-vitro envelope and the λ ≈ 0.953 cliff. |
| `docs/SUNDOG_V_CHAT.md` | Chat-experiment writeup; 5,670-trial result and methodology. |
| `docs/SUNDOG_V_CAPSET.md` | Cap-set / unit-distance staging ledger; evaluator vs substrate horizons. |
| `docs/SUNDOG_V_GRAVITY.md` | The pattern that the ledgers above mirror; candidate-template source. |
| `docs/COARSE_GRAINING_PROOF_ROADMAP.md` | Proof-trunk phase ladder (P0–P6 status). |
| `docs/prereg/structural-failure-coincidence/` | Pre-registered five-locus boundary map and its disposition record. |

**Public site and discipline:**

| File | Why it matters |
| --- | --- |
| `docs/APPLICATIONS.md` | Application map; per-workbench claim boundaries; cap-set "out-of-scope" statement. |
| `docs/SEO_AND_SOCIAL_READINESS_ROADMAP.md` | Per-page OG/JSON-LD/sitemap matrix; Phase 1 cleared 2026-05-21. |
| `docs/LEGAL_STANDING.md` | Current entity status (Stellar Aqua LLC) and future-entity transition path. |
| `docs/501c3/INITIATIVE_INDEX.md` | Working drafts for the planned 501(c)(3) ownership transfer. |

## Historical Context
## Historical Context

The older theorem materials in `notebooks/` and
`sundog_alignment_theorem_final_fixed/` preserve the original conceptual
language around `H(x) = dS/dtau`, bloom collapse, shadow geometry, torque, and
resonance. Those documents are useful for understanding the motivation, but the
current academic posture is narrower and more empirical.

The shift is deliberate: keep the poetry as origin, but let the paper stand on
claims that can be reproduced, attacked, and improved.
