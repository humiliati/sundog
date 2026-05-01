# Researcher Guide

This guide is for readers who want to inspect Sundog as a research artifact
without first reconstructing the project history.

## One-Sentence Summary

Sundog studies whether indirect environmental signals can be transformed into
usable control information for software agents.

## The Current Scientific Unit

The current paper-shaped contribution is the photometric mirror-alignment
experiment:

- A 2-DoF pole has a mirrored end-effector.
- A ceiling laser reflects from the mirror to a floor detector ring.
- The agent must align the reflected beam to detector 0.
- The agent sees detector intensities, joint angles, joint velocities, and
  torque proxies.
- The agent does not see the laser position, mirror hit point, target
  Cartesian position, or floor-hit geometry.

The controller uses a three-stage loop:

1. `SCAN`: sweep the bounded joint space with a Lissajous trajectory.
2. `SEEK`: move to the best intensity observed during the scan.
3. `TRACK`: refine using perturb-and-observe extremum seeking.

This is the measurable form of the older Sundog idea that shadow, torque, and
light can become an indirect alignment signal.

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

## Historical Context

The older theorem materials in `notebooks/` and
`sundog_alignment_theorem_final_fixed/` preserve the original conceptual
language around `H(x) = dS/dtau`, bloom collapse, shadow geometry, torque, and
resonance. Those documents are useful for understanding the motivation, but the
current academic posture is narrower and more empirical.

The shift is deliberate: keep the poetry as origin, but let the paper stand on
claims that can be reproduced, attacked, and improved.
