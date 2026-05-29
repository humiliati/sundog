# Yang-Mills Pre-Registration Holding Pen

Roadmap: [`SUNDOG_V_YANG_MILLS.md`](../../SUNDOG_V_YANG_MILLS.md)
Lit-pass: [`YANG_MILLS_LITPASS_MEMO.md`](../../YANG_MILLS_LITPASS_MEMO.md)

Filed: **2026-05-29 (PT)**

Status: **Phase 1 U(1) 2D and SU(2) 2D smokes passed 2026-05-29; SU(2) 3D
manifest filed 2026-05-29**. See
[`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md). This
directory remains the home for any future Yang-Mills pre-registrations
(Phase 1 runner manifests, Phase 2/3/4 phase specs, and any P0 amendments).
It still admits no Phase 2 scoring by itself.

## Filed Artifacts

- [`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md) —
  cell ladder, lattice slate, action, β slate, generator algorithms,
  burn-in / thinning / autocorrelation rules, signature vocabulary v1,
  held-out target vocabulary v1, seven-entry leakage controls battery,
  admission requirements, outcome-branch table, anti-scope-creep rule,
  public-language boundary, and external-reviewer category. Closes the
  five §11 open decisions in
  [`../../SUNDOG_V_YANG_MILLS.md`](../../SUNDOG_V_YANG_MILLS.md).
- [`PHASE1_U1_2D_gauge_invariance_smoke.md`](PHASE1_U1_2D_gauge_invariance_smoke.md)
  - runner manifest for the cheapest Abelian instrumentation cell:
  `U1_2D`, 16x16, beta 1.0, gauge-randomization and raw-link residual
  smoke only. No non-Abelian or certificate claim admitted.
  **Executed 2026-05-29 → `P1-A smoke_pass`.**
- [`PHASE1_SU2_2D_gauge_invariance_smoke.md`](PHASE1_SU2_2D_gauge_invariance_smoke.md)
  - runner manifest for the SU(2) 2D harness cell:
  `SU2_2D`, 16x16, beta 2.0, SU(2) Creutz heatbath + Kennedy-Pendleton +
  Brown-Woch overrelaxation (1 HB + 4 OR per combined sweep), identity +
  8 random SU(2) Haar gauge transforms per config. No Phase 2 or
  rank-locality claim admitted.
  **Executed 2026-05-29 -> `P1-A smoke_pass`.**
- [`PHASE1_SU2_3D_gauge_invariance_smoke.md`](PHASE1_SU2_3D_gauge_invariance_smoke.md)
  - runner manifest for the SU(2) 3D primary cell at the Phase 4 split
  partner: `SU2_3D`, 8x8x8, beta 2.4 (middle of slate, mid-crossover),
  same SU(2) Creutz+KP heatbath + Brown-Woch overrelaxation (1 HB + 4 OR
  per combined sweep), master seed 202605290103, identity + 8 random
  SU(2) Haar gauge transforms per config. Adds 3D-specific gates: per-
  orientation mean-plaquette isotropy spread ≤ 5% (new branch
  `YM-P1-QUAR-C orientation_anisotropy`) and the explicit link-unitarity
  Frobenius residual ≤ 1e-10 (new named branch
  `YM-P1-QUAR-D unitarity_drift`, promoted from SU2_2D's implicit check).
  No Phase 2 or rank-locality claim admitted. Runner not yet implemented.

## Required Next Artifact

The next artifact should be the minimal SU(2) 3D core module + entry
runner + package script required by
[`PHASE1_SU2_3D_gauge_invariance_smoke.md`](PHASE1_SU2_3D_gauge_invariance_smoke.md).
The existing U(1) and SU(2) 2D entries must remain bit-for-bit unchanged;
shared helpers (Sokal τ_int, mulberry32 substream factory, CSV writer,
hash finalizer, git info, CLI parser, SU(2) matrix primitives, quaternion
Haar sampler) may be copied or factored into a common library in the
same commit. Do not implement Phase 2 nearest-neighbor scoring, smearing,
blocking, topological observables, or 4D hooks in that runner.

## Guardrail

No Yang-Mills code run is admitted unless a phase manifest under this
directory exists, cites the P0 lock version, and fills every Admission
Requirement listed there. Exploratory notebooks or scratch checks, if
ever needed, must be labelled exploratory and cannot be cited as receipts.

## Current State

- 2026-05-29: holding pen opened.
- 2026-05-29: [`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md)
  filed. Receipt template at
  [`../../yang-mills/RECEIPT_TEMPLATE.md`](../../yang-mills/RECEIPT_TEMPLATE.md)
  and receipts dir at
  [`../../yang-mills/receipts/`](../../yang-mills/receipts/) stood up.
  Runner code remains admitted only after a phase manifest is filed.
- 2026-05-29: Phase 1 U(1) 2D gauge-invariance smoke manifest filed at
  [`PHASE1_U1_2D_gauge_invariance_smoke.md`](PHASE1_U1_2D_gauge_invariance_smoke.md).
  Runner subsequently executed against that exact manifest.
- 2026-05-29: Phase 1 U(1) 2D smoke executed and filed as
  [`../../yang-mills/receipts/2026-05-29_U1_2D_phase1_gauge_invariance_smoke_pos.md`](../../yang-mills/receipts/2026-05-29_U1_2D_phase1_gauge_invariance_smoke_pos.md).
  Verdict: `P1-A smoke_pass`. Pilot τ_int = 2.89 (well under 16),
  random-gauge signature residual max 6.66e-16 (machine epsilon),
  raw-link median normalized L2 = 0.495, wall clock 0.74 s. Mean
  plaquette 0.444 ≈ I₁(1)/I₀(1) ≈ 0.446.
- 2026-05-29: Phase 1 SU(2) 2D harness manifest filed at
  [`PHASE1_SU2_2D_gauge_invariance_smoke.md`](PHASE1_SU2_2D_gauge_invariance_smoke.md).
  Locks `SU2_2D` 16x16 at β=2.0 with Creutz+KP heatbath + Brown-Woch
  overrelaxation (1 HB + 4 OR), master seed 202605290102, 8 random SU(2)
  Haar gauge transforms per config, new `YM-P1-QUAR-B heatbath_pathology`
  branch for KP fallback >0.1% of link updates.
- 2026-05-29: Phase 1 SU(2) 2D smoke executed and filed as
  [`../../yang-mills/receipts/2026-05-29_SU2_2D_phase1_gauge_invariance_smoke_pos.md`](../../yang-mills/receipts/2026-05-29_SU2_2D_phase1_gauge_invariance_smoke_pos.md).
  Verdict: `P1-A smoke_pass`. Pilot tau_int = 0.404 (heatbath+overrelax
  killing decorrelation as predicted), random-gauge signature residual
  max 3.05e-16 (machine epsilon), raw-matrix median normalized L2 = 1.414,
  heatbath fallback fraction = 0 over 1,810,432 link updates, link
  unitarity max Frobenius residual 6.66e-16, mean plaquette 0.4332, wall
  clock 7.18 s. Code commit `8f1f25f3...`, gitDirty=true.
- 2026-05-29: Phase 1 SU(2) 3D primary-cell manifest filed at
  [`PHASE1_SU2_3D_gauge_invariance_smoke.md`](PHASE1_SU2_3D_gauge_invariance_smoke.md).
  Locks `SU2_3D` 8x8x8 at β=2.4 (mid-crossover for 3D SU(2)) with the
  same Creutz+KP heatbath + Brown-Woch overrelaxation (1 HB + 4 OR),
  master seed 202605290103, 8 random SU(2) Haar gauge transforms per
  config, two new 3D-specific quarantine branches
  (`YM-P1-QUAR-C orientation_anisotropy` for per-plane mean-plaquette
  spread > 5%; `YM-P1-QUAR-D unitarity_drift` for link Frobenius
  residual > 1e-10), and an explicit `plaquette_by_orientation.csv`
  output for the isotropy gate. Runner implementation pending.
