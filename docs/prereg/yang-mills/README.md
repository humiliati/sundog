# Yang-Mills Pre-Registration Holding Pen

Roadmap: [`SUNDOG_V_YANG_MILLS.md`](../../SUNDOG_V_YANG_MILLS.md)
Lit-pass: [`YANG_MILLS_LITPASS_MEMO.md`](../../YANG_MILLS_LITPASS_MEMO.md)

Filed: **2026-05-29 (PT)**

Status: **Phase 0 domain lock filed 2026-05-29**. See
[`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md). This
directory remains the home for any future Yang-Mills pre-registrations
(Phase 1 runner manifests, Phase 2/3/4 phase specs, and any P0 amendments).
It still admits no runner code by itself.

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
  rank-locality claim admitted. Runner not yet implemented.

## Required Next Artifact

The next artifact should be the minimal SU(2) core module + entry runner +
package script required by
[`PHASE1_SU2_2D_gauge_invariance_smoke.md`](PHASE1_SU2_2D_gauge_invariance_smoke.md).
The U(1) entry `scripts/yang-mills-phase1-gauge-smoke.mjs` must remain
U(1)-only and bit-for-bit unchanged; shared utilities (CSV writer, hash
finalizer, git info, CLI parser, Sokal tau_int, mulberry32 substream factory)
may be factored into a common library in the same commit. Do not implement
Phase 2 neighbor scoring, smearing, blocking, topological observables, or
4D hooks in that runner.

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
  No runner code yet.
- 2026-05-29: Phase 1 U(1) 2D gauge-invariance smoke manifest filed at
  [`PHASE1_U1_2D_gauge_invariance_smoke.md`](PHASE1_U1_2D_gauge_invariance_smoke.md).
  The next move is runner implementation against that exact manifest.
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
  branch for KP fallback >0.1% of link updates. Runner implementation
  pending.
