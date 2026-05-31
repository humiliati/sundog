# Yang-Mills Pre-Registration Holding Pen

Roadmap: [`SUNDOG_V_YANG_MILLS.md`](../../SUNDOG_V_YANG_MILLS.md)
Lit-pass: [`YANG_MILLS_LITPASS_MEMO.md`](../../YANG_MILLS_LITPASS_MEMO.md)

Filed: **2026-05-29 (PT)**

Status: **Phase 1 instrumentation closed across the full ladder 2026-05-29;
Phase 2 v0 / v1 / v2 / v3 all executed 2026-05-29 ->
`YM-P2-NEG-A no_rank_local_structure`; bounded-null synthesis filed
2026-05-29; Phase 2 v4 powered-target reopen executed 2026-05-31 ->
`YM-P2-UNDERPOWERED no_powered_target_in_envelope`; Phase 2 v5 symmetric
Polyakov audit executed 2026-05-31 ->
`YM-P2-UNDERPOWERED no_powered_target_in_envelope`; Phase 2 v6 finite-T
Polyakov pilot executed 2026-05-31 -> `Z beta_peak_unbracketed`; Phase 2 v6a
amended finite-T Polyakov run executed 2026-05-31 ->
`YM-P2-NEG-A no_rank_local_structure`**. See
[`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md). This
directory remains the home for any future Yang-Mills pre-registrations
(Phase 1 runner manifests, Phase 2/3/4 phase specs, and any P0 amendments).
The v0, v1, v2, and v3 Phase 2 receipts are named nulls and do not admit
Phase 3 observable-certificate work. The v4 receipt is quarantine-class
underpowered, not a named null: Stage 2 was not scored because no candidate
target was both powered and disjoint across all three beta values. Any
v5 continuation followed that pre-stated P0-amendment path and also landed
underpowered: the symmetric-cell Polyakov targets were disjoint but lacked
split-half power. The registered continuation then routed to the pre-stated
finite-temperature v6 Polyakov build, not to a silent symmetric-cell retry. The
first v6 pilot did not freeze a beta slate because the pilot peak landed on the
grid boundary; the continuation therefore required a dated pilot-grid or
susceptibility-metric amendment before ensemble generation.
The v6a amendment supplied that continuation, admitted a powered/disjoint
finite-T Polyakov target, and still landed `YM-P2-NEG-A`; no positive route to
Phase 3 is open.

## Filed Artifacts

- [`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md) â€”
  cell ladder, lattice slate, action, Î² slate, generator algorithms,
  burn-in / thinning / autocorrelation rules, signature vocabulary v1,
  held-out target vocabulary v1, seven-entry leakage controls battery,
  admission requirements, outcome-branch table, anti-scope-creep rule,
  public-language boundary, and external-reviewer category. Closes the
  five Â§11 open decisions in
  [`../../SUNDOG_V_YANG_MILLS.md`](../../SUNDOG_V_YANG_MILLS.md).
- [`PHASE1_U1_2D_gauge_invariance_smoke.md`](PHASE1_U1_2D_gauge_invariance_smoke.md)
  - runner manifest for the cheapest Abelian instrumentation cell:
  `U1_2D`, 16x16, beta 1.0, gauge-randomization and raw-link residual
  smoke only. No non-Abelian or certificate claim admitted.
  **Executed 2026-05-29 â†’ `P1-A smoke_pass`.**
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
  orientation mean-plaquette isotropy spread â‰¤ 5% (new branch
  `YM-P1-QUAR-C orientation_anisotropy`) and the explicit link-unitarity
  Frobenius residual â‰¤ 1e-10 (new named branch
  `YM-P1-QUAR-D unitarity_drift`, promoted from SU2_2D's implicit check).
  No Phase 2 or rank-locality claim admitted.
  **Executed 2026-05-29 -> `P1-A smoke_pass`.**
- [`PHASE2_SU2_3D_relative_locality_v0.md`](PHASE2_SU2_3D_relative_locality_v0.md)
  - the actual non-Abelian primary read. Binding v0 spec for the
  `SU2_3D` 12Â³ Ã— Î² slate `{2.0, 2.4, 2.8}` Ã— 32-configs-per-Î²
  relative-locality certificate. Per-Î² tertile bin freezing on per-config
  `Î³_held` (LS slope of `ln Re(W)` vs area on the held-out loops
  `W14, W23, W33`); Euclidean L2 in z-score-normalized 8-dim signature
  space; within-Î² k-NN bin-purity@k=5 as primary discrimination ratio
  (chance baseline 1/3), across-Î² bin-purity vs `CTRL_RAND_STRAT` as the
  coupling-triviality cross-check; six controls scored
  (`CTRL_META`, `CTRL_RAW`, `CTRL_RAND`, `CTRL_RAND_STRAT`, `CTRL_PERM`,
  `CTRL_GAUGE_RAND`), `CTRL_FINITE_SIZE` declared but deferred to
  Phase 4. New branch `YM-P2-NEG-D raw_dominates` introduced. Three
  per-Î² ensemble invocation commands + one aggregation invocation
  command all locked in the spec.
  **Executed 2026-05-29 -> `YM-P2-NEG-A no_rank_local_structure`.**
  Receipt:
  [`../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md).
- [`P0_AMENDMENT_2026-05-29_ape_smearing.md`](P0_AMENDMENT_2026-05-29_ape_smearing.md)
  - Amendment 1 to the P0 lock. Admits APE smearing as a primary-signature
  class (vocab v4) with frozen parameters `(Î±, N_sm) = (0.5, 10)`,
  closest-SU(2) projection after every iteration, applied to `SU2_2D` /
  `SU2_3D` cells. Held-out target unchanged. New
  `YM-P*-QUAR-E smearing_drift` quarantine branch. Triggered by the v0
  NEG-A receipt and the v1 smearing probe spec.
- [`PHASE2_SU2_3D_relative_locality_v1.md`](PHASE2_SU2_3D_relative_locality_v1.md)
  - binding v1 spec mirroring v0 with signature vocabulary v4 (smeared).
  Reuses the **same three v0 per-Î² ensemble dirs** as inputs (no new
  ensemble generation; same seeds `0201`/`0202`/`0203`, same configs,
  same bare Î³_held, same per-Î² tertile bin edges â€” asserted as v1
  bin-edge defense). One aggregation invocation only; runner reads
  v0 configs, applies smearing per the P0 amendment, computes v4
  signatures, scores against unchanged bare-link held-out target.
  Six controls scored on smeared signatures. Promotion thresholds and
  branch table inherited from v0; adds `YM-P2-QUAR-E smearing_drift`
  per the amendment. Compute estimate ~2-3 min for the single
  aggregation pass.
  **Executed 2026-05-29 -> `YM-P2-NEG-A no_rank_local_structure`.**
  Receipt:
  [`../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md).
- [`PHASE2_SU2_3D_relative_locality_v2.md`](PHASE2_SU2_3D_relative_locality_v2.md)
  - binding v2 spec promoting the v1 audit's T2 candidate: signature
  vocabulary v5 = 20-dim connected 2-point correlator of bare-link
  Wilson loops `{W11, W12, W13, W22}` at the locked five
  cubic-symmetry displacement classes `{(1,0,0), (1,1,0), (1,1,1),
  (2,0,0), (2,1,0)}`. Held-out target and per-Î² bin edges unchanged
  from v0/v1; correlators only (no marginal moments) to keep test
  scientifically clean. No P0 amendment needed (correlators are a new
  vocab class within the P0 fixed-loop framework, not smearing or
  blocking). Reuses v0 ensembles bit-for-bit; ONE aggregation
  invocation.
  **Executed 2026-05-29 -> `YM-P2-NEG-A no_rank_local_structure`.**
  Receipt:
  [`../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md).
- [`PHASE2_SU2_3D_relative_locality_v3.md`](PHASE2_SU2_3D_relative_locality_v3.md)
  - binding v3 spec selecting the v2 probe's pre-stated "different
  target derived from same loops" path. **Held-out target vocabulary
  v2** = per-config spatial variance `ÏƒÂ²_W33` of `(1/2) Re Tr U_loop_3x3`
  across all `12Â³ Ã— 3 = 5184` (position Ã— orientation) samples (biased
  estimator, no Bessel correction). **Signature unchanged** (vocab v1
  bare 8-dim mean+var, re-read from v0 signature CSV â€” NOT recomputed,
  hash-asserted against v0). New per-Î² tertile bin edges computed on
  ÏƒÂ²_W33 within each Î²; NOT asserted equal to v0's Î³_held edges (target
  is different). No P0 amendment needed. Reuses v0 ensembles
  bit-for-bit; ONE aggregation invocation; ~1-2 min compute. Pre-states
  v4 fallback if v3 also NEG-As: ÏƒÂ²_W14 / ÏƒÂ²_W23, Polyakov-target
  + P0 amendment 2, or PAUSE-and-synthesize.
  **Executed 2026-05-29 -> `YM-P2-NEG-A no_rank_local_structure`.**
  Receipt:
  [`../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v3_no_rank_local_structure.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v3_no_rank_local_structure.md).
- [`../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md)
  - bounded-null synthesis receipt for the four Phase 2 `SU2_3D`
  named nulls across three signature vocabularies and two target
  classes. Implements the v3 probe's PAUSE-and-synthesize fallback.
- [`PHASE2_SU2_3D_relative_locality_v4.md`](PHASE2_SU2_3D_relative_locality_v4.md)
  - binding v4 spec triggered by the dated powered-target probe spec
  [`../../yang-mills/specs/2026-05-31_phase2_v4_powered_target_probe.md`](../../yang-mills/specs/2026-05-31_phase2_v4_powered_target_probe.md).
  Stage 1 audits candidate held-out summaries `mean_W14`, `mean_W23`,
  `sigma2_W14`, `sigma2_W23`, and `ratio_W23_W14` for split-half power
  and signature disjointness before any rank-locality score. Reuses the
  v0 ensembles bit-for-bit and introduces `YM-P2-UNDERPOWERED
  no_powered_target_in_envelope` when no candidate is admitted.
  **Executed 2026-05-31 -> `YM-P2-UNDERPOWERED no_powered_target_in_envelope`.**
  Receipt:
  [`../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v4_underpowered.md`](../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v4_underpowered.md).
- [`P0_AMENDMENT_2026-05-31_polyakov.md`](P0_AMENDMENT_2026-05-31_polyakov.md)
  - Amendment 2 to the P0 lock. Admits the Polyakov loop as held-out
  target vocabulary v4 with summaries `abs_mean_P`, `mean_abs_P`, and
  `chi_P`, covering both the symmetric v5 audit on the v0 ensembles and
  the finite-temperature v6 setup. Topological charge remains deferred.
- [`PHASE2_SU2_3D_relative_locality_v5.md`](PHASE2_SU2_3D_relative_locality_v5.md)
  - binding v5 spec triggered by the dated Polyakov probe spec
  [`../../yang-mills/specs/2026-05-31_phase2_v5_polyakov_probe.md`](../../yang-mills/specs/2026-05-31_phase2_v5_polyakov_probe.md).
  Stage 1 audits symmetric Polyakov summaries `abs_mean_P`, `mean_abs_P`,
  and `chi_P` for transverse-parity split-half power and signature
  disjointness before any rank-locality score. Reuses the v0 ensembles
  bit-for-bit and re-audits `gamma_held` as the required self-validation
  prior.
  **Executed 2026-05-31 -> `YM-P2-UNDERPOWERED no_powered_target_in_envelope`.**
  Receipt:
  [`../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v5_polyakov_underpowered.md`](../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v5_polyakov_underpowered.md).
- [`PHASE2_SU2_3D_finite_t_polyakov_v6.md`](PHASE2_SU2_3D_finite_t_polyakov_v6.md)
  - binding v6 finite-temperature Polyakov spec, pre-stated as the route
  after v5 underpowering. It is a new build (`12^2 x 4`, asymmetric
  lattice support, finite-temperature ensembles) and is not a continuation
  of the symmetric-cell runner.
  **Pilot executed 2026-05-31 -> `Z beta_peak_unbracketed`.**
  Receipt:
  [`../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v6_pilot_unbracketed.md`](../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v6_pilot_unbracketed.md).
- [`PHASE2_SU2_3D_finite_t_polyakov_v6_AMENDMENT_2026-05-31_pilot_metric.md`](PHASE2_SU2_3D_finite_t_polyakov_v6_AMENDMENT_2026-05-31_pilot_metric.md)
  - binding v6 follow-up amendment after the pilot void. It keeps the
  finite-T cell and target pool unchanged, clarifies the pilot selector as
  ensemble-level order-parameter susceptibility
  `order_suscept_abs_mean_P = (12*12) * Var(abs_mean_P)`, and extends the
  pilot grid upward to `{6.0, 6.3, 6.55, 6.8, 7.1, 7.4, 7.7, 8.0}`.
  Executed 2026-05-31 with receipt:
  [`../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v6a_finite_t_polyakov_neg_a.md`](../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v6a_finite_t_polyakov_neg_a.md).

## Required Next Artifact

No automatic finite-T retry, symmetric-cell retry, or target/signature probe
ladder is admitted after v6a. Further Yang-Mills work requires fresh external
scientific motivation or reviewer feedback. The default next artifact is an
updated external-review packet / bounded-null synthesis addendum that includes
the powered finite-T Polyakov `NEG-A`.

The Phase 5 external-review packet is drafted and **owner-pending send**
to a lattice gauge theorist:
[`../../yang-mills/EXTERNAL_REVIEW_PACKET.md`](../../yang-mills/EXTERNAL_REVIEW_PACKET.md)
with email cover-letter draft at
[`../../yang-mills/EXTERNAL_REVIEW_EMAIL_DRAFT.md`](../../yang-mills/EXTERNAL_REVIEW_EMAIL_DRAFT.md).
Owner fills `[Name]`, `[reviewer specialty signal]`, `[link]`, `[Your name]`
per the email's "Owner Fill-In Checklist Before Sending" before sending.
The generality-gallery draft-handoff card admission is the other admissible
move, blocked only by external-review-pending banner discipline.

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
  Verdict: `P1-A smoke_pass`. Pilot Ï„_int = 2.89 (well under 16),
  random-gauge signature residual max 6.66e-16 (machine epsilon),
  raw-link median normalized L2 = 0.495, wall clock 0.74 s. Mean
  plaquette 0.444 â‰ˆ Iâ‚(1)/Iâ‚€(1) â‰ˆ 0.446.
- 2026-05-29: Phase 1 SU(2) 2D harness manifest filed at
  [`PHASE1_SU2_2D_gauge_invariance_smoke.md`](PHASE1_SU2_2D_gauge_invariance_smoke.md).
  Locks `SU2_2D` 16x16 at Î²=2.0 with Creutz+KP heatbath + Brown-Woch
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
  Locks `SU2_3D` 8x8x8 at Î²=2.4 (mid-crossover for 3D SU(2)) with the
  same Creutz+KP heatbath + Brown-Woch overrelaxation (1 HB + 4 OR),
  master seed 202605290103, 8 random SU(2) Haar gauge transforms per
  config, two new 3D-specific quarantine branches
  (`YM-P1-QUAR-C orientation_anisotropy` for per-plane mean-plaquette
  spread > 5%; `YM-P1-QUAR-D unitarity_drift` for link Frobenius
  residual > 1e-10), and an explicit `plaquette_by_orientation.csv`
  output for the isotropy gate.
- 2026-05-29: Phase 1 SU(2) 3D smoke executed and filed as
  [`../../yang-mills/receipts/2026-05-29_SU2_3D_phase1_gauge_invariance_smoke_pos.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase1_gauge_invariance_smoke_pos.md).
  Verdict: `P1-A smoke_pass`. Pilot tau_int = 0.807, random-gauge signature
  residual max 3.89e-16, raw-matrix median normalized L2 = 1.415, heatbath
  fallback fraction = 0 over 5,431,296 link updates, orientation spread =
  0.00294, link unitarity max Frobenius residual 1.10e-15, wall clock 37.03 s.
  Phase 1 instrumentation is closed across `U1_2D`, `SU2_2D`, and `SU2_3D`.
- 2026-05-29: Phase 2 v0 relative-locality spec filed at
  [`PHASE2_SU2_3D_relative_locality_v0.md`](PHASE2_SU2_3D_relative_locality_v0.md).
  Locks `SU2_3D` 12Â³ Ã— Î² slate `{2.0, 2.4, 2.8}` Ã— 32 configs/Î²; per-Î²
  tertile bin freezing on per-config `Î³_held` (LS slope of `ln Re(W)` vs
  area on `W14, W23, W33` with 1e-10 hard Îµ floor); Euclidean L2 in
  z-score-normalized 8-dim signature space; within-Î² bin-purity@k=5 as
  primary discrimination ratio (chance baseline 1/3); across-Î²
  bin-purity vs `CTRL_RAND_STRAT` as coupling-triviality cross-check.
  Six controls scored (`CTRL_META`, `CTRL_RAW`, `CTRL_RAND`,
  `CTRL_RAND_STRAT`, `CTRL_PERM`, `CTRL_GAUGE_RAND`), `CTRL_FINITE_SIZE`
  declared but deferred to Phase 4. Promotion thresholds: primary
  bin-purity@5 `>= 0.5` AND beats `CTRL_RAND`/`CTRL_META`/`CTRL_RAW`
  by margin `>= 0.10`; across-Î² primary beats `CTRL_RAND_STRAT` by
  margin `>= 0.05`. New branch `YM-P2-NEG-D raw_dominates` introduced.
  Per-Î² master seeds `202605290201` / `0202` / `0203`. Three per-Î²
  ensemble invocation commands + one aggregation invocation command
  all locked in the spec.
- 2026-05-29: Phase 2 v0 four-invocation verdict executed and filed as
  [`../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md).
  Verdict: `YM-P2-NEG-A no_rank_local_structure`. Within-beta primary
  bin-purity@5 = 0.3104 (below the registered 0.5 gate), primary-minus-random
  margin = 0.0104 (below the registered 0.10 gate), and metadata matched or
  beat the primary. All ensemble health gates passed. This named null blocks
  Phase 3 from v0 and routes the lane to a dated v1 probe spec.
- 2026-05-29: v1 smearing probe spec filed at
  [`../../yang-mills/specs/2026-05-29_phase2_v1_smearing_probe.md`](../../yang-mills/specs/2026-05-29_phase2_v1_smearing_probe.md).
  Diagnoses v0 NEG-A as UV-noise-dominance on bare small Wilson loops;
  selects T3 (APE smearing on signature only) over T1 (per-orientation)
  and T2 (connected correlator) for lit-pass support, classical
  lattice-QCD remedy, and lowest expected NEG-A repeat risk. Triggers
  the P0 amendment + Phase 2 v1 spec.
- 2026-05-29: P0 amendment 1 filed at
  [`P0_AMENDMENT_2026-05-29_ape_smearing.md`](P0_AMENDMENT_2026-05-29_ape_smearing.md).
  Admits APE smearing as primary-signature class (vocab v4) with
  frozen `(Î±, N_sm) = (0.5, 10)` and exact closest-SU(2) projection
  after every iteration, for `SU2_2D` / `SU2_3D` cells. Held-out target
  unchanged. New `YM-P*-QUAR-E smearing_drift` quarantine branch
  (cited by phase index in the consuming receipt).
- 2026-05-29: Phase 2 v1 binding spec filed at
  [`PHASE2_SU2_3D_relative_locality_v1.md`](PHASE2_SU2_3D_relative_locality_v1.md).
  Reuses the three v0 per-Î² ensemble dirs as inputs (no new ensemble
  generation; same seeds 0201/0202/0203, same configs, same bare
  Î³_held, asserted per-Î² bin-edge match to machine epsilon as a void
  gate). Applies APE smearing per the amendment; recomputes v4
  smeared signature; recomputes bare v1 signature for integrity check
  vs v0; six controls scored on smeared signature; YM-P2-QUAR-E
  added per amendment. ONE aggregation invocation only; estimated
  ~2-3 min compute.
- 2026-05-29: Phase 2 v1 APE-smearing aggregation executed and filed as
  [`../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md).
  Verdict: `YM-P2-NEG-A no_rank_local_structure`. Smearing health passed
  (`det` drift `6.66e-16`, unitarity `9.42e-16`, orientation spread
  `3.10e-4`) and gauge-randomized smeared residual was `1.44e-15`, but
  within-beta primary bin-purity@5 = `0.29375`, primary-minus-random =
  `-0.00208`, and metadata/raw controls beat the primary. This named null
  blocks Phase 3 from v1 and routes the lane to a dated v2 probe spec.
  Methodology note: an early draft of the APE implementation used the
  heatbath-staple orientation and was caught immediately by
  `CTRL_GAUGE_RAND` (`YM-P1-NEG-A gauge_leakage`) before any
  rank-locality score could be interpreted; the corrected implementation
  then produced the v1 receipt numerics. Receipt-quality save.
- 2026-05-29: v2 correlator probe spec filed at
  [`../../yang-mills/specs/2026-05-29_phase2_v2_correlator_probe.md`](../../yang-mills/specs/2026-05-29_phase2_v2_correlator_probe.md).
  Diagnoses v1 NEG-A: hypothesis 1 (UV-noise dominance) FALSIFIED
  (smeared primary slightly worse than bare v0, not better); hypothesis 2
  (small-loop mean+var summaries don't encode Î³_held) strongly favored
  â€” but tested only marginal-moment summaries so far. Promotes T2
  (connected 2-point correlator) from the v1 audit as the natural last
  small-loop richness test. Locks vocab v5 = 20-dim (4 loops Ã— 5
  cubic-symmetry displacement classes). No P0 amendment needed
  (correlators are new vocab class within fixed-loop framework). Pre-
  states v3 fallback (target redesign) if v2 also lands NEG-A.
- 2026-05-29: Phase 2 v2 binding spec filed at
  [`PHASE2_SU2_3D_relative_locality_v2.md`](PHASE2_SU2_3D_relative_locality_v2.md).
  Reuses v0 ensembles bit-for-bit; same bare Î³_held; same per-Î² bin
  edges asserted to v0 to 1e-12; vocab v5 connected correlators on bare
  links at frozen displacement slate `{(1,0,0), (1,1,0), (1,1,1),
  (2,0,0), (2,1,0)}`. Six controls scored on correlator signature; same
  promotion thresholds as v0/v1. ONE aggregation invocation only.
  Compute estimate ~1-2 min.
- 2026-05-29: Phase 2 v2 connected-correlator aggregation executed and
  filed as
  [`../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md).
  Verdict: `YM-P2-NEG-A no_rank_local_structure`. Bin-edge replay
  matched v0 exactly (per-beta `0`, global `0`) and gauge-randomized
  correlator residual was `3.33e-16`, but within-beta primary
  bin-purity@5 = `0.308333333333`, primary-minus-random =
  `0.020833333333`, and metadata beat the primary by `0.00625`.
  This named null blocks Phase 3 from v2. With v0/v1/v2 all null, the
  small-loop hypothesis is exhausted across marginal bare, marginal
  smeared, and bare spatial connected-correlator summaries; route to a
  dated v3 target-redesign probe spec per the pre-stated fallback table.
- 2026-05-29: v3 target-redesign probe spec filed at
  [`../../yang-mills/specs/2026-05-29_phase2_v3_target_redesign_probe.md`](../../yang-mills/specs/2026-05-29_phase2_v3_target_redesign_probe.md),
  followed by Phase 2 v3 binding spec
  [`PHASE2_SU2_3D_relative_locality_v3.md`](PHASE2_SU2_3D_relative_locality_v3.md).
  v3 isolates target redesign: unchanged v1 signature re-read from v0
  CSV with SHA-256 assertions; new held-out target vocab v2 =
  `sigma2_W33`, biased spatial variance over all 5184 W33 samples per
  config. No P0 amendment needed.
- 2026-05-29: Phase 2 v3 aggregation executed and filed as
  [`../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v3_no_rank_local_structure.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v3_no_rank_local_structure.md).
  Verdict: `YM-P2-NEG-A no_rank_local_structure`. Signature hashes,
  target spread, bin-edge timing, and gauge-randomization gates passed
  (`signature` residual `5.00e-13`, target residual `2.22e-16`), but
  within-beta primary bin-purity@5 = `0.329166666667`, primary-minus-
  random = `0.027083333333`, and across-beta primary trailed
  `CTRL_RAND_STRAT` by `0.041666666667`.
- 2026-05-29: bounded-null synthesis receipt filed as
  [`../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md).
  Disposition: PAUSE-and-synthesize. Four consistent Phase 2 named nulls
  across three signature vocabularies and two target classes are treated
  as a substantive cell-bounded null, not as permission for automatic
  v4 probe-ladder continuation. The later v4 reopen therefore required,
  and received, a fresh dated diagnostic spec.
- 2026-05-29: Phase 5 external-review packet drafted at
  [`../../yang-mills/EXTERNAL_REVIEW_PACKET.md`](../../yang-mills/EXTERNAL_REVIEW_PACKET.md)
  with email cover-letter draft at
  [`../../yang-mills/EXTERNAL_REVIEW_EMAIL_DRAFT.md`](../../yang-mills/EXTERNAL_REVIEW_EMAIL_DRAFT.md).
  Reviewer category locked at lattice gauge theorist per P0. Packet
  mirrors the riemann `EXTERNAL_REVIEW_PACKET.md` shape (reviewer
  snapshot, falsification surface, one-sentence ask, what we are/aren't
  asking, seven adapted reviewer questions, 10/25/60-minute review-budget
  tiers, four-receipt summary, files-to-read map, single-comment fallback,
  packet hygiene). Two questions are bounded-null-specific (probe-ladder
  completeness, pre-registration discipline as a methodological claim)
  beyond the five locked P0 reviewer questions. Email draft has short
  and slightly-warmer versions plus accept/decline follow-ups and an
  Owner Fill-In Checklist. Owner-pending: fill `[Name]`, `[reviewer
  specialty signal]`, `[link]`, `[Your name]` and send.
- 2026-05-31: powered-target v4 reopen filed at
  [`../../yang-mills/specs/2026-05-31_phase2_v4_powered_target_probe.md`](../../yang-mills/specs/2026-05-31_phase2_v4_powered_target_probe.md)
  and
  [`PHASE2_SU2_3D_relative_locality_v4.md`](PHASE2_SU2_3D_relative_locality_v4.md).
  The single locked invocation executed and filed as
  [`../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v4_underpowered.md`](../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v4_underpowered.md).
  Verdict: `YM-P2-UNDERPOWERED no_powered_target_in_envelope`. No
  candidate target cleared both power and disjointness across all three
  beta values; `gamma_held` failed the power self-validation as required;
  Stage 2 rank-locality scoring was not run.
- 2026-05-31: Polyakov amendment 2 and v5/v6 specs filed at
  [`P0_AMENDMENT_2026-05-31_polyakov.md`](P0_AMENDMENT_2026-05-31_polyakov.md),
  [`PHASE2_SU2_3D_relative_locality_v5.md`](PHASE2_SU2_3D_relative_locality_v5.md),
  and
  [`PHASE2_SU2_3D_finite_t_polyakov_v6.md`](PHASE2_SU2_3D_finite_t_polyakov_v6.md).
  The v5 symmetric Polyakov single locked invocation executed and filed as
  [`../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v5_polyakov_underpowered.md`](../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v5_polyakov_underpowered.md).
  Verdict: `YM-P2-UNDERPOWERED no_powered_target_in_envelope`. All three
  Polyakov candidates were disjoint, but none cleared split-half power
  across all three beta values; `gamma_held` failed the power
  self-validation and Polyakov gauge residual max was
  `1.6653345369377348e-16`. Stage 2 rank-locality scoring was not run;
  v6 finite-temperature Polyakov is the only registered continuation.
- 2026-05-31: v6 finite-temperature Polyakov runner invoked on the locked
  pilot grid `{6.0, 6.3, 6.55, 6.8, 7.1}` and filed as
  [`../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v6_pilot_unbracketed.md`](../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v6_pilot_unbracketed.md).
  Verdict: `Z beta_peak_unbracketed`. The pilot `mean_chi_P` peak landed at
  beta `6.0`, the lower boundary, so no finite-T beta slate was frozen and no
  ensembles were generated.
- 2026-05-31: v6 follow-up amendment filed at
  [`PHASE2_SU2_3D_finite_t_polyakov_v6_AMENDMENT_2026-05-31_pilot_metric.md`](PHASE2_SU2_3D_finite_t_polyakov_v6_AMENDMENT_2026-05-31_pilot_metric.md).
  It changes only the pre-generation pilot selector/grid: the selector is now
  ensemble-level `order_suscept_abs_mean_P`, and the grid extends upward to
  `{6.0, 6.3, 6.55, 6.8, 7.1, 7.4, 7.7, 8.0}`.
- 2026-05-31: v6a executed and filed as
  [`../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v6a_finite_t_polyakov_neg_a.md`](../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v6a_finite_t_polyakov_neg_a.md).
  The amended pilot froze `{6.3, 6.55, 6.8}`; `abs_mean_P` was admitted as
  powered/disjoint; Stage 2 landed `YM-P2-NEG-A no_rank_local_structure`.
