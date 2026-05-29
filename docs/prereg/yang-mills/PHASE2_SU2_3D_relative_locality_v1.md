# Yang-Mills Phase 2 v1 - SU2 3D Relative-Locality Certificate Spec (Smeared)

Status: **Phase 2 v1 spec filed 2026-05-29**. This is a pre-run binding
spec, not a result. It admits the v1 aggregation runner only after the
implementation reads the three v0 per-β ensemble directories, applies
APE smearing per the cited P0 amendment, and emits the runtime manifest
fields named below. It admits no Yang-Mills, Phase 3
observable-certificate, Phase 4 finite-size, continuum, confinement, or
mass-gap claim. A v1 pass is a **vocab v4 smeared-signature**
relative-locality receipt, scoped accordingly.

P0 lock: [`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md)
P0 amendment 1 (APE smearing):
[`P0_AMENDMENT_2026-05-29_ape_smearing.md`](P0_AMENDMENT_2026-05-29_ape_smearing.md)
v0 spec: [`PHASE2_SU2_3D_relative_locality_v0.md`](PHASE2_SU2_3D_relative_locality_v0.md)
v0 receipt:
[`../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md)
Triggering probe spec:
[`../../yang-mills/specs/2026-05-29_phase2_v1_smearing_probe.md`](../../yang-mills/specs/2026-05-29_phase2_v1_smearing_probe.md)
Receipt template:
[`../../yang-mills/RECEIPT_TEMPLATE.md`](../../yang-mills/RECEIPT_TEMPLATE.md)

## Claim Under Test

Inside the `SU2_3D` 12×12×12 primary cell, over the registered β slate
`{2.0, 2.4, 2.8}` with 32 retained configurations per β taken **from
the v0 ensembles**, the **v4 APE-smeared** small-Wilson-loop signature
should preserve the per-config bare-link held-out-Wilson-loop
area-law-decay tertile label (`γ_held`-bin) in nearest-neighbor rank
space, **beyond** every metadata, raw-link, random, β-stratified
random, label-permutation, and gauge-randomized control on the same
frozen neighbor graphs and frozen bin edges.

This is the second Phase 2 v0/v1 read on this cell. v0 (vocab v1, bare
loops) landed `YM-P2-NEG-A no_rank_local_structure`. v1 (vocab v4,
APE-smeared loops with frozen `(α, N_sm) = (0.5, 10)`) tests whether
UV-noise suppression on the same loop set, applied to the **same
ensembles**, recovers a within-β `γ_held` rank-locality signal that
beats every control.

A v1 pass is bounded by the same envelope as v0: cell-local
(`SU2_3D`, 12³), β-slate-local, signature-class-local (vocab v4,
smeared loops). It is **not** a Yang-Mills result, a confinement
proof, a mass-gap claim, a continuum statement, or evidence at any β,
lattice size, smearing parameter, or signature class outside the
registered envelope. A v1 null is filed as `YM-P2-NEG-A` / `NEG-B` /
`NEG-C` / `NEG-D` per the v0 branch table.

## Scope

In scope at this v1:

- application of APE smearing to the v0 per-β ensemble configurations
  at the cited P0 amendment parameters `(α, N_sm) = (0.5, 10)`;
- v4 smeared small-Wilson-loop signature (`W11`, `W12`, `W13`, `W22`
  mean/var on smeared links, position-and-orientation averaged);
- v1 held-out target unchanged (bare-link `{W14, W23, W33}` →
  `γ_held`);
- within-β k-NN rank-locality scoring as the **primary** read;
- across-β k-NN rank-locality scoring as the coupling-triviality
  cross-check;
- six controls scored (`CTRL_META`, `CTRL_RAW`, `CTRL_RAND`,
  `CTRL_RAND_STRAT`, `CTRL_PERM`, `CTRL_GAUGE_RAND`) on smeared
  signatures;
- one control declared but deferred (`CTRL_FINITE_SIZE`, gated to
  Phase 4);
- smearing-health gates introduced by the P0 amendment (smearing
  drift, post-projection unitarity, per-orientation smeared-plaquette
  isotropy);
- bootstrap 95% CIs on every mean bin-purity score; v1 gates remain
  point-estimate.

Out of scope at this v1:

- new per-β ensemble generation (v1 reuses v0 ensembles bit-for-bit);
- any `(α, N_sm)` other than `(0.5, 10)`;
- any blocking; any non-APE smearing variant (HYP, Stout, Wilson flow,
  etc.);
- smearing the held-out target loops;
- any β value or lattice size outside the registered v0 envelope;
- Phase 3, Phase 4, Phase 5 work;
- any retune of the per-β tertile bin edges after the v0 fixed values.

## Inputs From v0

This v1 spec is non-runnable without the three v0 per-β ensemble
directories. The v1 aggregation runner must validate them as inputs
before any smearing or scoring:

| Required input | Source |
| --- | --- |
| v0 per-β ensemble at β=2.0 | `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.0_ensemble_v0/` |
| v0 per-β ensemble at β=2.4 | `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.4_ensemble_v0/` |
| v0 per-β ensemble at β=2.8 | `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.8_ensemble_v0/` |

For each ensemble directory the runner must:

1. read and assert presence of `manifest.json`, `summary.json`,
   `configs/su2_links.jsonl`, `signatures/signature_vectors.csv`,
   `heldout/heldout_loop_values.csv`, `heldout/heldout_summary.csv`,
   `hashes.json`;
2. assert `summary.json` reports the Phase-1-inherited ensemble
   health thresholds passed (burn-in `>= 2000`, `τ_int <= 16`,
   `thinning/τ_int >= 2`, fallback `<= 0.001`, unitarity `<= 1e-10`,
   orientation spread `<= 5e-2`);
3. assert the per-ensemble `configs/su2_links.jsonl` SHA-256 matches
   the value recorded in the v0 ensemble's `hashes.json`;
4. record the read ensemble's manifest commit, dirty flag, and result
   directory in the v1 aggregation runtime manifest as
   `v0EnsembleSources[]`.

If any of the above fails, the run is a `Z void_run`.

## Smearing Application

For each retained configuration in each v0 per-β ensemble:

1. read the bare `su2_links` from `configs/su2_links.jsonl`;
2. apply APE smearing per the algorithm and projection in the P0
   amendment, with `α = 0.5`, `N_sm = 10`, synchronous update;
3. after every iteration, record:
   - max `|det(M) - 1|` across all post-projection link matrices in
     this iteration;
   - max `||M·M^† - I||_F` (link-unitarity Frobenius residual);
4. retain only the post-`N_sm = 10` smeared link state per
   configuration (intermediate iterations are not stored; only the
   per-iteration health maxes are recorded).

The smearing-health maxes across all 96 configurations × 10 iterations
× ~5184 links are reduced to two scalars per (β, config):
`max_det_drift` and `max_unitarity_residual`. These are aggregated to
ensemble-level maxes and gated per the P0 amendment.

## Signature Vocabulary v4

For each smeared configuration, the v4 signature is the same shape as
v1 but computed on smeared links:

```text
sig = ( W11_mean, W11_var,
        W12_mean, W12_var,
        W13_mean, W13_var,
        W22_mean, W22_var )
```

where each `Wnm_*` is the position-and-orientation average of
`(1/2) Re Tr U_loop^(smeared)` for an `n × m` rectangular Wilson loop
over the three plane orientations `{xy, xz, yz}` and all base
positions in the 12³ lattice, using smeared link products. Gauge
invariance is preserved by APE construction.

The runner must also recompute the **bare** v1 signature on the same
configurations for two reasons: (a) to confirm that the bare signature
agrees with the value v0 already emitted (a one-shot integrity check),
and (b) to enable a `bare-vs-smeared` signature delta to be reported in
the v1 receipt notes for diagnostic purposes only (not a gate).

## Held-Out Target And Bin Edges

The held-out target vocabulary v1 is unchanged from v0:

```text
heldout = ( W14_mean, W23_mean, W33_mean )   on bare links
γ_held  = - LS slope of ln(max(Wnm, 1e-10)) vs area
          over (A=4, A=6, A=9)
```

The runner reads `heldout/heldout_summary.csv` from each v0 per-β
ensemble dir; γ_held values are not recomputed (the v0 numerics are
the inputs).

Per-β tertile bin edges are recomputed from the same γ_held data by
the same linear-interpolation percentile convention as v0. The runner
must assert that the recomputed edges match v0's bin edges to machine
epsilon (max abs difference `<= 1e-12` per edge), reading v0's edges
from
`results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v0/aggregation/per_beta_bin_edges.json`.
A mismatch is a `Z void_run` (would indicate input drift).

The same applies to the global tertile edges from
`global_bin_edges.json`.

## Signature Distance Metric

Identical to v0 §"Signature Distance Metric": Euclidean L2 in
z-score-normalized 8-dim smeared-signature space; per-β normalization
for the within-β lane; combined-96 normalization for the across-β
lane. Normalization means and standard deviations are recomputed on
the **smeared** signatures (they will be numerically different from
v0's bare-signature normalizations); the values are recorded in the v1
aggregation receipt's `signature_normalization.json`.

`k` slate `{3, 5, 10}`; primary k=5; gate at k=5 only.

## Rank-Locality Scoring Metric

Identical to v0: bin-purity@k, discrimination ratio against chance
1/3, bootstrap 95% CI with B=1000 within-β resamples. Kendall τ
secondary metric.

## Controls Battery

Identical to v0 §"Controls Battery" with one clarification:

- `CTRL_RAW` in v1 means **NN graph using flat raw matrix-entry
  vectors of the bare (un-smeared, un-gauge-fixed) link variables**.
  This is unchanged from v0. The point of `CTRL_RAW` in v1 is to
  confirm that the smeared signature is doing something different
  from the raw bare-link representation; if `CTRL_RAW` matches or
  beats the smeared primary at the same margin, that is a
  `YM-P2-NEG-D raw_dominates` named null (smearing did not move the
  signal to a meaningfully different place).
- `CTRL_GAUGE_RAND` in v1 applies a random SU(2) Haar site-gauge
  transform to the **bare** link variables and re-runs the smearing
  + signature pipeline on the gauge-transformed bare configs. APE
  smearing preserves gauge invariance, so the resulting smeared
  signature must equal the original smeared signature to numerical
  precision. The gate `<= 1e-12` max abs residual is preserved; in
  practice the smeared pipeline accumulates more rounding than the
  bare one (each smearing iteration adds matrix product error), so
  the observed residual will be larger than v0's (rough estimate:
  `~ 1e-13` for `N_sm = 10` on 12³, still well under 1e-12 gate
  with margin). If the v1 observed residual exceeds 1e-12, this is
  `YM-P1-NEG-A gauge_leakage`, not a passive observation.

## Pass / Quarantine Thresholds

All point-estimate gates evaluated at `k = 5` on the within-β primary
lane unless otherwise noted.

| Quantity | Registered threshold | Branch if missed |
| --- | --- | --- |
| Three v0 ensemble dirs present + healthy | summary.json health passed; configs hash matches v0 hashes.json | `Z void_run` |
| Per-β tertile bin edges match v0 to machine epsilon | max abs edge difference `<= 1e-12` | `Z void_run` |
| Smearing drift (max `|det(M) - 1|`) per iteration | `<= 1e-10` | `YM-P2-QUAR-E smearing_drift` |
| Post-smearing-step link unitarity max Frobenius residual | `<= 1e-10` | `YM-P2-QUAR-E smearing_drift` |
| Per-orientation smeared mean-plaquette relative spread | `<= 5e-2` between any two of `{xy, xz, yz}` | `YM-P2-QUAR-C orientation_anisotropy` (existing P1 branch, extended to smeared) |
| Bare-signature recomputation integrity vs v0 | per-component max abs residual `<= 1e-12` | `Z void_run` (would indicate runner bug) |
| Primary within-β `mean_bin_purity_5` | `>= 0.5` (discrimination_ratio `>= 1.5`) | `YM-P2-NEG-A no_rank_local_structure` |
| Primary beats `CTRL_RAND` mean bin-purity | margin `>= 0.10` | `YM-P2-NEG-A no_rank_local_structure` |
| Primary beats `CTRL_META` mean bin-purity | margin `>= 0.10` | `YM-P2-NEG-B metadata_only` |
| Primary beats `CTRL_RAW` mean bin-purity | margin `>= 0.10` | `YM-P2-NEG-D raw_dominates` |
| `CTRL_PERM` mean bin-purity | within `0.05` of chance `1/3` | `Z graph_contamination` |
| `CTRL_GAUGE_RAND` recovers primary bin-purity to machine epsilon | match `<= 1e-12` in mean bin-purity | `YM-P1-NEG-A gauge_leakage` |
| Across-β primary `mean_bin_purity_5` beats `CTRL_RAND_STRAT` | margin `>= 0.05` | `YM-P2-NEG-C coupling_triviality` |
| Aggregation wall clock | `<= 10 minutes` | `Z void_run` |
| Missing runtime manifest field | none missing | `Z void_run` |

## Output Contract

Exact result directory:

```text
results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v1/
```

Required files (mirrors v0 aggregation output with smearing-specific
additions):

| Path under result directory | Role |
| --- | --- |
| `aggregation/manifest.json` | aggregation runtime manifest with locked fields, code commit, dirty flag, command line, wall clock, v0 ensemble sources, hashes |
| `aggregation/v0_ensemble_sources.json` | list of three v0 per-β ensemble dirs + their `configs/su2_links.jsonl` SHA-256 |
| `aggregation/smearing_health.csv` | per-(β, config, iteration) `max_det_drift` and `max_unitarity_residual` |
| `aggregation/smeared_signature_vectors.csv` | 96 v4 smeared signature vectors |
| `aggregation/bare_signature_integrity.csv` | bare signature recomputed on the same configs, per-component diff vs v0 emitted values |
| `aggregation/per_beta_bin_edges.json` | per-β tertile edges (asserted equal to v0's) |
| `aggregation/global_bin_edges.json` | global tertile edges (asserted equal to v0's) |
| `aggregation/signature_normalization.json` | z-score means/stds for smeared signature, both lanes |
| `aggregation/within_beta_nn_graphs.json` | per-β NN graphs on smeared signature at k=10 |
| `aggregation/across_beta_nn_graph.json` | combined NN graph on smeared signature |
| `aggregation/control_nn_graphs/<control_id>.json` | per-control NN graphs (six controls scored) |
| `aggregation/rank_locality_scores.csv` | one row per (lane, control_or_primary, k) with mean bin-purity, discrimination ratio, bootstrap 95% CI |
| `aggregation/kendall_tau.csv` | secondary metric, per-β + aggregated |
| `aggregation/v0_vs_v1_comparison.json` | per-(lane, control, k) diff between v0 and v1 mean bin-purity, for diagnostic interpretation |
| `aggregation/branch_inputs.json` | every gate's observed value and pass/fail flag |
| `aggregation/summary.json` | final Phase 2 v1 verdict |
| `aggregation/hashes.json` | SHA-256 hashes (excluding itself) |

Minimum `aggregation/manifest.json` fields:

```json
{
  "phase": "phase2",
  "phaseVersion": "v1",
  "cell": "SU2_3D",
  "latticeSize": [12, 12, 12],
  "betaSlate": [2.0, 2.4, 2.8],
  "perBetaConfigurations": 32,
  "totalConfigurations": 96,
  "signatureVocabularyVersion": "v4",
  "heldOutTargetVocabularyVersion": "v1",
  "p0Amendment": "P0_AMD_001_APE_SMEARING_2026-05-29",
  "smearingAlgorithm": "APE",
  "smearingAlpha": 0.5,
  "smearingIterations": 10,
  "smearingProjection": "complex_sqrt_det_branch_positive_realtr",
  "gammaHeldEpsilonFloor": 1e-10,
  "binConvention": "per_beta_tertile_linear",
  "distanceMetric": "euclidean_zscore",
  "kSlate": [3, 5, 10],
  "primaryK": 5,
  "bootstrapResamples": 1000,
  "controlsScored": [
    "CTRL_META",
    "CTRL_RAW",
    "CTRL_RAND",
    "CTRL_RAND_STRAT",
    "CTRL_PERM",
    "CTRL_GAUGE_RAND"
  ],
  "controlsDeferred": ["CTRL_FINITE_SIZE"],
  "v0EnsembleSources": null,
  "v0BinEdgesSource": "results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v0/aggregation/per_beta_bin_edges.json",
  "maxSmearingDetDrift": null,
  "maxSmearingUnitarityResidual": null,
  "perOrientationSmearedSpread": null,
  "codeCommit": null,
  "gitDirty": null,
  "commandLine": null,
  "wallClockSeconds": null
}
```

## Aggregation Invocation Manifest

The v1 aggregation runner consumes the three v0 per-β ensemble
directories and produces the Phase 2 v1 receipt inputs. It must use
the exact command line below.

```text
node scripts/yang-mills-phase2-v1-su2-3d-aggregate.mjs --cell SU2_3D --lattice-size 12x12x12 --beta-slate 2.0,2.4,2.8 --p0-amendment P0_AMD_001_APE_SMEARING_2026-05-29 --smearing-algorithm APE --smearing-alpha 0.5 --smearing-iterations 10 --in-beta-2.0 results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.0_ensemble_v0 --in-beta-2.4 results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.4_ensemble_v0 --in-beta-2.8 results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.8_ensemble_v0 --v0-bin-edges results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v0/aggregation/per_beta_bin_edges.json --distance-metric euclidean_zscore --k-slate 3,5,10 --primary-k 5 --bootstrap-resamples 1000 --bin-convention per_beta_tertile_linear --gauge-rand-seed-tag phase2_v1_aggregation --gauge-transforms-per-config 1 --out results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v1
```

Package script:

```text
npm run yang-mills:phase2:v1:su2-3d:aggregate
```

The aggregation runner must, in this exact order:

1. validate the three v0 ensemble dirs (manifests, health, configs
   hash);
2. read the 96 bare configurations and the 96 bare γ_held values;
3. read the v0 per-β bin edges file; verify the file timestamp is
   earlier than the v1 runner start time;
4. apply APE smearing per the cited P0 amendment to all 96 configs;
   record smearing-health maxes;
5. compute smeared v4 signature vectors;
6. recompute bare v1 signature vectors and assert agreement with v0
   to `<= 1e-12` per component;
7. compute per-β tertile edges from γ_held and assert match with v0
   edges to `<= 1e-12`;
8. compute z-score normalization on smeared signatures, write
   `signature_normalization.json`;
9. compute within-β and across-β NN graphs at k=10 on smeared
   signatures;
10. compute the six scored control NN graphs;
11. compute rank-locality bin-purity@k for k ∈ {3, 5, 10};
12. compute bootstrap 95% CIs;
13. compute Kendall τ secondary;
14. compute `v0_vs_v1_comparison.json` for diagnostic interpretation;
15. evaluate every gate; write `branch_inputs.json`, `summary.json`;
16. write `hashes.json` last.

Steps 3 and 7 are the bin-edge defense; if either fails the run is a
`Z void_run` and the receipt cannot be cited.

## Branch Table

| Branch | Trigger | Disposition |
| --- | --- | --- |
| `P2-A bounded_positive` | every Pass / Quarantine threshold passes on smeared signature | bounded relative-locality receipt at SU2_3D 12³ × β slate × **vocab v4 (smeared)**; Phase 3 manifest admitted |
| `YM-P2-NEG-A no_rank_local_structure` | smeared primary fails the 0.5 / 0.10-margin gate against RAND | named null; pushes weight onto "small-loop summaries don't carry γ_held information at all on this cell"; next probe spec must redesign target or signature class |
| `YM-P2-NEG-B metadata_only` | CTRL_META meets or beats smeared primary | named null; bin convention or scoring may need rework |
| `YM-P2-NEG-C coupling_triviality` | across-β primary fails RAND_STRAT margin | named null |
| `YM-P2-NEG-D raw_dominates` | CTRL_RAW matches or beats smeared primary at the margin | named null; smearing did not move the signal to a meaningfully different place |
| `YM-P1-NEG-A gauge_leakage` | CTRL_GAUGE_RAND breach > 1e-12 in mean bin-purity | quarantine; smearing or signature implementation rechecked |
| `YM-P2-QUAR-C orientation_anisotropy` | per-orientation smeared mean-plaquette spread > 5e-2 | quarantine; smeared signature isotropy bug |
| `YM-P2-QUAR-E smearing_drift` (new from P0 amendment) | det drift or post-projection unitarity > 1e-10 | quarantine; smearing projection bug |
| `Z graph_contamination` | CTRL_PERM != chance within 0.05 | aggregation runner bug |
| `Z void_run` | missing field, command drift, compute-cap breach, manifest/hash drift, v0 ensemble drift, bin-edge drift | output cannot be cited as a receipt |

The v1 receipt is filed as exactly one branch.

## Compute Cap

- Smearing: 96 configs × 10 iterations × 5184 links × few µs per
  link update ≈ 60-90 seconds total;
- Signature recomputation (bare + smeared): ~10 seconds;
- NN graphs + controls + bootstrap: ~30 seconds;
- Total estimated wall clock: ~2-3 minutes. Cap is 10 minutes per
  the P0 lock.

## Next Allowed Step

After this spec is filed, the next admitted engineering moves are, in
this order:

1. Implement `scripts/yang-mills-phase2-v1-su2-3d-aggregate.mjs`. The
   smearing routine lives in a new shared module
   `scripts/lib/yang-mills-su2-3d-smearing.mjs` to keep the aggregation
   entry focused on validation + scoring; the bare SU(2) 3D core at
   `scripts/lib/yang-mills-su2-3d-core.mjs` remains bit-for-bit
   unchanged. The v0 aggregation runner remains bit-for-bit unchanged.
2. Wire the npm script `yang-mills:phase2:v1:su2-3d:aggregate`.
3. Execute the aggregation invocation and file a single Phase 2 v1
   receipt under
   `docs/yang-mills/receipts/YYYY-MM-DD_SU2_3D_phase2_v1_<short-verdict>.md`
   citing the v1 aggregation directory and the three v0 ensemble dirs.
4. If the verdict is `P2-A bounded_positive`, draft
   `docs/prereg/yang-mills/PHASE3_SU2_3D_observable_certificate_v0.md`
   per P0 §8 Phase 3.
5. If the verdict is any `YM-P2-NEG-*` named null, file a new dated
   probe spec under `docs/yang-mills/specs/` proposing a v2 design
   change. Per the probe spec's "What A v1 Outcome Tells Us" table,
   the v2 design space depends on which null fired and what the
   `v0_vs_v1_comparison.json` shows.

Phase 4 (finite-size split against the 8³ partner) and Phase 5
(external-review packet) remain gated as in P0 §8.
