# Yang-Mills Phase 2 v3 - SU2 3D Relative-Locality Certificate Spec (σ²_W33 Target)

Status: **Phase 2 v3 spec filed 2026-05-29**. This is a pre-run binding
spec, not a result. It admits the v3 aggregation runner only after the
implementation reads the three v0 per-β ensemble directories, reuses
v0's emitted v1 signature vectors, computes the new held-out vocabulary
v2 (`σ²_W33`) on bare configurations, and emits the runtime manifest
fields named below. It admits no Yang-Mills, Phase 3
observable-certificate, Phase 4 finite-size, continuum, confinement, or
mass-gap claim. A v3 pass is a **vocab-v1 signature × vocab-v2
held-out target** relative-locality receipt, scoped accordingly.

P0 lock: [`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md)
v0 spec: [`PHASE2_SU2_3D_relative_locality_v0.md`](PHASE2_SU2_3D_relative_locality_v0.md)
v1 spec: [`PHASE2_SU2_3D_relative_locality_v1.md`](PHASE2_SU2_3D_relative_locality_v1.md)
v2 spec: [`PHASE2_SU2_3D_relative_locality_v2.md`](PHASE2_SU2_3D_relative_locality_v2.md)
v0 receipt:
[`../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md)
v1 receipt:
[`../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md)
v2 receipt:
[`../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md)
Triggering probe spec:
[`../../yang-mills/specs/2026-05-29_phase2_v3_target_redesign_probe.md`](../../yang-mills/specs/2026-05-29_phase2_v3_target_redesign_probe.md)

This spec does **not** invoke P0 amendment 1 (APE smearing): v3 uses
the bare-link v0 signature unchanged. v3 also does **not** trigger any
new P0 amendment, because the held-out vocabulary v2 (per-config
spatial variance of W33) uses the same held-out loop set
`{W14, W23, W33}` as P0's locked target vocabulary v1, only with a
different per-configuration summary statistic.

## Claim Under Test

Inside the `SU2_3D` 12×12×12 primary cell, over the registered β slate
`{2.0, 2.4, 2.8}` with 32 retained configurations per β taken from
the v0 ensembles, the unchanged **vocab v1 bare 8-dim signature**
should preserve the per-config **σ²_W33 spatial-variance tertile
label (held-out vocab v2)** in nearest-neighbor rank space, **beyond**
every metadata, raw-link, random, β-stratified random,
label-permutation, and gauge-randomized control on the same frozen
neighbor graphs and frozen bin edges.

This is the fourth Phase 2 read on this cell. v0 (signature vocab v1,
target γ_held LS slope) → NEG-A. v1 (signature vocab v4 smeared,
target γ_held) → NEG-A. v2 (signature vocab v5 correlator, target
γ_held) → NEG-A. v3 (**signature vocab v1**, **target vocab v2
σ²_W33**) tests whether the small-loop signature space resolves a
non-area-law held-out observable class even though it failed on the
area-law-mean class.

A v3 pass is bounded by the same lattice / β envelope as v0 / v1 / v2:
cell-local (`SU2_3D`, 12³), β-slate-local, signature-class-local
(vocab v1), target-class-local (vocab v2). It is **not** a Yang-Mills
result, a confinement proof, a mass-gap claim, a continuum statement,
or evidence at any β, lattice size, signature class, or target class
outside the registered envelope. A v3 null is filed as
`YM-P2-NEG-A` / `NEG-B` / `NEG-C` / `NEG-D` per the v0 / v1 / v2
branch table.

## Scope

In scope at this v3:

- vocab v1 signature unchanged (re-read from v0 ensemble outputs);
- **new** vocab v2 held-out target: per-config σ²_W33 spatial
  variance across all `12³ × 3 = 5184` (position × orientation)
  samples;
- per-β tertile bin edges on σ²_W33 (new; not the same as v0's γ_held
  bin edges);
- within-β k-NN rank-locality scoring as the **primary** read;
- across-β k-NN rank-locality scoring as the coupling-triviality
  cross-check;
- six controls scored (`CTRL_META`, `CTRL_RAW`, `CTRL_RAND`,
  `CTRL_RAND_STRAT`, `CTRL_PERM`, `CTRL_GAUGE_RAND`) on the new
  target labels;
- one control declared but deferred (`CTRL_FINITE_SIZE`, gated to
  Phase 4);
- bootstrap 95% CIs on every mean bin-purity score; v3 gates remain
  point-estimate.

Out of scope at this v3:

- new per-β ensemble generation (v3 reuses v0 ensembles bit-for-bit);
- any signature vocab other than v1 (vocab v4 smeared and v5
  correlator are NOT invoked at v3; combining smearing/correlator with
  the new target would be a compound design change and is deferred to
  later vN if v3 lands NEG-A);
- any held-out summary other than σ²_W33 (σ²_W14, σ²_W23 are NOT
  scored at v3; they are pre-stated v4 fallback options in the probe
  spec);
- assertion against v0's per-β bin edges — v3 has its own bins
  computed from σ²_W33, NOT from γ_held;
- any β value or lattice size outside the registered v0 envelope;
- Phase 3, Phase 4, Phase 5 work.

## Inputs From v0

| Required input | Source |
| --- | --- |
| v0 per-β ensemble at β=2.0 | `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.0_ensemble_v0/` |
| v0 per-β ensemble at β=2.4 | `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.4_ensemble_v0/` |
| v0 per-β ensemble at β=2.8 | `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.8_ensemble_v0/` |
| v0 emitted v1 signature vectors | each ensemble's `signatures/signature_vectors.csv` |

For each ensemble directory the runner must validate manifest, summary,
configs JSONL, hashes (matching v0's recorded hashes), and Phase 1
ensemble-health passing. Any failure → `Z void_run`.

The runner must also **re-read v0's emitted v1 signatures** from
`signatures/signature_vectors.csv` and use them directly as the v3
signature input — the v3 runner does NOT recompute v1 signatures. This
freezes the signature side: any difference in v3 vs v0 outcomes is
attributable entirely to the target redesign, not to floating-point
drift in signature recomputation.

The runner asserts the SHA-256 of the re-read v0 signature CSV matches
the value recorded in the v0 ensemble's `hashes.json`. Mismatch →
`Z void_run`.

## Signature Vocabulary

Unchanged from v0: vocab v1 bare 8-dim mean+var of
`{W11, W12, W13, W22}` position-and-orientation averaged. Re-read
from v0 outputs (not recomputed). No smearing, no correlators.

## Held-Out Target Vocabulary v2

Locked at the v3 probe spec; restated here for self-containment:

```text
For each retained configuration:

  S(x, o) = (1/2) Re Tr (ordered product of 12 links around the
                          3×3 Wilson loop anchored at base position
                          x in plane orientation o)

  μ(config)      = (1 / 5184) Σ_{x, o} S(x, o)
  σ²_W33(config) = (1 / 5184) Σ_{x, o} (S(x, o) − μ(config))²

where x runs over all 12³ = 1728 base positions and o runs over
the three plane orientations {xy, xz, yz}.

Biased variance estimator (divisor 5184, no Bessel correction).
```

Per-β tertile bin edges are recomputed in the v3 aggregation from the
σ²_W33 values (linear-interpolation percentile, exactly as v0 / v1 /
v2 did for γ_held). The edges are **new** and are NOT asserted equal
to v0's per-β bin edges (which were for γ_held, a different summary).
The runner writes the new edges to
`aggregation/per_beta_v3_bin_edges.json` **before** any NN scoring
happens. Aggregator must re-read at the NN step to defend against
in-memory edits.

Across-β tertile edges: same convention on the combined 96-config
σ²_W33 distribution.

## Signature Distance Metric

Identical to v0 §"Signature Distance Metric": Euclidean L2 in
z-score-normalized 8-dim signature space; per-β normalization from
that β's 32-config sample; across-β normalization on combined 96-
config sample. Normalization parameters re-computed in v3 from the
re-read v0 signature vectors (which are bit-for-bit unchanged) and
recorded in `aggregation/signature_normalization.json`.

`k` slate `{3, 5, 10}`; primary k=5; gate at k=5 only.

## Rank-Locality Scoring Metric

Identical to v0 / v1 / v2: bin-purity@k, discrimination ratio against
chance 1/3, bootstrap 95% CI with B=1000 within-β resamples. Kendall
τ secondary metric (now between signature distance rank and
|Δσ²_W33| rank, not |Δγ_held| rank).

## Controls Battery

Identical to v0 / v1 / v2 with one clarification:

- `CTRL_GAUGE_RAND` in v3 applies a random SU(2) Haar site-gauge
  transform to the bare link variables and re-runs **both** the v1
  signature pipeline (recompute mean+var of small loops on
  gauge-transformed bare configs) **and** the σ²_W33 target pipeline
  (recompute W33(x, o) across all positions and orientations on
  gauge-transformed bare configs, then take the variance). Both
  pipelines are gauge-invariant by construction; the gate is
  `<= 1e-12` max abs residual in mean bin-purity between the original
  and gauge-randomized NN graphs. A breach here is `YM-P1-NEG-A
  gauge_leakage`, not a passive observation.

## Pass / Quarantine Thresholds

All point-estimate gates evaluated at `k = 5` on the within-β primary
lane unless otherwise noted.

| Quantity | Registered threshold | Branch if missed |
| --- | --- | --- |
| Three v0 ensemble dirs present + healthy | summary.json health passed; configs hash matches v0 hashes.json | `Z void_run` |
| v0 signature CSV SHA-256 matches v0 hashes.json | exact match | `Z void_run` |
| Per-β σ²_W33 tertile edges written before scoring | non-empty `per_beta_v3_bin_edges.json` with timestamp earlier than first scoring artifact | `Z void_run` |
| Per-config σ²_W33 spread within β | spread `>= 1e-12` (degenerate-variance guard) | `Z bin_degenerate` for that β |
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
results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v3/
```

Required files:

| Path under result directory | Role |
| --- | --- |
| `aggregation/manifest.json` | aggregation runtime manifest with locked fields, code commit, dirty flag, command line, wall clock, v0 ensemble sources, hashes |
| `aggregation/v0_ensemble_sources.json` | list of three v0 per-β ensemble dirs + their `configs/su2_links.jsonl` SHA-256 and `signatures/signature_vectors.csv` SHA-256 |
| `aggregation/v3_target_summary.csv` | 96 per-config σ²_W33 values (with per-config μ also reported) |
| `aggregation/per_beta_v3_bin_edges.json` | per-β tertile edges on σ²_W33, frozen before scoring |
| `aggregation/global_v3_bin_edges.json` | global tertile edges on σ²_W33 |
| `aggregation/signature_normalization.json` | z-score means/stds for v1 signature (both lanes); will match v0's if v0 included this output, but recomputed here for self-containment |
| `aggregation/within_beta_nn_graphs.json` | per-β NN graphs on v1 signature at k=10 |
| `aggregation/across_beta_nn_graph.json` | combined NN graph on v1 signature |
| `aggregation/control_nn_graphs/<control_id>.json` | per-control NN graphs (six controls scored) |
| `aggregation/rank_locality_scores.csv` | one row per (lane, control_or_primary, k) with mean bin-purity, discrimination ratio, bootstrap 95% CI |
| `aggregation/kendall_tau.csv` | secondary metric, per-β + aggregated |
| `aggregation/v0_v3_comparison.json` | per-(lane, control, k) diff between v0 and v3 mean bin-purity, for diagnostic interpretation |
| `aggregation/branch_inputs.json` | every gate's observed value and pass/fail flag |
| `aggregation/summary.json` | final Phase 2 v3 verdict |
| `aggregation/hashes.json` | SHA-256 hashes (excluding itself) |

Minimum `aggregation/manifest.json` fields:

```json
{
  "phase": "phase2",
  "phaseVersion": "v3",
  "cell": "SU2_3D",
  "latticeSize": [12, 12, 12],
  "betaSlate": [2.0, 2.4, 2.8],
  "perBetaConfigurations": 32,
  "totalConfigurations": 96,
  "signatureVocabularyVersion": "v1",
  "signatureSource": "v0_reread",
  "heldOutTargetVocabularyVersion": "v2",
  "heldOutTargetSummary": "spatial_variance_W33",
  "heldOutTargetSampleCountPerConfig": 5184,
  "heldOutVarianceEstimator": "biased",
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
  "codeCommit": null,
  "gitDirty": null,
  "commandLine": null,
  "wallClockSeconds": null
}
```

## Aggregation Invocation Manifest

The v3 aggregation runner consumes the three v0 per-β ensemble
directories and produces the Phase 2 v3 receipt inputs. It must use
the exact command line below.

```text
node scripts/yang-mills-phase2-v3-su2-3d-aggregate.mjs --cell SU2_3D --lattice-size 12x12x12 --beta-slate 2.0,2.4,2.8 --signature-vocab v1 --signature-source v0_reread --heldout-vocab v2 --heldout-summary spatial_variance_W33 --heldout-variance-estimator biased --in-beta-2.0 results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.0_ensemble_v0 --in-beta-2.4 results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.4_ensemble_v0 --in-beta-2.8 results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.8_ensemble_v0 --distance-metric euclidean_zscore --k-slate 3,5,10 --primary-k 5 --bootstrap-resamples 1000 --bin-convention per_beta_tertile_linear --gauge-rand-seed-tag phase2_v3_aggregation --gauge-transforms-per-config 1 --out results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v3
```

Package script:

```text
npm run yang-mills:phase2:v3:su2-3d:aggregate
```

The aggregation runner must, in this exact order:

1. validate the three v0 ensemble dirs (manifests, health, configs
   hash, signatures CSV hash);
2. read the 96 bare configurations and the 96 v1 signature vectors
   (signature vectors are read from v0's emitted CSV, not recomputed);
3. for each configuration: compute `W33(x, o)` across all 12³ × 3
   positions × orientations using
   `scripts/lib/yang-mills-su2-3d-core.mjs`; compute per-config
   `σ²_W33` via the biased variance estimator;
4. write `v3_target_summary.csv` with per-config (μ, σ²_W33);
5. compute per-β tertile edges from σ²_W33; write
   `per_beta_v3_bin_edges.json`;
6. compute global tertile edges; write `global_v3_bin_edges.json`;
7. assert both edge files have timestamps earlier than the first
   `nn_graphs` write;
8. compute z-score normalization on the re-read v1 signatures; write
   `signature_normalization.json`;
9. compute within-β and across-β NN graphs at k=10 on v1 signatures;
10. compute the six scored control NN graphs;
11. compute rank-locality bin-purity@k for k ∈ {3, 5, 10} against
    the new v2 σ²_W33 labels;
12. compute bootstrap 95% CIs;
13. compute Kendall τ secondary;
14. compute `v0_v3_comparison.json` for diagnostic interpretation;
15. evaluate every gate; write `branch_inputs.json`, `summary.json`;
16. write `hashes.json` last.

## Branch Table

| Branch | Trigger | Disposition |
| --- | --- | --- |
| `P2-A bounded_positive` | every Pass / Quarantine threshold passes against the σ²_W33 target | bounded relative-locality receipt at SU2_3D 12³ × β slate × **vocab v1 signature × vocab v2 target (σ²_W33)**; Phase 3 manifest admitted |
| `YM-P2-NEG-A no_rank_local_structure` | v1 signature fails on σ²_W33 too | named null; four consistent NEG-As; routes to v4 per probe spec's pre-stated fallback (likely PAUSE-and-synthesize) |
| `YM-P2-NEG-B metadata_only` | CTRL_META meets or beats primary on σ²_W33 | named null; bin convention may need rework |
| `YM-P2-NEG-C coupling_triviality` | across-β primary fails RAND_STRAT margin | named null |
| `YM-P2-NEG-D raw_dominates` | CTRL_RAW matches or beats primary | named null; signature class too coarse even for non-area-law targets |
| `YM-P1-NEG-A gauge_leakage` | CTRL_GAUGE_RAND breach > 1e-12 | quarantine; signature or target implementation rechecked |
| `Z graph_contamination` | CTRL_PERM != chance within 0.05 | aggregation runner bug |
| `Z bin_degenerate` (per β) | β's σ²_W33 distribution has spread < 1e-12 | that β contributes no primary score; partial coverage reported |
| `Z void_run` | missing field, command drift, compute-cap breach, manifest/hash drift, v0 ensemble drift, bin-edge timestamp drift | output cannot be cited as a receipt |

The v3 receipt is filed as exactly one branch.

## Compute Cap

- Per-configuration W33 evaluation over `12³ × 3 = 5184` (position ×
  orientation) samples: 8-link matrix products per sample → modest
  per-config cost, ~few seconds for 96 configs;
- σ²_W33 reduction: O(N_samples) per config, negligible;
- NN graphs + controls + bootstrap on re-read 8-dim v1 signatures:
  same shape as v0, ~30-60 seconds;
- Total estimated wall clock: ~1-2 minutes. Cap is 10 minutes per
  the P0 lock.

## Next Allowed Step

After this spec is filed, the next admitted engineering moves are, in
this order:

1. Implement `scripts/yang-mills-phase2-v3-su2-3d-aggregate.mjs`. The
   per-position W33 evaluator lives in
   `scripts/lib/yang-mills-su2-3d-core.mjs` (already present from
   Phase 1 SU(2) 3D); the v3 runner imports and calls it for the
   target computation. No new shared module needed. v0, v1, v2
   aggregation runners + bare SU(2) 3D core + smearing module +
   correlator module all remain bit-for-bit unchanged.
2. Wire the npm script `yang-mills:phase2:v3:su2-3d:aggregate`.
3. Execute the aggregation invocation and file a single Phase 2 v3
   receipt under
   `docs/yang-mills/receipts/YYYY-MM-DD_SU2_3D_phase2_v3_<short-verdict>.md`
   citing the v3 aggregation directory and the three v0 ensemble dirs.
4. If the verdict is `P2-A bounded_positive`, draft
   `docs/prereg/yang-mills/PHASE3_SU2_3D_observable_certificate_v0.md`
   per P0 §8 Phase 3.
5. If the verdict is `YM-P2-NEG-A` again, file the **bounded-null
   synthesis receipt** per the v3 probe spec's pre-stated v4 fallback
   table; the lane sits at "bounded null on this envelope across four
   pre-registered probes" instead of continuing the probe ladder
   further within this cell. σ²_W14, σ²_W23, Polyakov-target probes
   remain admissible as future-dated specs if external context (e.g.
   a Polyakov-relevant scientific question) motivates them, but they
   are not the default next step.

Phase 4 and Phase 5 remain gated as in P0 §8.
