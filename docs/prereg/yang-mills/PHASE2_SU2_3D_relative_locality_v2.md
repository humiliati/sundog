# Yang-Mills Phase 2 v2 - SU2 3D Relative-Locality Certificate Spec (Connected Correlator)

Status: **Phase 2 v2 spec filed 2026-05-29**. This is a pre-run binding
spec, not a result. It admits the v2 aggregation runner only after the
implementation reads the three v0 per-β ensemble directories, computes
the v5 connected-correlator signature, and emits the runtime manifest
fields named below. It admits no Yang-Mills, Phase 3
observable-certificate, Phase 4 finite-size, continuum, confinement, or
mass-gap claim. A v2 pass is a **vocab v5 connected-correlator**
relative-locality receipt, scoped accordingly.

P0 lock: [`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md)
v0 spec: [`PHASE2_SU2_3D_relative_locality_v0.md`](PHASE2_SU2_3D_relative_locality_v0.md)
v1 spec: [`PHASE2_SU2_3D_relative_locality_v1.md`](PHASE2_SU2_3D_relative_locality_v1.md)
v0 receipt:
[`../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md)
v1 receipt:
[`../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md)
Triggering probe spec:
[`../../yang-mills/specs/2026-05-29_phase2_v2_correlator_probe.md`](../../yang-mills/specs/2026-05-29_phase2_v2_correlator_probe.md)
Receipt template:
[`../../yang-mills/RECEIPT_TEMPLATE.md`](../../yang-mills/RECEIPT_TEMPLATE.md)

This spec does **not** invoke P0 amendment 1 (APE smearing): v2 uses
bare-link Wilson loops, not smeared. v2 also does **not** trigger any
new P0 amendment, because connected 2-point correlators of bare loops
are a new vocabulary class within the P0 fixed-loop framework, not
smearing or blocking.

## Claim Under Test

Inside the `SU2_3D` 12×12×12 primary cell, over the registered β slate
`{2.0, 2.4, 2.8}` with 32 retained configurations per β taken from
the v0 ensembles, the **v5 bare-link connected 2-point correlator**
small-Wilson-loop signature should preserve the per-config bare-link
held-out-Wilson-loop area-law-decay tertile label (`γ_held`-bin) in
nearest-neighbor rank space, **beyond** every metadata, raw-link,
random, β-stratified random, label-permutation, and gauge-randomized
control on the same frozen neighbor graphs and frozen bin edges.

This is the third Phase 2 read on this cell. v0 (vocab v1, bare-loop
mean/var) → NEG-A. v1 (vocab v4, smeared-loop mean/var) → NEG-A. v2
(vocab v5, bare-loop connected 2-point correlators at frozen
displacement slate) tests the last unexplored small-loop richness
option before pivoting to target-side redesign.

A v2 pass is bounded by the same envelope as v0 / v1: cell-local
(`SU2_3D`, 12³), β-slate-local, signature-class-local (vocab v5,
bare-link connected correlators at the frozen displacement slate). It
is **not** a Yang-Mills result, a confinement proof, a mass-gap
claim, a continuum statement, or evidence at any β, lattice size,
displacement slate, or signature class outside the registered
envelope. A v2 null is filed as `YM-P2-NEG-A` / `NEG-B` / `NEG-C` /
`NEG-D` per the v0 / v1 branch table.

## Scope

In scope at this v2:

- v5 connected-correlator signature on bare-link Wilson loops
  `{W11, W12, W13, W22}` at the five locked cubic-symmetry
  displacement classes `{r1, r2, r3, r4, r5}` from the probe spec;
- v1 held-out target unchanged (bare-link `{W14, W23, W33}` →
  `γ_held`);
- within-β k-NN rank-locality scoring as the **primary** read;
- across-β k-NN rank-locality scoring as the coupling-triviality
  cross-check;
- six controls scored (`CTRL_META`, `CTRL_RAW`, `CTRL_RAND`,
  `CTRL_RAND_STRAT`, `CTRL_PERM`, `CTRL_GAUGE_RAND`) on the v5
  correlator signature;
- one control declared but deferred (`CTRL_FINITE_SIZE`, gated to
  Phase 4);
- bootstrap 95% CIs on every mean bin-purity score; v2 gates remain
  point-estimate.

Out of scope at this v2:

- new per-β ensemble generation (v2 reuses v0 ensembles bit-for-bit);
- any displacement slate other than the locked five classes;
- any loop set other than `{W11, W12, W13, W22}`;
- APE smearing of any kind (no use of P0 amendment 1 in this spec);
- non-Abelian higher-point correlators (n-point with n > 2);
- any β value or lattice size outside the registered v0 envelope;
- Phase 3, Phase 4, Phase 5 work;
- any retune of the per-β tertile bin edges from the v0 fixed values.

## Inputs From v0

Identical input requirements to v1 §"Inputs From v0", restated for
self-containment:

| Required input | Source |
| --- | --- |
| v0 per-β ensemble at β=2.0 | `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.0_ensemble_v0/` |
| v0 per-β ensemble at β=2.4 | `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.4_ensemble_v0/` |
| v0 per-β ensemble at β=2.8 | `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.8_ensemble_v0/` |
| v0 per-β bin edges | `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v0/aggregation/per_beta_bin_edges.json` |

For each ensemble directory the runner must validate manifest, summary,
configs JSONL, hashes (matching v0's recorded hashes), and Phase 1
ensemble-health passing. Any failure → `Z void_run`.

The v0 per-β bin edges file must be read and asserted against
re-computed per-β tertiles on the same bare γ_held values (max abs
edge difference `<= 1e-12`). Mismatch → `Z void_run`.

## Signature Vocabulary v5

Per the locked probe-spec definition:

```text
v5_signature(config) = ( C_W(r-class) for W in {W11, W12, W13, W22}
                                       for r-class in {r1, r2, r3, r4, r5} )

dimension = 4 * 5 = 20
```

Connected 2-point correlator per (loop, displacement-class):

```text
C_W(r-class) = mean over (position x, equivalent r in r-class)
               of [ Wbar(x) * Wbar(x + r) ]  -  <Wbar>^2

Wbar(x)  = (1/2) Re Tr U_loop(x)  averaged over 3 plane orientations
<Wbar>   = mean over all base positions x of Wbar(x)
x + r    = (x_a + r_a) mod 12 for each axis a
```

Displacement-class enumeration is the standard cubic-symmetry
expansion: for `r1 = (1, 0, 0)` the class contains
`{(±1, 0, 0), (0, ±1, 0), (0, 0, ±1)}` (6 vectors); for
`r2 = (1, 1, 0)` it contains all 12 sign-and-axis permutations; for
`r3 = (1, 1, 1)` all 8 corner sign permutations; for `r4 = (2, 0, 0)`
the same 6-vector class as `r1` but at distance 2; for `r5 = (2, 1, 0)`
all 24 sign-and-axis permutations.

For each loop class the runner first computes the position array
`Wbar(x)` over all 12³ = 1728 base positions (with 3-plane-orientation
averaging baked in). Then for each displacement class the connected
correlator is the position-class-and-r-class-averaged product minus
the squared mean. All four loops share the same per-position
precomputation; only the displacement-class indexing differs.

Forbidden as primary signature components, restated:

- raw link variables;
- gauge-fixed potentials;
- metadata fields (β, lattice size, cell label, seed);
- any held-out target loop (`W14`, `W23`, `W33`) copied into the
  signature;
- correlators at any displacement class not in the locked slate;
- smearing (P0 amendment 1 is not invoked at v2);
- blocking, topological proxies, Polyakov-loop proxies (still deferred
  per P0).

## Held-Out Target And Bin Edges

Identical to v0 / v1:

```text
heldout = ( W14_mean, W23_mean, W33_mean )   on bare links
γ_held  = - LS slope of ln(max(Wnm, 1e-10)) vs area
          over (A=4, A=6, A=9)
```

The runner reads `heldout/heldout_summary.csv` from each v0 per-β
ensemble dir; γ_held values are not recomputed.

Per-β tertile bin edges are recomputed from the same γ_held data by
the same linear-interpolation percentile convention as v0 / v1. The
runner must assert that the recomputed edges match v0's bin edges to
machine epsilon (max abs difference `<= 1e-12` per edge), reading
v0's edges from
`results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v0/aggregation/per_beta_bin_edges.json`.
Mismatch → `Z void_run`.

The same applies to the global tertile edges from
`global_bin_edges.json`.

## Signature Distance Metric

Same shape as v0 / v1 with the new dimension:

- Within-β NN graph (primary lane): Euclidean L2 in z-score-normalized
  **20-dim correlator signature space**; per-β normalization from
  that β's 32-config sample.
- Across-β NN graph (cross-check lane): Euclidean L2 in
  z-score-normalized 20-dim space; combined 96-config normalization.

Normalization means and standard deviations are recorded in
`aggregation/signature_normalization.json`.

`k` slate `{3, 5, 10}`; primary k=5; gate at k=5 only.

## Rank-Locality Scoring Metric

Identical to v0 / v1: bin-purity@k, discrimination ratio against
chance 1/3, bootstrap 95% CI with B=1000 within-β resamples. Kendall
τ secondary metric.

## Controls Battery

Identical to v0 / v1 with these clarifications:

- `CTRL_RAW` in v2 means **NN graph using flat raw matrix-entry
  vectors of the bare (un-smeared, un-gauge-fixed) link variables**,
  identical to v0 and v1's CTRL_RAW. The point of CTRL_RAW in v2 is
  to confirm the correlator signature is doing something different
  from the raw bare-link representation.
- `CTRL_GAUGE_RAND` in v2 applies a random SU(2) Haar site-gauge
  transform to the bare link variables and re-runs the correlator
  pipeline. Connected correlators of gauge-invariant traces are
  gauge invariant by construction; the gate is `<= 1e-12` max abs
  residual in the per-component signature, matching v0 / v1. Observed
  residual will be slightly larger than v0's bare mean+var (correlator
  products accumulate more rounding than marginal means) but should
  remain well under 1e-12.

## Pass / Quarantine Thresholds

All point-estimate gates evaluated at `k = 5` on the within-β primary
lane unless otherwise noted.

| Quantity | Registered threshold | Branch if missed |
| --- | --- | --- |
| Three v0 ensemble dirs present + healthy | summary.json health passed; configs hash matches v0 hashes.json | `Z void_run` |
| Per-β tertile bin edges match v0 to machine epsilon | max abs edge difference `<= 1e-12` | `Z void_run` |
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
results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v2/
```

Required files (mirrors v1 aggregation output with correlator-specific
additions):

| Path under result directory | Role |
| --- | --- |
| `aggregation/manifest.json` | aggregation runtime manifest with locked fields, code commit, dirty flag, command line, wall clock, v0 ensemble sources, hashes |
| `aggregation/v0_ensemble_sources.json` | list of three v0 per-β ensemble dirs + their `configs/su2_links.jsonl` SHA-256 |
| `aggregation/displacement_classes.json` | the five frozen displacement classes with their member vector lists; emitted for auditability |
| `aggregation/correlator_signature_vectors.csv` | 96 v5 connected-correlator signature vectors |
| `aggregation/per_beta_bin_edges.json` | per-β tertile edges (asserted equal to v0's) |
| `aggregation/global_bin_edges.json` | global tertile edges (asserted equal to v0's) |
| `aggregation/signature_normalization.json` | z-score means/stds for correlator signature, both lanes |
| `aggregation/within_beta_nn_graphs.json` | per-β NN graphs on correlator signature at k=10 |
| `aggregation/across_beta_nn_graph.json` | combined NN graph on correlator signature |
| `aggregation/control_nn_graphs/<control_id>.json` | per-control NN graphs (six controls scored) |
| `aggregation/rank_locality_scores.csv` | one row per (lane, control_or_primary, k) with mean bin-purity, discrimination ratio, bootstrap 95% CI |
| `aggregation/kendall_tau.csv` | secondary metric, per-β + aggregated |
| `aggregation/v0_v1_v2_comparison.json` | per-(lane, control, k) diff between v0 / v1 / v2 mean bin-purity, for diagnostic interpretation |
| `aggregation/branch_inputs.json` | every gate's observed value and pass/fail flag |
| `aggregation/summary.json` | final Phase 2 v2 verdict |
| `aggregation/hashes.json` | SHA-256 hashes (excluding itself) |

Minimum `aggregation/manifest.json` fields:

```json
{
  "phase": "phase2",
  "phaseVersion": "v2",
  "cell": "SU2_3D",
  "latticeSize": [12, 12, 12],
  "betaSlate": [2.0, 2.4, 2.8],
  "perBetaConfigurations": 32,
  "totalConfigurations": 96,
  "signatureVocabularyVersion": "v5",
  "heldOutTargetVocabularyVersion": "v1",
  "signatureLoopClasses": ["W11", "W12", "W13", "W22"],
  "displacementClassRepresentatives": [
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1],
    [2, 0, 0],
    [2, 1, 0]
  ],
  "signatureDimension": 20,
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
  "codeCommit": null,
  "gitDirty": null,
  "commandLine": null,
  "wallClockSeconds": null
}
```

## Aggregation Invocation Manifest

The v2 aggregation runner consumes the three v0 per-β ensemble
directories and produces the Phase 2 v2 receipt inputs. It must use
the exact command line below.

```text
node scripts/yang-mills-phase2-v2-su2-3d-aggregate.mjs --cell SU2_3D --lattice-size 12x12x12 --beta-slate 2.0,2.4,2.8 --signature-vocab v5 --signature-loops W11,W12,W13,W22 --displacement-classes 1-0-0,1-1-0,1-1-1,2-0-0,2-1-0 --in-beta-2.0 results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.0_ensemble_v0 --in-beta-2.4 results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.4_ensemble_v0 --in-beta-2.8 results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.8_ensemble_v0 --v0-bin-edges results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v0/aggregation/per_beta_bin_edges.json --distance-metric euclidean_zscore --k-slate 3,5,10 --primary-k 5 --bootstrap-resamples 1000 --bin-convention per_beta_tertile_linear --gauge-rand-seed-tag phase2_v2_aggregation --gauge-transforms-per-config 1 --out results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v2
```

Package script:

```text
npm run yang-mills:phase2:v2:su2-3d:aggregate
```

The aggregation runner must, in this exact order:

1. validate the three v0 ensemble dirs (manifests, health, configs
   hash);
2. read the 96 bare configurations and the 96 bare γ_held values;
3. read the v0 per-β bin edges file; verify the file timestamp is
   earlier than the v2 runner start time;
4. for each configuration: precompute the per-position Wilson-loop
   array `Wbar(x)` for each loop in `{W11, W12, W13, W22}` over all
   12³ base positions (with 3-plane-orientation averaging);
5. for each configuration and each (loop, displacement-class) entry,
   compute the connected correlator `C_W(r-class)`;
6. assemble the 20-dim v5 signature vector per configuration; emit
   to `correlator_signature_vectors.csv`;
7. compute per-β tertile edges from γ_held and assert match with v0
   edges to `<= 1e-12`;
8. compute z-score normalization on v5 signatures, write
   `signature_normalization.json`;
9. compute within-β and across-β NN graphs at k=10 on v5 signatures;
10. compute the six scored control NN graphs;
11. compute rank-locality bin-purity@k for k ∈ {3, 5, 10};
12. compute bootstrap 95% CIs;
13. compute Kendall τ secondary;
14. compute `v0_v1_v2_comparison.json` for diagnostic interpretation;
15. evaluate every gate; write `branch_inputs.json`, `summary.json`;
16. write `hashes.json` last.

Steps 3 and 7 are the bin-edge defense; if either fails the run is a
`Z void_run` and the receipt cannot be cited.

## Branch Table

| Branch | Trigger | Disposition |
| --- | --- | --- |
| `P2-A bounded_positive` | every Pass / Quarantine threshold passes on correlator signature | bounded relative-locality receipt at SU2_3D 12³ × β slate × **vocab v5 (correlator)**; Phase 3 manifest admitted |
| `YM-P2-NEG-A no_rank_local_structure` | correlator primary fails the 0.5 / 0.10-margin gate against RAND | named null; small-loop hypothesis exhausted across marginal + 2-point spatial structure, bare + smeared; routes to v3 target redesign per the probe spec's v3 fallback table |
| `YM-P2-NEG-B metadata_only` | CTRL_META meets or beats correlator primary | named null; bin convention or scoring may need rework |
| `YM-P2-NEG-C coupling_triviality` | across-β primary fails RAND_STRAT margin | named null |
| `YM-P2-NEG-D raw_dominates` | CTRL_RAW matches or beats correlator primary at the margin | named null; signature class needs to expand beyond pure small-loop information |
| `YM-P1-NEG-A gauge_leakage` | CTRL_GAUGE_RAND breach > 1e-12 in mean bin-purity | quarantine; correlator or NN implementation rechecked |
| `Z graph_contamination` | CTRL_PERM != chance within 0.05 | aggregation runner bug |
| `Z void_run` | missing field, command drift, compute-cap breach, manifest/hash drift, v0 ensemble drift, bin-edge drift | output cannot be cited as a receipt |

The v2 receipt is filed as exactly one branch.

## Compute Cap

- Per-configuration per-position Wilson-loop array (4 loops × 1728
  positions × 3 plane orientations): ~few hundred ms per config;
- Per-configuration correlator (4 loops × 5 displacement classes ×
  averaging over class members): ~few tens of ms per config;
- 96 configurations × above: ~30-60 seconds total signature
  computation;
- NN graphs + controls + bootstrap: ~30 seconds (same as v0 / v1);
- Total estimated wall clock: ~1-2 minutes. Cap is 10 minutes per
  the P0 lock.

## Next Allowed Step

After this spec is filed, the next admitted engineering moves are, in
this order:

1. Implement `scripts/lib/yang-mills-su2-3d-correlator.mjs`:
   - per-configuration per-position Wilson-loop array generator over
     the locked loop set;
   - per-configuration connected 2-point correlator at the locked
     five displacement classes;
   - reusable from any future probe spec that wants the same
     correlator vocabulary on different inputs.
2. Implement `scripts/yang-mills-phase2-v2-su2-3d-aggregate.mjs`.
   v0 and v1 aggregation runners + bare SU(2) 3D core +
   smearing module all remain bit-for-bit unchanged.
3. Wire the npm script `yang-mills:phase2:v2:su2-3d:aggregate`.
4. Execute the aggregation invocation and file a single Phase 2 v2
   receipt under
   `docs/yang-mills/receipts/YYYY-MM-DD_SU2_3D_phase2_v2_<short-verdict>.md`
   citing the v2 aggregation directory and the three v0 ensemble dirs.
5. If the verdict is `P2-A bounded_positive`, draft
   `docs/prereg/yang-mills/PHASE3_SU2_3D_observable_certificate_v0.md`
   per P0 §8 Phase 3.
6. If the verdict is any `YM-P2-NEG-*` named null, file a v3 probe
   spec per the v2 probe spec's pre-stated v3 fallback table
   (target-side redesign; likely a new P0 amendment for Polyakov or
   topological observables).

Phase 4 and Phase 5 remain gated as in P0 §8.
