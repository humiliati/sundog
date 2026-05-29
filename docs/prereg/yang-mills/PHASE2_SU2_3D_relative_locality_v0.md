# Yang-Mills Phase 2 v0 - SU2 3D Relative-Locality Certificate Spec

Status: **Phase 2 v0 spec filed 2026-05-29**. This is a pre-run binding
spec, not a result. It admits the SU(2) 3D 12³ ensemble runner only
after the implementation emits the runtime manifest fields named below
for the per-β invocations, and admits the aggregation runner only after
the three per-β ensemble receipts are filed under
`results/yang-mills/phase2/SU2_3D/`. It admits no Yang-Mills, Phase 3
observable-certificate, Phase 4 finite-size, continuum, confinement, or
mass-gap claim.

P0 lock: [`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md)
Roadmap: [`../../SUNDOG_V_YANG_MILLS.md`](../../SUNDOG_V_YANG_MILLS.md)
Receipt template:
[`../../yang-mills/RECEIPT_TEMPLATE.md`](../../yang-mills/RECEIPT_TEMPLATE.md)
Phase 1 instrumentation-closure receipts:
[`../../yang-mills/receipts/2026-05-29_U1_2D_phase1_gauge_invariance_smoke_pos.md`](../../yang-mills/receipts/2026-05-29_U1_2D_phase1_gauge_invariance_smoke_pos.md),
[`../../yang-mills/receipts/2026-05-29_SU2_2D_phase1_gauge_invariance_smoke_pos.md`](../../yang-mills/receipts/2026-05-29_SU2_2D_phase1_gauge_invariance_smoke_pos.md),
[`../../yang-mills/receipts/2026-05-29_SU2_3D_phase1_gauge_invariance_smoke_pos.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase1_gauge_invariance_smoke_pos.md).

## Claim Under Test

Inside the `SU2_3D` 12×12×12 primary cell, over the registered β slate
`{2.0, 2.4, 2.8}` with 32 retained configurations per β, the v1
gauge-invariant small-Wilson-loop signature should preserve the
per-config held-out-Wilson-loop area-law-decay tertile label
(`γ_held`-bin) in nearest-neighbor rank space, **beyond** every
metadata, raw-link, random, β-stratified random, label-permutation, and
gauge-randomized control on the same frozen neighbor graphs and frozen
bin edges.

This is the first non-Abelian relative-locality certificate read in
this lane. A pass establishes a finite-lattice, cell-local, lossy
gauge-invariant shadow that resolves a held-out area-law class better
than controls inside the registered envelope. A pass is **not** a Clay
result, a confinement proof, a mass-gap claim, a continuum statement, or
evidence at any β / lattice size outside the registered envelope. A
null is filed as `YM-P2-NEG-A` / `NEG-B` / `NEG-C` as appropriate.

## Scope

In scope at this v0:

- 12×12×12 ensemble generation on the SU(2) 3D primary cell at all
  three P0 β-slate points;
- v1 small-Wilson-loop signature (`W11`, `W12`, `W13`, `W22` mean/var,
  position-and-orientation averaged, locked at P0);
- v1 held-out target (`W14`, `W23`, `W33` per-config mean values,
  locked at P0) reduced to a per-config `γ_held` exponential-decay
  rate, then to a per-β tertile bin label;
- within-β `k`-NN rank-locality scoring as the **primary** read;
- across-β `k`-NN rank-locality scoring as a coupling-triviality
  cross-check (`YM-P2-NEG-C`);
- six controls scored (`CTRL_META`, `CTRL_RAW`, `CTRL_RAND`,
  `CTRL_RAND_STRAT`, `CTRL_PERM`, `CTRL_GAUGE_RAND`) on the same
  frozen neighbor graphs and bin edges;
- one control declared but deferred (`CTRL_FINITE_SIZE`, gated to
  Phase 4);
- bootstrap 95% confidence intervals reported around every mean
  bin-purity score; primary point-estimate gates remain the
  promotion/quarantine boundary at this v0.

Out of scope at this v0:

- 4D Yang-Mills, smearing, blocking, improved actions, topological
  proxies;
- any β value outside `{2.0, 2.4, 2.8}`;
- any lattice size other than 12³ on SU(2) 3D (the 8³ Phase 1
  ensemble exists but is reserved for Phase 4);
- the certificate cost / verifier program (Phase 3, separate spec);
- the finite-size split test (Phase 4, separate spec);
- the external-review packet (Phase 5, gated on Phase 2 or Phase 3
  earning a non-vacuous result);
- any per-β bin-edge revision after `rank-locality scoring begins`;
- any expansion of the signature or held-out vocabulary inside this
  spec — an expansion requires a dated `v1` spec under
  `docs/prereg/yang-mills/`.

## Locked Cell And Ensemble

| Field | Registered value |
| --- | --- |
| Cell label | `SU2_3D` |
| Lattice size | `12x12x12` (the larger of the P0 SU2_3D pair; Phase 1 used the smaller `8x8x8` partner for instrumentation) |
| Boundary condition | `periodic` in all three directions |
| Action | `Wilson` |
| Generator | SU(2) Creutz heatbath with Kennedy-Pendleton acceptance + Brown-Woch overrelaxation, **same `su2_heatbath_overrelax_v1` algorithm as Phase 1 SU2_3D**, with the 3D 4-plaquettes-per-link staple sum |
| Update mix ratio | 1 heatbath sweep + 4 overrelaxation sweeps per combined sweep |
| β slate | `{2.0, 2.4, 2.8}` (the full P0 SU2_3D slate; identical to the P0 lock; not revised) |
| Burn-in per β | `2000` combined sweeps minimum |
| Pilot per β | `512` combined sweeps; `τ_int(plaquette)` must be `<= 16.0`, thinning/τ_int `>= 2.0` (same as Phase 1) |
| Thinning per β | `32` combined sweeps |
| Retained configurations per β | `32` |
| Total retained configurations | `96` across all three β |
| Master seeds | `202605290201` (β=2.0), `202605290202` (β=2.4), `202605290203` (β=2.8); independent substreams per Phase 1 deriveSubstreamSeed factory |
| Signature vocabulary | `v1` (`W11`, `W12`, `W13`, `W22` mean+variance, `(1/2) Re Tr U_loop`, position-and-orientation averaged over `{xy, xz, yz}`) |
| Held-out target vocabulary | `v1` (`W14`, `W23`, `W33` mean values, same trace convention, same averaging) |
| Held-out summary | per-config exponential-decay slope `γ_held` (see § "γ_held Derivation" below) |
| Bin convention | per-β tertile of `γ_held` distribution over the 32 configurations at that β |
| Bin edge freezing point | recorded in aggregation receipt **before** any rank-locality scoring is computed; never revised after |

## Signature Vocabulary

Locked at P0 lock §"Primary Signature Vocabulary" v1; restated for
self-containment:

For each retained configuration, the v1 signature is the eight-component
vector

```text
sig = ( W11_mean, W11_var,
        W12_mean, W12_var,
        W13_mean, W13_var,
        W22_mean, W22_var )
```

where each `Wnm_*` is the position-and-orientation average of
`(1/2) Re Tr U_loop` for an `n × m` rectangular Wilson loop over the
three plane orientations `{xy, xz, yz}` and all base positions in the
12³ lattice.

The signature is **gauge invariant by construction** (loop traces of
SU(2) are gauge invariant). The runtime per-config signature is
recomputed after every `CTRL_GAUGE_RAND` transform and must agree to
machine epsilon with the pre-transform signature, exactly as in Phase 1.

## γ_held Derivation

For each retained configuration, the v1 held-out target is the
three-element vector

```text
heldout = ( W14_mean, W23_mean, W33_mean )
```

with the same `(1/2) Re Tr U_loop` convention and the same position-
and-orientation averaging as the signature. `W14` has area `4`, `W23`
has area `6`, `W33` has area `9`.

The per-config exponential-decay summary `γ_held` is the negative
least-squares slope of `ln(max(W, ε))` regressed against loop area `A`
on the three held-out points only:

```text
For point i in {W14, W23, W33}:
  A_i in {4, 6, 9}
  y_i = ln( max(Wi_mean, ε) ),   with ε = 1e-10 hard floor

γ_held = - ( N · Σ (A_i · y_i) - Σ A_i · Σ y_i )
         / ( N · Σ (A_i²) - (Σ A_i)² ),
where N = 3.
```

The ε hard floor is documented; any clamp event must be recorded in the
per-configuration γ_held output column so a downstream auditor can spot
configurations whose held-out W-values reached the clamp.

**Why this derivation respects the P0 signature-vs-target disjointness:**
the signature contains only loops with area `<= 4` (`W11`, `W12`, `W13`,
`W22`), and W22 (area 4) is geometrically distinct from W14 (also area
4) — same area, different rectangle. The held-out target contains only
loops with area `>= 4` (`W14` shares `W22`'s area but not its shape;
`W23` and `W33` have larger areas). Any area-law correlation between the
signature loop traces and the held-out γ_held is exactly the structure
under test — not assumed by construction.

**Why γ_held is the area-law proxy:** in any Wilson-loop ensemble where
the area law `<W(A)> ~ exp(-σ · A)` holds approximately,
`-d ln <W> / dA` is the per-configuration / per-ensemble string tension
proxy. Fitting that slope on three held-out points gives a per-config
scalar summary that is bounded by the geometry (no extreme outliers in
practice on a finite ensemble) and that varies meaningfully across both
β and within β.

## γ_held Bin Freezing Protocol

The per-β tertile bin edges are computed and frozen in the aggregation
step, before any rank-locality score is produced.

For each β:

1. Read the 32 per-configuration `γ_held` values from
   `results/yang-mills/phase2/SU2_3D/<beta-dir>/heldout/heldout_summary.csv`.
2. Compute the 33.333... and 66.666... percentile edges of those 32
   values using the **linear-interpolation** percentile convention
   (numpy `method='linear'`, identical to its default).
3. Assign each configuration a bin in `{1, 2, 3}`:
   - bin `1` if `γ_held <= low_edge`,
   - bin `2` if `low_edge < γ_held <= high_edge`,
   - bin `3` if `γ_held > high_edge`.
4. Record `(low_edge, high_edge)` and the per-configuration bin
   assignment in the aggregation receipt's
   `aggregation/per_beta_bin_edges.json` file. This file must be
   written **before** any neighbor graph is computed.

For the across-β cross-check lane (coupling triviality probe), a single
**global** tertile edge pair is also frozen at the same time, computed
on the combined 96-config `γ_held` distribution by the same convention.
The global edges are recorded alongside the per-β edges and likewise
frozen before scoring.

Forbidden, restated:

- recomputing tertile edges after seeing a rank-locality score;
- using a different percentile convention;
- using fewer or more than three bins;
- dropping a bin because it ends up with fewer than expected members.

If a per-β `γ_held` distribution is degenerate (all 32 values equal to
within floating-point tolerance `1e-10`), that β is marked
`Z bin_degenerate` and contributes no primary score; the other two β
proceed as scheduled and the verdict reports the partial coverage.

## Signature Distance Metric

Within-β NN graph (primary lane):

1. Take the 32 per-configuration signature vectors at one β.
2. z-score normalize per component using that β's 32-config mean and
   standard deviation.
3. Euclidean L2 distance in the resulting 8-dimensional space.

Across-β NN graph (cross-check lane):

1. Take the 96 per-configuration signature vectors across all three β.
2. z-score normalize per component using the combined 96-config mean
   and standard deviation.
3. Euclidean L2 distance in the resulting 8-dimensional space.

Both normalization passes are computed before any NN scoring; their
parameters (per-component means and standard deviations) are recorded
in `aggregation/signature_normalization.json`.

`k` slate: `{3, 5, 10}`. Primary metric is reported at `k = 5`; the
other two values are reported for robustness. Promotion gates are
evaluated at `k = 5` only; if the verdict would flip across the k slate
that fact must be reported in the receipt notes but does not change the
verdict.

## Rank-Locality Scoring Metric

For a query configuration `q` with bin label `B(q)`, let `N_k(q)` be
its `k` nearest neighbors (by the locked distance metric above,
excluding `q` itself, ties broken by configuration index).

```text
bin-purity@k(q) = (1 / k) · | { n in N_k(q) : B(n) = B(q) } |
mean_bin_purity_k = mean over all queries of bin-purity@k(q)
discrimination_ratio = mean_bin_purity_k / (1/3)
```

Chance baseline is `1/3` (uniform random over three bins). A
`discrimination_ratio` of 1 means no signal; values above 1 mean the
NN graph preserves bin structure better than chance.

Secondary metric (reported, not gated): **Kendall τ** between (a) the
rank of signature distance from `q` to each other configuration in the
same β and (b) the rank of `|γ_held(other) - γ_held(q)|`. Reported per
β and aggregated; positive values mean signature-space proximity
tracks γ_held-space proximity.

Bootstrap 95% confidence intervals on `mean_bin_purity_5` are computed
by resampling configurations within β with replacement (B = 1000
resamples). CI bounds are reported alongside the point estimate; v0
gates use point estimates, not CI bounds.

## Controls Battery

All seven P0 controls are declared. The first six are scored on the
same frozen neighbor graphs and frozen bin edges as the primary; the
seventh is deferred.

| Control id | Phase 2 v0 definition | Within-β behavior | Across-β behavior |
| --- | --- | --- | --- |
| `CTRL_META` | NN graph using only the metadata vector `(β, latticeSize)`; latticeSize is constant (12³) so the only distance is `\|β_i - β_j\|`. | degenerate within a single β (all distances zero, NN is index-order-tied) → bin-purity collapses to chance `1/3`; primary must beat this trivial floor | meaningful: β-stratified NN; primary must beat it |
| `CTRL_RAW` | NN graph using the flat 8-real-per-link raw matrix-entry vector (`Re a, Im a, Re b, Im b, Re c, Im c, Re d, Im d` per link, no gauge fixing); Euclidean L2 after the same per-component z-score normalization | scored | scored |
| `CTRL_RAND` | NN replaced by `k` uniform random configurations from the same β cohort | bin-purity collapses to ~chance in expectation; primary must beat with margin | bin-purity collapses to ~chance |
| `CTRL_RAND_STRAT` | NN replaced by `k` uniform random configurations from the same β cohort AND of the same primary β | identical to `CTRL_RAND` within-β | meaningful: β-stratified random; isolates the coupling-triviality concern |
| `CTRL_PERM` | bin labels permuted uniformly across the frozen primary NN graph | tests for graph contamination — must collapse to chance `1/3` | same |
| `CTRL_GAUGE_RAND` | primary signature recomputed after applying a random SU(2) Haar site-gauge transform to every configuration; NN graph rebuilt on the post-transform signatures | must match the primary signature-space NN graph to machine-epsilon tolerance; bin-purity must therefore equal the primary's exactly | same |
| `CTRL_FINITE_SIZE` | NN graph restricted to configurations from the partner lattice (`8x8x8`) | **deferred to Phase 4** — requires an 8³ Phase 2 ensemble at all three β, not in scope at v0 | same |

For each scored control, the receipt reports `mean_bin_purity_5`,
`discrimination_ratio`, and bootstrap 95% CI for the within-β primary
lane and (separately) the across-β cross-check lane. `CTRL_FINITE_SIZE`
is reported as `not_scored: phase4_reserved`.

`CTRL_RAW` deserves a specific note: in Phase 2 the raw matrix-entry
vector is the **un**-gauge-fixed flat link representation. Because raw
links are not gauge invariant, the NN graph built on them is itself
gauge-variant — but bin-purity computed against the (gauge-invariant)
γ_held labels is still a meaningful number to compare against the
primary. If `CTRL_RAW` beats the primary at the same `k`, that does not
mean Sundog has a Yang-Mills result via raw links; per the P0 lock
public-language boundary, the receipt must then state that the lossy
invariant shadow is too coarse and file `YM-P2-NEG-D raw_dominates` (a
new branch introduced by this v0 — see Branch Table).

## Pass / Quarantine Thresholds

All point-estimate gates evaluated at `k = 5` on the within-β primary
lane unless otherwise noted.

| Quantity | Registered threshold | Branch if missed |
| --- | --- | --- |
| Per-β ensemble: every Phase 1 admission requirement (burn-in, τ_int, thinning, fallback, unitarity, orientation isotropy) | inherited from Phase 1 SU2_3D manifest, applied at 12³ | per-β `Z void_run` or `YM-P1-*` quarantine inherited |
| Bin freezing recorded before scoring | non-empty `per_beta_bin_edges.json` with timestamp earlier than first scoring artifact | `Z void_run` |
| Primary within-β `mean_bin_purity_5` | `>= 0.5` (discrimination_ratio `>= 1.5`) | `YM-P2-NEG-A no_rank_local_structure` |
| Primary beats `CTRL_RAND` mean bin-purity | margin `>= 0.10` | `YM-P2-NEG-A no_rank_local_structure` |
| Primary beats `CTRL_META` mean bin-purity | margin `>= 0.10` | `YM-P2-NEG-B metadata_only` |
| Primary beats `CTRL_RAW` mean bin-purity | margin `>= 0.10` | `YM-P2-NEG-D raw_dominates` (new) |
| `CTRL_PERM` mean bin-purity | within `0.05` of chance `1/3` | `Z graph_contamination` |
| `CTRL_GAUGE_RAND` recovers primary bin-purity exactly to machine epsilon | match `<= 1e-12` in mean bin-purity | `YM-P1-NEG-A gauge_leakage` |
| Across-β primary `mean_bin_purity_5` (cross-check) beats `CTRL_RAND_STRAT` | margin `>= 0.05` | `YM-P2-NEG-C coupling_triviality` |
| Per-β `γ_held` distribution non-degenerate | spread `>= 1e-10` | `Z bin_degenerate` for that β |
| Aggregation wall clock | `<= 10 minutes` | `Z void_run` |
| Per-β ensemble wall clock | `<= 10 minutes` each | per-β `Z void_run` |
| Missing runtime manifest field | none missing | `Z void_run` |

A run that lands in any `NEG-A` / `NEG-B` / `NEG-C` / `NEG-D` branch is
a **named null**, filed as a Phase 2 receipt; the lane does not silently
revise the signature, the bin convention, or the β slate to look for a
better verdict.

## Output Contract

### Per-β ensemble runner (one per β)

Exact result directory per β:

```text
results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta<beta>_ensemble_v0/
```

where `<beta>` is `2.0`, `2.4`, or `2.8`. Required files (mirrors
Phase 1 SU2_3D output, with the Phase 2 v0 ensemble shape):

| Path under result directory | Role |
| --- | --- |
| `manifest.json` | runtime manifest; every locked field, code commit, dirty flag, command line, wall clock, per-orientation mean plaquette, max unitarity residual, fallback counts, hashes |
| `autocorr_pilot/plaquette_series.csv` | unthinned post-burn-in plaquette series |
| `autocorr_pilot/plaquette_by_orientation.csv` | per-orientation series for the isotropy gate |
| `configs/su2_links.jsonl` | 32 retained post-thinning configurations, quaternion-parametrized SU(2) link matrices |
| `signatures/signature_vectors.csv` | 32 v1 signature vectors |
| `heldout/heldout_loop_values.csv` | 32 per-config `(W14_mean, W23_mean, W33_mean)` rows |
| `heldout/heldout_summary.csv` | 32 per-config `γ_held` values, including a clamp flag column |
| `summary.json` | per-β ensemble health verdict (Phase 1 thresholds inherited) |
| `hashes.json` | SHA-256 hashes of all emitted artifacts except `hashes.json` itself |

### Aggregation runner

Exact aggregation result directory:

```text
results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v0/
```

Required files:

| Path under aggregation directory | Role |
| --- | --- |
| `aggregation/manifest.json` | aggregation runtime manifest (locked fields below, code commit, command line, wall clock, hashes) |
| `aggregation/per_beta_bin_edges.json` | per-β tertile edges, per-config bin assignments, frozen before scoring (with timestamp) |
| `aggregation/global_bin_edges.json` | across-β tertile edges, per-config bin assignments |
| `aggregation/signature_normalization.json` | per-component z-score means and standard deviations for both lanes |
| `aggregation/within_beta_nn_graphs.json` | per-β primary NN graphs (k=10 stored once; k=3 and k=5 derive) |
| `aggregation/across_beta_nn_graph.json` | combined NN graph |
| `aggregation/control_nn_graphs/<control_id>.json` | per-control NN graphs (`CTRL_META`, `CTRL_RAW`, `CTRL_RAND`, `CTRL_RAND_STRAT`, `CTRL_PERM`, `CTRL_GAUGE_RAND`) |
| `aggregation/rank_locality_scores.csv` | one row per (lane, control_or_primary, k) with mean bin-purity, discrimination ratio, bootstrap 95% CI low/high |
| `aggregation/kendall_tau.csv` | secondary metric, per-β + aggregated |
| `aggregation/branch_inputs.json` | every gate's observed value and pass/fail flag |
| `aggregation/summary.json` | final Phase 2 v0 verdict |
| `aggregation/hashes.json` | SHA-256 hashes (excluding itself) |

Minimum aggregation `manifest.json` fields:

```json
{
  "phase": "phase2",
  "cell": "SU2_3D",
  "latticeSize": [12, 12, 12],
  "betaSlate": [2.0, 2.4, 2.8],
  "perBetaConfigurations": 32,
  "totalConfigurations": 96,
  "signatureVocabularyVersion": "v1",
  "heldOutTargetVocabularyVersion": "v1",
  "gammaHeldEpsilonFloor": 1e-10,
  "binConvention": "per_beta_tertile_linear",
  "globalBinConvention": "global_tertile_linear",
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
  "perBetaReceiptPaths": null,
  "codeCommit": null,
  "gitDirty": null,
  "commandLine": null,
  "wallClockSeconds": null
}
```

## Per-β Ensemble Invocation Manifests

The three per-β ensemble runs are admitted as separate invocations
under the P0 10-minute-per-invocation compute cap. Each must use the
exact command line below; any divergence is a `Z void_run`.

| β | Master seed | Result directory | Exact command line |
| --- | --- | --- | --- |
| `2.0` | `202605290201` | `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.0_ensemble_v0` | `node scripts/yang-mills-phase2-su2-3d-ensemble.mjs --cell SU2_3D --lattice-size 12x12x12 --beta 2.0 --boundary periodic --action Wilson --generator su2_heatbath_overrelax_v1 --overrelax-per-heatbath 4 --seed 202605290201 --burn-in 2000 --pilot-sweeps 512 --thinning 32 --measurements 32 --signature-vocab v1 --heldout-vocab v1 --gamma-held-epsilon 1e-10 --out results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.0_ensemble_v0` |
| `2.4` | `202605290202` | `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.4_ensemble_v0` | `node scripts/yang-mills-phase2-su2-3d-ensemble.mjs --cell SU2_3D --lattice-size 12x12x12 --beta 2.4 --boundary periodic --action Wilson --generator su2_heatbath_overrelax_v1 --overrelax-per-heatbath 4 --seed 202605290202 --burn-in 2000 --pilot-sweeps 512 --thinning 32 --measurements 32 --signature-vocab v1 --heldout-vocab v1 --gamma-held-epsilon 1e-10 --out results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.4_ensemble_v0` |
| `2.8` | `202605290203` | `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.8_ensemble_v0` | `node scripts/yang-mills-phase2-su2-3d-ensemble.mjs --cell SU2_3D --lattice-size 12x12x12 --beta 2.8 --boundary periodic --action Wilson --generator su2_heatbath_overrelax_v1 --overrelax-per-heatbath 4 --seed 202605290203 --burn-in 2000 --pilot-sweeps 512 --thinning 32 --measurements 32 --signature-vocab v1 --heldout-vocab v1 --gamma-held-epsilon 1e-10 --out results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.8_ensemble_v0` |

Package scripts:

```text
npm run yang-mills:phase2:su2-3d:beta-2.0
npm run yang-mills:phase2:su2-3d:beta-2.4
npm run yang-mills:phase2:su2-3d:beta-2.8
```

Each ensemble runner reuses the SU(2) 3D core
`scripts/lib/yang-mills-su2-3d-core.mjs` from Phase 1 SU2_3D. The
existing Phase 1 entry
`scripts/yang-mills-phase1-su2-3d-gauge-smoke.mjs` must remain
bit-for-bit unchanged; the Phase 2 ensemble entry is a new file
`scripts/yang-mills-phase2-su2-3d-ensemble.mjs` that drops the
gauge-randomization read (deferred to the aggregation runner per
`CTRL_GAUGE_RAND` semantics) and adds the held-out summary computation.

## Aggregation Invocation Manifest

The aggregation runner consumes the three per-β ensemble directories
and produces the Phase 2 v0 receipt inputs. It must use the exact
command line below.

```text
node scripts/yang-mills-phase2-su2-3d-aggregate.mjs --cell SU2_3D --lattice-size 12x12x12 --beta-slate 2.0,2.4,2.8 --in-beta-2.0 results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.0_ensemble_v0 --in-beta-2.4 results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.4_ensemble_v0 --in-beta-2.8 results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.8_ensemble_v0 --distance-metric euclidean_zscore --k-slate 3,5,10 --primary-k 5 --bootstrap-resamples 1000 --bin-convention per_beta_tertile_linear --gauge-rand-seed-tag phase2_aggregation --gauge-transforms-per-config 1 --out results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v0
```

Package script:

```text
npm run yang-mills:phase2:su2-3d:aggregate
```

The aggregation runner is responsible for, in this exact order:

1. validating that all three per-β ensemble directories exist and that
   each `summary.json` reports the Phase-1-inherited ensemble health
   thresholds passed;
2. reading the 96 signature vectors and 96 held-out summaries;
3. computing per-β z-score normalization and global z-score
   normalization, writing `signature_normalization.json`;
4. computing per-β tertile edges and global tertile edges, writing
   `per_beta_bin_edges.json` and `global_bin_edges.json` (these writes
   must complete **before** any neighbor graph is computed; the runner
   must assert the timestamps);
5. computing the within-β primary NN graph and the across-β NN graph
   at `k = 10`;
6. computing the six scored control NN graphs at `k = 10`;
7. computing rank-locality bin-purity@k for `k in {3, 5, 10}` for
   primary and every scored control on both lanes;
8. computing bootstrap 95% CIs;
9. computing the secondary Kendall τ;
10. evaluating every gate in the Pass / Quarantine Threshold table and
    writing `branch_inputs.json` and `summary.json`;
11. writing `hashes.json` last.

The aggregation runner must **not** revise per-β bin edges after step
4; that step must be on disk before step 5 begins, and the runner must
re-read the file at step 5 to defend against accidental in-memory edits.

## Branch Table

| Branch | Trigger | Disposition |
| --- | --- | --- |
| `P2-A bounded_positive` | every Pass / Quarantine threshold passes | bounded relative-locality receipt at SU2_3D 12³ × β slate; Phase 3 manifest admitted |
| `YM-P2-NEG-A no_rank_local_structure` | primary `mean_bin_purity_5 < 0.5` OR primary beats `CTRL_RAND` by margin `< 0.10` | named null `no_rank_local_structure` |
| `YM-P2-NEG-B metadata_only` | primary beats `CTRL_META` by margin `< 0.10` | named null `metadata_only` |
| `YM-P2-NEG-C coupling_triviality` | across-β primary beats `CTRL_RAND_STRAT` by margin `< 0.05` | named null `coupling_triviality` |
| `YM-P2-NEG-D raw_dominates` | primary beats `CTRL_RAW` by margin `< 0.10` | named null `raw_dominates` (new branch introduced at v0) |
| `YM-P1-NEG-A gauge_leakage` | `CTRL_GAUGE_RAND` bin-purity differs from primary by `> 1e-12` | quarantine, lane voided until signature implementation rechecked |
| `Z graph_contamination` | `CTRL_PERM` bin-purity differs from chance by `> 0.05` | aggregation runner bug; receipt cannot be cited |
| `Z bin_degenerate` (per β) | β's γ_held distribution has spread `< 1e-10` | that β contributes no primary score; partial coverage reported |
| `YM-P4-DEFERRED_FINITE_SIZE` | `CTRL_FINITE_SIZE` declared but not scored at v0 | recorded as the Phase 4 gate, not a Phase 2 failure |
| `Z void_run` | missing field, command drift, compute-cap breach, manifest/hash drift, or bin-edge timestamp drift | output cannot be cited as a receipt |

Phase 2 v0 admits at most one verdict per receipt; the receipt is
filed as exactly one branch.

## Compute Cap

Per-β ensemble extrapolation from Phase 1 SU2_3D
(`8³ × 3536 combined sweeps in 37.03 s`):
- 12³ has `(12/8)³ = 3.375x` more links;
- same 3536 combined sweeps;
- ≈ 125 seconds raw ensemble generation per β;
- v1 signature + held-out passes ≈ 5-10 seconds per β;
- target per-β wall clock ≈ 2-3 minutes; cap is 10 minutes.

Aggregation extrapolation:
- 96 configs total, signature vectors are 8-dim, held-out summaries
  are scalars;
- z-score normalization, bin freezing, NN graphs at `k = 10` over
  96-config sets, six controls, bootstrap 1000 resamples;
- target aggregation wall clock under 1 minute; cap is 10 minutes.

The Phase 2 v0 total compute budget across the four invocations is
therefore ≈ 7-10 minutes, well inside the four × 10-minute cap.

## Next Allowed Step

After this spec is filed, the next admitted engineering moves are, in
this order:

1. Implement `scripts/yang-mills-phase2-su2-3d-ensemble.mjs` and wire
   the three per-β npm scripts. Reuse `scripts/lib/yang-mills-su2-3d-core.mjs`
   for lattice + heatbath + overrelaxation + signature; the only new
   work is the held-out summary computation (`γ_held` derivation and
   CSV emission), bumping the lattice from 8³ to 12³, and dropping the
   Phase 1 gauge-randomization read.
2. Execute the three per-β ensemble invocations; each must land
   `summary.json` with Phase-1-inherited ensemble health passing.
3. Implement `scripts/yang-mills-phase2-su2-3d-aggregate.mjs` and wire
   the aggregation npm script. The aggregator is **the only place** in
   the Phase 2 v0 pipeline that touches bin freezing, NN graphs,
   bin-purity scoring, controls, and the branch table.
4. Execute the aggregation invocation and file a single Phase 2 v0
   receipt under
   `docs/yang-mills/receipts/YYYY-MM-DD_SU2_3D_phase2_<short-verdict>.md`
   citing the four result directories and the aggregation receipt
   directory.
5. If the verdict is `P2-A bounded_positive`, draft a Phase 3
   observable-certificate manifest under
   `docs/prereg/yang-mills/PHASE3_SU2_3D_observable_certificate_v0.md`
   per P0 §8 Phase 3.
6. If the verdict is any `YM-P2-NEG-*` named null, file a new dated
   probe spec under `docs/yang-mills/specs/` proposing a v1 design
   change (e.g., a richer signature, an alternative bin convention,
   smearing/blocking — each requiring a dated P0 amendment if it
   crosses the P0 lock). Do not silently revise this v0 spec.

Phase 4 (finite-size split against the 8³ partner) and Phase 5
(external-review packet) remain gated as in P0 §8.
