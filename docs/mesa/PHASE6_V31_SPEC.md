# Mesa Phase 6 v3.1 — Subspace Validation and Generalization

This document is the implementation-grade spec for Phase 6 v3.1 of
[`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md). Phase 6 v2 + v3 (see
[`PHASE6_V2_RESULTS.md`](PHASE6_V2_RESULTS.md)) localized the basin
attractor at `net.7` to a 5-dimensional subspace, decomposable into a
1-dimensional policy-offset component (PC1, variance-heavy and
mechanism-empty) and a 4-dimensional mechanism component (PCs 2-5).
That result rests on three load-bearing claims that v3.1 stress-tests:

1. **PC1 really is mechanism-empty.** Predicted but not directly
   tested. Direct test: patch only PCs 2-5 (skipping PC1) and check
   that patch_success is unchanged or higher than K=5 baseline.
2. **The 5-dim subspace generalizes beyond the cliff pair.** Predicted
   as a v3.1 candidate. Direct test: apply the cliff-pair PCA basis to
   other Phase 5 zoo policy pairs (hold/collapse) and check that it
   patches them too.
3. **The directional asymmetry is statistical, not seed-noise.** At
   K=3, P→C patch_success was 0.881 while C→P was 0.509 — a gap of
   0.37. Direct test: seed-bootstrap 95% CI on the gap.
4. **The PC2-5 directions have legible structure across net.7
   neurons.** Descriptive — sparsity / concentration analysis on the
   PC basis vectors.

Where this spec and the roadmap disagree, the roadmap wins. Where both
are silent, this spec is authoritative for Phase 6 v3.1.

## 1. Decision Lock

Phase 6 v3.1 starts with six pinned calls:

- **Four axes only.** Axis I (PC2-5 alone patching), Axis J
  (generalization to held-out zoo pairs), Axis K (PC neuron-sparsity
  decomposition), Axis L (directional-asymmetry seed-bootstrap). Per-
  neuron causal mediation, larger-tier replication, and cross-
  architecture transfer are deferred to v3.2.
- **The 5-dim subspace from v3 is the central artifact.** Load from
  the existing axis-h-pca-k5 run; do not recompute. v3.1 is a
  *validation pass*, not a re-derivation.
- **No new PPO training runs.** v3.1 reuses the existing Phase 5 zoo
  policies as-is. New training is deferred to v3.2 or later.
- **Held-out test pairs are three.** L-Reward-M ↔ L-Signature-M
  terminal (canonical reward-vs-signature cliff, *not* L-Mixed),
  L-Mixed-M-λ=0.9 ↔ L-Mixed-M-λ=0.99 (within-family cliff at a
  different λ), Curriculum-Reward→Terminal-Sig ↔ L-Sig-Terminal-S
  (Small-tier cross-family).
- **Bootstrap is 1000-resample seed-level.** Resample with replacement
  from the matched 64-seed slate; recompute median patch_success;
  report 2.5/97.5 percentiles as 95% CI.
- **PC sparsity metric is L2-concentration on top-32 neurons.** For
  each PC direction, compute `||top32(v)||² / ||v||²` where `top32(v)`
  is the 32 largest-absolute-value coordinates of v in net.7's
  256-dim space. Concentration > 50% = "sparse on a few neurons";
  < 20% = "genuinely distributed."

Total v3.1 compute: 0 new PPO runs, 1 small harness extension to
`phase6_v2_sae.py`, ~60-90 minutes for the four axes combined.

## 2. Scope

Phase 6 v3.1 owns:

- Small extension to `training/mesa/phase6_v2_sae.py` adding four
  subcommands: `axis-i-pc-mech`, `axis-j-generalization`,
  `axis-k-decompose`, `axis-l-bootstrap`.
- Generalization of the existing `run_subspace_patch_battery` helper
  to accept arbitrary protected/collapsed `PolicySpec` arguments (v3
  hard-coded the cliff pair). One-parameter change.
- Loading and reusing the trained 5-dim PCA basis from
  `results/mesa/phase6-v2-direction/axis-h-pca-k5/`.
- v3.1 aggregate report with: PC2-5-only patch_success, generalization
  patch_success table across three held-out pairs, PC sparsity vector,
  directional-asymmetry CI.

Phase 6 v3.1 does **not** own:

- New PPO training runs.
- Recomputation of the PCA basis (use the v3 axis-h-pca-k5 artifact).
- Per-neuron causal mediation at net.7 (deferred to v3.2; v3.1's
  Axis K is descriptive sparsity analysis only).
- Large-tier policies (deferred indefinitely).
- Cross-architecture transfer (deferred indefinitely).
- Phase 7 v2 envelope cross-product expansion.

## 3. Axes

### 3.1 Axis I — PC2-5 alone patching (the offset-removal test)

The cleanest direct test of "PC1 is mechanism-empty." Load the
cliff-pair PCA basis `Q ∈ R^(256 × 5)` from
`axis-h-pca-k5/manifest.json` (the principal-component columns are
recoverable from the variance-explained CSV plus the underlying SVD
left-singular vectors — in v3.1 we recompute the SVD on the same
matched-seed activation collection but emit `Q_pc2_to_5 ∈ R^(256 × 4)`
as the basis for patching).

**Protocol:** identical to v3 axis-h-pca with `Q = Q_pc2_to_5`. Run
the 4-forward × 64-seed battery on the cliff pair (L-Mixed-M-λ=0.95
vs λ=0.97). Compare patch_success against the v3 K=5 baseline.

**Predicted outcome (Y1):** patch_success with PC1 removed is within
±0.05 of the K=5 baseline in both directions. If PC1 is truly
mechanism-empty, removing it should not hurt patch effect; if PC1
carries hidden mechanism, removing it would drop patch_success
substantially.

**Tier coverage:** Medium cliff pair only.

### 3.2 Axis J — Generalization across the Phase 5 zoo

The headline v3.1 test. Take the cliff-pair PCA basis
`Q_cliff ∈ R^(256 × 5)` and apply it to other policy pairs from the
Phase 7 envelope, asking: *does the basin attractor of a held-out
collapse cell live in the same 5-dim subspace?*

**Three held-out test pairs:**

| Pair | Protected (hold) | Collapsed (collapse) | Provenance |
| --- | --- | --- | --- |
| J1 | `signature_terminal_medium` | `reward_lambda_1_0_medium_anchor` | canonical signature-vs-reward at Medium |
| J2 | `mixed_lambda_0_9_medium_v3` | `mixed_lambda_0_99_medium_v4` | within-L-Mixed family, different λ |
| J3 | `signature_terminal` | `curriculum_reward_then_terminal_sig_v3` | Small-tier cross-family |

**v3.1.1 integrity amendment:** J1 and J2 are dimension-compatible with
the Medium cliff-pair PCA basis (`Q_cliff` is 256D and both pairs expose a
256D `net.7` final hidden activation). J3 is retained as the cross-tier
question but is blocked in v3.1: Small-tier policies expose a 64D final
hidden activation (`net.3`), so applying the 256D Medium basis would be a
dimension error, not a meaningful generalization result. J3 routes to v3.2
with either a Small-tier PCA basis or an explicit cross-tier adapter.

**Protocol:** for each pair, run the 4-forward × 64-seed direction-
patch battery at net.7 using `Q_cliff` (the cliff-pair-derived basis,
*not* a fresh PCA on the test pair). The harness extension generalizes
the existing `run_subspace_patch_battery` to accept arbitrary
protected/collapsed policy specs.

The implementation must reject dimension-incompatible pairs explicitly. For
v3.1, only J1 and J2 count toward Y2. J3 is a recorded integrity block, not a
generalization failure.

**Predicted outcomes (Y2):**

- **Y2-strong:** all three pairs achieve median patch_success ≥ 0.85
  in both directions. The cliff-pair PCA basis spans the basin-
  attractor mechanism for the entire controller family at Medium
  capacity. This would be the largest single ratchet of the v3 result
  — "the cliff pair has a 5-dim subspace" → "the whole basin-
  attraction phenomenon shares the same 5-dim subspace."
- **Y2-partial (default expectation):** ≥ 1 pair achieves median
  ≥ 0.7 in at least one direction, but at least one pair lags
  substantially. The basis transfers within a family but not across
  families, or transfers at Medium but not Small. Still informative;
  routes to v3.2 per-family or per-tier basis-comparison work.
- **Y2-falsifier:** all three pairs show median ≤ 0.4 in both
  directions. The 5-dim subspace is cliff-pair-specific; each
  collapse-cell pair has its own basin-attractor subspace, even if
  all are 5-dim. Routes v3.2 to per-pair PCA basis computation and
  a CKA-style basis-similarity comparison.

**Why these three pairs:** J1 tests cross-family generalization
(signature vs reward, not just λ-sweep); J2 tests within-family
robustness (does the basis work at a *different* λ-cliff in the same
L-Mixed family?); J3 tests cross-tier transfer (does Medium-derived
basis work at Small?).

v3.1.1 narrows the runnable J slate to J1/J2. The J3 cross-tier question is
still valuable, but it requires a Small-tier 64D basis or an adapter before
it can be interpreted.

### 3.3 Axis K — PC sparsity decomposition

For each principal component vector `v_i ∈ R^256` (i = 1..5),
compute:

- **L2-concentration on top-K neurons:** sort `|v_i|` descending,
  compute `||v_i[top_K]||² / ||v_i||²` for K ∈ {1, 4, 16, 32, 64}.
- **Effective dimensionality:** `(Σ |v_ij|)² / Σ v_ij²` (the
  participation ratio; high = distributed, low = concentrated).
- **Per-neuron rank:** identify the top-8 neurons by `|v_i|` for each
  PC; report whether the same neurons recur across PCs 2-5
  (mechanism-shared) or differ (mechanism-distributed across distinct
  neuron subsets).

**Predicted outcome (Y3):** PCs 2-5 are *moderately* concentrated —
top-32 neurons capture 40-70% of each PC's L2 norm. Effective
dimensionality 60-120 (out of 256). The "mechanism subspace" is
not a few isolated neurons (would be ≤ 16) but is also not fully
distributed (would be ≥ 200).

**Falsifier-light:** if top-32 captures < 20% (each PC genuinely
distributed across all 256 neurons), the v3.2 per-neuron causal
mediation axis is much less tractable.

**Falsifier-heavy:** if top-8 captures > 80% of each PC (the
mechanism really is a few neurons), v3.2 should pin those neurons by
name and try per-neuron patching — small and tractable.

This axis is descriptive, not a load-bearing claim. Output goes into
the v3.1 results note and routes v3.2 design.

### 3.4 Axis L — Directional-asymmetry seed-bootstrap

At K=3 in v3, P→C patch_success median = 0.881 vs C→P median = 0.509.
That's a 0.37 gap. Question: is the gap statistically reliable across
the seed slate, or could it be seed-sampling noise?

**Protocol:**

1. Load the existing axis-h-pca-k3 per-seed patch_success CSV.
2. Resample seeds with replacement 1000 times.
3. For each resample, compute median patch_success in both directions
   and the gap (P→C median − C→P median).
4. Report 2.5/97.5 percentile CI on the gap.

**Predicted outcome (Y4):** 95% CI on the gap excludes zero, with
lower bound > 0.15. The asymmetry is reliable.

**Falsifier:** 95% CI includes zero. The K=3 asymmetry observed in v3
is seed-noise, not a structural property of the cliff. Would
deflate the "becoming protected is mechanically more constrained than
becoming collapsed" reading from v3.

Also run the bootstrap at K=5 for completeness: at K=5 the gap is
much smaller (0.092), so the question is whether even *that* gap is
reliable.

## 4. Pre-Registered Predictions

Four load-bearing predictions, each with explicit falsifier:

### 4.1 (Y1) PC1 removal does not change patch_success

PCs 2-5 alone produce median patch_success within ±0.05 of K=5
baseline (P→C: 0.922; C→P: 0.830) in both directions.

**Falsifier:** patch_success drops by > 0.15 in either direction.
Would mean PC1 carries hidden mechanism not just policy-offset.

### 4.2 (Y2) Cliff-pair basis generalizes to held-out pairs

At least one of the three held-out pairs (J1/J2/J3) achieves median
patch_success ≥ 0.7 in at least one direction using the cliff-pair
PCA basis.

**Falsifier:** all three pairs show median ≤ 0.4 in both directions.
The basin subspace is cliff-pair-specific.

**Strong-confirmation flag (Y2-strong):** all three pairs achieve
median ≥ 0.85 in both directions. The basis spans the whole
controller-family basin mechanism — the largest single v3.1 ratchet.

**v3.1.1 amendment:** Y2 is evaluated over the runnable dimension-compatible
pairs J1/J2 only. J3 is not included in the Y2 denominator until v3.2 adds a
Small-tier basis or an explicit cross-tier adapter.

### 4.3 (Y3) PC2-5 sparsity is moderate

Top-32 of 256 net.7 neurons capture 40-70% of each PC's L2 norm.

**Falsifier:** top-32 captures < 20% (fully distributed) or top-8
captures > 80% (very sparse). Either falsifier routes v3.2 design.

### 4.4 (Y4) Directional asymmetry is statistically reliable

95% bootstrap CI on the K=3 directional gap (P→C median − C→P median)
excludes zero with lower bound > 0.15.

**Falsifier:** 95% CI includes zero. K=3 asymmetry is seed-noise.

## 5. Held-Out Policy Manifest

Phase 6 v3.1 evaluates the cliff-pair PCA basis on five policy slugs
beyond the cliff pair:

| policy_id | label | tier | role |
| --- | --- | --- | --- |
| `signature_terminal_medium` | L-Sig-Terminal-M | Medium | J1 hold side |
| `reward_lambda_1_0_medium_anchor` | L-Reward-M λ=1.0 | Medium | J1 collapse side |
| `mixed_lambda_0_9_medium_v3` | L-Mixed-M λ=0.9 | Medium | J2 hold side |
| `mixed_lambda_0_99_medium_v4` | L-Mixed-M λ=0.99 | Medium | J2 collapse side |
| `signature_terminal` | L-Sig-Terminal-S | Small | J3 hold side |
| `curriculum_reward_then_terminal_sig_v3` | Curriculum reward→terminal-sig | Small | J3 collapse side |

All checkpoints exist in `results/mesa/phase2-matched-capacity/checkpoints/`
(cross-reference `policies-inventory.csv` for canonical training slugs).

J3 checkpoints exist, but their architecture is dimension-incompatible with
the Medium cliff-pair basis. The harness should fail fast if J3 is attempted
with `--layer net.7` or with a 256D basis.

## 6. Metrics

### 6.1 Axis I (PC2-5 patching)

- `patch_success_pc2_to_5_mean / median / ratio_of_means` per direction.
- Δ vs K=5 baseline (P→C: 0.922 / C→P: 0.830).

### 6.2 Axis J (generalization)

Per held-out pair:
- `patch_success_mean / median / ratio_of_means` per direction.
- Δ vs cliff-pair K=5 baseline.
- `baseline_gap` (clean collapsed obp − clean protected obp) for the
  pair (different pairs will have different gaps; report alongside
  patch_success to avoid misreading large gaps as outsized effect).

### 6.3 Axis K (sparsity)

Per PC i ∈ {1, 2, 3, 4, 5}:
- `l2_top_K_concentration` for K ∈ {1, 4, 16, 32, 64}.
- `effective_dimensionality` (participation ratio).
- `top_8_neuron_indices` (which net.7 neurons dominate this PC).
- `pc_overlap_jaccard` (Jaccard similarity between top-8 neurons of
  PC_i and PC_j for all i ≠ j ∈ {2, 3, 4, 5}; high overlap →
  mechanism-shared neurons).

### 6.4 Axis L (bootstrap)

For K ∈ {3, 5}:
- `bootstrap_median_gap_mean`, `bootstrap_median_gap_lo_95`,
  `bootstrap_median_gap_hi_95`.
- `gap_excludes_zero` (boolean).

## 7. Harness Extensions

`training/mesa/phase6_v2_sae.py` gains four subcommands:

```bash
# Axis I: PC2-5 alone (skip PC1)
python -m training.mesa.phase6_v2_sae axis-i-pc-mech \
    --seeds 64 --layer net.7

# Axis J: apply cliff-pair basis to held-out pairs
python -m training.mesa.phase6_v2_sae axis-j-generalization \
    --pair J1  # or J2, J3
    --seeds 64 --layer net.7

# Axis K: PC structure decomposition (descriptive)
python -m training.mesa.phase6_v2_sae axis-k-decompose

# Axis L: directional-asymmetry seed-bootstrap
python -m training.mesa.phase6_v2_sae axis-l-bootstrap \
    --k 3  # or 5
    --resamples 1000
```

Implementation notes:

- **Axis I** reuses `axis_h_pca_patch` machinery, just drops the first
  column of the orthonormal basis Q before passing to
  `run_subspace_patch_battery`. ~20 LOC.
- **Axis J** requires generalizing `run_subspace_patch_battery` to
  accept arbitrary `PolicySpec` args for protected/collapsed instead
  of hard-coding the cliff pair. Then a new dispatch function looks
  up the (J1/J2/J3) pair, loads the cliff-pair PCA basis from
  axis-h-pca-k5 manifest, and runs the patch battery. ~80 LOC
  including the PolicySpec lookup table for the held-out policies.
- **Axis K** loads `Q_cliff` and computes the sparsity / participation
  / overlap metrics. Pure numpy, no env rollouts. ~30 LOC.
- **Axis L** loads `axis-h-pca-k3/axis-h-pca-patch.csv` (per-seed
  data), bootstraps with numpy, reports CI. ~40 LOC.

Total harness extension: ~170 LOC of additions to the existing
phase6_v2_sae.py.

## 8. Outputs

```
results/mesa/phase6-v3-1-validation/
  manifest.json
  axis-i-pc-mech/
    pc-mech-patch.csv
    pc-mech-patch-aggregate.csv
    v3-vs-v3-1-comparison.csv      # K=5 baseline vs PC2-5 alone
  axis-j-generalization/
    j1-patch.csv
    j2-patch.csv
    j3-patch.csv
    generalization-summary.csv     # rows: pair × direction; cols: patch_success
  axis-k-decompose/
    pc-sparsity-table.csv          # rows: PC index; cols: l2 concentrations
    pc-top8-neurons.csv            # rows: PC index; col: neuron indices
    pc-overlap-jaccard.csv         # pairwise Jaccard
  axis-l-bootstrap/
    bootstrap-gap-k3.csv
    bootstrap-gap-k5.csv
    bootstrap-summary.json
  reports/
    summary.json                    # Y1-Y4 outcomes
    v3-1-cascade.md                 # narrative summary for upstream
```

## 9. Execution Order

Recommended sequencing for Phase 6 v3.1:

1. **Harness extensions land first.** Generalize
   `run_subspace_patch_battery` to accept arbitrary policy pairs.
   Smoke-test with the cliff pair to verify identity to v3 axis-h
   output.
2. **Axis K decompose** first — pure numpy, no env rollouts. ~30 sec.
   Surfaces sparsity finding before patch batteries run, informs how
   to interpret subsequent axes.
3. **Axis L bootstrap** second — also no env rollouts; runs against
   the existing axis-h-pca-k{3,5} CSVs. ~10 sec.
4. **Axis I PC2-5 patch** third — quickest of the patch batteries
   because it's the same pair as v3 K=5. ~10 min.
5. **Axis J generalization** last — three pair runs, each ~10 min.
   ~30 min total. J1 (canonical signature-vs-reward) is the most
   informative; run it first as a smoke gate. If J1 shows
   patch_success ≤ 0.2 in both directions, the basis is
   cliff-pair-specific and J2/J3 are sanity-only; if J1 succeeds, run
   J2 and J3.
   v3.1.1 amendment: run J2 after a positive J1; do not run J3 until a
   Small-tier basis or cross-tier adapter exists.
6. **v3.1 result note** at `docs/mesa/PHASE6_V31_RESULTS.md` once all
   four axes land.

Total wall-clock: ~60-90 min for the full v3.1 slate.

## 10. Exit Criterion

Phase 6 v3.1 complete when:

- All four subcommands land in `phase6_v2_sae.py` and produce the
  named output CSVs.
- Axis K sparsity decomposition is complete (one of Y3-confirm,
  Y3-falsifier-light, Y3-falsifier-heavy).
- Axis L bootstrap CI on K=3 gap is reported (Y4 confirmed or
  falsified).
- Axis I PC2-5 patch battery is complete (Y1 confirmed or falsified).
- Axis J generalization is complete for at least J1 (canonical
  signature-vs-reward); if J1 clears the Y2 threshold, J2 and J3 are
  also run.
  v3.1.1 amendment: J2 completes the runnable Medium generalization slate;
  J3 is recorded as dimension-blocked and deferred to v3.2.
- The Y2 outcome is classified as one of: strong-confirm, partial,
  or strong-falsifier.
- v3.1 result note (`docs/mesa/PHASE6_V31_RESULTS.md`) is written
  with the four Y-prediction outcomes as the headline.

## 11. Cross-References

- **Phase 6 v2/v3 spec / results:** the 5-dim subspace claim v3.1
  validates. [`PHASE6_V2_SPEC.md`](PHASE6_V2_SPEC.md),
  [`PHASE6_V2_RESULTS.md`](PHASE6_V2_RESULTS.md).
- **Phase 6 v1 spec / results:** the layer-level locus.
  [`PHASE6_SPEC.md`](PHASE6_SPEC.md),
  [`PHASE6_RESULTS.md`](PHASE6_RESULTS.md).
- **Phase 7 v1 results:** envelope class membership for held-out
  policies in axis-J. [`PHASE7_RESULTS.md`](PHASE7_RESULTS.md).
- **Phase 5 v4 results:** behavioral context for the cliff.
  [`PHASE5_RESULTS.md`](PHASE5_RESULTS.md).
- **Roadmap:** [`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md).

## 12. What v3.2 and Phase 7 v2 Inherit

If Phase 6 v3.1 confirms Y1+Y2+Y4, three downstream consumers gain
load-bearing surface:

- **The gravity claim's mechanistic anchor ratchets again** —
  "5-dim subspace within net.7 at the cliff pair" upgrades to "the
  controller family shares a 5-dim basin-attractor subspace at
  net.7." Cascades into PROMO_HIGHLIGHTS, claims-and-scope,
  SUNDOG_V_MESA, mesa.html.
- **The mesa.html §The Locus surface gains a J-axis sub-panel** —
  generalization patch_success across three held-out pairs becomes
  a small grid alongside the K-sweep chart.
- **v3.2 work routes naturally to per-neuron causal mediation at
  net.7** if Axis K shows moderate sparsity (Y3 confirmed). Test
  per-neuron interventions on the top-32 net.7 neurons identified
  by Axis K.

If Y2 falsifies (basis is cliff-pair-specific):

- The mechanistic claim stays at the cliff-pair-specific form
  established by v2/v3. The gravity claim's mechanistic anchor does
  not ratchet further at v3.1.
- v3.2 routes to per-pair PCA basis computation and CKA-style
  basis-similarity analysis to ask: *are the per-pair bases similar
  to each other in subspace overlap, even if they aren't identical?*
  That would still suggest a shared mechanism, just one expressed in
  pair-specific coordinates.

## 13. Versioning

- **v3.1 (2026-05-12)** — initial pin. Four axes: I (PC2-5 alone),
  J (generalization), K (sparsity decomposition), L (asymmetry
  bootstrap). No new training, ~170 LOC of harness extensions.
  Headline test is Axis J generalization to three held-out pairs;
  strong-confirmation outcome (Y2-strong) would ratchet the gravity
  claim's mechanistic anchor from cliff-pair-specific to
  controller-family-wide.
- **v3.1.1 (2026-05-12)** - integrity amendment. J3 is marked
  dimension-blocked because Small-tier final hidden activations are 64D
  while the Medium cliff-pair PCA basis is 256D. v3.1 runs J1/J2 and
  defers cross-tier transfer to v3.2 with a Small-tier basis or adapter.
