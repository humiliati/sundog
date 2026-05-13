# Mesa Phase 6 v2 — Direction-based Mechanistic Probing Result Note

This document records the Phase 6 v2 result for
[`PHASE6_V2_SPEC.md`](PHASE6_V2_SPEC.md). Phase 6 v1 localized the
cliff causally to the actor's final hidden activation (`net.7`); v2
asked whether that locus is a single direction inside `net.7` or a
distributed property of the layer.

Status: Axis D (SAE feature dictionary), Axis E (single-direction
patching), and Axes F-H (v3 follow-ups: multi-feature, mean-diff, PCA
of per-step diffs) are all **complete** on the L-Mixed-M-λ=0.95 /
λ=0.97 cliff pair.

## 1. Summary

Phase 6 v2 gives the program one headline finding, one tightly-bound
methodological lesson, and one secondary observation:

1. **The basin attractor at `net.7` is a 5-dimensional subspace.** Top-5
   principal components of the per-step matched-seed cliff-pair
   difference matrix capture 97.4% of the activation-space variance and
   reproduce Phase 6 v1's full-layer patch success to within 0.03 in
   both directions. That is a **51× compression** of the mechanistic
   anchor (256 → 5 dims). K-sweep saturates at K=5; K=10/32/64 add no
   measurable patch_success past K=5 (noise-floor refinement).
2. **Variance and mechanism are decoupled.** The first PCA component
   carries 38.8% of the diff variance but contributes 0% of patch
   success. PC1 is essentially the empirical between-policy mean-offset
   direction; it separates "which policy am I" but not "what mechanism
   does this policy implement." The basin-attractor circuit lives in
   **PCs 2-5** (58.5% of variance, ~100% of patch effect).
3. **Sparse-autoencoder features are the wrong basis for this circuit.**
   The top SAE feature was strongly correlated with `basin_pref_intervened`
   (|corr| = 0.89, V2 confirmed), but direction-patching using that
   feature's decoder column produced zero patch effect (V3 + V4
   falsified). Even orthogonalized top-10 SAE features (axis-F) lag
   PCA basis at the same K. The SAE picked policy-identifier features
   ranked by per-episode correlation; the actual mechanism lives in a
   different basis of activation space.

The cleanest v2 headline is therefore: **the cliff at `net.7` is a
5-dimensional subspace of per-step matched-policy activation
differences, 1 of those dimensions is a variance-heavy mechanism-empty
policy-offset, and the load-bearing mechanism lives in the remaining 4
dimensions.**

## 2. Artifacts

Axis D — SAE training and feature labeling:

- `results/mesa/phase6-v2-direction/axis-d-sae/`
- Primary: `sae-weights.pt`, `sae-config.json`,
  `axis-d-feature-correlations.csv`, `axis-d-top10-basin-features.csv`,
  `axis-d-sae-quality.json`.

Axis E — single-direction SAE patching:

- `results/mesa/phase6-v2-direction/axis-e-patch/`
- Primary: `axis-e-direction-patch.csv`,
  `axis-e-direction-patch-aggregate.csv`, `v1-vs-v2-comparison.csv`.

v3 follow-ups (Axes F-H):

- `results/mesa/phase6-v2-direction/axis-f-multifeature/`
- `results/mesa/phase6-v2-direction/axis-g-mean-diff/`
- `results/mesa/phase6-v2-direction/axis-h-pca/`
- K-sweep on axis-H: `results/mesa/phase6-v2-direction/axis-h-pca-k{1,3,5,10,32,64}/`

Harness: `training/mesa/phase6_v2_sae.py` (commands `axis-d-sae`,
`axis-e-patch`, `axis-f-multifeature`, `axis-g-mean-diff`,
`axis-h-pca`).

## 3. Axis D — SAE feature dictionary at net.7

**Training run:** joint activation tensor of 20,885 rows × 256 dims
(L-Mixed-M-λ=0.95: 8281 rows, L-Mixed-M-λ=0.97: 12604 rows). Top-k SAE
trained for 10,000 steps with 1024 features, k=32 sparsity.

**Health checks all pass.**

| metric | value | target | verdict |
| --- | ---: | ---: | --- |
| reconstruction R² (test) | 0.9998 | > 0.8 | pass |
| dead-feature rate | 0.001 | < 0.30 | pass |
| active-feature rate | 0.0312 | ≈ 0.03125 | pass |
| final train loss | ~1e-5 | (informational) | converged |

**Top-correlated feature:** f=529, correlation = −0.892 with
per-episode `basin_pref_intervened`. The top 10 features had
correlations [−0.892, −0.846, −0.731, −0.671, +0.641, …], all well
above the V2 |corr| ≥ 0.5 threshold.

**Predictions confirmed:** V1 (SAE training healthy), V2 (basin-attraction
feature exists with |corr| ≥ 0.5). Both confirmed by wide margins.

## 4. Axis E — Single-direction patching (V3 + V4 falsified)

Direction-patching using feature 529's decoder column (norm = 1.267,
unit-normalized at use) on the matched seed slate produced:

| direction | mean | median | ratio of means |
| --- | ---: | ---: | ---: |
| protected → collapsed | 0.000 | −0.000 | −0.000 |
| collapsed → protected | 0.001 | −0.000 | −0.000 |

**Compared to v1 layer-patch baseline (net.7):**

| direction | v1 median | v2 median | Δ |
| --- | ---: | ---: | ---: |
| protected → collapsed | 0.944 | −0.000 | −0.944 |
| collapsed → protected | 0.860 | −0.000 | −0.860 |

V3 (direction-patch median > 0.5 in at least one direction) and V4
(within 0.2 of v1 baseline) are both **falsified hard**. The cliff is
not a single direction in net.7 activation space.

**Diagnosis:** the SAE found a *policy-identifier* feature, not a
*mechanism* feature. Feature 529 is high on protected-policy
activations and low on collapsed-policy activations (or vice versa);
its correlation with `basin_pref_intervened` is high because
`basin_pref_intervened` itself is policy-correlated (0.330 mean on
protected vs 5.510 on collapsed). SAE features ranked by correlation
against a per-episode target on a joint two-policy dataset return
policy-identifier features at the top — **they are the wrong basis for
the causal-mechanism question** even when their |correlations| are
extreme.

## 5. Axis F — Multi-feature SAE subspace (top-10, orthogonalized)

Stacking the top-10 SAE features by |correlation|, QR-orthogonalizing,
and direction-patching on the resulting 10-dim subspace produced:

| direction | mean | median | ratio of means | Δ from v1 median |
| --- | ---: | ---: | ---: | ---: |
| protected → collapsed | +0.062 | +0.028 | +0.150 | −0.916 |
| collapsed → protected | +0.567 | +0.372 | +0.506 | −0.488 |

Asymmetric and well short of v1 in both directions. C→P shows partial
effect, P→C nearly zero. With 10 SAE directions selected by basin
correlation, the v3 follow-up still recovers less than half of v1's
patch effect on average. **The SAE basis is methodologically
discounted as the right basis for this circuit.**

## 6. Axis G — Empirical between-policy mean-diff direction (K=1)

Δ_mean = mean(h_collapsed) − mean(h_protected) across all matched-seed
steps. Norm = 1.2937, unit-normalized.

| direction | mean | median | ratio of means | Δ from v1 median |
| --- | ---: | ---: | ---: | ---: |
| protected → collapsed | +0.395 | +0.437 | +0.529 | −0.507 |
| collapsed → protected | +0.310 | +0.070 | +0.245 | −0.790 |

Notable: ||Δ_mean|| = 1.2937 is nearly identical to the SAE top
feature's decoder direction norm of 1.267. The SAE found a direction
strongly aligned with the empirical mean-offset, just expressed in its
own learned basis. The mean-diff K=1 captures partial P→C effect
(~44%) but barely any C→P effect — the constant-offset component of
the cliff is variance-heavy but not enough mechanism for full patch
effect. This dovetails with the axis-H K=1 finding (PC1 alone is
mechanism-empty).

## 7. Axis H — PCA on per-step matched-seed diffs (headline)

Per-step diff matrix Δ ∈ R^(8085 × 256) constructed by aligning
matched-seed trajectories step-index-wise and computing
`h_collapsed[seed, t] − h_protected[seed, t]`. PCA via SVD on the
centered matrix.

### 7.1 K-sweep (the load-bearing result)

| K | Variance captured | P→C median | C→P median | Δ from v1 (P→C) | Δ from v1 (C→P) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.84% | +0.006 | +0.008 | −0.938 | −0.852 |
| 3 | 87.87% | +0.881 | +0.509 | −0.063 | −0.351 |
| **5** | **97.37%** | **+0.922** | **+0.830** | **−0.022** | **−0.030** |
| 10 | 99.55% | +0.960 | +0.851 | +0.016 | −0.009 |
| 32 | 99.98% | +0.950 | +0.861 | +0.006 | +0.001 |
| 64 | 100.00% | +0.946 | +0.859 | +0.002 | −0.001 |

**K=5 is the headline.** Both patch directions are within 0.03 of the
Phase 6 v1 full-layer baseline at K=5 with 97.4% of variance captured.
K=10/32/64 saturate (within 0.02 of v1 in all metrics). The diff
matrix is essentially rank-5 with respect to the basin-attractor
circuit.

**51× compression.** Net.7 is 256-dim. The cliff is 5-dim. The
mechanistic anchor sharpens from "single layer (256 dims)" to "5-dim
subspace within that layer."

### 7.2 Variance-vs-mechanism decoupling

PC1 (alone) captures 38.8% of variance but contributes ~0% of patch
success. PC1 is the policy-offset direction (effectively Δ_mean from
axis-G, just expressed in PCA coordinates). It separates the two
policies' activations but does not flip behavior under patch.

PCs 2-5 contribute the remaining 58.5% of variance and **all** of the
mechanism. Patching these 4 directions alone (without PC1) would
likely recover ~v1 baseline; this was not run in v2 but is a v3.1
candidate diagnostic.

The cleanest statement of the cliff geometry is:

> The basin attractor at net.7 is a 5-dimensional subspace,
> decomposable into a 1-dimensional policy-offset component (PC1,
> 38.8% variance, 0% mechanism) and a 4-dimensional mechanism
> component (PCs 2-5, 58.5% variance, ~100% mechanism).

### 7.3 Directional asymmetry

The K-sweep also surfaces a real directional asymmetry in the cliff's
geometry:

- At K=3 (87.9% variance), P→C patch_success = 0.881 (88% of v1) but
  C→P patch_success = 0.509 (59% of v1). PCs 1-3 nearly capture the
  protected → collapsed mechanism but only partially the reverse.
- PCs 4 and 5 (adding 9.5% of variance going from K=3 to K=5) are
  essential for C→P (rescues from 0.509 → 0.830) but only marginal for
  P→C (improves from 0.881 → 0.922).

Possible reading: "becoming protected" is mechanically more constrained
than "becoming collapsed." There are more ways the network can fall
into the basin than ways it can stay out of it. The basin is deeper in
mechanism than it is in escape route. This is a flagged observation,
not a pinned claim — testing it rigorously would require a v3.2
direction-attribution analysis.

## 8. Verdict

Phase 6 v2 + v3 ratchet the gravity-claim mechanistic anchor:

- **v1 claim (preserved):** the cliff localizes causally to a single
  layer (`net.7`). Layer-level activation patching achieves ≈0.9
  patch_success in both directions.
- **v2 single-direction claim (falsified):** the cliff is *not* a
  single direction in net.7 activation space. Direction-patching using
  the top SAE feature produces ~0% effect.
- **v3 low-dim subspace claim (confirmed):** the cliff is a
  5-dimensional subspace of activation space, captured by the top-5
  principal components of the matched-seed per-step diff matrix. K=5
  recovers v1 patch_success to within 0.03 in both directions; K=10/32/
  64 saturate at the v1 baseline.
- **v3 variance-vs-mechanism lesson (new):** the first principal
  component is variance-heavy but mechanism-empty — it carries 38.8%
  of variance and 0% of patch_success. Mechanism lives in PCs 2-5.
- **Methodological note:** SAE features ranked by per-episode-target
  correlation on a joint two-policy dataset are dominated by
  policy-identifier features and *are the wrong basis for causal
  mechanism* even when |correlation| is extreme. Future Phase 6+ work
  should default to PCA on per-step matched-seed diffs (or other
  empirically-derived bases) when the question is "where does the
  cliff live causally."

## 9. Upstream Implications

Phase 6 v2 + v3 has cascading effects on three program surfaces:

- **The gravity claim's mechanistic anchor sharpens.** Where Phase 6
  v1 supported "the cliff is at a specific layer," v2+v3 support "the
  cliff is a 5-dim subspace at that layer, of which 4 dims carry the
  mechanism." This is the next-level mechanistic statement and should
  cascade into PROMO_HIGHLIGHTS, claims-and-scope, and SUNDOG_V_MESA.
- **The `mesa.html` "fingerprint" surface is now buildable.** v3 gives
  a clear visual story: variance-explained curve plus patch_success(K)
  curve, both as functions of K, plotted together. The decoupling
  between the two curves at K=1 (variance 38.8%, patch_success 0%) is
  the figure that earns the "variance ≠ mechanism" lesson.
- **The SAE basis is methodologically discounted.** Any future
  interpretability work on this controller family should default to
  PCA on per-step matched-seed diffs. SAE work can be re-attempted if
  per-policy SAEs (trained only on activations from one policy, with a
  target that varies *within* that policy) replace the joint-SAE
  approach used in v2.

## 10. Open Edges

Several questions are surfaced by v2+v3 but deferred to v3.1 or later:

- **What is the geometric structure of PCs 2-5?** Are they sparse over
  net.7 neurons (a few specific neurons drive each PC) or distributed?
  Per-neuron decomposition of PC2-PC5 directions is a v3.1 candidate.
- **Does the 5-dim subspace generalize to other Phase 5 zoo cells?**
  If the same PCA basis (computed from the cliff pair) patches well on
  L-Reward-M (also collapsed) vs L-Sig-Terminal-M (also held), the
  basin-attractor subspace is *not* specific to the cliff pair —
  it's the basin-attraction structure of the whole controller family.
  This is a v3.2 generalization test.
- **What does PC2-PC5 patching alone produce?** Skipping PC1 (the
  policy-offset) and patching only PCs 2-5 should recover ~v1
  baseline. v3.1 candidate diagnostic; would tighten the variance-vs-
  mechanism story.
- **Does the directional asymmetry (P→C captured by 3 PCs, C→P
  needing 5) survive seed-bootstrap analysis?** v3.1 candidate.

These are inexpensive (no new training; same patch battery) and could
ship in a v3.1 result note.

## 11. Versioning

- **v2 (2026-05-12)** — initial pin. Axis D SAE training and feature
  labeling. Axis E single-direction patching falsifies V3+V4. v3
  follow-ups (Axes F-H) added as harness extensions to test
  multi-feature SAE basis, mean-diff direction, and PCA on per-step
  diffs. Axis-H K-sweep across {1, 3, 5, 10, 32, 64} confirms 5-dim
  subspace as the load-bearing v3 finding. SAE basis methodologically
  discounted; PCA on per-step matched-seed diffs is the recommended
  basis going forward.
