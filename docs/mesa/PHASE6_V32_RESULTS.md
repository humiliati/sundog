# Mesa Phase 6 v3.2 — Top-k Neuron Mediation Result Note (Negative)

This document records the Phase 6 v3.2 result for
[`PHASE6_V32_SPEC.md`](PHASE6_V32_SPEC.md). v3.2 asked whether the
entangled 5-dimensional basin-attractor subspace at `net.7`
(established in v3 and characterized in v3.1) could be approximated
by patching only the top-k most-contributing neurons. The spec
included a smoke gate that, if failed, would deliberately stop the
full k-sweep and record a negative result.

Status: **smoke gate failed cleanly**. Z1 falsified by a wide margin.
The full k-sweep was deliberately not run per the spec. v3.3 routes to
non-linear neuron-attribution methods.

## 1. Summary

Phase 6 v3.2 gives the program one tightly-bound negative result with
a methodological lesson that stacks on v2's SAE-wrong-basis lesson:

1. **Top-k neuron-restricted patching does not deliver the v3 K=5
   patch effect.** Even at top-32 (12.5% of net.7's neurons,
   capturing 33.6% of the aggregate L2 across PCs 1-5), the patch
   produces ~0% effect in both directions. The 5D subspace is **not
   linearly decomposable into top-k neuron contributions** — masking
   the delta to a subset of neurons does not deliver a proportional
   fraction of the mechanism.
2. **The mechanism is non-additive across the neuron substrate.** The
   patched neurons need the unmasked neurons to remain coordinated
   for the post-patch activation pattern to drive basin behavior;
   leaving 87.5% of neurons at their unpatched values interferes with
   the partial patch even though those neurons hold only ~66% of the
   subspace L2.
3. **Methodological lesson stacks with v2.** v2 said *feature-
   availability rankings (e.g., SAE features ranked by per-episode
   correlation) do not surface mechanism even at strong correlation*.
   v3.2 says *linear additive top-k neuron restriction does not
   surface mechanism even with the right basis*. Future Phase 6+
   interpretability work in this regime should default to non-linear
   attribution methods.

The cleanest v3.2 headline: **the entangled 5D basin subspace at
`net.7` is implemented holistically across the layer's neuron
substrate; linear additive top-k mediation is not a tractable lens
for this circuit, and the gravity-claim mechanistic anchor stays at
"5D entangled subspace" without further decomposition to a few
specific neurons.**

## 2. Artifacts

Harness: `training/mesa/phase6_v2_sae.py` subcommand
`axis-m-neuron-mediation`.

Smoke gate output:

```
results/mesa/phase6-v3-2-neuron-mediation/axis-m-cliff-pair/top-32/
  patch.csv
  patch-aggregate.csv
  v1-vs-v3-comparison.csv
  neuron-ids-top-32.csv
  pc-l2-capture.csv
  manifest.json
```

PCA basis loaded from the v3.1 canonical reconstruction at
`results/mesa/phase6-v3-1-validation/pca-basis/cliff-pca-net7-seed10000-n64-h200-k5.*`.

Full k-sweep directories (k ∈ {8, 16, 32, 64, 256}) were **not
created** — the smoke gate stopped the v3.2 sweep before they would
have run.

## 3. Smoke Gate Outcome

Smoke run: `--top-k 32 --seeds 8 --layer net.7 --pair cliff`.

| metric | value | gate requirement |
| --- | ---: | --- |
| Top-32 captured L2 fraction (aggregate across PCs 1-5) | 0.3363 | (informational) |
| P→C median patch_success | **−0.006** | ≥ 0.37 |
| C→P median patch_success | +0.001 | (gate is P→C only) |

The gate is one-sided on P→C per spec §3.4: if any direction
concentrates, it should be the basin-inducing direction (per v3.1
Axis J showing basin-inducing generalizes more cleanly than
basin-resisting). The P→C result of −0.006 is essentially zero — not
a partial recovery, not a noisy positive trend, just a complete
failure to deliver mechanism through the top-32 mask.

**Per spec §3.4: stop. The full k-sweep is deliberately not run.**
Per spec §12 Z1-falsifier branch: route v3.3 to non-linear
attribution methods.

## 4. Mechanistic Interpretation

The 33.6% L2 capture vs 0% patch effect contrast is the load-bearing
pair of numbers. If the mechanism were *linearly distributed* across
neurons (each neuron contributing patch effect proportional to its
L2 share of the PCA basis), top-32 would deliver ~30-40% of patch
success. Instead it delivers 0%.

Two non-mutually-exclusive readings:

- **Coordination reading.** The post-patch activation pattern works
  as a *whole*: the patched neurons need the unmasked neurons to be
  at their *patched* values too, or downstream weights' linear
  combinations across the layer produce action logits that don't
  reflect the intended subspace shift. Masking 87.5% of neurons
  leaves them at unpatched values, which interferes with the
  intended pattern at the patched neurons even though those
  unmasked neurons hold only ~66% of basis L2.
- **Non-linear-readout reading.** The policy_head downstream of
  `net.7` may apply a Tanh + Linear transformation whose output is
  sensitive to *configurations* of net.7 activations rather than
  individual coordinates. Linear additive substitution in a 5D
  subspace works (v3 K=5); linear additive substitution restricted
  to a neuron subset doesn't because the same subspace shift,
  delivered partially, produces a different downstream configuration
  than the subspace shift delivered fully.

Both readings predict that *non-linear* attribution methods —
zero-ablation (per-neuron removal scored by behavioral cost),
integrated gradients (per-neuron gradient credit along the integration
path), or causal scrubbing (replacing activations with structurally-
matched samples) — should produce different, more informative results.
That is the v3.3 routing.

## 5. Reconciliation with v3.1 Axis K

v3.1 Axis K reported per-PC top-32 L2-concentration in the range
0.377-0.663 (PC1 had the highest concentration; PCs 2-5 spanned the
range). v3.2's aggregate top-32 captures 0.336 of total L2 across
all five PCs.

Why is 0.336 lower than the per-PC range's lower bound of 0.377?

The per-PC top-32 mask is *different per PC* — PC1's top-32 neurons
need not be the same as PC2's top-32. v3.2's aggregate ranking picks
one fixed top-32 set by `Σ_i v_ij²`, which is the *intersection-ish*
of the per-PC top-32 sets weighted by total contribution. A neuron
that's #1 in PC2 but #200 in PC1/PC3/PC4/PC5 has low aggregate L2
even though it dominates PC2.

The 0.336 / 0.377-0.663 numbers are therefore mutually consistent and
do not conflict. v3.2's aggregate ranking is the *most parsimonious*
single-mask version of "top neurons across all five directions"; it
captures less L2 than any per-PC top-32 ranking but cleanly answers
the v3.2 question of "does a single neuron subset mediate the v3 K=5
patch?"

Alternative rankings — per-PC top-32 union, signed-L2 per PC, weighted
by patch_success-per-PC — are v3.3 candidates if the program decides
to revisit top-k mediation with a more sophisticated ranking. v3.2's
clean failure suggests this is low-priority.

## 6. Methodological Lesson (stacks with v2)

The Phase 6 program now has two stacked methodological lessons about
mechanism-localization in this controller family:

- **v2 lesson:** *Feature-availability rankings on a joint two-policy
  dataset return policy-identifier features rather than mechanism
  features.* SAE features with |corr| = 0.89 against `basin_pref_
  intervened` produced zero patch effect because the correlation was
  dominated by between-policy mean differences rather than the
  causal-mechanism subspace.
- **v3.2 lesson:** *Linear additive top-k restriction within the
  correct basis (PCA on per-step matched-seed diffs) still does not
  surface mechanism.* The basin-attractor circuit is implemented
  holistically across the neuron substrate; partial delivery of the
  subspace shift fails even when the basis is correct.

Together these lessons say: for mechanism in this circuit,
**non-linear attribution is mandatory; linear analysis methods (probes,
SAE features, top-k subspace restriction, top-k neuron restriction)
all fall short**.

The result is preserved as a methodological note for future
mesa-trap work and for any reviewer who reaches for the
linear-interpretability toolkit first.

## 7. v3.3 Routing

Per spec §12 Z1-falsifier branch:

> v3.2 records the negative result: the 5D subspace is mediated by
> a non-linearly-distributed neuron pattern that linear top-k
> ablation can't reproduce.
> v3.3 routes to non-linear attribution methods: zero-ablation
> attribution scores per neuron, integrated gradients on net.7 →
> action logits, or causal scrubbing.

Three candidate v3.3 axes, in order of compute-tractability:

1. **Axis N — Zero-ablation attribution.** For each neuron j ∈
   {0..255}, set neuron j's activation to zero at the patch step
   (analogous to dropout at a single neuron), rerun the v3 K=5
   patch battery, measure how much patch_success drops. The
   "critical" neurons are those whose ablation drops patch_success
   most. Compute: 256 neurons × 8-16 seeds × 4 forwards ≈ ~15-30
   minutes wall-clock. This is the dual of v3.2: instead of asking
   "which neuron subset is *sufficient*," ask "which neurons are
   *necessary*." Critical-neuron sets often differ from
   high-contribution sets in non-linear systems.
2. **Axis O — Integrated gradients on net.7 → action logits.**
   Standard gradient-based attribution. For each (seed, step),
   compute gradient of action logit w.r.t. net.7 activation,
   integrate from a baseline (zero or matched-protected activation),
   sum per-neuron credit. Compute: ~10 minutes with PyTorch autograd.
   Surfaces per-step neuron credit independent of the patching
   apparatus; tests whether basin-resisting and basin-inducing have
   different gradient-attribution structure.
3. **Axis P — Causal scrubbing.** Replace `net.7` activations with
   structurally-matched samples drawn from a hypothesized circuit
   model (e.g., "the basin attractor is implemented as a downstream
   readout that gates on the projection along PC2"). More work; more
   informative when there's a specific hypothesis to test. v3.3.1
   candidate after Axis N / Axis O surface targets.

Recommended v3.3 sequencing: Axis N first (cheapest, dual to v3.2,
direct comparison possible), then Axis O if Axis N surfaces no
critical neurons.

## 8. Open Edges

v3.2 closed cleanly but left two questions worth flagging for v3.3:

- **Does the directional asymmetry from v3.1 carry into zero-ablation
  attribution?** v3.1 Y4 confirmed P→C and C→P have statistically
  different patch_success structure. If zero-ablation attribution
  surfaces different critical-neuron sets for P→C vs C→P, that
  generalizes the asymmetry to neuron-level mechanism.
- **What is the relationship between aggregate-L2-rank and
  zero-ablation-rank?** v3.2 ranked neurons by L2; v3.3 Axis N will
  rank by ablation-cost. If the two rankings disagree substantially,
  that's a strong sign the L2-rank metric is misleading for this
  circuit and future interpretability work should default to
  ablation-based ranking.

These are v3.3 questions; not v3.2's to answer.

## 9. Versioning

- **v3.2 (2026-05-12)** — initial pin and negative result. Smoke gate
  ran with `--top-k 32 --seeds 8 --pair cliff`. Top-32 captured 33.6%
  of aggregate L2, delivered −0.006 P→C / +0.001 C→P median
  patch_success. Z1 falsified. Full k-sweep deliberately not run per
  spec §3.4. v3.3 routes to non-linear attribution methods (Axis N
  zero-ablation as first move).
