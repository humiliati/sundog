# Mesa Phase 6 v3.1 - Subspace Validation Result Note

This document records the Phase 6 v3.1 validation run for
[`PHASE6_V31_SPEC.md`](PHASE6_V31_SPEC.md). v3.1 stress-tests the
Phase 6 v2/v3 claim that the Medium cliff-pair basin attractor is carried by
a 5-dimensional PCA subspace at `net.7`.

Status: runnable v3.1 slate **complete** for Axis I, J1, J2, K, and L.
J3 is recorded as a spec-integrity block: it is Small-tier and 64D, while the
Medium cliff-pair PCA basis is 256D.

## 1. Summary

Phase 6 v3.1 gives four updates:

1. **Y1 falsified.** PC1 is not mechanism-empty. Removing PC1 and patching
   only PCs 2-5 drops median patch success from the K=5 baseline
   `0.922 / 0.830` to `0.291 / 0.121`.
2. **Y2 partially confirmed.** The cliff-pair basis generalizes strongly in
   the basin-inducing direction. J1 protected-to-collapsed median is `0.941`;
   J2 protected-to-collapsed median is `1.001`. Reverse transfer is weaker,
   especially J1 collapsed-to-protected at `0.162`.
3. **Y3 confirmed.** Top-32 neuron L2 concentration is moderate for every PC:
   `0.377` to `0.663`. No PC is fully distributed by the `<0.20` falsifier,
   and no PC is top-8 dominated by the `>0.80` falsifier.
4. **Y4 confirmed.** Directional asymmetry is statistically reliable. K=3
   bootstrap median-gap CI is `[0.251, 0.550]`, excluding zero with margin.

The mechanistic read sharpens: the 5D basis is not just a cliff-pair artifact,
but its most reliable transfer is "write basin attractor into a protected
policy," not "rescue a collapsed policy." PC1 carries real causal mechanism,
not merely policy offset.

## 2. Artifacts

Harness:

`training/mesa/phase6_v2_sae.py`

Outputs:

`results/mesa/phase6-v3-1-validation/`

Key files:

- `axis-i-pc-mech/pc-mech-patch-aggregate.csv`
- `axis-i-pc-mech/v3-vs-v3-1-comparison.csv`
- `axis-j-generalization/generalization-summary.csv`
- `axis-k-decompose/pc-sparsity-table.csv`
- `axis-l-bootstrap/bootstrap-summary.json`
- `pca-basis/cliff-pca-net7-seed10000-n64-h200-k5.npz`

The v3 artifact did not persist the PCA basis vectors themselves, so the
harness reconstructs the basis from the same matched cliff-pair activation
collection and caches it under the v3.1 output directory.

## 3. Axis I - PC2-5 Alone

| direction | K=5 baseline median | PC2-5 median | delta |
| --- | ---: | ---: | ---: |
| protected to collapsed | 0.922 | 0.291 | -0.631 |
| collapsed to protected | 0.830 | 0.121 | -0.709 |

Y1 falsified. PC1 is causally load-bearing. The prior "variance-heavy but
mechanism-empty" interpretation should be retired.

## 4. Axis J - Generalization

| pair | direction | median patch success | ratio of means |
| --- | --- | ---: | ---: |
| J1 signature-terminal-M vs reward-M | protected to collapsed | 0.941 | 0.917 |
| J1 signature-terminal-M vs reward-M | collapsed to protected | 0.162 | 0.366 |
| J2 mixed-0.9-M vs mixed-0.99-M | protected to collapsed | 1.001 | 1.006 |
| J2 mixed-0.9-M vs mixed-0.99-M | collapsed to protected | 0.631 | 0.684 |

Y2 partially confirmed. The basis generalizes strongly in the
protected-to-collapsed direction across both runnable held-out pairs. It does
not strongly rescue collapsed policies, especially across the J1
signature-vs-reward family boundary.

J3 is not counted. It is a dimension-blocked cross-tier target, not a failed
generalization result.

## 5. Axis K - PC Structure

| PC | top-8 concentration | top-32 concentration | participation ratio |
| ---: | ---: | ---: | ---: |
| 1 | 0.289 | 0.663 | 53.5 |
| 2 | 0.293 | 0.655 | 53.5 |
| 3 | 0.155 | 0.377 | 114.1 |
| 4 | 0.201 | 0.514 | 84.5 |
| 5 | 0.212 | 0.537 | 79.0 |

Y3 confirmed. PCs are moderately concentrated but not single-neuron sparse.
PC3 is the most distributed component; PCs 1 and 2 are the most concentrated.

## 6. Axis L - Bootstrap

| K | observed median gap | 95% CI |
| ---: | ---: | --- |
| 3 | 0.373 | [0.251, 0.550] |
| 5 | 0.092 | [0.030, 0.162] |

Y4 confirmed. The K=3 asymmetry is not seed noise. Even K=5 retains a small
but positive asymmetry.

## 7. Interpretation

v3.1 moves the claim from "a 5D subspace exists in the cliff pair" to a more
specific statement:

> The Medium basin-attractor mechanism is carried by a moderately concentrated
> 5D `net.7` subspace that generalizes in the basin-inducing direction across
> held-out Medium pairs. PC1 is causally necessary, and reverse rescue is
> weaker than collapse induction.

This supports the fixed-attractor / final-hidden-gating story, but narrows the
strongest version: the subspace is not a symmetric universal rescue basis.

## 8. Next

Phase 6 v3.2 should decide between:

- Small-tier basis for the J3 cross-tier question.
- Per-pair PCA basis comparison with CKA or principal-angle overlap.
- Per-neuron mediation inside the moderately concentrated PC1/PC2 directions.

## 9. Versioning

- **v1 (2026-05-12)** - initial v3.1 result note. Records Axis I/J/K/L,
  classifies J3 as dimension-blocked, and updates the mechanistic read.
