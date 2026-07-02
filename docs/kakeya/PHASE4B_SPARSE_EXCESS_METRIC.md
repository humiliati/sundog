# Kakeya Phase 4B - Sparse Excess-over-Control Metric (H-K4 redemption)

- Artifact id: `KAK-PHASE4B-SPARSE-EXCESS-METRIC`
- Date: 2026-06-29
- Status: internal measurement receipt. **Falsifier `SPARSE_EXCESS_NO_METRIC` CLEAR - the refined
  metric survives its own pre-registered kill gates.**
- Prior: [`PHASE4_ADAPTIVE_FIBERING_PANEL.md`](PHASE4_ADAPTIVE_FIBERING_PANEL.md) (the null this
  redeems: raw gap = density artifact; the only candidate fix named there is exactly this metric)
- Slate hook: [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md) (H-K4)
- Script: [`../../scripts/kakeya-sparse-excess-metric.mjs`](../../scripts/kakeya-sparse-excess-metric.mjs)
- Output: [`../../results/kakeya/sparse-excess-metric/manifest.json`](../../results/kakeya/sparse-excess-metric/manifest.json)

## The metric (as the null prescribed)

```
excess(K) = gap(K) - mean gap of size-matched random control,   restricted to |K| <= q^2/2
```

where `gap = adaptive_covered - fixed_best_covered` (PHASE4's statistic). The PHASE4 null showed
the *raw* gap is meaningless on dense bodies; this run pre-registered three kill gates for the
*excess* form and ran them at `q in {5, 7, 11}` (controls: 500 random bodies per size; holdout:
200 fresh; deterministic seeds).

## Pre-registered gates - all pass at every q

```text
q=5   tau=2  structured[k2:4]                    decoys[0]        fpr=0  boundary 13 (heuristic 12)
q=7   tau=3  structured[k2:6  k3:12]             decoys[0 0]      fpr=0  boundary 28 (heuristic 24)
q=11  tau=5  structured[k2:10 k3:20 k4:30 k5:40] decoys[0 0 0 0]  fpr=0  boundary 71 (heuristic 60)
KAK_SPARSE_EXCESS_METRIC falsifier=clear
```

- **G1 signal:** sparse `k`-direction line unions (`k >= 2`) score positive, strictly increasing
  excess. In fact the values are exact: **`excess = (k-1)(q-1)`** - in the sparse regime the
  control mean is 0 and a `k`-union's gap is `size - q = (k-1)(q-1)`. The metric has a closed
  form on its calibration family, the same texture as H-K3's `q(q^2-q+1)` law.
- **G2 specificity:** 200 fresh holdout random bodies per structured size (not used to fit the
  control): **false-positive rate 0** at threshold `tau_q` = half the smallest structured excess.
  The control generalizes; the sparse-regime signal is not an overfit.
- **G3 structure, not size:** broken decoys - same size as each `k`-union but every constituent
  line missing one point (replacements random, verified line-free) - score **exactly 0**. The
  metric responds to surviving multi-direction full-line structure, not to point count or
  clustering.

Report-only extra: the empirical density boundary (first size where the control mean leaves 0)
sits **above** the `q^2/2` heuristic at every q (13 vs 12, 28 vs 24, 71 vs 60), so the
pre-registered sparse fence is conservative - inside it the control is exactly quiet.

## What this redeems, precisely

PHASE4's verdict stands: the **raw** adaptive-over-fixed gap is not a metric. What is redeemed is
the panel's refinement clause: `gap - control`, sparse regime only, is a **controlled workbench
metric** - positive and exactly graded on multi-direction structure, zero on both random and
structure-broken same-size bodies. H-K4's "adaptive-fibering ambiguity" is now a displayable,
falsifier-fenced statistic rather than prose. The H-K5 table's methodological row ("a collision
number is only meaningful against a control") gains a constructive example: the control does not
just kill the artifact, it isolates the real signal.

## Interpretation Boundary

The metric counts surviving full-line multi-direction structure in a toy finite grid; "body
resistance" remains an interpretive gloss, not a theorem. Calibration family = k-direction line
unions and their broken decoys; other structured families may score differently and would need
their own gates before any claim. `q=11` is the largest registered field. No Euclidean Kakeya
content, no extremal-set search, no public claim; the HK6-K3 card's fence ("report gap minus
control, sparse regime only") is exactly the fence this receipt certifies.
