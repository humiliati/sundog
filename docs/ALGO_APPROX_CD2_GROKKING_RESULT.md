# C-D2 result — generalization-width threshold vs tropical complexity (INCONCLUSIVE / refined)

> **Slate hook:** [`ALGO_APPROX_CONJECTURE_SLATE.md`](ALGO_APPROX_CONJECTURE_SLATE.md) C-D2.
> **Verdict (2026-06-27): the naive prediction is NOT supported; the literal falsifier
> `GROKS_FAR_BELOW_COMPILED_SIZE` fires, but for two confounded reasons that are
> themselves the finding.** Empirical, CPU, deterministic. Scripts:
> `scripts/algo_approx_cd2_grokking.py` (smooth/essential + grokking dynamics),
> `scripts/algo_approx_cd2_jagged.py` (non-convex decisive run).

## Prediction tested

`compileToDag` proves an `N`-node tropical circuit compiles to a `≤ 4N`-gate ReLU net.
Naive C-D2 prediction: a trained ReLU MLP's **generalization-width threshold** (smallest
hidden width whose held-out error clears a 99%-variance bar) should grow `~ k-1`, linear in
the target's tropical piece-count `k` (`compileToDag` size as the upper-bound target).

## Result — threshold(k) for k = [2, 3, 4, 6, 8] (median over 3 seeds; predicted ~k−1)

| target family | threshold(k) | reads as |
|---|---|---|
| smooth (tangent-to-parabola; slope jumps shrink with k) | `[3, 3, 5, 4, 4]` | flat ~3–5, far below k |
| essential (fixed slope jumps; convex) | `[3, 3, 3, 4, 5]` | rises *sub-linearly*, far below k |
| jagged (non-convex, random slopes) | `[3, 3, 10, 3, 5]` | erratic, non-monotone |

Grokking-dynamics probe (small data + weight decay): **no delayed generalization** — test
error tracks train error from the same epoch. PL regression does not "grok" the way
modular arithmetic does.

## Why it is inconclusive — two named confounds (the actual finding)

1. **Exact size ≠ ε-approximation complexity (smooth/essential).** Both convex targets are
   smooth at coarse scale (even the "essential" one is the convex envelope of evenly-spaced
   slopes ≈ a discretized parabola), so a width-`w ≪ k` ReLU net clears the 99% bar by
   approximating the envelope. Generalization rightly happens far below the *exact* piece
   count because the *effective* (ε) complexity is far below it. This is the lane's
   **exact-vs-ε seam** (cf. H-A1/H-A4) showing up empirically.
2. **Existence ≠ trainability (jagged).** For the non-convex jagged target — built so every
   piece is essential (ε-complexity ≈ k) — the thresholds are *erratic* (`[3,3,10,3,5]`).
   The `k=4` outlier sat at rel-error ≈ 0.68 for widths 3–9 and only fit at `w=10`: **SGD
   failed to find a representable solution** at the capacity threshold. The measured number
   reflects optimization difficulty + per-instance variance, not representational capacity.
   This empirically hits the lane's **standing imported wall**: `compileToDag` bounds what a
   net *can* represent (existence); whether SGD *finds* it is never claimed and here visibly
   fails.

## Honest verdict

`GROKS_FAR_BELOW_COMPILED_SIZE` **fires literally** — generalization onset sits far below
the exact compiled size across all three families — but it **refines rather than refutes**
the theory: the exact compiled size is an *upper bound* that does **not** predict the
trained-generalization threshold. The operative quantities are (i) **ε-approximation
complexity** and (ii) **SGD trainability** — exactly the two walls the lane has named
throughout (exact-vs-ε; existence-vs-training). No promotion; this is a documented,
informative negative.

## What a clean test would need (not pursued)

A target with **ε-complexity ≈ exact complexity AND reliably trainable** at the capacity
threshold — hard to arrange in 1-D, since high-ε-complexity (jagged) targets are exactly the
ones SGD struggles to fit at minimal width. A fairer probe would decouple the two confounds:
fix a trainable architecture and measure *representational* threshold via a fitting oracle
(e.g. exhaustive small-width fit), not SGD. That is a separate experiment, not run here.
