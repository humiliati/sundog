# N-4 result — the ε-essential region count is the learnable invariant (REDEEMS, with a named residual)

> **Slate hook:** [`ALGO_APPROX_CONJECTURE_SLATE_2.md`](ALGO_APPROX_CONJECTURE_SLATE_2.md) N-4
> (the empirical redemption of the C-D2 grokking null). **Verdict (2026-06-29): REDEEMS** — the
> ε-essential region count tracks a ReLU net's generalization-width threshold where the *exact*
> tropical piece count `k` does not, and it moves with the measurement scale. One residual
> (the non-convex family's SGD-trainability gap) is named, not hidden. Empirical, CPU,
> deterministic. Script: `scripts/algo_approx_n4_eps_regions.py`.

## What C-D2 left open

C-D2 found that the exact piece count `k` does **not** predict the trained generalization-width
threshold, for two confounded reasons: **(1)** exact size ≠ ε-approximation complexity (convex
targets are coarsely smooth, so a net clears the bar far below `k`), and **(2)** existence ≠
trainability (the non-convex "jagged" target's threshold was erratic because SGD failed to find
representable solutions). N-4 tests whether the **ε-essential** count — the minimum pieces to
approximate the target *within the same variance bar the threshold uses* — is the predictor.

## Method (two upgrades over C-D2)

1. **ε-essential count** via an **optimal** piecewise-linear segmentation (DP over least-squares
   segment fits): the smallest `m` whose best `m`-segment fit leaves held-out frac-variance
   below the bar. Recomputed at each bar, so it *moves with ε* (exact `k` is constant).
2. A fitting **oracle**: best-of-6 restarts per hidden width, so the measured threshold reflects
   representational capacity rather than one SGD run's luck — the lever that addresses C-D2's
   trainability confound.

Families: smooth (tangent-to-parabola), essential (fixed slope jumps), jagged (non-convex random
slopes). `k ∈ {3,5,8}`, bars `∈ {0.10, 0.05, 0.02, 0.01}`. Prediction: `threshold ≈ ε-essential − 1`.

## Result (36 cells = 3 families × 3 k × 4 bars)

| comparison | within ±1 |
|---|---|
| threshold vs **(ε-essential − 1)** | **32/36 = 0.89** |
| threshold vs (exact `k` − 1) | 18/36 = 0.50 |

- **ε-essential spread across bars (mean per target): 1.33** — the invariant moves with the
  scale; exact `k` cannot (spread 0).
- **Per-family ε-tracking: smooth 1.00, essential 1.00, jagged 0.67.**

The convex families track the ε-essential count *exactly*. This is the clean redemption: the
C-D2 mystery — why smooth/essential generalize far below `k` — is now **explained**, not just
observed. They generalize at `≈ ε-essential − 1`, which is `≪ k` at coarse bars and rises toward
`k` as the bar tightens (e.g. smooth `k=8`: `threshold` 2→3 as `ε-essential` 2→4 over the bar
sweep). The exact piece count is the `bar → 0` limit, an upper bound that the operative threshold
only approaches asymptotically.

## The honest residual — the trainability confound, now isolated

The jagged family tracks at 0.67, with one clear outlier: jagged `k=5` sits at `threshold = 5`
across *every* bar while `ε-essential` is 2–3. There a low-width net *could* represent an adequate
approximation (the optimal 2–3-segment fit clears the bar) but the oracle's restarts did not find
it — exactly C-D2's **existence ≠ trainability** confound, surviving the best-of-restarts lever
on this non-convex shape. So N-4 **redeems confound (1)** — ε-essential is the right geometric
invariant — and **cleanly isolates confound (2)**: generalization onset = ε-essential region
count (geometry) **modulo** SGD trainability (a separate, non-convex-specific gap).

## Honest verdict

The slate's claim — *the ε-essential region count is the learnable invariant* — holds: it tracks
the trained threshold at 0.89 (vs 0.50 for exact `k`) and is the only one of the two that moves
with the operative scale. The falsifier `EPSILON_REGIONS_ALSO_FAIL` does **not** fire. The
qualifier, stated plainly: the prediction is of the **representational/generalization threshold**
up to **SGD trainability**, which remains a distinct factor on non-convex targets (the lane's
standing existence-vs-training wall). This converts C-D2's documented null into a positive: region
geometry *is* the predictor — once counted at the scale that matters, and read as geometry-plus-
trainability rather than one number for everything.
