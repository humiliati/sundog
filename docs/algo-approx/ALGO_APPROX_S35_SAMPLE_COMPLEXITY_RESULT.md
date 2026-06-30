# S3-5 — the ε-essential region count predicts *sample* complexity (result)

**Hook.** [Slate 3](ALGO_APPROX_CONJECTURE_SLATE_3.md) S3-5. N-4 showed the ε-essential region
count predicts a ReLU net's generalization-**width** threshold (the *capacity* axis), modulo SGD
trainability. S3-5 asks a second question: at **fixed, generous width** (capacity removed as the
bottleneck), does the ε-essential count also predict the **sample complexity** — the training-set
size needed to generalize within the bar? If region geometry is the learnable invariant, it should
govern *data* as well as capacity.

**Verdict: CONFIRMS** (both noiseless and noisy regimes). The ε-essential region count tracks the
sample threshold (Spearman 0.68 / 0.78); the exact piece count `k` does not (≈ −0.15 / 0.03). The
named falsifier `EPS_NOT_SAMPLE_PREDICTOR` did **not** fire.

Script: [`scripts/algo_approx_s35_sample_complexity.py`](../../scripts/algo_approx_s35_sample_complexity.py)
(deterministic, CPU; reuses the N-4 harness: `target_fn`, `eps_essential_pieces`, the `MLP`).

## Design (decoupled from N-4)

- **Width fixed** at `2k + 8` (far above any ε-essential here), so the net can always *represent*
  the target — the only thing varied is how much data it sees. This is the deliberate decoupling
  from N-4, whose story was about capacity.
- **Sweep `n_train`** over a geometric grid `{4, 8, 16, …, 2048}`; **best-of-4-restarts** oracle
  (the threshold reflects learnability from that much data, not one SGD run's luck); a **fixed,
  large (2000-pt) clean held-out test** per target for a stable frac-variance measurement.
- `sample_threshold(bar)` = smallest `n_train` whose held-out frac-variance-unexplained `< bar`.
- Two regimes: **noiseless** training labels, and **label noise** `N(0, (0.15·std f)²)` (the
  principled statistical-sample-complexity setting — to recover the true function under noise you
  must average noise out *per piece*, so the number of pieces genuinely drives the data need).
- Scored across the 3 families × 3 `k` × 4 bars (= 36 cells) by Spearman rank correlation of
  `sample_threshold` against ε-essential vs against exact `k`, plus a within-target monotonicity
  check (tighter bar ⇒ more data).

## Results

| regime | Spearman(st, ε-essential) | Spearman(st, exact k) | monotone (tighter bar → more data) |
|---|---|---|---|
| noiseless | **0.675** | −0.149 | 9/9 |
| noise = 0.15·std | **0.784** | 0.028 | 9/9 |

`st` = `sample_threshold`. The sample threshold spans `4 → 32` across cells (not floored), so the
axis genuinely discriminates. ε-essential is the clear predictor; exact `k` is flat-to-negative —
because `k` is constant across bars for a given target, while both the sample threshold and the
ε-essential count rise as the bar tightens, and they rise *together*. The noise regime is the
cleaner test: exact `k` collapses to correlation ≈ 0 (entirely uninformative), while ε-essential
strengthens to 0.78.

## Honest notes

- **A flooring artifact was caught and fixed.** The first cut used a grid starting at `n = 8`;
  every target cleared even the tightest bar at `n = 8`, so the threshold was floored and the axis
  was trivial (an *untestable*, not a null, regime — noiseless 1-D PL targets are learnable from a
  handful of points). Lowering the floor to `n = 4` (and adding the noisy regime) exposes the
  gradient. The script reports a `FLOORED` verdict explicitly when ≥ 60% of targets sit at the grid
  minimum, so this failure mode is self-declaring.
- **Scope.** This is an empirical result on 1-D PL target families with a fixed architecture and an
  SGD oracle — a directional confirmation that region geometry governs the *data* axis, consistent
  with the N-4 *capacity* result. It is not a sample-complexity *theorem*; the trainability factor
  is bounded (not removed) by the best-of-restarts oracle, exactly as in N-4.
- Together with N-4, the ε-essential region count is the learnable invariant on **both** axes it
  has been tested on: capacity (width) and data (samples). Exact `k` predicts neither.

> Cross-links: [`ALGO_APPROX_CONJECTURE_SLATE_3.md`](ALGO_APPROX_CONJECTURE_SLATE_3.md) ·
> [`ALGO_APPROX_N4_EPS_REGIONS_RESULT.md`](ALGO_APPROX_N4_EPS_REGIONS_RESULT.md) ·
> [`SUNDOG_V_ALGO_APPROX.md`](SUNDOG_V_ALGO_APPROX.md).
