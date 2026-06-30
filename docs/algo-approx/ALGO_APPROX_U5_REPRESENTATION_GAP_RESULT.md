# U-5 — The representation-vs-optimization gap (constructed net vs SGD)

**Result: REPRESENTATION_GAP — CONFIRMED.** Script:
[`scripts/algo_approx_u5_representation_gap.py`](../../scripts/algo_approx_u5_representation_gap.py).
Deterministic, CPU, best-of-5 restarts.

## The question

Every approximant in this lane is **constructed** — explicit weights with a machine-checked L∞
bound — never **trained**. The sharpest construction is the Telgarsky sawtooth for `x²` on `[0,1]`,

```
R_m(x) = x − Σ_{k=1}^m T^[k](x) / 4^k ,   T(x) = 1 − |2x − 1| ,
```

proved in `Sundogcert/SawtoothShared.lean` + `SawtoothDag.lean` to have L∞ error `≤ 1/(4·4^m)` at
depth `~m` and `O(m)` gates — accuracy **exponential in depth**. U-5 asks the optimization question:
at the *same depth*, does plain SGD find that accuracy, or does it plateau because it cannot exploit
depth?

## The measurement

Target `f(x) = x²` on `[0,1]`; L∞ error on a 2001-point grid; best-of-5 restarts.

| m | constructed `R_m` | `1/(4·4^m)` (proved) | SGD-deep (d=m, w=8) | SGD-shallow (d=1, matched params) | gap = SGD / constructed |
|--:|------------------:|---------------------:|--------------------:|----------------------------------:|------------------------:|
| 1 | 6.25e-2 | 6.25e-2 | 1.0e-2 | 1.0e-2 (w=8)   | 0.2× |
| 2 | 1.56e-2 | 1.56e-2 | 1.1e-2 | 5.4e-3 (w=32)  | 0.3× |
| 3 | 3.91e-3 | 3.91e-3 | 7.3e-3 | 3.5e-3 (w=56)  | 0.9× |
| 4 | 9.77e-4 | 9.77e-4 | 5.6e-3 | 2.0e-3 (w=80)  | 2.1× |
| 5 | 2.44e-4 | 2.44e-4 | 5.5e-3 | 2.6e-3 (w=104) | 10.5× |
| 6 | 6.11e-5 | 6.10e-5 | 3.0e-3 | 2.3e-3 (w=128) | 37.8× |
| 7 | 1.53e-5 | 1.53e-5 | 4.9e-3 | 1.7e-3 (w=152) | 111.4× |
| 8 | 3.87e-6 | 3.81e-6 | 3.8e-3 | 1.9e-3 (w=176) | **481.7×** |

- **Constructed L∞**, m=1→8: `6.25e-2 → 3.87e-6` (improved **16,132×**).
- **Best SGD L∞**, m=1→8: `1.01e-2 → 1.87e-3` (improved **5.4×**).
- **Gap**, m=1→8: `0.2× → 481.7×` (grew **~3000×**).

## Reading

- **Representability ≠ trainability.** The construction's accuracy decays *exponentially* in depth;
  SGD's best — across both a depth-`m` net **and** a matched-parameter shallow-wide net (the
  architecture SGD trains best), best-of-5 — **plateaus** around `2×10⁻³` and barely moves with depth.
  The depth-efficiency the construction exploits is **not accessed by plain SGD**. The construction is
  in Lean (`sq_sub_R_le`/`sawDag_polylog`); SGD is the optimization wall.
- **Where the gap turns on.** For `m ≤ 3` SGD matches or beats the construction (gap < 1) — when the
  target is easy and the construction is still coarse, optimization is not the bottleneck. The gap
  emerges only once the construction's exponential accuracy (`m ≥ 4`) outpaces the SGD plateau, and
  then grows sharply. So the gap is a *depth-efficiency* gap, not a blanket "SGD is worse."
- **A free cross-check.** The `constructed` column matches the proved bound `1/(4·4^m)` to ~3 digits
  at every `m` — a numerical confirmation of the machine-checked `SawtoothApprox.sq_sub_R_le`.

## Honest scope (and the falsifier)

- **Empirical, optimization-floor.** The SGD plateau is the best of a fixed budget (Adam, 3000
  epochs, modest nets, 5 restarts), not a lower bound. A heroically larger/better-tuned SGD run could
  lower the floor; the claim is the *qualitative* gap (exponential construction vs plateaued SGD,
  growing with depth), which is robust across depth, two architectures, and restarts — not a tight
  constant.
- **`NO_REPRESENTATION_GAP` did not fire.** It would have if SGD tracked the constructed error across
  depth (gap `~O(1)`); instead the gap grew ~3000×.
- This is **not** a learnability theorem and makes no P-vs-NP claim; it is the documented
  representation-vs-optimization gap, instantiated on the lane's own depth-efficient construction.

## Provenance

Slate-4 U-5. Reuses no external data; the target and the constructed approximant are the lane's own
(`x²`, the proved sawtooth). Cross-links: `Sundogcert/SawtoothShared.lean`,
`Sundogcert/SawtoothDag.lean`, [`ALGO_APPROX_CONJECTURE_SLATE_4.md`](ALGO_APPROX_CONJECTURE_SLATE_4.md),
and the empirical siblings [`ALGO_APPROX_N4_EPS_REGIONS_RESULT.md`](ALGO_APPROX_N4_EPS_REGIONS_RESULT.md)
/ [`ALGO_APPROX_S35_SAMPLE_COMPLEXITY_RESULT.md`](ALGO_APPROX_S35_SAMPLE_COMPLEXITY_RESULT.md).
