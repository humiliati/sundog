# PDE C1 Kolmogorov kNN Convergence-Check Receipt

**Status:** STRICTNESS_WITNESS_POSITIVE
**Preset:** `lock_v5`
**Adjudicator:** `knn-sweep`
**Interpretable verdict:** `True`

## Sweep (vs neighbourhood radius; primary statistic = mean_minority)

| k | r_k median | fidelity coverage | mean_minority | incompat fraction |
|---:|---:|---:|---:|---:|
| 10 | 0.0182286 | 1 | 0.012318 | 0.03476 |
| 15 | 0.0226449 | 1 | 0.016404 | 0.05698 |
| 20 | 0.0271133 | 1 | 0.01958 | 0.06234 |
| 25 | 0.030638 | 1 | 0.0213088 | 0.06844 |
| 30 | 0.0325759 | 1 | 0.0232327 | 0.06872 |
| 40 | 0.0372099 | 1 | 0.026356 | 0.08404 |
| 50 | 0.042367 | 1 | 0.0306996 | 0.09594 |

## Readout

- PRIMARY fit `mean_minority = a_mm + b * r_k_median` over `7` coverage-passing points: intercept `a_mm = -0.00077831`, slope `0.73702`
- secondary (diagnostic) `incompat_fraction` fit intercept: `-0.00214977` (grain-confounded; not gated)
- damp fraction (global): `0.2977`
- classification (pre-registered §6): `a_mm <= 0.005` => POSITIVE; `a_mm >= 0.015` => NEG-A; else INCONCLUSIVE
- elapsed seconds: `1607.427`

## Branch

`mean_minority` extrapolates to ~zero as `r_k -> 0` (`a_mm <= 0.005`): the observed mixing is a finite-radius boundary-straddling artifact around a clean decision surface. The proxy is control-sufficient on fibers at this cell (Reading-2 regime 2); the provisional v4 `PDE-C1-NEG-A` is **overturned**.

## Files

- `manifest.json`
- `knn-sweep.csv`
