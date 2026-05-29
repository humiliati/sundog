# PDE C1 Kolmogorov kNN Convergence-Check Receipt

**Status:** STRICTNESS_WITNESS_POSITIVE
**Preset:** `lock_v7_g200`
**Adjudicator:** `knn-sweep`
**Interpretable verdict:** `True`

## Sweep (vs neighbourhood radius; primary statistic = mean_minority)

| k | r_k median | fidelity coverage | mean_minority | incompat fraction |
|---:|---:|---:|---:|---:|
| 10 | 0.018229 | 1 | 0.012264 | 0.03486 |
| 15 | 0.0226535 | 1 | 0.0163893 | 0.05664 |
| 20 | 0.0271156 | 1 | 0.019529 | 0.0622 |
| 25 | 0.0306346 | 1 | 0.0212952 | 0.0684 |
| 30 | 0.0325761 | 1 | 0.0232107 | 0.06868 |
| 40 | 0.0372058 | 1 | 0.0262915 | 0.0834 |
| 50 | 0.0423703 | 1 | 0.0306324 | 0.09616 |

## Readout

- PRIMARY fit `mean_minority = a_mm + b * r_k_median` over `7` coverage-passing points: intercept `a_mm = -0.000792245`, slope `0.736096`
- secondary (diagnostic) `incompat_fraction` fit intercept: `-0.00227918` (grain-confounded; not gated)
- damp fraction (global): `0.3003`
- classification (pre-registered §6): `a_mm <= 0.005` => POSITIVE; `a_mm >= 0.015` => NEG-A; else INCONCLUSIVE
- elapsed seconds: `2418.494`

## Branch

`mean_minority` extrapolates to ~zero as `r_k -> 0` (`a_mm <= 0.005`): the observed mixing is a finite-radius boundary-straddling artifact around a clean decision surface. The proxy is control-sufficient on fibers at this cell (Reading-2 regime 2); the provisional v4 `PDE-C1-NEG-A` is **overturned**.

## Files

- `manifest.json`
- `knn-sweep.csv`
