# PDE C1 Kolmogorov kNN Convergence-Check Receipt

**Status:** STRICTNESS_WITNESS_POSITIVE
**Preset:** `lock_v7_g300`
**Adjudicator:** `knn-sweep`
**Interpretable verdict:** `True`

## Sweep (vs neighbourhood radius; primary statistic = mean_minority)

| k | r_k median | fidelity coverage | mean_minority | incompat fraction |
|---:|---:|---:|---:|---:|
| 10 | 0.019769 | 1 | 0.011988 | 0.03652 |
| 15 | 0.0253967 | 1 | 0.014876 | 0.04816 |
| 20 | 0.0306851 | 1 | 0.016937 | 0.04936 |
| 25 | 0.0326464 | 1 | 0.0190904 | 0.06076 |
| 30 | 0.0355433 | 1 | 0.0213187 | 0.0667 |
| 40 | 0.042201 | 1 | 0.0243645 | 0.076 |
| 50 | 0.0469411 | 1 | 0.0270216 | 0.08498 |

## Readout

- PRIMARY fit `mean_minority = a_mm + b * r_k_median` over `7` coverage-passing points: intercept `a_mm = 0.000577142`, slope `0.564177`
- secondary (diagnostic) `incompat_fraction` fit intercept: `0.000704953` (grain-confounded; not gated)
- damp fraction (global): `0.26878`
- classification (pre-registered §6): `a_mm <= 0.005` => POSITIVE; `a_mm >= 0.015` => NEG-A; else INCONCLUSIVE
- elapsed seconds: `2610.636`

## Branch

`mean_minority` extrapolates to ~zero as `r_k -> 0` (`a_mm <= 0.005`): the observed mixing is a finite-radius boundary-straddling artifact around a clean decision surface. The proxy is control-sufficient on fibers at this cell (Reading-2 regime 2); the provisional v4 `PDE-C1-NEG-A` is **overturned**.

## Files

- `manifest.json`
- `knn-sweep.csv`
