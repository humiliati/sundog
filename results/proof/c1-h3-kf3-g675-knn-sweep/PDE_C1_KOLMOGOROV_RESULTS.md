# PDE C1 Kolmogorov kNN Convergence-Check Receipt

**Status:** INCONCLUSIVE_CONVERGENCE
**Preset:** `lock_v7_g675_kf3`
**Adjudicator:** `knn-sweep`
**Interpretable verdict:** `False`

## Sweep (vs neighbourhood radius; primary statistic = mean_minority)

| k | r_k median | fidelity coverage | mean_minority | incompat fraction |
|---:|---:|---:|---:|---:|
| 10 | 0.194304 | 0.02728 | 0.014956 | 0.0417889 |
| 15 | 0.216436 | 0.00442 | 0.00784314 | 0.0316742 |
| 20 | 0.229599 | 0.00122 | 0 | 0 |
| 25 | 0.239594 | 0.00094 | 0 | 0 |
| 30 | 0.247879 | 6e-05 | 0 | 0 |
| 40 | 0.261215 | 0 | 0 | 0 |
| 50 | 0.271927 | 0 | 0 | 0 |

## Readout

- PRIMARY fit `mean_minority = a_mm + b * r_k_median` over `0` coverage-passing points: intercept `a_mm = 0`, slope `0`
- secondary (diagnostic) `incompat_fraction` fit intercept: `0` (grain-confounded; not gated)
- damp fraction (global): `0.30856`
- classification (pre-registered §6): `a_mm <= 0.005` => POSITIVE; `a_mm >= 0.015` => NEG-A; else INCONCLUSIVE
- elapsed seconds: `2659.030`

## Branch

`a_mm` in the ambiguous band `(0.005, 0.015)`, or too few coverage-passing sweep points to fit. `INCONCLUSIVE_CONVERGENCE` — a larger `N` or wider clean-`k` range is needed. No verdict filed.

## Files

- `manifest.json`
- `knn-sweep.csv`
