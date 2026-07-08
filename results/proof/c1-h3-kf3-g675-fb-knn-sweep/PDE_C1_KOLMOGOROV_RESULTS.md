# PDE C1 Kolmogorov kNN Convergence-Check Receipt

**Status:** INCONCLUSIVE_CONVERGENCE
**Preset:** `fallback_v7_g675_kf3`
**Adjudicator:** `knn-sweep`
**Interpretable verdict:** `False`

## Sweep (vs neighbourhood radius; primary statistic = mean_minority)

| k | r_k median | fidelity coverage | mean_minority | incompat fraction |
|---:|---:|---:|---:|---:|
| 10 | 0.163164 | 0.047105 | 0.0205074 | 0.0557266 |
| 15 | 0.178402 | 0.013695 | 0.017695 | 0.0638919 |
| 20 | 0.188526 | 0.004575 | 0.013224 | 0.0469945 |
| 25 | 0.196411 | 0.001815 | 0.00121212 | 0.00275482 |
| 30 | 0.202826 | 0.00078 | 0 | 0 |
| 40 | 0.213009 | 6e-05 | 0 | 0 |
| 50 | 0.221006 | 0 | 0 | 0 |

## Readout

- PRIMARY fit `mean_minority = a_mm + b * r_k_median` over `0` coverage-passing points: intercept `a_mm = 0`, slope `0`
- secondary (diagnostic) `incompat_fraction` fit intercept: `0` (grain-confounded; not gated)
- damp fraction (global): `0.30674`
- classification (pre-registered §6): `a_mm <= 0.005` => POSITIVE; `a_mm >= 0.015` => NEG-A; else INCONCLUSIVE
- elapsed seconds: `7015.134`

## Branch

`a_mm` in the ambiguous band `(0.005, 0.015)`, or too few coverage-passing sweep points to fit. `INCONCLUSIVE_CONVERGENCE` — a larger `N` or wider clean-`k` range is needed. No verdict filed.

## Files

- `manifest.json`
- `knn-sweep.csv`
