# PDE C1 Kolmogorov kNN Convergence-Check Receipt

**Status:** STRICTNESS_WITNESS_POSITIVE
**Preset:** `lock_v7_g300_n48_dt5`
**Adjudicator:** `knn-sweep`
**Interpretable verdict:** `True`

## Sweep (vs neighbourhood radius; primary statistic = mean_minority)

| k | r_k median | fidelity coverage | mean_minority | incompat fraction |
|---:|---:|---:|---:|---:|
| 10 | 0.0194527 | 1 | 0.011924 | 0.03342 |
| 15 | 0.0273 | 1 | 0.0141627 | 0.04522 |
| 20 | 0.0301781 | 1 | 0.016032 | 0.0469 |
| 25 | 0.0325084 | 1 | 0.019016 | 0.06228 |
| 30 | 0.0356736 | 1 | 0.021202 | 0.0665 |
| 40 | 0.0426115 | 1 | 0.023569 | 0.07242 |
| 50 | 0.0464793 | 1 | 0.02679 | 0.08454 |

## Readout

- PRIMARY fit `mean_minority = a_mm + b * r_k_median` over `7` coverage-passing points: intercept `a_mm = -7.96864e-05`, slope `0.568964`
- secondary (diagnostic) `incompat_fraction` fit intercept: `-0.00442651` (grain-confounded; not gated)
- damp fraction (global): `0.28548`
- classification (pre-registered §6): `a_mm <= 0.005` => POSITIVE; `a_mm >= 0.015` => NEG-A; else INCONCLUSIVE
- elapsed seconds: `7717.958`

## Branch

`mean_minority` extrapolates to ~zero as `r_k -> 0` (`a_mm <= 0.005`): the observed mixing is a finite-radius boundary-straddling artifact around a clean decision surface. The proxy is control-sufficient on fibers at this cell (Reading-2 regime 2); the provisional v4 `PDE-C1-NEG-A` is **overturned**.

## Files

- `manifest.json`
- `knn-sweep.csv`
