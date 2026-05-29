# PDE C1 Kolmogorov kNN Convergence-Check Receipt

**Status:** DEFERRED_VACUITY
**Preset:** `lock_v6`
**Adjudicator:** `knn-sweep`
**Interpretable verdict:** `False`

## Sweep (vs neighbourhood radius; primary statistic = mean_minority)

| k | r_k median | fidelity coverage | mean_minority | incompat fraction |
|---:|---:|---:|---:|---:|
| 10 | 0.0219895 | 0.90094 | 0 | 0 |
| 15 | 0.0290716 | 0.90048 | 0 | 0 |
| 20 | 0.0319099 | 0.90048 | 0 | 0 |
| 25 | 0.03521 | 0.90048 | 0 | 0 |
| 30 | 0.0396876 | 0.90048 | 0 | 0 |
| 40 | 0.0457603 | 0.9004 | 0 | 0 |
| 50 | 0.0503852 | 0.90024 | 0 | 0 |

## Readout

- PRIMARY fit `mean_minority = a_mm + b * r_k_median` over `7` coverage-passing points: intercept `a_mm = 0`, slope `0`
- secondary (diagnostic) `incompat_fraction` fit intercept: `0` (grain-confounded; not gated)
- damp fraction (global): `0.00446`
- classification (pre-registered §6): `a_mm <= 0.005` => POSITIVE; `a_mm >= 0.015` => NEG-A; else INCONCLUSIVE
- elapsed seconds: `1290.763`

## Branch

Proxy selector essentially constant; no verdict filed.

## Files

- `manifest.json`
- `knn-sweep.csv`
