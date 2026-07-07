# PDE C1 Kolmogorov kNN Convergence-Check Receipt

**Status:** STRICTNESS_WITNESS_POSITIVE
**Preset:** `lock_v7_g200_n48`
**Adjudicator:** `knn-sweep`
**Interpretable verdict:** `True`

## Sweep (vs neighbourhood radius; primary statistic = mean_minority)

| k | r_k median | fidelity coverage | mean_minority | incompat fraction |
|---:|---:|---:|---:|---:|
| 10 | 0.0183416 | 1 | 0.012474 | 0.0343 |
| 15 | 0.0222756 | 1 | 0.016416 | 0.05474 |
| 20 | 0.0272821 | 1 | 0.020383 | 0.0615 |
| 25 | 0.0308704 | 1 | 0.021928 | 0.0686 |
| 30 | 0.0326266 | 1 | 0.0227673 | 0.0675 |
| 40 | 0.0370384 | 1 | 0.02603 | 0.0821 |
| 50 | 0.0427758 | 1 | 0.0306012 | 0.09588 |

## Readout

- PRIMARY fit `mean_minority = a_mm + b * r_k_median` over `7` coverage-passing points: intercept `a_mm = 0.000168327`, slope `0.707451`
- secondary (diagnostic) `incompat_fraction` fit intercept: `-0.00275793` (grain-confounded; not gated)
- damp fraction (global): `0.30004`
- classification (pre-registered §6): `a_mm <= 0.005` => POSITIVE; `a_mm >= 0.015` => NEG-A; else INCONCLUSIVE
- elapsed seconds: `3241.631`

## Branch

`mean_minority` extrapolates to ~zero as `r_k -> 0` (`a_mm <= 0.005`): the observed mixing is a finite-radius boundary-straddling artifact around a clean decision surface. The proxy is control-sufficient on fibers at this cell (Reading-2 regime 2); the provisional v4 `PDE-C1-NEG-A` is **overturned**.

## Files

- `manifest.json`
- `knn-sweep.csv`
