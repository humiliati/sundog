# PDE C1 Kolmogorov kNN Convergence-Check Receipt

**Status:** SMOKE_ONLY
**Preset:** `smoke`
**Adjudicator:** `knn-sweep`
**Interpretable verdict:** `False`

## Sweep (vs neighbourhood radius; primary statistic = mean_minority)

| k | r_k median | fidelity coverage | mean_minority | incompat fraction |
|---:|---:|---:|---:|---:|
| 10 | 1.95655e-06 | 1 | 0 | 0 |
| 15 | 2.88767e-06 | 1 | 0 | 0 |
| 20 | 3.92644e-06 | 1 | 0 | 0 |
| 25 | 5.08918e-06 | 1 | 0 | 0 |
| 30 | 6.09185e-06 | 1 | 0 | 0 |
| 40 | 8.28032e-06 | 1 | 0 | 0 |
| 50 | 1.05873e-05 | 1 | 0 | 0 |

## Readout

- PRIMARY fit `mean_minority = a_mm + b * r_k_median` over `7` coverage-passing points: intercept `a_mm = 0`, slope `0`
- secondary (diagnostic) `incompat_fraction` fit intercept: `0` (grain-confounded; not gated)
- damp fraction (global): `0`
- classification (pre-registered §6): `a_mm <= 0.005` => POSITIVE; `a_mm >= 0.015` => NEG-A; else INCONCLUSIVE
- elapsed seconds: `6.003`

## Branch

Smoke-only run; no C1 verdict may be filed from this receipt.

## Files

- `manifest.json`
- `knn-sweep.csv`
