# PDE C1 Kolmogorov kNN Convergence-Check Receipt

**Status:** SMOKE_ONLY
**Preset:** `lock_v7_g200`
**Adjudicator:** `knn-sweep`
**Interpretable verdict:** `False`

## Sweep (vs neighbourhood radius; primary statistic = mean_minority)

| k | r_k median | fidelity coverage | mean_minority | incompat fraction |
|---:|---:|---:|---:|---:|
| 10 | 0.00859782 | 1 | 0 | 0 |
| 15 | 0.0124273 | 1 | 0 | 0 |
| 20 | 0.0171374 | 1 | 0 | 0 |
| 25 | 0.0216575 | 1 | 0 | 0 |
| 30 | 0.0256024 | 1 | 0 | 0 |
| 40 | 0.0351617 | 0.99 | 0 | 0 |
| 50 | 0.0444125 | 0.94 | 0 | 0 |

## Readout

- PRIMARY fit `mean_minority = a_mm + b * r_k_median` over `7` coverage-passing points: intercept `a_mm = 0`, slope `0`
- secondary (diagnostic) `incompat_fraction` fit intercept: `0` (grain-confounded; not gated)
- damp fraction (global): `0`
- classification (pre-registered §6): `a_mm <= 0.005` => POSITIVE; `a_mm >= 0.015` => NEG-A; else INCONCLUSIVE
- elapsed seconds: `67.573`

## Branch

Smoke-only run; no C1 verdict may be filed from this receipt.

## Files

- `manifest.json`
- `knn-sweep.csv`
