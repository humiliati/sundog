# PDE C1 Kolmogorov kNN Convergence-Check Receipt

**Status:** SMOKE_ONLY
**Preset:** `lock_v6`
**Adjudicator:** `knn-sweep`
**Interpretable verdict:** `False`

## Sweep (vs neighbourhood radius; primary statistic = mean_minority)

| k | r_k median | fidelity coverage | mean_minority | incompat fraction |
|---:|---:|---:|---:|---:|
| 10 | 0.241488 | 0 | 0 | 0 |
| 15 | 0.286754 | 0 | 0 | 0 |
| 20 | 0.34509 | 0 | 0 | 0 |
| 25 | 0.404439 | 0 | 0 | 0 |
| 30 | 0.46409 | 0 | 0 | 0 |
| 40 | 0.568387 | 0 | 0 | 0 |
| 50 | 0.657579 | 0 | 0 | 0 |

## Readout

- PRIMARY fit `mean_minority = a_mm + b * r_k_median` over `0` coverage-passing points: intercept `a_mm = 0`, slope `0`
- secondary (diagnostic) `incompat_fraction` fit intercept: `0` (grain-confounded; not gated)
- damp fraction (global): `0`
- classification (pre-registered §6): `a_mm <= 0.005` => POSITIVE; `a_mm >= 0.015` => NEG-A; else INCONCLUSIVE
- elapsed seconds: `2.436`

## Branch

Smoke-only run; no C1 verdict may be filed from this receipt.

## Files

- `manifest.json`
- `knn-sweep.csv`
