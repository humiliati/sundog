# PDE C1 Kolmogorov kNN Convergence-Check Receipt

**Status:** STRICTNESS_WITNESS_POSITIVE
**Preset:** `lock_v4`
**Adjudicator:** `knn-sweep`
**Interpretable verdict:** `True`

## Sweep (vs neighbourhood radius; primary statistic = mean_minority)

| k | r_k median | fidelity coverage | mean_minority | incompat fraction |
|---:|---:|---:|---:|---:|
| 10 | 0.0196454 | 1 | 0.012276 | 0.03492 |
| 15 | 0.0236601 | 1 | 0.0165787 | 0.05758 |
| 20 | 0.0287785 | 1 | 0.020576 | 0.06452 |
| 25 | 0.0321678 | 1 | 0.0215264 | 0.06936 |
| 30 | 0.0345984 | 1 | 0.0240753 | 0.07164 |
| 40 | 0.0393021 | 1 | 0.027631 | 0.0872 |
| 50 | 0.0448359 | 1 | 0.0311176 | 0.09738 |

## Readout

- PRIMARY fit `mean_minority = a_mm + b * r_k_median` over `7` coverage-passing points: intercept `a_mm = -0.00125459`, slope `0.729022`
- secondary (diagnostic) `incompat_fraction` fit intercept: `-0.00314121` (grain-confounded; not gated)
- damp fraction (global): `0.30014`
- classification (pre-registered §6): `a_mm <= 0.005` => POSITIVE; `a_mm >= 0.015` => NEG-A; else INCONCLUSIVE
- elapsed seconds: `1714.299`

## Branch

`mean_minority` extrapolates to ~zero as `r_k -> 0` (`a_mm <= 0.005`): the observed mixing is a finite-radius boundary-straddling artifact around a clean decision surface. The proxy is control-sufficient on fibers at this cell (Reading-2 regime 2); the provisional v4 `PDE-C1-NEG-A` is **overturned**.

## Files

- `manifest.json`
- `knn-sweep.csv`
