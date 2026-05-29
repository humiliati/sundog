# PDE C1 Kolmogorov kNN Convergence-Check Receipt

**Status:** SMOKE_ONLY
**Preset:** `smoke`
**Adjudicator:** `knn-sweep`
**Interpretable verdict:** `False`

## Sweep (incompat_fraction vs neighbourhood radius)

| k | r_k median | fidelity coverage | incompat fraction |
|---:|---:|---:|---:|
| 10 | 1.95655e-06 | 1 | 0 |
| 20 | 3.92644e-06 | 1 | 0 |
| 30 | 6.09185e-06 | 1 | 0 |
| 50 | 1.05873e-05 | 1 | 0 |
| 100 | 2.3256e-05 | 1 | 0 |

## Readout

- OLS fit `incompat_fraction = a + b * r_k_median`: intercept `a = 0`, slope `b = 0`
- min incompat fraction over sweep: `0` vs `delta_incompat = 0.01`
- damp fraction (global): `0`
- classification thresholds (pre-registered): `a > 0.02` & `min_incompat > delta_incompat` => NEG-A; `a < 0.01` => POSITIVE; else INCONCLUSIVE
- elapsed seconds: `5.781`

## Branch

Smoke-only run; no C1 verdict may be filed from this receipt.

## Files

- `manifest.json`
- `knn-sweep.csv`
