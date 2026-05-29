# PDE C1 Kolmogorov kNN Convergence-Check Receipt

**Status:** PDE-C1-NEG-A
**Preset:** `lock_v4`
**Adjudicator:** `knn-sweep`
**Interpretable verdict:** `True`

## Sweep (incompat_fraction vs neighbourhood radius)

| k | r_k median | fidelity coverage | incompat fraction |
|---:|---:|---:|---:|
| 10 | 0.0196454 | 1 | 0.03492 |
| 20 | 0.0287785 | 1 | 0.06452 |
| 30 | 0.0345984 | 1 | 0.07164 |
| 50 | 0.0448359 | 1 | 0.09738 |
| 100 | 0.0637917 | 0.4465 | 0.0582755 |

## Readout

- OLS fit `incompat_fraction = a + b * r_k_median`: intercept `a = 0.0458863`, slope `b = 0.507716`
- min incompat fraction over sweep: `0.03492` vs `delta_incompat = 0.01`
- damp fraction (global): `0.30014`
- classification thresholds (pre-registered): `a > 0.02` & `min_incompat > delta_incompat` => NEG-A; `a < 0.01` => POSITIVE; else INCONCLUSIVE
- elapsed seconds: `1833.247`

## Branch

`incompat_fraction` extrapolates to a positive value as `r_k -> 0` (intercept above threshold, and stays above `delta_incompat` across the sweep): the fiber-incompatibility survives the zero-radius limit and is **genuine**, not a finite-radius boundary artifact. The provisional v4 `PDE-C1-NEG-A` is confirmed.

## Files

- `manifest.json`
- `knn-sweep.csv`
