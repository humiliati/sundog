# PDE C1 Kolmogorov kNN Receipt

**Status:** PDE-C1-NEG-A
**Preset:** `lock_v4`
**Adjudicator:** `knn`
**Interpretable verdict:** `True`

## Readout

- samples: `50000`, k (effective): `30`
- `epsilon_K` (radius threshold): `0.0626036`
- fidelity coverage: `1` vs `S_pos = 0.5` (`50000` of `50000` within `epsilon_K`)
- `r_k` median / p95 / max: `0.0345984` / `0.0387776` / `0.0402362`
- `r_k` median among fidelity-passing: `0.0345984`
- damp fraction (global): `0.30014`
- incompatible fraction (of passing): `0.07164` vs `delta_incompat = 0.01` (`3582` samples)
- max passing minority fraction: `0.5` vs `delta_action = 0.1`
- elapsed seconds: `1549.335`

## Branch

A positive-mass fraction of fidelity-passing samples have local minority fraction above `delta_action`; file `PDE-C1-NEG-A` (fiber incompatibility under the kNN/disintegration adjudicator).

## Files

- `manifest.json`
- `knn-radius-histogram.csv`
