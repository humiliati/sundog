# PDE C1 Kolmogorov kNN Receipt

**Status:** SMOKE_ONLY
**Preset:** `smoke`
**Adjudicator:** `knn`
**Interpretable verdict:** `False`

## Readout

- samples: `200`, k (effective): `30`
- `epsilon_K` (radius threshold): `0.0221007`
- fidelity coverage: `1` vs `S_pos = 0.5` (`200` of `200` within `epsilon_K`)
- `r_k` median / p95 / max: `6.09185e-06` / `1.47679e-05` / `2.37171e-05`
- `r_k` median among fidelity-passing: `6.09185e-06`
- damp fraction (global): `0`
- incompatible fraction (of passing): `0` vs `delta_incompat = 0.01` (`0` samples)
- max passing minority fraction: `0` vs `delta_action = 0.1`
- elapsed seconds: `4.750`

## Branch

Smoke-only run; no C1 negative or positive may be filed from this receipt.

## Files

- `manifest.json`
- `knn-radius-histogram.csv`
