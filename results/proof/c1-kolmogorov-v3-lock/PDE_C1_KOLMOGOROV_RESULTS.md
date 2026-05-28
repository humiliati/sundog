# PDE C1 Kolmogorov v0 Receipt

**Status:** DEFERRED_VACUITY
**Preset:** `lock_v3`
**Interpretable verdict:** `False`

## Readout

- `S_eval`: `0` vs `S_pos = 0.5`
- occupied bins: `44065`
- evaluated bins: `0`
- incompatible bins: `0`
- damp fraction: `0`
- max evaluated minority fraction: `0`
- elapsed seconds: `1197.600`
- steps/sec: `2171.385`

## Branch

Proxy selector is essentially constant on the sampled support (damp_fraction outside `[delta_proxy_min, 1 - delta_proxy_min]`); the strictness predicate has no discriminative content. No verdict filed. No fall-back is admissible on this cell; re-pinning to a discriminative regime requires a new cell-set instance (e.g. v1).

## Files

- `manifest.json`
- `bin-summary.csv`
