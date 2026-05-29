# PDE C2 Sabra Cell v1 Receipt (objective-validity layer)

**Gate:** SMOKE_ONLY
**Preset:** `smoke`   **Model:** `sabra`   **Forcing:** `fixed-amplitude`

## Base-rate + per-block stationarity gates

- burst observable: ε(t) = ν Σ k_n²|u_n|²; label = held-out look-ahead-max quantile at target base rate `0.15`
- `E_burst`: `2.07204e-09`
- base rate train / val / test: `0` / `0` / `0`
- band `[0.05, 0.4]`: `DEFER`
- per-block pairwise max diff `0` vs tol `0.1`: `PASS`
- test query count: `230`
- elapsed: `32.9` s over `86000` steps

## Branch

Smoke / override run; no C2 result filed.

## Files

- `manifest.json`
