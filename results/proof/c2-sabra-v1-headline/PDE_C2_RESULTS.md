# PDE C2 Sabra Cell v1 Receipt (objective-validity layer)

**Gate:** PDE-C2-DEFERRED-BASERATE
**Preset:** `headline`   **Model:** `sabra`   **Forcing:** `fixed-amplitude`

## Base-rate + per-block stationarity gates

- burst observable: ε(t) = ν Σ k_n²|u_n|²; label = held-out look-ahead-max quantile at target base rate `0.15`
- `E_burst`: `4.27874e-09`
- base rate train / val / test: `0.1381` / `0` / `0`
- band `[0.05, 0.4]`: `DEFER`
- per-block pairwise max diff `0.1381` vs tol `0.1`: `DEFER`
- test query count: `19990`
- elapsed: `1787.1` s over `6300000` steps

## Branch

A block's burst base rate is outside the pre-registered band: the objective is degenerate at this cell. `PDE-C2-DEFERRED-BASERATE` (non-verdict). No threshold/rate rescue (would be `PDE-C2-NEG-B`).

## Files

- `manifest.json`
