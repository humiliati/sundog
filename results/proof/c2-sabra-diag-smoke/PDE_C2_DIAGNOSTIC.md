# PDE C2 Sabra v1 Stationarity Diagnostic

**Forcing:** `fixed-amplitude`   **Model:** `sabra`

## Stationarity

- total energy first-half / second-half mean: `0.998011` / `1.00001` (drift `0.002`)
- total energy min / max: `1.71586e-05` / `1.00001`
- plateaued: `True`   T_eq: `0` time units

## Burst timescale

- max-energy median / post-eq q98: `1` / `1`
- post-eq burst count: `3`   T_burst: `4.31` time units

## Suggested v1 cell lengths (pre-registered rule)

- warmup >= 3*T_eq -> `0` steps
- each block >= 50*T_burst -> `2155000` steps

## Read

Total energy plateaus (first/second-half drift < 10%): the cascade is statistically steady under this forcing. Pin the v1 cell lengths from the suggestions above and proceed to the verdict-bearing run.

## Files

- `manifest.json`
