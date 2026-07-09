# PDE C1 Twin-State Adaptive (Coverage-Adaptive Apparatus) Receipt

**Status (adaptive):** TWIN_STATE_ADAPTIVE_SLIVER  (covered_fraction_below_floor)
**Frozen comparator:** TWIN_STATE_DEFERRED_COVERAGE  (disagree `0.03394683482252278`, coverage `0.469195`)
**Preset:** `fallback_v7_g675_kf3`  **Adjudicator:** `twin-state-adaptive`
**Interpretable:** `False`

## Readout (Approach A; epsilon_K unchanged)

- `epsilon_K`: `0.0589334`  `delta_H`: `0.0174816`  `k_min`: `10`
- **covered fraction f**: `0.03644` (floor `0.1`)
- dense-count p05/p50/p95: `0.0` / `0.0` / `8.0`
- admitted witness pairs (unique): `35739`  witness sample fraction: `0.035915`
- **paired fiber-constancy**: `PAIRED_FIBER_UNDEFINED`  (disagree `0.029547553093259463`, threshold `0.1`)

## Regression cross-check (compact cells must reduce to frozen)

- frozen verdict `TWIN_STATE_DEFERRED_COVERAGE` vs adaptive `TWIN_STATE_ADAPTIVE_SLIVER`; f should be 1.000 on compact cells (here `0.03644`).

Non-promotional; apparatus-generality only. Spec: `docs/chatv2/NSE_COVERAGE_ADAPTIVE_APPARATUS_SPEC.md`.
