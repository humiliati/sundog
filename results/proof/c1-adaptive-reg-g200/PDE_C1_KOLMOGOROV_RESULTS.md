# PDE C1 Twin-State Adaptive (Coverage-Adaptive Apparatus) Receipt

**Status (adaptive):** TWIN_STATE_ADAPTIVE_CERTIFIED  (dense_fiber_high_mode_separated_twins)
**Frozen comparator:** TWIN_STATE_CERTIFIED  (disagree `0.036777106089302854`, coverage `1.0`)
**Preset:** `lock_v7_g200`  **Adjudicator:** `twin-state-adaptive`
**Interpretable:** `True`

## Readout (Approach A; epsilon_K unchanged)

- `epsilon_K`: `0.0605968`  `delta_H`: `0.0116667`  `k_min`: `10`
- **covered fraction f**: `1` (floor `0.1`)
- dense-count p05/p50/p95: `49.0` / `49.0` / `49.0`
- admitted witness pairs (unique): `693774`  witness sample fraction: `1`
- **paired fiber-constancy**: `PAIRED_FIBER_CONSTANCY_POSITIVE`  (disagree `0.036777106089302854`, threshold `0.1`)

## Regression cross-check (compact cells must reduce to frozen)

- frozen verdict `TWIN_STATE_CERTIFIED` vs adaptive `TWIN_STATE_ADAPTIVE_CERTIFIED`; f should be 1.000 on compact cells (here `1`).

Non-promotional; apparatus-generality only. Spec: `docs/chatv2/NSE_COVERAGE_ADAPTIVE_APPARATUS_SPEC.md`.
