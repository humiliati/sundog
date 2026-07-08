# PDE C1 Twin-State Adaptive (Coverage-Adaptive Apparatus) Receipt

**Status (adaptive):** SMOKE_ONLY  ()
**Frozen comparator:** SMOKE_ONLY  (disagree `0.0`, coverage `1.0`)
**Preset:** `smoke`  **Adjudicator:** `twin-state-adaptive`
**Interpretable:** `False`

## Readout (Approach A; epsilon_K unchanged)

- `epsilon_K`: `0.0221007`  `delta_H`: `9.39057e-05`  `k_min`: `10`
- **covered fraction f**: `1` (floor `0.1`)
- dense-count p05/p50/p95: `49.0` / `49.0` / `49.0`
- admitted witness pairs (unique): `0`  witness sample fraction: `0`
- **paired fiber-constancy**: `PAIRED_FIBER_UNDEFINED`  (disagree `nan`, threshold `0.1`)

## Regression cross-check (compact cells must reduce to frozen)

- frozen verdict `SMOKE_ONLY` vs adaptive `SMOKE_ONLY`; f should be 1.000 on compact cells (here `1`).

Non-promotional; apparatus-generality only. Spec: `docs/chatv2/NSE_COVERAGE_ADAPTIVE_APPARATUS_SPEC.md`.
