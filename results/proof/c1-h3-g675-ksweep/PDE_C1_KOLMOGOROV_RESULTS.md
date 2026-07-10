# PDE C1 Twin-State Adaptive (Coverage-Adaptive Apparatus) Receipt

**Status (adaptive):** SMOKE_ONLY  ()
**Frozen comparator:** SMOKE_ONLY  (disagree `0.02524653601298215`, coverage `0.40368`)
**Preset:** `recon_sweep_g675_kf3`  **Adjudicator:** `twin-state-adaptive`
**Interpretable:** `False`

## Readout (Approach A; epsilon_K unchanged)

- `epsilon_K`: `0.0663413`  `delta_H`: `0.00270657`  `k_min`: `10`
- **covered fraction f**: `0.008` (floor `0.1`)
- dense-count p05/p50/p95: `0.0` / `0.0` / `6.0`
- admitted witness pairs (unique): `1603`  witness sample fraction: `0.00784`
- **paired fiber-constancy**: `PAIRED_FIBER_UNDEFINED`  (disagree `0.0`, threshold `0.1`)

## Regression cross-check (compact cells must reduce to frozen)

- frozen verdict `SMOKE_ONLY` vs adaptive `SMOKE_ONLY`; f should be 1.000 on compact cells (here `0.008`).

Non-promotional; apparatus-generality only. Spec: `docs/chatv2/NSE_COVERAGE_ADAPTIVE_APPARATUS_SPEC.md`.
