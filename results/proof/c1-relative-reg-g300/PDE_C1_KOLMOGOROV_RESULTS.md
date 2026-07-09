# PDE C1 Twin-State Relative (Regime-Conditioned eps_K) Receipt

**Status (relative):** TWIN_STATE_RELATIVE_CERTIFIED  (relative_fiber_high_mode_separated_twins)
**Scale-consistency guard:** INCONSISTENT (inflation)  (tercile disagree [0.056780379277465834, 0.0046438535155560266, 0.051370383057467446], spread 0.05213652576190981, tol 0.03)
**Frozen comparator:** TWIN_STATE_CERTIFIED  (disagree `0.038185937291188056`)
**Preset:** `lock_v7_g300`  **Adjudicator:** `twin-state-relative`
**Interpretable:** `True`

## Readout (Approach B; eps_K(u) = 0.05*sqrt(2*E_low(u)))

- relative eps_K(u) p05/p50/p95: `0.06199` / `0.064586` / `0.066354` (frozen ref `0.066422`)
- **covered fraction f_rel**: `1` (floor `0.1`)  query-clip: `0.9999`
- witness pairs (unique): `942830`  disagree: `0.03818609929679794`
- **paired fiber-constancy**: `PAIRED_FIBER_CONSTANCY_POSITIVE`  tercile n: `[566340, 566340, 566338]`

A relative positive requires CERTIFIED + paired POSITIVE + scale-consistent; it is scoped to relative resolution and does not overturn the Approach-A absolute-resolution null. Spec: `docs/chatv2/NSE_REGIME_CONDITIONED_EPS_SCOPE.md`.
