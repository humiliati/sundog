# PDE C1 Twin-State Relative (Regime-Conditioned eps_K) Receipt

**Status (relative):** TWIN_STATE_RELATIVE_CERTIFIED  (relative_fiber_high_mode_separated_twins)
**Scale-consistency guard:** INCONSISTENT (inflation)  (tercile disagree [0.0453314068990428, 0.0, 0.06294607952960757], spread 0.06294607952960757, tol 0.03)
**Frozen comparator:** TWIN_STATE_CERTIFIED  (disagree `0.036777106089302854`)
**Preset:** `lock_v7_g200`  **Adjudicator:** `twin-state-relative`
**Interpretable:** `True`

## Readout (Approach B; eps_K(u) = 0.05*sqrt(2*E_low(u)))

- relative eps_K(u) p05/p50/p95: `0.059822` / `0.060238` / `0.060596` (frozen ref `0.060597`)
- **covered fraction f_rel**: `1` (floor `0.1`)  query-clip: `1`
- witness pairs (unique): `693774`  disagree: `0.036777106089302854`
- **paired fiber-constancy**: `PAIRED_FIBER_CONSTANCY_POSITIVE`  tercile n: `[437423, 437422, 437422]`

A relative positive requires CERTIFIED + paired POSITIVE + scale-consistent; it is scoped to relative resolution and does not overturn the Approach-A absolute-resolution null. Spec: `docs/chatv2/NSE_REGIME_CONDITIONED_EPS_SCOPE.md`.
