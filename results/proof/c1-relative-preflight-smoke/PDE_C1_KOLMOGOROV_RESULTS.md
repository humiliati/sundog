# PDE C1 Twin-State Relative (Regime-Conditioned eps_K) Receipt

**Status (relative):** SMOKE_ONLY  ()
**Scale-consistency guard:** None (underpowered)  (tercile disagree [], spread nan, tol 0.03)
**Frozen comparator:** SMOKE_ONLY  (disagree `0.0`)
**Preset:** `smoke`  **Adjudicator:** `twin-state-relative`
**Interpretable:** `False`

## Readout (Approach B; eps_K(u) = 0.05*sqrt(2*E_low(u)))

- relative eps_K(u) p05/p50/p95: `0.022097` / `0.022097` / `0.022097` (frozen ref `0.022101`)
- **covered fraction f_rel**: `1` (floor `0.1`)  query-clip: `1`
- witness pairs (unique): `0`  disagree: `nan`
- **paired fiber-constancy**: `PAIRED_FIBER_UNDEFINED`  tercile n: `[]`

A relative positive requires CERTIFIED + paired POSITIVE + scale-consistent; it is scoped to relative resolution and does not overturn the Approach-A absolute-resolution null. Spec: `docs/chatv2/NSE_REGIME_CONDITIONED_EPS_SCOPE.md`.
