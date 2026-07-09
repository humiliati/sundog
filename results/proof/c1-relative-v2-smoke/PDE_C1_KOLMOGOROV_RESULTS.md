# PDE C1 Twin-State Relative v2 (Inflation-Shell Guard) Receipt

**Status (relative):** SMOKE_ONLY  ()
**v2 inflation-shell guard:** None (SHELL empty -> no inflation to test)  (core disagree nan on 0 pairs; shell disagree nan on 0 pairs; margin 0.05)
**v1 tercile diagnostic (retired):** spread nan disagree []
**Frozen comparator:** SMOKE_ONLY  (disagree `0.0`)
**Preset:** `smoke`  **Adjudicator:** `twin-state-relative`
**Interpretable:** `False`

## Readout (Approach B; eps_K(u) = 0.05*sqrt(2*E_low(u)))

- relative eps_K(u) p05/p50/p95: `0.022097` / `0.022097` / `0.022097` (frozen ref `0.022101`)
- **covered fraction f_rel**: `1` (floor `0.1`)  query-clip: `1`
- witness pairs (unique): `0`  disagree: `nan`
- **paired fiber-constancy**: `PAIRED_FIBER_UNDEFINED`

A relative positive requires CERTIFIED + paired POSITIVE + inflation_clean in {True, None}; it is scoped to relative resolution and does not overturn the Approach-A absolute-resolution null. Spec: `docs/chatv2/NSE_REGIME_CONDITIONED_EPS_V2_SPEC.md`.
