# PDE C1 Twin-State Certificate Receipt

**Status:** SMOKE_ONLY
**Preset:** `smoke`
**Adjudicator:** `twin-state`
**Interpretable certificate:** `False`

## Readout

- samples: `80`, k (effective): `50`
- `epsilon_K` (signature radius): `0.0221125`
- `delta_H`: `9.50182e-05` (`0.05` x median high-mode norm, floor `1e-06`)
- high-mode norm median / min / max: `0.00190036` / `0.00185041` / `0.00196882`
- signature-near sample coverage: `1` vs `S_pos = 0.5` (`80` of `80`)
- candidate pairs unique / directed: `2296` / `3920`
- witness sample fraction: `1` vs `0.01` (`80` samples)
- witness pairs unique / directed: `1351` / `2030` vs min unique `100`
- witness high-distance p50 / p95: `0.00015501` / `0.000287028`
- elapsed seconds: `2.410`

## Branch

Smoke-only run; no support-level state-insufficiency certificate may be filed.

## Files

- `manifest.json`
- `twin-state-witnesses.csv`
