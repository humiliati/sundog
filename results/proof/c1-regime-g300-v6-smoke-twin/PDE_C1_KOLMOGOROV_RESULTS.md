# PDE C1 Twin-State Certificate Receipt

**Status:** SMOKE_ONLY
**Preset:** `lock_v6`
**Adjudicator:** `twin-state`
**Interpretable certificate:** `False`

## Readout

- samples: `120`, k (effective): `50`
- `epsilon_K` (signature radius): `0.152505`
- `delta_H`: `0.0249415` (`0.05` x median high-mode norm, floor `1e-06`)
- high-mode norm median / min / max: `0.49883` / `0.211151` / `1.20479`
- signature-near sample coverage: `1` vs `S_pos = 0.5` (`120` of `120`)
- candidate pairs unique / directed: `224` / `448`
- witness sample fraction: `1` vs `0.01` (`120` samples)
- witness pairs unique / directed: `224` / `448` vs min unique `100`
- witness high-distance p50 / p95: `0.0680381` / `0.12449`
- elapsed seconds: `2.525`

## Branch

Smoke-only run; no support-level state-insufficiency certificate may be filed.

## Files

- `manifest.json`
- `twin-state-witnesses.csv`
