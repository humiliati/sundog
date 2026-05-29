# PDE C1 Twin-State Certificate Receipt

**Status:** SMOKE_ONLY
**Preset:** `smoke`
**Adjudicator:** `twin-state`
**Interpretable certificate:** `False`

## Readout

- samples: `200`, k (effective): `50`
- `epsilon_K` (signature radius): `0.0221007`
- `delta_H`: `9.39057e-05` (`0.05` x median high-mode norm, floor `1e-06`)
- high-mode norm median / min / max: `0.00187811` / `0.00185502` / `0.00190162`
- signature-near sample coverage: `1` vs `S_pos = 0.5` (`200` of `200`)
- candidate pairs unique / directed: `5486` / `9800`
- witness sample fraction: `0` vs `0.01` (`0` samples)
- witness pairs unique / directed: `0` / `0` vs min unique `100`
- witness high-distance p50 / p95: `0` / `0`
- elapsed seconds: `11.430`

## Paired fiber-constancy

**Paired verdict:** `SMOKE_ONLY`

- witness-pair action disagreement (unique): `0` (`0` of `0`) vs `delta_action = 0.1`
- witness-pair action disagreement (directed): `0`
- candidate-pair action disagreement (unique, comparator): `0` (`0` of `5486`)

## Branch

Smoke-only run; no support-level state-insufficiency certificate may be filed.

## Files

- `manifest.json`
- `twin-state-witnesses.csv`
