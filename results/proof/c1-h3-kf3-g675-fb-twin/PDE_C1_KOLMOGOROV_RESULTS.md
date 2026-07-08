# PDE C1 Twin-State Certificate Receipt

**Status:** TWIN_STATE_DEFERRED_COVERAGE
**Preset:** `fallback_v7_g675_kf3`
**Adjudicator:** `twin-state`
**Interpretable certificate:** `False`

## Readout

- samples: `200000`, k (effective): `50`
- `epsilon_K` (signature radius): `0.0589334`
- `delta_H`: `0.0174816` (`0.05` x median high-mode norm, floor `1e-06`)
- high-mode norm median / min / max: `0.349631` / `0.185485` / `0.760476`
- signature-near sample coverage: `0.469195` vs `S_pos = 0.5` (`93839` of `200000`)
- candidate pairs unique / directed: `201908` / `403816`
- witness sample fraction: `0.46915` vs `0.01` (`93830` samples)
- witness pairs unique / directed: `179516` / `359032` vs min unique `100`
- witness high-distance p50 / p95: `0.037765` / `0.0609353`
- elapsed seconds: `6883.272`

## Paired fiber-constancy

**Paired verdict:** `PAIRED_FIBER_UNDEFINED`

- witness-pair action disagreement (unique): `0.0339468` (`6094` of `179516`) vs `delta_action = 0.1`
- witness-pair action disagreement (directed): `0.0339468`
- candidate-pair action disagreement (unique, comparator): `0.0314351` (`6347` of `201908`)

## Branch

No certified witness pairs (or no twin-state certificate), so the paired fiber-constancy test is undefined for this run.
Too few samples have a signature-near neighbour within `epsilon_K`; this run cannot adjudicate support-level non-injectivity.

## Files

- `manifest.json`
- `twin-state-witnesses.csv`
