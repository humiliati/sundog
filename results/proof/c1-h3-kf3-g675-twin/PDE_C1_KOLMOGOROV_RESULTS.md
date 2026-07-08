# PDE C1 Twin-State Certificate Receipt

**Status:** TWIN_STATE_DEFERRED_COVERAGE
**Preset:** `lock_v7_g675_kf3`
**Adjudicator:** `twin-state`
**Interpretable certificate:** `False`

## Readout

- samples: `50000`, k (effective): `50`
- `epsilon_K` (signature radius): `0.0589334`
- `delta_H`: `0.0176033` (`0.05` x median high-mode norm, floor `1e-06`)
- high-mode norm median / min / max: `0.352066` / `0.206756` / `0.75966`
- signature-near sample coverage: `0.4588` vs `S_pos = 0.5` (`22940` of `50000`)
- candidate pairs unique / directed: `43926` / `87852`
- witness sample fraction: `0.45876` vs `0.01` (`22938` samples)
- witness pairs unique / directed: `38577` / `77154` vs min unique `100`
- witness high-distance p50 / p95: `0.0368511` / `0.0605525`
- elapsed seconds: `2718.516`

## Paired fiber-constancy

**Paired verdict:** `PAIRED_FIBER_UNDEFINED`

- witness-pair action disagreement (unique): `0.0328693` (`1268` of `38577`) vs `delta_action = 0.1`
- witness-pair action disagreement (directed): `0.0328693`
- candidate-pair action disagreement (unique, comparator): `0.0305969` (`1344` of `43926`)

## Branch

No certified witness pairs (or no twin-state certificate), so the paired fiber-constancy test is undefined for this run.
Too few samples have a signature-near neighbour within `epsilon_K`; this run cannot adjudicate support-level non-injectivity.

## Files

- `manifest.json`
- `twin-state-witnesses.csv`
