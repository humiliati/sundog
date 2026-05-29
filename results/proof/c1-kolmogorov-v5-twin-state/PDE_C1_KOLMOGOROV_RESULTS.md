# PDE C1 Twin-State Certificate Receipt

**Status:** TWIN_STATE_CERTIFIED
**Preset:** `lock_v5`
**Adjudicator:** `twin-state`
**Interpretable certificate:** `True`

## Readout

- samples: `50000`, k (effective): `50`
- `epsilon_K` (signature radius): `0.0605976`
- `delta_H`: `0.0116667` (`0.05` x median high-mode norm, floor `1e-06`)
- high-mode norm median / min / max: `0.233334` / `0.220171` / `0.244182`
- signature-near sample coverage: `1` vs `S_pos = 0.5` (`50000` of `50000`)
- candidate pairs unique / directed: `1263121` / `2450000`
- witness sample fraction: `1` vs `0.01` (`50000` samples)
- witness pairs unique / directed: `693795` / `1312267` vs min unique `100`
- witness high-distance p50 / p95: `0.0153903` / `0.0209765`
- elapsed seconds: `1294.585`

## Branch

A positive-mass fraction of sampled states has a signature-near twin with high-mode separation above `delta_H`. This certifies `Phi_K` non-injective on the sampled SRB-like support for this cell.

## Files

- `manifest.json`
- `twin-state-witnesses.csv`
