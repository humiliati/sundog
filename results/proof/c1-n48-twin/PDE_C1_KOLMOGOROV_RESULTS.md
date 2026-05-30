# PDE C1 Twin-State Certificate Receipt

**Status:** TWIN_STATE_CERTIFIED
**Preset:** `lock_v5_n48`
**Adjudicator:** `twin-state`
**Interpretable certificate:** `True`

## Readout

- samples: `50000`, k (effective): `50`
- `epsilon_K` (signature radius): `0.0605953`
- `delta_H`: `0.0116667` (`0.05` x median high-mode norm, floor `1e-06`)
- high-mode norm median / min / max: `0.233335` / `0.220171` / `0.244183`
- signature-near sample coverage: `1` vs `S_pos = 0.5` (`50000` of `50000`)
- candidate pairs unique / directed: `1262205` / `2450000`
- witness sample fraction: `1` vs `0.01` (`50000` samples)
- witness pairs unique / directed: `689263` / `1305029` vs min unique `100`
- witness high-distance p50 / p95: `0.0154814` / `0.0197429`
- elapsed seconds: `2878.243`

## Paired fiber-constancy

**Paired verdict:** `PAIRED_FIBER_CONSTANCY_POSITIVE`

- witness-pair action disagreement (unique): `0.037691` (`25979` of `689263`) vs `delta_action = 0.1`
- witness-pair action disagreement (directed): `0.0367478`
- candidate-pair action disagreement (unique, comparator): `0.031872` (`40229` of `1262205`)

## Branch

**Paired fiber-constancy POSITIVE.** The state-separated (witness) pairs the certificate found almost all require the SAME proxy action: action disagreement on certified non-injective pairs is at or below `delta_action`. This composes the non-injectivity and control-sufficiency reads on the SAME pairs, not just at a matched radius.
A positive-mass fraction of sampled states has a signature-near twin with high-mode separation above `delta_H`. This certifies `Phi_K` non-injective on the sampled SRB-like support for this cell.

## Files

- `manifest.json`
- `twin-state-witnesses.csv`
