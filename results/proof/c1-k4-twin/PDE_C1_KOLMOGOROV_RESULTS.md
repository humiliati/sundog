# PDE C1 Twin-State Certificate Receipt

**Status:** TWIN_STATE_CERTIFIED
**Preset:** `lock_v5_k4`
**Adjudicator:** `twin-state`
**Interpretable certificate:** `True`

## Readout

- samples: `50000`, k (effective): `50`
- `epsilon_K` (signature radius): `0.0626036`
- `delta_H`: `0.00346867` (`0.05` x median high-mode norm, floor `1e-06`)
- high-mode norm median / min / max: `0.0693734` / `0.0642088` / `0.0781694`
- signature-near sample coverage: `1` vs `S_pos = 0.5` (`50000` of `50000`)
- candidate pairs unique / directed: `1265310` / `2450000`
- witness sample fraction: `1` vs `0.01` (`50000` samples)
- witness pairs unique / directed: `1140366` / `2200112` vs min unique `100`
- witness high-distance p50 / p95: `0.00749998` / `0.0114333`
- elapsed seconds: `2479.514`

## Paired fiber-constancy

**Paired verdict:** `PAIRED_FIBER_CONSTANCY_POSITIVE`

- witness-pair action disagreement (unique): `0.0347678` (`39648` of `1140366`) vs `delta_action = 0.1`
- witness-pair action disagreement (directed): `0.0341828`
- candidate-pair action disagreement (unique, comparator): `0.0324142` (`41014` of `1265310`)

## Branch

**Paired fiber-constancy POSITIVE.** The state-separated (witness) pairs the certificate found almost all require the SAME proxy action: action disagreement on certified non-injective pairs is at or below `delta_action`. This composes the non-injectivity and control-sufficiency reads on the SAME pairs, not just at a matched radius.
A positive-mass fraction of sampled states has a signature-near twin with high-mode separation above `delta_H`. This certifies `Phi_K` non-injective on the sampled SRB-like support for this cell.

## Files

- `manifest.json`
- `twin-state-witnesses.csv`
