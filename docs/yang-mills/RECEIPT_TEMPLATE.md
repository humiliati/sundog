# Yang-Mills Receipt Template

Use this template for every `sundog_v_yang_mills` phase or probe result.
Keep null and negative receipts. Do not rewrite a failed receipt into a
broader claim. A receipt that lands in a quarantine branch is filed as
that quarantine, not silently revised.

P0 lock: [`../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)

## Header

- Receipt id:
- Cell label: (one of `U1_2D`, `SU2_2D`, `SU2_3D`)
- Phase: (1, 2, 3, or 4)
- Date:
- Author / runner:
- Code commit:
- Result directory:
- P0 lock version:
- Phase manifest path:

## Registered Domain

- Lattice size: (must be one of the registered sizes for the cell)
- β value: (must be one of the registered slate points)
- Boundary: (must be `periodic`)
- Action: (must be `Wilson`)
- Generator algorithm + update mix:
- Random seed:
- Burn-in sweep count: (must be ≥ 2000)
- τ_int(plaquette) source and value:
- Registered thinning interval: (must satisfy `≥ 2 · τ_int`)
- Measurement count after thinning:
- Signature vocabulary version: (must be `v1`)
- Held-out target vocabulary version: (must be `v1`)
- γ_held bin edges (frozen before scoring):
- Control set used: (must include all seven `CTRL_*` entries)
- Compute wall-clock: (must be ≤ 10 minutes; over-cap quarantines as void)
- Exact command line:

## Claim Under Test

One sentence. It must be narrower than the P0 lock and must name the
cell × β × lattice-size envelope.

## Artifacts

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| Phase manifest | | | run lock |
| Ensemble configurations | | | post-burn-in, post-thinning |
| Autocorrelation pilot | | | τ_int registration evidence |
| Signature vectors | | | per-configuration `(W11, W12, W13, W22)` summaries |
| Held-out loop values | | | per-configuration `(W14, W23, W33)` summaries |
| γ_held bin assignments | | | label data |
| Neighbor graph | | | k-NN graph in signature space |
| Control neighbor graphs | | | k-NN graph for each `CTRL_*` |
| Rank-locality table | | | primary + 7 control scores |
| Gauge-randomization residuals | | | `CTRL_GAUGE_RAND` numerics |

## Observed Values

### Ensemble Quality (Phase 1 mandatory; Phase 2/3/4 inherited)

| Quantity | Registered threshold | Observed value | Pass / fail / quarantine |
| --- | --- | --- | --- |
| Burn-in sweeps | ≥ 2000 | | |
| τ_int(plaquette) | matches pilot ± tolerance | | |
| Thinning interval / τ_int | ≥ 2 | | |
| Mean plaquette | within published heatbath range for cell × β | | |
| Heatbath / overrelax acceptance | | | |
| `CTRL_GAUGE_RAND` invariance residual | ≤ numerical tolerance (e.g. 1e-12 on `W11`) | | |
| `CTRL_RAW` non-invariance | non-zero (raw must NOT be accidentally invariant) | | |

### Rank-Locality Scores (Phase 2 primary)

| Lane | k-NN rank-locality on γ_held | Notes |
| --- | --- | --- |
| Primary signature | | |
| `CTRL_META` | | metadata-shortcut check |
| `CTRL_RAW` | | gauge-variant diagnostic |
| `CTRL_RAND` | | floor |
| `CTRL_RAND_STRAT` | | β-stratified floor |
| `CTRL_PERM` | | label-permutation null |
| `CTRL_GAUGE_RAND` | | invariance witness |
| `CTRL_FINITE_SIZE` | | partner-size restriction |

### Certificate Cost (Phase 3 only)

| Quantity | Registered budget | Observed value | Pass / fail / quarantine |
| --- | --- | --- | --- |
| Verifier cost per configuration | | | |
| Source-binding integrity | all checks pass | | |
| Spoof control success rate | 0 | | |

## Falsifier Disposition

Choose exactly one:

- A — bounded positive: primary signature beats every control on
  held-out γ_held label, and `CTRL_GAUGE_RAND` confirms invariance.
- B — `YM-P2-NEG-B metadata_only`: `CTRL_META` matches or beats primary.
- C — `YM-P2-NEG-A no_rank_local_structure`: primary fails to beat
  `CTRL_RAND` or `CTRL_RAND_STRAT`.
- D — `YM-P1-NEG-A gauge_leakage`: `CTRL_GAUGE_RAND` breaks invariance.
- E — `YM-P3-NEG-A certificate_spoof / target_leakage`: held-out loop
  class derivable from signature without area-law mechanism.
- F — `YM-P4-DEFERRED_FINITE_SIZE`: signal disappears or reverses under
  `CTRL_FINITE_SIZE`.
- G — `YM-P2-NEG-C coupling_triviality`: result tracks β bin only.
- H — domain-leak quarantine: interpretation requires changing cell,
  lattice slate, β slate, signature vocabulary, or held-out target.
- Z — void run: compute-cap breach, missing admission field, or
  manifest / commit-hash drift.

## Verdict

Allowed values:

- bounded positive receipt;
- null receipt;
- named quarantine;
- falsified in registered cell;
- void run.

## Public Language Check

Confirm the receipt body and any derived public surface contain none of
the forbidden phrases (per P0 lock "Public Language Boundary"):

- [ ] does not say "Sundog has a Yang-Mills result"
- [ ] does not say "Sundog proves confinement"
- [ ] does not say "Sundog found a mass gap"
- [ ] does not imply continuum-limit reasoning
- [ ] competitor framing (L-CNN, bootstrap, equivariant diffusion) is
      treated as live baseline language, not foil

## Notes

Record interpretation, caveats, and next allowed step. If a domain
expansion is desired, name it as a new dated probe spec under
`docs/yang-mills/specs/` rather than modifying this receipt.
