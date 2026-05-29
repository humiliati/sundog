# Yang-Mills Phase 1 Receipt - SU2 3D Gauge-Invariance Smoke

- Receipt id: `2026-05-29_SU2_3D_phase1_gauge_invariance_smoke_pos`
- Cell label: `SU2_3D`
- Phase: 1
- Date: 2026-05-29
- Author / runner: Codex runner, local Windows workspace
- Code commit: `02248316ecf6489190b67b512ba8515a80d90add`
- Git dirty: `true`
- Result directory:
  `results/yang-mills/phase1/SU2_3D/2026-05-29_su2_3d_gauge_invariance_smoke_v0/`
- P0 lock version:
  [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
- Phase manifest:
  [`../../prereg/yang-mills/PHASE1_SU2_3D_gauge_invariance_smoke.md`](../../prereg/yang-mills/PHASE1_SU2_3D_gauge_invariance_smoke.md)

## Registered Domain

- Lattice size: `8x8x8`
- Beta value: `2.4`
- Boundary: `periodic`
- Action: `Wilson`
- Generator algorithm + update mix:
  `su2_heatbath_overrelax_v1`; Creutz / Kennedy-Pendleton heatbath plus
  Brown-Woch overrelaxation, dimension-3 staple sum, `1 HB + 4 OR` per
  combined sweep
- Random seed: `202605290103`
- Burn-in sweep count: `2000`
- tau_int(plaquette) source and value:
  `autocorr_pilot/plaquette_series.csv`; `0.8070525419094978`
- Registered thinning interval: `32`
- Measurement count after thinning: `32`
- Signature vocabulary version: `v1`
- Held-out target vocabulary version: `v1`
- gamma_held bin edges: `phase1_no_rank_scoring`
- Control set declared: all seven P0 controls
- Control set scored in Phase 1: `CTRL_RAW`, `CTRL_GAUGE_RAND`
- Compute wall-clock: `37.028111700000004` seconds
- Exact command line:

```powershell
npm run yang-mills:phase1:su2-3d-smoke
```

Runtime manifest command line:

```text
C:\Program Files\nodejs\node.exe C:\Users\hughe\Dev\sundog\scripts\yang-mills-phase1-su2-3d-gauge-smoke.mjs --cell SU2_3D --lattice-size 8x8x8 --beta 2.4 --boundary periodic --action Wilson --generator su2_heatbath_overrelax_v1 --overrelax-per-heatbath 4 --seed 202605290103 --burn-in 2000 --pilot-sweeps 512 --thinning 32 --measurements 32 --gauge-transforms 8 --signature-vocab v1 --heldout-vocab v1 --out results/yang-mills/phase1/SU2_3D/2026-05-29_su2_3d_gauge_invariance_smoke_v0
```

## Claim Under Test

Inside the `SU2_3D`, 8x8x8, beta 2.4 primary instrumentation cell, the v1
small-Wilson-loop signature, averaged over positions and the three plane
orientations `{xy, xz, yz}`, is invariant under random local SU(2) gauge
transformations, while raw link matrix-entry vectors are not accidentally
invariant.

This is not a Yang-Mills, confinement, mass-gap, continuum, or Phase 2
rank-locality result.

## Artifacts

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| Phase manifest | `docs/prereg/yang-mills/PHASE1_SU2_3D_gauge_invariance_smoke.md` | manifest filed 2026-05-29 | run lock |
| Runner source | `scripts/yang-mills-phase1-su2-3d-gauge-smoke.mjs` | `C17734B2DE21A67E348DB4B571D1834D521C8103C8903745B7362228B95EB2B8` | exact smoke entry |
| SU(2) 3D core source | `scripts/lib/yang-mills-su2-3d-core.mjs` | `3897F3503553E1A109295ACC59D20B042A2CC4E08370E048715C792ED66642CF` | lattice / heatbath / overrelaxation / signature / gauge transform core |
| Package script | `package.json` | `4A555DE82EF226B4BCCE7EC29522C6E514B2CCA8CC9C26005CCFAB74146EE736` | npm entrypoint |
| Runtime manifest | `manifest.json` | `a8fdf225a8faed74bf889d1c81029e173d336f00667de82f2d40876b00a358fd` | runtime lock |
| Summary | `summary.json` | `c9371918ee15c12dc60d404810a57eda8ab93d05b7ccd59581654f70ef4132ab` | verdict inputs |
| Ensemble configurations | `configs/su2_links.jsonl` | `54c60d4c5a84c62ede281dfcc4d6cdd5368e6ea48d61ba26b74f0f15036f39ec` | post-burn-in, post-thinning |
| Autocorrelation pilot | `autocorr_pilot/plaquette_series.csv` | `e44249e7dc47e9cd9e223ce6c96269ceee45978e7856c62671e2383a238ddf3c` | tau_int registration evidence |
| Orientation pilot | `autocorr_pilot/plaquette_by_orientation.csv` | `228001a22955ebec41725e360a88d85676da962f16ce4f1f180a65e5efe6fed8` | `YM-P1-QUAR-C` isotropy gate |
| Signature vectors | `signatures/signature_vectors.csv` | `9f336917af987da41057a4d3d5cb8261f1d87cb42e4dbf9475d64e804fda6183` | per-configuration v1 signature |
| Held-out loop values | `heldout/heldout_loop_values.csv` | `857ce6a0ddb44a5b7bcd173405d631b8f62662de9239bd7235012a276de23e4e` | format compatibility only |
| gamma_held bin status | `heldout/gamma_bin_edges.json` | `9f9c1ab71b91fd60541096a3aa5aa80676144300301fef1cff4207236067f9e9` | records no Phase 1 rank scoring |
| Signature residuals | `gauge_randomization/signature_residuals.csv` | `b3ce2a3524bdc04e9c7f8ac2465aad63dda3e3ce67f95dd9f1a5a2ea23dbcf43` | `CTRL_GAUGE_RAND` numerics |
| Raw-link residuals | `gauge_randomization/raw_link_residuals.csv` | `d5eaa1044c50db5c42cc378963c306d6b022ec7b990b88f836c6ec4503f16ab9` | `CTRL_RAW` numerics |

Full artifact hash map:
`results/yang-mills/phase1/SU2_3D/2026-05-29_su2_3d_gauge_invariance_smoke_v0/hashes.json`.

## Observed Values

### Ensemble Quality

| Quantity | Registered threshold | Observed value | Pass / fail / quarantine |
| --- | --- | --- | --- |
| Burn-in sweeps | `>= 2000` | `2000` | pass |
| tau_int(plaquette) | `<= 16.0` for this Phase 1 manifest | `0.8070525419094978` | pass |
| Thinning interval / tau_int | `>= 2.0` | `39.65045438589109` | pass |
| Mean plaquette | reported, no published-range gate in Phase 1 | `0.529036397399184` | reported |
| Per-orientation mean plaquette | relative spread `<= 5e-2` | `xy=0.5299769326584903`; `xz=0.5284172287276812`; `yz=0.5287150308113808`; spread `0.002942965692837478` | pass |
| Heatbath fallback fraction | `<= 0.001` | `0` | pass |
| Heatbath fallback count | reported | `0 / 5431296` link updates | reported |
| Identity-transform signature residual | `<= 1e-12` | `2.220446049250313e-16` | pass |
| Random-gauge signature residual | `<= 1e-12` | max `3.885780586188048e-16` | pass |
| Raw-matrix random-gauge residual | median `>= 1e-2`; at least 95 percent `> 1e-6` | median `1.4151164784034174`; fraction `1.0` | pass |
| Link unitarity Frobenius residual | `<= 1e-10` | max `1.0990647210786425e-15` | pass |
| Runtime wall clock | `<= 600` seconds | `37.028111700000004` | pass |

### Rank-Locality Scores

Not scored in Phase 1. The manifest declares all seven controls but interprets
only `CTRL_RAW` and `CTRL_GAUGE_RAND`. Any Phase 2 rank-locality result must
use a separate Phase 2 manifest on the `12x12x12` partner lattice with numeric
gamma_held bin edges frozen before scoring and all seven controls live.

### Certificate Cost

Not applicable to Phase 1.

## Falsifier Disposition

Disposition: `P1-A smoke_pass`.

The SU(2) 3D gauge-randomization smoke passes: the primary v1 signature is
invariant under the identity transform and eight random SU(2) Haar gauge
transforms per retained configuration; raw link matrices move under those
transforms; the heatbath fallback, orientation-anisotropy, and link-unitarity
quarantine gates are quiet.

## Verdict

Bounded positive receipt, Phase 1 instrumentation only.

Phase 1 instrumentation is now closed across the registered ladder:
`U1_2D`, `SU2_2D`, and `SU2_3D`. The admitted next research artifact is a
Phase 2 `SU2_3D` relative-locality manifest on the `12x12x12` partner lattice,
with numeric gamma_held bin edges frozen before scoring.

## Public Language Check

- [x] does not say "Sundog has a Yang-Mills result"
- [x] does not say "Sundog proves confinement"
- [x] does not say "Sundog found a mass gap"
- [x] does not imply continuum-limit reasoning
- [x] competitor framing is not used as a foil

## Notes

The runtime manifest records `gitDirty: true` because the runner, core module,
package script, and docs were present in the working tree at execution time.
Source hashes for those files are recorded above so this smoke remains
auditable before the code is committed.

No Phase 2 gamma_held bins were frozen or scored. The held-out loop file exists
only to verify the runner's output schema for later phases.
