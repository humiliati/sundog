# Yang-Mills Phase 1 Receipt - SU2 2D Gauge-Invariance Smoke

- Receipt id: `2026-05-29_SU2_2D_phase1_gauge_invariance_smoke_pos`
- Cell label: `SU2_2D`
- Phase: 1
- Date: 2026-05-29
- Author / runner: Codex runner, local Windows workspace
- Code commit: `8f1f25f3e67364e4fd3e2e9262bb0d0c5d5df1a4`
- Git dirty: `true`
- Result directory:
  `results/yang-mills/phase1/SU2_2D/2026-05-29_su2_2d_gauge_invariance_smoke_v0/`
- P0 lock version:
  [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
- Phase manifest:
  [`../../prereg/yang-mills/PHASE1_SU2_2D_gauge_invariance_smoke.md`](../../prereg/yang-mills/PHASE1_SU2_2D_gauge_invariance_smoke.md)

## Registered Domain

- Lattice size: `16x16`
- Beta value: `2.0`
- Boundary: `periodic`
- Action: `Wilson`
- Generator algorithm + update mix:
  `su2_heatbath_overrelax_v1`; Creutz / Kennedy-Pendleton heatbath plus
  Brown-Woch overrelaxation, `1 HB + 4 OR` per combined sweep
- Random seed: `202605290102`
- Burn-in sweep count: `2000`
- tau_int(plaquette) source and value:
  `autocorr_pilot/plaquette_series.csv`; `0.4038402078709045`
- Registered thinning interval: `32`
- Measurement count after thinning: `32`
- Signature vocabulary version: `v1`
- Held-out target vocabulary version: `v1`
- gamma_held bin edges: `phase1_no_rank_scoring`
- Control set declared: all seven P0 controls
- Control set scored in Phase 1: `CTRL_RAW`, `CTRL_GAUGE_RAND`
- Compute wall-clock: `7.1781799` seconds
- Exact command line:

```powershell
npm run yang-mills:phase1:su2-2d-smoke
```

Runtime manifest command line:

```text
C:\Program Files\nodejs\node.exe C:\Users\hughe\Dev\sundog\scripts\yang-mills-phase1-su2-gauge-smoke.mjs --cell SU2_2D --lattice-size 16x16 --beta 2.0 --boundary periodic --action Wilson --generator su2_heatbath_overrelax_v1 --overrelax-per-heatbath 4 --seed 202605290102 --burn-in 2000 --pilot-sweeps 512 --thinning 32 --measurements 32 --gauge-transforms 8 --signature-vocab v1 --heldout-vocab v1 --out results/yang-mills/phase1/SU2_2D/2026-05-29_su2_2d_gauge_invariance_smoke_v0
```

## Claim Under Test

Inside the `SU2_2D`, 16x16, beta 2.0 non-Abelian harness cell, the v1
small-Wilson-loop signature is invariant under random local SU(2) gauge
transformations, while raw link matrices are not accidentally invariant.

This is not a Yang-Mills, confinement, mass-gap, continuum, or Phase 2
rank-locality result.

## Artifacts

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| Phase manifest | `docs/prereg/yang-mills/PHASE1_SU2_2D_gauge_invariance_smoke.md` | manifest filed 2026-05-29 | run lock |
| Runner source | `scripts/yang-mills-phase1-su2-gauge-smoke.mjs` | `8ED7A7087EC0B002505702D00F9D7642B1FEDFEC1742A63152367BB6F73480E4` | exact smoke entry |
| SU(2) core source | `scripts/lib/yang-mills-su2-2d-core.mjs` | `B075EF634A9E97855FE36BD80511D12F4E677EAA252E15781A9257BAE54CD07D` | lattice / heatbath / overrelaxation / signature / gauge transform core |
| Package script | `package.json` | `A4EE2D96798299CEC0CBE98867037C346B5AEC3E47EE5E33BF98D02E87D10FCF` | npm entrypoint |
| Runtime manifest | `manifest.json` | `75a5078d213d19c79d3aedf87a86306a38f47da9ddfbf496c1ad05985361cca5` | runtime lock |
| Summary | `summary.json` | `e1ebd606c39e1486350507c2cb98ab03048163ed762140a5035b1829371b8f8b` | verdict inputs |
| Ensemble configurations | `configs/su2_links.jsonl` | `ae567d5c8c7b2967aff00d5f001fd3fd4f15f893c408e4fab1a8b73376db93bc` | post-burn-in, post-thinning |
| Autocorrelation pilot | `autocorr_pilot/plaquette_series.csv` | `7687eea5066dbb241e1dd303b4a03176c9702e192abcc4424f7468d3de242d05` | tau_int registration evidence |
| Signature vectors | `signatures/signature_vectors.csv` | `3a39492f76785de6703b138f41afbc499aad2b5e706284297518a70d4aa6ae6d` | per-configuration v1 signature |
| Held-out loop values | `heldout/heldout_loop_values.csv` | `4c6b771d5333cd801d29878f21b10997ab6403623119b1929a719478cbbf7f6e` | format compatibility only |
| gamma_held bin status | `heldout/gamma_bin_edges.json` | `9f9c1ab71b91fd60541096a3aa80676144300301fef1cff4207236067f9e9` | records no Phase 1 rank scoring |
| Signature residuals | `gauge_randomization/signature_residuals.csv` | `6152073cecc29ae6c088044d70a7c032731b624a8af4f8647529f5c9fb418d71` | `CTRL_GAUGE_RAND` numerics |
| Raw-link residuals | `gauge_randomization/raw_link_residuals.csv` | `e63a07dac73dcf57db7dac6ecba9cdcc082bb0c056ca9b6429b19727ce582499` | `CTRL_RAW` numerics |

Full artifact hash map:
`results/yang-mills/phase1/SU2_2D/2026-05-29_su2_2d_gauge_invariance_smoke_v0/hashes.json`.

## Observed Values

### Ensemble Quality

| Quantity | Registered threshold | Observed value | Pass / fail / quarantine |
| --- | --- | --- | --- |
| Burn-in sweeps | `>= 2000` | `2000` | pass |
| tau_int(plaquette) | `<= 16.0` for this Phase 1 manifest | `0.4038402078709045` | pass |
| Thinning interval / tau_int | `>= 2.0` | `79.23926190685161` | pass |
| Mean plaquette | reported, no published-range gate in Phase 1 | `0.43322159794351506` | reported |
| Heatbath fallback fraction | `<= 0.001` | `0` | pass |
| Heatbath fallback count | reported | `0 / 1810432` link updates | reported |
| Identity-transform signature residual | `<= 1e-12` | `2.220446049250313e-16` | pass |
| Random-gauge signature residual | `<= 1e-12` | max `3.0531133177191805e-16` | pass |
| Raw-matrix random-gauge residual | median `>= 1e-2`; at least 95 percent `> 1e-6` | median `1.4139401605938902`; fraction `1.0` | pass |
| Link unitarity residual | `<= 1e-10` | max `6.661338147750939e-16` | pass |
| Runtime wall clock | `<= 600` seconds | `7.1781799` | pass |

### Rank-Locality Scores

Not scored in Phase 1. The manifest declares all seven controls but interprets
only `CTRL_RAW` and `CTRL_GAUGE_RAND`. Any Phase 2 rank-locality result must
use a separate Phase 2 manifest with numeric gamma_held bin edges frozen before
scoring.

### Certificate Cost

Not applicable to Phase 1.

## Falsifier Disposition

Disposition: `P1-A smoke_pass`.

The SU(2) 2D gauge-randomization smoke passes: the primary v1 signature is
invariant under the identity transform and eight random SU(2) Haar gauge
transforms per retained configuration, raw link matrices move under those
transforms, and the registered `YM-P1-QUAR-B heatbath_pathology` fallback gate
is quiet.

## Verdict

Bounded positive receipt, Phase 1 instrumentation only.

The admitted next step is another Phase 1 manifest or runner for the next
ladder cell (`SU2_3D`). This receipt does not admit Phase 2 certificate
language.

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
