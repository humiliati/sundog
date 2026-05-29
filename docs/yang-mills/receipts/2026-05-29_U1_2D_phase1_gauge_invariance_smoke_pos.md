# Yang-Mills Phase 1 Receipt - U1 2D Gauge-Invariance Smoke

- Receipt id: `2026-05-29_U1_2D_phase1_gauge_invariance_smoke_pos`
- Cell label: `U1_2D`
- Phase: 1
- Date: 2026-05-29
- Author / runner: Codex runner, local Windows workspace
- Code commit: `c527eac45cd4ffdf7aee84340676986befee9393`
- Git dirty: `true`
- Result directory:
  `results/yang-mills/phase1/U1_2D/2026-05-29_u1_2d_gauge_invariance_smoke_v0/`
- P0 lock version:
  [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
- Phase manifest:
  [`../../prereg/yang-mills/PHASE1_U1_2D_gauge_invariance_smoke.md`](../../prereg/yang-mills/PHASE1_U1_2D_gauge_invariance_smoke.md)

## Registered Domain

- Lattice size: `16x16`
- Beta value: `1.0`
- Boundary: `periodic`
- Action: `Wilson`
- Generator algorithm + update mix: `u1_staple_metropolis_v1`; one Metropolis
  sweep per combined sweep; no overrelaxation
- Random seed: `202605290101`
- Burn-in sweep count: `2000`
- tau_int(plaquette) source and value:
  `autocorr_pilot/plaquette_series.csv`; `2.8876428130151304`
- Registered thinning interval: `32`
- Measurement count after thinning: `32`
- Signature vocabulary version: `v1`
- Held-out target vocabulary version: `v1`
- gamma_held bin edges: `phase1_no_rank_scoring`
- Control set declared: all seven P0 controls
- Control set scored in Phase 1: `CTRL_RAW`, `CTRL_GAUGE_RAND`
- Compute wall-clock: `0.7438203` seconds
- Exact command line:

```powershell
npm run yang-mills:phase1:u1-2d-smoke
```

Runtime manifest command line:

```text
C:\Program Files\nodejs\node.exe C:\Users\hughe\Dev\sundog\scripts\yang-mills-phase1-gauge-smoke.mjs --cell U1_2D --lattice-size 16x16 --beta 1.0 --boundary periodic --action Wilson --generator u1_staple_metropolis_v1 --proposal-half-width 0.75 --seed 202605290101 --burn-in 2000 --pilot-sweeps 512 --thinning 32 --measurements 32 --gauge-transforms 8 --signature-vocab v1 --heldout-vocab v1 --out results/yang-mills/phase1/U1_2D/2026-05-29_u1_2d_gauge_invariance_smoke_v0
```

## Claim Under Test

Inside the `U1_2D`, 16x16, beta 1.0 Abelian instrumentation cell, the v1
plaquette / small-loop signature is invariant under random lattice gauge
transformations, while raw link vectors are not accidentally invariant.

This is not a Yang-Mills, confinement, mass-gap, continuum, or non-Abelian
result.

## Artifacts

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| Phase manifest | `docs/prereg/yang-mills/PHASE1_U1_2D_gauge_invariance_smoke.md` | manifest filed 2026-05-29 | run lock |
| Runner source | `scripts/yang-mills-phase1-gauge-smoke.mjs` | `6ACC108DD57BE47DB9784961AE1813D34E389FA513A4F5A63616AB036830663C` | exact smoke entry |
| U(1) core source | `scripts/lib/yang-mills-u1-2d-core.mjs` | `085E59F355B39879FBE80B9F1742E56F3D6975D8270401487028C6C2F6F00EEE` | lattice / signature / gauge transform core |
| Package script | `package.json` | `E2D9D5742852B13985D7BB3ABFCCFE17472036B44C81F8BD0721D8B455D4D801` | npm entrypoint |
| Runtime manifest | `manifest.json` | `42103ad862b399092745a2c53d14c3bee4d60f42e5167dd402881433f5da698e` | runtime lock |
| Summary | `summary.json` | `7218309d9a5ed03a37e46de20d9f1829617284c8635318712349947d83a807b5` | verdict inputs |
| Ensemble configurations | `configs/u1_links.jsonl` | `64de2b74b4f9a8cff7ae97c8972d2a508d3adbc4e0a2261bc3005adaf58aac75` | post-burn-in, post-thinning |
| Autocorrelation pilot | `autocorr_pilot/plaquette_series.csv` | `d2c44b8ca6ad885c8e555b25d252247929e8b4d5fa0772356cafe94b2595641c` | tau_int registration evidence |
| Signature vectors | `signatures/signature_vectors.csv` | `eb5946d1500718547869142bec6c3342d0da848de3c1f6e80c9da85692b6514e` | per-configuration v1 signature |
| Held-out loop values | `heldout/heldout_loop_values.csv` | `1b8c97ee7530b1d88599decb1790eae78848d77926d9b1e5415d97d058f4f4af` | format compatibility only |
| gamma_held bin status | `heldout/gamma_bin_edges.json` | `8541f3529f9f33f07d606ceafbb006e1d1b53941577bf20f54a9e01e4ec93eac` | records no Phase 1 rank scoring |
| Signature residuals | `gauge_randomization/signature_residuals.csv` | `891cc5d36bba35d5b2ad98d9c0f20cd2afebd0c6af042c40213e77782d562bad` | `CTRL_GAUGE_RAND` numerics |
| Raw-link residuals | `gauge_randomization/raw_link_residuals.csv` | `1e34b2755d861f48d313fa28d28b156629ef37073a7891e4fe6a7528ab9755af` | `CTRL_RAW` numerics |

Full artifact hash map:
`results/yang-mills/phase1/U1_2D/2026-05-29_u1_2d_gauge_invariance_smoke_v0/hashes.json`.

## Observed Values

### Ensemble Quality

| Quantity | Registered threshold | Observed value | Pass / fail / quarantine |
| --- | --- | --- | --- |
| Burn-in sweeps | `>= 2000` | `2000` | pass |
| tau_int(plaquette) | `<= 16.0` for this Phase 1 manifest | `2.8876428130151304` | pass |
| Thinning interval / tau_int | `>= 2.0` | `11.081702991717048` | pass |
| Mean plaquette | reported, no published-range gate in Phase 1 | `0.4441463043421337` | reported |
| Metropolis acceptance | reported | burn-in `0.8589`; pilot `0.8596153259277344` | reported |
| Identity-transform signature residual | `<= 1e-12` | `0` | pass |
| Random-gauge signature residual | `<= 1e-12` | max `6.661338147750939e-16` | pass |
| Raw-link random-gauge residual | median `>= 1e-2`; at least 95 percent `> 1e-6` | median `0.4950600936853997`; fraction `1.0` | pass |
| Runtime wall clock | `<= 600` seconds | `0.7438203` | pass |

### Rank-Locality Scores

Not scored in Phase 1. The manifest declares all seven controls but interprets
only `CTRL_RAW` and `CTRL_GAUGE_RAND`. Any Phase 2 rank-locality result must
use a separate Phase 2 manifest with numeric gamma_held bin edges frozen before
scoring.

### Certificate Cost

Not applicable to Phase 1.

## Falsifier Disposition

Disposition: `P1-A smoke_pass`.

The gauge-randomization smoke passes: the primary v1 signature is invariant
under the identity transform and eight random U(1) gauge transforms per retained
configuration, and the raw-link diagnostic is not accidentally invariant.

## Verdict

Bounded positive receipt, Phase 1 instrumentation only.

The only admitted next step is another Phase 1 manifest or runner for the next
cell (`SU2_2D`). This receipt does not admit Phase 2 certificate language.

## Public Language Check

- [x] does not say "Sundog has a Yang-Mills result"
- [x] does not say "Sundog proves confinement"
- [x] does not say "Sundog found a mass gap"
- [x] does not imply continuum-limit reasoning
- [x] competitor framing is not used as a foil

## Notes

The runtime manifest records `gitDirty: true` because the runner, core module,
and package script were present in the working tree at execution time. Source
hashes for those files are recorded above so this smoke remains auditable before
the code is committed.

No Phase 2 gamma_held bins were frozen or scored. The held-out loop file exists
only to verify the runner's output schema for later phases.
