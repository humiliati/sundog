# Yang-Mills Phase 2 Receipt - SU2 3D Relative-Locality v0 Null

- Receipt id: `2026-05-29_SU2_3D_phase2_no_rank_local_structure`
- Cell label: `SU2_3D`
- Phase: 2
- Date: 2026-05-29
- Author / runner: Codex runner, local Windows workspace
- Code commit: aggregation manifest `bb44e53da27bc8e705b1c2245eff4bae95550c3c`
- Git dirty: `true`
- Result directories:
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.0_ensemble_v0/`
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.4_ensemble_v0/`
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.8_ensemble_v0/`
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v0/`
- P0 lock version:
  [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
- Phase manifest:
  [`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v0.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v0.md)

## Registered Domain

- Lattice size: `12x12x12`
- Beta slate: `{2.0, 2.4, 2.8}`
- Boundary: `periodic`
- Action: `Wilson`
- Generator algorithm + update mix:
  `su2_heatbath_overrelax_v1`; Creutz / Kennedy-Pendleton heatbath plus
  Brown-Woch overrelaxation, dimension-3 staple sum, `1 HB + 4 OR` per
  combined sweep
- Random seeds: `202605290201`, `202605290202`, `202605290203`
- Burn-in sweep count: `2000` per beta
- Pilot tau_int(plaquette):
  beta 2.0 = `0.6285869601718574`;
  beta 2.4 = `0.9412059512301147`;
  beta 2.8 = `0.862567636802007`
- Registered thinning interval: `32`
- Measurement count after thinning: `32` per beta, `96` total
- Signature vocabulary version: `v1`
- Held-out target vocabulary version: `v1`
- gamma_held bin edges: frozen in `aggregation/per_beta_bin_edges.json`
  before nearest-neighbor scoring; global edges frozen in
  `aggregation/global_bin_edges.json`
- Control set used: `CTRL_META`, `CTRL_RAW`, `CTRL_RAND`,
  `CTRL_RAND_STRAT`, `CTRL_PERM`, `CTRL_GAUGE_RAND`;
  `CTRL_FINITE_SIZE` declared and deferred to Phase 4
- Compute wall-clock:
  beta 2.0 = `235.5348082` s;
  beta 2.4 = `248.1261217` s;
  beta 2.8 = `244.4304717` s;
  aggregation = `7.5372065` s
- Exact command lines:

```powershell
npm run yang-mills:phase2:su2-3d:beta-2.0
npm run yang-mills:phase2:su2-3d:beta-2.4
npm run yang-mills:phase2:su2-3d:beta-2.8
npm run yang-mills:phase2:su2-3d:aggregate
```

## Claim Under Test

Inside the registered `SU2_3D`, 12^3, beta-slate finite-lattice envelope,
the v1 gauge-invariant small-Wilson-loop signature was tested for
nearest-neighbor rank-local preservation of frozen held-out gamma_held tertile
labels beyond all Phase 2 v0 controls.

## Artifacts

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| Phase manifest | `docs/prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v0.md` | filed 2026-05-29 | run lock |
| Ensemble runner source | `scripts/yang-mills-phase2-su2-3d-ensemble.mjs` | `9D5F0F2A68118E18B4A40F12BA1B809A21390AA77DBA36C3F0AED76E4E42EE8F` | per-beta ensemble entry |
| Aggregation runner source | `scripts/yang-mills-phase2-su2-3d-aggregate.mjs` | `079C9E18D92635CECA395AD627A132A57148894935E5FB7A0CDD5FE2625E2E6B` | bin freezing / NN / controls / branch |
| SU(2) 3D core source | `scripts/lib/yang-mills-su2-3d-core.mjs` | `3897F3503553E1A109295ACC59D20B042A2CC4E08370E048715C792ED66642CF` | lattice / update / loop core |
| Package script | `package.json` | `592E3C5BEC395DF195BC0993B143832F850BD1BE7D8167DE636158972B9E3B76` | npm entrypoints |
| Beta 2.0 manifest | `beta2.0/manifest.json` | `4206428599d33a25112a81b175d7e751e5ffb527f4ee5c5b1d8b84049169bc7b` | ensemble runtime lock |
| Beta 2.4 manifest | `beta2.4/manifest.json` | `24cf9f7d1711ab47b1c7d524af790e1f86f894c0db59a8b906c5274114de61d8` | ensemble runtime lock |
| Beta 2.8 manifest | `beta2.8/manifest.json` | `ef8b76359b2fdb042b1dd9509f384e1d726ae8a60b2f8e3c9c20bc1132e385c7` | ensemble runtime lock |
| Aggregation manifest | `aggregation/manifest.json` | `ae41a4cd7db9b7d4e2e1de4cefc4cd0ff5b7033051d661a7c8bb77659bddb863` | aggregation runtime lock |
| Per-beta bin edges | `aggregation/per_beta_bin_edges.json` | `94daea7306f6440e363fd9c8281e545dd6d5e33906da5957eaf1e00a9dbd9f14` | frozen labels |
| Global bin edges | `aggregation/global_bin_edges.json` | `27d45cdeebb956776b9ec7fe367ffd86aff85c0c8692f1fad7cf5fc16e8ea968` | across-beta frozen labels |
| Primary NN graph | `aggregation/within_beta_nn_graphs.json` | `cccfad7a948cb6a91a3814a926623ec8d0898a2304163ea0187808b627b685a3` | primary within-beta graph |
| Control NN graphs | `aggregation/control_nn_graphs/*.json` | see `aggregation/hashes.json` | six scored controls |
| Rank-locality table | `aggregation/rank_locality_scores.csv` | `a330a0756a38092ec24d94aeceb811650dd30b522b8850d574c9a78424b04599` | primary + controls |
| Branch inputs | `aggregation/branch_inputs.json` | `94567245a6344bcb1815f2a6465d4c06e2575e23ffb4e18d885505678579ee7a` | gate numerics |
| Aggregation summary | `aggregation/summary.json` | `a4bd14f1b42d933654cc5949bd78c99f3b0f5ae26173ef0f0ce973b71df89ec1` | final verdict |

Full artifact hash maps live in each per-beta `hashes.json` and in
`results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v0/aggregation/hashes.json`.

## Observed Values

### Ensemble Quality

| Quantity | Registered threshold | Observed value | Pass / fail / quarantine |
| --- | --- | --- | --- |
| Burn-in sweeps | `>= 2000` per beta | `2000` each | pass |
| tau_int(plaquette) | `<= 16.0` | beta 2.0 `0.629`; beta 2.4 `0.941`; beta 2.8 `0.863` | pass |
| Thinning interval / tau_int | `>= 2.0` | all `> 33` | pass |
| Orientation spread | `<= 0.05` | beta 2.0 `0.00261`; beta 2.4 `0.000885`; beta 2.8 `0.000292` | pass |
| Heatbath fallback fraction | `<= 0.001` | `0` for all betas | pass |
| Link unitarity Frobenius residual | `<= 1e-10` | max `9.42e-16` | pass |
| Runtime wall clock | `<= 600` seconds per invocation | all four invocations under cap | pass |

### Rank-Locality Scores

Primary gate at `k = 5` failed.

| Lane / control | mean bin-purity@5 | discrimination ratio | 95% bootstrap CI | Gate note |
| --- | --- | --- | --- | --- |
| Within-beta primary | `0.310416666667` | `0.93125` | `0.26875` to `0.34796875` | fails `>= 0.5` |
| Within-beta `CTRL_RAND` | `0.300000000000` | `0.9` | `0.260416666667` to `0.343802083333` | primary margin only `0.010416666667`, fails `>= 0.10` |
| Within-beta `CTRL_META` | `0.314583333333` | `0.94375` | `0.279166666667` to `0.345833333333` | metadata equals/beats primary |
| Within-beta `CTRL_RAW` | `0.297916666667` | `0.89375` | `0.264583333333` to `0.333333333333` | primary margin only `0.0125` |
| Within-beta `CTRL_PERM` | `0.312052083333` | `0.93615625` | `0.311037291667` to `0.3130903125` | within chance gate |
| Within-beta `CTRL_GAUGE_RAND` | `0.310416666667` | `0.93125` | `0.272864583333` to `0.352135416667` | exactly matches primary purity |
| Across-beta primary | `0.425` | `1.275` | `0.372916666667` to `0.477083333333` | cross-check passes random-strat margin |
| Across-beta `CTRL_RAND_STRAT` | `0.354166666667` | `1.0625` | `0.304166666667` to `0.402083333333` | primary margin `0.070833333333` |

Secondary Kendall tau:
beta 2.0 `0.0477`; beta 2.4 `0.0267`; beta 2.8 `-0.0347`; aggregate
`0.0133`.

### Certificate Cost

Not applicable to Phase 2.

## Falsifier Disposition

Disposition: `YM-P2-NEG-A no_rank_local_structure`.

The primary within-beta signature graph did not preserve the registered
gamma_held tertile label above chance and did not beat `CTRL_RAND` by the
registered margin. The v0 invariant signature is therefore too coarse for the
registered Phase 2 held-out area-law proxy read.

## Verdict

Named null receipt.

This Phase 2 v0 run does not admit Phase 3 observable-certificate work. The
next admitted research artifact is a new dated probe spec under
`docs/yang-mills/specs/` proposing a v1 design change, or a dated P0 amendment
if the proposed change crosses the locked signature / target / control
boundary.

## Public Language Check

- [x] does not say "Sundog has a Yang-Mills result"
- [x] does not say "Sundog proves confinement"
- [x] does not say "Sundog found a mass gap"
- [x] does not imply continuum-limit reasoning
- [x] competitor framing is not used as a foil

## Notes

`CTRL_PERM` is scored as the deterministic mean over 1000 uniform label
permutation draws on the frozen primary graph. This removes avoidable
single-draw Monte Carlo from a hard graph-contamination gate while preserving
the registered permutation-null control. The control file records the
permutation-control resample count and seed base.

Several held-out Wilson-loop values hit the registered `1e-10` gamma_held
floor: clamp counts were beta 2.0 `11`, beta 2.4 `7`, beta 2.8 `2`.
Those clamp events are recorded in each `heldout/heldout_summary.csv` and are
part of this v0 null rather than a reason to revise bins after scoring.
