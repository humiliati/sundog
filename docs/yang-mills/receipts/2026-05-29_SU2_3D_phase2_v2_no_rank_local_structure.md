# Yang-Mills Phase 2 v2 Receipt - SU2 3D Connected-Correlator Relative-Locality Null

- Receipt id: `2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure`
- Cell label: `SU2_3D`
- Phase: 2 v2
- Date: 2026-05-29
- Author / runner: Codex runner, local Windows workspace
- Code commit: aggregation manifest `360ac2b60168e03246bc383cee88603a492d96c1`
- Git dirty: `true`
- Result directory:
  `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v2/`
- Input ensemble directories:
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.0_ensemble_v0/`
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.4_ensemble_v0/`
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.8_ensemble_v0/`
- P0 lock:
  [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
- Phase manifest:
  [`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v2.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v2.md)
- Probe spec:
  [`../specs/2026-05-29_phase2_v2_correlator_probe.md`](../specs/2026-05-29_phase2_v2_correlator_probe.md)

## Registered Domain

- Lattice size: `12x12x12`
- Beta slate: `{2.0, 2.4, 2.8}`
- Input configurations: the 96 retained bare-link configurations from the
  Phase 2 v0 ensembles
- Signature vocabulary: `v5`, 20-dimensional connected 2-point correlators
  of bare-link Wilson loops `{W11, W12, W13, W22}`
- Displacement classes: `{(1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,1,0)}`
  with class sizes `{6, 12, 8, 6, 24}`
- Held-out target vocabulary: unchanged bare-link v1,
  `{W14, W23, W33} -> gamma_held`
- Bin edges: recomputed from v0 `gamma_held` and asserted equal to v0
  per-beta and global edges to `<= 1e-12`
- Primary gate: within-beta k-NN bin-purity@5 `>= 0.5` and margin
  `>= 0.10` over `CTRL_RAND`, `CTRL_META`, and `CTRL_RAW`
- Exact command:

```powershell
npm run yang-mills:phase2:v2:su2-3d:aggregate
```

## Claim Under Test

Inside the registered `SU2_3D`, 12^3, beta-slate finite-lattice envelope,
the v5 bare-link connected-correlator small-Wilson-loop signature was
tested for nearest-neighbor rank-local preservation of the unchanged
bare-link `gamma_held` tertile labels beyond all Phase 2 controls.

## Artifacts

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| Phase manifest | `docs/prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v2.md` | filed 2026-05-29 | run lock |
| Probe spec | `docs/yang-mills/specs/2026-05-29_phase2_v2_correlator_probe.md` | filed 2026-05-29 | diagnostic and fallback lock |
| Correlator helper source | `scripts/lib/yang-mills-su2-3d-correlator.mjs` | `6748CE9F98154E3283DD690D872C75898F93E15E71C4D87D34EF1B82D8F2E9DD` | v5 feature construction |
| Aggregation runner source | `scripts/yang-mills-phase2-v2-su2-3d-aggregate.mjs` | `3ED75D3CCAF4D5454E4647549490DFBDC506BF189C4358B75B1C29FC2BA9000E` | v2 aggregation |
| SU(2) 3D core source | `scripts/lib/yang-mills-su2-3d-core.mjs` | `3897F3503553E1A109295ACC59D20B042A2CC4E08370E048715C792ED66642CF` | unchanged bare core |
| Package script | `package.json` | `87DC6C7A0E3C63BB151189DDE01032C4FEE25264FD37B275BDD692E9BCFDB2F1` | npm entrypoint |
| Aggregation manifest | `aggregation/manifest.json` | `52b556ae43d2950e3d80b94ee349cb90bcbacac6a591d2a601fcab362edf9b33` | runtime lock |
| Displacement classes | `aggregation/displacement_classes.json` | `641b0582ff4ecdd1330ec22f92adb25daf36c3de2f06b374fdbbd8e388353194` | v5 class audit |
| Correlator signatures | `aggregation/correlator_signature_vectors.csv` | `bbfe6680d36f2c8441141be759d518ebd033dc91a69a9ec0292a526fcfce7237` | v5 features |
| Rank-locality table | `aggregation/rank_locality_scores.csv` | `21f3464e2a591d886adcd30fdedf0ade88a2fa42478c4c55fb3f08f9995efe05` | primary + controls |
| v0/v1/v2 comparison | `aggregation/v0_v1_v2_comparison.json` | `af9b6f2a212c3c10ba438a772216ee62db2f96a87980830b47ce641f9b0503ea` | diagnostic delta |
| Branch inputs | `aggregation/branch_inputs.json` | `94a3a9733f96e8e31162a2ef5df7547bf1fbeeeeadf86920aa9976fc9b4a98a2` | gate numerics |
| Aggregation summary | `aggregation/summary.json` | `b7af5eacd109f51bb2cc0eb9a3e44658a5949e38109cbe783379e92893d54adf` | final verdict |

Full artifact hashes live at
`results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v2/aggregation/hashes.json`.

## Observed Values

### Health And Integrity

| Quantity | Registered threshold | Observed value | Pass / fail / quarantine |
| --- | --- | --- | --- |
| v0 ensemble health | all three `P1-A ensemble_health_pass` | all three pass; config hashes match | pass |
| Displacement class sizes | `{6, 12, 8, 6, 24}` | `{6, 12, 8, 6, 24}` | pass |
| v0 bin-edge replay | max abs diff `<= 1e-12` | per-beta `0`, global `0` | pass |
| Gauge-randomized correlator residual | `<= 1e-12` | `3.3306690738754696e-16` | pass |
| Gauge-randomized bin-purity diff | `<= 1e-12` | `0` | pass |
| Runtime wall clock | `<= 600` seconds | `10.838720799999999` s | pass |

### Rank-Locality Scores

Primary gate at `k = 5` failed.

| Lane / control | mean bin-purity@5 | discrimination ratio | 95% bootstrap CI | Gate note |
| --- | --- | --- | --- | --- |
| Within-beta primary, v5 correlator | `0.308333333333` | `0.925` | `0.270833333333` to `0.345833333333` | fails `>= 0.5` |
| Within-beta `CTRL_RAND` | `0.2875` | `0.8625` | `0.252083333333` to `0.327083333333` | primary margin `0.020833333333`, fails `>= 0.10` |
| Within-beta `CTRL_META` | `0.314583333333` | `0.94375` | `0.28125` to `0.347916666667` | metadata beats primary |
| Within-beta `CTRL_RAW` | `0.297916666667` | `0.89375` | `0.260416666667` to `0.335416666667` | primary margin `0.010416666667`, fails `>= 0.10` |
| Within-beta `CTRL_PERM` | `0.31311875` | `0.93935625` | `0.311907916667` to `0.314293802083` | within chance gate |
| Within-beta `CTRL_GAUGE_RAND` | `0.308333333333` | `0.925` | `0.270833333333` to `0.345833333333` | exactly matches primary purity |
| Across-beta primary | `0.35625` | `1.06875` | `0.314583333333` to `0.4` | fails `CTRL_RAND_STRAT` margin |
| Across-beta `CTRL_RAND_STRAT` | `0.35` | `1.05` | `0.30625` to `0.391666666667` | primary margin `0.00625`, fails `>= 0.05` |

v0 / v1 / v2 diagnostic: within-beta primary@5 moved from `0.310416666667`
at v0 to `0.29375` at v1 to `0.308333333333` at v2. Thus v2 recovered only
`0.014583333333` over v1 and stayed `0.002083333334` below v0. Across-beta
primary@5 moved from `0.425` at v0 to `0.358333333333` at v1 to `0.35625`
at v2.

## Falsifier Disposition

Disposition: `YM-P2-NEG-A no_rank_local_structure`.

The v5 connected-correlator signature did not recover the registered
within-beta `gamma_held` rank-locality signal. The correlator implementation
passed the bin-edge replay, displacement-class, and gauge-randomization
integrity gates, but the primary v5 graph remained below chance and did not
beat the registered random, metadata, or raw-link controls by the required
margins.

## Verdict

Named null receipt.

This Phase 2 v2 run does not admit Phase 3 observable-certificate work. With
v0, v1, and v2 all landing `YM-P2-NEG-A`, the small-loop hypothesis is now
exhausted across marginal bare summaries, marginal APE-smeared summaries,
and bare spatial connected-correlator summaries on this cell. The next
admitted research artifact is a dated v3 probe spec for target-side redesign,
following the fallback table pre-stated in the v2 probe spec.

## Public Language Check

- [x] does not say "Sundog has a Yang-Mills result"
- [x] does not say "Sundog proves confinement"
- [x] does not say "Sundog found a mass gap"
- [x] does not imply continuum-limit reasoning
- [x] competitor framing is not used as a foil

## Notes

The v2 aggregate uses the same v0 ensemble configurations and unchanged
bare-link held-out target. No new ensembles, smearing parameters, loop sets,
or displacement classes were introduced after scoring. The exact-match
`CTRL_GAUGE_RAND` bin-purity and machine-epsilon correlator residual confirm
that the v5 construction behaved as a gauge-invariant observable rather than
a gauge-coordinate shortcut.
