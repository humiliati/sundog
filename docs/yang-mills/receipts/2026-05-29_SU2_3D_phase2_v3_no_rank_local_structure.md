# Yang-Mills Phase 2 v3 Receipt - SU2 3D W33-Variance Target Null

- Receipt id: `2026-05-29_SU2_3D_phase2_v3_no_rank_local_structure`
- Cell label: `SU2_3D`
- Phase: 2 v3
- Date: 2026-05-29
- Author / runner: Codex runner, local Windows workspace
- Code commit: aggregation manifest `f638b4d5f00db06e3717819369776b87ee294e5a`
- Git dirty: `true`
- Result directory:
  `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v3/`
- Input ensemble directories:
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.0_ensemble_v0/`
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.4_ensemble_v0/`
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.8_ensemble_v0/`
- P0 lock:
  [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
- Phase manifest:
  [`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v3.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v3.md)
- Probe spec:
  [`../specs/2026-05-29_phase2_v3_target_redesign_probe.md`](../specs/2026-05-29_phase2_v3_target_redesign_probe.md)

## Registered Domain

- Lattice size: `12x12x12`
- Beta slate: `{2.0, 2.4, 2.8}`
- Input configurations: the 96 retained bare-link configurations from the
  Phase 2 v0 ensembles
- Signature vocabulary: unchanged `v1`, bare 8-dimensional mean/variance
  signature for `{W11, W12, W13, W22}`, re-read from v0 CSV and SHA-256
  asserted against each v0 `hashes.json`
- Held-out target vocabulary: `v2`, `sigma2_W33`, the biased spatial
  variance of `(1/2) Re Tr U_loop_3x3` across all `12^3 x 3 = 5184`
  position-orientation samples per configuration
- Bin edges: new per-beta and global tertile edges computed on
  `sigma2_W33`; not asserted equal to v0 `gamma_held` edges because the
  target is different
- Primary gate: within-beta k-NN bin-purity@5 `>= 0.5` and margin
  `>= 0.10` over `CTRL_RAND`, `CTRL_META`, and `CTRL_RAW`
- Exact command:

```powershell
npm run yang-mills:phase2:v3:su2-3d:aggregate
```

## Claim Under Test

Inside the registered `SU2_3D`, 12^3, beta-slate finite-lattice envelope,
the unchanged v1 small-Wilson-loop signature was tested for nearest-
neighbor rank-local preservation of the new held-out `sigma2_W33` tertile
labels beyond all Phase 2 controls.

## Artifacts

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| Phase manifest | `docs/prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v3.md` | filed 2026-05-29 | run lock |
| Probe spec | `docs/yang-mills/specs/2026-05-29_phase2_v3_target_redesign_probe.md` | filed 2026-05-29 | diagnostic and fallback lock |
| Aggregation runner source | `scripts/yang-mills-phase2-v3-su2-3d-aggregate.mjs` | `0B573A01348EED0FF7C1F26C493A3BCD3558B3C8A2C879BA0C7B78379AA6061C` | v3 aggregation |
| SU(2) 3D core source | `scripts/lib/yang-mills-su2-3d-core.mjs` | `3897F3503553E1A109295ACC59D20B042A2CC4E08370E048715C792ED66642CF` | W33 target + controls |
| Package script | `package.json` | `CCA5D28ECE78F6F9800999DBC228827F0AB05582A3B3D0C692B61334572FC820` | npm entrypoint |
| Aggregation manifest | `aggregation/manifest.json` | `292be49e8f42c23277f11038e724ac9e19ab0a12948d08c2243f72ae5ee15c08` | runtime lock |
| v0 ensemble sources | `aggregation/v0_ensemble_sources.json` | `a18454f6e2760944e10ae3ec6f6f470bb5fe989052d94406af5cab94f4a1332e` | config + signature hash assertions |
| v3 target summary | `aggregation/v3_target_summary.csv` | `bbe66444d9767e8f6315215ffb2cb7f2504172ffae574cf9a9976a3e81614b3a` | `sigma2_W33` target values |
| Per-beta target edges | `aggregation/per_beta_v3_bin_edges.json` | `623c67e170e4524ecb81a9a68c7c29e3c60ea0d24d2fdea8b090fae0f36cfb8c` | frozen v3 bins |
| Global target edges | `aggregation/global_v3_bin_edges.json` | `b545acf6ac4bf3705e0b633ae40de15825d175906660562c319cc3a63ce25afd` | across-beta bins |
| Rank-locality table | `aggregation/rank_locality_scores.csv` | `56161053b467865b800ec6ee199f8e8a1a232deecbd74e7b158649425e4ee7bb` | primary + controls |
| v0/v3 comparison | `aggregation/v0_v3_comparison.json` | `5d582f6f305af3ca35e2db054b67be86fc1e82254cba96b05dfc8ed553eda715` | target-redesign diagnostic delta |
| Branch inputs | `aggregation/branch_inputs.json` | `2221039a88987d538f4d48c2b7947acb8892b653da29a75f7fc3438d75961ed7` | gate numerics |
| Aggregation summary | `aggregation/summary.json` | `bf888ecd9590484f82b60f8f42c3440317918a6014f19620e4969619012df3b0` | final verdict |

Full artifact hashes live at
`results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v3/aggregation/hashes.json`.

## Observed Values

### Health And Integrity

| Quantity | Registered threshold | Observed value | Pass / fail / quarantine |
| --- | --- | --- | --- |
| v0 ensemble health | all three `P1-A ensemble_health_pass` | all three pass; config and signature hashes match | pass |
| v3 bin-edge timestamp | written before first NN graph | `true` | pass |
| Per-beta `sigma2_W33` spread | `>= 1e-12` | no degenerate betas | pass |
| Gauge-randomized signature residual | `<= 1e-12` | `5.000444502911705e-13` | pass |
| Gauge-randomized target residual | `<= 1e-12` | `2.220446049250313e-16` | pass |
| Gauge-randomized bin-purity diff | `<= 1e-12` | `0` | pass |
| Runtime wall clock | `<= 600` seconds | `11.6402463` s | pass |

### Rank-Locality Scores

Primary gate at `k = 5` failed.

| Lane / control | mean bin-purity@5 | discrimination ratio | 95% bootstrap CI | Gate note |
| --- | --- | --- | --- | --- |
| Within-beta primary, v1 signature -> `sigma2_W33` | `0.329166666667` | `0.9875` | `0.29375` to `0.366666666667` | fails `>= 0.5` |
| Within-beta `CTRL_RAND` | `0.302083333333` | `0.90625` | `0.266666666667` to `0.337552083333` | primary margin `0.027083333333`, fails `>= 0.10` |
| Within-beta `CTRL_META` | `0.308333333333` | `0.925` | `0.275` to `0.339583333333` | primary margin `0.020833333333`, fails `>= 0.10` |
| Within-beta `CTRL_RAW` | `0.26875` | `0.80625` | `0.235416666667` to `0.30625` | primary margin `0.060416666667`, fails `>= 0.10` |
| Within-beta `CTRL_PERM` | `0.312714583333` | `0.93814375` | `0.311707916667` to `0.313729947917` | within chance gate |
| Within-beta `CTRL_GAUGE_RAND` | `0.329166666667` | `0.9875` | `0.29375` to `0.3625` | exactly matches primary purity |
| Across-beta primary | `0.320833333333` | `0.9625` | `0.283333333333` to `0.360416666667` | fails `CTRL_RAND_STRAT` margin |
| Across-beta `CTRL_RAND_STRAT` | `0.3625` | `1.0875` | `0.322864583333` to `0.4` | beats primary |

v0-to-v3 diagnostic: within-beta primary@5 moved from `0.310416666667`
on `gamma_held` to `0.329166666667` on `sigma2_W33`, a `+0.01875`
change that remains at chance and below the promotion gate.

## Falsifier Disposition

Disposition: `YM-P2-NEG-A no_rank_local_structure`.

Changing the held-out target from area-law slope (`gamma_held`) to
large-loop spatial variance (`sigma2_W33`) did not recover rank-local
structure. The signature-side freeze was successful: v3 re-read v0
signature CSVs with SHA-256 assertions rather than recomputing them.
Gauge-randomized signature and target recomputation also passed.

## Verdict

Named null receipt.

This Phase 2 v3 run does not admit Phase 3 observable-certificate work.
With v0, v1, v2, and v3 all landing `YM-P2-NEG-A`, this cell now has four
consistent named nulls across three signature vocabularies and two target
classes. Per the v3 probe spec, the default next step is PAUSE-and-
synthesize, not automatic v4 probe continuation.

## Public Language Check

- [x] does not say "Sundog has a Yang-Mills result"
- [x] does not say "Sundog proves confinement"
- [x] does not say "Sundog found a mass gap"
- [x] does not imply continuum-limit reasoning
- [x] competitor framing is not used as a foil

## Notes

The v3 aggregate uses the same v0 ensemble configurations and v0-emitted
signature vectors. The target-side change is isolated to `sigma2_W33` bin
assignment. The observed `sigma2_W33` values have modest within-beta spread
and overlapping beta ranges, so the null is not a degenerate-bin artifact.
