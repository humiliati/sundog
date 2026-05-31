# Yang-Mills Phase 2 v6a Receipt - SU2 3D Finite-T Polyakov NEG-A

- Receipt id: `2026-05-31_SU2_3D_phase2_v6a_finite_t_polyakov_neg_a`
- Cell label: `SU2_3D`
- Phase: 2 v6a
- Date: 2026-05-31
- Author / runner: Codex runner, local Windows workspace
- Code commit: aggregation manifest `def333ee0541ffc9280aec1baaf9879024fe1d7d`
- Git dirty: `true`
- Result directory:
  `results/yang-mills/phase2/SU2_3D/2026-05-31_su2_3d_finite_t_polyakov_v6a/`
- P0 lock:
  [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
- P0 amendment 2:
  [`../../prereg/yang-mills/P0_AMENDMENT_2026-05-31_polyakov.md`](../../prereg/yang-mills/P0_AMENDMENT_2026-05-31_polyakov.md)
- Parent v6 spec:
  [`../../prereg/yang-mills/PHASE2_SU2_3D_finite_t_polyakov_v6.md`](../../prereg/yang-mills/PHASE2_SU2_3D_finite_t_polyakov_v6.md)
- v6 follow-up amendment:
  [`../../prereg/yang-mills/PHASE2_SU2_3D_finite_t_polyakov_v6_AMENDMENT_2026-05-31_pilot_metric.md`](../../prereg/yang-mills/PHASE2_SU2_3D_finite_t_polyakov_v6_AMENDMENT_2026-05-31_pilot_metric.md)

## Registered Domain

- Geometry: `12x12x4`, periodic all directions
- Temporal direction: `mu = 2`
- Pilot beta grid: `{6.0, 6.3, 6.55, 6.8, 7.1, 7.4, 7.7, 8.0}`
- Pilot selector: `order_suscept_abs_mean_P = (12*12) * Var(abs_mean_P)`
  across pilot measurements
- Pilot scan: 800 burn-in sweeps, 96 measurements, thinning 4
- Frozen beta slate after pilot: `{6.3, 6.55, 6.8}`
- Finite-T ensembles: 32 configs per beta, 96 total
- Exact command:

```powershell
npm run yang-mills:phase2:v6:finite-t:polyakov
```

## Claim Under Test

v6a tested whether the amended finite-temperature Polyakov setup supplies a
powered, disjoint held-out target and, if so, whether the unchanged bare
small-loop v1 signature preserves that target's tertile structure in
relative-locality rank space beyond controls.

## Artifacts

| Artifact | Path | Hash / value | Role |
| --- | --- | --- | --- |
| Parent v6 spec | `docs/prereg/yang-mills/PHASE2_SU2_3D_finite_t_polyakov_v6.md` | `09B399514695C31FC83F8CF98C63EE12F1CF0EE36FE12BFA8DEF2D710DC54943` | parent run lock |
| v6a amendment | `docs/prereg/yang-mills/PHASE2_SU2_3D_finite_t_polyakov_v6_AMENDMENT_2026-05-31_pilot_metric.md` | `0B925A1C0BE15853BAB154C20E8072A2FE3FE21097722F3618EF5F2BB71AE0E9` | pilot metric/grid lock |
| Runner source | `scripts/yang-mills-phase2-v6-finite-t-polyakov.mjs` | `811C1330FB4BD759E4716F7C4FF45F8978CB05F1D20D7F5BBC90614F41C36BE7` | pilot + generation + audit runner |
| SU(2) 3D core source | `scripts/lib/yang-mills-su2-3d-core.mjs` | `E9D2659EB5697C06EA446DFFFEFF45FB851A962D1707BB8EE79A4252D5FA1BA9` | asymmetric lattice + temporal Polyakov core |
| Package script | `package.json` | `912D00C6B17CB289FECD85CC1024FE04D5EDF076D48AB1E3D4C44B23831A8B75` | npm entrypoint |
| Aggregation manifest | `aggregation/manifest.json` | `1cfb991ae4d514b20972128c24fa30d1712aae5d1ce16ef970a3f1199f9d505c` | runtime lock |
| Aggregation summary | `aggregation/summary.json` | `f0f155effbb73eb8519eb140cab4960e2db210501dc597bd826e5d0d3e3ae730` | final verdict |
| Pilot beta scan | `pilot_beta_scan/pilot_beta_scan.csv` | `a02bc6478ec0376743f4755b39b1e59ea9bac17124bb50b0871ff854f1c45b25` | beta-slate freeze |
| Frozen beta slate | `pilot_beta_scan/frozen_beta_slate.json` | `69d51d40689bb120469e280cae00c80ee98ae50aef852a5401f5587dde72b5a6` | pilot selection record |
| Admitted target | `aggregation/admitted_target.json` | `71b963b56f6be09ff8826ae49a51549687ef2ab05e2f48b983c450bd87dca6d4` | Stage-1 freeze |
| Target power audit | `aggregation/target_power_audit.csv` | `c2921300efde009ff1f8043624aed7ce099b95a0c956563c69947964346c5843` | Stage-1 gate read |
| Rank-locality scores | `aggregation/rank_locality_scores.csv` | `fde49ec638eb229cf5039c20fdddd4099e35aac9da3d13b0333d0336b4995e9b` | Stage-2 score table |

Full artifact hashes live at
`results/yang-mills/phase2/SU2_3D/2026-05-31_su2_3d_finite_t_polyakov_v6a/hashes.json`.

## Pilot Result

The amended pilot bracketed the `order_suscept_abs_mean_P` peak at beta `6.55`,
freezing the three-beta slate `{6.3, 6.55, 6.8}` before ensemble generation.

| beta | mean `abs_mean_P` | mean `mean_abs_P` | mean `chi_P` | `order_suscept_abs_mean_P` | mean plaquette |
| --- | --- | --- | --- | --- | --- |
| `6.0` | `0.333850391567` | `0.469073257904` | `0.163396269306` | `2.900872393036` | `0.824767553506` |
| `6.3` | `0.358968517551` | `0.476595339182` | `0.154784038781` | `2.481619186294` | `0.833689168897` |
| `6.55` | `0.37251232973` | `0.489316828193` | `0.150910454816` | `3.524815429123` | `0.839925546987` |
| `6.8` | `0.429667127358` | `0.507970150864` | `0.132121602086` | `2.230297167039` | `0.846432718154` |
| `7.1` | `0.460354698493` | `0.525973879214` | `0.120363023254` | `2.543143653185` | `0.853776184547` |
| `7.4` | `0.465524734421` | `0.530482942455` | `0.116265460599` | `3.300998921159` | `0.860263788066` |
| `7.7` | `0.515405655329` | `0.559680944051` | `0.10321386756` | `2.461224508664` | `0.865243781489` |
| `8.0` | `0.515888254828` | `0.557504394802` | `0.099774371479` | `2.296187237697` | `0.8713039857` |

Pilot wall clock: `188.02s`.

## Ensemble Health

All three finite-T ensembles passed health gates.

| beta | tau_int | thinning/tau | fallback fraction | max unitarity residual | wall clock |
| --- | --- | --- | --- | --- | --- |
| `6.3` | `0.953028433722` | `33.577172377763` | `0` | `7.850462293419e-16` | `64.98s` |
| `6.55` | `0.526961489527` | `60.725500128472` | `0` | `7.850462293419e-16` | `62.52s` |
| `6.8` | `0.731291980635` | `43.758171629654` | `0` | `7.850462293419e-16` | `60.29s` |

## Stage 1 - Powered Target Audit

All three Polyakov candidates were powered and disjoint across all three betas.
The signature-blind primary rule selected `abs_mean_P` because it had the
highest mean-over-beta ICC.

| target | mean ICC | mean leakage CV-R2 | admitted |
| --- | --- | --- | --- |
| `abs_mean_P` | `0.964866325746` | `-0.332186032365` | yes |
| `mean_abs_P` | `0.878629088076` | `-0.451505210653` | yes |
| `chi_P` | `0.834534229359` | `-0.452686588241` | yes |

`gamma_held` correctly failed the power self-validation. Polyakov gauge residual
max was `4.440892098500626e-16`; the selected target residual was
`3.3306690738754696e-16`.

## Stage 2 - Relative Locality

Primary target: `abs_mean_P`; k-primary `5`.

| lane / control | purity@5 |
| --- | --- |
| within-beta PRIMARY | `0.304166666667` |
| within-beta CTRL_RAND | `0.329166666667` |
| within-beta CTRL_META | `0.30625` |
| within-beta CTRL_RAW | `0.31875` |
| within-beta CTRL_PERM | `0.31309375` |
| within-beta CTRL_GAUGE_RAND | `0.304166666667` |
| across-beta PRIMARY | `0.345833333333` |
| across-beta CTRL_RAND_STRAT | `0.345833333333` |

Promotion failed the v0 gates:

- within-beta primary purity was below `0.5`;
- primary minus `CTRL_RAND` was `-0.025000000000000133`, not `>= 0.10`;
- primary minus `CTRL_META` was `-0.002083333333333437`, not `>= 0.10`;
- primary minus `CTRL_RAW` was `-0.01458333333333317`, not `>= 0.10`;
- across-beta primary minus `CTRL_RAND_STRAT` was effectively `0`, not `>= 0.05`.

Gauge-randomized signature scoring matched primary to machine precision
(`gaugeRandPurityDiff = 0`), as expected for a gauge-invariant signature.

## Falsifier Disposition

Disposition: `YM-P2-NEG-A no_rank_local_structure`.

This is an informative null, not an underpowered result. v6a supplied the first
powered and disjoint finite-T Polyakov Stage-2 read on this lane, and the
unchanged small-loop v1 signature still did not rank the admitted target beyond
controls. The result strengthens the Phase-2 bounded null from "unpowered target
envelope" to "powered finite-T Polyakov target also non-separating for this
signature on this cell."

## Verdict

`YM-P2-NEG-A no_rank_local_structure`.

The registered positive route to Phase 3 did not open. Further Yang-Mills
target/signature work now requires fresh external scientific motivation or
reviewer feedback; absent that, the disciplined next move is PAUSE / external
review packet update, not another automatic probe.

## Public Language Check

- [x] does not say "Sundog has a Yang-Mills result"
- [x] does not say "Sundog proves confinement"
- [x] does not say "Sundog found a mass gap"
- [x] does not imply continuum-limit reasoning
- [x] competitor framing is not used as a foil
