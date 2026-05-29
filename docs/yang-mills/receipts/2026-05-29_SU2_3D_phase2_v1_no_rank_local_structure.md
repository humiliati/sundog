# Yang-Mills Phase 2 v1 Receipt - SU2 3D APE-Smearing Relative-Locality Null

- Receipt id: `2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure`
- Cell label: `SU2_3D`
- Phase: 2 v1
- Date: 2026-05-29
- Author / runner: Codex runner, local Windows workspace
- Code commit: aggregation manifest `aee9e52198f6f2ac8399de89dd0cc4c967779ece`
- Git dirty: `true`
- Result directory:
  `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v1/`
- Input ensemble directories:
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.0_ensemble_v0/`
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.4_ensemble_v0/`
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.8_ensemble_v0/`
- P0 lock:
  [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
- P0 amendment:
  [`../../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md`](../../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md)
- Phase manifest:
  [`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v1.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v1.md)
- Probe spec:
  [`../specs/2026-05-29_phase2_v1_smearing_probe.md`](../specs/2026-05-29_phase2_v1_smearing_probe.md)

## Registered Domain

- Lattice size: `12x12x12`
- Beta slate: `{2.0, 2.4, 2.8}`
- Input configurations: the 96 retained bare-link configurations from the
  Phase 2 v0 ensembles
- Signature vocabulary: `v4`, APE-smeared `W11`, `W12`, `W13`, `W22`
  mean / variance
- Smearing: APE, `alpha = 0.5`, `N_sm = 10`, synchronous update,
  closest-SU(2) projection after every iteration
- Held-out target vocabulary: unchanged bare-link v1,
  `{W14, W23, W33} -> gamma_held`
- Bin edges: recomputed from v0 `gamma_held` and asserted equal to v0
  per-beta and global edges to `<= 1e-12`
- Primary gate: within-beta k-NN bin-purity@5 `>= 0.5` and margin
  `>= 0.10` over `CTRL_RAND`, `CTRL_META`, and `CTRL_RAW`
- Exact command:

```powershell
npm run yang-mills:phase2:v1:su2-3d:aggregate
```

## Claim Under Test

Inside the registered `SU2_3D`, 12^3, beta-slate finite-lattice envelope,
the v4 APE-smeared small-Wilson-loop signature was tested for
nearest-neighbor rank-local preservation of the unchanged bare-link
`gamma_held` tertile labels beyond all Phase 2 controls.

## Artifacts

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| P0 amendment | `docs/prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md` | filed 2026-05-29 | smearing-class lock |
| Phase manifest | `docs/prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v1.md` | filed 2026-05-29 | run lock |
| Smearing helper source | `scripts/lib/yang-mills-su2-3d-smearing.mjs` | `332677DC7A37D58851529F5821484542E43F81E7F59275C50254429B1400FA9D` | APE smearing |
| Aggregation runner source | `scripts/yang-mills-phase2-v1-su2-3d-aggregate.mjs` | `0E076DD501CDD23447C2C1CB81C29F6019502EE81C54AEE1C8B69375AA1C7C63` | v1 aggregation |
| SU(2) 3D core source | `scripts/lib/yang-mills-su2-3d-core.mjs` | `3897F3503553E1A109295ACC59D20B042A2CC4E08370E048715C792ED66642CF` | unchanged bare core |
| Package script | `package.json` | `C66D9F4FA49CCBFB1D22968E634043386F256446B17EDCE50800C47B8069BCAC` | npm entrypoint |
| Aggregation manifest | `aggregation/manifest.json` | `465a830de4e166a39480cc5e30078d68f5d1e9db9d2cc35c8e3684f2932ac3bb` | runtime lock |
| Smearing health | `aggregation/smearing_health.csv` | `97bed545e6047a878380b668bacd4f6ac1f2f5c370615c37f8fde55956e396fb` | smearing gates |
| Smeared signatures | `aggregation/smeared_signature_vectors.csv` | `db62abddd0fe5662b44d96b49c544f84de500a8c50cc277dae97e4b1e85adffa` | v4 features |
| Bare integrity | `aggregation/bare_signature_integrity.csv` | `11391a6866eb6d05288fc7925dbc96a088553991054516a4cf9348ef1589fc84` | v0 replay check |
| Rank-locality table | `aggregation/rank_locality_scores.csv` | `2ae580d242edab83293e81c9572a60e7b24aaabbdedcafc177cb1929aa39a8a5` | primary + controls |
| v0/v1 comparison | `aggregation/v0_vs_v1_comparison.json` | `2729b2512ad18a45c29a498403f1424ea87280e6a1b93b85a25ebc70dce6e4ea` | diagnostic delta |
| Branch inputs | `aggregation/branch_inputs.json` | `53c4416f918fbe3e0dbff36890e098a7f8296ad4f45249f60ace8abc60c67166` | gate numerics |
| Aggregation summary | `aggregation/summary.json` | `03d83f6fbf92c68b3157b0e5a517c4683cac76143ba7c92f1ec40201005d214a` | final verdict |

Full artifact hashes live at
`results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v1/aggregation/hashes.json`.

## Observed Values

### Health And Integrity

| Quantity | Registered threshold | Observed value | Pass / fail / quarantine |
| --- | --- | --- | --- |
| v0 ensemble health | all three `P1-A ensemble_health_pass` | all three pass; config hashes match | pass |
| v0 bin-edge replay | max abs diff `<= 1e-12` | per-beta `0`, global `0` | pass |
| Bare-signature replay | max abs residual `<= 1e-12` | `4.999889391399392e-13` | pass |
| Smearing det drift | `<= 1e-10` | `6.661338147750939e-16` | pass |
| Smeared unitarity residual | `<= 1e-10` | `9.420554752102651e-16` | pass |
| Smeared orientation spread | `<= 5e-2` | max `0.00030956955364345453` | pass |
| Gauge-randomized smeared signature residual | `<= 1e-12` | `1.4432899320127035e-15` | pass |
| Runtime wall clock | `<= 600` seconds | `25.777605799999996` s | pass |

### Rank-Locality Scores

Primary gate at `k = 5` failed.

| Lane / control | mean bin-purity@5 | discrimination ratio | 95% bootstrap CI | Gate note |
| --- | --- | --- | --- | --- |
| Within-beta primary, v4 smeared | `0.29375` | `0.88125` | `0.252083333333` to `0.33546875` | fails `>= 0.5` |
| Within-beta `CTRL_RAND` | `0.295833333333` | `0.8875` | `0.258333333333` to `0.333333333333` | primary margin `-0.002083333333`, fails `>= 0.10` |
| Within-beta `CTRL_META` | `0.314583333333` | `0.94375` | `0.281197916667` to `0.347916666667` | metadata beats primary |
| Within-beta `CTRL_RAW` | `0.297916666667` | `0.89375` | `0.258333333333` to `0.333333333333` | raw beats primary |
| Within-beta `CTRL_PERM` | `0.313002083333` | `0.93900625` | `0.312064427083` to `0.31397578125` | within chance gate |
| Within-beta `CTRL_GAUGE_RAND` | `0.29375` | `0.88125` | `0.252083333333` to `0.339583333333` | exactly matches primary purity |
| Across-beta primary | `0.358333333333` | `1.075` | `0.314583333333` to `0.4` | fails `CTRL_RAND_STRAT` margin |
| Across-beta `CTRL_RAND_STRAT` | `0.377083333333` | `1.13125` | `0.3375` to `0.422916666667` | beats primary |

v0-to-v1 diagnostic: within-beta primary@5 moved from `0.310416666667`
to `0.29375` (`delta = -0.016666666667`). Across-beta primary@5 moved
from `0.425` to `0.358333333333` (`delta = -0.066666666667`).

## Falsifier Disposition

Disposition: `YM-P2-NEG-A no_rank_local_structure`.

APE smearing did not recover the registered within-beta `gamma_held`
rank-locality signal. The smearing implementation passed the new P0
amendment health gates and the gauge-randomization control, but the primary
v4 graph remained below chance and did not beat random neighbors.

## Verdict

Named null receipt.

This Phase 2 v1 run does not admit Phase 3 observable-certificate work. The
next admitted research artifact is a new dated probe spec under
`docs/yang-mills/specs/` proposing a v2 design change, likely changing the
target or signature class rather than retuning the locked smearing parameters.

## Public Language Check

- [x] does not say "Sundog has a Yang-Mills result"
- [x] does not say "Sundog proves confinement"
- [x] does not say "Sundog found a mass gap"
- [x] does not imply continuum-limit reasoning
- [x] competitor framing is not used as a foil

## Notes

The v1 aggregate uses the same v0 ensemble configurations and unchanged
bare-link held-out target. Smearing is applied only inside the v1 aggregation
runner. The large bare-vs-smeared signature delta (`max = 0.7810538851649217`,
mean absolute delta `0.4125827352061768`) confirms that v1 materially changed
the signature representation; the null is therefore not an accidental replay
of v0.
