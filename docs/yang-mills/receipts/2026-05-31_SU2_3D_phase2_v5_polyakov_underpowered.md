# Yang-Mills Phase 2 v5 Receipt - SU2 3D Symmetric Polyakov Underpowered

- Receipt id: `2026-05-31_SU2_3D_phase2_v5_polyakov_underpowered`
- Cell label: `SU2_3D`
- Phase: 2 v5
- Date: 2026-05-31
- Author / runner: Codex runner, local Windows workspace
- Code commit: aggregation manifest `5c2812b9c11a8b1b1d7b36e0f08e2279cc72b89a`
- Git dirty: `true`
- Result directory:
  `results/yang-mills/phase2/SU2_3D/2026-05-31_su2_3d_relative_locality_v5/`
- Input ensemble directories:
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.0_ensemble_v0/`
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.4_ensemble_v0/`
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.8_ensemble_v0/`
- P0 lock:
  [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
- P0 amendment 2:
  [`../../prereg/yang-mills/P0_AMENDMENT_2026-05-31_polyakov.md`](../../prereg/yang-mills/P0_AMENDMENT_2026-05-31_polyakov.md)
- Phase manifest:
  [`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v5.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v5.md)
- Probe spec:
  [`../specs/2026-05-31_phase2_v5_polyakov_probe.md`](../specs/2026-05-31_phase2_v5_polyakov_probe.md)

## Registered Domain

- Lattice size: `12x12x12`
- Beta slate: `{2.0, 2.4, 2.8}`
- Input configurations: the 96 retained bare-link configurations from the
  Phase 2 v0 ensembles
- Signature vocabulary: unchanged `v1`, bare 8-dimensional mean/variance
  signature for `{W11, W12, W13, W22}`, re-read from v0 CSVs and SHA-256
  asserted against each v0 `hashes.json`
- Held-out target vocabulary: `v4`, symmetric Polyakov summaries admitted by
  P0 amendment 2
- Candidate pool: `abs_mean_P`, `mean_abs_P`, `chi_P`
- Prior target re-audited, reported but not admissible: `gamma_held`
- Split rule: transverse-site parity over `3 x 12^2 = 432` Polyakov loop
  samples per configuration
- Power gate: `ICC >= 0.50` and tertile agreement `>= 0.50` in all three
  beta values
- Disjointness gate: 5-fold OLS CV R2 `target | v1 signature <= 0.25`,
  using the diagnostic's NumPy `default_rng(0)` fold geometry
- Gauge health gate: each Polyakov summary after random gauge transform must
  match the original to `<= 1e-12`
- Exact command:

```powershell
npm run yang-mills:phase2:v5:su2-3d:aggregate
```

## Claim Under Test

Before any rank-locality score, v5 tested whether the existing symmetric
`SU2_3D`, `12^3`, beta-slate envelope contains a Polyakov target that is both
powered under a split-half transverse-parity audit and disjoint from the v1
signature.

## Artifacts

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| P0 amendment 2 | `docs/prereg/yang-mills/P0_AMENDMENT_2026-05-31_polyakov.md` | `91482A6DF656B30E873B98DCBC8D0973157D1699D314F3123351B4847F96F5AA` | admits Polyakov target class |
| Probe spec | `docs/yang-mills/specs/2026-05-31_phase2_v5_polyakov_probe.md` | `103A17890B3F3F87181BEC951198893391CC89CDDEAC10A32FB0DD939AD2BD1C` | diagnostic and v6 fallback lock |
| Phase manifest | `docs/prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v5.md` | `0E36967561CC4F4E4F49EE83C366174FDAA69EBE82BFE0CEECBBB60F460697F4` | run lock |
| Aggregation runner source | `scripts/yang-mills-phase2-v5-su2-3d-aggregate.mjs` | `1E169CDD1EE9943BB522F604B9852287FEC7D85455CEB458D64B7D54AB9EAA44` | v5 Stage 1 + optional Stage 2 aggregation |
| SU(2) 3D core source | `scripts/lib/yang-mills-su2-3d-core.mjs` | `3A43C97FAB35E788A2492F5CE7CC12277734E304DEFE2F5E507CBD8D3D727CE1` | Polyakov loop helper + gauge transform controls |
| Package script | `package.json` | `9DE6FB5AD26B15F838BB51592FB31A8BB7D4090B20E9F914D760C85906AC66CD` | npm entrypoint |
| Aggregation manifest | `aggregation/manifest.json` | `32ff0a764b33782a66dc3fdedb482fa9f9568dcb5d6850882c06d2adefa97f16` | runtime lock |
| v0 ensemble sources | `aggregation/v0_ensemble_sources.json` | `a18454f6e2760944e10ae3ec6f6f470bb5fe989052d94406af5cab94f4a1332e` | config + signature hash assertions |
| Polyakov target re-summaries | `aggregation/v5_polyakov_target_resummaries.csv` | `7d9157e264ca3cc67587e6e67f52e618dd525ba00c3179d30213a68aa484f7fd` | candidate target values |
| Target power audit | `aggregation/target_power_audit.csv` | `e268e3f33f984c3fa3d5c781d402ce1468390b9878152bd6e9a8ef5c299ac6ec` | Stage 1 gate table |
| Polyakov gauge health | `aggregation/polyakov_gauge_health.json` | `6d90c3ff9264913ed64ee221fed0c3fed9f597685f99242fa319fecf75e17dae` | target-side gauge-invariance audit |
| Admitted-target freeze | `aggregation/admitted_target.json` | `fa802b4f373f017442750488526f7d0518485e14256afbccb6e4fcab7e054d3d` | records no admitted target |
| Branch inputs | `aggregation/branch_inputs.json` | `b81c1d2634619acfbdca4977b68b73727f8545a12976ec2735cecec52e369313` | gate numerics |
| Aggregation summary | `aggregation/summary.json` | `cf084a150ba4f1a6d3af2c96cc34f3d0d2e67fae94b6dce4a66f8505d555021e` | final verdict |

Full artifact hashes live at
`results/yang-mills/phase2/SU2_3D/2026-05-31_su2_3d_relative_locality_v5/aggregation/hashes.json`.

## Observed Values

### Health And Integrity

| Quantity | Registered threshold | Observed value | Pass / fail / quarantine |
| --- | --- | --- | --- |
| v0 ensemble health | all three `P1-A ensemble_health_pass` | all three pass; config and signature hashes match | pass |
| Runtime wall clock | `<= 600` seconds | `2.3039848` s | pass |
| `gamma_held` power self-validation | must fail power gate | fails in all three beta values | pass |
| Polyakov gauge residual max | `<= 1e-12` | `1.6653345369377348e-16` | pass |
| Stage 2 rank-locality scoring | only if a primary target is admitted | not run | correct stop |

### Stage 1 Power / Disjointness Audit

No Polyakov candidate was both powered and disjoint in all three beta values.
All three candidates were disjoint in all three beta values, but none was
powered.

| Target | Gate summary | Disposition |
| --- | --- | --- |
| `abs_mean_P` | ICCs `0.468596413202`, `0.07384761975`, `-0.171214452016`; agreements `0.5`, `0.3125`, `0.21875`; all leakage CV-R2 values `<= 0.25` | rejected |
| `mean_abs_P` | ICCs `-0.165379384995`, `-0.166122890995`, `0.056170255782`; agreements `0.3125`, `0.21875`, `0.375`; all leakage CV-R2 values `<= 0.25` | rejected |
| `chi_P` | ICCs `-0.107521537987`, `-0.153139654337`, `-0.058599628777`; agreements `0.375`, `0.25`, `0.34375`; all leakage CV-R2 values `<= 0.25` | rejected |
| `gamma_held` (prior, not admissible) | ICCs `-0.227627567266`, `-0.227458834205`, `0.129928290911` | self-validation passes: target fails power |

`abs_mean_P` was the closest candidate: at beta 2.0 it reached agreement
`0.5` and ICC `0.468596413202`, just below the `0.50` ICC gate. It was still
far below the power gate at beta 2.4 and beta 2.8. Since admission requires
all three beta values, no primary target was frozen and no nearest-neighbor
artifact was scored.

## Falsifier Disposition

Disposition: `YM-P2-UNDERPOWERED no_powered_target_in_envelope`.

This is quarantine-class, not `YM-P2-NEG-A`: it says the symmetric `12^3`
Polyakov target class did not supply a target that could make a Stage 2 read
informative. It does not implicate the v1 signature, because no powered and
disjoint held-out target was admitted.

## Verdict

Underpowered-envelope receipt.

Phase 2 v5 does not admit Phase 3 observable-certificate work, and it does not
upgrade the prior named nulls into an informative signature-vacuity claim. Per
the pre-registered v5 probe and v6 binding spec, a continuation now routes to
the finite-temperature `12^2 x 4` Polyakov v6 build rather than another
symmetric-cell retry.

## Public Language Check

- [x] does not say "Sundog has a Yang-Mills result"
- [x] does not say "Sundog proves confinement"
- [x] does not say "Sundog found a mass gap"
- [x] does not imply continuum-limit reasoning
- [x] competitor framing is not used as a foil

## Notes

The v5 audit used the same v0 ensemble configurations and v0-emitted signature
vectors. The target-side change was limited to symmetric Polyakov loop
re-summaries admitted by P0 amendment 2. The exact diagnostic leakage estimator
was matched by using the same 5-fold OLS CV fold geometry as
`scripts/yang-mills-q1q5-controls.py`.
