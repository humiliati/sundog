# Yang-Mills Phase 2 v4 Receipt - SU2 3D Powered-Target Underpowered

- Receipt id: `2026-05-31_SU2_3D_phase2_v4_underpowered`
- Cell label: `SU2_3D`
- Phase: 2 v4
- Date: 2026-05-31
- Author / runner: Codex runner, local Windows workspace
- Code commit: aggregation manifest `43f4b34daf284decab21aaae86611dd536b3e26c`
- Git dirty: `true`
- Result directory:
  `results/yang-mills/phase2/SU2_3D/2026-05-31_su2_3d_relative_locality_v4/`
- Input ensemble directories:
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.0_ensemble_v0/`
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.4_ensemble_v0/`
  - `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta2.8_ensemble_v0/`
- P0 lock:
  [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
- Phase manifest:
  [`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v4.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v4.md)
- Probe spec:
  [`../specs/2026-05-31_phase2_v4_powered_target_probe.md`](../specs/2026-05-31_phase2_v4_powered_target_probe.md)

## Registered Domain

- Lattice size: `12x12x12`
- Beta slate: `{2.0, 2.4, 2.8}`
- Input configurations: the 96 retained bare-link configurations from the
  Phase 2 v0 ensembles
- Signature vocabulary: unchanged `v1`, bare 8-dimensional mean/variance
  signature for `{W11, W12, W13, W22}`, re-read from v0 CSVs and SHA-256
  asserted against each v0 `hashes.json`
- Held-out target vocabulary: `v3`, a pre-scoring audit over amendment-free
  re-summaries of `{W14, W23, W33}`
- Candidate pool: `mean_W14`, `mean_W23`, `sigma2_W14`, `sigma2_W23`,
  `ratio_W23_W14`
- Prior targets re-audited, reported but not admissible: `gamma_held`,
  `sigma2_W33`
- Split rule: site-coordinate parity over `12^3 x 3 = 5184`
  position-orientation samples per configuration
- Power gate: `ICC >= 0.50` and tertile agreement `>= 0.50` in all three
  beta values
- Disjointness gate: 5-fold OLS CV R2 `target | v1 signature <= 0.25`,
  using the diagnostic's NumPy `default_rng(0)` fold geometry
- Exact command:

```powershell
npm run yang-mills:phase2:v4:su2-3d:aggregate
```

## Claim Under Test

Before any rank-locality score, v4 tested whether the existing `SU2_3D`,
12^3, beta-slate envelope contains a held-out target that is both powered
under a split-half site-parity audit and disjoint from the v1 signature.
Only such a target was eligible for Stage 2 scoring.

## Artifacts

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| Probe spec | `docs/yang-mills/specs/2026-05-31_phase2_v4_powered_target_probe.md` | filed 2026-05-31 | diagnostic and v5 fallback lock |
| Phase manifest | `docs/prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v4.md` | filed 2026-05-31 | run lock |
| Aggregation runner source | `scripts/yang-mills-phase2-v4-su2-3d-aggregate.mjs` | `30AA06FCA09274C18BBD41E4E7455AC7C5BE3E899BC7D1A9E984AED06751A5BB` | v4 Stage 1 + optional Stage 2 aggregation |
| SU(2) 3D core source | `scripts/lib/yang-mills-su2-3d-core.mjs` | `3897F3503553E1A109295ACC59D20B042A2CC4E08370E048715C792ED66642CF` | gauge transform + signature controls |
| Package script | `package.json` | `D854C56D316EA9313086A330D45482852DF9EBBA9FC5C164B7FE18D862D930DF` | npm entrypoint |
| Aggregation manifest | `aggregation/manifest.json` | `e5f11557932cef732518418d0c286bd0010ccd040c60124016e9c10d3b02b6e9` | runtime lock |
| v0 ensemble sources | `aggregation/v0_ensemble_sources.json` | `a18454f6e2760944e10ae3ec6f6f470bb5fe989052d94406af5cab94f4a1332e` | config + signature hash assertions |
| v4 target re-summaries | `aggregation/v4_target_resummaries.csv` | `b52c277e378213587c5601d8769b4d4bde8aef731c1a61fdfec5a57d67551100` | candidate target values |
| Target power audit | `aggregation/target_power_audit.csv` | `df14ecc6b129c2ec9ffe9b205a5efdb73c8c98b8ad1b8d94d1fd8529b7f0a26e` | Stage 1 gate table |
| Admitted-target freeze | `aggregation/admitted_target.json` | `84a6445d666836c71aedcb3291761ab655214c535db66f3c8308802c81a47313` | records no admitted target |
| Branch inputs | `aggregation/branch_inputs.json` | `d13d3062bc2f6e830051ca0752ef0aa6daa4e52fbef3d766a81d73e372ae2891` | gate numerics |
| Aggregation summary | `aggregation/summary.json` | `8f96c2445f17e30284e830a5d2d9723494f413adc8c6588fe5eb67e125534c46` | final verdict |

Full artifact hashes live at
`results/yang-mills/phase2/SU2_3D/2026-05-31_su2_3d_relative_locality_v4/aggregation/hashes.json`.

## Observed Values

### Health And Integrity

| Quantity | Registered threshold | Observed value | Pass / fail / quarantine |
| --- | --- | --- | --- |
| v0 ensemble health | all three `P1-A ensemble_health_pass` | all three pass; config and signature hashes match | pass |
| Runtime wall clock | `<= 600` seconds | `3.8390122` s | pass |
| `gamma_held` power self-validation | must fail power gate | fails in all three beta values | pass |
| Stage 2 rank-locality scoring | only if a primary target is admitted | not run | correct stop |

### Stage 1 Power / Disjointness Audit

No candidate was both powered and disjoint in all three beta values.

| Target | Gate summary | Disposition |
| --- | --- | --- |
| `mean_W14` | beta 2.0 ICC `0.487065485911` < `0.50`; beta 2.8 leakage CV-R2 `0.576115423876` > `0.25` | rejected |
| `mean_W23` | beta 2.0 ICC `0.228285983404`, agreement `0.4375`; beta 2.8 ICC `0.474388999732`, leakage CV-R2 `0.323858569909` | rejected |
| `sigma2_W14` | ICCs `-0.124947696929`, `0.051799491182`, `0.160592260241` | rejected |
| `sigma2_W23` | ICCs `-0.110131195327`, `0.059390628129`, `0.126995759670` | rejected |
| `ratio_W23_W14` | ICCs `0.045725193612`, `0.314509733247`, `0.289334941474` | rejected |
| `gamma_held` (prior, not admissible) | ICCs `-0.227627567266`, `-0.227458834205`, `0.129928290911` | self-validation passes: target fails power |
| `sigma2_W33` (prior, not admissible) | ICCs `0.291530166337`, `-0.253043132687`, `0.054758297386` | reported; not powered |

`mean_W14` was the closest candidate: it cleared power and disjointness at
beta 2.4, and cleared power at beta 2.8, but it missed the beta 2.0 ICC
gate and failed the beta 2.8 leakage gate. Since admission requires all
three beta values, no primary target was frozen and no nearest-neighbor
artifact was scored.

## Falsifier Disposition

Disposition: `YM-P2-UNDERPOWERED no_powered_target_in_envelope`.

This is quarantine-class, not `YM-P2-NEG-A`: it says the registered
12^3 envelope did not provide a target that could make a Stage 2 read
informative. It does not implicate the v1 signature, because no powered
and disjoint held-out target was admitted.

## Verdict

Underpowered-envelope receipt.

Phase 2 v4 does not admit Phase 3 observable-certificate work, and it does
not upgrade the prior four named nulls into an informative signature-vacuity
claim. If the lane continues, the pre-stated path is the v5 fallback: a new
P0 amendment to a powered regime, such as a weaker beta slate, larger
volume, or a different target class, with the choice justified before code.

## Public Language Check

- [x] does not say "Sundog has a Yang-Mills result"
- [x] does not say "Sundog proves confinement"
- [x] does not say "Sundog found a mass gap"
- [x] does not imply continuum-limit reasoning
- [x] competitor framing is not used as a foil

## Notes

The v4 audit used the same v0 ensemble configurations and v0-emitted
signature vectors. The target-side change was limited to pre-scoring
held-out-loop re-summaries. The exact diagnostic leakage estimator was
matched by using the same 5-fold OLS CV fold geometry as
`scripts/yang-mills-q1q5-controls.py`.
