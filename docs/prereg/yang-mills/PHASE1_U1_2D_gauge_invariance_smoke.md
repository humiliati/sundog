# Yang-Mills Phase 1 - U1 2D Gauge-Invariance Smoke Manifest

Status: **runner manifest filed 2026-05-29**. This is a pre-run manifest, not a
result. It admits the cheapest Abelian instrumentation runner only after the
implementation emits the runtime manifest fields named below. It admits no
Yang-Mills or non-Abelian claim.

P0 lock: [`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md)
Roadmap: [`../../SUNDOG_V_YANG_MILLS.md`](../../SUNDOG_V_YANG_MILLS.md)
Receipt template:
[`../../yang-mills/RECEIPT_TEMPLATE.md`](../../yang-mills/RECEIPT_TEMPLATE.md)

## Claim Under Test

Inside the `U1_2D`, 16x16, beta 1.0 Abelian instrumentation cell, the v1
plaquette / small-loop signature should be invariant under random lattice gauge
transformations, while raw link vectors should not be accidentally invariant.

This is only an instrumentation and leakage smoke. A pass means the runner's
gauge-randomization plumbing is usable for the next cell. It is not evidence
for non-Abelian Yang-Mills, confinement, mass gap, or any continuum statement.

## Locked Cell

| Field | Registered value |
| --- | --- |
| Cell label | `U1_2D` |
| Lattice size | `16x16` |
| Beta value | `1.0` |
| Boundary condition | `periodic` |
| Action | `Wilson` |
| Generator | U(1) staple-based single-link Metropolis |
| Update mix ratio | 1 Metropolis sweep per combined sweep; no overrelaxation |
| Proposal | link angle proposal `theta -> theta + Uniform(-0.75, 0.75)` radians |
| Master seed | `202605290101` |
| Burn-in | `2000` sweeps minimum |
| Pilot source | `autocorr_pilot/plaquette_series.csv` emitted by this runner |
| Pilot tau_int registration | pass window requires `tau_int(plaquette) <= 16.0` sweeps |
| Thinning interval | `32` sweeps, valid only if pilot `tau_int <= 16.0` |
| Retained configurations | `32` post-burn-in, post-thinning configurations |
| Signature vocabulary | `v1` (`W11`, `W12`, `W13`, `W22` mean/variance) |
| Held-out vocabulary | `v1` recorded for format compatibility; not scored in Phase 1 |
| Gamma-held bin edges | `phase1_no_rank_scoring`; Phase 2 must freeze numeric edges separately |
| Beta-slate revision option | unused; P0 beta slate remains unchanged |

The runtime manifest must write the measured `tau_int(plaquette)` value. If it
exceeds `16.0`, the run is quarantined as autocorrelation underflow and the
gauge-randomization read is not interpreted.

## Gauge-Randomization Read

For each retained configuration, the runner applies:

1. one identity gauge transform sanity check;
2. eight random U(1) site-gauge transformations using deterministic substream
   seeds derived from the master seed, configuration index, and transform index.

U(1) link transform:

```text
U_mu(x) -> g(x) * U_mu(x) * conjugate(g(x + mu))
g(x) = exp(i alpha_x), alpha_x ~ Uniform(0, 2*pi)
```

The primary signature is recomputed after every transform. The raw-link
diagnostic vector is also compared before and after the transform.

## Controls Interpreted In Phase 1

P0's full seven-control battery remains binding for Phase 2. This Phase 1
smoke interprets only the controls that are meaningful before rank-locality
scoring exists:

| Control id | Phase 1 handling |
| --- | --- |
| `CTRL_GAUGE_RAND` | primary read; signature must remain invariant |
| `CTRL_RAW` | diagnostic read; raw links must change under random gauges |
| `CTRL_META` | declared but not scored in Phase 1 |
| `CTRL_RAND` | declared but not scored in Phase 1 |
| `CTRL_RAND_STRAT` | declared but not scored in Phase 1 |
| `CTRL_PERM` | declared but not scored in Phase 1 |
| `CTRL_FINITE_SIZE` | declared but not scored in Phase 1 |

No Phase 2 claim may cite this manifest's unscored controls. A Phase 2 manifest
must freeze numeric gamma-held bin edges and score all seven controls.

## Pass / Quarantine Thresholds

| Quantity | Registered threshold | Branch if missed |
| --- | --- | --- |
| Burn-in sweeps | `>= 2000` | `Z void_run` |
| Runtime wall clock | `<= 10 minutes` | `Z void_run` |
| Pilot `tau_int(plaquette)` | `<= 16.0` | `YM-P1-NEG-X autocorrelation_underflow` |
| Thinning / pilot tau_int | `>= 2.0` | `YM-P1-NEG-X autocorrelation_underflow` |
| Identity transform signature residual | `<= 1e-12` max absolute residual | `YM-P1-NEG-A gauge_leakage` |
| Random gauge signature residual | `<= 1e-12` max absolute residual over all transforms and signature components | `YM-P1-NEG-A gauge_leakage` |
| Raw-link random-gauge residual | median normalized L2 residual `>= 1e-2` and at least 95 percent of transforms `> 1e-6` | `YM-P1-QUAR-A suspicious_raw_invariance` |
| Missing runtime manifest field | none missing | `Z void_run` |

If a numerical residual is exactly on a threshold, it is treated as passing
only for the upper-bound invariance checks and failing for the lower-bound
raw-link non-invariance checks.

## Output Contract

Exact result directory:

```text
results/yang-mills/phase1/U1_2D/2026-05-29_u1_2d_gauge_invariance_smoke_v0/
```

Required files:

| Path under result directory | Role |
| --- | --- |
| `manifest.json` | runtime manifest; must include every locked field, code commit, dirty flag, command line, wall clock, and emitted artifact hashes |
| `autocorr_pilot/plaquette_series.csv` | unthinned post-burn-in plaquette series used to estimate tau_int |
| `configs/u1_links.jsonl` | retained post-thinning configurations |
| `signatures/signature_vectors.csv` | v1 signature before gauge randomization |
| `heldout/heldout_loop_values.csv` | v1 held-out loop values for format compatibility only |
| `heldout/gamma_bin_edges.json` | must record `status = phase1_no_rank_scoring` |
| `gauge_randomization/signature_residuals.csv` | identity and random-gauge signature residuals |
| `gauge_randomization/raw_link_residuals.csv` | raw-link residuals under random gauges |
| `summary.json` | branch assignment inputs and final Phase 1 verdict |
| `hashes.json` | SHA-256 hashes of all emitted artifacts except `hashes.json` itself |

Minimum `manifest.json` fields:

```json
{
  "phase": "phase1",
  "cell": "U1_2D",
  "latticeSize": [16, 16],
  "beta": 1.0,
  "boundary": "periodic",
  "action": "Wilson",
  "generator": "u1_staple_metropolis_v1",
  "updateMix": {"metropolis": 1, "overrelaxation": 0},
  "masterSeed": 202605290101,
  "burnInSweeps": 2000,
  "pilotTauIntPlaquette": null,
  "registeredThinning": 32,
  "retainedConfigurations": 32,
  "signatureVocabularyVersion": "v1",
  "heldOutTargetVocabularyVersion": "v1",
  "gammaHeldBinEdgeStatus": "phase1_no_rank_scoring",
  "controlSetDeclared": [
    "CTRL_META",
    "CTRL_RAW",
    "CTRL_RAND",
    "CTRL_RAND_STRAT",
    "CTRL_PERM",
    "CTRL_GAUGE_RAND",
    "CTRL_FINITE_SIZE"
  ],
  "controlSetScored": ["CTRL_RAW", "CTRL_GAUGE_RAND"],
  "codeCommit": null,
  "gitDirty": null,
  "commandLine": null,
  "wallClockSeconds": null
}
```

The `null` values above are placeholders for runtime-emitted values. If any
remain null in the actual result manifest, the run is void.

## Exact Command Line

The runner implementation must provide this exact command before execution:

```powershell
npm run yang-mills:phase1:u1-2d-smoke
```

The package script must expand to the locked invocation:

```powershell
node scripts/yang-mills-phase1-gauge-smoke.mjs --cell U1_2D --lattice-size 16x16 --beta 1.0 --boundary periodic --action Wilson --generator u1_staple_metropolis_v1 --proposal-half-width 0.75 --seed 202605290101 --burn-in 2000 --pilot-sweeps 512 --thinning 32 --measurements 32 --gauge-transforms 8 --signature-vocab v1 --heldout-vocab v1 --out results/yang-mills/phase1/U1_2D/2026-05-29_u1_2d_gauge_invariance_smoke_v0
```

If the implementation needs a different command line, it must be filed as a
dated manifest amendment before execution. Do not reinterpret output from a
different command under this manifest.

## Branch Table

| Branch | Trigger | Disposition |
| --- | --- | --- |
| `P1-A smoke_pass` | all burn-in, compute, autocorrelation, signature-invariance, and raw-non-invariance thresholds pass | runner plumbing admitted for the next Phase 1 cell |
| `YM-P1-NEG-A gauge_leakage` | primary signature changes under identity or random gauge transform above tolerance | fix signature implementation; no Phase 2 work |
| `YM-P1-QUAR-A suspicious_raw_invariance` | raw-link diagnostic is too invariant under random gauges | investigate implicit gauge fixing or raw-control bug |
| `YM-P1-NEG-X autocorrelation_underflow` | pilot tau or thinning rule fails | no interpretation; file a new manifest rather than silently changing thinning |
| `Z void_run` | missing field, command drift, compute-cap breach, or manifest/hash drift | output cannot be cited as receipt |

## Compute Cap

This is intentionally the cheapest cell. Expected wall clock is under two
minutes on the repo reference machine. The runner must abort before measurement
if a pilot timing estimate predicts the full invocation will exceed ten
minutes. An over-cap run is void, not a reason to trim outputs after the fact.

## Next Allowed Step

After this manifest is filed, the next allowed engineering move is to implement
only the runner and package script necessary to execute the exact command above.
Do not add SU(2), Phase 2 neighbor scoring, smearing, blocking, topological
observables, or 4D hooks inside this runner.

If this smoke passes, the next research artifact is a second Phase 1 manifest
for `SU2_2D`, not a Phase 2 certificate.
