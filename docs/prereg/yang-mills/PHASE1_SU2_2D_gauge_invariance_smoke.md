# Yang-Mills Phase 1 - SU2 2D Gauge-Invariance Smoke Manifest

Status: **runner manifest filed 2026-05-29**. This is a pre-run manifest, not a
result. It admits the SU(2) 2D harness-cell runner only after the
implementation emits the runtime manifest fields named below. It admits no
non-Abelian certificate or rank-locality claim.

P0 lock: [`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md)
Roadmap: [`../../SUNDOG_V_YANG_MILLS.md`](../../SUNDOG_V_YANG_MILLS.md)
Receipt template:
[`../../yang-mills/RECEIPT_TEMPLATE.md`](../../yang-mills/RECEIPT_TEMPLATE.md)
Prior cell receipt (`U1_2D` smoke pass):
[`../../yang-mills/receipts/2026-05-29_U1_2D_phase1_gauge_invariance_smoke_pos.md`](../../yang-mills/receipts/2026-05-29_U1_2D_phase1_gauge_invariance_smoke_pos.md)

## Claim Under Test

Inside the `SU2_2D`, 16x16, beta 2.0 harness cell, the v1
plaquette / small-loop signature - taken as the position-and-orientation
average of `(1/2) Re Tr U_loop` - should be invariant under random
lattice SU(2) gauge transformations, while raw matrix-entry vectors
should not be accidentally invariant.

This is the second instrumentation smoke after `U1_2D`. A pass means the
SU(2) heatbath / overrelaxation / matrix-trace / Haar-gauge plumbing is
usable for Phase 2 on the SU(2) cells. It is not evidence for confinement,
mass gap, area law, continuum behaviour, or any rank-locality structure.
The non-Abelian primary read remains Phase 2 on `SU2_3D`.

## Locked Cell

| Field | Registered value |
| --- | --- |
| Cell label | `SU2_2D` |
| Lattice size | `16x16` (smaller of the P0 SU2_2D pair; the 24x24 partner is the registered Phase 4 finite-size split) |
| Beta value | `2.0` (middle of the P0 SU2_2D slate `{1.5, 2.0, 2.5}`) |
| Boundary condition | `periodic` |
| Action | `Wilson` |
| Generator | SU(2) Creutz heatbath with Kennedy-Pendleton acceptance + Brown-Woch overrelaxation |
| Update mix ratio | 1 heatbath sweep + 4 overrelaxation sweeps per combined sweep |
| Heatbath sampling | Creutz a-coefficient draw with Kennedy-Pendleton rejection-correction; identity-staple guard returns Haar sample |
| Master seed | `202605290102` |
| Burn-in | `2000` combined sweeps minimum |
| Pilot source | `autocorr_pilot/plaquette_series.csv` emitted by this runner |
| Pilot tau_int registration | pass window requires `tau_int(plaquette) <= 16.0` combined sweeps |
| Thinning interval | `32` combined sweeps, valid only if pilot `tau_int <= 16.0` |
| Retained configurations | `32` post-burn-in, post-thinning configurations |
| Signature vocabulary | `v1` (`W11`, `W12`, `W13`, `W22` mean/variance, `(1/2) Re Tr U_loop`, position-and-orientation averaged) |
| Held-out vocabulary | `v1` recorded for format compatibility; not scored in Phase 1 |
| Gamma-held bin edges | `phase1_no_rank_scoring`; Phase 2 must freeze numeric edges separately |
| Beta-slate revision option | unused; P0 beta slate remains unchanged |
| Lattice-slate revision option | unused; the 24x24 partner is reserved for Phase 4 |

The runtime manifest must write the measured `tau_int(plaquette)` value. If it
exceeds `16.0`, the run is quarantined as autocorrelation underflow and the
gauge-randomization read is not interpreted.

`(1/2) Re Tr U_loop` is the locked SU(2) trace convention from the P0 lock
("Primary Signature Vocabulary"). For W11 the loop is one plaquette; for the
larger loops the trace is taken of the ordered product of links around the
rectangle, then halved and real-projected.

## Gauge-Randomization Read

For each retained configuration, the runner applies:

1. one identity gauge transform sanity check (`g(x) = I` everywhere);
2. eight random SU(2) site-gauge transformations using deterministic substream
   seeds derived from the master seed, configuration index, and transform index.

SU(2) link transform:

```text
U_mu(x) -> g(x) * U_mu(x) * g(x + mu)^dagger
g(x) ~ Haar measure on SU(2)
```

Random SU(2) sampling is the quaternion construction: draw four independent
standard normals `(z0, z1, z2, z3)`, normalize to a unit quaternion
`q = (z0, z1, z2, z3) / |z|`, and form

```text
g = [[ q0 + i q3,  q2 + i q1 ],
     [-q2 + i q1,  q0 - i q3 ]]
```

This is Haar-distributed on SU(2) by isotropy of the four-dimensional standard
normal. The substream seed for `(config m, transform t)` is
`derive(master_seed, "gauge", m, t)` using the same `deriveSubstreamSeed` factory
locked for `U1_2D`.

The primary signature is recomputed after every transform. The raw matrix-entry
vector (flat real array of `(Re a, Im a, Re b, Im b, Re c, Im c, Re d, Im d)`
per link, where the SU(2) matrix is `[[a, b], [c, d]]`) is also compared before
and after the transform.

## Controls Interpreted In Phase 1

P0's full seven-control battery remains binding for Phase 2. This Phase 1
smoke interprets only the controls that are meaningful before rank-locality
scoring exists:

| Control id | Phase 1 handling |
| --- | --- |
| `CTRL_GAUGE_RAND` | primary read; signature must remain invariant under SU(2) Haar gauges |
| `CTRL_RAW` | diagnostic read; raw matrix entries must change under random gauges |
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
| Burn-in combined sweeps | `>= 2000` | `Z void_run` |
| Runtime wall clock | `<= 10 minutes` | `Z void_run` |
| Pilot `tau_int(plaquette)` | `<= 16.0` combined sweeps | `YM-P1-NEG-X autocorrelation_underflow` |
| Thinning / pilot tau_int | `>= 2.0` | `YM-P1-NEG-X autocorrelation_underflow` |
| Identity transform signature residual | `<= 1e-12` max absolute residual | `YM-P1-NEG-A gauge_leakage` |
| Random gauge signature residual | `<= 1e-12` max absolute residual over all transforms and signature components | `YM-P1-NEG-A gauge_leakage` |
| Heatbath staple guard | identity-staple fallback must return a Haar-distributed sample; runtime must log fallback count and it must not exceed `0.1%` of link updates | `YM-P1-QUAR-B heatbath_pathology` |
| Raw matrix-entry random-gauge residual | median normalized L2 residual `>= 1e-2` and at least 95 percent of transforms `> 1e-6` | `YM-P1-QUAR-A suspicious_raw_invariance` |
| Missing runtime manifest field | none missing | `Z void_run` |

The random gauge residual threshold is kept at `1e-12` despite SU(2) matrix
products accumulating more rounding than U(1) phase additions. A bona-fide
gauge-invariance read on a 16x16 lattice with W22 loops (8-link products) has
accumulated error of order `1e-14` per signature entry, several orders below
the threshold. If the threshold fires, that is a real signal worth diagnosing,
not a numerical tightness artifact.

The new `YM-P1-QUAR-B heatbath_pathology` branch is specific to SU(2): the
Kennedy-Pendleton sampler degenerates when the staple sum is near the identity,
which can in principle happen on small lattices at low beta. The runtime must
log the fallback count and it must stay under `0.1%`.

If a numerical residual is exactly on a threshold, it is treated as passing
only for the upper-bound invariance checks and failing for the lower-bound
raw non-invariance checks.

## Output Contract

Exact result directory:

```text
results/yang-mills/phase1/SU2_2D/2026-05-29_su2_2d_gauge_invariance_smoke_v0/
```

Required files:

| Path under result directory | Role |
| --- | --- |
| `manifest.json` | runtime manifest; must include every locked field, code commit, dirty flag, command line, wall clock, fallback counts, and emitted artifact hashes |
| `autocorr_pilot/plaquette_series.csv` | unthinned post-burn-in plaquette series used to estimate tau_int |
| `configs/su2_links.jsonl` | retained post-thinning configurations, one JSON object per line with quaternion-parametrized SU(2) link matrices |
| `signatures/signature_vectors.csv` | v1 signature before gauge randomization |
| `heldout/heldout_loop_values.csv` | v1 held-out loop values for format compatibility only |
| `heldout/gamma_bin_edges.json` | must record `status = phase1_no_rank_scoring` |
| `gauge_randomization/signature_residuals.csv` | identity and random-gauge signature residuals |
| `gauge_randomization/raw_link_residuals.csv` | raw matrix-entry residuals under random gauges |
| `summary.json` | branch assignment inputs and final Phase 1 verdict |
| `hashes.json` | SHA-256 hashes of all emitted artifacts except `hashes.json` itself |

Minimum `manifest.json` fields:

```json
{
  "phase": "phase1",
  "cell": "SU2_2D",
  "latticeSize": [16, 16],
  "beta": 2.0,
  "boundary": "periodic",
  "action": "Wilson",
  "generator": "su2_heatbath_overrelax_v1",
  "updateMix": {"heatbath": 1, "overrelaxation": 4},
  "masterSeed": 202605290102,
  "burnInSweeps": 2000,
  "pilotTauIntPlaquette": null,
  "registeredThinning": 32,
  "retainedConfigurations": 32,
  "signatureVocabularyVersion": "v1",
  "heldOutTargetVocabularyVersion": "v1",
  "gammaHeldBinEdgeStatus": "phase1_no_rank_scoring",
  "heatbathFallbackCount": null,
  "heatbathFallbackFraction": null,
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
npm run yang-mills:phase1:su2-2d-smoke
```

The package script must expand to the locked invocation:

```powershell
node scripts/yang-mills-phase1-su2-gauge-smoke.mjs --cell SU2_2D --lattice-size 16x16 --beta 2.0 --boundary periodic --action Wilson --generator su2_heatbath_overrelax_v1 --overrelax-per-heatbath 4 --seed 202605290102 --burn-in 2000 --pilot-sweeps 512 --thinning 32 --measurements 32 --gauge-transforms 8 --signature-vocab v1 --heldout-vocab v1 --out results/yang-mills/phase1/SU2_2D/2026-05-29_su2_2d_gauge_invariance_smoke_v0
```

The U(1) entry `scripts/yang-mills-phase1-gauge-smoke.mjs` is U(1)-only and
must not be reused for this manifest. A new SU(2) entry
`scripts/yang-mills-phase1-su2-gauge-smoke.mjs` paired with a new core module
`scripts/lib/yang-mills-su2-2d-core.mjs` is the only admitted implementation
path. Shared utilities (CSV writer, hash finalizer, git info, CLI parser, Sokal
tau_int, mulberry32 substream factory) may be factored into a common library
in the same commit, but the U(1) command pinned by
[`PHASE1_U1_2D_gauge_invariance_smoke.md`](PHASE1_U1_2D_gauge_invariance_smoke.md)
must remain bit-for-bit unchanged.

If the implementation needs a different command line, it must be filed as a
dated manifest amendment before execution. Do not reinterpret output from a
different command under this manifest.

## Branch Table

| Branch | Trigger | Disposition |
| --- | --- | --- |
| `P1-A smoke_pass` | all burn-in, compute, autocorrelation, signature-invariance, heatbath-fallback, and raw-non-invariance thresholds pass | runner plumbing admitted for the next Phase 1 cell (`SU2_3D`) |
| `YM-P1-NEG-A gauge_leakage` | primary signature changes under identity or random gauge transform above tolerance | fix signature or gauge-transform implementation; no Phase 2 work |
| `YM-P1-QUAR-A suspicious_raw_invariance` | raw matrix-entry diagnostic is too invariant under random gauges | investigate implicit gauge fixing or raw-control bug |
| `YM-P1-QUAR-B heatbath_pathology` | Kennedy-Pendleton fallback count exceeds 0.1 percent of link updates | re-examine heatbath proposal at this beta, do not promote |
| `YM-P1-NEG-X autocorrelation_underflow` | pilot tau or thinning rule fails | no interpretation; file a new manifest rather than silently changing thinning |
| `Z void_run` | missing field, command drift, compute-cap breach, or manifest/hash drift | output cannot be cited as receipt |

## Compute Cap

Expected wall clock is well under five minutes on the repo reference machine -
SU(2) heatbath + overrelaxation on 16x16 with the 1+4 mix is a few-times more
expensive per link update than U(1) Metropolis, and W22 / W33 traces involve
2x2 complex matrix products instead of phase additions. The runner must abort
before measurement if a pilot timing estimate predicts the full invocation
will exceed ten minutes. An over-cap run is void, not a reason to trim outputs
after the fact.

## Next Allowed Step

After this manifest is filed, the next allowed engineering move is to
implement only the SU(2) core module, entry runner, and package script
necessary to execute the exact command above. Do not add Phase 2 neighbor
scoring, smearing, blocking, topological observables, alternative gauge
groups, or 4D hooks inside this runner.

If this smoke passes, the next research artifact is a third Phase 1 manifest
for `SU2_3D` (the primary cell), not a Phase 2 certificate on `SU2_2D`. 2D
SU(2) is harness instrumentation only and cannot carry a non-Abelian Phase 2
result by itself.

If this smoke fails, the disposition is filed as the named quarantine and a
new dated manifest is drafted under
`docs/prereg/yang-mills/` rather than silently revising this one.
