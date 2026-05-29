# Yang-Mills Phase 1 - SU2 3D Gauge-Invariance Smoke Manifest

Status: **runner manifest filed 2026-05-29**. This is a pre-run manifest,
not a result. It admits the SU(2) 3D primary-cell runner only after the
implementation emits the runtime manifest fields named below. It admits no
Phase 2 certificate or rank-locality claim.

P0 lock: [`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md)
Roadmap: [`../../SUNDOG_V_YANG_MILLS.md`](../../SUNDOG_V_YANG_MILLS.md)
Receipt template:
[`../../yang-mills/RECEIPT_TEMPLATE.md`](../../yang-mills/RECEIPT_TEMPLATE.md)
Prior cell receipts:
[`../../yang-mills/receipts/2026-05-29_U1_2D_phase1_gauge_invariance_smoke_pos.md`](../../yang-mills/receipts/2026-05-29_U1_2D_phase1_gauge_invariance_smoke_pos.md)
and
[`../../yang-mills/receipts/2026-05-29_SU2_2D_phase1_gauge_invariance_smoke_pos.md`](../../yang-mills/receipts/2026-05-29_SU2_2D_phase1_gauge_invariance_smoke_pos.md).

## Claim Under Test

Inside the `SU2_3D`, 8x8x8, beta 2.4 primary cell, the v1 small-Wilson-loop
signature - taken as the position-and-orientation average of
`(1/2) Re Tr U_loop` across all positions AND the three plane orientations
{xy, xz, yz} - should be invariant under random local SU(2) gauge
transformations, while raw matrix-entry vectors should not be accidentally
invariant.

This is the third and final Phase 1 instrumentation smoke. A pass means
the SU(2) heatbath / overrelaxation / matrix-trace / Haar-gauge plumbing
extends cleanly to the 3D plaquette geometry (four plaquettes per link
instead of two) and the three-plane-orientation signature averaging. It is
not evidence for confinement, mass gap, area law, continuum behaviour, or
any rank-locality structure. The non-Abelian primary read remains a future
Phase 2 manifest on the same cell at the **12³** partner lattice with
γ_held bin edges frozen before scoring; this Phase 1 manifest scores no
held-out target.

## Locked Cell

| Field | Registered value |
| --- | --- |
| Cell label | `SU2_3D` |
| Lattice size | `8x8x8` (smaller of the P0 SU2_3D pair; the 12x12x12 partner is the registered Phase 4 finite-size split partner, also the registered Phase 2/3 primary lattice) |
| Beta value | `2.4` (middle of the P0 SU2_3D slate `{2.0, 2.4, 2.8}`; sits in the standard 3D SU(2) confinement-to-perturbative crossover region) |
| Boundary condition | `periodic` in all three directions |
| Action | `Wilson` |
| Generator | SU(2) Creutz heatbath with Kennedy-Pendleton acceptance + Brown-Woch overrelaxation, **dimension-3 staple sum** (four plaquettes per link instead of two) |
| Update mix ratio | 1 heatbath sweep + 4 overrelaxation sweeps per combined sweep (same as SU2_2D) |
| Heatbath sampling | Creutz a-coefficient draw with Kennedy-Pendleton rejection-correction; identity-staple guard returns Haar sample (same algorithm; only the staple sum input differs from SU2_2D) |
| Master seed | `202605290103` |
| Burn-in | `2000` combined sweeps minimum |
| Pilot source | `autocorr_pilot/plaquette_series.csv` emitted by this runner |
| Pilot tau_int registration | pass window requires `tau_int(plaquette) <= 16.0` combined sweeps |
| Thinning interval | `32` combined sweeps, valid only if pilot `tau_int <= 16.0` |
| Retained configurations | `32` post-burn-in, post-thinning configurations |
| Signature vocabulary | `v1` (`W11`, `W12`, `W13`, `W22` mean/variance, `(1/2) Re Tr U_loop`, averaged over positions AND over the three plane orientations `{xy, xz, yz}`) |
| Held-out vocabulary | `v1` recorded for format compatibility; not scored in Phase 1 |
| Gamma-held bin edges | `phase1_no_rank_scoring`; Phase 2 must freeze numeric edges separately |
| Beta-slate revision option | unused; P0 beta slate remains unchanged |
| Lattice-slate revision option | unused; the 12³ partner is reserved for Phase 2/3 primary + Phase 4 split |

The runtime manifest must write the measured `tau_int(plaquette)` value. If
it exceeds `16.0`, the run is quarantined as autocorrelation underflow and
the gauge-randomization read is not interpreted.

The mean plaquette must average over the **three** plane orientations
{xy, xz, yz}. In equilibrium with isotropic periodic boundaries the three
orientations are equivalent in expectation; the per-orientation means and
variances must be reported separately in the runtime summary, and a
runtime quarantine `YM-P1-QUAR-C orientation_anisotropy` fires if any pair
of per-orientation mean plaquettes differs by more than `5e-2` relative.

## Gauge-Randomization Read

For each retained configuration, the runner applies:

1. one identity gauge transform sanity check (`g(x, y, z) = I` everywhere);
2. eight random SU(2) site-gauge transformations using deterministic
   substream seeds derived from the master seed, configuration index, and
   transform index.

SU(2) link transform (unchanged from 2D except the site grid is 3D):

```text
U_mu(x, y, z) -> g(x, y, z) * U_mu(x, y, z) * g(x + mu_hat)^dagger
g(x, y, z) ~ Haar measure on SU(2)
```

Random SU(2) sampling is the same quaternion construction as in SU2_2D:
draw four independent standard normals `(z0, z1, z2, z3)`, normalize to a
unit quaternion `q = (z0, z1, z2, z3) / |z|`, and form

```text
g = [[ q0 + i q3,  q2 + i q1 ],
     [-q2 + i q1,  q0 - i q3 ]]
```

The substream seed for `(config m, transform t)` is
`derive(master_seed, "gauge", m, t)` using the same `deriveSubstreamSeed`
factory locked for `U1_2D` and `SU2_2D`.

The primary signature is recomputed after every transform. The raw
matrix-entry vector (flat real array of `(Re a, Im a, Re b, Im b, Re c, Im
c, Re d, Im d)` per link, where the SU(2) matrix is `[[a, b], [c, d]]`)
is also compared before and after the transform.

## Controls Interpreted In Phase 1

P0's full seven-control battery remains binding for Phase 2. This Phase 1
smoke interprets only the controls that are meaningful before
rank-locality scoring exists:

| Control id | Phase 1 handling |
| --- | --- |
| `CTRL_GAUGE_RAND` | primary read; signature must remain invariant under SU(2) Haar gauges |
| `CTRL_RAW` | diagnostic read; raw matrix entries must change under random gauges |
| `CTRL_META` | declared but not scored in Phase 1 |
| `CTRL_RAND` | declared but not scored in Phase 1 |
| `CTRL_RAND_STRAT` | declared but not scored in Phase 1 |
| `CTRL_PERM` | declared but not scored in Phase 1 |
| `CTRL_FINITE_SIZE` | declared but not scored in Phase 1 |

No Phase 2 claim may cite this manifest's unscored controls. A Phase 2
manifest must freeze numeric gamma_held bin edges, run at 12³, score all
seven controls, and pair against the 8³ partner for `CTRL_FINITE_SIZE`.

## Pass / Quarantine Thresholds

| Quantity | Registered threshold | Branch if missed |
| --- | --- | --- |
| Burn-in combined sweeps | `>= 2000` | `Z void_run` |
| Runtime wall clock | `<= 10 minutes` | `Z void_run` |
| Pilot `tau_int(plaquette)` | `<= 16.0` combined sweeps | `YM-P1-NEG-X autocorrelation_underflow` |
| Thinning / pilot tau_int | `>= 2.0` | `YM-P1-NEG-X autocorrelation_underflow` |
| Identity transform signature residual | `<= 1e-12` max absolute residual | `YM-P1-NEG-A gauge_leakage` |
| Random gauge signature residual | `<= 1e-12` max absolute residual over all transforms and signature components | `YM-P1-NEG-A gauge_leakage` |
| Heatbath staple guard | identity-staple fallback fraction `<= 0.1%` of link updates | `YM-P1-QUAR-B heatbath_pathology` |
| Raw matrix-entry random-gauge residual | median normalized L2 residual `>= 1e-2` and at least 95 percent of transforms `> 1e-6` | `YM-P1-QUAR-A suspicious_raw_invariance` |
| Link unitarity residual | max `||U U^dagger - I||_F <= 1e-10` over all post-update links | `YM-P1-QUAR-D unitarity_drift` |
| Per-orientation mean-plaquette relative spread | `<= 5e-2` between any two of `{xy, xz, yz}` | `YM-P1-QUAR-C orientation_anisotropy` |
| Missing runtime manifest field | none missing | `Z void_run` |

The 1e-12 random-gauge residual threshold is kept from SU2_2D. On 8³ with
W22 (8-link products) accumulated matrix-product error is of order `1e-15`
per signature entry, ~3 orders below the threshold; W33 (12-link product)
pushes that to ~`1.2e-15`, still well below.

The new `YM-P1-QUAR-C orientation_anisotropy` branch is specific to 3D:
in equilibrium with isotropic Wilson action and isotropic periodic
boundaries, the three plane-orientation mean plaquettes must agree in
expectation. A 5% relative-spread floor between any pair is generous (in
practice the spread should be of order `1/sqrt(N_planes · N_sites)` ≈ 1.4%
on 8³, fluctuating); a violation would point at a per-orientation bug in
the staple sum or signature loop iteration.

The new `YM-P1-QUAR-D unitarity_drift` branch is an explicit promotion of
the SU2_2D receipt's implicit link-unitarity check to a named gate. On
SU2_2D the observed max `||U U^dagger - I||_F` was `6.66e-16`; the 1e-10
threshold is six orders of magnitude looser, gating only against a real
algorithmic drift, not floating-point dust.

If a numerical residual is exactly on a threshold, it is treated as
passing only for the upper-bound invariance checks and failing for the
lower-bound raw non-invariance checks.

## Output Contract

Exact result directory:

```text
results/yang-mills/phase1/SU2_3D/2026-05-29_su2_3d_gauge_invariance_smoke_v0/
```

Required files:

| Path under result directory | Role |
| --- | --- |
| `manifest.json` | runtime manifest; must include every locked field, code commit, dirty flag, command line, wall clock, fallback counts, per-orientation mean plaquettes, max unitarity residual, and emitted artifact hashes |
| `autocorr_pilot/plaquette_series.csv` | unthinned post-burn-in plaquette series (averaged over the three plane orientations) used to estimate tau_int |
| `autocorr_pilot/plaquette_by_orientation.csv` | per-orientation mean plaquette series (three columns `xy`, `xz`, `yz`) for the anisotropy gate |
| `configs/su2_links.jsonl` | retained post-thinning configurations, one JSON object per line with quaternion-parametrized SU(2) link matrices over `(mu, x, y, z)` indexing |
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
  "cell": "SU2_3D",
  "latticeSize": [8, 8, 8],
  "beta": 2.4,
  "boundary": "periodic",
  "action": "Wilson",
  "generator": "su2_heatbath_overrelax_v1",
  "updateMix": {"heatbath": 1, "overrelaxation": 4},
  "masterSeed": 202605290103,
  "burnInSweeps": 2000,
  "pilotTauIntPlaquette": null,
  "registeredThinning": 32,
  "retainedConfigurations": 32,
  "signatureVocabularyVersion": "v1",
  "heldOutTargetVocabularyVersion": "v1",
  "gammaHeldBinEdgeStatus": "phase1_no_rank_scoring",
  "heatbathFallbackCount": null,
  "heatbathFallbackFraction": null,
  "linkUnitarityMaxFrobenius": null,
  "perOrientationMeanPlaquette": {"xy": null, "xz": null, "yz": null},
  "perOrientationRelativeSpread": null,
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

The `null` values above are placeholders for runtime-emitted values. If
any remain null in the actual result manifest, the run is void.

## Exact Command Line

The runner implementation must provide this exact command before
execution:

```powershell
npm run yang-mills:phase1:su2-3d-smoke
```

The package script must expand to the locked invocation:

```powershell
node scripts/yang-mills-phase1-su2-3d-gauge-smoke.mjs --cell SU2_3D --lattice-size 8x8x8 --beta 2.4 --boundary periodic --action Wilson --generator su2_heatbath_overrelax_v1 --overrelax-per-heatbath 4 --seed 202605290103 --burn-in 2000 --pilot-sweeps 512 --thinning 32 --measurements 32 --gauge-transforms 8 --signature-vocab v1 --heldout-vocab v1 --out results/yang-mills/phase1/SU2_3D/2026-05-29_su2_3d_gauge_invariance_smoke_v0
```

The existing U(1) and SU(2) 2D entries must remain bit-for-bit unchanged:

- `scripts/yang-mills-phase1-gauge-smoke.mjs` is U(1)-only.
- `scripts/yang-mills-phase1-su2-gauge-smoke.mjs` is SU(2)-2D-only.

A new SU(2) 3D entry `scripts/yang-mills-phase1-su2-3d-gauge-smoke.mjs`
paired with a new core module `scripts/lib/yang-mills-su2-3d-core.mjs` is
the only admitted implementation path. The 3D core may share helper
utilities (CSV writer, hash finalizer, git info, CLI parser, Sokal
tau_int, mulberry32 substream factory, SU(2) matrix primitives,
quaternion Haar sampler) with the 2D core via copy or via a
common-utility library, but the SU2_2D npm script and its existing
runner output must remain bit-for-bit unchanged.

If the implementation needs a different command line, it must be filed as
a dated manifest amendment before execution. Do not reinterpret output
from a different command under this manifest.

## Branch Table

| Branch | Trigger | Disposition |
| --- | --- | --- |
| `P1-A smoke_pass` | all burn-in, compute, autocorrelation, signature-invariance, heatbath-fallback, raw-non-invariance, link-unitarity, and orientation-isotropy thresholds pass | Phase 1 instrumentation closed across the full ladder; the next admitted research artifact is a Phase 2 manifest on `SU2_3D` at the 12³ partner with γ_held bin edges frozen before scoring |
| `YM-P1-NEG-A gauge_leakage` | primary signature changes under identity or random gauge transform above tolerance | fix signature or gauge-transform implementation; no Phase 2 work |
| `YM-P1-QUAR-A suspicious_raw_invariance` | raw matrix-entry diagnostic is too invariant under random gauges | investigate implicit gauge fixing or raw-control bug |
| `YM-P1-QUAR-B heatbath_pathology` | Kennedy-Pendleton fallback count exceeds 0.1 percent of link updates | re-examine heatbath proposal at this beta, do not promote |
| `YM-P1-QUAR-C orientation_anisotropy` | per-orientation mean plaquettes disagree by more than 5 percent relative | investigate staple sum or signature loop iteration; do not interpret invariance read |
| `YM-P1-QUAR-D unitarity_drift` | any post-update link drifts from SU(2) by more than `1e-10` in Frobenius norm | re-examine heatbath / overrelax reprojection; do not interpret invariance read |
| `YM-P1-NEG-X autocorrelation_underflow` | pilot tau or thinning rule fails | no interpretation; file a new manifest rather than silently changing thinning |
| `Z void_run` | missing field, command drift, compute-cap breach, or manifest/hash drift | output cannot be cited as receipt |

## Compute Cap

Expected wall clock based on SU2_2D's `7.18 s` for 3536 combined sweeps
on 16² (≈ 1.28 M sub-sweep link updates): the 8³ lattice has 1536 links
× 5 sub-sweeps = 7680 per combined sweep, and each link touches four
plaquettes instead of two. Naive extrapolation gives ≈ 90-150 seconds
total for the same combined-sweep count. Signature passes (8³ × 3 plane
orientations × loop set) add ≈ 5-10 seconds. Gauge-randomization passes
add ≈ 10-20 seconds. Total estimate ≈ 2-4 minutes, well inside the
ten-minute cap.

The runner must still abort before measurement if a pilot timing estimate
predicts the full invocation will exceed ten minutes. An over-cap run is
void, not a reason to trim outputs after the fact.

## Next Allowed Step

After this manifest is filed, the next allowed engineering move is to
implement only the SU(2) 3D core module, entry runner, and package
script necessary to execute the exact command above. Do not add Phase 2
neighbor scoring, smearing, blocking, topological observables,
alternative gauge groups, or 4D hooks inside this runner.

If this smoke passes, Phase 1 instrumentation is closed across the full
P0 cell ladder (U1_2D, SU2_2D, SU2_3D). The next research artifact is a
**Phase 2 manifest** under `docs/prereg/yang-mills/`:

```text
PHASE2_SU2_3D_relative_locality_v0.md
```

It must run on the **12³** partner lattice (Phase 1 8³ instrumentation is
not a Phase 2 result), freeze numeric `γ_held` bin edges before any
nearest-neighbor scoring, and score all seven entries of the P0 leakage
controls battery on the same frozen neighbor graph and held-out labels.
That manifest is out of scope for this Phase 1 smoke.

If this smoke fails, the disposition is filed as the named quarantine and
a new dated manifest is drafted under `docs/prereg/yang-mills/` rather
than silently revising this one.
