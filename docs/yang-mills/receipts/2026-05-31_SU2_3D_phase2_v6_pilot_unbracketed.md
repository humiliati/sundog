# Yang-Mills Phase 2 v6 Receipt - SU2 3D Finite-T Polyakov Pilot Unbracketed

- Receipt id: `2026-05-31_SU2_3D_phase2_v6_pilot_unbracketed`
- Cell label: `SU2_3D`
- Phase: 2 v6 pilot
- Date: 2026-05-31
- Author / runner: Codex runner, local Windows workspace
- Code commit: aggregation manifest `def333ee0541ffc9280aec1baaf9879024fe1d7d`
- Git dirty: `true`
- Result directory:
  `results/yang-mills/phase2/SU2_3D/2026-05-31_su2_3d_finite_t_polyakov_v6/`
- P0 lock:
  [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
- P0 amendment 2:
  [`../../prereg/yang-mills/P0_AMENDMENT_2026-05-31_polyakov.md`](../../prereg/yang-mills/P0_AMENDMENT_2026-05-31_polyakov.md)
- Phase manifest:
  [`../../prereg/yang-mills/PHASE2_SU2_3D_finite_t_polyakov_v6.md`](../../prereg/yang-mills/PHASE2_SU2_3D_finite_t_polyakov_v6.md)

## Registered Domain

- Geometry: `12x12x4`, periodic all directions
- Temporal direction: `mu = 2`
- Pilot beta grid: `{6.0, 6.3, 6.55, 6.8, 7.1}`
- Literature anchor: `beta_c = 6.53661(13)` for SU(2) 2+1D at `N_t=4`
  in the `beta = 4/g^2 a` convention
- Pilot scan: 600 burn-in sweeps, 64 measurements, thinning 4, selection
  metric `mean_chi_P`
- Exact command:

```powershell
npm run yang-mills:phase2:v6:finite-t:polyakov
```

## Claim Under Test

Before finite-temperature ensemble generation, v6 tested whether the locked
coarse pilot grid brackets a Polyakov-susceptibility peak so a three-beta
finite-T slate can be frozen without silently choosing an off-grid regime.

## Artifacts

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| Phase manifest | `docs/prereg/yang-mills/PHASE2_SU2_3D_finite_t_polyakov_v6.md` | `09533A72F42EBEF9E210CFC6A1F85F30CBD92B9CBF4C5152B2B28B0F7267CFD3` | run lock |
| Runner source | `scripts/yang-mills-phase2-v6-finite-t-polyakov.mjs` | `B869A707443F7FA5EC5A247FB2451F5EFE514CD78AF82B59584402581751BD8B` | pilot + finite-T generation + audit runner |
| SU(2) 3D core source | `scripts/lib/yang-mills-su2-3d-core.mjs` | `E9D2659EB5697C06EA446DFFFEFF45FB851A962D1707BB8EE79A4252D5FA1BA9` | asymmetric lattice + temporal Polyakov core |
| Package script | `package.json` | `B5B519B9E9213FE794FF93777444579A1202D7CA65ED8CA938B0BE8317CBDB2F` | npm entrypoint |
| Aggregation manifest | `aggregation/manifest.json` | `1d8e397f24452eaebabbc549a373b8d987c4c9faf4f7f0c29d2360202d03d759` | runtime lock |
| Aggregation summary | `aggregation/summary.json` | `458f0c18b89574b03bf4a8d01046c9f271db402db2fc9a82dce9b24331adab44` | final pilot verdict |
| Pilot beta scan | `pilot_beta_scan/pilot_beta_scan.csv` | `47f40373ebbf6e72410aa429f6c1881604478196ae3a3a69eaab974dbce3d063` | pilot readback |
| Frozen beta slate file | `pilot_beta_scan/frozen_beta_slate.json` | `d2d587d556b1c7049ee4b1fc05f537c233a6ad621c41fb63ac64bd1b61006dd6` | records no frozen slate |

Full artifact hashes live at
`results/yang-mills/phase2/SU2_3D/2026-05-31_su2_3d_finite_t_polyakov_v6/hashes.json`.

## Observed Values

No finite-T ensembles were generated, and no Stage 1 or Stage 2 score was run.
The runner stopped before generation because the pilot peak was not bracketed.

| beta | mean `abs_mean_P` | mean `mean_abs_P` | mean `chi_P` | variance `abs_mean_P` |
| --- | --- | --- | --- | --- |
| `6.0` | `0.345288840872` | `0.47453677953` | `0.164830968923` | `0.014876592688` |
| `6.3` | `0.372772129585` | `0.477504178246` | `0.147662708979` | `0.016675731155` |
| `6.55` | `0.385968003677` | `0.489934965354` | `0.146999183288` | `0.019531658795` |
| `6.8` | `0.420575279737` | `0.501230746107` | `0.131773641281` | `0.017154054015` |
| `7.1` | `0.441336406997` | `0.518839571006` | `0.123826229908` | `0.024682753714` |

The selection metric `mean_chi_P` was largest at beta `6.0`, the lower boundary
of the locked pilot grid. Therefore no three-beta slate can be said to bracket
the pilot peak.

## Falsifier Disposition

Disposition: `Z beta_peak_unbracketed`.

This is a void pilot, not a `YM-P2-UNDERPOWERED` result. It says only that the
pilot grid/metric did not admit a frozen finite-T beta slate. It does not test
whether finite-T Polyakov targets are powered or disjoint, and it does not
implicate the v1 signature.

## Verdict

Void pilot receipt.

The next admissible move is a dated v6 follow-up amendment/spec that either
expands the pilot grid or clarifies the finite-T susceptibility metric before
any finite-T ensemble generation. A silent retry with a hand-picked beta slate
is not admitted.

## Public Language Check

- [x] does not say "Sundog has a Yang-Mills result"
- [x] does not say "Sundog proves confinement"
- [x] does not say "Sundog found a mass gap"
- [x] does not imply continuum-limit reasoning
- [x] competitor framing is not used as a foil

## Notes

The runner successfully exercised the pilot machinery and stopped before
expensive generation. The outcome is scientifically useful because it prevents
freezing `{6.0, 6.3, 6.55}` or `{6.3, 6.55, 6.8}` after seeing a boundary peak.
