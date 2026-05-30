# Yang-Mills Phase 2 Bounded-Null Synthesis - SU2 3D

- Synthesis id: `2026-05-29_SU2_3D_phase2_bounded_null_synthesis`
- Cell label: `SU2_3D`
- Phase: 2 synthesis
- Date: 2026-05-29
- Basis receipts:
  - [`2026-05-29_SU2_3D_phase2_no_rank_local_structure.md`](2026-05-29_SU2_3D_phase2_no_rank_local_structure.md)
  - [`2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md`](2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md)
  - [`2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md`](2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md)
  - [`2026-05-29_SU2_3D_phase2_v3_no_rank_local_structure.md`](2026-05-29_SU2_3D_phase2_v3_no_rank_local_structure.md)
- P0 lock:
  [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
- v3 probe fallback:
  [`../specs/2026-05-29_phase2_v3_target_redesign_probe.md`](../specs/2026-05-29_phase2_v3_target_redesign_probe.md)

## Scope

This is a synthesis receipt, not a new run. It summarizes the bounded
Phase 2 null on the registered `SU2_3D`, `12^3`, beta slate
`{2.0, 2.4, 2.8}`, 32-configs-per-beta envelope. It admits no Phase 3,
Phase 4, continuum, confinement, mass-gap, or Clay-problem claim.

## Four-Receipt Matrix

| Probe | Signature | Target | Primary@5 | RAND margin | Verdict |
| --- | --- | --- | --- | --- | --- |
| v0 | v1 bare mean/var | `gamma_held` LS slope | `0.310416666667` | `0.010416666667` | `YM-P2-NEG-A` |
| v1 | v4 APE-smeared mean/var | `gamma_held` LS slope | `0.29375` | `-0.002083333333` | `YM-P2-NEG-A` |
| v2 | v5 bare connected correlator | `gamma_held` LS slope | `0.308333333333` | `0.020833333333` | `YM-P2-NEG-A` |
| v3 | v1 bare mean/var, v0 reread | `sigma2_W33` spatial variance | `0.329166666667` | `0.027083333333` | `YM-P2-NEG-A` |

All four primary scores are near the chance baseline `1/3` and fail both
registered promotion criteria: absolute primary bin-purity@5 `>= 0.5` and
margin `>= 0.10` over random neighbors.

## Synthesis

Inside this finite envelope, the tested small-loop invariant signatures do
not preserve usable within-beta rank-local structure for either tested
held-out target class:

- area-law mean structure, via `gamma_held` tertile labels;
- large-loop spatial inhomogeneity, via `sigma2_W33` tertile labels.

This is not evidence against Yang-Mills, confinement, a mass gap, lattice
gauge theory, or other lattice sizes/couplings/signature classes. It is a
cell-bounded named null: the Sundog small-loop signature lane did not earn a
relative-locality certificate on this registered `SU2_3D` cell.

## PAUSE Disposition

Disposition: **PAUSE-and-synthesize**.

Per the v3 probe spec's pre-stated v4 fallback table, automatic probe-ladder
continuation stops here. `sigma2_W14`, `sigma2_W23`, Polyakov-loop targets,
or smeared signatures against `sigma2_W33` remain admissible only as
future-dated specs with fresh external scientific motivation. They are not
the default next move and should not be filed merely to search for a positive
result.

## Public Language Check

- [x] says "bounded finite-lattice null" rather than "Yang-Mills result"
- [x] does not say "Sundog proves confinement"
- [x] does not say "Sundog found a mass gap"
- [x] does not imply continuum-limit reasoning
- [x] leaves future target classes gated by new motivation, not by p-hunting
