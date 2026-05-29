# Yang-Mills Receipts

Reviewed receipts for the `sundog_v_yang_mills` finite-lattice
certificate program live here.

- P0 lock: [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
- Template: [`../RECEIPT_TEMPLATE.md`](../RECEIPT_TEMPLATE.md)
- Roadmap: [`../../SUNDOG_V_YANG_MILLS.md`](../../SUNDOG_V_YANG_MILLS.md)
- Lit-pass: [`../../YANG_MILLS_LITPASS_MEMO.md`](../../YANG_MILLS_LITPASS_MEMO.md)

Filing convention:

```text
YYYY-MM-DD_<cell>_<phase>_<short-verdict>.md
```

where:

- `<cell>` ∈ `U1_2D`, `SU2_2D`, `SU2_3D`
- `<phase>` ∈ `phase1`, `phase2`, `phase3`, `phase4`
- `<short-verdict>` is a hyphen-joined verdict tag, e.g.
  `gauge_invariance_smoke_pos`, `metadata_only_null`,
  `gauge_leakage_quarantine`, `void_compute_cap`.

Raw outputs live under `results/yang-mills/<phase>/<cell>/<receipt-id>/`
and every receipt here must cite its raw result directory.

No receipt is admitted unless it cites the P0 lock version and fills
the [receipt template](../RECEIPT_TEMPLATE.md).

## Filed Receipts

- [`2026-05-29_U1_2D_phase1_gauge_invariance_smoke_pos.md`](2026-05-29_U1_2D_phase1_gauge_invariance_smoke_pos.md)
  - `P1-A smoke_pass` for the Abelian `U1_2D`, 16x16, beta 1.0
  gauge-randomization smoke. Instrumentation only; no non-Abelian or
  certificate claim.
- [`2026-05-29_SU2_2D_phase1_gauge_invariance_smoke_pos.md`](2026-05-29_SU2_2D_phase1_gauge_invariance_smoke_pos.md)
  - `P1-A smoke_pass` for the non-Abelian `SU2_2D`, 16x16, beta 2.0
  gauge-randomization smoke with Creutz / Kennedy-Pendleton heatbath plus
  Brown-Woch overrelaxation. Instrumentation only; no Phase 2 certificate
  claim.
- [`2026-05-29_SU2_3D_phase1_gauge_invariance_smoke_pos.md`](2026-05-29_SU2_3D_phase1_gauge_invariance_smoke_pos.md)
  - `P1-A smoke_pass` for the non-Abelian `SU2_3D`, 8x8x8, beta 2.4
  gauge-randomization smoke with three-plane signature averaging. Closes
  Phase 1 instrumentation across the registered ladder; no Phase 2
  certificate claim.
- [`2026-05-29_SU2_3D_phase2_no_rank_local_structure.md`](2026-05-29_SU2_3D_phase2_no_rank_local_structure.md)
  - `YM-P2-NEG-A no_rank_local_structure` for the non-Abelian `SU2_3D`,
  12x12x12, beta slate `{2.0, 2.4, 2.8}` relative-locality v0 read.
  All ensemble health gates passed, but within-beta bin-purity@5 was
  `0.310416666667`, below the registered `0.5` promotion gate and only
  `0.010416666667` above `CTRL_RAND`. Named null; no Phase 3 artifact
  is admitted from this v0 run.
- [`2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md`](2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md)
  - `YM-P2-NEG-A no_rank_local_structure` for the APE-smeared vocab v4
  `SU2_3D`, 12x12x12, beta slate `{2.0, 2.4, 2.8}` v1 read on the
  same v0 ensembles. Smearing health and gauge-randomization gates passed,
  but within-beta bin-purity@5 was `0.29375`, below the registered `0.5`
  gate and `0.002083333333` below `CTRL_RAND`. Named null; no Phase 3
  artifact is admitted from this v1 run.
