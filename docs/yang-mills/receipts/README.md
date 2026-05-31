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
- [`2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md`](2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md)
  - `YM-P2-NEG-A no_rank_local_structure` for the bare connected-
  correlator vocab v5 `SU2_3D`, 12x12x12, beta slate `{2.0, 2.4, 2.8}`
  v2 read on the same v0 ensembles. Bin-edge replay and gauge-
  randomization gates passed, but within-beta bin-purity@5 was
  `0.308333333333`, below the registered `0.5` gate and only
  `0.020833333333` above `CTRL_RAND`. Named null; no Phase 3 artifact
  is admitted from this v2 run.
- [`2026-05-29_SU2_3D_phase2_v3_no_rank_local_structure.md`](2026-05-29_SU2_3D_phase2_v3_no_rank_local_structure.md)
  - `YM-P2-NEG-A no_rank_local_structure` for the target-redesign v3
  read: unchanged v1 bare signature, re-read from v0 CSVs, scored
  against new held-out target vocab v2 `sigma2_W33`. Signature hashes,
  target spread, edge timing, and gauge-randomization gates passed, but
  within-beta bin-purity@5 was `0.329166666667`, below the registered
  `0.5` gate and only `0.027083333333` above `CTRL_RAND`.
- [`2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md`](2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md)
  - PAUSE-and-synthesize receipt for the four Phase 2 `SU2_3D` named
  nulls. Records the bounded cell-local null across three signature
  vocabularies and two target classes; no automatic v4 probe-ladder
  continuation is admitted without fresh external scientific motivation.
  The later v4 reopen therefore required a new dated diagnostic spec.
- [`2026-05-31_SU2_3D_phase2_v4_underpowered.md`](2026-05-31_SU2_3D_phase2_v4_underpowered.md)
  - `YM-P2-UNDERPOWERED no_powered_target_in_envelope` for the reopened
  powered-target v4 audit. No candidate among `mean_W14`, `mean_W23`,
  `sigma2_W14`, `sigma2_W23`, and `ratio_W23_W14` was both powered and
  disjoint in all three beta values; `gamma_held` correctly failed the
  power self-validation. Quarantine-class, not a `NEG-A`; Stage 2 was not
  scored.
- [`2026-05-31_SU2_3D_phase2_v5_polyakov_underpowered.md`](2026-05-31_SU2_3D_phase2_v5_polyakov_underpowered.md)
  - `YM-P2-UNDERPOWERED no_powered_target_in_envelope` for the symmetric
  Polyakov v5 audit. All candidates (`abs_mean_P`, `mean_abs_P`, `chi_P`)
  were disjoint, but none was powered in all three beta values; `gamma_held`
  correctly failed the power self-validation and Polyakov gauge residual max
  was `1.6653345369377348e-16`. Quarantine-class, not a `NEG-A`; Stage 2 was
  not scored. The registered continuation is finite-temperature v6.
- [`2026-05-31_SU2_3D_phase2_v6_pilot_unbracketed.md`](2026-05-31_SU2_3D_phase2_v6_pilot_unbracketed.md)
  - `Z beta_peak_unbracketed` for the finite-temperature Polyakov v6 pilot.
  The locked grid `{6.0, 6.3, 6.55, 6.8, 7.1}` did not bracket the pilot
  `mean_chi_P` peak; it landed at beta `6.0`, the lower boundary. Void pilot;
  no finite-T ensembles, Stage 1 audit, or Stage 2 score were run.
- [`2026-05-31_SU2_3D_phase2_v6a_finite_t_polyakov_neg_a.md`](2026-05-31_SU2_3D_phase2_v6a_finite_t_polyakov_neg_a.md)
  - `YM-P2-NEG-A no_rank_local_structure` for the amended finite-temperature
  Polyakov v6a run. The amended pilot froze beta slate `{6.3, 6.55, 6.8}`;
  `abs_mean_P` was powered and disjoint (mean ICC `0.964866`, mean leakage
  CV-R2 `-0.332186`); Stage 2 scored at/below controls (`PRIMARY@5 0.304167`,
  `CTRL_RAND 0.329167`, across-beta matching `CTRL_RAND_STRAT`).
