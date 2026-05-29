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
