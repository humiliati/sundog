# P-vs-NP Verification Project Index

This folder holds working artifacts for the `SUNDOG_V_P_V_NP` roadmap.

Main roadmap:

- [`../SUNDOG_V_P_V_NP.md`](../SUNDOG_V_P_V_NP.md)

Lit-pass:

- [`../P_V_NP_LITPASS_MEMO.md`](../P_V_NP_LITPASS_MEMO.md) - 2026-05-28
  prior-art and citation spine for promise-bounded verifier scaffolding.

Phase specs:

- [`PHASE1_TOY_VERIFIER_SPEC.md`](PHASE1_TOY_VERIFIER_SPEC.md) - draft spec
  for the first formal toy verifier, including promise domain, certificate
  schema, baselines, cost accounting, and falsifier mapping.
- [`PHASE1_V0_SLATE.md`](PHASE1_V0_SLATE.md) - frozen first implementation
  slate with split namespaces, calibration insulation, attacker budgets, and
  success thresholds.

Templates:

- [`RECEIPT_TEMPLATE.md`](RECEIPT_TEMPLATE.md)

Result convention:

- Computational outputs should write under `results/pvnp/<phase-or-probe>/`.
- `results/pvnp/` is treated as transient run output. Durable conclusions
  belong in dated receipt notes under `docs/pvnp/receipts/` once reviewed.
- A run without a manifest, verifier-access declaration, baseline comparison,
  and falsifier disposition is not a receipt.

Receipts:

- [`receipts/README.md`](receipts/README.md) - receipt index. No Phase 1
  execution receipt is filed yet.

Current state:

- Roadmap opened: 2026-05-28.
- Lit-pass filed: 2026-05-28.
- Project scaffold opened: 2026-05-28.
- Phase 1 toy-verifier spec drafted: 2026-05-28.
- Phase 1 v0 slate frozen: 2026-05-28.
- No verifier implementation or receipt has been filed.
