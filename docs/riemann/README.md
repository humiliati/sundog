# Riemann Project Index

This folder holds the working artifacts for the `sundog_v_riemann` ledger.

Main ledger:

- [`../SUNDOG_V_RIEMANN.md`](../SUNDOG_V_RIEMANN.md)

Pre-registration:

- [`../prereg/riemann/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../prereg/riemann/P0_DOMAIN_AND_RECEIPT_LOCK.md)

Probe specs:

- [`PROBE_01_ISOTROPY_ZERO_PAIRS_SPEC.md`](PROBE_01_ISOTROPY_ZERO_PAIRS_SPEC.md)

Templates:

- [`RECEIPT_TEMPLATE.md`](RECEIPT_TEMPLATE.md)

Result convention:

- Computational outputs should write under `results/riemann/<phase-or-probe>/`.
- `results/riemann/` is ignored by git; durable conclusions belong in a dated
  receipt note under `docs/riemann/receipts/` once reviewed.
- A run without a manifest, source declaration, and falsifier disposition is not
  a receipt.

Current state:

- Ledger opened: 2026-05-28.
- No zero-data probe has run.
- Front A is design-admitted.
- Front B is horizon only.
