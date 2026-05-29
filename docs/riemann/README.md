# Riemann Project Index

This folder holds the working artifacts for the `sundog_v_riemann` ledger.

Main ledger:

- [`../SUNDOG_V_RIEMANN.md`](../SUNDOG_V_RIEMANN.md)

Synthesis (read this for the conclusion):

- [`RIEMANN_BOUNDED_NULL_SYNTHESIS.md`](RIEMANN_BOUNDED_NULL_SYNTHESIS.md) -
  three lanes, three identified vacuity causes, no structural-zero edge; the
  bounded null is the result. Public surface gated on external review.

Lit-pass (2026-05-28):

- [`../RIEMANN_LITPASS_MEMO.md`](../RIEMANN_LITPASS_MEMO.md) - prior-art
  reference grounding the four-probe ranking; gap claims time-stamped.

Pre-registration:

- [`../prereg/riemann/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../prereg/riemann/P0_DOMAIN_AND_RECEIPT_LOCK.md)

Bridge scoping:

- [`REPRESENTATION_BRIDGE_NOTES.md`](REPRESENTATION_BRIDGE_NOTES.md) -
  resolves the v0.3h-D3 vs Riemann-Z₂ admissibility question for Probe 01.
- [`NONLINEAR_PAIR_CORRELATION_BRIDGE_NOTES.md`](NONLINEAR_PAIR_CORRELATION_BRIDGE_NOTES.md)
  - nonlinear sibling; admits S₂ gap-pair swap (as a reversibility test, NEG-A
  expected), quarantines C3 triple (relabeling), downgrades residual bins.

Front-A reading:

- [`FRONT_A_FUNCTIONAL_EQUATION_READING.md`](FRONT_A_FUNCTIONAL_EQUATION_READING.md)
  - draft reading note on functional-equation reflection as a receipt scaffold
  for smoothed explicit formulae.

Cell sets:

- [`RIEMANN_C1_CELLSET_V0.md`](RIEMANN_C1_CELLSET_V0.md)
  - v0 explicit-formula parity scaffold; unreviewed and unrun.

Scoping notes:

- [`NONLINEAR_PAIR_CORRELATION_LANE.md`](NONLINEAR_PAIR_CORRELATION_LANE.md)
  - nonlinear zero-statistics lane opened by the C1 linearity audit.

Probe specs:

- [`PROBE_01_ISOTROPY_ZERO_PAIRS_SPEC.md`](PROBE_01_ISOTROPY_ZERO_PAIRS_SPEC.md)
  - admitted bridge is Path (i) Z₂-descent per the bridge notes above.
- [`PROBE_05_NONLINEAR_ZERO_STATISTICS_SPEC.md`](PROBE_05_NONLINEAR_ZERO_STATISTICS_SPEC.md)
  - nonlinear gap-pair reversibility test; S2 hook only, NEG-A expected.

Templates:

- [`RECEIPT_TEMPLATE.md`](RECEIPT_TEMPLATE.md)

Result convention:

- Computational outputs should write under `results/riemann/<phase-or-probe>/`.
- `results/riemann/` is ignored by git; durable conclusions belong in a dated
  receipt note under `docs/riemann/receipts/` once reviewed.
- A run without a manifest, source declaration, and falsifier disposition is not
  a receipt.

Receipts:

- [`receipts/2026-05-28_probe01_pathi_parity_decomposition.md`](receipts/2026-05-28_probe01_pathi_parity_decomposition.md)
  - first Probe 01 v1 receipt under Path (i), using Odlyzko `zeros1`, `N=5000`.
- [`receipts/2026-05-28_probe05_reversibility_null.md`](receipts/2026-05-28_probe05_reversibility_null.md)
  - Probe 05 S2 gap-pair reversibility null (`D=-0.0064`, inside floor `0.0424`);
  `R-NL-NEG-A` as predicted.

Current state:

- Ledger opened: 2026-05-28.
- Lit-pass filed: 2026-05-28.
- Bridge admission (Path (i) Z₂-descent) selected: 2026-05-28.
- Probe 01 v1 Path (i) receipt filed: 2026-05-28.
- Front-A functional-equation reading note drafted: 2026-05-28.
- Riemann C1 cell-set v0 filed: 2026-05-28.
- Nonlinear pair-correlation lane scoped: 2026-05-28.
- Nonlinear bridge notes filed (S₂ admitted / C3 quarantined / bins downgraded): 2026-05-28.
- Probe 05 v0 nonlinear reversibility-test spec filed: 2026-05-28.
- Probe 05 executed; `R-NL-NEG-A` bounded reversibility null filed: 2026-05-28.
- Front A is design-admitted under reduced apparatus (Z₂-descent).
- Front B is horizon only.
- Three lanes (Path-i Z₂ / C1 explicit-formula / nonlinear S₂) have each
  returned a clean documented null by a distinct identified cause; no
  structural-zero edge found.
- Cross-lane bounded-null synthesis filed: 2026-05-28. Public surface gated on
  external review.
