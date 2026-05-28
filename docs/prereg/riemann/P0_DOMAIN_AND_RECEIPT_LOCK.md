# Riemann P0 - Domain and Receipt Lock

Status: design lock. This is not an executed pre-registration and admits no
claim by itself.

Purpose:

> Freeze the shape of `sundog_v_riemann` receipts before the first zero-data
> run, so the project cannot drift from "bounded stress test" into "RH-adjacent
> narrative" after seeing output.

## Scope

This P0 lock covers Front A only:

- zero spacings;
- pair-correlation features;
- isotropy v0.3 / induced-representation classification;
- structural-zero and quarantine receipts.

Front B projection work is horizon and requires its own P0-style lock before
execution.

## Initial Domain Envelope

Admissible first-run target:

- Data type: non-trivial Riemann zeta zeros on the critical line as supplied by
  a named public table or reproducible generator.
- Window: first `N` zeros, with candidate `N = 5000` and maximum height below
  about `1e4`.
- Primary statistic: nearest-neighbor unfolded spacings.
- Secondary statistic: pair-correlation features under one registered pair
  window.
- Excluded: higher n-point functions, high-zero tables, unsmoothed explicit
  formula experiments, Galois/class-field actions, and any claim outside the
  first registered window.

The candidate `N = 5000` is intentionally a candidate, not an execution result.
The actual run manifest must freeze `N` before data inspection.

## Admission Requirements

No Probe 01 run is admitted unless the manifest states:

- exact zero source;
- exact `N`;
- arithmetic precision;
- zero-validation rule;
- unfolding formula;
- pair-window definition;
- bin edges;
- representation bridge;
- residual and structural-zero thresholds;
- quarantine predicates;
- output path;
- code commit hash;
- command line or notebook path.

If any field is missing, the output is exploratory and cannot be cited as a
receipt.

## Outcome Branches

| Branch | Trigger | Disposition |
| --- | --- | --- |
| A - clean bounded catalog | all source, bridge, and threshold checks pass | bounded Front A receipt |
| B - structural-zero receipt | clean-domain rows satisfy the structural-zero predicate by construction | structural-zero receipt; no positive alignment claim |
| C - bridge quarantine | representation bridge fails before or during classification | named quarantine |
| D - residual breach | registered residual or variance threshold is crossed | falsified in registered cell |
| E - scope leak | interpretation requires changing `N`, statistic, source, or threshold | domain-leak quarantine |

## Anti-Scope-Creep Rule

The first run may fail. If it fails, the next action is a new dated probe spec
with a new falsifier, not a silent domain expansion.

Examples:

- If `N = 5000` is noisy, do not simply rerun at `N = 50000` and call the old
  claim supported.
- If nearest-neighbor spacings fail, do not switch to a higher n-point statistic
  inside the same receipt.
- If a source table and generator disagree, freeze an adjudication rule before
  selecting whichever produces a cleaner catalog.

## Receipt Storage

Raw outputs:

- `results/riemann/probe01-isotropy-zero-pairs/`

Reviewed notes:

- `docs/riemann/receipts/YYYY-MM-DD_probe01_<short-verdict>.md`

The reviewed note must cite the raw result directory and fill the receipt
template in [`../../riemann/RECEIPT_TEMPLATE.md`](../../riemann/RECEIPT_TEMPLATE.md).

## Current State

- 2026-05-28: project opened from roadmap draft.
- 2026-05-28: bridge admission resolved as **Path (i) — Z₂-descent under
  functional-equation reflection** per
  [`../../riemann/REPRESENTATION_BRIDGE_NOTES.md`](../../riemann/REPRESENTATION_BRIDGE_NOTES.md);
  the "representation bridge" admission-requirement field is satisfied for
  Probe 01 v1 by Path (i). Branch B (structural-zero receipt) is **not
  reachable** under Path (i); Branches A, C, D, E remain reachable.
- 2026-05-28: lit-pass filed at
  [`../../RIEMANN_LITPASS_MEMO.md`](../../RIEMANN_LITPASS_MEMO.md) with
  time-stamped gap claims; re-audit windows live there.
- 2026-05-28: Probe 01 v1 Path (i) run admitted and filed against Odlyzko
  `zeros1`, `N = 5000`, max-height ceiling `1e4`, nearest-neighbor pairs,
  reflection-residual threshold `1e-12`, and spacing sign-component threshold
  `1e-12`. Raw output: `results/riemann/probe01-isotropy-zero-pairs/`.
  Reviewed receipt:
  [`../../riemann/receipts/2026-05-28_probe01_pathi_parity_decomposition.md`](../../riemann/receipts/2026-05-28_probe01_pathi_parity_decomposition.md).
