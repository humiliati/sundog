# Probe 01 - Isotropy v0.3 on Low-Lying Zero Pair Data

Status: specification draft. Not execution-admitted. No run has occurred.

Bridge admission (2026-05-28): per
[`REPRESENTATION_BRIDGE_NOTES.md`](REPRESENTATION_BRIDGE_NOTES.md), the
admitted bridge for Probe 01 v1 is **Path (i) — Z₂-descent under the
functional-equation reflection `s ↔ 1-s`**. The v0.3h structural-zero
discipline does **not** carry across this descent (the D3-standard sector
that pins "absent by construction" has no counterpart under Z₂). Probe 01 v1
is reinterpreted accordingly as a parity-decomposition of zero-pair
statistics under reflection, not a v0.3h-strength structural-zero claim.
Path (ii) S₃-via-triple is deferred to a future Probe 01b spec.

Purpose:

> Test whether the isotropy v0.3 machinery — restricted to its Z₂ sub-
> representations under Path (i) descent — produces a usable, falsifiable
> catalog on low-lying Riemann-zero pair data, and record either alignment,
> null, or quarantine receipts without upgrading the result into an RH claim
> or into a v0.3h-strength structural-zero claim.

## Claim Boundary

This probe does not test RH. It tests whether a Sundog apparatus built for
choreographic symmetry and structural-zero receipts can be cleanly applied to a
registered zero-data window.

Allowed outcome language (under Path (i) Z₂-descent):

- "Probe 01 produced a parity-decomposition receipt under functional-equation
  reflection on the registered zero window."
- "Probe 01 returned a null receipt under the registered zero window."
- "Probe 01 returned a catalog-asymmetry receipt under `(12)`-action."
- "Probe 01 failed by named falsifier."
- "Probe 01 is inconclusive because the representation bridge did not admit
  at execution time" (Path (iii) quarantine).

Forbidden outcome language:

- "Evidence for RH."
- "Evidence against RH."
- "The zeros have Sundog choreography."
- "A spectral realization has been found."
- "Structural-zero receipt under v0.3h on RH" — not reachable under Path (i);
  reaching it requires a defended Path (ii) bridge in a separate Probe 01b spec.

## Input Lock

The run inherits the admission requirements in
[`../prereg/riemann/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../prereg/riemann/P0_DOMAIN_AND_RECEIPT_LOCK.md).

Before execution, the maintainer must freeze:

- zero source: one named table or one named generator path, never mixed after
  inspection;
- exact `N`;
- arithmetic precision;
- zero validation rule;
- unfolding method;
- pair construction rule;
- binning and normalization;
- structural-zero and quarantine thresholds;
- output directory;
- code commit hash.

The default candidate domain is the first `5000` non-trivial zeros, with maximum
height below the initial ledger envelope of about `1e4`, but this number is not
execution-frozen until copied into the run manifest.

## Data Products

Required outputs under `results/riemann/probe01-isotropy-zero-pairs/`:

- `manifest.json` - source, versions, thresholds, command, commit hash.
- `zeros.csv` - zero index, imaginary part, source precision, validation
  status.
- `unfolded_spacings.csv` - normalized nearest-neighbor spacings and local
  density estimate.
- `pair_features.csv` - registered pair-correlation features.
- `isotropy_records.csv` - per-record v0.3 classification, residuals, flags.
- `structural_zero_summary.csv` - counts by representation case and action.
- `quarantine.csv` - any rows excluded from clean receipts, with reason.
- `README.md` - human-readable run note and falsifier disposition.

Durable reviewed receipts should be summarized in `docs/riemann/receipts/` using
[`RECEIPT_TEMPLATE.md`](RECEIPT_TEMPLATE.md).

## Pipeline Sketch

1. Acquire or generate zeros.
2. Verify zero order, uniqueness, and precision.
3. Unfold nearest-neighbor spacings using the registered density rule.
4. Build pair-correlation features under the registered pair window.
5. Translate features into the representation bridge admitted for the run.
6. Apply isotropy v0.3 operators:
   - twist operator;
   - induced-representation `d_i` cases;
   - F_beta template;
   - tau-flag;
   - structural-zero classifier.
7. Compute catalog asymmetry under registered actions, including bare `(12)` if
   admitted.
8. Dispose the run against the falsification surface.

## Primary Falsifiers

Probe 01 directly exercises:

- **Mode 2: Isotropy v0.3 structural failure.**
- **Mode 5: Domain leakage / scope creep.**

It may also trigger:

- **Mode 1: Invariant mismatch after alignment**, if an alignment subpass is
  admitted.

## Disposition Table

| Outcome | Disposition |
| --- | --- |
| Clean parity-decomposition catalog with residuals inside thresholds | bounded Front A positive receipt under Path (i) |
| One reflection-sector empty after the registered clean-domain rule | null / catalog-asymmetry receipt under `(12)`; no positive alignment claim |
| Path (i) bridge fails admission at execution (Z₂ sector degenerates) | named quarantine, Path (iii); probe does not run or is voided |
| Threshold exceeded after execution begins | failure under Mode 2 or Mode 1 |
| Need to expand `N`, statistic, or smoothing after seeing result | Mode 5 quarantine |
| Path (ii) S₃-via-triple silently invoked to rescue a failed Path (i) run | Mode 5 quarantine; Path (ii) must live in a separate Probe 01b spec |

## Review Gate

Probe 01 is not complete until:

- artifacts exist under the registered result path;
- the receipt template is filled;
- one maintainer pass checks source/version/threshold consistency;
- the main ledger is updated with a one-paragraph status note.
