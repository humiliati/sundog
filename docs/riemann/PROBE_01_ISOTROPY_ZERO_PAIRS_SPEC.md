# Probe 01 - Isotropy v0.3 on Low-Lying Zero Pair Data

Status: specification draft. Not execution-admitted. No run has occurred.

Purpose:

> Test whether the isotropy v0.3 machinery produces a usable, falsifiable
> catalog on low-lying Riemann-zero pair data, and record either alignment,
> structural-zero, or quarantine receipts without upgrading the result into an
> RH claim.

## Claim Boundary

This probe does not test RH. It tests whether a Sundog apparatus built for
choreographic symmetry and structural-zero receipts can be cleanly applied to a
registered zero-data window.

Allowed outcome language:

- "Probe 01 produced a catalog under the registered zero window."
- "Probe 01 failed by named falsifier."
- "Probe 01 returned structural-zero / quarantine receipts."
- "Probe 01 is inconclusive because the representation bridge did not admit."

Forbidden outcome language:

- "Evidence for RH."
- "Evidence against RH."
- "The zeros have Sundog choreography."
- "A spectral realization has been found."

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
| Clean catalog with residuals inside thresholds | bounded Front A positive receipt |
| Structural-zero rows dominate under the registered clean-domain rule | structural-zero receipt; no positive alignment claim |
| Representation bridge fails admission | named quarantine; probe does not run or is voided |
| Threshold exceeded after execution begins | failure under Mode 2 or Mode 1 |
| Need to expand `N`, statistic, or smoothing after seeing result | Mode 5 quarantine |

## Review Gate

Probe 01 is not complete until:

- artifacts exist under the registered result path;
- the receipt template is filled;
- one maintainer pass checks source/version/threshold consistency;
- the main ledger is updated with a one-paragraph status note.
