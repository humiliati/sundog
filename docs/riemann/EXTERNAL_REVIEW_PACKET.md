# Riemann Bounded-Null External Review Packet

> Minimal packet for an external sanity check. This is not a public page and not
> an RH-adjacent claim. It exists to make it easy for a reviewer to say "yes,
> this null is correctly characterized," "no, this is still overstated," or
> "you missed this standard issue."

**Date:** 2026-05-28  
**Status:** draft reviewer packet  
**Primary synthesis:** [`RIEMANN_BOUNDED_NULL_SYNTHESIS.md`](RIEMANN_BOUNDED_NULL_SYNTHESIS.md)

## Reviewer Snapshot

Repository map:

- Primary working packet: `docs/riemann/`.
- Main ledger: [`../SUNDOG_V_RIEMANN.md`](../SUNDOG_V_RIEMANN.md).
- P0 domain lock only: [`../prereg/riemann/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../prereg/riemann/P0_DOMAIN_AND_RECEIPT_LOCK.md).
- Current conclusion: [`RIEMANN_BOUNDED_NULL_SYNTHESIS.md`](RIEMANN_BOUNDED_NULL_SYNTHESIS.md).

Why we looked at Riemann:

> Riemann-zero statistics were used as a bounded stress test for whether the
> Sundog structural-zero / isotropy apparatus can find a forced-absent
> representation sector in a famous mathematical substrate. It did not. The
> review question is whether that negative result is framed honestly.

Current status:

- Three lanes are closed as nulls: Z2 reflection, explicit-formula parity, and
  nonlinear consecutive-gap S2 reversibility.
- No RH proof, disproof, or evidence claim is live.
- The public surface is blocked until an external sanity check confirms or
  corrects the bounded-null framing.

Core mapping:

| Sundog concept | Riemann instantiation | Current read |
| --- | --- | --- |
| Structural-zero / absent-sector receipt | forced absence in a finite representation catalog | not found in the registered Riemann substrates |
| Z2 reflection | functional-equation reflection / mirrored zero pairs | identity-zero on registered gap and center features |
| Parity split | explicit-formula test-function parity | linearity makes odd cancellation free |
| Nonlinear symmetry | S2 swap of consecutive unfolded gap pairs | real statistic, but expectation appears GUE / sine-kernel pinned |

Ways to falsify or weaken this packet:

- Show that GUE / sine-kernel symmetry does not justify the Probe 05
  reversibility-null read.
- Show that the C1 explicit-formula parity diagnosis depends on a nonstandard
  normalization or misses a standard term.
- Show that windowing, unfolding, or `N = 5000` is a better explanation than
  the claimed substrate-level causes.
- Identify a legitimate finite-group / representation-sector reading we
  prematurely dismissed.
- Recommend replacing "substrate-level" with a narrower "finite-window" null.

## One-Sentence Ask

Please sanity-check whether our bounded-null conclusion is correctly framed:
three Riemann-adjacent lanes produced no Sundog structural-zero edge, and each
null has a distinct substrate-level cause rather than being a tuning artifact.

## What We Are Not Asking

- Not asking for a review of a proof of RH.
- Not asking whether this is evidence for or against RH.
- Not asking for endorsement of Sundog, the broader apparatus, or any public
  presentation.
- Not asking the reviewer to debug the entire repository.

## What We Are Asking

Please check the following three calls:

1. **Probe 05 / nonlinear S2 call:** Is `R-NL-NEG-A` the right disposition? In
   other words, does the sine-kernel / GUE baseline already predict
   reversibility for consecutive unfolded gap pairs, so the observed
   `D = -0.0064` inside floor `0.0424` is just a standard null rather than any
   Sundog edge?
2. **C1 explicit-formula call:** Is the linearity verdict faithful to standard
   explicit-formula practice? Namely, does odd-test-function cancellation on a
   symmetric `+-gamma` zero multiset amount to an identity/linearity fact rather
   than an empirical structural-zero receipt?
3. **Window / unfolding call:** Are the three null diagnoses plausibly substrate
   features, or is any one of them better explained as an artifact of using the
   first `5000` Odlyzko `zeros1` ordinates, the local-density unfolding, the
   smoothing scale, or another implementation choice?

A short reply is enough. The most useful possible answers are:

```text
Yes, the bounded-null framing is basically right.
No, this overstates X.
I would quarantine Y until you address Z.
This is standard/known as A; cite B.
```

## Reviewer Time Budget

Suggested review paths:

- **10 minutes:** read the synthesis load-bearing statement, the three-lane
  table, and the External Review Gate.
- **25 minutes:** additionally inspect the Probe 05 receipt observed-values
  table and the C1 linearity addendum.
- **60 minutes:** inspect the bridge notes and runner/spec details.

## Core Claim To Audit

From the synthesis:

> Across three independent lanes -- the functional-equation Z2 reflection, the
> linear explicit formula, and the nonlinear gap-pair statistic -- the Sundog
> structural-zero / isotropy apparatus produced no structural-zero edge on the
> first 5,000 Odlyzko `zeros1` ordinates. Each lane returned a clean null for a
> distinct, identified reason, and in each case the reason is a structural
> feature of the substrate, not a tuning artifact. The bounded null, with its
> named causes, is the durable result.

This sentence is the thing under review. If it is too strong, the packet should
be revised before any public surface exists.

## Three-Lane Summary

| Lane | Result | Why the null is not a Sundog structural-zero edge |
| --- | --- | --- |
| Probe 01 Path (i): Z2 reflection | Identity-zero null | The reflected pair preserves gap and absolute center, so the relevant residuals vanish by construction. |
| C1 explicit formula | Desk-audited linearity null | The explicit formula is linear in the test function; parity decomposition is free on a symmetric zero multiset. |
| Probe 05 nonlinear S2 | Bounded reversibility null | The statistic is non-forced per sample, but GUE / sine-kernel reflection invariance predicts zero mean. |

## Files To Read

Primary:

- [`RIEMANN_BOUNDED_NULL_SYNTHESIS.md`](RIEMANN_BOUNDED_NULL_SYNTHESIS.md) -
  capstone conclusion and review gate.
- [`receipts/2026-05-28_probe05_reversibility_null.md`](receipts/2026-05-28_probe05_reversibility_null.md)
  - nonlinear S2 run receipt.
- [`RIEMANN_C1_CELLSET_V0.md`](RIEMANN_C1_CELLSET_V0.md) - explicit-formula
  cell set and linearity addendum.

Secondary:

- [`receipts/2026-05-28_probe01_pathi_parity_decomposition.md`](receipts/2026-05-28_probe01_pathi_parity_decomposition.md)
  - Probe 01 Path (i) receipt.
- [`NONLINEAR_PAIR_CORRELATION_BRIDGE_NOTES.md`](NONLINEAR_PAIR_CORRELATION_BRIDGE_NOTES.md)
  - S2 admitted, C3 quarantined, residual bins downgraded.
- [`REPRESENTATION_BRIDGE_NOTES.md`](REPRESENTATION_BRIDGE_NOTES.md) - original
  Z2 / S3 / quarantine bridge.
- [`PROBE_05_NONLINEAR_ZERO_STATISTICS_SPEC.md`](PROBE_05_NONLINEAR_ZERO_STATISTICS_SPEC.md)
  - Probe 05 pre-registration.

Runnable / audit support:

- [`../../scripts/riemann-probe05-reversibility.mjs`](../../scripts/riemann-probe05-reversibility.mjs)
  - runner used for Probe 05.
- `results/riemann/probe05-nonlinear-zero-statistics/` - local run artifacts
  (ignored by git; include separately if sending a ZIP).

## If The Reviewer Has Only One Comment

Ask them to answer this:

> Is there any legitimate finite-group / representation-sector reading in
> these three Riemann substrates that we are prematurely dismissing, or is the
> "no structural-zero edge here" conclusion the right conservative call?

## Output We Want From Review

Any of:

- "OK as bounded internal null; keep public claims blocked or cautious."
- "Quarantine the synthesis because X."
- "Probe 05 should be called standard reversibility check, not GUE-pinned null,
  because X."
- "C1 linearity is right, but cite/phrase it as X."
- "The window/unfolding is too small to support substrate-level language; say
  finite-window null only."

## Packet Hygiene

Do not send a polished public page first. Send this packet or a PDF rendering of
it. A `riemann.html` page can be made later only if it carries the same load-
bearing null statement and a clear "external review pending" banner.
