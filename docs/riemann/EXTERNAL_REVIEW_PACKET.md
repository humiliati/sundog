# Riemann Bounded-Null Record (review questions resolved in-house)

> Minimal record of a closed bounded-null. This is not a public page and not an
> RH-adjacent claim. It began as a reviewer packet — a request for an external
> sanity check on three framing questions. As of 2026-05-30 all of those
> questions have been answered in-house by controls, a proof, and a desk
> pre-check (see "Resolution" and "Review Questions — Resolved In-House"). It is
> retained as a self-contained record: sendable to a reviewer as a finished
> result with the controls attached, not as an open set of questions.

**Date:** 2026-05-28 (review questions resolved in-house 2026-05-30)  
**Status:** bounded-null closed; external review no longer gating the framing  
**Primary synthesis:** [`RIEMANN_BOUNDED_NULL_SYNTHESIS.md`](RIEMANN_BOUNDED_NULL_SYNTHESIS.md)  
**Closure receipts:** [`receipts/2026-05-30_category_a_controls_summary.md`](receipts/2026-05-30_category_a_controls_summary.md) · [`receipts/2026-05-30_pathii_triple_desk_precheck.md`](receipts/2026-05-30_pathii_triple_desk_precheck.md)

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
> review question — whether that negative result is framed honestly — has been
> answered in-house (2026-05-30; see "Resolution" below and the closure receipts).

Current status:

- Three lanes are closed as nulls: Z2 reflection, explicit-formula parity, and
  nonlinear consecutive-gap S2 reversibility.
- No RH proof, disproof, or evidence claim is live.
- **The three review questions below are now answered in-house** (2026-05-30): Q1
  by a sine-kernel/GUE control, Q2 by proof, Q3 by a window/unfolding sweep; the
  residual representation-sector question is closed by the Path-(ii) triple desk
  pre-check. External review is no longer gating the framing.
- The public-surface gate (whether to publish a `riemann.html`) is a **separate
  decision** left to the owner; this closure resolves the *framing* questions,
  not the publication call.

Core mapping:

| Sundog concept | Riemann instantiation | Current read |
| --- | --- | --- |
| Structural-zero / absent-sector receipt | forced absence in a finite representation catalog | not found; the natural triple / S3 extension is closed at desk (2026-05-30) |
| Z2 reflection | functional-equation reflection / mirrored zero pairs | identity-zero **proved** (odd function over a symmetric ±γ multiset) |
| Parity split | explicit-formula test-function parity | linearity makes odd cancellation free — **proved** identity |
| Nonlinear symmetry | S2 swap of consecutive unfolded gap pairs | GUE / sine-kernel pinned — **confirmed by control**: observed D is a z=−0.85 draw from a synthetic sine-kernel D-distribution |

Falsification routes (each now run in-house, 2026-05-30):

- ~~GUE / sine-kernel symmetry does not justify the Probe 05 reversibility read.~~
  → **Run:** a synthetic CUE/sine-kernel + i.i.d. Wigner control reproduces the
  null; the observed D is a typical (z=−0.85) draw. Confirmed.
- ~~The C1 explicit-formula parity diagnosis depends on a nonstandard
  normalization or misses a standard term.~~ → **Proved** an identity,
  independent of normalization, arithmetic, and RH.
- ~~Windowing, unfolding, or `N = 5000` is a better explanation.~~ → **Swept:**
  the null persists and deepens to 100,000 ordinates and is unfolding-insensitive
  (RvM / raw / smoothed agree).
- ~~A legitimate finite-group / representation-sector reading was prematurely
  dismissed.~~ → **Desk pre-check:** the natural triple / S3 sectors are
  forced-vacuous or fabricated-symmetry relabelings; genuine 3-point statistics
  are GUE-pinned. Only a defended (research-level) prime-side selector could
  reopen this.
- Replace "substrate-level" with the more precise "GUE/sine-kernel reversibility
  null (universal) + identity/linearity parity nulls" — **adopted** (being more
  conservative needs no sign-off).

## Resolution (was: One-Sentence Ask)

The original ask — "sanity-check whether the bounded-null conclusion is correctly
framed" — has been **answered in-house** (2026-05-30). The three lanes produced no
Sundog structural-zero edge, and each null's cause is now established rather than
asserted: the S2 reversibility null **is** the GUE / sine-kernel prediction
(control: observed D at z=−0.85 of the synthetic distribution), and the two parity
nulls are **proved** identities. The "substrate-level vs tuning-artifact" worry is
settled in favor of substrate-level — specifically the universal GUE/sine-kernel
reversibility null — and confirmed robust to 20× more ordinates. A reviewer is
still welcome to flag a missed angle, but no reviewer judgment is required to
stand behind the framing.

## What We Are Not Asking

- Not asking for a review of a proof of RH.
- Not asking whether this is evidence for or against RH.
- Not asking for endorsement of Sundog, the broader apparatus, or any public
  presentation.
- Not asking the reviewer to debug the entire repository.

## Review Questions — Resolved In-House (2026-05-30)

The three calls this packet originally asked a reviewer to make have each been
answered by an in-house Category-A control. Full detail:
[`receipts/2026-05-30_category_a_controls_summary.md`](receipts/2026-05-30_category_a_controls_summary.md).

1. **Probe 05 / nonlinear S2 call** — *Is `R-NL-NEG-A` the right disposition; is
   `D = -0.0064` inside floor `0.0424` just the standard GUE/sine-kernel null?*
   → **CONFIRMED.** A synthetic CUE/sine-kernel control (the exact bulk process,
   Montgomery–Odlyzko) plus an i.i.d. Wigner contrast, run through the identical
   D statistic, place the observed value at **z = −0.85 / 19th percentile** of the
   sine-kernel D-distribution (mean ≈ 0); **100%** of sine-kernel realizations sit
   within the registered floor, and the probe's own bootstrap floor recomputed on
   real sine-kernel (0.0234) matches the zeta value (0.0204). A standard null.
   (`scripts/riemann-q1-sinekernel-control.py`)
2. **C1 explicit-formula call** — *Is the odd-cancellation an identity/linearity
   fact, not a structural-zero receipt?*
   → **PROVED.** The explicit-formula functional is linear and the ordinate
   multiset is `±γ`-symmetric (unconditionally, from `ζ(s̄) = conj ζ(s)`), so
   `Σ_γ h_odd(γ) = 0` term-by-term — an identity, independent of RH and arithmetic.
   (`receipts/2026-05-30_Q2_parity_identity_proof.md`)
3. **Window / unfolding call** — *Is the null an artifact of the first 5,000
   ordinates, the unfolding, or the smoothing?*
   → **NO.** Re-running the identical statistic to the full **100,000** Odlyzko
   ordinates keeps `|D|` inside the (shrinking) floor at every window, and the
   value deepens toward 0 (|D| → 0.001 at 100k vs floor 0.0095); three unfoldings
   (RvM / raw / smoothed) agree to ~0.001. (`scripts/riemann-q3-window-sweep.py`)

## Reviewer Time Budget

If sent to a reviewer as a finished result:

- **5 minutes:** read "Resolution" and "Review Questions — Resolved In-House."
- **15 minutes:** additionally read the controls-summary receipt (Q1 numbers, Q3
  sweep table) and the Q2 proof.
- **30 minutes:** additionally read the Path-(ii) triple desk pre-check and the
  bridge notes.

## Core Claim To Audit

From the synthesis:

> Across three independent lanes -- the functional-equation Z2 reflection, the
> linear explicit formula, and the nonlinear gap-pair statistic -- the Sundog
> structural-zero / isotropy apparatus produced no structural-zero edge on the
> first 5,000 Odlyzko `zeros1` ordinates. Each lane returned a clean null for a
> distinct, identified reason, and in each case the reason is a structural
> feature of the substrate, not a tuning artifact. The bounded null, with its
> named causes, is the durable result.

This sentence was the thing under review. It has been audited in-house
(2026-05-30): the controls confirm the "structural feature of the substrate, not a
tuning artifact" clause (specifically the GUE/sine-kernel reversibility null,
robust to 100,000 ordinates), and the two parity causes are proved identities.
"Structural feature of the substrate" is retained but is most precisely read as
"the universal GUE/sine-kernel reversibility null."

## Three-Lane Summary

| Lane | Result | Why the null is not a Sundog structural-zero edge | In-house verification (2026-05-30) |
| --- | --- | --- | --- |
| Probe 01 Path (i): Z2 reflection | Identity-zero null | The reflected pair preserves gap and absolute center, so the relevant residuals vanish by construction. | **Proved** identity (Q2) |
| C1 explicit formula | Desk-audited linearity null | The explicit formula is linear in the test function; parity decomposition is free on a symmetric zero multiset. | **Proved** identity (Q2) |
| Probe 05 nonlinear S2 | Bounded reversibility null | The statistic is non-forced per sample, but GUE / sine-kernel reflection invariance predicts zero mean. | **Control confirms** (Q1, z=−0.85); robust to 100k ordinates (Q3) |

## Files To Read

Primary:

- [`RIEMANN_BOUNDED_NULL_SYNTHESIS.md`](RIEMANN_BOUNDED_NULL_SYNTHESIS.md) -
  capstone conclusion and review gate.
- [`receipts/2026-05-30_category_a_controls_summary.md`](receipts/2026-05-30_category_a_controls_summary.md)
  - **closure: Q1/Q2/Q3 resolved in-house** (sine-kernel control, parity proof,
  window/unfolding sweep).
- [`receipts/2026-05-30_pathii_triple_desk_precheck.md`](receipts/2026-05-30_pathii_triple_desk_precheck.md)
  - **closure: the representation-sector / triple question, decided at desk.**
- [`receipts/2026-05-30_Q2_parity_identity_proof.md`](receipts/2026-05-30_Q2_parity_identity_proof.md)
  - the parity / reflection identity proof.
- [`receipts/2026-05-28_probe05_reversibility_null.md`](receipts/2026-05-28_probe05_reversibility_null.md)
  - nonlinear S2 run receipt.
- [`RIEMANN_C1_CELLSET_V0.md`](RIEMANN_C1_CELLSET_V0.md) - explicit-formula cell
  set and linearity addendum.

Secondary:

- [`receipts/2026-05-28_probe01_pathi_parity_decomposition.md`](receipts/2026-05-28_probe01_pathi_parity_decomposition.md)
  - Probe 01 Path (i) receipt.
- [`NONLINEAR_PAIR_CORRELATION_BRIDGE_NOTES.md`](NONLINEAR_PAIR_CORRELATION_BRIDGE_NOTES.md)
  - S2 admitted, C3 quarantined, residual bins downgraded.
- [`REPRESENTATION_BRIDGE_NOTES.md`](REPRESENTATION_BRIDGE_NOTES.md) - original Z2
  / S3 / quarantine bridge.
- [`PROBE_05_NONLINEAR_ZERO_STATISTICS_SPEC.md`](PROBE_05_NONLINEAR_ZERO_STATISTICS_SPEC.md)
  - Probe 05 pre-registration.

Runnable / audit support:

- [`../../scripts/riemann-probe05-reversibility.mjs`](../../scripts/riemann-probe05-reversibility.mjs)
  - runner used for Probe 05.
- [`../../scripts/riemann-q1-sinekernel-control.py`](../../scripts/riemann-q1-sinekernel-control.py)
  - Q1 sine-kernel / GUE control → `results/riemann/probe05-q1-sinekernel-control/`.
- [`../../scripts/riemann-q3-window-sweep.py`](../../scripts/riemann-q3-window-sweep.py)
  - Q3 window / unfolding sweep → `results/riemann/probe05-q3-window-sweep/`.
- `results/riemann/probe05-nonlinear-zero-statistics/` - local Probe 05 run
  artifacts (ignored by git; include separately if sending a ZIP).

## Representation-Sector Question — Closed at Desk (2026-05-30)

This was the last question that genuinely needed a human ("is there a finite-group
/ representation-sector reading we are prematurely dismissing?"). A desk pre-check
([`receipts/2026-05-30_pathii_triple_desk_precheck.md`](receipts/2026-05-30_pathii_triple_desk_precheck.md))
adjudicates every natural triple / S3 construction:

- the C3 cyclic triple fabricates an adjacency the linearly-ordered sequence does
  not have (a relabeling; even a nonzero result is meaningless);
- the pure-zero S3 triple is symmetric-polynomial-trivial (the parity identity one
  dimension up);
- genuine 3-point statistics are GUE-pinned by the same reflection-invariance that
  pins the pair null (reinforced by the 100k deepening-null), or are standard
  Rudnick–Sarnak n-level correlations;
- the only non-forced direction (a prime-side selector) escapes triviality only by
  importing the linear explicit formula plus an arbitrary selector whose own
  admission gate predicts a "known device."

So the conservative call ("no structural-zero edge here") is the right one, and
reopening requires genuine research (a defended, non-arbitrary prime-conjugate
selector), not a reviewer's paragraph.

## If Still Sent to a Reviewer

The framing no longer depends on review, so the only remaining useful reviewer
output is a pointer we could not generate ourselves — e.g.:

- "Your prime-conjugate selector idea (ii-b/c) is / isn't a known explicit-formula
  device; see X." (the one research-level reopen path)
- "The GUE/sine-kernel control should cite or compare against X."
- "A specific published result already states this null; cite B."

We are no longer asking a reviewer to adjudicate whether the bounded null is honest
— the controls and the desk pre-check settle that.

## Packet Hygiene

Do not send a polished public page first. Send this record or a PDF rendering of
it. A `riemann.html` page can be made later only if it carries the same
load-bearing null statement; the prior "external review pending" banner can be
replaced with "bounded null; review questions resolved in-house 2026-05-30," since
the framing no longer depends on a pending review. Whether to publish at all
remains the owner's separate call.
