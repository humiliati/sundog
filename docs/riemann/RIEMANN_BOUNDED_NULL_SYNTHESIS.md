# Riemann Front-A — Bounded-Null Synthesis

> Cross-lane synthesis for the `sundog_v_riemann` ledger. Filed 2026-05-28 after
> three independent probe lanes each returned a clean documented null. This is
> not RH progress, not a structural-zero result, and not evidence for or against
> RH. It is the consolidated reading of *where the Sundog apparatus has no edge
> on low-lying Riemann zero statistics, and why.*

**Date:** 2026-05-28
**Status:** Reviewed-internally synthesis. External sanity check is the
registered gate before any public-facing claim (see "External Review Gate").

## The load-bearing statement

> Across three independent lanes — the functional-equation Z₂ reflection, the
> linear explicit formula, and the nonlinear gap-pair statistic — the Sundog
> structural-zero / isotropy apparatus produced **no** structural-zero edge on
> the first 5,000 Odlyzko `zeros1` ordinates. Each lane returned a clean null
> for a **distinct, identified** reason, and in each case the reason is a
> structural feature of the substrate, not a tuning artifact. The bounded null,
> with its named causes, is the durable result.

Any public copy must trace back to this. Collapsing it to "Sundog studied the
Riemann zeros" or "Sundog found structure in the zeros" silently retires the
discipline that makes the null worth filing. The three causes appear together
with the conclusion, exactly as the isotrophy ledger keeps "20/21 plus the named
O_617 quarantine" in one sentence.

## The three lanes

| Lane | Substrate | Admitted symmetry | Why it returned a null | Forced or non-forced? | Receipt |
| --- | --- | --- | --- | --- | --- |
| Probe 01 v1 — Path (i) | unfolded nearest-neighbor zero spacings | Z₂ functional-equation reflection `s ↔ 1-s` | the reflection acts as the **identity** on `(gap, \|center\|)` features (mirrored pair has the same gap and absolute center), so residuals are identity-zero | **forced** (algebraic identity) | [`receipts/2026-05-28_probe01_pathi_parity_decomposition.md`](receipts/2026-05-28_probe01_pathi_parity_decomposition.md) |
| C1 cell set | smoothed explicit formula | Z₂ test-function parity | the explicit formula is **linear** in the test function, so odd test functions vanish on the symmetric zero multiset by identity; parity decomposition is free | **forced** (linearity) | scaffold: [`RIEMANN_C1_CELLSET_V0.md`](RIEMANN_C1_CELLSET_V0.md) (desk-audited, not run; the audit *is* the finding) |
| Probe 05 — nonlinear S₂ | consecutive unfolded gap pairs | S₂ gap-pair swap `(s_n, s_{n+1}) ↦ (s_{n+1}, s_n)` | the swap-antisymmetric sector is a real time-reversibility statistic, but the sine-kernel / GUE baseline is reflection-invariant and **pins its expectation to zero** | **non-forced** (real statistic, pinned by baseline) | [`receipts/2026-05-28_probe05_reversibility_null.md`](receipts/2026-05-28_probe05_reversibility_null.md) |

The progression is the point. Each lane was opened *because the previous lane's
vacuity was diagnosed*, and each escaped the previous failure mode only to meet
a new one:

- Path (i) failed by **algebraic forcing** → so C1 moved to a substrate where
  the symmetry is not a feature-level identity;
- C1 failed by **linearity** → so the nonlinear lane moved to a genuinely
  nonlinear statistic where parity is not free;
- the nonlinear S₂ statistic was genuinely non-forced — it *could* have moved —
  but its expectation is **fixed by the GUE baseline** for standard reasons, so
  confirming it earns no edge.

Probe 05 is therefore the most informative of the three: the observable was real
(`s_n ≠ s_{n+1}` per sample; `D = −0.0064` at `0.45σ`, inside the registered
floor `0.0424`), and the block bootstrap independently reproduced the GUE
anti-persistence signature (`tau_boot < tau_ind`). It is a null that *had room to
be otherwise*.

## The unifying diagnosis

A v0.3h-style structural-zero receipt requires a **finite representation catalog
with a privileged sector whose absence is forced by the group structure** — in
K_facet, the standard D3 2D irreducible absent from `ker(M_i − I)` for 20 of 21
strict G.2 choreographies. The Riemann substrates examined here do not supply
such an object:

1. **The natural symmetry is too small.** The functional-equation reflection is
   Z₂; its irreducibles are only trivial + sign. There is no privileged
   higher-dimensional sector that can be "absent by construction," so there is
   nothing for a structural-zero receipt to attest to. (Original bridge notes:
   [`REPRESENTATION_BRIDGE_NOTES.md`](REPRESENTATION_BRIDGE_NOTES.md).)
2. **The richest available identity is linear.** The explicit formula — the one
   place primes and zeros meet with full structure — is a linear functional of
   the test function. Linearity makes parity decomposition free: the odd sector
   is forced, not discovered. (Audit:
   [`RIEMANN_C1_CELLSET_V0.md`](RIEMANN_C1_CELLSET_V0.md).)
3. **The nonlinear statistics are pinned by GUE.** Where the object is genuinely
   nonlinear (gap pairs, ratios, n-point counts), a real finite symmetry exists
   (S₂ swap), but the local statistics are sine-kernel to the accessible order,
   and the sine kernel's reflection invariance fixes the informative sector's
   expectation. The apparatus shares any signal it could find with standard
   reversibility / GUE-deviation tests. (Bridge:
   [`NONLINEAR_PAIR_CORRELATION_BRIDGE_NOTES.md`](NONLINEAR_PAIR_CORRELATION_BRIDGE_NOTES.md).)

So the obstruction is not "we didn't look hard enough." It is that the
structural-zero discipline needs a forced-absent sector in a finite group
catalog, and the low-lying zero statistics — at the registered windows and
statistics — either lack a non-trivial catalog (Z₂), trivialize it by linearity
(explicit formula), or have the relevant sector's expectation set by GUE
(nonlinear gap statistics). This is a *characterized* boundary, not a vague
negative.

## What this is — and is not

**Is:**

- a durable bounded-null with three independent, dated, reproducible receipts;
- a precisely located failure boundary: the apparatus has no structural-zero
  edge on Z₂-reflection, linear-explicit-formula, or GUE-pinned-nonlinear
  substrates at the first-`5000`-zeros window;
- a faithful realization of the ledger's own hook — "Publish the projection.
  Publish where it breaks. The failure boundary is the credential."

**Is not:**

- a proof or disproof of RH, or any evidence bearing on RH;
- a structural-zero receipt for Riemann zeros (Branch B was unreachable in every
  admitted lane);
- a claim that GUE / Montgomery–Odlyzko / Rudnick–Sarnak / Connes statistics are
  replaced, extended, or improved;
- a statement that *no* Sundog coupling to RH-adjacent mathematics can ever
  exist — only that the three lanes drilled here do not yield one, for reasons
  named above.

## What would reopen the program

All currently closed or horizon. None is in flight; each would need its own
pre-registration before any run.

1. **A defended Path (ii) S₃-via-triple bridge** with a non-arbitrary
   anchor selector that does *not* trivialize the structural-zero discipline
   (the relabeling failure mode from
   [`REPRESENTATION_BRIDGE_NOTES.md`](REPRESENTATION_BRIDGE_NOTES.md) §ii and the
   C3 quarantine in the nonlinear bridge). No concrete construction is known.
2. **A higher-order statistic whose deviation from GUE is *not*
   reflection-symmetric** — i.e. a place where the arithmetic corrections to the
   sine kernel break the symmetry the S₂ null relied on. This is beyond the
   accessible order at `N = 5000` and would need a much larger / higher table,
   making it a different domain with its own P0 lock.
3. **An external reviewer identifying an angle** the desk audits missed — the
   registered gate below.

## External Review Gate

Before any public-facing surface (page, paper section, outreach packet) cites
this synthesis, an external sanity check should confirm:

1. that `R-NL-NEG-A` was the correct call for Probe 05 — i.e. the GUE baseline
   genuinely pins the gap-reversibility expectation at the accessible order, and
   the test adds nothing beyond a standard reversibility check;
2. that the C1 linearity verdict is faithful to a standard explicit-formula
   normalization (the cell set's `R-C1-NEG-A` / `R-C1-NEG-D` questions);
3. that none of the three nulls is an artifact of the unfolding or windowing
   choices rather than a substrate feature.

Target: a point-process / random-matrix or analytic-number-theory reviewer
familiar with Montgomery–Odlyzko statistics, consecutive-gap tests, and the
Weil explicit formula. Named at the time of review.

Reviewer materials:

- [`EXTERNAL_REVIEW_PACKET.md`](EXTERNAL_REVIEW_PACKET.md) - minimal review
  packet and three-question sanity-check ask.
- [`EXTERNAL_REVIEW_EMAIL_DRAFT.md`](EXTERNAL_REVIEW_EMAIL_DRAFT.md) - email
  drafts for requesting the review without implying endorsement.

Until that review lands, this synthesis is an internal ledger artifact. The
bounded null may be discussed inside the Sundog research record; it may not be
elevated to a public RH-adjacent posture (main-ledger Falsification Mode 5 /
coupling-overreach guard).

## Cross-references

- [`../SUNDOG_V_RIEMANN.md`](../SUNDOG_V_RIEMANN.md) — main ledger; this synthesis
  is the consolidated reading of Probes 01 and 05 plus the C1 audit.
- [`../RIEMANN_LITPASS_MEMO.md`](../RIEMANN_LITPASS_MEMO.md) — prior art; the GUE /
  sine-kernel and explicit-formula baselines the nulls lean on.
- [`REPRESENTATION_BRIDGE_NOTES.md`](REPRESENTATION_BRIDGE_NOTES.md) — Path (i)/(ii)/(iii)
  bridge; the Z₂-too-small finding.
- [`FRONT_A_FUNCTIONAL_EQUATION_READING.md`](FRONT_A_FUNCTIONAL_EQUATION_READING.md)
  and [`RIEMANN_C1_CELLSET_V0.md`](RIEMANN_C1_CELLSET_V0.md) — the linear lane and
  its linearity audit.
- [`NONLINEAR_PAIR_CORRELATION_LANE.md`](NONLINEAR_PAIR_CORRELATION_LANE.md) and
  [`NONLINEAR_PAIR_CORRELATION_BRIDGE_NOTES.md`](NONLINEAR_PAIR_CORRELATION_BRIDGE_NOTES.md)
  — the nonlinear lane and its S₂-only admission.
- [`receipts/2026-05-28_probe01_pathi_parity_decomposition.md`](receipts/2026-05-28_probe01_pathi_parity_decomposition.md)
  and [`receipts/2026-05-28_probe05_reversibility_null.md`](receipts/2026-05-28_probe05_reversibility_null.md)
  — the two dated run receipts.
- [`../SUNDOG_V_ISOTROPHY_KFACET.md`](../SUNDOG_V_ISOTROPHY_KFACET.md) — the
  structural-zero discipline whose forced-absent-sector requirement this
  synthesis shows the Riemann substrates do not meet.

## Status

- 2026-05-28: three lanes closed with dated nulls; cross-lane synthesis filed.
- Structural-zero edge: not found; not expected in the examined substrates.
- Reopen paths: all horizon (defended S₃ bridge / non-symmetric higher-order
  deviation / external angle).
- Public surface: blocked on external review.
