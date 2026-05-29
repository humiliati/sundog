# Nonlinear Pair-Correlation — Representation Bridge Notes

> Bridge scoping note for the nonlinear lane in
> [`NONLINEAR_PAIR_CORRELATION_LANE.md`](NONLINEAR_PAIR_CORRELATION_LANE.md).
> Answers the gate question that lane named before any Probe 05 can run:
> **which finite symmetry is actually present in the nonlinear statistic, and
> what would count as a nontrivial sector rather than a relabeling?**
> Status: bridge scoping draft. Not execution-admitted. No probe run.

**Date:** 2026-05-28
**Trigger:** the C1 linearity audit
([`RIEMANN_C1_CELLSET_V0.md`](RIEMANN_C1_CELLSET_V0.md)) showed the explicit
formula is the wrong substrate for a structural-zero edge *because it is
linear*; this note adjudicates whether the nonlinear gap statistics carry a
real finite symmetry.

## The bridge problem (restated for the nonlinear lane)

The original bridge notes
([`REPRESENTATION_BRIDGE_NOTES.md`](REPRESENTATION_BRIDGE_NOTES.md)) found that
the v0.3h D3 structural-zero discipline does not carry across the Riemann Z₂
reflection: the "absent-by-construction" sector has no counterpart. The C1 cell
set then showed the explicit-formula parity decomposition is forced by
linearity (odd test functions vanish on the symmetric zero multiset by
identity, not by discovery).

The nonlinear lane's hope is that gap pairs, gap ratios, and two-point counts
are **not** linear functionals of a single test function over individual zeros,
so a finite symmetry acting on them might have a sector that is informative
rather than forced. The bridge question is whether any of the three proposed
hooks is a **genuine symmetry of the registered statistic** (whose nontrivial
sector is a real, fluctuating observable) or merely a **relabeling** (whose
nontrivial sector is forced or meaningless).

The discriminator used throughout this note:

```text
genuine sector   = nontrivial under the action AND not algebraically forced to
                   a constant AND its expectation is not fixed by the baseline
                   for a reason unrelated to the apparatus
relabeling sector = forced to a constant by construction, OR corresponds to no
                   symmetry the underlying ordered process actually has
```

This is stricter than the C1 test. C1 only had to avoid algebraic forcing.
Here a hook can clear algebraic forcing and still fail, if the baseline (GUE /
sine kernel) already fixes its expectation for standard reasons — that is
`R-NL-NEG-A` (GUE dominance), distinct from `R-NL-NEG-B` (representation
triviality).

## Hook 1 — S2 gap-pair swap (admitted, with a pinned honest framing)

**Action.** On consecutive unfolded gaps, `tau: (s_n, s_{n+1}) -> (s_{n+1}, s_n)`,
`S2 = {id, tau}`. Decompose any pair feature `f`:

```text
f_sym(s_n, s_{n+1})  = (f(s_n, s_{n+1}) + f(s_{n+1}, s_n)) / 2
f_anti(s_n, s_{n+1}) = (f(s_n, s_{n+1}) - f(s_{n+1}, s_n)) / 2
```

**Is the antisymmetric sector forced?** No. For a single realization
`s_n != s_{n+1}` generically, so `f_anti` is generically nonzero per sample.
This is the decisive difference from C1, where every odd-sector term was an
algebraic identity-zero. Here the per-sample antisymmetric content is real;
only its *sample mean* is in question. Hook 1 therefore **clears `R-NL-NEG-B`**.

**What does the antisymmetric sector actually measure?** Time-reversibility of
the gap sequence. The mean of `f_anti` is nonzero exactly when
`(s_n, s_{n+1})` is *not* equal in distribution to `(s_{n+1}, s_n)` — i.e. when
the unfolded gap sequence has an arrow of time. A canonical desk-auditable
observable:

```text
D = ( #{n : s_n > s_{n+1}} - #{n : s_n < s_{n+1}} ) / (N - 2)
```

`E[D] = 0` iff the consecutive-gap law is swap-symmetric. Note `D` is **not**
forced to zero by stationarity alone: a stationary-but-irreversible sequence
can have `P(s_n > s_{n+1}) != 1/2`. So `D` is a genuine reversibility statistic.

**Why it most likely lands in `R-NL-NEG-A` anyway.** The registered baseline is
the sine-kernel / GUE bulk process. That process is reflection-invariant
(`x -> -x`; its kernel `K(x,y) = sin(pi(x-y))/(pi(x-y))` depends only on
`|x-y|`). Reflection invariance of the point process implies the gap sequence
is reversible, hence `(s_n, s_{n+1}) =d (s_{n+1}, s_n)` and `E[D] = 0` under the
baseline. The leading arithmetic corrections to GUE for the zeros
(Bogomolny–Keating / Berry lower-order terms) enter the two-point function as a
function of `|distance|` and remain reflection-symmetric at accessible order, so
they do **not** introduce expected gap-irreversibility a finite `N = 5000`
window could resolve. Therefore:

- a **null** `D ~ 0` within the sampling floor is the predicted outcome, and it
  is `R-NL-NEG-A`: GUE already predicts reversibility, so the apparatus earns no
  edge from confirming it;
- a **nonzero** `D` beyond the floor would be a genuine deviation from
  baseline reversibility — interesting — but it is detectable by *any*
  time-reversal test, so the apparatus would not uniquely own it; it would be
  reported as a GUE-anomaly flag, not a Sundog structural-zero.

**Disposition for Hook 1: ADMITTED for a Probe 05 v0**, on the explicit
pre-registration that the expected verdict is `R-NL-NEG-A` and the receipt is a
*bounded reversibility-test receipt*, not a structural-zero claim. This is the
nonlinear analog of the Path (i) admission: real, runnable, auditable, and
honest about its small reach.

## Hook 2 — C3 cyclic triple (quarantined as a relabeling)

**Action.** On triples, `rho: (s_n, s_{n+1}, s_{n+2}) -> (s_{n+1}, s_{n+2}, s_n)`,
`C3 = {id, rho, rho^2}`.

**Adjudication: relabeling.** The cyclic action wraps `s_n` to the end of a
**linearly ordered** triple. The shift the process actually admits is
`(s_n, s_{n+1}, s_{n+2}) -> (s_{n+1}, s_{n+2}, s_{n+3})` (stationarity), which is
*not* `rho` — `rho` reuses `s_n` in the third slot, fabricating a cyclic
adjacency between `s_{n+2}` and `s_n` that the ordered sequence does not have.
The two nontrivial C3 characters therefore measure a quantity defined by an
action the data does not possess. This is `R-NL-NEG-B` (representation
triviality) in its purest form: not algebraically forced to zero, but
corresponding to no symmetry of the underlying process — exactly the
"relabeling" failure the lane note warned about, and the same trap the original
bridge notes flagged for the S₃-via-triple Path (ii-a) construction.

**Disposition for Hook 2: NOT ADMITTED.** File `R-NL-NEG-B` if a future spec
tries to invoke it without first defending a real cyclic structure on the gap
triple (none is known). Held strictly weaker than the S₂ admission.

## Hook 3 — baseline-residual sign bins (downgraded; no structural content)

**Object.** Decompose the pair-correlation residual `data - sine_kernel` into
registered local feature bins and call the finite bin catalog the
"representation object."

**Adjudication: no structural-zero content.** The residual against the sine
kernel *is* the standard pair-correlation comparison the Track A literature
already computes (Montgomery, Odlyzko, Rudnick–Sarnak). Binning it introduces
catalog vocabulary but no "absent-by-construction" sector: a bin's residual is
whatever the data-minus-baseline is, with no symmetry forcing any bin to a
privileged zero. So a clean result is `R-NL-NEG-A` (GUE dominance) by
construction, and there is no structural-zero notion to earn.

**Disposition for Hook 3: DOWNGRADED.** Retain it only as ordinary descriptive
residual analysis with no representation claim and no Sundog edge language. It
must not be dressed as a finite-catalog structural-zero apparatus.

## Default for Probe 05 v0

Admit **Hook 1 (S2 swap) only**, framed as a reversibility-test receipt:

1. Statistic: consecutive unfolded gap pairs `(s_n, s_{n+1})`; observable `D`
   (and optionally the antisymmetric part of the joint 2D gap histogram).
2. Baseline: sine-kernel / GUE, whose prediction `E[D] = 0` is registered
   *before* the run as the expected (NEG-A) outcome.
3. Floor: a bootstrap / finite-`N` sampling floor on `D` fixed before
   inspection (per `R-NL-NEG-C`); the run reports `|D|` against this floor.
4. Allowed verdicts:
   - `|D| <= floor` → bounded reversibility-test receipt; disposition
     `R-NL-NEG-A` (GUE dominance; apparatus earns no edge) — **the predicted
     result**, recorded as a clean null;
   - `|D| > floor` → GUE-reversibility-anomaly flag; explicitly *not* a Sundog
     structural-zero; escalate only to an independent reversibility cross-check,
     never to an RH claim;
   - sampling floor cannot be fixed pre-run → `R-NL-NEG-C`.
5. Forbidden: invoking Hook 2 (C3) or Hook 3 (residual bins) inside the same
   receipt to rescue a null (`R-NL-NEG-B` / `R-NL-NEG-D` guard).

This is deliberately small. Its honest selling point is the same as Probe 01
v1's: it establishes where the apparatus has *no* edge, with a real (not
algebraically forced) statistic this time.

## Falsifier coupling

Mapping to the lane note's named negatives:

- **`R-NL-NEG-A` (GUE dominance)** — the predicted landing for the admitted
  Hook 1; a null `D` confirms baseline reversibility the apparatus did not earn.
- **`R-NL-NEG-B` (representation triviality)** — fires for Hook 2 immediately
  (cyclic action on linearly-ordered gaps is not a real symmetry), and for any
  attempt to upgrade Hook 1's S₂ into a claimed larger group on the same data.
- **`R-NL-NEG-C` (sampling-floor failure)** — fires if the finite window cannot
  pin a `D` floor independent of the observed value.
- **`R-NL-NEG-D` (bridge overreach)** — fires if S₂/C3 *local gap* symmetries
  are silently rewritten as D3/S₃ or as the functional-equation Z₂; the S₂
  swap is a symmetry of the *gap-pair feature*, not of the critical line.

## Honest summary

Of the three hooks, **only the S₂ gap-pair swap is a genuine, non-forced
symmetry of the registered statistic** — and it is a time-reversibility test
whose expected value is pinned to zero by the GUE baseline for standard reasons.
So the nonlinear lane is a real improvement over C1 (the observable is no longer
an algebraic identity-zero) but it is **not** a path to a structural-zero edge:
its clean-null outcome is `R-NL-NEG-A`, and its only "positive" would be a
baseline-reversibility anomaly the apparatus shares with any reversibility test.
C3 is a relabeling; residual bins are standard stats. If the program's goal is a
*structural-zero* edge specifically, this lane most likely also returns a clean
documented null — which is a publishable Sundog outcome, but the user should
enter Probe 05 expecting `R-NL-NEG-A`, exactly as the lane note anticipated.

## Cross-references

- [`NONLINEAR_PAIR_CORRELATION_LANE.md`](NONLINEAR_PAIR_CORRELATION_LANE.md)
  — the lane scoping note this bridge answers; named negatives R-NL-NEG-A…D.
- [`REPRESENTATION_BRIDGE_NOTES.md`](REPRESENTATION_BRIDGE_NOTES.md)
  — the original Path (i)/(ii)/(iii) bridge; this note is its nonlinear sibling
  and reuses the admit / quarantine / downgrade structure.
- [`RIEMANN_C1_CELLSET_V0.md`](RIEMANN_C1_CELLSET_V0.md)
  — the linear cell set whose audit opened this lane.
- [`../RIEMANN_LITPASS_MEMO.md`](../RIEMANN_LITPASS_MEMO.md)
  — Track A prior art (Montgomery / Odlyzko / Rudnick–Sarnak) and the GUE /
  sine-kernel baseline this note leans on.
- [`../SUNDOG_V_RIEMANN.md`](../SUNDOG_V_RIEMANN.md) — main ledger.

## Current state

- 2026-05-28: bridge problem restated for the nonlinear lane; three hooks
  adjudicated.
- Hook 1 (S₂ swap) admitted for a Probe 05 v0 as a reversibility-test receipt,
  NEG-A pre-registered as the expected outcome.
- Hook 2 (C3 triple) not admitted — relabeling (`R-NL-NEG-B`).
- Hook 3 (residual bins) downgraded — no structural-zero content.
- Probe 05 v0 spec filed:
  [`PROBE_05_NONLINEAR_ZERO_STATISTICS_SPEC.md`](PROBE_05_NONLINEAR_ZERO_STATISTICS_SPEC.md).
- No execution.
