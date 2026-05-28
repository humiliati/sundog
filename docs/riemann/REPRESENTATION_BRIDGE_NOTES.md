# Riemann — Representation Bridge Notes

> Pre-execution scoping note for Probe 01 of [`SUNDOG_V_RIEMANN.md`](../SUNDOG_V_RIEMANN.md).
> Resolves the load-bearing question raised by
> [`RIEMANN_LITPASS_MEMO.md`](../RIEMANN_LITPASS_MEMO.md) Track C and Probe 01 spec
> [`PROBE_01_ISOTROPY_ZERO_PAIRS_SPEC.md`](PROBE_01_ISOTROPY_ZERO_PAIRS_SPEC.md):
> **what is the representation bridge from `(pair of Riemann zeros, unfolded spacing)`
> into the v0.3h K_facet apparatus?**

**Date:** 2026-05-28
**Status:** Bridge scoping draft. Not execution-admitted. No probe run.

## The bridge problem

The v0.3h K_facet apparatus
([`isotrophy/SUNDOG_V_ISOTROPHY_KFACET.md`](../isotrophy/SUNDOG_V_ISOTROPHY_KFACET.md))
is built on the **D3 representation acting on the 21 strict G.2 single-curve
choreographies at `m3 = 1`**. The load-bearing object is the **standard D3
2D irreducible**: a row `i` produces a structural-zero receipt `c_i = d_i = 0`
*by construction* precisely when the standard D3 sector is **absent** in
`ker(M_i - I)`. This absence-by-construction is what the entire
twist operator → induced-representation `d_i` → F_beta template → tau-flag →
structural-zero classifier chain attests to.

The natural symmetry on Riemann zeros is the **Z₂ reflection `s ↔ 1-s`**
induced by the functional equation. D3 has order 6; the natural Riemann Z₂
has order 2. The standard D3 irreducible, restricted to a Z₂ subgroup,
decomposes into one trivial + one sign character. Both restricted characters
are *present* in any Z₂ representation built from real pair data; there is
no "absent by construction" target for a structural-zero receipt to attest to.

In compact form: **the v0.3h structural-zero discipline does not carry across
the natural Riemann reflection symmetry. Probe 01 admissibility hinges on
whether a candidate bridge restores a meaningful "absent-by-construction"
sector.**

## Three candidate bridges

### Path (i) — Z₂-descent (default; admits with reduced apparatus)

Bridge: take the functional-equation reflection `s ↔ 1-s` as the literal Z₂
symmetry. Apply the v0.3h apparatus *with only its Z₂ sub-representations
admitted*.

What the apparatus carries across:
- Procrustes alignment + gauge cocycle (representation-agnostic);
- closure-relative residuals (representation-agnostic);
- F_beta template's *sign-sector* component (Z₂ admits sign);
- catalog asymmetry detection under bare `(12)`-style actions, since `(12)`
  is itself a Z₂ element.

What is **lost**:
- structural-zero receipts in the D3-standard sector — there is no D3-
  standard sector under Z₂, so `c_i = d_i = 0` by construction has nothing
  to attest to;
- the induced-representation `d_i` formula loses its load (no induction step
  to perform when the source representation is already the whole group);
- the twist operator collapses to a trivial sign-flip.

**Expected receipt category under Path (i):** null receipt or
catalog-asymmetry receipt under `(12)`. *Not* a structural-zero receipt. The
P0 outcome branches need to be read with `c_i = d_i = 0` reinterpreted as
"present in trivial sector, absent in sign sector" — which is the
parity-decomposition of pair statistics under reflection.

**Honest position:** under Path (i), Probe 01 v1 is a parity-decomposition
of zero-pair statistics under functional-equation reflection. This is closer
to a linear-statistics-of-zeros computation (cf.
[math/0208220](https://arxiv.org/abs/math/0208220)) than to the v0.3h
K_facet result, and Sundog's edge is the pre-registration / receipt
discipline, not a new structural-zero category. **This is the recommended
admission for Probe 01 v1**, but the ledger should not advertise it as
"v0.3h applied to RH."

### Path (ii) — S₃-via-triple (horizon; admission not concretely available)

Bridge: group consecutive zero triples `(γ_n, γ_{n+1}, γ_{n+2})` and seek
an S₃ action on the triple.

Difficulty: zeros are inherently *ordered by height*. The natural S₃ action
is on indices, but the data — spacings `s₁ = γ_{n+1} − γ_n`,
`s₂ = γ_{n+2} − γ_{n+1}`, products / ratios — is computed from ordered data
and is not permutation-equivariant. A cyclic Z₃ action on `(s₁, s₂, s₁+s₂)`
is mechanically definable but discards the reflection-half of S₃; a full S₃
needs a construction that simultaneously admits 3-cycles and reflections on
the *invariants*, not just on the indices.

Candidate concrete constructions (none currently defended in the indexed
literature):
- **(ii-a)** Unfolded-spacing triple with a *symmetric* invariant set
  `(e_1, e_2, e_3) = (s_1+s_2+s_3, s_1 s_2 + s_2 s_3 + s_1 s_3, s_1 s_2 s_3)`
  treated as a single S₃-symmetric tuple. The S₃ action on the underlying
  `(s_1, s_2, s_3)` is then trivial on the elementary symmetric polynomials,
  which removes the structural-zero discipline (every row lives in the
  trivial sector).
- **(ii-b)** Triple `(γ_n, 1 − γ_n, γ_{n'})` for some matched-prime-side
  selector `n'`. Mixes zero data with prime-side data via the explicit
  formula. The S₃ action permutes the three "anchor types"
  (zero, reflected zero, prime-conjugate). Conceptually novel but the action
  is non-canonical — the prime-conjugate selector has degrees of freedom
  that the structural-zero receipt would inherit.
- **(ii-c)** D3-via-(zero, prime, reflection) composite, generalizing (ii-b)
  to the full D3 group with the reflection acting on the zero/reflected-zero
  pair and the 3-cycle rotating across the three anchor types.

**Status of Path (ii):** none of (ii-a, ii-b, ii-c) is concretely admissible
today. (ii-a) trivializes the structural-zero discipline. (ii-b) and (ii-c)
need a defended selector for the prime-conjugate anchor. Without such a
defense, Path (ii) is not a usable bridge for a first probe run. It is held
as **horizon for a future Probe 01b** specification, separately scoped from
Probe 01 v1.

### Path (iii) — explicit quarantine (clean failure receipt)

Bridge: declared but not admitted. If neither Path (i) nor a concrete
Path (ii) instance can be defended at execution time, Probe 01 returns a
**named quarantine** under P0 lock outcome branch C — bridge quarantine — and
does not run.

This is a *valid* Sundog outcome and matches the v0.3h O_617 precedent
(a row whose bridge direction sits outside the valid D3 representation
yields a named quarantine, not a silent failure). Path (iii) is the receipt
that gets filed if a maintainer reads these notes and decides Path (i) is
too descoped to be worth running and Path (ii) is not yet defensible.

## Default for Probe 01 v1

**Path (i) — Z₂-descent**, with explicit re-scoping of the spec language:

1. Update [`PROBE_01_ISOTROPY_ZERO_PAIRS_SPEC.md`](PROBE_01_ISOTROPY_ZERO_PAIRS_SPEC.md)
   under "Representation bridge" to name Path (i) as the admitted bridge.
2. Update the **Disposition Table** to expect a *null receipt* or
   *catalog-asymmetry receipt under `(12)`* as the bounded-positive outcome
   under Path (i), not a structural-zero receipt.
3. Update the **Allowed outcome language** to forbid "structural-zero
   receipt under v0.3h" and admit "parity-decomposition receipt under
   functional-equation reflection."
4. Reinterpret the **P0 lock outcome branches** B and C accordingly:
   under Path (i), Branch B (structural-zero) is *not reachable* — only
   Branches A (clean bounded catalog under Z₂), C (bridge quarantine, i.e.
   Path (iii)), D (residual breach), and E (scope leak) are reachable.

The honest claim made by Probe 01 v1 if Path (i) admits cleanly:

> The functional-equation reflection Z₂ admits the Sundog
> closure-residual + gauge-cocycle apparatus on unfolded nearest-neighbor
> spacings inside a registered zero window. Parity decomposition under the
> reflection produces a receipt; the v0.3h structural-zero discipline does
> not carry across this descent and is not invoked.

This is a much smaller claim than "v0.3h on RH," and the lit-pass memo
already flags that the Sundog edge in this lane is *operational discipline*,
not a new structural category.

## Probe 01b horizon (Path (ii) future spec)

If Probe 01 v1 lands cleanly and a defended (ii-b) or (ii-c) anchor selector
becomes available, a separate Probe 01b spec under
`docs/riemann/PROBE_01B_S3_TRIPLE_BRIDGE_SPEC.md` would stage the S₃
extension. Probe 01b is **horizon only**; it does not gate Probe 01 v1 and
does not appear in the current Probe Shortlist.

Pre-conditions for Probe 01b admission (not yet met):
- a defended anchor selector for the prime-conjugate slot in (ii-b/c) with
  named falsifiers for selector arbitrariness;
- an explicit S₃-equivariant invariant construction on the triple that does
  not trivialize the structural-zero discipline (i.e., not (ii-a));
- a lit re-check that the selector is not already a known explicit-formula
  device in disguise (Front-A vacuity check).

## Falsifier coupling

The bridge falsification surface plugs into the main ledger's
[Falsification Surface](../SUNDOG_V_RIEMANN.md) in named ways:

- **Mode 2** (isotropy v0.3 structural failure) fires if Path (i) is
  admitted and the parity-decomposition under reflection is degenerate
  (one sector empty) on the registered window.
- **Mode 2 → Path (iii) quarantine** fires if neither Path (i) nor Path (ii)
  admits at execution time.
- **Mode 5** (domain leakage / scope creep) fires if a Path (i) result is
  silently rewritten as a Path (ii) or v0.3h-strength claim after seeing the
  output. The named guard: the "Allowed outcome language" forbids language
  that would smuggle in either upgrade.

## Cross-references

- [`../SUNDOG_V_RIEMANN.md`](../SUNDOG_V_RIEMANN.md) — main ledger; this
  note resolves the bridge gap surfaced under Falsification Mode 2 and
  Front A apparatus list.
- [`../RIEMANN_LITPASS_MEMO.md`](../RIEMANN_LITPASS_MEMO.md) — lit-pass memo
  Track C; bridge problem is named there.
- [`../prereg/riemann/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../prereg/riemann/P0_DOMAIN_AND_RECEIPT_LOCK.md)
  — admission requirements; the "representation bridge" field there is
  satisfied by Path (i) for Probe 01 v1.
- [`PROBE_01_ISOTROPY_ZERO_PAIRS_SPEC.md`](PROBE_01_ISOTROPY_ZERO_PAIRS_SPEC.md)
  — needs Path (i) re-scoping per the "Default for Probe 01 v1" section
  above. Update is staged in the meta-audit step, not in these notes.
- [`../isotrophy/SUNDOG_V_ISOTROPHY_KFACET.md`](../isotrophy/SUNDOG_V_ISOTROPHY_KFACET.md)
  — the apparatus being descended; load-bearing for the D3 vs Z₂ argument.

## Current state

- 2026-05-28: bridge problem named, three paths staged.
- Path (i) selected as default for Probe 01 v1, pending spec update.
- Path (ii) deferred to a future Probe 01b spec, not currently in flight.
- Path (iii) reserved as the named-quarantine receipt category.
- No execution.
