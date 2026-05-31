# Riemann Q2 — Parity / Reflection Identity Proof (Category-A resolution)

## Header

- Receipt id: `2026-05-30_Q2_parity_identity_proof`
- Resolves: External Review Packet **Question 2** — "does an odd statistic
  summed over a symmetric `±γ` zero multiset amount to an identity / linearity
  fact rather than a Sundog structural zero?"
- Lanes covered: **C1 explicit-formula** (`C1-O` odd row) and **Probe 01
  Path (i)** (Z2 reflection feature map).
- Type: desk proof (no run). Self-contained; needs no external sign-off.
- Author / runner: Claude (Opus 4.8)
- Source framing: `RIEMANN_C1_CELLSET_V0.md` §"Linearity Verdict";
  `receipts/2026-05-28_probe01_pathi_parity_decomposition.md` §v1.1.

## Claim resolved

The packet asks whether the two parity nulls are *identities* (forced by the
symmetry of the construction) rather than empirical structural-zero edges. They
are identities. Two short proofs follow, one per lane. Neither uses RH, the
location of the zeros, or any arithmetic input.

## Standing fact (unconditional symmetry of the ordinate multiset)

Because `ζ` has real Dirichlet coefficients, `ζ(s̄) = conj(ζ(s))`, so the
nontrivial zeros are symmetric about the real axis: if `ρ = β + iγ` is a zero,
so is `ρ̄ = β − iγ`. Hence the ordinate multiset

```
Γ = { γ : ζ(β + iγ) = 0 }   is invariant under   γ ↦ −γ.
```

This is **unconditional** — RH would put `β = 1/2`, but the `±γ` symmetry holds
regardless. Both probes register the symmetric window `Γ_N = {±γ_1, …, ±γ_N}`
(first `N = 5000` positive ordinates, mirrored), so the symmetry is exact on the
registered data, not approximate.

## Proof A — explicit-formula linearity (`C1-O` lane)

The smoothed explicit formula evaluates a **linear** functional of the test
function on the zero side:

```
Z[h] = Σ_{γ ∈ Γ} h(γ).
```

Split `h` into parity components (the cell set's own definitions):

```
h_E(t) = (h(t) + h(−t)) / 2 ,   h_O(t) = (h(t) − h(−t)) / 2 .
```

Pair each ordinate with its mirror and use `h_O(−t) = −h_O(t)`:

```
Z[h_O] = Σ_{γ ∈ Γ} h_O(γ)
       = Σ_{γ > 0} [ h_O(γ) + h_O(−γ) ]
       = Σ_{γ > 0} [ h_O(γ) − h_O(γ) ]
       = 0.
```

The cancellation is **term-by-term and exact**. It consumes only (i) the `±γ`
symmetry of `Γ` and (ii) the oddness of `h_O`. The prime / archimedean side of
the explicit formula never enters, so no arithmetic content is tested. The
odd-sector row `C1-O` is therefore identically zero "before any arithmetic
enters," exactly as the cell-set linearity addendum states. Any structural-zero
content can live only in `Z[h_E]`, which the odd projector annihilates by
construction — so the odd projection **cannot in principle** expose a
structural-zero edge. ∎

## Proof B — reflection-feature invariance (Probe 01 Path (i) lane)

The registered feature map on a consecutive pair `(γ_i, γ_{i+1})` is
`φ = (g, s)` with gap `g = γ_{i+1} − γ_i` and unfolded spacing
`s = g · ρ(center)`, `center = (γ_i + γ_{i+1})/2`, density
`ρ(t) = log(|t| / 2π) / 2π`. The Z2 action is the reflection `R : γ ↦ −γ`,
sending `(γ_i, γ_{i+1}) ↦ (−γ_{i+1}, −γ_i)`. Then:

```
gap     ↦ (−γ_i) − (−γ_{i+1}) = γ_{i+1} − γ_i = g        (invariant)
center  ↦ −center , so |center| invariant , so ρ invariant
s       ↦ g · ρ(center) = s                               (invariant)
```

Hence the reflection residual `φ − R·φ = (g − g, s − s) = 0` **identically,
term-by-term** — which is precisely the Probe 01 v1.1 receipt's *measured* "max
reflection residual = 0 … identity-zero by construction" (not an empirical
near-zero). The only nonzero odd coordinate in that run is the signed height
`center / maxHeight`, i.e. the reflection coordinate itself: trivially odd, and
carrying no pair-structure information. ∎

## Unifying statement

Both lanes are one elementary identity:

> An odd function summed over an origin-symmetric multiset vanishes; equivalently,
> a reflection-invariant feature has zero reflection residual.

The zero is a property of the **symmetry of the construction**, not of the
arithmetic of the zeta zeros, and is independent of RH.

## What this does and does not settle

- **Settles (Q2):** the parity / reflection nulls are identity / linearity
  facts, not structural-zero edges. Affirmative, provable, no reviewer needed.
- **Does NOT touch:** the even-sector residual `C1-E` (the parity projector
  cannot see it), nor the nonlinear Probe 05 reversibility null (that is Q1).
  The only honest residue of the *parity* construction is whether the **even**
  sector hides structure — a separate probe, not a Q2 matter.

## Verdict / disposition

Q2 → **Category A, RESOLVED IN-HOUSE (proved).** Drop from the external-review
ask. This upgrades the packet's own pre-registered dispositions
`R-C1-NEG-A` (Front-A vacuity) and the Probe 01 identity-zero precedent from
*expected* to *proved*. The packet's odd-sector rows (`C1-O`, the odd part of
`C1-M`, the Probe 01 reflection residuals) must remain labeled identity-zero
negative controls and must never be presented as empirical evidence
(`R-C1-NEG-B` laundering guard).
