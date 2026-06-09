# Sundog Lean Certificate - Five Machine-Checked Cores

> **Deductive complement to the public Sundog lanes.** The public Lean repo now
> carries five worked examples of the same discipline: machine-check the
> deductive core, then name the imported wall. The first is the
> [P-vs-NP certificate-syndrome lane](SUNDOG_V_P_V_NP.md); the second is a
> real-analysis shadow-decay example; the third is the halo minimum-deviation
> geometry behind the 22-degree halo; the fourth is the Aharonov-Bohm
> gauge-invariance (a gradient's closed-loop circulation is zero); the fifth is
> the general characteristic-function law behind the shadow decay (real analysis
> again — five examples across four kinds of math).

**Public and reproducible:**
[`github.com/humiliati/sundogcert`](https://github.com/humiliati/sundogcert) -
`lake build` re-certifies every theorem; `#print axioms` shows only
`[propext, Classical.choice, Quot.sound]` (no `sorry`, no `native_decide`, no
trusted compiler step). Lean `v4.30.0`, mathlib `v4.30.0`.

Working hook:

> A claim gets smaller and cleaner when its proof goes into Lean and its
> assumptions stay outside the proof, named in plain view.

## Five worked examples

| Lean surface | kind of math | checked deductive core | imported wall |
|---|---|---|---|
| `Certificate` / `Instance` / `Scaling` / `Looseness` / `Degradation` / `CheckCost` | finite-field algebra | syndrome-certificate soundness, exact algebraic lossiness, reject-bound behavior, and check-cost scaling | decoding hardness / SIS one-wayness |
| `ShadowDecay` | real analysis | a lossy averaged shadow washes out a continuous variable while keeping a shared discrete label | that a real system instantiates the averaging model |
| `HaloGeometry` | geometric optics / calculus | `dev_value`, `min_deviation_stationary`, and `min_deviation_isLocalMin` (the symmetric ray is a genuine minimum) | 60-degree ice-prism geometry, measured `n ~= 1.31`, Snell refraction, ray exit, and the observed bright ring at the deviation extremum |
| `FaradayAB` | vector calculus / topology | `gauge_circulation_zero`, `gauge_integrand_eq`, and `gauge_invariant_loop` — a gradient (gauge) field's closed-loop circulation is zero, so the loop observable is gauge-invariant | that the vector potential `A` enters as a loop integral, `grad chi` is the gauge freedom, the Aharonov-Bohm phase / Faraday loop EMF *is* that integral, and the loop encloses the flux (the `H^1` period) |
| `ShadowDecayGeneral` | real analysis (generalizing example 2) | `shadow_decay_charFun` + the determine/resist corollaries — averaging over any probability measure factors through its characteristic function; resist ⟺ `‖charFun‖→0` (Riemann–Lebesgue), determine ⟺ a finite centered mean (two independent conditions); the Gaussian discharges both | that a real system instantiates the characteristic-function averaging; the Cauchy separator is named, not built (mathlib lacks `charFun_cauchy`) |

The third example matters because it breaks the first two examples' shared
shape. The certificate and shadow-decay examples both involve a lossy shadow
that keeps one invariant while losing another. The halo proof is not that. It
is a pure extremization statement: the deviation

```text
dev n A r = arcsin (n * sin r) + arcsin (n * sin (A - r)) - A
```

is stationary at the symmetric ray `r = A/2`, under the explicit no-total-
internal-reflection differentiability hypothesis. So the demonstrated method is
not "one motif formalized once"; it is a portable way to separate deduction
from import across different mathematical shapes.

The fourth example, the Aharonov-Bohm gauge-invariance, *rejoins* that shared
shape — but topologically: a gradient (gauge) field's circulation around a
closed loop is exactly zero, so the loop observable keeps only the enclosed
flux (an `H^1` period) while the gauge freedom `grad chi` washes out. The fifth,
the general characteristic-function law, *is* the shadow-decay shape stated in
full — it names which spectral condition of the averaging measure governs each
half. So four of the five share the deeper shape across a finite-field coset, a
measure-theoretic label and its characteristic-function spectrum, and a
topological period, and the halo stands apart as the pure-extremization breaker
— the method is tied neither to one motif nor to one mathematical structure.

## P-vs-NP certificate core

The first Lean surface is still the deductive core of the P-vs-NP
certificate-syndrome lane. It machine-checks the certificate's **soundness and
lossiness** in Lean 4 - `sorry`-free, axiom-clean, **referee-free**. The kernel
re-checks every theorem in seconds, so the *validity* of this core is
author-independent. The decoding-hardness assumption (information-set decoding
/ SIS) is **imported, not proven** - Lean certifies the deductive core, never
the hardness.

## What is machine-checked in the certificate

- **Lossiness by algebra.** The syndrome `H(sG + e) = He` is independent of the
  secret `s` (`syndrome_independent_of_secret`); every message maps to the same
  syndrome; there are `|F|^k` bodies per syndrome (`secret_bits_lost`). The
  shadow discards `k*log|F|` bits - forced by the algebra, not assumed.
- **Soundness.** `accept -> Safe` - the only route to *accept* is an exhibited
  light witness, which *is* the proof (`accept_sound`); no accepted body is
  unsafe (`no_passing_unsafe`); `reject -> not Safe` under a sound lower bound
  (`reject_sound`).

The trust surface is small: the scheme definitions and the meaning of *Safe*.
Everything above them is kernel-checked. Peer review shrinks from "trust the
proof" to "audit the statement."

## The wall, named

Lean certifies **soundness + lossiness only** for the certificate lane. The
certificate's security rests on a decoding-hardness assumption that is
**imported, not proven** - hardness is not a mathlib theorem. Every
"Lean-verified" claim here means the deductive core, never the hardness.

## The reject bound, fully characterized

The load-bearing reject bound `colWeightLb` is pinned down from every direction,
each fact kernel-verified:

| regime | behavior | theorem |
|---|---|---|
| any basis | **sound** - never exceeds the true witness weight | `colWeightLb_sound` |
| uniform `H` | **tight** - equals the true distance; reject threshold scales linearly, `tau = n/2 - 1` | `scaling_law` |
| denser `H`, same code | **loose** - collapses to `0`, purely from the basis | `looseness` |
| general | capped by `||syndrome|| / density` | `colWeightLb_le_card_div` |

Items (loose) and (general) are **completeness** phenomena, not soundness
breaks: a collapsed bound still never over-claims - it quarantines where it
cannot reject. Soundness never depends on the basis; only the bound's *strength*
does.

## The frontier

A *cheap, basis-robust, tight* reject bound - one that does not degrade when the
parity-check is chosen adversarially - would return the true minimum coset
weight on every basis. That **would be a fast decoder**: it would solve the very
problem (information-set decoding) whose hardness the certificate imports. So
the basis-dependence of `colWeightLb` is not a defect to be patched away - it is
the visible edge of the hardness assumption. The honest open question is
quantitative: how large is the gap between a cheap bound and the true coset
weight, as a function of the decoding margin.

## Relation to the P-vs-NP lane

The certificate-syndrome receipts (v1-v6) measure the **empirical** side -
cheaper to check than to find (op-count cost certificate `0.949 <= 1.0`), safety
green. This ledger is the **deductive** complement: the soundness and lossiness
those receipts rely on are machine-checked, axiom-clean. The two are orthogonal,
and neither proves the decoding hardness - which both import.

The newer `ShadowDecay`, `ShadowDecayGeneral`, `HaloGeometry`, and `FaradayAB`
modules are method demonstrations, not extra P-vs-NP evidence. They show that the
same public Lean discipline spans finite-field algebra, real analysis, geometric
optics, and vector calculus / topology without turning any one imported wall into
a theorem.

## Status

**PUBLIC, REPRODUCIBLE, FIVE-EXAMPLE LEAN METHOD CORE.** The P-vs-NP
certificate soundness + lossiness are machine-checked; the shadow-decay (the
concrete Gaussian decay and its general characteristic-function law), halo
minimum-deviation, and Aharonov-Bohm gauge-invariance examples extend the method
to three more mathematical shapes — five worked examples across four kinds of
math. Hardness, model realization, physical optics,
and the physical gauge field remain imported. Not a cryptographic one-wayness
claim; not a claim about P versus NP; not a claim that Lean proves the sky
realizes the halo or that nature realizes the Aharonov-Bohm effect.
