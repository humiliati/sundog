# Sundog Lean Certificate - Eight Machine-Checked Cores

> **Deductive complement to the public Sundog lanes.** The public Lean repo now
> carries eight worked examples of the same discipline: machine-check the
> deductive core, then name the imported wall. The first is the
> [P-vs-NP certificate-syndrome lane](SUNDOG_V_P_V_NP.md); the second is a
> real-analysis shadow-decay example; the third is the halo minimum-deviation
> geometry behind the 22-degree halo; the fourth is the Aharonov-Bohm
> gauge-invariance (a gradient's closed-loop circulation is zero); the fifth is
> the general characteristic-function law behind the shadow decay (real analysis
> again); the sixth is a machine-checked Karp reduction
> `3SAT <= 3DM <= X3C <= decoding` that anchors the certificate's
> decoding-hardness import to 3SAT (reduction correctness only); the seventh
> is a finite audit game — a proved-cheap full-access audit paired with
> ∀-verifier blindness of the pooled-mean channel; and the eighth is
> `Sundogcert/CircuitNet.lean` — an exact tropical / piecewise-linear
> circuit-to-ReLU compilation (`compile_eval`) with a linear depth bound
> (`compile_depth_le`) and an exact min-plus Bellman-Ford gate
> (`bellmanStep_compiles_exactly`), the ε = 0 piecewise-linear core for
> arXiv:2606.26705 Thm 3.2 / Cor 5.1, in the AxiomAudit gate. Eight examples
> across seven kinds of math; not a learnability claim and not a full
> analytic-gate approximation theorem.

**Public and reproducible:**
[`github.com/humiliati/sundogcert`](https://github.com/humiliati/sundogcert) -
`lake build` re-certifies every theorem; `#print axioms` shows only
`[propext, Classical.choice, Quot.sound]` (no `sorry`, no `native_decide`, no
trusted compiler step). Lean `v4.30.0`, mathlib `v4.30.0`.

Working hook:

> A claim gets smaller and cleaner when its proof goes into Lean and its
> assumptions stay outside the proof, named in plain view.

## Eight worked examples

| Lean surface | kind of math | checked deductive core | imported wall |
|---|---|---|---|
| `Certificate` / `Instance` / `Scaling` / `Looseness` / `Degradation` / `CheckCost` | finite-field algebra | syndrome-certificate soundness, exact algebraic lossiness, reject-bound behavior, and check-cost scaling | decoding hardness / SIS one-wayness |
| `ShadowDecay` | real analysis | a lossy averaged shadow washes out a continuous variable while keeping a shared discrete label | that a real system instantiates the averaging model |
| `HaloGeometry` | geometric optics / calculus | `dev_value`, `min_deviation_stationary`, and `min_deviation_isLocalMin` (the symmetric ray is a genuine minimum) | 60-degree ice-prism geometry, measured `n ~= 1.31`, Snell refraction, ray exit, and the observed bright ring at the deviation extremum |
| `FaradayAB` | vector calculus / topology | `gauge_circulation_zero`, `gauge_integrand_eq`, and `gauge_invariant_loop` — a gradient (gauge) field's closed-loop circulation is zero, so the loop observable is gauge-invariant | that the vector potential `A` enters as a loop integral, `grad chi` is the gauge freedom, the Aharonov-Bohm phase / Faraday loop EMF *is* that integral, and the loop encloses the flux (the `H^1` period) |
| `ShadowDecayGeneral` | real analysis (generalizing example 2) | `shadow_decay_charFun` + the determine/resist corollaries — averaging over any probability measure factors through its characteristic function; resist ⟺ `‖charFun‖→0` (Riemann–Lebesgue), determine ⟺ a finite centered mean (two independent conditions); the Gaussian discharges both | that a real system instantiates the characteristic-function averaging; the Cauchy separator is named, not built (mathlib lacks `charFun_cauchy`) |
| `SATReduction*` / `VarWheel` / `ClauseGadget` (on `MatchingNPHard` / `DecodingNPHard`) | combinatorics / computational complexity | `sat_iff_decodes` — a machine-checked Karp reduction `3SAT ≤ 3DM ≤ X3C ≤ bounded-weight GF(2) decoding`, both directions proved; a 3-CNF formula is satisfiable iff its decoding image decodes within the weight bound | the NP class, the poly-time-ness of the reduction maps, and 3SAT's own NP-hardness (Cook–Levin) — hardness imported, any "NP-hard" reading conditional on P ≠ NP, no P-vs-NP claim |
| `AuditCost` | finite decidability / audit game | `pooled_channel_blind` + `no_verifier_checks_perUnit` — a proved-cheap full-access audit (sound + complete against an adversarial reporter at `auditCost ≤ 3n+2`) paired with ∀-verifier blindness: an explicit same-mean fiber pair defeats *every* decidable channel verifier at any prescribed per-unit `δ` | that the pooled-mean channel is the operative observation model (non-vacuity proved: `n = 1` determines; the second moment separates the blind pair) |
| `CircuitNet` (`AxiomAudit`-gated) | tropical / piecewise-linear algorithmic approximation | `compile_eval` — an exact (ε = 0) compilation of tropical / piecewise-linear circuits to ReLU networks; `compile_depth_le` — a linear depth bound; `bellmanStep_compiles_exactly` — an exact min-plus / Bellman-Ford gate compiling to a ReLU sub-network. The ε = 0 piecewise-linear core for [arXiv:2606.26705](https://arxiv.org/abs/2606.26705) Thm 3.2 / Cor 5.1 | the **linear gate-count** claim, which still needs the DAG / sharing follow-up; the **analytic** reciprocal / radical gates (and any non-piecewise-linear primitive), which remain approximate / imported; learnability and the full Kratsios analytic-approximation theorem (neither is asserted by this core) |

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
half. So five of the eight share the deeper shape across a finite-field coset, a
measure-theoretic label and its characteristic-function spectrum, a topological
period, and a pooled-mean audit channel that loses every per-unit claim; the
halo stands apart as the pure-extremization breaker, and the Karp reduction —
`3SAT <= ... <= decoding` — stands apart too, a combinatorial equivalence whose
import is hardness, not a model. The method is tied neither to one motif nor to
one mathematical structure.

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

## The hardness wall, pushed inward (the reduction chain)

The decoding-hardness assumption is no longer opaque. A machine-checked Karp
reduction now connects it to the canonical NP-complete problem:

> **3SAT <= 3DM <= X3C <= bounded-weight GF(2) decoding**

is formalized end to end, its top-level correctness an `iff`
(`SATReductionMain.sat_iff_decodes`): a 3-CNF formula `phi` is satisfiable **if
and only if** the decoding instance it maps to decodes within the weight bound.
Both directions are proved - the forward builds the perfect matching from a
satisfying assignment (Garey-Johnson variable-wheel, clause, and garbage
gadgets, the leftover tips absorbed by a counted bijection), the reverse reads
an assignment back out of any perfect matching. Axiom-clean, like the rest.

What is machine-checked is the reduction's **correctness** - the many-one / Karp
equivalence between the SAT instance and its decoding image. The complexity
wrapping stays imported, because mathlib has no complexity-theory framework: the
**NP class** itself, the **poly-time-ness** of the reduction maps (each built and
proved correct, but never timed), and 3SAT's **own NP-hardness** (Cook-Levin, in
no proof assistant to date). So the certificate's "decoding is hard" import is
now *anchored* - at least as hard as 3SAT, modulo the named wrapping - while any
"NP-hard" reading stays **conditional on P != NP**. This is **not** a claim about
P versus NP, and not a proof that decoding is hard.

## Relation to the P-vs-NP lane

The certificate-syndrome receipts (v1-v6) measure the **empirical** side -
cheaper to check than to find (op-count cost certificate `0.949 <= 1.0`), safety
green. This ledger is the **deductive** complement: the soundness and lossiness
those receipts rely on are machine-checked, axiom-clean. The two are orthogonal,
and neither proves the decoding hardness - which both import.

The newer `ShadowDecay`, `ShadowDecayGeneral`, `HaloGeometry`, `FaradayAB`, and
`AuditCost` modules are method demonstrations, not extra P-vs-NP evidence. They
show that the same public Lean discipline spans finite-field algebra, real
analysis, geometric optics, vector calculus / topology, and a finite audit game
without turning any one imported wall into a theorem. The reduction-chain modules (`SATReduction*`, on `MatchingNPHard` /
`DecodingNPHard`) sit closer to this lane: they *anchor* the certificate's
decoding-hardness import to 3SAT by a checked reduction. But they too leave the
hardness imported (the NP class, poly-time-ness, and Cook-Levin), so they sharpen
the import rather than discharge it - still not P-vs-NP evidence.

## Status

**PUBLIC, REPRODUCIBLE, SEVEN-EXAMPLE LEAN METHOD CORE.** The P-vs-NP
certificate soundness + lossiness are machine-checked; the shadow-decay (the
concrete Gaussian decay and its general characteristic-function law), halo
minimum-deviation, and Aharonov-Bohm gauge-invariance examples extend the method
to three more mathematical shapes; a machine-checked Karp reduction
`3SAT <= 3DM <= X3C <= decoding` anchors the certificate's decoding-hardness
import to 3SAT; a finite audit game proves a cheap full-access audit blind to
every decidable channel verifier; and `Sundogcert/CircuitNet.lean` machine-checks
an exact tropical / piecewise-linear circuit-to-ReLU compilation (with a linear
depth bound and an exact min-plus Bellman-Ford gate) — the ε = 0
piecewise-linear core for arXiv:2606.26705 Thm 3.2 / Cor 5.1, in the AxiomAudit
gate — eight worked examples across seven kinds of math. Hardness, model
realization, physical optics, the physical gauge field, the NP-class /
poly-time / Cook-Levin complexity wrapping, the **linear gate-count** claim
(still waiting on the DAG / sharing follow-up), and the **analytic** reciprocal /
radical gates (which remain approximate / imported) all remain named walls. Not
a cryptographic one-wayness claim; not a claim about P versus NP; not a claim
that Lean proves the sky realizes the halo or that nature realizes the
Aharonov-Bohm effect; not a learnability claim; not a full analytic-gate
approximation theorem.
