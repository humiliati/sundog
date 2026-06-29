# The Parity Dictionary — toy theorem ↔ Sarnak/Chowla (a NOTE, explicitly not a proof)

**Parity-barrier hooks slate** (`PARITY_BARRIER_HOOKS_SLATE.md`), Rank-1 **P-1**, the
dictionary half. The machine-checked half lives in the public Lean development:
`sundogcert/Sundogcert/ParityNoSufficientStat.lean` (axiom-clean, gated in
`Sundogcert/AxiomAudit.lean`).

> **Status of this document.** This is a *definitions-level correspondence*, not a
> theorem and not a reduction. It records *why* the lab's H9-strong toy and the
> arithmetic parity barrier are the **same object class**, and — equally important —
> *where the correspondence stops*. Nothing here is proved about the Liouville
> function `λ`. Sarnak's conjecture and Chowla's conjecture are named as the
> **imported wall**; they remain open and are not touched.

---

## 1. What is actually PROVED (the toy, in Lean)

The H9-strong toy: uniform bits `b : Fin n → ZMod 2`. The readout latent rides the
**total parity** `T b = ∑ i, b i` (the causal state — it needs the whole history). A
"finite-order statistic" is a **partial parity** `pS A b = ∑ i∈A, b i` over a fixed
index set `A`.

| Lean name (`Sundog.ParityNoSufficientStat`) | Statement |
| --- | --- |
| `parity_split` | `T b = pS A b + pS Aᶜ b` — total parity = partial ⊕ complementary. |
| `pS_flip` | flipping a coordinate `j ∉ A` leaves the partial parity over `A` fixed. |
| `pS_compl_flip` | the same flip toggles the complementary parity — it is a free fair coin. |
| `partial_parity_underdetermines_total` | for any proper `A` and any `b`, the flip gives `b'` with the SAME partial parity but the OPPOSITE total parity. |
| `partial_not_sufficient` | **headline** — for `A ≠ univ` there exist two inputs agreeing on the partial parity yet disagreeing on the total: a finite-order partial parity is **not a sufficient statistic** for the full-history total parity. |

The mechanism is purely parity-structural: because `A` is proper, its complement is
non-empty, and the complementary parity randomizes the total no matter what the partial
parity is. This is a real theorem **about the toy**, where the driver is uniform
constructed bits we own.

---

## 2. The dictionary (toy ↔ arithmetic)

| Toy object (PROVED) | Arithmetic analog (IMPORTED, open) |
| --- | --- |
| uniform bits `b : Fin n → ZMod 2` | the multiplicative driver: `Ω(n) mod 2`, built from the primes |
| total parity `T b` (causal state) | the Liouville function `λ(n) = (−1)^Ω(n)` |
| partial parity `pS A b` over finite `A` (a finite-order statistic) | a finite-complexity / zero-entropy observable (the sieve-accessible part) |
| `partial_not_sufficient` (no proper-subset parity determines `T`) | **Selberg parity problem** — sieves alone cannot fix the sign of `λ` |
| complementary parity `pS Aᶜ b` = free fair coin (`pS_compl_flip`) | the **parity content** no zero-entropy observable can see |
| `parity_split`: `T = pS A + pS Aᶜ` | splitting `λ` into a sieve-visible part and a parity-invisible part |
| autocorrelations of `T` vanish under resampling (P-2 own-R² ≈ 0) | **Chowla**: `∑ λ(n)λ(n+h₁)…λ(n+hₖ) = o(N)` |
| `T` disjoint from any finite-order predictor (P-2 ladder flat) | **Sarnak**: `λ` disjoint from every zero-entropy sequence |

- **Sarnak's conjecture** = the load-bearing *determine* statement for arithmetic
  parity: `λ` correlates with **no** deterministic (zero-entropy) sequence. The toy's
  `partial_not_sufficient` is the finite, provable **shadow** of this — "no finite-order
  statistic determines the parity" — with `zero-entropy / deterministic` standing in for
  `finite-order`.
- **Chowla's conjecture** = the *self-correlations → 0* statement. P-2's empirical
  result (own-R²(λ, k) ≈ 0 for every accessible `k`, linear and nonlinear) is the
  computational face of Chowla at `N = 10⁷`; see `PARITY_BARRIER_HOOKS_SLATE.md` ▸ P-2.

---

## 3. Where the correspondence STOPS (the non-transfer fence)

**The toy's positive does NOT transfer to `λ`.** The Lean proof of
`partial_not_sufficient` works because the complementary bits are a **free, uniform fair
coin** — a fact we may *assert* for the toy because we constructed its driver. For `λ`,
the driver is the **primes**, whose causal state we cannot compute; asserting the
analogous "the parity-invisible part is free" *is precisely the open content* of
Sarnak/Chowla. The toy is legible **because we own its driver**; `λ` is a wall **because
we do not**.

Consequently:

1. **No claim is made about `λ`.** The theorem quantifies over uniform `b`, not over the
   Liouville sequence. The dictionary is a map of analogies, not a deduction.
2. **Breaching the wall would need a parity-sensitive Type-II / bilinear estimate**
   (Friedlander–Iwaniec, Heath-Brown, Bombieri's asymptotic sieve). The lab's
   determine-decoder is **parity-blind** (sieve-side) and contributes none — see the P-4
   kill record in the slate.
3. **Hard fence.** Under no circumstance is any of this to be presented as progress on
   twin primes, bounded gaps, Sarnak, Chowla, or the parity barrier. The deliverable is a
   *reframe + instrumentation*: a machine-checked toy theorem (P-1) and an empirical
   confirmation at accessible scale (P-2), both of which **confirm the barrier and cannot
   breach it**.

---

## 4. Cross-references

- Lean core: `sundogcert/Sundogcert/ParityNoSufficientStat.lean`; axiom gate:
  `sundogcert/Sundogcert/AxiomAudit.lean` (`parity_split`, `partial_not_sufficient`).
- Empirical companion: `docs/parity/PARITY_BARRIER_HOOKS_SLATE.md` ▸ P-2 RESULT;
  `scripts/parity_p2_liouville_ordermeter.py`.
- Seed: the H9-strong result in `docs/atlas/H9S_EPSILON_MACHINE_RESULT.md` /
  `SUNDOG_V_SHADOW.md` (Shadow-Invertibility Phase-5).
