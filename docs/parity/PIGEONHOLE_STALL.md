# Where the Pigeonhole Stalls — the parity barrier, made concrete (P-3)

**Parity-barrier hooks slate** (`PARITY_BARRIER_HOOKS_SLATE.md`), Rank-3 **P-3**. A
**didactic receipt**, not an attack. It locates — on real integers, with exact counts —
the point where the Maynard–Tao tuple pigeonhole forces "almost prime" but the parity
barrier blocks "prime," and contrasts it with the congruence obstruction a sieve *can*
see.

> **Hard fence.** This document does **not** attempt twin primes, bounded gaps, Chowla,
> Sarnak, or to breach the parity barrier. Everything load-bearing is an **imported,
> named wall**. The deliverable is *visualization of the stall*, nothing more. Receipt:
> `scripts/parity_p3_pigeonhole_stall.py`.

---

## 1. The Maynard–Tao pigeonhole (setup; the arithmetic is imported)

Fix an admissible `k`-tuple `H = {h₁,…,h_k}`. Maynard–Tao attach sieve weights `w_n ≥ 0`
(squares of divisor sums supported on `∏ dᵢ < R = N^{θ/2}`, `θ` = level of distribution)
and form two sums over `N ≤ n < 2N`:

- `S₁ = ∑_n w_n` (total weight);
- `S₂ = ∑_n (∑_{i=1}^k 𝟙[n+hᵢ prime]) · w_n` (weight × number of primes in the tuple).

**The pigeonhole.** If `S₂ > m · S₁` then infinitely many `n` have **≥ m+1** of the
`n+hᵢ` prime — otherwise every `n` contributes ≤ m and `S₂ ≤ m·S₁`. Asymptotically
`S₂/S₁ → (θ/2)·M_k`, where `M_k = sup_F (∑ᵢ Jᵢ(F))/I(F)` over smooth `F` on the simplex
(`I = ∫F²`, `Jᵢ = ∫(∫F dxᵢ)²`).

**Imported, named (NOT proved here):** Bombieri–Vinogradov gives `θ = 1/2`
unconditionally; Maynard (Annals 2015) proved `M_k ≥ log k − 2 → ∞`; the prime
indicator's level of distribution is what lets `S₂` count *actual primes*.

---

## 2. Stall 1 — tuple size: why not twins

`M_k` grows only like `log k`, so the pigeonhole can be triggered (`S₂/S₁` above a fixed
threshold) **only for `k` large**. The twin tuple is `k = 2`, `H = {0,2}` — and `M₂` is a
small constant far below any useful threshold. **You can never force both `n` and `n+2`
prime.** The pigeonhole yields *two primes somewhere among many shifts within a bounded
window* (bounded gaps), never a *specified* pair. Twins are out of reach by the size of
`M_k`, not by anything the method can tighten.

---

## 3. Stall 2 — the parity principle, instrumented exactly

The deeper reason a sieve cannot lower-bound primes in a *single* sparse sequence: a sieve
of level `D = N^θ` only constrains prime factors `≤ D`. Among the **survivors** in `[N,2N]`
(smallest prime factor `> D`), it computes the **count** but is **blind to the parity of
`Ω(n)`**. For `θ > 1/3` the survivors are exactly primes (`P₁`, `Ω=1`) and semiprimes
(`P₂`, `Ω=2`), so with `λ(n) = (−1)^{Ω(n)}`:

```
count       = P₁ + P₂           (a level-D divisor sum — sieve-computable)
parity_sum  = Σ λ(n) = P₂ − P₁  (NOT a level-D quantity — the parity barrier)
⇒  P₁ = (count − parity_sum) / 2
```

**Measured on `[10⁶, 2·10⁶]`** (`scripts/parity_p3_pigeonhole_stall.py`; sanity:
`π(2N)−π(N) = 70435`):

| θ | D = N^θ | survivors = count | parity_sum = Σλ | P₁ | P₂ | P₃ | (count−Σλ)/2 |
|---|---|---|---|---|---|---|---|
| 0.50 | 1000 | 74197 | −66673 | 70435 | 3762 | 0 | **70435 = P₁ (exact)** |
| 0.40 | 251 | 100725 | −40145 | 70435 | 30290 | 0 | **70435 = P₁ (exact)** |
| 0.34 | 110 | 116386 | −24508 | 70435 | 45939 | 12 | 70447 ≠ P₁ (P₃ enters: >2 buckets) |
| 0.30 | 63 | 131523 | −13713 | 70435 | 58905 | 2183 | 72618 ≠ P₁ (P₃ enters) |

**The stall, in one line.** The sieve hands you `count`. To extract primes you need
`P₁ = (count − parity_sum)/2`, and `parity_sum = Σλ` is precisely the parity-sensitive sum
the sieve is structurally blind to. That dividing **`/2` is the factor-of-2 loss**: with
`count` alone you get only the `P_{≤2}` ("almost prime") bound — the same ceiling that
limited classical sieves to **Chen's `P₂`**.

And it gets *worse* as the sieve weakens: as `θ` drops, the almost-prime contamination
`P₂` the sieve cannot remove explodes (`3762 → 30290 → 45939 → 58905`). For `θ ≤ 1/3` even
the clean two-bucket identity breaks as `P₃` enters — more parity structure the level-`D`
sieve cannot resolve.

---

## 4. The two stalls are one wall

Classical sieves hit **Stall 2** directly and could only reach almost-primes. GPY–Maynard
*sidestep* Stall 2 by never pinning a single sequence — many shifts, claim only that *some*
`n+hᵢ` is prime (via the prime level of distribution), never which — and the price is
**Stall 1**: `k` must be large, so `k=2` (twins) is unreachable. Either way twins stay out.
The numpy table makes the parity face of that single wall tangible.

---

## 5. Contrast — the obstruction a sieve CAN see (admissibility)

A `k`-tuple is **admissible** iff for every prime `p` it misses some residue class mod `p`.
This is a *congruence/local* obstruction — exactly what sieves handle. Computed:

| H | verdict |
|---|---|
| `{0,2,4}` | **inadmissible** — covers all residues mod 3 (always contains a multiple of 3) |
| `{0,2,6}` | admissible |
| `{0,2,6,8,12}` | admissible |

The point of the contrast: admissibility is the obstruction the method *resolves*; the
parity barrier of §3 is the one it *cannot*. Knowing the difference is the whole discipline
of the lane.

---

## 6. Honest scope & imported walls

- **Non-claims.** No twin-prime, bounded-gap, Chowla, or Sarnak claim; no barrier breach.
  Breaching would require a parity-sensitive **Type-II / bilinear** estimate
  (Friedlander–Iwaniec, Heath-Brown, Bombieri's asymptotic sieve) — the lab contributes
  none (see the P-4 kill record in the slate).
- **Imported, named:** Bombieri–Vinogradov (`θ=1/2`); Maynard `M_k ≥ log k − 2`; the
  Selberg **parity principle** / Bombieri **asymptotic sieve** (the §3 blindness); Chen's
  `P₂`. This receipt only *exercises* these; it proves none of them.
- **What is genuinely the lab's:** the exact instrumentation — the identity
  `P₁ = (count − Σλ)/2` measured on real integers, which renders the factor-of-2 parity
  loss as a concrete computation rather than a slogan.
- **Companions:** P-1 (`PARITY_DICTIONARY.md` + `sundogcert` `ParityNoSufficientStat.lean`,
  the toy theorem) and P-2 (`parity_p2_liouville_ordermeter.py`, λ to 10⁷). P-3 is the
  lowest-strength, purely expository member of the slate: it proves nothing, it teaches.
