# Parity-barrier hooks slate (2026-06-28)

> Generate→adversarially-vet→rank, in the house format (cf. `docs/atlas/FRESH_HYPOTHESES_SLATE_2.md`). The
> seed: the H9-strong result (`docs/atlas/H9S_EPSILON_MACHINE_RESULT.md`) found a determine-type latent
> **load-bearing vs ALL finite-order surrogates** by riding the **parity** of a driving process — a quantity
> with no finite-order sufficient statistic. That is the same object class as the **Liouville function**
> `λ(n) = (−1)^Ω(n)` (the parity of the prime-factor count) and the **Selberg parity problem** of sieve theory.
> This slate asks what genuine, bounded, NON-crank hooks the Sundog framework has into the parity barrier.
> **Framing discipline (load-bearing): this lane REFRAMES and INSTRUMENTS the barrier; it does NOT attempt to
> breach it. The parity barrier / Chowla / Sarnak is the named IMPORTED WALL.** NOT public-eligible. A clean
> result either way (especially a clean NULL or a clean "instrument confirms the barrier") is bankable.

## The bridge (a definitions-level correspondence, NOT a theorem)
The rhyme is not loose — λ(n) is literally a parity-of-a-driver, exactly the H9-strong construction with the
prime-factorization process in place of fair coins:

| H9-strong artifact (`epsilon_machine_shadow.py`) | analytic number theory |
|---|---|
| parity causal state `P_t = b_1⊕…⊕b_t` (no finite-order sufficient statistic) | `λ(n) = (−1)^Ω(n)` / the Möbius parity |
| order-k surrogate ladder (structured, low-complexity foils) | sieve axioms / low-complexity (zero-entropy) dynamical observables |
| "load-bearing vs ALL finite order" | **Sarnak's conjecture** — λ disjoint from every zero-entropy sequence |
| determine signature + self-correlations → 0 | **Chowla's conjecture** — λ behaves like random ±1 |
| negative-control "order-meter" (finite-order parity IS caught at k=d−1) | sieves DO capture the finite-complexity (congruence/major-arc) part of primes; they fail ONLY on the parity content |

**The single honest sentence:** *Sarnak's conjecture is the H9-strong load-bearing-determine statement for
arithmetic parity, against the zero-entropy surrogate class — and it is OPEN because, unlike our toy, the driver
is the primes (we cannot compute the causal state).*

**The barrier, stated for the record.** Selberg's parity problem: a sieve — a parity-blind, structured weight —
cannot distinguish Ω(n) even from Ω(n) odd, so it returns an upper bound a factor ≈2 too large and no matching
lower bound (the "can't get more than ~50%" intuition: the parity-blind weight splits its mass evenly across the
two parity classes). This is why sieves reach **P₂** (Chen: every large even = p + P₂; infinitely many p with
p+2 a P₂) but never **P₁ = prime**. Everything known to BREACH the barrier injected a parity-sensitive
**bilinear / Type-II** estimate (Friedlander–Iwaniec, primes a²+b⁴; Heath-Brown, a³+2b³; via Bombieri's
asymptotic sieve) — never a cleverer sieve. **The bar for any Sundog hook to be more than reframing: it must
contribute a Type-II ingredient. The determine-shadow decoder does not (see the kill record, P-4).**

## Survivors (ranked)

### Rank 1 — P-1: the parity barrier as a no-finite-order-sufficient-statistic theorem (strength 6)
**Lean-formalize the H9-strong toy and pin the Sarnak/Chowla dictionary as an explicit definitions-level
correspondence, with the parity barrier as the named wall.** The contentful, buildable, fully-honest hook.
- **Core (machine-checkable):** in `sundogcert`, formalize "a strictly-sofic PARITY process has no finite-order
  sufficient statistic for the readout latent" — i.e. for any finite index set `S`, the parity over `S` is
  independent of the readout `c` because the complementary parity is an independent fair coin (the
  complementary-parity argument the red-team verified empirically at L=200k in H9-strong). This is the toy half;
  it is a real theorem about the toy, not about λ.
- **The dictionary (a NOTE, explicitly not a proof):** a definitions-level map from {causal state, order-k
  surrogate ladder, load-bearing-vs-all-finite-order} to {λ(n), zero-entropy observables, Sarnak/Chowla}. State
  Sarnak as the IMPORTED WALL; state that the toy's positive does NOT transfer (we own the toy's driver, not the
  primes').
- **Imported wall (named, NOT proved):** Sarnak's conjecture (λ ⟂ zero-entropy systems); Chowla (Tao's
  logarithmic two-point Chowla + Matomäki–Radziwiłł is the current frontier).
- **Attack:** `sundogcert/Sundogcert/ParityNoSufficientStat.lean` mirroring the FoldCancellation / parity cores;
  axiom audit; the dictionary as a fenced `docs/parity/PARITY_DICTIONARY.md`. **Kill if** the "correspondence"
  smuggles in ANY claim about λ itself (the moment it does, it is crank); if the Lean core needs a sorry; or if
  the no-sufficient-statistic statement collapses to a restatement of independence of fair coins with no parity
  content.
- **Why it survived:** it is the only candidate that is simultaneously buildable in bounded hours, machine-
  checkable on the toy half, and structurally incapable of overclaiming (the wall is named, the non-transfer is
  explicit). Caps at 6 — not 6.5+ — because the Lean core is a modest extension of the existing parity work and
  the dictionary is exposition, not new mathematics.

### Rank 2 — P-2: the Liouville "order-meter" experiment (strength 5.5)
**Run the exact H9-strong order-k surrogate ladder on the REAL Liouville sequence `λ(n)`, n ≤ ~10⁷–10⁸.** A
genuinely runnable, falsifiable empirical confirmation that the toy is faithful and the barrier is real at
accessible scale.
- **Prediction (the barrier holds):** every structured order-k predictor stays at own-R² ≈ 0 across the ladder —
  λ is invisible at all accessible orders, the empirical face of "no finite-order sufficient statistic."
- **The order-meter control (the apparatus is not broken):** inject a KNOWN finite-order arithmetic signal (e.g.
  the residue n mod small q, or a short-block factor-count feature) — the ladder MUST catch it at the
  corresponding order, exactly as the H9-strong negative control crossed at k=d−1. This proves a null on λ means
  "invisible," not "dead apparatus."
- **Imported wall (named, NOT proved):** Chowla (the conjecture this experiment is consistent-with, never proves).
- **Attack:** sieve λ(n) up to N (linear sieve of Ω parity); reuse `epsilon_machine_shadow.own_r2` + a Markov /
  block-feature ladder; report own-R² vs k + the arithmetic-control crossing. **Kill / interest-flip:** if some
  order-k predictor beats chance on λ, it MUST be shown to be a known finite-n bias (Pólya / Liouville-summatory
  drift `L(n)`, Mertens-style fluctuation, small-prime density) BEFORE any excitement — almost certainly an
  artifact, not a barrier breach. **Honest label: confirms the barrier empirically; cannot breach it.**
- **Why it survived:** real data, real falsifiability, directly exercises the lab's own apparatus on the actual
  arithmetic object, and the expected clean confirmation is a bankable "the toy is faithful" receipt. Caps at
  5.5 because the expected outcome is confirmatory (a null-shaped positive), not a discovery.

### Rank 3 — P-3: "where the pigeonhole stalls" instrument (strength 4.5)
**Instrument the Maynard–Tao tuple pigeonhole at small admissible k-tuples and make the factor-of-2 parity loss
concrete — show numerically the exact step where P₂→P₁ is blocked.** A didactic receipt, not an attack.
- **Content:** take a small admissible tuple, run the multidimensional-sieve weight pigeonhole (GPY/Maynard
  form), and display where the bound forces "≥1 almost-prime in the tuple" (→ bounded gaps) but cannot force
  "exactly prime" — the parity barrier's location, visualized.
- **Imported wall (named, NOT proved):** the parity barrier as the reason Maynard–Tao yields bounded gaps, not
  twins (the user's pigeonhole instinct is correct AND terminates exactly here).
- **Attack:** small-tuple numeric sieve weights (numpy); a receipt `docs/parity/PIGEONHOLE_STALL.md`. **Kill if**
  it drifts from "instrument the stall" toward "close the gap" — plaster it with "does NOT attempt twin primes."
- **Why it survived (barely):** purely expository value — it makes the barrier tangible and pre-empts the crank
  reading of the user's own (correct) pigeonhole intuition. Lowest strength: it proves nothing, it teaches.

## The kill record (the discipline is part of the deliverable)

- **P-4 — "does the determine-shadow decoder add a Type-II / bilinear term?" — KILLED (strength 2).** The Shadow
  framework's engine is RECOVERY of a latent from a lossy shadow: a structured, parity-BLIND decoder, i.e. a
  sieve-shaped operation. It therefore lives on the WRONG side of the barrier by construction. Breaching the
  barrier requires a parity-sensitive bilinear (Type-II) estimate (Friedlander–Iwaniec / Heath-Brown / Bombieri
  asymptotic sieve); the determine-decoder contributes none. **KILLED, and the kill is a deliverable** — it is
  exactly the overreach that would turn this lane crank. (Reopen ONLY if a concrete Type-II hook is exhibited,
  which none of the lab's assets currently provides.)
- **"Transfer the H9-strong positive to λ" — KILLED (self-refuting).** The H9-strong win required OWNING the
  driver (fair coins, computable full-history parity). λ's driver is the primes; the causal state is exactly what
  we cannot compute — that is the entire content of Chowla/Sarnak. The toy models WHY the barrier is hard; its
  positive does not transfer. Any candidate that quietly assumes the transfer is dead on arrival.
- **"Shadow-invertibility ⟹ a statement about prime gaps" — KILLED (category error).** The Shadow law is about
  recoverability of a latent from an ensemble shadow; prime gaps are not a lossy-shadow recovery problem. No
  honest map exists; the surface resemblance is the word "determine."
- **"Möbius randomness as a charFun-resist" — KILLED (wrong axis).** charFun-resist is the H8 SNAPSHOT-shadow
  phase phenomenon (washes under jitter); λ is a DETERMINE-side, trajectory/causal-state object (H9 side). Filing
  it under resist is the same axis confusion the H8 capstone corrected.

## Recommendation
**P-1 (the no-finite-order-sufficient-statistic theorem + named-wall dictionary)** is the strongest, most honest,
most buildable: a real machine-checkable core on the toy half, a definitions-level dictionary that is
structurally incapable of overclaiming, and Sarnak/Chowla correctly imported as the wall. **P-2 (the Liouville
order-meter)** is the best empirical companion — it exercises the lab's own apparatus on the actual arithmetic
object and the expected clean confirmation is bankable. Do P-1 + P-2 together; treat P-3 as optional exposition.
**Under no circumstance present any of this as progress on twin primes or the barrier — the wall is imported, and
the kill record (P-4) is the proof of discipline.**

## Honest scope & boundaries
- This lane is **REFRAME + INSTRUMENT**, frozen-as-portfolio, NOT public-eligible. It cannot and does not attempt
  to prove twin primes, Chowla, or Sarnak, or to breach the parity barrier.
- Everything load-bearing rests on **imported walls named in-line** (Sarnak, Chowla, the parity barrier,
  Friedlander–Iwaniec/Heath-Brown for what breaching takes). The lab contributes the toy, the apparatus, and the
  dictionary — not a Type-II estimate.
- Attribution: Selberg (parity problem); Bombieri (asymptotic sieve / parity principle); Chen (P₂);
  Friedlander–Iwaniec (a²+b⁴), Heath-Brown (a³+2b³) (parity-barrier breaches via Type-II); Chowla; Sarnak;
  Matomäki–Radziwiłł, Tao (logarithmic Chowla frontier); Goldston–Pintz–Yıldırım, Zhang, Maynard–Tao (bounded
  gaps / the pigeonhole); Crutchfield & Young (ε-machines / causal states); the Sundog H8/H9 lineage.
