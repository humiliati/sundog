# C1 Result — the Reed–Solomon evaluation certificate (a new sundogcert Lean pillar)

> **2026-06-08.** Rank-1 survivor of fresh slate #2 (`wm13nclfe`). The **evaluation-dual** of the
> parity-check/syndrome certificate, and the lab's machine-checked certificate program's newest pillar.
> Lean leg = candidate public sundogcert pillar (built + axiom-clean locally, **owner-gated**, not pushed).
> NOT public-eligible (empirical). Attribution: Reed–Solomon 1960; MacWilliams–Sloane (the two RS views);
> Guruswami–Vardy 2005 (list-decoding NP-hardness); the lab's certificate thesis.

## Headline — BOUNDED-POSITIVE

**The same Reed–Solomon code, viewed through its EVALUATION face instead of the syndrome's parity-check
face, is a second cheap-CHECK ≫ hard-FIND certificate — machine-checked axiom-clean in Lean.** `k`
evaluations DETERMINE the unique degree-`<k` message; the cheap check is sound and (within the radius)
unique; yet many corrupted words decode to the SAME message (DETERMINE the message, LOSE the corruption) —
the determining-shadow split, on a fresh mathlib core the syndrome lane never touches.

## Lean pillar (the headline) — `sundogcert/Sundogcert/RSCertificate.lean`

Lean 4.30 + mathlib v4.30; full `lake build` **0 warnings (3529 jobs)**, wired into the `Sundogcert` root;
every theorem **axiom-clean** (`[propext, Classical.choice, Quot.sound]`, no `sorryAx`).

| theorem | statement |
|---|---|
| `rs_unique` | **CORE (cheap-check soundness)**: two `natDegree < k` polys agreeing on `≥ k` distinct nodes are EQUAL — `k` evaluations DETERMINE the message. A thin specialization of mathlib `Polynomial.eq_of_natDegree_lt_card_of_eval_eq` |
| `accept_sound` | accept ⟹ Safe — the exhibited decoding IS a witness (the cheap forward check is sound) |
| `unique_decoding` | within the radius `2τ + k ≤ n`, any two valid decodings of a word are EQUAL (inclusion–exclusion on the agreement sets + `rs_unique`) |
| `corruption_fiber_nontrivial` | **the NEW determine/lose theorem**: for degree-`<k` `f` and `τ ≥ 1`, ≥2 distinct words decode to the SAME `f` (clean codeword + a 1-symbol corruption) — DETERMINE the message, LOSE the corruption |
| `nodes_distinct_cert` | node-distinctness is a one-determinant certificate (`det (vandermonde nodes) ≠ 0 ↔ Injective nodes`, via `det_vandermonde_ne_zero_iff`) |

`RSScheme` = `n` distinct nodes + degree bound `k` + radius `τ`; `encode f = i ↦ f(nodes i)`;
`Decodes f y = (deg < k ∧ ≤ τ disagreements)`; `Safe y = ∃ f, Decodes f y`. The cheap verifier exhibits a
degree-`<k` `f` matching `≥ n − τ` positions in one `O(n·k)` forward pass.

**The imported wall (named, NOT proved):** producing a low-degree polynomial agreeing with a CORRUPTED
word *past* the unique-decoding radius is list decoding — NP-hard for general RS (Guruswami–Vardy 2005).
The ISD analog of the syndrome certificate.

**Why it earns a place vs the syndrome pillar (not a near-duplicate):** it's the *evaluation/interpolation*
view of the *same* RS code (the two classical RS faces), and its headline `corruption_fiber_nontrivial`
is the determine/lose split proved on a DIFFERENT mathlib core — **polynomial root-counting** — that the
parity-check syndrome lane never touches.

## Empirical sanity — `scripts/rs_cert.py` (+ frozen test 6/6), GF(7), n=5, k=2, τ=1
- **rs_unique:** 49 deg-`<2` polys → **49 distinct codewords** (encoding injective).
- **corruption_fiber_nontrivial:** message (3,2) → codeword (3,5,0,2,4) → **31 distinct words all decode to
  the same f** (clean + 30 single-symbol corruptions within τ=1) — determine the message, lose the corruption.
- **unique_decoding:** `d = n−k+1 = 4`, `2τ = 2 < d` ⇒ within radius; **no** second deg-`<2` poly decodes any
  fiber word.
- **cheap check:** verifying a claimed `f` = `n·k = 10` field ops vs the `7² = 49` message space (find = search).

## Boundaries
- The cheap-check is `O(n·k)` forward, polynomial in the witness; the find-hardness is **imported, not
  proven** (no complexity lower bound here) — the certificate framing is verify-vs-find, not P-vs-NP.
- The Lean pillar is a candidate public sundogcert pillar — owner-gated (built + axiom-clean locally,
  not pushed), like `ShadowDecayGeneral`/`SortingCert`.

## Files
- `sundogcert/Sundogcert/RSCertificate.lean` — the pillar (5 theorems, axiom-clean, in root).
- `scripts/rs_cert.py` (+ `test_rs_cert.py`) — the GF(7) numerical witness.
- `docs/atlas/FRESH_HYPOTHESES_SLATE_2.md` — the slate this came from (the pre-registration).
