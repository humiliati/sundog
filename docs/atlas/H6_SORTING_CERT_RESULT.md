# H6 Result — the sorting-network verify≪search certificate (the 0-1 principle)

> **2026-06-08.** Hypothesis #6 (final) of slate `ww6koomb1`. The **combinatorics** worked example of the
> lab's referee-free thesis: machine-check the deductive cheap-CHECK core, name the imported hard-FIND
> wall. Two legs: an empirical demonstration (`scripts/sorting_cert.py`, internal) and a **machine-checked
> Lean pillar** (`sundogcert/Sundogcert/SortingCert.lean`, candidate public sundogcert pillar — built +
> axiom-clean locally, **owner-gated**, not pushed). NOT public-eligible (empirical). Attribution: Knuth,
> *TAOCP* vol. 3 §5.3.4 (the 0-1 principle); the lab's certificate thesis.

## Headline — BOUNDED-POSITIVE

**Sorting networks are a clean verify≪search certificate instance, with the 0-1 principle as the
cheap-check SOUNDNESS — now machine-checked in Lean.** A comparator network sorts *every* input over
*every* linear order iff it sorts all `2ⁿ` **binary** inputs, so verification is `O(2ⁿ·size)` (polynomial
in the witness) while *finding* a minimal network is a needle-in-a-haystack search (the imported wall,
exactly like ISD-hardness in the syndrome certificate).

## Lean pillar (the headline) — `sundogcert/Sundogcert/SortingCert.lean`

Built on Lean 4.30.0 + mathlib v4.30.0; full `lake build` **0 warnings (3528 jobs)**, wired into the
`Sundogcert` root; every theorem **axiom-clean** (`[propext, Quot.sound]`, no `sorryAx`).

| theorem | statement |
|---|---|
| `comp_comm` | **CORE**: a comparator commutes with any monotone map — `comp i j (f∘x) = f∘(comp i j x)` (from mathlib `Monotone.map_min`/`map_max`) |
| `runNet_comm` | a whole network commutes with any monotone map (induction over the comparator list) |
| `sorts_of_sorts_bool` | **THE 0-1 PRINCIPLE (cheap-check soundness)**: if a network sorts all `2ⁿ` binary inputs, it sorts every input over any linear order `α` |
| `sorts_nat_of_sorts_bool` | concrete corollary — the binary check certifies sorting over `ℕ` |

`comp` = a comparator `(i,j)` sending wire `i→min, j→max`; `runNet` = a `List` of comparators folded over
the wires `Fin n → α`; `Monotone (runNet net x)` = "sorted." The proof: a failure `runNet x j < runNet x i`
at positions `i ≤ j` is exposed by the monotone threshold `thr (runNet x i) = (· ≥ runNet x i)`, which the
network commutes with — yielding a *binary* input the network fails to sort. The single imported analytic
input is mathlib's monotone-commutes-with-min/max; everything else is elementary.

**The imported wall (named, NOT proved):** finding a *minimal*-size/depth sorting network is a hard
combinatorial search (optimal sizes open / required massive SAT effort for `n ≳ 10`). The search-hardness
is named, exactly as the syndrome certificate imports ISD/decoding hardness.

## Empirical leg — `scripts/sorting_cert.py` (+ frozen test 12/12)

- **0-1 principle SOUNDNESS** (the cheap check is sound): for n=4,5,6, the `2ⁿ` binary check **agrees with
  the full n! permutation check** on optimal, bubble, and broken networks; a broken network (optimal − 1
  comparator) is **caught** by the cheap check.
- **The verify≪search asymmetry:**

  | n | opt size s | verify ops 2ⁿ·s | #size-s nets | P(random sorts) | ~search 1/P |
  |---|---|---|---|---|---|
  | 4 | 5 | 80 | 7.8e3 | 1.4e-3 | 7.0e2 |
  | 5 | 9 | 288 | 1.0e9 | 2.4e-4 | 4.2e3 |
  | 6 | 12 | 768 | 1.3e14 | **0 / 200k** | huge |

  Verify costs ~10²–10³ ops (poly in the witness); finding by random search collapses (P→0, space
  super-exponential). Verify is a single fast pass (0.9 ms at n=8, scaling as `2ⁿ`).

## Boundaries
- Verification is `O(2ⁿ·size)` — cheap *relative to search*, and polynomial in the witness, but still
  exponential in `n` (the 0-1 principle improves `n!` → `2ⁿ`, the best general bound). The certificate
  framing is verify-vs-find, not P-vs-NP.
- The find-hardness is **imported, not proven** (no complexity lower bound here).
- The Lean pillar is a candidate public sundogcert pillar — owner-gated (built + axiom-clean locally,
  not pushed).

## Files
- `sundogcert/Sundogcert/SortingCert.lean` — the 0-1-principle pillar (4 theorems, axiom-clean, in root).
- `scripts/sorting_cert.py` (+ `test_sorting_cert.py`) — the empirical verify≪search demonstration.
