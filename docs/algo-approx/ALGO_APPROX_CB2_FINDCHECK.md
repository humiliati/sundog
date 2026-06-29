# C-B2 — the in-model find-vs-check gap (with ShortestPathCert as verifier)

> **Slate hook:** [`ALGO_APPROX_CONJECTURE_SLATE.md`](ALGO_APPROX_CONJECTURE_SLATE.md) C-B2.
> **Verdict (2026-06-27): bounded — `CHEAP_MONOTONE_CONSTRUCTOR` *fires* for the
> shortest-path instance.** The cheap-CHECK side is fully proved (`ShortestPathCert` /
> `StraightLineCost`), but shortest path is in **P**, so it supplies only a *polynomial*
> find/check gap, not the superpolynomial separation the hook hoped for. A superpolynomial
> in-model gap requires a **monotone-hard** function (Jerrum–Snir), whose lower bound is
> **imported** — that is C-B1's `monotone_transfer`. An *unconditional* superpolynomial
> find ≫ check separation is **not** Lean-reachable (it would be a standalone complexity
> lower bound, which neither mathlib nor the field has). Analytical close; the Lean it
> needs — the cheap verifier — already exists.

## What the cost ledger proves vs imports, on the find/check axis

The `StraightLineCost` ledger now has three instances, and they share one shape:
**the CHECK side is provable; the FIND side is imported.**

| instance | CHECK (proved, in-ledger) | FIND (the hard side) |
|---|---|---|
| syndrome certificate | `verifyCost ≤ 2mn+n+m+2`, `O(mn)` | decoding hardness — **imported** (ISD/SIS) |
| **shortest path** (`ShortestPathCert`) | `verifyCost ≤ 2\|E\|+1`, `O(E)` | Bellman–Ford `O(VE)` — **polynomial** (SP ∈ P) |
| ReLU construction (`CircuitNet`) | — (construction *is* the object) | monotone lower bound — **imported** (C-B1) |

So C-C1 gave a *real, proved, cheap* verifier for shortest paths; C-B2 asks whether the
matching **find** side is provably much larger *in-model*. It is not — for two reasons.

## Why shortest paths cannot give a superpolynomial gap

1. **Shortest path is in P.** Bellman–Ford computes the distances (and hence a feasible
   potential + tree) in `O(VE)`; Dijkstra in `O(E log V)`. So the constructor is
   *polynomial*, the verifier is `O(E)`, and the in-model gap is at most a **polynomial**
   factor (`~V`), never superpolynomial. The slate's falsifier `CHEAP_MONOTONE_CONSTRUCTOR`
   — "every candidate also has a small constructor" — *fires*: SP's constructor is cheap.
2. **Even the polynomial gap is not a *lower-bound* separation.** Proving "every tropical
   circuit computing the SP distances needs `≥ B` gates" is a circuit lower bound; for SP
   (which has near-linear algorithms) no superlinear unconditional lower bound is known.
   One can *exhibit* a constructor (the `V`-round min-plus Bellman–Ford circuit, `Θ(VE)`
   gates) that is `Θ(V)×` the verifier — an **asymmetry exhibit**, not a separation, and
   only polynomial.

## Where the superpolynomial gap actually lives — and stays imported

A superpolynomial in-model find ≫ check gap needs a function that is **monotone-hard**
(superpolynomial in the `(min,+)`/`(max,+)` circuit model) yet has a **cheap witness
verifier**. Jerrum–Snir supplies such functions (e.g. `(min,+)` permanent/clique-style),
but their lower bound is exactly the **imported** content of C-B1's `monotone_transfer`:
*given* the Jerrum–Snir bound `B ≤ maxCount(any IsMono circuit for f)`, every
cancellation-free ReLU constructor for `f` has `≥ B` gates, while a witness verifier is
cheap. So:

> **The superpolynomial find/check separation = C-B1's transfer (find-bound imported) +
> a cheap verifier (C-C1's `ShortestPathCert` pattern).** It is conditional on the imported
> monotone lower bound; the unconditional version would be a complexity breakthrough.

## Honest verdict

C-B2 does **not** add a new unconditional separation. What it establishes:

- The cost ledger's **cheap-CHECK side is proved** for shortest paths (`ShortestPathCert`,
  `O(E)`), the reusable contribution — and it is the verifier side any in-model find/check
  statement would use.
- **Shortest paths cannot witness a superpolynomial gap** (SP ∈ P → `CHEAP_MONOTONE_CONSTRUCTOR`
  fires). The gap there is polynomial and is an exhibit, not a lower bound.
- The **superpolynomial in-model gap is exactly C-B1** (monotone-hard `f`, imported
  Jerrum–Snir bound) paired with a cheap verifier; **unconditional** is out of reach.

This is the same wall the whole lane respects and the P-vs-NP sibling lane is built on
([`SUNDOG_V_P_V_NP.md`](../SUNDOG_V_P_V_NP.md)): *verification is provably cheap; the hardness
of finding is the imported assumption.* C-B2 makes that decomposition explicit on the
find/check axis and locates the one place (C-B1, imported) a superpolynomial gap could come
from. No promotion.
