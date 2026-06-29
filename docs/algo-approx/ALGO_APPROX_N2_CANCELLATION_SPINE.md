# N-2 — The cancellation spine: one coordinate, honestly bounded (synthesis retrospective)

> **What this is.** Slate-2 hook **N-2**, tier **[SYNTHESIS]**. It is a *retrospective lens*
> over the Algorithmic-Approximation lane, **not** a reduction theorem. The reading: almost
> every wall the lane meets is the *same* wall — **cancellation**. This document states that
> reading, marks exactly which part of it is machine-checked and which part is an organizing
> analogy, names the falsifier, and records the honest limit. Nothing here is promoted beyond
> what its receipts support.
>
> **Provenance.** Authored from the lane's own closed results (slates 1 and 2). The Lean anchor
> is `Sundogcert/CancellationSpine.lean` (`isMono_tame`); the negative half is
> `FoldCancellation.isMono_not_iterTent`.

## The coordinate

A ReLU network is a tropical **rational** map: `relu` plus affine is min/max-plus arithmetic
*with subtraction* — the negative scale. The classical results that transfer cleanly live over
tropical **polynomials**: the monotone, cancellation-free fragment (`IsMono` — no negative
scale). So one yes/no question sits under the lane's findings:

> **Does the computation cancel?** — i.e., does it use the negative scale / subtraction /
> division that turns a tropical polynomial into a tropical rational map?

The conjecture N-2 records is that the lane's *entire* hardness-import surface is this one
coordinate: every imported wall is "the target requires cancellation," typed by *which kind*
of cancellation (additive / subtractive / division / fold).

## The map — wall by wall, with honest status

| Wall (slate id) | Cancellation type | Status | Anchor |
|---|---|---|---|
| Monotone ⊊ general (C-B1) | negative scale | **PROVEN** | `abs_not_isMono` (`abs = max(x,−x)` needs the negative scale), `monotone_of_isMono` |
| Folding / depth expressivity (C-D1, N-1) | fold = `abs` = negative scale | **PROVEN** | `isMono_not_iterTent` (no cancellation-free circuit folds), `tent_iterate_dyadic` (cancellation ⇒ `2^d`) |
| Region growth (N-1, circuit-level) | negative scale | **PROVEN** | `isMono_hasPieceCover` (no cancellation ⇒ region count linear in leaves); `isMono_tame` bundles monotone + convex + this |
| 3SUM (C-C2) | additive cancellation | *organizing reading* | analytical (`ALGO_APPROX_CC2_FINEGRAINED.md`); 3SUM fell **outside** the tropical cost-family — not a formal reduction |
| `n^ω` / Strassen matmul (C-C2) | subtractive / algebraic | *organizing reading* | analytical; the `n^ω` route uses subtractive cancellation, also outside the family |
| Analytic gates: reciprocal, radical (H-A1/A2) | division | *organizing reading* | exact compilation held on the cancellation-free PL fragment; the division gates are exactly where exactness broke and `ε` entered |
| Depth-vs-**width** lower bound (C-D1) | — | **IMPORTED** | Telgarsky/Eldan–Shamir; this is the named import the lane never proves, cancellation-typed or not |

## The proven core vs. the organizing reading

**Machine-checked (the spine's load-bearing half).** On the *monotone / convex / region / fold*
axes the coordinate is not a metaphor — it is a theorem pair:

- **Positive:** `isMono_tame` — a cancellation-free circuit is *uniformly* tame (monotone **and**
  convex **and** region-polynomial), all from the single hypothesis `IsMono`.
- **Negative:** `isMono_not_iterTent` — the canonical cancellation-using function (the `d`-fold
  tent, built from `abs`) is **unreachable** cancellation-free.

So absence of cancellation ⇒ all the good behaviour, and the canonical cancelling behaviour ⇒
unreachable without it. On these axes, "cancellation is the coordinate" is earned.

**Organizing reading (the spine's conjectural half).** The 3SUM (additive), `n^ω` (subtractive),
and analytic-gate (division) walls are *typed* by cancellation by analogy — each invokes the
same primitive (subtraction or division) — but there is **no formal reduction** putting them on
equal footing with the fold axis. They are arranged under the coordinate, not derived from it.

## What the lens predicts

Used as a heuristic (not a theorem), the coordinate predicts the *shape* of any future wall the
lane meets: **ask first whether the target cancels.** If it is cancellation-free, expect it to
be tame (the lane's positive machinery applies); if it cancels, expect a wall, and classify it
by the *kind* of cancellation to guess which imported result it will lean on.

## Falsifier — `WALL_WITHOUT_CANCELLATION`

A single wall the lane genuinely relies on that is provably **not** cancellation-reducible — a
hardness the cancellation-free fragment *also* exhibits — collapses the claim. **Status: not
fired.** No such wall has surfaced; every wall met so far is cancellation-typed. But absence of a
counterexample is not a proof, and the additive/subtractive/division axes are reading, not
reduction, so the conjecture stays exactly that.

## Honest limit — why this is a lens, not a theorem

"Reducible to cancellation" has **no precise common definition** that covers additive,
subtractive, division, and fold cancellation on equal footing. Without one, the four axes cannot
be unified into a single statement, and N-2 stays an (elegant, predictive) retrospective rather
than a result. What it would take to promote it:

1. a formal predicate `CancellationReducible (wall)` general enough to instance all four axes;
2. a machine-checked reduction of each wall to it.

Step 2 would, in general, require the imported walls (3SUM-hardness, `n^ω` optimality,
analytic-gate inapproximability, depth-vs-width) to be *formalized* — which they are not, in any
proof assistant. So full promotion is out of reach by the same wall the rest of the lane
respects. The fold / monotone / region axes are the exception: there the reduction *is* the
machine-checked `isMono_tame` / `isMono_not_iterTent`.

## Status

N-2 **deliverable complete** (synthesis written + Lean-anchored); the **claim** is a typed
conjecture — *proven* on the fold/monotone/region axes, *organizing reading* on the
additive/subtractive/division axes, *not promoted* overall. This closes the Slate-2 hardening
order (N-1 → N-3 → N-4 → N-2) except for the empirical N-4 (ε-essential region count).

> Cross-links: [`ALGO_APPROX_CONJECTURE_SLATE_2.md`](ALGO_APPROX_CONJECTURE_SLATE_2.md) (the
> slate) · `Sundogcert/CancellationSpine.lean` (`isMono_tame`) ·
> `Sundogcert/FoldCancellation.lean` (`isMono_not_iterTent`) ·
> `Sundogcert/CancellationFree.lean` (`abs_not_isMono`) ·
> `Sundogcert/RegionPoly.lean` (`isMono_hasPieceCover`) ·
> [`ALGO_APPROX_CC2_FINEGRAINED.md`](ALGO_APPROX_CC2_FINEGRAINED.md) (the 3SUM / `n^ω` analysis).
