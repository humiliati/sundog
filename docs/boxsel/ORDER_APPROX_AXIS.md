# Approximation as an Order-Relative Axis (E2) — exactness is order-relative, ε-approximation is not

**Status:** expedition E2 landed and machine-checked. A fresh instance family for the Order-Relative
Resolution Law, off the algorithmic-approximation lane (`ExactRepr` / `UniversalApprox`). Lean:
`Sundogcert/OrderRelativeApprox.lean`. Internal; frozen-as-portfolio.

The question: is "approximating a function" an order-relative phenomenon — a determine/resist split on
some budget? The break-first answer turns on **which axis you pick**.

---

## The collapse — ε-approximation is NOT order-relative

If the budget is *approximation precision* (resolve = "ε-approximable by a ReLU net"), the axis
**collapses**: by the universal-approximation capstone (`UniversalApprox.continuous_relu_approximable`)
**every** continuous `f` on `[0,1]` is ε-approximable for **every** `ε > 0`. So "resolvable within
budget" is always satisfiable — nothing ever resists. Universality kills the determine/resist
structure on this axis (`approx_axis_collapses` re-exposes the capstone as exactly this collapse).

## The axis that survives — EXACTNESS (`ε = 0`)

Move the budget to *exact representability* (resolve = "**is** a finite ReLU net", `ε = 0`). Now the
determine/resist split is real and grounded:

- **DETERMINE** — a continuous-piecewise-linear function (any net's realization) is exactly a finite
  net, so it has **finite** exactness-order (`exactProblem`, ord `0`).
- **RESIST** — `x²` is only approximable, never exact: exactness-order `⊤` (`sqResistProblem`). This
  is the **earned** resist pole:
  - `sq_no_pieceCover` — `x²` has no finite piece cover (not continuous-PL). Any finite cut set leaves
    a gap interval beyond its max on whose interior `x²` would be affine — impossible by strict
    convexity (the 3-point midpoint test gives `(a−b)² = 0`).
  - `net_hasPieceCover` (from `ExactRepr`) — every finite ReLU net is piecewise-linear.
  - ⇒ `sq_not_exactly_net` — `x²` is not the realization of any finite net.

`exact_determine_vs_resist` packages the split: a PL function has finite exactness-order, `x²` has `⊤`.

---

## The lesson

**Approximation is order-relative on exactness, not on ε-approximation — and universality is precisely
what forces that move.** The interesting object is `x²`: *approximable* (the analytic gate,
`AnalyticGate` / `SawtoothApprox`) yet *not exactly representable* — the gap between "ε for every
ε > 0" and "ε = 0" is exactly where the determine/resist structure lives. This is the same shape as
the rest of the ledger: the resist pole is a thing you can get arbitrarily close to but never reach.

All four headlines are axiom-clean (`[propext, Classical.choice, Quot.sound]`) and build-gated.

---

## Continuation — graded order + a second resist (`OrderRelativeApproxGraded`)

**Graded by piece count.** The exactness order is not just `{finite, ⊤}` — grade the determine side
by the **number of affine pieces** (`ord f =` least `k` with `HasPieceCover f k`). Then it is a real
ladder:

- `id` — order **1** (one affine piece), `id_pieceProblem`;
- `ReLU = max x 0` — order **2** (one breakpoint), `relu_pieceProblem` (earned both ways:
  `relu_affineAway` gives the 2-piece cover, `relu_not_affine` rules out 1 piece);
- `step2 = ReLU x + ReLU (x-1)` — order **3** (two breakpoints), `step2_pieceProblem`
  (`OrderRelativeApproxLadder`); the lower bound `step2_not_pieceCover_two` is a two-interval
  pigeonhole (the disjoint windows around `0` and `1` each need a cut);
- `x²`, `eˣ` — order **⊤** (no finite cover).

`graded_exactness` packages `1 < 2 < ⊤`; `ladder3` packages the explicit climb `1 < 2 < 3`. So the
determine side is a genuine ladder, with the resist pole on top — not a flat binary.

## The ladder is unbounded — the general `k`-breakpoint family (`OrderRelativeApproxLadderK`)

The rungs `id`, `ReLU`, `step2` are the `k = 0, 1, 2` cases of one explicit family:

> `sumRelu k = Σ_{i<k} ReLU(x - (i+1))` — `k` breakpoints at `1, …, k`, `k+1` affine pieces
> (slopes `0, 1, …, k`), **exactness-order exactly `k+1`** (`sumRelu_order_eq`, `ladderK`).

Both bounds are general in `k`:

- **Upper** (`sumRelu_hasPieceCover`): the cut set `{1, …, k}` works. Proved by `AffineAway`
  additivity (`affineAway_add`: a sum of two functions is affine off the union of their cut sets)
  folded over the `Finset` sum — each `ReLU(x-(i+1))` is affine off `{i+1}`.
- **Lower** (`sumRelu_not_pieceCover`): a genuine **`k`-fold pigeonhole**. `sumRelu k` *bends* at
  every breakpoint `j` (`sumRelu_bend`: the second difference at `j`, spacing `½`, equals `½ ≠ 0` —
  a one-term sum, since only the `j`-th ReLU is crossed there). The `k` windows `(j-½, j+½)` are
  disjoint, so each forces its own cut; an `InjOn` count over `{1, …, k}` gives `≥ k` cuts, i.e.
  `≥ k+1` pieces.

So for every `k ≥ 1` there is an explicit PL function of exactness-order exactly `k+1`: the determine
side is an **unbounded** ladder `1 < 2 < 3 < ⋯`, with the resist pole `⊤` (`x²`, `eˣ`) sitting above
the entire chain. The finite/`⊤` law isn't a binary dressed up as a grade — it is a real `ℕ ∪ {⊤}`
order, every finite level inhabited.

**A second resist witness — `eˣ` (transcendental).** `exp_no_pieceCover` / `exp_not_exactly_net`:
`eˣ` joins `x²` on the resist pole, earned by **strict convexity** (`strictConvexOn_exp` — on any gap
interval `eˣ` would be affine, but the midpoint sits strictly below the chord). It shows the resist is
generic to *curvature*, not special to polynomials. (`√x`, `x³` are traps — domain / odd-symmetry
fool the three-point test; strict convexity everywhere is what makes `eˣ` clean.)

---

*Sundog Research Lab — expedition E2. Approximation as an Order-Relative axis: the ε-approximation
budget collapses (universality), the exactness budget (`ε = 0`) carries the determine/resist split —
PL determines (graded by piece count, an explicit **unbounded** ladder: `id` 1, `ReLU` 2, `step2` 3,
and the general `sumRelu k` at order `k+1`), `x²` and `eˣ` resist (earned via no-piece-cover; `eˣ` by
strict convexity). Lean: `sq_no_pieceCover`, `sq_not_exactly_net`, `exact_determine_vs_resist`,
`approx_axis_collapses`, `graded_exactness`, `exp_no_pieceCover`, `exp_not_exactly_net`,
`step2_not_pieceCover_two`, `ladder3`; general-`k`: `affineAway_add`, `sumRelu_bend`,
`sumRelu_not_pieceCover`, `sumRelu_hasPieceCover_iff`, `sumRelu_order_eq`, `ladderK`. Internal;
frozen-as-portfolio.*
