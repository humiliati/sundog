# Algorithmic Approximation — Fresh Conjecture Slate 2 (opened 2026-06-28)

> **What this is.** Slate 1 ([`ALGO_APPROX_CONJECTURE_SLATE.md`](ALGO_APPROX_CONJECTURE_SLATE.md))
> closed eight hooks. This slate mines what those closures actually *found*. The
> discovery: **almost every wall slate 1 hit was the same wall — cancellation.**
> One coordinate (does the computation cancel?) appears to control hardness,
> fine-grained membership, exact-vs-`ε`, **and** depth's expressive power. This
> slate tracks four hooks off that spine.
>
> **Discipline (inherited).** Nothing here is promoted. Each entry is stated so it
> can come back **NULL**: a claim, the traction the lane's machine-checked
> arithmetic gives, the classical question it touches, a **named falsifier**, a
> **tier**, and a **first move**. Status is earned only by a receipt (experiment)
> or a Lean core. All entries below are **PROPOSED — not yet run**.
>
> **Provenance.** Authored inline from live session synthesis across the closed
> slate-1 results, not a cold subagent fan-out; any one theme can be spun into a
> focused deep-dive on request.

## Tier legend

- **[FORMALIZABLE]** — a Lean target on the existing
  `CircuitNet`/`StraightLineCost`/`RegionCount`/`CancellationFree`/`DepthSeparation`
  surface; could become an axiom-clean core.
- **[IMPORT-AND-CHECK]** — the classical theorem exists; the work is verifying the
  bridge is faithful *and naming exactly where it breaks*.
- **[EMPIRICAL]** — a runnable frontier-ML experiment with a falsifiable prediction.
- **[SYNTHESIS]** — a retrospective organizing claim over results already in hand;
  falsifiable in principle, but framing more than a single Lean/experiment target.
- **[DAYDREAM]** — general/ambitious; mapped to mark the frontier, not yet actionable.

## The spine — cancellation is the controlling coordinate

A ReLU net is a tropical **rational** map (`relu` + affine = max-plus *with*
subtraction); the cleanly-transferable classical results live over tropical
**polynomials** (monotone, no cancellation). Slate 1 kept hitting that one seam
from four directions:

- **C-B1** — monotone (cancellation-free, `IsMono`) ⊊ general ReLU. The
  monotone-vs-general / natural-proofs wall *is* cancellation: tropical-polynomial
  vs tropical-rational. `abs = max(x,−x)` is the witness that needs the negative
  scale.
- **C-C2** — the two fine-grained cornerstones that fell **outside** the tropical
  cost-family were **3SUM** (additive cancellation) and the **`n^ω`** matmul route
  (Strassen's subtractive cancellation). Everything cancellation-free unified;
  the cancellation problems didn't.
- **H-A1 / H-A2** — exact compilation worked precisely on the cancellation-free PL
  fragment; the analytic gates (reciprocal, radical) — the division/cancellation
  gates — are exactly where exactness broke and `ε` entered.
- **C-D1** — the tent's `2^d` regions come from `abs = max(x,−x)`, which *requires*
  the negative scale. Folding needs cancellation.

So one predicate — **does the computation cancel?** — seems to sit under hardness
(C-B1), fine-grained membership (C-C2), exact-vs-`ε` (H-A1/A2), and depth's
expressivity (C-D1). The four hooks below test whether that spine is real.

---

## N-1 — Monotone depth is region-*polynomial*; cancellation depth is region-*exponential*. [FORMALIZABLE] *(the gem)* — SHARP FORM + QUALITATIVE + QUANTITATIVE-COMPOSITION CORES LANDED 2026-06-28 (axiom-clean), `Sundogcert/FoldCancellation.lean` + `Sundogcert/PieceCover.lean`. *(Open: per-gate lift to arbitrary `IsMono` circuits.)*

*Claim.* For a cancellation-free (`IsMono`) circuit, depth cannot create
exponentially many linear regions: a depth-`d`, width-`w` `IsMono` circuit
realizes a function with `O(d·w)` linear regions. The exponential expressivity of
depth (the tent's `2^d`) **requires** cancellation. Equivalently: **folding — and
hence all exponential expressivity from depth — requires cancellation.**

*Mechanism (why it should be provable, 1-D).* An `IsMono` circuit computes a
**monotone** (non-decreasing) function (`monotone_of_isMono`, already proved in
`CancellationFree.lean`). Compose a monotone outer `g` (`m` pieces) with a monotone
inner `h` (`n` pieces): the breakpoints of `g∘h` are the breakpoints of `h` plus
the points where `h` hits a breakpoint *value* of `g`. Because `h` is monotone,
each such value is hit at a **single** point (generically), so pieces **add**:
`pieces(g∘h) ≤ n + (m−1)`. Iterate over depth `d` ⇒ `O(d·w)` regions. The tent is
**non-monotone** (it folds via `abs`), so each outer breakpoint value is hit at
**two** inner points (up-leg and down-leg) ⇒ pieces **multiply** ⇒ `2^d` (exactly
`tent_iterate_dyadic`). Monotonicity is the no-fold condition, and `IsMono` is the
no-cancellation grammar, so the two coincide.

*Traction / what fuses.* This is a single positive theorem that unifies three
landed cores: **C-A1** (`RegionCount.lean`, regions as an exact intrinsic) +
**C-B1** (`CancellationFree.lean`, `monotone_of_isMono`) + **C-D1**
(`DepthSeparation.lean`, the tent's `2^d`). The Lean target: a new lemma
`isMono_regionCount_add` / `isMono_regions_poly` — "the 1-D region count adds (not
multiplies) under composition of `IsMono` circuits," on the same 1-D surface
`realize1_compile` already lives on. `monotone_of_isMono` is the hypothesis;
the breakpoint-pullback argument is the proof.

*Classical hook.* Depth separations (Telgarsky); monotone-circuit expressivity;
"depth = computation" — sharpened to "depth = computation **only with**
cancellation."

*Falsifier* (`MONOTONE_FOLDS`): a cancellation-free (`IsMono`) circuit whose 1-D
linear-region count is super-polynomial in depth. A single such witness ends it.

*Result — SHARP FORM LANDED 2026-06-28, axiom-clean (`Sundogcert/FoldCancellation.lean`,
`lake build` green, deps only `[propext, Classical.choice, Quot.sound]`).* The tightest
form of the contrast is machine-checked: **no cancellation-free circuit computes the
`d`-fold tent for `d ≥ 1`** (`isMono_not_iterTentFun`, and `isMono_not_iterTent` against
the circuit), via **`isMono_no_fold`** — a monotone circuit cannot be `1` at `1/2^d` and `0`
at the larger `2/2^d`. It fuses `monotone_of_isMono` (C-B1) with `tent_iterate_dyadic` (C-D1):
the *same witness* that achieves `2^d` regions is **unrealizable cancellation-free**, so the
exponential expressivity of depth is cancellation-essential — "depth = computation **only with
cancellation**."

*Positive half — QUALITATIVE CORE LANDED 2026-06-28, axiom-clean (same module, same gate).* The
mechanism — *monotone cannot fold, at any level* — is now machine-checked, lifting `isMono_no_fold`
from one witnessed fold to **every** level: **`isMono_realize1_monotone`** (the 1-D realization of an
`IsMono` circuit is `Monotone`, from `monotone_of_isMono`) ⇒ **`isMono_superlevel_isUpperSet`** (every
super-level set `{x | c ≤ f x}` is an `IsUpperSet` — once `f` reaches `c` it never returns below it, i.e.
exactly one rising crossing per level, **no oscillation**). The matching contrast
**`tent_superlevel_not_isUpperSet`** shows the tent's `{x | 1/2 ≤ T^[2] x}` is *not* an upper set
(`1/4` in, `1/2` out), so no `IsMono` circuit has the tent's level structure — folding *is* the
upper-set failure. This is the structural reason monotone region counts stay small (pieces *add*, they
cannot *multiply*).

*Quantitative half — COMPOSITION CORE LANDED 2026-06-28, axiom-clean (`Sundogcert/PieceCover.lean`,
9th Lean module, in the AxiomAudit gate).* Built via the **safer target**: instead of defining the
exact tropical-chamber count (the C-A1-deferred cardinality), a **bounded piece certificate**
`HasPieceCover f k` (`f` is affine off a finite cut set of size `≤ k−1`, i.e. `≤ k` affine pieces) —
an *upper bound* on the region count with no exact chamber object. On it, the load-bearing lemma is
machine-checked: **`hasPieceCover_comp_mono`** — `HasPieceCover h n → HasPieceCover g m → Monotone h →
HasPieceCover (g ∘ h) (n + m)`: composing a **monotone** inner map makes pieces *add*. The proof is
exactly the mechanism the draft predicted — monotonicity pins each `g`-cut's preimage to a single
upward ray (`{x | β ≤ h x} = Ici (crossing)`), so crossings can't multiply. Iterating gives the
headline **`hasPieceCover_iterate`**: a monotone map with `k` pieces composed with itself `d` times has
`≤ d·k + 1` pieces — **linear in depth**, the exact quantitative contrast to the tent's `2^d`
(`DepthSeparation`). So monotone depth is region-*polynomial*; cancellation depth is
region-*exponential* — now a theorem on the 1-D depth axis. **Remaining (narrower):** the only gap to a
fully general `isMono_regions_poly` over *arbitrary* `IsMono` circuits is the per-gate piece lemmas
(`+`, `max`, `scale`) over `Trop` — `HasPieceCover` for each gate — to lift the function-level
composition core to circuit structure; the depth/composition axis (the one that produces the `2^d`
blow-up) is done.

---

## N-2 — Cancellation is the single imported-wall coordinate. [SYNTHESIS]

*Claim.* *Every* imported wall in the lane — the monotone-circuit lower bound
(C-B1), 3SUM and the `n^ω` route (C-C2), the analytic/division gates (H-A1/A2),
and the depth-vs-width bound (C-D1) — is reducible to one predicate: **the target
requires cancellation.** If true, the lane's entire hardness-import surface
collapses to a single coordinate, and every "imported wall" is the same wall.

*Traction.* It would retro-organize the whole lane: instead of five distinct named
imports, one — "cancellation-required" — with the others as instances on different
axes (additive: 3SUM; subtractive/algebraic: `n^ω`; division: analytic gates;
fold: depth). It also predicts the *shape* of any future wall the lane meets:
ask first whether the target cancels.

*Classical hook.* Monotone vs general circuits; the natural-proofs barrier as the
"cancellation is where lower bounds stop being natural" statement.

*Falsifier* (`WALL_WITHOUT_CANCELLATION`): an imported wall the lane genuinely
relies on that is provably **not** cancellation-reducible — a hardness the
cancellation-free fragment also exhibits. One such wall falsifies the collapse.

*Honest tier note.* More framing than theorem: "reducible to cancellation" needs a
precise common definition that covers additive / subtractive / division / fold on
equal footing, or it stays an elegant retrospective rather than a result. Best
pursued *after* N-1 gives the fold↔cancellation half a proof.

---

## N-3 — The find/check ledger is a general certificate theory. [FORMALIZABLE]

*Claim.* The three `StraightLineCost` instances (syndrome / shortest-path / ReLU
gate-count) are cases of one structure: **a witness whose verifier is a cheap
straight-line program, soundness proved, find-hardness imported.** Generalize to a
`Certifies` class and spawn new instances — each a small Lean cert module like
`ShortestPathCert`. Conjecture: a broad class of P/NP problems carry tropical/PL
verification certificates expressible in the ledger.

*Candidate instances (each a buildable module).* **max-flow / min-cut**
(LP-duality cert: a cut certifies a flow's optimality) · **bipartite matching**
(König / a vertex cover certifies a maximum matching) · **2-SAT** (the
implication-graph SCC certificate) · **Pratt primality** (the classic succinct
NP certificate). Each gives another `HasStraightLineCost` instance and another
`O(·)` cheap-check theorem.

*Traction.* The most *buildable* hook on the slate — it extends the proven
`ShortestPathCert` pattern (`feasible_le_walk` dual bound + `cert_isLeast` exact
optimality) to new domains, each closing as its own axiom-clean module. Grows the
ledger from 3 to N instances and tests the breadth of "check is cheap, find is the
imported wall."

*Classical hook.* Certifying algorithms; the NP "verification is easy" half,
machine-checked across optimization and decision problems; LP duality / König /
implication graphs.

*Falsifier* (`VERIFIER_NOT_PL`): a natural certifying problem whose verifier is
**not** straight-line / PL-expressible in the ledger — the certificate exists but
the cost model can't carry it. Names the boundary of the ledger's reach.

*First move.* Pick one (max-flow/min-cut is the cleanest LP-dual mirror of
`ShortestPathCert`) and write the module: a `Cut`/`Feasible` structure, a
`weak-duality` lower-bound lemma, the exact-optimality capstone, and the
`HasStraightLineCost` instance.

---

## N-4 — The `ε`-essential region count is the *learnable* invariant. [EMPIRICAL]

*Claim.* C-A1 proved the **exact** region count is an intrinsic invariant; C-D2
found it does **not** drive generalization (`ε`-approximability + trainability did,
not exact `k`). The redemption: an **`ε`-smoothed** region count — counting only
regions *wider than `ε`* — *is* the generalization predictor. The exact-vs-`ε` seam
that confounded C-D2 becomes the measured quantity.

*Traction.* Turns C-D2's documented INCONCLUSIVE into a sharp, testable positive on
the same harness: re-run the threshold sweep measuring `ε`-essential pieces instead
of exact `k`, and test whether `threshold ~ (ε-essential count)` where it failed to
track exact `k`. Directly tests the lane's standing "exact size ≠ `ε`-complexity"
confound as a hypothesis rather than a footnote.

*Classical hook.* Effective / `ε`-region complexity vs exact region count; the
approximation-theoretic side of expressivity; generalization bounds keyed to
*effective* rather than exact capacity.

*Falsifier* (`EPSILON_REGIONS_ALSO_FAIL`): the `ε`-essential region count also fails
to track the trained-generalization threshold (or tracks no better than the exact
`k` it replaces) → region geometry, smoothed or not, is not the learnable invariant
and the operative quantity is purely trainability.

*First move.* Reuse the C-D2 grokking harness; add an `ε`-essential piece counter
(merge adjacent pieces whose width `< ε`), sweep `ε`, and re-fit `threshold(k)`
against the `ε`-essential count. Decouple the trainability confound with the
fitting-oracle step C-D2 already named as its honest next move.

---

## What this slate is NOT

- **Not a Millennium attack.** Scope = tropical / piecewise-linear / monotone
  computation and ML theory. The cancellation spine is a *typed coordinate* over
  the lane's own results, not a route to an unconditional separation.
- **Not an unconditional P-vs-NP claim.** N-1 is a lower bound *in the
  cancellation-free model* (it says depth can't help *there*); it does not bound
  general nets, which is exactly the natural-proofs boundary the lane respects.
  N-2/N-3 keep find-hardness imported.
- **Not promoted.** Tiers are aspirations; every entry is PROPOSED until a Lean
  core or an experiment receipt lands. Slate 1's discipline carries over unchanged.

## Recommended first attacks (ranked)

1. **N-1 — the gem. [FORMALIZABLE] — ✅ SHARP FORM LANDED 2026-06-28** (axiom-clean,
   `Sundogcert/FoldCancellation.lean`): `isMono_no_fold` + `isMono_not_iterTentFun` /
   `isMono_not_iterTent` — no cancellation-free circuit computes the `d`-fold tent
   (`d ≥ 1`); the depth-separation witness is cancellation-essential, fusing
   `monotone_of_isMono` (C-B1) + `tent_iterate_dyadic` (C-D1). The mechanism the
   draft predicted (monotone can't fold) is now machine-checked. **Positive half —
   qualitative core also LANDED 2026-06-28** (`isMono_realize1_monotone` /
   `isMono_superlevel_isUpperSet` / `tent_superlevel_not_isUpperSet`): monotone ⇒
   super-level sets are upper sets (no fold at *any* level), with the tent as the
   not-an-upper-set contrast. **Quantitative composition core also LANDED 2026-06-28**
   (`Sundogcert/PieceCover.lean`): the bounded piece certificate `HasPieceCover` +
   `hasPieceCover_comp_mono` (monotone composition *adds* pieces, `n + m`) +
   `hasPieceCover_iterate` (`d` self-compositions ⇒ `≤ d·k + 1` pieces, **linear in
   depth** vs the tent's `2^d`). **Remaining = the per-gate lift** (`+`, `max`,
   `scale` `HasPieceCover` lemmas over `Trop`) to reach a fully general
   `isMono_regions_poly` over arbitrary circuits; the depth axis is done.
2. **N-3 — the safest builder. [FORMALIZABLE]** If the goal is more Lean closures,
   this is the highest-yield: each new certificate (max-flow/min-cut first) is an
   independent axiom-clean module on the proven `ShortestPathCert` pattern, growing
   the find/check ledger with low risk.
3. **N-4 — the empirical redemption. [EMPIRICAL]** Cheapest to run and it converts a
   documented null into a falsifiable positive on an existing harness; do it when
   you want a measured result rather than a proof.
4. **N-2 — the elegant retrospective. [SYNTHESIS]** Highest framing value, but it is
   organizing rather than provable until "reducible to cancellation" is defined
   uniformly across the four axes. Best written *after* N-1 lands the
   fold↔cancellation half, so it rests on a theorem rather than a vibe.

> Cross-links: [`ALGO_APPROX_CONJECTURE_SLATE.md`](ALGO_APPROX_CONJECTURE_SLATE.md)
> (slate 1, the closed results this mines) ·
> [`SUNDOG_V_ALGO_APPROX.md`](SUNDOG_V_ALGO_APPROX.md) (parent lane) ·
> [`SUNDOG_V_P_V_NP.md`](SUNDOG_V_P_V_NP.md) (find/check sibling) ·
> `Sundogcert/CancellationFree.lean` · `Sundogcert/RegionCount.lean` ·
> `Sundogcert/DepthSeparation.lean` · `Sundogcert/StraightLineCost.lean`.
