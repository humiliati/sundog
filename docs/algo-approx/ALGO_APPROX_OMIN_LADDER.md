# The o-minimality ladder (opened 2026-07-02, off Slate-4 U-4)

> **What this is.** U-4 landed the PL finiteness-modulus (`DefinableRate.lean`) and exposed that the
> "no o-minimal substrate in mathlib" blocker was a mis-tiering for that hook. The owner's follow-up
> question — *can we get to formalizing o-minimality itself?* — gets an honest graded answer here:
> a ladder from what the lane's `PieceCover` calculus already supports up to the genuinely large
> walls, with each rung falsifier-fenced and sized. Rungs are Lean cores in the public `sundogcert`
> repo; nothing is promoted without a build receipt.
>
> **Prior-art status (checked 2026-07-02).** mathlib v4.30.0 has **no** o-minimal structures, no
> semialgebraic sets, no cell decomposition (verified by grep; the only "definability" is FO-logic
> `ModelTheory.Definability`, which carries no tameness consequences). A web sweep found **no Lean
> o-minimality formalization effort**; the formal-methods prior art is Cohen–Mahboubi's Coq
> quantifier elimination for real closed fields. Dimension-one o-minimality in Lean appears to be
> new ground.

## The design key: the finite-frontier characterization

A subset of ℝ is a finite union of points and open intervals **iff** its frontier is finite
(⇐ endpoints; ⇒ finitely many frontier points cut ℝ into open intervals on which the set is clopen,
hence full or empty by connectedness). So the dimension-one o-minimality axiom is stateable with
zero new substrate: `Tame s := (frontier s).Finite`, and mathlib's frontier calculus
(`frontier_compl`, `frontier_union_subset`) gives the boolean algebra for free.

## R1 — Dimension-one o-minimality, two structures. ✅ LANDED 2026-07-02

`Sundogcert/OMinimalOne.lean` (34th module). Axiom-clean, gated, zero warnings.

- **`Tame`** + boolean algebra (`tame_compl`/`tame_union`/`tame_inter`).
- **`net_continuous`** — every 1-D ReLU net realizes a continuous function.
- **`affineAway_levelSet_tame`** (the load-bearing core) — a continuous function affine away from a
  finite cut set has a finite-frontier level set. Proof: away from the cuts, mapping a frontier
  point to the set of cuts below it is **injective** — two frontier points over the same cut
  pattern would pin their whole stretch to the constant (`line_pinned`), and pinned stretches admit
  no frontier point — so the frontier injects into `S ∪ powerset S`.
- **`netDef_tame`** — **o-minimality of the ReLU/semilinear structure**: every set built by
  complement/union from strict superlevel sets `{x | c < realize1N g x}` is tame. Sublevel sets are
  inside the algebra (`scale (−1)` mirrors), hence level sets, rays, intervals, points (`var` = id).
- **`polyDef_tame`** — **o-minimality of quantifier-free semialgebraic sets** (the one-variable,
  quantifier-free shadow of Tarski): boolean combinations of `{x | c < p.eval x}` are tame, via
  `Polynomial.finite_setOf_isRoot`.

*Falsifier that did not fire* (`TAME_NEEDS_SUBSTRATE`): the dimension-one axiom could not be stated
or used without an o-minimal framework. It could — finite-frontier is enough.

## R2 — The dimension-one consequences on the tame base. [FORMALIZABLE — SCOPED 2026-07-02]

> **Recon receipts (mathlib v4.30.0, verified by grep).** Present: `Set.OrdConnected.isPreconnected`
> + `isPreconnected_iff_ordConnected` (intervals-are-preconnected is free), `Finset.sort`,
> `Finset.eq_of_subset_of_card_le` (chain-card injectivity), `Set.InjOn.encard_image` /
> `ncard_le_ncard` (counting). Absent: any "preconnected + frontier-disjoint ⇒ full-or-empty" split
> (≈15-line direct proof from the `IsPreconnected` definition); sorted-list *adjacency* lemmas
> (hand-rolled — the known grind). Strike order = cheapest first, shared machinery last.

### R2-M — Monotonicity theorem, PL instance (warm-up). [~30 lines]

*Target.* `net_mono_or_anti_between_cuts : ∀ g : Net 1, ∃ S : Finset ℝ, ∀ a b, a ≤ b →
(∀ s ∈ S, s ∉ Ioo a b) → MonotoneOn (realize1N g) (Icc a b) ∨ AntitoneOn (realize1N g) (Icc a b)`
— on every cut-free stretch the net is monotone or antitone (affine, by sign of the slope; pure
algebra off `AffineAway`, no derivatives). *Honest fence:* this is the per-stretch instance; the
**global sorted decomposition** ("piecewise monotone on an explicit finite partition") is the R2-N
rider — claiming it here would be the failure mode. No mathematical falsifier; pure assembly.

### R2-Q — Quantitative tameness: the frontier modulus. [~100 lines]

*Target (core, exact `S`-form — the gate target).*
`(frontier {x | f x = c}).ncard ≤ 2 * S.card + 1` for continuous `f`, `AffineAway f S`.
*The sharpening over R1:* R1's injection lands in `powerset S` (a `2^|S|` bound, finiteness only).
But the image of `x ↦ S.filter (· < x)` consists only of **initial segments** — a `⊆`-chain — and
card is injective on a chain (`eq_of_subset_of_card_le`), so there are at most `|S| + 1` fibers:
`|S|` cut points + `|S|+1` one-per-fiber points = `2|S|+1` exact.
*Corollaries (numbers reconciled):* `HasPieceCover k` gives `S.card + 1 ≤ k`, so the level and
superlevel frontiers have `ncard ≤ 2k − 1`; for nets, `≤ 2 · netPieceBound g − 1` (ℕ-safe:
`netPieceBound ≥ 1` always). This ties U-4's **piece modulus** to an **o-minimality (frontier)
modulus** — the definability rate and the tameness rate become one graded story, inheriting the
representation-dependence fence (linear additive / `2^d` folding) unchanged.
*Falsifier* (`RATE_NOT_TIGHTER`): the chain/initial-segment argument fails to close in Lean and
the honest bound stays `|S| + 2^|S|` — reported as the result if so.

### R2-N — Normal form + the strong decomposition (the grind). [~200–250 lines]

*Target.* `Tame s ↔ ∃ (P : Finset ℝ) (𝒥 : Finset (Set ℝ)), (∀ J ∈ 𝒥, IsOpen J ∧ J.OrdConnected) ∧
s = ↑P ∪ ⋃₀ ↑𝒥` — the docstring equivalence becomes a theorem: tame **is** "finite union of points
and open intervals" (open + `OrdConnected` = open interval, rays and `univ` included).
*⇐ (easy):* frontier of an open `OrdConnected` set ⊆ `{sInf s, sSup s}` (junk-value-safe: the
inclusion holds with ℝ's junk `sInf`/`sSup` since a frontier point of an open interval *is* a
one-sided bound in closure, hence the genuine `csInf`/`csSup`); points contribute themselves;
`tame_union` closes.
*⇒ (the content):* sort the frontier (`Finset.sort`), form the `|F|+1` **gaps** (two rays +
adjacent-pair `Ioo`s); each gap misses `frontier s`, so `s ∩ gap` is clopen-in-gap and the gap is
preconnected ⇒ `s ∩ gap = ∅` or the whole gap (the 15-line split lemma); `s` = (its frontier
points in `s`) ∪ (the full gaps). The grind is the sorted-adjacency cover ("every non-frontier
point lies in exactly one gap; gaps miss `F`") — mechanical, hand-rolled.
*Rider (shared machinery):* the **strong** piecewise-monotonicity for nets — `realize1N g` is
monotone-or-antitone on each gap of one explicit sorted partition — falls out of R2-M + the gap
enumeration.
*Falsifier* (`NORMAL_FORM_NEEDS_COMPONENTS`): if sorted-adjacency stalls, ship the ⇐ direction +
the per-gap split only, and report that ⇒ is where dimension one earns its name.

*Not in R2:* the polynomial strict-monotonicity twin (needs derivative-sign machinery —
`strictMonoOn_of_deriv_pos` did not surface in the recon grep; name-risk, deferred); anything about
projection (R3) or the abstract structure (R4).

## R3 — Projection / quantifier elimination. [FORMALIZABLE-HARD → the real wall]

Closure under projection from ℝ² is where the structures earn the name:

- **Semilinear case** (ReLU): Fourier–Motzkin elimination — genuinely formalizable, bounded effort;
  would make `NetDef` a real (projection-closed) structure in dimensions 1–2.
- **Semialgebraic case**: Tarski–Seidenberg QE — the big one. Formalized in **Coq**
  (Cohen–Mahboubi); porting to Lean is a months-scale project and the natural mathlib contribution.

## R4 — The abstract theory. [EXPEDITION — mathlib-scale]

`structure OMinimalStructure` (van den Dries axioms: boolean/product/projection closure +
dimension-one tameness) and the first abstract theorems: the Monotonicity Theorem, then Cell
Decomposition. This is the genuine "formalize o-minimality" project — new ground for Lean per the
prior-art sweep, multi-month, and worth its own slate if opened. R1's `Tame` is designed to be the
`S₁` axiom of that structure unchanged.

## Honest scope

- R1 is o-minimality **of two concrete structures in dimension one** — not the abstract theory, no
  projection closure, no cell decomposition. Every claim beyond R1 is sized, not started.
- The deep-learning hook (ReLU families are tame/o-minimal — recent literature frames deep learning
  as "disciplined construction of tame objects") is a *reading*, cited not claimed.

> Cross-links: `Sundogcert/OMinimalOne.lean` · `Sundogcert/DefinableRate.lean` ·
> [`ALGO_APPROX_CONJECTURE_SLATE_4.md`](ALGO_APPROX_CONJECTURE_SLATE_4.md) (U-4) ·
> `Sundogcert/PieceCover.lean` (the cut-set calculus R1 stands on).
