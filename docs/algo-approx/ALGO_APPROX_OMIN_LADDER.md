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

## R2 — The dimension-one consequences on the tame base. [FORMALIZABLE, next]

Three self-contained targets, roughly in order of effort:

1. **Normal form**: `Tame s ↔` s is a finite union of points and open intervals (the ⇒ direction
   is the components argument; makes the docstring equivalence a theorem).
2. **Quantitative tameness**: `frontierBound` tied to `netPieceBound` (U-4's modulus) — frontier
   card ≤ `2·netPieceBound g + 1`-ish, making the o-minimality *rate* checkable, in the same spirit
   as the piece modulus (representation-dependent, honestly fenced).
3. **Monotonicity theorem, PL instance**: every `NetDef`-definable function is piecewise monotone
   with an explicit finite decomposition — the first o-minimal *consequence* theorem, nearly free
   from `AffineAway`.

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
