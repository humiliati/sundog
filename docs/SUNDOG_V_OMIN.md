# SUNDOG_V_OMIN — the o-minimality lane (opened 2026-07-02)

> **What this is.** The machine-checked o-minimality program in the PUBLIC `sundogcert` repo,
> grown out of Slate-4 U-4 (the PL finiteness-modulus) in one day of climbing: the complete
> dimension-one theory plus a projection-closed semilinear structure in dimensions 1–2. This
> roadmap holds the lane's two long-horizon targets: **TS-QE** (Tarski–Seidenberg quantifier
> elimination — on the todo list, not started) and the **R4 expedition** (the abstract o-minimal
> structure and its first theorems — scoped below, awaiting go). Both are months-scale: the lane
> is deliberately long-horizon where most prior SUNDOG_V lanes closed in about a week.
>
> **Prior-art status (checked 2026-07-02).** No Lean o-minimality formalization found anywhere;
> mathlib v4.30.0 has no o-minimal/semialgebraic substrate. Formal-methods benchmark:
> Cohen–Mahboubi's Coq quantifier elimination for real closed fields. This lane appears to be
> new ground for the Lean ecosystem.
>
> **Discipline (inherited).** Nothing promoted without a receipt; every stage falsifier-fenced;
> axiom-clean gated modules in `Sundogcert/`; detailed rung receipts in
> [`algo-approx/ALGO_APPROX_OMIN_LADDER.md`](algo-approx/ALGO_APPROX_OMIN_LADDER.md).

## Status ledger (receipts in the ladder doc)

| Rung | Content | Status |
|---|---|---|
| R1 | Dim-1 o-minimality via finite-frontier `Tame`; ReLU/semilinear + QF-semialgebraic structures tame | ✅ 2026-07-02 |
| R2 | Monotonicity PL instance; frontier modulus `≤ 2\|S\|+1` (U-4 bridge); `tame_iff_normalForm` | ✅ 2026-07-02 |
| R3-semilinear | Constructive boolean closure + Fourier–Motzkin projection 2→1; semilinear = a genuine structure in dims 1–2 | ✅ 2026-07-02 |
| R3-semialgebraic | Tarski–Seidenberg QE | **TODO** (below) |
| R4-A | Abstract `OMinStructure` + definable calculus + `S₁ = Tame` capstone | ✅ 2026-07-02 |
| R4-B | **`semilinearStructure : OMinStructure` — the first machine-checked o-min structure** | ✅ 2026-07-03 |
| R4-C/D | Monotonicity Theorem / Cell Decomposition | **SCOPED** (below) |

Modules: `OMinimalOne` … `SemilinearStructure` (34th–42nd), 24 gated headline theorems, all
axiom-clean. Classical anchor for what's landed:
Fourier–Motzkin **is** quantifier elimination for the ordered ℝ-vector-space reduct — the
semilinear structure result is the machine-checked linear fragment of Tarski.

---

## TODO-1 — TS-QE: Tarski–Seidenberg quantifier elimination. [MONTHS — NOT STARTED]

*Claim (when taken up).* Every projection of a semialgebraic set is semialgebraic: quantifier
elimination for the real field, landing the full semialgebraic structure as o-minimal (dim-1
landing already banked: `polyDef_tame`).

*Recon receipts (2026-07-02).* mathlib now has **`IsRealClosed`** (`FieldTheory/IsRealClosed/`)
— the class exists but is young (squares/odd-degree material; no polynomial sign-change/Sturm
machinery, no root-counting). No Sturm chains anywhere in mathlib. `ModelTheory/Algebra/Field/`
has the FO treatment of fields incl. ACF — an in-tree template for the model-theoretic route.

*Staging.*
- **TS-0 (route decision, days):** effective-algebraic (Cohen–Mahboubi pseudo-remainder + sign
  determination, port the Coq design) vs Sturm/CAD-style vs model-theoretic (FO RCF theory over
  `ModelTheory`, following the ACF template). Deliverable: a one-page pre-registered route memo
  with the kill-criteria for each.
- **TS-1:** one-variable sign determination over an ordered/real-closed field (root counting —
  Sturm chains or Cohen–Mahboubi's remainder-sequence bookkeeping). The substrate mathlib lacks.
- **TS-2:** the single-quantifier elimination step (the projection operator on sign conditions).
- **TS-3:** iterate to full QE; close the semialgebraic structure (booleans landed in the
  `PolyDef` style + projection from TS-2); o-min landing via the `polyDef_tame` generalization.

*Falsifier* (`QE_COMBINATORICS_WALL`): the sign-determination combinatorics (the case tables that
make Coq's development large) fail to stay tractable in Lean at this lane's single-owner scale —
in which case the honest fallback is TS-1 alone (one-variable sign determination is independently
valuable and unblocks Sturm-adjacent mathlib gaps).

*Honest scope.* This is the lane's long wall, deliberately queued behind R4-A/B (which produce
the frame TS-QE's result plugs into). Not started; nothing here claims progress.

---

## NEXT — R4: the abstract o-minimal structure (the expedition). [SCOPED 2026-07-02 — AWAITING GO]

*The design keys, fixed up front.*
1. Dimension-indexed definables over `Set (Fin n → ℝ)`.
2. **Substitution closure in one axiom**: definables closed under preimage along `(· ∘ σ)` for
   every `σ : Fin m → Fin n` — this single axiom yields cylinders, coordinate permutations, and
   diagonals at once (the standard slick formulation).
3. Projection along the drop-last coordinate map; booleans per dimension; the order atom
   `{f | f 0 < f 1}` definable.
4. **Rung 1's `Tame` slots in unchanged as the `S₁` axiom** (designed for this from the start).

### R4-A — The interface + definable calculus. ✅ LANDED 2026-07-02

`Sundogcert/OMinimalStructure.lean` (39th module), axiom-clean, 4 gate entries, full build green,
public-safety clean. Delivered, per the pre-registered design keys:

- **`OMinStructure`** — booleans; **substitution closure as one axiom** (`{f | f ∘ σ ∈ A}` for
  every `σ : Fin m → Fin n` = cylinders + permutations + diagonals at once); projection in the
  working `∃`/`Fin.snoc` form; order + singleton atoms; **rung 1's `Tame` as the `S₁` axiom,
  unchanged**.
- **Definable calculus**: inter/diff/finite unions; reversed order, diagonal; rays `Ioi`/`Iio`
  and `Ioo` (the two-variable project-a-pinned-constant constructions); `DefinableFun` (graphs)
  closed under identity, constants, and **composition** (the three-variable projection argument);
  preimages; level/superlevel sets; and the tameness payoffs (`DefinableFun.tame_levelSet` — the
  abstract echo of rung 1's `netDef_tame`, holding in *every* o-minimal structure).
- **The capstone** (`s1_eq_tame` / `definable_dim_one_iff_tame`): **dimension-one definables are
  EXACTLY the tame sets.** ⊇ runs through rung 2's `normalForm_of_tame` + the new **shape lemma**
  (`isOpen_ordConnected_shape`: an open `OrdConnected` set is ∅ / `univ` / `Ioi` / `Iio` / `Ioo`,
  by boundedness cases with `csInf`/`csSup` endpoints) — rung 2 is now load-bearing for the
  abstract interface, as designed.
- **`INTERFACE_MISFITS` did not fire** — the transport friction was contained in `toOne` +
  `toOne_eval`, fixed once. One new build gotcha for the ledger: long qualified names wrap
  `#print axioms` output past the 120-col pretty-printer width and break `#guard_msgs` — gate
  via short aliases (`s1_eq_tame`, `defFun_tame_level`).

*Unchanged fence:* no instance yet — **non-vacuity is R4-B's deliverable.**

### R4-B — The canonical instance: n-dimensional semilinear. [SCOPED 2026-07-02 — 1–2 WEEKS]

*Deliverable.* `semilinearStructure : OMinStructure` — **the first machine-checked nontrivial
o-minimal structure** (classically: QE for the ordered ℝ-vector space / divisible ordered abelian
groups). Discharges R3's "n-dim = named, not claimed" fence and makes R4-A non-vacuous; R4-A's
payoff theorems (`s1_eq_tame`, `DefinableFun.tame_levelSet`, composition) instantiate to free
corollaries about a real structure.

*Recon receipts (mathlib v4.30.0, verified).* `Fin.sum_univ_castSucc` (the front/last sum split —
the load-bearing lemma for eliminating the last variable); `Finset.sum_fiberwise_eq_sum_filter` /
`sum_fiberwise_of_maps_to` (the substitution reindexing); `Fin.sum_univ_one/two`, `Finset.mul_sum`
/ `sum_sub_distrib` (linearity glue). `DecidableEq (Fin n)` is genuine, so the substitution layer
stays fully computable (no classical sign tests needed there — only FM's sign splits are
classical, as in dims 2).

*Syntax.* `AtomN n` = coefficient row `a : Fin n → ℝ` + constant + kind `{gt, eq}`; value
`(∑ i, a i * x i) + c`; `CellN`/`SLN` as list conjunction/union; `Definable A := ∃ S, A = toSet S`
(presentation-as-predicate — the instance wraps syntax in one `∃`).

*Stages (each with its own build receipt).*
- **B1 — syntax + booleans** ✅ LANDED 2026-07-03: `Sundogcert/SemilinearN.lean` (40th module),
  axiom-clean, 2 gate entries (`slInterN_holds`, `slComplN_holds`), GREEN on the first build.
  `AtomN/CellN/SLN` + holds; the R3 boolean-closure proofs ported verbatim with the atom value
  abstracted — the single new fact is `neg_val` (negating row + constant negates the value),
  proved name-risk-free from `sum_add_distrib` + `sum_eq_zero` (the `sum_neg_distrib` name is
  unstable in this mathlib — routed around).
- **B2 — substitution** ✅ LANDED 2026-07-03 (same module, appended; GREEN on the first build):
  `substAtomN σ` sums coefficients over fibers; correctness via `reindex_sum`
  (`Finset.sum_mul` + `sum_fiberwise_of_maps_to`, exactly as recon'd); headline
  `substSLN_toSet` lands in the binding axiom shape `{f | f ∘ σ ∈ toSet S}` verbatim. Fully
  computable (genuine `DecidableEq (Fin n)`), gated, axiom-clean.
- **B3 — n-dim Fourier–Motzkin** ✅ LANDED 2026-07-03: `Sundogcert/FourierMotzkinN.lean` (41st
  module), axiom-clean, 2 gate entries (`projCellN_correct`, `projSLN_toSet`), green on the
  second build (one `field_simp`-residue line — the ledger's known gotcha). The port went
  exactly as pre-registered: `val_snoc` (`Fin.sum_univ_castSucc`) is the only place the last
  variable is split out; **one shared linearity lemma** (`combo_val`) serves the pairwise atoms
  and both substitution kinds; the value lemmas restated with opaque `P`
  (`pos_iff_bnd_lt`/`pos_iff_lt_bnd`/`pairP_lt_iff` + pure-algebra `substP_gt`/`substP_eq`);
  `listMax`/`exists_witness` imported verbatim from the dims-2 module. Headline `projSLN_toSet`
  lands the `definable_proj` axiom shape `{g | ∃ y, Fin.snoc g y ∈ toSet S}` verbatim.
  **`FMN_FRONT_LEAK` did not fire** — no 2-D proof needed the concrete `a·x + c` shape.
- **B4 — atoms + the instance** ✅ LANDED 2026-07-03: `Sundogcert/SemilinearStructure.lean`
  (42nd module), axiom-clean, 3 gate entries, green on the second build.
  **`semilinearStructure : OMinStructure` exists — the first machine-checked nontrivial
  o-minimal structure** (classically: QE for the ordered ℝ-vector space). Every axiom assembled
  from its shape-exact supplier (B1 booleans, B2 `substSLN_toSet`, B3 `projSLN_toSet`); atoms =
  one-cell presentations (`lt` = row `![-1,1]`, singleton = `eq ![1] (−r)`); `tame_dim_one` via
  the `AtomN 1 → Atom₁` bridge into R3's `slHolds₁_tame`. Instantiated payoffs landed:
  `semilinear_s1_eq_tame` (its dim-1 definables are EXACTLY the tame sets) and
  `semilinear_defFun_tame`. **R4-A's non-vacuity fence and R3's n-dim fence are both
  discharged.** Build gotcha for the ledger: dot-notation `.toSet` fails on a `([] : SLN n)`
  literal (the abbrev unfolds to `List`) — use the explicit `SLN.toSet` form.

**R4-B COMPLETE 2026-07-03** — scoped at 1–2 weeks, landed in two days (B1+B2 day one, B3+B4 day
two). None of the three pre-registered falsifiers fired.

*Falsifiers.*
- `NDIM_BOOKKEEPING_WALL` (pre-registered at lane opening): the reindex closure fails to stay
  list-constructive — recon says it shouldn't (decidable fibers), fallback = dims ≤ 3 by hand.
- `FMN_FRONT_LEAK` (new): some 2-D FM proof secretly used the concrete `a·x + c` shape where the
  opaque front `P` doesn't substitute (candidate spots: the pin's `linear_combination`
  identities, which become sum-linearity chains) — fallback = per-spot `Finset.sum` expansion,
  costed as extra grind not a wall.
- `INSTANCE_GLUE_WALL` (new): the `∃`-presentation wrapper makes the axiom-by-axiom set-equality
  glue fight elaboration — mitigation baked into the design: prove everything at the syntax level
  (holds-iff), wrap once per axiom.

*Modules.* Three, for gate hygiene: `SemilinearN.lean` (B1+B2), `FourierMotzkinN.lean` (B3),
`SemilinearStructure.lean` (B4) — 40th–42nd.

*Statement-shape reconciliation (binding).* The instance's axiom forms must match `OMinStructure`
verbatim: `{f | f ∘ σ ∈ A}` (subst), `{g | ∃ y, Fin.snoc g y ∈ A}` (proj),
`{f : Fin 2 → ℝ | f 0 < f 1}` (lt), `{f : Fin 1 → ℝ | f 0 = r}` (singleton),
`Tame {x | (fun _ => x) ∈ A}` (dim-1); each B-stage lands its glue lemma in that exact shape.

### R4-C — The Monotonicity Theorem, abstract. [SCOUTED 2026-07-03 — 2–4 WEEKS]

*Target (C1, house cut-set form).* For every `OMinStructure` `S` and `S`-definable `φ`:
`∃ F : Finset ℝ, ∀ a b, a < b → (∀ s ∈ F, s ∉ Ioo a b) → (φ constant on Ioo a b) ∨
StrictMonoOn φ (Ioo a b) ∨ StrictAntiOn φ (Ioo a b)` — van den Dries Ch. 3 §1 without
continuity. C2 (+ continuity on the pieces) is a separate later scope.

*Scout findings (the architecture, adapted to our machinery).*
1. **The hidden cost-center is definability plumbing**, not mathematics: sets like
   `{x | ∃ v > x, ∀ y ∈ (x,v), φ y > φ x}` cost 50–100 lines each of raw
   substitution/projection. **C0 fixes this once**: a reflected formula layer `Fml` (atoms =
   definable sets pulled back along coordinate maps; `¬`/`∧`/`∃`-last; `∀` derived) with one
   induction `Fml.definable` — after which every vdD "clearly definable" is a term. C0 also
   subsumes parametric slicing (constants = singleton atoms + `∃`).
2. **The pointwise trichotomy is cheap**: at each `x`, the right-sign sets
   `{y > x | φ y ≷/= φ x}` are parametric only in the *real* `φ x` — already covered by R4-A's
   level/superlevel machinery; the three tame sets cover a right-window, their finitely many
   frontiers avoid a small `(x, x+ε)`, and `preconnected_split` (R2-N's engine) forces one to
   contain it. Every point has an eventual right sign (and left, mirrored).
3. **The two-sided local-behavior sets** `D_const/D_inc/D_dec` are formulas ⇒ tame; the bad set
   `B` is their complement. If `B` is infinite it contains an interval
   (`tame_infinite_contains_Ioo`, a normal-form rider); refining by the one-sided sign classes
   gives a subinterval with uniform (left, right) signs; **coherent combos glue, mixed combos
   (local-min-everywhere, etc.) must be killed** — vdD's Lemma-1.3-style sup/inf + tame
   arguments. This kill step is the intricate ~20% and is exactly where
   `MONOTONICITY_STALLS` lives.
4. **The gluing lemmas are pure real analysis, no continuity needed** (verified in scout: the
   sup-chaining closes through overlap midpoints for neighbor-sense local monotonicity, and the
   two-sided version dodges the one-sided dead end that sawtooth counterexamples exploit).
   Mathlib recon: **no local-to-global monotonicity lemma exists** — hand-rolled, ~60–80 lines
   each; `Set.Ioo_infinite` / `StrictMonoOn` / `StrictAntiOn` present.

*Stages.*
- **C0 — the formula layer** ✅ LANDED 2026-07-03: `Sundogcert/OMinimalFormula.lean` (43rd
  module), axiom-clean, 2 gate entries (`Fml.definable`, `Fml.tame_right_inc`). `Fml S n`
  (atoms = definable sets pulled back along coordinate maps; `¬`/`∧`/`∃`-last), **one induction
  `Fml.definable`** (each constructor = one structure axiom, four lines each), derived
  `∨`/`→`/`∀`, convenience atoms (`ltAt`, `eqConstAt`, `memAt`, `graphAt`, and the two-`∃`
  `ltGraph` for `φ(x_i) < φ(x_j)`), and `Fml.tame_one` landing dimension-one formulas in `Tame`.
  **Receipt: `tame_right_inc`** — C1's first D-set (`{x | ∃v > x, ∀y ∈ (x,v), φ x < φ y}`
  tame) as a six-line formula term. `FORMULA_LAYER_LEAKS` did not fire (one real issue: the
  `eval` index must be part of the recursion, not a `variable` binder). Ledger gotchas: literal
  `Fin` indices still need the ascribed snoc helpers; `and_imp` closes the curried/uncurried
  gap after `eval` simp.
- **C1a — toolkit riders** ✅ LANDED 2026-07-03: `Sundogcert/OMinimalTrichotomy.lean` (44th
  module), axiom-clean, 3 gate entries. `tame_infinite_contains_Ioo` (normal form + a ball
  inside a nonempty open piece); `tame_sublevelSet` (the R4-A payoff gap-fill by tame
  booleans); `exists_right_window`/`exists_left_window` (frontier-avoiding windows via
  `Finset.min'`/`max'`); **`eventual_right_sign` / `eventual_left_sign`** — the pointwise
  trichotomies, exactly per scout: three tame comparison sets parametric only in the real
  `φ x`, a window avoiding their finitely many frontier points, `preconnected_split` forcing
  full-or-empty, the midpoint picking the sign. Ledger gotcha: `₊`/`₋` are not legal Lean
  identifier characters (subscript digits are).
- **C1b — sign partition + refinement** ✅ LANDED 2026-07-03: `OMinimalSignPartition.lean`
  (45th module), axiom-clean, 2 gate entries, **green on the first build**. Design improvement
  over the scout: the two-sided D-sets need **no new formulas** — neighbor-sense
  locally-increasing is exactly `rightAbove ∩ leftBelow` — so the six one-sided sign-class sets
  come from **two formula templates** (`tame_right_template`/`tame_left_template`; the six tame
  proofs are two-liners), the D-sets are tame by intersection, and `eqGraph` (one-`∃` value
  equality) is the only new combinator. Headline **`sign_partition`**: a finite cut-set (the
  four frontiers) with exactly one behavior class — `locConst`/`locInc`/`locDec`/`badSet` — per
  gap, by the full-or-empty engine + midpoint election. Pointwise covers
  (`right_sign_cover`/`left_sign_cover`) banked for C1d.
- **C1c — the gluing lemmas** ✅ LANDED 2026-07-03: `OMinimalGluing.lean` (46th module),
  axiom-clean, 3 gate entries, green on the second build. Design improvement over the scout:
  **one engine covers all three classes** — the sup-chaining argument only uses transitivity,
  so `rel_propagate` is proved once over an abstract transitive relation (`T = {z ∈ [u,w] |
  (u,z] absorbed}`; the sup is absorbed through its left window via a chained `T`-member; `s <
  w` is impossible via the right window) and instantiated with `<`, flipped-`<`, flipped-`=`:
  `strictMonoOn_of_locInc`, `strictAntiOn_of_locDec`, `eqOn_of_locConst`. No continuity
  anywhere; no `-φ` mirroring needed. Two fix rounds (the `subst` rename; `le_or_gt` again).
- **C1d — mixed-sign kills + assembly** [~1–2 sessions, THE RISK]: local-extremum-everywhere
  impossibilities; assemble C1. Falsifier `MONOTONICITY_STALLS` scoped precisely here;
  fallback = C1-weak (per-gap uniform sign classes), honestly reported as the partial.

Total ≈ 4½–5½ sessions, consistent with the 2–4-week pre-registration. C2 scoped separately
after C1 lands.

### R4-D — Cell decomposition, dimension 2 first. [2–4 WEEKS; general n = the long pole]
Decompose any definable `A ⊆ ℝ²` into finitely many cells (points/intervals over graphs and
bands of definable continuous functions), via the **uniform finiteness lemma** (fiberwise
boundary counts are uniformly bounded — where o-minimality genuinely bites; vdD's I₂/II₂).
General n (the full induction) is the expedition's long pole and is *not* promised by this scope.
*Falsifier* (`UNIFORM_FINITENESS_WALL`): the fiber-bound lemma resists at dim 2 — the honest null,
worth publishing as the located wall.

### R4-E — The payoff hook (rides the others). [GARNISH]
Cell decomposition ⇒ definable families have finite VC dimension (Pillay–Steinhorn / NIP) — the
bridge back to this lane's origin: ReLU families are definable in an o-minimal structure, so the
approximation lane's objects carry uniform finiteness for free. Recorded as the *why it matters
here*; cited, not claimed, until R4-D exists.

*Order:* A → B → C → D (E rides). A+B are bankable in ~2 weeks even if C/D stall; every stage has
a standalone receipt.

## What this lane is NOT

- Not a Millennium/foundations claim; the theorems are classical — the contribution is
  machine-checked cores where the Lean ecosystem has none.
- Not promoted beyond receipts: TS-QE is a TODO, R4 is a scope; only R1–R3-semilinear are landed.
- Not a strategy document: statuses and fences only.

> Cross-links: [`algo-approx/ALGO_APPROX_OMIN_LADDER.md`](algo-approx/ALGO_APPROX_OMIN_LADDER.md)
> (rung receipts + gotchas) · `Sundogcert/OMinimalOne.lean` … `Sundogcert/FourierMotzkin.lean` ·
> [`algo-approx/ALGO_APPROX_CONJECTURE_SLATE_4.md`](algo-approx/ALGO_APPROX_CONJECTURE_SLATE_4.md)
> (U-4, the origin).
