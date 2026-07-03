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
| R4-B/C/D | n-dim semilinear instance / Monotonicity Theorem / Cell Decomposition | **SCOPED** (below) |

Modules: `OMinimalOne` / `OMinimalRate` / `OMinimalNormalForm` / `Semilinear` / `FourierMotzkin`
/ `OMinimalStructure` (34th–39th), 16 gated headline theorems, all axiom-clean. Classical anchor for what's landed:
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
- **B1 — syntax + booleans** [~1 session]: `AtomN/CellN/SLN` + holds; port the R3 boolean-closure
  proofs with the atom value abstracted to one opaque real (they never used the 2-var shape);
  atom complement negates the coefficient row pointwise (`sum` linearity glue).
- **B2 — substitution** [~½ session]: `substAtom σ` sums coefficients over fibers
  (`b j := ∑ i ∈ univ.filter (σ · = j), a i`); correctness = the fiberwise-sum lemma +
  `Finset.sum_mul`. Yields the `definable_subst` axiom, matching `{f | f ∘ σ ∈ A}` exactly.
- **B3 — n-dim Fourier–Motzkin** [~1–2 sessions]: eliminate the LAST variable: split each atom's
  row via `Fin.sum_univ_castSucc` into an opaque front value `P := ∑ front + c` and the last
  coefficient `b`; the entire 2-D architecture ports with `P` replacing `a·x + c` — eq-pin
  substitution (×`b²` sign-free trick, rows combine linearly), lower/upper split, division-free
  pairwise atoms, and the explicit between-the-bounds witness. `listMax`/`listMin`/
  `exists_witness` import VERBATIM from `FourierMotzkin`; the three value-lemmas
  (`holds_iff_bnd_lt`, `pair_lt_iff`, …) are restated with opaque `P` (cleaner than the 2-D
  originals). Yields `definable_proj` in the `∃`/`snoc` form exactly.
- **B4 — atoms + the instance** [~½–1 session]: `lt` = row `![-1, 1]` (`Fin.sum_univ_two`);
  singleton = `eq ![1] (−r)` (`Fin.sum_univ_one`); `tame_dim_one` by bridging `AtomN 1` to R3's
  `Atom₁` and reusing `slHolds₁_tame`. Assemble `semilinearStructure : OMinStructure`; land the
  instantiated payoff corollaries as gates.

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

### R4-C — The Monotonicity Theorem, abstract. [2–4 WEEKS]
For any `OMinStructure` and definable `f : ℝ → ℝ`: a finite partition of ℝ into points and open
intervals on which `f` is constant or strictly monotone (stage C1), and continuous (stage C2) —
van den Dries Ch. 3 §1. The proof is elementary but intricate: the sets "locally constant /
locally increasing / locally decreasing at x" are definable (projection + booleans), o-minimality
partitions, and infimum arguments glue. This is the first genuine *theorem of the abstract
theory* in Lean.
*Falsifier* (`MONOTONICITY_STALLS`): the infimum-gluing arguments fight conditional-completeness
API; staged fallback = C1 without continuity, honestly reported.

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
