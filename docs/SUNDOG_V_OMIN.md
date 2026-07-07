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
| R4-C | **THE MONOTONICITY THEOREM, with continuity** (`monotonicity_theorem_continuous`) | ✅ 2026-07-04 |
| R4-D | **UNIFORM FINITENESS (the Finiteness Lemma)** — `uniform_finiteness`; the pile-wall dissolved via the classical tube | ✅ 2026-07-06 |
| R4-D5 | **CELL DECOMPOSITION FOR ℝ² (CDT₂)** — `cell_decomposition : ∃ cells, IsCellDecomp cells A` | ✅ 2026-07-06 |

Modules: `OMinimalOne` … `OMinimalCellDecomp` (34th–61st), 80 gated headline theorems, all
axiom-clean; audits 8540 → 8605 GREEN. Classical anchors for what's landed:
Fourier–Motzkin **is** quantifier elimination for the ordered ℝ-vector-space reduct (the
machine-checked linear fragment of Tarski); the Monotonicity Theorem, the Finiteness Lemma,
and CDT₂ are van den Dries Ch. 3 over the abstract structure. Commit state 2026-07-06:
sundogcert committed AND pushed through `757222e`; this roadmap committed `5e2ef6f4`.

---

## TS-QE — Tarski–Seidenberg quantifier elimination. [OPENED 2026-07-06; MONTHS]

*Claim.* Every projection of a semialgebraic set is semialgebraic: quantifier elimination for
the real field, landing `semialgebraicStructure : OMinStructure` — the second (and first
genuinely nonlinear) witness, through which Monotonicity, UF, and CDT₂ instantiate at the real
field for free.

### TS-0 — ROUTE DECIDED 2026-07-06: Cohen–Hörmander, set-level, over concrete ℝ.

*The decision.* Follow the **Cohen–Hörmander sign-matrix proof** (the simplest known effective
TS; Harrison's HOL Light route), formalized **set-level over the concrete reals** — not a
reflective decision procedure, not FO model theory. Parametric coefficients handled by
**leading-coefficient branch trees** over `MvPolynomial.finSuccEquiv`
(`MvPolynomial (Fin (n+1)) ℝ ≃ₐ Polynomial (MvPolynomial (Fin n) ℝ)` — in mathlib, verified).
Working over ℝ (house-fenced ℝ-specific devices allowed) buys the univariate core by *real
analysis* — `Polynomial.finite_setOf_isRoot`, `Polynomial.continuous` + IVT, Rolle-adjacent
monotonicity — instead of the abstract-RCF algebra that makes the Coq development large.

*Routes rejected.* Cohen–Mahboubi port (subresultants + BKR sign determination: the full
abstract-RCF algebra stack, none of it in mathlib — longest runway); CAD (strictly heavier);
model-theoretic FO route (`ModelTheory` has Presburger only, no RCF; and our target is
set-level `OMinStructure`, so FO syntax is an impedance mismatch).

*Prior-art receipts (re-checked 2026-07-06).* No Lean 4 TS/QE/semialgebraic anywhere; mathlib's
`IsRealClosed` still embryonic (squares/odd powers); no Sturm; ModelTheory QE = Presburger only.
Coq: Cohen–Mahboubi LMCS 2012 (+ a Coq CAD). Isabelle/HOL: a complete real QE (2022).
HOL Light: Harrison's Cohen–Hörmander. Lean remains open ground.
Key mathlib assets verified: `finSuccEquiv`, `EuclideanDomain ℝ[X]` (mod/div),
`Polynomial.finite_setOf_isRoot`, `Polynomial.continuous`, `MvPolynomial.rename`.
Already banked in-lane: `polyDef_tame` (R1: QF dim-1 tameness), the whole abstract theory
(R4) waiting as the payoff socket.

*The restaged ladder.*
- **TS-1 — the univariate sign partition** ✅ LANDED 2026-07-06: `PolySignPartition.lean`
  (62nd module), axiom-clean, 3 gate entries (`poly_sign_constant`, `family_sign_partition`,
  `family_sign_eq`), green on the second build (one fix: `push Not` already normalizes
  `¬(0 < ·)` to `≤`). New arc namespace `Sundog.TarskiQE`. The IVT core
  (`poly_sign_constant`: a sign change without a root manufactures one via
  `intermediate_value_Ioo`/`Ioo'` — 20 lines, exactly the TS-0 payoff of working over ℝ);
  `finite_familyRoots`; `family_sign_partition` (ONE Finset cut set, C-avoiding interface
  matching the R4 plumbing so TS-2 composes); `family_sign_eq` (sign-vector constancy — the
  clause TS-2d's correctness cites); `tame_zeroSet` (exact-sign atoms complete R1's
  `polyDef_tame` picture).
- **TS-2 — the parametric elimination step** [MONTHS — the pole proper]. Representation
  `SAN n` (sign-condition lists on `MvPolynomial (Fin n) ℝ`, in the `SLN` style); head-view
  through `finSuccEquiv`. Sub-rungs:
  - **TS-2a** — branch trees ✅ LANDED 2026-07-06: `PolyBranchTrees.lean` (63rd module),
    axiom-clean, 3 gate entries (`spec_eval_cons`, `resolve_spec`, `spec_natDegree_eq`),
    green round 3 (dite/ite mismatch in WF-def unfolds — keep dite, use `dif_pos/dif_neg`;
    `eraseLead_natDegree_le` lands at `natDegree − 1`; beta-unreduced motives need `change`;
    `famDegrees` noncomputable). `spec g` (coefficient evaluation) bridged to
    `MvPolynomial.eval (Fin.cons y g)` by mathlib's `eval_eq_eval_mv_eval'` — verbatim, no
    work; `resolve g` (strip vanishing leading coefficients, WF on support size) with a
    reusable `resolve_cases` induction principle; specialization preserved
    (`resolve_spec`) and DEGREE-EXACT on resolutions (`spec_natDegree_eq`,
    `spec_eq_zero_iff`, `spec_leadingCoeff_eq` via `natDegree_map_of_leadingCoeff_ne_zero`);
    the finite `truncChain` every branch lands in + compositional branch conditions
    (`resolve_eq_self_iff`, `resolve_of_lead_vanish`); `famDegrees` + mathlib's
    Dershowitz–Manna well-founded order (in-tree! `instWellFoundedIsDershowitzMannaLT`) as
    the pre-registered 2d measure. First-vs-last variable mismatch (finSuccEquiv vs snoc)
    noted: recovered by a rename at TS-3.
  - **TS-2b** — the mod-trick ✅ LANDED 2026-07-06: `PseudoRemainder.lean` (64th module),
    axiom-clean, 3 gate entries (`pseudoModExp_identity`, `emod_eval_at_root`,
    `sign_transfer`), green round 4 (classical decidability for ring-equality ites;
    `ring`/`linear_combination` treat `a^k` with opaque `k` as an atom — normalize
    `pow_succ` first AND keep `pstep P Q` atomic, feeding its definition as a separate
    equation with coefficient `−c^k`). **Pseudo-division built from scratch, generic over
    any `CommRing`** (mathlib has none — giftable): `pstep` exact top-cancellation (no
    domain hypothesis, ~coeff bashing over `coeff_X_pow_mul'` + `degree_lt_iff_coeff_zero`),
    `pseudoModExp` WF recursion on a zero-guarded degree measure, identity
    `∃ S, C c^k · Q = S·P + rem` with degree guard. **The even-exponent trick** (`emod`/
    `eexp`): pad `k` to even — the transfer factor `c(g)^k` is strictly positive on the
    live branch (`Even.pow_pos`), so classical Cohen–Hörmander's sign-twist bookkeeping
    disappears. `emod_eval_at_root` (the `S·P` term dies at roots) and the headline
    `sign_transfer`: three-way sign equality of `Q` and its evenized remainder at every
    root of `spec g P` — the elimination's degree-descent engine.
  - **TS-2c** — between-roots signs ✅ LANDED 2026-07-06: `PolyDerivZones.lean`
    (65th module), axiom-clean, 3 gate entries (`spec_derivative`,
    `strictMonoOn_sign_zones`, `deriv_free_sign_zones`), **GREEN ON THE FIRST BUILD**.
    `spec_derivative` (the derivative family is parametric — `derivative_map`, one line);
    derivative sign ⇒ strict monotonicity on the closed interval
    (`strictMonoOn_of_deriv_pos` + `Polynomial.deriv`); the sign-zone trichotomies
    (endpoint trichotomy + IVT: all-positive / all-negative / ONE interior root with clean
    strict zones, both orientations); the capstone `deriv_free_sign_zones` — on any
    derivative-root-free interval a nonzero polynomial has at most one root and one of six
    diagrams (TS-1's `poly_sign_constant` on the derivative picks the orientation; the
    constant case dies by `eq_C_of_derivative_eq_zero`); end behavior:
    `eventually_pos/neg_atTop` (tendsto asymptotics; constants separate) and the `−∞`
    mirrors with the `(−1)^deg` parity twist via `comp (−X)` + `leadingCoeff_comp`.
    The 2d recursion's real-analysis inputs are now complete.
  - **TS-2d** — the recursion assembly + correctness. STAGED (summit push, 2026-07-06):
    - **TS-2d-1 — base camp** ✅ LANDED 2026-07-06: `SignDiagrams.lean` (66th module),
      axiom-clean, 3 gate entries (`exists_diagram`, `elim_of_diagramPartition`,
      `diagramPartition_of_constants`), green round 3 (SignType's constructor order is
      zero-first; `Finset.sort_sorted_lt` → `sortedLT_sort` + `.pairwise`; a goal-side
      rewrite needed `conv_lhs`). **`SADef`** (QF-semialgebraic parameter sets, `PolyDef`
      style) with the full closure algebra (inter/zero/`signEq` atoms/finite list unions);
      **the sign-diagram representation**: `signVec`, `ColsFrom` (structural recursion on
      the sample list with an `Option`-barrier — `∀ l ∈ lo, l < y` makes `none` vacuous),
      `Realizes` (sorted samples containing every root of every nonzero member);
      `exists_diagram` (sorted root Finset + IVT gap constancy); `realizes_exists_iff`
      (columns are exactly the attained sign vectors — cells are nonempty);
      **`DiagramPartition` = THE SUMMIT STATEMENT** (finitely many `SADef` branches, each
      carrying one diagram valid branch-wide — branch-wide validity is load-bearing: a
      cover with per-point diagrams would break the union argument);
      `elim_of_diagramPartition` (the elimination is a filter-union corollary);
      `diagramPartition_of_constants` (the induction's base case via the `allSignVecs`
      enumeration).
    - **TS-2d-2 — the reconstruction step** [in progress; the wall watches here].
      Pitches: 2a padding removal ✅ / 2b P-column augmentation ◐ (foothold landed) /
      2c root-insertion splice ✅ (toolkit landed) / 2d-3 walk ✅ (see below).
      - **2d-2a ✅ LANDED 2026-07-06**: `DiagramNormalize.lean` (67th module), axiom-clean,
        3 gate entries (`dropPadding_paddingFree`, `realizes_dropPadding`,
        `realizes_samples_root`), green round 4. `dropPadding` merges away point-columns
        with no zero entry — the diagram itself knows its padding, so the operation is
        branch-uniform; realization is preserved (in the drop case NO member can be zero
        at `g` — a zero member puts `zero` in every column — the dropped sample is a root
        of nothing, and the flanking columns provably agree by `signVec_eq_of_gap` across
        it); the output is padding-free, and in a padding-free realized diagram EVERY
        sample is a root of some member — the TS-2b transfer reaches every sample.
        Formalization lesson (banked): WF defs with LIST PATTERNS don't yield usable
        unfold equations — use fuel-structured auxiliaries (structural recursion, clean
        `simp` equations) + a fuel-irrelevance congruence + a thin wrapper with per-shape
        rewrite lemmas.
      - **2d-2b ◐ FOOTHOLD LANDED 2026-07-06**: `DiagramAugment.lean` (68th module),
        axiom-clean, 4 gate entries, green round 2. The load-bearing design move:
        **prefix projection** (`realizes_project_prefix`) — with the recursive family
        laid out `base ++ base.map (emod · P)`, a realized diagram of the whole family
        restricts to a realized diagram of the base (columns truncated to the prefix
        length, same samples), and a `dropPadding` pass on the PROJECTED diagram makes
        every sample a BASE root. This dissolves the remainder-only-sample worry: a
        sample where only a remainder vanishes can't be merged in the FULL diagram (the
        remainder's own entry flips), but after projection its column carries no zero
        and drops cleanly. Plus `sign_transfer_signType` (TS-2b's three-way transfer as
        one SignType equation) and `spec_eq_zero_of_gap_roots` (interval-vanishing ⇒
        zero polynomial — the degeneracy that turns a zero P'-gap-column into "P is
        constant", banked for 2c). Integration receipt `augment_foothold`: live base
        leads ⇒ a padding-free realized base-diagram exists at EVERY sample of which
        P's sign equals the sign of `emod q P` for a base member q vanishing there.
        Remaining in 2b: the augmentation as diagram DATA (branch-uniform column
        rewriting via positions — the column-to-member indexing), P's sign on gap/ray
        columns from flanking samples + monotonicity.
      - **2d-2c ✅ TOOLKIT LANDED 2026-07-06**: `DiagramGraft.lean` (69th module),
        axiom-clean, 4 gate entries (`colsFrom_insert`, `gap_cross_mono`,
        `ray_left_cross_mono`, `ray_right_cross_mono`), GREEN ON THE FIRST BUILD
        (round 2 = two unused-binder lints). The complete per-gap/per-ray graft kit:
        **keyed gap zones** (all-pos / all-neg / one-root-with-zones SELECTED by the
        flank signs — the walk never refutes mismatched disjuncts of the TS-2c
        trichotomies), **ray monotonicity** on `Iic`/`Ici` from derivative sign (shrink
        to `Icc`), **keyed ray zones** (boundary-sample sign + end sign from
        `eventually_*` + branch data; a crossing manufactures its far witness from the
        eventual bound via `min`/`max` and grafts exactly one root), and the
        **`ColsFrom`-level graft mechanism**: `colsFrom_insert` splits the first gap
        into `[gap, point, gap]`, `colsFrom_insert_last` splits the terminal ray;
        `colsFrom_graft_root` = the receipt that for the old family the three columns
        just repeat (nothing moves but the sample). Constant case needs no new lemma
        (`spec_eq_zero_of_gap_roots` + `eq_C_of_derivative_eq_zero`). Because `P'` is a
        base member, at most one `P`-root per gap/ray — the kit covers every case the
        walk meets.
    - **TS-2d-3 — the walk + the descent** [walk ✅]:
      - **THE WALK ✅ LANDED 2026-07-06**: `DiagramWalk.lean` (70th module), axiom-clean,
        3 gate entries (`colsFrom_pairwise`, `graftWalk_colsFrom`, `realizes_graftWalk`),
        **GREEN ON THE FIRST BUILD** (~300 lines, the arc's hardest single construction).
        `GapPlan` (`flow s` / `graft s₁ s₂`) + per-sample sign list = the annotation as
        DIAGRAM-LEVEL DATA; `graftWalk sP plans D` = the `P :: F` diagram as a **pure
        list function** of the annotation and the old columns — the branch-uniformity
        payload: branch-constant inputs give ONE branch-wide output diagram, and only
        the validity contract (`GraftData`, `ColsFrom`-shaped Option-barrier recursion)
        mentions the point `g`. The walk theorem produces `ξ'` (old samples + grafted
        roots) with `ColsFrom g (P::F) lo ξ' (graftWalk sP plans D)`, keeps old samples,
        and catches EVERY root of a nonzero `spec g P`: a root inside any flow/zone
        forces that zone's sign to 0, so P vanishes on an interval and dies by 2d-2b's
        degeneracy lemma — no strictness side conditions needed. `P = 0` needs no
        special case (all-zero annotation is valid; coverage is `≠ 0`-guarded; Realizes
        exempts zero members). Capstone `realizes_graftWalk`; sortedness recovered by
        the new utility `colsFrom_pairwise` (samples clear the barrier + strictly
        increasing, by barrier-chaining induction).
      - **THE ANNOTATION ✅ LANDED 2026-07-06**: `DiagramAnnotate.lean` (71st module),
        axiom-clean, 3 gate entries (`gapValid_incr`, `gapValid_left_incr`,
        `exists_graftData`), **GREEN ON THE FIRST BUILD** (~470 lines of casework, zero
        fix rounds). `(sP, plans)` computed from what the machinery provides:
        `BotSign`/`TopSign` end-sign contracts (existence from leading-coefficient
        trichotomy via TS-2c's `eventually_*`; the `zero` case = the zero polynomial,
        dissolved against strict monotonicity wherever it conflicts); decision functions
        `gapPlanMono`/`gapPlanAnti` (flank signs select flow-vs-graft); TEN validity
        lemmas — bounded gap / left ray / terminal ray / whole line × incr/anti, plus
        the constant-P pair — each discharged by exactly one 2d-2c keyed zone lemma
        (the keyed design paid off: no disjunct refutation anywhere); sign-constancy
        on root-free rays/line from the de-privatized `sign_eq_of_no_root_Icc`.
        Capstone `exists_graftData`: derivative-roots-among-samples ⇒ a valid
        annotation exists (per-gap trichotomy via TS-1 `poly_sign_constant`), so
        `realizes_graftWalk` fires end-to-end.
      - **The descent [next]**: make the annotation branch-CONSTANT (sample signs via
        the 2d-2b transfer = remainder column entries; end signs via resolve-branch
        lead-sign/parity conditions as `SADef` refinements); the Dershowitz–Manna
        induction wrapping 2d-2; degree bookkeeping through `resolve`-branches;
        `DiagramPartition` for every family; `elim_signVector` unconditionally.
  *Falsifier* (`QE_COMBINATORICS_WALL`, carried over): the sign-matrix bookkeeping fails to
  stay tractable at single-owner scale — fallback = TS-1 + TS-2b/2c as independently valuable
  univariate machinery, wall published with location.
- **TS-3 — the structure** [2–3 sessions once TS-2 lands]: `semialgebraicStructure`:
  booleans (list ops), substitution (`MvPolynomial.rename`/composition), `definable_lt`/
  `definable_singleton` (linear polynomials), projection (TS-2), `tame_dim_one` (TS-1).
  Capstone `semialgebraic_s1_eq_tame`; payoffs: Monotonicity/UF/CDT₂ at the real field.

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
- **C1d — mixed-sign kills + assembly** ✅ LANDED 2026-07-04: `OMinimalMonotonicity.lean` (47th
  module), axiom-clean, 2 gate entries (`badSet_finite`, `monotonicity_theorem`), green on the
  third build (an import path + a paren — no mathematical fight). **`MONOTONICITY_STALLS` did
  not fire.** The kills, per scout: a bad interval carries uniform one-sided signs on a
  frontier-free subwindow (`elect3` twice); the three coherent combos contradict badness; the
  four eq-mixed combos die by short window contradictions (`kill_rightEq_left` /
  `kill_leftEq_right`, each handling both strict directions through an abstract `P` with a
  `ne`-extractor); the two extremum combos die by **countability** — strict neighbor-extrema of
  any real function inject into ℚ × ℚ via rational witness windows (`countable_min_pts`; two
  minima sharing a window each dominate the other), `Ioo` is uncountable
  (`Cardinal.mk_Ioo_real` + `aleph0_lt_continuum`), and the max case is the min case for `-φ`.
  *Honest note:* the countability kill is ℝ-specific — over a general RCF, van den Dries's
  constant-or-injective route is required; named refinement, not claimed.

**R4-C1 = THE MONOTONICITY THEOREM, COMPLETE 2026-07-04** (`monotonicity_theorem`): for every
o-minimal structure and definable `φ`, a finite cut-set outside of which `φ` is piecewise
constant, strictly increasing, or strictly decreasing. Scoped at 2–4 weeks; landed in ~2 days
(C0+C1a day one, C1b+C1c+C1d day two). One falsifier partially engaged (`MONOTONICITY_STALLS`
routed via the ℝ-specific kill, honestly fenced); the rest never fired.

### R4-C2 — continuity on the pieces. [SCOPED 2026-07-04 — ~2½–3 SESSIONS]

*Target.* `monotonicity_theorem_continuous`: the C1 statement with `ContinuousOn φ (Ioo a b)`
added to every gap (the constant case is trivially continuous, so the conclusion is
`(constant ∨ StrictMonoOn ∨ StrictAntiOn) ∧ ContinuousOn`).

*Two traps flushed at scoping (both would have been landmines):*
1. **No arithmetic atoms** — the structure has only order + singletons + graphs, so ε-δ
   continuity is *not expressible*. Route: ℝ's topology is the order topology, so
   `ContinuousAt` has a pure `<`-characterization —
   `∀ p q, p < φ x < q → ∃ u v, u < x < v ∧ ∀ y ∈ (u,v), p < φ y < q` — which IS a formula.
   Bridge lemma `continuousAt_iff_order` via `nhds_basis_Ioo` + `HasBasis.tendsto_iff`
   (recon: both present; ℝ has `NoMaxOrder`/`NoMinOrder`).
2. **The `-φ` trick is unavailable wherever definability is consumed** — negation is not an
   atom, so `DefinableFun (-φ)` is not derivable (C1d's `kill_max` got away with it because
   the countability kill needed no definability). All decreasing cases in C2 must be mirrored
   manually.

*Stages.*
- **C2a — bridge + the discontinuity set** ✅ LANDED 2026-07-04: `OMinimalContinuity.lean`
  (48th module), axiom-clean, 2 gate entries (`continuousAt_iff_order`, `tame_discSet`), green
  with zero warnings by the third build (three one-line fixes). **Neither falsifier fired**:
  the `nhds_basis_Ioo` + `tendsto_iff` plumbing went through directly (`ORDER_BRIDGE_GAP` no),
  and the depth grind was tamed by a design refinement — **split into usc/lsc**: two
  five-coordinate formulas (`x, ∀q, ∃u, ∃v, ∀y`) instead of one six-coordinate one, recovering
  `ContinuousAt` as the conjunction (`continuousAt_iff_usc_lsc`; the max/min window
  intersection lives at the meta level where arithmetic is free). New combinators
  `ltCoordGraph`/`ltGraphCoord` (`x_j ≶ φ(x_i)`); snoc battery extended to depths 3→4, 4→5.
  The five-layer eval-rewrites went through in one `simp only` each.
- **C2b — the kill** ✅ LANDED 2026-07-04 (same module, appended; green on the second build —
  two rewrite-orientation fixes from the `by_contra`/`not_lt` equation flip). `tame_image`
  (the image formula: `∃z, z ∈ Ioi c ∧ z ∈ Iio d ∧ φ z = p` — `memAt` over `definable_Ioi/Iio`
  earning their keep); `strictMonoOn_exists_continuityPt` + the hand-mirrored
  `strictAntiOn_exists_continuityPt` — image tame + infinite (`infinite_image_iff` + `injOn`)
  ⇒ contains `(p,q)` (`tame_infinite_contains_Ioo`, fourth use) ⇒ the midpoint's preimage is
  order-continuous by the squeeze; plus `constOn_continuousAt` (free). `IMAGE_KILL_GAP` did
  not fire.
- **C2c — assembly** ✅ LANDED 2026-07-04, **green on the first build**: `discSet_finite`
  (interval of discontinuities → C1-free sub-gap → constant-or-monotone → a continuity point,
  contradiction) and the headline **`monotonicity_theorem_continuous`**.

**R4-C COMPLETE 2026-07-04** — `monotonicity_theorem_continuous`: for every o-minimal
structure and definable `φ`, a finite cut-set outside of which `φ` is piecewise constant,
strictly increasing, or strictly decreasing, **and continuous** — van den Dries Ch. 3 §1 in
full, machine-checked. C1 scoped 2–4 weeks / landed 2 days; C2 scoped 2½–3 sessions / landed
in 2. Module `OMinimalContinuity.lean` (48th); 6 gate entries across C2. Remaining R4: **D —
cell decomposition, dimension 2** (the uniform finiteness lemma), then the lane's long walls
(TS-QE, general-n cells).

### R4-D — Cell decomposition, dimension 2. [SCOUTED 2026-07-04 — THE WALL, STAGED]

*Core target (UF, the o-min bite).* A definable `A ⊆ ℝ²` with every fiber finite has uniformly
bounded fibers. CDT₂ then follows as assembly (finite-fiber part = finitely many graphs; the
rest = bands between consecutive graph curves) — D5, scoped after UF.

*Scout verdict — a confidence gradient, stated honestly.* Unlike C1/C2, the finisher is **not
fully nailed in-scout**. What is nailed:
1. **D0 — slicing + fiber tameness** ✅ LANDED 2026-07-04: `OMinimalSlice.lean` (49th module),
   axiom-clean, 3 gate entries (`tame_slice`, `definable_fiberFrontier`,
   `fiberFrontier_fiber_finite`), **green on the first build, zero warnings after two unused
   simp args**. `tame_slice` discharges C0's slicing IOU (`∃z, z = x ∧ (z,y) ∈ A`);
   `mem_closure_iff_order` (the `Ioo`-basis closure bridge); `definable_fiberFrontier` via a
   parameterized closure block (`closBlock`, C1b-template style) instantiated at the `A`-atom
   and its negation, glued by `frontier_eq_closure_inter_closure`;
   `fiberFrontier_fiber_finite` — the fiber of `Y` at `x` *is* `frontier(A_x)` and `Tame` is
   frontier-finiteness, so the finiteness is one rewrite.
2. **D1 — the counting formulas** ✅ LANDED 2026-07-04: `OMinimalCounting.lean` (50th module),
   axiom-clean, 3 gate entries (`Fml.eval_reindex`, `tame_countSet`,
   `infinite_fiber_of_mem_all`), green by the third build (a dotted-notation qualification, a
   `notMem` rename, an `omega`). **Formula-layer upgrade shipped**: `Fml.reindex` — pullback
   of *whole formulas* along coordinate maps (`∃` threads through
   `extendMap σ = snoc (castSucc ∘ σ) last`; semantics by one clean pattern-based induction,
   no literal indices). The counting recursion: semantic `AboveCount` mirror + syntactic
   `gtCount` recursing through `reindex ![0,2]` at the freshly bound point; `eval_gtCount`
   closed in ONE `simp only` per case. Chain kit banked for D2: `countSet_anti`,
   `aboveCount_exists_finset` (strictly-increasing witnesses ⇒ distinct),
   `aboveCount_of_finset` (`min'`-peeling), `mem_countSet_of_finset`, and
   `infinite_fiber_of_mem_all`.
3. **D2 — the dichotomy kill + rank functions** ✅ LANDED 2026-07-04:
   `OMinimalPointFns.lean` (51st module), axiom-clean, 3 gate entries
   (`exists_interval_in_countSet`, `isNth_exists`, `definableFun_nthFn`), green by the third
   build (omega needed the beta-reduced sInf link spelled out; `Finset.exists_subset_card_eq`
   is the current name; `open Classical in` must precede the docstring). **The dichotomy,
   machine-checked exactly as scouted**: infinite `B_k` → interval directly
   (`tame_infinite_contains_Ioo`, fifth use); finite `B_k` → the tail's cardinalities take
   their `sInf` at some `B_{k+j₀}`, every later set equals it by
   `Set.eq_of_subset_of_ncard_le`, its point lies in every `B_{k'}` →
   `infinite_fiber_of_mem_all` contradicts fiber-finiteness. **Rank functions**: `BelowCount`
   mirror (`ltCount` formula, `max'`-peeling construction, extraction); exact-rank
   `IsNth A j x z` (member + exactly `j` below) — uniqueness is four lines (the lower of two
   rank-`j` points hands the higher an illegal `(j+1)`-th below-witness); existence by
   min-peeling above the previous rank point, with the two "overflow" kills running through
   `Finset.pred_card_le_card_erase`; the totalized `nthFn` (0 off-domain) has definable graph
   via a 4-piece formula (`(∃-rank ∧ rank) ∨ (¬∃-rank ∧ z = 0)`) — `definableFun_nthFn` is
   D3's direct input to the continuous Monotonicity Theorem.
4. **D3 — the k-curve extraction** ✅ LANDED 2026-07-04: `OMinimalCurves.lean` (52nd module),
   axiom-clean, 3 gate entries (`isNth_lt`, `isNth_exists_of_mem_countSet`,
   `exists_k_ordered_curves`), green on the SECOND build (one fix: a `BelowCount`-match
   behind a metavariable won't unify — pin `(k' := j'+1)` explicitly). Headline
   `exists_k_ordered_curves`: for every `k`, one interval on which the `k` rank functions
   are **genuine** (exact rank, graphs in `A`), **continuous**, and **strictly ordered** —
   `k` pairwise disjoint definable curves over one base. Assembly = D2's dichotomy interval
   + `monotonicity_theorem_continuous` per rank (via `definableFun_nthFn`) + `choose` over
   the `k` cut-sets + one `Finset.induction` avoidance window (`avoid_finset`: a finite set
   can't block every subinterval — split at the offending point, keep the left piece).
   `isNth_lt` = the disjointness engine: equality or inversion of ranks both manufacture an
   illegal below-witness (four lines each).
5. **D4 — THE WALL, DISSOLVED (replan receipt 2026-07-05).** Prior-art recovery (vdD Ch. 3
   Finiteness Lemma via the Ghorbani REU exposition, fetched + read in full) shows the
   classical proof kills the pile with a device our scout missed: **the
   least-non-normal-height tube**. Define *normal* points by boxes (empty, or meeting `A` in
   exactly the graph of a continuous selection); the bad set `B` (parameters with a
   non-normal height, ±∞ included) is definable; if `B` is infinite it contains an interval,
   where the three definable functions `β⁻ < β < β⁺` (least non-normal height + its fiber
   neighbors) turn continuous after a Monotonicity shrink — and then every `(x, β(x))` is
   provably normal (the tube between the fiber neighbors is A-free, so would-be pile
   invaders ARE the continuous selection), contradicting β's definition. `B` finite then
   gives locally-constant counts on the complement intervals and the bound. All order-only,
   single-structure, every ingredient banked. The pre-registered
   `UNIFORM_FINITENESS_WALL` falsifier is RETIRED — replaced by the classical route:
   - **D4a — the normality layer** ✅ LANDED 2026-07-05: `OMinimalNormal.lean` (53rd module),
     axiom-clean, 3 gate entries (`definable_normal`, `normal_slice_isOpen`,
     `tame_normalTop`), **GREEN ON THE FIRST BUILD** (504 lines). Box predicates
     (`IsEmptyBox`/`IsThinBox`/`IsSelContBox`, shapes tuned to their formulas so the eval
     lemmas close by one both-sides `simp`); `Normal`, `NormalTop/Bot`; definability via the
     deepest formula of the ladder (selection-continuity at ambient 12) over a
     python-generated 77-lemma snoc battery (`attribute [local simp]`) + general
     `compPair`/`compQuad` collapses + `Fml.reindex` templates; `eqAt` combinator;
     `normal_slice_isOpen` (graph boxes certify neighbors by carving empty boxes from the
     continuity clause), `notNormal_slice_isClosed`, `tame_notNormal_slice`.
   - **D4b — the β-machinery** ✅ LANDED 2026-07-06: `OMinimalBeta.lean` (54th module),
     axiom-clean, 3 gate entries (`definableFun_selFn`, `definableFun_betaFn`,
     `tame_badFin`), green on the first build (round 2 only stripped six redundant simp
     args). **The generic selector engine `selFn`**: any definable *functional* relation
     `R ⊆ ℝ²` totalizes (by 0) to a `DefinableFun` — D2's `nthFn` 4-piece pattern abstracted
     over the atom, proved ONCE, instantiated FIVE times (β, fiber max/min, β⁻/β⁺). `betaFn`
     exists on the bad set via `IsClosed.csInf_mem` + the new ray-normality bounds
     (`notNormal_bddBelow/Above`: bottom/top-ray normality pushes far heights into empty
     boxes). The neighbor relations `maxBelowSet`/`minAboveSet` feed **betaFn's own graph
     back into the formula layer as an atom** — definable functions compose. Bad sets
     `tame_badFin`/`tame_notNormalTop`/`tame_notNormalBot` all tame. D4c inputs complete.
   - **D4c — the tube kill** ✅ LANDED 2026-07-06: `OMinimalBadFinite.lean` (55th module),
     axiom-clean, 3 gate entries (`notNormalTop_finite`, `badFin_finite`,
     `abnormal_finite`), green round 4 (arity pins on restated `Fml.eval`s; `open Topology`
     for `𝓝`; C1 already owns the name `badSet_finite` — master renamed `abnormal_finite`;
     one `▸`→`rw`). **THE BAD SETS ARE FINITE.** Ray kills: fiber-max/min continuity +
     one constant caps a window (the offending A-point itself witnesses the spec — no
     domain bookkeeping). The tube proper: interval of bad parameters → dodge finitely many
     `¬NormalBot` (β exists) → `split_interval` (new: tame split with the disjunction
     packed inside the existential, deferring the 2×2×2 case tree) on the three tame flags
     (below-β/above-β/graph) → ONE avoidance shrink over the three cut-sets → constants
     `c < β < d` between the fiber neighbors → **the tube is A-free except the β-graph**
     (`htube`) → graph box (thin + β-continuity = the selection) or empty box →
     `(a*, β a*)` normal, contradicting leastness. `avoid_finset` de-privatized (the
     ladder's workhorse).
   - **D4d — count constancy + UF** ✅ LANDED 2026-07-06: `OMinimalUniformFiniteness.lean`
     (56th module), axiom-clean, 3 gate entries (`count_locally_constant`,
     `uniform_finiteness`, `exists_empty_countSet`), green round 3.
     **THE FINITENESS LEMMA IS COMPLETE — vdD Ch. 3 (1.7) machine-checked over the abstract
     `OMinStructure`**: `uniform_finiteness : ∃ N, ∀ x, |A_x|.ncard ≤ N` for definable `A`
     with all fibers finite; equivalently some counting set is empty.
     `count_locally_constant`: `≥` by a min-peeling strips induction (one fresh separator
     per step; the selection-continuity bracket traps each selection below its separator,
     the recursion floor keeps the rest above — sort-free); `≤` by ray boxes + Heine–Borel
     on the compact leftover `K = Icc ∖ strips` (ℝ-specific, fenced like C1d's countability)
     + thin-box injection into the fiber of `a`. Counting-set frontiers ⊆ the finite
     abnormal set. The chain kill: witnesses off the abnormal set, infinitely many share
     one abnormal-gap (pigeonhole on below-sets via `Finite.exists_infinite_fiber` —
     sort-free), `preconnected_split` (R2-N, its third life) propagates membership across
     the gap, and the earliest witness lands in every counting set — an infinite fiber,
     contradiction. The `countSet ⊆ abnormal` side-case dies in two lines: D2's dichotomy
     puts an interval inside a finite set.

   **R4-D COMPLETE (2026-07-04 → 2026-07-06): UNIFORM FINITENESS from the axioms in seven
   modules (D0 slices/fiber-frontier, D1 counting formulas, D2 dichotomy + rank functions,
   D3 k-curves, D4a normality, D4b β-machinery, D4c tube kill, D4d UF).**

### R4-D5 — CDT₂: cell decomposition for `ℝ²` from UF (spine registered 2026-07-06)

The classical order restored: UF in hand, CDT₂ is bookkeeping over the rank functions of
`fiberFrontier A`. Stages:

1. **D5a — the uniform frontier bound + exact-count partition** ✅ LANDED 2026-07-06:
   `OMinimalFrontierBound.lean` (57th module), axiom-clean, 3 gate entries
   (`frontier_ncard_bound`, `ranks_enumerate`, `frontier_partition`), **GREEN ON THE FIRST
   BUILD**. The arc closes on itself: `uniform_finiteness` applied to D0's `fiberFrontier`
   (definable + automatically finite fibers) gives `|∂(A_x)| ≤ N` for EVERY definable `A`,
   no hypothesis. Plus: `exists_isNth_of_mem` (every point of a finite fiber is the rank of
   its below-count — `Set.ncard_lt_ncard` on the strict below-subset), `ranks_enumerate`
   (membership ⟺ one of the first `ncard` rank values), the tame `exactSet` classes, and
   the capstone `frontier_partition` (over the `k`-class, the `k` rank functions of
   `fiberFrontier A` enumerate every fiber frontier). The countSet–ncard bridge
   de-privatized + hypothesis-localized in D4d.
2. **D5b — the master refinement** ✅ LANDED 2026-07-06: `OMinimalRefinement.lean`
   (58th module), axiom-clean, 3 gate entries (`tame_bandIn`, `tame_const_on`,
   `master_refinement`), **GREEN ON THE FIRST BUILD** (round 2 dropped one lint).
   `master_refinement`: ONE bound `N` + ONE finite cut set `C`; on every `C`-avoiding
   interval — a single exact-count class, all `N` rank functions of `fiberFrontier A`
   continuous, and every structural test constant: `graphIn`, `bandIn` between consecutive
   ranks, `rayLowIn`/`rayHighIn`, `allIn` — **both polarities via the complement trick**
   (`bandOut A = bandIn Aᶜ`, one tame lemma each, instantiated at `A` and
   `S.definable_compl hA`; the polarity quantifier `∀ B' ∈ {A, Aᶜ}` keeps the statement
   flat). Five parameterized tame lemmas (rank-graph atoms + `exists_eq_left'` collapse);
   `tame_const_on` = `preconnected_split` on frontier-free intervals (pure topology);
   `C` = one `Finset` bundling rank cut-sets + class frontiers + test frontiers, with
   the membership chains factored into seven inclusion lemmas.
3. **D5c — band triviality (the CDT₂ core)** ✅ LANDED 2026-07-06:
   `OMinimalBandTriviality.lean` (59th module), axiom-clean, 3 gate entries
   (`band_dichotomy_fiber`, `regions_cover`, `band_triviality`), **GREEN ON THE FIRST
   BUILD** (round 2 dropped two unused hypotheses — `regions_cover` turned out to be pure
   order combinatorics, no class membership needed). The falsifier never fired. Per-fiber
   dichotomies: `mem_rank_of_frontier` (every fiber-frontier point is a rank value, via
   `ranks_enumerate` + the class count) makes each region frontier-free, and
   `preconnected_split` (its 5th/6th/7th/8th lives: `Ioo`, `Iio`, `Ioi`, `univ`) forces
   all-in/all-out per fiber. `regions_cover`: `Nat.findGreatest` on the largest rank
   ≤ target — rays + graphs + bands exhaust the line. Capstone `band_triviality`: one
   `N`, one `C`; per `C`-avoiding interval — one class `k ≤ N`, ranks continuous +
   strictly ordered, and **every region uniformly in or out of `A`** (D5b constancy +
   the fiber dichotomies; the `Aᶜ`-polarity clauses turned out unneeded — dichotomy +
   A-side-out already forces the complement side). Cell decomposition for `ℝ²` in
   concrete form: `A ∩ (I×ℝ)` = the union of the selected graphs and bands.
4. **D5d — the packaged CDT₂** [owner go 2026-07-06: pack Cell₂; honest 2-session split]:
   - **D5d-1 — the packaging layer** ✅ LANDED 2026-07-06: `OMinimalCells.lean`
     (60th module), axiom-clean, 3 gate entries (`baseCells_covers`, `baseCells_pairwise`,
     `univ_uniform`). `Cell₁` (pt/ioo/iio/ioi/univ) and `Cell₂` (graph/band/bandLow/
     bandHigh/full over a base) with `toSet`, `WellFormed` (continuity + ordering on the
     base), `IsCellDecomp` (covers + pairwise disjoint + adapted + well-formed; cell
     definability inherited from construction, not in the predicate). **`baseCells`** — the
     line decomposition from a cut Finset, SORT-FREE: points + adjacent gaps (pairs from
     `C ×ˢ C` filtered by `adjPair` = no cut point between) + outer rays; proved covering
     (filter-max'/min' pick the enclosing gap) and pairwise disjoint (`gap_disjoint`: two
     overlapping adjacent gaps are equal — adjacency tested at each other's endpoints, no
     WLOG; a self-contained `pairwise_filterMap_of_forall` carrying source memberships
     replaced the missing library lemma). Ray/univ two-point uniformity upgrades banked.
     Gotchas: the `ite` inside a `Classical`-defined function must be restated under
     `open Classical in` or instance mismatch blocks `rw`; `Option.noConfusion` on
     `none = some c` hits universe metavariables — use `simp at`; `Set.mem_singleton_iff`
     needs the `toSet` match-def unfolded first (`simp only [Cell₁.toSet, …]`).
   - **D5d-2 — the assembly** ✅ LANDED 2026-07-06: `OMinimalCellDecomp.lean`
     (61st module), axiom-clean, 2 gate entries, **green on the second build** (a private
     lemma to de-privatize, one `-`-in-term-pattern). **THE ARC'S TERMINAL THEOREM:**
     `cell_decomposition : ∃ cells : List Cell₂, IsCellDecomp cells A` — every definable
     planar set admits a finite, covering, pairwise-disjoint, adapted, well-formed cell
     decomposition; `cell_decomposition_mem` = the union form (membership in `A` ⟺
     membership in an inside-cell). The generic per-base constructor
     `cells_over_uniform` (bandLow · graphs · bands · bandHigh, or one full column for the
     frontier-free class; covering = `regions_cover`, disjointness = strict rank ordering,
     adaptedness = the selections, well-formedness = rank continuity) instantiated over the
     `baseCells` partition: per-fiber dichotomies at point bases, `band_triviality`
     directly on the cut-avoiding gaps, the two-point uniformity upgrades (with the class
     matched across overlapping intervals via a shared interior point) on the rays and the
     `C = ∅` line; `flatMap` + cross-base disjointness (cells project to their bases).

**R4-D5 COMPLETE — CDT₂ MACHINE-CHECKED (2026-07-06, one day, five modules 57–61).**
**The whiteboard triple is done: abstract o-minimal structures (R4-A/B) + the Monotonicity
Theorem (R4-C) + Uniform Finiteness (R4-D) + Cell Decomposition for ℝ² (R4-D5), all
axiom-clean over the abstract `OMinStructure`, with `semilinearStructure` as the standing
nontrivial witness.** Remaining lane TODO: TS-QE (the long pole); natural extensions:
definable-cell clauses in `IsCellDecomp`, multi-set-adapted decompositions, dimension
theory from cells.

   TS-QE remains the long-pole TODO.

General-`n` cells remain outside this scope. Sizing: D0–D3 ≈ 3–4 sessions, bankable
independently; D4 is the variance.

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
