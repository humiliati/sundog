# A machine-checked o-minimality program in Lean 4, with Tarski–Seidenberg

**Write-up of the SUNDOG_V_OMIN lane** — factual account, 2026-07-08.
Repository: `Dev\sundogcert` (public), Lean 4.30.0 + mathlib. Companion roadmap with
per-stage receipts: `docs/SUNDOG_V_OMIN.md`.

---

## 1. What was built

A machine-checked development of one-variable o-minimality over ℝ, an abstract
o-minimal-structure interface with the three classical Chapter-3 theorems proved once
over the interface, two concrete structures witnessing it — semilinear and
semialgebraic — and, as the engine of the second witness, a complete set-level proof
of **Tarski–Seidenberg quantifier elimination** by the Cohen–Hörmander sign-diagram
method. A Pillay–Steinhorn-style corollary carries the results back to the
approximation-theory lane that motivated the program.

Headline theorems (all `#print axioms`-gated to `[propext, Classical.choice,
Quot.sound]`, enforced in CI by `Sundogcert/AxiomAudit.lean`):

| Theorem | Content |
|---|---|
| `monotonicity_theorem_continuous` | The Monotonicity Theorem (with continuity), over any `OMinStructure` |
| `uniform_finiteness` | The Finiteness Lemma (van den Dries (1.7)), over any `OMinStructure` |
| `cell_decomposition` | Cell decomposition for ℝ², over any `OMinStructure` |
| `semilinearStructure` | The first witness: ordered ℝ-vector-space reduct (Fourier–Motzkin projection) |
| `elim_signVector` | **Tarski–Seidenberg, elimination form**: `{g \| ∃ y, signVec F g y = σ}` is semialgebraic |
| `semialgebraicStructure` | The second witness: the real field (projection = `elim_signVector`) |
| `semialgebraic_cell_decomposition` etc. | The Chapter-3 theorems instantiated at the real field |
| `definable_family_nip` | Pillay–Steinhorn, bounded form: definable planar families shatter ≤ 2·cells+1 points |

## 2. What is and is not claimed

- Everything is **set-level over concrete ℝ**: sets of `Fin n → ℝ`, no first-order
  syntax, no abstract real-closed fields, no reflective decision procedure.
- The o-minimality results are the **dimension ≤ 2 core** of van den Dries Chapter 3
  (Monotonicity, Uniform Finiteness, CDT for ℝ²), not the full inductive tower for
  all dimensions.
- The NIP result is the **one-parameter planar case** with an explicit bound, not the
  general Pillay–Steinhorn theorem.
- Prior art (checked 2026-07-02 and re-checked 2026-07-06): no Lean development of
  o-minimality or Tarski–Seidenberg existed; the comparable formalizations are
  Cohen–Mahboubi (Coq, algebraic route via subresultants + BKR), a complete real QE
  in Isabelle/HOL (2022), and Harrison's Cohen–Hörmander in HOL Light. This appears
  to be the first machine-checked Tarski–Seidenberg and the first machine-checked
  o-minimal structures in Lean.

## 3. Architecture

```
R1  dim-1 o-minimality: Tame s := (frontier s).Finite; finite-frontier calculus
R2  monotonicity PL instance; frontier modulus; tame_iff_normalForm
R3  semilinear: constructive booleans + Fourier–Motzkin projection 2→1
R4-A  OMinStructure (8 axioms: booleans, substitution, snoc-projection,
      lt/singleton atoms, tame_dim_one) + definable calculus + S₁ = Tame
R4-B  semilinearStructure : OMinStructure          (first witness)
R4-C  THE MONOTONICITY THEOREM (plain + continuous)
R4-D  UNIFORM FINITENESS (the classical tube argument)
R4-D5 CELL DECOMPOSITION for ℝ²
R4-E  Pillay–Steinhorn NIP/VC bridge (this write-up's last entry)
TS    Tarski–Seidenberg (17 modules, below) → semialgebraicStructure (second witness)
```

The abstract layer was built *first* so that the hard theorem (TS) would buy the
Chapter-3 results at the real field by instantiation. That bet paid exactly as
designed: TS-3 is ~300 lines, and the real-field Monotonicity/UF/CDT₂ are one-line
corollaries.

## 4. The Tarski–Seidenberg arc (TS-0 … TS-3, 2026-07-06 → 07-07)

**Route (TS-0).** Cohen–Hörmander sign diagrams, set-level, over concrete ℝ.
Rejected: a Cohen–Mahboubi port (the abstract-RCF algebra stack — subresultants, BKR —
has no mathlib support), CAD (strictly heavier), and the model-theoretic route
(mathlib's `ModelTheory` had Presburger only, and the target is set-level).
Working over ℝ buys the univariate core by real analysis — `finite_setOf_isRoot`,
continuity + IVT, derivative monotonicity — instead of algebra.

**The ladder** (modules 62–78, one `AxiomAudit` gate set per module):

- **TS-1** univariate sign partition: one root-cut Finset gives constant strict signs
  per interval (IVT core).
- **TS-2a** parametric branch trees: `spec g P` (coefficient specialization),
  `resolve` (strip vanishing leads, well-founded on support), `truncChain`,
  degree-exactness on live leads; the Dershowitz–Manna measure (mathlib's in-tree
  well-foundedness).
- **TS-2b** pseudo-division from scratch over any `CommRing` (mathlib had none), an
  even-exponent trick that eliminates Hörmander's sign-twist bookkeeping, and
  `sign_transfer`: at roots of the divisor, the dividend's sign is its remainder's.
- **TS-2c** derivative zones: strict monotonicity between derivative roots, the
  six-shape between-roots diagram, end behavior with the `(−1)^deg` parity twist.
- **TS-2d-1** sign diagrams: `signVec`, `ColsFrom` (Option-barrier column recursion),
  `Realizes`, and `DiagramPartition` — finitely many semialgebraic branches each
  carrying ONE diagram valid branch-wide. Branch-wide validity (vs. per-point) is
  load-bearing: the final union argument fails without it.
- **TS-2d-2** the reconstruction surgery: padding removal (`dropPadding`), the
  prefix-projection foothold (the transfer reaches every sample of the projected
  diagram), and the per-gap/per-ray graft toolkit (keyed zone lemmas — outcomes
  *selected* by flank signs, never refuted post hoc).
- **TS-2d-3** the walk and the descent: `graftWalk` — the `P :: F` diagram as a
  **pure function** of (annotation, old columns); the reads (`readPtSign`,
  `readAnnot`, `baseDrop`) making the annotation a pure function of column data; the
  fused master (`colsFrom_master`) with the pending-gap accumulator;
  `realizes_readAnnot` — the Cohen–Hörmander reconstruction step, machine-checked.
- **TS-2 closure** — the selection transport (`realizes_selectFam`: membership
  suffices, killing all permutation machinery), the vanishing-lead branch layer
  (resolve-fibers are semialgebraic in the parameters), and the Dershowitz–Manna
  assembly: `diagramPartition_all`, then `elim_signVector`.
- **TS-3** the structure: sign-vector characterization of semialgebraic sets, the
  snoc→cons rotation into the `finSuccEquiv` frame, projection from
  `elim_signVector`; `tame_dim_one` from TS-1's cut partition.

The pre-registered falsifier `QE_COMBINATORICS_WALL` (the sign-matrix bookkeeping
fails to stay tractable at single-owner scale) was carried from TS-0 and **retired**
at TS-2's closure.

## 5. Design findings worth keeping

1. **The pending-gap accumulator.** The naive fused-induction statement for the
   reconstruction is *false*: the annotation cannot be restated at intermediate
   dropped barriers, because a grafted root may lie left of a dropped sample
   (concrete counterexample in the receipts). The repair — carry the last kept
   barrier, its flank tag, and the pending projected gap column; discharge plan
   validity only at keep time — made the ~460-line master land essentially first-try.
2. **Keyed zone lemmas.** Stating interval analyses with the outcome selected by
   endpoint signs (rather than as trichotomies to be refuted) removed all continuity
   arguments from the assembly layer.
3. **Membership-selection transport.** Duplicate family members carry equal sign
   entries, so re-reading columns through first-occurrence indices transports
   realized diagrams to any family whose members occur in the original — one lemma
   replaces member-dropping, reordering, and permutation machinery.
4. **Purity as the uniformity mechanism.** Branch-wide diagram validity is obtained
   by making every construction a pure function of diagram-level data
   (`graftWalk ∘ readAnnot ∘ baseDrop`), with the point `g` confined to validity
   side conditions. No canonicity or uniqueness theorems were needed.
5. **Fuel-structured vs. keep-later recursions.** Well-founded definitions with list
   patterns don't yield usable unfold equations; fuel-structured auxiliaries fix
   this. Better still, choosing the *keep-later* merge orientation made the walk
   functions plainly structural — no fuel at all — and turned the master's drop case
   into pure plumbing.
6. **Probe before writing.** A scratch `#check` file for uncertain mathlib names
   before each module saved roughly a fix round per module in the late arc.

## 6. Statistics

- **Modules 34–79** of `sundogcert` (46 ladder modules; the TS arc is 17 of them),
  every headline theorem `#guard_msgs`-gated; final audit **8624 jobs GREEN**
  (expected; 8623 before R4-E).
- Axioms throughout: `[propext, Classical.choice, Quot.sound]` — no `sorry`, no
  custom axioms, no `native_decide`.
- Timeline: R1–R3 landed 2026-07-02 (one day); R4-A…D5 by 2026-07-06 (the
  "whiteboard triple", five days); TS-QE opened 2026-07-06 and closed 2026-07-07;
  R4-E 2026-07-08.
- Reproduction: `lake build Sundogcert.AxiomAudit` at the repo root.

## 7. The bridge back (R4-E) and open directions

`definable_family_nip`: in any `OMinStructure`, a definable planar set — read as a
family `x ↦ {y | (x,y) ∈ A}` of subsets of the line — cannot shatter more than
`2·(cells) + 1` points. Instantiated both ways: `semialgebraic_family_nip`
(polynomial families) and `semilinear_family_nip` (piecewise-linear, i.e.
ReLU-class families). This is the tameness→VC-finiteness direction that connects to
the approximation lane's interest in learning-theoretic bounds for tame function
classes (cf. the "deep learning as tame objects" literature, arXiv 2509.18025).

Natural extensions, none started: the dimension tower above 2 (CDT for ℝⁿ),
definable-cell clauses in `IsCellDecomp`, dimension theory from cells, a
`Polynomial.Chebyshev`-style API extraction of the pseudo-division and sign-diagram
layers for mathlib, and a paper-length write-up (ITP/CPP shape) of the TS arc.

---

*Status note: `sundogcert` commits/pushes are owner-side; at the time of writing the
public repo's history ends before TS-1, and modules 62–79 exist locally, all green.*
