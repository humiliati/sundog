## A. Preface V0.9
 legal disclaimer 
	MIT licence or Property of Stellar Aqua unless Stellar Aqua is being sued for improper toilet installation in which case it's property of Sundog Research Lab whichever makes the most sense at the time.
 dont try this at home.
This is the paper-only deliverable — ready for review before any code or registration. We've tried to be honest about where we're confident vs. where the load-bearing detail still needs work (Section 7 enumerates the five open verification points). 

This is written with ink and quill don't go around erasing things.

## B. ~Table of Contents~

Line |_| title || sum ||| wtf+sum+title-_i__e iff (I)__o__a__ q (I)
0001 |A| law || no erase ||| pn (a)nd ppr
0009 |B| ToC || Table of Contents ||| we really doing this?
0027 |1| Setup & notation || Sundog Isotrophy Wire Maze Premise ||| 21 wires, symp to target, ez piano kill, questionable anti-symp order, and goal
0040 |2| Piano Trio twist








---


## 1. Setup & notation

Let `C_i` be one of the 21 strict single-curve choreographies at `m_3 = 1`, with period `T_i` and base point `y_i(0) ∈ Ξ`, where `Ξ ≅ ℝ^{12}` is phase space after symplectic reduction by translation and total linear momentum (these reductions are clean: they preserve all spatial and discrete isotropy). Let `M_i := Dφ_{T_i}(y_i(0))` be the symplectic monodromy.

Isotropy: `Iso(C_i) ⊇ ⟨σ_3, F_β⟩` with
- `σ_3 = ((123), T_i/3, +, I)` symplectic, order 3 → `[M_i, ρ(σ_3)] = 0`;
- `F_β = ((12), τ, R_π)` **anti-symplectic** (because τ is anti-symplectic), order 2 → by the standard reversible-systems result, `M_i · ρ(F_β) = ρ(F_β) · M_i^{-1}`. **This is the fact most easily mis-imported from the symplectic case as "[M_i, ρ(F_β)] = 0"; that import is wrong and would corrupt everything downstream.**

Piano-trio target: `α_I = ((12), T_i/2, +, I)`, **NOT** in `Iso(C_i)` (G.2 + v0.3a established this for all 21).

Forced symmetry breaking: `m_3 → 1 + ε` breaks `S_3 → ⟨(12)⟩`. All other Hamiltonian symmetries (`O(3)`, `τ`, `S¹`, energy) survive.

Goal: count `α_I`-fixed periodic orbits bifurcating from `C_i` as `ε` turns on, modulo structural-continuation directions (deformations of `C_i` *within* its F_β-equivariant family — those continue the parent, they don't branch a piano-trio).

## 2. The piano-trio twist `G_i` as a fiber operator

Because `α_I ∉ Iso(C_i)`, `α_I` does not naively act on `T_{y_i(0)}`. We construct the induced fiber operator via the F_β-mediated identification of the partner orbit.

**Loop-space action.** Bifurcation analysis lives on the based loop space `Λ_i = {γ : [0,T_i] → Ξ, γ(0) = γ(T_i)}`. `α_I` acts by `(α_I · γ)(t) := ρ((12)) · γ(t + T_i/2)` (time mod `T_i`).

**Fiber reduction.** Modulo forced Hamiltonian neutrals (§4), the loop-space bifurcation kernel is canonically isomorphic to the reduced fiber kernel

```
K_i^{fib} := ker(M_i − I) / N_i,    N_i := span{ ẏ_i(0), J·∇H(y_i(0)) }
```

via the extension `v ↦ δγ_v(t) := Dφ_t(y_i(0)) · v` (periodic ⟺ `M_i v = v`).

**Induced `G_i`.** For `v ∈ K_i^{fib}`, the loop `α_I · δγ_v` has new base point `α_I · y_i(0) = ρ((12)) · y_i(T_i/2)`. Using F_β-isotropy of `C_i` (`F_β · y_i(s) = y_i(s)` for *all* s, hence `ρ((12)) · y_i(s) = ρ(R_π·τ) · y_i(s)`), this equals `ρ(R_π·τ) · y_i(T_i/2)` — a point of the partner orbit `Y_i := (12)·C_i = (R_π·τ)·C_i` at parameter `T_i/2`. Identifying `T_{Y_i(T_i/2)}` back to `T_{y_i(0)}` via (i) half-period back-flow along `Y_i`, then (ii) `ρ(R_π·τ)^{-1} = ρ(τ·R_π)` (since `Y_i(0) = (R_π·τ)·y_i(0)`), and using F_β's anti-symplectic intertwining to relate back-flow along `Y_i` to forward-flow along `C_i`, the composite operator on `T_{y_i(0)}` is:

```
G_i := ρ(F_β) · ρ(R_π) · ρ(τ) · Φ_{T/2,i},     Φ_{T/2,i} := Dφ_{T_i/2}(y_i(0)).
```

(`Φ_{T/2,i}` is interpreted as a fiber endomorphism via the F_β-mediated transport above. **Verification point 1, §7.**)

## 3. `[M_i, G_i] = 0` — the τ-cancellation

Using:
- (a) `M_i · ρ(F_β) = ρ(F_β) · M_i^{-1}` (F_β anti-symplectic isotropy);
- (b) `M_i · ρ(R_π) = ρ(R_π) · M_i` (R_π proper symplectic Hamiltonian symmetry);
- (c) `M_i · ρ(τ) = ρ(τ) · M_i^{-1}` (τ reversibility);
- (d) `M_i · Φ_{T/2,i} = Φ_{T/2,i} · M_i` (autonomous flow, modulo the F_β-mediated identification of §2);

```
M_i · G_i  = M_i · ρ(F_β) · ρ(R_π) · ρ(τ) · Φ_{T/2,i}
           = ρ(F_β) · M_i^{-1} · ρ(R_π) · ρ(τ) · Φ_{T/2,i}     [by (a) — F_β twists M ↔ M⁻¹]
           = ρ(F_β) · ρ(R_π) · M_i^{-1} · ρ(τ) · Φ_{T/2,i}     [by (b)]
           = ρ(F_β) · ρ(R_π) · ρ(τ) · M_i · Φ_{T/2,i}          [by (c) — τ twists M⁻¹ ↔ M]
           = ρ(F_β) · ρ(R_π) · ρ(τ) · Φ_{T/2,i} · M_i          [by (d)]
           = G_i · M_i.
```

The two anti-symplectic intertwinings — F_β's `M ↔ M^{-1}` and τ's `M ↔ M^{-1}` — **cancel exactly**, giving a clean commutation. This is the load-bearing structural fact and it has a precise reason (each anti-symplectic factor contributes one `M ↔ M^{-1}` twist; F_β has one and ρ((12)) decomposes through one more τ, total two, even number → cancellation). Standard symplectic equivariant-bifurcation machinery applies; no detour into reversible-bifurcation theory required.

**`G_i^2 = I` on `K_i^{fib}`.** Since `α_I^2 = (e, 0, +, I)` on the loop space (modulo `T_i`-periodicity), the induced fiber `G_i` is involutive. So `K_i^{fib} = K_i^{+} ⊕ K_i^{-}` cleanly.

**Symplecticity of `G_i`.** Parities of the factors: F_β anti, R_π sym, τ anti, Φ sym. Product parity: anti·sym·anti·sym = sym (even number of anti-symplectic factors). **`G_i` is a symplectic involution.** Its `+1` eigenspace `K_i^{+}` is therefore a symplectic subspace of `K_i^{fib}` (symplectic involutions split the space into symplectic ± blocks). This matters in §5.

## 4. Bifurcation kernel and structural removal

Piano-trio branches correspond to `α_I`-fixed kernel modes: `K_i^{+}`. The F_β-structural continuation of `C_i` (deformations within the parent's F_β-symmetric, σ_3-trivial family under the perturbation) also lives in `K_i^{+}` and must be subtracted — those are continuations of the parent, not piano-trio branches.

**Structural sector.** 

```
B_i := { v ∈ K_i^{fib} : ρ(F_β) v = +v  AND  ρ(σ_3) v = +v }
     = (F_β-even) ∩ (σ_3-trivial isotypic component).
```

Both projectors commute appropriately with `M_i` on `K_i^{fib}` (σ_3 cleanly; F_β preserves the ±-grading because `F_β² = I`), so `B_i` is well-defined. **Subtract only the intersection** — F_β-even alone is too coarse (kills genuine σ_3-nontrivial branches); σ_3-trivial alone is too coarse (kills genuine F_β-odd branches). The intersection is the structural-continuation projection precisely.

**Genuine piano-trio sector.**

```
K_i^{PT} := K_i^{+} ⊖ B_i      (symplectic-orthogonal complement of B_i inside K_i^{+}, w.r.t. ω)
```

Symplectic-orthogonal, not Euclidean-orthogonal, because `B_i` is generically Lagrangian-flavored (F_β-even eigenspace of an anti-symplectic involution is Lagrangian — standard). The symplectic-orthogonal complement of `B_i ∩ K_i^{+}` inside the symplectic subspace `K_i^{+}` is well-defined and inherits symplectic structure from `K_i^{+}`. **Verification point 2, §7** (the precise definition when `B_i ∩ K_i^{+}` is not coisotropic in `K_i^{+}`).

## 5. Multiplicity-parity rule

A symmetry-breaking pitchfork in a Hamiltonian system bifurcates through **symplectic-paired** kernel directions: a `+1` Floquet eigenvalue of `M_i` participating in a bifurcation comes with its symplectic-conjugate partner, also at `+1`, paired by the symplectic form `ω`. The pair represents **one** branching event but contributes **2** to `dim_ℝ`.

Since `K_i^{PT}` is symplectic (inherits from `K_i^{+}`), `dim_ℝ K_i^{PT}` is even, and:

```
d_i = (1/2) · dim_ℝ K_i^{PT}.
```

**Equivariant L-S justification.** The reduced bifurcation equation `B^{red}(a, ε) = 0` on `K_i^{PT}` is symplectic+`α_I`-equivariant. `α_I` acts as `+I` on `K_i^{PT}` (by definition: `K_i^{+}` is the `α_I`-fixed eigenspace of `G_i`), so the equivariance group reduces to the *symplectic* `O(2)`-rotation in each 2D symplectic-paired direction. `O(2)`-equivariant bifurcations from a center give **one** branch per symplectic pair → factor of `1/2`.

**Edge cases (verification point 3, §7):**
- If `K_i^{PT}` has a Krein-collision structure where the symplectic pair is *not* a clean `O(2)` rotation block (e.g., a nilpotent +1 Jordan block of size 4 instead of two size-2 blocks), the count needs the Jordan-form refinement.
- If `dim_ℝ K_i^{PT}` is odd for some `i` despite the symplectic argument (would indicate the symplectic structure on `K_i^{PT}` is degenerate — should not happen generically, but check), the formula must be refined or the row flagged for manual review.

## 6. The v0.3 d_i formula (provisional, pending §7 closures)

```
G_i := ρ(F_β) · ρ(R_π) · ρ(τ) · Φ_{T/2,i}                              [piano-trio twist, uniform F_β-mediated]
K_i^{fib} := ker(M_i − I) / span{ ẏ_i(0), J·∇H(y_i(0)) }               [forced neutrals quotiented]
K_i^{+}  := { v ∈ K_i^{fib} : G_i v = +v }                             [α_I-fixed sector]
B_i      := { v ∈ K_i^{fib} : ρ(F_β) v = +v  AND  ρ(σ_3) v = +v }     [F_β-structural continuation]
K_i^{PT} := K_i^{+}  ⊖  B_i                                            [genuine piano-trio sector, symplectic ⊖]
d_i      := (1/2) · dim_ℝ K_i^{PT}                                     [symplectic-paired pitchfork count]
K_facet_v0.3 := Σ_{i=1}^{21} d_i.                                    [supercalifragilisticexpialidocious 22]
```

**Inputs are computable per row from `M_i`** plus the closed-form linear actions of `ρ(F_β), ρ(σ_3), ρ(R_π), ρ(τ), Φ_{T/2,i}`. All 21 rows are uniformly induced-case under the F_β cocycle, so the formula is **one expression**, not a 21-row case analysis.

## 7. Open verification points (must close before code)

These are the loose ends my derivation sketches but does not fully nail. Paper-level work, each.

1. **F_β-mediated identification of `Φ_{T/2,i}` as a fiber endomorphism.** I asserted the back-flow along `Y_i = (12)·C_i`, conjugated by `ρ(R_π·τ)`, yields exactly `Φ_{T/2,i}` interpreted on `T_{y_i(0)}`. The justification uses F_β anti-symplecticity + the commutation chain in §3, but the explicit identification — particularly how `ρ(τ)`'s anti-symplecticity interacts with the back-flow direction — should be written out Lie-algebraically. **A subtle wrong factor here (e.g., `M_i^{1/2}` vs `M_i^{-1/2}`) would silently corrupt `K_i^{+}`.** This is the single most important verification to nail.

2. **`K_i^{PT} = K_i^{+} ⊖ B_i`, precise definition.** Symplectic-orthogonal complement is well-defined when `B_i ∩ K_i^{+}` is *coisotropic in `K_i^{+}`*. If it's only isotropic, the symplectic-orthogonal is well-defined but lives in a different subspace; if mixed, the formulation needs Lagrangian-reduction care. Verify the coisotropic property holds, or refine the formula.

3. **Multiplicity-parity edge cases.** Generic case is the `1/2 · dim_ℝ` rule. Krein collisions and degenerate symplectic structure on `K_i^{PT}` need refinement. Detectable empirically once `M_i` is computed (look for repeated `+1` eigenvalues, Jordan blocks of order >1). Specify the refined rule for each pathology *before* running, not after.

4. **"+1" eigenvalue tolerance.** Identify `+1` Floquet multipliers of `M_i` using closure-relative criterion (`|λ−1| ≤ k · eigenvalue_floor`, with `k` and `eigenvalue_floor` derived from the integrator + Procrustes precision in the same spirit as G.2's `to_closure` discipline). **Do not use absolute thresholds** — that recurring pathology has bitten the workbench in three different forms; pre-empt it here.

5. **`B_i` projector concreteness.** Compute `ρ(F_β)|_{T_{y_i(0)}}` (an explicit 12×12 anti-symplectic involution, derivable from `(12) · τ · R_π` acting on phase space) and `ρ(σ_3)|_{T_{y_i(0)}}` (12×12 symplectic of order 3). Their `+1`-and-trivial intersection is a standard linear projection. **Sanity: verify `dim B_i ≥ 2`** — if `B_i` is empty, the structural-continuation subtraction is vacuous, which would itself be informative (means `C_i` has no F_β-σ_3-symmetric continuation direction beyond trivial neutrals — likely indicates I've miscounted neutrals or B_i).

---

If (1)–(5) close cleanly, `K_facet_v0.3` is locked. The runner becomes mechanical: variational integrate each of the 21 to get `M_i` (~minutes), apply the linear-algebra recipe of §6, sum to `K_facet_v0.3`, freeze, *then* compare against supplementary-B's `K_emp`. No more theory after that — every load-bearing question is in §7.

**The single highest-priority verification is (1)** — without the precise back-flow identification, `G_i` as a fiber operator is the wrong object and everything else cascades. I'd close that first on paper before touching (2)–(5).

---

Want me to take a first pass at (1) — write out the Lie-algebraic verification of the F_β-mediated identification of `Φ_{T/2,i}` explicitly — as the next paper-only deliverable, since it's the single load-bearing closure that gates the rest?