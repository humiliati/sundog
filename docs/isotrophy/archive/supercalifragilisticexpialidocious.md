## A. Preface V0.9
 legal disclaimer 
	Copyright © 2026 Stellar Aqua LLC. All rights reserved. Historical joke
	licensing language removed during the rights-posture cleanup.
 dont try this at home.
This is the paper-only deliverable — ready for review before any code or registration. We've tried to be honest about where we're confident vs. where the load-bearing detail still needs work (Section 7 enumerates the five open verification points). 

This is written with ink and quill don't go around erasing things.

## B. ~Table of Contents~

Line |_| title || sum ||| wtf+sum+title-_i__e iff (I)__o__a__ q (I)
0001 |A| law || no erase ||| pn (a)nd ppr
0009 |B| ToC || Table of Contents ||| we really doing this?
0027 |1| Setup & notation || Sundog Isotrophy Wire Maze Premise ||| 21 wires, symp to target, ez piano kill, questionable anti-symp order, and goal
0040 |2| Piano twist G_i || Fiber operator
0063 |3| M_i G_i τ-cancellation |







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

## 8. Codex review, 2026-05-20

**Verdict:** keep this as the right draft shape, but do not freeze
`K_facet_v0.3` or authorize monodromy code from it yet. The new direction is
much stronger than v0.2 because it is no longer the static equivariance-null,
and it correctly centers the anti-symplectic `F_beta` chain. But two items are
currently blockers, not ordinary polish.

### Blocker 1: the neutral quotient is degenerate as written

The draft defines

```text
N_i := span{ ydot_i(0), J*grad H(y_i(0)) }.
```

In canonical Hamiltonian coordinates, `ydot = X_H = J*grad H`. Those two named
vectors are the same vector, not the autonomous Hamiltonian two-plane. The
forced neutral removal therefore removes only one written direction while
claiming to remove the flow/energy block.

Before code, replace this with a precise neutral construction. Acceptable
routes:

1. Work on a fixed-energy Poincare section and remove only the flow direction
   by construction.
2. Keep the full fiber but name the second neutral as the generalized
   period/energy direction, e.g. a vector `u_E` satisfying the appropriate
   `(M_i - I)u_E = c_i X_H(y_i(0))` relation, not `J*grad H`.
3. If no clean generalized vector is available numerically, pre-register that
   rows with a non-semisimple unit multiplier block are manual-review rows,
   not silently counted.

This is the same family of bug as the old "right quantity, wrong gate" issue:
the formula names the thing we want, but the linear object currently written is
not that thing.

### Blocker 2: `G_i` is not yet a proven single-fiber endomorphism

The draft correctly identifies verification point 1 as load-bearing, but the
status should be stronger: section 3's commutation chain is provisional until
the transport map is written with source and target fibers.

As written, `Phi_{T/2,i} = Dphi_{T_i/2}(y_i(0))` maps
`T_{y_i(0)}` to `T_{y_i(T_i/2)}`. It is not an endomorphism of
`T_{y_i(0)}` by itself. Likewise,

```text
M_i * Phi_{T/2,i} = Phi_{T/2,i} * M_i
```

only makes sense after identifying `T_{y_i(T_i/2)}` back to `T_{y_i(0)}`.
The autonomous-flow identity is really a square of maps between different
fibers:

```text
Dphi_T(y_i(T/2)) * Dphi_{T/2}(y_i(0))
  = Dphi_{T/2}(y_i(0)) * Dphi_T(y_i(0)).
```

Those are not the same statement unless the half-period transport has already
been fixed. The review standard should be: write the operator as a typed
composition through

```text
V_0 = T_{y_i(0)}Xi
V_h = T_{y_i(T_i/2)}Xi
V_partner,h = T_{alpha_I*y_i(0)}Xi
```

then compose the maps back to `V_0`. Only after that should we assert
`[M_i, G_i]=0`.

One more reason to be strict: if all factors are treated as same-fiber matrices,
then `rho(F_beta) * rho(Rpi) * rho(tau)` algebraically collapses back toward
the bare `(12)` action. That is either harmless bookkeeping or a sign that the
transport is doing all the work. The derivation must say which.

### Blocker 3: `G_i^2 = I` must be proved after the cocycle is chosen

Loop-level `alpha_I^2 = e` is necessary, but it does not automatically prove
that the chosen single-fiber representative squares to identity. A nontrivial
transport cocycle can leave holonomy behind. After the typed transport in
Blocker 2 is written, explicitly compute `G_i^2` on `K_i^{fib}`. If it equals
`I`, the `K_i^+ + K_i^-` split is legitimate. If it equals a structural
operator such as `F_beta` or a phase-shift remnant, the eigenspace split changes.

### Blocker 4: structural subtraction should be a quotient, not an informal complement

The intent is right: subtract only the structural continuation directions. The
current notation

```text
K_i^{PT} := K_i^+ symplectic-complement B_i
```

is too loose. Until containment is proven, use

```text
B_i^+ := B_i cap K_i^+.
```

Then specify either:

```text
K_i^{PT} := K_i^+ / B_i^+
```

if this is a branch-count quotient, or the symplectic reduction

```text
K_i^{PT} := (B_i^+)^omega / B_i^+
```

if `B_i^+` is isotropic/coisotropic in the needed way. The phrase
"symplectic-orthogonal complement" is not enough because anti-symplectic
fixed sets are commonly Lagrangian-flavored; complements are not canonical
unless the reduction structure is pinned.

### Blocker 5: `1/2 * dim` is a candidate multiplicity, not yet a branch count

The factor-of-two instinct is good, but the statement "one branch per
symplectic pair" needs the crossing form / Lyapunov-Schmidt nondegeneracy
condition. A `+1` multiplier pair gives a candidate direction. It becomes a
counted branch only if the parameter derivative crosses the symmetry-breaking
equation transversely and the Jordan/Krein structure is clean.

Pre-registration-safe version:

```text
d_i_candidate = 1/2 * dim_R K_i^{PT}
```

Then define the branch-validity gate before any run:

```text
d_i = d_i_candidate only for semisimple +1 blocks with nonzero crossing form;
otherwise row i is a manual-review / refined-rule row.
```

### Notation cleanup before canonical docs

Use a typed spacetime-action notation for `F_beta`. The sentence
`F_beta * y_i(s) = y_i(s) for all s` is easy to misread as a pointwise
phase-space fixed condition. Safer:

```text
A_F y_i(-s) = y_i(s)
```

where `A_F` is the phase-space linear part of `F_beta`. That keeps the time
reversal explicit and prevents a future implementer from dropping a minus sign.

### Recommended next artifact

Do not code yet. Write a one-page "typed transport lemma" with only maps and
fibers:

1. Define `A_F`, `A_tau`, `A_Rpi`, `P12`, and `X(t)=Dphi_t(y_i(0))`.
2. State the reversible identities with domains and codomains.
3. Construct the alpha-induced map on `V_0` through the partner fiber.
4. Prove or refute `G_i^2=I`.
5. Only then revisit `[M_i,G_i]=0`, `B_i^+`, and the multiplicity rule.

If that lemma closes, the draft becomes a real v0.3 registration candidate.
If it does not, the failure is valuable: it means the induced-representation
functional still has hidden holonomy and should not be flattened into a
single-fiber `d_i`.

---

## 9. Author response / disposition, 2026-05-20

All five review blockers are accepted. The v0.3 draft remains useful as a map,
but it is not a frozen formula and not code-authorizing.

### Accepted corrections

1. **Neutral quotient.** The old
   `span{ydot_i(0), J*grad H(y_i(0))}` is wrong because these name the same
   Hamiltonian vector. The corrected neutral block is the standard
   two-dimensional generalized unit-multiplier block:

   ```text
   N_i := span{ X_H(y_i(0)), u_E }
   (M_i - I) u_E = c_i X_H(y_i(0)),  c_i != 0
   ```

   Route 2 is preferred for the paper: make `u_E` explicit. If `u_E` cannot be
   computed cleanly on a row, that row becomes a pre-registered manual-review
   row rather than a silently counted branch.

2. **Typed transport.** The section 3 commutation chain is provisional because
   it treated `Phi_{T/2,i}`, `rho(tau)`, and `M_i` as same-fiber endomorphisms.
   The next derivation must keep source and target fibers explicit. A later
   collapse to a single-fiber operator is allowed only after the transport
   cocycle is written and checked.

3. **`G_i^2`.** Loop-level `alpha_I^2 = e` is not enough. The chosen fiber
   representative must be squared directly. If it leaves holonomy behind
   (`F_beta`, a phase-shift remnant, or another structural operator), the
   eigenspace split is not the simple `+/-` split.

4. **Structural subtraction.** Replace the informal complement with a typed
   quotient/reduction. First define

   ```text
   B_i^+ := B_i cap K_i^+.
   ```

   Then choose, after the typed transport is known, either the branch-count
   quotient

   ```text
   K_i^{PT} := K_i^+ / B_i^+
   ```

   or the symplectic reduction

   ```text
   K_i^{PT} := ((B_i^+)^omega) / B_i^+.
   ```

5. **Multiplicity.** The safe pre-registration language is candidate-first:

   ```text
   d_i_candidate = (1/2) * dim_R K_i^{PT}
   d_i = d_i_candidate only when:
     (i)  the +1 block of M_i is semisimple beyond N_i,
     (ii) the Lyapunov-Schmidt crossing form on K_i^{PT} is nondegenerate,
     (iii) the symplectic structure on K_i^{PT} is nondegenerate.
   ```

   Otherwise row `i` is a manual-review / refined-rule row.

### Notation lock

Use the typed spacetime-action notation for the structural symmetry:

```text
A_F y_i(-s) = y_i(s)
```

where `A_F` is the phase-space linear part of `F_beta`. This keeps time
reversal explicit and prevents the sign-drop bug that would come from writing
`F_beta*y_i(s)=y_i(s)` pointwise.

### Next deliverable

The next artifact is exactly one paper-level lemma, not a runner:

1. Define `A_F`, `A_tau`, `A_Rpi`, `P12`, and `Phi_t := Dphi_t(y_i(0))`, with
   `V_0`, `V_h`, `V'_0`, and `V'_h` explicit.
2. State the reversible identities with domains and codomains.
3. Construct the alpha-induced map on `V_0` through the partner fiber without
   collapsing intermediate fibers.
4. Prove or refute `G_i^2 = I`.
5. Only then revisit `[M_i,G_i]`, `B_i^+`, and the multiplicity rule.

If the typed lemma reveals holonomy that cannot be absorbed into a canonical
single-fiber operator, v0.3 should stop there. That would mean the
induced-representation functional has hidden cocycle dependence and must not be
flattened into a single-fiber `d_i`.

---

## 10. Typed transport lemma draft / pair-orbit proposal, 2026-05-20

New proposal received after the author response:

- The typed transport attempt finds no canonical single-fiber alpha operator
  `G_i` on `V_0`.
- The old claims `[M_i,G_i]=0` and `G_i^2=I` are therefore artifacts of a
  hidden cocycle choice.
- The replacement object is the pair-orbit alpha-fixed kernel on the parent
  choreography and its bare-`(12)` partner.
- In that pair picture, alpha-fixed pairs are proposed to be parameterized by
  the parent periodic kernel `K_i^{fib}` directly.
- The proposed v0.3 count becomes an `A_F`-even, `sigma3`-nontrivial sector of
  `K_i^{fib}`, after quotienting the structural
  `A_F`-even / `sigma3`-trivial continuation sector.

Proposed formula from the draft, **not locked**:

```text
N_i        := span{ X_H(p_i), u_E },  (M_i - I)u_E = c_i X_H(p_i), c_i != 0
K_i^{fib}  := ker(M_i - I) / N_i
B_i^+      := (A_F-even) cap (sigma3-trivial) cap K_i^{fib}
K_i^{PT}   := ((A_F-even) cap K_i^{fib}) / B_i^+
d_i^{cand} := (1/2) * dim_R K_i^{PT}
```

The draft explicitly asks for review before registration. Good instinct. The
single-fiber `G_i` obstruction is real enough to take seriously, but the
replacement pair-orbit formula is not yet safe.

## 11. Codex review of typed transport lemma, 2026-05-20

**Verdict:** this is a productive simplification, but not lockable. It likely
found a real failure in the single-fiber `G_i` program. It has not yet proved
the replacement count. The next version should be a pair-orbit / dihedral
representation lemma, not monodromy code.

### Blocker A: F_beta is not pointwise fixed along the whole curve

The draft states:

```text
A_F y_i(-t) = y_i(t)
therefore A_F y_i(t) = y_i(t) for all t.
```

The conclusion does not follow. Replacing `t` by `-t` gives

```text
A_F y_i(t) = y_i(-t).
```

So `A_F` maps the tangent fiber at `y_i(t)` to the tangent fiber at
`y_i(-t)`. It is a same-fiber endomorphism only at symmetry phases such as
`t=0` and `t=T/2` (modulo the period), not at every parent point. The base
identity

```text
M_i A_F^{V_0} = A_F^{V_0} M_i^{-1}
```

can still be valid, but the draft must delete the "F_beta fixes every point"
sentence. Otherwise a future implementation will silently use `A_F` at the
wrong fiber.

### Blocker B: the pair-orbit construction still has a base-phase issue

The draft correctly notices that alpha sends a parent tangent loop to the
partner orbit at the half-period shifted partner fiber `V_h'`, not `V_0'`.
But section 6 then writes the pair space as if it were simply
`T_{C_i}Lambda + T_{Y_i}Lambda`.

That is not enough for based loops. Either:

1. define the second component as the shifted partner `Y_i^h(t)=Y_i(t+T/2)`;
2. work on unbased/free loops modulo phase; or
3. add an explicit phase transport from `V_h'` to `V_0'`.

Without one of those choices, the pair-orbit picture has merely moved the
cocycle from `G_i` into the definition of the partner loop space. This is
fixable, but it must be explicit.

### Blocker C: the square-root ambiguity is overstated in the wrong place

The `(C2)` calculation writes the half-step as `M_i^{-1/2}` and then flags a
matrix-square-root / Maslov branch ambiguity. The actual orbit gives a typed
map `Dphi_{-T/2}` directly. There is no numerical need to choose a square root
of `M_i` if the half-period flow is integrated as a flow map.

There may still be holonomy or Maslov-index content if one tries to represent
the half-step purely as a matrix function of `M_i`, but the paper should not
make that the primary ambiguity. The primary ambiguity is the choice of
identification between based fibers, not a free-floating square-root branch.

### Blocker D: alpha-fixed pair parameterization needs a proof after quotienting

For a clean involution swapping two identical vector spaces, fixed pairs are
graphs and are parameterized by one component. That part is plausible. Here the
two components are phase-shifted loop fibers plus Hamiltonian neutral
quotients. The draft needs to prove the graph parameterization descends through
`N_i = span{X_H,u_E}` and respects the chosen based/free-loop convention.

Until then, "alpha-fixed is automatic" is a strong candidate, not a theorem.

### Blocker E: `A_F` and `sigma3` form a dihedral representation, not two
commuting projectors

This is probably the biggest replacement-formula issue. `F_beta` contains a
transposition, so it should conjugate the 3-cycle to its inverse:

```text
A_F sigma3 A_F^{-1} = sigma3^{-1}
```

up to the time/spatial factors. That means `A_F` generally normalizes the
`Z3` action rather than commuting with it. The trivial `sigma3` sector is
stable, but the nontrivial `sigma3` sector is a real two-dimensional irrep
where `A_F` acts as a reflection. So the phrase

```text
(A_F-even) cap (sigma3-nontrivial)
```

needs representation-theoretic precision. The right object is likely the
nontrivial real isotypic component of the dihedral group generated by
`sigma3` and `F_beta`, with `A_F` selecting a reflection-even line inside each
real `Z3` block. Do not implement this as two commuting projector masks unless
the commutation relation is proved.

### Blocker F: the `1/2 * dim` candidate lost its symplectic proof

The previous `1/2 * dim` argument depended on a symplectic `K_i^{PT}`. But
`A_F` is anti-symplectic, and its fixed subspace is typically Lagrangian-like,
not symplectic. After switching to `A_F`-even sectors, the old symplectic-pair
argument no longer transfers automatically.

So the safe draft language is:

```text
d_i_candidate = dim_R K_i^{PT}              # or 1/2 dim, pending derivation
```

with no factor frozen until the dihedral/fixed-sector Lyapunov-Schmidt
reduction says which dimension counts branches.

### Revised next artifact

Write a new lemma with this target:

1. Replace "F_beta fixes every point" with the typed identity
   `A_F: V_t -> V_{-t}` and record the fixed phases separately.
2. Choose the loop convention: shifted partner, free-loop quotient, or explicit
   phase transport. No hidden based-loop identification.
3. Prove the alpha-fixed pair graph descends through the neutral quotient.
4. Derive the representation of `<sigma3,F_beta>` on `K_i^{fib}` as a real
   dihedral representation.
5. Define the branch candidate sector from that representation.
6. Only then decide whether multiplicity is `dim`, `1/2 dim`, or a
   crossing-form-gated refinement.

If that closes, v0.3 survives in a cleaner pair-orbit form. If it does not,
the typed lemma has still done real work: it killed the noncanonical
single-fiber `G_i` without pretending the replacement was free.

---

## 12. Pair-orbit / dihedral draft 2, received 2026-05-20

Draft 2 narrows the v0.3 target to a representation problem:

```text
N_i        := span{ X_H(p_i), u_E },  (M_i - I)u_E = c_i X_H(p_i), c_i != 0
K_i^{fib}  := ker(M_i - I) / N_i
K_i^{fib} ~= a_i*T + b_i*S + c_i*E       # real D3 irreps
d_i^{cand} := c_i                         # multiplicity of standard 2D irrep E
```

The draft chooses an `F_beta`-fixed anchor, avoids `M^{1/2}` language by using
typed half-flows, and recasts `<sigma3,F_beta>` as a real `D3` representation.
It also proposes three gates:

```text
G1: anchored operators satisfy F_beta sigma3 F_beta^{-1} = sigma3^{-1}
G2: each E-block has the right nondegenerate symplectic / L-S crossing form
G3: the neutral block N_i lies entirely in the trivial isotypic component
```

This is the best candidate shape so far, but it still needs review before
registration.

## 13. Codex review of pair-orbit / dihedral draft 2, 2026-05-20

**Verdict:** the `D3` representation turn is the right mathematical target.
The standard-irrep count `c_i` is a clean candidate, and it is a better object
than either v0.2's static containment count or the noncanonical single-fiber
`G_i`. But the draft still overclaims the alpha-fixed graph step. Do not lock
or code.

### Passes / improvements

1. **F_beta anchor discipline improved.** Choosing an actual `F_beta` fixed
   phase is the right repair for the earlier pointwise-fixedness mistake.
2. **No matrix square root.** Replacing `M^{-1/2}` with typed half-flow is the
   right correction. Keep that.
3. **Dihedral structure is the right basis.** `F_beta` should conjugate
   `sigma3` to `sigma3^{-1}` on the anchored representation, so `D3` irreps
   are the natural language. This is the first version that names the right
   representation-theoretic object.
4. **Multiplicity as `c_i` is plausible.** In a real standard `D3` block, the
   reflection-fixed line is one-dimensional. Counting standard blocks is a
   defensible candidate, pending the crossing-form gate.

### Blocker 1: alpha still does not define a based pair endomorphism

The draft anchors pair loops at

```text
delta gamma_C(0) in V_0
delta gamma_Y(0) in V_0'
```

but `alpha_I` sends the parent component at `t=0` to the shifted partner fiber
`V_h'`, and sends the partner component to the shifted parent fiber `V_h`. The
line

```text
v_C = P12 * Phi_{T/2}^Y * v_Y
```

therefore equates a vector in `V_0` with a vector in `V_h` unless a phase
transport, shifted-loop convention, or free-loop quotient has already been
chosen. This is the same base-phase issue as before, now inside the pair
picture.

Fix options:

1. Define the pair as `(C_i, Y_i^h)` with the partner already half-shifted.
2. Work on free loops modulo phase, then handle the flow neutral carefully.
3. Add explicit phase transport from `V_h` to `V_0` and from `V_h'` to `V_0'`.

Until one is chosen, the conclusion

```text
alpha-fixed pair-kernel = ker(M_i - I) / N_i
```

is not proved.

### Blocker 2: the `P12 Phi^Y P12 Phi^C = M_i` calculation is mistyped

The draft uses the equivariance relation for `Phi_{T/2}^Y`, but the typed
composition must keep domains:

```text
Phi_{T/2}^Y : V_0' -> V_h'
P12         : V_h' -> V_h
```

After applying `P12 Phi_{T/2}^Y P12 Phi_{T/2}^C`, the intermediate fiber is
not automatically `V_0`. The algebra may reduce to `M_i` after choosing a
based/free-loop convention, but the draft performs that reduction before the
convention is specified.

### Blocker 3: the `F_beta` fixed anchor must be certified per row

The catalog ansatz strongly suggests the stored initial condition is an
`F_beta` fixed epoch, and the previous `F_beta` receipt supports the structural
claim. Still, the v0.3 registration should state the anchor rule:

```text
anchor p_i^F is accepted only when F_beta residual is closure-tight at phase 0
or after an explicitly recorded phase shift to the nearest F_beta fixed epoch
```

If a row needs a phase shift, all `sigma3` and half-flow operators must be
built from the shifted anchor. No silent use of the supplementary-A epoch.

### Blocker 4: the partner anchor is fixed by a conjugate symmetry, not
necessarily the same `F_beta`

The partner `Y_i=(12)C_i` generally carries the conjugated reversing symmetry

```text
P12 F_beta P12^{-1}
```

not automatically the same `F_beta` operator in the parent frame. Draft 2 says
`P12 p_i^F` is the partner's analog `F_beta` anchor by equivariance; that is
probably right, but the operator should be named as the conjugate symmetry.
This matters when deriving the pair alpha action and the `D3` relation.

### Blocker 5: `N_i` must be decomposed as a `D3` subrepresentation

Superseded by v0.3f: `X_H` is not trivial. At an `F_beta` fixed anchor,
`A_F X_H = -X_H`, so the flow direction lives in the sign irrep `S`. The
generalized energy vector `u_E` should be normalized as the trivial line `T`.
The registration still needs either a normalization rule for `u_E` that makes
this canonical, or a proof that the quotient is independent of the remaining
kernel ambiguity.

Gate G3 should therefore be:

```text
G3: prove / normalize N_C = T*u_E + S*X_H as a D3 subrepresentation with no E leakage;
    otherwise row is manual-review
```

### Blocker 6: `d_i=c_i` is a candidate, not a branch count

Counting standard real irreps is clean as a representation candidate. It is
not yet a bifurcation count. The crossing-form / Lyapunov-Schmidt gate must
decide whether each standard block yields one branch, two sign-related
branches, or a ghost direction.

Safe language:

```text
d_i_candidate := c_i
d_i := d_i_candidate only after G1, G2, G3 and the crossing-form gate pass
```

### Scratchpad conjecture quarantine

The brainstorm note `internal/quarantine/scratchpad_brainstorm_notes.md`
contains the "couplings surface / Kenny Wheeler / magnesium / 1/phi^3"
conjecture. Its rigorous version is: inspect the Lyapunov-Schmidt crossing form
on the standard `D3` isotypic sector and see whether its spectrum has a
recognizable coupling scale. This is not a premise of v0.3 and not part of
`d_i`. It can become an exploratory diagnostic only after the branch candidate
sector is defined.

### Revised next gate

Write the v0.3e lemma with four hard choices made explicitly:

1. Choose the loop convention: shifted partner, free-loop quotient, or explicit
   phase transport.
2. Certify the `F_beta` anchor and partner conjugate anchor.
3. Build the anchored `D3` representation on `K_i^{fib}` and prove G1.
4. Define a `D3`-equivariant neutral quotient and prove G3.
5. State `d_i_candidate=c_i`, with G2/crossing-form as the branch-validity
   gate.

Then, and only then, the variational runner becomes meaningful.

---

## 14. v0.3f neutral-block refinement, 2026-05-20

New finding: the neutral block is `D3`-equivariant, but it is not wholly
trivial-isotypic. It splits canonically as one trivial line plus one sign line:

```text
N_C = T*u_E^C  +  S*X_H(p_i^F)
```

Reason:

- `X_H(p_i^F)` is `sigma3`-invariant, but `F_beta` reverses the Hamiltonian
  flow at the `F_beta` fixed anchor, so `A_F X_H = -X_H`. Therefore `X_H`
  lives in the sign irrep `S`, not the trivial irrep `T`.
- The canonical energy/Jordan partner `u_E^C` can be chosen
  `sigma3`-invariant and `F_beta`-even, so it lives in the trivial irrep `T`.
  This choice must still be written as a normalization rule, because Jordan
  partners are only defined modulo kernel vectors.

Pre-quotient:

```text
K_i^{fib*} ~= a_i*T + b_i*S + c_i*E
```

Post-neutral quotient:

```text
K_i^{fib} ~= (a_i-1)*T + (b_i-1)*S + c_i*E
```

So the standard-irrep count `c_i` is invariant under the corrected neutral
quotient. That is the load-bearing payoff: the previous gate G3 should no
longer say "prove `N_C` is trivial"; it should say:

```text
G3: prove / normalize the neutral quotient as T*u_E + S*X_H, and verify no
    neutral direction leaks into E.
```

Consequences:

- Structural sector after quotient is `(a_i-1)*T`, not `a_i*T`.
- The sign sector `(b_i-1)*S` is a separate `F_beta`-odd, `sigma3`-trivial
  deformation sector. It is not structural-T continuation and not a piano-trio
  candidate under the ansatz convention.
- `d_i_candidate = c_i` survives this correction.

Still pending:

1. **Loop convention / half-flow typing.** Keep based loops at the
   `F_beta`-fix anchor and write
   `Phi_T^C = Phi_{T/2}^{y_i(T/2)} o Phi_{T/2}^{y_i(0)}` with both halves
   typed. Do not write `(Phi_{T/2})^2` without naming the transport.
2. **Anchor certification.** Prove `(12)p_i^F` is fixed by the partner
   conjugate anchor. Because `P12` commutes with `A_tau` and spatially uniform
   `A_Rpi`, `[P12,A_F]=0`, so `A_F(P12 p_i^F)=P12 p_i^F`; still record the
   discrete anchor choice in the receipt.
3. **G1.** Prove the anchored `D3` relation
   `F_beta sigma3 F_beta^{-1}=sigma3^{-1}` at the operator level.
4. **G2.** Crossing-form / Lyapunov-Schmidt gate remains the branch-validity
   gate. The scratchpad coupling conjecture stays quarantined as a possible
   diagnostic after this sector exists.

Updated next gate: v0.3f should address the five points in order:

1. declare the based-loop convention at `p_i^F`;
2. type the half-flow composition explicitly;
3. certify the parent and partner anchors;
4. carry the neutral decomposition `N_C=T*u_E+S*X_H` through the `D3` quotient;
5. reserve G2 crossing form as a per-row branch-validity check.

## 15. v0.3g crossing-form / non-degeneracy gate, 2026-05-20

The v0.3g draft is the right next paper gate: it moves the surviving candidate
`d_i_candidate=c_i` through the branch-validity question instead of treating
standard-irrep multiplicity as a branch count.

The useful structural claim is:

```text
Delta H = partial_epsilon H |_{epsilon=0}
Delta H = Delta H_T + Delta H_E
Delta H has no S component because it is F_beta-even.
```

The consequence is also the right shape: the `D3`-symmetric `T` part should
drive structural continuation, while the `E` component of the perturbation is
the only part that can drive piano-trio branch creation from the standard
sector.

Do not lock the scalar per-block `gamma_i^(k)` form yet. The crossing form has
to be defined on the neutral-quotiented, anchored `D3` standard sector first.
If `c_i > 1`, the load-bearing object is likely a crossing matrix on the
standard-irrep multiplicity space, not independent hand-picked scalar diagonal
entries:

```text
Gamma_i : R^{c_i} -> R^{c_i}          # crossing form on the E-multiplicity space
d_i     := rank_floor(Gamma_i)        # candidate locked only if the paper gate proves this rule
```

The scalar recipe `gamma_i^(k)` is a special case after one of two facts is
proved: either `c_i=1`, or the crossing form has been diagonalized by a
canonical `D3`-compatible basis and the off-diagonal E-copy mixing is shown not
to change the branch count. Until then, the per-block receipt fields should be
treated as a proposed readout, not as the definition of the gate.

Additional blockers to record before code:

1. **Reduced Hamiltonian convention.** The formula for `Delta H` must be checked
   in the actual symplectic-reduced coordinates. If the center-of-mass or
   momentum reduction depends on the mass parameter, `partial_epsilon H` can
   pick up terms not visible in the unreduced shorthand.
2. **Phase-space `F_beta` invariance.** The no-`S` claim must be verified as a
   tangent / Hamiltonian-vector-field statement, not just as a configuration
   function statement. Momentum convention and time-reversal signs matter.
3. **Correct Floquet crossing form.** The expression should be the quotient
   Floquet crossing form, likely involving the standard Hamiltonian pairing
   with the induced monodromy variation, not an untyped
   `omega(xi, partial_epsilon M xi)` written before neutral representatives are
   chosen.
4. **Schur collapse scope.** The `T` part vanishing on a Lagrangian
   `F_beta`-fixed line is plausible, but it must be proved for the exact
   quotient crossing form. It should not be imported from the informal scalar
   expression.
5. **E-copy mixing.** If `c_i>1`, the crossing form may have off-diagonal
   terms between standard-irrep copies. The branch count should be a
   matrix/rank statement unless a diagonal scalar rule is proved.
6. **Anchor independence.** The two `F_beta` anchors should give the same
   `Gamma_i` up to conjugacy, or the receipt must carry the anchor convention.
7. **Gamma floor.** The floor must be empirical and closure-relative, derived
   from reproducibility / step refinement or a symmetry-equivalent anchor check,
   not inferred from `rtol`/`atol` alone.

The scratchpad `1/phi^3` coupling conjecture remains quarantined. It can become
a diagnostic only if the crossing matrix has near-zero or degenerate directions.

Updated next gate: write v0.3g as a paper-only crossing-form definition on the
`E` multiplicity space. Close the reduced-Hamiltonian, no-S, quotient-crossing,
Schur-collapse, E-copy-mixing, anchor-independence, and floor gates before any
monodromy code.

## 16. v0.3h rank-matrix / G2.6 prep, 2026-05-20

The matrix formulation is cleaner than the scalar one and should become the
next paper deliverable. Instead of treating `gamma_i^(k)` as independent
per-block scalars, define the crossing form on the `c_i`-dimensional
`F_beta`-even standard sector:

```text
Gamma_i^(k,k') := omega(xi_k, (partial_epsilon M_i) xi_k')
d_i            := rank_floor(Gamma_i)
rank_floor     := #{ singular values of Gamma_i > k_gamma * gamma_floor }
```

This recovers the scalar rule only when `c_i=1`, or when a structural
orthogonality / diagonalization result proves the matrix is block-diagonal in
the chosen `E`-copy basis.

New sub-gate:

```text
G2.6: symplectic block-orthogonality of the E-isotypic sector.
```

The v0.3g `T`-Schur collapse only handled diagonal entries:
`omega(xi_k, xi_k)=0`. Off-diagonal entries
`omega(xi_k, xi_k')` for `k != k'` can be nonzero unless the `E` copies are
symplectically orthogonal. If G2.6 holds, the `T` component contributes zero to
all of `Gamma_i`, and the scalar/block-diagonal formula becomes a justified
operating special case. If G2.6 fails, `Gamma_i` has real off-diagonal
structure and the full SVD/rank gate is mandatory.

Consolidated gates after v0.3h:

1. **G2.1:** operator-level `F_beta` invariance of `D(J grad Delta H)`.
2. **G2.2:** real Schur for the `D3` standard rep, `End_D3(E)=R`.
3. **G2.3:** E-copy mixing is absorbed into `Gamma_i`, not post-processed.
4. **G2.4:** anchor independence of `Gamma_i`, or an explicit anchor convention.
5. **G2.5:** closure-relative `gamma_floor`.
6. **G2.6:** E-isotypic symplectic block-orthogonality.

Receipt schema extension:

```text
Gamma_i_matrix
Gamma_i_singular_values
Gamma_i_rank_floor
gamma_floor
symplectic_block_orthogonal_E
d_i = Gamma_i_rank_floor
```

The `symplectic_block_orthogonal_E` flag comes from the E-isotypic
`omega`-Gram matrix. If true across all 21 rows, the scalar rule is empirically
the operating regime. If false for any row, that row uses the full rank-matrix
gate.

The scratchpad coupling conjecture remains quarantined and is eligible only as
a degeneracy diagnostic if `Gamma_i` has near-zero singular directions. No
monodromy code is authorized until v0.3h is written as a paper lemma with G2.6,
anchor independence, and the floor specified.

## 17. v0.3h draft review: rank survives, G2.6 reframed, 2026-05-20

The v0.3h draft's rank-matrix object survives:

```text
Gamma_i^(k,k') := omega(xi_k, (partial_epsilon M_i) xi_k')
d_i            := rank_floor(Gamma_i)
```

But the proposed proof of G2.6 does not close as written. It diagonalizes
`M_i` on the `E`-isotypic part of `K_i^{fib}`. That cannot be the canonical
split, because:

```text
K_i^{fib} = ker(M_i - I) / N_i
```

so `M_i` acts as the identity on the space being counted. There are no distinct
Floquet multipliers inside `K_i^{fib}` to choose an `M_i+sigma3` basis. If a
receipt records `M_i_E_floquet_multipliers`, that field must refer to the
ambient E-isotypic variational spectrum, not to the kernel/multiplicity space
used for `Gamma_i`.

The better proof is simpler and stronger. For the `T` component:

1. `(partial_epsilon M_i)_T` is `D3`-equivariant, so it commutes with
   `F_beta`.
2. Therefore it maps `Fix(F_beta)` to itself.
3. `F_beta` is anti-symplectic, so `Fix(F_beta)` is isotropic:
   `omega(u,v)=0` for every pair of `F_beta`-fixed vectors.

Thus:

```text
Gamma_i^T(k,k') = omega(xi_k, (partial_epsilon M_i)_T xi_k') = 0
```

for all `k,k'`, including off-diagonals, without any E-block
symplectic-orthogonality assumption. This makes G2.1 the load-bearing proof:
we must verify operator-level `F_beta` preservation for the `T` component.

G2.6 should be reframed. It is not needed to make `Gamma_i^T` vanish. It is a
basis-conditioning and scalar-interpretation diagnostic: if a chosen basis has
orthogonal E blocks, the diagonal/scalar readouts are easier to explain; if it
does not, the full SVD rank still gives the basis-invariant count.

Anchor independence also needs one word fixed. A change of `F_beta` anchor
transports the bilinear form by congruence, not necessarily similarity:

```text
Gamma_i -> S^T Gamma_i S
```

Rank is invariant under invertible congruence, so `d_i` remains anchor
independent once the transport is typed.

G2.5 is still not closed. The proposed `gamma_floor_i` is a good shape, but
`k_gamma=3` and `k_int=10` need a pre-registered sentinel calibration receipt
before any run. That calibration can be the first empirical step only after its
negative/branch conditions are written.

Updated v0.3h-final target:

1. Prove operator-level `F_beta` preservation of `(partial_epsilon M_i)_T`.
2. Use anti-symplectic fixed-space isotropy to prove `Gamma_i^T=0`.
3. Define `Gamma_i` as a basis-invariant matrix/rank object on the
   `F_beta`-even standard multiplicity space.
4. Treat E-block orthogonality as diagnostic/conditioning, not as the lock.
5. Correct anchor-independence to rank under congruence.
6. Pre-register the `gamma_floor` sentinel calibration before any code.

## 18. v0.3h G2.6d disposition, 2026-05-20

The corrected operational stance is:

```text
G2.6d: do not canonicalize for the count.
```

`Gamma_i` is the binding object and its rank is invariant under basis changes
on the `F_beta`-even standard multiplicity space. Therefore:

```text
d_i = rank_floor(Gamma_i)
```

is the count. The diagonal entries `Gamma_i^(k,k)` are basis-dependent scalar
diagnostics. They can be recorded for debugging and human reading, but only with
the basis convention in the manifest, and they do not enter `K_facet_v0.3`.

The optional paper follow-up is:

```text
G2.6b: Phi_{T/2}^C as an involutive canonicalizer on K_i^{fib}.
```

Because `(Phi_{T/2}^C)^2=M_i=I` on the kernel, the half-period map could give a
meaningful `+/-` split tied to `alpha_I`. This is the natural research follow-up
for a physically meaningful per-block diagnostic, but it is not required for
the rank count.

Locked / nearly locked landscape:

- `Gamma_i^T=0` entry-wise by anti-symplectic `Fix(F_beta)` isotropy, once
  operator-level `F_beta` preservation of `(partial_epsilon M_i)_T` is written.
- `Gamma_i=Gamma_i^E` as the `c_i x c_i` bilinear form on the
  `F_beta`-even standard multiplicity space.
- `d_i=rank_floor(Gamma_i)` as the basis-invariant count.
- G2.4 anchor independence by rank under congruence.
- G2.5 formula shape survives; constants `k_gamma` and `k_int` still require a
  pre-registered sentinel calibration receipt before any run.

Pending:

- G2.1/G2.2 citation and operator-level write-up.
- Replace the incorrect "Krein in `M_i`" flag with degeneracy / bimodality of
  `(partial_epsilon M_i)_E` or `Gamma_i` singular values.
- Write the sentinel calibration receipt with negative/branch conditions before
  running it.

## 19. v0.3i sentinel calibration pre-registration, 2026-05-20

The remaining concrete deliverable is a one-row calibration pre-registration.
It is the smallest empirical move compatible with the rank-matrix posture.

First, separate the two degeneracy flags that v0.3h had conflated:

```text
dE_perturbation_spectral_degeneracy_E
  Degenerate spectrum of (partial_epsilon M_i)_E on the E-isotypic sector.
  This is a basis-choice / scalar-diagnostic flag for G2.6b. It is not a count
  ambiguity by itself.

gamma_singular_bimodality_clean
  Clean or marginal singular-value split for Gamma_i. This is the rank-gate
  flag and therefore count-relevant.
```

Sentinel choice:

```text
primary: O_62
backup:  O_264, O_468, O_574, ... in canonical-strict period order
```

Rationale: `O_62` is shortest-period among the strict rows and was
bi-orientationally closure-tight in G.2, so it should minimize accumulation of
variational integration error. If `O_62` returns `c_i=0`, step through the
pre-registered ladder until the first `c_i>0` row. If the whole ladder returns
`c_i=0`, disposition as structural negative: `K_facet_v0.3=0` is forced by the
catalog, not by calibration failure.

Constants, fixed before any sentinel:

```text
k_gamma = 3
k_int   = 10
```

The sentinel verifies separator behavior; it does not tune either constant.

Bimodality falsifier:

```text
above:    sigma_j > k_gamma * gamma_floor
below:    sigma_j < gamma_floor
marginal: gamma_floor <= sigma_j <= k_gamma * gamma_floor
```

If the marginal band is nonempty, v0.3i fails and halts. The successor work is a
better floor/error analysis, not row-wise retuning.

Authorized by v0.3i only after this receipt is accepted:

```text
one sentinel variational integration
rtol = atol = 1e-12
n_samples = 1009
phase_grid = 73
receipt dir = results/isotrophy/k-facet-v03-sentinel-calibration-O62/
```

Receipt schema:

```text
Gamma_i_matrix
Gamma_i_singular_values
gamma_floor_i
gamma_singular_to_floor_i
d_i
dE_perturbation_spectral_degeneracy_E
gamma_singular_bimodality_clean
diagonal_gamma_i_k        # diagnostic only
branch_status_i_k         # diagnostic only
basis_convention
gate_version = "v0.3i"
```

v0.3i does not authorize more rows, supplementary-B, or freezing
`K_facet_v0.3`. A pass allows a separate v0.3j full-21 authorization. A fail is
structured: marginal singular values mean floor analysis, while all-ladder
`c_i=0` means a structural negative.

## 20. v0.3i-runner spec review, 2026-05-20

The sentinel runner spec is accepted as a paper-only pre-implementation
contract. It should implement one command:

```text
npm run isotrophy:kfacet:sentinel --indices 62 ... --out results/isotrophy/k-facet-v03-sentinel-calibration-O62
```

Scope is still exactly one sentinel row. Backup ladder is allowed only if the
current sentinel returns `c_i=0`. Auto-refinement A is allowed only for a
marginal singular-value outcome. No full-21 run, no supplementary-B comparison,
and no `K_facet_v0.3` freeze.

The ansatz anchor lemma is useful and should go into the runner notes:
`p_i^F=y_i(0)` is in `Fix(A_F)` in closed form. The runner does not search for
F_beta fixed points; the second fixed epoch is `y_i(T/2)` by reversibility.

Runner shape:

1. parse row and anchor at `p_i^F`;
2. integrate orbit and `M_i`;
3. optionally integrate the bare `(12)` partner as a sanity receipt for the
   F_beta cocycle;
4. integrate `partial_epsilon M_i`;
5. compute `ker(M_i-I)`, quotient `N_C=T*u_E+S*X_H`, and form `K_i^{fib}`;
6. build typed `rho(sigma3)` and closed-form `rho(F_beta)`;
7. project `D3`, compute `c_i`, and build the `F_beta`-even standard basis;
8. compute `Gamma_i`, SVD, `gamma_floor_i`, rank, and both degeneracy flags;
9. write the protected receipt.

Pre-implementation gates:

- **Typed D3 products.** The projectors need `rho(F_beta sigma3^k)` with
  explicit ordering. Once `rho(sigma3)` and `rho(F_beta)` are typed
  endomorphisms on the anchor fiber, matrix products are fine, but the runner
  must verify the `D3` relation instead of silently constructing private paths.
- **Reduced symplectic form.** Verify the coordinate reduction preserves
  canonical `omega`; otherwise store the reduced `J` used for `Gamma_i`.
- **Basis convention.** The count is rank-invariant, but scalar diagnostics
  need a deterministic basis convention in the receipt.
- **SVD/eigen vocabulary.** `gamma_singular_bimodality_clean` uses singular
  values of `Gamma_i`; `dE_perturbation_spectral_degeneracy_E` must explicitly
  state whether it uses eigenvalues, Schur blocks, or singular values of
  `(partial_epsilon M_i)_E`.
- **Refinement A precision.** If `1e-14` is below practical integrator
  precision, record a refinement failure rather than interpreting noise as
  structure.

This review clears the runner spec for implementation planning, not execution.
