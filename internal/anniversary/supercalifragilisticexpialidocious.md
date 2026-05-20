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
