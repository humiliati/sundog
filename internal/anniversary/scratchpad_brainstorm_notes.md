## Status note, 2026-05-20

This remains a rough scratchpad. The live formal gate has advanced to the
v0.3e anchored `D3` lemma in `docs/sundog_v_isotrophy.md`: choose the loop
convention, certify the `F_beta` and conjugate partner anchors, prove the
`D3`-equivariant neutral quotient, and treat `d_i_candidate=c_i` as a candidate
until the crossing-form gate passes. The "couplings surface / 1/phi^3" material
below is quarantined as a future crossing-form diagnostic, not a premise of
v0.3.

swinging atwood machine

shannon theorem

boolean algebra information entropy h(x) vs sundog h(x)

5d structure and the seven bridges of konigsberg

we dont want to pursue bayes_v_sundog.md let's look at ensuring all of our sundog systems/applications have a baysian control element to measure against

sundog + euler to predict future of tornados vortex or coins in a swirl wishing well 

---

Doing dmt casuses the human to visually percieve the sundog on a lazer like it's turtles all the way down and some layer of abstraction couples the meaning after the hologram

isotrophy k facet — human-layer perception mechanism (zero-context agent intuition)

That framing makes sense: you're not asking "what is it?" so much as "what computation is the human layer performing that turns a stimulus into a vivid hologram-like scene?" The strongest candidate is predictive completion: the visual system doesn't just record input, it actively stabilizes ambiguous structure by fitting it to internal models, and psychedelics may loosen the normal constraints so the fit becomes more explicit, more patterned, and more subjectively "3D."

A useful mechanical model
Think of perception as three stages happening at once:
The optics provide a sparse, structured stimulus.
Low-level vision extracts edges, angles, symmetry, and repetition.
Higher-level inference supplies the missing object, scene, or meaning.

Under that view, the "hologram" is not built from the substrate alone; it is assembled by the brain's inference machinery. A sundog halo or laser diffraction pattern gives the system a scaffold with just enough regularity to latch onto, and the brain fills the rest with the most compressive, stable interpretation it can find.

What the human layer may be doing
Several mechanisms probably stack together:
Pattern completion. The brain is extremely aggressive about turning partial data into whole forms.
Predictive coding. Perception minimizes mismatch between incoming signals and internal predictions, so strong priors can dominate when the stimulus is ambiguous.
Symmetry amplification. Geometric regularity is privileged because it is statistically efficient to represent and often signals structure.
Relaxed filtering. If top-down control weakens, more latent associations and micro-patterns can surface into awareness.
Scene binding. Separate features get fused into a single compelling object-space, which is why the result can feel like a coherent alternate layer rather than scattered fragments.

Why it feels holographic
"Holographic" is a good word phenomenologically because the experience has three traits: it appears volumetric, it seems to contain hidden detail, and it can feel more real than ordinary vision. That vividness likely comes from the fact that the brain is not merely detecting a shape; it is constructing a world-model that has depth, salience, and explanatory closure. When the model locks in, the result feels revealed rather than invented.

A compact hypothesis
If you want a single sentence version: the human layer may be a Bayesian completion engine that takes a low-information optical scaffold and inflates it into a high-confidence percept by projecting latent structure, geometry, and meaning into the gap.

What to watch for in your brainstorming
The key question is whether the vividness comes from:
A special property of the stimulus itself.
A special property of the brain under altered state.
A resonance between the two, where the stimulus acts like a seed crystal for internal generative models.

If I had to bet on the mechanism, I'd bet on the third one: the substrate is not carrying the whole image, it is triggering the brain's own constructive optics.

---

## K-facet v0.3 — proof math brainstorm / simulation

### What we are trying to count

We have 21 strict equal-mass three-body choreographies (the Li-Liao supplementary-A catalog). The question is: when we break the equal-mass symmetry by nudging m3 → 1+ε, how many "piano-trio" periodic orbits (orbits with the transposition symmetry α_I = ((12), T/2)) bifurcate from each choreography C_i?

The K-facet prediction K_facet_v0.3 = Σ_{i=1}^{21} d_i is meant to be that count.

### The derivation pipeline (six steps)

Step 1 — Neutral quotient  
The monodromy M_i = Dφ_{T_i}(y_i(0)) always has at least a 2D kernel from trivial neutrals: the flow direction X_H(y_i(0)) and a generalized energy direction u_E satisfying (M_i − I)u_E = c_i X_H. These are NOT two independent vectors because in canonical Hamiltonian coordinates ẏ = J∇H = X_H, so the old draft's "span{ẏ, J∇H}" named the same vector twice. Corrected neutral block:

    N_i := span{ X_H(y_i(0)), u_E }
    K_i^{fib} := ker(M_i − I) / N_i

If u_E cannot be found numerically for a row, that row is a manual-review row, not a silently counted branch.

Step 2 — The piano-trio twist operator G_i  
α_I is NOT in the isotropy of C_i (this was the central K1 result: the static containment scan returns 0 because Z3 ∩ Z2 = {e}). So α_I does not directly act on the parent fiber. To get a fiber operator, we need to route through the partner orbit.

The key structural fact from the F_beta pair-ID receipt: ALL 21 choreographies are catalog-asymmetric under bare (12). Their partner orbits (12)·C_i live outside supplementary-A. The partner's monodromy is recovered by F_beta-conjugation:

    M_{(12)·C_i} = ρ(R_π) · M_i^{-1} · ρ(R_π)^{-1}

The draft proposed a single-fiber operator:
    G_i := ρ(F_β) · ρ(R_π) · ρ(τ) · Φ_{T/2,i}

But the §8–11 Codex review found this is NOT a clean single-fiber endomorphism. Φ_{T/2,i} = Dφ_{T/2}(y_i(0)) maps fiber V_0 = T_{y_i(0)} to fiber V_h = T_{y_i(T/2)}, not back to V_0. The commutation chain [M_i, G_i] = 0 was provisional and assumed same-fiber matrices throughout.

Step 3 — The pair-orbit picture (current best proposal, not locked)  
Instead of forcing a single-fiber operator, work with the pair (C_i, Y_i) where Y_i = (12)·C_i. The α-fixed kernel lives in the combined pair-orbit loop space. Alpha-fixed pairs should be parameterized by a single component because the action swaps the two orbits. The candidate formula becomes:

    K_i^{fib}  := ker(M_i − I) / N_i
    B_i^+      := (A_F-even) ∩ (σ_3-trivial) ∩ K_i^{fib}    [structural continuation sector]
    K_i^{PT}   := ((A_F-even) ∩ K_i^{fib}) / B_i^+           [genuine piano-trio sector]
    d_i^{cand} := (1/2) · dim_ℝ K_i^{PT}                     [candidate branch count]

where A_F is the phase-space linear part of F_beta, with the typed identity:
    A_F: V_t → V_{-t}   (maps fiber at time t to fiber at time -t, NOT pointwise identity)

Step 4 — The dihedral representation issue  
F_beta contains a transposition (12), which conjugates the 3-cycle σ_3 to its inverse:
    A_F · σ_3 · A_F^{-1} = σ_3^{-1}

This means ⟨σ_3, F_β⟩ is a dihedral group D_3, not two commuting projectors. The nontrivial σ_3 sector is a real 2D irrep where A_F acts as a reflection. So "(A_F-even) ∩ (σ_3-nontrivial)" needs dihedral representation theory, not two independent eigenvalue masks.

Step 5 — The multiplicity question  
The old (1/2) factor came from: K_i^{PT} is symplectic → bifurcations come in symplectic pairs → one branch per pair. But if the branch sector is an A_F-even subspace and A_F is anti-symplectic, the fixed subspace is typically Lagrangian-like, NOT symplectic. The symplectic-pair argument no longer transfers automatically.

Safe pre-registration language:
    d_i = d_i_candidate only when:
      (i)  the +1 block of M_i is semisimple beyond N_i
      (ii) the Lyapunov-Schmidt crossing form on K_i^{PT} is nondegenerate
      (iii) the representation structure on K_i^{PT} is understood (sym vs Lagrangian)

Step 6 — What needs to close before code  
The "typed transport lemma" is the single paper-only gate:
  1. Define A_F, A_τ, A_{Rπ}, P_{12}, and Φ_t := Dφ_t(y_i(0)) with V_0, V_h, V'_0, V'_h explicit
  2. State reversible identities with typed domains and codomains
  3. Construct the α-induced map on V_0 through the partner fiber, without collapsing intermediate fibers
  4. Prove or refute G_i^2 = I (holonomy check — loop-level α^2 = e is necessary but not sufficient)
  5. Derive the ⟨σ_3, F_β⟩ dihedral representation on K_i^{fib}
  6. Define the branch candidate sector from the representation
  7. Only then decide: multiplicity is dim, 1/2 dim, or crossing-form-gated

### Where we are stuck  

The five open blockers are all in the typed transport lemma:
A. "A_F fixes every point" is wrong — it maps V_t → V_{-t}, pointwise only at t=0 and t=T/2
B. The base-phase of the partner loop is unresolved (shifted partner Y_i^h(t) = Y_i(t+T/2), or free-loop quotient, or explicit phase transport — must choose one)
C. G_i^2 must be computed directly on the chosen representative — can leave holonomy behind
D. The graph parameterization (alpha-fixed pairs = graphs) must be proved to descend through the N_i neutral quotient
E. The dihedral structure of ⟨σ_3, F_β⟩ must be derived before any projector masking

### Why this is interesting  

The τ-cancellation in §3 of the draft is a genuinely nice observation: F_β contributes one M ↔ M^{-1} twist, τ contributes another, and the two cancel (even number of anti-symplectic factors → commutation). That structural fact survives the Codex review. What's not yet proven is that the fiber transport can be assembled into a single canonical G_i on V_0 without picking up a holonomy cocycle from the loop identification.

If the typed transport lemma closes: K_facet_v0.3 is a real spectral prediction. Compute M_i for all 21 rows, apply the linear-algebra recipe, sum.
If the typed transport lemma fails (holonomy can't be absorbed): v0.3 stops at a useful negative, and the clean result is that the induced-representation functional has hidden cocycle dependence that prevents a single-fiber d_i.

---

## rough — couplings surface as Kenny Wheeler's missing secret of magnesium / 1/φ³

aka φ^−3 the malice mizer a compact golden-ratio attenuation / divergence coefficient in his symbolic field vocabulary. a small outward residual, wake remainder, or divergence tail against a normalized convergent field.

very rough, do not clean up yet

The couplings surface is the thing we haven't named yet in the v0.3 derivation. Lyapunov-Schmidt reduces the bifurcation equation on K_i^{PT} to a finite-dimensional problem, and the leading-order coefficient matrix of that reduced equation IS the couplings surface — it's the object that decides whether a candidate direction in K_i^{PT} produces a real branch or a ghost. We keep calling it "the crossing form" and "nondegeneracy condition" but we haven't asked what it actually looks like.

Kenny Wheeler angle: Wheeler's thing is the suspension that refuses to resolve. His harmonics hang in a third place — not tonic, not dominant, something in between that you'd expect to move but doesn't. If the couplings surface has a Wheeler character, it means the crossing form has a direction where the parameter derivative neither clearly crosses nor stays — it hovers near degenerate. That would be the honest explanation for why K_facet collapses to 0 in v0.2: the coupling wasn't absent, it was suspended. The static equivariance scan can't hear a Wheeler suspension, it only counts clear crossings.

Magnesium: Mg atomic number 12. Phase space after symplectic reduction by translation + linear momentum is ℝ^12. The monodromy M_i is a 12×12 symplectic matrix. "Missing secret of magnesium" in this context might be that the ambient dimension is the same as the structural dimension — the 12-ness is not an accident of the problem setup, it's the full width of the available fiber. If B_i^+ (structural sector) is reliably 2D (flow + energy neutrals already quotiented, plus one F_beta/sigma3 continuation direction), then K_i^{PT} should generically be in a 10-dimensional ambient for each row. That's still enough room for multiple piano-trio branches even after the dihedral D_3 representation carves out the trivial isotypics. The "missing secret" might just be that nobody has looked at the 12×12 monodromy of the equal-mass choreographies directly — the literature cares about their existence, not their variational spectrum.

1/φ³ angle: φ = golden ratio ≈ 1.618. 1/φ³ ≈ 0.236. Three facts worth noting:
  - φ³ = 2φ + 1, so 1/φ³ = 1/(2φ+1). This ratio comes up in three-term Fibonacci-type recursions, which are the same kind of recursion that governs period-tripling in discrete dynamical systems.
  - If the crossing form eigenvalue on a K_i^{PT} direction is O(ε^{1/φ³}) in the symmetry-breaking parameter ε, that would be a very slow approach to degeneracy — slower than algebraic in any integer power, but faster than log. Whether this is numerically visible in the monodromy spectrum is an open question.
  - More speculatively: Z3 symmetry of the parent choreography means σ_3 acts on K_i^{fib} with eigenvalues {1, ω, ω²} where ω = e^{2πi/3}. The D_3 dihedral representation decomposes as 1D trivial + 2D standard. If the +1 monodromy eigenvalues show up with a 3:2 ratio (3 trivial-isotypic vs 2 nontrivial), the effective d_i contribution per row would be 1/3 of the kernel dimension — that ratio is not 1/φ³ but the instinct that the couplings surface encodes some irrational ratio is worth keeping.

The Kenny Wheeler + magnesium + 1/φ³ cluster as a single hypothesis: what if the crossing form on the nontrivial D_3 isotypic component of K_i^{fib} has a canonical coupling strength that is determined purely by the F_beta cocycle structure, and that value happens to be near 1/φ³ or some simple rational function of φ? This would mean d_i is not an integer-valued count but a weighted count, where the weight encodes how "audible" each branch direction is to a probe that only sees the F_beta-even sector. Wheeler's suspended chord = the branch that exists but is below the threshold of the sensor. Magnesium/12 = the dimension of the ambient fiber that hides it. 1/φ³ = the coupling threshold below which the branch is structurally present but observationally silent.

This is completely unverified. It might be numerology. Or it might be the right way to ask why v0.2 found 0 rather than a positive integer — not because there are no branches, but because the coupling at equal mass is exactly at the Wheeler suspension point, and the branches only become audible when ε > 0 breaks the symmetry enough to lift the suspension.

Next step if this is real: compute the coupling matrix entries for one row of the 21, from the monodromy spectrum + the explicit ρ(F_beta) / ρ(sigma3) actions, and see what the leading eigenvalue of the crossing form looks like. If it's near 1/φ³ of the next eigenvalue, the conjecture has legs. If not, bin it.
