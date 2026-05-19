# Sundog v. Isotrophy

**Test:** Z₃ → Z₂ descent of equal-mass 3D choreographies into (m₁=m₂=1, m₃≠1) piano-trios.
**Status:** workbench, open. First-prediction stage.
**Owner:** sundog
**Related:** `./isotrophy/files.math` — symbolic + numerical scratch.

**Spelling note:** `isotrophy` is the workbench name; the mathematical object
under test is spacetime isotropy.

---

## 0. One-paragraph posture

The sundog theorem must do strictly more than equivariant bifurcation theory or it isn't a theorem. The cheapest place it can earn its keep is the Z₃-choreography → Z₂-piano-trio descent in Li & Liao 2025: equivariance alone says "Z₃ ∩ Z₂ = {e}, generic descent breaks all symmetry"; the sundog refinement claims the surviving piano-trio count is fixed by the *additional* residual Z₂s living in each choreography's full spacetime isotropy. The number 273 (piano-trios) / 21 (equal-mass choreographies) ≈ 13 is the empirical pin we are trying to thread.

---

## 1. The objects

Newtonian 3-body, 3D, point masses m_i, periodic orbit X(t) = (x₁(t), x₂(t), x₃(t)) ∈ ℝ⁹, period T.

The full discrete symmetry group of the autonomous Hamiltonian, including reparameterization symmetries, is

  G_full = S₃ (body relabel) × S¹ (time translation) × ⟨τ⟩ (time reversal) × O(3) (spatial)

A given orbit X has *spacetime isotropy* Iso(X) ⊆ G_full = set of g ∈ G_full such that g · X = X (as a parameterized loop, modulo the obvious S¹ phase).

**Choreography:** Iso(X) ⊇ Z₃ = ⟨σ_3⟩ where σ_3 = ((123), T/3, +, I).
**Piano-trio (Liao convention, m₁=m₂):** Iso(X) ⊇ Z₂ = ⟨σ_2⟩ where σ_2 = ((12), T/2, +, I).

---

## 2. Mass-perturbation symmetry breakdown

At m₁=m₂=m₃=1: body-relabel group is S₃, |S₃| = 6.
At m₁=m₂=1, m₃=1+ε: body-relabel stabilizer is Z₂ = ⟨(12)⟩, |Z₂| = 2.

The S¹, τ, O(3) factors of G_full survive the perturbation untouched. So the *surviving* discrete spacetime symmetries are

  G_ε = Z₂ × S¹ × ⟨τ⟩ × O(3)

A choreography orbit at ε=0 with isotropy Iso(X) ⊆ G_full continues into ε≠0 iff Iso(X) ∩ G_ε is sufficient to constrain a non-degenerate continuation problem.

---

## 3. The four residual Z₂ generators

Among elements of G_ε whose body-relabel component is (12), exactly four are independent under the always-on dynamical symmetries:

| Name | Permutation | Time shift | Time reversal | z-reflection |
|------|-------------|------------|---------------|--------------|
| α    | (12)        | T/2        | no            | no           |
| β    | (12)        | 0          | **yes**       | no           |
| γ    | (12)        | T/2        | no            | **yes**      |
| δ    | (12)        | 0          | **yes**       | **yes**      |

α is the "pure" piano-trio generator (Liao's σ_2).
β is the "isoscele time-reversal" — body 1's forward trajectory equals body 2's reversed trajectory.
γ, δ are the chirality-flipped analogs (only nontrivial in 3D, vacuous in planar case).

These four exhaust the order-2 elements of G_ε with body-relabel component (12), up to redundancy with σ_3-conjugation inside the choreography itself. (Sanity: σ_3 · α · σ_3⁻¹ = ((23), 5T/6, +, I), which is α conjugated to a different transposition — same Z₂ orbit, not a new element.)

---

## 4. The prediction

Let the 21 equal-mass 3D choreographies be C₁, ..., C₂₁. For each C_i, classify Iso(C_i) ∩ G_ε. Define

  S_i := { ρ ∈ {α, β, γ, δ} : ρ ∈ Iso(C_i) }

Possible strata:

- **Type 0:** S_i = ∅ — Iso(C_i) ∩ G_ε is trivial. C_i descends to a *generic* (no surviving symmetry) orbit under ε≠0. **No piano-trio daughter.**
- **Type α / β / γ / δ:** |S_i| = 1 — exactly one residual Z₂. C_i descends to **one** piano-trio family parameterized by ε. (β/δ daughters appear as piano-trios after the parameterization is taken modulo time-reversal, which is invisible at the spatial-trace level Liao reports.)
- **Type 2:** |S_i| = 2 — two residual Z₂s present. **Two** piano-trio families. Implies an enhanced isotropy beyond generic, candidate for special-position choreographies.
- **Type ≥ 3:** higher enhancement, expected rare/empty.

Let

  K := #{ i : S_i ≠ ∅ }, weighted as needed by |S_i|

**Hard prediction (sundog theorem v0.1):**

  # distinct piano-trio families = K

This is the falsifier. It is not predicted by equivariant bifurcation theory, which only says daughters exist generically when (and only when) a residual subgroup survives the broken-symmetry parameter.

---

## 5. The empirical pin

Liao (2025) supplementary:
- supplementary-A.txt: 10,059 3D periodic orbits with m₁=m₂=1, m₃ = 0.1·n for n ∈ {1, ..., 20}. Of these, 21 sit at the equal-mass slice n=10 with full Z₃ choreography property.
- supplementary-B.txt: 273 3D piano-trio orbits across the same m₃ grid.

Naïve ratio: 273 / 21 ≈ 13.

We want either:
- **(P1)** # of *distinct piano-trio families* K_emp matches the structurally-predicted K_pred. (Each family generically sampled at multiple m₃ values, average ~273/K_emp samples per family.)
- **(P2)** If K_emp ≠ K_pred, the discrepancy is itself a structured object — should factor through one of: (i) family bifurcation as m₃ crosses critical values, (ii) chirality doubling we missed, (iii) under-discovered choreographies among the 21.

Either P1 or P2 is informative. The only outcome that *kills* the theorem is K_pred and K_emp being unrelated in a way no rep-theoretic refinement repairs.

---

## 6. Check procedure (numerical)

For each of the 21 choreographies:

1. **Ingest** initial conditions and period from supplementary-A. The file uses a compact fixed ansatz, not full 18D state rows: expand `O_index(m3), z0, vx, vy, vz, T, stability` into the three positions and velocities before integrating. Filter the `m3=1` slice and validate `sigma3` before treating a row as one of the 21 choreographies.
2. **Discretize** X(t) on N samples per period (N ~ 10³ initially).
3. **For each ρ ∈ {α, β, γ, δ}:**
   - Construct ρ·X(t) by permuting bodies, time-shifting, optionally reversing time, optionally flipping z.
   - Compute residual r_ρ(X) := min over rotation R ∈ SO(3) and time-phase φ ∈ S¹ of  ||ρ·X(·) − R · X(· + φ)||_∞ / scale(X).
   - Mark ρ ∈ Iso(C_i) if r_ρ(X) < tolerance.
4. Tabulate S_i for each C_i. Sum to get K_pred.

For piano-trio family counting:

5. **Ingest** supplementary-B. For each piano-trio orbit, identify the residual Z₂ type by the same procedure restricted to body-relabel (12).
6. **Cluster** piano-trios into families by continuation in m₃: orbits with same residual type, same free-group-word, and continuously-deformable initial conditions belong to one family.
7. Count K_emp = # of distinct families.

Compare K_pred ↔ K_emp.

---

## 7. Sources of error to design around

- **SO(3) gauge:** every orbit lives in a 3-parameter SO(3) gauge orbit; isotropy checks must min over rotation. Standard fix: align orbits to principal axes of inertia tensor at t=0 before comparing.
- **Time phase:** likewise S¹ phase. Fix by anchoring at, e.g., the configuration of minimal moment of inertia.
- **Reflection ambiguity in 3D:** improper rotations are a separate factor; γ and δ require checking against O(3) not just SO(3) gauge.
- **Period sampling:** time discretization should be coprime with T/2 and T/3 to avoid alias-induced false isotropies.
- **Liao's "scale-invariant averaged period":** confirm normalization before doing any cross-orbit comparisons.

---

## 8. Open questions / blowout directions

- **8.1 Lagrange vs figure-eight, formalized.** The persistence-vs-isolation diagnostic from the brainstorm: confirm the moduli-dimension criterion holds across all known equivariant 3-body families (Euler collinear, Lagrange equilateral, Broucke–Hadjidemetriou–Henon, Chenciner–Montgomery 8, the 695 planar, the 21 3D choreographies). Build a 2D table: (family, isotropy moduli dim, observed mass-range width).
- **8.2 Higher-N generalization.** For N=4, S₄ has more subgroups; predict the analog descent multiplicities and check against the smaller-but-extant N=4 catalog (Chen et al.).
- **8.3 Stability-as-coherence.** Among confirmed descended piano-trios, predict the *stable* subset using a Bragg-style integer matching between the Floquet exponents at ε=0 and the descended period ratio. Liao reports 1,996 / 10,059 ≈ 20% stable in 3D; sub-question: is the stable fraction among piano-trios systematically higher, lower, or equal?
- **8.4 The mesa hook.** Once K_pred = K_emp is in hand (if it is), draft the mesa-analog: NN substrate hosts inner optimizers iff loss-landscape has a residual subgroup of the data-symmetry surviving the training perturbation. Make precise enough to be falsifiable on toy NN setups.
- **8.5 Shape-sphere visualization.** Plot the 21 + 273 + 10k orbits on the 3D analog of Montgomery's shape sphere (the shape *space* for 3D triangles is the closed half-space ℝ³₊ with metric from the inertia tensor). Look for the predicted halo-angle clustering.

---

## 9. Data

- Numerical Tank page statement: the 2025 3D catalog reports 10,059 periodic
  orbits for `m1=m2=1, m3=0.1*n`, including 1,996 linearly stable orbits, 21
  equal-mass 3D choreographies, and 273 piano-trio orbits.
- Supplementary A (initial conditions, 3D periodic orbits): https://numericaltank.sjtu.edu.cn/three-body/three-body/gif/supplementary-A.txt
- Supplementary B (piano-trio orbits): https://numericaltank.sjtu.edu.cn/three-body/three-body/gif/supplementary-B.txt
- Full catalog mirror: https://github.com/sjtu-liao/three-body
- Paper: arXiv:2508.08568v1

---

## 10. Log

| Date       | Note                                                                     |
|------------|--------------------------------------------------------------------------|
| 2026-05-18 | Skeleton drafted. Prediction K_pred = #{C_i : S_i ≠ ∅} stated.           |
| (next)     | Ingest supplementary A, run isotrophy detector on 21 choreographies.    |
| (next)     | Ingest supplementary B, cluster into families, compute K_emp.            |
| (next)     | Compare. Iterate.                                                        |

---

## 11. Citation

Li, X. & Liao, S. *Discovery of 10,059 new three-dimensional periodic orbits of the general three-body problem.* arXiv:2508.08568 (2025).
