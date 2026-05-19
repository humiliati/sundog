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

## 4. The ansatz facet and corrected prediction

The compact Li-Liao supplementary ansatz is not an unbiased sample of full 3D
orbit space. It is a symmetry-conditioned slice:

```text
r1 = (-1, 0, 0)
r2 = ( 1, 0, 0)
r3 = ( 0, 0, z0)

v1 = (vx, vy,  vz)
v2 = (vx, vy, -vz)
v3 = (-2vx/m3, -2vy/m3, 0)
```

Let `Rpi = diag(-1, -1, 1)`, the 180-degree rotation around the z-axis. At
`t=0`, the state is fixed by the beta-type operation

```text
F := ((12), time reversal, Rpi)
```

because `Rpi r2 = r1`, `Rpi r1 = r2`, `Rpi r3 = r3`, and the velocity
condition for time reversal gives `-Rpi v2 = v1`, `-Rpi v1 = v2`,
`-Rpi v3 = v3`. Since the Newtonian flow with `m1=m2` commutes with `F`, every
trajectory produced by this ansatz carries this structural residual `Z2`.

So the original unconditioned estimator is void:

```text
K_raw := #{ i : S_i != empty }
```

The catalog cannot populate the "Type 0" stratum, because generic asymmetric
orbits are outside the facet by construction. The ansatz is the crystal cut.
The 21 choreographies and 273 piano-trios are the sundogs visible through that
cut, not all possible sundogs in the sky.

Corrected working prediction (sundog theorem v0.2):

For each equal-mass choreography `C_i` in the ansatz facet, classify its
*enhanced* residual spacetime isotropy relative to the structural facet
generator `F`. Define

```text
S_i := { rho in {alpha, beta, gamma, delta} : rho in Iso(C_i) }
E_i := S_i / <F>     # inequivalent emergent residual generators inside facet
d_i := facet-conditioned daughter multiplicity for C_i
```

The empirical target is now

```text
# distinct piano-trio families visible through the Li-Liao ansatz facet
  = K_facet := sum_i d_i
```

The immediate workbench task is to make `d_i` operational: determine which of
`alpha/beta/gamma/delta` is structurally baked in by the ansatz, which are
emergent, and which are equivalent under `sigma3` conjugation and spatial gauge.
The algebra above pins the structural generator as beta-type. The numerical
detector should verify that before reading any daughter count.

---

## 5. The empirical pin

Liao (2025) supplementary:
- supplementary-A.txt: 10,059 3D periodic orbits with m₁=m₂=1, m₃ = 0.1·n for n ∈ {1, ..., 20}. Of these, 21 sit at the equal-mass slice n=10 with full Z₃ choreography property.
- supplementary-B.txt: 273 3D piano-trio orbits across the same m₃ grid.

Naïve ratio: 273 / 21 ≈ 13. After the ansatz correction, this is not an
unconditioned branching ratio. It is a facet-conditioned multiplicity: roughly
how many sampled piano-trio rows are visible per equal-mass choreography through
the Li-Liao isosceles/time-reversal facet.

We want either:
- **(P1)** # of *facet-visible distinct piano-trio families* `K_emp` matches the structurally-predicted `K_facet`. Each family is sampled at multiple `m3` values, so `273/K_emp` remains a sampling-density diagnostic, not the prediction itself.
- **(P2)** If `K_emp != K_facet`, the discrepancy is itself a structured object and should factor through one of: (i) family bifurcation as `m3` crosses critical values, (ii) chirality doubling we missed, (iii) equivalence collapse from `sigma3` conjugation or spatial gauge, (iv) an ansatz facet that enforces more symmetry than beta-type alone.

Either P1 or P2 is informative. The outcome that kills the theorem is `K_facet`
and `K_emp` being unrelated after the facet conditioning is accounted for.

---

## 6. Check procedure (numerical)

For each of the 21 choreographies:

1. **Ingest** initial conditions and period from supplementary-A. The file uses a compact fixed ansatz, not full 18D state rows: expand `O_index(m3), z0, vx, vy, vz, T, stability` into the three positions and velocities before integrating. Filter the `m3=1` slice.
2. **Discretize** X(t) on N samples per period (N ~ 10³ initially).
3. **Facet precondition:** verify the beta-type ansatz symmetry `F` and the choreography condition `sigma3` before treating a row as one of the 21 choreographies.
4. **For each ρ ∈ {α, β, γ, δ}:**
   - Construct ρ·X(t) by permuting bodies, time-shifting, optionally reversing time, optionally flipping z.
   - Compute residual r_ρ(X) := min over rotation R ∈ SO(3) and time-phase φ ∈ S¹ of  ||ρ·X(·) − R · X(· + φ)||_∞ / scale(X).
   - Mark ρ ∈ Iso(C_i) if r_ρ(X) < tolerance.
5. Tabulate `S_i`, quotient out the structural facet generator `F`, and compute `d_i`. Sum to get `K_facet`.

For piano-trio family counting:

6. **Ingest** supplementary-B. For each piano-trio orbit, identify the residual Z₂ type by the same procedure restricted to body-relabel (12).
7. **Cluster** piano-trios into families by continuation in m₃: orbits with same residual type, same free-group-word, and continuously-deformable initial conditions belong to one family.
8. Count `K_emp` = # of distinct facet-visible families.

Compare `K_facet` ↔ `K_emp`.

---

## 7. Sources of error to design around

- **SO(3) gauge:** every orbit lives in a 3-parameter SO(3) gauge orbit; isotropy checks must min over rotation. Standard fix: align orbits to principal axes of inertia tensor at t=0 before comparing.
- **Time phase:** likewise S¹ phase. Fix by anchoring at, e.g., the configuration of minimal moment of inertia.
- **Reflection ambiguity in 3D:** improper rotations are a separate factor. The classifier should keep the main gauge at SO(3) and apply z-reflection only through the γ/δ candidate operators; otherwise a free O(3) minimization can collapse γ/δ into α/β.
- **Period sampling:** time discretization should be coprime with T/2 and T/3 to avoid alias-induced false isotropies.
- **Liao's "scale-invariant averaged period":** confirm normalization before doing any cross-orbit comparisons.

---

## 8. Open questions / blowout directions

- **8.1 Lagrange vs figure-eight, formalized.** The persistence-vs-isolation diagnostic from the brainstorm: confirm the moduli-dimension criterion holds across all known equivariant 3-body families (Euler collinear, Lagrange equilateral, Broucke–Hadjidemetriou–Henon, Chenciner–Montgomery 8, the 695 planar, the 21 3D choreographies). Build a 2D table: (family, isotropy moduli dim, observed mass-range width).
- **8.2 Higher-N generalization.** For N=4, S₄ has more subgroups; predict the analog descent multiplicities and check against the smaller-but-extant N=4 catalog (Chen et al.).
- **8.3 Stability-as-coherence.** Among confirmed descended piano-trios, predict the *stable* subset using a Bragg-style integer matching between the Floquet exponents at ε=0 and the descended period ratio. Liao reports 1,996 / 10,059 ≈ 20% stable in 3D; sub-question: is the stable fraction among piano-trios systematically higher, lower, or equal?
- **8.4 The mesa hook.** Once `K_facet = K_emp` is in hand (if it is), draft the mesa-analog: NN substrate hosts inner optimizers iff loss-landscape has a residual subgroup of the data-symmetry surviving the training perturbation. Make precise enough to be falsifiable on toy NN setups.
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
| 2026-05-18 | Corrected Section 4 after supplement-format grounding: Li-Liao ansatz is a beta-type symmetry facet, so raw K_pred is void and replaced by facet-conditioned K_facet. |
| (next)     | Ingest supplementary A, run isotrophy detector on 21 choreographies.    |
| (next)     | Ingest supplementary B, cluster into families, compute K_emp.            |
| (next)     | Compare. Iterate.                                                        |

---

## 11. Citation

Li, X. & Liao, S. *Discovery of 10,059 new three-dimensional periodic orbits of the general three-body problem.* arXiv:2508.08568 (2025).
