# v0.7 Mechanism Preregistration: gamma_1 Direction-of-Instability

Status: **REGISTERED 2026-05-24** (parent; no compute yet). This document
locks the v0.7 body / projection / observable and the non-circularity
discipline. Form locks for v0.7a (audit) and v0.7b (held-out predictor,
conditional on v0.7a passing) will be separately registered children
that pin specific operational definitions, binning, df, and falsifier
choices.

Audience: v0.7a/v0.7b form-lock author; paper-side reviewer of the
v0.6 -> v0.7 transition (catalog-coordinate close + orbit-dynamics open).

Companions:

- `kfacet_v06_writeup.md` -- v0.6 conditional-independence chapter close
  (the joint v0.4 + v0.5 + v0.6 catalog-coordinate envelope that opened
  v0.7).
- `kfacet_v04b_gamma3_form.md` (retired) and
  `kfacet_v04b_gamma3prime_form.md` (falsified) -- v0.4b disallowed-
  feature provenance.
- `kfacet_v04a_domain_map_preregistration.md` -- supp-B monodromy
  computation source; per-row M_i and eigenstructure are derivable from
  this receipt.
- v0.3 gamma_1 sidecar in the v0.3h audit chain -- the (sidecar) origin
  of the gamma_1 quantity, now promoted to primary registration.

Frame: v0.4 ruled out the Z_2 tangent/orbit-gauge-rigidity shadow as a
stability projector. v0.5 ruled out the 2-bit catalog branch shadow as
a held-out predictor. v0.6 ruled out the continuous catalog-coordinate
(E, |L|) shadow beyond branch labeling. **v0.7 leaves catalog-coordinate
space**: it asks whether the row's per-orbit monodromy operator has a
geometric structure — specifically the DIRECTION of its
largest-real-part eigenvector — that stratifies stability.

## What v0.7 Is Not

v0.7 is NOT:

- **A Floquet-magnitude-based feature.** The unstable eigenvalue's
  modulus IS the stability indicator (or trivially equivalent to it).
  v0.4b's disallowed-feature list explicitly retired such features
  for stability circularity. v0.7 uses the eigenvector DIRECTION as a
  geometric quantity in phase space, not the eigenvalue magnitude or
  the count of off-unit-circle eigenvalues.

- **A within-U-rows analysis.** Conditioning the audit on stability =
  U is direct circularity. v0.7's gamma_1 quantity must be defined
  for ALL 273 rows (S and U) under a single operational rule, and
  the audit asks whether the geometric feature stratifies S vs U
  across the full catalog (or a stratum thereof).

- **A re-use of v0.4b's K_fib tangent-isotypic decomposition.** v0.4b
  retired the tangent isotypic split because `F_beta` does not
  preserve K_fib on supp-B. v0.7 must NOT use the
  K_fib-isotypic-projection step. The gamma_1 direction is computed
  from the raw monodromy M_i, not from the K_fib slice.

- **A re-projection onto the v0.5a (m_3, z_0) branch shadow or the
  v0.6 (E, |L|) shadow.** v0.7's projection axis is the per-row
  eigenvector DIRECTION in phase space, not the catalog-coordinate
  features that v0.4/v0.5/v0.6 already exhausted.

## Body / Projection / Observable

```text
Body:
  supplementary-B piano-trio orbit as primary orbit-dynamics object.
  domain: 273 rows, already verified Z2_clean by v0.4a; per-row M_i
  and Floquet eigenstructure are derivable from the v0.4a receipt.

Projection:
  orbit-dynamics direction shadow
    row -> gamma_1(row), the eigenvector DIRECTION in phase space
           of the row's largest-real-part Floquet eigenvalue,
           projected onto a pre-registered geometric reference frame.

Observable:
  per-row gamma_1 direction feature (a scalar or a small-cardinality
  bin label derived from the direction) joined with the supp-B
  stability label S/U.

Per-row computational cost:
  monodromy M_i and its eigenstructure are already computed in the
  v0.4a two-pass classifier; v0.7 may reuse those numerics under
  the disallowed-feature audit. No new orbit integration. Per-row
  feature extraction is seconds.
```

The projection family is **orbit-dynamics**: the monodromy M_i carries
information about how the orbit's tangent space evolves under one
period of the dynamics. The largest-real-part Floquet eigenvector
captures the orbit's most-dynamically-active direction in tangent
space — for U rows this is the unstable direction; for S rows it is
the direction along which neutral perturbations are largest.

## Operational Definition Challenges

gamma_1's operational definition is the load-bearing design problem
of this parent registration. Three concerns:

1. **Floquet eigenvalue choice for S rows.** Stable orbits have all
   Floquet eigenvalues on the unit circle, paired by conjugation and
   typically degenerate at +1 (translations, time direction,
   energy-conservation Jordan block, etc.). Picking ONE eigenvalue
   among the degenerate ones is arbitrary unless a deterministic
   tie-break is locked.

2. **Eigenvector direction ambiguity.** Eigenvectors are defined up
   to sign (or up to a complex phase for complex eigenvalues). The
   form lock must specify a deterministic normalization.

3. **Reference frame.** gamma_1 lives in 6D phase space (delta-r,
   delta-v) at the orbit anchor. Projecting onto a pre-registered
   geometric axis requires choosing the reference frame (anchor body
   frame, center-of-mass frame, Z_2-symmetry frame, etc.).

These concerns are NOT resolved by the parent registration. They are
flagged so the v0.7a form-lock author can take direction and lock a
specific operational definition before any compute.

## Candidate Operational Definitions of gamma_1 (Sketches for v0.7a)

The v0.7a form lock will pick one of the following (or a registered
variant), with explicit non-circularity provenance:

```text
D1. Largest-real-part eigenvalue:
    gamma_1(row) = (unit-normalized) eigenvector of the Floquet
                   eigenvalue whose REAL PART is largest.
    - For S rows: picks the eigenvalue closest to +1 from the right
      (real part = 1 if all eigenvalues are unit modulus; tie-break
      below).
    - For U rows: picks the unstable eigenvalue (real > 1 by
      definition).
    - tie-break: among eigenvalues with equal real part, pick the
      one with smallest absolute imaginary part. Further ties go
      to smallest argument.

D2. Largest-magnitude eigenvalue:
    gamma_1(row) = eigenvector of the eigenvalue with largest |lambda|.
    - For S rows: |lambda| = 1 for all; arbitrary among unit-circle
      eigenvalues.
    - For U rows: unique unstable eigenvalue (largest |lambda| > 1).
    - tie-break: argument closest to 0, then smallest absolute
      imaginary part.

D3. Restricted to "well-defined" rows; companion sidecar for ill-defined rows:
    gamma_1(row) defined only when the eigenvalue choice is
    unambiguous (e.g., a unique largest-|lambda| eigenvalue exists
    above a pre-registered floor). Other rows go to a "gamma_1_undefined"
    bin which is recorded but report-only.
    - Risk: the "well-defined" subset may correlate with stability
      (most U rows are well-defined; many S rows are not), which
      reintroduces circularity through the back door.
    - Pre-registered guard: alignment-tightness scalar between
      "gamma_1_defined" and the v0.5a branch label, with a stop
      condition at alignment > 0.8.

D4. Z_2-isotypic phase-space split of gamma_1:
    Decompose gamma_1 into its Z_2-even and Z_2-odd components under
    the supp-B (12)-swap symmetry. Use the EVEN/ODD FRACTION as the
    feature (continuous in [0, 1]).
    - Domain-non-circular: uses the catalog's known Z_2 symmetry
      (verified by v0.4a's domain map), not the stability label.
    - Risk: re-uses the Z_2 projection that v0.4 already ruled out.
      Promotion requires explicit paper-side justification that the
      DIRECTION-level Z_2 information differs from the
      tangent-isotypic-DIMENSION information v0.4 tested.

D5. Phase-space angle decomposition (spatial vs velocity):
    gamma_1 in 6D space splits naturally into a 3D delta-r component
    and a 3D delta-v component. Use the ratio
      |gamma_1_velocity| / (|gamma_1_velocity| + |gamma_1_spatial|)
    as the feature (continuous in [0, 1]).
    - Non-circular: ratio is a geometric scalar of the eigenvector.
    - Risk: may be degenerate (a Hamiltonian flow's tangent dynamics
      naturally mixes position and velocity); the feature may not
      vary across rows.
```

The v0.7a form lock must lock ONE of D1-D5 (or a registered variant)
with EXPLICIT non-circularity provenance. Multiple parallel definitions
are NOT permitted under a single v0.7a registration.

## Non-Circularity Audit (Load-Bearing)

```text
The Floquet eigenstructure of M_i is computed from the row's
monodromy operator, which in turn is computed from the orbit's
tangent dynamics under one period. This computation:
  - uses the row's published initial conditions (m_i, r_i, v_i),
  - uses the row's published period T,
  - uses the three-body Hamiltonian dynamics,
  - does NOT use the supp-B stability label S/U.

Therefore the EIGENVECTOR of M_i is a function of the orbit and
its anchor-period only; it does not inspect S/U.

The CIRCULARITY risk is in three places, and the v0.7a form lock
must address all three:

1. EIGENVALUE-CHOICE CIRCULARITY:
   - "Pick the eigenvalue WITH MAGNITUDE > 1" picks only on U rows.
     This IS the stability label. DISALLOWED.
   - "Pick the largest-magnitude eigenvalue regardless of whether
     it's > 1" is allowed IF a deterministic tie-break is locked
     pre-data for the unit-circle case.
   - "Pick the eigenvalue with the largest real part" is allowed
     under the same tie-break discipline.

2. WELL-DEFINEDNESS CIRCULARITY:
   - "gamma_1 is defined only when..." where the condition correlates
     with stability (e.g., "only when there's an unstable eigenvalue")
     reintroduces circularity through the back door.
   - If the form lock uses a definedness-condition, an alignment-
     tightness scalar between "gamma_1_defined" and the stability
     label MUST be pre-registered with a stop condition.

3. FEATURE-EXTRACTION CIRCULARITY:
   - "Use the unstable Lyapunov rate as the feature" IS the stability
     label. DISALLOWED.
   - "Use the eigenvector DIRECTION's projection onto a pre-registered
     geometric axis" is allowed.
   - "Use the eigenvalue's MAGNITUDE or any function of it" is
     DISALLOWED (this is the v0.4b inheritance).

The non-circularity provenance must be argued explicitly in the
v0.7a form lock document. A v0.7a registration whose operational
definition does not pass this audit is rejected pre-compute.
```

## Disallowed Features (Inheritance from v0.4b + v0.7-specific)

The v0.7 form locks may NOT use:

```text
v0.4b inheritance:
  - the stability label S/U (or any function of it),
  - spectral radius, unit-circle count, max |lambda|, or any
    function of M_i eigenvalues' MAGNITUDES,
  - Krein signature,
  - "has off-unit-circle Floquet pair" (this IS S/U),
  - any feature derived from the K_fib tangent-isotypic
    decomposition retired in v0.4b's gamma_3 form lock.

v0.5/v0.6 inheritance:
  - the v0.5a branch hash output (any function of (m_3, z_0)
    indicator bits),
  - Q_E or Q_|L| or any function of the v0.6a quartile assignments,
  - E, |L|, or any conserved-quantity scalar -- v0.6 closed on
    these and they are retired for v0.7.

v0.7-specific:
  - The MAGNITUDE of any Floquet eigenvalue (re-asserted).
  - Any operational definition where gamma_1's definedness
    condition correlates with stability at alignment > 0.8 (this
    is a back-door circularity).
  - Any feature that requires conditioning on S or U separately
    (within-S or within-U analyses are NOT permitted at the
    primary audit level; only post-hoc diagnostics).
```

## Inheritance Discipline From v0.5 / v0.6

The v0.5 and v0.6 chapter closes locked six register-shaping lessons.
v0.7 inherits all six:

1. **Audit-then-predictor separation.** v0.7a registers a chi-squared
   independence audit on a binned gamma_1 feature; v0.7b separately
   registers a held-out predictor. An audit pass does NOT license a
   predictor; the predictor must pass its own pre-registered
   held-out gate.

2. **Asymmetric falsifier for the predictor.** v0.7b requires BOTH
   `McNemar p <= 0.01` AND positive accuracy delta against the
   always-U baseline. Inherited from v0.5b.

3. **Held-out partition discipline.** v0.7b defaults to
   leave-one-m_3-bin-out (12 bins with N >= 5 on supp-B), matching
   v0.5b. Alternative partitions must be motivated paper-side and
   pre-registered.

4. **Tie rule and absent-bin fallback.** v0.7b inherits the
   conservative tie rule (predict U) and the global-majority
   fallback for absent bins, with `absent_bin_fallback_used` flagged
   on the receipt.

5. **Alignment-tightness guard (v0.6 inheritance).** v0.7a MUST
   pre-register an alignment-tightness scalar between the gamma_1
   feature's bins and the v0.5a branch label, with a stop
   condition at alignment > 0.8 (alignment-warning verdict) or
   alignment > 0.95 (re-registration mandate). v0.6a's value of
   0.956 demonstrated this guard is load-bearing for any
   catalog-correlated feature.

6. **Sparse-cell fallback tree (v0.6 inheritance).** v0.7a and v0.7b
   MUST pre-register the three-way fallback (chi-squared / exact
   permutation / sparse-inconclusive) with locked thresholds and
   locked permutation seed. This is permanent discipline.

## Candidate Audit Forms (Sketches for v0.7a Registration)

The v0.7a form lock will pick one of the following audit forms with
specific binning and df:

```text
A. Direction-projection quartile audit:
   Project gamma_1 onto a pre-registered geometric axis (e.g., body-3
   spatial axis, center-of-mass frame z-axis, or Z_2-even projection).
   Quartile-bin the projection magnitude over supp-B's 273-row
   distribution. 4-bin x 2-class contingency.
   df = 3 (or occupied_bins - 1 if sparse-cell fallback fires).

B. Velocity-fraction quartile audit:
   Use the gamma_1_velocity / (gamma_1_velocity + gamma_1_spatial)
   ratio (continuous in [0, 1]). Quartile-bin over supp-B.
   4-bin x 2-class contingency, df = 3.

C. Z_2 isotypic-fraction quartile audit:
   Use the gamma_1_even / (gamma_1_even + gamma_1_odd) Z_2 fraction.
   Quartile-bin. 4-bin x 2-class. df = 3.
   Requires explicit non-circularity vs v0.4b's retired isotypic
   dimension feature.

D. Direction-angle binary audit:
   "Predict S iff gamma_1 lies in pre-registered angular sector X."
   2-bin x 2-class. df = 1, critical 6.63 at p = 0.01.
   Coarser test; more power if the angular structure is sharp.
```

All four forms must include the alignment-tightness guard against
the v0.5a branch label and the sparse-cell fallback tree.

## Open Design Questions (Resolved in v0.7a/v0.7b Form Locks)

1. **Operational definition (D1-D5).** Which definition of gamma_1?
   D1 (largest real part) and D2 (largest magnitude) are most-natural;
   D4 (Z_2 isotypic) and D5 (velocity fraction) are derived features.

2. **Tie-break discipline.** Which specific deterministic tie-break
   handles the S-row degeneracy? Pre-registered, not data-driven.

3. **Reference frame.** Anchor frame, center-of-mass frame, or
   Z_2-symmetry frame for the geometric projection?

4. **Audit form (A-D).** Which projection of gamma_1 is the primary
   audit feature?

5. **Sparse-cell threshold inheritance.** v0.6b used min_expected = 5
   and min_occupied_bin_count = 2 with permutation seed 20260523.
   v0.7a inherits these unless explicitly re-registered.

6. **|L|-loud-signal threshold from v0.6.** A residual question
   from v0.6: the |L| sidecar at p = 0.074 was below the
   loud-signal threshold for fresh registration. v0.7 explicitly
   does NOT re-register |L|; conserved quantities are retired.

## Pre-Mortem (Recorded Now)

```text
Risk 1: gamma_1's operational definition is ill-defined for S rows
        in a way that DOES align with stability.
        Mitigation: alignment-tightness scalar pre-registered with
        stop condition at alignment > 0.8 against the stability
        label (NOT just against the v0.5a branch).

Risk 2: gamma_1's eigenvector direction depends on numerical
        precision of the v0.4a monodromy computation. Stability of
        the eigenvector direction across the v0.4a Pass 1 vs Pass 2
        tolerances must be checked on the 24 Pass 2-rescued rows
        as a sanity gate.

Risk 3: The non-circularity audit may fail. If the v0.7a operational
        definition cannot be made non-circular under reasonable
        framings, the v0.7 chapter closes as
        "operational_definition_failed_non_circularity_audit" without
        any compute, and the program either:
          - opens v0.7' with a fresh operational definition,
          - opens v0.8 with a different mechanism family
            (action-angle / KAM, m_3 = 0.4 sub-catalog targeted,
             or close-without-v0.8).

Risk 4: gamma_1 may be near-constant on supp-B. If the per-row
        gamma_1 distribution under the locked operational definition
        has insufficient spread, the constant-feature retirement
        rule (v0.5/v0.6 inheritance) fires and the audit cannot run.
        Mitigation: the v0.7a runner emits a per-row gamma_1
        distribution as a pre-audit diagnostic; the form lock must
        check distribution spread before claiming a verdict.
```

## Stop Conditions

1. **Non-circularity audit failure.** If the v0.7a operational
   definition cannot be argued non-circular, the form lock is
   rejected pre-compute. The v0.7 chapter closes immediately.

2. **Constant gamma_1 on supp-B.** If the per-row gamma_1 feature
   has insufficient variation (e.g., all rows project to the same
   bin), constant-feature retirement fires. The chapter closes as
   "gamma_1_constant_on_supp_b".

3. **Alignment-tightness firing.** If the v0.7a binning has
   alignment > 0.8 against the v0.5a branch label, v0.7b cannot
   inherit the default leave-one-m_3-bin-out partition. v0.7b must
   re-register with an alignment-breaking partition (within-branch
   audit, mirroring v0.6b's pattern).

4. **v0.7a fails the chi-squared / permutation test.** The chapter
   closes as a structural-negative on the gamma_1 direction shadow.
   v0.7c (alternative gamma_1 formulation) may be registered
   separately or the program closes.

## Lock-In Statement

This parent registration is committed before:

- any per-row gamma_1 has been computed under any operational
  definition,
- any specific binning has been locked,
- the non-circularity audit has been argued in writing for any
  specific operational definition.

The body / projection / observable, the non-circularity audit
framing, the disallowed-feature list, the inheritance discipline
from v0.5/v0.6, and the stop conditions are locked.

Any change to those after the v0.7a form lock is a re-registration
of the parent, not a refinement of a child form.

Implementation may proceed to:

1. Add a per-row monodromy + Floquet eigenstructure computation
   against the v0.4a manifest. The v0.4a Pass 1/Pass 2 receipts
   already carry M_i numerically; v0.7 may reuse those numerics
   under the disallowed-feature audit (no K_fib decomposition is
   used).
2. Sanity-check the eigenvector direction stability across Pass 1
   vs Pass 2 tolerances on the 24 Pass 2-rescued rows.
3. Emit a per-row gamma_1 distribution as a pre-audit diagnostic.

No audit verdict is licensed until a v0.7a form lock is separately
registered with:
- an explicit operational definition of gamma_1 (D1-D5 or a
  registered variant),
- an explicit non-circularity argument that addresses all three
  circularity risks above,
- a deterministic tie-break for the S-row degeneracy,
- an alignment-tightness guard against the v0.5a branch label,
- a sparse-cell fallback tree.

## Doc Trail

- `kfacet_v06_writeup.md` -- v0.6 conditional-independence chapter close
  (opens v0.7).
- `kfacet_v06a_energy_quartile_audit_form.md` -- v0.6a catalog audit
  (alignment-tightness guard provenance).
- `kfacet_v06b_within_branch_energy_audit_form.md` -- v0.6b within-branch
  audit (sparse-cell fallback tree provenance).
- `kfacet_v05a_branch_map_form.md`, `kfacet_v05b_branch_predictor_form.md`
  -- v0.5 audit+predictor pattern (asymmetric falsifier inheritance).
- `kfacet_v04b_gamma3_form.md` (retired) and
  `kfacet_v04b_gamma3prime_form.md` (falsified) -- v0.4b disallowed-
  feature provenance for Floquet-magnitude features.
- `kfacet_v04a_domain_map_preregistration.md` -- supp-B monodromy
  source (per-row M_i is derivable from this receipt).

---

Parent registration locked. v0.7a (operational definition + audit
form) and v0.7b (held-out predictor) form locks pending. The
non-circularity audit is the load-bearing design problem before any
compute.
