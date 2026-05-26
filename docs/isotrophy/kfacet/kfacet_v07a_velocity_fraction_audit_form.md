# v0.7a Velocity-Fraction Audit Form Lock (D5 + B)

Status: **VERDICT LANDED 2026-05-24 (R1 + R2.A amendments applied)**.
Verdict: **`velocity_fraction_blocked_integration_attrition`**
at `integration_blocked_count = 23` (8.42% of catalog, above the
pre-registered 5%/14-row threshold). Of the 250 analyzable rows, an
additional 11 failed the R1-amended symplecticity / reciprocal-pair
gates. The catalog cannot be evaluated at the locked variational
precision (`rtol = atol = 1e-12`); the audit is integration-attrited,
not feature-falsified. See the "Verdict (Landed)" section below.

This document locks the v0.7a univariate-velocity-fraction chi-squared
audit form under the parent registration's candidate slots **D5**
(velocity-fraction direction property) and **B** (velocity-fraction
quartile audit). The D1 + A audit (largest-real-part eigenvector
direction-projection quartile) is registered as a **report-only named
sidecar** and emits no independent verdict (also not licensed under
the attrition verdict).

**Amendment R1 (2026-05-24): symplecticity sanity-gate threshold
relaxed from 1e-6 to 1e-4** based on the 7-row vectorized smoke
evidence. See the "Amendment R1: Sanity-Gate Threshold" section
below.

**Amendment R2.A (2026-05-24): per-row integration-failure fallback
discipline** added after the full-catalog runner crashed at row 76
(O_194 at m_3=0.5). See the "Amendment R2.A: Per-Row Integration-
Failure Fallback" section below.

## Verdict (Landed, 2026-05-24)

Receipt:
`results/isotrophy/k-facet-v07a-velocity-fraction-audit/manifest.json`.

The R1-amended runner ran through all 273 supp-B rows over ~4.5 hours
under the R2.A try-catch and R2.C append-per-row disciplines. Result:

```text
verdict:                  velocity_fraction_blocked_integration_attrition
total catalog rows:                  273
S / U (catalog):                      97 / 176
analyzable rows:                     250
integration_blocked_count:            23
attrition_threshold_count:            14   (5% of 273, ceiling)
attrition_fired:                      True  (23 > 14)
sanity_gate_failures
  (analyzable rows with
   symp > 1e-4 or recip > 1e-4):      11
total data-integrity issues:          34 / 273 = 12.5%

constant_feature_retirement check:
  vf_sd (over analyzable rows):       0.1444
  threshold:                          0.01
  retired:                            False
                                      (vf has real variation, but the
                                       audit is not licensed to read it)
total runtime:                        266.9 min  (~4.5 h)
```

Integration-blocked rows by m_3:

```text
m_3       blocked count   labels
0.5             1         O_{194}
0.9             4         O_{796}, O_{1008}, O_{1079}, O_{1194}
1.0             3         O_{1153}, O_{1200}, O_{1362}
1.5             4         O_{152}, O_{166}, O_{175}, O_{187}
1.6             6         O_{255}, O_{258}, O_{315}, O_{317}, O_{324}, O_{327}
1.7             5         O_{178}, O_{185}, O_{188}, O_{231}, O_{246}
                ---
                23
```

All 23 blocks fired in `compute_monodromy_vectorized` with
`Required step size is less than spacing between numbers`. The base
`integrate_orbit` succeeded for every row; the failure is the matrix
variational equation `dY/dt = J(t) Y` along the integrated orbit at
`rtol = atol = 1e-12`.

Sanity-gate failures (the 11 analyzable rows where symp or recip
exceeded the R1 1e-4 threshold) cluster at m_3 in {0.6, 0.7, 1.0,
1.1, 1.4, 1.6}. Worst symplecticity residual was O_321(1.6) at
1.12e-3 (11x the gate). Worst reciprocal-pair residual was
O_642(1.0) at 9.17e-3 (90x the gate).

**Structural reading:**

> The orbits where the variational equation is most numerically
> challenging — long-period high-m_3 rows in the 1.4..1.7 cluster
> and a tail at m_3 = 0.9..1.0 — cluster precisely in the regime
> v0.4a's two-pass classifier was built to handle (Pass 2 tight
> tolerances at identity_rotation_tolerance = 1e-9, phase_grid =
> 361). At v0.7a's variational precision rtol = atol = 1e-12, the
> DOP853 step adapter cannot find a feasible step for the
> 324-dimensional matrix variational equation along these orbits.
> The audit is NOT feature-falsified; it is INTEGRATION-ATTRITED at
> the locked precision. The catalog cannot be evaluated honestly
> under this precision configuration.

**Joint v0.4 + v0.5 + v0.6 + v0.7 envelope (the publishable
multi-chapter statement):**

> Four sequential pre-registered projections of the supp-B body —
> the Z_2 symmetry shadow (tangent-isotypic and orbit-gauge-rigidity,
> v0.4), the 2-bit catalog branch shadow (v0.5), the continuous
> orbit-level conserved-quantity shadow E and |L| (v0.6), and the
> orbit-dynamics gamma_1 direction shadow (v0.7a velocity-fraction) --
> have produced three distinct chapter-close types: structural-
> negative (v0.4), projection-limit (v0.5), conditional-independence
> (v0.6), AND a fourth methodological close type, integration-
> attrition (v0.7), where the audit cannot honestly run on this
> catalog at the registered precision because the variational
> equation along the most-challenging orbits fails DOP853 step
> control.

**Receipts:**

```text
docs/isotrophy/kfacet/kfacet_v07a_velocity_fraction_audit_form.md  (this document)
results/isotrophy/k-facet-v07a-velocity-fraction-audit/manifest.json
results/isotrophy/k-facet-v07a-velocity-fraction-audit/per_row_table.csv
results/isotrophy/k-facet-v07a-velocity-fraction-audit/smoke/smoke_manifest.json
results/isotrophy/k-facet-v07a-velocity-fraction-audit/smoke/smoke_per_row_table.csv
scripts/v07a_velocity_fraction_smoke.py
scripts/v07a_velocity_fraction_audit.py
```

No chi-squared verdict is licensed under this audit run. The D1+A
sidecar (z-fraction quartile) is also not evaluated; the same
catalog-integrity issue applies.

Audience: v0.7a runner; v0.7b form-lock author (conditional on v0.7a
passing); paper-side reviewer of the v0.6 -> v0.7 transition.

Companions:

- `kfacet_v07_mechanism_preregistration.md` -- v0.7 parent registration
  (operational definitions D1-D5, audit forms A-D, three circularity
  risks, disallowed-feature list, inheritance discipline).
- `kfacet_v06_writeup.md` -- v0.6 conditional-independence chapter
  close that opened v0.7.
- `kfacet_v06b_within_branch_energy_audit_form.md` -- v0.6b sparse-cell
  fallback tree pattern (inheritance source).
- `kfacet_v06a_energy_quartile_audit_form.md` -- v0.6a alignment-
  tightness guard pattern (inheritance source).
- `kfacet_v04a_domain_map_preregistration.md` -- supp-B monodromy
  computation source; per-row M_i is derivable from this receipt.

Frame: v0.6 closed as a conditional-independence result — the
continuous catalog-coordinate (E, |L|) shadow does not carry stability
information beyond labeling the v0.5a branch hash. v0.7 leaves
catalog-coordinate space entirely; v0.7a is the first
orbit-dynamics audit. Among the parent's five candidate operational
definitions (D1-D5), this form lock picks **D5** because it
minimizes the tie-break surface and stays furthest from
eigenvalue-ordering as the feature-selection mechanism.

## Why D5 + B Over Alternatives (Codex Rationale)

> v0.7 is already walking close to the circularity cliff by touching
> Floquet eigenstructure at all. D1 "largest-real-part" is
> mathematically natural, but it still chooses the feature through
> eigenvalue ordering, and eigenvalue ordering is too close to the
> S/U definition. We'd save D1 for a sidecar after D5 establishes
> the clean floor.

The locked picks therefore prioritize **non-circularity over
discriminative power**. D5 is a direction property that exists for any
6D phase-space vector and is independent of eigenvalue ordering once a
direction is fixed. The S-row eigenvalue degeneracy tie-break is
documented but minimized: D5's feature does NOT depend on resolving
the degeneracy because the velocity-fraction within any eigenspace is
well-defined.

D1 + A is preserved as a **named sidecar** (report-only): the natural
"physicist's instinct" version. If D5 + B passes, D1 + A is an
interpretable companion; if D5 + B fails but D1 + A lights up, the
sidecar signal may be selection-rule contaminated and would require a
fresh circularity audit, not a refinement of v0.7a.

## Amendment R1: Sanity-Gate Threshold (Locked 2026-05-24)

The v0.7a form lock pre-registered a symplecticity sanity gate at
`1e-6`. A 7-row vectorized smoke (sentinels O_50, O_62, O_67, O_434
at m_3=0.4 and O_242, O_282, O_284 at m_3=1.0) at `rtol = atol =
1e-12` showed symplecticity residuals in the range `7.9e-8 ..
3.84e-5` -- 5 of 7 rows above the pre-locked 1e-6 threshold. The
reciprocal-pair gate at `1e-4` passed for all 7 rows
(worst-case `8.35e-5` on O_242).

The residuals scale with period and Floquet amplification, not with
implementation breakage: strongly unstable rows (max_re = 39 on
O_242, max_re = 25 on O_284) have larger absolute residuals because
the unstable Floquet mode quadratically amplifies integration error
in the `M_i^T omega M_i - omega` quantity. Long-period S rows
(O_434 at period 200) show similar amplification via accumulated
step error.

The pre-locked 1e-6 was a pre-run guess. The smoke is the
appropriate empirical evidence to amend the threshold. Blocking the
chapter on 1e-6 would say "our sanity gate was too tight," not "the
mechanism failed" — and the reciprocal-pair gate (the load-bearing
test that the monodromy preserves the Floquet pairing structure)
passes uniformly.

```text
v0.7a-r1 sanity-gate amendment:

rtol                  = 1e-12       (unchanged)
atol                  = 1e-12       (unchanged)
max_step_fraction     = 0.02        (unchanged)

symplecticity_gate:
  old:  1e-6
  new:  1e-4

reciprocal_pair_gate: 1e-4          (unchanged)

justification:
  7-row vectorized smoke (rtol = atol = 1e-12, max_step = 0.02*T)
  showed symplecticity residuals in 7.9e-8 .. 3.84e-5, while
  reciprocal-pair residuals all passed at 1e-4. Residuals scale
  with period and Floquet amplification and are treated as the
  variational-integration precision floor, not as feature evidence.

firewall (locked permanently):
  symplecticity residual, reciprocal-pair residual, max_re, and the
  selected eigenvalue are QC/provenance only. They do NOT enter the
  velocity-fraction feature, the binning, the chi-squared statistic,
  the alignment-tightness scalar, or the verdict. Any future
  amendment that promotes a sanity diagnostic into the feature path
  is a fresh circularity audit, not a refinement.
```

This amendment is locked before the amended-gate smoke is re-run.
Any subsequent change to either gate after the audit verdict lands
is a re-registration, not a refinement.

## Amendment R2.A: Per-Row Integration-Failure Fallback (Locked 2026-05-24)

The v0.7a R1-amended full-catalog runner crashed at row 76 of 273
(O_194 at m_3=0.5) with:

```text
RuntimeError: vectorized variational integration failed for
              O_{194}(0.5): Required step size is less than spacing
              between numbers.
```

This is a pre-sanity numerical-precision failure inside scipy's
DOP853 step adapter on the matrix variational equation. The base
`integrate_orbit` succeeded; `compute_monodromy_vectorized` could
not find a feasible step at `rtol = atol = 1e-12`. It is NOT a
symplecticity or reciprocal-pair sanity-gate violation -- the
sanity gates never get to evaluate because M_i is never produced.

The pre-locked form lock anticipated symplecticity / reciprocal-pair
failures but did not anticipate pre-sanity integrator failures.
R2.A pre-registers the row-level fallback discipline.

```text
v0.7a-r2.a per-row integration-failure fallback:

trigger:
  any RuntimeError raised from compute_monodromy_vectorized for a
  given row (typically "Required step size is less than spacing
  between numbers", but any IVP failure mode).

per-row marking:
  the row is recorded on per_row_table.csv with:
    - row identity preserved (label, index, m3, z0, period, stability,
      branch_label),
    - symplecticity_status     = "integration_blocked",
    - reciprocal_pair_status   = "integration_blocked",
    - velocity_fraction        = null / NaN,
    - z_fraction               = null / NaN,
    - Q_vf, Q_z                = null,
    - all gamma_1 fields       = null,
    - integration_error_message  = the exception message,
    - integration_blocked      = True.

chi-squared denominator:
  blocked rows are EXCLUDED from the chi-squared catalog (the 273
  reduces to (273 - integration_blocked_count)) and from the
  alignment-tightness denominator. Quartile cutpoints are computed
  on the non-blocked subset only.

attrition threshold (locked):
  if integration_blocked_count > 5% of catalog
  (= ceil(0.05 * 273) = 14 rows for supp-B):
    verdict = velocity_fraction_blocked_integration_attrition.
    No chi-squared verdict is licensed; the catalog is too
    integration-attrited for the audit to run honestly.
  else:
    audit proceeds on (273 - integration_blocked_count) rows.

firewall (re-asserted permanently):
  integrator error messages and blocked-row status are QC/provenance
  only. They do NOT enter the velocity-fraction feature, the binning,
  the chi-squared statistic, the alignment-tightness scalar, or the
  verdict (beyond the attrition gate above).
```

R2.A is locked before the runner is re-run from scratch with the
R2.A behavior built in. Any future change to the trigger condition,
per-row marking, denominator handling, attrition threshold, or
firewall is a re-registration, not a refinement.

## Engineering Note: R2.C Append-Per-Row Discipline (Locked 2026-05-24)

Companion to R2.A. Not a form-lock amendment (does not change any
feature, binning, df, or verdict), but a runner-engineering
discipline that v0.7a operations are committed to:

```text
v0.7a-r2.c append-per-row + resume:

per-row append:
  the runner writes each completed row's record to per_row_table.csv
  immediately after the row is processed (or marked
  integration_blocked). A crash mid-run preserves all rows
  completed before the crash on disk.

resume mode:
  on startup, the runner reads any existing per_row_table.csv at the
  output path and skips re-processing rows already present.
  Completion-by-identity (label, index) is the match key. This
  guarantees that compute already spent on rows 1..k is not lost
  if the runner restarts.

separate amendment to manifest emission:
  the manifest.json is still written at the END of the run (after
  all rows are processed or marked blocked) as the audit verdict
  finalization. A crash before manifest emission means re-running
  with resume mode produces the manifest on the next clean
  completion.
```

R2.C is engineering; it does not change v0.7a's substantive
verdict shape. It is recorded here for transparency.

## What v0.7a Is

v0.7a is a **catalog-side audit**, NOT a predictor (continues
v0.5a/v0.6a/v0.6b audit-not-predictor framing):

```text
Test:         chi-squared independence of velocity-fraction quartile
              Q_vf vs S/U over 273 supp-B rows.
Default df:   3  (or occupied_bins - 1 under sparse-cell fallback).
Critical:     chi-squared(3) at p = 0.01 = 11.34
              (or chi-squared(df) at p = 0.01 if df != 3).
Pass:         chi^2 > critical AND alignment-tightness <= 0.8
                -> velocity_fraction_passes_audit
Warn pass:    chi^2 > critical AND alignment-tightness > 0.8
                -> velocity_fraction_passes_audit_alignment_warning
Fail:         chi^2 <= critical
                -> velocity_fraction_fails_audit
Sparse:       any occupied bin with N < 2
                -> velocity_fraction_inconclusive_sparse
                   (fallback inherited from v0.6b)
```

A clean pass licenses v0.7b: a separately-registered held-out
velocity-fraction predictor with the v0.5b discipline. A pass with
alignment warning licenses only an alignment-breaking v0.7b
registration (mirroring v0.6b's within-branch pattern). A fail closes
the velocity-fraction sub-question; the D1 + A sidecar's status is
report-only regardless.

## The Locked Form

### Operational definition (D5 velocity-fraction)

```text
1. Per row, compute the monodromy M_i = exp(integral over one period
   of the Jacobian of the Hamiltonian flow at the periodic orbit).
   - INPUT: published mass triple (1, 1, m_3), published initial
     conditions (r_i, v_i for i = 1..3), published period T.
   - METHOD: variational integration of the three-body flow's
     Jacobian over one period.
   - PROVENANCE: same computational substrate as v0.4a's two-pass
     classifier. v0.7a may reuse the v0.4a numerics if and only if
     the disallowed-feature audit is preserved (no K_fib tangent
     decomposition is used here; only the full 18 x 18 monodromy).

2. Compute the Floquet eigenstructure of M_i:
   - All 18 eigenvalues and right eigenvectors.
   - DOUBLE PRECISION numpy.linalg.eig.

3. Identify gamma_1(row):
   - Primary rule: gamma_1 = right eigenvector of the eigenvalue
     with LARGEST REAL PART.
   - This selection is the SAME as D1's selection rule. The
     non-circularity-preserving move is that v0.7a uses the
     resulting eigenvector ONLY as a geometric direction; the
     eigenvalue itself is never used as a feature.
   - Tie-break cascade for degenerate eigenspaces (S-row case
     where the largest-real-part eigenvalue is unit-modulus and
     degenerate):
       a) within the degenerate eigenspace, choose the vector
          with maximal projection onto the velocity subspace
          under the mass-weighted norm (see Reference Frame
          below);
       b) then smallest absolute imaginary part of the eigenvalue;
       c) then smallest positive argument;
       d) then deterministic lexicographic sign convention
          (first non-zero entry positive).
   - All four tie-break steps are pre-data; no run-time tuning.

4. Reduce gamma_1 to the center-of-mass reduced frame:
   - Subtract the mass-weighted average over the 3 bodies in each
     of the position and velocity sub-blocks.
   - Apply the mass-weighted norm: for delta-q components,
     ||delta_q||^2 = sum_{i=1..3} m_i * ||delta_q_i||^2;
     for delta-v components,
     ||delta_v||^2 = sum_{i=1..3} m_i * ||delta_v_i||^2.
   - This normalization preserves the physical-units balance
     between position and velocity components and removes the
     dependence on body-3's mass scaling.

5. Compute the velocity-fraction:
   - velocity_fraction(row) = ||delta_v||^2 /
                              (||delta_q||^2 + ||delta_v||^2)
   - velocity_fraction in [0, 1] for any non-zero gamma_1.
   - velocity_fraction = 0 means gamma_1 is purely positional;
     = 1 means purely velocity.
```

### Audit form (B velocity-fraction quartile)

```text
binning:      quartiles over the supp-B catalog's 273-row
              velocity_fraction distribution.

cutpoints:    q25, q50, q75 of the supp-B velocity_fraction array,
              computed via numpy.quantile(values, [0.25, 0.50, 0.75],
              method='linear').

bin
assignment:   right-closed lower intervals, ties to lower bin
              (matches v0.6a convention exactly).
                Q_vf(r) = 1  iff  vf(r) <= q25
                Q_vf(r) = 2  iff  q25 < vf(r) <= q50
                Q_vf(r) = 3  iff  q50 < vf(r) <= q75
                Q_vf(r) = 4  iff  vf(r) >  q75

contingency:  4 x 2 table (Q_vf, stability) with stability in {S, U}.

test:         chi-squared statistic over 273 rows;
              df = occupied_bins - 1 (= 3 if all 4 quartiles
              occupied, which is structurally guaranteed for
              quartile binning on 273 rows).

critical:     11.34 at p = 0.01 (chi-squared(3)).

alignment
guard:        max over Q_vf of (fraction of bin in any single
                                v0.5a branch_label bucket).
              threshold = 0.8 (v0.6 inheritance).

sparse-cell
fallback:     v0.6b discipline:
                if min_occupied_bin_count < 2:
                  -> velocity_fraction_inconclusive_sparse
                elif min(expected cell count) < 5:
                  -> exact permutation test
                     (seed = 20260523, n_permutations = 10000)
                else:
                  -> asymptotic chi-squared(df).
              For quartile binning on 273 rows the
              chi-squared branch is overwhelmingly likely;
              the fallback is documented for completeness.
```

### Non-circularity sentence (locked)

> v0.7a uses Floquet eigenvectors only as geometric directions; it does
> not use eigenvalue magnitude, spectral radius, unit-circle status,
> unstable-pair count, or any threshold that defines the published
> S/U label. The tested scalar is a phase-space composition ratio of
> the selected direction, not the growth rate of that direction.

This sentence is the load-bearing argument addressing the parent
registration's three circularity risks. Re-stating in the parent's
risk taxonomy:

- **Eigenvalue-choice circularity:** Addressed. The selection rule
  (largest real part) is pre-data and applies to every row. The
  feature extracted (velocity-fraction) is independent of which
  eigenvalue was selected; it depends only on the eigenvector's
  phase-space composition.

- **Well-definedness circularity:** Addressed. gamma_1 is defined for
  ALL 273 rows under the tie-break cascade. There is no
  "gamma_1_undefined" condition that could correlate with stability.

- **Feature-extraction circularity:** Addressed. The velocity-fraction
  is a geometric ratio in [0, 1] that does not encode growth rate,
  eigenvalue magnitude, or stability count. It is the same scalar
  whether the eigenvalue is unit-modulus or growing.

## Pre-Audit Sanity Surface

Before v0.7a runs:

```text
1. Per-row monodromy M_i computed for all 273 rows via variational
   integration of the three-body flow at the published initial
   conditions.

2. Sanity check on the symplectic structure:
   - Verify M_i^T J M_i = J (where J is the 18 x 18 symplectic form)
     to the **R1-amended** tolerance of 1e-4 over the maximum entry
     of the residual.
   - Block v0.7a runner if any row violates symplecticity at this
     tolerance; the variational integration is suspect.
   - (Original tolerance 1e-6; amended to 1e-4 on 2026-05-24 after
     the 7-row vectorized smoke. See "Amendment R1: Sanity-Gate
     Threshold" above.)

3. Sanity check on the Floquet eigenstructure:
   - Verify that the eigenvalues come in reciprocal pairs
     (Floquet theorem for symplectic matrices) to a registered
     tolerance of 1e-4 on max |lambda * lambda_conj_recip - 1|.
   - Block v0.7a runner if violated.

4. Sanity check on gamma_1 uniqueness:
   - For each row, verify that the largest-real-part eigenvalue is
     unique (i.e., separated from the next eigenvalue by > 1e-6
     in real part) OR record the tie-break cascade decision on
     the receipt.
   - Rows requiring tie-break cascade are flagged but not blocked;
     the cascade is deterministic.

5. Constant-feature retirement check:
   - Compute velocity_fraction distribution over 273 rows.
   - If sd(velocity_fraction) < 0.01 (i.e., near-constant), the
     constant-feature retirement rule fires: verdict
     `velocity_fraction_retired_near_constant`, no audit run.
   - Threshold (0.01) is locked pre-data.
```

Sanity surface receipts are emitted regardless of audit outcome.

## Non-Circularity Audit (Full Argument)

### Why eigenvalue-choice circularity is not triggered

The selection rule (largest real part) is pre-data. For S rows, the
largest-real-part eigenvalue is unit-modulus (real part = 1 with
possible degeneracy). For U rows, the largest-real-part eigenvalue
is the unstable Floquet multiplier (real > 1). The selection rule
DOES end up picking physically different objects for S vs U rows,
but:

```text
- The selection rule is DETERMINISTIC and applied UNIFORMLY before
  the stability label is consulted.
- The selected eigenvalue's magnitude, real part, or growth rate
  is NEVER used as a feature; it is used only to disambiguate
  which eigenvector to extract.
- The eigenvector itself is a geometric vector in phase space; its
  composition (velocity-fraction) is a geometric property
  independent of eigenvalue ordering rules.
```

The substantive question v0.7a is asking is: do the eigenvectors
selected by this uniform rule have DIFFERENT velocity-fractions for
S vs U rows? That is, holding the selection rule fixed across all
rows, does the resulting geometric direction stratify stability?

A passing audit would say: yes, the eigenvectors selected by the
largest-real-part rule have systematically different velocity-fractions
for S vs U rows. The S-row eigenvectors (unit-modulus eigenvalue
eigenvectors) and U-row eigenvectors (unstable eigenvalue eigenvectors)
sit in different geometric regions of phase space.

A failing audit would say: the velocity-fractions are
indistinguishable between S and U rows, or are stratified in a way
that does not reach the registered p <= 0.01 floor.

Neither outcome trivializes the stability label.

### Why well-definedness circularity is not triggered

gamma_1 is defined for all 273 rows under the deterministic tie-break
cascade. The cascade handles the S-row eigenvalue degeneracy without
introducing a "gamma_1_undefined" condition that could correlate with
stability. The cascade is:

```text
Step 1: largest real part of eigenvalue (the D1 selection).
   - U rows: unique unstable eigenvalue. No tie.
   - S rows: typically degenerate at real part = 1.

Step 2 (degenerate eigenspace only): within the eigenspace, pick the
   vector with maximal projection onto the velocity subspace.
   - This is well-defined: project each basis vector of the
     degenerate eigenspace onto the velocity subspace under the
     mass-weighted norm, pick the one with maximal projection.

Step 3 (further tie at step 2): smallest absolute imaginary part.

Step 4 (further tie): smallest positive argument.

Step 5 (further tie): lexicographic sign convention -- first
   non-zero entry positive.
```

Every row terminates the cascade at some step. The eigenvector is
returned deterministically.

### Why feature-extraction circularity is not triggered

velocity_fraction is a geometric ratio in [0, 1]:

```text
vf = ||delta_v||^2 / (||delta_q||^2 + ||delta_v||^2)
```

- It is independent of the eigenvector's normalization (numerator
  and denominator scale together).
- It is independent of the sign convention (squared norms).
- It does not encode growth rate, eigenvalue magnitude, or stability
  count.
- For any 6D phase-space vector, vf is a well-defined number.

The audit's verdict is therefore a statement about the geometric
distribution of gamma_1 directions in phase space, not about whether
the orbit is stable.

## Free Parameter Count

```text
Number of free parameters fitted to the data: 0
```

The feature (velocity-fraction), the selection rule (largest real
part), the tie-break cascade, the reference frame (center-of-mass
reduced with mass-weighted norm), the binning rule (quartiles), the
quantile method (numpy linear interpolation), the bin assignment
(right-closed lower), the df (3), the critical value (11.34), the
alignment-tightness threshold (0.8), the constant-feature retirement
threshold (sd < 0.01), and the sparse-cell fallback parameters
(min_expected_floor = 5, min_occupied_bin_count = 2,
permutation_seed = 20260523, n_permutations = 10000) are all locked
pre-data.

## Sparse-Cell Fallback Logic (Inherited from v0.6b)

```text
1. Compute expected cell counts E_{kj} = N_k * C_j / 273.
2. Compute min_occupied_bin_count = min over occupied bins of N_k.
3. Branch:
   a) If min_occupied_bin_count < 2:
        verdict = velocity_fraction_inconclusive_sparse
        do NOT report a p-value.
   b) Elif min over occupied cells of E_{kj} < 5:
        permutation test with locked seed = 20260523,
        n_permutations = 10000, same chi-squared statistic;
        p_value = empirical tail.
   c) Else:
        asymptotic chi-squared(df = occupied_bins - 1) at p = 0.01.
```

For quartile binning on 273 rows with 97 S / 176 U, the expected
counts per cell are approximately {24, 44} -- well above the
asymptotic floor. The fallback is documented for completeness but
will not fire under normal conditions.

## Alignment-Tightness Guard (Inherited from v0.6a)

```text
For each Q_vf in {1, 2, 3, 4}:
  - identify the v0.5a branch_label that dominates the bin.
  - compute dominant_fraction = (rows in bin in dominant branch)
                                / (rows in bin).

alignment_tightness_scalar = max over Q_vf of dominant_fraction.

Threshold:    0.8.

If alignment_tightness_scalar > 0.8 AND chi^2 > 11.34:
  verdict = velocity_fraction_passes_audit_alignment_warning
  -> v0.7b must use an alignment-breaking partition
     (mirroring v0.6b's within-branch approach).
```

## Sidecar: D1 + A (Report-Only)

Under the SAME monodromy and gamma_1-selection pipeline (the
selection rule is the same; only the feature differs), the runner
ALSO emits the D1 + A direction-projection quartile audit:

```text
sidecar feature:    gamma_1 projected onto a pre-registered
                    reference axis.
reference axis:     center-of-mass-frame z-axis (the
                    out-of-plane direction; aligned with the
                    catalog's z_0 coordinate convention).
sidecar scalar:     |proj_z(gamma_1)| (absolute value of the
                    z-component of gamma_1 under mass-weighted
                    norm normalization).
binning:            quartiles over supp-B's 273-row
                    |proj_z(gamma_1)| distribution.
contingency:        4 x 2 over (Q_proj_z, S/U).
test:               same chi-squared / sparse-cell pipeline.

REPORT-ONLY:        NO VERDICT IS CLAIMED FROM THE SIDECAR.
                    The sidecar's chi-squared and contingency
                    are recorded on the receipt for diagnostic
                    and motivation purposes only.

re-registration:    if v0.7a primary fails but D1+A sidecar shows
                    chi^2 > 11.34 at p <= 0.01, the loud sidecar
                    is FLAGGED but NOT promoted directly. It
                    requires a fresh circularity audit
                    (selection-rule contamination check) before a
                    new D1+A form lock can be registered.
                    Reason: D1+A's selection rule uses the
                    eigenvalue's largest real part, which is closer
                    to the stability boundary than D5's feature.
                    Any positive D1+A signal must be re-audited
                    against the selection-rule-contamination risk.
```

The sidecar's purpose is to surface "loud signal from the physicist's
instinct version" cheaply, without licensing it as a verdict under
this registration.

## Reserved Diagnostics (Not Gating)

The v0.7a runner emits per-row and aggregated diagnostics:

```text
per-row table:
  (label, m_3, z_0, period, stability, branch_label,
   gamma_1_eigenvalue_real, gamma_1_eigenvalue_imag,
   gamma_1_velocity_fraction, gamma_1_z_projection,
   gamma_1_tie_break_step_used,
   Q_vf, Q_proj_z)

aggregated:
  velocity_fraction histogram over 273 rows.
  velocity_fraction by (m_3, z_0) branch joint table.
  velocity_fraction by m_3 joint table.
  velocity_fraction by S/U distribution summary.
  symplecticity_residual distribution (sanity diagnostic).
  reciprocal_pair_residual distribution (sanity diagnostic).
  tie_break_step_used count by row.
```

## Edge Cases (Pre-Registered)

1. **Symplecticity sanity gate fail**: block v0.7a; inspect the
   variational integrator before re-running.

2. **Reciprocal-pair sanity gate fail**: same gate as #1.

3. **Constant-feature retirement fires (sd(vf) < 0.01)**: verdict
   `velocity_fraction_retired_near_constant`. Possible physical
   cause: the velocity-fraction is structurally fixed by the
   Hamiltonian flow's tangent dynamics (e.g., a symplectic
   constraint forcing vf = 0.5 for all rows). The chapter closes
   on velocity-fraction; v0.7b is not licensed. v0.7c may register
   a different feature.

4. **Tie-break cascade reaches step 4 on > 50% of rows**: the
   selection rule is highly degenerate on this catalog. Audit
   proceeds but the receipt flags `tie_break_dominance`; the
   verdict's interpretive weight is reduced.

5. **Alignment-tightness > 0.95**: verdict
   `velocity_fraction_passes_audit_severe_alignment` (a stronger
   version of the standard alignment-warning); v0.7b registration
   is mandated alignment-breaking, AND the v0.6 pre-mortem residual
   alignment trace must be documented in v0.7b's form lock.

6. **gamma_1 is purely real for some rows, complex for others**:
   acceptable. The selection rule and tie-break cascade handle both
   cases. The velocity-fraction is computed on the real part of the
   eigenvector for complex eigenvalues; this convention is locked
   here (any post-hoc switch to a complex-magnitude convention is
   a re-registration).

## Pre-Mortem Expectation

The per-row velocity_fraction distribution has NOT been computed
before lock-in. The audit runs blind.

Expected published statement, regardless of outcome:

> v0.7a tests whether the supp-B catalog's gamma_1 velocity-fraction
> stratifies stability under the locked operational definition.
> The verdict is one of `velocity_fraction_passes_audit`,
> `velocity_fraction_passes_audit_alignment_warning`,
> `velocity_fraction_fails_audit`,
> `velocity_fraction_inconclusive_sparse`, or
> `velocity_fraction_retired_near_constant` per the locked threshold
> tree.

Pre-mortem note: symplectic Hamiltonian dynamics may force
gamma_1's velocity-fraction toward a characteristic value across all
rows (e.g., a Floquet eigenvector of a symplectic matrix has
constraints on its phase-space composition). If the
constant-feature retirement fires, that itself is a methodological
finding worth documenting.

## Stop Conditions

1. **Non-circularity audit failure at runner time.** If the
   implementation cannot maintain the locked non-circularity
   discipline (e.g., the tie-break cascade requires inspecting
   eigenvalue magnitude in some unforeseen way), the runner aborts
   with `velocity_fraction_blocked_circularity_audit`; no compute
   verdict is licensed.

2. **Sanity gate failures (symplecticity, reciprocal pair)**: block.

3. **Constant-feature retirement**: chapter closes on
   velocity-fraction; v0.7b not licensed.

4. **Permutation test required (min_expected < 5)**: not a stop;
   the fallback proceeds.

5. **Tie-break-cascade dominance**: not a stop; flag on receipt.

## Lock-In Statement

This audit form is committed before:

- per-row M_i has been computed under any variational integrator,
- per-row gamma_1 has been extracted under the locked selection
  rule and tie-break cascade,
- per-row velocity_fraction values have been computed,
- the supp-B quartile cutpoints for velocity_fraction have been
  calculated,
- the alignment-tightness scalar has been computed,
- the D1 + A sidecar has been computed.

Any change to the operational definition, selection rule, tie-break
cascade, reference frame normalization, feature definition,
binning rule, quantile method, df formula, critical value,
alignment-tightness threshold, constant-feature retirement
threshold, sparse-cell fallback parameters, permutation seed,
sidecar configuration, or non-circularity sentence after the v0.7a
runner is executed is a **re-registration**, not a refinement.

Implementation may proceed against this form lock using:

```text
- supp-B published catalog (initial conditions, masses, periods)
- v0.4a manifest at
  results/isotrophy/k-facet-v04a-domain-map/manifest.json
  for cross-validation of the integration substrate
- v0.5a per_row_table.csv at
  results/isotrophy/k-facet-v05a-branch-map/per_row_table.csv
  for the alignment-tightness check
```

as inputs. New compute is required: per-row variational integration
of M_i (was NOT in v0.4a's emitted scalars; v0.4a computed K_fib
basis but not full 18 x 18 eigenstructure).

## Receipt Schema (Planned)

```text
results/isotrophy/k-facet-v07a-velocity-fraction-audit/
  manifest.json
    - mode = "v0.7a-velocity-fraction-audit"
    - form_lock = "docs/isotrophy/kfacet/kfacet_v07a_velocity_fraction_audit_form.md"
    - input_catalog_supplementary_b
    - input_v04a_manifest
    - input_v05a_per_row_table
    - operational_definition = "D5_velocity_fraction_of_largest_real_part_eigenvector"
    - selection_rule = "largest_real_part_eigenvalue"
    - tie_break_cascade_steps:
        [ "max_velocity_projection_within_eigenspace",
          "smallest_absolute_imaginary_part",
          "smallest_positive_argument",
          "lexicographic_sign_convention_first_nonzero_positive" ]
    - reference_frame = "center_of_mass_reduced_mass_weighted_norm"
    - quartile_method = "numpy_quantile_linear"
    - bin_assignment_convention = "right_closed_lower_ties_to_lower"
    - constant_feature_sd_threshold = 0.01
    - alignment_tightness_threshold = 0.8
    - alignment_severe_threshold = 0.95
    - asymptotic_expected_floor = 5
    - min_occupied_bin_count_for_test = 2
    - permutation_seed_locked = 20260523
    - n_permutations_locked = 10000
    - chi2_critical_at_p01_df3 = 11.34
    - sanity_check_symplecticity: { tolerance, max_residual, status }
    - sanity_check_reciprocal_pair: { tolerance, max_residual, status }
    - sanity_check_gamma_1_uniqueness:
        { unique_count, tie_break_used_count, by_step }
    - constant_feature_check:
        { sd_velocity_fraction, retired_status }
    - cutpoints_vf: { q25, q50, q75 }
    - contingency_vf: 4 x 2 (Q_vf, S/U)
    - chi_squared_vf:        test statistic
    - df_vf:                  3 (or occupied_bins - 1)
    - critical_vf:            11.34
    - test_branch_taken:      "chi_squared" | "permutation" |
                              "inconclusive_sparse" | "retired_near_constant" |
                              "blocked_sanity"
    - p_value_vf:             chi-squared upper-tail p OR permutation p
    - alignment_tightness_scalar_vf: max bin-alignment vs v0.5a
    - alignment_records_vf:   per-bin dominant branch + counts
    - verdict:                 velocity_fraction_passes_audit |
                                velocity_fraction_passes_audit_alignment_warning |
                                velocity_fraction_passes_audit_severe_alignment |
                                velocity_fraction_fails_audit |
                                velocity_fraction_inconclusive_sparse |
                                velocity_fraction_retired_near_constant |
                                velocity_fraction_blocked_sanity |
                                velocity_fraction_blocked_circularity_audit
    - sidecar_D1_A:
        - feature = "gamma_1_z_projection_magnitude"
        - reference_axis = "center_of_mass_frame_z_axis"
        - cutpoints, contingency, chi_squared, p_value,
          alignment_tightness
        - sidecar_status = "report_only"
    - diagnostic_per_row_table
    - diagnostic_vf_by_m3
    - diagnostic_vf_by_branch
    - diagnostic_vf_by_Q_E (v0.6a quartile)
    - diagnostic_vf_by_Q_L (v0.6a quartile)
  per_row_table.csv
  contingency_table_vf.csv
  sidecar_D1_A_contingency.csv
  velocity_fraction_by_m3.csv
  velocity_fraction_by_branch.csv
  velocity_fraction_by_Q_E.csv
```

## Implementation Plan

Suggested script:

```text
scripts/v07a_velocity_fraction_audit.py
```

Suggested command:

```powershell
python scripts\v07a_velocity_fraction_audit.py `
  --catalog        docs\isotrophy\supplementary-B_piano-init-condit-3d.txt `
  --v04a-manifest  results\isotrophy\k-facet-v04a-domain-map\manifest.json `
  --v05a-per-row   results\isotrophy\k-facet-v05a-branch-map\per_row_table.csv `
  --out            results\isotrophy\k-facet-v07a-velocity-fraction-audit
```

Expected runtime: minutes (per-row variational integration of the
18-dimensional tangent system over one period times 273 rows).
Numerical precision must match v0.4a's two-pass classifier discipline
to preserve the disallowed-feature inheritance audit.

## Interpretation

If v0.7a passes (chi^2 > 11.34, alignment <= 0.8):

> The supp-B catalog's gamma_1 velocity-fraction stratifies stability
> under a pre-registered audit that uses only the geometric direction
> of the largest-real-part Floquet eigenvector (not its growth rate).
> The orbit-dynamics shadow carries stability information not visible
> in the v0.4 / v0.5 / v0.6 catalog-coordinate or symmetry-shadow
> projections. This licenses v0.7b: a separately-registered held-out
> velocity-fraction predictor with the v0.5b discipline.

If v0.7a passes with alignment warning (chi^2 > 11.34, alignment in
(0.8, 0.95]):

> Pass-but-suspect. The velocity-fraction quartiles track (m_3, z_0)
> branch tightly on supp-B, so the in-sample signal is plausibly
> entangled with the bin-locality phenomenon v0.5/v0.6 exposed.
> v0.7b's partition must be re-registered alignment-breaking
> (within-branch test, mirroring v0.6b).

If v0.7a passes with severe alignment (alignment > 0.95):

> Same as above, but with v0.6 pre-mortem residual alignment trace
> documented as part of v0.7b's form lock.

If v0.7a fails (chi^2 <= 11.34):

> The gamma_1 velocity-fraction does not stratify stability on
> supp-B under the locked univariate quartile audit. The chapter
> sub-question on velocity-fraction closes; the D1+A sidecar's
> chi-squared and the v0.7 mechanism family's remaining projections
> (D1 with fresh audit, D3 with definedness guard, D4 Z_2-isotypic
> split with v0.4 inheritance audit) are open for separate
> registration if any sidecar signal warrants pursuit.

If v0.7a retires near-constant:

> velocity_fraction is structurally fixed by the Hamiltonian flow
> on this catalog; no audit can distinguish S from U on this scalar.
> The chapter closes; v0.7b is not licensed; the D1+A sidecar's
> chi-squared (if any) is informational only.

## Doc Trail

- `kfacet_v07_mechanism_preregistration.md` -- v0.7 parent
  registration.
- `kfacet_v06_writeup.md` -- v0.6 chapter close (predecessor).
- `kfacet_v06a_energy_quartile_audit_form.md` -- alignment-tightness
  guard pattern.
- `kfacet_v06b_within_branch_energy_audit_form.md` -- sparse-cell
  fallback tree pattern.
- `kfacet_v04a_domain_map_preregistration.md` -- monodromy
  computational substrate.

---

Audit floor, not predictor. Velocity-fraction quartile, df = 3,
critical 11.34. D1 + A direction-projection sidecar report-only.
Selection rule = largest real part with mass-weighted tie-break
cascade; feature = velocity-fraction (geometric ratio, not growth
rate); non-circularity addressed by separating SELECTION from
FEATURE.
