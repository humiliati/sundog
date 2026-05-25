# v0.8 Mechanism Preregistration: Floquet Direction-Purity

Status: **REGISTERED 2026-05-24** (parent; no compute yet). This document
locks the v0.8 body / projection / observable and the non-circularity
discipline. Form locks for v0.8a (audit: purity-binned audit or monotone
purity predictor) and v0.8b (held-out predictor with attrition carried as
domain restriction) are separately registered children that pin specific
binning, df, falsifier, and held-out partition choices.

Audience: v0.8a/v0.8b form-lock author; paper-side reviewer of the
v0.7 -> v0.8 transition (qualified-positive close + direction-purity
mechanism open).

Companions:

- `kfacet_v07_writeup.md` -- v0.7 chapter close (qualified-positive
  on restricted domain) that opened v0.8.
- `kfacet_v07a_velocity_fraction_audit_form.md` -- v0.7a parent audit;
  source of the velocity-fraction operational definition and the
  locked non-circularity sentence (carries forward unchanged).
- `kfacet_v07a_prime_restricted_scope_form.md` -- v0.7a' restricted-
  scope confirmation; source of the analyzable subset (250 rows) and
  the U-shaped direction-purity signature.
- `kfacet_v05b_branch_predictor_form.md` -- v0.5b held-out predictor
  pattern; inheritance source for v0.8b's discipline.

Frame: v0.7a' produced a statistically-significant (chi^2 = 16.43,
p = 9.3e-4) branch-independent (alignment 0.698) **non-monotone
U-shaped** signal on the velocity-fraction quartile contingency:
Q1 (gamma_1 mostly positional) 49% S, Q2 (mixed, slightly velocity)
18% S, Q3 29% S, Q4 (gamma_1 mostly velocity) 43% S. The U-shape
suggests the load-bearing axis is **direction-purity**, not raw
velocity-fraction. v0.8 promotes purity to the primary operational
quantity and asks whether it predicts held-out stability on the
analyzable subset.

## What v0.8 Is Not

v0.8 is NOT:

- **A retreat from the v0.7a non-circularity sentence.** The
  non-circularity sentence carries forward verbatim: Floquet
  eigenvectors are used as geometric directions only; eigenvalue
  magnitude and stability thresholds never enter the feature. The
  purity scalar `abs(vf - 0.5)` is a deterministic function of vf,
  inheriting vf's non-circularity provenance.

- **A v0.7c that promotes the sidecar.** The v0.7a' z-fraction
  sidecar (D1 + A) did NOT pass the loud-signal threshold and is
  not promoted under v0.8. If the v0.8 chapter wants to extend to
  z-direction structure, that is a separate fresh registration with
  its own circularity audit.

- **A catalog-wide claim.** The domain is fixed at the 250
  analyzable rows from v0.7a (rows with `integration_blocked ==
  False`). The 23 integration-blocked rows + 11 R1-sanity-failed
  rows are **not** in the v0.8 domain. Attrition is recorded
  permanently on every v0.8 receipt; v0.8 makes no claims about
  rows outside the analyzable subset.

- **A re-binning of the v0.7a' Q_vf quartiles.** v0.8 uses purity
  as a fresh feature, with its own binning rule locked at v0.8a.
  Borrowing v0.7a' quartile cutpoints would conflate the registered-
  positive feature (vf quartile) with the proposed mechanism
  (purity). v0.8 needs its own binning to evaluate purity on its
  own terms.

## Body / Projection / Observable

```text
Body:
  the 250-row analyzable supp-B subset from v0.7a (catalog rows
  where compute_monodromy_vectorized completed successfully at
  rtol = atol = 1e-12, max_step_fraction = 0.02). Attrition carried
  as a permanent domain restriction.

Projection:
  orbit-dynamics direction-purity shadow
    row -> purity(row) = abs(vf(row) - 0.5)
  where vf is the v0.7a velocity-fraction (gamma_1's mass-weighted
  velocity-norm divided by total mass-weighted norm under CoM
  reduction), inheriting v0.7a's selection rule, tie-break cascade,
  and reference frame.

  Equivalently: purity = min(vf, 1 - vf) reversed and shifted (purity
  high <-> direction near a pure end {gamma_1 mostly positional or
  gamma_1 mostly velocity}; purity low <-> direction maximally mixed
  at vf = 0.5).

Observable:
  per-row pair (purity, stability) on 250 analyzable rows, joined
  with the v0.5a branch_label for alignment-tightness diagnostics.

Per-row computational cost:
  zero. purity is derived from existing v0.7a per_row_table.csv;
  no new variational integration is needed.
```

The projection axis is **direction-purity**, a monotone transform of
the v0.7a U-shape signature. Where v0.7a' tested whether the vf
quartile distribution stratifies S/U (non-monotone signal), v0.8
tests whether purity stratifies S/U (expected-monotone signal: high
purity -> high S, low purity -> low S).

## Operational Definition of Purity

```text
For each analyzable row r:

  purity(r) = abs(vf(r) - 0.5)

  range: [0, 0.5]

  purity(r) = 0    iff  vf(r) = 0.5  (maximally mixed direction)
  purity(r) = 0.5  iff  vf(r) in {0, 1} (perfectly pure direction;
                                         numerically rare for any
                                         real Floquet eigenvector).
```

The definition is **deterministic, monotone in `abs(vf - 0.5)`, and
inherits all of v0.7a's non-circularity provenance**. No new
operational quantities are introduced.

Equivalent forms (recorded but not used as the primary feature):

```text
  purity_alt(r) = 1 - 2 * min(vf(r), 1 - vf(r))
                = 2 * abs(vf(r) - 0.5)
                ∈ [0, 1]   (scaled version; same ordering)

  purity_signed(r) = vf(r) - 0.5
                   ∈ [-0.5, 0.5]   (signed; preserves which "pure end")
```

`purity_signed` is **recorded as a diagnostic** for the U-shape
asymmetry check (do Q1-pure orbits behave the same as Q4-pure
orbits?), but is **not the primary feature**. Using the signed version
as primary would silently re-test the v0.7a' Q1-vs-Q4 split, which
is the conditioning the v0.7a' result already used.

## Non-Circularity Audit

```text
purity is a monotone deterministic function of vf, so its
non-circularity provenance is exactly v0.7a's:

  vf is computed from gamma_1, the eigenvector of the largest-
  real-part Floquet eigenvalue of M_i, under the locked tie-break
  cascade. The eigenvalue ORDERING is used to disambiguate which
  eigenvector to extract; the eigenvalue MAGNITUDE never enters
  the feature.

  vf is a geometric ratio of mass-weighted-norm components after
  CoM reduction; purity is the distance from the mid-point of
  this ratio. Neither vf nor purity uses the stability label, the
  Floquet spectral radius, the unstable-pair count, the Krein
  signature, the v0.4a Pass classification, the v0.5a branch hash
  output, or the v0.6 conserved-quantity quartile.

  The v0.5a branch_label is joined ONLY to compute the alignment-
  tightness scalar (a diagnostic guard against branch-shadow
  re-projection), not to construct purity.

  S/U is the held-back test column, joined at the chi-squared step.
```

The non-circularity sentence from v0.7a is **re-asserted** for v0.8:

> v0.8 uses Floquet eigenvectors only as geometric directions; it
> does not use eigenvalue magnitude, spectral radius, unit-circle
> status, unstable-pair count, or any threshold that defines the
> published S/U label. The tested scalar is a distance-from-mixed
> phase-space composition ratio of the selected direction, not the
> growth rate of that direction.

## Disallowed Features (Inheritance + v0.8-Specific)

The v0.8 form locks may NOT use:

```text
v0.4b / v0.5 / v0.6 / v0.7 inheritance:
  - the stability label S/U or any function of it,
  - any function of M_i eigenvalue magnitudes (spectral radius,
    unit-circle count, max |lambda|, Krein signature, etc.),
  - the row's v0.4a Pass 1 / Pass 2 provenance,
  - the v0.5a branch hash output as a feature (it remains
    available as the alignment-tightness comparison axis only),
  - Q_E / Q_|L| / any v0.6a quartile,
  - E or |L| as scalars (retired in v0.6 close).

v0.8-specific:
  - the SIGNED form purity_signed as the primary feature
    (re-tests the v0.7a' Q1-vs-Q4 split; available as diagnostic
    only).
  - vf itself as the primary feature (v0.7a/v0.7a' already used
    this; v0.8 tests purity, not raw vf).
  - the v0.7a' Q_vf quartile assignments as features (would
    conflate v0.7a' positive with v0.8 mechanism test).
  - any quantity derived from the 23 integration-blocked rows or
    the 11 R1-sanity-failed rows (they are outside the v0.8
    domain).
```

## Inheritance Discipline From v0.5 / v0.6 / v0.7

v0.8 inherits all prior register-shaping disciplines:

1. **Audit-then-predictor separation.** v0.8a registers an audit
   on a binned (or monotone) purity feature; v0.8b separately
   registers a held-out predictor. An audit pass does NOT license
   a predictor; the predictor must pass its own pre-registered
   held-out gate.

2. **Asymmetric falsifier for the predictor.** v0.8b requires BOTH
   `McNemar p <= 0.01` AND positive accuracy delta against the
   always-U baseline. Inherited from v0.5b.

3. **Held-out partition discipline.** v0.8b defaults to
   leave-one-m_3-bin-out (over m_3 bins with N >= 5 on the 250-row
   analyzable subset). Alternatives must be motivated paper-side
   and pre-registered. Per the user's direction, **attrition is
   carried as part of the domain, not hidden**: the held-out folds
   are the m_3 bins of the analyzable subset, and the receipt
   explicitly records that the catalog-wide audit was attrited at
   8.4%.

4. **Tie rule and absent-bin fallback.** v0.8b inherits the
   conservative tie rule (predict U) and the global-majority
   fallback for absent bins.

5. **Alignment-tightness guard.** v0.8a MUST pre-register an
   alignment-tightness scalar against the v0.5a branch hash with
   the locked 0.8 warning / 0.95 severe thresholds. The v0.7a'
   alignment was 0.698 -- below the 0.8 warning floor -- but the
   purity feature is a transform, not the same as vf; the alignment
   must be re-evaluated on purity quartiles (or whatever binning
   v0.8a locks).

6. **Sparse-cell fallback tree.** v0.8a and v0.8b inherit the
   v0.6b discipline (chi-squared / exact permutation /
   sparse-inconclusive) with locked thresholds and locked
   permutation seed (20260523, n_permutations = 10000).

7. **Constant-feature retirement.** If purity has sd < 0.01 on the
   250-row subset, constant-feature retirement fires. Pre-mortem
   estimate from v0.7a': vf_sd was 0.144, so |vf - 0.5| should have
   sd > 0.05 (well above the floor).

## Candidate Audit Forms (Sketches for v0.8a Registration)

The v0.8a form lock will pick ONE of the following (or a registered
variant). All forms inherit the alignment-tightness guard, the
sparse-cell fallback tree, and the constant-feature retirement
threshold.

```text
A. Purity-quartile audit (analogue of v0.7a's B):
   Quartile-bin purity over the 250-row subset. 4 x 2 contingency
   over (Q_purity, S/U). df = 3, critical 11.34 at p = 0.01.
   Expected shape if v0.7a' signature is monotone in purity:
     Q1 (low purity, mixed direction)        ->  low S_fraction
     Q4 (high purity, near a pure direction) ->  high S_fraction
   Cleanest extension of the v0.7a' pattern.

B. Monotone purity-threshold predictor (single-cut audit):
   "predict S iff purity > purity_threshold"
   threshold = pre-registered quantile (e.g., median, q60, q75).
   2 x 2 contingency vs S/U, df = 1, critical 6.63 at p = 0.01.
   Coarser test but a direct monotone-direction claim.

C. Continuous correlation audit:
   Spearman rank correlation between purity and S (treating S as
   binary 1, U as 0). One-sample test of H_0: rho = 0 vs H_1:
   rho > 0 (one-sided since v0.7a' predicts positive correlation).
   Closest to the "distance from mixed" intuition; no binning
   choice.

D. Two-bin median-split audit (cleanest minimal test):
   Median-split purity over the 250-row subset. 2 x 2 contingency,
   df = 1, critical 6.63. Pre-registered direction: above-median
   purity has higher S_fraction. The form lock can include a
   binomial-tail test as a power-conservative alternative.
```

The v0.8a form lock must lock ONE of A-D (or a registered variant)
BEFORE running the audit. Multiple parallel audits are NOT permitted
under a single v0.8a registration.

## Candidate Predictor Forms (Sketches for v0.8b Registration)

If v0.8a passes, v0.8b registers a held-out predictor. Candidate
shapes:

```text
E. Bin-majority on the v0.8a purity binning (mirrors v0.5b):
   train per-bin majority on training folds (leave-one-m_3-bin-out);
   predict per-bin majority on test fold.

F. Threshold rule on purity (mirrors a fixed v0.8a-style cut):
   "predict S iff purity > purity_threshold"
   threshold pre-registered as a quantile of the training fold's
   purity distribution.

G. Logistic regression on purity (1 free parameter):
   single-feature logistic predictor; held-out McNemar against
   always-U.
   Free parameter must be counted explicitly; df adjustments
   recorded.
```

v0.8b's form lock must pre-register exactly one shape and one
partition (default leave-one-m_3-bin-out on the 250-row analyzable
subset). Free parameters must be counted explicitly.

## Open Design Questions (Resolved in v0.8a/v0.8b Form Locks)

1. **v0.8a audit form (A/B/C/D).** Most natural extension of v0.7a'
   is A (purity-quartile audit), mirroring v0.7a's B audit form.
   D (median-split) is a power-conservative two-bin alternative.
   C (Spearman correlation) avoids the binning choice entirely but
   introduces a different test statistic.

2. **Purity binning method.** Quartile (4 bins) vs median split
   (2 bins) vs continuous. Pre-registered before any chi-squared
   computation.

3. **v0.8b partition.** Default leave-one-m_3-bin-out on the 250
   analyzable rows. The m_3 bin distribution after attrition needs
   to be checked: some bins may have < 5 rows after attrition.
   v0.8a's receipt MUST report the per-m_3-bin analyzable row
   counts before v0.8b registration.

4. **U-shape vs monotone diagnostic.** v0.7a' showed a U-shape; v0.8
   tests monotone-in-purity. v0.8a's receipt MUST emit the
   diagnostic purity_signed quartile contingency to check whether
   Q1-pure and Q4-pure orbits behave the same in S-fraction (if
   not, the U-shape is asymmetric and the purity transform may not
   be the cleanest mechanism).

## Pre-Mortem (Recorded Now)

```text
Risk 1: purity could be a near-constant on the 250-row analyzable
        subset if vf is concentrated near 0 or 1.
        Mitigation: constant-feature retirement at sd < 0.01.
        v0.7a' vf_sd was 0.144; purity sd should be similar order
        (vf is roughly uniform over the subset, so purity is roughly
        uniform over [0, 0.5]).

Risk 2: the v0.7a' U-shape may be asymmetric -- Q1-pure orbits and
        Q4-pure orbits may have different S-fractions for physical
        reasons. If asymmetric, the purity transform conflates two
        different physical mechanisms.
        Mitigation: v0.8a receipt MUST emit the purity_signed
        contingency as a diagnostic. If the asymmetry is large
        (e.g., one tail has 60% S and the other 30%), v0.8b may
        need re-registration with a non-purity feature.

Risk 3: the attrition (23 blocked + 11 sanity-failed) is concentrated
        at long-period high-m_3. The held-out leave-one-m_3-bin-out
        partition for v0.8b on the analyzable subset will have very
        thin folds at high m_3 (m_3 = 1.7 has only ~5 analyzable
        rows). Low statistical power on those folds.
        Mitigation: v0.8a's receipt reports per-m_3 analyzable row
        counts. v0.8b may need to drop high-m_3 bins from the
        gating partition (recording them as report-only) if N < 5
        after attrition.

Risk 4: v0.8a passes but v0.8b fails (the v0.5/v0.6 pattern). This
        would close the chapter as another projection-limit, but
        with the difference that the v0.8a positive is the FIRST
        non-branch-aligned positive in the chain. A v0.8b fail
        does not undo v0.7's qualified-positive close.
        Mitigation: pre-mortem expectation recorded; the chapter
        close framing handles all outcomes cleanly.
```

## Stop Conditions

1. **Constant-feature retirement on purity.** Chapter closes on
   purity; the v0.8 mechanism cannot be evaluated under this
   feature.

2. **v0.8a fails the chi-squared / permutation test.** Chapter
   closes as a structural-negative on the purity shadow.

3. **v0.8a alignment-tightness > 0.8.** v0.8b cannot inherit the
   default leave-one-m_3-bin-out partition; an alignment-breaking
   re-registration is required.

4. **v0.8b fails the held-out predictor gate.** Chapter closes as
   a projection-limit on the purity shadow (mirroring v0.5).

5. **U-shape asymmetry diagnostic large (signed-quartile spread
   > 0.2).** v0.8a's receipt flags the issue; v0.8b may register
   a non-purity feature instead.

## Lock-In Statement

This parent registration is committed before:

- any v0.8a binning has been chosen,
- the purity distribution over the 250-row subset has been
  computed,
- the per-m_3 analyzable row counts have been examined,
- any v0.8b partition has been locked.

The body / projection / observable, operational definition of
purity, non-circularity audit framing, disallowed-feature list, and
inheritance discipline are locked. Any change after the v0.8a form
lock is a re-registration of the parent, not a refinement of a
child form.

Implementation may proceed to:

1. Compute purity = abs(vf - 0.5) for each of the 250 analyzable
   rows in v0.7a per_row_table.csv. Catalog-only; seconds of
   compute total.
2. Emit a per-row purity diagnostic table (label, m_3, z_0, vf,
   purity, purity_signed, stability, branch_label).
3. Emit per-m_3-bin analyzable row counts on the v0.8a receipt
   before any predictor partition is locked.

No audit verdict is licensed until a v0.8a form lock is separately
registered.

## Doc Trail

- `kfacet_v07_writeup.md` -- v0.7 qualified-positive chapter close
  (opens v0.8).
- `kfacet_v07a_velocity_fraction_audit_form.md` -- v0.7a parent
  audit (non-circularity sentence + amendments).
- `kfacet_v07a_prime_restricted_scope_form.md` -- v0.7a' restricted-
  scope confirmation (source of the U-shape signature).
- `kfacet_v05b_branch_predictor_form.md` -- v0.5b held-out
  predictor pattern (inheritance source for v0.8b).
- `kfacet_v06b_within_branch_energy_audit_form.md` -- sparse-cell
  fallback tree (inheritance source).
- `kfacet_v06a_energy_quartile_audit_form.md` -- alignment-tightness
  guard (inheritance source).

---

Parent registration locked. v0.8a (audit) and v0.8b (held-out
predictor) form locks pending. The non-circularity sentence carries
forward unchanged; the domain is fixed at the 250 analyzable rows;
attrition is permanent on the receipt and is carried as part of
the domain, not hidden.
