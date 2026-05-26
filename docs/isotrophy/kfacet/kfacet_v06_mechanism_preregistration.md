# v0.6 Mechanism Preregistration: Conserved-Quantity (E, |L|) Stratification

Status: **PARENT REGISTERED 2026-05-23**. v0.6a child verdict landed
2026-05-23: `energy_quartile_passes_audit_alignment_warning` (chi^2 =
33.70 vs 11.34, alignment = 0.956 > 0.8; see
`kfacet_v06a_energy_quartile_audit_form.md`). **v0.6b child verdict
landed 2026-05-24**: `within_branch_energy_fails_audit` at chi^2 =
6.90, permutation p = 0.029 (the sparse-cell fallback fired because
min_expected = 2.66 < 5; permutation test with seed 20260523 and
n_permutations = 10000 was used). |L| sidecar likewise below the
loud-signal threshold (chi^2 = 4.46, permutation p = 0.074). See
`kfacet_v06b_within_branch_energy_audit_form.md`. The within-branch
test shows the v0.6a in-sample positive was dominated by branch-shadow
content; the energy shadow does not carry stability information
beyond the v0.5a branch hash on supp-B. v0.6c (held-out predictor)
is NOT licensed under this form lock chain; the chapter is positioned
to close on a clean conditional-independence result. This document
locks the v0.6 body / projection / observable and the non-circularity
discipline. Form locks for v0.6a (audit) and v0.6b (within-branch
audit) are separately registered children that pin specific binning,
df, and falsifier choices.

Audience: v0.6a/v0.6b form-lock author; paper-side reviewer of the
v0.5 -> v0.6 transition.

Companions:

- `kfacet_v05_writeup.md` -- v0.5 chapter close (projection-limit) that
  opened v0.6.
- `kfacet_v04_writeup.md` -- v0.4 chapter close (structural-negative on
  the Z_2 shadow).
- `kfacet_v05a_branch_map_form.md`, `kfacet_v05b_branch_predictor_form.md`
  -- v0.5 audit + predictor form locks (inherited discipline).
- v0.3 cross-m_3 receipts / invariant implementation -- the sentinel
  rows used for E and |L| sanity-check provenance.

Frame: v0.4 ruled out the Z_2 shadow as a stability projector on supp-B
piano-trios. v0.5 ruled out the 2-bit catalog branch shadow as a
held-out predictor. v0.6 asks whether a **higher-dimensional
catalog-coordinate projection — orbit-level conserved quantities
(E, |L|) — carries held-out stability**.

## What v0.6 Is Not

v0.6 is NOT:

- A re-run of v0.5 with continuous features on the same axis. v0.5b
  failed because the m_3 axis stratification was bin-local. v0.6
  promotes the projection to orbit-level invariants that are
  catalog-derivable but not aligned with the m_3 catalog parameter.
- A Floquet-derived feature set. E and |L| are computed from initial
  conditions and the Hamiltonian, BEFORE any monodromy or stability
  inspection. They are orbit-input invariants, not orbit-output
  invariants.
- A held-back z_0 or v_z gate. The retired bits b3 (`abs(v_z) < 1e-6`)
  and b4 (`m_3 z_0^2 < 2`) stay retired on supp-B. If a future
  substrate (mesa, geometry, freeze-A choreographies) lights them up,
  THAT substrate gets its own registration.

## Body / Projection / Observable

```text
Body:
  supplementary-B piano-trio orbit as primary conserved-quantity object.
  domain: 273 rows, already verified Z2_clean by v0.4a.

Projection:
  orbit-level conserved-quantity shadow
    row -> (E(row), |L|(row))
  where E and |L| are computed at the row's published initial conditions
  using the three-body Hamiltonian and the published mass triple.

Observable:
  per-row pair (E, |L|) joined with the supp-B stability label S/U,
  binned into a pre-registered (n_E x n_L) contingency grid.

Per-row computational cost:
  seconds (sum of kinetic energy + sum of pairwise gravitational
  potential + cross-product sum). NO orbit integration required.
  Catalog-only quantities.
```

The projection family is **catalog-coordinate but orbit-invariant**:
E and |L| are functions of the published initial conditions, but they
carry orbit-level physical content (total energy and total angular
momentum) that is invariant under the Hamiltonian flow. This is
strictly richer than the v0.5 2-bit branch hash (which used only
`m_3` and `z_0`) without invoking Floquet spectra or symmetry
decompositions.

## Operational Definition of E and |L|

For a supp-B piano-trio row with published mass triple `(m_1, m_2,
m_3) = (1, 1, m_3)` and published 3D initial state per body
`(r_i, v_i)`:

```text
E(row) = sum_{i=1..3} 0.5 * m_i * |v_i|^2
       - sum_{1 <= i < j <= 3} (G * m_i * m_j) / |r_i - r_j|

L(row) = sum_{i=1..3} m_i * (r_i x v_i)        (vector cross-product)

|L|(row) = euclidean norm of L(row)
```

With the v0.3 convention `G = 1` (Li-Liao normalization). The
published initial conditions are read from the supp-B parser used in
v0.3 and v0.4a; no re-extraction is required.

Sanity checks (must pass before v0.6a registration):

```text
1. E and |L| are computed from the SAME initial conditions used by
   the v0.4a two-pass classifier and the v0.5a branch hash.
   No new orbit integration is required.

2. The 7 v0.3 cross-m_3 sentinel rows (O_50, O_62, O_67, O_434
   at m_3=0.4; O_242, O_282, O_284 at m_3=1.0) reproduce the
   v0.3 cross-m_3 receipt values for E and |L| at the v0.3
   floor (per-row residual < 1e-6 against the v0.3 receipt).

3. The per-row table emits both E and |L| for all 273 rows.
   No row drops out under the operational definition.

4. The per-row signed pre-mortem: |L_z| / |L| is recorded as a
   diagnostic but not used as a gating feature (z-axis component
   of angular momentum is reserved for chirality diagnostics
   only, mirroring the v0.5 treatment of sign(v_z)).
```

## Non-Circularity Audit

```text
E and |L| are functions of:
  - the published row mass triple (m_1, m_2, m_3),
  - the published row initial conditions (r_i, v_i for i=1..3).

E and |L| are NOT functions of:
  - the row's Floquet spectrum or monodromy operator,
  - the row's stability label S/U,
  - any v0.4a Pass 1 / Pass 2 classification,
  - any v0.5a branch hash output.

The supp-B stability label S/U is supplied as the independent test
column. E and |L| are catalog-derivable; the stability label is the
held-back observable.

The non-circularity provenance is the same as the v0.5a branch hash:
catalog-only features paired with an independent stability label.
The substantive difference is that E and |L| are CONTINUOUS
orbit-invariant scalars instead of catalog-coordinate indicator bits.
```

The conserved-quantity projection is therefore strictly richer than
the v0.5 branch hash (more information content per row) but uses no
additional stability information in the predictor.

## Disallowed Features (Inheritance from v0.4b)

The v0.6 form locks may NOT use:

```text
- the row's stability label S/U (or any function of it),
- the row's Floquet spectrum or any function of M_i eigenvalues
  (spectral radius, unit-circle count, Krein signature, etc.),
- the row's v0.4a Pass 1 / Pass 2 provenance,
- the row's v0.5a branch hash output,
- the row's K_fib tangent decomposition,
- any tangent-isotypic projection of M_i.
```

These are explicitly disallowed because they would either re-circulate
v0.4/v0.5 information or directly use the held-back observable. The
v0.6 family must construct its predictor from `(m_1, m_2, m_3, r_i,
v_i)` per row, plus operational quantities derivable from these via the
Hamiltonian (E, |L|, and any explicitly registered scalar).

## Inheritance Discipline From v0.5

The v0.5 chapter close locked five register-shaping lessons. v0.6
inherits all five:

1. **Audit-then-predictor separation.** v0.6a registers a chi-squared
   independence audit on a binned (E, |L|) grid; v0.6b separately
   registers a held-out predictor. An audit pass does NOT license a
   predictor; the predictor must pass its own pre-registered
   held-out gate.

2. **Asymmetric falsifier for the predictor.** v0.6b requires
   BOTH `McNemar p <= 0.01` AND positive accuracy delta against the
   always-U baseline. "Differs from baseline" alone is not a pass.

3. **Held-out partition that respects bifurcation structure.** v0.6b
   defaults to leave-one-m_3-bin-out (12 bins with N >= 5 on supp-B),
   matching v0.5b. If a v0.6b form locks an alternative partition
   (e.g., leave-one-E-band-out), that alternative must be motivated
   paper-side and pre-registered, and the leave-one-m_3 partition
   is recorded as a sidecar.

4. **Tie rule and absent-bin fallback.** v0.6b inherits the
   conservative tie rule (predict U) and the global-majority
   fallback for absent bins, with `absent_bin_fallback_used` flagged
   on the receipt.

5. **Constant-feature retirement.** If any feature derived from
   (E, |L|) is constant on supp-B (e.g., all rows bound, all rows
   prograde), it is recorded on the receipt and retired from df.
   Cross-substrate comparability is preserved.

## Candidate Audit Forms (Sketches for v0.6a Registration)

The v0.6a form lock will pick one of the following audit forms with
specific binning constants. All forms share a chi-squared(df)
falsifier at p = 0.01:

```text
A. Univariate E audit:
   bin E by quartiles over supp-B -> 4-bin x 2-class contingency.
   df = 3, critical = 11.34. Tests whether E quartile alone stratifies.

B. Univariate |L| audit:
   bin |L| by quartiles over supp-B -> 4-bin x 2-class contingency.
   df = 3, critical = 11.34. Tests whether |L| quartile alone stratifies.

C. Joint (E, |L|) audit:
   bin both E and |L| by pre-registered cutoffs (median split, terciles,
   or quartiles) -> n_E x n_L x 2 contingency, collapsed by
   occupied-cell count.
   df = (occupied_cells) - 1; critical from chi-squared(df) at p = 0.01.

D. Energy-ratio audit (dimensionless):
   compute K_total / |V_total| (kinetic-to-virial ratio) per row,
   bin by quartiles -> 4-bin x 2-class. df = 3, critical = 11.34.

E. Angular-momentum scaled audit (dimensionless):
   compute |L| / sqrt(|E| * I_tot) or similar dimensionless ratio
   with explicitly pinned reference scales. df = 3, critical = 11.34.
```

v0.6a must lock ONE of A-E (or a registered variant) BEFORE running
the audit. Multiple parallel audits are NOT permitted under one v0.6a
registration; each separately-registered audit gets its own form lock,
its own falsifier, and its own receipt directory.

## Candidate Predictor Forms (Sketches for v0.6b Registration)

If v0.6a passes, v0.6b registers a held-out predictor. Candidate
shapes:

```text
F. Bin-majority on the v0.6a binning:
   train per-bin majority on training folds, predict per-bin majority
   on test fold. Mirrors v0.5b's branch-majority pattern.

G. Threshold rule on a univariate quantile:
   "predict S iff E > E_threshold" (or |L| analog). Threshold
   pre-registered as a quantile (e.g., median, q25, q75) of the
   training fold's gating subset.

H. Two-feature threshold rule:
   "predict S iff E > E_threshold AND |L| > L_threshold" (or
   any pre-registered logical conjunction/disjunction of two
   univariate quantile thresholds).

I. Linear discriminant on (E, |L|):
   fit a per-fold LDA on training; this introduces 2-3 free
   parameters per fold. Requires explicit free-parameter
   accounting on the receipt and df adjustment.
```

v0.6b's form lock must pre-register exactly one shape and one
partition (default leave-one-m_3-bin-out). Free parameters must be
counted explicitly; LDA / regression-based forms get separate
treatment from zero-parameter threshold rules.

## Open Design Questions (Resolved in v0.6a/v0.6b Form Locks)

These are flagged here so the form-lock authors can take direction:

1. **Audit form (A/B/C/D/E).** Which projection of (E, |L|) carries
   the first audit? The univariate energy audit (A) is the simplest
   and most defensible non-circularly; the joint (E, |L|) audit (C)
   has more df but more discrimination power; the dimensionless
   audits (D, E) require pinned reference scales (I_tot for E,
   the energy-virial denominator for D).

2. **Binning strategy.** Quartile bins are conservative and assumption-
   free; physically-motivated cutoffs (bound vs. unbound, prograde vs.
   retrograde) are sharper but require justification before v0.6a is
   locked.

3. **Predictor form (F/G/H/I).** Threshold rules (G, H) are
   zero-parameter and inherit v0.5b discipline most cleanly. Bin-
   majority (F) mirrors v0.5b. LDA / regression (I) introduces free
   parameters and requires explicit accounting.

4. **Partition.** Default is leave-one-m_3-bin-out (inherited from
   v0.5b). Alternatives (leave-one-E-band-out, deterministic random
   half) are sidecars unless explicitly motivated.

5. **Dimensionless ratios.** Quantities like `K_total / |V_total|`,
   `|L| / sqrt(|E| * I)`, `T / T_kepler`, etc. require pinned
   reference scales. v0.6 carries forward the v0.5 demotion of
   T / T_kepler and mu_eff^(1/2) products until references are
   registered.

## Pre-Mortem (Recorded Now)

```text
Risk 1: E and |L| are smooth functions of (m_3, z_0) on supp-B.
        If (E, |L|) tracks (m_3, z_0) too closely, v0.6 inherits
        v0.5b's bin-locality failure mode.
        Mitigation: report the per-bin scatter of (m_3, z_0) within
        each (E, |L|) bin on the v0.6a receipt; if (E, |L|) bins
        are nearly disjoint in (m_3, z_0) space, the audit pass
        does NOT license confidence in held-out generalization.

Risk 2: |L_z| (z-component) is sign-degenerate on chirality-symmetric
        sub-catalogs. Treat sign as diagnostic only; magnitude
        |L| as the gating scalar.

Risk 3: E is negative for bound orbits and positive for unbound
        configurations. supp-B is all bound (periodic), so E sign
        is constant. The retirement rule applies: sign(E) RETIRED
        if constant on supp-B, recorded on receipt for cross-
        substrate comparability.
```

## Stop Condition

If v0.6a fails the chi-squared audit, v0.6b is not registered. The
chapter closes immediately as a structural-negative on the (E, |L|)
projection.

If v0.6a passes but the (E, |L|) bins are tightly aligned with the
v0.5b (m_3, z_0) buckets (per the Risk 1 mitigation check), v0.6b
must register an alternative partition (leave-one-E-band-out, or a
sub-catalog with constant m_3) BEFORE running the held-out test.
Inheriting v0.5b's leave-one-m_3-out partition without checking
(E, |L|)-to-(m_3, z_0) alignment is a discipline violation.

If v0.6b fails the held-out predictor gate, the chapter closes as a
projection-limit (mirroring v0.5). The published statement would
read: "the (E, |L|) projection is associated with stability in-sample
but does not generalize across held-out folds." v0.7 (if any) would
target a non-catalog-coordinate projection (e.g., tangent-bundle
geometry that uses orbit dynamics).

## Lock-In Statement

This parent registration is committed before any v0.6a binning,
v0.6b partition, or runner work. The body / projection / observable,
operational definitions of E and |L|, non-circularity provenance,
disallowed-feature list, and inheritance discipline from v0.5 are
locked.

Any change to those after the v0.6a form lock is a re-registration of
the parent, not a refinement of a child form.

Implementation may proceed to:

1. Add a per-row E and |L| computation against the existing supp-B
   parser output (catalog-only, no orbit integration).
2. Sanity-check against the 7 v0.3 cross-m_3 sentinel rows.
3. Emit a per-row (E, |L|) table for v0.6a registration to bin.

No audit verdict is licensed until a v0.6a form lock is separately
registered with explicit binning constants.

## Doc Trail

- `kfacet_v06a_energy_quartile_audit_form.md` -- v0.6a child form lock
  (audit form A, univariate E quartiles, chi-squared(3) at p = 0.01;
  |L| quartile sidecar report-only).
- `kfacet_v05_writeup.md` -- v0.5 projection-limit chapter close (opens v0.6).
- `kfacet_v05a_branch_map_form.md` -- v0.5a audit (catalog-coordinate ancestor).
- `kfacet_v05b_branch_predictor_form.md` -- v0.5b held-out predictor (inheritance source).
- `kfacet_v04_writeup.md` -- v0.4 structural-negative chapter close.
- `kfacet_v06b_within_branch_energy_audit_form.md` -- v0.6b
  alignment-breaking within-branch audit (alignment-warning child).
- v0.3 cross-m_3 receipts / invariant implementation -- E, |L|
  sentinel sanity-check provenance.

---

Parent registration locked. v0.6a verdict landed
(`kfacet_v06a_energy_quartile_audit_form.md`): `chi^2_E = 33.703158`
versus critical `11.34`, `p = 2.29e-7`, but
`alignment_tightness_scalar_E = 0.955882 > 0.8`. The |L| sidecar was
loud (`chi^2_|L| = 28.954252`, report-only) and similarly aligned
(`0.956522`). v0.6b verdict landed 2026-05-24
(`kfacet_v06b_within_branch_energy_audit_form.md`):
`within_branch_energy_fails_audit` at `chi^2 = 6.904`, permutation
`p = 0.029` (sparse-cell fallback fired with `min_expected = 2.655`).
|L| sidecar likewise fell short of the loud-signal threshold
(`chi^2 = 4.465`, permutation `p = 0.074`, report-only). v0.6c
(held-out predictor) is NOT licensed; the energy-shadow sub-question
closes as a conditional-independence result. v0.6 chapter is
positioned for chapter-close writeup.
