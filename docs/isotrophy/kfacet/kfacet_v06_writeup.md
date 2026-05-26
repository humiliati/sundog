# K_facet v0.6 Methodology + Result Handoff

Status: closed conditional-independence (catalog audit passes, within-branch
audit fails), 2026-05-24.
Audience: collaborators, paper-side writers, future coding agents.
Canonical sources: `kfacet_v06_mechanism_preregistration.md`,
`kfacet_v06a_energy_quartile_audit_form.md`,
`kfacet_v06b_within_branch_energy_audit_form.md`, and
[`../../threebody/CROSS_SUBSTRATE_NOTES.md`](../../threebody/CROSS_SUBSTRATE_NOTES.md)
sections 6-7 (projection-language framing).
Companion (closed v0.5 chapter): `kfacet_v05_writeup.md`.

## One-Line Read

v0.6 closes as a **conditional-independence result** on the
catalog-coordinate conserved-quantity (E, |L|) shadow of the Li-Liao
supplementary-B piano-trio catalog. The univariate energy quartile audit
passes the registered chi-squared gate decisively in-sample
(`chi^2 = 33.70` vs critical 11.34, `p ~= 2.29e-7`, v0.6a) but trips the
pre-registered alignment-tightness guard (`alignment = 0.956 > 0.8`). The
alignment-breaking within-branch audit then fails the registered
permutation-test floor (`chi^2 = 6.90`, `permutation p = 0.029`,
v0.6b), confirming the v0.6a in-sample positive was dominated by
branch-shadow content. Energy quartile is essentially a 1-to-1 label
for m_3 sub-bin within the v0.5a stable branch; it does not carry
stability information independent of the v0.5a branch hash on this
catalog. Distinct from v0.4 (structural-negative) and v0.5
(projection-limit); a third chapter-close type.

## What v0.6 Tested

v0.6 opened after v0.5's projection-limit close. The v0.5 chapter
established that the 2-bit catalog branch shadow `(m_3 < 1, z_0 < 0.3)`
stratifies supp-B in-sample (chi^2 = 34.99) but does not predict held-out
stability across m_3 bins (accuracy delta -0.019). v0.6 promoted the
projection from indicator-bit features to continuous orbit-level
conserved quantities:

> v0.6 promotes the projection from a 2-bit catalog hash to continuous
> catalog-coordinate orbit invariants (E, |L|), and asks whether the
> body's stability structure is visible in this richer projection.

Three pre-registered chapters, all paper-side first:

```text
v0.6a: univariate energy-quartile catalog audit
       Body:        supp-B piano-trio orbit as primary conserved-
                    quantity object.
       Projection:  row -> E(row); supp-B-relative quartiles.
       Observable:  4 x 2 contingency Q_E vs S/U over 273 rows.
       Question:    chi-squared independence of Q_E vs S/U at p=0.01,
                    with pre-registered alignment-tightness guard.
       Result:      energy_quartile_passes_audit_alignment_warning
                    (chi^2 = 33.70 vs 11.34, p ~= 2.29e-7,
                     alignment-tightness = 0.956 > 0.8).
                    Pass with the guard tripped.

v0.6b: within-branch energy-quartile audit (alignment-breaking)
       Stratum:     v0.5a (m_3 < 1, z_0 < 0.3) branch (113 rows /
                    63 S / 50 U). v0.5a's most-stable branch.
       Feature:     Q_E under v0.6a's GLOBAL supp-B cutpoints (no
                    within-branch re-binning).
       Test:        chi-squared with sparse-cell fallback to exact
                    permutation test (seed = 20260523,
                    n_permutations = 10000) or
                    within_branch_energy_inconclusive_sparse verdict.
       Sidecar:     within-branch Q_|L| audit (report-only).
       Question:    does Q_E carry stability info beyond branch?
       Result:      within_branch_energy_fails_audit
                    (chi^2 = 6.90, permutation p = 0.029;
                     |L| sidecar p = 0.074 below loud-signal floor).
```

The audit chain inherited and extended v0.5's discipline with one new
register-shaping addition.

## Audit Chain Methodology

The v0.6 chain inherited the v0.5 discipline (audit-then-predictor
separation, asymmetric falsifier on the predictor, default
leave-one-m_3-bin-out partition, conservative tie rule,
constant-feature retirement) and added:

**Alignment-tightness guard.** v0.6a pre-registered a scalar measuring
how tightly the audit feature's bins align with the v0.5a branch
hash. If `max over Q_E of (fraction in any single branch_label bucket)
> 0.8`, the audit pass is **bin-locality-suspect** and a default
held-out predictor partition is no longer inherited cleanly. The
alignment guard fired hard (0.956), confirming the v0.6a in-sample
positive was branch-shadow-aligned and licensing v0.6b as an
alignment-breaking child audit instead of v0.6c as a held-out
predictor.

**Sparse-cell fallback discipline.** v0.6b pre-registered a three-way
fallback tree before any compute: (a) chi-squared(occupied_bins - 1)
if min expected cell >= 5; (b) exact permutation test (seed locked
pre-data) if min expected cell < 5 but min occupied bin count >= 2;
(c) `within_branch_energy_inconclusive_sparse` verdict if any
occupied bin has < 2 rows. The permutation branch fired (min_expected =
2.655 < 5; min_occupied_bin_count = 6 >= 2), giving an exact
distribution-free p-value of 0.029 — below the chi-squared(2)
critical-implied threshold but above the registered p <= 0.01 floor.

Three stages, all paper-side first:

1. **v0.6 parent registration**
   (`kfacet_v06_mechanism_preregistration.md`). Locked body /
   projection / observable, operational definitions of E and |L|,
   non-circularity provenance, disallowed-feature list (no Floquet,
   no stability, no v0.5a output, no K_fib decomposition),
   inheritance discipline from v0.5, candidate audit forms (A-E)
   and predictor forms (F-I).

2. **v0.6a child form lock + verdict**
   (`kfacet_v06a_energy_quartile_audit_form.md`). Selected candidate
   form A (univariate E quartiles) over B (|L| sidecar-only), C
   (joint, df dilution), D (K/|V|, reference scale unpinned), and E
   (|L|-scaled, reference scale unpinned). Locked supp-B-relative
   quartile cutpoints via `numpy.quantile(..., method='linear')`,
   4 x 2 contingency, df = 3, critical = 11.34, alignment-tightness
   scalar with 0.8 threshold. Sanity gates (v0.3 cross-m_3 sentinel
   parity at < 1e-6, bound-orbit check E < 0 for all 273) passed.

3. **v0.6b child form lock + verdict**
   (`kfacet_v06b_within_branch_energy_audit_form.md`). Alignment-
   breaking by **conditioning on branch**: stratum =
   v0.5a (m_3 < 1, z_0 < 0.3), 113 rows. Reused v0.6a's GLOBAL supp-B
   cutpoints (no re-binning within branch). Pre-registered sparse-cell
   fallback. Within-branch |L| audit as report-only sidecar.

## Result And Conditional-Independence Verdict

```text
v0.6a verdict:   energy_quartile_passes_audit_alignment_warning
  chi^2_E = 33.70 vs critical 11.34 (chi-squared(3), p = 0.01).
  p_value ~= 2.29e-7 (~3x threshold).
  273 rows, 97 S / 176 U; 4 quartile bins.
  Q-bin S fractions: Q1 30.4%, Q2 17.6%, Q3 30.9%, Q4 63.2%.
  alignment_tightness_scalar_E = 0.956 > 0.8.
  Q4 holds 65 of 68 rows in the v0.5a (m_3<1, z_0<0.3) branch
  (95.6%); the in-sample positive is bin-locality-suspect.
  |L| sidecar (report-only): chi^2 = 28.95, p = 2.29e-6, alignment 0.957.

v0.6b verdict:   within_branch_energy_fails_audit
  Stratum: 113 rows / 63 S / 50 U (v0.5a stable branch).
  Sanity passed.
  Within-branch Q_E contingency:
    Q1 N=0     (entirely outside stratum -- tightest catalog orbits
                all sit in other branches).
    Q2 N=6   (2 S / 4 U, 33.3% S)
    Q3 N=42  (18 S / 24 U, 42.9% S)
    Q4 N=65  (43 S / 22 U, 66.2% S)
  Within-branch direction MONOTONE in same direction as v0.6a.
  min_expected_cell = 2.655 (< 5) -> sparse-cell fallback fired.
  Permutation test (seed = 20260523, n_permutations = 10000):
    chi^2 = 6.904, permutation p = 0.0292.
  Verdict: fails registered p <= 0.01 floor.

  Within-branch Q_E x m_3 joint diagnostic:
    Q4 = m_3 in {0.4, 0.5}   (52 of 65 Q4 rows are m_3 = 0.4)
    Q3 = m_3 in {0.5, 0.6, 0.7, 0.8}
    Q2 = m_3 in {0.8, 0.9}
  Within stratum, Q_E is essentially a 1-to-1 label for m_3 sub-bin.

  |L| sidecar (report-only):
    chi^2 = 4.465, permutation p = 0.0741.
    Does NOT meet loud-signal threshold (p <= 0.01) for a fresh
    v0.6c |L| form lock.
```

The publishable v0.6 statement:

> v0.6 registered a univariate energy-quartile audit on the supp-B
> piano-trio catalog and tested it under two pre-registered
> protocols. The catalog audit (v0.6a, chi-squared independence with
> alignment guard) passed the registered chi-squared gate decisively
> (chi^2 = 33.70, p ~= 2.29e-7) but tripped the alignment-tightness
> guard at 0.956 (> 0.8 threshold). The alignment-breaking
> within-branch audit (v0.6b, on the v0.5a (m_3<1, z_0<0.3) stratum)
> failed under the pre-registered sparse-cell fallback's exact
> permutation test (chi^2 = 6.90, permutation p = 0.029 vs threshold
> p <= 0.01). The within-branch direction is monotone in the same
> direction as the catalog-wide finding, but Q_E within the stratum
> is essentially a 1-to-1 label for m_3 sub-bin and does not
> generalize as a within-branch stratifier at the registered floor.
> **Energy quartile does not carry stability information beyond
> labeling the v0.5a branch hash on supp-B.** The |L| sidecar at
> permutation p = 0.074 likewise falls short of the loud-signal
> threshold; no fresh |L| form lock is licensed. v0.6c (held-out
> predictor) is not licensed; the chapter closes as a
> conditional-independence result, distinct from v0.4
> (structural-negative) and v0.5 (projection-limit).

Three structural sub-results preserved in receipts and worth carrying
into v0.7:

1. **Audit-with-alignment-guard as a methodological register**. A
   chi-squared independence audit at p ~= 1e-7 is operationalized as
   bin-locality-suspect by a pre-registered alignment-tightness
   scalar against v0.5a's branch label. This is the third register-
   shaping discipline added to the isotrophy methodology after v0.5's
   audit-then-predictor separation and asymmetric McNemar+delta
   falsifier. The alignment guard fires BEFORE a held-out test runs,
   forcing alignment-breaking re-registration rather than letting
   bin-locality survive into a predictor stage.

2. **Sparse-cell fallback tree as a permanent discipline**. The
   v0.6b form lock introduced an explicit three-way fallback (chi-
   squared / exact permutation / sparse-inconclusive) with locked
   thresholds and a locked permutation seed. This generalizes to any
   future within-strata audit on small N and avoids the standard
   chi-squared assumption violations (expected cell >= 5) without
   silently inflating type-I error.

3. **Energy quartile = m_3 sub-bin within the v0.5a stable branch
   on supp-B**. Q4 (loosest catalog orbits) is dominated by m_3 = 0.4
   (52 of 65 Q4 rows); Q3 is mid-m_3; Q2 is high-m_3 within the
   branch; Q1 is entirely outside the branch (the tightest catalog
   orbits all sit in other branches). The conserved-quantity shadow
   does not add information beyond the (m_3, z_0) branch hash on
   this catalog at the registered precision.

Joint v0.4 + v0.5 + v0.6 envelope on supp-B (the publishable
multi-chapter statement):

> Three sequential pre-registered low-dimensional projections of the
> supp-B body — the Z_2 symmetry shadow (tangent-isotypic and
> orbit-gauge-rigidity, v0.4), the 2-bit catalog branch shadow
> (v0.5), and the continuous orbit-level conserved-quantity shadow
> E and |L| (v0.6) — none carries held-out or branch-conditional
> stability information on this catalog at the registered floors.
> Stability on supp-B is bin-local to the m_3 = 0.4 cluster; no
> catalog-coordinate or symmetry-shadow projection tested so far
> lifts that locality to a generalizable mechanism. Any v0.7
> mechanism must either leave catalog-coordinate space entirely,
> target the m_3 = 0.4 cluster as a substrate-specific object, or
> retire the program.

## Reproducibility Surface

Primary scripts (catalog-only; total compute is seconds):

```bash
# v0.6a (catalog-only, seconds):
python scripts/v06a_energy_quartile_audit.py

# v0.6b (catalog-only, seconds + sub-minute permutation):
python scripts/v06b_within_branch_energy_audit.py
```

Key receipt directories:

```text
results/isotrophy/k-facet-v06a-energy-quartile-audit/manifest.json
results/isotrophy/k-facet-v06a-energy-quartile-audit/contingency_table_E.csv
results/isotrophy/k-facet-v06a-energy-quartile-audit/contingency_table_L.csv
results/isotrophy/k-facet-v06a-energy-quartile-audit/energy_quartile_by_branch.csv
results/isotrophy/k-facet-v06a-energy-quartile-audit/energy_quartile_by_m3.csv
results/isotrophy/k-facet-v06a-energy-quartile-audit/energy_quartile_by_z0_bucket.csv
results/isotrophy/k-facet-v06a-energy-quartile-audit/l_quartile_by_branch.csv
results/isotrophy/k-facet-v06a-energy-quartile-audit/per_row_table.csv

results/isotrophy/k-facet-v06b-within-branch-energy/manifest.json
results/isotrophy/k-facet-v06b-within-branch-energy/contingency_table_within_branch_E.csv
results/isotrophy/k-facet-v06b-within-branch-energy/contingency_table_within_branch_L.csv
results/isotrophy/k-facet-v06b-within-branch-energy/within_branch_Q_E_by_m3.csv
results/isotrophy/k-facet-v06b-within-branch-energy/within_branch_Q_E_by_Q_L.csv
results/isotrophy/k-facet-v06b-within-branch-energy/per_row_table.csv
```

Load-bearing constants live in the form-lock documents:

```text
kfacet_v06a_energy_quartile_audit_form.md
                            chi-squared critical = 11.34 (df = 3, p = 0.01);
                            alignment-tightness threshold = 0.8;
                            operational definitions of E and |L|;
                            quartile method = numpy linear interpolation.

kfacet_v06b_within_branch_energy_audit_form.md
                            sparse-cell fallback tree;
                            permutation seed = 20260523;
                            n_permutations = 10000;
                            stratum = v0.5a (m_3<1, z_0<0.3) branch;
                            sidecar = within-branch |L|, report-only.
```

## Where v0.7 Opens

The v0.6 conditional-independence close adds a third chapter-close
register to the isotrophy methodology:

```text
v0.3:  domain-of-applicability (Gamma_i predicts 0 on D_3-strict
       domain, supp-B daughters are Z_2-or-smaller -> out of domain).
v0.4:  structural-negative (Z_2 shadow's two registered predictors
       both fail).
v0.5:  projection-limit (audit passes in-sample at p ~= 1e-7,
       predictor fails held-out by accuracy_delta = -0.019).
v0.6:  conditional-independence (catalog audit passes with alignment
       warning, within-branch audit fails permutation p = 0.029 vs
       p <= 0.01 floor).
```

Each chapter closes a distinct family of catalog-coordinate or
symmetry-shadow projections. The cumulative envelope says no such
projection tested so far lifts the m_3 = 0.4 stable cluster's
bin-locality to a generalizable mechanism.

**v0.7 opens with the gamma_1 direction-of-instability family**
(codex direction, 2026-05-24). Re-registers the v0.3 gamma_1 sidecar
(direction of largest-real-part Floquet eigenvector) as a primary
mechanism. This required a careful non-circularity audit because
v0.4b explicitly disallowed Floquet-derived features for stability
circularity reasons. **Parent registration landed 2026-05-24**
at `kfacet_v07_mechanism_preregistration.md`; **v0.7a child form
lock landed 2026-05-24** at
`kfacet_v07a_velocity_fraction_audit_form.md`. The parent locks
body / projection / observable, sketches five candidate operational
definitions (D1-D5), explicitly identifies three circularity risks,
and inherits all v0.5/v0.6 discipline. The v0.7a form lock picked
**D5 + B** (velocity-fraction quartile audit) over D1 + A because
D5 minimizes the tie-break surface and avoids using eigenvalue
ordering as the feature-selection mechanism; D1 + A is preserved as
a named report-only sidecar. The locked non-circularity sentence:
*"v0.7a uses Floquet eigenvectors only as geometric directions; it
does not use eigenvalue magnitude, spectral radius, unit-circle
status, unstable-pair count, or any threshold that defines the
published S/U label. The tested scalar is a phase-space composition
ratio of the selected direction, not the growth rate of that
direction."* v0.7a runner pending implementation
(`scripts/v07a_velocity_fraction_audit.py`).

## Doc Trail

- `kfacet_v06_mechanism_preregistration.md` -- v0.6 parent registration.
- `kfacet_v06a_energy_quartile_audit_form.md` -- v0.6a catalog audit form
  lock + verdict.
- `kfacet_v06b_within_branch_energy_audit_form.md` -- v0.6b within-branch
  audit form lock + verdict.
- `kfacet_v07_mechanism_preregistration.md` -- v0.7 parent registration
  (gamma_1 direction-of-instability; opens v0.7).
- `kfacet_v05_writeup.md` -- v0.5 projection-limit chapter close (companion).
- `kfacet_v04_writeup.md` -- v0.4 structural-negative chapter close
  (grand-companion).
- `kfacet_v03h_writeup.md` -- v0.3 domain-of-applicability close
  (great-grand-companion).
- `docs/threebody/CROSS_SUBSTRATE_NOTES.md` sections 6-7 -- projection-
  language framing (body / shadow / projection-limit / conditional-
  independence vocabulary).

The chapter closes cleanly. Receipts are durable. v0.6a's catalog
audit is preserved as a real finding (and as a methodological
demonstration of the alignment-tightness guard); v0.6b's within-branch
fail is preserved as the corresponding conditional-independence
result. The chain has produced four sequential pre-registered
chapters (v0.3, v0.4, v0.5, v0.6) of progressively richer projections
on the Li-Liao supplementary-B catalog, each with reproducible
receipts and explicit chapter-close discipline.
