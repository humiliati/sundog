# v0.4b gamma_3' Functional Form Lock (Orbit-Pass2 Baseline)

Status: **VERDICT LANDED 2026-05-23**. Locked paper-side 2026-05-23 after the
tangent-isotypic gamma_3 baseline retired and the cocycle-rescue probe surfaced
`(R_i, phi_i) = (I, 0)` uniformly. The locked orbit-pass2 baseline was tested
against the v0.4a manifest directly (no new compute). **Falsified by
chi^2 = 1202.32 vs critical value 26.22** (46x threshold). Rule's accuracy
63.74% is slightly below the always-U baseline 64.47%, confirming
gauge-rigidity is statistically independent of stability on supp-B. See the
**Verdict (Landed)** section below.
Audience: v0.4b implementation; v0.4c reader of the verdict.
Companion: `kfacet_v04b_gamma3_form.md` (retired tangent baseline),
`kfacet_v04b_mechanism_preregistration.md` (parent registration), and the
v0.4a manifest at
`results/isotrophy/k-facet-v04a-domain-map/manifest.json`.

## Verdict (Landed, 2026-05-23)

The locked baseline was tested against the v0.4a manifest directly. No new
compute. Per-row pass2_required boolean was read from
`results/isotrophy/k-facet-v04a-domain-map/manifest.json#per_row_table[*].provenance`.

```text
Total rows:                          273
Predicted S (pass2_required = True):  24   ( 8.79% of catalog)
Predicted U:                         249  (91.21%)
Observed S:                           97  (35.53%)
Observed U:                          176  (64.47%)

Accuracy:                174 / 273 = 63.74%
Always-U baseline:       176 / 273 = 64.47%
                         (rule is SLIGHTLY WORSE than always-U)

chi^2 test statistic:    1202.32
critical value:            26.22   (chi-squared(12), p = 0.01)
verdict:                 falsifies_baseline_orbit_pass2_rule
                         (chi^2 / critical = 45.86)
```

Per-m_3 contributions to chi-squared:

```text
m_3   N    S_obs  U_obs  pass2_count  P_S_pred  chi2_contrib
0.4   55   35     20     1            0.0182    1177.41
0.5   31   15     16     0            0.0000       7.26
0.6   22    6     16     0            0.0000       1.64
0.7   18    4     14     0            0.0000       0.89
0.8   18    6     12     0            0.0000       2.00
0.9   25    8     17     0            0.0000       2.56
1.0   38    7     31     0            0.0000       1.29
1.1   23    1     22     0            0.0000       0.04
1.2    7    2      5     0            0.0000       0.57
1.5    8    4      4     8            1.0000       2.00
1.6    8    7      1     7            0.8750       0.00
1.7   10    0     10     4            0.4000       6.67
                                                ---------
                                       chi^2 =  1202.32
```

Diagnostic bins (N < 5, non-gating): m_3 in {1.3, 1.4, 1.9}.

The m_3 = 0.4 bin alone contributes `1177` of the total chi-squared.
35 observed stable orbits at m_3 = 0.4 vs only 1 predicted stable
(O_434, the worked-example Pass 2 rescue). The rule under-predicts S
catastrophically at low m_3 and over-predicts S at m_3 in {1.5, 1.6, 1.7}.

**Structural reading:**

> Orbit-level gauge-rigidity, as measured by Pass 2 reclassification in the
> v0.4a two-pass classifier, is statistically independent of stability on the
> supp-B catalog. The two pieces of information do not predict one another;
> they are orthogonal projections of the body.

This falsifies the registered baseline. The wording per Codex sign-off:
**falsifies this baseline orbit-pass2 rule**, not gamma_3'_orbit_features
globally. A subsequent gamma_3'_T_threshold, gamma_3'_z0_threshold, or
gamma_3'_combined registration could be tested separately if the diagnostic
residuals suggest a different orbit feature carries stability information.

### Receipts

```text
internal/anniversary/kfacet_v04b_gamma3prime_form.md
results/isotrophy/k-facet-v04b-gamma3prime-orbit-pass2/manifest.json
  - per_m3_table:        per-m_3 contingency + chi-squared contributions
  - per_row_predictions: 273 rows with pass2_required + predict + correct
  - verdict:             falsifies_baseline_orbit_pass2_rule
```

## The Locked Form

`gamma_3'_orbit_pass2` is a **zero-parameter threshold rule** on a single
orbit-level Z_2 feature recorded by the v0.4a two-pass classifier:

```text
gamma_3'_orbit_pass2(row) =
  +1.0  (predict S)  iff  pass2_required(row) == True
   0.0  (predict U)  otherwise
```

Where:

```text
pass2_required(row) == True
   iff the row's Pass 1 classification (default gauge tolerances:
       identity_rotation_tolerance = 1e-6, phase_grid = 73) placed it
       in any band other than Z2_clean, AND the row's final classification
       (after the Pass 2 tight rerun at identity_rotation_tolerance = 1e-9,
       phase_grid = 361) is Z2_clean.

   Equivalently: the v0.4a manifest's per_row_table[row]["provenance"]
   field reads "pass2".
```

24 of 273 supplementary-B rows have `pass2_required == True`; 249 do not.

## Interpretation

> *Orbits whose F_beta closure required tight gauge minimization to land in
> `Z2_clean` are predicted stable; orbits whose closure was already clean at
> default tolerances are predicted unstable.*

This tests the **orbit-level gauge-rigidity hypothesis**: that an orbit's
resistance to coarse-gauge classification is a structural property
correlating with stability. The hypothesis is registered, not asserted.

## Non-Circularity Audit

```text
pass2_required is determined entirely by F_beta closure residual at two
gauge tolerances, BEFORE any stability label is consulted.

The pass2_required boolean was computed by the v0.4a two-pass sweep
which ran with no knowledge of stability and emitted a Z_2-clean/not
decision per row.

The stability label S/U is from the supp-B catalog, an independent
input parameter.

Predicting stability from pass2_required uses no stability information
in the predictor.

m_3 enters only as a stratification variable (chi-squared bins),
NOT as a fitted feature. The rule fires on pass2_required alone.
```

The rule could be wrong; it cannot be circular.

## Free Parameter Count

```text
Number of free parameters fitted to the data: 0
```

The threshold `pass2_required == True` is a binary feature defined
pre-data by the v0.4a two-pass tolerance schedule. The functional form
maps that boolean directly to a S/U prediction.

## Chi-Squared Degrees of Freedom

```text
df = (number of gating m_3 bins with N >= 5) - (number of free parameters)
   = 12 - 0 = 12
```

Reference distribution `chi-squared(12)`. Critical value `26.22` at
`p = 0.01`. Test statistic `chi^2 > 26.22` **falsifies this baseline
orbit-pass2 rule**.

Wording note (per Codex sign-off): the failure condition is recorded as
"falsifies this baseline orbit-pass2 rule", **not** "falsifies
gamma_3'_orbit_features globally". A subsequent gamma_3'_T_threshold
or gamma_3'_z0_threshold could be separately registered if the
diagnostic residuals suggest a different orbit feature.

## Bin-Level Predicted Stable Fraction

For each m_3 bin:

```text
P_S_predicted(m_3) = (count of rows in bin where pass2_required is True)
                     / N(m_3)
```

This is a deterministic function of v0.4a manifest data; no additional
fitting.

## Edge Cases (Pre-Registered)

1. **m_3 bins with 0 pass2_required rows**: predicted_S = 0 for all rows in
   that bin. If observed_S > 0, the bin contributes positively to
   chi-squared.
2. **m_3 bins with all pass2_required rows**: predicted_S = 1 for all rows
   in that bin (m_3=1.5, m_3=1.6, m_3=1.7 all have pass2 fractions visible
   in the v0.4a Pass 2 reclassification table). If observed_S < N(m_3),
   the bin contributes positively to chi-squared.
3. **Small bins (N < 5)**: m_3 in {1.3, 1.4, 1.9}. Reported as diagnostic
   only; not gating.

## Pre-Mortem Expectation

Before the verdict computation, the registered baseline has a known
pre-mortem signal:

```text
24 pass2_required rows, 11 S + 13 U:
  the rule labels all 24 as S; accuracy on these is 11/24 = 45.8%.

249 non-pass2_required rows, 86 S + 163 U:
  the rule labels all 249 as U; accuracy on these is 163/249 = 65.5%.

Overall accuracy: (11 + 163) / 273 = 63.7%.
Always-U baseline accuracy: 176 / 273 = 64.5%.

The rule is slightly worse than the trivial always-U predictor. So the
baseline is expected to fail chi-squared by a substantial margin; the
question is how much.
```

The registered baseline is therefore expected to falsify cleanly. That
falsification IS the v0.4b sub-result: "orbit-level gauge-rigidity as
measured by Pass 2 reclassification does not predict stability."

## Lock-In Statement

This functional form is committed before the verdict is computed. Any
change to the form after the verdict is a re-registration, not a
refinement.

Implementation may proceed against this form lock using the v0.4a
manifest at `results/isotrophy/k-facet-v04a-domain-map/manifest.json`
as the sole input; no new compute required.
