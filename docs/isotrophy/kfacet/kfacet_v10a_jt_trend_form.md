# v0.10a Jonckheere–Terpstra Trend Test Form Lock

Status: **VERDICT LANDED 2026-05-29: `jt_trend_monotone_registered`** — exact
fixed-margin enumeration one-sided **p = 7.304e-3 < 0.01**. Locked 2026-05-29; the
adjustment pass made the primary p-value
a deterministic **exact fixed-margin enumeration** (the 1,340-table
multivariate-hypergeometric null over S-allocations across zone sizes 19/165/66 at
fixed total S=87), demoted the 10,000-shuffle permutation to a **secondary sanity
check**, and strengthened the frozen-input gate to assert **exact** zone/S/U counts
and recompute each row's zone from `velocity_fraction` under {0.25, 0.50}. No
v0.10a runner has been written and no v0.10a command has been run at lock time.
This document locks the test, statistic, exact-enumeration primary p-value,
secondary permutation sanity check, significance floor, one-sided direction,
non-circularity argument, and verdict tree **before** the J statistic is computed.

**Naming.** This is **v0.10a**, the program's canonical post-pause reopening number
(the v0.9 program-pause doc registers reopening avenues 4/5 as "a v0.10 chapter").
It executes **candidate D** of the v0.9 parent registration (Jonckheere–Terpstra),
which v0.9a held back. The name `v0.9b` is left **reserved** for the held-out
predictor (avenue 2), which v0.9a's fail did not license; that is the planned
follow-on, not this chapter.

## Verdict (Landed 2026-05-29)

Receipt: `results/isotrophy/k-facet-v10a-jt-trend/manifest.json`.

```text
verdict:                jt_trend_monotone_registered
frozen-input check:     PASSED (zones N=19/165/66, S=2/56/29, U=17/109/37 = 250/87/163;
                        zones recomputed from velocity_fraction under {0.25,0.50} match)
J_observed:             8760.5

exact fixed-margin enumeration (PRIMARY, binding):
  feasible tables:      1340
  probability mass:     1.0000000000   (enumeration self-check clean)
  one-sided p:          7.304e-3   <  alpha = 0.01   -> PASS

permutation sanity (secondary, seed 20260523, n=10000):
  one-sided p:          7.499e-3   (agrees with exact)
normal-approx (context, moments from the exact null):
  z = 2.471, one-sided p = 6.731e-3
```

**Reading.** The right statistic finds the signal the blunt one missed: v0.9a's
χ²-of-independence gave p=0.0245 (missed the 0.01 floor); the ordered J-T trend test
gives **p=0.0073** (clears it). The monotone-increasing velocity-fraction → stability
relationship (S-fraction 0.105 → 0.339 → 0.439) is now a **registered ordered trend**.

**Bound (per the Integrity Caveat).** This is an **in-sample** registration of a
post-hoc-discovered ordered direction — NOT held-out prediction, NOT an independent
confirmation, NOT a theorem-facing claim, NOT a change to the v0.3h K_facet
structural-null. The held-out predictor (the reserved `v0.9b`) remains the only path
to a predictive claim.

## Frame

The v0.4–v0.9 isotrophy chain produced one load-bearing positive: **v0.7a′
velocity-fraction → stability** (χ²=16.43, p=9.3e-4, branch-independent, on the
250-row analyzable supp-B subset). v0.9a then tested a **U-shape** hypothesis on
three physical zones and **failed at χ²=7.42** (vs critical 9.21, p=0.0245) — but
discovered the real pattern is **monotone-increasing** in velocity-fraction:

```text
zone                          N    S_fraction
positional-dominant (vf<0.25) 19    0.1053
mixed (0.25 ≤ vf < 0.50)     165    0.3394
velocity-heavy (vf ≥ 0.50)    66    0.4394
```

χ²-of-independence is a **blunt tool against an ordered claim**: it spends df on
arbitrary departures from independence and ignores the ordering of the zones. A
one-sided **Jonckheere–Terpstra trend test** is the statistic that matches the
hypothesis "S-fraction increases monotonically with velocity-fraction." This is the
phenomenon-aligned-test discipline (cf. the Navier-Stokes lesson: a fixed-threshold
diagnostic went vacuous until replaced by a phenomenon-aligned objective).

## Integrity Caveat (Load-Bearing — Disclosed Pre-Lock)

The **monotone direction was discovered in v0.9a**, not pre-registered before the
data. Candidate D was originally framed in the v0.9 parent under the v0.7a′
**U-shape** hypothesis ("pass if J-T *fails* to find monotonicity"); the v0.9a
discovery flipped the expectation. Therefore:

> v0.10a is an **in-sample, post-hoc** trend test of a data-derived ordered
> hypothesis on the same 250 rows. A PASS registers an **in-sample ordered trend**
> under the correct statistic; it is **NOT** held-out prediction and **NOT** an
> independent confirmation. The held-out predictor (avenue 2, the reserved `v0.9b`)
> remains the only path to a predictive claim.

This is allowed under the v0.9 parent's anti-circular rule because it breaks the
**test statistic** (J-T ≠ χ²-of-independence). The one-sided direction (increasing)
is post-hoc and is disclosed here; it is locked **before** the v0.10a runner reads
the data.

## What v0.10a Is

A trend test on the **frozen v0.9a receipt** — no new variational compute, no
re-derivation of velocity-fraction.

```text
Frozen input:   results/isotrophy/k-facet-v09a-signed-vf-three-zone/per_row_table.csv
                250 analyzable rows, columns: zone, stability (S/U),
                velocity_fraction. Inherited verbatim from v0.9a (which inherited
                vf + the 250-row analyzable subset from v0.7a). No re-binning:
                the three ordered zones are consumed as written.

Ordered groups: positional-dominant (0) < mixed (1) < velocity-heavy (2).
                Ordering is the locked vf-zone ordering, NOT data-driven.

Response:       stability coded S = 1, U = 0 (S = "more stable").

Hypotheses:     H0: no increasing monotone trend in S across the ordered zones.
                H1 (one-sided): S-fraction increases across positional → mixed →
                    velocity-heavy.

Statistic:      Jonckheere–Terpstra J = sum over ordered zone pairs (u < v) of
                    U_{uv}, where
                    U_{uv} = #{(i in u, j in v): y_j > y_i}
                             + 0.5 * #{(i in u, j in v): y_j = y_i}.
                With binary y this is
                    U_{uv} = (#U in u)(#S in v)
                             + 0.5 * [(#S in u)(#S in v) + (#U in u)(#U in v)].
                Larger J = stronger increasing trend.

p-value
(primary):      EXACT ENUMERATION of the finite fixed-margin permutation null.
                Enumerate every feasible allocation of S counts (s0, s1, s2)
                across the frozen zone sizes (19, 165, 66), preserving total
                S = 87 and U = 163. Weight each table by its multivariate-
                hypergeometric probability under random assignment of fixed
                S/U labels to fixed zone sizes:

                    P(s0,s1,s2) =
                      C(19,s0) C(165,s1) C(66,s2) / C(250,87)

                with ui = ni - si. Recompute J for each table and report the
                one-sided upper tail:

                    p_exact = sum_{J_table >= J_obs} P(table)

                This is deterministic, seedless, assumption-free under the
                fixed-margin null, and cheap here because only 1,340 feasible
                tables exist.

p-value
(secondary):    SEEDED MONTE CARLO PERMUTATION SANITY CHECK. Shuffle the zone
                labels across the 250 rows (preserving the per-zone counts and
                S/U multiset), recompute J, and take the one-sided upper tail
                    p_mc = (1 + #{J_perm >= J_obs}) / (1 + n_permutations).
                Locked for reproducibility only: seed = 20260523,
                n_permutations = 10000 (the program's v0.6b-locked permutation
                parameters). This is NOT binding unless the exact-enumeration
                implementation fails its probability-mass sanity check.

p-value
(context):      Normal approximation z = (J - E[J]) / sqrt(Var[J]) with the
                tie-corrected J-T null moments, reported for context only. Given
                the binary response, the exact-enumeration p is the binding
                verdict; the normal-approx p is disclosed as approximate.

Significance:   alpha = 0.01, ONE-SIDED (increasing). The program's locked floor.

Frozen-input
cross-check:    Before computing J, the runner recomputes the per-zone counts and
                S/U counts and ASSERTS exact count matches against the v0.9a
                manifest:

                    positional-dominant: N=19,  S=2,  U=17
                    mixed:               N=165, S=56, U=109
                    velocity-heavy:      N=66,  S=29, U=37
                    total:               N=250, S=87, U=163

                The runner recomputes S-fractions from those exact counts. It
                also recomputes each row's zone from `velocity_fraction` using
                the frozen cutpoints {0.25, 0.50} and ASSERTS it matches the
                stored `zone` column. A mismatch aborts the run (the input is not
                the frozen v0.9a receipt).
```

## Verdict Tree

```text
if exact per-zone counts / S-U counts != v0.9a manifest:
    ABORT  (frozen-input cross-check failed; not the v0.9a receipt)
elif recomputed zones from velocity_fraction != stored zone column:
    ABORT  (frozen-input cross-check failed; stale or inconsistent zone labels)
elif exact-enumeration p < 0.01 (one-sided increasing):
    verdict = jt_trend_monotone_registered
              -> the velocity-fraction -> stability trend is a registered
                 monotone (increasing) ordered trend under the aligned statistic
                 (IN-SAMPLE; not held-out).
else:
    verdict = jt_trend_not_significant
              -> the ordered trend does not clear the p=0.01 floor even under the
                 phenomenon-aligned test; the v0.9a meta-finding does not promote.
```

## Non-Circularity Argument

The v0.9 parent requires any reopening to break at least one of (feature
derivation, test statistic, evaluation domain) relative to v0.7a′'s
χ²-of-independence on vf quartiles.

```text
Feature derivation:   inherited (vf zones from v0.9a). NOT broken.
Evaluation domain:    inherited (same 250 analyzable rows). NOT broken.
Test statistic:       BROKEN. Jonckheere-Terpstra is a one-sided ordered-trend
                      test, NOT chi^2 of independence. It uses the zone ORDERING
                      (which chi^2 discards) and tests a directional alternative.
```

Exactly one condition broken (statistic), which the locked rule permits. The
**direction** of the one-sided alternative is post-hoc from v0.9a and is disclosed
in the Integrity Caveat; the verdict is explicitly an in-sample trend registration,
never an independent or held-out positive.

The locked Floquet non-circularity sentence carries forward verbatim:

> v0.10a uses Floquet eigenvectors only as geometric directions; it does not use
> eigenvalue magnitude, spectral radius, unit-circle status, unstable-pair count,
> or any threshold that defines the published S/U label.

## Free Parameter Count

```text
Number of free parameters fitted to the data: 0
```

The statistic (J-T), the zone ordering (positional < mixed < velocity-heavy), the
binary coding (S=1/U=0), the exact fixed-margin enumeration, the secondary Monte
Carlo permutation parameters (seed 20260523, n=10000), the significance floor
(0.01), the one-sidedness (increasing), and the verdict tree are all locked before
the runner reads the data. The one-sided direction is post-hoc-from-v0.9a but is
disclosed and fixed before this runner executes.

## Claim Boundary

```text
A PASS (jt_trend_monotone_registered):
  - registers an IN-SAMPLE monotone (increasing) ordered trend of
    velocity-fraction -> stability on the 250-row supp-B subset, under the
    phenomenon-aligned statistic that v0.9a's chi^2 lacked the power to detect.
  - does NOT claim held-out prediction (the reserved v0.9b held-out predictor is
    the next rung).
  - does NOT make isotrophy theorem-facing or restore a "sundog theorem predicts
    the catalog" claim.
  - does NOT change the v0.3h K_facet structural-null (this is the supp-B
    Z_2-projection thread; the m_3=1 D_3 K_facet gate is untouched).
  - does NOT touch three-body, Phase 15+, or any other substrate.

A FAIL (jt_trend_not_significant):
  - the v0.9a monotone meta-finding does not promote even under the correct test;
    the chapter closes the trend thread honestly.
```

## Receipt Schema (Planned)

```text
results/isotrophy/k-facet-v10a-jt-trend/
  manifest.json
    - mode = "v0.10a-jt-trend"
    - form_lock = "docs/isotrophy/kfacet/kfacet_v10a_jt_trend_form.md"
    - input_v09a_per_row_table
    - frozen_input_crosscheck: { per_zone_N, per_zone_S, per_zone_U,
                                 per_zone_S_fraction,
                                 zones_recomputed_from_velocity_fraction: bool,
                                 matches_v09a_manifest: bool }
    - zone_order: [positional-dominant, mixed, velocity-heavy]
    - response_coding: { S: 1, U: 0 }
    - J_observed
    - exact_enumeration: { feasible_table_count, probability_mass_total,
                           J_null_mean, J_null_p95, p_value_one_sided }
    - permutation_sanity: { seed: 20260523, n: 10000, J_null_mean, J_null_p95,
                            p_value_one_sided }
    - normal_approx: { E_J, Var_J_tie_corrected, z, p_value_one_sided }  (context)
    - alpha: 0.01
    - verdict: jt_trend_monotone_registered | jt_trend_not_significant
  jt_trend_receipt.json   (same content, flat)
```

## Lock-In Statement

This form is committed before the J statistic, the exact null, or the
normal-approximation moments are computed. Any change to the statistic, zone
ordering, response coding, exact-enumeration rule, secondary permutation
parameters, significance floor, one-sidedness, frozen-input cross-check, or
verdict tree after the v0.10a runner is executed is a re-registration, not a
refinement.

Implementation may proceed against
`results/isotrophy/k-facet-v09a-signed-vf-three-zone/per_row_table.csv` as the sole
input. No new variational compute is required. Expected runtime: seconds.

## Doc Trail

- `kfacet_v09_mechanism_preregistration.md` — parent registration; candidate D
  (J-T) source; anti-circular discipline.
- `kfacet_v09a_signed_vf_three_zone_form.md` — v0.9a (χ² three-zone, failed;
  monotone meta-finding); the frozen input for v0.10a; reserved `v0.9b` for the
  held-out predictor.
- `kfacet_v07a_prime_restricted_scope_form.md` — v0.7a′ PASS (χ²=16.43); the
  load-bearing positive whose shape this trend test characterizes.
- `kfacet_isotrophy_program_pause.md` — reopening avenues 1 (this) and 2 (held-out
  predictor); v0.10 reopening-chapter convention.

---

v0.10a: one-sided Jonckheere–Terpstra trend test on the frozen v0.9a 250-row
receipt. Breaks the test statistic (ordered-trend, not χ²-of-independence);
direction post-hoc-from-v0.9a and disclosed. Deterministic exact fixed-margin
enumeration primary; seeded 10,000-shuffle permutation sanity check and
normal-approx secondary; α=0.01 one-sided. PASS = registered in-sample monotone
trend (not held-out); held-out predictor (`v0.9b`) is the next rung. Zero free
parameters fitted to the data.
