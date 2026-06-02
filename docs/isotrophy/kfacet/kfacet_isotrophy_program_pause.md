# K_facet Isotrophy Program Pause (2026-05-24)

Status: **PAUSED 2026-05-24** at end-of-v0.9, then **ADVANCED off-pause
2026-05-29 -> 2026-06-02** with the v0.10 frontier pair (v0.10a + v0.10b), the
v0.11 conditional-rank close, and the v0.12 -> v0.16 external-transfer arc, which
reached a **clean Tier-2 external PASS at v0.16** (`tail_resolved_transfer_passes_clean`,
AUC_cond 0.647, p 1e-5): the velocity-fraction signal transfers to an independent
catalog once read as a continuous tail-resolved score rather than the coarse v0.11 zone.
The chain ran from v0.3 (December 2025 / January 2026, pre-anniversary)
through v0.9 (May 2026) as seven sequential pre-registered chapters with
seven distinct chapter-close types and one load-bearing in-sample positive
(v0.7a' chi^2 = 16.43 on signed velocity-fraction quartiles, 250 analyzable
rows), then the frontier-advance matured that positive into a **publishable
conditional-positive sequence**:

- **v0.10a (the positive):** the in-sample monotone velocity-fraction ->
  stability trend, registered by the phenomenon-aligned J-T one-sided trend
  test (exact fixed-margin enumeration, p = 7.3e-3 < 0.01) that the blunt
  chi^2-of-independence had missed.
- **v0.10b (the boundary):** the trend does NOT survive as a mass-MARGINAL
  held-out predictor (pooled leave-one-m_3-bin-out AUC = 0.41 < 0.5), yet it
  DOES rank within each held-out mass bin (per-fold AUC mean ~0.65; the
  observed pooled AUC still beats the within-m_3-shuffled null of 0.265 at
  p = 1e-4). The signal lives at the (m_3, zone) JOINT level.
- **v0.11 (the conditional close):** the explicit within-m_3 rank test passes on
  the locked 229-row primary domain (`AUC_cond = 0.6783`, exact stratified
  `p = 2.046e-7`; permutation sanity consistent). This registers the
  conditional rank signal while preserving v0.10b's global held-out null.
- **v0.12 (the external-transfer boundary):** the conditional rule does NOT yet
  transfer to an independent catalog. The nearest target, same-paper
  supplementary-A, is `external_transfer_blocked_by_attrition`: an unbiased uniform
  probe (300 rows, 6 shards) reads attrition 0.3433, Wilson95 [0.2919, 0.3987] --
  the whole CI above the locked 0.20 block gate. NOT a falsification: the frozen
  v0.7 D5 measurement is numerically intractable on ~1/3 of supp-A. The ~2.5 h
  probe spared the projected 6.7-day full run.
- **v0.13 (the target search):** a signal-blind search returned an independent-target
  landscape negative -- only Tier-2 Li/Liao 2021 (135,445 non-hierarchical orbits, same
  lineage) was schema-viable. v0.13a bounded cross-ansatz leakage at 0.0 but found the
  raw dominant-direction vf FEATURE frame-relative; v0.13b priced the coarse v0.11 ZONE
  as frame-stable enough to test (supp-B 4.35%, liao2021 pooled ~1.3%).
- **v0.14 (coverage-undecidable):** the frozen coarse-zone rule on a 1280-row sampled
  liao2021 draw landed `sample_transfer_undecidable_coverage` -- only 7/16 mass cells
  could host a within-cell stable-vs-unstable comparison (liao2021 is overwhelmingly
  unstable and zone-2-saturated).
- **v0.15 (directional-weak):** an outcome-balanced 80/80 case-control draw over the 7
  stable-support cells defeated the coverage wall but landed
  `stable_support_transfer_directional_weak` (AUC_cond 0.5125, p 3e-4): direction
  transfers and is significant, but the coarse zone is saturated (98.4% zone-2) so the
  effect sits below the 0.55 floor -- the binning, not the projection, is the bottleneck.
- **v0.16 (the external PASS):** a tail-resolved continuous 4-frame ensemble-median vf
  score on a fresh doubly-held-out 80/80 draw landed
  `tail_resolved_transfer_passes_clean` (AUC_cond 0.647, p 1e-5, attrition 0, frame
  gate clean). On the SAME rows the coarse zone reads 0.510 -- direct proof the binning
  was the bottleneck. Pooled and mass-cell-heterogeneous (5 of 7 cells in-direction, 2
  reversed; leave-one-out 0.593 still clears the floor), Tier-2 and bounded, not
  theorem-facing.

The mature, falsifiable claim is **"yes, but only after conditioning"**:
velocity-fraction stratifies three-body stability, but as a function of the
mass ratio, not marginally over it. That is a stronger and more honest
statement than either a bare in-sample positive or a bare held-out null —
the held-out chapter localized the signal rather than killing it.

The program remains **paused at the lab-bandwidth level**, not retired. The
within-m_3 conditional rung that v0.10b motivated is closed (v0.11); the
external-catalog rung is now **CONFIRMED at Tier-2** -- v0.12's supp-A D5-attrition
wall and v0.13's landscape-negative left Li/Liao 2021 as the only viable target, and
the v0.14 (undecidable) -> v0.15 (directional-weak) -> v0.16 (clean PASS) arc carried
it to a held-out external confirmation of the continuous velocity-fraction projection.
Remaining axes: the v0.16 mass-cell heterogeneity (which regions carry the transfer,
why two reverse), a larger supported-region or multi-feature chapter, or new-mechanism
work -- not another rephrasing of the same table.

Audience: paper-side writers, future coding agents, isotrophy-
program lab initiates, anyone who wants to know what's left.

## What the Isotrophy Program Did

Sundog v. Isotrophy tested whether a "sundog theorem" makes
non-tautological predictions about Li & Liao 2025's three-body
catalog (specifically the supplementary-B piano-trio catalog of
273 m_3-parametrized periodic orbits, half stable and half
unstable). The program ran as seven sequential pre-registered
chapters testing progressively richer low-dimensional projections
of the catalog body:

```text
v0.3:  Gamma_i rank gate (D_3-equivariant; m_3 = 1 strict G.2).
       Outcome: domain-of-applicability. 20/21 m_3 = 1 strict-G.2
       rows resolve as structural zeros; O_617 is a clean
       opposite-strict row with a bridge direction outside the
       valid D3 representation. Daughter catalog (supp-B) is
       Z_2-or-smaller and OUT of D_3-domain.

v0.4:  Z_2 shadow on supp-B (tangent-isotypic + orbit-gauge-
       rigidity).
       Outcome: structural-negative. Both registered Z_2-shadow
       predictors fail. gamma_3 retired pre-sweep (F_beta does
       not preserve K_fib on supp-B). gamma_3'_orbit_pass2
       falsified at chi^2 = 1202 (vs critical 26.22).

v0.5:  2-bit catalog branch shadow on (m_3 < 1, z_0 < 0.3).
       Outcome: projection-limit (audit-passes / predictor-fails).
       v0.5a audit passes chi^2 = 34.99, p = 1.23e-7. v0.5b
       held-out predictor fails: accuracy delta -0.019, McNemar
       p = 1.0. Branch hash describes but does not predict.

v0.6:  Continuous catalog-coordinate (E, |L|) shadow.
       Outcome: conditional-independence. v0.6a catalog audit
       passes chi^2 = 33.70 but alignment-tightness 0.956 trips
       the warning guard. v0.6b within-branch audit (alignment-
       breaking) fails permutation p = 0.029. Energy quartile is
       a 1-to-1 label for m_3 sub-bin within the stable branch.

v0.7:  Floquet velocity-fraction direction shadow.
       Outcome: integration-attrition + qualified-positive on
       restricted domain. v0.7a catalog-wide audit attrited at
       8.4% (variational integrator step-size wall at high m_3).
       v0.7a' restricted-domain confirmation on 250 analyzable
       rows PASSES at chi^2 = 16.43, p = 9.3e-4, alignment 0.698.
       FIRST non-branch-aligned positive in the chain.

v0.8:  Unsigned direction-purity = abs(vf - 0.5).
       Outcome: structural-negative. v0.8a fails at chi^2 = 4.94.
       Diagnostic confirms the v0.7a' signal is signed (linear-
       transform invariance of vf preserves the quartile chi^2).
       Density-asymmetric vf distribution (74% below 0.5)
       violates the purity transform's symmetry assumption.

v0.9:  Signed three-zone direction-composition under physical
       cutpoints {0.25, 0.50}.
       Outcome: structural-negative on U-shape hypothesis +
       monotone-increasing meta-finding. v0.9a fails at
       chi^2 = 7.42 (vs critical 9.21); U-shape pattern check
       also fails. Under physical zones, the actual pattern is
       monotone-increasing in vf (positional 11% S, mixed 34% S,
       velocity-heavy 44% S). The v0.7a' U-shape was a quartile-
       boundary artifact at vf ~ 0.297.

v0.10a: Jonckheere-Terpstra one-sided trend test on the v0.9a
        three-zone contingency (phenomenon-aligned statistic;
        the Navier-Stokes "stop using a blunt diagnostic against
        the wrong shape" lesson applied in-program).
        Outcome: registered-positive (in-sample). Exact fixed-
        margin enumeration (1,340-table multivariate-hypergeometric
        null) one-sided p = 7.304e-3 < 0.01. The monotone trend the
        chi^2-of-independence (p = 0.0245) missed is registered under
        the aligned ordered statistic. Direction post-hoc from v0.9a
        -> in-sample only, never held-out.

v0.10b: leave-one-m_3-bin-out held-out predictor (zone-only
        weighted-PAVA monotone risk score) on the 234 primary rows.
        Outcome: conditional-positive / mass-marginal null. Pooled
        held-out AUC = 0.4125 <= 0.5 FAILS the registered gate, but
        per-fold (mass-conditioned) AUC is mostly > 0.5 (mean ~0.65;
        within-m_3 permutation p = 1e-4 beats the shuffled null 0.265).
        The vf -> stability signal is real but lives at the (m_3, zone)
        JOINT level, not the marginal zone level; a zone-only score
        cannot use the mass base rate that dominates marginal ranking.
        Pairs with v0.10a as the publishable "yes, but only after
        conditioning" result.

v0.11:  within-m_3 conditional vf rank test on the locked 229-row
        both-class primary domain.
        Outcome: conditional-rank positive. AUC_cond = 0.6783,
        exact stratified p = 2.046e-7, exact probability mass = 1.0,
        and the within-stratum permutation sanity sidecar is consistent.
        This registers the conditional signal explicitly, while leaving
        v0.10b's mass-marginal held-out null intact.

v0.12:  external transfer of the frozen v0.11 rule to same-paper
        supplementary-A (10,059 rows; 8,320 candidates after overlap
        quarantine + strict-m3=1 exclusion; reflection-overlap 0).
        Outcome: external_transfer_blocked_by_attrition. An unbiased
        uniform probe (300 rows, 6 shards, seed 20260523) reads
        attrition_fraction = 0.3433, Wilson95 [0.2919, 0.3987] -- the
        entire CI above the locked 0.20 block gate, systemic across m3.
        The frozen v0.7 D5 measurement (DOP853, 1e-12) is numerically
        intractable on ~1/3 of supp-A. NOT a falsification of the vf
        hypothesis. The ~2.5 h probe spared the projected 6.7-day run.

v0.13:  external target-search chapter (pre-feature source selection).
        Outcome: independent-target landscape NEGATIVE -- every Tier-3 catalog
        failed a hard gate (equal-mass / restricted / too small); only Tier-2
        Li/Liao 2021 (135,445 non-hierarchical orbits, same lineage) was
        schema-viable. Four-layer anti-target-shopping firewall held.

v0.13a: liao2021 adapter + cross-ansatz leakage preflight. Leakage bounded at
        0.0 (expansion-only adapter), BUT the raw dominant-direction vf FEATURE
        is frame-relative (select_gamma_1 argmax flips under rotation). The
        coarse v0.11 ZONE, not raw vf, is the transferable object.

v0.13b: frame-zone stability audit. coarse_zone_rule_frame_stable_enough_to_test
        (supp-B zone-change 4.35%, liao2021 pooled ~1.3%; fragility localized to
        two mass bands). Cleared the coarse zone for transfer.

v0.14:  sampled coarse-zone transfer (1280 rows, 16 mass cells).
        sample_transfer_undecidable_coverage -- only 7/16 cells hosted both
        stability classes (liao2021 is overwhelmingly unstable + zone-2-saturated);
        coverage gate (>=10 cells / >=800 rows) not met. No transfer read licensed.

v0.15:  stable-support case-control transfer (7 cells x 80 S + 80 U, v0.14-held-out).
        Coverage wall defeated (7/7 primary, 0 attrition) ->
        stable_support_transfer_directional_weak (AUC_cond 0.5125, p 3e-4). Direction
        transfers + significant, but the coarse zone is saturated so the effect is
        below the 0.55 floor. Diagnosis: the binning, not the projection.

v0.16:  tail-resolved transfer (4-frame ensemble-median continuous vf, fresh
        double-holdout 80/80). tail_resolved_transfer_passes_clean: AUC_cond 0.647,
        p 1e-5, attrition 0, frame-spread gate clean. SAME-rows coarse zone reads
        0.510 -> the coarse binning was the bottleneck; the continuous projection
        transfers. First clean external (Tier-2, held-out) confirmation. Pooled +
        mass-cell-heterogeneous (5/7 in-direction, 2 reversed; leave-one-out 0.593);
        Tier-2, bounded, not theorem-facing.
```

## The Load-Bearing Findings

After seven chapters plus the v0.10-v0.13 frontier-advance, three findings
carry the program's substantive output (the second is now a publishable
conditional profile, bounded by an external-transfer null):

1. **The v0.7a' chi^2 = 16.43 positive**. On the 250-row analyzable
   supp-B subset (post v0.7a integration-attrition), the velocity-
   fraction quartile contingency is statistically significant
   (p = 9.3e-4) AND branch-independent (alignment 0.70 against the
   v0.5a branch hash). This is the FIRST non-branch-aligned
   positive in the chain and remains uncontradicted by v0.8 and
   v0.9 (both of which contextualized it without invalidating it).

2. **The v0.10a + v0.10b + v0.11 conditional profile -- "yes, but only after
   conditioning."** The v0.9a monotone-increasing meta-finding
   (S_fraction 11% / 34% / 44% across positional / mixed / velocity-
   heavy zones under physical cutpoints {0.25, 0.50}) was matured into
   a registered result and then bounded:

   - **v0.10a (the positive).** The Jonckheere-Terpstra one-sided
     trend test — the ordered-alternative statistic the chi^2-of-
     independence was too blunt to see — registers the trend at exact
     fixed-margin enumeration p = 7.304e-3 < 0.01 (permutation sanity
     7.50e-3; normal-approx 6.73e-3). The signal is real and ordered,
     in-sample. (Direction post-hoc from v0.9a, disclosed; in-sample
     only, never a held-out claim.)
   - **v0.10b (the boundary).** As a mass-MARGINAL held-out predictor
     the signal FAILS: a zone-only weighted-PAVA monotone risk score
     under leave-one-m_3-bin-out lands at pooled AUC = 0.4125 < 0.5,
     missing the registered AUC>0.5 gate. But CONDITIONED on mass it
     holds — per-fold held-out AUC is mostly > 0.5 (m_3 0.8->0.84,
     0.9->0.90, 1.0->0.81, 1.1->0.82; mean ~0.65), and the observed
     pooled AUC still beats the within-m_3-shuffled null of 0.265 at
     permutation p = 1e-4. The failure is cross-bin CALIBRATION, not a
     within-bin signal failure: the vf -> stability relationship lives
     at the (m_3, zone) JOINT level, and the mass base rate — which a
     zone-only score cannot use — dominates any marginal ranking.
   - **v0.11 (the conditional close).** The explicit within-m_3
     conditional rank test PASSES on the frozen 229-row primary domain:
     `AUC_cond = 0.6783`, exact stratified `p = 2.046e-7`, exact-null
     mass = 1.0, permutation sanity consistent. This is still in-sample
     and conditional, not an external-catalog predictor and not a v0.10b
     rescue.

   Together these are a **publishable conditional profile** with a mature, falsifiable
   shape: **velocity-fraction stratifies three-body stability, but only
   after conditioning on the mass ratio — it is a conditional predictor,
   not a marginal one.** This is a stronger and more honest statement
   than either a bare in-sample positive or a bare held-out null; the
   held-out chapter LOCALIZED the signal rather than killing it, and v0.11
   registered that localization under the exact conditional rank gate.

   **The external-transfer arc (v0.12 -> v0.16): from supp-B-internal to a clean Tier-2
   PASS.** v0.12 took the frozen rule to same-paper supplementary-A and hit
   `external_transfer_blocked_by_attrition` (uniform probe attrition 0.3433, the whole
   CI above the 0.20 gate; a measurement-feasibility limit, not a falsification).
   v0.13's signal-blind search found only Tier-2 Li/Liao 2021 viable; v0.13a bounded
   leakage at 0.0 but exposed raw vf as frame-relative; v0.13b certified the coarse
   v0.11 zone as frame-stable enough to test. The transfer then ran in three rungs on
   liao2021: v0.14 `undecidable_coverage` (only 7/16 cells hostable), v0.15
   `directional_weak` (AUC_cond 0.5125, p 3e-4 -- direction transfers but the coarse
   zone is 98.4% saturated, below the 0.55 floor), and v0.16
   `tail_resolved_transfer_passes_clean` (**AUC_cond 0.647, p 1e-5**) once the feature
   was read as a continuous 4-frame ensemble-median vf rather than the coarse zone. On
   the SAME held-out rows the coarse zone reads 0.510 -- direct proof the binning, not
   the projection, was the bottleneck. **The conditional positive now has a held-out
   external (Tier-2) confirmation of the underlying velocity-fraction projection** --
   pooled and mass-cell-heterogeneous (5 of 7 cells in-direction, two reversed;
   leave-one-out 0.593 still clears the floor), bounded to Tier-2 / stable-support /
   not theorem-facing. The coarse v0.11 zone FORM does not transfer (v0.15); its
   continuous tail-resolved relative does (v0.16).

3. **The methodology surface**. The chain produced and locked
   a comprehensive set of pre-registration discipline tools that
   are reusable across mechanism-search programs:

   - Closure-relative gate discipline (v0.3 origin).
   - Two-pass classifier with empirically-calibrated tolerances
     (v0.4a).
   - Projection-language vocabulary (body / shadow / projection-
     limit, from CROSS_SUBSTRATE_NOTES.md).
   - Audit-then-predictor separation (v0.5).
   - Asymmetric McNemar + positive-delta falsifier for held-out
     predictors (v0.5b).
   - Alignment-tightness guard against prior-chapter feature
     shadows (v0.6a, with thresholds 0.8 warning / 0.95 severe).
   - Sparse-cell fallback tree (v0.6b: chi-squared / exact
     permutation / inconclusive_sparse, with locked permutation
     seed 20260523 and n_permutations = 10000).
   - Non-circularity sentence template for Floquet-derived
     features (v0.7a: "use eigenvectors as geometric directions
     only").
   - Per-row integration-failure fallback with attrition gate
     at 5% of catalog (v0.7a R2.A).
   - Append-per-row resume mode for crash resilience (v0.7a R2.C).
   - Anti-circular framing discipline for sequential audits on
     the same data (v0.9 parent).
   - Phenomenon-aligned statistic substitution: replace a blunt
     omnibus test with the ordered-alternative statistic that matches
     the discovered shape (v0.10a J-T — the Navier-Stokes lesson
     applied in-program), using exact fixed-margin enumeration as a
     deterministic primary p-value with permutation/normal-approx as
     sanity sidecars.
   - Pooled-vs-per-fold held-out diagnostic: report per-fold AUC
     alongside the pooled gate so a cross-bin calibration artifact is
     exposed as a conditional positive rather than hidden inside a
     failing marginal statistic — and so the goalposts stay locked
     (pooled AUC primary) while the per-fold structure is read as a
     diagnostic, not a post-hoc rescue (v0.10b).

## Concrete Reopening Avenues for Lab Initiates

Things that could be done if a lab initiate has bandwidth to "waste
compute on." Each is a real research direction that the principal
deemed not worth the bandwidth tradeoff against three-body
Phase 15+, but which would yield publishable substance if
followed up.

### 1. Jonckheere-Terpstra trend test on v0.9a three-zone data (LOW EFFORT, HIGH SUBSTANCE)

**Status**: **EXECUTED 2026-05-29 as v0.10a** — `kfacet_v10a_jt_trend_form.md`,
`scripts/isotrophy_jt_trend.py`, `results/isotrophy/k-facet-v10a-jt-trend/`. Verdict
**`jt_trend_monotone_registered`**: exact fixed-margin enumeration (1,340-table
multivariate-hypergeometric null) one-sided **p = 7.304e-3 < 0.01** (permutation
sanity 7.50e-3; normal-approx 6.73e-3). The monotone velocity-fraction → stability
trend that v0.9a's χ² (p=0.0245) missed is registered under the aligned ordered
statistic — **in-sample only** (the direction was post-hoc from v0.9a). The held-out
predictor (avenue 2, reserved `v0.9b`) remains the next rung for a predictive claim.

**Hypothesis**: the three-zone contingency `S_fraction = 0.105 /
0.339 / 0.439` shows a monotone-increasing trend in vf that would
pass at p < 0.01 under a J-T trend test (which has more power than
chi-squared for ordered alternatives).

**Implementation**: ~30 minutes paper-side + seconds runner. Read
`results/isotrophy/k-facet-v09a-signed-vf-three-zone/per_row_table.csv`,
apply J-T trend test on zone-ordered S/U using scipy.stats.
Pre-register the one-sided test against H_0 = no monotone trend.

**Non-circularity**: this breaks condition #2 (test statistic)
of the v0.9 parent's anti-circular discipline; J-T is NOT chi-
squared of independence. Allowed under the locked framework.

**Why deferred**: another chapter's worth of paper-side work after
seven chapters had already been written; bandwidth shifted to
Phase 15+.

### 2. v0.7a' held-out predictor on the analyzable subset (LOW-MEDIUM EFFORT)

**Status**: **EXECUTED 2026-05-30 as v0.10b** —
`kfacet_v10b_monotone_vf_heldout_predictor_form.md`,
`scripts/isotrophy_vf_heldout_predictor.py`,
`results/isotrophy/k-facet-v10b-monotone-vf-heldout/`. Verdict
**`monotone_vf_predictor_fails_heldout`** — a clean **projection-limit** result, the
audit-vs-predictor pattern for the THIRD time (cf. v0.5). The pre-registered primary
gate (pooled leave-one-`m_3`-bin-out held-out AUC of a zone-only weighted-PAVA risk
score) lands at **AUC = 0.4125 ≤ 0.5 → FAIL** (gate needs AUC>0.5 AND permutation
p≤0.01). The mandated per-fold diagnostic shows the failure is a **cross-bin
calibration artifact, not a within-bin signal failure**: per-fold held-out AUCs are
mostly >0.5 (m_3 = 0.8→0.84, 0.9→0.90, 1.0→0.81, 1.1→0.82; mean ~0.65), and the
within-`m_3` permutation p is tiny (observed 0.41 beats the shuffled-label null of
0.265 → genuine zone structure). What fails is the **pooled** cross-bin ranking — the
zone-only score is not comparable across `m_3` bins because the mass-bin base rate
(which the predictor cannot use) dominates global ranking. The v0.10a in-sample /
within-bin monotone trend is real but does **not** yield a globally-calibrated
held-out risk score. Per the locked verdict tree the goalposts stay where locked
(pooled AUC primary); the encouraging per-fold AUCs are diagnostic, not a rescue. A
within-`m_3` or `m_3`-conditional predictor would be a **fresh** chapter; that
fresh chapter is now v0.11, recorded above as the conditional-rank close. The
exact-`m_3`=0.4 risk noted below materialized as predicted (its fold AUC 0.671 is
strong but does not transfer pooled). Hard-label sidecar (non-gating) confirms the
base-rate trap: accuracy delta vs always-U = −0.167, McNemar one-sided p = 0.99999.

_Original registration text (pre-execution) follows._

**Status (original)**: parent registration mentions v0.7b as a candidate;
NEVER licensed under the v0.7a attrition verdict; could be
licensed under v0.7a' restricted-scope PASS retrospectively if
re-registered.

**Hypothesis**: the v0.7a' chi^2 = 16.43 in-sample positive is
held-out-predictive at p <= 0.01 under leave-one-m_3-bin-out on
the 250 analyzable rows.

**Implementation**: re-register `kfacet_v07b_*` paper-side as a
held-out predictor on the restricted scope (v0.7a' domain). Use
v0.5b's discipline verbatim (asymmetric McNemar, positive-delta
gate, leave-one-m_3-bin-out partition). Per-fold accuracy + pooled
McNemar at p <= 0.01.

**Risk**: m_3 = 0.4 is the load-bearing fold; held-out it may flip
the bucket-majority direction (same failure mode v0.5b exhibited).
Lab initiate should run v0.5b-style diagnostic per fold.

**Non-circularity**: domain is the v0.7a' subset (already used in
v0.7a' for in-sample chi-squared); held-out partition breaks the
in-sample-vs-held-out condition cleanly.

### 3. v0.7a relaxed-precision re-run (MEDIUM-HIGH EFFORT)

**Status**: never registered. The v0.7a R1 amendment locked
symplecticity gate at 1e-4 (relaxed from 1e-6) on the variational
precision 1e-12 substrate; the integration attrition came from
DOP853 step-adapter failures, not sanity-gate failures.

**Hypothesis**: at variational precision 1e-10 or 1e-9 (matching
v0.4a Pass 2), the DOP853 step adapter clears the 23 attrited
rows, allowing a catalog-wide v0.7a verdict.

**Implementation**: register a v0.7c paper-side at relaxed
precision. Re-run the full audit (~3-5 hours expected). The
restricted-domain v0.7a' verdict would be either confirmed (the
attrited rows match the analyzable subset's pattern) or contested
(the attrited rows have systematically different vf values).

**Risk**: precision change affects every per-row vf value; the
catalog-wide chi-squared at the new precision is its own audit and
cannot be compared to v0.7a' directly without explicit precision
disclosure on every receipt.

**Non-circularity**: catalog-wide claim under the new precision is
a fresh audit, not a re-test of v0.7a'.

### 4. m_3 = 0.4 sub-catalog targeted mechanism investigation (MEDIUM EFFORT, HIGH SUBSTANCE)

**Status**: never registered. The m_3 = 0.4 cluster has been the
load-bearing positive throughout the chain (v0.7a Q1 49% S; v0.5a
(m_3<1, z_0<0.3) branch 56% S).

**Hypothesis**: the m_3 = 0.4 sub-catalog (55 rows / 35 S / 20 U)
has a substrate-specific mechanism that the catalog-wide audits
average out. A within-m_3-bin audit on this single mass ratio
would reveal what makes the bin special.

**Implementation**: register a v0.10 chapter on the m_3 = 0.4
sub-catalog. Use any of the prior mechanism families (vf, |L|,
joint, gamma_1 direction) but restrict the domain to 55 rows.
Apply v0.6b-style within-stratum discipline.

**Risk**: low N (55 rows) limits chi-squared power; sparse-cell
fallback will likely fire frequently. The substantive return is
high if the mechanism is clean; the statistical return is
power-limited.

**Non-circularity**: the m_3 = 0.4 sub-catalog is a domain
restriction; conditioning on m_3 = 0.4 uses catalog parameter
only, not the stability label.

### 5. Joint (vf, Q_E) audit on the 250 analyzable rows (MEDIUM EFFORT)

**Status**: never registered. v0.6a (E quartile, chi^2 = 33.70)
and v0.7a' (vf quartile, chi^2 = 16.43) both passed in-sample;
the relationship between the two has not been characterized.

**Hypothesis**: vf and Q_E carry overlapping but partially
independent stability information. A joint 4x4x2 contingency
(Q_vf x Q_E x S/U) would reveal whether one is a proxy for the
other or whether they explain different variance in S/U.

**Implementation**: register a v0.10 chapter on joint
(vf, Q_E) stratification on the 250 analyzable rows. Compute the
4x4x2 contingency; run a stratified chi-squared (Cochran-Mantel-
Haenszel) to test conditional independence of Q_vf and S/U given
Q_E.

**Risk**: joint contingency is high-df; many cells will be sparse;
sparse-cell fallback fires.

**Non-circularity**: both Q_vf and Q_E are inherited from prior
chapters; the joint test is a NEW hypothesis (conditional
independence) that v0.6a and v0.7a' did not pre-register.

### 6. Cross-substrate transfer test (HIGH EFFORT)

**Status**: cross-substrate notes exist at
`docs/CROSS_SUBSTRATE_NOTES.md` sections 6-7.

**Hypothesis**: the v0.7a' velocity-fraction shadow is substrate-
specific to supp-B; applying the same mechanism to a different
three-body sub-catalog (e.g., free-fall, choreography, or mesa
analog) would test whether the signal is supp-B-specific or
generalizes.

**Implementation**: identify a different three-body sub-catalog
with comparable structure (stability labels + periodic-orbit ICs).
Apply the v0.7a operational definition + the v0.9a three-zone
audit. Compare verdicts.

**Risk**: identifying a comparable catalog is non-trivial;
re-implementation of variational integration substrate is real
work. High effort, but high publishable substance if it works.

**Non-circularity**: cross-substrate transfer is by design a fresh
test on new data.

### 7. Action-angle / KAM-style decomposition (HIGH EFFORT, THEORETICAL)

**Status**: mentioned as a candidate v0.7 mechanism family before
v0.7 picked gamma_1 direction-of-instability; never registered.

**Hypothesis**: the supp-B orbits have action-angle representations
that stratify stability via integrability proximity. KAM theory
says nearly-integrable orbits are stable; computing action-angle
variables per row would test this directly.

**Implementation**: heavy theoretical machinery. Per-row
action-angle decomposition requires extended variational
integration over multiple periods (not just one) plus
diagonalization in canonical coordinates. Compute cost: weeks.

**Non-circularity**: action-angle variables are new geometric
quantities, not derived from prior-chapter features.

### 8. Full-catalog audit at variational precision 1e-10 with v0.7a R2.A discipline

**Status**: candidate alternative to avenue 3. If attrition at
1e-10 is below 5%, the catalog-wide v0.7a verdict could be
recovered without re-registering precision.

**Implementation**: re-run `scripts/v07a_velocity_fraction_audit.py`
with rtol = atol = 1e-10 (modify the locked constants paper-side
first). Compare attrition rates and pattern stability against
the 1e-12 result.

**Risk**: precision change is a re-registration; needs paper-side
discipline.

## Reusable Methodology Surface (For Programs Beyond Isotrophy)

The discipline tools listed under "load-bearing findings" item 3
are not isotrophy-specific. They apply to any mechanism-search
program where:

- Multiple candidate mechanisms compete on the same data,
- Sequential audits risk silent re-testing under different labels,
- Catalog-coordinate vs orbit-dynamics axes need to be distinguished
  for non-circularity,
- Integration-precision walls may limit catalog-wide claims,
- Held-out generalization is the publishable predictor gate.

Future mesa, geometry, or three-body Phase 15+ work that follows
the same pattern is invited to inherit these disciplines as
appropriate.

## Bandwidth Redirect

Per the principal's pre-v0.7 direction (which the v0.7a' surprise
positive temporarily overrode), bandwidth redirects to:

- **three-body Phase 15+** (survival pocket investigation, the
  current high-ROI path).
- **Coarse-graining proof trunk** (P0/P1 priority per
  `docs/TODO.md`).
- **Public surface updates** including `threebody.html` (flagged
  as seriously stale during v0.7 close).

The isotrophy section's pause does NOT release the program's
discipline; any future re-opening must respect the locked
methodology surface AND the anti-circular framing discipline for
sequential audits on the v0.7a' analyzable subset.

## Doc Trail

- `kfacet_v03h_writeup.md` -- v0.3 chapter close.
- `kfacet_v04_writeup.md` -- v0.4 chapter close.
- `kfacet_v05_writeup.md` -- v0.5 chapter close.
- `kfacet_v06_writeup.md` -- v0.6 chapter close.
- `kfacet_v07_writeup.md` -- v0.7 chapter close (load-bearing
  qualified-positive).
- `kfacet_v08_writeup.md` -- v0.8 chapter close.
- `kfacet_v09_writeup.md` -- v0.9 chapter close (this chapter's
  companion).
- `kfacet_v10a_jt_trend_form.md` -- v0.10a ordered-trend positive.
- `kfacet_v10b_monotone_vf_heldout_predictor_form.md` -- v0.10b
  mass-marginal held-out null / conditional diagnostic.
- `kfacet_v11_m3_conditional_vf_rank_form.md` -- v0.11 within-m3
  conditional rank pass.
- `docs/CROSS_SUBSTRATE_NOTES.md` -- cross-substrate
  framing (avenue 6).
- `docs/isotrophy/sundog_v_isotrophy.md` -- full chapter-by-chapter dated log.

---

Isotrophy program paused at end-of-v0.9, then reopened through the v0.10/v0.11
conditional-rank frontier. The live result is now sharper: velocity-fraction is a
real in-sample, within-m3 stability rank signal, but not a mass-marginal held-out
predictor. Reopening is invited; mechanism-laundering is not.
