# K_facet v0.5 Methodology + Result Handoff

Status: closed projection-limit (audit-passes-predictor-fails), 2026-05-23.
Audience: collaborators, paper-side writers, future coding agents.
Canonical sources: `kfacet_v05a_branch_map_form.md`,
`kfacet_v05b_branch_predictor_form.md`, and
[`../../CROSS_SUBSTRATE_NOTES.md`](../../CROSS_SUBSTRATE_NOTES.md)
sections 6-7 (projection-language framing).
Companion (closed v0.4 chapter): `kfacet_v04_writeup.md`.

## One-Line Read

v0.5 closes as a **projection-limit on the catalog-coordinate branch
shadow** of the Li-Liao supplementary-B piano-trio catalog. The 2-bit
branch hash on `(m_3 < 1, z_0 < 0.3)` is decisively associated with
stability in-sample (`chi^2 = 34.99` vs critical 11.34, `p ~= 1.23e-7`,
v0.5a) but does **not** predict held-out stability across mass bins
under a pre-registered fold-trained branch-majority rule
(`accuracy_delta = -0.019` vs always-U, `McNemar p = 1.0`, v0.5b). The
branch hash is a descriptive catalog partition, not a predictive
mechanism. The asymmetric result is itself the methodological finding:
a positive chi-squared audit does not license a held-out predictor.

## What v0.5 Tested

v0.5 opened after v0.4 closed as a structural-negative on the Z_2 shadow
as a stability projector. The v0.5 frame, locked at chapter open:

> Stop asking what symmetry the row has. Start asking which branch the
> row belongs to.

This trades the Z_2 tangent/orbit projection for a catalog-coordinate
indicator-bit projection (a "branch hash") and asks whether the supp-B
body's stability structure is visible in that projection.

Two pre-registered chapters, both paper-side first:

```text
v0.5a: branch-shadow audit
       Body:        supp-B piano-trio orbit as primary branch object
       Projection:  catalog branch hash on indicator bits drawn from
                    {b1 = m_3 < 1, b2 = z_0 < 0.3,
                     b3 = abs(v_z) < 1e-6, b4 = m_3 * z_0^2 < 2}
                    with deterministic constant-bit retirement.
       Observable:  contingency table[branch_label][stability]
       Question:    chi-squared independence of branch_label vs S/U.
       Result:      PASS at chi^2 = 34.99 vs critical 11.34
                    (chi-squared(3), p ~= 1.23e-7, 3.1x threshold).

v0.5b: branch-shadow predictor (held-out)
       Body:        same primary branch body.
       Projection:  same active (b1, b2) branch hash.
       Predictor:   fold-trained branch-majority on the 4-bucket hash.
       Partition:   leave-one-m_3-bin-out over 12 bins with N >= 5.
       Baseline:    always-U.
       Observable:  pooled paired accuracy on 263 gating rows.
       Question:    exact one-sided McNemar p <= 0.01 AND positive
                    accuracy delta?
       Result:      FAIL. accuracy_model = 0.6198 vs always-U 0.6388;
                    accuracy_delta = -0.019; win=28/loss=33/p=1.0.
```

## Audit Chain Methodology

The v0.5 chain inherited v0.3 + v0.4 discipline (closure-relative gates,
pre-registered constants, deterministic outcome categories, pre-mortem
discipline) and extended it with one new register:

**Audit-then-predictor separation.** v0.5a is registered as an AUDIT
(chi-squared independence on the full catalog), explicitly NOT a
predictor. A pass does not by itself establish a mechanism; it
licenses a separately-registered held-out predictor. v0.5b is
registered as the predictor with leave-one-m_3-bin-out folds, a
pre-registered tie rule (predict U), a pre-registered absent-branch
fallback (global majority, tie -> U), an asymmetric falsifier
(McNemar p <= 0.01 AND positive accuracy delta), and a sidecar
discipline that keeps the single-rule `S iff (m_3 < 1 and z_0 < 0.3)`
and a deterministic random-half split as report-only.

The asymmetric falsifier (both p AND positive delta required) prevents
"differ-from-baseline" passes that aren't "better-than-baseline."
The fold-trained majority avoids baking in the v0.5a observed
direction. The leave-one-m_3-bin-out partition respects bifurcation
structure; the deterministic random-half is sidecar only because
random splits can leak mass-bin structure across train and test.

Three stages, all paper-side first:

1. **v0.5a form lock (`kfacet_v05a_branch_map_form.md`)**. 4-bit
   candidate signature with deterministic constant-bit retirement.
   Pre-registered chi-squared(df) falsifier with `df =
   occupied_branch_count - 1`.

2. **v0.5a runner and verdict**. `scripts/v05a_branch_map_audit.py`
   ran the locked audit against the v0.4a manifest's 273 rows. Catalog
   degeneracy check retired b3 (constant FALSE on supp-B) and b4
   (constant TRUE on supp-B); active signature `(b1, b2)` gave 4
   occupied buckets {87, 17, 56, 113}. Verdict
   `branch_hash_passes_audit` at chi^2 = 34.99.

3. **v0.5b form lock + held-out predictor + verdict**.
   `kfacet_v05b_branch_predictor_form.md` locked the leave-one-m_3-bin-out
   fold-trained branch-majority rule before any held-out compute. Pre-mortem
   flagged m_3 = 0.4 as the load-bearing fold (35 S of 55 rows; the
   catalog's most-stable bin). `scripts/v05b_branch_predictor.py`
   reproduced the receipt under the locked rule. Verdict
   `branch_predictor_fails_heldout`; m_3 = 0.4 fold was indeed
   load-bearing.

## Result And Projection-Limit Verdict

```text
v0.5a verdict:   branch_hash_passes_audit
  chi^2 = 34.99 vs critical 11.34 (chi-squared(3), p = 0.01).
  p_value ~= 1.23e-7  (3.1x threshold).
  273 rows, 97 S / 176 U; 4 occupied branch buckets.
  audit-dominant bucket: (m_3 < 1, z_0 < 0.3) at 113 rows / 55.75% S.
  other buckets: 87 / 20.69% S, 17 / 29.41% S, 56 / 19.64% S.
  catalog mean: 35.53% S. small-mass low-z corner concentrates
  stability.

v0.5b verdict:   branch_predictor_fails_heldout
  263 gating rows over 12 leave-one-m_3-bin-out folds.
  accuracy_model     = 0.6198
  accuracy_always_U  = 0.6388
  accuracy_delta     = -0.019  (predictor LOSES to always-U)
  McNemar  win=28, loss=33, n_disc=61, p=1.0
  load-bearing fold: m_3 = 0.4 (55 rows / 35 S / 63.6% S).
  on its held-out fold, the residual five m_3 < 1 bins train the
  (m_3 < 1, z_0 < 0.3) bucket as U-majority (28 S / 33 U); the
  predictor cannot capture the m_3 = 0.4 stable cluster on its own
  held-out fold.
```

The publishable v0.5 statement:

> v0.5 registered a 2-bit catalog branch hash on the supp-B piano-trio
> catalog and tested it under two pre-registered protocols. The audit
> (v0.5a, chi-squared independence) passed decisively: the catalog
> branch is associated with stability at p ~= 1.23e-7. The held-out
> predictor (v0.5b, leave-one-m_3-bin-out fold-trained branch-majority
> vs always-U baseline) failed: accuracy delta -0.019, McNemar p = 1.0.
> **The branch hash describes the supp-B distribution but does not
> generalize across mass bins.** The in-sample positive is bin-local
> to the m_3 = 0.4 stable cluster; under leave-one-bin-out it cannot
> reproduce its own observed direction. The catalog branch shadow is a
> descriptive partition, not a predictive mechanism.

Three structural sub-results preserved in receipts and worth carrying
into v0.6:

1. **Bin-locality of the m_3 = 0.4 stable cluster on supp-B**. m_3 = 0.4
   has 35 S of 55 rows (63.6% stable) — the catalog's most-stable bin
   by a wide margin. Held-out, the rest of the m_3 < 1 sub-catalog
   cannot reproduce this cluster's S-majority direction. Either the
   mechanism is mass-specific to m_3 = 0.4, or the relevant feature is
   along an axis not aligned with the m_3 < 1 indicator.

2. **Audit-vs-predictor asymmetry as a methodological finding**. A
   chi-squared independence audit at p ~= 1e-7 did not survive a
   leave-one-bin-out predictor with the same active feature set. This
   is a load-bearing register for subsequent isotrophy chapters:
   passing audits license predictors, they do not establish them. The
   pre-registered v0.5b form caught this cleanly because it locked the
   asymmetric falsifier (both p AND positive accuracy delta required)
   and the partition (leave-one-m_3-bin-out, not random-half) before
   any held-out compute.

3. **Joint v0.4 + v0.5 projection-limit envelope on supp-B**. Stability
   information on this catalog is NOT carried by the Z_2 shadow at
   either tangent-isotypic or orbit-gauge-rigidity granularity (v0.4),
   AND is NOT carried as a held-out generalizable signal by the
   catalog-coordinate 2-bit branch shadow (v0.5). Any predictor with
   held-out stability content must look beyond both projections.

## Reproducibility Surface

Primary scripts:

```bash
# v0.5a (catalog-only, seconds):
python scripts/v05a_branch_map_audit.py

# v0.5b (catalog-only, seconds):
python scripts/v05b_branch_predictor.py \
  --input  results/isotrophy/k-facet-v05a-branch-map/per_row_table.csv \
  --out    results/isotrophy/k-facet-v05b-branch-predictor
```

Key receipt directories:

```text
results/isotrophy/k-facet-v05a-branch-map/manifest.json
results/isotrophy/k-facet-v05a-branch-map/contingency_table.csv
results/isotrophy/k-facet-v05a-branch-map/per_row_table.csv

results/isotrophy/k-facet-v05b-branch-predictor/manifest.json
results/isotrophy/k-facet-v05b-branch-predictor/per_fold_table.csv
results/isotrophy/k-facet-v05b-branch-predictor/per_row_predictions.csv
```

Load-bearing constants live in the form-lock documents:

```text
kfacet_v05a_branch_map_form.md   chi-squared critical = 11.34 (df = 3, p = 0.01),
                                  4-bit candidate with deterministic
                                  constant-bit retirement,
                                  active signature (b1, b2) on supp-B.

kfacet_v05b_branch_predictor_form.md
                                  leave-one-m_3-bin-out partition,
                                  fold-trained branch-majority predictor,
                                  tie rule predict U, absent-branch fallback,
                                  asymmetric falsifier (McNemar p <= 0.01 AND
                                  positive accuracy delta),
                                  inconclusive verdict at n_disc < 10.
```

## Where v0.6 Opens

The v0.5 projection-limit close is asymmetric, not a pure
structural-negative. The audit pass is a real catalog stratification
finding; the held-out fail is a real generalization gap. Together they
constrain the next chapter:

> v0.6 must use a projection richer than a 2-bit catalog hash, or it
> must be a different mechanism family entirely. Continuous-feature
> promotion within the catalog-coordinate axis (v0.5c) is the obvious
> path but inherits v0.5b's bin-locality risk. A different projection
> — orbit-level conserved quantities, bifurcation tracks across m_3,
> direction-of-instability sidecar, or action-angle / KAM-style
> decomposition — is the natural alternative.

Candidate v0.6 mechanism families, all paper-side first, in rough order
of inheritance from existing receipts:

1. **Conserved-quantity (E, |L|) stratification.** Orbit-level
   invariants verified tangent to v_bridge directions in v0.3; their
   per-row joint distribution may stratify supp-B in a stability-relevant
   way without invoking new groups. Builds directly on v0.3 cross-m_3
   receipts; per-row E and |L| are cheap when the v0.3 sentinel ran on
   the row.

2. **Bifurcation-track projection.** The per-m_3 S/U distribution shows
   visible discrete transitions (m_3 = 1.0 has 7 S / 31 U, m_3 = 1.1
   has 1 S / 22 U, m_3 = 1.5 has 4 S / 4 U, m_3 = 1.7 has 0 S / 10 U).
   A bifurcation-track mechanism would project the catalog onto the
   transition structure rather than an equivariant subspace. Risk:
   the m_3 axis is precisely the axis that failed held-out in v0.5b.

3. **Direction-of-instability (gamma_1) sidecar.** The v0.3 gamma_1
   sidecar tracked unstable directions per row; v0.4 retired the
   related gamma_3 family. A re-registration of gamma_1 as a primary
   mechanism would ask whether the direction of instability (not just
   its presence) carries predictive content.

4. **Action-angle / KAM-style decomposition.** The heaviest theoretical
   machinery: project orbits onto torus structure and ask whether
   nearly-integrable rows are stable. Theoretically the most aligned
   with the substantive question; technically the most expensive.

5. **Closing without v0.6.** If the next workstream targets a different
   substrate (mesa, geometry) rather than another K_facet refinement,
   the isotrophy program retires at end-of-v0.5 with publishable
   methodology + three negative findings + one descriptive positive
   + one explicit projection-limit finding.

Codex direction will pick. No runner work proceeds until the v0.6
mechanism family is registered paper-side.

**Update 2026-05-23:** v0.6 has opened with the **conserved-quantity
(E, |L|) stratification** family. Parent registration locked at
`kfacet_v06_mechanism_preregistration.md`. The chosen mechanism is
strictly richer than the v0.5 2-bit branch hash (continuous orbit-
level invariants vs. catalog-coordinate indicator bits) while
remaining catalog-derivable (computed from initial conditions + the
Hamiltonian; no orbit integration required). The parent registration
locks body / projection / observable, operational definitions of
E and |L|, non-circularity provenance, disallowed-feature inheritance,
and the v0.5 discipline (audit-then-predictor, asymmetric falsifier,
held-out partition, tie rule, constant-feature retirement). v0.6a
(audit form lock with explicit binning) and v0.6b (held-out predictor
form lock) are pending child registrations.

## Doc Trail

- `kfacet_v05a_branch_map_form.md` -- v0.5a audit form lock + verdict.
- `kfacet_v05b_branch_predictor_form.md` -- v0.5b held-out predictor
  form lock + verdict.
- `kfacet_v06_mechanism_preregistration.md` -- v0.6 parent registration
  (conserved-quantity (E, |L|) stratification; opens v0.6).
- `kfacet_v04_writeup.md` -- v0.4 chapter close (companion).
- `kfacet_v03h_writeup.md` -- v0.3 chapter close (grand-companion).
- `docs/CROSS_SUBSTRATE_NOTES.md` sections 6-7 -- projection-
  language framing (body / shadow / projection-limit vocabulary).

The chapter closes cleanly. Receipts are durable. v0.5a's catalog
stratification is preserved as a real finding; v0.5b's held-out failure
is preserved as a real generalization-limit. The chain has now produced
three sequential pre-registered chapters (v0.3, v0.4, v0.5) of
projection-limit results on the Li-Liao supplementary-B catalog, each
with reproducible receipts and explicit chapter-close discipline.
