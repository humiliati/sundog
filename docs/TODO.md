# Sundog Active TODO

Last assembled: 2026-05-18.

This is the operator queue for outstanding roadmap work, experiment gates, and
blockers that were getting scattered across docs, anniversary notes, and public
site passes. Source specs and pre-registrations remain authoritative. If this
file conflicts with a spec, the spec wins and this file should be corrected.

Status tags:

- `compute-blocked`: the next result needs a long run or a runner plan.
- `operator-blocked`: the next result needs a human artifact, review, or manual
  measurement.
- `design-blocked`: the next result needs a sharper spec before execution.
- `public-surface`: the next result affects site copy, Ask Sundog, or launch
  posture.
- `deferred`: worth keeping visible, but not a launch gate.

## P0/P1 Load-Bearing Queue

### 1. Coarse-Graining Proof Trunk

Sources:
[`COARSE_GRAINING_PROOF_ROADMAP.md`](COARSE_GRAINING_PROOF_ROADMAP.md),
[`proof/PHASE4_THREEBODY.md`](proof/PHASE4_THREEBODY.md),
[`proof/PHASE4_BAYESIAN_FLOOR_BUILDOUT.md`](proof/PHASE4_BAYESIAN_FLOOR_BUILDOUT.md),
[`proof/PHASE6_LAMBDA_CONTROL.md`](proof/PHASE6_LAMBDA_CONTROL.md).

Status: `compute-blocked`, active.

Current state:

- Phases 0-3 are positive on the proof track: definitions lock, LQG proof,
  finite-MDP proof, and boundary theorem.
- Phase 4 has a three-body measured-substrate spec and a smoke-passed
  Bayesian-floor controller path, but the gate is not closed.
- BF-4b's first off-set calibration was a useful negative: the off-set cell
  gave regret CI `[0, 0]`, while the satisfiability probe showed oracle-signature
  headroom. That means the next diagnostic is real work, not bookkeeping.
- Phase 5 cross-substrate sameness is blocked on a positive Phase 4.
- Phase 6 lambda-control is staged, with empirical result still open.

Blocker:

The compute-unconstrained Information-Accessibility Diagnostic and BF-5
sharded/full-lock path need a long-run plan. Treat this as multiple-day compute
until measured otherwise. Do not abandon the trunk for a cheaper target just to
get a faster public result.

Next actions:

1. Add the exact runnable PowerShell commands to
   [`proof/PHASE4_THREEBODY.md`](proof/PHASE4_THREEBODY.md), including module or
   npm form, output directory, resume behavior, and read-back manifest path.
2. Run only a capped rate probe inline, if it is under the repo's ~10 minute
   rule. Record per-cell or per-trial rate in the spec.
3. Choose the runner for the real lock: local operator over multiple days,
   GitHub Actions `workflow_dispatch`, or self-hosted runner. Do not use an
   interactive coding-agent session for a multi-hour/multi-day lock.
4. Pre-register the branch table before the run:
   measurable-set regret goes to zero -> Phase 5 entry;
   measurable-set regret remains bounded away -> halt/falsify on real substrate;
   off-set regret goes to zero -> reopen the boundary theorem.
5. After Phase 4 closes, decide whether Phase 5 compares mesa, geometry, and a
   third toy, or stays strictly mesa plus geometry.

### 2. Structural Failure Boundary Map / Cut 3

Sources:
[`prereg/structural-failure-coincidence/BOUNDARY_MAP.md`](prereg/structural-failure-coincidence/BOUNDARY_MAP.md),
[`prereg/structural-failure-coincidence/PUBLICATION_PLAN.md`](prereg/structural-failure-coincidence/PUBLICATION_PLAN.md),
[`prereg/structural-failure-coincidence/P2_CUT3_ADMISSION.md`](prereg/structural-failure-coincidence/P2_CUT3_ADMISSION.md),
[`prereg/structural-failure-coincidence/P2_CUT3_H0_CALIBRATION.md`](prereg/structural-failure-coincidence/P2_CUT3_H0_CALIBRATION.md),
[`prereg/structural-failure-coincidence/P2_CUT3_H0_2_SCHEMA.md`](prereg/structural-failure-coincidence/P2_CUT3_H0_2_SCHEMA.md).

Status: `operator-blocked`, active.

Current state:

- P0 boundary map is frozen and passed.
- P1 admission passed.
- P2 first cut was corrected as vacuous.
- Cut 2 produced closed-form regime-separability, not a controller result.
- Cut 3 rendered escalation is opened but execution is held.

Blocker:

H0 angular calibration needs real-frame measurement. The operator has to mark or
draw the relevant anchors on actual pictures/renders, not on modeled stubs:
sun center, scale ticks or angular map, anchor loci, scored feature, valid span,
and any filename/configuration fields that might leak `h`.

Next actions:

1. Prepare an image packet for operator annotation, with one sidecar template per
   frame and a short instruction sheet.
2. Identify the known-PASS full-span fixture from real HaloSim output.
3. Fill measured sidecars for the Phase-15 known-FAIL frames, the known-PASS
   fixture, and each proposed Cut-3 corpus frame.
4. Generate the anchor-residual table from the sidecars.
5. Record the operator decision on whether HaloSim compound codes such as
   `e13` encode sun elevation and should trip the H-leak rule.
6. Re-run Cut-3 admission only after the render corpus manifest, H0 residual
   table, concrete agent path, baselines, edit operators, and write-path guard
   all exist.

Stop condition:

If honest angular coverage fails, Cut 3 stays blocked. Do not stretch the span,
substitute anchors, or score off-ruler features to force a result.

### 3. Ask Sundog Claim-Map Freshness

Sources:
[`SUNDOG_V_CHAT.md`](SUNDOG_V_CHAT.md),
[`WEBSITE_DEVELOPMENT.md`](WEBSITE_DEVELOPMENT.md),
[`../chat/contents.json`](../chat/contents.json),
[`../chat/claim_map.json`](../chat/claim_map.json).

Status: `public-surface`, launch-sensitive.

Current state:

The widget has deterministic trace and boundary machinery, but the site changed
quickly: atlas/legend disambiguation, alignment page, anniversary posture,
commercial translation language, and homepage proof panels all moved. The next
Ask Sundog pass should be isolated so it does not casually rewrite or contaminate
the eval.

Blocker:

`chat/claim_map.json`, prompt gold slates, and static answers may be behind the
current public copy.

Next actions:

1. Inventory new claim phrases and routes from `index.html`, `about.html`,
   `alignment.html`, `sundog.html`, `legend.html`, `applications-gallery.html`,
   `mesa.html`, and `structural-failure.html`.
2. Update `chat/claim_map.json` and the relevant prompt slates only where the
   public copy now invites a real visitor question.
3. Rebuild generated data with `npm run chat:index`.
4. Run, at minimum:

   ```powershell
   npm run chat:eval:static
   npm run chat:eval:phase3
   npm run chat:eval:phase3:adversarial
   npm run chat:eval:phase3:differential
   npm run chat:eval:phase4
   ```

5. Publish only if the gate preserves evidence tiers, source support, and active
   boundaries under adversarial pressure.

### 4. Mesa Phase 7 v3 / Basin-Attractor Caveat

Sources:
[`mesa/PHASE7_V2_RESULTS.md`](mesa/PHASE7_V2_RESULTS.md),
[`SUNDOG_V_MESA.md`](SUNDOG_V_MESA.md),
[`../internal/anniversary/fix_roadmap.md`](../internal/anniversary/fix_roadmap.md).

Status: `design-blocked`, active.

Current state:

Large Phase 7 v2 shows a U-shape recovery at `lambda=0.99`, but the eval summary
does not compute `old_basin_pref`. High terminal alignment is consistent with
both "reaches the basin" and "collapses onto a fixed attractor that co-points
with the basin direction."

Blocker:

We need a Phase 4-style intervention battery on Large checkpoints before saying
the recovery avoids the mesa-trap attractor.

Next actions:

1. Draft Phase 7 v3 with `old_basin_pref` or an equivalent intervention-confirmed
   basin-avoidance metric.
2. Reuse the coherent-signal framing: signature-pure and reward-pure may both be
   coherent classes; mixed-signal controllers destabilize when the mixture
   creates inference noise.
3. Keep public language at "basin-reaching by current eval metric, not yet
   verified basin-attractor-avoiding."
4. Measure compute cost before running a Large-tier battery.

### 5. Three-Body Phase 15 Full Lock

Sources:
[`threebody/PHASE15_SPEC.md`](threebody/PHASE15_SPEC.md),
[`threebody/PHASE15_RESULTS.md`](threebody/PHASE15_RESULTS.md),
[`../copilot_test_readme.md`](../copilot_test_readme.md).

Status: `compute-blocked`, operator-gated.

Current state:

The readback blocker is cleared, but the full `npm run threebody:phase15` lock is
still operator-gated. Recorded estimate is roughly 12,960 trials and about 75
hours. Interactive cloud agents are structurally unfit for this run.

Blocker:

Needs long-budget execution, not a coding-agent session.

Next actions:

1. Decide local multi-day operator run vs `workflow_dispatch` vs self-hosted.
2. Record runner command, expected wall-clock, output path, and manifest readback
   before starting.
3. Keep full-lock interpretation separate from coarse-graining Phase 4 unless
   Phase 4 explicitly chooses this evidence path.

## P2 Active Follow-Ups

### 6. Bayesian Floors Across Workbenches

Sources:
[`BAYESIAN_FLOOR_PROFILE_TEMPLATE.md`](BAYESIAN_FLOOR_PROFILE_TEMPLATE.md),
[`../alignment.html`](../alignment.html),
[`../internal/anniversary/attack_vectors.md`](../internal/anniversary/attack_vectors.md).

Status: `design-blocked`, cross-cutting.

Blocker:

The Bayes reference should be a same-observation baseline inside each workbench,
not a separate `bayes_v_sundog` storyline. Some workbenches still need their
Bayesian-floor profile and readout shape.

Next actions:

1. Add or update a Bayesian-floor row in each active workbench roadmap.
2. State the observation lane, privileged oracle lane, random/naive lane, and
   Bayes lane separately.
3. Keep public comparisons in the form "Sundog did X; Bayes did Y under the same
   information lane; oracle did Z with privileged access."

### 7. Geometry Calibration / Re-Anchoring Debt

Sources:
[`SUNDOG_V_GRAVITY.md`](SUNDOG_V_GRAVITY.md),
[`calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md`](calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md),
[`MESA_CROSSOVER_NOTE.md`](MESA_CROSSOVER_NOTE.md).

Status: `operator-blocked`, not a launch blocker unless reopened by Cut 3.

Blocker:

The tangent/supralateral/pyramidal route still has specialist-photo and
rendered-vs-anchored debt. The known discipline is to mark it as settled only at
the strength earned by anchored receipts.

Next actions:

1. If reopened, package p2/p7/p13/p27 anchors and the post-audit wording for
   specialist review.
2. Do not promote named-only or weak halo families into evidence without new
   anchored photos or HaloSim receipts.
3. Keep `sundog.html` and `legend.html` split: atlas for visual geometry,
   legend for vocabulary.

### 8. Application Evidence Assets

Sources:
[`../applications-gallery.html`](../applications-gallery.html),
[`UI_UX_THEME_FOUNDATION.md`](UI_UX_THEME_FOUNDATION.md),
[`WEBSITE_DEVELOPMENT.md`](WEBSITE_DEVELOPMENT.md).

Status: `public-surface`, active.

Outstanding assets:

- Photometric Alignment: experiment video, graph package, short reviewer-facing
  summary.
- EyesOnly / Gone Rogue: matched-seed multi-policy study, compressed-perception
  ablation, volatility sweep, behavior clips.
- Dungeon Gleaner: orbit telemetry capture, side-by-side planner comparison,
  tuning-sensitivity sweep.
- Money Bags: frozen terrain fixtures, disturbance scripts, metric glossary.
- Pressure Mines / Balance / Pushable Occluder: keep rail posters and clips tied
  to their owning evidence docs; do not activate missing-evidence cards just to
  fill space.

Next actions:

1. Prioritize one chart or clip per application that clarifies the evidence tier.
2. Keep gallery language as "surfaces" or "workbenches" unless a result doc earns
   stronger wording.
3. Replace decorative placeholders with claim-boundary visuals only when the
   visual adds interpretation.

### 9. Anniversary Rollout Aftercare

Sources:
[`../internal/anniversary/anni_spam_roadmap.md`](../internal/anniversary/anni_spam_roadmap.md),
[`../internal/anniversary/first_public_statement.md`](../internal/anniversary/first_public_statement.md),
[`../internal/anniversary/attack_vectors.md`](../internal/anniversary/attack_vectors.md),
[`ANNIVERSARY_ROADMAP_TRIAGE.md`](ANNIVERSARY_ROADMAP_TRIAGE.md).

Status: `public-surface`, active after posting.

Blocker:

The promotional circuit needs critique capture, not just more copy. Human
feedback is already pointing at comprehension and commercial-value questions.

Next actions:

1. Collect serious objections into `internal/anniversary/attack_vectors.md`.
2. Record recurring plain-language failures separately from technical objections.
3. Keep every broad public post attached to the failure-boundary sentence.
4. Add a young-reader/plain-language path only after the main copy stops
   accumulating contradiction and redundancy.

### 10. Canonical Routes / Deploy Verification

Sources:
[`WEBSITE_DEVELOPMENT.md`](WEBSITE_DEVELOPMENT.md),
[`../public/_redirects`](../public/_redirects),
[`../scripts/check-public-routes.mjs`](../scripts/check-public-routes.mjs).

Status: `public-surface`, pre-deploy check.

Blocker:

Local route smoke cannot prove Cloudflare `_redirects` actually emits 301s for
legacy `.html` URLs. A static dev server can pass by serving the old files
directly.

Next actions:

1. On the next Cloudflare preview deploy, run:

   ```powershell
   npm run site:routes -- --base <preview>.pages.dev
   ```

2. Consider tightening the route checker so legacy `.html` paths must return a
   redirect to the canonical extensionless route, not just any `200`.
3. Keep sitemap, canonical tags, `og:url`, and `_redirects` in agreement before
   anniversary traffic.

## Deferred, But Do Not Lose

### Vortex / Wishing-Well Toy

Source: [`../internal/anniversary/postulations.md`](../internal/anniversary/postulations.md).

Status: `deferred`.

This is a tempting cheap fourth test for coarse-graining language, but it should
not replace the proof trunk or the structural-failure boundary work. Keep it as
a candidate workbench only if it gets a Bayesian floor and a pre-registered
failure boundary.

### Repo Map / Agent Navigation Surface

Source: [`../repo-map.html`](../repo-map.html).

Status: `deferred`.

A living repo map could become a useful human and agent navigation surface, but
it should not sit on top of the hero until it is accurate enough to route a
visitor to evidence tiers, not just folders.

## v0.3 Cross-m_3 Sentinel (II)

### V0.3 Gamma cross-m_3 sentinel sweep (~3-4 hours staged compute)

Sources:
[`../internal/anniversary/kfacet_v03_gamma_crossm3_preregistration.md`](../internal/anniversary/kfacet_v03_gamma_crossm3_preregistration.md),
[`../internal/anniversary/kfacet_v03_freeze_b_comparison.md`](../internal/anniversary/kfacet_v03_freeze_b_comparison.md),
[`../internal/anniversary/kfacet_v03h_writeup.md`](../internal/anniversary/kfacet_v03h_writeup.md).

Status: **VERDICT LANDED 2026-05-22**. Sentinel sweep executed against
the pre-registration; joint verdict `(Q1.D, Q2.D) = gate pathology on
both axes`. Resolved by the targeted `sigma_3-scan` symmetry probe:
all seven supp-B sentinels fail `sigma_3`; the initial coarse-gauge
read placed six of seven in `Z_2 = (12)`-swap and flagged `O_434(0.4)`
as smaller-symmetry, but v0.4a0 later showed `O_434` was a gauge
artifact and v0.4a classified all 273 supp-B rows as `Z2_clean`.
v0.3 `Gamma_i` is `D_3`-equivariant by construction and structurally
inapplicable to this catalog. **Domain-of-applicability finding**, not
within-domain falsification. v0.3 epilogue closed; v0.4 chapter opens
on `Z_2`-equivariant mechanisms (paper-side design first, no runner
code until representation theory closes).

Current state:

- `alpha` closed the strict-G.2 m_3 = 1 cell of a **2 x 2 design** as
  structural null: zero daughters predicted from 20 resolved strict-G.2
  rows against 273 observed piano-trios. II tests the remaining two
  testable cells.
- The 2 x 2 has rows `(m_3 = 1, m_3 != 1)` and columns
  `(strict G.2 family, piano-trio family)`. Strict G.2 is only defined
  at m_3 = 1; the off-equal-mass strict-G.2 cell is empty by
  construction.
- The two slices answer two structurally separate questions:
  - **Q1 (m_3 axis):** does `Gamma` wake up where supp-B is densest?
    m_3 = 0.4 carries 55 of 273 rows.
  - **Q2 (family axis):** do piano-trio orbits carry the standard-E
    signal even though strict-G.2 seeds did not? The 38 supp-B rows
    at m_3 = 1 are entirely disjoint from the strict G.2 21 indices.
- Pre-registration freezes the sentinel subset (4 rows at m_3 = 0.4,
  3 rows at supp-B m_3 = 1.0), the per-axis outcome alphabet, and
  the joint verdict table over `(Q1, Q2)` before any run. The four
  main cases are
  `(A,A) = everywhere null`,
  `(A,B) = family-dependent`,
  `(B,A) = m_3-dependent`,
  `(B,B) = G.2 corner is unique`.

Blocker:

Each sentinel row takes 20-45 minutes wall-clock at rtol=1e-12. Seven
sentinels total = ~3-4 hours of staged compute. Above the inline
~10-minute rule.

Next actions:

1. Operator verifies parse: `npm run isotrophy:parse:b` returns 273
   rows; sentinel indices match the pre-registration tables.
2. Operator runs the 7 sentinel commands listed in the pre-registration
   (4 at m_3=0.4 + 3 at supp-B m_3=1.0). Each is staged separately.
3. Operator runs the reprocessor and bridge-audit commands over the
   sentinel receipts.
4. Read the per-axis outcomes `(Q1, Q2)` against the pre-registered
   alphabet `{A null, B signal, C quarantine, D gate-pathology}`.
   Resolve any C / D outcomes via per-row WHY-dive / runner diagnosis
   so the joint verdict reads over `(Q1, Q2) in {A, B}`. Then branch
   on the joint verdict:
   - `(A, A)` everywhere null -> stage full 55-row m_3 = 0.4 sweep as
     a follow-on operator command; if also null, write the v0.3
     epilogue.
   - `(A, B)` family-dependent -> halt; open family-axis review
     (piano-trios vs strict G.2).
   - `(B, A)` m_3-dependent -> halt; open m_3-axis review (F_beta
     cocycle off equal mass).
   - `(B, B)` G.2 corner is unique -> halt; open joint review (alpha
     verdict may be a rigidity artifact of the strict G.2 m_3 = 1
     cell).

Stop condition:

If sentinels surface a positive standard-E sector at any m_3, do not
extend to full slices before the redesign review lands. The alpha
verdict on m_3=1 strict-G.2 stands either way.

## v0.4a Domain Map (VERDICT LANDED)

### V0.4a Z_2 domain map for supplementary-B (~85-120 min staged)

Status: **VERDICT LANDED 2026-05-22**. Sweep executed 22:21 -> 02:46
(~4 h 25 min). Final verdict: **outcome_A_all_Z2_clean**. 273/273 rows in
`Z2_clean` after Pass 2 (24 rows rescued from provisional
`smaller_symmetry`, validating the two-pass design). `marginal_Z2 /
smaller_symmetry / undefined` bands all empty. v0.4 body now locked as
**"supplementary-B piano-trio orbit as primary Z_2 object; domain: all
273 rows"**. See the **Verdict (Landed)** section of the pre-registration
for the per-row provenance and Pass 2 rescue table.

Methodology rule recorded: long-period rows (T >= ~170) at m_3 in
{1.4, 1.5, 1.6, 1.7} stress the coarse SO(3) gauge minimizer; treat
default tolerances as a screening pass, not a final classifier.

Next: **v0.4b mechanism predictor** (closed structural-negative; see the
section below).

### V0.4a pre-registration reference

Sources:
[`../internal/anniversary/kfacet_v04a_domain_map_preregistration.md`](../internal/anniversary/kfacet_v04a_domain_map_preregistration.md)
(includes the Verdict (Landed, 2026-05-22) section with per-row
provenance and the Pass 2 rescue table),
[`threebody/CROSS_SUBSTRATE_NOTES.md`](threebody/CROSS_SUBSTRATE_NOTES.md)
§7.2 (projection-language framing for the v0.3 -> v0.4 transition).

Receipts:

```text
results/isotrophy/k-facet-v04a-domain-map/manifest.json
results/isotrophy/k-facet-v04a-domain-map/pass1/{aggregator_manifest, flagged_for_pass2.csv, m3eq*/}
results/isotrophy/k-facet-v04a-domain-map/pass2/m3eq{0.4, 1.4, 1.5, 1.6, 1.7}/O{idx}/
```

## v0.4b Mechanism Predictor (chapter closed)

### V0.4b gamma_3 / gamma_3' non-circular structural predictors (closed structural-negative)

Sources:
[`../internal/anniversary/kfacet_v04b_mechanism_preregistration.md`](../internal/anniversary/kfacet_v04b_mechanism_preregistration.md),
[`../internal/anniversary/kfacet_v04a_domain_map_preregistration.md`](../internal/anniversary/kfacet_v04a_domain_map_preregistration.md)
(domain locked: 273/273 Z2_clean),
[`../internal/anniversary/kfacet_v04b_gamma3prime_form.md`](../internal/anniversary/kfacet_v04b_gamma3prime_form.md),
[`../internal/anniversary/kfacet_v04_writeup.md`](../internal/anniversary/kfacet_v04_writeup.md),
[`threebody/CROSS_SUBSTRATE_NOTES.md`](threebody/CROSS_SUBSTRATE_NOTES.md)
§7.2 (projection-language framing).

Status: **CHAPTER CLOSED 2026-05-23, structural-negative**. Both
registered Z_2-shadow predictors failed their pre-registered falsifiers
without committing the compute-blocked 2-hour sweep. `gamma_3`
(tangent-isotypic) retired pre-sweep when the 7-row sanity probe found
`F_beta` does not preserve `K_fib` on supp-B (verdict
`form_precondition_failed`). `gamma_3'_orbit_pass2` (orbit
gauge-rigidity) registered and immediately tested against the v0.4a
manifest (no new compute); falsified by `chi^2 = 1202.32` vs critical
`26.22` (46x threshold). Rule accuracy 63.74% lies slightly below the
always-U baseline 64.47%. Stability is statistically independent of the
Z_2 shadow at both tested granularities. Full chapter close at
[`../internal/anniversary/kfacet_v04_writeup.md`](../internal/anniversary/kfacet_v04_writeup.md);
gamma_3'_orbit_pass2 verdict receipt at
`results/isotrophy/k-facet-v04b-gamma3prime-orbit-pass2/manifest.json`.

Current state:

- v0.4a closed with verdict `outcome_A_all_Z2_clean`. The 273 supp-B
  piano-trio rows uniformly carry `Z_2 = (12)`-swap symmetry after the
  pre-registered two-pass gauge classifier.
- v0.4b found a new structural gap: v0.4a proves `F_beta` is an
  orbit-level symmetry of supplementary-B piano-trios, but the
  at-anchor `F_beta` operator used by the locked form is **not** a
  tangent-level symmetry of `K_fib = ker(M_i - I) / N_C` on the tested
  rows.
- Seven distinct sanity rows all show `F_beta` leakage out of `K_fib`
  at order `1e-1` (median approximately `0.45`) and projector overcount
  of `+1` to `+3` directions. The naive `(I +/- F_beta_K)/2`
  projectors are therefore not valid isotypic projectors on supp-B
  `K_fib`.
- The locked threshold rule `predict S iff F_beta_even_dim >=
  F_beta_odd_dim` is retired **before** the 273-row sweep. It is not a
  failed chi-squared result; the precondition failed first.
- `gamma_3'` replacement selected the orbit-feature lane:
  `gamma_3'_orbit_pass2`, predicting S iff the row required Pass 2
  rescue in the v0.4a two-pass gauge classifier. It was tested inline
  from the v0.4a manifest and falsified (`chi^2 = 1202.32` vs
  `critical = 26.22`; accuracy 63.74%, below always-U 64.47%).

Closed blockers:

1. ~~**Paper-side** gamma_3 form lock~~ -- DONE 2026-05-22, then
   RETIRED 2026-05-23. Threshold-rule baseline locked at
   [`../internal/anniversary/kfacet_v04b_gamma3_form.md`](../internal/anniversary/kfacet_v04b_gamma3_form.md):
   `predict S iff F_beta_even_dim >= F_beta_odd_dim`. Zero free
   parameters. Chi-squared df = 12, critical value `26.22` at
   `p = 0.01`. Retired because `F_beta` does not preserve `K_fib` on
   the sanity rows.
2. ~~**Paper-side**: choose and pre-register a replacement `gamma_3'`
   projection target.~~ DONE 2026-05-23. The `orbit_features` lane was
   selected after the cocycle-rescue probe returned identity cocycles.
3. ~~**Implementation/compute**: deferred.~~ The original 273-row
   `kfacet-row-z2-sweep` plan is historical only and should not be run
   for the retired baseline; the selected `gamma_3'_orbit_pass2` verdict
   used the existing v0.4a manifest with no new compute.

Next actions:

1. ~~Lock gamma_3 form paper-side.~~ DONE 2026-05-22, retired
   2026-05-23 by `form_precondition_failed`.
2. ~~Choose replacement lane.~~ DONE 2026-05-23. The cocycle-rescue probe
   surfaced `(R_i, phi_i) = (I, 0)` uniformly, ruling out
   `anticomm`-style lanes. `orbit_features` selected; baseline locked at
   `gamma_3'_orbit_pass2` (predict S iff Pass 2 rescue required).
3. ~~Run the verdict.~~ DONE 2026-05-23, inline from the v0.4a manifest
   (no new compute). Verdict: **falsifies_baseline_orbit_pass2_rule** at
   `chi^2 = 1202.32` vs `critical = 26.22`.
4. ~~Land the chapter close.~~ DONE 2026-05-23. v0.4 epilogue
   writeup at
   [`../internal/anniversary/kfacet_v04_writeup.md`](../internal/anniversary/kfacet_v04_writeup.md).
5. ~~**Paper-side v0.5 direction (open).**~~ DONE 2026-05-23. v0.5
   opens as a **branch-shadow audit**: catalog-only branch hash on
   `(m_3, z_0)` indicator bits, chi-squared independence vs S/U at
   `df = occupied_branch_count - 1`. Frame: stop asking what symmetry
   the row has, start asking which branch the row belongs to. See
   [`../internal/anniversary/kfacet_v05a_branch_map_form.md`](../internal/anniversary/kfacet_v05a_branch_map_form.md).
   See the v0.5a section below for the runner state.

Stop condition:

If a proposed `gamma_3'` still requires an isotypic decomposition on a
subspace not preserved by the relevant `F_beta` action, retire it at the
precondition stage rather than launching a catalog sweep.

## v0.5a Branch-Shadow Audit (registered)

### V0.5a branch-shadow audit (chi-squared independence on the active branch hash)

Status: **VERDICT LANDED 2026-05-23**. The 4x2 branch-shadow audit passes:
`chi^2 = 34.986` vs critical `11.34` (`df = 3`, p ~= `1.23e-7`).
Receipt:
`results/isotrophy/k-facet-v05a-branch-map/manifest.json`.

Frame: stop asking what symmetry the row has, start asking which branch
the row belongs to. v0.5a is an AUDIT (chi-squared independence of a
branch hash against S/U), NOT a predictor. A pass licenses v0.5b
registration as a true predictor with held-out test; a fail joins
v0.4 as another projection-limit negative and the chain moves to the
next mechanism family.

Locked form
(`internal/anniversary/kfacet_v05a_branch_map_form.md`):

```text
4-bit candidate signature, with deterministic constant-bit retirement:
  b1 = (m_3 < 1)              ACTIVE  (169 / 273 true on supp-B)
  b2 = (z_0 < 0.3)            ACTIVE  (130 / 273 true on supp-B)
  b3 = (abs(v_z) < 1e-6)      RETIRED (constant FALSE on supp-B)
  b4 = (m_3 * z_0^2 < 2)      RETIRED (constant TRUE  on supp-B)

active signature on supp-B: (b1, b2)
occupied buckets: 4
df = occupied_branch_count - 1 = 3
critical: chi-squared(3) at p = 0.01 = 11.34
verdicts:
  chi^2 >  11.34: branch_hash_passes_audit
  chi^2 <= 11.34: branch_hash_fails_audit
```

Catalog-only input; no new dynamical compute is required. The audit
runs against the v0.4a manifest at
`results/isotrophy/k-facet-v04a-domain-map/manifest.json`.

4x2 result:

```text
branch_label                  N    S    U    S_fraction
m_3 >= 1, z_0 >= 0.3          87   18   69   0.2069
m_3 >= 1, z_0 <  0.3          17    5   12   0.2941
m_3 <  1, z_0 >= 0.3          56   11   45   0.1964
m_3 <  1, z_0 <  0.3         113   63   50   0.5575
```

Next actions:

1. ~~Implement the audit aggregator.~~ DONE:
   [`../scripts/v05a_branch_map_audit.py`](../scripts/v05a_branch_map_audit.py).
2. ~~Run the audit blind.~~ DONE. Verdict:
   `branch_hash_passes_audit`.
3. ~~Land the verdict.~~ DONE in
   [`../internal/anniversary/kfacet_v05a_branch_map_form.md`](../internal/anniversary/kfacet_v05a_branch_map_form.md).
4. Register v0.5b paper-side with held-out m_3 bins or
   leave-one-branch-out CV. No predictor claim is licensed until v0.5b
   is separately registered and tested.

Stop condition:

Any post-audit narrowing of the active signature, threshold change, or
df-formula change is a **re-registration**, not a refinement. The
audit form is locked.

## v0.5b Branch Predictor (held-out draft)

### V0.5b branch-shadow predictor (leave-one-m_3-bin-out)

Status: **DRAFTED FOR SIGN-OFF 2026-05-23**. Paper-side form:
[`../internal/anniversary/kfacet_v05b_branch_predictor_form.md`](../internal/anniversary/kfacet_v05b_branch_predictor_form.md).
No compute has been run under this form.

Locked draft shape:

```text
primary partition:  leave-one-m_3-bin-out over the 12 bins with N >= 5
predictor form:     fold-trained branch-majority on active (m_3 < 1, z_0 < 0.3)
tie rule:           predict U
baseline:           always-U
primary gate:       one-sided exact McNemar/binomial test on discordant rows
pass threshold:     p <= 0.01 AND positive accuracy delta
continuous fields:  diagnostic-only; reserved for v0.5c
```

The random catalog-half split and the single rule `predict S iff
(m_3 < 1 and z_0 < 0.3)` are registered only as sidecars. The primary
gate respects the mass-bin structure rather than mixing same-`m_3`
rows across train and test.

Next actions:

1. Sign off or revise the held-out predictor form before any runner is
   executed.
2. Implement `scripts/v05b_branch_predictor.py` against
   `results/isotrophy/k-facet-v05a-branch-map/per_row_table.csv`.
3. Run the seconds-scale held-out predictor and land the receipt at
   `results/isotrophy/k-facet-v05b-branch-predictor/`.
4. Update this section with the verdict.

## Onboarding / Polish (Run-Friendly)

### V0.3h K_facet Tooling Polish (~10-hour initiation)

Sources (paper trail / table of contents):

1. [`../internal/anniversary/kfacet_v03h_writeup.md`](../internal/anniversary/kfacet_v03h_writeup.md)
   — Methodology + result hand-off. **Start here.** Sets the audit chain
   (sentinel → adaptive-floor reprocessor → bridge audit) and the
   load-bearing result: `20 of 21 m_3=1 strict G.2 rows are structural
   zeros; O_617 is the single quarantined row`.
2. [`../internal/anniversary/kfacet_v03h_o617_deep_dive.md`](../internal/anniversary/kfacet_v03h_o617_deep_dive.md)
   — The quarantined-row companion. Reads the six-probe deep dive plus
   the WHY-dive addendum that landed `bridge_approx_sign_isotypic` as
   the corrected disposition.
3. [`../internal/anniversary/kfacet-runner-spec.md`](../internal/anniversary/kfacet-runner-spec.md)
   — Full runner spec. Includes the adaptive-floor algorithm, the bridge
   audit outcome categories, and the disposition for O_617. Long; use the
   table-of-contents at the top to navigate.
4. [`sundog_v_isotrophy.md`](sundog_v_isotrophy.md)
   log lines 1305-1310 — Dated v0.3h log entries. Each line is one round
   of the audit chain ratcheting.
5. [`SUNDOG_V_THREEBODY.md`](SUNDOG_V_THREEBODY.md)
   bridge-audit section around line 933 — Parent-spine mirror of the
   same closure.
6. [`../scripts/isotrophy_workbench.py`](../scripts/isotrophy_workbench.py)
   — Workbench with three landed subcommands (`kfacet-sentinel`,
   `kfacet-reprocess-floor`, `kfacet-bridge-audit`) plus the pre-registered
   constants (`ADAPTIVE_FLOOR_*`, `BRIDGE_*`).
7. [`../scripts/o617_deep_dive.py`](../scripts/o617_deep_dive.py),
   [`../scripts/o617_why_dive.py`](../scripts/o617_why_dive.py),
   [`../scripts/catalog_near_t_separator.py`](../scripts/catalog_near_t_separator.py)
   — Three one-shot diagnostic scripts. These are the candidates for
   promotion in this polish item.

Status: `deferred`, polish-ready. ~10-hour budget. No compute lock; all
diagnostic scripts run in seconds against existing receipts under
`results/isotrophy/k-facet-v03-*/`.

Current state:

- The v0.3h audit chain is **closed** for `m_3=1`. Three workbench
  subcommands + three one-shot Python scripts produce the load-bearing
  result with deterministic, pre-registered classifications.
- Diagnostic scripts (`o617_deep_dive.py`, `o617_why_dive.py`,
  `catalog_near_t_separator.py`) are functional one-shots but
  hardcoded to `O_617` (deep-dive and why-dive) or to the full 21-row
  catalog (separator). They are not discoverable via `npm run`.
- The catalog separator has a known projector double-counting artifact
  (T + S + E exceeds kernel dim by 1 for `O_617`) that should be fixed
  by switching from character projectors to simultaneous (sigma_3,
  F_beta) joint eigendecomposition.
- The interpretive lesson from the (II) round (always compute signed
  `<v, F_beta v>`, never just parity magnitudes) is currently a comment
  in the WHY-dive script; it should be hoisted into a workbench helper
  used by all three diagnostics.

Verify-the-picture step (1 hour budget):

1. Run `npm run isotrophy:kfacet:sentinel:gamma` on `O_62` (control) and
   inspect `gate_receipt.json`. Expect all gates PASS, `c_i = d_i = 0`,
   `T(2)+S(5)+E(0)`.
2. Run `npm run isotrophy:kfacet:reprocess-floor`. Manifest summary
   should report `21/21 resolved or recorded` with no suspicious rows.
3. Run `npm run isotrophy:kfacet:bridge-audit`. Manifest should report
   `20 no_bridge_present + 1 defective_E_block_confirmed (O_617)`.
4. Run `python scripts/o617_deep_dive.py`. Six probes; expect
   `near_T_edge=[617]` and `e_rotation=0` per the existing receipt.
5. Run `python scripts/o617_why_dive.py`. Expect outcome
   `bridge_approx_sign_isotypic` (the corrected label after the
   isotypic-projector check).
6. Run `python scripts/catalog_near_t_separator.py`. Expect
   `edge_E_other=[617]` only; all 20 other rows pure `clean_T(2) +
   clean_S(5 or 6)`.

If any of (1)-(6) drift from the recorded receipts, stop and consult
[`../internal/anniversary/kfacet_v03h_writeup.md`](../internal/anniversary/kfacet_v03h_writeup.md)
before continuing.

Polish items:

1. **Generalize `o617_deep_dive.py` to a workbench subcommand**
   (~2 hours).
   - Add `kfacet-row-anatomy --row INDEX [--source A --m3 1.0]`.
   - Make all six probes accept an arbitrary row, not just `O_617`.
   - Receipt path: `results/isotrophy/k-facet-v03-row-anatomy/O{idx}/`.
   - npm: `isotrophy:kfacet:row-anatomy -- --row 617`.
   - Smoke: run on `O_62` (control) and `O_617` (quarantined); both
     should produce structured receipts without errors.
2. **Generalize `o617_why_dive.py` to a workbench subcommand**
   (~2 hours).
   - Add `kfacet-why-dive --row INDEX`.
   - Keep the corrected outcome hierarchy:
     `bridge_in_wrong_frame`, `bridge_frame_partial`,
     `bridge_approx_sign_isotypic`, `bridge_approx_trivial_isotypic`,
     `bridge_is_quasi_kernel`,
     `bridge_eigenvector_off_unit_circle`, `bridge_genuinely_non_D3`.
   - Include direct signed `<v, sigma_3 v>` and `<v, F_beta v>` in the
     C1/C3 receipts (the interpretive-error fix from the (II) round).
   - npm: `isotrophy:kfacet:why-dive -- --row 617`.
3. **Generalize the catalog separator and fix the projector
   double-counting** (~3 hours).
   - Add `kfacet-near-T-separator [--rows IDX,IDX,...]`.
   - Replace the character-projector decomposition with a
     simultaneous-eigenvector decomposition of `(sigma_3, F_beta)` on
     the kernel (joint diagonalization). T = (lambda_s3=+1,
     lambda_fb=+1) eigenspace, S = (+1, -1), E = (omega, omega^2) pair.
     This fixes the `T + S + E = 9` vs `kernel_dim = 8` artifact for
     `O_617`.
   - Keep the existing classification thresholds (alignment 0.99,
     closure 1e-3) intact.
   - npm: `isotrophy:kfacet:near-T-separator`.
4. **Add a shared workbench helper `compute_d3_alignments`**
   (~30 minutes). Encapsulate the "always signed inner product" lesson:
   given a direction `v`, return `<v, sigma_3 v>` and `<v, F_beta v>`
   (signed). Use this helper from the three new subcommands; this is
   the codified version of the interpretive lesson the (II) round
   surfaced.
5. **Smoke tests** (~1 hour). One small `scripts/test_kfacet_diagnostics.py`
   or pytest module that runs each new subcommand on `O_62` and `O_617`,
   asserts on a few headline fields (outcome name, isotypic dims,
   bridge alignment sign), and emits a pass/fail summary. Wire it under
   `npm run isotrophy:kfacet:diagnostics:test`.
6. **Doc updates** (~1 hour).
   - In
     [`../internal/anniversary/kfacet_v03h_o617_deep_dive.md`](../internal/anniversary/kfacet_v03h_o617_deep_dive.md)
     and
     [`../internal/anniversary/kfacet-runner-spec.md`](../internal/anniversary/kfacet-runner-spec.md),
     replace the `python scripts/o617_*.py` references with the new
     `npm run isotrophy:kfacet:*` commands and `--row` parameterization.
   - Add a one-line note in
     [`../internal/anniversary/kfacet_v03h_writeup.md`](../internal/anniversary/kfacet_v03h_writeup.md)
     pointing at the new row-anatomy / why-dive subcommands.
7. **Buffer / unblocking room** (~30 minutes). Reserved for friction
   on argparse, receipt-schema drift, or the joint eigendecomposition
   subtleties.

Acceptance criteria:

- All three new npm scripts run cleanly on `O_62` and `O_617` and emit
  versioned receipts under `results/isotrophy/k-facet-v03-*/`.
- The `kfacet-near-T-separator` no longer over-counts: `T + S + E ==
  kernel_dim` exactly for every row, including `O_617`.
- The `kfacet-why-dive` receipt records signed
  `<v, sigma_3 v>` and `<v, F_beta v>` explicitly. The outcome label
  is `bridge_approx_sign_isotypic` on `O_617`.
- `npm run isotrophy:kfacet:diagnostics:test` passes locally; receipts
  archived for the audit trail.
- Documentation references the new npm-script entry points; the
  one-shot scripts can be marked deprecated but remain runnable.

Known issues / interpretive guardrails:

- **Signed-vs-magnitude trap**: the original WHY-dive inferred
  `F_beta v approx +v` from parity-magnitude inspection of
  `F_beta v_bridge`. The actual signed alignment is `-0.9999970`.
  Always use signed inner products for representation-theoretic
  classification; magnitudes alone discard the irrep label.
- **Catalog-admission orientation**: `O_617` is admitted to the strict-21
  catalog via the *opposite* sigma_3 orientation (residual `1.01e-8`,
  catalog-normal). It is not a weak-admission row. The defective E(1)
  bridge is a structural property of the orbit at this kernel boundary,
  not a tolerance artifact.
- **The 20/21 firewall**: the polish work does **not** touch the
  load-bearing v0.3h evidence claim. The new subcommands are
  reproductions of existing one-shot scripts with `--row` parameters
  and one methodology fix (joint eigendecomposition instead of
  character projectors). The 20-row structural-zero result and the
  O_617 quarantine should be byte-identical across the rerun.

Out of scope for this polish round:

- Cross-`m_3` or supplementary-B verification.
- Catalog-admission tightening (already settled as unnecessary; `O_617`
  is not a weak-admission row).
- Paper-side write of the audit chain (see
  [`../internal/anniversary/kfacet_v03h_writeup.md`](../internal/anniversary/kfacet_v03h_writeup.md)
  for the existing methodology hand-off; a paper draft would be a
  separate deferred item).
- Advancing to v0.4 / induced-rep `d_i` derivation.

Exit deliverable: a single PR (or its equivalent) that adds three new
workbench subcommands, three new npm scripts, one shared
`compute_d3_alignments` helper, one smoke-test module, and the doc
updates listed above. Reviewer should be able to verify the exit
deliverable by running `npm run isotrophy:kfacet:diagnostics:test` and
inspecting the new manifest under
`results/isotrophy/k-facet-v03-row-anatomy/`.

## Quick Re-Inventory Commands

Use these when this TODO starts to feel stale:

```powershell
rg -n "BLOCK|HELD|HOLD|blocked|open|Next actions|Next Gate|operator|compute-unconstrained|old_basin_pref|Bayesian floor" . -g "*.md" -g "*.html" -g "*.json"
git status --short --untracked-files=all
npm run chat:index
```
