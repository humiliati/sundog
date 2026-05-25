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

## v0.5b Branch Predictor (held-out verdict)

### V0.5b branch-shadow predictor (leave-one-m_3-bin-out)

Status: **VERDICT LANDED 2026-05-23**. Paper-side form:
[`../internal/anniversary/kfacet_v05b_branch_predictor_form.md`](../internal/anniversary/kfacet_v05b_branch_predictor_form.md).
Receipt:
`results/isotrophy/k-facet-v05b-branch-predictor/manifest.json`.

Verdict: **branch_predictor_fails_heldout**. The v0.5a branch hash
stratifies S/U in-sample, but the fold-trained branch-majority predictor
does not beat always-`U` under leave-one-`m_3`-bin-out.

Locked form:

```text
primary partition:  leave-one-m_3-bin-out over the 12 bins with N >= 5
predictor form:     fold-trained branch-majority on active (m_3 < 1, z_0 < 0.3)
tie rule:           predict U
baseline:           always-U
primary gate:       one-sided exact McNemar/binomial test on discordant rows
pass threshold:     p <= 0.01 AND positive accuracy delta
continuous fields:  diagnostic-only; reserved for v0.5c
```

Primary result:

```text
gate rows:       263  (95 S / 168 U)
model accuracy:  0.619772
always-U acc:    0.638783
accuracy delta: -0.019011
McNemar:         win = 28, loss = 33, n_disc = 61, p = 1.0
verdict:         branch_predictor_fails_heldout
```

Reading: `m_3 = 0.4` was the load-bearing fold. Holding it out flips the
low-mass/low-`z_0` branch training direction to `U` (`28 S / 33 U`),
missing the large stable `0.4` block. In the remaining low-mass folds,
the same branch usually predicts `S`, but its held-out wins and losses
nearly cancel.

The random catalog-half split and the single rule `predict S iff
(m_3 < 1 and z_0 < 0.3)` are registered only as sidecars. The primary
gate respects the mass-bin structure rather than mixing same-`m_3`
rows across train and test.

Next actions:

1. ~~Sign off or revise the held-out predictor form before any runner is
   executed.~~ DONE 2026-05-23.
2. ~~Implement `scripts/v05b_branch_predictor.py` against
   `results/isotrophy/k-facet-v05a-branch-map/per_row_table.csv`.~~
   DONE 2026-05-23.
3. ~~Run the seconds-scale held-out predictor and land the receipt at
   `results/isotrophy/k-facet-v05b-branch-predictor/`.~~ DONE 2026-05-23.
4. ~~Land the chapter close.~~ DONE 2026-05-23. v0.5 epilogue writeup
   at [`../internal/anniversary/kfacet_v05_writeup.md`](../internal/anniversary/kfacet_v05_writeup.md).
   Chapter type: **projection-limit (audit-passes-predictor-fails)**.
   The v0.5a in-sample positive (chi^2 = 34.99) and v0.5b held-out
   negative (accuracy_delta = -0.019) together constrain the branch
   shadow as descriptive, not predictive.
5. ~~**Paper-side v0.6 direction (open).**~~ DONE 2026-05-23. v0.6
   opens with the **conserved-quantity (E, |L|) stratification**
   family. Parent registration at
   [`../internal/anniversary/kfacet_v06_mechanism_preregistration.md`](../internal/anniversary/kfacet_v06_mechanism_preregistration.md).
   See the v0.6 section below for the next-action queue.

Stop condition:

If a proposed v0.6 mechanism requires re-using the same active feature
set that failed v0.5b held-out, retire it at the registration stage
rather than launching a sweep. v0.5b's pre-registered asymmetric
falsifier (McNemar p AND positive delta) is the inheritance discipline.

## v0.6 Conserved-Quantity (E, |L|) Stratification (parent + v0.6a + v0.6b verdicts landed)

### V0.6 conserved-quantity (E, |L|) mechanism family

Status: **PARENT REGISTERED 2026-05-23; v0.6a VERDICT LANDED
2026-05-23; v0.6b VERDICT LANDED 2026-05-24**. Parent registration:
[`../internal/anniversary/kfacet_v06_mechanism_preregistration.md`](../internal/anniversary/kfacet_v06_mechanism_preregistration.md).
**v0.6a verdict**: `energy_quartile_passes_audit_alignment_warning`
(chi^2 = 33.70 vs 11.34, alignment = 0.956). Form/verdict:
[`../internal/anniversary/kfacet_v06a_energy_quartile_audit_form.md`](../internal/anniversary/kfacet_v06a_energy_quartile_audit_form.md).
**v0.6b verdict**: `within_branch_energy_fails_audit` (chi^2 = 6.90,
permutation p = 0.029; sparse-cell fallback fired with min_expected =
2.66; |L| sidecar at permutation p = 0.074 also below loud-signal
threshold). Form/verdict:
[`../internal/anniversary/kfacet_v06b_within_branch_energy_audit_form.md`](../internal/anniversary/kfacet_v06b_within_branch_energy_audit_form.md).
v0.6c (held-out predictor) is NOT licensed; v0.6 chapter is positioned
for chapter-close writeup as a clean conditional-independence result.

Frame: v0.4 ruled out the Z_2 shadow as a stability projector.
v0.5 ruled out the 2-bit catalog branch shadow as a held-out
predictor. v0.6 asks whether a higher-dimensional catalog-coordinate
projection — orbit-level conserved quantities `(E, |L|)` —
carries held-out stability.

Locked parent shape:

```text
Body:        supp-B piano-trio orbit as primary conserved-quantity
             object (273 rows, Z2_clean per v0.4a).
Projection:  row -> (E(row), |L|(row)) computed from initial
             conditions and the three-body Hamiltonian.
Observable:  per-row (E, |L|) joined with S/U stability label,
             binned into a pre-registered (n_E x n_L) contingency.
Compute:     catalog-only, seconds (no orbit integration required).
Discipline:  inherits v0.5 (audit-then-predictor, asymmetric
             McNemar+delta falsifier, leave-one-m_3-bin-out partition
             default, conservative tie rule, constant-feature
             retirement).
```

v0.6a locked shape:

```text
primary feature:    E(row) = total energy at IC, computed from
                    (m_1, m_2, m_3, r_i, v_i) and Hamiltonian.
binning:            quartiles over supp-B's 273-row E distribution.
quantile method:    numpy.quantile(E, [0.25, 0.50, 0.75], method='linear').
contingency:        4 x 2 over (Q_E, S/U), df = 3.
critical:           11.34 at p = 0.01.
verdicts:           energy_quartile_passes_audit (chi^2 > 11.34) |
                    energy_quartile_passes_audit_alignment_warning
                      (chi^2 > 11.34 AND max-bin-alignment > 0.8) |
                    energy_quartile_fails_audit (chi^2 <= 11.34).
sidecar:            |L| quartile chi-squared under same shape,
                    REPORT-ONLY (no verdict claim).
alignment guard:    receipt MUST report
                    max over Q_E of (fraction in any single
                    v0.5a branch_label bucket).
```

v0.6a result:

```text
receipt:       results/isotrophy/k-facet-v06a-energy-quartile-audit/manifest.json
sanity:        PASS  (max |Delta E| = 0, max |Delta |L|| = 0)
bound check:   PASS  (E < 0 for all 273 rows)

E audit:       chi^2 = 33.703158, p = 2.29e-7, critical = 11.34
alignment_E:  0.955882  (> 0.8 warning threshold)
verdict:      energy_quartile_passes_audit_alignment_warning

|L| sidecar:   chi^2 = 28.954252, p = 2.29e-6, report-only
alignment_|L|: 0.956522
```

Interpretation: E quartiles carry strong in-sample S/U signal, but the
registered alignment scalar says the signal is tightly entangled with
the v0.5a `(m_3, z_0)` branch shadow. v0.6b must therefore be
registered with an alignment-breaking partition (for example
leave-one-E-quartile-out, or a constant-m_3 subcatalog design), not the
default leave-one-m_3-bin-out partition.

Completed actions:

1. Per-row E and |L| computation implemented in
   `scripts/v06a_energy_quartile_audit.py`.
2. Sanity check passed. The receipt records that local cross_m_3
   receipts do not carry scalar E/|L| fields, so the parity reference
   is the pre-existing v0.3 `scripts.isotrophy_workbench`
   invariant implementation.
3. Bound-orbit check passed.
4. Per-row `(label, m_3, z_0, E, |L|, Q_E, Q_|L|, branch_label,
   stability)` table emitted.
5. v0.6a verdict landed.

v0.6b result:

```text
receipt:           results/isotrophy/k-facet-v06b-within-branch-energy/manifest.json
stratum:           113 rows  (63 S / 50 U)
sanity:            PASS  (N, S, U match locked counts)

within-branch Q_E contingency:
  Q1:  N=0   (Q1 entirely outside stratum -- tightest catalog
              orbits all sit in other branches)
  Q2:  N=6   S=2,  U=4   S_frac=0.333
  Q3:  N=42  S=18, U=24  S_frac=0.429
  Q4:  N=65  S=43, U=22  S_frac=0.662

within-branch Q_E x m_3:
  Q4 = m_3 in {0.4, 0.5}   (52 of 65 are m_3 = 0.4)
  Q3 = m_3 in {0.5, 0.6, 0.7, 0.8}
  Q2 = m_3 in {0.8, 0.9}
  -- Q_E is essentially a label for m_3 sub-bin inside the stratum.

E primary:    chi^2 = 6.904228
              fallback fired (min_expected = 2.655 < 5);
              permutation test seed = 20260523, n_permutations = 10000;
              permutation p = 0.0292
verdict:      within_branch_energy_fails_audit   (p > 0.01)

|L| sidecar:  chi^2 = 4.464524
              permutation p = 0.0741
              report-only; does NOT meet the loud-signal threshold
              (p <= 0.01) for a fresh v0.6c |L| form lock.
```

Interpretation: the within-branch direction is monotone in the same
direction as the catalog-wide v0.6a finding (Q2: 33% S, Q3: 43% S,
Q4: 66% S), but the permutation p = 0.029 does not clear the
pre-registered p <= 0.01 floor. The v0.6a in-sample chi-squared was
dominated by branch-shadow content; energy quartile is essentially a
1-to-1 label for m_3 sub-bin within the stratum, and the residual
within-branch stratification is below the gating floor. The
energy-shadow-as-mechanism sub-question closes; the |L| sidecar at
p = 0.074 does NOT trigger a fresh form lock either.

Completed actions:

1. ~~Implement `scripts/v06b_within_branch_energy_audit.py`.~~ DONE
   2026-05-24.
2. ~~Run the v0.6b audit blind; land verdict.~~ DONE 2026-05-24.

Next actions:

1. ~~Author the v0.6 chapter close writeup.~~ DONE 2026-05-24. v0.6
   epilogue at
   [`../internal/anniversary/kfacet_v06_writeup.md`](../internal/anniversary/kfacet_v06_writeup.md).
   Chapter type: **conditional-independence close** (distinct from
   v0.4 structural-negative and v0.5 projection-limit).
2. ~~**Paper-side v0.7 direction (open).**~~ DONE 2026-05-24. v0.7
   opens with the **gamma_1 direction-of-instability** family. Parent
   registration at
   [`../internal/anniversary/kfacet_v07_mechanism_preregistration.md`](../internal/anniversary/kfacet_v07_mechanism_preregistration.md).
   See the v0.7 section below for the runner state.

Stop condition:

If the within-branch chi-squared has a permutation p > 0.01 but the
residual structure suggests a coarser binning would pass, the
pre-registered re-registration path is a fresh form lock with
explicitly justified coarser bins (e.g., median-split with df=1) --
NOT a post-hoc loosening of v0.6b. v0.6b's sparse-cell fallback
discipline is permanent.

## v0.7 gamma_1 Direction-of-Instability (chapter closed, qualified-positive)

### V0.7 gamma_1 direction-of-instability mechanism family

Status: **CHAPTER CLOSED 2026-05-24, qualified-positive on restricted
domain.** Parent registration:
[`../internal/anniversary/kfacet_v07_mechanism_preregistration.md`](../internal/anniversary/kfacet_v07_mechanism_preregistration.md).
Chapter close:
[`../internal/anniversary/kfacet_v07_writeup.md`](../internal/anniversary/kfacet_v07_writeup.md).

**v0.7a catalog-wide verdict**:
`velocity_fraction_blocked_integration_attrition` at
`integration_blocked_count = 23` (8.42% of catalog > 5% threshold).
Two amendments applied: R1 (symplecticity gate 1e-6 -> 1e-4) and
R2.A (per-row integration-failure fallback). Form lock + verdict:
[`../internal/anniversary/kfacet_v07a_velocity_fraction_audit_form.md`](../internal/anniversary/kfacet_v07a_velocity_fraction_audit_form.md).

**v0.7a' restricted-scope confirmation verdict**:
`velocity_fraction_restricted_passes_audit` [PASS] at `chi^2 = 16.43`
(critical 11.34, df=3, p = 9.3e-4) on the 250 analyzable rows, with
alignment-tightness 0.698 against the v0.5a branch hash (below the
0.8 warning floor). Non-monotone U-shape signature: Q1 (gamma_1
mostly positional) 49% S, Q2 (mixed) 18% S, Q3 29% S, Q4 (gamma_1
mostly velocity) 43% S. Form lock + verdict:
[`../internal/anniversary/kfacet_v07a_prime_restricted_scope_form.md`](../internal/anniversary/kfacet_v07a_prime_restricted_scope_form.md).

The Floquet velocity-fraction direction shadow is the **first**
low-dimensional projection in the v0.4 / v0.5 / v0.6 / v0.7 envelope
to produce a non-branch-aligned positive signal at p < 0.01 on the
analyzable sub-catalog. Catalog-wide generalizability is limited by
the integration-attrition.

## v0.8 Floquet Direction-Purity (parent + v0.8a verdict landed)

### V0.8 Floquet direction-purity mechanism family

Status: **PARENT REGISTERED 2026-05-24; v0.8a VERDICT LANDED
2026-05-24 (FAIL).** Parent registration:
[`../internal/anniversary/kfacet_v08_mechanism_preregistration.md`](../internal/anniversary/kfacet_v08_mechanism_preregistration.md).
**v0.8a verdict**: `purity_quartile_fails_audit` at `chi^2 = 4.94`,
`p = 0.176` (df=3, critical 11.34). Form lock + verdict:
[`../internal/anniversary/kfacet_v08a_purity_quartile_audit_form.md`](../internal/anniversary/kfacet_v08a_purity_quartile_audit_form.md).
The unsigned purity transform `abs(vf - 0.5)` does NOT capture the
v0.7a' positive at the registered floor; the diagnostic
purity_signed contingency reproduces v0.7a's `chi^2 = 16.43`
exactly, confirming the signal lives in the **signed** direction.
The failure is explained by the vf distribution's density asymmetry
(184/250 rows below vf = 0.5; max vf = 0.87 -- the "vf near 1" pure
end is structurally under-populated). v0.8b (held-out predictor) is
NOT licensed under the fails verdict.

Frame: v0.7a' produced a non-monotone U-shaped signal (direction-
purity correlates with stability; mixed directions correlate with
instability). v0.8 promotes purity = `abs(vf - 0.5)` to the primary
operational quantity and asks whether it predicts S/U **held-out**
on the 250-row analyzable subset.

Locked parent shape:

```text
Body:         the 250-row analyzable supp-B subset from v0.7a
              (rows where compute_monodromy_vectorized completed).
              Attrition carried as a permanent domain restriction.
Projection:   row -> purity(row) = abs(vf(row) - 0.5)
              inheriting v0.7a's vf operational definition (gamma_1
              selection rule, tie-break cascade, CoM reduction,
              mass-weighted norm).
Observable:   per-row (purity, S/U) on 250 rows joined with v0.5a
              branch_label for alignment-tightness diagnostics.
Discipline:   inherits v0.5/v0.6/v0.7 (audit-then-predictor,
              asymmetric McNemar+delta predictor falsifier,
              alignment-tightness guard, sparse-cell fallback,
              conservative tie rule, constant-feature retirement).
Non-circ:     re-asserts v0.7a's locked sentence on Floquet
              eigenvectors used as geometric directions only.
```

Completed actions:

1. ~~Pick v0.8a form.~~ DONE 2026-05-24. Locked A (purity-quartile
   audit).
2. ~~Implement `scripts/v08a_purity_audit.py`.~~ DONE 2026-05-24.
3. ~~Run blind; land verdict.~~ DONE 2026-05-24. Verdict:
   `purity_quartile_fails_audit`.

Result interpretation:

```text
v0.8a purity quartile contingency:
  Q1  N=63  S=21  U=42  33.3% S  (vf ~ 0.44-0.56, mid)
  Q2  N=62  S=18  U=44  29.0% S
  Q3  N=62  S=19  U=43  30.6% S
  Q4  N=63  S=29  U=34  46.0% S  (vf far from 0.5, high purity)
                                  -----
                          chi^2 =  4.94  (p = 0.176)

v0.8a diagnostic purity_signed contingency
  (reproduces v0.7a' exactly since linear transform of vf):
  chi^2 = 16.43

asymmetry: |S(Q1_signed) - S(Q4_signed)| = 0.064 (< 0.2 flag)
asymmetric_u_shape_flag: False

vf distribution:
  vf < 0.5: 184 / 250 (73.6%)
  vf > 0.5:  66 / 250 (26.4%)
  vf max:    0.8653  (no rows near vf = 1)
```

The U-shape signal IS real (purity_signed reproduces v0.7a's
chi^2 = 16.43) but is **NOT captured by unsigned direction-purity**.
The signal lives in the signed vf direction. Density asymmetry of
the vf distribution (74% below 0.5) violates the purity transform's
symmetry assumption.

Next action: ~~paper-side direction call.~~ DONE 2026-05-24. v0.8
chapter closed as structural-negative on direction-purity (chapter
close at
[`../internal/anniversary/kfacet_v08_writeup.md`](../internal/anniversary/kfacet_v08_writeup.md));
v0.9 opened on signed Floquet direction-composition with explicit
anti-circular framing.

## v0.9 Signed Floquet Direction-Composition (parent + v0.9a verdict landed)

### V0.9 signed Floquet direction-composition mechanism family

Status: **PARENT REGISTERED 2026-05-24; v0.9a VERDICT LANDED
2026-05-24 (FAIL on both chi^2 and U-shape pattern; substantively
informative).** Parent registration:
[`../internal/anniversary/kfacet_v09_mechanism_preregistration.md`](../internal/anniversary/kfacet_v09_mechanism_preregistration.md).
**v0.9a verdict**: `signed_vf_three_zone_fails_audit_chi2` at
chi^2 = 7.42 vs critical 9.21 (df=2, p = 0.0245). Form lock + verdict:
[`../internal/anniversary/kfacet_v09a_signed_vf_three_zone_form.md`](../internal/anniversary/kfacet_v09a_signed_vf_three_zone_form.md).
The U-shape pattern check ALSO fails (mixed > positional, not
< positional). Substantive read: **the v0.7a' U-shape was a
quartile-boundary artifact at vf ~ 0.297**; under physical cutpoint
at vf = 0.25 the pattern is **monotone-increasing** (positional
11% S, mixed 34% S, velocity-heavy 44% S) but does not reach the
chi-squared floor. The v0.7a' chi^2 = 16.43 catalog-coordinate
positive remains valid; the U-shape MECHANISM hypothesis is
invalidated. v0.9b is NOT licensed.

Frame: v0.7a' produced chi^2 = 16.43 (p = 9.3e-4) on vf quartiles
vs S/U. v0.8a tested unsigned purity = `abs(vf - 0.5)` and failed;
the v0.8a diagnostic confirmed the signal is signed. v0.9 promotes
signed direction-composition (raw vf without folding) and asks
**what specific pattern in vf vs S/U is testable beyond v0.7a's
chi^2 of independence**.

**Anti-circular framing discipline (load-bearing, locked by parent):**

Any v0.9a form that:
1. Uses vf-ordering as feature, AND
2. Uses chi^2 of independence as test statistic, AND
3. Tests on the v0.7a 250-row analyzable subset

**reproduces v0.7a's chi^2 = 16.43 by linear-transform invariance**
and is REJECTED PRE-COMPUTE. To be honest, v0.9a must break at
least one of (feature derivation, test statistic, evaluation
domain).

Candidate v0.9a forms:

```text
A. Ordered quartile audit.          REJECTED (re-test of v0.7a').
B. Binary vf < 0.5 threshold.        REJECTED (binary collapse).
C. Three-zone audit with PHYSICAL
   cutpoints (not v0.7a' quartiles). LIVE (cutpoint motivation
                                          required).
D. Ordinal trend + nonmonotonicity
   test (e.g., Jonckheere-Terpstra). LIVE (cleanest, tests
                                           hypothesis-specific
                                           pattern not chi^2 of
                                           independence).
```

Completed actions:

1. ~~Pick v0.9a form.~~ DONE 2026-05-24. Locked C with physical
   cutpoints {0.25, 0.50} (positional-dominant / mixed /
   velocity-heavy). Anti-circular discipline: two of three
   conditions broken (feature derivation via physical cutpoints,
   test derivation via pattern-specific verdict tree).
2. ~~Implement `scripts/v09a_signed_vf_three_zone_audit.py`.~~ DONE.
3. ~~Run blind; land verdict.~~ DONE.

Result:

```text
verdict:  signed_vf_three_zone_fails_audit_chi2
chi^2:    7.42  vs critical 9.21 (df=2, p=0.01)
p_value:  0.0245
alignment_tightness:  0.5697

three-zone contingency:
  positional-dominant (vf < 0.25):  N=19   S_frac=0.1053
  mixed (0.25 <= vf < 0.50):        N=165  S_frac=0.3394
  velocity-heavy (vf >= 0.50):      N=66   S_frac=0.4394

U-shape pattern check:
  mixed < positional?   FALSE  (0.34 > 0.11)
  mixed < velocity?     TRUE   (0.34 < 0.44)
  u_shape_confirmed?    FALSE
```

Substantive interpretation: the v0.7a' U-shape was a quartile-
boundary artifact at vf ~ 0.297. Under physical cutpoint vf = 0.25,
the positional-dominant zone shrinks from 63 to 19 rows and its
S-fraction collapses from 49% to 11%. The actual physical pattern
is **monotone-increasing** in vf: positional < mixed < velocity-
heavy. The v0.7a' chi^2 = 16.43 catalog-coordinate result remains
valid; the U-shape MECHANISM hypothesis is invalidated.

Next action: ~~paper-side direction call.~~ DONE 2026-05-24. v0.9
chapter closed; isotrophy program paused at end-of-v0.9. Chapter
close at
[`../internal/anniversary/kfacet_v09_writeup.md`](../internal/anniversary/kfacet_v09_writeup.md);
program pause at
[`../internal/anniversary/kfacet_isotrophy_program_pause.md`](../internal/anniversary/kfacet_isotrophy_program_pause.md).

## Isotrophy Program PAUSED (2026-05-24)

The K_facet isotrophy program paused at end-of-v0.9 after seven
sequential pre-registered chapters with seven distinct chapter-close
types. Load-bearing substantive finding: v0.7a' chi^2 = 16.43
positive on the 250-row analyzable subset (branch-independent at
alignment 0.70). Load-bearing meta-finding: under physical zones,
the actual pattern is monotone-increasing in vf (positional 11% S,
mixed 34% S, velocity-heavy 44% S); but chi-squared on three zones
(7.42) does not pass p = 0.01.

Program pause document:
[`../internal/anniversary/kfacet_isotrophy_program_pause.md`](../internal/anniversary/kfacet_isotrophy_program_pause.md).

Bandwidth redirects to three-body Phase 15+ (survival pocket) and
the coarse-graining proof trunk per the original direction call.

### Lab Initiate Reopening Avenues (waste compute on)

Eight concrete reopening directions for any lab initiate with
bandwidth to spare. Full descriptions, hypotheses, implementation
sketches, and non-circularity arguments at
[`../internal/anniversary/kfacet_isotrophy_program_pause.md`](../internal/anniversary/kfacet_isotrophy_program_pause.md).

```text
1. Jonckheere-Terpstra trend test on v0.9a three-zone data
   (LOW EFFORT, HIGH SUBSTANCE). Tests the monotone-increasing
   pattern directly; J-T has more power than chi^2 for ordered
   alternatives. Non-circular by breaking test-statistic condition.

2. v0.7a' held-out predictor on the 250 analyzable rows
   (LOW-MEDIUM EFFORT). Re-register v0.7b on the restricted
   scope; leave-one-m_3-bin-out + asymmetric McNemar + positive-
   delta gate.

3. v0.7a relaxed-precision re-run at rtol = atol = 1e-10
   (MEDIUM-HIGH EFFORT). Attempts to clear the 23 attrited rows
   and recover catalog-wide v0.7a claims.

4. m_3 = 0.4 sub-catalog targeted mechanism investigation
   (MEDIUM EFFORT, HIGH SUBSTANCE). The m_3 = 0.4 cluster has
   been the load-bearing positive throughout; a within-bin
   audit may reveal substrate-specific structure.

5. Joint (vf, Q_E) audit on the 250 analyzable rows
   (MEDIUM EFFORT). Cochran-Mantel-Haenszel-style stratified
   test of Q_vf perp S/U given Q_E.

6. Cross-substrate transfer test (HIGH EFFORT). Apply the v0.7a
   velocity-fraction mechanism to a different three-body sub-
   catalog (mesa analog, free-fall, choreography) to test
   substrate-specificity.

7. Action-angle / KAM-style decomposition (HIGH EFFORT,
   THEORETICAL). Per-row action-angle variables stratify by
   integrability proximity; KAM-stable orbits are predicted
   stable.

8. Full-catalog audit at rtol = atol = 1e-10 with v0.7a R2.A
   discipline (alternative to avenue 3 if precision relaxation
   is sufficient to clear attrition).
```

All eight avenues respect the locked methodology surface AND the
anti-circular framing discipline. Reopening the program at any
of these is invited; mechanism-laundering (silently re-testing
v0.7a' under different labels) is not.

Stop condition:

If a lab initiate's reopening avenue runs and produces a clean
verdict (pass, fail, or sparse-inconclusive) under the locked
discipline, the program may UNPAUSE at that point and the chain
extends. Until then, the isotrophy section is closed for active
investigation.

Frame: v0.4 ruled out the Z_2 shadow. v0.5 ruled out the 2-bit catalog
branch shadow. v0.6 ruled out the (E, |L|) catalog-coordinate shadow
beyond branch labeling. **v0.7 leaves catalog-coordinate space**: it
asks whether the row's per-orbit monodromy operator has a geometric
structure (specifically the DIRECTION of its largest-real-part Floquet
eigenvector, projected onto a pre-registered geometric reference frame)
that stratifies stability.

Locked parent shape:

```text
Body:        supp-B piano-trio orbit as primary orbit-dynamics object
             (273 rows, M_i derivable from v0.4a receipt).
Projection:  row -> gamma_1(row), the eigenvector DIRECTION in phase
             space of the row's largest-real-part Floquet eigenvalue,
             projected onto a pre-registered geometric reference frame.
Observable:  per-row gamma_1 direction feature joined with S/U.
Compute:     monodromy + eigenstructure already in v0.4a; feature
             extraction is seconds per row.
Discipline:  inherits v0.5 (audit-then-predictor, asymmetric McNemar
             + delta falsifier) and v0.6 (alignment-tightness guard,
             sparse-cell fallback tree).
```

Candidate operational definitions (one to be locked by v0.7a):

```text
D1. Largest-real-part eigenvalue's eigenvector.
D2. Largest-magnitude eigenvalue's eigenvector.
D3. Restricted to "well-defined" rows; companion sidecar for ill-defined.
D4. Z_2-isotypic phase-space split of gamma_1 (even/odd fraction).
D5. Phase-space angle decomposition (velocity vs spatial component
    ratio).
```

Candidate audit forms (one to be locked by v0.7a):

```text
A. Direction-projection quartile audit.
B. Velocity-fraction quartile audit.
C. Z_2 isotypic-fraction quartile audit.
D. Direction-angle binary audit (df = 1, critical 6.63).
```

The v0.7a form lock must address three explicit circularity risks
(eigenvalue-choice, well-definedness, feature-extraction) and lock a
deterministic tie-break for S-row eigenvalue degeneracy, a reference
frame for the geometric projection, an alignment-tightness guard
against the v0.5a branch label, and the sparse-cell fallback tree.

v0.7a result:

```text
receipt:       results/isotrophy/k-facet-v07a-velocity-fraction-audit/manifest.json
verdict:       velocity_fraction_blocked_integration_attrition

total rows:               273  (S=97 / U=176)
analyzable:               250
integration_blocked:       23  (8.42%)
attrition threshold:       14  (5% of 273)
attrition fired:           True

sanity-gate failures
  (analyzable but symp>1e-4 or recip>1e-4):  11
total data-integrity issues:  34 / 273 = 12.5%

vf_sd:        0.144  (would have passed constant-feature retirement at 0.01,
                      but no chi-squared verdict is licensed under the
                      attrition gate)

runtime:      266.9 min (~4.5 h)
```

Blocked rows cluster at long-period high-m_3 (the same regime
v0.4a's two-pass classifier was built to handle). At v0.7a's
variational precision rtol = atol = 1e-12, the DOP853 step adapter
cannot find feasible steps for the matrix variational equation
along the most-challenging orbits. The audit is integration-attrited,
NOT feature-falsified.

v0.7a locked shape (pre-verdict):

```text
operational definition:  D5 velocity-fraction direction property.
gamma_1 selection:       largest-real-part Floquet eigenvalue's
                         eigenvector (D1's selection rule used only to
                         disambiguate; feature is independent of which
                         eigenvalue is picked).
tie-break cascade:       within degenerate eigenspace, maximal projection
                         onto velocity subspace; then smallest |Im(lambda)|;
                         then smallest positive argument; then lex sign.
reference frame:         center-of-mass reduced frame with mass-weighted
                         norm (body-3 mass variation does not fake a
                         direction effect).
feature:                 vf(row) = ||delta_v||^2 /
                                   (||delta_q||^2 + ||delta_v||^2)
                         after CoM reduction and mass-weighted norm.

audit form:              B = quartile audit of vf vs S/U.
binning:                 supp-B quartiles via numpy.quantile,
                         linear interpolation; right-closed lower bins.
contingency:             4 x 2 (Q_vf, S/U), df = 3.
critical:                11.34 at p = 0.01 (chi-squared(3)).

alignment guard:         max over Q_vf of fraction-in-any-single-
                         v0.5a-branch_label, threshold 0.8 (warning),
                         0.95 (severe).

sparse-cell fallback:    v0.6b inheritance verbatim
                         (seed = 20260523, n_permutations = 10000).

constant-feature
retirement threshold:    sd(vf) < 0.01.

verdicts:
  velocity_fraction_passes_audit                        |
  velocity_fraction_passes_audit_alignment_warning      |
  velocity_fraction_passes_audit_severe_alignment       |
  velocity_fraction_fails_audit                         |
  velocity_fraction_inconclusive_sparse                 |
  velocity_fraction_retired_near_constant               |
  velocity_fraction_blocked_sanity                      |
  velocity_fraction_blocked_circularity_audit.

named sidecar (D1 + A):  gamma_1 z-projection magnitude under CoM
                         frame, quartile-binned. Report-only; a loud
                         sidecar would require a fresh circularity
                         audit, NOT a refinement of v0.7a.

locked non-circularity sentence (paper-side):
  "v0.7a uses Floquet eigenvectors only as geometric directions; it
  does not use eigenvalue magnitude, spectral radius, unit-circle
  status, unstable-pair count, or any threshold that defines the
  published S/U label. The tested scalar is a phase-space composition
  ratio of the selected direction, not the growth rate of that
  direction."
```

Completed actions (during the v0.7a run):

1. ~~Implement `scripts/v07a_velocity_fraction_smoke.py` and run 7-row
   sentinel smoke.~~ DONE 2026-05-24. Initial smoke showed integrator
   was column-by-column slow; user signed off on vectorized
   `compute_monodromy` helper. Subsequent vectorized smoke + R1
   amendment showed 7/7 sentinels pass at 1e-4 symplecticity gate.
2. ~~First full-catalog run crashed at row 76 (O_194 at m_3=0.5)
   with "Required step size is less than spacing between numbers"
   inside `compute_monodromy_vectorized`.~~ User signed off on R2.A
   per-row integration-failure fallback + R2.C append-per-row resume.
3. ~~Implement `scripts/v07a_velocity_fraction_audit.py` with R2.A
   try/catch + R2.C append-per-row + resume mode.~~ DONE 2026-05-24.
4. ~~Re-run full audit blind under amendments.~~ DONE 2026-05-24,
   ~4.5 hours runtime, 23 integration-blocked rows, attrition fired.

Next action:

Pre-mortem flagged action items if the run hadn't completed cleanly:

1. Implement `scripts/v07a_velocity_fraction_audit.py`:
   - Per-row variational integration of the 18-dimensional tangent
     system over one period (new compute; was not in v0.4a's emitted
     scalars).
   - Symplecticity sanity gate (M_i^T J M_i - J in tolerance 1e-6).
   - Reciprocal-pair sanity gate (Floquet eigenvalue pairing within
     1e-4).
   - gamma_1 extraction under the locked selection rule and tie-break
     cascade.
   - Velocity-fraction computation under CoM reduction + mass-weighted
     norm.
   - Constant-feature retirement check (sd(vf) < 0.01).
   - Quartile binning + chi-squared + alignment-tightness guard +
     sparse-cell fallback.
   - D1 + A sidecar emitted under the same selection rule, different
     feature.
   - Receipt at
     `results/isotrophy/k-facet-v07a-velocity-fraction-audit/manifest.json`.
2. Run v0.7a blind; land verdict.
3. Conditional on `velocity_fraction_passes_audit` (clean):
   register v0.7b with default leave-one-m_3-bin-out partition.
   Conditional on `..._alignment_warning` or `..._severe_alignment`:
   register v0.7b with alignment-breaking partition (within-branch
   mirroring v0.6b).
   Conditional on `..._fails_audit`:
   close the velocity-fraction sub-question; the D1+A sidecar's
   chi-squared is informational only; v0.7c may register a fresh
   form lock with a different D + A combination after a circularity
   re-audit.
   Conditional on `..._inconclusive_sparse` or
   `..._retired_near_constant` or `..._blocked_*`:
   inspect and re-register as appropriate.

Stop condition:

If the v0.7a runner cannot maintain the locked non-circularity
discipline at compute time (e.g., the tie-break cascade requires
inspecting eigenvalue magnitude in some unforeseen way), abort with
`velocity_fraction_blocked_circularity_audit`. No compute verdict is
licensed.

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
