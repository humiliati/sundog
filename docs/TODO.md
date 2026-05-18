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

## Quick Re-Inventory Commands

Use these when this TODO starts to feel stale:

```powershell
rg -n "BLOCK|HELD|HOLD|blocked|open|Next actions|Next Gate|operator|compute-unconstrained|old_basin_pref|Bayesian floor" . -g "*.md" -g "*.html" -g "*.json"
git status --short --untracked-files=all
npm run chat:index
```
