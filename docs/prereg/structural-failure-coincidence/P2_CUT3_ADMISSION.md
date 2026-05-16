# Structural Failure Coincidence — Cut 3 Admission Check

Run spec under review: [`P2_CUT3_RUN_SPEC.md`](P2_CUT3_RUN_SPEC.md)
Parent: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md)
Cut-2 disposition: [`P2_CUT2_WAVE42_DISPOSITION.md`](P2_CUT2_WAVE42_DISPOSITION.md)
Phase-15 hazard source:
[`SPECULATIVE_HALO_PROOFS.md`](../../calibration/SPECULATIVE_HALO_PROOFS.md)
Filed: **2026-05-16 (PT)**. Reviewer: Codex audit.
Status: **HOLD — CUT-3 SPEC ADMITTED IN PRINCIPLE; EXECUTION NOT
ADMITTED**. This check reviews the Cut-3 specification and the
pixel-to-degree hazard. It does not approve a corpus, script, training
run, or controller evaluation.

## Purpose

Cut 3 may execute only if the rendered-signal protocol proves, before
any agent run, that the Phase-15 pixel-to-degree / centering hazard is
handled. This admission check therefore separates two questions:

1. Is the Cut-3 specification shaped correctly?
2. Does a concrete render corpus already satisfy the angular-calibration
   gate?

The first answer is **yes in principle**. The second answer is **not yet**:
no Cut-3 corpus manifest exists. Execution remains held.

## Admission checklist

| requirement | ruling | notes |
| --- | --- | --- |
| Cut-3 trigger is valid | **PASS** | Wave-4.2 filed the alpha+gamma disposition: closed-form Cut 2 can only certify regime-separability plus escalation. |
| No Wave-4 receipt reopened | **PASS** | The Cut-3 spec treats rendered signal as a new cut, not a retune of C3-A. |
| H0 angular-calibration gate resolves the Phase-15 hazard at protocol level | **PASS** | The spec requires per-frame sun-centered angular calibration, valid span, anchor residuals, hashes, and feature exclusion outside span. |
| Phase-15 negative is absorbed honestly | **PASS** | The pyramidal Scale attempt is named as a negative example: Scale works as an instrument, but a short-span scale receipt cannot admit a feature field. |
| Corpus exists with H0 records | **HOLD** | No `cut3-render-corpus-manifest.json` or equivalent exists yet. |
| H0 anchor residual table exists | **HOLD** | No per-frame angular residual table has been filed. |
| Agent-under-test path is bound | **HOLD** | The spec allows either existing-controller rendered adapter or learned image agent, but the concrete path is not yet selected. |
| Route and correlate baselines are operational | **HOLD** | Baseline roles are specified, but no scripts or artifacts exist yet. |
| Counterfactual edit operators are frozen | **HOLD** | Route-edit and nuisance-edit semantics are specified, but no concrete image edit operators are frozen. |
| Public-language constraint preserved | **PASS** | The spec forbids pass/theorem/traceability language until an admitted run passes quantities 1-3. |

## Rulings

### A — What "resolves the P15 hazard" means

The Phase-15 hazard is not resolved by declaring a single
pixels-per-degree scale. HaloSim Camera View can auto-zoom when the
crystal population changes; split-sky / plan projections and short
Scale rulers can also make a visually plausible feature unmeasurable.

For Cut 3, the hazard is resolved only by an **H0 angular-calibration
manifest**. Each admitted frame must carry a sun-centered angular map
with a valid span and anchor residuals. The admission gate must reject
any frame whose scored feature lies outside that span. This turns the
Phase-15 failure into a reusable guardrail instead of a buried caveat.

### B — Phase-15 Scale receipt ruling

HaloSim-native Scale is admissible as a *method*, not as an automatic
pass. The Phase-15 pyramidal scale-stamped frames showed exactly why:
the ruler located the sun and supplied HaloSim-native angular ticks, but
its span was shorter than the ring field and the 22/46 degree anchors
were beyond the ruler tip. Those frames therefore fail H0 for a
quantitative residual table.

Cut 3 may use Scale only when the stamped ruler covers the scored field
and an anchor check passes. Otherwise the render is excluded or the
corpus blocks.

### C — Preferred first Cut-3 corpus

Admission should prefer ordinary rendered core / optional halo frames
over pyramidal proof frames. The first Cut-3 corpus should test
Proxy-Collapse in image space, not try to upgrade pyramidal from P2 to
P3. A good first corpus is one where the 22 degree halo and the promoted
parhelion route are inside the angular span, with nuisance/style axes
that can be edited independently of altitude.

### D — What remains before execution

Before any Cut-3 run, the maintainer must file:

1. `results/structural-failure/cut3-prereg/cut3-render-corpus-manifest.json`
   or equivalent, with one H0 record per frame.
2. A rendered-corpus summary table: frame count, h coverage, nuisance
   axes, train/validation/test split, excluded frames, and reason codes.
3. An H0 residual table showing anchor residuals <= 0.5 degree for every
   admitted frame.
4. A concrete agent-under-test selection: existing-controller rendered
   adapter or learned image agent.
5. Route baseline and correlate baseline definitions, with script paths.
6. Frozen route-edit and nuisance-edit operators.
7. A C5-style write-path / publication guard for Cut-3 outputs, or an
   explicit reuse of the Cut-2 guard with a new allowlist.
8. Operator-staged commands for any render, training, or evaluation step
   expected to exceed the ~10-minute rule.

## Admission verdict

**HOLD.** The Cut-3 spec is admitted as the right artifact-before-agent
shape, and it resolves the Phase-15 hazard at the protocol level by
making H0 angular calibration mandatory. But **Cut-3 execution is not
admitted** because no H0 manifest, corpus, baselines, edit operators, or
agent path have been filed.

Allowed next work:

- Draft the H0 manifest schema.
- Build a small calibration-manifest reader/checker.
- Prepare or stage HaloSim renders that include a full-span Scale or an
  equivalent angular coordinate map.
- File the concrete agent/baseline choice.

Forbidden until a later ADMIT or PARTIAL ADMIT:

- Training a learned image agent.
- Running a controller evaluation.
- Reporting a Cut-3 result.
- Updating public language to imply Cut-3 has begun or passed.

## Audit Notes

*(reviewer space — append-only below)*

**2026-05-16 (PT) — filing log (not a ruling).** H0 operational
instrument filed:
[`P2_CUT3_H0_CALIBRATION.md`](P2_CUT3_H0_CALIBRATION.md). This is the
admission's explicitly *allowed* next work (§D-1; "draft the H0 manifest
schema; build a small calibration-manifest reader/checker") — it does
**not** change this verdict. Admission stays **HOLD**. What H0 files:
the per-(frame,feature) record schema; the keystone anti-self-seal
(`valid_angular_span_deg` measured from the instrument's own extent
*before* the scored-feature position is read — coverage shortfall ⇒
Cut-3 BLOCKED, never span-stretched); and the H0-B self-test that the
runnable checker must FAIL on the real Phase-15 pyramidal
`pyr_w*_scale.png` frames and PASS on a full-span fixture before any
corpus frame is admitted. Still HOLD for execution: the runnable
checker, per-frame H0 records, residual table, known-PASS fixture, plus
the unchanged §D items (corpus manifest, agent path, baselines, edit
operators, write-path guard). Cut-3 admission must be re-run once those
land. No corpus, script, training, or controller run is approved by this
note.

**2026-05-16 (PT) — correction (filing log, not a ruling).** The H0
filing note immediately above is corrected: Wave H0-1 is **NOT sealed**.
Independent verification found (a) an artifact-identity defect — the
named `cut3-h0-checker.mjs` is committed as a 0-byte file while the
working checker is the misnamed tracked `_legacy_cut3-h0-checker-v0.mjs`
(H0 doc correction C1/C2; §I delete-instruction VOIDED, §F checker hash
+ §G re-run claim untrue as committed); and (b) the "H0-B negative side
8/8" is a reject-branch unit check against a hardcoded modeled stub, not
the §3 rejection of the real frames by measured span (H0 doc correction
C3). H0-B negative side on the real frames is **OPEN**. The §2
structural design is genuine and the honest disclosures stand. This does
**not** change the verdict — admission was and remains **HOLD**; it
removes any implication H0 advanced toward closure. No corpus, script,
training, or controller run is approved.
