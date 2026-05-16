# Structural Failure Coincidence — Cut 3 Rendered-Signal Run Spec

Pre-registration: [`README.md`](README.md)
Parent run spec: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md)
Cut-2 disposition: [`P2_CUT2_WAVE42_DISPOSITION.md`](P2_CUT2_WAVE42_DISPOSITION.md)
Admission check: [`P2_CUT3_ADMISSION.md`](P2_CUT3_ADMISSION.md)
Phase-15 hazard source:
[`SPECULATIVE_HALO_PROOFS.md`](../../calibration/SPECULATIVE_HALO_PROOFS.md)
Controller binding: [`P2_CUT2_C1_CONTROLLER_BINDING.md`](P2_CUT2_C1_CONTROLLER_BINDING.md)
Filed: **2026-05-16 (PT)**. Status: **CUT-3 SPEC FILED — EXECUTION
HELD**. This document opens the Cut-3 documentation front as an
artifact-before-agent record. It runs nothing, trains nothing, and admits
no controller by itself.

## Purpose

Cut 2 closed the closed-form line honestly: C3-A-R removed the Cut-1
vacuity, but C3-A-T/B blocked because the closed-form decoy cannot
compete with the promoted parhelion route where that route is eligible.
That is the gamma regime-separability finding. Cut 3 is the
pre-registered rendered-signal escalation: test whether a learned or
image-native correlate can compete in the eligible band where scalar
closed-form decoys structurally could not.

Cut 3 is not a consolation run. It is a different signal class with a
known hard dependency from Phase 15: the pixel-to-degree / centering
hazard. This spec resolves that hazard at the protocol level by making
angular calibration an admission requirement. If the render corpus
cannot satisfy that requirement, Cut 3 is blocked.

## H0 — angular-calibration gate (the Phase-15 hazard)

Every rendered frame admitted to Cut 3 must have a per-frame angular
calibration record that maps image pixels to sun-centered angular
coordinates before the controller or any learned baseline sees the
image.

The calibration record must include:

1. `sun_px`: the image-space sun center or HaloSim-native sun-origin
   receipt.
2. `projection`: the projection family used by the frame.
3. `theta_map`: a deterministic function or tabulated ruler that maps
   pixel location to angular distance from the sun, in degrees.
4. `valid_angular_span_deg`: the angular span over which the map is
   calibrated.
5. `anchor_check`: a cross-check against at least one known halo locus
   inside the valid span, preferably the 22 degree halo; if a scored
   feature lies beyond the span, the frame is inadmissible for that
   feature.
6. `source_hashes`: hashes for the render and its calibration artifact.

Allowed ways to satisfy H0:

- **HaloSim-native Scale, full-span.** A scale-stamped render whose ruler
  covers every scored feature and at least one anchor locus. This is the
  preferred HaloSim path because the ruler is drawn from the sun and
  carries HaloSim's own projection.
- **Renderer-native angular metadata.** A renderer or export mode that
  directly emits a sun-centered angular coordinate map. The coordinate
  map must be saved as an artifact and hashed.
- **Independently validated 2D angular fit.** A non-split render with at
  least two known loci inside the fitted range, such as 22 and 46
  degrees, plus residuals below the tolerance below.

Disallowed as H0:

- A global pixels-per-degree constant copied between renders.
- Any scale-lock that changes with HaloSim auto-zoom but is not
  re-measured per frame.
- A ruler or fitted map whose span is shorter than the scored feature
  field.
- A center chosen by whichever downstream detector gives the desired
  result.
- Any frame where the filename, directory, overlay text, or embedded
  metadata gives the hidden altitude to the agent.

H0 tolerance: anchor residuals must be **<= 0.5 degree** for every anchor
locus used to admit a frame. A feature outside the valid angular span is
not scored; if too many features are outside span, Cut 3 remains blocked
for that corpus.

Phase-15 precedent: the HaloSim Scale method is accepted as a real
instrument, but the pyramidal scale-stamped frames are a negative H0
example for Cut 3 because the ruler span was shorter than the ring field
and the 22/46 degree anchors were beyond the calibrated tip. The lesson
is not "Scale always passes"; it is "a scale receipt must cover the
feature field or the run blocks."

## Signal and corpus

Cut 3 input is a rendered image, not the closed-form feature bundle.
The hidden cause remains sun altitude `h`. The agent must not receive
`h`, altitude-coded filenames, render-order cues, or calibration sidecar
values as inputs.

Minimum corpus before execution:

- `h` grid: at least **0 to 40 degrees** with step no coarser than
  **2 degrees**, with denser samples within **+/- 3 degrees** of the L1,
  L2, and L3 boundaries when feasible.
- Split: train / validation / held-out test by altitude and nuisance
  condition, with no identical render seed appearing in both train and
  test.
- Nuisance axes: at least two render/style axes that can be varied
  independently of `h` for counterfactual tests. Examples: ray seed,
  crop, color mode, contrast curve, non-scored crystal population, or
  Monte Carlo noise level.
- Calibration manifest: one H0 record per frame.

The first admissible corpus should prefer ordinary rendered core handles
over pyramidal proof frames. Pyramidal Phase 15 remains a P2/P3 proof
chain with its own failure modes; Cut 3 needs an image-domain
Proxy-Collapse test, not a pyramidal residual table.

## Agent and baselines

The agent-under-test path must be named in Cut-3 admission before any
run. Two families are allowed, but must not be mixed silently:

- **Existing-controller rendered adapter.** The bound
  `PhotometricAgent` receives a scalar photometric objective derived
  from the rendered image. The adapter must be fixed before execution.
- **Learned image agent.** A learned policy or regressor receives the
  rendered image. Architecture, optimizer, training budget, and
  early-stop rule must be frozen before training.

Required baselines:

- **Route baseline.** A transparent route extractor that uses only
  admitted angular handles from the H0-calibrated image, with the same
  abstain rules as BOUNDARY_MAP.
- **Correlate baseline.** A deliberately image-correlate baseline that
  is allowed to use nuisance/style features but not the angular route.
  It must be strong enough to move under nuisance edits; otherwise the
  decoy battery is inconclusive.
- **Analytic reference.** The closed-form parhelion route on the
  calibration sidecar, used only for scoring and sanity checks, never as
  an agent input.

## Quantities

The P2 four-quantity structure carries forward. Quantities 1-3 are
traceability; quantity 4 is efficiency/reporting only.

1. **Rendered convergence.** On H0-admitted and BOUNDARY_MAP-eligible
   frames, the agent's altitude estimate must be within **1.5 degrees**
   of hidden `h` on at least **90%** of held-out test frames.
2. **Counterfactual steerability.** Two edits are required:
   - *Route edit:* alter the rendered angular handle while holding
     nuisance/style cues fixed. Pass iff the agent follows the edited
     route within **2.0 degrees**.
   - *Nuisance edit:* alter style/metadata/image nuisance while holding
     the angular route fixed. Pass iff the agent moves by **<=0.5
     degree**, while the correlate baseline moves by at least **2.0
     degrees**. If the correlate baseline does not move, the edit is too
     weak and the result is inconclusive, not pass.
3. **Failure-boundary coincidence.** The agent must degrade, abstain, or
   switch at the BOUNDARY_MAP loci:
   - L1 low leverage / invalid parhelion route.
   - L2 CZA disappearance at the operative 32 degree coded guard.
   - L3 tangent-to-circumscribed merge at 29 degrees.
   - L4 supralateral remains a non-handle.
   - L5 rendered does not equal anchored; unanchored rendered primitives
     may act as nuisance cues but never as route evidence.
4. **Efficiency and robustness.** Report samples, wall-clock, render
   count, and failure modes. No pass/fail threshold.

## Outcome mapping

The P2 outcome table still governs:

- Fails convergence: convergence null.
- Converges but follows nuisance edits or ignores route edits:
  Proxy-Collapse / opaque-correlate confirmation.
- Converges and steers but sails through BOUNDARY_MAP boundaries:
  boundary failure; no traceability claim.
- Passes convergence, steerability, and boundary coincidence:
  narrow apparatus result only, never theorem language.

Cut 3 must also report an H0 disposition:

- **H0 PASS:** corpus is angularly calibrated and may be used for Cut-3
  admission.
- **H0 BLOCK:** corpus lacks valid angular calibration; no controller
  or learned-agent run may start.
- **H0 PARTIAL:** some frames/features are admitted and others excluded;
  execution may start only if coverage remains sufficient after
  exclusions.

## Publication and language guard

Until an admitted Cut-3 run passes quantities 1-3, public surfaces may
say only:

> Cut 3 is the rendered-signal escalation and is gated on per-frame
> angular calibration.

They may not say Cut 3 has begun, that a rendered traceability harness
passes, or that the controller is traceable. If H0 blocks, the block is
the result.

## Execution discipline

No Cut-3 script, corpus generator, controller training run, or long
HaloSim sweep may run until [`P2_CUT3_ADMISSION.md`](P2_CUT3_ADMISSION.md)
records ADMIT or a narrower PARTIAL ADMIT. Any run expected to exceed
the repository's ~10-minute rule must be staged as exact PowerShell
commands with wall-clock estimates and read-back paths.

## Audit Notes

*(reviewer space — append-only below)*
