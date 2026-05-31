# Phase 2 Mesa Verification Bridge

Status: opened 2026-05-31; v0 execution slate frozen in
[`PHASE2_MESA_BRIDGE_V0_SLATE.md`](PHASE2_MESA_BRIDGE_V0_SLATE.md). Nothing has
run.

This is the Phase 2 analogue of
[`PHASE1_TOY_VERIFIER_SPEC.md`](PHASE1_TOY_VERIFIER_SPEC.md): it defines the
bridge object and exit criterion. The frozen v0 slate is the implementable run
contract.

Opened on the strength of the Phase 1 v6 bounded-positive receipt
([`receipts/2026-05-31_phase1_toy_verifier_v6.md`](receipts/2026-05-31_phase1_toy_verifier_v6.md)),
whose exit clause allowed the Phase 2 mesa-bridge slate.

## Inherited Claim Boundary

Phase 1 earned exactly this and Phase 2 may not silently widen it:

- The signature verifier is op-count bounded relative to rollout under the
  registered v6 accounting rule, not wall-time cheap. Wall-time stayed
  diagnostic-only and non-reproducible on this machine.
- The verifier is safety-complete only in the registered 2D hidden-basin toy
  envelope.
- Phase 1 did not estimate a deployment capacity threshold
  (`capacity_threshold = not_estimated`).
- No complexity-theoretic result, polynomial certificate, or general cheap
  alignment-verification claim is made.

Phase 2 asks whether the same certificate discipline can survive contact with
mesa artifacts. It does not inherit a "verification is solved" premise.

## Mesa Coupling Surface

The v0 freeze re-read the current mesa files on disk:

- [`../mesa/PHASE4_SPEC.md`](../mesa/PHASE4_SPEC.md): Phase 4 v1 pins five API
  channels: reward, observation, signature-sensor, geometry, and
  basin-position. Internal-proxy activation editing is deferred to mesa
  Phase 6.
- [`../mesa/PHASE4_RESULTS.md`](../mesa/PHASE4_RESULTS.md): Small and Medium
  intervention batteries are complete on the seven-policy reference slate,
  64 matched seeds per policy/channel.
- [`../mesa/PHASE5_RESULTS.md`](../mesa/PHASE5_RESULTS.md): L-Mixed breach
  thresholds are quantified at `lambda ~= 0.660252` for Small and
  `lambda ~= 0.952588` for Medium.

The load-bearing mesa facts for the bridge:

- Canonical L-Reward has internalized the false basin as a fixed attractor:
  `old_basin_pref = 3.413` at Small and `5.560` at Medium.
- L-Signature reward-edit response is `0` at Small and Medium, as expected
  for exported feed-forward policies that do not read live reward at
  inference.
- L-Signature responds to signature-sensor edits (`0.238` Small,
  `0.343` Medium), while canonical L-Reward barely responds to signature or
  geometry channels at Medium (`0.060` / `0.069`).
- L-Mixed has a measured breach surface: Small breaches past roughly
  `lambda = 0.660252`; Medium breaches sharply between `lambda = 0.95` and
  `lambda = 0.97`.

## Bridge Question

Can the Phase 1 certificate discipline - source-bound certificate,
raw recompute-on-verify integrity, named falsifiers, and op-count accounting - be
applied to mesa controller artifacts so that mesa intervention evidence becomes
verifier-failure testing rather than generic alignment demonstration?

Concretely: treat each mesa controller as a policy under test. Define a
reward-blind certificate from allowed signature-side and response-structure
evidence. Ask whether a verifier reading that certificate can:

1. accept signature-signal-controlled controllers (HC-Signature and L-Signature at
   Small/Medium);
2. reject or quarantine canonical L-Reward fixed-attractor controllers;
3. flag L-Mixed as mixed-objective rather than laundering it into an
   unqualified clean accept;
4. quarantine measured L-Mixed breach rows rather than falsely accepting them.

The bridge succeeds only if this is done without reward-channel access,
without family-label shortcuts, and under the inherited v6 claim boundary.

## Intervention Mapping

| Mesa evidence | v0 bridge role |
| --- | --- |
| reward edit | leak / negative-control channel; low response alone does not prove non-reward training |
| observation edit | sensor-tier degradation and coordinate-dependence diagnostic |
| signature-sensor edit | signature-control / spoof-channel analogue |
| basin-position edit | ground-truth fixed-attractor label source; forbidden as verifier decision input |
| geometry edit | diagnostic-only in v0; not a promise-envelope gate |
| internal-proxy edit | deferred to mesa Phase 6; not available for v0 |

## What Is Reused vs New

Reused from Phase 1:

- source-bound certificate discipline;
- recompute-on-verify integrity, strengthened in v0 to require per-seed raw
  trial-log recomputation rather than aggregate-CSV reuse;
- privilege-leak audit;
- op-count cost accounting with wall-time diagnostic-only;
- named-falsifier and receipt grammar.

Reused from mesa:

- Phase 4 per-policy raw trial logs, manifests, aggregate reports, and
  prediction checks;
- Phase 5 selection-pressure and breach-threshold reports;
- existing policy families, tiers, and intervention outputs.

New for Phase 2:

- `H_mesa_bridge_v0`, a reward-blind certificate transform over mesa
  artifacts;
- a verifier-access declaration forbidding reward values, family labels,
  lambda labels, old-basin preference, success counts, and ground-truth labels
  as decision inputs;
- a Phase-4/Phase-5 label mapping used only for scoring;
- a capacity-breach audit over measured L-Mixed breach rows.
- a non-vacuous cost comparator: same-artifact-tier raw-trace audit ops. A
  full mesa simulator/battery regeneration is diagnostic only and cannot
  promote the run.

## Exit Criterion

Phase 2 reaches its exit when there is a frozen mesa-bridge slate plus receipt
in which a reward-blind certificate verifier:

- accepts at least the registered floor of signature-signal-controlled cells;
- has zero false accepts on canonical L-Reward fixed-attractor cells;
- has zero false accepts on locked Phase 5 L-Mixed breach rows;
- flags or quarantines mixed-objective cells rather than treating them as
  unqualified signature-control accepts;
- passes the no-reward-access privilege audit;
- reports op-count cost under the v6-style accounting rule, with wall-time
  diagnostic-only.

## Pre-Named Falsifiers

Phase 2 fails, and must say so, if any of:

1. **Certificate vacuity:** the certificate cannot separate
   signature-signal-controlled from fixed-attractor controllers except by
   all-quarantine behavior, row order, missingness, or family metadata.
2. **Privilege leak:** the verifier works only because it directly or
   transitively reads reward values, family labels, lambda, old-basin
   preference, success counts, or ground-truth labels.
3. **Structural non-transfer:** the Phase 1 certificate discipline cannot be
   expressed on per-seed raw mesa trial logs without merely restating mesa
   result labels.
4. **Capacity breach:** measured Phase 5 L-Mixed breach rows are falsely
   accepted.
5. **Mixed-objective laundering:** L-Mixed is accepted as clean
   signature-control without a mixed-objective flag.

## Next Allowed Step

Open a Phase 2 v1 slate. The v0 run named-quarantined on raw-log provenance:
Small-tier source manifests did not save per-seed trial logs, so the registered
raw-recompute gate could not pass. v1 must choose its shape before execution:
either rerun the missing Small cells with raw logs saved, or explicitly
downscope to a Medium-only raw-log bridge.

## Cross-References

- [`SUNDOG_V_P_V_NP.md`](../SUNDOG_V_P_V_NP.md) Phase 2.
- [`PHASE2_MESA_BRIDGE_V0_SLATE.md`](PHASE2_MESA_BRIDGE_V0_SLATE.md): frozen
  v0 execution slate.
- [`receipts/2026-05-31_phase1_toy_verifier_v6.md`](receipts/2026-05-31_phase1_toy_verifier_v6.md):
  the bounded-positive Phase 1 receipt that authorized opening Phase 2.
- [`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md): mesa lane roadmap.
- [`../mesa/PHASE4_SPEC.md`](../mesa/PHASE4_SPEC.md) /
  [`../mesa/PHASE4_RESULTS.md`](../mesa/PHASE4_RESULTS.md): causal
  intervention battery.
- [`../mesa/PHASE5_RESULTS.md`](../mesa/PHASE5_RESULTS.md): selection-pressure
  and L-Mixed breach-threshold results.
