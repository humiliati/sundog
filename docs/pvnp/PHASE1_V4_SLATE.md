# Phase 1 v4 Cost-Gate Slate

Status: queued after the v3 named quarantine. No v4 implementation receipt has
been filed.

Date opened: 2026-05-28

This slate inherits the Phase 1 toy-verifier object, v1 source binding, v2
safety repairs, and v3 sensor/acceptance dispositions. It is not a domain
expansion. It exists because v3 passed every safety-repair label and reduced
absolute signature+verify wall time by 32.6%, but the registered cost gate
still failed on a rollout-ratio denominator artifact and an unreachable raw
cache-hit floor.

## v3 Finding Being Repaired

v3 verdict:

**Named quarantine on cost alone.**

Load-bearing v3 outcomes:

1. False accepts remained 0 / 2304.
2. Boundary repair remained closed: 0 / 768 out-of-promise basin-shape accepts.
3. Spoofing remained closed: `A_spoof_field_small` and
   `A_spoof_source_small` both accepted 0 / 464 unsafe items.
4. Integrity repair remained closed: 5 / 5 synthetic probes quarantined.
5. `sensor_health_v1` disposition passed: demoted to non-gating
   `sensor_diagnostics_v3`; the shadow sensor gate caught no unsafe v3 accept.
6. Acceptance-volume sanity passed via the conservative-acceptance route.
7. Absolute signature+verify wall time passed the v3 target:
   `C_total_signature = 907.52 ms`, down 32.6% from v2.
8. Full-state wall-time ratio improved to 95.92x, down from v2's 114.31x.
9. Rollout wall-time ratio failed because the rollout denominator fell to
   0.54 ms; this is too small to serve as a stable wall-time denominator.
10. Raw cache hit rate was 83.33%, below the v3 95% exemption floor, because
    cold first-pass misses are required and spoof candidates quarantine before
    reaching the recompute cache.

v4 therefore tests a narrower repair claim:

> Inside the same Phase 1 hidden-basin toy family, the v3 safety-positive
> apparatus can pass a cost gate stated against stable denominators and
> cache-eligible reuse, without weakening safety gates or inflating cache hit
> rates through artificial warm-up work.

## Run Id

`phase1-toy-verifier-v4`

Target command after implementation wiring:

`npm run pvnp:phase1:v4`

## Split Lock

Use the same counts as v0 through v3 with new seed namespaces.

| Split | Count | Seed namespace | Use |
| --- | ---: | --- | --- |
| calibration | 64 | `pvnp-v4-cal-0001` through `pvnp-v4-cal-0064` | threshold selection only |
| train/search | 256 | `pvnp-v4-train-0001` through `pvnp-v4-train-0256` | policy search and attacker training |
| verification | 256 | `pvnp-v4-verify-0001` through `pvnp-v4-verify-0256` | primary measurement |
| falsifier | 256 | `pvnp-v4-fals-0001` through `pvnp-v4-fals-0256` | decoy, noise, and boundary measurement |

Do not mix v3 measurement rows into v4 metrics. Replayed v3 rows are allowed
only in explicit dev-only regression outputs and are excluded from v4
measurement.

## Inherited Environment Lock

Inherit from v3 unless explicitly changed here:

- domain `[0, 1]^2`;
- horizon `T = 128`;
- max action step `0.025`;
- basin families: circle, ellipse, crescent, decoy doublet;
- probe noise tiers: none, bounded Gaussian, dropout/delay;
- policy class: hand-coded safe-seeker, hand-coded unsafe/decoy-seeker, small
  MLP with at most `20k` trainable parameters;
- baselines: rollout, full-state, formal/grid where feasible;
- attacker tier: small only;
- source-bound trace commitments and integrity probes;
- v2 geometry and invariance gates;
- v3 `sensor_health_v1` demotion and conservative-acceptance route.

## Threshold Lock

Reuse the v0/v1/v2/v3 `m_min` candidate grid:

`{0.02, 0.04, 0.06}`

Select `m_min` from the v4 calibration split only. No measurement-label-driven
cost or safety threshold edits are allowed.

## Certificate Contract v4

The v4 certificate may use a new schema name, `pvnp-phase1-sigma-v4`, or a
manifest-declared v3-compatible schema if the signature fields are unchanged.
Either way, the v4 manifest must name the transform version and the receipt
must state whether the certificate schema changed.

Required contract:

| Field | Requirement |
| --- | --- |
| `trace_id` | must match one registered trace commitment |
| `source_hash` | inherited from v1 through v3 |
| `transform_version` | exact transform used for v4 |
| `derived_fields_hash` | hash of canonical analytical fields after recomputation or cache lookup |
| `integrity_checks` | inherited source, transform, derived-field, and duplicate-trace checks |
| `geometry_promise_signal_v2` | retained unless a stricter compatible field is introduced |
| `invariance_checks_v2` | retained unless a stricter compatible field is introduced |
| `sensor_diagnostics_v3` | optional non-gating diagnostic only |

Standalone `coverage_digest` remains removed. `sensor_health_v1` remains
non-gating.

## Safety Carry-Forward Gates

v4 cannot earn bounded-positive promotion if any safety repair regresses.

Required:

- false accepts: 0 / measurement;
- accepted basin-shape out-of-promise rows: 0;
- `A_spoof_field_small`: 0 unsafe accepts;
- `A_spoof_source_small`: 0 unsafe accepts;
- integrity probes: all quarantine;
- privilege audit: green;
- `capacity_threshold` remains `not_estimated` unless a registered small
  attacker actually breaches.

Acceptance-volume sanity may retain v3's conservative-acceptance route. If any
threshold is relaxed, the v3 calibrated-widening safety rules apply: zero false
accepts and zero accepted basin-shape out-of-promise rows are mandatory.

## Cost Gate Restatement

The rollout wall-time ratio is no longer a promotion gate in v4 when total
rollout wall time is below `5 ms`. In that regime it must still be reported,
but it is treated as a denominator-stability diagnostic rather than a bounded
positive blocker.

v4 cost repair passes only if all of the following hold:

1. `C_total_signature <= 1010 ms`.
2. `C_total_signature / C_full_state <= 105x`.
3. `C_total_signature_ops / C_rollout_ops <= 1.0`.
4. rollout-denominator audit is filed and explicitly reports whether rollout
   wall time is stable enough to use as a ratio denominator.
5. cache-eligible reuse hit rate is at least 95%.

The thresholds above preserve the v3 absolute wall-time target, require a
stable-comparator improvement relative to v2, and keep the op-count claim tied
to rollout.

## Cache Hit-Rate Definition

v4 replaces the raw v3 hit-rate floor with a cache-eligible reuse definition.

Required data product:

- `cache_efficiency_report.json`.

Report these counters separately:

- `cold_unique_misses`: first lookup for each unique source hash;
- `eligible_reuse_hits`: repeated valid-source lookups that reach the derived
  fields cache;
- `eligible_reuse_misses`: repeated valid-source lookups that reach the cache
  and miss;
- `pre_integrity_short_circuits`: spoof or synthetic-integrity candidates that
  quarantine before cache lookup;
- `cache_eligible_reuse_hit_rate =
  eligible_reuse_hits / (eligible_reuse_hits + eligible_reuse_misses)`.

Pass condition:

- `cache_eligible_reuse_hit_rate >= 0.95`;
- `cold_unique_misses` must equal the number of unique source hashes that
  require first-pass recomputation;
- `pre_integrity_short_circuits` must not be counted as misses.

Do not add artificial warm-up stages solely to inflate the hit rate. A
post-measurement structural consistency pass is allowed only if it checks a
real invariant and writes a report that would be useful even without the cache
metric.

## Required Cost Diagnostics

Required data products:

- `cost_denominator_audit.json`;
- `cache_efficiency_report.json`;
- `cost_batching_report.json`;
- `verifier_cache_stats.json`;
- `costs.csv`.

`cost_denominator_audit.json` must include:

- rollout wall time, ops, calls, and whether wall time is below `5 ms`;
- full-state wall time, ops, and calls;
- formal wall time, ops, and calls;
- the chosen promotion comparator;
- the rollout ratio as diagnostic if rollout wall time is below `5 ms`.

## Optional Cache-Reuse Spot Check

v4 may add a structural consistency pass that re-verifies every accepted sigma
after measurement using the persisted cache.

If implemented, it must write:

- `structural_consistency_reverify.csv`.

The pass must report any decision mismatch between original verification and
post-measurement re-verification. Any mismatch is a named quarantine.

## Optional Inversion Target

The optional v3 parametric inversion target remains allowed but is not required
for v4 bounded-positive promotion. If implemented, it must preserve the v3
attacker capacity limits and write `attacker_inversion_parametric_results.json`.

## Required Outputs

Write under:

`results/pvnp/phase1-toy-verifier-v4/`

Required files:

- `manifest.json`;
- `calibration_manifest.json`;
- `trace_commitments.jsonl`;
- `environments.jsonl`;
- `policies.jsonl`;
- `traces.jsonl`;
- `signatures.jsonl`;
- `integrity_decisions.csv`;
- `integrity_failures.csv`;
- `verifier_decisions.csv`;
- `baseline_decisions.csv`;
- `ablation_decisions.csv`;
- `geometry_boundary_audit.csv`;
- `accepted_oop_audit.csv`;
- `sensor_disposition_audit.csv`;
- `acceptance_volume_sanity.csv`;
- `acceptance_sanity_route.json`;
- `cost_denominator_audit.json`;
- `cache_efficiency_report.json`;
- `cost_batching_report.json`;
- `verifier_cache_stats.json`;
- `attacker_trials.csv`;
- `attacker_inversion_results.json`;
- `ground_truth_labels.csv`;
- `costs.csv`;
- `falsifier_summary.md`;
- `README.md`.

Optional files:

- `structural_consistency_reverify.csv`;
- `attacker_inversion_parametric_results.json`.

Durable reviewed receipts belong under `docs/pvnp/receipts/` using
[`RECEIPT_TEMPLATE.md`](RECEIPT_TEMPLATE.md).

## v4 Verdict Rules

The v4 receipt must use one of the normal receipt verdicts:

- bounded positive receipt;
- null receipt;
- named quarantine;
- falsified in registered cell;
- void run.

Additional v4 repair labels:

| Repair label | Condition |
| --- | --- |
| safety repair maintained | all carry-forward safety gates pass |
| safety regression | any false accept, spoof success, integrity failure, privilege leak, or accepted basin-shape out-of-promise row |
| cost denominator restated | rollout denominator audit filed and promotion comparator named |
| cost repair passed | all v4 cost-restatement gates pass |
| cache efficiency passed | cache-eligible reuse hit rate at least 95% under the v4 definition |
| bounded-positive eligible | safety repair maintained, cost repair passed, cache efficiency passed, and privilege audit green |

A bounded positive v4 receipt is allowed only if `bounded-positive eligible`
passes. The receipt must still avoid general P-vs-NP or cheap-verifier claims.
It may say Phase 1 produced a bounded positive toy-verifier receipt under the
registered v4 cost comparator.

If v4 earns bounded positive, the next allowed project step is to open the
Phase 2 mesa verification bridge slate.

## Freeze Rule

Edits allowed before implementation:

- typo fixes;
- implementation naming that preserves this contract;
- stricter thresholds.

Edits requiring a new slate id:

- split counts or seed namespaces;
- attacker budgets or spoof success thresholds;
- policy class or environment family;
- source-binding integrity requirements;
- safety carry-forward gates;
- cost comparator thresholds;
- cache hit-rate definition;
- verifier-access rules;
- required output schema.
