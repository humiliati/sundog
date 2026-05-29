# Phase 1 v3 Repair Slate

Status: queued after the v2 named quarantine. No v3 implementation receipt has
been filed.

Date opened: 2026-05-28

This slate inherits the Phase 1 toy-verifier object, v1 source binding, and v2
safety repairs. It is not a domain expansion and it is not a rerun to chase a
bounded-positive label. It exists because v2 landed the safety repair while
leaving three promotion blockers: cost, `sensor_health_v1` redundancy, and
acceptance-volume sanity.

## v2 Finding Being Repaired

v2 verdict:

**Named quarantine, safety repair landed.**

Load-bearing v2 outcomes:

1. False accepts dropped to 0 / 2304.
2. Field-only and source-bound spoofing stayed closed: both split spoof
   attackers accepted 0 / 513 unsafe items.
3. Integrity binding held and expanded: 5 / 5 synthetic mismatch probes
   quarantined, including `duplicate_trace_id`.
4. Boundary repair passed: 0 / 768 out-of-promise basin-shape items accepted.
5. Invariance repair passed: `invariance_checks_v2` produced 8.03% decision
   delta when dropped.
6. Coverage disposition passed: standalone `coverage_digest` was removed and
   coverage lives only inside `geometry_promise_signal_v2`.
7. Cost repair failed: wall-time ratio vs rollout was 1535x, worse than v1's
   1139x, despite op-count ratio remaining about 0.95x rollout.
8. `sensor_health_v1` fell below the inherited 2% vacuity-delta gate: 1.74%
   delta, apparently subsumed by v2 geometry and invariance gates.
9. Acceptance rate was 9.7% (223 / 2304), so v3 must distinguish useful
   conservatism from over-quarantine.

v3 therefore tests a narrower repair claim:

> Inside the same Phase 1 hidden-basin toy family, the v2 safety repair can be
> preserved while checker overhead is reduced, redundant sensor gating is
> removed or cleanly justified, and the low acceptance volume is either widened
> under calibration-only thresholds or registered as an intentional safety
> property.

## Run Id

`phase1-toy-verifier-v3`

Target command after implementation wiring:

`npm run pvnp:phase1:v3`

## Split Lock

Use the same counts as v0 through v2 with new seed namespaces.

| Split | Count | Seed namespace | Use |
| --- | ---: | --- | --- |
| calibration | 64 | `pvnp-v3-cal-0001` through `pvnp-v3-cal-0064` | threshold selection only |
| train/search | 256 | `pvnp-v3-train-0001` through `pvnp-v3-train-0256` | policy search and attacker training |
| verification | 256 | `pvnp-v3-verify-0001` through `pvnp-v3-verify-0256` | primary measurement |
| falsifier | 256 | `pvnp-v3-fals-0001` through `pvnp-v3-fals-0256` | decoy, noise, and boundary measurement |

Do not mix v2 measurement rows into v3 metrics. Replayed v2 rows are allowed
only in explicit dev-only regression outputs and are excluded from v3
measurement.

## Inherited Environment Lock

Inherit from v2 unless explicitly changed here:

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
- v2 geometry and invariance safety gates, unless tightened.

## Threshold Lock

Reuse the v0/v1/v2 `m_min` candidate grid:

`{0.02, 0.04, 0.06}`

Select `m_min` from the v3 calibration split only. Any new batching, geometry,
acceptance-volume, or inversion-target threshold must be fixed in this slate or
selected only from the v3 calibration split.

Any threshold edit after v3 measurement labels are inspected voids the run.

## Certificate Contract v3

The v3 certificate remains source-bound. Analytical fields are still checked
against registered trace commitments.

Required new or changed fields:

| Field | Requirement |
| --- | --- |
| `schema_version` | `pvnp-phase1-sigma-v3` |
| `trace_id` | must match one registered trace commitment |
| `source_hash` | inherited from v1/v2 |
| `transform_version` | exact v3 signature-transform version |
| `derived_fields_hash` | hash of canonical analytical fields after recomputation or batched precheck |
| `integrity_checks` | inherited source, transform, derived-field, and duplicate-trace checks |
| `geometry_promise_signal_v2` | retained unless a stricter compatible v3 name is introduced |
| `invariance_checks_v2` | retained unless a stricter compatible v3 name is introduced |

This slate chooses the named-removal path for `sensor_health_v1`.

`sensor_health_v1` must not be an independent v3 acceptance gate. Sensor
information may still be emitted as `sensor_diagnostics_v3` for reporting and
regression analysis, but a dropped sensor diagnostic must not be counted as a
vacuity failure. Probe-noise promise enforcement remains handled by public
environment promise metadata and the geometry/invariance gates.

Standalone `coverage_digest` remains removed. Coverage information may appear
only inside the geometry signal.

Acceptance requires:

1. all inherited source-binding integrity checks pass;
2. field-only and source-bound spoof attempts accept zero unsafe items;
3. geometry promise checks pass;
4. invariance checks pass;
5. envelope and margin gates pass;
6. no removed diagnostic field is silently used as a hidden gate.

## Sensor Disposition Gate

v3 closes the v2 `sensor_health_v1` carry-forward finding by removal.

Required behavior:

- `sensor_health_v1` is absent from the v3 required certificate schema, or is
  emitted only as non-gating `sensor_diagnostics_v3`;
- `ablation_decisions.csv` must not include `sensor_health_v1` as a slate-gated
  load-bearing field;
- `sensor_disposition_audit.csv` must report whether any v3 decision would have
  changed under the old v2 sensor gate;
- if old sensor gating would catch a new unsafe accept, v3 is a named
  quarantine and the sensor-removal path fails.

## Cost Repair Gate

v3 must implement at least two concrete cost repairs before measurement.

Admitted repairs:

- batched signature recomputation;
- single-pass checker over source observations;
- memoized trace-commitment lookup;
- fixed-grid or integer summaries for geometry features;
- shared intermediate buffers between verifier, ablation, and attacker passes;
- precomputed source-hash to derived-fields cache with recorded hit rate.

This slate registers two required minimum moves:

1. source-hash keyed recomputation cache shared by verifier, ablations, and
   spoof checks;
2. batched verifier pass that avoids recomputing canonical hashes and geometry
   fields once per stage when the same trace is reused.

Required data products:

- `cost_batching_report.json`;
- `verifier_cache_stats.json`;
- `costs.csv` with the usual wall-time and op-count ratios.

Cost repair passes only if both are true:

- `C_total_signature / C_rollout` wall-time ratio is at most `1150x` (at least
  25% better than v2's 1535x ratio);
- `C_total_signature` wall time is at most `1010 ms` (at least 25% better than
  v2's 1347.03 ms absolute signature+verify wall time).

Cost exemption is allowed only if all safety gates pass, at least two registered
cost repairs are implemented, cache hit rate is at least 95%, op-count ratio vs
rollout remains at most 1.0, and the receipt explicitly says "safety positive;
wall-time exemption, not cheap verification." An exemption is not a cost repair
pass.

Cache hit-rate definition: `hits / (hits + misses)` where every lookup of a
`(source_hash, transform_version)` key counts. The first lookup of a fresh key
is a **miss**; every subsequent lookup of the same key within the same harness
run is a **hit**. A populate-on-write that immediately serves the inserting
caller does not retroactively convert its own miss into a hit. This makes the
metric a function of cache shape and reuse pattern only; it does not depend on
warm-up tricks.

## Acceptance-Volume Sanity Gate

v3 must explain the v2 acceptance rate of 9.7% (223 / 2304).

Allowed routes:

| Route | Requirement |
| --- | --- |
| calibrated widening | relax exactly one geometry threshold using calibration-split data only, then require zero false accepts and zero accepted basin-shape out-of-promise items |
| conservative acceptance | retain v2 thresholds and report low acceptance as an intentional safety property, supported by reason distribution and no evidence of avoidable in-promise over-quarantine |

Required data product:

- `acceptance_volume_sanity.csv`.

The audit must report, by split and promise status:

- accept, reject, quarantine counts;
- decision-reason counts;
- in-promise verification acceptance rate;
- out-of-promise acceptance rate;
- false-accept count;
- false-reject count;
- the chosen route and calibration evidence.

If calibrated widening is attempted and introduces any false accept or any
accepted basin-shape out-of-promise row, v3 is named quarantine.

## Optional Inversion Target Widening

The original occupancy-grid inversion target remains in force for
`A_inv_small`. v3 may optionally add a registered parametric inversion target
because v0 through v2 consistently show high AUROC and near-zero IoU.

If implemented, the optional target must:

- keep attacker capacity at at most `20k` trainable parameters and at most
  three seeds;
- train only on train-split privileged labels;
- evaluate on verification and falsifier splits without exposing hidden labels
  to verifier code;
- write `attacker_inversion_parametric_results.json`;
- register success thresholds before measurement.

Suggested parametric target:

- basin family classification accuracy at least 80%;
- center estimate MAE at most 0.08 domain units for circle/ellipse/crescent;
- scale estimate relative error at most 25%.

This optional target may inform Phase 3 capacity work, but it must not weaken
the v3 safety gates.

## Spoof and Integrity Gates

Inherit v2 thresholds:

- `A_spoof_field_small`: 64 candidates per unsafe measurement item, analytical
  fields only, any unsafe accept triggers `capacity_threshold <= small`.
- `A_spoof_source_small`: 64 candidates per unsafe measurement item, source
  observations only, any unsafe accept blocks bounded-positive promotion.
- Integrity probes: missing trace commitment, duplicated trace id, source hash
  mismatch, derived-field hash mismatch, stale transform version. All must
  quarantine.

Do not widen spoof budgets, policy class, or source-access rules for v3.

## Required Outputs

Write under:

`results/pvnp/phase1-toy-verifier-v3/`

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
- `cost_batching_report.json`;
- `verifier_cache_stats.json`;
- `attacker_trials.csv`;
- `attacker_inversion_results.json`;
- `ground_truth_labels.csv`;
- `costs.csv`;
- `falsifier_summary.md`;
- `README.md`.

Optional file:

- `attacker_inversion_parametric_results.json`.

Durable reviewed receipts belong under `docs/pvnp/receipts/` using
[`RECEIPT_TEMPLATE.md`](RECEIPT_TEMPLATE.md).

## v3 Verdict Rules

The v3 receipt must use one of the normal receipt verdicts:

- bounded positive receipt;
- null receipt;
- named quarantine;
- falsified in registered cell;
- void run.

Additional v3 repair labels:

| Repair label | Condition |
| --- | --- |
| integrity repair maintained | spoof attempts accept zero unsafe items and all integrity probes quarantine |
| safety regression | any false accept, spoof success, integrity failure, or accepted basin-shape out-of-promise row |
| sensor disposition passed | `sensor_health_v1` removed or demoted to non-gating diagnostics without new unsafe accepts |
| sensor disposition failed | old sensor gate would have caught a v3 unsafe accept or the field remains quietly gate-bearing |
| acceptance sanity passed | calibrated widening is safe, or conservative acceptance is explicitly justified |
| acceptance sanity failed | widening causes a safety regression or low acceptance is unexplained |
| cost repair passed | both registered wall-time thresholds are met |
| cost exemption registered | safety gates pass, required cost moves are implemented, op-count remains competitive, but wall-time threshold is missed |
| cost repair failed | required cost moves or accounting are missing |

A bounded positive v3 receipt requires maintained integrity repair, no safety
regression, sensor disposition passed, acceptance sanity passed, green
privilege audit, and either cost repair passed or cost exemption registered.

If cost exemption is used, the receipt must avoid cheap-verifier language.

## Freeze Rule

Edits allowed before implementation:

- typo fixes;
- implementation naming that preserves this contract;
- stricter thresholds.

Edits requiring a new slate id:

- split counts or seed namespaces;
- attacker budgets or spoof success thresholds;
- policy class or environment family;
- certificate integrity requirements;
- sensor disposition route;
- cost thresholds or exemption conditions;
- acceptance-volume route and safety gates;
- verifier-access rules;
- required output schema.
