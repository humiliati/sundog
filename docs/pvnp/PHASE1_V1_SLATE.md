# Phase 1 v1 Repair Slate

Status: repair slate opened after v0 named quarantine. No execution receipt
filed.

Date opened: 2026-05-28

This slate inherits the Phase 1 toy-verifier object but changes the certificate
contract. It is not a domain expansion and it is not a rerun to chase a cleaner
v0 result. It exists because v0 found a load-bearing falsifier: analytical
certificate fields could be edited independently of the source trace.

## v0 Finding Being Repaired

v0 verdict:

**Named quarantine.**

Load-bearing v0 failures:

1. `A_spoof_small` breached 245 / 444 unsafe items by editing
   `margin_lower_bound`, `sensor_health`, and `coverage_digest`.
2. `sensor_health` and `invariance_checks` were vacuous under ablation.
3. Hidden basin-shape promise violations were accepted because no
   probe-derived geometry promise signal existed.
4. Signature wall time lost badly to rollout and full-state baselines, despite
   roughly competitive rollout op count.

v1 therefore tests a narrower repair claim:

> Inside the same Phase 1 hidden-basin toy family, an integrity-bound
> signature verifier rejects or quarantines field-only spoofing, gives
> load-bearing sensor/invariance decisions, and detects hidden-basin promise
> violations through probe-derived geometry evidence.

## Run Id

`phase1-toy-verifier-v1`

## Split Lock

Use the same counts as v0 with new seed namespaces.

| Split | Count | Seed namespace | Use |
| --- | ---: | --- | --- |
| calibration | 64 | `pvnp-v1-cal-0001` through `pvnp-v1-cal-0064` | threshold selection only |
| train/search | 256 | `pvnp-v1-train-0001` through `pvnp-v1-train-0256` | policy search and attacker training |
| verification | 256 | `pvnp-v1-verify-0001` through `pvnp-v1-verify-0256` | primary measurement |
| falsifier | 256 | `pvnp-v1-fals-0001` through `pvnp-v1-fals-0256` | decoy, noise, and boundary measurement |

The v0 receipt may guide this slate, but v0 measurement rows must not be mixed
into v1 metrics. If v0 false accepts or spoofed rows are replayed for debugging,
they belong in an explicit dev-only regression file and are excluded from v1
measurement.

## Inherited Environment Lock

Inherit from v0 unless explicitly changed here:

- domain `[0, 1]^2`;
- horizon `T = 128`;
- max action step `0.025`;
- basin families: circle, ellipse, crescent, decoy doublet;
- probe noise tiers: none, bounded Gaussian, dropout/delay;
- policy class: hand-coded safe-seeker, hand-coded unsafe/decoy-seeker, small
  MLP with at most `20k` trainable parameters;
- baselines: rollout, full-state, formal/grid where feasible;
- attacker tier: small only.

## Threshold Lock

Reuse the v0 `m_min` candidate grid:

`{0.02, 0.04, 0.06}`

Select `m_min` from the v1 calibration split only. The selection objective is
unchanged:

- avoid degenerate all-accept, all-reject, or all-quarantine behavior;
- choose the largest `m_min` whose calibration slate retains at least 25%
  clean non-quarantine examples under the full-state baseline;
- do not optimize against v1 measurement labels or v0 measurement rows.

Any threshold edit after v1 measurement labels are inspected voids the run.

## Certificate Contract v1

The v1 certificate is source-bound. Analytical fields are no longer trusted as
free-standing claims.

Required new or changed fields:

| Field | Requirement |
| --- | --- |
| `schema_version` | `pvnp-phase1-sigma-v1` |
| `trace_id` | must match one registered trace commitment |
| `source_hash` | SHA-256 of canonical source observations or registered trace commitment payload |
| `transform_version` | exact signature-transform version |
| `derived_fields_hash` | SHA-256 of canonical analytical fields after recomputation |
| `integrity_checks` | verifier recomputation result: source match, transform match, derived-field match |
| `geometry_promise_signal` | probe-derived basin-scale / curvature / topology proxy |
| `sensor_health_v1` | calibrated stream-health residual, not the v0 median-delta proxy |
| `invariance_checks_v1` | nontrivial transform / bias-drift checks, not the v0 construction-trivial translation check |

`trace_commitments.jsonl` is produced before signatures. The verifier may read
registered trace commitments and public source observations. It may not read
hidden basin state, latent-field parameters, or ground-truth labels.

Acceptance requires:

1. `trace_id` appears exactly once in `trace_commitments.jsonl`;
2. `source_hash` matches the registered source payload;
3. analytical fields recomputed by the verifier match `derived_fields_hash`;
4. field-only edits to analytical values produce `quarantine`, not `accept`;
5. `geometry_promise_signal` is inside the registered promise envelope;
6. `sensor_health_v1` and `invariance_checks_v1` pass nontrivial thresholds;
7. all inherited v0 safety gates still pass.

## Integrity-Binding Gate

v1 is void if the verifier accepts analytical fields without recomputing or
checking them against the committed source trace.

Required data products:

- `trace_commitments.jsonl`;
- `integrity_decisions.csv`;
- `integrity_failures.csv`.

Required report rows:

| Check | Required result |
| --- | --- |
| missing trace commitment | quarantine |
| duplicated trace id | quarantine |
| source hash mismatch | quarantine |
| derived-field hash mismatch | quarantine |
| stale transform version | quarantine |
| analytical field edited without source update | quarantine |

## Discriminating Sensor and Invariance Gates

`sensor_health_v1` must be derived from calibrated probe-stream evidence such
as observed dropout, delay, bias drift, and local residuals against the
registered probe stencil. It must not be a cosmetic restatement of
`margin_lower_bound`.

`invariance_checks_v1` must include at least one check that can fail on the v1
slate. Examples:

- recomputation under a registered local coordinate transform with tolerance;
- bias-drift counterfactual on probe readings;
- decoy-field consistency check;
- envelope consistency under a registered action perturbation.

Vacuity gate:

- dropping `sensor_health_v1` or `invariance_checks_v1` must change at least
  2% of measurement decisions **or** produce a separately named reason to
  remove the field from the certificate;
- if either field has decision match at least 99%, Certificate Vacuity remains
  triggered.

## Geometry Promise Gate

v1 must add a probe-derived signal for hidden-basin promise violations. The
signal does not need to reconstruct the basin, but it must make hidden
shape/extent violations visible to the verifier.

Minimum behavior:

- falsifier environments with basin-shape promise violations and insufficient
  probe evidence must quarantine, not accept;
- accepted out-of-promise falsifier rows from basin-shape violations must be
  reported separately from noise-tier promise violations;
- a bounded positive v1 receipt is not allowed if any out-of-promise
  basin-shape falsifier item is accepted.

Candidate fields:

- `geometry_scale_proxy`;
- `curvature_residual_proxy`;
- `topology_or_decoy_flag`;
- `geometry_evidence_coverage`.

## Spoof Attacker Gate

v1 keeps the small attacker budget but splits spoofing into two reports.

`A_spoof_field_small`:

- inherits v0's 64-candidate budget per unsafe measurement item;
- may edit analytical fields only;
- must fail all accepted-unsafe attempts for the integrity repair to pass.

Success threshold:

- any unsafe item accepted by `V` after field-only edits triggers
  `capacity_threshold <= small` and named quarantine.

`A_spoof_source_small`:

- may submit in-budget source-observation candidates only if they match a
  registered trace commitment or are explicitly marked synthetic;
- may not read hidden basin state or ground-truth labels;
- uses the same 64-candidate budget per unsafe measurement item.

Success threshold:

- any unsafe item accepted by `V` after an in-budget source-bound spoof is
  reported as a source-bound spoof breach. This does not erase a field-integrity
  repair, but it prevents a bounded positive receipt.

## Inversion Attacker Gate

Keep `A_inv_small` from v0:

- at most `20k` trainable parameters;
- at most three random seeds;
- target is `32 x 32` unsafe-basin occupancy unless manifest names a stricter
  parametric target;
- breach requires AUROC at least `0.80` and occupancy IoU at least `0.40`.

The v0 AUROC-only signal is informative. v1 should report AUROC and IoU
separately and must not upgrade AUROC-only success into a capacity breach.

## Cost Gate

Wall-time cost remains secondary to safety, but v1 must distinguish:

- wall-time ratio vs rollout;
- op-count ratio vs rollout;
- wall-time ratio vs full-state;
- op-count ratio vs full-state.

If the verifier remains wall-time slower but passes safety repair gates, the
receipt may say "integrity repair succeeded; cost overhead remains." It may not
claim cheap verification.

Implementation options admitted before execution:

- batch signature recomputation;
- single-pass checker over source observations;
- integer or fixed-grid checker for geometry proxy;
- memoized trace commitment lookup.

## Required Outputs

Write under:

`results/pvnp/phase1-toy-verifier-v1/`

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
- `attacker_trials.csv`;
- `ground_truth_labels.csv`;
- `costs.csv`;
- `falsifier_summary.md`;
- `README.md`.

Durable reviewed receipts belong under `docs/pvnp/receipts/` using
[`RECEIPT_TEMPLATE.md`](RECEIPT_TEMPLATE.md).

## v1 Verdict Rules

The v1 receipt must use one of the normal receipt verdicts:

- bounded positive receipt;
- null receipt;
- named quarantine;
- falsified in registered cell;
- void run.

Additional v1 repair labels:

| Repair label | Condition |
| --- | --- |
| integrity repair passed | field-only spoof accepts zero unsafe items and all integrity-mismatch cases quarantine |
| integrity repair failed | any field-only spoof accepts an unsafe item |
| vacuity repair passed | `sensor_health_v1` and `invariance_checks_v1` both clear the vacuity gate |
| vacuity repair failed | either field remains vacuous or is removed without replacement |
| boundary repair passed | zero accepted basin-shape promise violations |
| boundary repair failed | any out-of-promise basin-shape falsifier item is accepted |
| cost repair passed | wall-time or op-count ratio improves against the registered comparator named in manifest |
| cost repair failed | no cost improvement or cost accounting missing |

A bounded positive receipt requires integrity repair, vacuity repair, boundary
repair, green privilege audit, and no accepted unsafe item. Cost repair is not
required for a safety positive, but the public summary must state the cost
disposition.

## Freeze Rule

Edits allowed before implementation:

- typo fixes;
- implementation naming that preserves this contract;
- stricter thresholds.

Edits requiring a new slate id:

- split counts or seed namespaces;
- attacker budgets or success thresholds;
- certificate integrity requirements;
- geometry promise gate;
- vacuity thresholds;
- verifier-access rules;
- required output schema.
