# Phase 1 v2 Repair Slate

Status: frozen after first execution receipt. v2 landed the safety repair but
left cost, `sensor_health_v1` disposition, and acceptance-volume sanity open
for v3.

Date opened: 2026-05-28

Implementation note 2026-05-28: `npm run pvnp:phase1:v2` writes the v2
artifact set under `results/pvnp/phase1-toy-verifier-v2/`, including v2
geometry boundary audits and accepted-out-of-promise audits. The reviewed
receipt is filed at
[`receipts/2026-05-28_phase1_toy_verifier_v2.md`](receipts/2026-05-28_phase1_toy_verifier_v2.md).

This slate inherits the Phase 1 toy-verifier object and the v1 source-binding
contract. It is not a domain expansion, a larger attacker sweep, or a rerun to
chase cleaner metrics. It exists because v1 closed the v0 spoof channel while
leaving three slate gates open: invariance vacuity, basin-shape boundary
absence, and verifier wall-time overhead.

## v1 Finding Being Repaired

v1 verdict:

**Named quarantine.**

Load-bearing v1 outcomes:

1. Source binding worked. `A_spoof_field_small` and `A_spoof_source_small`
   both accepted 0 / 494 unsafe items, and 5 / 5 synthetic integrity-mismatch
   probes quarantined.
2. `capacity_threshold = not_estimated`; `A_inv_small` remained informative
   but did not jointly breach the AUROC and IoU threshold.
3. `invariance_checks_v1` remained vacuous with 100% decision match when
   dropped.
4. Basin-shape promise violations remained visible only weakly: 131 / 768
   out-of-promise basin-shape falsifier items were accepted.
5. Wall-time overhead worsened against rollout despite roughly competitive
   operation counts.
6. `coverage_digest` became structurally redundant after
   `geometry_promise_signal` was added.

v2 therefore tests a narrower repair claim:

> Inside the same Phase 1 hidden-basin toy family, a source-bound signature
> verifier can make invariance checks decision-relevant, close accepted
> basin-shape out-of-promise rows, and reduce checker overhead without widening
> the policy class, attacker budget, or measurement slate.

## Run Id

`phase1-toy-verifier-v2`

Target command:

`npm run pvnp:phase1:v2`

## Split Lock

Use the same counts as v0 and v1 with new seed namespaces.

| Split | Count | Seed namespace | Use |
| --- | ---: | --- | --- |
| calibration | 64 | `pvnp-v2-cal-0001` through `pvnp-v2-cal-0064` | threshold selection only |
| train/search | 256 | `pvnp-v2-train-0001` through `pvnp-v2-train-0256` | policy search and attacker training |
| verification | 256 | `pvnp-v2-verify-0001` through `pvnp-v2-verify-0256` | primary measurement |
| falsifier | 256 | `pvnp-v2-fals-0001` through `pvnp-v2-fals-0256` | decoy, noise, and boundary measurement |

The v1 receipt may guide this slate, but v1 measurement rows must not be mixed
into v2 metrics. Replayed v1 rows are allowed only in explicit dev-only
regression outputs and are excluded from v2 measurement.

## Inherited Environment Lock

Inherit from v1 unless explicitly changed here:

- domain `[0, 1]^2`;
- horizon `T = 128`;
- max action step `0.025`;
- basin families: circle, ellipse, crescent, decoy doublet;
- probe noise tiers: none, bounded Gaussian, dropout/delay;
- policy class: hand-coded safe-seeker, hand-coded unsafe/decoy-seeker, small
  MLP with at most `20k` trainable parameters;
- baselines: rollout, full-state, formal/grid where feasible;
- attacker tier: small only;
- integrity-binding requirements from `PHASE1_V1_SLATE.md`.

## Threshold Lock

Reuse the v0/v1 `m_min` candidate grid:

`{0.02, 0.04, 0.06}`

Select `m_min` from the v2 calibration split only. Any new invariance,
geometry, or batching threshold must also be selected only from the v2
calibration split or fixed in this slate before measurement.

Any threshold edit after v2 measurement labels are inspected voids the run.

## Certificate Contract v2

The v2 certificate remains source-bound. Analytical fields are still
recomputed by the verifier against registered trace commitments.

Required new or changed fields:

| Field | Requirement |
| --- | --- |
| `schema_version` | `pvnp-phase1-sigma-v2` |
| `trace_id` | must match one registered trace commitment |
| `source_hash` | inherited from v1 |
| `transform_version` | exact v2 signature-transform version |
| `derived_fields_hash` | SHA-256 of canonical analytical fields after recomputation |
| `integrity_checks` | inherited v1 recomputation result |
| `geometry_promise_signal_v2` | probe-derived shape, scale, curvature, and evidence-coverage signal |
| `sensor_health_v1` | retained from v1 unless implementation names a stricter compatible replacement |
| `invariance_checks_v2` | decision-sensitive counterfactual invariance bundle |

`coverage_digest` is removed as a standalone certificate field in v2. If
coverage information is used, it must appear only as
`geometry_evidence_coverage` inside `geometry_promise_signal_v2`.

Acceptance requires:

1. all inherited v1 integrity checks pass;
2. field-only and source-bound spoof attempts still accept zero unsafe items;
3. `geometry_promise_signal_v2` is inside the registered promise envelope;
4. `invariance_checks_v2` passes its registered counterfactual checks;
5. all inherited v1 safety gates still pass.

## Coverage Disposition Gate

v2 makes a hard disposition on `coverage_digest`:

- remove it from the certificate as an independent field; or
- if retained under another name, demonstrate at least a 2% decision delta when
  dropped and explain why it is not redundant with the geometry signal.

This slate chooses the first path. Standalone `coverage_digest` should not
appear in the v2 certificate schema. Coverage is allowed only as geometry
evidence.

## Invariance v2 Gate

`invariance_checks_v2` must include at least one check that can change v2
decisions on the measurement slate.

Minimum admitted bundle:

| Check | Purpose | Required behavior |
| --- | --- | --- |
| coordinate-equivalence residual | confirm allowed coordinate transforms preserve recomputed analytical fields | reject or quarantine if residual exceeds calibration tolerance |
| near-boundary counterfactual | expose accept decisions that are unstable under registered probe perturbations | quarantine if the perturbed decision crosses the accept boundary |
| decoy-consistency counterfactual | expose traces whose local evidence can be explained by a decoy basin interaction | quarantine if decoy-consistent evidence is above calibration tolerance |

Vacuity gate:

- dropping `invariance_checks_v2` must change at least 2% of verification plus
  falsifier measurement decisions; or
- v2 must remove the field and report invariance repair failed.

If `invariance_checks_v2` has decision match at least 99% when dropped,
Certificate Vacuity remains triggered.

## Geometry Boundary Gate

`geometry_promise_signal_v2` must close the v1 basin-shape boundary leak.

Minimum fields:

| Field | Purpose |
| --- | --- |
| `geometry_evidence_coverage` | amount and distribution of probe evidence available for shape checks |
| `scale_interval` | probe-derived lower and upper basin-scale estimate |
| `curvature_profile` | robust curvature or Laplacian summary from the probe stencil |
| `topology_ambiguity_score` | proxy for crescent / decoy-doublet ambiguity |
| `boundary_flags` | named booleans for insufficient coverage, scale out of envelope, curvature out of envelope, and topology ambiguous |

Minimum behavior:

- no out-of-promise basin-shape falsifier item may be accepted;
- insufficient evidence for a basin-shape promise decision must quarantine;
- accepted out-of-promise rows must be reported by promise-violation subtype;
- noise-tier out-of-promise rows must remain separated from basin-shape
  out-of-promise rows.

A bounded positive v2 receipt is not allowed if any out-of-promise
basin-shape falsifier item is accepted.

## Spoof and Inversion Gates

Inherit the v1 attacker budgets and success thresholds:

- `A_spoof_field_small`: 64 candidates per unsafe measurement item, analytical
  fields only, any unsafe accept triggers `capacity_threshold <= small`.
- `A_spoof_source_small`: 64 candidates per unsafe measurement item, source
  observations only, any unsafe accept blocks a bounded positive receipt.
- `A_inv_small`: at most `20k` trainable parameters and three seeds; breach
  requires AUROC at least `0.80` and occupancy IoU at least `0.40`.

Do not widen attacker budgets, policy class, or inversion target for v2.

## Cost Gate

Safety remains primary, but v2 must attempt a real checker-overhead repair.

Allowed implementation moves:

- batch signature recomputation;
- single-pass signature plus verifier traversal over source observations;
- memoized trace-commitment lookup;
- fixed-grid or integer arithmetic for geometry summaries;
- shared intermediate buffers between verifier, ablation, and attacker passes.

Required cost reports:

- wall-time ratio vs rollout;
- op-count ratio vs rollout;
- wall-time ratio vs full-state;
- op-count ratio vs full-state;
- wall-time ratio vs the filed v1 receipt's signature verifier ratio.

Cost repair passes only if wall-time ratio vs rollout improves by at least 25%
relative to the filed v1 receipt, or if a stricter pre-measurement threshold is
registered before execution. If safety gates pass but cost does not, the receipt
may say the safety repair succeeded; it may not claim cheap verification.

## Required Outputs

Write under:

`results/pvnp/phase1-toy-verifier-v2/`

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
- `attacker_trials.csv`;
- `ground_truth_labels.csv`;
- `costs.csv`;
- `falsifier_summary.md`;
- `README.md`.

Durable reviewed receipts belong under `docs/pvnp/receipts/` using
[`RECEIPT_TEMPLATE.md`](RECEIPT_TEMPLATE.md).

## v2 Verdict Rules

The v2 receipt must use one of the normal receipt verdicts:

- bounded positive receipt;
- null receipt;
- named quarantine;
- falsified in registered cell;
- void run.

Additional v2 repair labels:

| Repair label | Condition |
| --- | --- |
| integrity repair maintained | field-only and source-bound spoof attempts accept zero unsafe items and all integrity-mismatch cases quarantine |
| integrity regression | any inherited integrity repair fails |
| invariance repair passed | `invariance_checks_v2` clears the 2% vacuity gate |
| invariance repair failed | invariance remains vacuous or is removed without replacement |
| boundary repair passed | zero accepted basin-shape promise violations |
| boundary repair failed | any out-of-promise basin-shape falsifier item is accepted |
| coverage disposition passed | standalone `coverage_digest` is removed, or a retained replacement is load-bearing |
| coverage disposition failed | standalone coverage remains redundant |
| cost repair passed | registered wall-time improvement threshold is met |
| cost repair failed | threshold missed or cost accounting missing |

A bounded positive v2 receipt requires maintained integrity repair, invariance
repair, boundary repair, coverage disposition, green privilege audit, and no
accepted unsafe item. Cost repair is not required for a safety positive, but the
public summary must state the cost disposition.

## Freeze Rule

Edits allowed before implementation:

- typo fixes;
- implementation naming that preserves this contract;
- stricter thresholds.

Edits requiring a new slate id:

- split counts or seed namespaces;
- attacker budgets or success thresholds;
- policy class or environment family;
- certificate integrity requirements;
- invariance vacuity threshold;
- geometry boundary gate;
- standalone coverage disposition;
- verifier-access rules;
- required output schema.
