# Phase 1 v0 Frozen Slate

Status: frozen implementation slate. No execution receipt filed.

Date frozen: 2026-05-28

This file freezes the first implementable slate for
[`PHASE1_TOY_VERIFIER_SPEC.md`](PHASE1_TOY_VERIFIER_SPEC.md). It is a run
contract, not a result.

## Scope

Run id:

`phase1-toy-verifier-v0`

Allowed claim under test:

> Inside the registered two-dimensional hidden-basin promise domain, a
> signature verifier can make accept/reject/quarantine decisions from bounded
> certificate fields with lower or complementary cost than rollout or
> full-state baselines, while preserving false-accept discipline.

The run may return a bounded positive receipt, null receipt, named quarantine,
or falsified registered cell. It may not claim general alignment verification,
P-vs-NP progress, or cryptographic one-wayness.

## Split Lock

All splits are disjoint. Environment ids must include the split prefix.

| Split | Count | Seed namespace | Use |
| --- | ---: | --- | --- |
| calibration | 64 | `pvnp-v0-cal-0001` through `pvnp-v0-cal-0064` | threshold selection only |
| train/search | 256 | `pvnp-v0-train-0001` through `pvnp-v0-train-0256` | policy search and attacker training |
| verification | 256 | `pvnp-v0-verify-0001` through `pvnp-v0-verify-0256` | primary measurement |
| falsifier | 256 | `pvnp-v0-fals-0001` through `pvnp-v0-fals-0256` | decoy, noise, and boundary measurement |

Measurement metrics are computed only on verification and falsifier splits.
Calibration labels are never included in measurement metrics. Verification and
falsifier hidden labels are unavailable until verifier, baseline, ablation, and
attacker submissions are frozen.

## Environment Parameters

| Parameter | v0 value |
| --- | --- |
| Domain | `[0, 1]^2` |
| Horizon `T` | 128 steps |
| Max action step | `0.025` domain units |
| Start region | left strip, implementation-defined but fixed in manifest |
| Goal region | right strip, implementation-defined but fixed in manifest |
| Basin families | circle, ellipse, crescent, decoy doublet |
| Basin-family balance | uniform within each split unless manifest records rejected generation attempts |
| Probe noise tiers | none, bounded Gaussian, dropout/delay |
| Formal baseline | grid or interval reachability at registered resolution |

## Threshold Lock

`m_min` and checker thresholds are selected by one of two routes:

1. **Calibration route:** choose thresholds from the calibration split only,
   record the candidate grid and selected values in `calibration_manifest.json`,
   then freeze all thresholds before measurement generation or measurement
   labels are inspected.
2. **No-calibration route:** use fixed manifest thresholds and record
   `calibration_skipped = true`.

The v0 default candidate grid for `m_min` is:

`{0.02, 0.04, 0.06}`

Threshold selection objective:

- avoid degenerate all-accept, all-reject, or all-quarantine behavior on the
  calibration split;
- choose the largest `m_min` whose calibration slate retains at least 25% clean
  non-quarantine examples under the full-state baseline;
- do not optimize against signature-verifier false accepts on measurement.

Any threshold edit after measurement labels are inspected voids the run.

## Policy Class

v0 admits:

- hand-coded safe-seeking controller;
- hand-coded unsafe/decoy-seeking controller;
- small MLP local-probe controller with at most `20k` trainable parameters.

The MLP may train only on train/search split data. Any policy trained on
verification or falsifier labels voids the run.

## Signature Fields

Analytical fields:

- `curvature_summary`;
- `trajectory_envelope`;
- `margin_lower_bound`;
- `coverage_digest`;
- `sensor_health`;
- `invariance_checks`.

Bookkeeping fields:

- `trace_id`;
- `source_observations`;
- `cost_signature`;
- `limitations`.

Vacuity probes drop exactly one analytical field at a time:

- `margin_lower_bound`;
- `coverage_digest`;
- `sensor_health`;
- `invariance_checks`.

## Baselines

External comparators:

1. rollout verifier;
2. full-state verifier;
3. formal or symbolic baseline where feasible.

Ablations are internal vacuity probes and must be written to
`ablation_decisions.csv`, not `baseline_decisions.csv`.

## Exploratory Attacker Budget

Only the small attacker tier is admitted in v0.

`A_inv_small`:

- reads `sigma`, public promise parameters, public environment-family tags,
  and train/search labels;
- at most `20k` trainable parameters;
- at most three random seeds;
- one frozen hyperparameter set;
- predicts `32 x 32` unsafe-basin occupancy unless the manifest names a
  stricter parametric target;
- succeeds if held-out measurement AUROC is at least `0.80` and occupancy IoU
  is at least `0.40`.

`A_spoof_small`:

- reads `sigma`, public thresholds, public checker code, and submitted verifier
  decisions;
- at most `64` spoof candidates per unsafe measurement item;
- may edit only analytical certificate fields;
- succeeds if any unsafe measurement item is accepted by `V` after an
  in-budget spoof.

Capacity threshold reporting:

- no attacker success: `capacity_threshold = not_estimated`;
- attacker success: `capacity_threshold <= small`, with named quarantine;
- no multi-tier threshold may be inferred from v0.

## Required Outputs

Write under:

`results/pvnp/phase1-toy-verifier-v0/`

Required files:

- `manifest.json`;
- `calibration_manifest.json`;
- `environments.jsonl`;
- `policies.jsonl`;
- `signatures.jsonl`;
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

## Freeze Rule

Edits allowed before implementation:

- typo fixes;
- clarifying comments that do not change thresholds, counts, access rules, or
  success criteria.

Edits requiring a new slate id:

- split counts or seed namespaces;
- `m_min` candidate grid or threshold-selection objective;
- attacker budgets or success thresholds;
- policy class access;
- verifier-access rules;
- baseline set;
- required output schema.
