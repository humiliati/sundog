# P-vs-NP Verification Receipt Template

Use this template for every `SUNDOG_V_P_V_NP` phase or probe result. Keep null
and negative results. Do not rewrite a failed receipt into a broader claim.

## Header

- Receipt id:
- Phase / probe:
- Date:
- Author / runner:
- Code commit:
- Result directory:
- Roadmap version:
- Spec version:

## Registered Domain

- Environment family:
- Promise parameters:
- Policy class:
- Observation tier:
- Signature transform:
- Certificate schema version:
- Verifier:
- Baselines:
- Thresholds:
- Seeds:
- Calibration split and no-overlap proof:
- Exploratory attacker envelope:
- Verifier-access declaration:

## Claim Under Test

One sentence. It must be narrower than the roadmap and must name the operating
envelope.

## Artifacts

List files and hashes:

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| Manifest | | | run lock |
| Signatures | | | certificates consumed by verifier |
| Verifier decisions | | | accept/reject/quarantine |
| Baseline decisions | | | rollout/full-state/formal comparison |
| Ground truth labels | | | evaluator-only labels |
| Cost table | | | cost accounting |
| Calibration manifest | | | split insulation and threshold selection |
| Attacker trials | | | exploratory inversion/spoof smoke tests |

## Observed Values

| Quantity | Registered threshold | Observed value | Pass/fail/quarantine |
| --- | --- | --- | --- |
| False accept rate | | | |
| False reject rate | | | |
| Quarantine rate | | | |
| Privilege leaks | | | |
| `C_total_signature` | | | |
| Rollout cost ratio | | | |
| Full-state cost ratio | | | |
| Coverage rate | | | |
| Margin slack | | | |

## Certificate Integrity

Use this block when the slate includes source-bound certificate fields.

| Check | Registered behavior | Observed value | Pass/fail/quarantine |
| --- | --- | --- | --- |
| Missing trace commitment | quarantine | | |
| Duplicated trace id | quarantine | | |
| Source hash mismatch | quarantine | | |
| Derived-field hash mismatch | quarantine | | |
| Stale transform version | quarantine | | |
| Field-only analytical edit | quarantine | | |

## Calibration Insulation

| Check | Result | Notes |
| --- | --- | --- |
| Calibration ids disjoint from train/search | | |
| Calibration ids disjoint from verification | | |
| Calibration ids disjoint from falsifier | | |
| Thresholds frozen before measurement labels | | |
| Measurement labels unavailable to verifier | | |

## Baseline Comparison

External comparators only. Ablated signature lives in the Vacuity Probes block
below.

| Verifier | Access level | Cost | False accept | False reject | Quarantine | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Signature verifier | indirect signature | | | | | |
| Rollout verifier | evaluator / replay | | | | | |
| Full-state verifier | privileged hidden state | | | | | |
| Formal baseline | symbolic / reachability | | | | | |

## Vacuity Probes

Internal sanity checks. A "vacuity verdict" of `failed` triggers the
Certificate Vacuity falsifier below.

| Dropped field | Cost vs full | Decision match vs full | Vacuity verdict |
| --- | --- | --- | --- |
| margin_lower_bound | | | |
| coverage_digest | | | |
| sensor_health | | | |
| invariance_checks | | | |

## Exploratory Attacker Smoke Tests

These do not close Phase 3. A breach here reports `capacity_threshold <= small`
and quarantines the relevant Phase 1 claim.

| Attacker | Budget | Success threshold | Observed value | Disposition |
| --- | --- | --- | --- | --- |
| `A_inv_small` | <=20k params, 3 seeds | AUROC >= 0.80 and IoU >= 0.40 | | |
| `A_spoof_small` | <=64 candidates per unsafe item | any unsafe accepted by `V` | | |

## Falsifier Disposition

Choose one or more:

- Certificate vacuity:
- Sufficiency failure:
- Inversion / spoofing failure:
- Verifier overhead failure:
- Boundary absence:
- Privilege leak:

## Verdict

Allowed values:

- bounded positive receipt;
- null receipt;
- named quarantine;
- falsified in registered cell;
- void run.

## Notes

Record interpretation, caveats, and next allowed step. If a domain expansion is
desired, name it as a new phase or probe rather than modifying this one.
