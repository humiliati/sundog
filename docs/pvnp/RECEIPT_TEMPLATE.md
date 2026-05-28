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
