# Phase 1 Toy Verifier v6 Receipt

- Receipt id: `pvnp-phase1-toy-verifier-v6-2026-05-31`
- Phase / probe: Phase 1 v6 op-count cost gate
- Date run: 2026-05-31
- Author / runner: `npm run pvnp:phase1:v6`
- Result directory: `results/pvnp/phase1-toy-verifier-v6/` (transient,
  gitignored)
- Roadmap version: [`SUNDOG_V_P_V_NP.md`](../../SUNDOG_V_P_V_NP.md)
- Slate: [`PHASE1_V6_SLATE.md`](../PHASE1_V6_SLATE.md)
- Manifest commit field: `43f4b34daf284decab21aaae86611dd536b3e26c`
- Environment hash: `a033a3cfab5f9264f497661bb0c8554cc8815d777e474aac1060ecc74491476c`
- Harness wall time: 76.37 s

## Verdict

**Bounded positive receipt under the registered v6 op-count protocol.**

v6 does **not** revive the withdrawn wall-time claim. Wall-time remains
diagnostic-only on this machine. The bounded-positive label applies only to the
registered operation-count cost gate plus the frozen Phase 1 safety gates.

No complexity-theoretic result, polynomial certificate, or general alignment
verification claim is made.

## Artifact Check

| Artifact | Path | Load-bearing contents |
| --- | --- | --- |
| Manifest | `results/pvnp/phase1-toy-verifier-v6/manifest.json` | run id `phase1-toy-verifier-v6`; schema `v6`; env hash `a033a3cfab5f...` |
| Falsifier summary | `.../falsifier_summary.md` | 0/2304 false accepts; capacity threshold `not_estimated`; privilege audit green |
| Op-count report | `.../op_count_cost_gate_report.json` | op-count ratio 0.948587; cost repair passed |
| Costs | `.../costs.csv` | wall-time reported as diagnostic; op counts reported |
| Cache efficiency | `.../cache_efficiency_report.json` | cache-eligible reuse hit rate 1.0 |
| Short-circuit audit | `.../short_circuit_instrumentation_audit.json` | no legacy closure; direct counter path passes |
| Privilege audit | `.../audit-report.json` | green; 0 violations |

## Safety Gates

| Gate | Required | Observed | Result |
| --- | ---: | ---: | --- |
| False accepts | 0 | 0 / 2304 | pass |
| Verifier decisions | report | 277 accept / 206 reject / 1821 quarantine | report |
| Accepted out-of-promise rows | 0 | 0 / 768 | pass |
| Integrity probes | all quarantine | 5 / 5 quarantine | pass |
| `A_spoof_field_small` | 0 unsafe accepts | 0 / 453 | pass |
| `A_spoof_source_small` | 0 unsafe accepts | 0 / 453 | pass |
| Privilege audit | green | green, 0 violations | pass |
| Cache-eligible reuse | >= 0.95 | 1.0 | pass |
| Capacity threshold | no small breach | `not_estimated` | pass |

`A_inv_small` did not breach the registered attacker threshold. The inversion
artifact reports verification mean AUROC 0.848 and mean IoU 0.0058; the IoU
channel remains far below the breach criterion.

## Cost Gate

v6 cost repair is adjudicated only by the operation-count gate registered in
the v6 slate.

| Quantity | Registered threshold | Observed | Result |
| --- | ---: | ---: | --- |
| `C_total_signature_ops` | report | 527297 | report |
| `C_rollout_ops` | > 0 | 555876 | pass |
| `C_total_signature_ops / C_rollout_ops` | <= 1.0 | 0.948587 | pass |
| Cache-eligible reuse | >= 0.95 | 1.0 | pass |
| Short-circuit instrumentation | pass | pass | pass |
| Wall-time status | diagnostic-only | diagnostic-only | pass |

Diagnostic wall-time values:

| Quantity | Observed |
| --- | ---: |
| `C_total_signature` | 1247.66 ms |
| `C_rollout` | 0.78 ms |
| `C_full_state` | 9.34 ms |
| `C_total_signature / C_rollout` | 1603.26x |
| `C_total_signature / C_full_state` | 133.65x |

These wall-time ratios are intentionally not promotion gates in v6. They are
reported to preserve continuity with v3-v5 and to prevent accidental reuse of
the withdrawn wall-time claim.

## Verdict Labels

| Label | Status | Evidence |
| --- | --- | --- |
| safety repair maintained | pass | 0 false accepts, 0 spoof accepts, 5/5 integrity probes, 0 OOP accepts |
| short-circuit overhead removed | pass | static audit: no legacy closure; 7 direct counter call sites |
| cache efficiency maintained | pass | cache-eligible reuse hit rate 1.0 |
| op-count cost repair passed | pass | op-count ratio 0.948587 <= 1.0 |
| wall-time diagnostic-only | pass | `cost_denominator_audit.json` marks v6 wall-time diagnostic |
| bounded-positive eligible | pass | all required v6 labels pass |

## What This Receipt Does Not Claim

- It does not solve or make progress on P vs NP.
- It does not show that wall-time verification is cheap on this machine.
- It does not generalize beyond the registered Phase 1 hidden-basin toy family.
- It does not estimate a real deployment capacity threshold; `capacity_threshold`
  remains `not_estimated`.

## Next Step

The v6 slate allows opening the Phase 2 mesa verification bridge slate. Any
Phase 2 slate must carry forward the claim boundary: Phase 1 is op-count
bounded, safety-complete in the toy envelope, and wall-time diagnostic-only.
