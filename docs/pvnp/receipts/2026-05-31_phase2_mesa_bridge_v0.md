# Phase 2 Mesa Bridge v0 Receipt

- Receipt id: `pvnp-phase2-mesa-bridge-v0-2026-05-31`
- Phase / probe: Phase 2 mesa bridge v0
- Date run: 2026-05-31
- Author / runner: `npm run pvnp:phase2:mesa-bridge:v0`
- Result directory: `results/pvnp/phase2-mesa-bridge-v0/` (transient,
  gitignored)
- Roadmap version: [`SUNDOG_V_P_V_NP.md`](../../SUNDOG_V_P_V_NP.md)
- Slate: [`PHASE2_MESA_BRIDGE_V0_SLATE.md`](../PHASE2_MESA_BRIDGE_V0_SLATE.md)
- Manifest commit field: `5c2812b9c11a8b1b1d7b36e0f08e2279cc72b89a`

## Verdict

**Named quarantine.** The v0 bridge implementation is wired and reads mesa
Phase 4 raw trial logs, but the frozen raw-recompute gate does not pass because
the Small-tier source manifests record `trial_logs_saved=false`.

This is not a bounded-positive Phase 2 receipt. The useful positive signal is
only partial: Medium raw-log cells recompute, fixed-attractor false accepts are
closed, measured capacity-breach rows are caught, mixed-objective laundering is
closed, and the same-artifact-tier op-count comparator passes. The run cannot
promote because raw provenance is incomplete for the registered population.

## Artifact Check

| Artifact | Path | Load-bearing contents |
| --- | --- | --- |
| Manifest | `results/pvnp/phase2-mesa-bridge-v0/manifest.json` | run id `phase2-mesa-bridge-v0`; schema `pvnp-phase2-mesa-bridge-sigma-v0`; verdict `named_quarantine` |
| Raw recompute audit | `.../raw_recompute_audit.json` | 7/15 cells fail raw recompute; all failures are Small-tier / Small Phase 5 cells with raw logs unavailable |
| Verifier decisions | `.../verifier_decisions.csv` | 2/4 signature-signal cells accepted; canonical Medium L-Reward rejected |
| Capacity audit | `.../capacity_breach_audit.csv` | 0 accepted capacity-breach rows without breach/quarantine flag |
| Mixed audit | `.../mixed_objective_audit.csv` | 0 unqualified L-Mixed accepts |
| Privilege audit | `.../privilege_audit.json` | reward-blind access maintained; forbidden decision inputs not used |
| Op-count report | `.../op_count_cost_gate_report.json` | raw-trace comparator ratio 0.734877; wall-time diagnostic-only |
| Falsifier summary | `.../falsifier_summary.md` | named-quarantine gate summary |

## Gate Results

| Gate | Required | Observed | Result |
| --- | --- | --- | --- |
| Raw recomputation | all registered cells from per-seed raw logs | 7/15 cells fail | fail |
| Integrity | no missing required channels in accepted rows / registered cells | 24 channel-level raw-complete failures | fail |
| Signature accept floor | at least 3/4 HC-Signature/L-Signature cells accepted | 2/4 | fail |
| Fixed-attractor false accepts | 0 canonical L-Reward accepts | 0 | pass |
| Capacity-breach false accepts | 0 locked breach rows accepted without flag | 0 | pass |
| Mixed-objective laundering | 0 unqualified L-Mixed accepts | 0 | pass |
| Cost gate | `C_total_certificate_ops / C_raw_trace_audit_ops <= 1.0` | 0.734877 | pass |
| Wall-time | diagnostic-only | diagnostic-only | pass |

Raw recompute failures are provenance failures, not semantic verifier wins or
losses. The affected cells are:

- `hc_signature_small`;
- `l_signature_small`;
- `l_reward_small`;
- `l_mixed_small`;
- `l_mixed_lambda_0_5_small`;
- `l_mixed_lambda_0_7_small`;
- `l_mixed_lambda_0_9_small`.

Their Phase 4 manifests were present, but the per-seed `trials/*.jsonl` logs
required by the v0 slate were not saved.

## Cost

| Quantity | Observed |
| --- | ---: |
| `C_total_certificate_ops` | 3,660,129 |
| `C_raw_trace_audit_ops` | 4,980,604 |
| ratio | 0.734877 |
| wall-time | 6208.07 ms |

The comparator is the same-artifact-tier raw-trace audit required by the slate,
not full mesa battery regeneration. Wall-time remains diagnostic-only.

## Claim Boundary

This run tests reward-blind detection of **signature-signal-control versus
fixed-attractor control** from raw mesa trial logs. It does not detect literal
reward-training status, does not claim mesa-general verification, and does not
extend the Phase 1 toy-domain bounded-positive result.

## Next Step

Open a Phase 2 v1 slate rather than editing this receipt. The v1 decision is a
design choice: either rerun the missing Small-tier Phase 4 cells with raw trial
logs explicitly saved, or downscope to a Medium-only raw-log bridge and state
that boundary before execution.
