# Phase 1 v5 Toy Verifier - PROVISIONAL Receipt

## PROVISIONAL / Correction Banner

This receipt withdraws the earlier v5 cost conclusion. The first v5 write-up
claimed a stable cost-only quarantine from Run A's median-of-3 result
(`C_total_signature = 889.7 ms`, full-state ratio `108.21 x`, spread `2.41 %`).
A later clean run disagreed by roughly 3.5x (`C_total_signature = 3185 ms`,
full-state ratio about `191 x`, spread about `29 %`) and the
`environments.jsonl` hash drifted despite deterministic seeds
(`4934d752...` vs `5549b4c4...`). That combination invalidates any v5 cost
verdict until the environment-generation determinism and timing stability are
reproduced on a quiescent machine.

Durable status: **PROVISIONAL named quarantine - safety-complete,
cost-unadjudicated**. Do not cite the 889.7 ms / 108.21x Run A value as stable,
and do not use this v5 receipt to unlock v6 or Phase 2 until the required
rerun below either finalizes or voids it.

## Header

- Receipt id: `pvnp-phase1-toy-verifier-v5-2026-05-28-provisional`
- Phase / probe: Phase 1 v5 cost-closure slate - provisional execution record
- Date: 2026-05-28
- Author / runner: harness `scripts/pvnp-phase1-harness.mjs` invoked via
  `npm run pvnp:phase1:v5`
- Result directory: `results/pvnp/phase1-toy-verifier-v5/` (transient;
  not a durable source of truth)
- Roadmap version: [`SUNDOG_V_P_V_NP.md`](../../SUNDOG_V_P_V_NP.md)
- Spec version: [`PHASE1_TOY_VERIFIER_SPEC.md`](../PHASE1_TOY_VERIFIER_SPEC.md)
  constrained by [`PHASE1_V5_SLATE.md`](../PHASE1_V5_SLATE.md)
- Predecessors: [`..._v0`](2026-05-28_phase1_toy_verifier_v0.md) through
  [`..._v4`](2026-05-28_phase1_toy_verifier_v4.md)

## Implementation Delta

v5 implemented the intended cost-closure code changes:

- Removed the hot-path `noteShortCircuit` closure allocation from
  `certificateIntegritySourceBound`; pre-integrity short-circuits now call the
  cache counter directly behind a hoisted guard.
- Produced the slate-required short-circuit instrumentation audit and
  `cost_multirun_report.json`.
- Wired schema `pvnp-phase1-sigma-v5` through run config, package scripts,
  harness stages, and version dispatch. The field set remains identical to
  v3/v4.

The code change is statically verified, but the cost measurement is not yet
trusted.

## Safety Summary

The v5 safety side is retained as a strong provisional result:

| Check | Provisional result |
| --- | --- |
| False accepts | `0` |
| Field spoof channel | `0 / 509` |
| Source spoof channel | `0 / 509` |
| Integrity probes | `5 / 5` quarantined |
| OOP basin-shape accepts | `0 / 768` |
| Privilege-leak audit | green |

These outcomes are timing-independent and consistent with the v2-v4 safety
repair arc. They should still be reconfirmed in the required rerun, but the
known defect is cost/determinism measurement, not a discovered safety breach.

## Cost Non-Adjudication

Cost is explicitly **not adjudicated** for v5.

| Run | Reported cost median | Reported full-state ratio | Spread | Env hash prefix |
| --- | ---: | ---: | ---: | --- |
| A | `889.7 ms` | `108.21 x` | `2.41 %` | `4934d752` |
| B | `3185 ms` | about `191 x` | about `29 %` | `5549b4c4` |

The hash drift is the more serious signal: with deterministic seeds, a changed
`environments.jsonl` hash should not happen. Until that is explained, neither
Run A nor Run B is a valid basis for a bounded-positive, cost-only quarantine,
or target-reset claim.

## Environment Incidents

1. **Hardlinked result directories - resolved.** During v5 work,
   `results/pvnp/phase1-toy-verifier-v4` and
   `results/pvnp/phase1-toy-verifier-v5` shared inodes, so a v5 write mutated
   v4 transient artifacts. This was caught by inode and manifest-hash checks
   and resolved by deleting the transient v5 result directory and rerunning.
   Durable receipts remain the record; `results/` is transient.
2. **Cost non-reproducibility and environment-hash drift - open.** The two v5
   clean runs reported incompatible cost medians and different environment
   hashes. This blocks all v5 cost conclusions.

## Verdict

**PROVISIONAL named quarantine - safety-complete, cost-unadjudicated.**

v5 earned a code-level cost-instrumentation repair and another green safety
record. It did **not** earn a cost verdict, a stable 108.21x claim, a
full-state-target reset, v6 promotion, or Phase 2 promotion.

## Required Next Step

Before v6 or Phase 2, rerun v5 twice on a quiescent machine:

```powershell
Remove-Item -LiteralPath results/pvnp/phase1-toy-verifier-v5 -Recurse -Force
npm run pvnp:phase1:v5
```

Then repeat the same clean run and compare:

- `environments.jsonl` hash must be identical across runs. If it drifts, fix
  environment generation determinism first.
- `cost_multirun_report.json` must show small spread and stable medians. If it
  does not, keep v5 cost-unadjudicated.
- Once both checks pass, replace this provisional receipt with a finalized v5
  receipt or void v5 explicitly.

## Not Allowed From This Receipt

- Do not cite Run A as "stable 889.7 ms" or "stable 108.21x".
- Do not claim "7 of 8 cost clauses pass" for v5.
- Do not open v6 or Phase 2 on the basis of the current v5 cost numbers.
