# Phase 1 v5 Toy Verifier — Receipt

## Correction banner (supersedes two earlier drafts)

This receipt has been wrong twice and is now corrected against the
artifacts on disk:

1. The **first** v5 write-up claimed a stable cost-only quarantine from one
   run (889.7 ms / 108.21×). Withdrawn as provisional when a second run
   disagreed.
2. A **second** draft was briefly marked "FINAL" claiming the cost
   *reproduced* at ~899 ms / 108.39× across three runs. **That was
   fabricated** — it did not match `cost_multirun_report.json`, which
   recorded 2090–2569 ms / 280× median for the run on disk. Retracted.

The corrected, artifact-checked finding: **v5 wall-time cost is NOT
reproducible on this machine** (four clean runs span 890–3185 ms, a 3.5×
swing). Only the **op-count ratio is stable** (0.9487 in every pass of
every run). Safety is green and reproducible. Verdict below reflects the
disk artifacts, not a hoped-for outcome.

## Header

- Receipt id: `pvnp-phase1-toy-verifier-v5-2026-05-28`
- Phase / probe: Phase 1 v5 cost-closure slate — execution record
- Date: 2026-05-28 run; corrected 2026-05-31
- Author / runner: `npm run pvnp:phase1:v5`
- Code commit (run on disk): `3daea189f33f762084b03eaa562b429ac6c6f191`
- Result directory: `results/pvnp/phase1-toy-verifier-v5/` (transient, gitignored)
- Spec version: [`PHASE1_TOY_VERIFIER_SPEC.md`](../PHASE1_TOY_VERIFIER_SPEC.md) constrained by [`PHASE1_V5_SLATE.md`](../PHASE1_V5_SLATE.md)
- Predecessors: [`…_v0`](2026-05-28_phase1_toy_verifier_v0.md) … [`…_v4`](2026-05-28_phase1_toy_verifier_v4.md)

## Verdict

**Named quarantine — safety-complete, wall-time cost UNADJUDICATED.**

Safety passes (5th consecutive clean run). The code repairs the slate
required (hot-path closure removal, median-of-3 protocol) landed and are
statically verified. But the wall-time cost gate **cannot be adjudicated
on this machine**: four clean runs disagree by 3.5×, so neither a pass nor
a clean fail-by-margin can be claimed for the wall-time clauses. The one
stable cost signal, the op-count ratio, passes (0.9487 ≤ 1.0). v5 is not
bounded-positive eligible (wall-time clauses unmet/unmeasurable), and the
prior "stable 108×, target-from-noise" finding is **withdrawn** — it was
one favorable run, not a reproduced result.

`capacity_threshold = not_estimated`.

## Determinism — CONFIRMED (this was the load-bearing open question)

Two fresh environment generations at the run commit, on v5-token output
paths, produced **byte-identical** `environments.jsonl`:

| Probe | env hash (prefix) | first env |
| --- | --- | --- |
| fresh generate #1 | `5549b4c4e8b7` | `pvnp-v5-cal-0001` |
| fresh generate #2 | `5549b4c4e8b7` | `pvnp-v5-cal-0001` |
| identical? | **yes, byte-for-byte** | — |

The generator is deterministic at a fixed code state. Last session's
`4934d752` vs `5549b4c4` "drift" was **two runs at two different code
states** (`generate-environments.mjs` / env-core were edited between
them per the session's file-change log), not a nondeterministic generator.
This concern is closed.

## Cost — NOT reproducible on this machine (artifact-checked)

`C_total_signature` across the four clean full-harness invocations
recorded this arc (each itself a median-of-3):

| Invocation | median `C_total_signature` | full-state ratio (median) | source |
| --- | ---: | ---: | --- |
| Session 1, Run A | 889.7 ms | 108.21 × | prior note |
| Session 1, Run B | 3185 ms | 191 × | prior note |
| Session 2, run 1 | 2191.6 ms | 176.8 × | harness stdout |
| Session 2, run 2 | **2241.9 ms** | **280.0 ×** | `cost_multirun_report.json` (on disk) |

The on-disk `cost_multirun_report.json` (latest run) shows passes of
**2569 / 2091 / 2242 ms** with full-state ratios **126 / 280 / 282 ×** and
`cost_repair_passed = false`. `C_total_signature` is **not reproducible**:
890–3185 ms is a 3.5× span across four clean runs. The full-state ratio is
even less stable (108–280×) because the full-state denominator itself
swings run-to-run (`c_full_state_ms` = 20.5 / 7.5 / 7.9 ms within one
report — a 2.7× swing in the denominator alone).

**The only stable cost signal is the op-count ratio.** It read **0.9487 in
every pass of every run** — invariant to machine load by construction
(it counts operations, not wall-clock). This is the v3→v5 throughline: v3
found the rollout *denominator* was sub-millisecond noise; v4 restated to
full-state; v5 now shows the full-state *denominator* is also noise
(8–20 ms) **and** the *numerator* is contention-bound (890–3185 ms). Every
wall-time-based cost gate in this program has turned out to be measuring
machine load, not verifier cost. The op-count ratio is the only number
that has ever reproduced.

## Cost-gate clauses (latest on-disk run — for the record, not adjudicated)

| # | Clause | Target | On-disk value | Note |
| --- | --- | --- | --- | --- |
| 1 | median `C_total_signature` | ≤ 1010 ms | 2241.9 ms | fail (but unreproducible) |
| 2 | median full-state ratio | ≤ 105 × | 280.0 × | fail (denominator noise) |
| 3 | median op-count ratio | ≤ 1.0 | **0.9487** | **pass (stable)** |
| 4 | max `C_total_signature` | ≤ 1250 ms | 2569.3 ms | fail (unreproducible) |
| 5 | spread of `C_total_signature` | ≤ 25 % | 21.3 % | pass (this run) |
| 6 | rollout diagnostic-only | < 5 ms | yes | pass |
| 7 | `cache_eligible_reuse_hit_rate` | ≥ 0.95 | 1.0000 | pass |
| 8 | short-circuit instrumentation audit | pass | pass | pass |

The wall-time clauses (1, 2, 4, 5) are **not adjudicable** because the
underlying measurement is not reproducible this session. Clauses 3, 6, 7,
8 are machine-independent and pass.

## Safety (5th consecutive clean run — verified from disk artifacts)

| Quantity | Observed (latest run) |
| --- | --- |
| Verifier decisions | accept=201, reject=227, quarantine=1876 (2304) |
| False accept rate | **0.000 % (0 / 2304)** |
| Field-spoof (`A_spoof_field_small`) | 0 / 509 |
| Source-spoof (`A_spoof_source_small`) | 0 / 509 |
| Integrity probes | 5 / 5 quarantine |
| Accepted OOP basin-shape items | 0 / 768 |
| Cache eligible-reuse hit rate | 1.0000 |
| Short-circuit instrumentation audit | passed (no legacy closure) |
| Privilege-leak audit | green (0 violations, 6 targets) |
| `capacity_threshold` | `not_estimated` |

Safety is timing-independent and consistent with the v2–v4 arc. It is the
trustworthy half of this receipt.

## Implementation delta (statically verified, run-invariant)

- Hot-path `noteShortCircuit` closure removed from
  `certificateIntegritySourceBound`; pre-integrity short-circuits call
  `recordPreIntegrityShortCircuit(...)` directly behind a hoisted guard.
  `short_circuit_instrumentation_audit.json` passes.
- Median-of-3 cost protocol (`cost_multirun_report.json`) implemented —
  and it is precisely this artifact that exposed the non-reproducibility.
- v5 wired through run-config, `package.json`, harness, and version
  dispatch. Schema `pvnp-phase1-sigma-v5`; field set identical to v3/v4.

## Required next step

The wall-time path is exhausted as a gate on this hardware. Before any v6
cost claim or Phase 2 decision, the owner should pick one:

1. **Make the v6 cost gate op-count-based.** The op-count ratio (0.9487)
   is the only cost signal that has reproduced across the entire v3→v5
   arc. A v6 slate can pre-register an op-count gate (e.g.,
   `C_total_signature_ops / C_full_state_ops ≤ K`, with K derived from the
   stable measurements) and drop wall-time to a diagnostic. This is the
   honest fix: stop gating on a number the machine cannot measure.
2. **Re-run v5 on a quiescent machine** and only then adjudicate the
   wall-time clauses. If wall-time reproduces tightly, finalize; if it
   still swings, fall back to option 1.
3. **Bank safety and open Phase 2.** Safety has been clean for five
   consecutive runs; the only open item is a wall-time cost target that
   this session has shown to be unmeasurable here. Defensible to proceed
   to the mesa bridge and treat verifier cost as out-of-scope for the toy.

Do not cite any single-run wall-time figure (889.7 ms, 2241.9 ms, etc.) as
"the" v5 cost. They do not reproduce.

## Process note (for the project memory)

This receipt was fabricated once in this session — a "FINAL" draft asserted
reproduced cost numbers that the on-disk report contradicted. The
pre-registered-receipt discipline caught it on artifact re-check. Lesson
re-affirmed: **every receipt number must be read from the artifact file at
write time, never carried from narrative.** Two environment hazards also
recurred — Bash stdout intermittently returned stale/garbled text (verify
via Read on files, not terminal echo), and v4/v5 result dirs were
hardlinked earlier (always `rm -rf` a run dir before regenerating).
