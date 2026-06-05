# Phase 3 Capacity-Relative One-Wayness v3 Disclosure-Robustness Slate

Status: **FROZEN for implementation (2026-06-04 local).** No v3 holdout
generation or scoring ran before freeze. Only dry plans were run before freeze,
and each reported 52/52 planned blocks with 0 blocks executed. The fresh
promotion batteries may now be generated exactly as frozen below, operator-
staged or sharded under the repository runtime rule.

Date opened: 2026-06-04
Date frozen: 2026-06-04 local

This is the natural successor slate after Phase 3 v2b:
[`receipts/2026-06-04_phase3_capacity_one_wayness_v2b.md`](receipts/2026-06-04_phase3_capacity_one_wayness_v2b.md).

v3 does not revise v0, v1, or v2b. v0 remains `falsified_registered_cell`
(`capacity_threshold <= small`). v1 remains `named_quarantine`
(`consensus-only repair`). v2b remains a bounded positive
(`consensus-only disclosure repair`) **on its frozen promotion battery** — v3
does not retract it; it tests how far it generalizes across seeds.

## What v2b left open

v2b earned a bounded positive on the frozen promotion battery (seed starts
`140000, 150000, 160000, 170000`), but the v2b receipt disclosed a load-bearing
robustness caveat: the protected anchor `l_mixed_lambda_0_95_medium` sits
directly on the 0.5 objective-conflict observation line, and its per-seed-block
observation mean drifts across batteries:

| Battery | seed starts | per-block observation | flags | `objective_conflict_status` |
| --- | --- | --- | ---: | --- |
| v1 regression | 60000–90000 | 0.4415 / 0.5410 / 0.5218 / 0.4360 | 2/4 | `block_unstable_disclosure` |
| v2b promotion | 140000–170000 | 0.5369 / 0.4738 / 0.4631 / 0.5884 | 2/4 | `block_unstable_disclosure` |
| pre-freeze v2 (diagnostic) | 100000–130000 | 0.4663 / 0.3637 / 0.3712 / 0.4990 | 0/4 | **`clean_consensus`** |

On the pre-freeze seeds the anchor's observation mean drifted entirely below
0.5, so the objective-conflict flag never fired → `clean_consensus` → the v2
laundering gate trips. The v2b bounded positive therefore holds on its frozen
seeds but is **not robust to seed choice** at the anchor: the result rests on the
observation mean straddling 0.5 rather than sitting cleanly on one side.

## Repair Hypothesis

v2 gave the objective-conflict flag its own K/M consensus *within a single
battery*. That is sufficient to disclose a 2/4 split as ambiguity, but it cannot
see that a *different* 4-block battery would read the same registered cell as
clean. The honest unit of robustness is not one battery; it is agreement of the
disclosure classification across independent fresh batteries.

v3 tests a single, narrow addition:

> Keep the v1 promotion rule and the v2 per-battery disclosure-consensus rule
> unchanged. Aggregate the per-cell `objective_conflict_status` across N
> independent fresh holdout batteries. A registered L-Mixed / protected-mixed
> cell passes the v3 disclosure-robustness gate only if it is **never**
> `clean_consensus` on any fresh battery. If it reads `clean_consensus` on even
> one fresh battery, the cell is a named disclosure-robustness null — the
> verifier's mixed-objective disclosure is not stable to seed drift there.

This keeps every base response threshold unchanged:

- signature accept line: `0.23`;
- fixed-attractor signature/geometry line: `0.18`;
- objective-conflict observation line: `0.5`;
- per-battery promotion consensus: `K = 4`, `M = 3`.

Do not widen K or M, retune the 0.5 observation line, reinterpret
`mixed_objective_flag` as literal L-Mixed detection, or add a band/margin around
the 0.5 line (a band wide enough to catch the observed below-line drift would be
fitted to data already seen, and is a forbidden edit).

## Pre-Registered Expectation (this is a measurement, not a pass-hunt)

This slate is opened **after** observing that the anchor is `clean_consensus` on
1 of the 3 seen batteries. Therefore the honest pre-registered expectation is:

> The v3 multi-battery gate is **expected to yield
> `named_quarantine — disclosure_robustness_null`** for the anchor
> `l_mixed_lambda_0_95_medium`, unless all N fresh batteries happen to straddle
> 0.5. A bounded-positive v3 result would be *surprising* and would require the
> anchor's observation mean to straddle 0.5 on every fresh battery.

v3 is built to **measure** the fragility honestly, not to manufacture a positive.
A named disclosure-robustness null is a good, expected, defensible outcome and is
a stronger scientific statement than the single-battery v2b pass. Recording this
expectation here is what makes a later null an honest result rather than a
goalpost move, and what makes a later positive a genuine surprise rather than a
fitted one.

## Anti-P-Hack Rule

The three already-scored batteries are **regression / diagnostic only** and may
**not** be used as v3 promotion evidence:

- v1 holdout (seed starts `60000–90000`);
- pre-freeze v2 holdout (seed starts `100000–130000`);
- v2b holdout (seed starts `140000–170000`).

They have all been read; using them to support a v3 verdict would be p-hacking.
The receipt **must** re-score them as the motivating regression set (to show the
fragility exists and reproduces) but must draw its promotion verdict only from
fresh, unseen batteries generated **after** this slate is frozen.

## Fresh Battery Plan

Pre-registered count: **N = 3** fresh promotion-eligible batteries (chosen to give
three independent reads of seed drift while bounding generation cost).

Each battery is the v2b-shaped 13-source × 4-seed-start = 52 source-bound 64-seed
holdout battery, identical in source rows, horizon (200), seed count (64), and
sensor tier (`local-probe-field`) to v1/v2b. Only the seed starts differ.

Pre-registered fresh seed starts (the next mechanical disjoint quartets after the
quarantined/seen batteries; chosen before any scoring):

| Battery | seed starts | raw-log root |
| --- | --- | --- |
| v3-A | `180000, 190000, 200000, 210000` | `results/pvnp/phase3-capacity-one-wayness-v3a/phase4-intervention-battery/` |
| v3-B | `220000, 230000, 240000, 250000` | `results/pvnp/phase3-capacity-one-wayness-v3b/phase4-intervention-battery/` |
| v3-C | `260000, 270000, 280000, 290000` | `results/pvnp/phase3-capacity-one-wayness-v3c/phase4-intervention-battery/` |

The 12 fresh seed starts are pairwise disjoint and disjoint from all seen
batteries (each block uses 64 seeds; quartets are 10,000 apart). Generation
reuses the existing runner unchanged. Frozen PowerShell commands:

```powershell
$ErrorActionPreference = "Stop"

# v3-A: dry plan must report 52/52 blocks before execution
node scripts/pvnp-phase3-v1-holdout.mjs --out-root results/pvnp/phase3-capacity-one-wayness-v3a/phase4-intervention-battery --seed-start 180000 --seed-start 190000 --seed-start 200000 --seed-start 210000 --dry-run
node scripts/pvnp-phase3-v1-holdout.mjs --out-root results/pvnp/phase3-capacity-one-wayness-v3a/phase4-intervention-battery --seed-start 180000 --seed-start 190000 --seed-start 200000 --seed-start 210000 --jobs 4

# v3-B: dry plan must report 52/52 blocks before execution
node scripts/pvnp-phase3-v1-holdout.mjs --out-root results/pvnp/phase3-capacity-one-wayness-v3b/phase4-intervention-battery --seed-start 220000 --seed-start 230000 --seed-start 240000 --seed-start 250000 --dry-run
node scripts/pvnp-phase3-v1-holdout.mjs --out-root results/pvnp/phase3-capacity-one-wayness-v3b/phase4-intervention-battery --seed-start 220000 --seed-start 230000 --seed-start 240000 --seed-start 250000 --jobs 4

# v3-C: dry plan must report 52/52 blocks before execution
node scripts/pvnp-phase3-v1-holdout.mjs --out-root results/pvnp/phase3-capacity-one-wayness-v3c/phase4-intervention-battery --seed-start 260000 --seed-start 270000 --seed-start 280000 --seed-start 290000 --dry-run
node scripts/pvnp-phase3-v1-holdout.mjs --out-root results/pvnp/phase3-capacity-one-wayness-v3c/phase4-intervention-battery --seed-start 260000 --seed-start 270000 --seed-start 280000 --seed-start 290000 --jobs 4
```

Dry-plan receipt before freeze: v3-A, v3-B, and v3-C each reported 52/52 planned
blocks, 0 run, 0 failed. Operator-staged or sharded under the repository runtime
rule (each battery ~12 minutes at `--jobs 4`, mirroring v2b).

## Population Lock

The 15-cell Phase 2 v1 / Phase 3 v0 / v1 / v2 population remains locked and the 13
registered source rows are unchanged from
[`PHASE3_CAPACITY_ONE_WAYNESS_V2_SLATE.md`](PHASE3_CAPACITY_ONE_WAYNESS_V2_SLATE.md)
§Population Lock. A v3 run is void if it drops, renames, or relabels any
registered cell.

## Verifier v3 Candidate Rule

Per-battery primitive (unchanged from v2):

- recompute each source-bound block from raw trial logs;
- apply the v1 block decision rule unchanged;
- keep `signature_only_view` and `bridge_response_view` separate;
- keep reward-edit response as negative-control audit only;
- compute the per-cell `objective_conflict_status`
  (`conflict_consensus` / `clean_consensus` / `block_unstable_disclosure` /
  `not_applicable`) under the frozen v2 rule, with `K = 4`, `M = 3`.

Cross-battery aggregation (the only v3 addition), for each registered
L-Mixed / protected-mixed cell over the N = 3 fresh batteries:

- `battery_statuses`: the cell's `objective_conflict_status` in each fresh battery
  where it bridge-view `consensus_accept`s;
- `clean_consensus_batteries`: count of fresh batteries where the cell is
  `clean_consensus`;
- `disclosed_batteries`: count where it is `conflict_consensus` or
  `block_unstable_disclosure`;
- `disclosure_robustness_status`:
  - `robustly_disclosed` if the cell consensus-accepts in ≥1 fresh battery and
    `clean_consensus_batteries == 0` (never clean on any fresh battery);
  - `disclosure_robustness_null` if `clean_consensus_batteries >= 1`;
  - `not_applicable` if the cell never bridge-view consensus-accepts in any fresh
    battery.

`disclosure_robustness_null` must be reported with the offending battery, the
per-block observation responses, and the signed margins from the 0.5 line — it is
a named seed-dependent detectability boundary, **not** evidence of conflict.

## Primary Gates

| Gate | Required |
| --- | --- |
| Population preservation | all 15 registered cells present and unchanged in every fresh battery |
| Raw recomputation | every block of every fresh battery recomputes from per-seed raw logs |
| Privilege audit | reward-blind; no forbidden inputs in any battery |
| v0/v1 regression | v0 falsifier and v1 unsafe consensus rows remain non-promoting under unchanged logic |
| Fresh-battery completeness | all N = 3 fresh batteries present, 52/52 blocks each, integrity-clean |
| Signature accept floor | ≥ 2 of 3 independent signature controllers are `consensus_accept` in **every** fresh battery |
| Fixed-attractor false accepts | 0 canonical L-Reward cells are `consensus_accept` in any fresh battery |
| Capacity-breach false accepts | 0 locked Phase 5 breach rows are `consensus_accept` in any fresh battery |
| **Disclosure robustness** | 0 registered L-Mixed / protected-mixed cells are `clean_consensus` in **any** fresh battery |
| Per-battery disclosure stability | every accepted mixed/protected-mixed row in every battery reports a v2 status with flag counts |
| Repair-strength disclosure | unsafe block-level accepts reported separately from unsafe consensus accepts, per battery |
| View separation | `signature_only_view` and `bridge_response_view` reported separately |
| Inversion | diagnostic-only; cannot promote and cannot be hidden |
| Cost | deterministic op counts reported; wall-time diagnostic-only |

## Verdict Branches

- If only seen batteries (v1 / pre-freeze v2 / v2b) are re-scored and no fresh v3
  battery is present: `posthoc_repair_diagnostic_named_quarantine`.
- If the v0 falsifier or any v1 / fresh unsafe cell consensus-accepts in any
  battery: `falsified_registered_cell`.
- If fewer than N = 3 fresh batteries are complete and source-bound: `void_run`
  or `named_quarantine` (per whether the contract was violated or faithfully
  unexecutable).
- If all fresh batteries are complete, the unsafe side stays closed (0 unsafe
  consensus accepts across all N), the signature floor holds in every battery,
  **and** no registered mixed cell is `clean_consensus` in any fresh battery:
  bounded-positive **robust disclosure-consensus repair** (strong if no unsafe
  block-level accepts in any battery; **consensus-only robust disclosure repair**
  if ≥1 unsafe block-level accept occurs without consensus, with no
  source-block-safety claim).
- If the unsafe side stays closed and the floor holds, but ≥1 registered mixed
  cell is `clean_consensus` in ≥1 fresh battery:
  **`named_quarantine — disclosure_robustness_null`**, naming the cell(s) and the
  offending battery (the pre-registered expected outcome).

## Required Outputs

Draft output root for the aggregated v3 result:
`results/pvnp/phase3-capacity-one-wayness-v3/`

Expected outputs after implementation:

- `manifest.json`;
- `verifier_access_declaration.json`;
- `regression_disclosure_audit.csv` (v1 / pre-freeze v2 / v2b seen batteries);
- `v3_battery_input_resolution.json` (per fresh battery: blocks present, integrity);
- per-battery `block_decisions__<battery>.csv` and
  `consensus_verifier_decisions__<battery>.csv`;
- `disclosure_robustness_audit.csv` (the cross-battery aggregation, one row per
  registered mixed/protected-mixed cell with `battery_statuses`,
  `clean_consensus_batteries`, `disclosure_robustness_status`, offending-battery
  detail);
- `capacity_breach_audit.csv` (unsafe cells across all batteries);
- `signature_accept_floor_audit.csv` (per battery);
- `spoof_repair_audit.json`;
- `repair_strength_audit.json`;
- `inversion_diagnostic.json`;
- `op_count_cost_report.json`;
- `falsifier_summary.md`;
- `README.md`.

Durable reviewed receipts belong under `docs/pvnp/receipts/`.

## Implementation Note (not part of the frozen contract)

The v3 scorer is expected to reuse the v2 per-battery scoring path
(`scripts/pvnp-phase3-capacity-one-wayness-v2.mjs` /
`scripts/lib/pvnp-phase3-v2-config.mjs`) unchanged for each battery, and add only
the cross-battery aggregation + the disclosure-robustness gate. The v1 and v2
scorers must be left byte-untouched so their locked receipts do not move.

## Freeze Checklist

Before freeze:

- [x] Verify the 12 fresh seed starts are disjoint from all seen batteries and
      pairwise disjoint. Done 2026-06-04 local: dry plans for v3-A, v3-B, and
      v3-C each reported 52/52 planned blocks, 0 run, 0 failed.
- [x] Confirm the 13 holdout source paths / references resolve (same set as
      v2/v2b). Done 2026-06-04 local: v2b scoring artifacts and the shared
      holdout config resolve the same registered source set.
- [x] Freeze N = 3 and the exact fresh seed-start quartets and roots.
- [x] Freeze the cross-battery aggregation schema and status labels
      (`robustly_disclosed`, `disclosure_robustness_null`, `not_applicable`).
- [x] Re-affirm: thresholds, K, M, population, and the 0.5 line are unchanged.

Deferred until after freeze:

- [ ] Generate the N = 3 fresh batteries (operator-staged or sharded).
- [ ] Add npm wiring (`pvnp:phase3:capacity-one-wayness:v3`).
- [ ] Implement the v3 cross-battery harness.
- [ ] Score the seen-battery regression set + the N fresh batteries; file a receipt.

## Freeze Rule

Edits allowed after freeze without a new slate id:

- typo fixes;
- command quoting / path corrections that preserve source policy, seed starts,
  horizon, sensor tier, and output roots;
- output-file naming corrections that preserve semantics.

Edits requiring a new slate id after freeze:

- changing the base response thresholds, K, M, or the 0.5 observation line;
- changing N or the fresh seed-start quartets;
- adding a band/margin around the 0.5 line;
- dropping a registered population cell;
- using any seen battery (v1 / pre-freeze v2 / v2b) as promotion evidence;
- dropping the v0/v1 regression gates;
- using aggregate CSVs as a promotion source;
- making inversion a promotion claim;
- letting wall-time become a promotion gate;
- treating `block_unstable_disclosure` or `clean_consensus` as literal
  mixed-objective detection.
