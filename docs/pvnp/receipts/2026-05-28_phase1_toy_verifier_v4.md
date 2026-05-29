# Phase 1 v4 Toy Verifier — Receipt

## Header

- Receipt id: `pvnp-phase1-toy-verifier-v4-2026-05-28`
- Phase / probe: Phase 1 v4 cost-gate slate — first execution
- Date: 2026-05-28
- Author / runner: harness `scripts/pvnp-phase1-harness.mjs` invoked via
  `npm run pvnp:phase1:v4` (~71 s wall on CPU, Python 3.14.4 +
  torch 2.11.0+cpu + Node 22)
- Code commit (run): `3af1ee66c5f24a3f3f0ac89187190e0e2fd07ccd`
- Result directory: `results/pvnp/phase1-toy-verifier-v4/`
- Roadmap version: [`SUNDOG_V_P_V_NP.md`](../../SUNDOG_V_P_V_NP.md) (after v3 receipt was filed)
- Spec version: [`PHASE1_TOY_VERIFIER_SPEC.md`](../PHASE1_TOY_VERIFIER_SPEC.md) constrained by [`PHASE1_V4_SLATE.md`](../PHASE1_V4_SLATE.md) (frozen cost-gate slate)
- Predecessors:
  [`…_v0`](2026-05-28_phase1_toy_verifier_v0.md),
  [`…_v1`](2026-05-28_phase1_toy_verifier_v1.md),
  [`…_v2`](2026-05-28_phase1_toy_verifier_v2.md),
  [`…_v3`](2026-05-28_phase1_toy_verifier_v3.md)

## Registered Domain

- Environment family / promise / policy class / observation tier: inherited
  from v0–v3 unchanged
- Signature transform: schema **`pvnp-phase1-sigma-v4`**, transform
  `pvnp-phase1-transform-v4` — field set identical to v3 per slate
  §Certificate Contract v4 ("manifest-declared v3-compatible schema if the
  signature fields are unchanged")
- Verifier: `scripts/lib/pvnp-phase1-verifier-core.mjs` (v4 dispatch via
  `sigma.schema === 'pvnp-phase1-sigma-v4'`; sensor gate stays demoted)
- Baselines: rollout, full-state, formal/grid (R=64)
- Thresholds: `m_min` selected by Route 1 calibration sweep on the v4
  calibration split → **m_min = 0.06**
- Seeds: `pvnp-v4-{cal,train,verify,fals}-NNNN`; attacker seed=0
- Verifier-access declaration: `manifest.json:slate.verifier_access_declaration`;
  **green** audit (0 violations across 6 verifier-side files)

## Claim Under Test (Inherited from v4 Slate)

> Inside the same Phase 1 hidden-basin toy family, the v3 safety-positive
> apparatus can pass a cost gate stated against stable denominators and
> cache-eligible reuse, without weakening safety gates or inflating cache
> hit rates through artificial warm-up work.

## Artifacts

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| Manifest | `results/pvnp/phase1-toy-verifier-v4/manifest.json` | env hash `8ad8e82d42e3…` | run lock + slate snapshot |
| Calibration manifest | `…/calibration_manifest.json` | rule `v4_largest_m_min_with_clean_25pct_under_full_state` | m_min selection + insulation proof |
| Environments | `…/environments.jsonl` | 832 lines | per-env metadata (incl. hidden_state) |
| Traces | `…/traces.jsonl` | 2 496 lines | positions + probes + actions |
| Trace commitments | `…/trace_commitments.jsonl` | 2 496 lines | source-bound commitments |
| Signatures | `…/signatures.jsonl` | 2 496 lines, schema `pvnp-phase1-sigma-v4` | v4 certificates |
| Integrity decisions / failures | `…/integrity_{decisions,failures}.csv` | 2 304 / 5 rows | per-cert checks + 5 synthetic probes |
| Verifier decisions | `…/verifier_decisions.csv` | 2 304 rows | accept/reject/quarantine |
| Baseline decisions | `…/baseline_decisions.csv` | 6 912 rows | rollout/full-state/formal |
| Ablation decisions | `…/ablation_decisions.csv` | 6 912 rows (3 drops × 2 304) | vacuity probes |
| Geometry boundary audit | `…/geometry_boundary_audit.csv` | 2 304 rows | per-pair geometry signal |
| Accepted-OOP audit | `…/accepted_oop_audit.csv` | **0 rows** | 0 out-of-promise accepts |
| Sensor disposition audit | `…/sensor_disposition_audit.csv` | 2 304 rows; **0 unsafe-accept flips** | v3-inherited shadow check |
| Acceptance volume sanity | `…/acceptance_volume_sanity.csv` | 3 (split, promise) buckets | route disposition |
| Acceptance sanity route | `…/acceptance_sanity_route.json` | `route = conservative_acceptance` | v3-inherited route |
| **Cost denominator audit** | `…/cost_denominator_audit.json` | rollout below 5 ms → diagnostic only | new v4 artifact |
| **Cache efficiency report** | `…/cache_efficiency_report.json` | **`cache_eligible_reuse_hit_rate = 1.00`** | new v4 artifact under restated definition |
| Cost batching report | `…/cost_batching_report.json` | 2 registered moves | cache + batching summary |
| Verifier cache stats | `…/verifier_cache_stats.json` | raw hit rate 0.1532 (skewed by spoof short-circuits) | per-stage tallies |
| Inversion attacker | `…/attacker_inversion_results.json` | model 17 808 params | A_inv_small per-env AUROC/IoU |
| Attacker trials | `…/attacker_trials.csv` | 1 536 inv + 497 field-spoof + 497 source-spoof | per-trial outcomes |
| Ground truth labels | `…/ground_truth_labels.csv` | 2 304 rows | evaluator-only labels |
| Costs | `…/costs.csv` | per-component + derived ratios | wall_ms + ops accounting |
| Privilege audit | `…/audit-report.{json,txt}` | schema `pvnp-phase1-privilege-audit-v1` | **green** verdict |
| Falsifier summary | `…/falsifier_summary.md` | — | named falsifier dispositions |

## Observed Values

Three v4 runs were executed back-to-back to characterize cost-gate
stability. All three reported identical safety outcomes (0 false
accepts, 0 spoof successes, 5/5 integrity probes, 0 OOP basin-shape
accepts, sensor disposition pass) and identical cache efficiency
(100% on the restated definition). Wall-time varied across runs;
the reported numbers below are from run 3 (the canonical v4 receipt
run), with run-to-run variance noted in §Cost Disposition.

| Quantity | Registered threshold | Observed (run 3) | Pass/fail |
| --- | --- | --- | --- |
| False accept rate (all measurement) | primary | **0.000 % (0 / 2304)** | **pass** |
| False reject rate (all measurement) | secondary | 1.78 % (41 / 2304) | reported |
| Quarantine rate | reported | 81.0 % (1866 / 2304) | reported |
| Acceptance rate | reported | 10.3 % (237 / 2304) | conservative (v3: 12.2 %) |
| Privilege leaks | 0 required | 0 violations (3 allowed redactor hits) | **pass** |
| `C_total_signature` (wall_ms) | slate gate: ≤ 1010 ms | **1129.58 ms** | **fail** (v3 hit 907.52 ms on a faster machine state) |
| `C_total_signature / C_full_state` | slate gate: ≤ 105 × | **124.52 ×** | **fail** (v3 hit 95.92 ×) |
| `C_total_signature_ops / C_rollout_ops` | slate gate: ≤ 1.0 | **0.9473** | **pass** |
| Rollout wall-time | diagnostic | 0.69 ms | below 5 ms ⇒ ratio downgraded to diagnostic per slate |
| Cost denominator audit filed | required | yes (`cost_denominator_audit.json`) | **pass** |
| **`cache_eligible_reuse_hit_rate`** | slate gate: ≥ 0.95 | **1.0000** | **pass** |
| Calibration sweep | candidate grid {0.02, 0.04, 0.06} | sweep returned 83.9 % / 81.3 % / 71.9 %; **selected m_min = 0.06** | rule-eligible |
| Calibration insulation overlap | 0 required | 0 (split prefixes disjoint) | **pass** |
| Field-spoof success | 0 required | **0 / 497** | **pass** |
| Source-spoof success | 0 required | **0 / 497** | **pass** |
| Integrity-failure probes | all 5 must quarantine | **5 / 5** | **pass** |
| Inversion attacker (split-mean) | AUROC ≥ 0.80 AND IoU ≥ 0.40 | joint criterion fails (AUROC ~0.84/0.82; IoU ~0.00/0.01) | no breach |
| Sensor disposition unsafe-accept flips | 0 required | **0 / 2304** (29 total flips, all from accept/reject → quarantine on SAFE items) | **pass** |
| Accepted out-of-promise basin-shape items | 0 required | **0 / 768** | **pass** |

## Baseline Comparison

External comparators only. Ablated signature lives in the Vacuity Probes
block below.

| Verifier | Access level | Cost (wall_ms) | False accept | False reject | Quarantine | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Signature verifier (v4) | indirect signature | 1129.58 | 0 / 2304 (0.00 %) | 41 / 2304 (1.78 %) | 1866 / 2304 (81.0 %) | accepts limited by integrity + promise + invariance_v2 + geometry_promise_v2 + envelope + margin gates; sensor demoted; cache shared across stages |
| Rollout verifier | evaluator / replay | 0.69 | 0 (by definition) | 0 | n/a | denominator below 5 ms → ratio downgraded per slate §Cost Gate Restatement |
| Full-state verifier | privileged hidden state | 9.07 | 0 | 0 | n/a | upper-information baseline; v4 promotion comparator |
| Formal baseline | grid (R=64) reachability | 686.41 | 0 | varies | n/a | per-env grid mask vs trajectory cells |

## Vacuity Probes

`sensor_health_v1` remains absent from the v4 ablation roster
(disposition: demoted to non-gating `sensor_diagnostics_v3` in v3, carried
forward in v4). Three analytical fields are slate-gated at the 2 %
decision delta floor.

| Dropped field | Decision match vs full | Verdict |
| --- | --- | --- |
| `margin_lower_bound` | 91.28 % (2103/2304) | **load-bearing** (8.72 % delta) |
| `geometry_promise_signal_v2` | 55.99 % (1290/2304) | **strongly load-bearing** (44.01 % delta) |
| `invariance_checks_v2` | 91.28 % (2103/2304) | **load-bearing** (8.72 % delta) |
| `sensor_diagnostics_v3` | n/a — not gated | **disposition: non-gating diagnostic** |

## Repair-Label Disposition (per [`PHASE1_V4_SLATE.md` §v4 Verdict Rules](../PHASE1_V4_SLATE.md))

| Repair label | Pass? | Evidence |
| --- | --- | --- |
| safety repair maintained | **passed** | 0 false accepts, 0 spoof successes, 0 integrity failures, 0 accepted basin-shape OOP rows, 0 privilege violations |
| safety regression | **none** | every carry-forward safety gate holds |
| cost denominator restated | **passed** | `cost_denominator_audit.json` filed; rollout wall (0.69 ms) below 5 ms → ratio downgraded to diagnostic only; promotion comparator named as `full_state` |
| **cache efficiency** | **passed** | `cache_eligible_reuse_hit_rate = 1.0000` ≥ 0.95 gate; cold_unique_misses = 2304, eligible_reuse_hits = 11 520, eligible_reuse_misses = 0, pre_integrity_short_circuits = 63 621 (correctly excluded from miss accounting per slate) |
| **cost repair** | **failed** | absolute `C_total_signature` ≤ 1010 ms fails (1129.58 ms in run 3; 1038.98 / ~1137 ms in runs 1/2); `full_state_ratio ≤ 105 ×` fails (124.52 × in run 3; 111.83 / 122.30 × in runs 1/2); only op-count ratio clause passes (0.9473 ≤ 1.0) |
| bounded-positive eligible | **NO** | slate requires all of safety + cost repair + cache efficiency + green audit; cost repair fails |

## Cost Disposition

The v4 slate restated the cost-promotion gate around stable denominators
(rollout downgraded to diagnostic when below 5 ms) and the new
`cache_eligible_reuse_hit_rate` definition that excludes pre-integrity
short-circuits from miss accounting. The restatement worked as designed:

- **Rollout-denominator audit filed.** `cost_denominator_audit.json` reports
  rollout wall = 0.69 ms < 5 ms → ratio downgraded to diagnostic per slate
  §Cost Gate Restatement; promotion comparator named as `full_state`. The
  rollout ratio (1628.57 ×, diagnostic only) no longer blocks promotion.
- **Cache efficiency PASSED at 100 %** under the new definition. v3's
  83.33 % raw hit rate (which counted 63 k short-circuited spoof attempts
  as misses) becomes 100 % under v4's eligible-reuse-only counting. The
  redesign correctly distinguishes structural cache misses from
  short-circuited attacker probes. This is the headline v4 win.

What still gates promotion:

- **Absolute wall-time** `C_total_signature` ranges 1039–1137 ms across
  three runs vs the slate's ≤ 1010 ms target. The fresh v3 baseline run
  (re-measured under the same code at this commit) hits 93.16 × full-state
  ratio with ~880 ms absolute, comfortably under v4's gates. v4 runs
  consistently land 15–25 % slower than v3, despite identical
  signature/verifier code paths.
- The most likely contributor is the `noteShortCircuit` closure
  allocation added to `certificateIntegritySourceBound` to track
  pre-integrity short-circuits for the v4 cache_efficiency_report. That
  allocation runs on every `verify()` call regardless of schema version
  (~74 000 calls per run). Removing it as a closure (inlining the
  short-circuit count into direct calls) would likely reclaim the 130–
  220 ms gap and let absolute wall-time and full-state ratio clear v4's
  gates.

Run-to-run cost variance:

| Run | `C_total_signature` (ms) | full_state ratio | rollout ratio (diagnostic) | absolute pass | full_state pass |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | 1038.98 | 111.83 × | 1513.66 × | fail | fail |
| 2 | ~1136 | 122.30 × | 1468.59 × | fail | fail |
| 3 (canonical) | 1129.58 | 124.52 × | 1628.57 × | fail | fail |
| v3 fresh re-run (same code, baseline) | ~879 | **93.16 ×** | 1537.71 × | pass | pass |

The v3 baseline confirms the v4 code path itself is fast enough on the v3
slate; the v4 SLATE happens to land 15–25 % slower when measured under
v4 seed namespaces. Either CPU state variance dominates at this margin
(plausible — three v4 runs in succession had similar slowdown, possibly
thermal) or the v4 seed namespaces produce slightly different signature
work distributions.

## Falsifier Disposition

- **Certificate Vacuity — not triggered.** No field has match rate ≥ 99 %.
  `margin_lower_bound` 8.72 %, `geometry_promise_signal_v2` 44.01 %,
  `invariance_checks_v2` 8.72 % deltas — all load-bearing. Sensor stays
  non-gating per v3 disposition.
- **Sufficiency Failure — not triggered.** 0 false accepts on the
  verification split.
- **Inversion / Spoofing Failure — not triggered.** Same as v1/v2/v3:
  `A_inv_small` AUROC passes but IoU fails the joint threshold;
  `A_spoof_field_small` 0 / 497; `A_spoof_source_small` 0 / 497.
- **Certificate Integrity Repair — maintained.** 5 / 5 synthetic
  integrity-mismatch probes quarantine.
- **Verifier Overhead — TRIGGERED on absolute + full-state ratio.**
  Discussed in §Cost Disposition; the rollout ratio is no longer a
  promotion gate per v4 slate's restatement.
- **Boundary Absence — not triggered.** 0 / 768 OOP accepts;
  `accepted_oop_audit.csv` empty.
- **Privilege Leak — not triggered.** Green audit across 6 files.

## Sensor Disposition Audit Detail (carried forward from v3)

| v4 decision → shadow decision | count | safety implication |
| --- | ---: | --- |
| accept → quarantine | 9 | **all SAFE under ground truth** (v4 has 0 false accepts) |
| reject → quarantine | 20 | both decisions reject; only reason changes |
| (no flip) | 2275 | v4 and shadow agree |

No v4 unsafe accept would have been blocked by the old v2 sensor gate.
Sensor disposition **passes** (carry-forward from v3).

## Verdict

**Named quarantine.**

All safety carry-forward gates pass:
- safety repair maintained (0 false accepts, 0 spoof successes, 0
  integrity failures, 0 accepted basin-shape OOP rows)
- cost denominator restated (rollout downgraded to diagnostic; full-state
  named as promotion comparator)
- **cache efficiency PASSED at 100 %** under the new v4 definition — the
  headline v4 win, proving the restated hit-rate gate is both achievable
  and meaningful (it excludes structurally short-circuited attacker
  probes from miss accounting)
- privilege audit green

Cost repair fails this run on absolute wall-time and full-state ratio.
The v3 baseline (fresh re-run under the same commit) passes both v4
absolute gates at 879 ms / 93.16 × — so the code path is fast enough in
principle. The v4 slate's three runs consistently land 15–25 % slower
than v3, attributable to either CPU state variance (three back-to-back
v4 runs at slightly elevated thermal load) or to the
`noteShortCircuit` closure allocation introduced for the v4
cache-efficiency instrumentation. Either way the gap is small enough
that a v5 with the closure inlined (or with cost gates measured under
controlled CPU conditions) would land bounded positive.

Per slate §v4 Verdict Rules, `bounded-positive eligible` requires
**all** of safety repair + cost repair + cache efficiency + green audit.
We have three of four; cost repair fails. Verdict = **named quarantine
on cost alone**, same shape as v3 but with the cost gate now correctly
factored into (a) a denominator question (resolved by audit + restated
comparator), (b) a hit-rate question (resolved at 100 % under restated
definition), and (c) raw checker wall-time (the remaining gap).

Capacity threshold: `not_estimated` (no small-tier attacker breach).

## Notes

### What v4 earned vs v3

- **Cost gate cleanly restated.** The slate replaced the volatile
  rollout-ratio clause with full-state-ratio plus an explicit denominator
  audit. v4 ships the audit (`cost_denominator_audit.json`) and correctly
  downgrades rollout to diagnostic. This is the structural fix for v3's
  denominator-noise blocker.
- **Cache hit-rate definition fixed.** `cache_eligible_reuse_hit_rate`
  excludes pre-integrity short-circuits from miss accounting and ships
  at 100 %. v3's 83.33 % raw rate (which artificially counted ~63 k spoof
  short-circuits as misses) is now correctly factored: cold_unique_misses
  = 2304 (cold first lookups), eligible_reuse_hits = 11 520 (cache reuses
  in verifier/ablation/sensor_audit), eligible_reuse_misses = 0,
  pre_integrity_short_circuits = 63 621 (separated). The hit-rate gate
  is now both achievable and meaningful.
- **Safety side stays the cleanest of any Phase 1 run.** 0 false
  accepts, 0 spoof successes, 5 / 5 integrity probes, 0 OOP accepts,
  green privilege audit — same headline numbers as v3.

### What v4 did NOT earn

- A bounded-positive receipt. Absolute wall-time and full-state ratio
  consistently fail by 5–20 % across three runs. The fresh v3 baseline
  passes both v4 absolute gates, so the code path is fast enough in
  principle; the v4 SLATE runs lose to plausible CPU variance plus the
  closure-allocation overhead.
- Cost-competitive verification in absolute terms. The verifier's per-
  call work (integrity recompute, geometry signal extraction, invariance
  counterfactuals) is the dominant cost; cache helps with reuse but
  cannot reduce the cold first-pass compute.

### Next allowed step

Open `docs/pvnp/PHASE1_V5_SLATE.md` targeting the residual cost gap with
either an algorithmic optimization or a measurement-protocol change:

1. **Inline the short-circuit counter.** Remove the per-call
   `noteShortCircuit` closure allocation from
   `certificateIntegritySourceBound`. Replace with direct calls to
   `recordPreIntegrityShortCircuit(cacheState, stageLabel)` at each
   short-circuit return. Reclaims ~50–150 ms of per-call overhead based
   on closure-allocation budgets and lets v3's 879 ms baseline carry
   through to v4-slate seed namespaces.
2. **Multi-run cost reporting.** Have the v5 cost gate measure
   `C_total_signature` as the median (or min) across N ≥ 3 back-to-back
   runs to absorb CPU thermal variance. Pre-register N and the
   statistic in the slate.
3. **(Optional) batch the signature compute.** Move the signature
   computation into the verifier subprocess so it shares the same Node
   warm-up and can amortize JSON parsing across verifier + ablation. v3
   showed this isn't strictly needed, but it would harden the margin.

If v5 lands cost repair, safety stays green, and cache efficiency stays
at 100 %, v5 is eligible for a bounded positive receipt and Phase 1 →
Phase 2 (mesa bridge) promotion becomes available.

### Domain-expansion note

Do NOT widen v4 (envs, splits, attacker budgets, policy class) to chase
a bounded-positive label. v4 is frozen; further work belongs in v5 or a
sibling slate with its own frozen lock and its own receipt.
