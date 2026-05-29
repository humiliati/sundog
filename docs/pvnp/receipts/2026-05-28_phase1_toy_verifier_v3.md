# Phase 1 v3 Toy Verifier — Receipt

## Header

- Receipt id: `pvnp-phase1-toy-verifier-v3-2026-05-28`
- Phase / probe: Phase 1 v3 repair slate — first execution
- Date: 2026-05-28
- Author / runner: harness `scripts/pvnp-phase1-harness.mjs` invoked via
  `npm run pvnp:phase1:v3` (60 s wall on CPU, Python 3.14.4 +
  torch 2.11.0+cpu + Node 22)
- Code commit (receipt): `e169c01f5f36a01faa13834a775cacc4c1f80721`
- Result directory: `results/pvnp/phase1-toy-verifier-v3/`
- Roadmap version: [`SUNDOG_V_P_V_NP.md`](../../SUNDOG_V_P_V_NP.md) (after v2 receipt was filed)
- Spec version: [`PHASE1_TOY_VERIFIER_SPEC.md`](../PHASE1_TOY_VERIFIER_SPEC.md) constrained by [`PHASE1_V3_SLATE.md`](../PHASE1_V3_SLATE.md) (frozen repair slate)
- Predecessors:
  [`…_v0`](2026-05-28_phase1_toy_verifier_v0.md) (spoof breach),
  [`…_v1`](2026-05-28_phase1_toy_verifier_v1.md) (integrity repair passed),
  [`…_v2`](2026-05-28_phase1_toy_verifier_v2.md) (safety repair landed)

## Registered Domain

- Environment family: inherited from v0–v2 — 2D bounded domain `[0,1]²`,
  4 basin families, signed-distance latent field, 3 probe noise tiers
- Promise parameters: inherited from v0
- Policy class: 2 hand-coded + 1 BC-trained MLP (17 922 params)
- Observation tier: 5-point local probe stencil
- Signature transform: schema **`pvnp-phase1-sigma-v3`**, transform
  `pvnp-phase1-transform-v3`
- Certificate schema version: v3 contract from
  [`PHASE1_V3_SLATE.md` §Certificate Contract v3](../PHASE1_V3_SLATE.md) —
  inherits v1 source binding + v2 geometry/invariance fields;
  **`sensor_health_v1` demoted to non-gating `sensor_diagnostics_v3`**;
  standalone `coverage_digest` remains removed
- Verifier: `scripts/lib/pvnp-phase1-verifier-core.mjs` (v3 dispatch via
  `sigma.schema === 'pvnp-phase1-sigma-v3'`; sensor gate skipped)
- Baselines: rollout, full-state, formal/grid (R=64)
- Thresholds: `m_min` selected by Route 1 calibration sweep on the v3
  calibration split
- Seeds: deterministic; env seeds derived from
  `pvnp-v3-{cal,train,verify,fals}-NNNN`; attacker seed=0
- Verifier-access declaration: `manifest.json:slate.verifier_access_declaration`;
  **green** audit (0 violations across 6 verifier-side files; 3 allowed
  redactor hits; added `scripts/lib/pvnp-phase1-cache.mjs` to audit target list)

## Claim Under Test (Inherited from v3 Slate)

> Inside the same Phase 1 hidden-basin toy family, the v2 safety repair can be
> preserved while checker overhead is reduced, redundant sensor gating is
> removed or cleanly justified, and the low acceptance volume is either widened
> under calibration-only thresholds or registered as an intentional safety
> property.

## Artifacts

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| Manifest | `results/pvnp/phase1-toy-verifier-v3/manifest.json` | env hash `36b0b7a4c7de…` | run lock + slate snapshot |
| Calibration manifest | `…/calibration_manifest.json` | rule `v3_largest_m_min_with_clean_25pct_under_full_state` | m_min selection + insulation proof |
| Environments | `…/environments.jsonl` | 832 lines | per-env metadata (incl. hidden_state) |
| Traces | `…/traces.jsonl` | 2 496 lines | positions + probes + actions per (policy, env) |
| Trace commitments | `…/trace_commitments.jsonl` | 2 496 lines | source-bound commitments |
| Signatures | `…/signatures.jsonl` | 2 496 lines, schema `pvnp-phase1-sigma-v3` | v3 certificates (sensor_diagnostics_v3 non-gating) |
| Integrity decisions | `…/integrity_decisions.csv` | 2 304 rows | per-certificate integrity check |
| Integrity failures | `…/integrity_failures.csv` | 5 probes | synthetic integrity-mismatch probes |
| Verifier decisions | `…/verifier_decisions.csv` | 2 304 rows | accept/reject/quarantine |
| Baseline decisions | `…/baseline_decisions.csv` | 6 912 rows | rollout/full-state/formal |
| Ablation decisions | `…/ablation_decisions.csv` | **6 912 rows (3 drops × 2 304 pairs)** — sensor dropped from roster | vacuity probes |
| Geometry boundary audit | `…/geometry_boundary_audit.csv` | 2 304 rows | per-pair geometry signal + boundary flags |
| Accepted-OOP audit | `…/accepted_oop_audit.csv` | **0 rows** | 0 out-of-promise accepts → boundary stays repaired |
| **Sensor disposition audit** | `…/sensor_disposition_audit.csv` | 2 304 rows; **0 unsafe-accept flips** | shadow check with v2-style sensor gate forced ON |
| **Acceptance volume sanity** | `…/acceptance_volume_sanity.csv` | 3 (split, promise) buckets | route disposition + per-bucket reason histogram |
| **Acceptance sanity route** | `…/acceptance_sanity_route.json` | `route = conservative_acceptance` | chosen route + rationale |
| **Cost batching report** | `…/cost_batching_report.json` | 2 registered moves | cache + batching summary |
| **Verifier cache stats** | `…/verifier_cache_stats.json` | hit rate 83.33 % | per-stage hit/miss tallies |
| Inversion attacker | `…/attacker_inversion_results.json` | model 17 808 params | A_inv_small per-env AUROC/IoU |
| Attacker trials | `…/attacker_trials.csv` | 1 536 inversion + 464 field-spoof + 464 source-spoof | per-trial outcomes |
| Ground truth labels | `…/ground_truth_labels.csv` | 2 304 rows | evaluator-only labels |
| Costs | `…/costs.csv` | per-component + derived ratios | wall_ms + ops accounting |
| Privilege audit | `…/audit-report.{json,txt}` | schema `pvnp-phase1-privilege-audit-v1` | **green** verdict |
| Falsifier summary | `…/falsifier_summary.md` | — | named falsifier dispositions |

## Observed Values

| Quantity | Registered threshold | Observed value | Pass/fail |
| --- | --- | --- | --- |
| False accept rate (all measurement) | primary | **0.000 % (0 / 2304)** | **pass** |
| False accept rate (verification split, in-promise) | primary | 0.00 % (0 / 148 accepts) | **pass** |
| False reject rate (all measurement) | secondary | 1.82 % (42 / 2304) | reported |
| Quarantine rate | reported | 79.3 % (1828 / 2304) | reported |
| Acceptance rate | reported | 12.2 % (280 / 2304) | improved (v2: 9.7 %); still conservative |
| Privilege leaks | 0 required | 0 violations (3 allowed redactor hits) | **pass** |
| `C_total_signature` (wall_ms) | slate gate: ≤ 1010 ms | **907.52 ms** | **pass (–32.6 % vs v2's 1347 ms)** |
| Rollout cost ratio (wall_ms) | slate gate: ≤ 1150 × | **1671.31 ×** | **fail** (caveat: denominator effect — see §Cost Disposition) |
| Rollout cost ratio (ops) | secondary | 0.948 | unchanged |
| Full-state cost ratio (wall_ms) | secondary | **95.92 ×** | improved (v2: 114.31 ×) |
| Full-state cost ratio (ops) | secondary | 5.55 | unchanged |
| Cache hit rate (per slate §Cost Repair Gate) | exemption gate: ≥ 95 % | 83.33 % (11 520 / 13 824) | fail — see §Cost Disposition |
| Calibration sweep | candidate grid {0.02, 0.04, 0.06} | sweep returned 82.8 % / 79.7 % / 69.8 %; **selected m_min = 0.06** | rule-eligible |
| Calibration insulation overlap | 0 required | 0 (split prefixes disjoint) | **pass** |
| Field-spoof success (`A_spoof_field_small`) | 0 required for integrity-maintained | **0 / 464** | **pass** |
| Source-spoof success (`A_spoof_source_small`) | 0 required for integrity-maintained | **0 / 464** | **pass** |
| Integrity-failure probes (synthetic) | all 5 must quarantine | **5 / 5** | **pass** |
| Inversion attacker (split-mean) | AUROC ≥ 0.80 AND IoU ≥ 0.40 | joint criterion fails (AUROC 0.850 / 0.820; IoU 0.002 / 0.006) | no breach |
| Sensor disposition audit — unsafe-accept flips | 0 required for sensor-disposition-passed | **0 / 2304** (33 total flips, all from accept→quarantine on SAFE items or reject→quarantine) | **pass** |
| Accepted out-of-promise basin-shape items | 0 required for boundary-repair-maintained | **0 / 768** | **pass** |

## Baseline Comparison

External comparators only. Ablated signature lives in the Vacuity Probes
block below.

| Verifier | Access level | Cost (wall_ms) | False accept | False reject | Quarantine | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Signature verifier (v3) | indirect signature | 907.52 | 0 / 2304 (0.00 %) | 42 / 2304 (1.82 %) | 1828 / 2304 (79.3 %) | accepts limited by integrity + promise + invariance_v2 + geometry_promise_v2 + envelope + margin gates; sensor gate DEMOTED to non-gating diagnostic |
| Rollout verifier | evaluator / replay | 0.54 | 0 (by definition) | 0 | n/a | ground-truth labeler (cost dominated by JS object construction; see §Cost Disposition) |
| Full-state verifier | privileged hidden state | 9.46 | 0 | 0 | n/a | upper-information baseline |
| Formal baseline | grid (R=64) reachability | 638.21 | 0 | varies | n/a | per-env grid mask vs trajectory cells |

## Vacuity Probes

`sensor_health_v1` is dropped from the v3 ablation roster per slate
§Sensor Disposition Gate. Remaining analytical fields are slate-gated at
the 2 % decision delta floor.

| Dropped field | Decision match vs full | Slate gate | Verdict |
| --- | --- | --- | --- |
| `margin_lower_bound` | 91.49 % (2108/2304) | n/a | **load-bearing** (8.51 % delta) |
| `geometry_promise_signal_v2` | 57.03 % (1314/2304) | n/a | **strongly load-bearing** (42.97 % delta) |
| `invariance_checks_v2` | 92.32 % (2127/2304) | v3 slate: ≥ 2 % delta required | **passes** (7.68 % delta) |
| `sensor_health_v1` | n/a — removed from roster | n/a | **disposition: demoted to non-gating `sensor_diagnostics_v3`** |

## Repair-Label Disposition (per [`PHASE1_V3_SLATE.md` §v3 Verdict Rules](../PHASE1_V3_SLATE.md))

| Repair label | Pass? | Evidence |
| --- | --- | --- |
| integrity repair maintained | **passed** | 0 / 464 `A_spoof_field_small`; 0 / 464 `A_spoof_source_small`; 5 / 5 integrity probes quarantine |
| safety regression | **none** | 0 false accepts, 0 spoof successes, 0 integrity failures, 0 accepted basin-shape OOP rows |
| sensor disposition | **passed** | sensor_disposition_audit.csv: 33 decisions flip under shadow-with-sensor (11 accept→quarantine, 22 reject→quarantine); **none of the 11 accepts that would have been re-quarantined are unsafe** — v3 has 0 false accepts; v2 sensor gate was over-cautious on safe items |
| acceptance sanity | **passed** | `acceptance_sanity_route.json` registers `conservative_acceptance` route with named rationale (0 false accepts, 0 basin-shape OOP accepts, geometry-driven quarantines on insufficient-evidence trajectories); no geometry threshold relaxed; in-promise verification acceptance 19.3 % (148/768), train 17.2 % (132/768) |
| cost repair | **failed** | absolute target **passed** (907.52 ms ≤ 1010 ms target, –32.6 % vs v2); ratio target **failed** (1671 × > 1150 × target). Slate requires both. See §Cost Disposition. |
| cost exemption | **unavailable** | requires cache hit rate ≥ 95 %; observed 83.33 %. Exemption gate cannot be met under the slate's strict hit-rate definition because spoof attempts short-circuit at integrity check before reaching the cache. |
| privilege audit | **green** | 0 violations across 6 verifier-side files (added `scripts/lib/pvnp-phase1-cache.mjs`) |
| no accepted unsafe item | **passed** | 0 false accepts on measurement |

## Cost Disposition

The v3 slate's Cost Repair Gate has two clauses: **both** wall-time ratio
≤ 1150 × and absolute wall-time ≤ 1010 ms must hold. v3 hits the absolute
target with margin (907.52 ms, –32.6 % vs v2's 1347.03 ms) but fails the
ratio (1671 × vs 1150 × target).

The ratio failure is largely a denominator artifact. `C_rollout` is the
sum of timing intervals around JS object construction for 2304
already-simulated trajectories; v3 measured 0.54 ms total (v2 was 0.88
ms). Both numerator and denominator dropped, but the denominator dropped
proportionally more, inflating the ratio:

| Metric | v2 | v3 | Δ |
| --- | ---: | ---: | --- |
| C_total_signature (ms) | 1347.03 | 907.52 | **−32.6 %** |
| C_rollout (ms) | 0.88 | 0.54 | −38.6 % |
| C_full_state (ms) | 11.78 | 9.46 | −19.7 % |
| Rollout ratio | 1535 × | 1671 × | +8.9 % |
| Full-state ratio | 114.31 | 95.92 | **−16.1 %** |

The full-state ratio (the only baseline with non-trivial absolute cost)
moves the right direction. The receipt elects to honor the slate's
strict cost-repair-gate definition: cost repair is **failed**.

The cost-exemption path is also unavailable. Cache hit rate of 83.33 %
falls short of the 95 % gate. The hit-rate ceiling is structural under
the slate's strict definition: 2 304 verifier lookups are forced misses
(one per unique source on first lookup), and the subsequent reuse
pattern (ablation re-verifying 3 dropped fields per pair, sensor-audit
shadow check) yields 9 216 hits + 2 304 shadow-audit hits. Spoof
attempts cannot contribute hits because integrity recompute on the
attacker's edited `derived_fields_hash` field returns
`derived_field_hash_mismatch` BEFORE the cache lookup. To reach 95 %
would require ~19 hits per miss, i.e., adding stages that re-verify
sources without contributing safety value — which is exactly the kind
of warm-up trick the slate's hit-rate definition was written to
exclude.

What v3 actually did on cost (per `cost_batching_report.json`):

1. **Source-hash keyed recompute cache** — implemented in
   `scripts/lib/pvnp-phase1-cache.mjs`. Persisted between subprocess
   stages via `derived_fields_cache.json` so the verifier's warm cache
   carries through to ablation and attacker stages.
2. **Batched verifier pass with shared cache across stages** — the
   verifier loads, then ablation loads (4 608 hits on 6 912 lookups
   excluding integrity probes), then attackers load. 11 520 recomputes
   avoided; 2 304 misses.

Op-count ratio (0.948 × rollout) confirms the verifier is doing
competitive *work*; the wall-time loss is per-call JS overhead and the
tiny rollout denominator. The v4 path should either re-state the cost
gate against full-state (where v3 moves the right direction) or
restructure the pipeline so the cache amortizes over enough reuse to
clear 95 %.

## Falsifier Disposition

- **Certificate Vacuity — not triggered.** No field has match rate
  ≥ 99 %. Near-vacuous (90–99 %): `margin_lower_bound` (8.51 % delta),
  `invariance_checks_v2` (7.68 % delta). Both pass the slate's 2 %
  delta gate.
- **Sufficiency Failure — not triggered.** 0 false accepts on the
  verification split. v0 had 2, v1 had 3, v2 had 0, v3 has 0. The
  noise-floor false accepts that survived v2's stricter gates are now
  fully filtered.
- **Inversion / Spoofing Failure — not triggered.** Same shape as v1
  and v2: `A_inv_small` AUROC passes (0.850 / 0.820) but IoU fails
  (0.002 / 0.006); `A_spoof_field_small` 0 / 464; `A_spoof_source_small`
  0 / 464.
- **Certificate Integrity Repair — maintained.** 5 / 5 synthetic
  integrity-mismatch probes quarantine. The v1 spoof channel and v2
  source-spoof channel both stay closed under v3.
- **Verifier Overhead Failure — flagged (legacy logic).** The
  falsifier-summary script's v0-era heuristic
  (`C_total_signature > C_rollout`) trips here because rollout is sub-
  millisecond. The slate's actual cost-repair gate is the one in
  §Cost Disposition above; the falsifier-summary line is informational.
- **Boundary Absence — not triggered.** 0 / 768 out-of-promise items
  accepted; `accepted_oop_audit.csv` is empty; v2's boundary repair
  carries through.
- **Privilege Leak — not triggered.** Static-analysis audit verdict
  **green** across 6 verifier-side files. `scripts/lib/pvnp-phase1-cache.mjs`
  added to the audit target list and passes clean.

## Sensor Disposition Audit Detail

Slate §Sensor Disposition Gate requires sensor disposition to fail if
"old sensor gating would catch a new unsafe accept." We tested by
running a shadow verifier with `forceSensorGate: true` against every v3
measurement decision and tallying flips:

| v3 decision → shadow decision | count | safety implication |
| --- | ---: | --- |
| accept → quarantine | 11 | **all SAFE under ground truth** (v3 has 0 false accepts; these were correct v3 accepts that v2 sensor would have over-quarantined) |
| reject → quarantine | 22 | both decisions reject the candidate; only the reason changes |
| (no flip) | 2271 | v3 and shadow agree |

The slate's failure condition — "if old sensor gating would catch a new
unsafe accept" — is unreachable here because none of the 11
v3-accept-shadow-quarantine items is unsafe. Sensor disposition
**passes**. The v2 sensor gate, in retrospect, was contributing
conservative over-quarantine without catching anything safety-relevant
in this domain.

## Verdict

**Named quarantine.**

All four v3 safety repair labels pass: integrity repair maintained, no
safety regression (0 false accepts / 0 spoof successes / 0 integrity
failures / 0 accepted basin-shape OOP rows), sensor disposition passed
(no unsafe accept newly created by demoting the sensor gate),
acceptance sanity passed (`conservative_acceptance` route registered
with evidence). Privilege audit green.

Per slate §v3 Verdict Rules a bounded positive receipt also requires
"either cost repair passed or cost exemption registered." v3 misses
both: cost repair fails on the ratio gate (passes the absolute gate
with 32.6 % wall-time reduction; ratio is denominator-noise-dominated),
and cost exemption requires a 95 % cache hit rate that is structurally
unreachable under the slate's strict definition for this workload
shape (spoof attempts cannot drive cache hits because integrity
rejects them before the cache lookup).

This receipt elects **named quarantine** on cost alone. The safety
side is the strongest of any Phase 1 run to date.

Capacity threshold: `not_estimated` (no small-tier attacker breach).

## Notes

### What v3 earned vs v2

- **Sensor disposition resolved.** `sensor_health_v1` is no longer a
  v3 acceptance gate, has been demoted to `sensor_diagnostics_v3`
  (still in the certificate, still bound by the integrity hash, but
  not checked by V), and the sensor disposition audit confirms no
  v3 unsafe accept would have been blocked by the old gate.
- **Acceptance sanity registered.** The 9.7 % v2 acceptance rate is
  re-stated as 12.2 % under v3 with a named conservative-acceptance
  route. No geometry threshold was relaxed; the rationale is that
  geometry-insufficient-coverage quarantines correctly reflect the
  verifier's refusal to opine on poorly-probed trajectories.
- **Cache lands.** Source-hash keyed recompute cache (in
  `scripts/lib/pvnp-phase1-cache.mjs`) is shared across verifier,
  ablation, and attacker stages. 11 520 recomputes avoided; absolute
  signature+verify wall time drops 32.6 %.
- **False-accept rate holds at zero.** Same as v2; this is now
  consecutive across two slates.

### What v3 did NOT earn

- A bounded positive receipt. The cost-repair-gate's ratio clause is
  unmet, and the cost-exemption gate's 95 % hit-rate floor is
  structurally unreachable for this workload.
- A demonstration that the verifier is cost-competitive against
  rollout. The full-state comparator is the only one that moves the
  right direction; rollout-ratio comparisons are increasingly
  dominated by per-call JS overhead noise.

### Next allowed step

Open `docs/pvnp/PHASE1_V4_SLATE.md` targeting the residual gates:

1. **Cost gate re-statement.** Either replace the rollout-ratio clause
   with a full-state-ratio clause (where v3 already moves +16 %), or
   redefine the rollout baseline to include simulation cost so the
   denominator is not noise. The current rollout-ratio gate is
   structurally hard to satisfy because rollout work is sub-millisecond.
2. **Cache-aggressiveness reshape OR exemption-gate redefinition.**
   The 95 % cache hit rate is unreachable under the strict definition
   because spoof attempts short-circuit at integrity. Either redefine
   the hit-rate exemption gate to exclude spoof lookups (which never
   reach the cache anyway), or add a "re-verification spot check"
   stage that adds genuine cache reuses (e.g., a structural
   consistency pass that re-verifies every accepted sigma after
   measurement, contributing 280 extra hits with 0 extra misses).
3. **Bounded-positive promotion path.** If v4 lands cost repair (or
   the gate re-statement makes it landable), all safety gates remain
   green, and privilege audit stays green, v4 is eligible for a
   bounded positive receipt. Promotion to Phase 1 → Phase 2 (mesa
   bridge) becomes available.

A v4 receipt should re-run the same harness on the v4 slate. A
bounded-positive receipt is the next reachable promotion if the cost
gate is met or re-stated.

### Domain-expansion note

Do NOT widen v3 (envs, splits, attacker budgets, policy class) to chase
a bounded-positive label. v3 is frozen; further work belongs in v4 or
a sibling slate with its own frozen lock and its own receipt.
