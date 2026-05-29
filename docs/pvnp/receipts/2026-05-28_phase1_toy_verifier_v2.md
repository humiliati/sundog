# Phase 1 v2 Toy Verifier — Receipt

## Header

- Receipt id: `pvnp-phase1-toy-verifier-v2-2026-05-28`
- Phase / probe: Phase 1 v2 repair slate — first execution
- Date: 2026-05-28
- Author / runner: harness `scripts/pvnp-phase1-harness.mjs` invoked via
  `npm run pvnp:phase1:v2` (Python 3.14.4 + torch 2.11.0+cpu + Node 22)
- Code commit (run): `f56383c3454bd2019d56d030e75c29d3519a98ed`
- Code commit (receipt): `8177a578c404ef37fa964303913b6dae6ed34491`
- Result directory: `results/pvnp/phase1-toy-verifier-v2/`
- Roadmap version: [`SUNDOG_V_P_V_NP.md`](../../SUNDOG_V_P_V_NP.md) (after v1 receipt was filed)
- Spec version: [`PHASE1_TOY_VERIFIER_SPEC.md`](../PHASE1_TOY_VERIFIER_SPEC.md) constrained by [`PHASE1_V2_SLATE.md`](../PHASE1_V2_SLATE.md) (frozen repair slate inheriting v1 source-binding contract)
- Predecessors:
  [`2026-05-28_phase1_toy_verifier_v0.md`](2026-05-28_phase1_toy_verifier_v0.md) (spoof breach),
  [`2026-05-28_phase1_toy_verifier_v1.md`](2026-05-28_phase1_toy_verifier_v1.md) (integrity repair passed)

## Registered Domain

- Environment family: inherited from v0/v1 — 2D bounded domain `[0,1]²`,
  4 basin families (circle, ellipse, crescent, decoy_doublet),
  signed-distance latent field, 3 probe noise tiers
- Promise parameters: inherited from v0/v1
- Policy class: 2 hand-coded + 1 BC-trained MLP (≤ 20 k params)
- Observation tier: 5-point local probe stencil
- Signature transform: `scripts/lib/pvnp-phase1-signature-core.mjs`,
  schema **`pvnp-phase1-sigma-v2`**
- Certificate schema version: v2 contract from
  [`PHASE1_V2_SLATE.md` §Certificate Contract v2](../PHASE1_V2_SLATE.md) —
  inherits v1 integrity binding; **removes standalone `coverage_digest`**;
  adds `geometry_promise_signal_v2` (shape, scale, curvature, evidence-
  coverage, topology, boundary flags) and `invariance_checks_v2` (coordinate-
  equivalence residual, near-boundary counterfactual, decoy consistency)
- Verifier: `scripts/lib/pvnp-phase1-verifier-core.mjs` (v2 dispatch via
  `sigma.schema === 'pvnp-phase1-sigma-v2'`)
- Baselines: rollout, full-state, formal/grid (R=64); ablated signature
  reported separately under Vacuity Probes
- Thresholds: `m_min` selected by Route 1 calibration sweep on the v2
  calibration split (split-prefixed `pvnp-v2-cal-*`)
- Seeds: deterministic; env seeds derived from
  `pvnp-v2-{cal,train,verify,fals}-NNNN`; attacker seed=0
- Verifier-access declaration:
  `manifest.json:slate.verifier_access_declaration`; **green** audit
  (0 violations across 5 verifier-side files; 3 allowed redactor hits)

## Claim Under Test (Inherited from v2 Slate)

> Inside the same Phase 1 hidden-basin toy family, a source-bound signature
> verifier can make invariance checks decision-relevant, close accepted
> basin-shape out-of-promise rows, and reduce checker overhead without
> widening the policy class, attacker budget, or measurement slate.

## Artifacts

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| Manifest | `results/pvnp/phase1-toy-verifier-v2/manifest.json` | env hash `2ad8ce0193c2…` | run lock + slate snapshot |
| Calibration manifest | `…/calibration_manifest.json` | rule `v2_largest_m_min_with_clean_25pct_under_full_state` | m_min selection + insulation proof |
| Environments | `…/environments.jsonl` | 832 lines | per-env metadata (incl. hidden_state) |
| Traces | `…/traces.jsonl` | 2 496 lines | positions + probes + actions per (policy, env) |
| Trace commitments | `…/trace_commitments.jsonl` | 2 496 lines | source-bound commitments (v1 + v2 integrity checks) |
| Signatures | `…/signatures.jsonl` | 2 496 lines, schema `pvnp-phase1-sigma-v2` | certificates consumed by verifier |
| Integrity decisions | `…/integrity_decisions.csv` | 2 304 rows | per-certificate integrity check |
| Integrity failures | `…/integrity_failures.csv` | **5 probes** (added `duplicate_trace_id`) | synthetic integrity-mismatch probes |
| Verifier decisions | `…/verifier_decisions.csv` | 2 304 rows | accept/reject/quarantine |
| Baseline decisions | `…/baseline_decisions.csv` | 6 912 rows (3 baselines × 2 304 pairs) | rollout/full-state/formal |
| Ablation decisions | `…/ablation_decisions.csv` | 9 216 rows (4 drops × 2 304 pairs) | vacuity probes |
| Geometry boundary audit | `…/geometry_boundary_audit.csv` | 2 304 rows | per-pair geometry signal + boundary flags |
| Accepted-OOP audit | `…/accepted_oop_audit.csv` | **0 rows (header only)** | zero out-of-promise accepts → boundary repair PASSED |
| Inversion attacker | `…/attacker_inversion_results.json` | model 17 808 params | A_inv_small per-env AUROC/IoU |
| Attacker trials | `…/attacker_trials.csv` | 1 536 inversion + 513 field-spoof + 513 source-spoof | per-trial outcomes |
| Ground truth labels | `…/ground_truth_labels.csv` | 2 304 rows | evaluator-only labels |
| Costs | `…/costs.csv` | per-component + derived ratios | wall_ms + ops accounting |
| Privilege audit | `…/audit-report.{json,txt}` | schema `pvnp-phase1-privilege-audit-v1` | **green** verdict |
| Falsifier summary | `…/falsifier_summary.md` | — | named falsifier dispositions |

## Observed Values

| Quantity | Registered threshold | Observed value | Pass/fail |
| --- | --- | --- | --- |
| False accept rate (all measurement) | primary | **0.000 % (0 / 2304)** | **pass** |
| False accept rate (verification split, in-promise) | primary | **0.00 % (0 / 223 accepts)** | **pass** |
| False reject rate (all measurement) | secondary | 1.78 % (41 / 2304) | reported |
| Quarantine rate | reported | 81.3 % (1873 / 2304) | reported — very conservative |
| Acceptance rate | reported | 9.7 % (223 / 2304) | reported — very conservative |
| Privilege leaks | 0 required | 0 violations (3 allowed redactor hits) | **pass** |
| `C_total_signature` (wall_ms) | reported | 1 347.03 ms | reported |
| Rollout cost ratio (wall_ms) | slate gate: ≥ 25 % improvement vs v1 (= 854 ×) | **1 535.08 × (worse than v1's 1139.05 ×)** | **fail** |
| Rollout cost ratio (ops) | secondary | 0.948 (≈ v0, v1) | unchanged |
| Full-state cost ratio (wall_ms) | secondary | 114.31 × (vs v1's 81.06 ×) | regression |
| Full-state cost ratio (ops) | secondary | 5.55 (≈ v0, v1) | unchanged |
| Calibration sweep | candidate grid {0.02, 0.04, 0.06}; rule = largest m_min with calibration-clean ≥ 25 % under full-state | sweep returned 87.0 % / 80.2 % / 76.6 %; **selected m_min = 0.06** | rule-eligible |
| Calibration insulation overlap | 0 required | 0 (split prefixes disjoint) | **pass** |
| Field-spoof success (`A_spoof_field_small`) | 0 required for integrity-maintained | **0 / 513** | **pass** |
| Source-spoof success (`A_spoof_source_small`) | 0 required for integrity-maintained | **0 / 513** | **pass** |
| Integrity-failure probes (synthetic) | all 5 must quarantine (added `duplicate_trace_id`) | **5 / 5 quarantined** | **pass** |
| Inversion attacker (split-mean) | AUROC ≥ 0.80 AND IoU ≥ 0.40 | joint criterion fails (high AUROC, low IoU) | no breach |
| Inversion attacker (per-env) | informational | 1 / 1536 per-env joint hit | informational |
| Accepted out-of-promise basin-shape items | 0 required for boundary repair | **0 / 768** | **pass** |

## Baseline Comparison

External comparators only. Ablated signature lives in the Vacuity Probes
block below.

| Verifier | Access level | Cost (wall_ms) | False accept | False reject | Quarantine | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Signature verifier (v2) | indirect signature | 1 347.03 | 0 / 2304 (0.00 %) | 41 / 2304 (1.78 %) | 1873 / 2304 (81.3 %) | accepts limited by integrity + promise + invariance_v2 + sensor_v1 + geometry_promise_v2 + envelope + margin gates |
| Rollout verifier | evaluator / replay | 0.88 | 0 (by definition) | 0 | n/a | ground-truth labeler |
| Full-state verifier | privileged hidden state | 11.78 | 0 | 0 | n/a | upper-information baseline |
| Formal baseline | grid (R=64) reachability | 837.11 | 0 | varies (conservative) | n/a | per-env grid mask vs trajectory cells |

## Vacuity Probes

Internal sanity checks. Standalone `coverage_digest` is **removed in v2 per
the slate's Coverage Disposition Gate** — see §Coverage Disposition below.

| Dropped field | Decision match vs full | Slate gate | Verdict |
| --- | --- | --- | --- |
| `margin_lower_bound` | 90.97 % (2096/2304) | n/a | **load-bearing** (9.03 % delta) |
| `sensor_health_v1` | **98.26 % (2264/2304)** | inherited v1 gate: ≥ 2 % delta required, or named removal | **borderline fail** — 1.74 % delta < 2 % gate; slate-derived disposition: subsumed by stricter v2 geometry/invariance checks, name for v3 removal |
| `invariance_checks_v2` | 91.97 % (2119/2304) | v2 slate: ≥ 2 % delta required | **passes** (8.03 % delta) |
| `geometry_promise_signal_v2` | 58.46 % (1347/2304) | n/a | **strongly load-bearing** (41.54 % delta) |

## Repair-Label Disposition (per [`PHASE1_V2_SLATE.md` §v2 Verdict Rules](../PHASE1_V2_SLATE.md))

| Repair label | Pass? | Evidence |
| --- | --- | --- |
| integrity repair maintained | **passed** | 0 / 513 `A_spoof_field_small`; 0 / 513 `A_spoof_source_small`; 5 / 5 synthetic integrity-mismatch probes (incl. new `duplicate_trace_id`) quarantined |
| invariance repair | **passed** | `invariance_checks_v2` 8.03 % delta when dropped, well above 2 % slate gate |
| boundary repair | **passed** | 0 / 768 out-of-promise falsifier-split items accepted; `accepted_oop_audit.csv` has 0 rows; 956 quarantines via `geometry_insufficient_coverage` (573), `geometry_topology_ambiguous` (368), `geometry_curvature_out_of_envelope` (15) all fired in `geometry_promise_signal_v2` |
| coverage disposition | **passed** | standalone `coverage_digest` removed from v2 certificate per slate §Coverage Disposition Gate; coverage information appears only as `geometry_evidence_coverage` inside `geometry_promise_signal_v2` |
| cost repair | **failed** | wall-time ratio vs rollout = 1 535 × (slate required ≥ 25 % improvement on v1's 1 139 × → target ≤ 854 ×; observed regression to 1.35 × of v1) |
| privilege audit | **green** | 0 violations across 5 verifier-side files |
| no accepted unsafe item | **passed** | **0 false accepts on measurement** (v0 had 2; v1 had 3) |

## Verifier-Decision Reason Distribution (measurement)

| Reason | Count | Class |
| --- | --- | --- |
| `geometry_insufficient_coverage` | 573 | quarantine — new v2 geometry signal |
| `geometry_topology_ambiguous` | 368 | quarantine — new v2 topology check |
| `invariance_failed` | 327 | quarantine — `invariance_checks_v2` finally firing |
| `env_noise_exceeds_promise` | 252 | quarantine — falsifier-split noise tier |
| `env_dropout_exceeds_promise` | 252 | quarantine — falsifier-split dropout tier |
| `all_checks_pass` | 223 | **accept** |
| `sensor_health_v1_failed` | 64 | quarantine |
| `geometry_curvature_out_of_envelope` | 15 | quarantine — new v2 geometry signal |
| `margin_*_below_m_min_0.06` (multiple thresholds) | 208 | **reject** |

## Falsifier Disposition

- **Certificate Vacuity — not formally TRIGGERED at the 99 % match
  threshold**, but `sensor_health_v1` reports 98.26 % match (1.74 %
  delta). This is below the v1-inherited 2 % vacuity-delta gate. The
  honest disposition is identical in shape to v1's coverage_digest
  finding: `sensor_health_v1` is being subsumed by the stricter v2 gates
  that fire earlier (invariance_v2 quarantines 327 items, geometry_v2
  quarantines 956). v3 should remove `sensor_health_v1` from the
  certificate with this receipt cited as the named-removal reason, or
  reorder so sensor_health runs before the v2 geometry/invariance
  checks. The v1 coverage-redundancy pattern repeats one gate deeper.
- **Sufficiency Failure — not triggered.** 0 false accepts on
  verification (or any other split). v0 had 2, v1 had 3, v2 has 0.
- **Inversion / Spoofing Failure — not triggered.**
  - `A_inv_small`: split-mean AUROC + IoU joint criterion fails (high
    AUROC, near-zero IoU); 1 / 1536 per-env accidental hit is informational.
  - `A_spoof_field_small`: 0 / 513 unsafe items spoofed. Integrity
    binding from v1 holds.
  - `A_spoof_source_small`: 0 / 513 unsafe items spoofed. Source-bound
    spoof still cannot manufacture a registered trace commitment within
    the small-tier budget.
- **Certificate Integrity Repair — maintained and extended.** v1's
  4-probe battery (missing trace commitment, source-hash mismatch,
  derived-field hash mismatch, stale transform version) all quarantine.
  v2 added the `duplicate_trace_id` probe per slate §Integrity-Binding
  Gate row 2; it also quarantines. 5 / 5 integrity probes pass.
- **Verifier Overhead Failure — TRIGGERED, worse than v1.**
  `C_total_signature / C_rollout` = 1 535 (wall_ms), up from 1 139 in v1
  (which was already up from 228 in v0). The slate's cost-repair gate
  asked for ≥ 25 % wall-time improvement vs v1 (= ratio ≤ 854); observed
  is +34 % regression. Op-count ratios are unchanged (0.95 × rollout,
  5.55 × full-state). The v2 signature transform and verifier added more
  per-call work (geometry signal, invariance counterfactuals, integrity
  recompute) without applying the slate-admitted batching / single-pass /
  memoization options. Cost repair will roll into v3.
- **Boundary Absence — NOT TRIGGERED.** 0 / 768 out-of-promise
  falsifier-split items accepted; `accepted_oop_audit.csv` is empty.
  The new `geometry_promise_signal_v2` catches the v1 leak: 573
  `geometry_insufficient_coverage` + 368 `geometry_topology_ambiguous`
  + 15 `geometry_curvature_out_of_envelope` = 956 geometry-driven
  quarantines on out-of-promise envs. v1's 131 / 768 acceptance gap is
  closed.
- **Privilege Leak — not triggered.** Static-analysis audit verdict
  **green** across 5 verifier-side files. `hidden_state` appears 3 times
  inside the allowed redactor pattern; 8 other forbidden tokens appear
  0 times in verifier code.

## Coverage Disposition

Per slate §Coverage Disposition Gate, v2 elected the first allowed path:
**remove standalone `coverage_digest` from the certificate.** Coverage
information now appears only as `geometry_evidence_coverage` inside
`geometry_promise_signal_v2`. The ablation roster confirms removal —
`coverage_digest` is no longer present as a vacuity probe row.

This closes the v1 receipt's coverage-redundancy disposition finding.

## Verdict

**Named quarantine.**

The v2 slate's bounded-positive safety criteria are technically all
satisfied: integrity repair maintained, invariance repair passed,
boundary repair passed, coverage disposition passed, privilege audit
green, zero accepted unsafe items. By slate §v2 Verdict Rules a bounded
positive receipt would be filable on safety alone, with the cost
disposition stated separately.

This receipt elects **named quarantine** for three reasons:

1. **Cost repair failed against an explicit slate gate.** The slate's
   Cost Gate registered "≥ 25 % wall-time improvement vs v1" as the
   repair-passed criterion. Observed is a +34.7 % regression. Even
   though cost is not formally bounded-positive-blocking, claiming a
   bounded positive while the registered cost gate failed (and went
   the wrong direction across three slates) would understate the
   remaining work.
2. **`sensor_health_v1` fell below the v1-inherited 2 % vacuity-delta
   gate** (1.74 % delta). The slate's coverage-disposition language
   admits a "named removal" path; this receipt names sensor_health_v1
   for v3 removal-or-subsumption. Filing bounded positive without first
   disposing of a sub-gate field would carry forward a quiet
   redundancy.
3. **Acceptance rate is 9.7 % (223 / 2304)** — very conservative. v3
   should verify the geometry quarantine thresholds aren't over-firing
   on in-promise envs before promoting to a public bounded-positive
   claim.

Per slate, the receipt may say **the safety repair succeeded** — and
this receipt says exactly that, three times: integrity holds, boundary
closes, invariance fires. It does not claim cheap verification, and it
does not promote.

Capacity threshold: `not_estimated` (no small-tier attacker breach).

## Notes

### What v2 earned vs v1

- **Boundary repair landed clean.** 131 / 768 v1 acceptance gap on
  out-of-promise basin-shape items → 0 / 768 in v2. The new
  `geometry_promise_signal_v2` fields (`scale_interval`,
  `curvature_profile`, `topology_ambiguity_score`,
  `boundary_flags`) and the dedicated `accepted_oop_audit.csv` close
  the v1 receipt's "Boundary Absence — TRIGGERED" finding.
- **Invariance got teeth.** `invariance_checks_v2` produces 327
  measurement quarantines (`invariance_failed`) — up from 0 in v1.
  Ablation confirms 8.03 % decision delta, well above the slate's 2 %
  gate. v1's "Certificate Vacuity TRIGGERED for invariance_checks_v1
  (100 % match)" is closed.
- **Integrity stayed clean and grew one probe.** v1's 4-probe integrity
  battery extended with `duplicate_trace_id`; all 5 quarantine.
- **Coverage disposition done.** Standalone `coverage_digest` removed
  from the v2 certificate; subsumed as `geometry_evidence_coverage`
  inside `geometry_promise_signal_v2`. v1's coverage-redundancy
  disposition closed.
- **Zero false accepts.** v0 had 2, v1 had 3, v2 has 0. Adding stricter
  gates removed the noise-floor offenders from v1.

### What v2 did NOT earn

- A bounded-positive receipt. Cost regressed and `sensor_health_v1` fell
  below its inherited vacuity-delta gate.
- Cost-competitive verification. Wall-time ratio worsened from v1's
  1 139 × to v2's 1 535 × against rollout. None of the slate-admitted
  cost moves (batch signature recompute, single-pass checker, integer/
  fixed-grid geometry, memoized commitment lookup, shared intermediate
  buffers) were applied in v2.
- A demonstration that the verifier is not over-conservative. 81 % of
  measurement pairs quarantine. v3 should sanity-check whether the
  geometry thresholds quarantine in-promise envs that a domain expert
  would accept.

### Next allowed step

Open `docs/pvnp/PHASE1_V3_SLATE.md` targeting the residual gates:

1. **Cost repair.** Pick at least two of the slate's admitted options —
   batched signature recompute, single-pass checker over source
   observations, memoized trace-commitment lookup, fixed-grid geometry,
   shared intermediate buffers. Register a stricter pre-measurement
   wall-time threshold so a v3 cost-repair pass is unambiguous.
2. **`sensor_health_v1` disposition.** Either remove it from the v3
   certificate with this receipt cited as named-redundancy reason, or
   reorder check sequence so sensor health runs before geometry/
   invariance to restore load-bearing status. Either move closes the
   carry-forward finding from this receipt.
3. **Acceptance-volume sanity.** Either widen the in-promise acceptance
   by relaxing one geometry threshold pinned in v3 calibration (still
   from calibration-split data only), or report the conservative
   acceptance rate as a load-bearing safety property and move on.
4. **Inversion attacker — optional widening.** v0-through-v2 have shown
   AUROC consistently passes (≥ 0.83) but IoU consistently fails
   (≤ 0.01). v3 may either add an alternate parametric inversion target
   the slate names (per v0 spec language: "stricter registered basin-
   parameter target"), or accept that the current target is too hard for
   a 17 808-param attacker and freeze the disposition.

A v3 receipt should re-run the same harness on the v3 slate. A
bounded-positive receipt requires every safety gate to remain green
AND cost repair to land OR an explicit registered exemption.

### Domain-expansion note

Do NOT widen v2 (envs, splits, attacker budgets, policy class) to chase
a bounded-positive label here. v2 is frozen; further work belongs in v3
or a sibling slate with its own frozen lock and its own receipt.
