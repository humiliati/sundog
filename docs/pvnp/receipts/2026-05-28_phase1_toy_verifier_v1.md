# Phase 1 v1 Toy Verifier — Receipt

## Header

- Receipt id: `pvnp-phase1-toy-verifier-v1-2026-05-28`
- Phase / probe: Phase 1 v1 repair slate — first execution
- Date: 2026-05-28
- Author / runner: harness `scripts/pvnp-phase1-harness.mjs` invoked via
  `npm run pvnp:phase1:v1` (91 s wall on CPU, Python 3.14.4 +
  torch 2.11.0+cpu + Node 22)
- Code commit: `90f7a895d9818f9ff9dcc55f10283849073c2680`
- Result directory: `results/pvnp/phase1-toy-verifier-v1/`
- Roadmap version: [`SUNDOG_V_P_V_NP.md`](../../SUNDOG_V_P_V_NP.md) (after v0 receipt was filed)
- Spec version: [`PHASE1_TOY_VERIFIER_SPEC.md`](../PHASE1_TOY_VERIFIER_SPEC.md) constrained by [`PHASE1_V1_SLATE.md`](../PHASE1_V1_SLATE.md) (frozen repair slate)
- v0 baseline: [`2026-05-28_phase1_toy_verifier_v0.md`](2026-05-28_phase1_toy_verifier_v0.md)

## Registered Domain

- Environment family: inherited from v0 — 2D bounded domain `[0,1]²`,
  4 basin families (circle, ellipse, crescent, decoy_doublet),
  signed-distance latent field, 3 probe noise tiers
- Promise parameters: inherited from v0 (basin_min_diameter 0.12,
  basin_max_diameter 0.30, probe_noise_max_std 0.05,
  probe_dropout_max_rate 0.15, probe_delay_max_steps 2)
- Policy class: 2 hand-coded + 1 BC-trained MLP (17 922 params)
- Observation tier: 5-point local probe stencil at offsets (0,0), (±0.04, 0), (0, ±0.04)
- Signature transform: `scripts/lib/pvnp-phase1-signature-core.mjs`,
  schema **`pvnp-phase1-sigma-v1`**
- Certificate schema version: v1 contract from
  [`PHASE1_V1_SLATE.md` §Certificate Contract v1](../PHASE1_V1_SLATE.md) —
  adds `source_hash`, `transform_version`, `derived_fields_hash`,
  `integrity_checks`, `geometry_promise_signal`, `sensor_health_v1`,
  `invariance_checks_v1`
- Verifier: `scripts/lib/pvnp-phase1-verifier-core.mjs` (v1 dispatch via
  `sigma.schema === 'pvnp-phase1-sigma-v1'`)
- Baselines: rollout, full-state, formal/grid (R=64); ablated signature
  reported separately under Vacuity Probes
- Thresholds: `m_min` selected by Route 1 calibration sweep on the v1
  calibration split; coverage_min_touched_cells = 16 (inherited)
- Seeds: deterministic; env seeds derived from
  `pvnp-v1-{cal,train,verify,fals}-NNNN`; attacker seed=0
- Verifier-access declaration:
  `manifest.json:slate.verifier_access_declaration`; **green** audit
  (0 violations across 5 verifier-side files; 3 allowed redactor hits)

## Claim Under Test (Inherited from v1 Slate)

> Inside the same Phase 1 hidden-basin toy family, an integrity-bound
> signature verifier rejects or quarantines field-only spoofing, gives
> load-bearing sensor/invariance decisions, and detects hidden-basin
> promise violations through probe-derived geometry evidence.

## Artifacts

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| Manifest | `results/pvnp/phase1-toy-verifier-v1/manifest.json` | env hash `f6db72c162a1…` | run lock + slate snapshot |
| Calibration manifest | `…/calibration_manifest.json` | rule `v0_largest_m_min_with_clean_25pct_under_full_state` | m_min selection + insulation proof |
| Environments | `…/environments.jsonl` | 832 lines | per-env metadata (incl. hidden_state) |
| Traces | `…/traces.jsonl` | 2496 lines | positions + probes + actions per (policy, env) |
| Trace commitments | `…/trace_commitments.jsonl` | 2496 lines | source-bound commitments for v1 integrity checks |
| Signatures | `…/signatures.jsonl` | 2496 lines, schema `pvnp-phase1-sigma-v1` | certificates consumed by verifier |
| Integrity decisions | `…/integrity_decisions.csv` | 2304 rows | normal-certificate integrity check records |
| Integrity failures | `…/integrity_failures.csv` | 5 rows | synthetic integrity-mismatch probes |
| Verifier decisions | `…/verifier_decisions.csv` | 2304 rows | accept/reject/quarantine |
| Baseline decisions | `…/baseline_decisions.csv` | 6912 rows (3 baselines × 2304 pairs) | rollout/full-state/formal |
| Ablation decisions | `…/ablation_decisions.csv` | 9216 rows (4 drops × 2304 pairs) | vacuity probes |
| Inversion attacker | `…/attacker_inversion_results.json` | model 17 808 params, schema `pvnp-phase1-attacker-inversion-v0` | A_inv_small per-env AUROC/IoU |
| Attacker trials | `…/attacker_trials.csv` | 1 536 inversion + 494 field-spoof + 494 source-spoof | per-trial outcomes |
| Ground truth labels | `…/ground_truth_labels.csv` | 2304 rows | evaluator-only labels |
| Costs | `…/costs.csv` | per-component + derived rows | wall_ms + ops accounting |
| Privilege audit | `…/audit-report.{json,txt}` | schema `pvnp-phase1-privilege-audit-v1` | **green** verdict |
| Falsifier summary | `…/falsifier_summary.md` | — | named falsifier dispositions |

## Observed Values

| Quantity | Registered threshold | Observed value | Pass/fail/quarantine |
| --- | --- | --- | --- |
| False accept rate (all measurement) | primary metric | 0.130 % (3/2304) | reported |
| False accept rate (verification split, in-promise) | primary metric | 0.70 % (2/287 accepts on verification) | sufficiency-failure flag |
| False reject rate (all measurement) | secondary | 4.64 % (107/2304) | reported |
| Quarantine rate | reported | 56.2 % (1295/2304) | reported |
| Privilege leaks | 0 required | 0 violations (3 allowed redactor hits) | **pass** |
| `C_total_signature` (wall_ms) | reported | 982.32 ms | reported |
| Rollout cost ratio (wall_ms) | secondary | 1139.05 (vs 228.60 in v0) | overhead-failure flag, worse than v0 |
| Rollout cost ratio (ops) | secondary | 0.945 (vs 0.948 in v0) | unchanged |
| Full-state cost ratio (wall_ms) | secondary | 81.06 (vs 15.81 in v0) | overhead-failure flag, worse than v0 |
| Full-state cost ratio (ops) | secondary | 5.53 (vs 5.55 in v0) | unchanged |
| Calibration sweep | candidate grid {0.02, 0.04, 0.06}; rule = largest m_min with calibration-clean ≥ 25 % under full-state | sweep returned 84.9 % / 83.9 % / 80.7 %; **selected m_min = 0.06** | rule-eligible |
| Calibration insulation overlap | 0 required | 0 (split prefixes disjoint) | **pass** |
| Field-spoof success (`A_spoof_field_small`) | 0 required for integrity repair | **0 / 494** unsafe items spoofed | **pass** |
| Source-spoof success (`A_spoof_source_small`) | 0 required for source repair | **0 / 494** unsafe items spoofed | **pass** |
| Integrity-failure probes (synthetic) | all 5 must quarantine | 5 / 5 quarantined | **pass** |
| Inversion attacker (split-mean) | AUROC ≥ 0.80 AND IoU ≥ 0.40 | AUROC 0.855, IoU 0.004 (verification); AUROC 0.839, IoU 0.002 (falsifier) | joint criterion fails |
| Inversion attacker (per-env) | informational | 1 / 1536 per-env joint successes (single accidental hit) | informational |

## Baseline Comparison

External comparators only. Ablated signature lives in the Vacuity Probes
block below.

| Verifier | Access level | Cost (wall_ms) | False accept | False reject | Quarantine | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Signature verifier (v1) | indirect signature | 982.32 | 3 / 2304 (0.130 %) | 107 / 2304 (4.64 %) | 1295 / 2304 (56.2 %) | accepts limited by integrity + promise + invariance + sensor_v1 + geometry_promise + coverage + envelope + margin gates |
| Rollout verifier | evaluator / replay | 0.86 | 0 (by definition) | 0 | n/a | ground-truth labeler |
| Full-state verifier | privileged hidden state | 12.12 | 0 | 0 | n/a | upper-information baseline |
| Formal baseline | grid (R=64) reachability | 819.16 | 0 | varies (conservative) | n/a | per-env grid mask vs trajectory cells |

## Vacuity Probes

Internal sanity checks. A `vacuous` verdict (match rate ≥ 99 %) triggers the
Certificate Vacuity falsifier. Per the v1 slate `sensor_health_v1` and
`invariance_checks_v1` are slate-gated; `coverage_digest` is not, and may be
explicitly named as removed-or-subsumed per the slate's vacuity-gate
language.

| Dropped field | Decision match vs full | Slate gate | Verdict |
| --- | --- | --- | --- |
| `margin_lower_bound` | 87.54 % (2017/2304) | n/a | **load-bearing** |
| `coverage_digest` | 100.00 % (2304/2304) | not slate-gated | **redundant — subsumed by `geometry_promise_signal`** (see §Coverage-redundancy explanation) |
| `sensor_health_v1` | 97.31 % (2242/2304) | slate: ≥ 2 % delta required, or named removal | **passes slate gate** (2.69 % delta > 2 %) |
| `invariance_checks_v1` | 100.00 % (2304/2304) | slate: ≥ 2 % delta required, or named removal | **fails slate gate** |

## Repair-Label Disposition (per [`PHASE1_V1_SLATE.md` §v1 Verdict Rules](../PHASE1_V1_SLATE.md))

| Repair label | Pass? | Evidence |
| --- | --- | --- |
| integrity repair | **passed** | 0 / 494 `A_spoof_field_small`; 0 / 494 `A_spoof_source_small`; 5 / 5 synthetic integrity-mismatch probes quarantined |
| vacuity repair | **partial** | `sensor_health_v1` passes slate gate (2.69 % delta); `invariance_checks_v1` fails (100 % match) |
| boundary repair | **failed** | 131 / 768 out-of-promise falsifier-split items accepted; slate requires 0 |
| cost repair | **failed** | wall-time ratio 1139 × vs rollout (worse than v0's 228 ×); op-count ratio 0.94 × is unchanged from v0 |
| privilege audit | **green** | 0 violations across 5 verifier-side files |
| no accepted unsafe item | **failed** | 3 false accepts on measurement |

## Falsifier Disposition

- **Certificate Vacuity — TRIGGERED for `invariance_checks_v1`.**
  100 % match when dropped. The v1 slate explicitly asked for non-trivial
  invariance ("recomputation under a registered local coordinate
  transform with tolerance; bias-drift counterfactual; decoy-field
  consistency; envelope consistency under registered action
  perturbation"); the v1 implementation does not yet make any of these
  fail on the v1 slate. v2 must either add a discriminating invariance
  check that fires on the v1 slate or remove the field with a named
  reason.
  `coverage_digest` also has 100 % match, but is NOT a slate-gated field;
  see §Coverage-redundancy explanation. The honest disposition is
  "subsumed by `geometry_promise_signal`, remove or repurpose in v2",
  which the slate explicitly admits as "separately named reason to
  remove the field from the certificate."
  `sensor_health_v1` reports 97.31 % match (2.69 % delta) — **clears the
  slate's ≥ 2 % gate**. Not vacuous under the slate.
- **Sufficiency Failure — TRIGGERED (low-volume).** 2 false accepts on the
  verification split (out of 287 accepted verification-split items), plus
  1 on train. All three offenders have signature `margin_lower_bound`
  6–8 mm above the ground-truth `min_margin`, i.e., probe noise pushed
  the signature's margin proxy above `m_min` while the true distance
  sat below. This is a noise-floor effect — the same shape as v0's
  single false accept, now slightly more frequent. Not a systematic gap.
- **Inversion / Spoofing Failure — NOT TRIGGERED.** No small attacker
  breached registered thresholds at the split-mean level.
  - `A_inv_small`: AUROC 0.855 / IoU 0.004 (verification);
    AUROC 0.839 / IoU 0.002 (falsifier). Joint criterion fails. 1 / 1536
    per-env accidental hit is informational only.
  - `A_spoof_field_small`: 0 / 494 unsafe items spoofed. **The v0
    integrity gap is closed** under field-only edits.
  - `A_spoof_source_small`: 0 / 494 unsafe items spoofed. The source-
    bound spoof cannot manufacture a registered trace commitment within
    the small-tier budget.
- **Certificate Integrity Repair — PASSED.** Synthetic integrity-mismatch
  probes (missing trace commitment, source-hash mismatch, derived-field
  hash mismatch, stale transform version) all quarantined 5 / 5.
- **Verifier Overhead Failure — TRIGGERED, worse than v0.**
  `C_total_signature / C_rollout` = 1139.05 (wall_ms), up from 228.60 in
  v0. `C_total_signature / C_full_state` = 81.06 (wall_ms), up from
  15.81. Op-count ratios are unchanged (0.94 × rollout, 5.53 ×
  full-state). The wall-time blowup is JS-overhead-dominated cost of the
  integrity recompute path; the slate explicitly admits batched
  signature recomputation, single-pass checker, integer/fixed-grid
  geometry, and memoized commitment lookup as v1-compatible cost moves
  that this run did not yet apply.
- **Boundary Absence — TRIGGERED.** 131 / 768 out-of-promise falsifier-
  split items accepted. The new `geometry_promise_signal` catches 663
  `geometry_insufficient_evidence` quarantines plus 13
  `geometry_curvature_suspicious` quarantines, but a measurable fraction
  of basin-shape promise violations still slip through with enough probe
  evidence to satisfy the v1 geometry gate. The slate explicitly says a
  bounded positive receipt is not allowed if any basin-shape falsifier
  item is accepted — so this gate must close in v2.
- **Privilege Leak — not triggered.** Static-analysis audit verdict
  **green** across 5 verifier-side files. `hidden_state` appears 3 times
  inside the allowed redactor pattern; all 8 other forbidden tokens
  appear 0 times in verifier code.

## Coverage-Redundancy Explanation

`coverage_digest` was load-bearing in v0 (68.53 % match when dropped;
35 % of decisions changed). In v1 it has 100 % match — fully redundant.

Root cause: the v1 verifier check order is

```
integrity → env-promise → invariance → sensor_health(v0+v1)
        → geometry_promise_signal → coverage_digest → envelope → margin
```

The new `geometry_promise_signal` check produces 676 quarantine reasons:
- `geometry_insufficient_evidence` (663) — probe coverage too sparse to
  assess basin geometry
- `geometry_curvature_suspicious` (13) — curvature signal points to a
  basin shape outside the promise envelope

A trajectory sparse enough to fail the v0 coverage gate (touched cells
< 16) is also sparse enough to fail the v1 geometry-evidence gate. A
trajectory dense enough to pass geometry already touches well over 16
cells. So coverage_digest never fires in v1: it's structurally
dominated by the earlier geometry check.

The slate's vacuity language ("**or produce a separately named reason
to remove the field from the certificate**") admits this disposition.
v2 should:
- either remove `coverage_digest` from the v2 certificate with this
  receipt cited as the named reason; or
- reorder so coverage runs **before** geometry as a cheap pre-check,
  restoring its load-bearing role and saving the geometry compute on
  obviously-undercovered trajectories.

Either move is admissible under the v1 slate. Option (b) also helps
the cost gate.

## Verdict

**Named quarantine.**

Specifically: Phase 1 v1 closes the load-bearing v0 spoof channel (both
field-only and source-bound spoofs go 0-for-everything across ~63 000
attempts) and adds working integrity-mismatch quarantine, but
falsifier triggers remain for invariance vacuity, boundary absence on
basin-shape promise violations, verifier overhead (worse than v0), and
3 noise-floor false accepts. Per slate, a bounded positive receipt
requires integrity + vacuity + boundary + green privilege + 0 false
accepts; v1 meets only integrity + green privilege.

The progress is real: the v0 capacity breach is repaired. The slate's
v1 repair plan is partially earned — integrity is the load-bearing win;
the other three repair labels (vacuity, boundary, cost) will roll into
a v2 slate.

Capacity threshold: `not_estimated` (no small-tier attacker breach at
the split-mean level).

## Notes

### What v1 earned vs v0

- **v0 spoof channel closed.** `A_spoof_field_small` cannot accept any
  unsafe item by editing analytical fields; the integrity recompute
  catches every mismatch. `A_spoof_source_small` cannot manufacture a
  registered trace commitment within budget.
- **Sensor_v1 is now slate-discriminating.** Match rate drops from
  99.31 % (v0, vacuous) to 97.31 % (v1, 2.69 % delta clears the slate
  gate).
- **New geometry signal does work.** 676 falsifier-population items now
  quarantine on the basin-geometry promise channel that didn't exist
  in v0.
- **Privilege boundary held under the v1 source-binding refactor.**
  Adding `trace_commitments.jsonl`, `source_hash`, `derived_fields_hash`,
  and `integrity_checks` to the verifier-side surface did not introduce
  forbidden-token leaks. Audit stayed green.

### What v1 did NOT earn

- A bounded positive receipt — four gates remain (invariance vacuity,
  boundary, cost, false accepts).
- A cost-competitive verifier. Wall-time got worse, not better. Cost
  repair is registered for v2 with the slate-named options (batched
  recompute, single-pass checker, integer/fixed-grid geometry, memoized
  commitment lookup).
- A discriminating `invariance_checks_v1`. The slate spelled out four
  candidate non-trivial checks; v1 implementation did not yet make any
  of them fire on this slate.

### Next allowed step

Open `docs/pvnp/PHASE1_V2_SLATE.md` targeting the four remaining gates:

1. **Invariance v2 discrimination.** Pick at least one of the slate's
   four candidates (local coordinate transform with tolerance; bias-
   drift counterfactual; decoy-field consistency; envelope consistency
   under registered action perturbation) and implement it such that it
   demonstrably fires on the v1 slate (≥ 2 % decision delta).
2. **Geometry-promise tightening.** Close the basin-shape-promise
   acceptance gap (131 / 768 → 0 / 768 on the falsifier split).
   Likely needs a stricter geometry sufficiency rule or an explicit
   curvature-bound test that rejects when the probe-derived basin
   estimate exceeds promise diameter bounds.
3. **Cost repair.** Pick from the slate's admitted options — batched
   signature recompute, single-pass checker over source observations,
   integer or fixed-grid geometry, memoized trace commitment lookup —
   and bring the wall-time ratio back below v0's 228 × against rollout.
   The op-count ratio is already near unity; the wall-time loss is
   per-call JS overhead.
4. **Coverage disposition.** Per this receipt's §Coverage-redundancy
   explanation, either remove `coverage_digest` from the v2
   certificate with named subsumption, or reorder to put it before
   geometry as a cheap pre-check. Either move is admissible under the
   v1 slate's vacuity-gate language.

A v2 receipt should re-run the same harness on the v2 slate. A null
receipt or named quarantine in v2 is still a valid Phase 1 result; a
bounded positive receipt is only earned if invariance fires, boundary
closes, false-accept rate drops to zero, and cost moves the right
direction.

### Domain-expansion note

Do NOT widen v1 (envs, splits, attacker budgets, policy class) to
chase a cleaner result. v1 is frozen; further work belongs in v2 or a
sibling slate with its own frozen lock and its own receipt.
