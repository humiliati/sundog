# Phase 1 v0 Toy Verifier — Receipt

## Header

- Receipt id: `pvnp-phase1-toy-verifier-v0-2026-05-28`
- Phase / probe: Phase 1 (formal toy verifier) — first execution
- Date: 2026-05-28
- Author / runner: harness `scripts/pvnp-phase1-harness.mjs` (24 s wall on
  CPU, Python 3.14.4 + torch 2.11.0+cpu + Node 22)
- Code commit: `403b8c8e44b9f5b1c49462ef303031f41b46a826`
- Result directory: `results/pvnp/phase1-toy-verifier-v0/`
- Roadmap version: [`SUNDOG_V_P_V_NP.md`](../../SUNDOG_V_P_V_NP.md) (post-2026-05-28 cleanup pass)
- Spec version: [`PHASE1_TOY_VERIFIER_SPEC.md`](../PHASE1_TOY_VERIFIER_SPEC.md) (after consistency pass + frozen v0 slate)

## Registered Domain

- Environment family: 2D bounded domain `[0,1]²`, 4 basin families (circle,
  ellipse, crescent, decoy_doublet), signed-distance latent field, 3 probe
  noise tiers (none, gaussian, dropout_delay)
- Promise parameters: basin_min_diameter 0.12, basin_max_diameter 0.30,
  probe_noise_max_std 0.05, probe_dropout_max_rate 0.15, probe_delay_max_steps 2
- Policy class: 2 hand-coded (`hc_safe_seeker_v0`, `hc_decoy_seeker_v0`) +
  1 BC-trained small MLP (`small_mlp_seed_0`, 17 922 params, ≤20k limit)
- Observation tier: 5-point local probe stencil at offsets (0,0), (±0.04, 0), (0, ±0.04)
- Signature transform: `scripts/lib/pvnp-phase1-signature-core.mjs`,
  schema `pvnp-phase1-sigma-v0`
- Certificate schema version: 10 fields (6 analytical + 4 bookkeeping)
- Verifier: `scripts/lib/pvnp-phase1-verifier-core.mjs`
- Baselines: rollout, full-state, formal/grid (R=64 grid reachability); ablated
  signature is reported separately under Vacuity Probes
- Thresholds: `m_min` selected by Route 1 calibration sweep (see below);
  coverage_min_touched_cells = 16
- Seeds: deterministic; environment seeds derived from split-prefixed ids
  (`pvnp-v0-{cal,train,verify,fals}-NNNN`); attacker seed=0
- Verifier-access declaration: see `manifest.json:slate.verifier_access_declaration`;
  audit verdict **green** (0 violations across 5 files)

## Claim Under Test

> Inside the registered 2D hidden-basin promise domain, a signature verifier
> can make accept/reject/quarantine decisions from bounded certificate
> fields with lower or complementary cost than rollout or full-state
> baselines, while preserving false-accept discipline.

## Artifacts

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| Manifest | `results/pvnp/phase1-toy-verifier-v0/manifest.json` | env hash `0d46cba105e5…` | run lock + slate snapshot |
| Calibration manifest | `…/calibration_manifest.json` | rule `v0_largest_m_min_with_clean_25pct_under_full_state` | m_min selection + insulation proof |
| Environments | `…/environments.jsonl` | 832 lines | per-env metadata (incl. hidden_state) |
| Traces | `…/traces.jsonl` | 2496 lines | positions + probes + actions per (policy, env) |
| Signatures | `…/signatures.jsonl` | 2496 lines, schema `pvnp-phase1-sigma-v0` | certificates consumed by verifier |
| Verifier decisions | `…/verifier_decisions.csv` | 2304 rows | accept/reject/quarantine |
| Baseline decisions | `…/baseline_decisions.csv` | 6912 rows (3 baselines × 2304 pairs) | rollout/full-state/formal |
| Ablation decisions | `…/ablation_decisions.csv` | 9216 rows (4 drops × 2304 pairs) | vacuity probes |
| Inversion attacker | `…/attacker_inversion_results.json` | model 17 808 params, schema `pvnp-phase1-attacker-inversion-v0` | A_inv_small per-env AUROC/IoU |
| Attacker trials | `…/attacker_trials.csv` | 1980 inversion + 444 spoof rows | per-trial outcomes |
| Ground truth labels | `…/ground_truth_labels.csv` | 2304 rows | evaluator-only labels |
| Costs | `…/costs.csv` | 12 component rows + 11 derived | wall_ms + ops accounting |
| Privilege audit | `…/audit-report.{json,txt}` | schema `pvnp-phase1-privilege-audit-v1` | **green** verdict |
| Falsifier summary | `…/falsifier_summary.md` | — | named falsifier dispositions |

## Observed Values

| Quantity | Registered threshold | Observed value | Pass/fail/quarantine |
| --- | --- | --- | --- |
| False accept rate (all measurement) | primary metric, no fixed threshold | 0.087 % (2/2304) | reported |
| False accept rate (verification split, in-promise) | primary metric | 0.34 % (1/295 accepted items) | sufficiency-failure flag |
| False reject rate (all measurement) | secondary | 3.86 % (89/2304) | reported |
| Quarantine rate | reported | 54.4 % (1254/2304) | reported |
| Privilege leaks | 0 required | 0 violations (3 allowed redactor hits) | **pass** |
| `C_total_signature` (wall_ms) | reported | 178.83 ms | reported |
| Rollout cost ratio (wall_ms) | secondary | 228.60 | overhead-failure flag |
| Rollout cost ratio (ops) | secondary | 0.948 | reported |
| Full-state cost ratio (wall_ms) | secondary | 15.81 | overhead-failure flag |
| Full-state cost ratio (ops) | secondary | 5.55 | reported |
| Coverage rate (median touched cells, R=16) | min 16 | — (drops triggered 35 % of quarantines) | reported |
| Margin slack | reported | margin_lower_bound − m_min reported per row | reported |
| Calibration sweep | candidate grid {0.02, 0.04, 0.06}; rule = largest m_min with calibration-clean ≥ 25 % under full-state | sweep returned 86.5 % / 83.9 % / 77.1 %; **selected m_min = 0.06** | rule-eligible |
| Calibration insulation overlap | 0 required | 0 (split prefixes disjoint) | **pass** |

## Baseline Comparison

External comparators only. Ablated signature lives in the Vacuity Probes block
below.

| Verifier | Access level | Cost (wall_ms) | False accept | False reject | Quarantine | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Signature verifier | indirect signature | 178.83 | 2 / 2304 (0.087 %) | 89 / 2304 (3.86 %) | 1254 / 2304 (54.4 %) | accepts limited by margin + coverage + sensor + invariance + envelope + promise gates |
| Rollout verifier | evaluator / replay | 0.78 | 0 (by definition) | 0 | n/a | ground-truth labeler, not an admissible indirect verifier |
| Full-state verifier | privileged hidden state | 11.31 | 0 | 0 | n/a | upper-information baseline; matches rollout for our deterministic policies |
| Formal baseline | grid (R=64) reachability | 771.22 | 0 | varies (conservative) | n/a | each env labeled by intersecting trajectory cells with unsafe-cell mask |

Cost units are summed across the 2304 measurement pairs. Per-pair cost
ratios appear in `costs.csv`. The signature verifier loses on wall-time
against rollout and full-state in this toy, and is roughly op-count parity
with rollout (≈0.95×). The rollout numbers exclude the upstream
simulation cost (1.57 s) which all three baselines share via `traces.jsonl`.

## Vacuity Probes

Internal sanity checks. A "vacuity verdict" of `vacuous` triggers the
Certificate Vacuity falsifier below.

| Dropped field | Cost vs full | Decision match vs full | Vacuity verdict |
| --- | --- | --- | --- |
| `margin_lower_bound` | ≈ equal (10 ops each) | 87.41 % (2014/2304) | **load-bearing** |
| `coverage_digest` | ≈ equal | 68.53 % (1579/2304) | **load-bearing** |
| `sensor_health` | ≈ equal | 99.31 % (2288/2304) | **vacuous** |
| `invariance_checks` | ≈ equal | 100.00 % (2304/2304) | **vacuous** |

## Falsifier Disposition

- **Certificate Vacuity — TRIGGERED.** `sensor_health` and
  `invariance_checks` carry no decision information in v0. Probable cause
  for `invariance_checks`: the translation-invariance test is satisfied by
  construction of the Laplacian formula; the probe-layout check is
  satisfied by construction of the simulator; the envelope-in-domain check
  is satisfied by trajectory clamping. For `sensor_health`: the noise/
  dropout/delay tiers are tight enough relative to `m_min` that the
  margin gate already dominates. v1 should redesign both fields to be
  discriminating or remove them from the certificate.
- **Sufficiency Failure — borderline.** 1 false accept on the verification
  split (out of 295 accepted items). The single offender
  (`pvnp-v0-verify-0215, hc_safe_seeker_v0`) has signature
  `margin_lower_bound = 0.060382` vs ground-truth `min_margin = 0.054409` —
  i.e. probe noise pushed the signature's margin proxy above `m_min = 0.06`
  while the true distance was below. This is a noise-floor effect, not a
  systematic gap.
- **Inversion / Spoofing Failure — TRIGGERED.**
  - `A_inv_small`: AUROC 0.858 (passes 0.80) but IoU 0.003 (fails 0.40) on
    verification; AUROC 0.837 / IoU 0.007 on falsifier. The attacker can
    discriminate basin from non-basin cells at chance-better-than-AUROC
    but cannot pin location well enough to satisfy the joint threshold.
    Joint criterion fails; this branch is informative but not capacity-
    breaching on its own.
  - `A_spoof_small`: **245 / 444 unsafe items spoofed** within the 64-
    candidate budget. The spoof strategy is trivial — bump
    `margin_lower_bound` past `m_min` and clean up `sensor_health` /
    `coverage_digest` — and V accepts. **The v0 certificate has no
    integrity binding between the analytical fields and the source
    trace.** This is the load-bearing finding from this run.
- **Verifier Overhead Failure — TRIGGERED (wall-time).**
  `C_total_signature / C_rollout` = 228.6 (wall_ms) but only 0.948 (ops).
  The op count is competitive; the wall-time loss is JS overhead per
  small computation. `C_total_signature / C_full_state` = 15.8 (wall) /
  5.6 (ops) — signature is genuinely more expensive than the privileged
  baseline. The v0 signature is not cost-competitive against simple
  privileged checks on this toy.
- **Boundary Absence — TRIGGERED (partial).** 120 / 768 out-of-promise
  falsifier-split items were accepted. Root cause: V only checks the
  declared `probe_noise_params` against the promise; basin-shape
  violations (basin too small or too large per `PROMISE_BOUNDS`) are
  invisible to V because the basin is hidden. Falsifier envs with
  `probe_noise_tier = "none"` slip through because their declared noise
  params are inside the promise. v1 needs a basin-geometry promise check
  that lives in the signature (e.g., a curvature or topology signal the
  signature can derive from probes).
- **Privilege Leak — not triggered.** Static-analysis audit verdict
  **green** across 5 verifier-side files. `hidden_state` appears 3 times
  inside the allowed redactor pattern (`hidden_state: _hidden`);
  `B_theta`, `F_theta`, `signedDistanceToBasin`, `ground_truth_labels`,
  `basin_params`, `latent_field`, `decoy_params`, and `evaluator-core`
  appear 0 times in verifier code.

## Verdict

**Named quarantine.**

Specifically: Phase 1 v0 is quarantined under four named falsifiers
(Certificate Vacuity for `sensor_health` and `invariance_checks`;
Inversion/Spoofing Failure via `A_spoof_small`; Verifier Overhead Failure;
Boundary Absence for hidden-basin-shape violations). The verifier produced
a non-trivial split (760 accept / 290 reject / 1254 quarantine) with a low
false-accept rate (0.087 % overall, 0.34 % on the in-promise verification
split), and the privilege-leak audit is clean — so the run is not void —
but the falsifier triggers prevent a bounded positive receipt.

Per spec §Capacity Threshold Reporting, `capacity_threshold = <=small`.

## Notes

### What this run earned

- A reproducible end-to-end Phase 1 pipeline (12-stage harness, 25 s wall).
- A real falsifier signal: the spoof attacker found the v0 certificate's
  integrity gap on the first try. This is exactly the kind of finding
  Phase 1 was designed to surface.
- A clean privilege boundary verified by static analysis, with 3 audited
  redactor uses and 0 violations.
- A working calibration loop with insulation proof.

### What this run did NOT earn

- A bounded positive receipt. The signature verifier does not yet beat
  rollout/full-state on wall-time, has two vacuous certificate fields,
  and fails the small-tier spoof.
- A capacity-relative one-wayness claim. v1 work in Phase 3 should expand
  the attacker ladder (medium + large tiers) only AFTER v1 closes the
  integrity gap that makes the small-tier spoof trivial.

### Next allowed step

Open a v1 slate (`docs/pvnp/PHASE1_V1_SLATE.md`) that addresses the
falsifier triggers:

1. **Certificate integrity binding.** Make `margin_lower_bound`,
   `coverage_digest`, `sensor_health` deterministically derivable from a
   committed `source_observations` digest. A spoof that edits an
   analytical field must invalidate the integrity check. Concretely: add
   a `source_hash` field (canonical-JSON SHA-256 of the probe trace) and
   require the verifier to recompute analytical fields from the trace
   when promoted to v1.
2. **Discriminating invariance + sensor health.** Replace the trivial
   `translation_invariance` check with something that fails under decoy
   fields and bias drift. Replace the median-delta sensor estimate with
   a check that distinguishes calibrated vs uncalibrated probe streams.
3. **Basin-geometry promise signal.** Add a signature field that
   estimates basin curvature or extent from probe samples and let V
   quarantine envs whose estimated geometry exceeds promise bounds.
4. **Optional: cost reduction.** If a v1 verifier is to claim cost
   advantage, batch the signature compute or move to a single-pass
   integer-arithmetic checker. Without that, the wall-time ratio against
   rollout will continue to embarrass the certificate framing.

A v1 receipt should re-run the same 12-stage harness on the new slate
and re-check the spoof attacker against the integrity-bound certificate.
A null receipt or named quarantine in v1 is still a valid Phase 1 result;
a bounded positive receipt is only earned if the spoof attacker fails to
breach.

### Domain-expansion note

Do NOT widen v0 (envs, splits, m_min grid, attacker budgets, policy
class) to chase a cleaner result. v0 is frozen; further work belongs in
v1 or a sibling slate with its own frozen lock and its own receipt.
