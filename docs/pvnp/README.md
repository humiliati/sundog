# P-vs-NP Verification Project Index

This folder holds working artifacts for the `SUNDOG_V_P_V_NP` roadmap.

Main roadmap:

- [`../SUNDOG_V_P_V_NP.md`](../SUNDOG_V_P_V_NP.md)

Lit-pass:

- [`../P_V_NP_LITPASS_MEMO.md`](../P_V_NP_LITPASS_MEMO.md) - 2026-05-28
  prior-art and citation spine for promise-bounded verifier scaffolding.

Phase specs:

- [`PHASE1_TOY_VERIFIER_SPEC.md`](PHASE1_TOY_VERIFIER_SPEC.md) - draft spec
  for the first formal toy verifier, including promise domain, certificate
  schema, baselines, cost accounting, and falsifier mapping.
- [`PHASE1_V0_SLATE.md`](PHASE1_V0_SLATE.md) - frozen first implementation
  slate with split namespaces, calibration insulation, attacker budgets, and
  success thresholds.
- [`PHASE1_V1_SLATE.md`](PHASE1_V1_SLATE.md) - repair slate opened after the
  v0 named quarantine, focused on certificate integrity binding, non-vacuous
  sensor/invariance checks, and probe-derived geometry promise detection.
- [`PHASE1_V2_SLATE.md`](PHASE1_V2_SLATE.md) - repair slate opened after the
  v1 named quarantine, focused on decision-relevant invariance checks,
  basin-shape boundary closure, coverage disposition, and checker overhead.
- [`PHASE1_V3_SLATE.md`](PHASE1_V3_SLATE.md) - repair slate opened after the
  v2 safety-repair-landed quarantine, focused on cost batching,
  `sensor_health_v1` disposition, acceptance-volume sanity, and optional
  inversion-target widening.
- [`PHASE1_V4_SLATE.md`](PHASE1_V4_SLATE.md) - cost-gate slate opened after
  the v3 cost-only quarantine, focused on stable cost comparators,
  cache-eligible reuse accounting, and bounded-positive promotion criteria.
- [`PHASE1_V5_SLATE.md`](PHASE1_V5_SLATE.md) - cost-closure slate opened
  after the v4 cost-only quarantine, focused on short-circuit instrumentation
  overhead and median-of-3 cost promotion.
- [`PHASE1_V6_SLATE.md`](PHASE1_V6_SLATE.md) - op-count cost slate opened
  after the corrected v5 named quarantine, focused on using the only
  reproducible cost signal while keeping wall-time diagnostic-only.
- [`PHASE2_MESA_BRIDGE.md`](PHASE2_MESA_BRIDGE.md) - Phase 2 mesa
  verification bridge spec / charter (opened 2026-05-31). Carries forward
  the v6 claim boundary and maps mesa intervention evidence to verifier-failure
  tests. The frozen v0 slate below supersedes its initial "no slate frozen"
  status.
- [`PHASE2_MESA_BRIDGE_V0_SLATE.md`](PHASE2_MESA_BRIDGE_V0_SLATE.md) -
  frozen first mesa-bridge execution slate. Re-reads the current mesa Phase 4
  and Phase 5 artifacts from disk, gates only a reward-blind bridge-smoke, and
  uses the measured L-Mixed breach thresholds as the capacity-breach falsifier.
  Certificates must recompute from per-seed raw trial logs, and the op-count
  comparator must be a same-artifact-tier raw-trace audit rather than a full
  mesa battery regeneration.
- [`PHASE2_MESA_BRIDGE_V1_SLATE.md`](PHASE2_MESA_BRIDGE_V1_SLATE.md) -
  frozen provenance-repair slate after the v0 named quarantine. Chooses
  raw-logged Small reruns over Medium-only downscope.
- [`PHASE3_CAPACITY_ONE_WAYNESS_V0_SLATE.md`](PHASE3_CAPACITY_ONE_WAYNESS_V0_SLATE.md) -
  frozen capacity-relative one-wayness slate after the Phase 2 v1 bounded
  positive. Registers the attacker/capacity battery and source-bound
  seed-extension plan before any execution.
- [`PHASE3_CAPACITY_ONE_WAYNESS_V1_SLATE.md`](PHASE3_CAPACITY_ONE_WAYNESS_V1_SLATE.md) -
  frozen block-consensus repair slate after the v0 falsified registered cell.
  Targets source-bound seed-block mean drift without changing the base response
  thresholds.
- [`PHASE3_CAPACITY_ONE_WAYNESS_V2_SLATE.md`](PHASE3_CAPACITY_ONE_WAYNESS_V2_SLATE.md) -
  opened disclosure-consensus repair slate after the v1 named quarantine.
  Targets block-unstable objective-conflict disclosure while keeping v1
  promotion consensus unchanged.

Templates:

- [`RECEIPT_TEMPLATE.md`](RECEIPT_TEMPLATE.md)

Result convention:

- Computational outputs should write under `results/pvnp/<phase-or-probe>/`.
- `results/pvnp/` is treated as transient run output. Durable conclusions
  belong in dated receipt notes under `docs/pvnp/receipts/` once reviewed.
- A run without a manifest, verifier-access declaration, baseline comparison,
  and falsifier disposition is not a receipt.

Receipts:

- [`receipts/README.md`](receipts/README.md) - receipt index, including the
  Phase 1 v0-v5 named-quarantine receipts, the v6 op-count positive receipt,
  the Phase 2 v0 mesa-bridge named quarantine, the Phase 2 v1 mesa-bridge
  bounded-positive receipt, the Phase 3 v0 falsified-cell receipt, and the
  Phase 3 v1 named-quarantine receipt.

Current state:

- Roadmap opened: 2026-05-28.
- Lit-pass filed: 2026-05-28.
- Project scaffold opened: 2026-05-28.
- Phase 1 toy-verifier spec drafted: 2026-05-28.
- Phase 1 v0 slate frozen: 2026-05-28.
- Phase 1 v0 harness executed: 2026-05-28. Verdict = named quarantine;
  `capacity_threshold = <=small` after field-only spoof breach.
- Phase 1 v1 repair slate opened: 2026-05-28.
- Phase 1 v1 harness executed: 2026-05-28. Verdict = named quarantine;
  v0 spoof channel closed, but invariance vacuity, boundary leak, and cost
  overhead remain open.
- Phase 1 v2 repair slate opened: 2026-05-28.
- Phase 1 v2 harness executed: 2026-05-28. Verdict = named quarantine with
  safety repair landed; false-accept rate 0%, `capacity_threshold =
  not_estimated`, zero accepted basin-shape out-of-promise rows, but cost
  overhead, `sensor_health_v1` disposition, and acceptance-volume sanity
  remain open.
- Phase 1 v3 repair slate opened: 2026-05-28.
- Phase 1 v3 harness executed: 2026-05-28. Verdict = named quarantine on
  cost alone; all safety-repair labels pass, absolute signature+verify wall
  time improves to 907.52 ms, but the rollout-ratio and raw cache-hit gates
  remain unsuitable for bounded-positive promotion.
- Phase 1 v4 cost-gate slate opened: 2026-05-28.
- Phase 1 v4 harness executed: 2026-05-28. Verdict = named quarantine on
  cost alone; safety, denominator restatement, and cache-eligible reuse all
  pass, but absolute wall-time and full-state-ratio cost gates miss by a small
  margin.
- Phase 1 v5 cost-closure slate opened: 2026-05-28.
- Phase 1 v5 harness executed: 2026-05-28. Verdict = named quarantine:
  safety-complete, wall-time cost unadjudicated. The earlier stable cost claim
  is withdrawn after artifact re-check: four clean full-harness invocations
  span `C_total_signature` 890 / 2192 / 2242 / 3185 ms, while the on-disk
  `cost_multirun_report.json` shows 2569 / 2091 / 2242 ms and
  `cost_repair_passed=false`. Determinism is confirmed by two fresh
  byte-identical v5 environment generations (`5549b4c4e8b7`; first env
  `pvnp-v5-cal-0001`). The stable cost signal is the op-count ratio 0.9487,
  which became the pre-registered v6 cost gate.
- Phase 1 v6 op-count cost slate opened: 2026-05-31. v6 keeps the v5 safety
  gates frozen, demotes wall-time to diagnostic-only, and gates cost on
  `C_total_signature_ops / C_rollout_ops <= 1.0`.
- Phase 1 v6 harness executed: 2026-05-31. Verdict = bounded positive under
  the registered v6 op-count protocol; wall-time remains diagnostic-only.
  Cost gate passed (`527297 / 555876 = 0.948587 <= 1.0`), cache reuse was 100%,
  short-circuit audit passed, privilege audit was green, and safety gates stayed
  clean (0/2304 false accepts, 0/453 field spoofs, 0/453 source spoofs, 5/5
  integrity probes, 0/768 OOP accepts, `capacity_threshold = not_estimated`).
  Independent disk re-verification 2026-05-31 confirmed all of the above and
  flagged one conservative op-count imprecision (numerator counts 2496
  signature calls incl. 64 calibration envs x 3 policies vs 2304 measurement
  rollout calls) that makes the reported ratio slightly higher than the clean
  measurement-only ~0.879 — directionally conservative, does not change the
  pass.
- Phase 2 mesa verification bridge opened: 2026-05-31. Spec/charter
  [`PHASE2_MESA_BRIDGE.md`](PHASE2_MESA_BRIDGE.md) filed; carries forward the
  v6 claim boundary.
- Phase 2 mesa bridge v0 slate frozen: 2026-05-31. Slate
  [`PHASE2_MESA_BRIDGE_V0_SLATE.md`](PHASE2_MESA_BRIDGE_V0_SLATE.md) locks
  source artifacts, verifier-access rules, mesa label mapping, op-count
  accounting, and pre-named falsifiers.
- Phase 2 mesa bridge v0 executed: 2026-05-31. Verdict = named quarantine.
  The bridge implementation recomputes from per-seed raw trial logs where they
  exist, keeps reward-blind access, catches the registered fixed-attractor /
  capacity-breach / mixed-objective falsifiers, and passes the same-artifact
  op-count comparator (`0.734877`). It cannot promote because 7/15 registered
  cells lack raw trial logs (`trial_logs_saved=false` for Small-tier source
  manifests), leaving raw recomputation, integrity, and the 3/4 signature
  accept floor failed.
- Phase 2 mesa bridge v1 slate frozen for implementation: 2026-05-31.
  [`PHASE2_MESA_BRIDGE_V1_SLATE.md`](PHASE2_MESA_BRIDGE_V1_SLATE.md) selects
  the raw-logged Small rerun repair, preserves the v0 population and
  thresholds, records full-size timing probes, and stages exact Phase 4 rerun
  commands.
- Phase 2 mesa bridge v1 executed: 2026-05-31. Verdict = bounded positive
  under the frozen v1 mesa-bridge contract. The run regenerates 6/6 Small raw
  policy batteries, preserves the original 15-cell population, recomputes
  15/15 cells from per-seed raw logs, accepts 4/4 signature cells, closes the
  registered fixed-attractor / capacity-breach / mixed-objective falsifiers,
  and passes the same-artifact raw-trace op-count comparator (`0.73760368`).
  Wall-time remains diagnostic-only; the result is not mesa-general,
  body-resistance, wall-time-cheap, or P-vs-NP evidence.
- Phase 3 capacity-relative one-wayness v0 slate frozen for implementation:
  2026-05-31 local / 2026-06-01 UTC. The slate carries forward the v1 caveats
  as registered Phase 3 design constraints: Small-tier near-threshold
  fragility, signature-only vs full-bridge separation, and conservative
  objective-conflict flag overfire. It also aligns with the updated
  cross-substrate note: mesa remains marginal on dimensional body-resistance,
  AB is the earned exact topological regime-2 witness, and Phase 3 is only a
  mesa-local inversion/spoof capacity battery.
- Phase 3 capacity-relative one-wayness v0 executed: 2026-05-31 local /
  2026-06-01 UTC. Verdict = **falsified in a registered cell**;
  `capacity_threshold <= small` for this mesa bridge battery. The harness
  (`scripts/pvnp-phase3-capacity-one-wayness-v0.mjs`, plus
  `scripts/pvnp-phase3-seed-extension.mjs` and the
  `scripts/lib/pvnp-phase3-*.mjs` cores) preserved the 15-cell population,
  reproduced the v1 verifier decisions byte-for-byte, and scored 24 source-bound
  seed-extension blocks. One block of the registered unsafe controller
  `phase5_l_mixed_lambda_0_7_small` was accepted by the bridge verifier without a
  breach/quarantine flag (signature 0.23545148, geometry 0.28284839), a spoof
  success at the smallest attacker tier on both views, found at 6.25 % of the
  small-tier candidate budget. Inversion also succeeded at the small tier (AUROC
  0.96–0.98) but is near-tautological. The v1 repair slate opens a
  block-consensus repair for per-seed-block mean drift and keeps inversion
  diagnostic-only. Receipt:
  [`receipts/2026-05-31_phase3_capacity_one_wayness_v0.md`](receipts/2026-05-31_phase3_capacity_one_wayness_v0.md).
- Phase 3 capacity-relative one-wayness v1 repair slate frozen and wired:
  2026-06-01. It does not revise v0; `capacity_threshold <= small` remains the
  v0 verdict. The repair keeps the base response thresholds unchanged but
  removes single-block promotion: accept requires 3-of-4 source-bound holdout
  blocks, while mixed or unstable block patterns quarantine. The harness is
  `scripts/pvnp-phase3-capacity-one-wayness-v1.mjs`, exposed as
  `npm run pvnp:phase3:capacity-one-wayness:v1`. The pre-holdout dry run
  correctly named-quarantines with 0/52 holdout blocks present, while the v0
  falsifier regression no longer consensus-accepts. The 52-block holdout
  battery remains operator-staged under the repository's runtime rule.
- Phase 3 capacity-relative one-wayness v1 executed: 2026-06-01. Verdict =
  **named quarantine**, repair strength **consensus-only repair**. The 52-block
  holdout battery was generated (`scripts/pvnp-phase3-v1-holdout.mjs`, 52/52
  integrity-clean) and scored by the consensus harness. The repair works on the
  unsafe side: the v0 falsifier no longer consensus-accepts and 0 unsafe cells
  reach `consensus_accept`; two unsafe single blocks still cross by seed-block
  drift (`l_mixed_lambda_0_7_small` seed 70000, `l_mixed_lambda_0_9_small` seed
  90000) but neither reaches 3-of-4, the registered consensus-only signature.
  The run cannot promote because the mixed-objective-laundering gate fails on the
  protected anchor `l_mixed_lambda_0_95_medium`, whose objective-conflict flag is
  block-unstable (2/4 < M=3) — the same drift mechanism as v0, now in the
  disclosure flag. Signature floor passed 3/3. v0 stays falsified;
  `capacity_threshold <= small` unrevised. Receipt:
  [`receipts/2026-06-01_phase3_capacity_one_wayness_v1.md`](receipts/2026-06-01_phase3_capacity_one_wayness_v1.md).
- Phase 3 capacity-relative one-wayness v2 disclosure slate opened for review
  and then provenance-corrected/frozen: 2026-06-04. It does not revise v0 or
  v1. The draft keeps v1's promotion consensus unchanged and targets only the
  disclosure aggregation failure: a bridge-view consensus accept is unqualified
  only if the clean/no-conflict side itself has stable consensus; a 2/2
  objective-conflict split is reported as block-unstable disclosed ambiguity.
  A full seed-100000/110000/120000/130000 holdout battery ran before the slate
  status was frozen; it is clean on disk but diagnostic-only and cannot promote.
  The promotion-eligible successor path is `phase3-capacity-one-wayness-v2b`
  with fresh seeds `140000, 150000, 160000, 170000`; its corrected fresh holdout
  battery is now complete on disk (52/52 blocks, 0 failed, 52/52 raw trial logs
  saved). No v2/v2b verifier verdict has been scored yet.
