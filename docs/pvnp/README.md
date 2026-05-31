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
  the v6 claim boundary; maps the mesa Phase 4 causal interventions
  (3 of 5 executed) to verifier-failure tests; uses the Phase 5 Large-tier
  reward-proxy emergence as the capacity-breach falsifier. No v0 execution
  slate frozen yet.

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
  Phase 1 v0-v5 named-quarantine receipts and the v6 op-count positive receipt.

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
  v6 claim boundary. No v0 slate frozen and nothing run; next step is to
  freeze `PHASE2_MESA_BRIDGE_V0_SLATE.md`.
