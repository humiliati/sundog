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
  Phase 1 v0-v4 named-quarantine receipts and the v5 provisional receipt.

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
- Phase 1 v5 harness executed: 2026-05-28. Receipt is PROVISIONAL:
  safety-complete, cost-unadjudicated. The earlier stable cost claim is
  withdrawn after incompatible v5 reruns and `environments.jsonl` hash drift.
  Before v6 or Phase 2, rerun v5 twice on a quiescent machine and finalize or
  void the receipt.
