# Sundog Certificate Problem — Syndrome Certificate v2 Stronger-ISD Slate

Status: **DRAFT — opened for review, NOT frozen.** No v2 attacker run may execute
against the frozen regime until this slate is frozen and each attacker has been
smoke-validated on a throwaway regime. (Freeze-before-execute, per the lane.)

Date opened: 2026-06-04

This is the **stronger-attacker** successor to
[`SUNDOG_CERTIFICATE_SYNDROME_V1_SLATE.md`](SUNDOG_CERTIFICATE_SYNDROME_V1_SLATE.md)
(Prange ISD → bounded positive, measured `C`, receipt
[`receipts/2026-06-04_certificate_syndrome_v1.md`](receipts/2026-06-04_certificate_syndrome_v1.md)).
v1's measured `C` was a threshold **against Prange ISD** — an explicit upper bound.
v2 tightens that bound: it measures the witness-recovery threshold against a ladder
of stronger attackers on the **same frozen instance**, and tests whether `C` drops
monotonically as the better-attack work factors predict.

## What v2 measures

On the **same frozen `[128,64] w=12` regime and the same `T=64` targets as v1**,
the witness-recovery (invert-`e`) capacity threshold `C` for a **ladder of attacker
classes** — Prange (the v1 baseline, reused), Lee-Brickell (p=2), and Stern
(p=2, l=4). The single question:

> Does `C` drop monotonically as attacker strength increases, by the predicted
> work factors? (Stronger finder → lower `C` → tighter upper bound on the
> certificate's one-wayness.)

**Target-manifest caveat (load-bearing):** v1 did not emit a target manifest, and
its harness interleaved target sampling with Prange attack draws. Therefore v2 must
first reconstruct the v1 target set by replaying the v1 deterministic stream and
persisting `target_manifest.json` (target id, `z` hash / value, and scoring labels in
a labels-only section). All v2 attackers consume the manifest's `z` values only. If
the v1 target set cannot be reconstructed byte-identically, the "same targets as v1"
claim is unavailable and the slate must either downscope to a new target set or get a
new id.

The certificate, the existence safety predicate `Safe(y) := ∃ e* : He*=Hy ∧
wt(e*)≤τ`, the witness-verifier, and the `invert-s`-is-vacuous / spoof-is-structural
facts are **unchanged from v1**. Only the attacker class is strengthened.

## Capacity unit: operations

The attackers do different work per iteration (Stern does more per iteration but
far fewer iterations), so the only fair cross-attacker capacity unit is **operations**.
`C_attacker := the measured op-count budget at which that attacker reaches 50%
witness recovery` over the `T` targets. Each attacker also reports its natural
iteration budget and its valid-iteration audit. The ladder claim is
`C_Prange > C_LB > C_Stern` (in ops), monotone, matching the predicted work factors.

## Frozen regime (identical to v1 — do not change)

`[n=128, k=64], w=12, τ=12`, GF(2), `code_seed=2026128`, body seed base `7000000`,
`T=64` targets. Full enumeration space `C(128,12)=2.37×10¹⁶` (non-enumerable). Using
the **same** regime, seeds, and targets as v1 is what makes the cross-attacker `C`
comparison apples-to-apples; changing any of them voids the comparison.

## The attacker ladder (frozen, named classes)

All three find a light same-syndrome witness `e*` (`He*=z`, `wt(e*)≤τ`) given only
`z`. **Valid-iteration convention** (generalizing v1's rank-valid rule): each
attacker counts only iterations whose required Gaussian-elimination step is
full-rank; rank-deficient draws are charged to measured ops and audited as
`rank_fail_draws`, **not** counted as iterations. The analytic curves are read
against valid iterations; the cross-attacker comparison is in measured ops. For
Stern, a full-rank draw with zero useful list collisions is still a valid iteration
and still charged to ops; collision/list work is never free.

1. **Prange** (baseline, reused from v1): random size-`k` information set; candidate
   error supported on the `n−k` complement; success iff its weight `≤ τ`.
2. **Lee-Brickell, p=2** (frozen `p=2`): Prange, but allow exactly `p=2` error
   positions inside the information set — enumerate the `C(k,2)` weight-2 info-set
   patterns, add each to the candidate, accept if total weight `≤ τ`.
3. **Stern, p=2, l=4** (frozen `p=2, l=4`): split the information set into two halves
   of size `k/2`; on an `l=4`-bit window of the redundancy, build the two lists of
   weight-`p` half-patterns and their window-syndromes; for each collision on the
   window, form the full candidate and accept if weight `≤ τ`.

No MMT/BJMM/representation attacks in v2 (a future slate). Implementations are frozen
to these algorithms with these parameters; each must reproduce a valid witness
(`He*=z`, `wt≤τ`) before its curve is trusted.

## Pre-registered prediction (the comparator)

Draft work-factor target on `[128,64] w=12`, per-iteration linear-algebra cost
modeled at `≈ (n−k)³ = 2.62×10⁵` ops:

| Attacker | iters | per-iter ops | work `W` (ops) | predicted `C` drop vs Prange |
| --- | ---: | ---: | ---: | ---: |
| Prange (v1) | 7,224 | 2.62×10⁵ | **1.9×10⁹** | 1× (v1 measured `C≈5007` trials / ~3×10⁹ ops at 50%) |
| Lee-Brickell p=2 | 78 | 3.91×10⁵ | **3.0×10⁷** | **≈ 62×** |
| Stern p=2, l=4 | 38 | 3.28×10⁵ | **1.2×10⁷** | **≈ 153×** |

**Draft expected ladder:** `C_Stern ≈ C_Prange / 153 < C_LB ≈ C_Prange / 62 <
C_Prange`. The run measures each attacker's witness-recovery curve vs an op budget,
locates each 50%-breakpoint `C(ops)`, and reports whether the measured drops match
the frozen prediction. The prose factors above are review targets, not the freeze
comparator; the exact tolerance must be in `prediction_lock.json`. A measured drop
far from the locked prediction is itself the finding (the better-attack model is off
for this regime).

**Freeze requirement:** before any frozen-regime attacker run, replace this draft
planning table with a machine-readable `prediction_lock.json` that records, for each
attacker, the exact success-probability formula, valid-iteration definition, op-count
formula, calibrated constants from throwaway smoke only, predicted 50% `C(ops)`, and
the frozen tolerance. Updating constants after seeing frozen-target results voids the
run. The prose table is not enough for freeze.

## Primary gates

| Gate | Required |
| --- | --- |
| Regime identical to v1 | `(n,k,w,τ,code_seed,body-base,T)` byte-equal to v1 |
| Target manifest | v1 `T=64` targets reconstructed and persisted before attacker runs; every attacker reads only `z` from it |
| Code validity | `G Hᵀ = 0` |
| Verifier cheap + unchanged | check op-count flat (`16,576` ops, as v1) `≪ C` for every attacker |
| Witness validity | every attacker, on success, exhibits a valid `e*` (`He*=z`, `wt≤τ`) — verified |
| Privilege / label audit | every attacker receives **only** `z`; planted `e`/`wt(e)`/`s` scoring-only |
| Valid-iteration audit | per attacker: `rank_fail_draws`, valid iterations, measured ops (incl. overhead) |
| Monotone ladder | `C_Prange > C_LB > C_Stern` in ops, each curve a clean `0→1` transition |
| Prediction match | each measured `C(ops)` within the frozen `prediction_lock.json` tolerance |
| Determinism | byte-identical re-run under the frozen seeds (or established by construction + smoke, per v1) |
| Cost reported | per-attacker ops, the `C` ladder, the find-vs-check gap at each `C` |

## Verdict branches

- **bounded-positive measured attacker-hierarchy one-wayness** — the three curves
  rise `0→1` with monotone breakpoints `C_Prange > C_LB > C_Stern` (ops) matching
  the predicted work factors (within tolerance), the verifier stays cheap, and the
  audits pass: report the tightened upper bound `C_Stern` (the threshold against the
  best tested attacker) and the full Prange/LB/Stern `C`-ladder.
- **model-deviation (named, not a failure)** — the ladder is monotone but a measured
  `C` deviates from its predicted work factor beyond tolerance: report the deviation
  honestly (the better-attack model is mis-calibrated for this regime); the tightened
  bound still stands at the measured `C_Stern`.
- **6.1 vacuity / 6.4 overhead** — the verifier op-count is not below the cheapest
  attacker's effort: falsified.
- **non-monotone / no clean threshold (6.5 boundary)** — an attacker fails to beat a
  weaker one, or a curve has no locatable breakpoint: named quarantine; investigate
  the implementation (a stronger attacker that does not beat a weaker one is most
  likely an implementation bug, which the smoke must have caught).
- **void_run** — regime not identical to v1, labels leaked, an attacker given more
  than `z`, a non-validated attacker (no exhibited witness), or non-deterministic.

## Required outputs (under `results/pvnp/certificate-syndrome-v2/`)

- `manifest.json` (frozen regime = v1, attacker ladder, git sha, verdict);
- `target_manifest.json` (reconstructed v1 target set; attacker-visible `z` and
  labels-only scoring fields separated);
- `prediction_lock.json` (formulas, constants, predicted `C(ops)`, tolerance,
  smoke calibration provenance);
- `verifier_access_declaration.json` (every attacker sees only `z`; labels scoring-only);
- `attacker_ladder_curve.csv` (per attacker, per op-budget: witness-recovery success,
  iterations, measured ops);
- `capacity_ladder.json` (per attacker: measured `C(ops)`, predicted work factor,
  measured/predicted ratio, drop vs Prange);
- `valid_iteration_audit.json` (per attacker: valid iterations, `rank_fail_draws`,
  measured ops incl. overhead);
- `witness_validity_audit.json` (per attacker: a sample of recovered `e*` re-checked
  `He*=z ∧ wt≤τ`);
- `op_count_report.json` (verifier ops flat; attacker ops at each `C`; gaps);
- `prediction_vs_measured.json`; `falsifier_summary.md`; `README.md`.

(Per the v1 receipt's artifact-packaging note: this slate **pre-registers** the
per-file output bundle above; the harness must emit it, not a single consolidated
JSON.)

## Anti-P-Hack rule

- Reuse v1's exact regime, seeds, and targets; materialize the target manifest before
  attacker runs; freeze the attacker parameters (`LB p=2`, `Stern p=2,l=4`) and
  `prediction_lock.json` **before** the frozen run; do not retune parameters or
  constants to make the ladder land on the prediction.
- Every attacker receives **only** `z`; planted `e`/`wt(e)`/`s` are scoring-only.
- Each attacker is smoke-validated on a **throwaway** regime first (it must recover
  valid witnesses and match its analytic curve) before the frozen run. Smoke may
  calibrate op constants only because it is disjoint from the frozen regime; the
  calibrated constants must be locked before frozen targets are scored.
- Deterministic seeds; report determinism.
- The measured `C_Stern` is still an **upper bound** (BJMM/MMT would lower it) — a
  new slate, not a v2 edit.

## Freeze checklist

- [ ] Reconstruct and hash the v1 `T=64` target manifest; confirm the regime/seeds
      are byte-equal to v1 and all attackers will consume only manifest `z`.
- [ ] Freeze `LB p=2` and `Stern p=2, l=4`, plus `prediction_lock.json` with exact
      formulas, calibrated constants, predicted 50% `C(ops)`, and tolerance.
- [ ] Implement the three attackers (Prange reused) sharing the v1 verifier + GF(2)
      core; smoke each on a throwaway regime (recover valid witnesses; curve matches
      analytic) before the frozen run.
- [ ] Re-affirm: every attacker sees only `z`; labels scoring-only; per-file outputs.

## Freeze rule

Edits allowed without a new slate id: typo/path/output-naming corrections preserving
semantics. Edits requiring a new slate id: changing the regime/seeds/targets (would
break the v1 comparison); changing the attacker set or the frozen `p`/`l`; adding a
representation attack (MMT/BJMM); giving an attacker more than `z`; using
`wt(e)`/`s` as an input; retuning parameters after seeing the ladder.
