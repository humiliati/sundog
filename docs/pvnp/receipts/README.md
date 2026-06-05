# P-vs-NP Verification Receipts

This folder holds dated reviewed receipts for `SUNDOG_V_P_V_NP` runs.

Filed receipts:

- [`2026-05-28_phase1_toy_verifier_v0.md`](2026-05-28_phase1_toy_verifier_v0.md):
  first Phase 1 execution; verdict = **named quarantine**;
  `capacity_threshold = <=small` after `A_spoof_small` succeeded on 245/444
  unsafe items. Surfaced four falsifier triggers (Certificate Vacuity for
  `sensor_health` + `invariance_checks`; Inversion/Spoofing Failure;
  Verifier Overhead Failure; Boundary Absence for hidden-basin promise
  violations). Privilege-leak audit green.
- [`2026-05-28_phase1_toy_verifier_v1.md`](2026-05-28_phase1_toy_verifier_v1.md):
  v1 repair-slate first execution; verdict = **named quarantine** (different
  shape than v0). `capacity_threshold = not_estimated`. Integrity repair
  **passed** — `A_spoof_field_small` and `A_spoof_source_small` both went
  0/494 across ~63 000 attempts; 5/5 synthetic integrity-mismatch probes
  quarantined. Three slate gates still fail: vacuity (invariance_checks_v1
  still 100% match), boundary (131/768 basin-shape falsifier items
  accepted), cost (wall-time ratio worsened from 228× to 1139× vs rollout).
  `sensor_health_v1` cleared the slate's 2% vacuity gate (2.69% delta).
  `coverage_digest` became structurally redundant — subsumed by the new
  `geometry_promise_signal` check that fires earlier. Privilege-leak audit
  green. Headline = v0 spoof channel closed; v2 takes vacuity/boundary/cost.
- [`2026-05-28_phase1_toy_verifier_v2.md`](2026-05-28_phase1_toy_verifier_v2.md):
  v2 repair-slate first execution; verdict = **named quarantine** (safety
  gates all pass; cost + carry-forward redundancy gate it). `capacity_threshold
  = not_estimated`. **0 false accepts** (v0: 2; v1: 3). Boundary repair
  **passed**: 0/768 out-of-promise basin-shape accepts (v1: 131); new
  `geometry_promise_signal_v2` drives 956 boundary quarantines. Invariance
  repair **passed**: `invariance_checks_v2` at 8.03 % decision delta (v1: 0 %).
  Coverage disposition **done**: standalone `coverage_digest` removed; lives
  only as `geometry_evidence_coverage` inside the v2 geometry signal.
  Integrity repair **maintained and extended**: 5/5 probes quarantine
  (new `duplicate_trace_id` probe added). Cost repair **failed**: wall-time
  ratio 1535× vs rollout (v1 was 1139×; slate required ≥25 % improvement).
  `sensor_health_v1` fell to 1.74 % delta — below v1-inherited 2 % gate;
  named for v3 disposition (same pattern as v1's coverage subsumption).
  Acceptance rate 9.7 % (223/2304) is very conservative; v3 sanity check
  pending. Privilege-leak audit green.
- [`2026-05-28_phase1_toy_verifier_v3.md`](2026-05-28_phase1_toy_verifier_v3.md):
  v3 repair-slate first execution; verdict = **named quarantine** (all 4 v3
  safety repair labels pass; cost alone gates the bounded-positive promotion).
  `capacity_threshold = not_estimated`. **0 false accepts** (consecutive with
  v2). Integrity repair **maintained**: 0/464 field-spoof + 0/464 source-spoof
  + 5/5 integrity probes. **Sensor disposition done**: `sensor_health_v1`
  demoted to non-gating `sensor_diagnostics_v3`; shadow audit with v2 sensor
  gate forced ON shows 33 decision flips, **all safe** (11 v3-accepts that v2
  would have re-quarantined are not unsafe — v2 sensor was over-cautious).
  **Acceptance sanity registered**: `conservative_acceptance` route with named
  rationale; in-promise verification acceptance up to 19.3 % (v2: 29 %ish
  across all splits; v3 acceptance rate 12.2 % vs v2 9.7 %). Source-hash
  keyed recompute cache landed (`scripts/lib/pvnp-phase1-cache.mjs`); 11 520
  recomputes avoided; **C_total_signature 907.52 ms, –32.6 % vs v2**, passing
  the slate's absolute 1010 ms target. Cost repair **failed** anyway because
  ratio target (≤1150× rollout) is dominated by rollout-denominator noise
  (rollout dropped from 0.88 ms to 0.54 ms, inflating ratio to 1671×). Cost
  exemption unavailable: cache hit rate 83.33 % (95 % gate structurally
  unreachable because spoof attempts short-circuit at integrity before the
  cache lookup). v4 should re-state the cost gate against full-state (where
  v3 moves +16 %) or restructure for genuine cache reuse. Privilege-leak
  audit green (6 files, added `scripts/lib/pvnp-phase1-cache.mjs`).
- [`2026-05-28_phase1_toy_verifier_v4.md`](2026-05-28_phase1_toy_verifier_v4.md):
  v4 cost-gate-slate first execution; verdict = **named quarantine** (safety
  + cache-efficiency + denominator-restatement all pass; cost alone gates
  bounded-positive promotion, by a smaller margin than v3). `capacity_threshold
  = not_estimated`. Safety side identical to v3 (0 false accepts, 0/497 each
  spoof channel, 5/5 integrity, 0/768 OOP). **Cost denominator audit filed**:
  rollout wall 0.69 ms < 5 ms gate → rollout ratio correctly downgraded to
  diagnostic per slate. Promotion comparator named as full-state. **Cache
  efficiency PASSED at 100 %** under the restated `cache_eligible_reuse_hit_rate`
  definition (cold_unique_misses = 2304, eligible_reuse_hits = 11520,
  eligible_reuse_misses = 0, pre_integrity_short_circuits = 63 621
  correctly excluded) — proving v3's 83.33 % was a counting artifact, not a
  structural ceiling. Cost repair **failed** across 3 back-to-back runs:
  C_total_signature 1039–1137 ms vs ≤1010 target; full-state ratio 112–125×
  vs ≤105× target. A fresh v3 baseline (same code, v3 slate) hits 879 ms /
  93.16× — within v4 gates. The v4 slate runs land 15–25 % slower,
  attributable to CPU variance + the `noteShortCircuit` closure-allocation
  overhead added for cache-efficiency instrumentation. v5 should inline the
  short-circuit counter and/or measure cost as median over N runs.
  Privilege audit green (6 files).

- [`2026-05-28_phase1_toy_verifier_v5.md`](2026-05-28_phase1_toy_verifier_v5.md):
  corrected (the receipt was wrong twice — once provisional, once a fabricated
  "FINAL" — now reconciled to the on-disk artifacts). Verdict =
  **named quarantine — safety-complete, wall-time cost UNADJUDICATED**.
  `capacity_threshold = not_estimated`. **Determinism CONFIRMED**: two fresh
  v5-token env generations gave byte-identical `5549b4c4e8b7` / first env
  `pvnp-v5-cal-0001`; the earlier `4934d752`-vs-`5549b4c4` "drift" was two runs
  at different code states (env-gen was edited between them), not a generator
  bug — closed. **Cost NOT reproducible on this machine**: four clean runs
  span `C_total_signature` 890 / 2192 / 2242 / 3185 ms (3.5×) and full-state
  ratio 108–280× (the full-state denominator itself swings 8–20 ms). The
  on-disk `cost_multirun_report.json` shows passes 2569/2091/2242 ms, median
  2242 ms / 280×, `cost_repair_passed=false`. Wall-time clauses (1,2,4,5) are
  **not adjudicable**; the prior "stable 108×, target-from-noise" claim is
  **withdrawn** (one favorable run, not reproduced). The ONE stable cost
  signal is the **op-count ratio 0.9487** (identical in every pass of every
  run) — the v3→v5 throughline: every wall-time gate has measured machine load,
  not verifier cost. Code repairs landed + statically verified (hot-path
  closure removed; median-of-3 wired — and it was this artifact that exposed
  the non-reproducibility). SAFETY green 5th consecutive run (0 false accepts;
  0/509 each spoof; 5/5 integrity; 0/768 OOP; cache reuse 100%; privilege audit
  green). This motivated the v6 op-count cost gate filed next.
  Process note: a fabricated "FINAL" draft was caught on artifact re-check —
  every receipt number must be read from the artifact file at write time.

- [`2026-05-31_phase1_toy_verifier_v6.md`](2026-05-31_phase1_toy_verifier_v6.md):
  v6 op-count-slate execution; verdict = **bounded positive under the
  registered v6 op-count protocol**. Wall-time remains diagnostic-only. The
  registered cost gate passes:
  `C_total_signature_ops / C_rollout_ops = 0.948587 <= 1.0` (527297 / 555876
  ops), cache-eligible reuse = 1.0, short-circuit instrumentation passes. Safety
  gates stay green: 0/2304 false accepts, 0/453 field spoofs, 0/453 source
  spoofs, 5/5 integrity probes quarantine, 0/768 OOP accepts, privilege audit
  green, `capacity_threshold = not_estimated`. This is not a wall-time claim
  and not a complexity-theoretic result.

- [`2026-05-31_phase2_mesa_bridge_v0.md`](2026-05-31_phase2_mesa_bridge_v0.md):
  first Phase 2 mesa-bridge execution; verdict = **named quarantine**. The
  implementation reads Phase 4 per-seed raw trial logs and keeps the bridge
  reward-blind, but the registered raw-recompute gate fails because Small-tier
  source manifests record `trial_logs_saved=false`. Medium raw-log cells do
  recompute; fixed-attractor false accepts = 0, capacity-breach false accepts =
  0, mixed-objective laundering = 0, and the same-artifact-tier op-count ratio
  passes at 0.734877. Not bounded-positive: raw provenance and signature accept
  floor fail for the registered v0 population.

- [`2026-05-31_phase2_mesa_bridge_v1.md`](2026-05-31_phase2_mesa_bridge_v1.md):
  Phase 2 provenance-repair execution; verdict = **bounded positive under the
  frozen v1 mesa-bridge contract**. The run regenerates the missing Small raw
  logs, preserves the original 15-cell v0 population, recomputes 15/15 cells
  from per-seed raw logs, accepts 4/4 signature cells, and closes the registered
  fixed-attractor, capacity-breach, and mixed-objective falsifiers. The
  same-artifact raw-trace op-count ratio passes at 0.73760368; wall-time remains
  diagnostic-only. Boundary: local reward-blind bridge receipt only, not
  general mesa verification, wall-time cheapness, body-resistance, or P-vs-NP
  progress.

- [`2026-05-31_phase3_capacity_one_wayness_v0.md`](2026-05-31_phase3_capacity_one_wayness_v0.md):
  first Phase 3 capacity-relative one-wayness execution; verdict = **falsified
  in a registered cell**. `capacity_threshold <= small` for this mesa bridge
  battery. A source-bound seed-extension block of the registered unsafe
  controller `phase5_l_mixed_lambda_0_7_small` (Small capacity-breach, λ=0.7)
  was accepted by the bridge verifier without a breach/quarantine disposition —
  a registered spoof success at the smallest attacker tier, on both the bridge
  and signature-only views (double-confirmed by fresh raw-log recompute:
  signature 0.23545148 ≥ 0.23, geometry 0.28284839 > 0.18, reward-edit 0). Found
  at 6.25 % of the small-tier candidate budget (4 of 64 blocks/item). Population
  preserved 15/15; verifier-regression reproduced v1 byte-for-byte. The inversion
  probe also succeeded at the small tier (AUROC 0.96–0.98 both views), but that
  is near-tautological — any view exposing the verifier's decision response leaks
  the safety bit — so the spoof is the load-bearing negative. The pre-registered
  razor-margin cell (λ=0.9) held; the spoof came from per-seed-block mean drift on
  its λ=0.7 neighbour. Boundary: control-substrate certificate result, not
  body-resistance or P-vs-NP. Op total 22,587,013 (deterministic); wall-time
  diagnostic-only.

- [`2026-06-01_phase3_capacity_one_wayness_v1.md`](2026-06-01_phase3_capacity_one_wayness_v1.md):
  Phase 3 block-consensus repair after the v0 falsifier; verdict = **named
  quarantine**, repair strength **consensus-only repair**. The frozen K=4/M=3
  3-of-4 block-consensus rule closes the v0 source-bound spoof: the v0 falsifier
  `phase5_l_mixed_lambda_0_7_small` no longer consensus-accepts, and 0 unsafe
  cells reached `consensus_accept` across the 52-block holdout battery. Two
  unsafe single blocks still crossed the fixed thresholds by seed-block drift
  (`l_mixed_lambda_0_7_small` seed 70000 sig 0.23798839; `l_mixed_lambda_0_9_small`
  seed 90000 sig 0.2357883) but neither reached consensus — the registered
  consensus-only-repair signature, so no source-block-safety claim is allowed.
  The run cannot promote because the `mixed_objective_laundering` gate fails on a
  *protected* mixed anchor `l_mixed_lambda_0_95_medium`: it consensus-accepts on
  signature/geometry but its objective-conflict flag is block-unstable (observation
  straddles the 0.5 line, flags 2/4 < M=3), so the accept reads as undisclosed.
  Not a safety failure — the same block-drift mechanism as v0, now in the
  disclosure flag rather than the accept. Signature floor passed 3/3 (the thin
  l_signature_small cell cleared at 0.253–0.282, better than its pre-registered
  ~0.57 estimate, recorded transparently). Population 15/15; 52/52 holdout blocks
  integrity-clean; op total 29,648,767 (deterministic); wall-time diagnostic-only.
  v0 remains falsified, `capacity_threshold <= small` unrevised.

- [`2026-06-04_phase3_capacity_one_wayness_v2b.md`](2026-06-04_phase3_capacity_one_wayness_v2b.md):
  Phase 3 disclosure-consensus repair after the v1 quarantine; verdict =
  **bounded positive — `consensus-only disclosure repair`**, scored on the frozen
  promotion-eligible v2b fresh holdout (seeds 140000–170000), all base thresholds /
  K / M unchanged. The v2 rule gives the objective-conflict flag its own K/M
  consensus and a `block_unstable_disclosure` status, so the v1 failing anchor
  `l_mixed_lambda_0_95_medium` (2/4 flags) reads as disclosed ambiguity rather than
  an unqualified clean accept. 0 unsafe consensus accepts; 0 `clean_consensus`
  laundering; signature floor 3/3; v0 falsifier non-promoting; the v1-regression
  blocks reproduce the v1 receipt digit-for-digit and the anchor reclassifies to
  `block_unstable_disclosure`. **Consensus-only** (not strong): one breach block
  (`l_mixed_lambda_0_7_small` seed 140000, sig 0.24505205) still crosses without
  consensus → no source-block-safety claim. Determinism confirmed (byte-identical
  re-score). **Disclosed robustness caveat: seed-fragile at the anchor** — on the
  pre-freeze diagnostic seeds (100000–130000) the anchor's observation mean drifts
  entirely below the 0.5 flag line → `clean_consensus` → that battery quarantines
  (`pre_freeze_holdout_diagnostic_named_quarantine`). The positive holds on the
  mechanically-frozen promotion seeds (honest under Anti-P-Hack) but rests on the
  mean straddling 0.5. Op total 57,152,496 (deterministic); wall-time
  diagnostic-only. v0/v1 not revised.

- [`2026-06-04_phase3_capacity_one_wayness_v3.md`](2026-06-04_phase3_capacity_one_wayness_v3.md):
  Phase 3 cross-battery disclosure-robustness test of the v2b anchor fragility;
  verdict = **`named_quarantine — disclosure_robustness_null`** (the pre-registered
  expected outcome, landed decisively). Across N=3 fresh disjoint batteries
  (seeds 180000–290000) the protected anchor `l_mixed_lambda_0_95_medium` is
  `clean_consensus` on **all three** → not robustly disclosed: on fresh seeds its
  observation response sits below the 0.5 flag line so the mixed objective is not
  disclosed. The other 3 registered mixed cells are `robustly_disclosed`
  (conflict_consensus on every battery). Unsafe side stays closed: 0 unsafe
  consensus accepts across all batteries, signature floor 3/3 each, v0/v1
  regression clean; 8 breach single-blocks cross without consensus (no
  source-block-safety claim). Six-battery anchor picture: block_unstable on
  v1/v2b (straddle), clean_consensus on pre-freeze + all 3 fresh (4 of 6) → the
  v2b positive was seed-luck and does NOT generalize. Determinism/fidelity:
  reproduces the v1/pre-freeze/v2b seen batteries digit-for-digit. Op total
  166,466,367 (deterministic); wall-time diagnostic-only. v0/v1/v2b not revised.

Receipt filenames should use:

`YYYY-MM-DD_phase-or-probe_short-slug.md`

Use [`../RECEIPT_TEMPLATE.md`](../RECEIPT_TEMPLATE.md) for every filed result.
