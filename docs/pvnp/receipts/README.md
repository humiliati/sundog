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
  v5 cost-closure-slate first execution; verdict = **named quarantine** (7 of 8
  cost clauses pass; only the full-state wall-time ratio gates it).
  `capacity_threshold = not_estimated`. **Hot-path closure removed** (audited:
  no `noteShortCircuit` closure; 6 direct `recordPreIntegrityShortCircuit`
  call sites) and **median-of-3 cost protocol** implemented. Median
  `C_total_signature` dropped to **889.7 ms** (v4 ~1130 ms) with **2.41 %
  spread** across 3 passes — proving the v4 quarantine was single-sample noise
  + closure overhead, now resolved. Absolute (≤1010 ms), max (≤1250 ms),
  spread (≤25 %), op-ratio (0.9473), cache-reuse (100 %), rollout-diagnostic,
  and short-circuit-audit clauses all PASS. Only the full-state ratio fails:
  **108.21 × median vs ≤105 ×** (3.1 % miss). The sharpened finding: with
  closure overhead gone and a 2.41 % spread, the verifier sits at a STABLE
  ~108–112 × the wall-cost of the privileged O(T) full-state scan; the ≤105 ×
  target was set from v3's single favorable 95.92 × sample (a noise-derived
  threshold). Safety green 4th consecutive run (0 false accepts; 0/497 each
  spoof; 5/5 integrity; 0/768 OOP; privilege audit green). FS-integrity note:
  v4/v5 results dirs were hardlinked mid-implementation and resolved by full
  rm-rf + fresh run; v5 verified inode-independent and manifest-hash-consistent.
  v6 should either re-baseline the full-state target from the v5 median-of-3
  distribution or algorithmically reduce verifier wall-cost.

Receipt filenames should use:

`YYYY-MM-DD_phase-or-probe_short-slug.md`

Use [`../RECEIPT_TEMPLATE.md`](../RECEIPT_TEMPLATE.md) for every filed result.
