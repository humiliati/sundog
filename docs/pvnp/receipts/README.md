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

Receipt filenames should use:

`YYYY-MM-DD_phase-or-probe_short-slug.md`

Use [`../RECEIPT_TEMPLATE.md`](../RECEIPT_TEMPLATE.md) for every filed result.
