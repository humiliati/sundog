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

Receipt filenames should use:

`YYYY-MM-DD_phase-or-probe_short-slug.md`

Use [`../RECEIPT_TEMPLATE.md`](../RECEIPT_TEMPLATE.md) for every filed result.
