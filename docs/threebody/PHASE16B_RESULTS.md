# Three-Body Phase 16B - Radius Warning Re-Pose Results

Status: **spec drafted 2026-05-29; pending operator lock review.** No Phase 16B
code has been written and no Phase 16B command has been run.

Phase 16B will re-pose the Phase 15 warning-quality readout with `radius` as the
warning score, using the completed Phase 16 passive lock receipt. It is an
offline verdict-reconciliation pass only: no new simulation, no controller
retune, no new hazard label, and no revision of the Phase 15 Fail-Magnitude
mechanism verdict.

Pending lock artifacts:

- [`PHASE16B_SPEC.md`](PHASE16B_SPEC.md)
- future output directory:
  `results/threebody/phase16b-radius-warning-repose/`

## Pending Command

```powershell
npm run threebody:phase16b:repose
```

## Pending Readback

Record the Phase 16 source receipt, favorable-pocket coverage, radius
Phase-15-style mean AUROC, energy baseline mean AUROC, per-velocity summary, and
the selected Branch A/B/C from the locked spec.
