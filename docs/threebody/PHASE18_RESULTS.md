# Three-Body Phase 18 - Radius-Only Controller Control Results

Status: **spec locked 2026-05-29; implementation pending.** No Phase 18 code has
been written and no Phase 18 command has been run.

Phase 18 will test whether guarded TRACK's survival envelope reduces to a
controller gated solely on radius. It is a mechanism diagnostic only: it does not
retune guarded TRACK, change the inherited radius threshold, alter the hazard
label, revise the Phase 15 Fail-Magnitude verdict, or broaden the gravity claim.

Locked artifacts:

- [`PHASE18_SPEC.md`](PHASE18_SPEC.md)

## Pending Commands

```powershell
npm run threebody:phase18:smoke
npm run threebody:phase18:calibrate
npm run threebody:phase18:control
```

## Pending Readback

Record:

- smoke command, wall-clock, and mode sign sanity;
- hard-void Phase 13/14 gate status after the additive mode implementation;
- calibration DeltaV table over `m in {0.1,0.2,0.3,0.4}`;
- matched inward magnitude and DeltaV ratio, recorded before the measurement
  lock;
- measurement-lock candidate-cell Jaccard and survival-band fraction for
  `track_radius_guard` and matched `track_radius_inward`;
- per-mass-ratio split, inward m-sweep sensitivity, and branch A/B/C.
