# Three-Body Phase 17 - Hazard-Aligned Counterfactual Results

Status: **spec locked 2026-05-29; pending implementation.** No Phase 17 code has
been written and no Phase 17 command has been run.

Phase 17 will test whether guarded TRACK actions move the state away from the
same frozen hazard boundary used by the Phase 15/16 oracle label:
`r3 > 4 OR minPrimaryDistance < 0.08`. It is a mechanism audit only. It does not
retune the controller, alter the hazard label, or revise the locked Phase 15
Fail-Magnitude verdict under any branch — verdict revision would require a fresh
pre-registered lock with the geometric metrics fixed in advance (spec §1, §6).

Pending lock artifact:

- [`PHASE17_SPEC.md`](PHASE17_SPEC.md)

## Pending Commands

```powershell
npm run threebody:phase17:hazard-cf-smoke
npm run threebody:phase17:hazard-cf
```

The lock is expected to exceed the inline-agent ten-minute rule unless the smoke
rate proves otherwise. Record the smoke wall-clock and extrapolated lock cost
before starting any full run.

## Pending Readback

Record the smoke column-flow receipt, hard-void gate status, lock wall-clock,
candidate/non-candidate mode x horizon tables, guarded-vs-delay and
guarded-vs-sign-flip separation, escape-vs-close subtype diagnostic, and the
selected Branch from the locked spec.
