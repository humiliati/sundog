# Three-Body Phase 18 - Radius-Only Controller Control Results

Status: **spec locked 2026-05-29; implementation + smoke + gates complete;
calibration → measurement lock pending.** Phase 18 is a mechanism diagnostic
only: it does not retune guarded TRACK, change the inherited radius threshold,
alter the hazard label, revise the Phase 15 Fail-Magnitude verdict, or broaden
the gravity claim.

## Implementation Receipt

Additive (two new modes), no audit instrumentation:

- `public/js/threebody-core.mjs`: `radiusInwardMagnitude` default (0.4);
  `track_radius_guard` (rung 1: radius-only guard predicate, otherwise the exact
  guarded-TRACK noisy-tidal-gradient thrust) and `track_radius_inward` (rung 2:
  radius-gate + fixed radially-inward thrust `−m·(x3,y3)/radius`, no tidal
  sensing); both registered in `KNOWN_CONTROLLER_MODES`, **not** in
  `PHASE14_ABLATION_MODES`. No edit to existing modes, `shouldRunGuardedTrack`, or
  `deriveGuardThresholds`.
- `scripts/threebody-operating-envelope.mjs`: `--radius-inward-magnitude` flag +
  config wire.
- `scripts/threebody-phase18-calibrate.mjs`: matched-duty calibration wrapper.
- `package.json`: `threebody:phase18:{smoke,calibrate,control}`.
- `node --check` clean on core, harness, calibrate wrapper.

## Hard-void gates (post-implementation)

Re-run 2026-05-29 — **both byte-identical** (additive modes are non-perturbing):

- phase13 → 3,456 / 88·324 / bounded·escape·close 1154·2030·272 ✓
- phase14 → 6,048 / 130·648 / 1269·4616·163 ✓

## Smoke Readback

`npm run threebody:phase18:smoke` — 4 modes, 1 cell, 1 seed, duration 4; all
bounded. Per-mode duty (totalDeltaV):

| mode | totalDeltaV | note |
|---|--:|---|
| `track_sensor_accel_guarded` | 0.808 | reference |
| `track_radius_guard` (rung 1) | 0.914 | **fires more** than guarded (dropped 2 guard conditions) ✓ |
| `track_radius_inward` (rung 2, m=0.4) | 0.754 | already near guarded's duty at top of grid |

**Sign sanity (rung 2, direct controller check):** `track_radius_inward` thrusts
**radially inward** (`dot(thrust, position) < 0`) in every tested direction
(E/N/NW), magnitude = m, and is **gated off** when `radius < trackGuardMinRadius`.
Correct by construction.

## Calibration (pending)

```powershell
npm run threebody:phase18:calibrate    # favorable pocket; selects matched-m by ΔV, blind to survival
```

<!-- matched-m + ΔV table to be recorded HERE before the measurement lock -->

## Measurement Lock + Readback (pending)

```powershell
npm run threebody:phase18:control -- --radius-inward-magnitude <matched-m>
```

<!-- branch A/B/C; candidate-cell Jaccard + survival-band fraction for rung 1 and
matched rung 2; per-mass-ratio split; inward m-sweep sensitivity; v=0.95 control -->.
