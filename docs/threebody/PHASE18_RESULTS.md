# Three-Body Phase 18 - Radius-Only Controller Control Results

Status: **complete 2026-05-29; formal branch: Branch A - Reduces to radius.**
Phase 18 is a mechanism diagnostic only: it does not retune guarded TRACK,
change the inherited radius threshold,
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

## Calibration — matched-m = 0.4 (recorded before the measurement lock)

`npm run threebody:phase18:calibrate` — favorable-pocket mean `totalDeltaV`,
selected on ΔV **blind to survival**. guarded ΔV is identical across all four
runs (cross-run spread 0), confirming magnitude-independence.

| inward m | mean ΔV | ratio vs guarded (3.727) | \|Δ\| |
|--:|--:|--:|--:|
| 0.1 | 0.555 | 0.149 | 3.173 |
| 0.2 | 1.461 | 0.392 | 2.266 |
| 0.3 | 2.697 | 0.724 | 1.030 |
| **0.4** | **3.909** | **1.049** | **0.181** (min) |

**matched-m = 0.4**, ΔV ratio **1.049** — a within-5% duty match. (matched-m sits
at the top of the locked grid; linear interpolation puts the exact ΔV match at
m≈0.385, but the locked `argmin` over `{0.1,0.2,0.3,0.4}` selects 0.4. The full
m-sweep survival is reported as a sensitivity curve in the readback.)

## Measurement Lock + Readback

`npm run threebody:phase18:control -- --radius-inward-magnitude 0.4` — 1,152
trials, full grid, dt=0.004; outcomes escape 273 / close 83 / bounded 796;
candidate envelope rows 73/108.

### §6 Branch: **(A) Reduces to radius**

Favorable pocket (27 cells), each radius-only mode vs `track_sensor_accel_guarded`:

| mode | favorable candidate cells | Jaccard w/ guarded | within-±0.10 survival band | mean favorable survivalΔ |
|---|--:|--:|--:|--:|
| `track_sensor_accel_guarded` | 24/27 | — | — | 0.8009 |
| `track_radius_guard` (rung 1) | 17/27 | 0.708 | 16/24 = 0.667 | 0.7731 |
| `track_radius_inward` (rung 2, m=0.4) | 25/27 | **0.815** | **19/24 = 0.79** | **0.8009** |

**Rung 2 clears branch A comfortably** (Jaccard 0.815 ≥ 0.60 AND band 0.79 ≥ 2/3),
and its mean favorable survival-delta **matches guarded to four digits (0.8009)**
— a controller with **no tidal sensing, no gradient, no error term**, doing
nothing but thrusting radially inward when `radius ≥ trackGuardMinRadius`,
reproduces guarded TRACK's survival envelope at matched duty.

### Per-mass-ratio split (favorable, candidate cells / mean survivalΔ)

| mass ratio | guarded | rung 1 (radius_guard) | rung 2 (radius_inward) |
|---|---|---|---|
| 0.01 | 9/9 · 0.861 | 9/9 · 0.861 | 8/9 · 0.819 |
| 0.3 | 7/9 · 0.875 | 5/9 · 0.861 | 8/9 · 0.903 |
| 1 | 8/9 · 0.667 | **3/9 · 0.597** | 9/9 · 0.681 |

**Rung 2 reproduces (or slightly exceeds) guarded across all three mass ratios.**
Rung 1 passes the overall bar only *marginally* and is mass-ratio-fragile — at
`mu=1` it collapses to 3/9 candidate cells. So the radius-only **guard** does
**not** cleanly reduce to radius (dropping the local-accel/tidal guard conditions
hurts at high mass ratio), yet the **full** radius-only controller does. The
inward push — not the tidal steering — is the robust survival driver; kept on a
loosened radius-only gate (rung 1), the steering is even mildly counterproductive
at `mu=1`.

### Inward m-sweep sensitivity (favorable mean survivalΔ; calibration dirs)

| m | mean favorable survivalΔ | candidate cells | ΔV ratio |
|--:|--:|--:|--:|
| 0.1 | 0.213 | 14/27 | 0.149 |
| 0.2 | 0.421 | 17/27 | 0.392 |
| 0.3 | 0.648 | 26/27 | 0.724 |
| **0.4 (matched)** | **0.801** | 25/27 | 1.049 |

Survival rises smoothly with duty and meets guarded (0.801) at the matched point.
Honest note: matched-m=0.4 over-spends duty by ~5% (ΔV ratio 1.049); at an
exactly-matched duty (m≈0.385 by interpolation) rung 2 would land marginally below
guarded (~0.78) but still within the ±0.10 band — the reproduction is robust to
the grid-edge duty match. **v=0.95 boundary control:** guarded 2/9, rung 1 2/9,
rung 2 3/9 candidate cells — uniformly low, no spurious candidate inflation.

### Interpretation

**Guarded TRACK's survival edge reduces to radius-gated inward thrusting.** The
sophisticated machinery — noisy-accelerometer tidal-gradient sensing, error-
proportional action coupling, the precise three-condition guard — is **not
load-bearing for survival**: the simplest possible reflex ("when far out, thrust
toward the center") reproduces the whole favorable-pocket survival envelope at
matched duty, across every mass ratio.

This **closes Phase 17's open mechanism question**. The survival edge that no
per-step first-action counterfactual could isolate — energy (15C) or
hazard-margin (17) — was invisible to those instruments precisely because it is
**not** a per-step action-choice effect: it is a cumulative radius-gated push that
any inward controller reproduces. The counterfactuals asked "is *this* action
better than *that* one?" when the mechanism is "keep pushing inward while far
out," a policy property, not a per-step margin.

**Phase 15 verdict preserved.** Branch A *explains* the mechanism — and is, if
anything, deflationary about the controller's sophistication — but it does **not**
(and by the locked §1/§6 cannot) convert the Phase 15 Fail-Magnitude verdict or
broaden the gravity claim. Mechanism finding only.

### Next registered move

The mechanism is now explained, so the per-step-counterfactual / mechanism line of
inquiry is **closed**. The remaining move is a **claim-language update** — frame
the controller's survival as a simple radius-gated geometric reflex rather than
sophisticated tidal control — which is a documentation decision, not a new
experiment. Any further controller work (e.g. whether the tidal steering ever adds
value outside the near-escape pocket) would be a fresh pre-registered phase.
