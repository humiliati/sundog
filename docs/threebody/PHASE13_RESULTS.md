# Three-Body Phase 13 - Longer-Horizon Result Note

This document records Phase 13 results for
[`PHASE13_SPEC.md`](PHASE13_SPEC.md). Phase 13 asks whether the Phase 11
positive pocket survives a longer rollout, or whether guarded accelerometer
TRACK mostly delays failures inside the original 8-second horizon.

Status: smoke run complete; full lock running or staged.

## 1. Smoke Run

Command:

```bash
npm run threebody:phase13:smoke
```

Output:

- `results/threebody/phase13-long-horizon-smoke/`
- 32 trials at `duration=16`
- 2 candidate envelope rows out of 12
- Terminal outcomes: 13 bounded, 15 escape, 4 close approach

The smoke covered mass ratio `1`, timestep `0.01`, radii `1.025` and `1.075`,
velocities `0.9` and `1.1`, four modes, and two seeds.

## 2. Smoke Read

The larger-radius, high-velocity cell (`radiusScale=1.075`,
`velocityScale=1.1`) remains promising at the longer horizon. Guarded
accelerometer TRACK bounded both seeds while passive bounded none, with mean
time delta `+9.85` seconds against passive and mean delta-v `1.948`.

The smaller-radius, high-velocity cell (`radiusScale=1.025`,
`velocityScale=1.1`) is neutral: passive already bounded both seeds for the
full 16-second horizon, so guarded TRACK does not earn a survival delta there.

Both low-velocity boundary cells (`velocityScale=0.9`) are risky. In those
cells, passive often survives longer than the controllers, and the dominant
failure mechanism is `controller_destabilized_or_shortened_passive`.

## 3. Claim Impact

The smoke does not yet justify upgrading the public claim. It supports running
the full Phase 13 lock because the high-velocity larger-radius pocket still
exists at 16 seconds, but it also confirms that horizon extension makes the
failure boundary harder, not softer.

Current earned wording remains:

> In the tested planar restricted setup, a guarded accelerometer-proxy TRACK
> controller improves survival over passive and naive local baselines in a
> robust high-velocity near-escape operating pocket.

The next ratchet requires the full Phase 13 run to show that this pocket remains
positive across mass ratio, timestep, and nearby high-velocity cells at the
longer horizon, without a large delta-v-per-second or late-failure penalty.

## 4. Full Lock Readback

Command:

```bash
npm run threebody:phase13
```

Expected slate: 3 mass ratios x 3 timesteps x 3 radii x 4 velocity scales x 4
modes x 8 seeds at `duration=16`, for 3,456 trials.

Fill after completion:

- Wall-clock runtime:
- Trial count:
- Terminal outcomes:
- Candidate-envelope rows:
- Best-cell class balance:
- Favorable-pocket read:
- Boundary-cell read:
- Delta-v-per-second read:
- Late-failure read:
- Claim branch: pass / partial / fail
