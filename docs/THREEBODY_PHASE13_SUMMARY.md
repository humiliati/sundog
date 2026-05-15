# Three-Body Phase 13 Summary

Phase 13 asks whether the Phase 11 positive pocket survives a longer rollout,
or whether guarded accelerometer TRACK mostly delays failures inside the
original 8-second horizon.

## Smoke Run

Command:

```bash
npm run threebody:phase13:smoke
```

Output:

- `results/threebody/phase13-long-horizon-smoke/`
- 32 trials at `duration=16`
- 2 candidate envelope rows out of 12
- Terminal outcomes: 13 bounded, 15 escape, 4 close approach

The smoke deliberately covered a tiny pocket-plus-boundary slate: mass ratio
`1`, timestep `0.01`, radii `1.025` and `1.075`, velocities `0.9` and `1.1`,
four modes, and two seeds.

## Read

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

## Claim Impact

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

## Staged Full Run

Command:

```bash
npm run threebody:phase13
```

This runs 3 mass ratios x 3 timesteps x 3 radii x 4 velocity scales x 4 modes x
8 seeds at `duration=16`, for 3,456 trials. The 32-trial smoke took about 34
seconds on the current machine, so a linear estimate for the full run is about
one hour. Treat that as a staged operator run rather than an inline agent task.

Decision branches:

- Pass: high-velocity pocket stays positive against passive and naive baselines,
  late failures do not dominate, and delta-v-per-second remains comparable to
  Phase 11.
- Partial: survival improves but cost-rate or late-failure behavior suggests
  bounded delay rather than durable longer-horizon control.
- Fail: the Phase 11 pocket disappears or becomes controller-destabilized at 16
  seconds; redesign the guard/controller before larger sweeps.
