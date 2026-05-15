# Three-Body Phase 13 - Longer-Horizon Lock Spec

This document is the implementation-grade spec for Phase 13 of
[`../SUNDOG_V_THREEBODY.md`](../SUNDOG_V_THREEBODY.md). Phase 11 showed that
guarded accelerometer TRACK improves survival in a robust high-velocity
near-escape pocket. Phase 13 asks whether that result survives a doubled
rollout horizon, or whether it mainly delays failures that arrive just outside
the original 8-second window.

Where this spec and the roadmap disagree, the roadmap wins. Where both are
silent, this spec is authoritative for Phase 13.

## 1. Decision Lock

Phase 13 starts with six pinned calls:

- **Horizon extension is the main axis.** The canonical comparison moves from
  `duration=8` to `duration=16`; it does not change controller code.
- **Same-mode comparison.** Passive, naive local acceleration, guarded
  accelerometer TRACK, and privileged heuristic oracle stay on the same slate.
- **Boundary cells stay in.** The long-horizon slate must include low-velocity
  boundary cells so the result cannot silently select only the winning pocket.
- **Cost-rate is load-bearing.** Report total delta-v and delta-v per simulated
  second; a survival gain that buys time by exploding effort is a partial, not a
  pass.
- **Late failures are separated.** Track cells that survive the original
  horizon but fail before 16 seconds, because those are delay evidence rather
  than durable-control evidence.
- **No indefinite-stability claim.** A pass supports only a longer tested
  horizon inside a mapped restricted setup.

## 2. Scope

Phase 13 owns:

- `npm run threebody:phase13:smoke`
- `npm run threebody:phase13`
- Outputs under `results/threebody/phase13-long-horizon-smoke/`
- Outputs under `results/threebody/phase13-long-horizon-lock/`
- Result note [`PHASE13_RESULTS.md`](PHASE13_RESULTS.md)
- Roadmap and writeup receipt bullets after the lock lands

Phase 13 does **not** own:

- Controller redesign
- Spatial/3D extension
- New sensor models
- Orbit-family maintenance claims
- Retuning based on the full lock after observing its result

## 3. Commands

Smoke:

```bash
npm run threebody:phase13:smoke
```

Full lock:

```bash
npm run threebody:phase13
```

The smoke covers mass ratio `1`, timestep `0.01`, radii `1.025` and `1.075`,
velocities `0.9` and `1.1`, four controller modes, and two seeds at
`duration=16`.

The full lock covers:

- mass ratios: `0.01`, `0.3`, `1`
- timesteps: `0.008`, `0.01`, `0.012`
- radii: `1.025`, `1.05`, `1.075`
- velocities: `0.95`, `1.05`, `1.1`, `1.15`
- modes: `off`, `naive`, `track_sensor_accel_guarded`, `oracle`
- seeds: `8`
- duration: `16`

That is 3,456 trials. The 32-trial smoke took about 34 seconds on the current
machine, so the linear full-run estimate is about one hour. Treat the full lock
as a staged operator run under the repository's long-run rule.

## 4. Metrics

Read these files first:

- `aggregate-envelope.csv`
- `candidate-envelope.csv`
- `best-by-cell.csv`
- `cell-class-map.csv`
- `trial-outcomes.csv`

Primary metrics:

- Candidate envelope rows out of total envelope rows
- Best-cell class balance: promising, mixed, neutral, risky, negative
- Survival delta versus passive
- Worsened rate versus passive
- Mean simulated-time delta versus passive
- Mean delta-v
- Mean delta-v per simulated second
- Dominant failure mechanism

Phase 13-specific checks:

- Favorable high-velocity cells should remain positive against passive and
  naive local control.
- Low-velocity boundary cells should remain explicit even if they are negative.
- Cells where passive already survives should be labeled neutral rather than
  counted as evidence for the controller.
- Any cell with high survival delta but sharply worse delta-v-per-second should
  be labeled partial.

## 5. Pre-Registered Branches

**Pass:** guarded accelerometer TRACK remains positive against passive and naive
baselines in the high-velocity pocket, has no large late-failure cliff, and its
delta-v-per-second does not materially explode relative to Phase 11.

**Partial:** survival improves at 16 seconds but cost-rate or late failures grow
enough that the claim remains "delay and survival improvement in a bounded
pocket," not durable longer-horizon control.

**Fail:** the Phase 11 pocket disappears or becomes naive/oracle-dominated at
the longer horizon. In that case, Phase 13 becomes a useful negative result and
the next project step should be controller redesign, not bigger sweeps.

## 6. Readback Template

After the full lock finishes, update [`PHASE13_RESULTS.md`](PHASE13_RESULTS.md)
with:

- command and wall-clock runtime
- total trial count and terminal outcome counts
- candidate-envelope count
- best-cell class balance
- favorable-pocket read
- boundary-cell read
- cost-rate read
- late-failure read
- claim wording to preserve, upgrade, or weaken
