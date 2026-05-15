# Three-Body Phase 13 - Longer-Horizon Result Note

This document records Phase 13 results for
[`PHASE13_SPEC.md`](PHASE13_SPEC.md). Phase 13 asks whether the Phase 11
positive pocket survives a longer rollout, or whether guarded accelerometer
TRACK mostly delays failures inside the original 8-second horizon.

Status: smoke run and full lock complete.

Cross-doc verification (2026-05-15): result trail confirmed consistent across
`SUNDOG_V_THREEBODY.md` (claim ladder, rollup, receipt), `threebody-writeup.md`,
and `CROSS_SUBSTRATE_NOTES.md`. Metrics (3,456 / 88·324 / 81·108 / 100·108 /
0.229 vs 0.218) agree on every surface; no Phase 12 bleed or transposition.
Reconciled the Phase 13 claim-ladder article in `SUNDOG_V_THREEBODY.md` and
added the pre-registered-branch verdict to `threebody-writeup.md`; low-velocity
boundary negatives carried in `CROSS_SUBSTRATE_NOTES.md` as designed (no numeric
trail expected there).

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

## 3. Smoke Claim Impact

The smoke alone did not justify upgrading the public claim. It supported
running the full Phase 13 lock because the high-velocity larger-radius pocket
still existed at 16 seconds, but it also confirmed that horizon extension makes
the failure boundary harder, not softer.

Current earned wording remains:

> In the tested planar restricted setup, a guarded accelerometer-proxy TRACK
> controller improves survival over passive and naive local baselines in a
> robust high-velocity near-escape operating pocket.

The full lock below supplies the ratchet test: the pocket must remain positive
across mass ratio, timestep, and nearby high-velocity cells at the longer
horizon, without a large delta-v-per-second or late-failure penalty.

## 4. Full Lock Readback

Command:

```bash
npm run threebody:phase13
```

Slate: 3 mass ratios x 3 timesteps x 3 radii x 4 velocity scales x 4 modes x 8
seeds at `duration=16`, for 3,456 trials.

Output:

- `results/threebody/phase13-long-horizon-lock/`
- 3,456 trials
- Terminal outcomes from the runner: 1,154 bounded, 2,030 escape, 272 close
  approach
- `trial-outcomes.csv` contains 2,592 paired non-passive rows: 864 each for
  naive, guarded TRACK, and oracle
- Candidate envelope rows: 88 / 324
- Candidate rows by mode: guarded TRACK 77, oracle 11, naive 0
- Best-cell class balance: 81 promising, 17 mixed, 5 risky, 5 negative
- Best-cell controller: guarded TRACK in 100 / 108 cells, oracle in 8 / 108

## 5. Full Lock Read

Phase 13 passes the long-horizon lock for the scoped high-velocity pocket. The
result is not merely an 8-second delay: guarded accelerometer TRACK still
dominates the favorable band at `duration=16`.

Velocity-class read:

- `velocityScale=1.05`: 27 / 27 best cells are promising.
- `velocityScale=1.15`: 27 / 27 best cells are promising.
- `velocityScale=1.1`: 19 / 27 best cells are promising, 8 / 27 are mixed, and
  none are risky or negative.
- `velocityScale=0.95`: 8 / 27 promising, 9 / 27 mixed, 5 / 27 risky, and
  5 / 27 negative. This remains the long-horizon boundary.

Mode read:

- Naive local acceleration remains a failed baseline: 0 candidate rows and 0
  bounded paired trials out of 864, while its matched passive rows bounded 144
  times.
- Guarded TRACK bounded 749 / 864 paired trials; its matched passive rows
  bounded 144 / 864.
- Oracle bounded 261 / 864 paired trials; useful as a privileged heuristic
  reference, but still not an optimal controller.

Cost-rate read:

- Guarded TRACK candidate rows have mean delta-v `3.665` over 16 seconds, or
  about `0.229` delta-v per simulated second.
- Phase 11 guarded TRACK candidate rows had mean delta-v `1.741` over 8
  seconds, or about `0.218` delta-v per simulated second.
- The total control cost roughly doubles with horizon, but the cost rate does
  not materially explode.

Late-failure read:

- Guarded TRACK has 29 / 864 paired trials that survive past 8 seconds but fail
  before 16 seconds.
- Of those late failures, 17 are
  `controller_destabilized_or_shortened_passive`, 6 are
  `control_effort_or_saturation`, and 6 have no classified mechanism.
- This is not a large late-failure cliff; most guarded TRACK failures still
  occur before the original horizon or in the known low-velocity boundary.

Boundary read:

- All 5 negative best cells occur at `velocityScale=0.95` and mass ratio `1`.
- The `velocityScale=0.95` band also contains all risky best cells.
- The long-horizon result therefore strengthens the positive high-velocity
  pocket while preserving the low-velocity/equal-mass warning boundary.

## 6. Claim Impact

Pre-registered branch: **pass, with boundary sharpening**.

Updated earned wording:

> In the tested planar restricted setup, a guarded accelerometer-proxy TRACK
> controller improves survival over passive and naive local baselines across a
> mapped high-velocity near-escape pocket through a 16-second tested horizon.
> The result is not global: the low-velocity boundary, especially equal-mass
> cells near `velocityScale=0.95`, still exposes controller harms.

Do not upgrade this to indefinite stability, orbit-family maintenance, or
general three-body control. Phase 13 only supports a longer tested horizon in a
restricted planar operating envelope.
