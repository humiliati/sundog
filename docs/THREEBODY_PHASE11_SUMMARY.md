# Three-Body Phase 11 Summary

Phase 11 tested whether the Phase 9 near-escape pocket survives guard-quantile
changes, nearby outside-pocket expansion, and matched comparison against simple
baselines.

## Runs

```bash
npm run threebody:phase11:guard-quantiles
npm run threebody:phase11:outside-pocket
npm run threebody:phase11:compare
```

Outputs:

- `results/threebody/phase11-guard-quantiles/`
- `results/threebody/phase11-outside-pocket/`
- `results/threebody/phase11-comparison/`

## Guard-Quantile Sweep

The guard-quantile sweep emitted 7,056 trials and 300 candidate envelope rows
out of 588.

By quantile:

| Guard quantile | Candidate rows | Avg survival delta | Avg mean delta-v | Read |
| --- | ---: | ---: | ---: | --- |
| `0.5` | 89 / 196 | 0.1165 | 0.5531 | Conservative, cheapest, many neutral cells. |
| `0.75` | 100 / 196 | 0.1565 | 1.1231 | Middle setting from Phase 9; still viable. |
| `0.9` | 111 / 196 | 0.2338 | 1.8192 | Strongest survival delta, highest effort. |

Best-cell selection favored `0.5` in 32 cells, `0.9` in 14 cells, and `0.75`
in 3 cells. This means the positive pocket is not an artifact of the single
Phase 9 quantile, but quantile choice controls the survival/effort trade.

## Outside-Pocket Expansion

The outside-pocket sweep emitted 6,912 trials and 157 candidate envelope rows
out of 432. Across 144 best cells:

- 96 promising;
- 36 mixed;
- 9 negative;
- 3 neutral.

Velocity remains the clearest boundary. Velocity scale `1.0` is mostly
promising, `0.95` is still mostly promising, and `0.85`/`0.9` contain most of
the mixed and negative cases.

Mass ratio also matters. Low mass ratio `0.01` is easiest, mass ratio `0.3` is
intermediate, and equal mass ratio `1` carries all 9 negative best cells.
Timestep did not change the broad picture: each tested timestep had 32
promising cells, with similar mixed/negative counts.

Failure mechanisms across controller rows:

- `controller_destabilized_or_shortened_passive`: 353;
- `control_effort_or_saturation`: 281;
- `unclassified_harm`: 5.

## Comparison Slate

The comparison run emitted 2,592 trials over the high-velocity near-escape
pocket and compared passive, naive local acceleration, guarded accelerometer
TRACK, and the privileged heuristic oracle.

Envelope summary:

| Controller | Candidate rows | Avg survival delta | Avg mean delta-v | Region classes |
| --- | ---: | ---: | ---: | --- |
| `naive` | 0 / 81 | -0.1343 | 0.7977 | 27 negative, 42 neutral, 12 risky |
| `oracle` | 34 / 81 | 0.3503 | 0.8270 | 34 promising, 15 mixed, 27 neutral, 4 negative, 1 risky |
| `track_sensor_accel_guarded` | 81 / 81 | 0.8056 | 1.7412 | 81 promising |

The oracle here is a privileged heuristic, not an optimal controller. The useful
result is that the guarded accelerometer-proxy controller dominates the naive
local baseline in this favorable pocket and remains consistently positive on
the matched slate, while spending more delta-v than the heuristic oracle.

## Current Claim

Phase 11 supports the stronger bounded claim:

> In the tested planar restricted setup, the guarded accelerometer-proxy TRACK
> controller improves survival over passive and naive local baselines in a
> robust high-velocity near-escape pocket. The result is not global: lower
> velocity and equal-mass boundary cells still expose controller harms, mostly
> through controller-shortened passive survival and control effort/saturation.

Do not generalize this to arbitrary three-body control, long-term chaos
prediction, or physical sensor validation.
