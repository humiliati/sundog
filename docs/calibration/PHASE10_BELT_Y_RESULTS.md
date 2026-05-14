# Phase 10 Belt-Y Results

Result date: 2026-05-13

Spec: [`PHASE10_BELT_Y_SPEC.md`](PHASE10_BELT_Y_SPEC.md)

Machine artifacts:

- [`belt-y-residuals.csv`](../../results/calibration/phase10-belt-y/belt-y-residuals.csv)
- [`summary.json`](../../results/calibration/phase10-belt-y/summary.json)

Residual convention in this result note follows the spec:

```text
belt_y_residual_px = observed_parhelic_belt_y_px - predicted_belt_y_px
predicted_belt_y_px = sun_y + (-0.05 * R22_px)
```

Older calibration notes sometimes quote the script convention
`predicted - observed`. Under that convention, the sign is reversed.

## Classification

| prediction | verdict | reason |
| --- | --- | --- |
| FF1: low-h belt-y residual replicates as a pattern | **falsified** | Three photos reach `|residual| >= 5 px`, but they do not share a sign: p13 and p26 are negative, p25 is positive. Spearman `rho(h, residual) = +0.086`, inside the neutral `[-0.3, +0.3]` band. |
| FF2: residual is altitude-dependent | **not gated** | FF2 only triggers if FF1 confirms. |
| FF3: residual does not breach primitive threshold | **passes** | No anchored low-h photo reaches `|residual| >= 12 px`. |

Bottom line: p13's belt-y residual does not promote into a low-h rule.
The watch-list flag can be retired as photo-specific / anchor-local rather
than escalated into a primitive failure or a per-altitude correction.

## Residual Table

| photo | h inferred | R22 px | predicted belt y | observed belt y | residual px | residual / R22 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p27 | 0.00 | 219.0 | 548.0 | 548.0 | -0.0 | -0.0002 |
| p22 | 5.69 | 505.0 | 427.8 | 428.0 | +0.2 | +0.0005 |
| p13 | 6.81 | 211.0 | 361.4 | 351.0 | -10.4 | -0.0495 |
| p26 | 8.97 | 323.0 | 187.8 | 177.0 | -10.8 | -0.0336 |
| p30 | 11.15 | 650.0 | 901.5 | 899.0 | -2.5 | -0.0038 |
| p25 | 11.36 | 300.0 | 171.0 | 176.0 | +5.0 | +0.0167 |

## Notes

- p25 is a caveated robustness anchor. The right-side reference is cleaner
  than the foreground / flare-contaminated left side, so it should not be
  treated as a clean deciding vote by itself.
- p26 and p30 carry parhelic-tilt flags because their left/right parhelion
  y values differ by at least 5 px.
- The result remains useful even though FF1 falsifies: the pre-registered
  replication test prevented p13's single-photo residual from becoming a
  new atlas rule without support.

## Cascade

Allowed cascade under the spec:

- Update the belt-y row in `RICH_DISPLAY_OVERLAY_NOTES.md`.
- Retire the watch-list flag.
- Do not change inversion-route promotions, logo vocabulary, CZA/tangent
  verdicts, or the global `--parhelic-y-offset-r22` value.
