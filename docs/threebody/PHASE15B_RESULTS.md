# Three-Body Phase 15B - Counterfactual Normalizer Audit Results

Status: spec created 2026-05-27; additive harness receipts implemented behind
`--counterfactual-audit`. Smoke passed as an instrumentation check; lock pending.

Phase 15B is a diagnostic follow-up to the Phase 15 **Fail-Magnitude** verdict.
It does not revise Phase 15, retune the controller, or upgrade the claim.

## Smoke Readback

Run 2026-05-27:

```powershell
npm run threebody:phase15b:normalizer-smoke
```

Result:

- 6 trials written to `results/threebody/phase15b-normalizer-audit-smoke/`
- wall-clock ≈ 60 s
- all new counterfactual audit columns present in `paired.csv`,
  `trial-outcomes.csv`, and `aggregate-envelope.csv`
- candidate envelope rows: `0 / 5`; outcomes `{"bounded":5,"escape":1}`

Smoke aggregate read (one cell, duration 4; instrumentation sanity only):

| mode | eligible steps | floor-hit rate | floor-hit raw effect | floor-hit positive rate | floor-hit mean score |
|---|---:|---:|---:|---:|---:|
| `track_sensor_accel_guarded` | 506 | 1.000 | −5.35e-7 | 0.500 | 0.000 |
| `track_sensor_accel_signal_delay` | 360 | 1.000 | +6.67e-7 | 0.614 | +0.228 |
| `track_sensor_accel_signal_shuffle` | 248 | 1.000 | −3.97e-6 | 0.367 | −0.266 |
| `track_sensor_accel_action_shuffle` | 234 | 1.000 | −3.88e-6 | 0.380 | −0.239 |
| `track_sensor_accel_sign_flip` | 669 | 0.830 | −1.03e-5 | 0.000 | −1.000 |

The smoke confirms the audit instrument is live and that denominator-floor
collapse can be total in at least one Phase 15B cell: for TRACK, delay, and both
shuffle arms the raw oracle/no-op denominator is at the `1e-9` floor on every
eligible step. This is not a branch result because the smoke is one cell and
shorter than the lock, but it strongly supports staging the 1,728-trial lock
unchanged.

## Lock Readback

Pending:

```powershell
npm run threebody:phase15b:normalizer
```

The lock is operator-staged under the long-run rule. Formal branch will be one
of: Floor-collapse supported, Floor-collapse rejected, or Mixed / partial
diagnostic.
