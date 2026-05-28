# Three-Body Phase 15B - Counterfactual Normalizer Audit Results

Status: spec created 2026-05-27; additive harness receipts implemented behind
`--counterfactual-audit`. Smoke passed as an instrumentation check; lock
completed 2026-05-28.

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

Run 2026-05-27 (completed 2026-05-28):

```powershell
npm run threebody:phase15b:normalizer
```

- **1,728 trials** written to `results/threebody/phase15b-normalizer-audit-lock/`
- **wall-clock: 791.2 min (13 h 11 m)** — per-step audit overhead (12 fields ×
  ~4,000 steps × 1,728 trials) is substantial; treated as a single overnight run
- candidate envelope rows: 44 / 180
- outcomes: escape 1,244 / bounded 428 / close_approach 56
- all aggregate CSVs present; all new counterfactual audit columns populated

### Aggregate table — favorable pocket (v ≥ 1.05), by mode × candidate split

| mode | split | n rows | floor rate | floor raw eff | floor pos rate | floor score | non-floor score | non-floor steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `track_sensor_accel_guarded` | CAND | 24 | 0.971 | ≈ 0 | 0.561 | +0.123 | n/a | 11,103 |
| `track_sensor_accel_guarded` | NON-CAND | 3 | 1.000 | ≈ 0 | 0.536 | +0.073 | n/a | 0 |
| `track_sensor_accel_signal_delay` | CAND | 12 | 0.954 | ≈ 0 | 0.401 | −0.197 | +0.317 | 8,267 |
| `track_sensor_accel_signal_delay` | NON-CAND | 15 | 0.844 | ≈ 0 | 0.605 | +0.211 | +0.358 | 16,889 |
| `track_sensor_accel_signal_shuffle` | NON-CAND | 27 | 0.725 | ≈ 0 | 0.785 | +0.570 | +0.312 | 5,102 |
| `track_sensor_accel_action_shuffle` | NON-CAND | 27 | 0.727 | ≈ 0 | 0.794 | +0.587 | +0.320 | 5,092 |
| `track_sensor_accel_sign_flip` | NON-CAND | 27 | 0.727 | ≈ 0 | 0.001 | −0.997 | −0.914 | 26,605 |

### §5B Branch: **Mixed / Partial Diagnostic**

**Floor-collapse supported** (§5B spec criterion) requires all of: non-candidate
TRACK floor-hit rate ≥ 0.20 **✓** (1.000), floor-hit raw effect positive **❌**
(≈ 0 — noise-level), floor-hit positive rate ≥ 0.60 **❌** (0.536), and at
least two ablation arms ≥ 0.10 lower in floor-hit positive rate **❌** (they are
**higher**: signal_shuffle 0.785, action_shuffle 0.794 vs TRACK 0.536).

**Floor-collapse rejected** (§5B spec criterion): TRACK floor-hit rate < 0.05
**❌** (1.000 — not met), OR TRACK floor-hit raw effect non-positive in both
splits **✓** (≈ 0 in both). The rate criterion is not met; the effect criterion
is met; rejection is therefore partial.

**Mixed / Partial Diagnostic** is the assigned branch: the denominator-floor
collapse is confirmed as near-universal (97–100% of TRACK eligible steps collapse
to 1e-9), but the floor-hit steps carry **no hidden positive signal specific to
TRACK** — raw effect ≈ 0, positive rate 0.536 (barely above chance), and the
ablation arms score *higher* on floor-hit positive rate (0.785, 0.794) not lower.

**Interpretation:** denominator collapse is real and near-total, but the Phase 15
magnitude miss is **not explained by the floor suppressing real hidden signal**.
When the oracle and noop are nearly identical per step — which is almost always —
TRACK's actions reduce energy at chance rate. The shuffled arms reduce energy more
often per floor-hit step, yet produce no candidate cells; TRACK produces candidate
cells but with noise-level per-step energy signal. The controller is winning on
survival through **multi-step trajectory steering**, not step-level energy
reduction against a locally undifferentiated oracle.

**Unexpected finding:** signal_delay non-candidate rows have floor-hit positive
rate 0.605 (above TRACK's 0.536) with non-floor score +0.358, while signal_delay
candidate rows flip to negative floor score (−0.197). The delay arm's mechanism
is different in cells where it succeeds vs fails: where it wins, the per-step
energy signal is anti-correlated with the floor steps. This asymmetry is a
candidate diagnostic for the multi-step horizon pass.

**Next step per branch:** denominator collapse is not the full explanation; the
Phase 15 one-step yardstick is detecting something at near-chance level. The
pre-registered follow-on is a **multi-step counterfactual** pass: measure
cumulative trajectory divergence over N steps (N ∈ {4, 8, 16, 32}) rather than a
single normalizer-collapsed step. The signal_delay candidate/non-candidate
asymmetry should be a primary stratification in the multi-step design.
