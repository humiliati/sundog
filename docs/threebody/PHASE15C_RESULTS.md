# Three-Body Phase 15C - Multi-Step Counterfactual Horizon Audit Results

Status: spec created 2026-05-28; implementation commit complete 2026-05-28.
Smoke passed as an instrumentation check. Operator-signed lock started
2026-05-29; shard readback is in progress below.

Phase 15C is a diagnostic follow-on to the Phase 15B **Mixed/Partial Diagnostic** verdict.
It does not revise Phase 15, retune the controller, or upgrade the claim.

## Smoke Readback

Run 2026-05-28:

```bash
npm run threebody:phase15c:multistep-smoke
```

Result:

- 6 trials written to `results/threebody/phase15c-multistep-counterfactual-smoke/`
- wall-clock ≈ 77 s (startedAt 10:56:40Z, completedAt 10:57:57Z)
- all 36 new horizon columns present in `paired.csv`, `trial-outcomes.csv`
  (9 fields × 4 horizons = 36 per trial)
- candidate envelope rows: 0 / 5; outcomes `{"bounded":5,"escape":1}`

### Smoke column-flow sanity table (one non-candidate cell, duration 4)

| mode | eligible steps | H4 pos rate | H4 score | H8 pos rate | H8 score | H32 pos rate | H32 score | H4 floor rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `track_sensor_accel_guarded` | 506 | 0.496 | −0.008 | 0.492 | −0.016 | 0.458 | −0.083 | 1.000 |
| `track_sensor_accel_signal_delay` | 360 | 0.603 | +0.206 | 0.594 | +0.189 | 0.514 | +0.028 | 1.000 |
| `track_sensor_accel_signal_shuffle` | 248 | 0.363 | −0.274 | 0.343 | −0.315 | 0.294 | −0.411 | 1.000 |
| `track_sensor_accel_action_shuffle` | 234 | 0.376 | −0.248 | 0.363 | −0.274 | 0.312 | −0.376 | 1.000 |
| `track_sensor_accel_sign_flip` | 669 | 0.000 | −1.000 | 0.000 | −1.000 | 0.000 | −1.000 | 0.830 |

**Observations (smoke-cell instrumentation check only — non-candidate cell, no branch weight):**

- All 36 horizon fields populated and populated correctly (mean_score = 2 × pos_rate − 1 when floor_rate = 1 ✓).
- Multi-step normalizer floor rate is 1.000 for TRACK and all shuffle arms across all horizons, same as Phase 15B. The terminal oracle/noop gap at horizon N remains at the 1e-9 floor even after 32 steps of continuation; the normalizer collapse is total at multi-step horizons in this cell.
- With floor_rate = 1, mean_score = 2 × positive_rate − 1 exactly (score collapses to sign). This is consistent with Phase 15B's one-step finding.
- TRACK positive rate and score decrease with increasing horizon (H4=0.496→H32=0.458; score H4=−0.008→H32=−0.083). This cell is non-candidate; candidate-cell behavior requires the lock.
- signal_delay positive rates are higher than TRACK at all horizons in this cell (0.603→0.514 vs 0.496→0.458), consistent with Phase 15B's non-candidate asymmetry observation.
- sign_flip is flat at −1.000 across all horizons (anti-oracle thrust dominates unconditionally regardless of horizon).

The smoke confirms the multi-step audit instrument is live. Column-flow check passes. Lock is staged.

## Lock Plan

### Rate probe

Smoke: 77 s wall for 6 trials at duration=4. Rate: **12.9 s / trial at duration=4**.

Extrapolated to duration=16 (4× more eligible steps): **~52 s / trial**.

Lock totals:

| approach | trials | estimated wall |
|---|---:|---|
| Monolithic | 1,728 | ~24.8 h |
| 12 shards × 1-concurrent | 144 / shard | ~2.1 h / shard (~24.8 h serial) |
| 12 shards × 4-concurrent | — | ~6.2 h elapsed |
| 12 shards × 6-concurrent | — | ~4.1 h elapsed |

### Shard command

```bash
node scripts/threebody-phase15c-shard.mjs --mass-ratio <X> --velocity-scale <Y>
# or: npm run threebody:phase15c:multistep -- --mass-ratio <X> --velocity-scale <Y>
```

Shard partition (12 total): `massRatio ∈ {0.01, 0.3, 1}` × `velocityScale ∈ {0.95, 1.05, 1.1, 1.15}`.
Each shard writes to `results/threebody/phase15c-shard-mu{X}-v{Y}/`.

### Resume / readback path

Per-shard outputs are independent CSVs in their own output dirs. Full readback
aggregates across all 12 shard dirs. Each shard produces `aggregate-envelope.csv`
stratified by `mode × candidateEnvelope × horizon` (via the standard harness summarizer).
No merge script required; per-shard aggregate tables can be combined by the operator.

### Operator-run plan

**Recommended:** 4-6 shards concurrent, staged as 2–3 overnight batches. Each
shard ~2.1 h → 2–3 batches completes the full lock in 2 overnights. The shard
wrapper (`threebody-phase15c-shard.mjs`) is the intended launch vehicle.

**HARD STOP:** do not start the lock until the operator signs off on this plan.

## Lock Readback

Operator sign-off 2026-05-29. Lock running as 12 shards in 4 waves of 3
(favorable pocket first). Actual per-shard wall is ~63–85 min (faster than the
~2.1 h smoke projection — eligible-step counts per cell ran lower than the
duration-4 smoke extrapolation).

### Shard run log

| # | shard | trials | wall (min) | candidate rows | outcomes (bounded·escape·close) | wave |
|--:|---|--:|--:|--:|---|:--:|
| 1 | mu1-v1p1 | 144 | 84.5 | 5/15 | 50·93·1 | 1 |
| 2 | mu0p3-v1p1 | 144 | 81.5 | 3/15 | 31·109·4 | 1 |
| 3 | mu0p01-v1p1 | 144 | 63.4 | 3/15 | 20·124·0 | 1 |
| 4 | mu1-v1p15 | 144 | 82.0 | 6/15 | 36·108·0 | 2 |
| 5 | mu0p3-v1p15 | 144 | 84.1 | 5/15 | 26·117·1 | 2 |
| 6 | mu0p01-v1p15 | 144 | 59.1 | 3/15 | 18·126·0 | 2 |

### Interim directional note (provisional — NOT a branch read)

Across the 3 completed v=1.1 shards, **trial-pooled** (not candidate-split)
per-horizon TRACK guarded mean score declines monotonically with horizon
(H4≈0.094 → H32≈0.028), i.e. no horizon lift on the pooled view. The shuffle
arms score *higher* on this pooled metric (≈0.45), which is the same
all-cells degeneracy Phase 15 identified: the 1e-9 normalizer floor collapses
in non-candidate cells, making shuffled arms appear positive. The binding read
is the **candidate-split** §6 readback after all 12 shards land — this note is
recorded only to track the run, not to assign a branch.

<!-- §6 branch readback to be filled after all 12 shards complete -->
