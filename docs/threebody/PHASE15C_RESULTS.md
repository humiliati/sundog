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
| 7 | mu1-v1p05 | 144 | 95.1 | 3/15 | 61·83·0 | 3 |
| 8 | mu0p3-v1p05 | 144 | 91.3 | 5/15 | 41·96·7 | 3 |
| 9 | mu0p01-v1p05 | 144 | 84.5 | 3/15 | 24·120·0 | 3 |
| 10 | mu1-v0p95 | 144 | 107.4 | 0/15 | 63·54·27 | 4 |
| 11 | mu0p3-v0p95 | 144 | 109.8 | 6/15 | 41·96·7 | 4 |
| 12 | mu0p01-v0p95 | 144 | 92.3 | 2/15 | 17·118·9 | 4 |

**Lock complete: 12/12 shards, 1,728 trials, 44/180 candidate envelope rows**
(matches Phase 15B's 44/180 exactly — same slate, same candidate classification).
Per-shard wall 59–110 min; total ~17.4 core-hours across 4 waves of 3-concurrent.
All shards verified: 144 trials and 36 horizon columns each.

### §6 Branch: **Multi-step steering REJECTED**

Binding candidate-split read across all 1,728 trials (1,440 paired controlled
trials; `off` is the passive reference). Each trial joined to its
`candidateEnvelope` flag via `(mode, massRatio, dt, radiusScale, velocityScale)`;
per-horizon metrics from `paired.csv`; trials with zero eligible steps excluded.

#### Mean normalized score — mode × candidate × horizon

| mode | split | n | H4 | H8 | H16 | H32 |
|---|---|--:|--:|--:|--:|--:|
| `guarded` | CAND | 205 | **+0.100** | +0.091 | +0.072 | **+0.032** |
| `signal_delay` | CAND | 127 | −0.174 | −0.185 | −0.207 | −0.248 |
| `signal_shuffle` | CAND | 8 | −0.095 | −0.103 | −0.131 | −0.153 |
| `action_shuffle` | CAND | 8 | +0.064 | +0.061 | +0.041 | +0.008 |
| `guarded` | non | 78 | +0.056 | +0.048 | +0.033 | +0.001 |
| `signal_delay` | non | 155 | +0.110 | +0.103 | +0.089 | +0.061 |
| `signal_shuffle` | non | 275 | +0.422 | +0.418 | +0.411 | +0.396 |
| `action_shuffle` | non | 275 | +0.435 | +0.432 | +0.425 | +0.409 |
| `sign_flip` | non | 283 | −0.969 | −0.970 | −0.971 | −0.972 |

#### Positive rate — mode × candidate × horizon

| mode | split | n | H4 | H8 | H16 | H32 |
|---|---|--:|--:|--:|--:|--:|
| `guarded` | CAND | 205 | **0.550** | 0.546 | 0.536 | **0.516** |
| `signal_delay` | CAND | 127 | 0.424 | 0.419 | 0.408 | 0.388 |
| `action_shuffle` | CAND | 8 | 0.542 | 0.540 | 0.531 | 0.514 |
| `guarded` | non | 78 | 0.526 | 0.522 | 0.514 | 0.498 |
| `signal_delay` | non | 155 | 0.574 | 0.571 | 0.564 | 0.551 |
| `signal_shuffle` | non | 275 | 0.747 | 0.746 | 0.742 | 0.735 |
| `action_shuffle` | non | 275 | 0.754 | 0.752 | 0.749 | 0.742 |
| `sign_flip` | non | 283 | 0.005 | 0.004 | 0.004 | 0.004 |

Guarded-TRACK normalizer floor rate on candidate cells is **0.970 at every
horizon** (4 → 32). The multi-step oracle reference is therefore as degenerate
as the one-step: on 97% of eligible steps the oracle/no-op terminal-energy gap
stays below `1e-9` even after 32 frozen-continuation steps, so the normalized
score collapses to the sign statistic (`score ≈ 2·posRate − 1`, confirmed by the
two tables).

#### Branch criteria (§6 spec)

- **Multi-step steering supported** requires TRACK candidate rows to show a
  *monotone or material horizon lift* from N=4 to ≥1 of {N=16, N=32}. **FAILS:**
  guarded-CAND declines monotonically at every horizon — score +0.100 → +0.032,
  positive rate 0.550 → 0.516. The multi-step edge *erodes* toward chance, the
  opposite of lift.
- **Multi-step steering rejected:** "TRACK candidate rows remain near chance or
  *do not improve with horizon*." **MET:** no horizon lift exists for any arm;
  guarded-CAND's small above-chance edge at N=4 decays with N.
- **Mixed / partial:** "lift appears only in non-candidate / only in signal_delay
  / only at one horizon." Not applicable — there is *no* horizon lift anywhere to
  localize.

**Assigned branch: Multi-step steering REJECTED.** Phase 15's survival pocket is
**not** explained by cumulative trajectory steering over 4–32 step horizons. The
energy counterfactual is no less local at multi-step than at one-step; whatever
makes guarded TRACK win on survival does not register as a compounding
energy-reduction effect in this counterfactual family.

#### Ablation separation (survival result remains real & sensitive)

On candidate cells, guarded TRACK is the **best-scoring arm at every horizon**
(+0.100/+0.091/+0.072/+0.032; positive rate 0.550→0.516), clearly separated from
the ablations: `signal_delay`-CAND is *negative* (−0.174 → −0.248), and both
shuffles barely produce candidate cells at all (8 candidate trials each vs 205
for guarded). So Phase 15's signal/timing-sensitivity finding is intact — TRACK
*does* carry a candidate-specific signal; it simply does **not** grow with
horizon. The rejection is specifically of the *horizon-lift / multi-step-steering*
hypothesis, not of TRACK's survival edge.

#### Signal-delay asymmetry (elevated primary diagnostic — reproduced at multi-step)

The Phase 15B one-step tell reproduces at every multi-step horizon:

- `signal_delay` **CAND** (n=127): **negative** score (−0.174 → −0.248), positive
  rate **below chance** (0.424 → 0.388).
- `signal_delay` **non** (n=155): **positive** score (+0.110 → +0.061), positive
  rate **above chance** (0.574 → 0.551).

Where the delay arm *succeeds* (produces candidate cells) its multi-step energy
signal is anti-correlated; where it *fails* it is positive. The delay arm "buys
survival while scoring negative" — confirmed now over 4–32 step horizons, not just
one step. This is the strongest evidence that the survival mechanism is
orthogonal to the energy counterfactual yardstick.

#### Non-candidate control

Non-candidate shuffle arms score high (+0.42/+0.43, positive rate ~0.75) — the
all-cells floor-collapse degeneracy Phase 15 flagged — yet produce essentially no
candidate cells (8/283 and 8/283). Non-candidate guarded TRACK is modest and also
declines (+0.056 → +0.001). The horizon-lift pattern is absent in both splits, so
the non-candidate control is satisfied: there is no candidate-specific multi-step
mechanism to contrast against.

### Next registered move

Per the **rejected** branch, the next move is **not** another counterfactual
variant. The pre-registered options are: (i) **hazard-score audit** —
Phase 15's oracle-hazard AUROC missed the 0.70 bar (favorable-pocket mean 0.683);
re-examine the hazard score family directly; (ii) **event-warning-quality rerun**;
or (iii) treat the gap as a **controller-design limitation** and pursue a
mechanism that the energy counterfactual *can* see. Given that both the one-step
(15) and multi-step (15C) energy counterfactuals are normalizer-degenerate (~97%
floored) on the cells where TRACK wins, the energy-counterfactual yardstick is
likely the wrong instrument for this controller, and the hazard-score audit (i) is
the highest-value next pass.

**Phase 15 formal verdict (Fail-Magnitude) is preserved.** Phase 15C is a
mechanism diagnostic; it does not convert Phase 15 to Pass, retune the controller,
or broaden the gravity claim.
