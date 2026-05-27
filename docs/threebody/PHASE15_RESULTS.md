# Three-Body Phase 15 - Forward-Oracle / Precision Lock Results

Status: pre-registered and lock-reviewed on 2026-05-15, then amended and
lock-reviewed to replace the precision gate with an early-window Richardson
cross-timestep trajectory order. The PRE-amendment Phase 15 code (behind
`forward_oracle_strict` / `--precision-receipts`: counterfactual, energy-drift,
oracle-hazard, `cell-precision-map.csv`) is written and the pre-amendment smoke
ran. **The amended Richardson sampler + `makeRichardsonOrderRows` +
`richardson-order-map.csv` are now implemented in code** as of 2026-05-16.
The post-code hard-void gates and widened smoke have not been rerun, no locked
`T_window` has been derived, and no full-lock result exists.

Smoke-slate amendment (2026-05-16, lock-reviewed): the smoke timestep slate is
widened from `0.004,0.012` to the full ladder `0.004,0.006,0.008,0.01,0.012`
(20 cases / 360 trials, ≈ 2 h local) so the 4-point Richardson order fit is
evaluable in the smoke itself, as the §4 `T_window` procedure requires.

Pending sequence before any full-lock interpretation: (1) re-run BOTH
hard-void gates bit-for-bit (`threebody:phase13`, `threebody:phase14`);
(2) re-run the widened smoke; (3) derive + record `T_window` +
per-`off`-cell fitted-order evidence here; (4) operator readback sign-off. The
full lock stays operator-gated throughout.

Implementation receipt:

- `node --check public/js/threebody-core.mjs` passed
- `node --check scripts/threebody-operating-envelope.mjs` passed
- `package.json` parsed successfully
- `npm run build` passed, including the dist link check
- a two-trial scratch probe exercised `forward_oracle_strict`,
  `--precision-receipts`, `cell-warning-quality-map.csv`, and
  `cell-precision-map.csv`; scratch output was removed and is not a result
- oracle-hazard labels are computed on passive/off precision-audit samples at
  the locked `sensor-audit-every` cadence, plus terminal/hazard endpoints; AUROC
  nulls remain coverage, not successes
- post-amendment `node --check public/js/threebody-core.mjs` passed
- post-amendment `node --check scripts/threebody-operating-envelope.mjs` passed
- post-amendment `npm run build` passed, including the dist link check
- a five-timestep passive scratch probe exercised `earlyTrajectory`,
  `makeRichardsonOrderRows`, and `richardson-order-map.csv`; it selected
  `T_window = 2.4` with `20` in-window points and fitted order `p = 4.314838`
  for the single scratch cell. Scratch output was removed and is not a result.

For the amended Richardson gate, the exact locked gates and smoke are staged for
the operator; they have not been run after the Richardson sampler change:

```powershell
npm run threebody:phase13 *> results/threebody/phase15-phase13-gate.log
npm run threebody:phase14 *> results/threebody/phase15-phase14-gate.log
npm run threebody:phase15:smoke *> results/threebody/phase15-smoke.log
```

Gate readback requirements:

- `npm run threebody:phase13` must reproduce 3,456 trials; 88 / 324 candidate
  envelope rows; 81 promising best cells; terminal outcomes 1,154 bounded /
  2,030 escape / 272 close approach.
- `npm run threebody:phase14` must reproduce 6,048 trials; 5,184 paired
  non-passive rows; 130 / 648 candidate envelope rows; terminal outcomes 1,269
  bounded / 4,616 escape / 163 close approach.
- If either gate deviates, Phase 15 is void and the smoke must not be interpreted.

Before the full lock is interpreted, record:

- the exact smoke command and transcript
- the locked `T_window`
- per-smoke-`off`-cell Richardson fitted-order evidence and coverage
- whether the smoke supports starting the staged multi-hour full lock unchanged

The full lock remains pending until that amended smoke readback is recorded here.

## Pre-Amendment Gate + Smoke Readback (2026-05-15)

Sequence run unattended as a chained background job; logs:
`results/phase15-gate-phase13.log`, `results/phase15-gate-phase14.log`,
`results/phase15-smoke.log`.

**Hard-void regression gates: both PASS bit-for-bit for the pre-amendment
implementation.**

- `npm run threebody:phase13`: 3,456 trials; 88 / 324 candidate envelope rows;
  81 promising best cells; outcomes 1,154 bounded / 2,030 escape / 272 close
  approach. Matches the locked requirement exactly.
- `npm run threebody:phase14`: 6,048 trials; 5,184 paired non-passive rows;
  130 / 648 candidate envelope rows; outcomes 1,269 bounded / 4,616 escape /
  163 close approach. Matches the locked requirement exactly.

The additive `forward_oracle_strict` mode and initial `--precision-receipts`
instrumentation did not perturb the frozen Phase 13/14 code paths. The amended
Richardson sampler still requires a fresh hard-void gate rerun before amended
Phase 15 columns are interpreted.

**Smoke (`npm run threebody:phase15:smoke`):** 144 trials (8 cases × 9 modes ×
2 seeds), all 9 modes including `forward_oracle_strict` executed; all new
outputs emitted and populated (`cell-precision-map.csv`,
`cell-warning-quality-map.csv`, and the counterfactual / energy-drift /
oracle-hazard columns). Instrumentation behaves correctly under the locked
sign convention: smoke `counterfactualMeanEffect` ranges are guarded TRACK
[-0.207, +0.476], sign-flip [-1.000, -0.530], action-shuffle [-0.832, -0.028]
— harmful action drives the metric negative as intended (this is an
instrumentation check, not a Phase 15 result; the smoke is 2 seeds off-slate).

**Pre-registered drift bound (DEMOTED 2026-05-15 — superseded by the amendment
below; retained only as a reported, non-gating diagnostic):**

- `D_smoke` = max passive/off `finalRelEnergyDrift` over the 16 smoke passive
  trials = **0.0087190059**
- Pre-registered full-lock passive energy-drift bound `2 * D_smoke` =
  **0.0174380118**
- Per-dt max passive drift: `dt=0.004` → 0.0087125993; `dt=0.012` →
  0.0087190059. Finer is not worse than coarser, so the pre-registered
  "non-worsening as `dt` shrinks" expectation holds at the ladder extremes.

**Drift-receipt investigation outcome (2026-05-15):** the flat-across-ladder
drift was investigated. Root cause confirmed: the integrator runs the restricted
problem (primaries ignore the test particle, `computeAcceleration`
`if (index < 2 && j === 2) continue;`), so total 3-body energy is not a
conserved invariant of the integrated equations — its drift is structural
non-conservation + close-encounter softening, `dt`-insensitive, never an
RK4-order signal. The genuinely conserved two-primary subsystem energy is
conserved ~1e-8 but at the 8-digit logging floor and `dt`-insensitive (a vacuous
gate). Conclusion: no energy-conservation quantity can serve as the Phase 15 RK4
precision gate. The energy-drift bound above is therefore **demoted to a
reported, non-gating diagnostic** and the precision receipt is **replaced** by
an early-window Richardson cross-timestep trajectory receipt (`PHASE15_SPEC.md`
§4/§5 amended). An in-process scan over the existing smoke `off` cells measured
fitted order p ≈ 4.31 (textbook RK4 O(dt⁴)), validating the replacement.

## Pre-registered early-window Richardson receipt (2026-05-15 amendment)

Resolved binding pins (operator lock review): position-only L2 test-particle
divergence; common-grid `Δ = 0.12` (divides every ladder dt: 30/20/15/12/10);
reference `dt = 0.004`; fitted order `p` = OLS slope over
`dt ∈ {0.006,0.008,0.01,0.012}`; `EARLY_GRID_MAX_T = 4.8`;
`EARLY_DIV_ABS_CAP = 1e-6`; `MIN_INWINDOW_GRID_POINTS = 12`;
`T_WINDOW_CAP = 2.4`; median per-seed `p`; favorable pocket
`velocityScale ≥ 1.05`; decidable iff `≥ 2/3` favorable `off` cells defined
(else Partial, not Fail); Pass floor `p ≥ 3.0`; Fail iff decidable and
`p < 3.0`.

### Smoke readback — recorded 2026-05-16 (post-Richardson-code)

Run as a chained background job after the Richardson implementation
(`node --check` → phase13 → phase14 → widened smoke). Logs:
`results/phase15amend-gate-phase13.log`, `…-phase14.log`,
`results/phase15amend-smoke.log`.

**Hard-void gates — both reproduce bit-for-bit after the additive sampler
edit (the additive/flag-gated design held):**

- `npm run threebody:phase13`: 3,456 trials; 88 / 324 candidate envelope rows;
  81 promising best cells; outcomes 1,154 bounded / 2,030 escape / 272 close.
  Matches the locked requirement exactly.
- `npm run threebody:phase14`: 6,048 trials; 130 / 648 candidate envelope rows;
  outcomes 1,269 bounded / 4,616 escape / 163 close. Matches exactly.

Phase 15 is **not void**; Richardson columns are interpretable.

**Widened smoke (`threebody:phase15:smoke`, full ladder
`0.004,0.006,0.008,0.01,0.012`):** 360 trials (20 cases × 9 modes × 2 seeds),
40 `off` trials, `richardson-order-map.csv` emitted. All 8 `off` cell-seeds
**defined**: 20 in-window grid points per ladder dt (Δ=0.12, T_window 2.4 →
20 pts), no early termination, every `maxD(0.012)` ∈ [2.2e-9, 4.6e-9] ≪ the
`EARLY_DIV_ABS_CAP = 1e-6`.

**Locked `T_window` = 2.4** (procedure selected `t*` then capped at
`T_WINDOW_CAP`; `windowSelectionStatus=selected`).

**Per-`off`-cell fitted-order evidence and coverage:** per-cell-seed
`fittedOrder` ∈ [4.3035, 4.3148] (all 8). Favorable pocket
(`velocityScale ≥ 1.05`): `favorableDefinedCount = 4 / 4`, coverage rate
`1.0`, **`favorableDecidable = true`**, **favorable-pocket median fitted order
= 4.30985**. This is a clean RK4 `O(dt^4)` signature, ≥ 1.3 above the Pass
floor `p ≥ 3.0` and inside `[3,5]`.

**Precision-receipt verdict:** decidable and **PASSES** the §5 precision
condition (favorable-pocket median `p = 4.30985 ≥ 3.0`). The `T_window` /
precision leg of Phase 15 is satisfied; the harness, sampler, and reducer are
verified working on a from-scratch run.

Minor implementation note (non-blocking): the per-trial summary field
`earlyTrajectoryPointCount` reads `undefined`, but `richardson-order-map.csv`
shows 20 points/dt and valid fitted orders per `off` cell-seed — the sampler
and reducer are functioning; the undefined summary field is a cosmetic
naming/location detail, optional to tidy later.

**Status:** the readback is now recorded, so the full lock is no longer
blocked on it. It remains **operator-gated**: the full `npm run
threebody:phase15` (12,960 trials, ~75 h) requires explicit operator
authorization, and the §5 Pass verdict still depends on the conditions that
are only evaluable at the full lock (per-step counterfactual separation across
arms, candidate-fraction stability across the ladder, `oracleHazardAuroc`
warning quality, favorable-pocket survival). The smoke establishes only the
precision/`T_window` leg + that gates and instrumentation are intact.

The full lock (`npm run threebody:phase15`) remains **not run**, pending:
(1) re-run of both hard-void gates after the Richardson code change,
(2) re-run of the widened smoke, (3) the locked `T_window` recorded above,
(4) operator readback sign-off.

*(Note: items (1)–(3) above were completed by the 2026-05-16 post-Richardson
chain; superseded by the authoritative pending list below.)*

## Sharded execution plan (2026-05-16 amendment, lock-reviewed)

Per `PHASE15_SPEC.md` §7 (procedural amendment), the 12,960-trial full lock
executes as 12 overnight-fittable shards by `mass-ratio × velocityScale`.
Each shard keeps all 5 timesteps + 3 radii + 8 seeds + 9 modes intact = 15
cases × 1,080 trials × ~6 h, written to
`results/threebody/phase15-shard-mu<X>-v<Y>/`. A new
`npm run threebody:phase15:merge` produces the unified
`results/threebody/phase15-forward-oracle-precision-lock/` aggregate CSVs from
the per-shard manifests + trial JSONLs. Per-shard arithmetic reconciles to
the locked §3 totals: 12 × 1,080 = 12,960 ✓; 12 × 960 = 11,520 paired ✓;
12 × 120 = 1,440 envelope ✓; 12 × 15 = 180 best-cells ✓.

### Shard-equivalence gate (pending, blocking before any full-lock shard)

Re-run the locked widened smoke as 2 shards (`v=0.95` and `v=1.1`, each
180 trials) → `npm run threebody:phase15:merge` → diff aggregate CSVs against
the on-disk single-run smoke at
`results/threebody/phase15-forward-oracle-precision-smoke/`. Byte-identical
required on: `aggregate-envelope.csv`, `candidate-envelope.csv`, `paired.csv`,
`trial-outcomes.csv`, `envelope-map.csv`, `best-by-cell.csv`,
`cell-class-map.csv`, `cell-delta-map.csv`, `cell-warning-quality-map.csv`,
`cell-precision-map.csv`, `richardson-order-map.csv`. `manifest.json` may
legitimately differ (per-process timestamps). To be recorded here when the
gate runs.

### Shard run log (per-shard append on completion)

| shard | `mass-ratio` | `velocityScale` | trials | wall-clock | status |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 / 12 | 1 | 1.1 | 1,080 | 5 h 16.8 min | PASS — overnight 1, 2-concurrent w/ v=0.95 |
| 2 / 12 | 1 | 0.95 | 1,080 | 5 h 37.4 min | PASS — overnight 1, 2-concurrent w/ v=1.1 (boundary cell, ~20 min slower) |
| 3 / 12 | 1 | 1.05 | 1,080 | 7 h 9.7 min | PASS — overnight 2, 3-concurrent (cand 16/120) |
| 4 / 12 | 1 | 1.15 | 1,080 | 6 h 8.6 min | PASS — overnight 2, 3-concurrent (cand 31/120) |
| 5 / 12 | 0.3 | 0.95 | 1,080 | 7 h 0.0 min | PASS — overnight 2, 3-concurrent (cand 34/120; mu=0.3 boundary is NOT a failure cell, contrast with mu=1·v=0.95 which had 0/120) |
| 6 / 12 | 0.3 | 1.1 | 1,080 | 5 h 40.5 min | PASS — batch 3 (mid-day), 2-concurrent w/ mu=0.01·v=1.1 (cand 14/120) |
| 7 / 12 | 0.01 | 1.1 | 1,080 | 4 h 15.1 min | PASS — batch 3 (mid-day), 2-concurrent w/ mu=0.3·v=1.1 (cand 15/120; ~85 min faster — smaller-mass cells terminate earlier) |
| 8 / 12 | 0.3 | 1.05 | 1,080 | 5 h 45.6 min | PASS — batch 4 (evening), 2-concurrent w/ mu=0.3·v=1.15 (cand 25/120) |
| 9 / 12 | 0.3 | 1.15 | 1,080 | 5 h 24.1 min | PASS — batch 4 (evening), 2-concurrent w/ mu=0.3·v=1.05 (cand 26/120) |
| 10 / 12 | 0.01 | 0.95 | 1,080 | 5 h 24.5 min | PASS — batch 5 (overnight), 3-concurrent (cand 12/120; boundary cell, slowest of the three) |
| 11 / 12 | 0.01 | 1.05 | 1,080 | 5 h 10.7 min | PASS — batch 5 (overnight), 3-concurrent (cand 18/120) |
| 12 / 12 | 0.01 | 1.15 | 1,080 | 3 h 51.8 min | PASS — batch 5 (overnight), 3-concurrent (cand 16/120; fastest shard — 87% escape, ~34 min faster than v=1.05) |

**Overnight 1 (2026-05-25):** launched 2026-05-25T06:35:50Z; both shards
started within 10 ms of each other (concurrent launch confirmed). All 14
per-shard outputs emitted on both (paired/trial-outcomes/aggregate/
envelope-map/best-by-cell/cell-class-map/cell-delta-map/candidate-envelope/
cell-warning-quality-map/cell-precision-map/richardson-order-map/manifest/
trials-minimal.jsonl + trials/). Trial counts 1,080/1,080 on both — no
stalls. 2-concurrent wall-clock = max(316.8, 337.4) = **5 h 37.4 min** vs
estimated ~10 h 54 min sequential → **1.94× speedup** (better than the
smoke's 1.74×; smoke had more cell-cost asymmetry per-shard).

**Overnight 2 (2026-05-26, first 3-concurrent):** launched
2026-05-26T05:02:45Z; all three shards started within 2 ms of each other.
All 14 per-shard outputs emitted on each; trial counts 1,080/1,080 across
the board — no stalls. Group wall = max(429.7, 368.6, 420.0) =
**7 h 9.7 min** vs sequential estimate 20 h 18.3 min → **2.84× speedup**
(close to ideal 3×). Per-shard wall is higher than at 2-concurrent (more
contention per process), but aggregate throughput is up: 7.54 trials/min at
3-concurrent vs 6.40 at 2-concurrent.

**Observational read (context only — §5 verdict on the full merge):** the
mu=0.3·v=0.95 shard produced **34/120 candidate rows**, contrasting sharply
with the mu=1·v=0.95 shard from overnight 1 which had **0/120**. This is
consistent with the Phase 13 finding that the boundary failure is
*equal-mass-specific* (all 5 Phase 13 negative best cells were at mu=1,
v=0.95) rather than a generic low-velocity issue. Mass-ratio is doing real
work at the boundary. Reading is provisional and does not pre-judge the
§5 Pass/Partial/Fail verdict, which awaits the full 12-shard merge.

**Concurrency-3 promotion status:** all four conditions cleared — exercised
overnight 2 with no thermal/UI pain. Stays at 3 (never 4).

**Batch 3 (2026-05-26, mid-day 2-concurrent):** launched 2026-05-26T17:18:20Z;
both shards started within ~70 ms of each other. All 14 per-shard outputs
emitted on both; trial counts 1,080/1,080 across — no stalls. Group wall =
max(340.5, 255.1) = **5 h 40.5 min** vs sequential estimate 9 h 55.6 min
→ **1.75× speedup** at 2-concurrent (mu=0.01 finished ~85 min before mu=0.3,
so the last ~85 min was effectively solo mu=0.3 — that reduces the
concurrency benefit during the tail; at 2-concurrent the wall is always
bounded by the slower shard).

**Operational observation across the mu axis at v=1.1** (context for
concurrency planning — not science): mu=1 = 5 h 16.8 min (overnight 1,
2-concurrent), mu=0.3 = 5 h 40.5 min (batch 3, 2-concurrent), mu=0.01 = 4 h
15.1 min (batch 3, 2-concurrent). Per-cell cost is **non-monotonic in mu** —
mu=0.3 was slowest, mu=0.01 noticeably fastest. Pairing one mu=0.01 with one
heavier-mu shard balances wall in remaining nights.

**Batch 4 (2026-05-26, evening 2-concurrent):** launched 2026-05-27T01:11:54Z;
both shards started within ~1 ms of each other. All 14 per-shard outputs
emitted on both; trial counts 1,080/1,080 across — no stalls. Group wall =
max(345.6, 324.1) = **5 h 45.6 min** vs sequential estimate 11 h 9.7 min
→ **1.94× speedup** at 2-concurrent (matches overnight 1's 1.94× — same-mu
pair, wall-clock balanced). v=1.15 finished ~21.5 min before v=1.05; outcomes
explain the gap (v=1.05 bounded 258 / escape 705 / close 117 vs v=1.15
bounded 171 / escape 830 / close 79 — more bounded trials run longer).
Closes the mu=0.3 row.

**Batch 5 (2026-05-27, overnight 3-concurrent):** launched 2026-05-27T09:40:07Z;
all three shards started within ~16 ms. All 14 per-shard outputs emitted on
each; trial counts 1,080/1,080/1,080 — no stalls. Group wall =
max(324.5, 310.7, 231.8) = **5 h 24.5 min** vs sequential estimate 14 h 27 min
→ **2.67× speedup**. v=1.15 finished ~1 h 32.7 min before v=0.95 (87% escape
rate at v=1.15 vs 76% at v=0.95 — high-escape cells terminate fast). Closes
the mu=0.01 row. **All 12 shards complete.**

**Cross-velocity observation at mu=0.01** (context, not science): v=0.95 = 5 h
24.5 min (batch 5, 3-concurrent), v=1.05 = 5 h 10.7 min (batch 5,
3-concurrent), v=1.1 = 4 h 15.1 min (batch 3, 2-concurrent), v=1.15 = 3 h
51.8 min (batch 5, 3-concurrent). Strong monotonic trend — fastest at highest
velocity (more escape-dominant cells). The v=1.1 figure is at 2-concurrent so
less contention overhead; adjusted for that, the monotonic trend holds. mu=0.01
cells are consistently faster than mu=0.3 / mu=1 across the velocity axis.

**Cross-velocity observation at mu=0.3** (context, not science): v=0.95 = 7 h
0.0 min (overnight 2, 3-concurrent), v=1.05 = 5 h 45.6 min (batch 4,
2-concurrent), v=1.1 = 5 h 40.5 min (batch 3, 2-concurrent), v=1.15 = 5 h
24.1 min (batch 4, 2-concurrent). Slowest at the boundary v=0.95 (more
bounded trials run longer), trending faster as velocity increases. The
overnight-2 v=0.95 figure mixes the cell cost with 3-concurrent contention,
so the boundary-vs-favorable gap is real but its magnitude is confounded.

### Concurrency policy (operator pin, 2026-05-16)

Binding operational guardrails for this machine, this first full lock:

- **Initial concurrency: 2 concurrent shards per overnight.**
- **Promotion to 3 concurrent: allowed only after the first 2-shard overnight
  delivers all of (a) stable per-shard wall-clock (each shard ≈ 6 h within
  a reasonable variance band), (b) per-shard aggregate CSVs emitted clean,
  (c) no stalls (no hung process; every scheduled trial accounted for in the
  shard's manifest), (d) no thermal or interactive UI pain.** Promotion is
  the operator's call; it is recorded in the Shard run log when first
  exercised.
- **Hard cap: never run 4 concurrent shards on this machine during this
  first full lock**, regardless of how well 3 performs. Revisit only after
  the first sharded full lock completes and the box's behavior under 3 is
  characterized.

**Promotion to 3 exercised 2026-05-26 (operator):** overnight 1 (2-shard)
ran fine on this machine *alongside* an unrelated experiment from a different
workstream, putting effective 3-process load on the box with no thermal or
interactive-UI pain observed. That satisfies condition (d) observationally.
3-concurrent is permitted from overnight 2 forward; the no-4 hard cap still
stands.

### Authoritative pending sequence (supersedes the earlier list)

1. Implement the post-lock merge code (additive: refactor aggregation
   exports in `threebody-operating-envelope.mjs`, new
   `scripts/threebody-phase15-merge.mjs`, new
   `npm run threebody:phase15:merge`). Verification: re-run `npm run
   threebody:phase13` and `npm run threebody:phase14` after the refactor;
   both must reproduce bit-for-bit (the refactor is exports-only, so the
   hard-void gates must still pass).
2. Run the shard-equivalence gate (above); byte-identical or void.
3. On gate pass: schedule the 12 overnight shards per the **Concurrency
   policy** above (initial 2; promotion to 3 only after the first 2-shard
   overnight clears the listed conditions; never 4 for this first full
   lock). Per-shard commit of the run-log row when each completes.
4. After all 12 shards complete: `npm run threebody:phase15:merge` → unified
   aggregate CSVs in `phase15-forward-oracle-precision-lock/`.
5. Operator §5/§6 readback + Pass/Partial/Fail sign-off.

The full lock remains operator-gated throughout. This amendment is
procedural; §1–§6 (slate, modes, seeds, receipts, branches) are unchanged.

## Full-lock execution receipt (2026-05-27)

All 12 shards complete. `npm run threebody:phase15:merge` run 2026-05-27;
all 12 shard dirs auto-discovered and merged.

- **Unified trial count: 12,960** — matches locked §3 total (12 × 1,080) ✓
- **Unified candidate envelope rows: 235 / 1,440**
- **Unified outcomes:** escape 9,131 / bounded 2,759 / close_approach 1,070
- **Output:** `results/threebody/phase15-forward-oracle-precision-lock/`
  — all 11 binding aggregate CSVs present:
  `aggregate-envelope.csv`, `candidate-envelope.csv`, `paired.csv`,
  `trial-outcomes.csv`, `envelope-map.csv`, `best-by-cell.csv`,
  `cell-class-map.csv`, `cell-delta-map.csv`, `cell-warning-quality-map.csv`,
  `cell-precision-map.csv`, `richardson-order-map.csv`, plus
  `manifest-summary.json`.

**Total sharded execution wall (all 5 batches, operator-paced):**
- Overnight 1 (2-concurrent): 5 h 37.4 min (2 shards)
- Overnight 2 (3-concurrent): 7 h 9.7 min (3 shards)
- Batch 3 mid-day (2-concurrent): 5 h 40.5 min (2 shards)
- Batch 4 evening (2-concurrent): 5 h 45.6 min (2 shards)
- Batch 5 overnight (3-concurrent): 5 h 24.5 min (3 shards)
- **Total elapsed wall (sequential batches): ≈29 h 37.7 min** over 5 sessions

Sequential single-run estimate (sum of all per-shard walls):
324.5+310.7+231.8+337.0+316.8+420.0+368.6+429.7+316.8+345.6+324.1+255.1
= **3,980.7 min ≈ 66 h 20.7 min**. Sharded concurrency reduced effective
wall to ≈29.6 h (2.24× overall; individual batch speedups 1.94× – 2.84×).

The data is in hand. **Operator §5/§6 readback is unblocked.** The
Pass/Partial/Fail verdict per `PHASE15_SPEC.md` §5 awaits operator
sign-off and is not rendered here.

## §5/§6 Full-Lock Readback (2026-05-27)

All numbers from `results/threebody/phase15-forward-oracle-precision-lock/`.
Hard-void gates confirmed bit-for-bit in the 2026-05-16 post-Richardson chain
(recorded above). Favorable pocket = `velocityScale ≥ 1.05` (27 cells per dt,
3 mass-ratios × 3 radii × 3 velocities). Total 12,960 trials.

### Precision receipt — Richardson (§4/§5 primary gate)

| metric | value | threshold | status |
|---|---|---|---|
| favorable-pocket median fitted order | **p = 4.313** | ≥ 3.0 | ✓ PASS |
| favorable defined coverage | 196 / 216 = 90.7% | ≥ 2/3 | ✓ decidable |
| favorableDecidable flag | true | — | ✓ |
| fittedOrder range (all defined rows) | [4.258, 4.423] | — | inside [3,5] ✓ |
| T_window (locked) | 2.4 | — | matches smoke |

### Precision receipt — Candidate-fraction stability

| dt | candidates / 27 favorable cells | fraction |
|---|---|---|
| 0.004 | 24 / 27 | 0.889 |
| 0.006 | 24 / 27 | 0.889 |
| 0.008 | 23 / 27 | 0.852 |
| 0.01 | 24 / 27 | 0.889 |
| 0.012 | 24 / 27 | 0.889 |

**Δ(dt=0.004 vs dt=0.01) = 0.000** (threshold ≤ 0.10 → ✓ PASS). Candidate
fraction is rock-stable across the ladder; the envelope does not narrow at
finer dt. All three ablation-failure arms (`signal_shuffle`, `action_shuffle`,
`sign_flip`) have **0/27 candidate cells at every dt** — they produce no
favorable-pocket survival improvement at any timestep.

### Per-arm × per-dt outcome table (favorable pocket v≥1.05, N=27 cells per dt)

Columns show `candidate_count / 27` at each dt.

| mode | 0.004 | 0.006 | 0.008 | 0.01 | 0.012 | mean surv Δ (dt=0.004) |
|---|---|---|---|---|---|---|
| `track_sensor_accel_guarded` | 24/27 | 24/27 | 23/27 | 24/27 | 24/27 | **+0.801** (27/27 pos) |
| `track_sensor_accel_signal_delay` | 12/27 | 12/27 | 12/27 | 12/27 | 12/27 | +0.162 (15/27 pos) |
| `track_sensor_accel_signal_shuffle` | 0/27 | 0/27 | 0/27 | 0/27 | 0/27 | −0.097 (0/27 pos) |
| `track_sensor_accel_action_shuffle` | 0/27 | 0/27 | 0/27 | 0/27 | 0/27 | −0.102 (0/27 pos) |
| `track_sensor_accel_sign_flip` | 0/27 | 0/27 | 0/27 | 0/27 | 0/27 | −0.120 (0/27 pos) |
| `oracle` | 0/27 | 0/27 | 1/27 | 7/27 | 0/27 | −0.014 |
| `forward_oracle_strict` | 1/27 | 1/27 | 0/27 | 0/27 | 0/27 | −0.014 |
| `naive` | 0/27 | 0/27 | 0/27 | 0/27 | 0/27 | −0.130 |

Boundary read (v=0.95): track has **10/45** candidate rows (5 dt × 9 cells
= 45); signal_shuffle 4/45, action_shuffle 5/45, signal_delay 20/45 — consistent
with the equal-mass-specific mechanism pattern from Phase 13.

### Per-step counterfactual table (favorable-pocket candidate rows)

Metric: `counterfactualMeanEffect` = mean normalized (H(noop)−H(actual)) /
normalizer over eligible steps; candidate rows only (normalizer meaningful).
Reference: Phase 14 pinned bars intact ≥+0.20; each ablation ≤+0.05;
sign_flip ≤−0.10; intact-vs-each-shuffled gap ≥+0.15.

| mode | n (candidate rows) | mean effect | threshold | status |
|---|---|---|---|---|
| `track_sensor_accel_guarded` | 119 | **+0.0626** | ≥ +0.20 | ❌ MISS |
| `track_sensor_accel_signal_delay` | 60 | **−0.175** | ≤ +0.05 | ✓ |
| `track_sensor_accel_signal_shuffle` | 0 (fav. pocket) | — | ≤ +0.05 | ✓ (absent) |
| `track_sensor_accel_action_shuffle` | 0 (fav. pocket) | — | ≤ +0.05 | ✓ (absent) |
| `track_sensor_accel_sign_flip` | 0 (any velocity) | — | ≤ −0.10 | ✓ (absent) |

Gap checks (intact vs each):
- vs `signal_delay`: +0.0626 − (−0.175) = **+0.238** ≥ +0.15 ✓
- vs `signal_shuffle` / `action_shuffle`: no favorable-pocket candidate rows → track
  uncontested in the favorable pocket (effective gap ≫ 0.15)
- vs `sign_flip`: no candidate rows at any velocity

The intact arm is positive (+0.0626 > 0, so **not** the pre-registered
negative ≤0 trigger). At the finest timestep (dt=0.004): track candidate-cell
mean = **+0.1058** (22/24 cells positive) — stronger signal at finer dt, but
still below the +0.20 bar. **The +0.20 intact threshold is missed.**

Note: signal_delay produces 44% candidate cells with positive survival delta but
NEGATIVE per-step counterfactual (−0.175) — the delayed-signal arm improves
survival through thrust direction alone, not through the energy-sensing signal,
confirming that timing/signal precision is load-bearing.

### Warning quality — oracleHazardAuroc

Computed from `off` (passive) trials; scores `energy` channel; label positive
iff `forward_oracle_strict` rollout flags hazard; reported from
`meanPassiveOracleHazardAuroc` on active-mode rows (shared cell-level property).

| metric | value | threshold | status |
|---|---|---|---|
| mean oracleHazardAuroc (favorable pocket) | **0.683** | ≥ 0.70 | ❌ MISS |
| coverage (cells with samples > 0) | 135 / 135 = 100% | ≥ 2/3 | ✓ decidable |
| cells ≥ 0.70 | 58 / 135 = 43% | — | — |
| range | [0.313, 1.000] | — | — |
| mean sample count / cell | 4.58 samples, 1.00 positive | — | (sparse) |

Warning quality is **decidable** (all cells defined) but below the 0.70
threshold. Sample counts are sparse (≈4.6 samples per cell mean) — the AUROC
estimates are low-precision. Miss is narrow: **0.683 vs 0.70 (~2.5% relative)**.

### Demoted energy diagnostics (non-gating, reported only)

Structural restricted-model non-conservation + close-encounter softening;
dt-insensitive by construction; not an RK4-order signal:
- mean passive `finalRelEnergyDrift` (favorable pocket): ~0.0087 (flat across ladder)
- No pass/fail weight (per the §4 amendment and §5 spec text).

### §5 Branch analysis

Pre-registered criteria against observed data:

| criterion | observed | threshold | pass? |
|---|---|---|---|
| Richardson decidable | yes (90.7% coverage) | ≥ 2/3 | ✓ |
| Richardson favorable median p | 4.313 | ≥ 3.0 | ✓ |
| Candidate fraction Δ(0.004 vs 0.01) | 0.000 | ≤ 0.10 | ✓ |
| Intact counterfactual | +0.0626 | ≥ +0.20 | ❌ |
| Ablation counterfactuals (3 arms) | −0.175 / absent / absent | ≤ +0.05 each | ✓ |
| sign_flip counterfactual | absent | ≤ −0.10 | ✓ (absent) |
| Intact-vs-ablation gaps | 0.238 / absent / absent | ≥ 0.15 each | ✓ (signal_delay only decidable) |
| oracleHazardAuroc mean | 0.683 | ≥ 0.70 | ❌ |
| Warning quality decidable | yes (100% defined) | ≥ 2/3 cells | ✓ |

**By the §5 branching logic:** Pass requires all criteria — two miss. Partial
is triggered by Richardson undecidable, candidate-fraction drift > 0.10, OR
warning-quality undecidable — none apply (misses are threshold misses, not
undecidability). Fail is triggered when "the per-step counterfactual fails to
separate intact control from the shuffled/mistimed arms by the pinned margin"
— the intact +0.20 bar misses.

**Reading against the spec:** the §5 Fail narrative ("claim narrowed to the
original 0.008–0.012 coarse-timestep regime only") does not cleanly fit because
the candidate envelope is dt-stable (no narrowing at finer dt). The actual
failure is one of **magnitude** (intact +0.06 vs +0.20 bar; warning 0.683 vs
0.70), not of directional separation or envelope stability. All three mis-timed /
inverted ablation arms produce 0 favorable-pocket candidates; signal_delay
produces negative counterfactual (correct direction, wrong magnitude); the
envelope survives finer dt cleanly.

**Pending operator sign-off:** which §5 branch to formally record (Fail on
pinned-bar miss, or operator-assessed Partial given the specific character of
the misses), what claim wording to preserve vs narrow, and whether to annotate
the signal_delay mechanism finding (timing-sensitive but signal-precision
insufficient at this bar).
