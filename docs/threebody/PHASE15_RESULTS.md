# Three-Body Phase 15 - Forward-Oracle / Precision Lock Results

Status: pre-registered and lock-reviewed on 2026-05-15. Additive Phase 15 core
and harness code has been written behind `forward_oracle_strict` and
`--precision-receipts`; no Phase 15 smoke has run, and no full-lock result exists
yet.

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

Per the repository long-run rule, the exact locked gates and smoke are staged for
the operator rather than silently run by the agent:

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
- `D_smoke`, the maximum passive/off `finalRelEnergyDrift` observed in the smoke
- the pre-registered full-lock drift bound `2 * D_smoke`
- whether the smoke supports starting the staged multi-hour full lock unchanged

The full lock remains pending until that smoke readback is recorded here.

## Gate + Smoke Readback (2026-05-15)

Sequence run unattended as a chained background job; logs:
`results/phase15-gate-phase13.log`, `results/phase15-gate-phase14.log`,
`results/phase15-smoke.log`.

**Hard-void regression gates — both PASS bit-for-bit:**

- `npm run threebody:phase13`: 3,456 trials; 88 / 324 candidate envelope rows;
  81 promising best cells; outcomes 1,154 bounded / 2,030 escape / 272 close
  approach. Matches the locked requirement exactly.
- `npm run threebody:phase14`: 6,048 trials; 5,184 paired non-passive rows;
  130 / 648 candidate envelope rows; outcomes 1,269 bounded / 4,616 escape /
  163 close approach. Matches the locked requirement exactly.

The additive `forward_oracle_strict` mode and `--precision-receipts`
instrumentation did not perturb the frozen Phase 13/14 code paths. Phase 15
columns are interpretable.

**Smoke (`npm run threebody:phase15:smoke`):** 144 trials (8 cases × 9 modes ×
2 seeds), all 9 modes including `forward_oracle_strict` executed; all new
outputs emitted and populated (`cell-precision-map.csv`,
`cell-warning-quality-map.csv`, and the counterfactual / energy-drift /
oracle-hazard columns). Instrumentation behaves correctly under the locked
sign convention: smoke `counterfactualMeanEffect` ranges are guarded TRACK
[-0.207, +0.476], sign-flip [-1.000, -0.530], action-shuffle [-0.832, -0.028]
— harmful action drives the metric negative as intended (this is an
instrumentation check, not a Phase 15 result; the smoke is 2 seeds off-slate).

**Pre-registered drift bound:**

- `D_smoke` = max passive/off `finalRelEnergyDrift` over the 16 smoke passive
  trials = **0.0087190059**
- Pre-registered full-lock passive energy-drift bound `2 * D_smoke` =
  **0.0174380118**
- Per-dt max passive drift: `dt=0.004` → 0.0087125993; `dt=0.012` →
  0.0087190059. Finer is not worse than coarser, so the pre-registered
  "non-worsening as `dt` shrinks" expectation holds at the ladder extremes.

**Observation requiring an operator call before the full lock:** passive
relative energy drift is essentially **flat across the timestep ladder**
(≈0.00871 at both `dt=0.004` and `dt=0.012`) rather than shrinking with RK4
order (a true O(dt⁴) truncation receipt would differ by ≈81× between those
steps). The drift therefore appears dominated by a `dt`-insensitive source —
most likely close-encounter softening from the `r < 0.01` acceleration cutoff
and the `0.01` potential-energy cutoff near hazards — not integration order.
This does not void the pre-registered procedure (bound derived from the smoke;
finer-not-worse satisfied) and the harness is sound, but it means the full-lock
"precision lock" receipt would largely test softening robustness, not RK4
numerical order. How the precision receipt should be read at the 12,960-trial
full lock is an operator decision (proceed and interpret with this caveat
recorded; or investigate the drift metric / softening before committing the
multi-hour run).

The full lock (`npm run threebody:phase15`) remains **not run**, pending
operator go/no-go on the observation above.
