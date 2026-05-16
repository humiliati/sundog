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

To be recorded here after the re-run smoke, before the full lock is interpreted:

- the locked `T_window` (largest grid `t* ≤ 4.8` meeting order ∈ [3,5],
  `max D(0.012) < 1e-6`, no early termination on any smoke `off` cell-seed;
  capped at 2.4)
- per-smoke-`off`-cell fitted-order evidence and coverage
- confirmation both hard-void gates still reproduce bit-for-bit after the
  additive sampler edit

The full lock (`npm run threebody:phase15`) remains **not run**, pending:
(1) operator lock review of the `PHASE15_SPEC.md` amendment, (2) additive
implementation of the in-`runTrial` sampler + `richardson-order-map.csv`,
(3) re-run of both hard-void gates + the smoke, (4) the locked `T_window`
recorded above, (5) operator readback sign-off.
