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
