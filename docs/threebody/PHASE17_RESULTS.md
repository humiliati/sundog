# Three-Body Phase 17 - Hazard-Aligned Counterfactual Results

Status: **spec locked 2026-05-29; implementation + smoke complete; lock pending
operator go.** Phase 17 is a mechanism audit only. It does not retune the
controller, alter the hazard label, or revise the locked Phase 15 Fail-Magnitude
verdict under any branch — verdict revision would require a fresh pre-registered
lock with the geometric metrics fixed in advance (spec §1, §6).

## Implementation Receipt

Additive + flag-gated (`--hazard-counterfactual-audit`); no oracle, no
`--precision-receipts`, no `--track-action-coupling` (audit observes; it does not
change dynamics, so candidate-envelope classification is unchanged):

- `public/js/threebody-core.mjs`: `hazardCounterfactualAudit` default;
  `hazardMargins` (reuses the exact `stateHasTerminalHazard` geometry);
  `computeHazardCounterfactualHorizon` (2-arm matched actual-vs-noop rollout,
  horizons `{1,4,8,16,32}`, raw terminal margin effect + escape/close
  decomposition + cumulative `hazardAvoided`); 11 trial-level fields × 5 horizons.
- `scripts/threebody-operating-envelope.mjs`: `--hazard-counterfactual-audit`
  flag + config wire; `hazardCf*` columns in `paired.csv` (55), `trial-outcomes.csv`
  (55), `aggregate-envelope.csv` (50, presence-guarded so phase13/14 byte-identical).
- `package.json`: `threebody:phase17:hazard-cf-smoke`, `threebody:phase17:hazard-cf`,
  `threebody:phase17:shard`; `scripts/threebody-phase17-shard.mjs` (12-shard
  mass×velocity wrapper).
- `node --check` passed on core, harness, and shard wrapper.

## Smoke Readback

Run 2026-05-29: `npm run threebody:phase17:hazard-cf-smoke`

- 6 trials (one cell, 6 modes, 1 seed, duration 4), wall ≈ 2.7 s
- all 55 `hazardCf` columns present in `paired.csv` and `trial-outcomes.csv`;
  50 aggregate columns in `aggregate-envelope.csv`
- candidate envelope rows 0/5 (single-seed benign cell); outcomes
  `{"bounded":5,"escape":1}`
- **sign sanity:** `sign_flip` is the most negative arm at every horizon
  (margin effect ↓ and positive rate 0.30–0.35) — confirms the sign convention
  (anti-oracle is worst), no inverted-sign bug. Magnitudes are tiny (~1e-5) in
  this benign non-candidate cell; `guarded` is near-zero/slightly negative
  (pos ≈ 0.44) and `signal_delay` positive (pos ≈ 0.68). **This is not a result**
  — it is the same non-candidate-cell noise the 15C smoke showed (the particle is
  far from any boundary, so the first-action margin perturbation is near-zero
  noise). The binding read is the candidate split over the full lock.

## Lock Plan (pending operator go)

- Rate probe: ≈ 0.45 s/trial at duration 4 → ≈ 1.8 s/trial at duration 16 →
  **≈ 52 min** for the full 1,728-trial lock — over the inline rule, so use the
  12-shard pattern (mass-ratio × velocity), ≈ 4–5 min/shard.
- Hard-void gates 13/14 re-run after the core touch (see below) must be
  byte-identical before any lock interpretation.
- Command: `npm run threebody:phase17:shard -- --mass-ratio <X> --velocity-scale <Y>`
  for each of the 12 cells, then aggregate at readback.

## Lock Readback

<!-- To be filled after the 12-shard lock: branch (per §6); candidate/non-candidate
mode×horizon tables; favorable-pocket guarded TRACK read at N=8,16,32;
guarded-vs-delay and guarded-vs-sign-flip separation (sign_flip via the
favorable-pocket fallback); escape-vs-close subtype diagnostic. -->

### Hard-void gates (pre-lock)

<!-- phase13 → 3,456 / 88·324 / 1154·2030·272; phase14 → 6,048 / 130·648 / 1269·4616·163 -->

