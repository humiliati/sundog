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

Operator go 2026-05-29. Running as 12 shards in 4 concurrent waves of 3
(favorable pocket first), per the locked spec.

### Shard run log

| # | shard | trials | wall (min) | hazardCf cols | candidate | outcomes (bnd·esc·close) | wave |
|--:|---|--:|--:|--:|--:|---|:--:|
| 1 | mu1-v1p1 | 144 | 7.6 | 55 | 5/15 | 50·93·1 | 1 |
| 2 | mu0p3-v1p1 | 144 | 7.2 | 55 | 3/15 | 31·109·4 | 1 |
| 3 | mu0p01-v1p1 | 144 | 5.7 | 55 | 3/15 | 20·124·0 | 1 |
| 4 | mu1-v1p15 | 144 | 5.4 | 55 | 6/15 | 36·108·0 | 2 |
| 5 | mu0p3-v1p15 | 144 | 5.2 | 55 | 5/15 | 26·117·1 | 2 |
| 6 | mu0p01-v1p15 | 144 | 3.8 | 55 | 3/15 | 18·126·0 | 2 |
| 7 | mu1-v1p05 | 144 | 6.0 | 55 | 3/15 | 61·83·0 | 3 |
| 8 | mu0p3-v1p05 | 144 | 5.7 | 55 | 5/15 | 41·96·7 | 3 |
| 9 | mu0p01-v1p05 | 144 | 5.1 | 55 | 3/15 | 24·120·0 | 3 |

**Frozen-slate check:** wave-1 candidate rows + outcomes are **identical to the
Phase 15C wave-1 shards** (15C: mu1 5/15·50·93·1; mu0.3 3/15·31·109·4; mu0.01
3/15·20·124·0). Confirms the `--hazard-counterfactual-audit` instrumentation is a
passive observer — dropping `--precision-receipts`/`--track-action-coupling` did
not change dynamics or candidate classification.

<!-- §6 candidate-split branch readback to be filled after all 12 shards land -->

### Hard-void gates (pre-lock)

### Hard-void gates (pre-lock)

Re-run 2026-05-29 after the core touch — **both byte-identical**:

- phase13 → 3,456 trials / 88·324 candidate rows / bounded·escape·close 1154·2030·272 ✓
- phase14 → 6,048 trials / 130·648 candidate rows / 1269·4616·163 ✓

The `--hazard-counterfactual-audit` path is additive and non-perturbing.

