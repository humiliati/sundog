# Three-Body Phase 17 - Hazard-Aligned Counterfactual Results

Status: **complete 2026-05-29; formal branch: Hazard-directed mechanism
REJECTED.** Phase 17 is a mechanism audit only. It does not retune the
controller, alter the hazard label, or revise the locked Phase 15 Fail-Magnitude
verdict under any branch; verdict revision would require a fresh pre-registered
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
| 10 | mu1-v0p95 | 144 | 4.6 | 55 | 0/15 | 63·54·27 | 4 |
| 11 | mu0p3-v0p95 | 144 | 4.6 | 55 | 6/15 | 41·96·7 | 4 |
| 12 | mu0p01-v0p95 | 144 | 3.8 | 55 | 2/15 | 17·118·9 | 4 |

**Lock complete: 12/12 shards, 1,728 trials, 44/180 candidate envelope rows —
bit-identical to Phase 15C** (every shard's candidate count + outcome mix matches
15C exactly), confirming the audit instrumentation is non-perturbing. Per-shard
wall 3.8–7.6 min; ~22 min total at 3-concurrent.

**Frozen-slate check:** wave-1 candidate rows + outcomes are **identical to the
Phase 15C wave-1 shards** (15C: mu1 5/15·50·93·1; mu0.3 3/15·31·109·4; mu0.01
3/15·20·124·0). Confirms the `--hazard-counterfactual-audit` instrumentation is a
passive observer — dropping `--precision-receipts`/`--track-action-coupling` did
not change dynamics or candidate classification.

### §6 Branch: **Hazard-directed mechanism REJECTED**

Binding candidate-split read across all 1,728 trials (1,440 controlled paired
trials; `off` is the passive reference). Favorable pocket (v ≥ 1.05); each trial
joined to its `candidateEnvelope` flag; trials with zero eligible steps excluded.

#### Positive rate (hazardMarginEffect > 0) — mode × candidate × horizon

| mode | split | n | H1 | H4 | H8 | H16 | H32 |
|---|---|--:|--:|--:|--:|--:|--:|
| `guarded` | CAND | 189 | 0.738 | 0.738 | **0.738** | **0.738** | **0.737** |
| `signal_delay` | CAND | 95 | 0.922 | 0.921 | 0.920 | 0.917 | 0.910 |
| `guarded` | non | 24 | 0.709 | 0.709 | 0.708 | 0.707 | 0.701 |
| `signal_delay` | non | 118 | 0.879 | 0.879 | 0.879 | 0.880 | 0.880 |
| `signal_shuffle` | non | 213 | 0.919 | 0.919 | 0.920 | 0.921 | 0.923 |
| `action_shuffle` | non | 213 | 0.932 | 0.932 | 0.932 | 0.932 | 0.931 |
| `sign_flip` | non | 213 | 0.139 | 0.133 | 0.125 | 0.110 | 0.083 |

Mean margin effects are tiny and grow only with horizon divergence (guarded CAND
+1.5e-6 → +9.7e-5 over H1→H32; an ~0.01% perturbation on an O(1) margin).
`signal_shuffle` and `sign_flip` produce **no** candidate cells (non-split only);
the `sign_flip` comparison uses the favorable-pocket fallback per §6.

#### Branch criteria (§6, "supported" requires all four on guarded CAND)

1. ≥2/3 primary horizons `hazardMarginEffect > 0` — **✓** (all positive).
2. ≥2/3 primary horizons positive rate ≥ 0.60 — **✓** (0.738 flat).
3. guarded exceeds `signal_delay` by ≥0.10 positive rate at ≥2/3 — **✗ FAILS**:
   `signal_delay` (0.92) **exceeds** guarded (0.74) by ~0.18 at every horizon.
4. `sign_flip` non-positive or ≥0.10 below guarded — **✓** (negative effect;
   rate 0.08–0.13 ≪ 0.74).

Criterion 3 fails decisively, and the **rejected** branch's explicit trigger is
met: *signal_delay / shuffle arms match or exceed guarded TRACK at the primary
horizons.* `action_shuffle` (0.93) and `signal_shuffle` (0.92) also exceed
guarded on the non-split.

#### Subtype honesty (§5) — the whole signal is escape-margin

For guarded **and** delay candidate rows alike, at every horizon:
`escapePositiveRate = 1.000`, `closePositiveRate = 0.000`,
`hazardAvoidedRate = 0.0000`. The thrust **uniformly** raises escape margin and
**uniformly** lowers close-approach margin — it pushes the test particle inward.
The net `hazardMargin` positive rate (0.74 guarded / 0.92 delay) is therefore
*entirely* an escape-margin effect, diluted only by the fraction of steps where
the close boundary is the binding `min`. No first action flips a terminal hazard
within 32 steps in candidate cells (`hazardAvoided = 0`), so the horizon is too
short to register actual avoidance.

### Interpretation

Even with the **geometrically correct** observable that Phase 16 earned (the
hazard margin, replacing energy), the multi-step first-action counterfactual
**still cannot isolate guarded TRACK's mechanism**. The margin effect is (a) a
generic consequence of thrusting inward (escape-margin only — true of every
non-inverted arm), (b) negligible in magnitude, and (c) **lower** for guarded
TRACK than for the `signal_delay` and shuffle ablations. The shuffles score
highest yet produce no candidate cells (no survival); guarded produces the
candidate cells but scores below the mistimed/misdirected arms — the same
survival-≠-local-counterfactual dissociation that 15B/15C found with energy, now
reproduced with the right yardstick.

**The conclusion the chain has earned: guarded TRACK's survival advantage is not
a per-step first-action effect** — not in energy (15C) and not in the
hazard-aligned geometric margin (17). It is consistent with a holistic /
cumulative property (timing, policy structure) that a matched-continuation
single-action counterfactual cannot decompose.

**Phase 15 verdict preserved.** A rejected Phase 17 does not (and by the locked
§1/§6 cannot) promote Phase 15; the Fail-Magnitude verdict and the gravity claim
stand unchanged. Phase 16/16B repaired the **warning** side (radius warns);
Phase 17 shows the **counterfactual-mechanism** side remains genuinely
unexplained by per-step counterfactuals.

### Next registered move

Per §7, the rejected branch points away from further per-step counterfactuals.
The highest-value next move is the **radius-only / matched-duty controller
control**: does a controller gated solely on `radius` (the warnable channel)
reproduce guarded TRACK's survival envelope? If yes, the mechanism reduces to
radius-gated thrusting; if no, the survival edge is a genuinely multi-factor
policy effect. Either outcome would be a **new pre-registered phase** — Phase 17
closes the per-step-counterfactual line of inquiry.

### Hard-void gates (pre-lock)

### Hard-void gates (pre-lock)

Re-run 2026-05-29 after the core touch — **both byte-identical**:

- phase13 → 3,456 trials / 88·324 candidate rows / bounded·escape·close 1154·2030·272 ✓
- phase14 → 6,048 trials / 130·648 candidate rows / 1269·4616·163 ✓

The `--hazard-counterfactual-audit` path is additive and non-perturbing.
