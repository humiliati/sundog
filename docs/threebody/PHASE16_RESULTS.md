# Three-Body Phase 16 - Hazard-Score Channel Audit Results

Status: spec locked 2026-05-29; additive implementation written after lock.
Phase 16 is a diagnostic hazard-score audit only. It does not revise Phase 15 /
15B / 15C, retune the controller, or upgrade the earned claim.

## Implementation Receipt

Verification:

- `--hazard-channel-audit` flag added to the operating-envelope harness
- passive-only `hazardSamples` receipt added behind the flag
- `scripts/threebody-phase16-hazard-audit.mjs` added for offline AUROC,
  trajectory-bootstrap CI, and fixed-fold logistic-combo analysis
- npm scripts added:
  - `npm run threebody:phase16:hazard-smoke`
  - `npm run threebody:phase16:hazard`
  - `npm run threebody:phase16:analyze`
  - `npm run threebody:phase16:analyze-smoke`
- `node --check public/js/threebody-core.mjs` passed
- `node --check scripts/threebody-operating-envelope.mjs` passed
- `node --check scripts/threebody-phase16-hazard-audit.mjs` passed
- `package.json` parsed successfully

## Smoke Readback

Run 2026-05-29:

```powershell
npm run threebody:phase16:hazard-smoke
npm run threebody:phase16:analyze-smoke
```

Result:

- smoke wall-clock: ≈ 2 s (`manifest` internal runtime ≈ 0.6 s)
- 1 passive trial written to
  `results/threebody/phase16-hazard-channel-audit-smoke/`
- `trials-minimal.jsonl` contains `hazardSamples`
- 6 hazard samples emitted; 0 positive labels in this capped smoke
- first sample includes all 10 locked channels: `energy`, `kineticEnergy`,
  `potentialEnergy`, `virial`, `inertia`, `tidalMagnitude`,
  `localAccelerationMagnitude`, `radius`, `minPrimaryDistance`, `speed`
- analysis wrote `hazard-channel-audit.csv` and
  `hazard-channel-audit-manifest.json`
- smoke analysis branch is `C_mixed_provisional` only because the one-trial smoke
  has one label class (0 positives); this is an expected estimator-sanity
  outcome, not a Phase 16 result

Estimated lock cost from the smoke is low enough to run locally but above the
inline-agent rule once scaled to 288 duration-16 passive trials; stage it for the
operator unchanged.

## Lock Readback

Run 2026-05-29:

```powershell
npm run threebody:phase16:hazard     # 288 passive trials, dt=0.004, wall 1.5 min
npm run threebody:phase16:analyze
```

- **288 passive (`off`) trials**, hard-void gates re-confirmed byte-identical
  before the run (phase13 88/324·1154·2030·272; phase14 130/648·1269·4616·163).
- Favorable pocket (v ≥ 1.05): **216 trajectories, 1,460 hazard samples, 214
  positive (14.7%)** — far denser than Phase 15's ~1 positive/cell, so the pooled
  estimator is well-defined. Full grid: 2,146 samples, 274 positive (12.8%).
- Trajectory cluster-bootstrap fully decidable: radius and energy both
  2,000/2,000 valid resamples.
- Outcome mix: **escape 235 / close_approach 5 / bounded 48** — the hazard label
  is ~98% the *escape* subtype on this slate (see OR-label caveat below).

### §6 Branch: **(A) Hazard warnable**

Favorable-pocket pooled directional AUROC (energy baseline first; pre-registered
sign in the `dir` column; pass = lower 95% CI ≥ 0.70):

| channel | dir | AUROC | 95% CI | pass | discrim. | per-cell | full-grid |
|---|:--:|--:|---|:--:|--:|--:|--:|
| `energy` (baseline) | + | 0.724 | [0.681, 0.761] | ✗ | 0.724 | 0.656 | 0.746 |
| `radius` | + | **0.995** | **[0.985, 1.000]** | **✓ PASS** | 0.995 | 0.997 | 0.978 |
| `virial` | + | 0.660 | [0.617, 0.699] | ✗ | 0.660 | 0.491 | 0.681 |
| `inertia` | + | 0.499 | [0.458, 0.546] | ✗ | 0.501 | 0.997 | 0.459 |
| `speed` | + | 0.407 | [0.364, 0.446] | ✗ | 0.593 | 0.233 | 0.399 |
| `kineticEnergy` | + | 0.274 | [0.239, 0.309] | ✗ | 0.726 | — | 0.259 |
| `potentialEnergy` | − | 0.190 | [0.162, 0.221] | ✗ | 0.810 | — | 0.180 |
| `localAccelerationMagnitude` | + | 0.041 | [0.031, 0.053] | ✗ | 0.959 | — | 0.054 |
| `tidalMagnitude` | + | 0.028 | [0.020, 0.040] | ✗ | 0.972 | — | 0.043 |
| `minPrimaryDistance` | − | 0.022 | [0.013, 0.033] | ✗ | 0.978 | — | 0.038 |

Fitted L2-logistic combo (held-out, grouped CV by seed): **held-out AUROC 0.930
[0.914, 0.946], PASS**; in-sample 0.950; **overfit gap 0.020** (healthy);
converged on all folds.

### Interpretation

**Energy was the wrong channel — confirmed.** Under the Phase-16 pooled estimator
energy scores 0.724 with lower CI **0.681 < 0.70** — it misses, reproducing
Phase 15's 0.683. The instantaneous *energy* yardstick genuinely cannot warn at
the bar.

**`radius` clears the bar overwhelmingly (0.995, lower CI 0.985).** A single
instantaneous geometric observable predicts the oracle's 32-step hazard call
near-perfectly. This is **Branch A**: instantaneous hazard warning *is* achievable.

**Honest caveat — the win is escape-subtype-skewed (the locked §4 OR-label caveat,
made manifest).** The hazard label is `r₃ > 4` **OR** `minPrimaryDistance < 0.08`,
and on this near-escape slate the escape subtype dominates 235:5. `radius` is
essentially the escape half of the label evaluated at the present step, so its
near-1.0 AUROC is strong but close to geometrically tautological for the dominant
subtype. It is **not** target leakage (radius is a legitimate instantaneous
observable, not a future value or an oracle output), but the result should be read
as *"current radius forecasts imminent escape,"* not as a deep nonlinear
discovery. The close-approach subtype (5 cases) is too rare to audit on its own;
per the locked spec no subtype relabeling or subtype branch is taken here.

**Sign-misregistration worked as designed.** Five "distance-from-primary" proxies
(`minPrimaryDistance`, `tidalMagnitude`, `localAccelerationMagnitude`,
`potentialEnergy`, `kineticEnergy`) are strongly predictive in the *reverse* of
their pre-registered direction (discriminability 0.73–0.98) — because they encode
proximity to a primary, and the dominant hazard is *escape* (far from both
primaries), not close approach. They are flagged, not counted as pre-registered
passes (a deployable warning must commit to a direction a priori). This is
consistent evidence for the escape-dominance reading.

**The fitted combo passes but does not beat `radius` alone** (0.930 < 0.995): L2
regularization + standardization dilute radius's near-saturated signal with
less-informative channels. The pass is driven by a clean single channel, so it is
**not** a fitted-only/provisional pass.

### Next registered move (per Branch A)

The follow-on **warning-quality re-pose with `radius` as the hazard score** is
complete as [`PHASE16B_RESULTS.md`](PHASE16B_RESULTS.md). It formalizes Phase
16's per-cell column in Phase 15's verdict shape: `radius` clears at `0.996624`
with `27/27` favorable cells defined, while the within-16B energy baseline misses
at `0.655508`. This is a *separate* warning-instrument repair and **does not
upgrade Phase 15**: warnability (can an instantaneous observable forecast the
oracle's hazard call?) is orthogonal to the controller's survival *mechanism*,
which Phase 15C showed is not a multi-step energy-steering effect. Phase 16
establishes that the Phase 15 warning miss was an **instrument choice** (energy),
not a structural limit - but the gravity claim and the Phase 15 Fail-Magnitude
verdict are preserved unchanged.
