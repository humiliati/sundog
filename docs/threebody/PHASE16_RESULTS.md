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

Pending operator-staged run:

```powershell
npm run threebody:phase16:hazard
npm run threebody:phase16:analyze
```

Formal branch will be one of:

- (A) Hazard warnable
- (B) Warnability capped
- (C) Mixed / provisional
