# Three-Body Phase 16 - Hazard-Score Channel Audit Spec

Phase 15 ended **Fail-Magnitude**. Two diagnostics followed: Phase 15B
(normalizer audit) and Phase 15C (multi-step counterfactual — **steering
rejected**). Both showed the energy counterfactual is normalizer-degenerate
(~97% floored) on exactly the cells where guarded TRACK wins. Phase 15C's
pre-registered next move is the **hazard-score audit**: in Phase 15 the
oracle-hazard warning quality missed its bar — favorable-pocket mean
`oracleHazardAuroc` = **0.683 vs the ≥ 0.70 target**.

Phase 16 is that audit. It is motivated by a structural mismatch: the hazard
**label** is a purely *geometric* escape / close-approach event
(`stateHasTerminalHazard`: `r₃ > escapeRadius (4)` **OR**
`minPrimaryDistance < closeApproachRadius (0.08)`), but the **score** tested in
Phase 15 was instantaneous **energy** — an indirect thermodynamic proxy.
Instantaneous observables dimensionally closer to the label (radius,
min-primary-distance, speed) were never tested. Phase 16 asks: **does any
instantaneous observable — single channel, or a fitted combination — predict the
strict oracle's 32-step `hazardReached` label well enough to clear 0.70?**

Phase 16 does **not** revise the Phase 15 / 15B / 15C verdicts, retune the
controller, change guard thresholds, or upgrade the earned claim. It is a
mechanism / instrument diagnostic only. A positive read supports a follow-on
warning-quality re-pose with the winning channel; it cannot by itself convert
Phase 15 to Pass or broaden the gravity claim.

## 1. Decision Lock

Operator lock: **2026-05-29**. The adjustment pass pinned the fitted-combo
solver details, grouped folds, bootstrap undefined handling, and OR-label
diagnostic caveat. No Phase 16 harness code has been written or run at lock time.

- **Frozen label & oracle.** Reuse the Phase 15 strict forward oracle
  (`computeStrictOracleDetails`, 32 steps × 8 substeps) and its boolean
  `hazardReached` label unchanged. No change to the oracle, the hazard geometry
  (`escapeRadius=4`, `closeApproachRadius=0.08`), or the audit cadence
  (`--sensor-audit-every 240`). The label and its sampling are byte-identical to
  Phase 15.
- **Passive only.** Oracle-hazard samples are emitted only on `off` (passive)
  trajectories, exactly as Phase 15. The controller is never invoked, retuned, or
  scored here.
- **Same cell slate (dt=0.004).** Mass ratios `0.01,0.3,1`, radius scales
  `1.025,1.05,1.075`, velocity scales `0.95,1.05,1.1,1.15`, `near_escape`,
  thrust limit `0.4`, sensor noise `0`, eight seeds, duration `16`, `dt=0.004`.
  This is the Phase 15B/15C lock geometry restricted to `off` mode: **288 passive
  trials** (3 × 3 × 4 × 8). The **favorable pocket** is `velocityScale ≥ 1.05`
  (216 trials); `velocityScale = 0.95` is reported as a control, never pooled into
  the primary read.
- **Audit only / additive.** All Phase 13/14/15/15B/15C paths remain unchanged.
  New receipts are written only behind a new explicit `--hazard-channel-audit`
  flag. The existing energy `oracleHazardAuroc` path is untouched.
- **No peeking.** This spec is locked **before** the Phase 16 passive run. No
  channel, sign, threshold, model, or estimator may be changed after any Phase 16
  data is seen; any change requires a filed amendment and a fresh run.
- **No claim upgrade.** All branches preserve the Phase 15 formal verdict.

## 2. Scope

Phase 16 owns:

- additive per-sample hazard-channel logging in `public/js/threebody-core.mjs`
  (trial-level `hazardSamples` array, behind `--hazard-channel-audit`);
- the `--hazard-channel-audit` flag in `scripts/threebody-operating-envelope.mjs`;
- a new offline analysis script `scripts/threebody-phase16-hazard-audit.mjs`
  (all AUROC / bootstrap / cross-validation; no re-simulation);
- future npm scripts:
  - `npm run threebody:phase16:hazard-smoke`
  - `npm run threebody:phase16:hazard`
  - `npm run threebody:phase16:analyze`
- outputs under `results/threebody/phase16-hazard-channel-audit-smoke/` and
  `…-lock/` (incl. `hazardSamples` inside `trials-minimal.jsonl`) plus
  `hazard-channel-audit.csv` from the analysis script;
- this spec and a future `PHASE16_RESULTS.md`.

Phase 16 does **not** own controller redesign, guard retuning, alternate hazard
*labels* (the oracle label is frozen), lookahead-warning implementation,
spatial/3D extension, or any public-copy change.

## 3. Command Shape

Reserved but not runnable until the implementation commit adds the flag, the
`hazardSamples` receipt, the analysis script, and the npm scripts.

Smoke (column-flow + estimator sanity only; capped under the ten-minute rule):

```bash
npm run threebody:phase16:hazard-smoke
```

Intended shape: `--modes off`, `massRatio=1`, `dt=0.004`, `radius=1.025`,
`velocity=1.1`, one seed, duration `4`, `--sensor-audit-every 240`,
`--precision-receipts 1 --hazard-channel-audit 1`.

Lock (passive grid, 288 trials; exceeds the inline ten-minute rule — operator
staged):

```bash
npm run threebody:phase16:hazard      # writes trials-minimal.jsonl with hazardSamples
npm run threebody:phase16:analyze     # offline AUROC / bootstrap / CV → readback
```

The implementation commit must record a capped rate probe and wall-clock estimate
from the smoke before the lock is started.

## 4. Hazard-Score Channel Definition

At each passive oracle-hazard audit step, log the boolean label
`hazardReached` together with a fixed vector of **instantaneous** channels, all
already in scope at that point (no new physics calls, no oracle-derived
features):

| channel | source | pre-registered hazard direction |
|---|---|---|
| `energy` (baseline) | `signatures.energy` | + (higher → escape) |
| `kineticEnergy` | `signatures.kineticEnergy` | + |
| `potentialEnergy` | `signatures.potentialEnergy` | − (more negative → closer to primary) |
| `virial` | `signatures.virial` | + |
| `inertia` | `signatures.inertia` | + |
| `tidalMagnitude` | `tidal.magnitude` | + |
| `localAccelerationMagnitude` | `localAccelerationMagnitude` | + |
| `radius` | `events.testParticleRadius` | + (higher → escape) |
| `minPrimaryDistance` | `events.minPrimaryDistance` | − (lower → close approach) |
| `speed` | ‖(vx₃,vy₃)‖ from `state` | + |

The label and the oracle are **unchanged** from Phase 15. Energy is re-measured
inside Phase 16 under the identical estimator; the binding baseline is energy's
**Phase-16 pooled AUROC**, not the historical 0.683 (per-cell-mean over the dt
ladder), which is cited as context only.

Interpretive caveat: the frozen `hazardReached` label is an OR of two geometric
failure modes (escape and close approach). A channel may be physically aligned
with one subtype and diluted against the combined label. Phase 16 may report this
as diagnostic prose if visible in the data, but no subtype relabeling or
subtype-specific pass branch is allowed in this phase.

## 5. Metrics & Estimator

**Primary statistic — pooled directional AUROC.** For each channel, pool all
favorable-pocket samples and compute the Mann–Whitney AUROC of the
pre-registered-directed channel value against the label (negative-direction
channels are sign-flipped per §4 before pooling). Reuse the existing
`computeAuroc` rank logic.

**Confidence interval — cluster bootstrap by trajectory.** Resample passive
trials (one trajectory = one `cell × seed`; 216 in the favorable pocket) with
replacement, pool their samples, recompute the pooled AUROC; `B = 2000` valid
resamples; report the 2.5 / 97.5 percentiles. Bootstrap RNG seed is fixed to
`160016`. If a resample has only one label class, discard and redraw until 2000
valid resamples are collected, capped at 20,000 attempts. If fewer than 2000
valid resamples are available after the cap, the CI is **undecidable** and the
branch is Mixed / provisional. Clustering by trajectory respects
within-trajectory sample autocorrelation.

**Secondary / diagnostic.**
- **Discriminability** = `max(AUROC, 1−AUROC)` (sign-agnostic). If a channel's
  reversed direction is strongly predictive (discriminability ≥ 0.70 but
  pre-registered-directional AUROC < 0.5), flag it **"sign-misregistered"** —
  informative, but **not** a pre-registered pass (a deployable warning must commit
  to a direction a priori).
- **Per-cell mean AUROC** (Phase-15-style), reported for continuity only.
- **Full-grid** pooled AUROC (incl. `velocityScale = 0.95`) as a control.

**Fitted combination (integrity-gated).**
- Features: the §4 whitelist, standardized on the **train fold only**. No
  oracle-derived features.
- Model: L2-regularized logistic regression, **fixed `C = 1.0`** (no
  hyperparameter search). Objective is mean logistic loss plus
  `(1 / (2C)) * ||w||_2^2`; intercept is included and unpenalized; no class
  weights. Fit deterministically from zero weights with a full-batch optimizer
  (Newton / IRLS or equivalent deterministic second-order solve), `maxIter=200`,
  gradient tolerance `1e-8`; if convergence fails, report the combo as
  undecidable rather than changing solver settings.
- Cross-validation: **grouped 4-fold by seed** with fixed folds
  `{0,1}`, `{2,3}`, `{4,5}`, `{6,7}` (leave-2-seeds-out). All samples from a
  trajectory remain in one fold, and all trajectories sharing a seed are held out
  together.
- Report **held-out pooled AUROC** (pool all held-out fold predictions → one
  AUROC) + the same trajectory cluster-bootstrap CI; also report **in-sample
  AUROC** as an overfit-gap diagnostic.
- For a fair head-to-head, every single channel is **also** scored on the same
  held-out folds (identical evaluation sets), alongside its full-pool AUROC.

Trial-level receipt (`hazardSamples`, per passive trial): array of
`{time, label, channels:{…10 fields…}}`, lightly rounded (~8 digits; AUROC is
rank-based). Analysis-level outputs (`hazard-channel-audit.csv`): per channel —
directional AUROC, [lo, hi] CI, discriminability, per-cell-mean AUROC, full-grid
AUROC, held-out-fold AUROC; and for the combo — held-out AUROC, [lo, hi] CI,
in-sample AUROC, overfit gap.

## 6. Pre-Registered Branches

Pass bar: **a channel's favorable-pocket pooled directional AUROC with lower 95%
CI ≥ 0.70.**

**(A) Hazard warnable.** At least one **single** channel clears 0.70 (lower CI).
Interpretation: instantaneous hazard warning *is* achievable; energy was the
wrong channel. Recommend a follow-on warning-quality re-pose with the winning
channel. (Does not upgrade Phase 15.)

**(B) Warnability capped.** No channel — single or fitted combination — clears
0.70 (lower CI). Interpretation: instantaneous hazard warnability is structurally
limited on these cells; the next move is lookahead-based warning or a
controller-design reframe, not another instantaneous score.

**(C) Mixed / provisional.** Any of: a channel's CI straddles 0.70; **only** the
fitted combination clears while all single channels miss (overfitting-suspect —
flagged provisional, not a clean pass); or a channel clears on the full grid but
not on the favorable pocket. Interpretation: Phase 16 localizes a candidate but
does not cleanly establish instantaneous warnability.

All branches preserve the Phase 15 formal verdict.

## 7. Readback

After the smoke, record: command and wall-clock; trial count; presence of
`hazardSamples` with all 10 channels + boolean label in `trials-minimal.jsonl`;
that the analysis script runs end-to-end; a capped per-trial rate to estimate the
lock runtime.

After the lock, record: branch (A/B/C); the per-channel favorable-pocket pooled
directional AUROC + CI table (energy baseline first), with discriminability and
sign-misregistration flags; the fitted-combination held-out AUROC + CI +
overfit gap; the full-grid control; and whether the next registered move is a
warning-quality re-pose (A), a lookahead-warning / controller-design reframe (B),
or further localization (C).
