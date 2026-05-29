# Three-Body Phase 16B - Radius Warning Re-Pose Spec

Phase 16 answered the channel question: the frozen strict-oracle hazard label is
warnable by an instantaneous geometric observable, and `radius` was the clean
single-channel winner. Phase 16B is the narrow follow-on promised by Branch A:
**would the Phase 15 warning-quality verdict have flipped if the warning score
had been `radius` instead of energy?**

This is a verdict-reconciliation replay, not a new discovery experiment. The
channel was selected by Phase 16, so Phase 16B is explicitly post-selection. Its
job is to re-pose the Phase 15 warning-quality readout with the right
instrument, preserve the Phase 15 / 15B / 15C mechanism verdicts, and keep the
claim boundary honest.

## 1. Decision Lock

Status: **drafted 2026-05-29; pending operator lock review.** No Phase 16B code
has been written and no Phase 16B command has been run at draft time.

- **Frozen source receipt.** Use the already-completed Phase 16 lock receipt:
  `results/threebody/phase16-hazard-channel-audit-lock/trials-minimal.jsonl`.
  No new simulation is owned by this phase. If the source receipt is missing,
  malformed, or not paired with the Phase 16 Branch-A manifest, the run is void.
- **Frozen label and oracle.** Reuse the Phase 15 / Phase 16 strict
  forward-oracle label `hazardReached` unchanged. The hazard geometry remains
  `r3 > 4 OR minPrimaryDistance < 0.08`; no subtype relabeling is allowed.
- **Frozen score.** Primary warning score is `radius`, direction `+` (larger
  radius means higher escape-hazard risk). Energy is recomputed only as the
  comparison baseline.
- **Frozen slate.** Analyze the Phase 16 passive `off` grid at `dt=0.004`:
  mass ratios `0.01,0.3,1`, radius scales `1.025,1.05,1.075`, velocity scales
  `0.95,1.05,1.1,1.15`, eight seeds, duration `16`, audit cadence `240`.
  Primary favorable pocket is `velocityScale >= 1.05` (27 cells, 216 passive
  trajectories). `velocityScale = 0.95` is reported as a boundary control.
- **Phase-15-style verdict.** The primary statistic is a per-cell warning-quality
  mean, matching the shape of Phase 15's warning verdict rather than Phase 16's
  pooled Branch-A statistic.
- **No claim upgrade.** Phase 16B may repair the warning-instrument read. It does
  not revise Phase 15's Fail-Magnitude branch, retune the controller, alter the
  survival envelope, or convert the three-body claim to a full mechanism pass.

## 2. Scope

Phase 16B owns:

- a small offline reducer over the Phase 16 lock `hazardSamples` receipt;
- future npm command:
  - `npm run threebody:phase16b:repose`
- outputs under `results/threebody/phase16b-radius-warning-repose/`:
  - `radius-warning-quality-map.csv`
  - `radius-warning-summary.csv`
  - `manifest.json`
- this spec and a future `PHASE16B_RESULTS.md`.

Phase 16B does **not** own a harness change, a controller run, new oracle calls,
new channel selection, guard retuning, subtype-specific hazard labels, or public
copy changes.

## 3. Command Shape

Reserved but not runnable until the offline reducer and npm script are added:

```bash
npm run threebody:phase16b:repose
```

Intended script shape:

```bash
node scripts/threebody-phase16b-radius-warning.mjs \
  --in results/threebody/phase16-hazard-channel-audit-lock \
  --out results/threebody/phase16b-radius-warning-repose
```

This is an offline analysis over a 3.4 MB JSONL receipt and should run well
under the ten-minute rule. No hard-void Phase 13 / 14 gates are required because
no simulation or shared harness code is touched; instead, the reducer must verify
the Phase 16 manifest branch is `A_hazard_warnable` and that the source contains
288 passive trials.

## 4. Metric Definition

For each cell `(massRatio, radiusScale, velocityScale)` at `dt=0.004`, pool all
passive `hazardSamples` from the eight seeds.

For each score (`radius` primary, `energy` baseline):

1. Apply the pre-registered direction (`radius` positive; `energy` positive).
2. Compute Mann-Whitney AUROC against the boolean `hazardReached` label.
3. Mark the cell **defined** only if it contains at least one positive and one
   negative label.

Primary readout:

- `radiusDefinedCells / 27` in the favorable pocket (`velocityScale >= 1.05`);
- arithmetic mean of defined favorable-pocket `radius` AUROCs;
- the same mean and coverage for `energy`, as the baseline comparison.

Secondary diagnostics:

- median / min / max cell AUROC for `radius`;
- per-velocity table, including the `v=0.95` control column;
- pooled favorable-pocket AUROC for continuity with Phase 16;
- subtype mix (`escape`, `close_approach`, `bounded`) copied from the Phase 16
  manifest or recomputed from trial outcomes if present.

No fitted combination, bootstrap threshold, or sign-agnostic discriminability
can create a Phase 16B pass. This phase is specifically the Phase-15-style
warning verdict with `radius` substituted for energy.

## 5. Pre-Registered Branches

The Phase 15 warning-quality bar is reused: **mean favorable-pocket AUROC >=
0.70 with >= 2/3 favorable cells defined** (at least 18/27 cells).

**(A) Warning verdict flips under radius.** `radius` clears the Phase-15-style
bar. Interpretation: Phase 15's warning-quality miss was an instrument-choice
failure; the frozen strict-oracle hazard label was warnable, but energy was the
wrong instantaneous score. This still does **not** revise the Phase 15
Fail-Magnitude mechanism verdict.

**(B) Warning verdict does not flip.** `radius` is decidable and misses the bar.
Interpretation: Phase 16's pooled Branch-A result does not survive the
Phase-15-style cell-mean readout; the warning repair is weaker than it looked.

**(C) Mixed / provisional.** Coverage is below 18/27 cells, source receipt
validation is incomplete, or `radius` passes only on pooled AUROC while the
cell-mean readout is undefined or split. Interpretation: Phase 16B localizes the
warning repair but does not cleanly flip the warning verdict.

All branches preserve the controller-mechanism verdict from Phase 15 / 15B /
15C.

## 6. Readback

Record:

- source receipt path and Phase 16 manifest branch;
- passive trial count, favorable-pocket trajectory count, sample count, and
  positive-label count;
- `radius` favorable-pocket coverage and mean per-cell AUROC;
- `energy` baseline coverage and mean per-cell AUROC under the same reducer;
- per-velocity summary table;
- branch (A/B/C);
- exact prose boundary: warning-instrument repair only, no controller retune and
  no Phase 15 mechanism upgrade.
