# Three-Body Phase 15B - Counterfactual Normalizer Audit Spec

Phase 15 ended as **Fail-Magnitude**: the guarded TRACK survival envelope was
dt-stable and ablations collapsed, but the pre-registered one-step privileged
counterfactual missed the `+0.20` magnitude bar and oracle-hazard AUROC missed
`0.70`. Phase 15B is a mechanism-resolution audit for the most immediate
post-readback hypothesis: the `1e-9` normalizer floor in
`counterfactualScore = clamp(effectVsNoop / max(|H(noop)-H(oracleStrict)|, 1e-9), -1, 1)`
may suppress or distort signal in cells where the one-step oracle and no-op
states are nearly identical.

Phase 15B does **not** revise the Phase 15 verdict, retune the controller, or
upgrade the earned claim. It asks whether the failed counterfactual magnitude bar
was partly a measurement artifact of the one-step denominator.

## 1. Decision Lock

- **Frozen controller.** Reuse the Phase 13/15 guarded TRACK controller and Phase
  13 passive-derived guard thresholds. No guard retuning, thrust retuning, sensor
  retuning, or new controller mode.
- **Audit only.** Additive receipts are written only behind
  `--counterfactual-audit`; default Phase 13/14/15 paths remain unchanged.
- **Candidate and non-candidate rows both count.** The audit is explicitly not
  candidate-only. Non-candidate rows are the first place a denominator-collapse
  artifact could hide signal.
- **Same denominator floor.** The audited floor is the Phase 15 floor,
  `counterfactualNormalizerFloor = 1e-9`; alternative floors are not swept in
  this phase.
- **Primary unit.** Read per controller-mode/cell aggregate rows, then split by
  `candidateEnvelope=true/false`. Trial rows are supporting detail.
- **No claim upgrade.** A positive floor-collapse finding can motivate a later
  multi-step counterfactual, but cannot convert Phase 15 to Pass.

## 2. Scope

Phase 15B owns:

- additive `counterfactualAudit` receipts in `public/js/threebody-core.mjs`
- additive CSV columns in `paired.csv`, `trial-outcomes.csv`, and
  `aggregate-envelope.csv`
- `npm run threebody:phase15b:normalizer-smoke`
- `npm run threebody:phase15b:normalizer`
- outputs under `results/threebody/phase15b-normalizer-audit-smoke/`
- outputs under `results/threebody/phase15b-normalizer-audit-lock/`
- this spec and [`PHASE15B_RESULTS.md`](PHASE15B_RESULTS.md)

Phase 15B does not own multi-step counterfactual horizons, alternate hazard
scores, warning-quality reruns, spatial/3D extension, or controller redesign.

## 3. Commands

Smoke:

```bash
npm run threebody:phase15b:normalizer-smoke
```

The smoke is a capped 6-trial instrumentation check: `massRatio=1`,
`dt=0.004`, radius `1.025`, velocity `1.1`, modes `off`,
`track_sensor_accel_guarded`, `signal_delay`, `signal_shuffle`,
`action_shuffle`, `sign_flip`, one seed, duration `4`. It is a column-flow and
sanity check only; the full-horizon diagnostic is the staged lock command.

Lock:

```bash
npm run threebody:phase15b:normalizer
```

The lock is a staged long run: all Phase 15 mass ratios, all Phase 15 radii, all
Phase 15 velocities, `dt=0.004`, the six audit modes above, eight seeds, duration
`16` = 1,728 trials. It is operator-staged under the repository long-run rule.

## 4. Metrics

New trial-level audit fields:

- `counterfactualMeanEffectVsNoop`
- `counterfactualMeanAbsEffectVsNoop`
- `counterfactualMeanRawNormalizer`
- `counterfactualMinRawNormalizer`
- `counterfactualNormalizerFloor`
- `counterfactualNormalizerFloorHits`
- `counterfactualNormalizerFloorRate`
- `counterfactualFloorMeanEffectVsNoop`
- `counterfactualFloorPositiveRate`
- `counterfactualFloorMeanScore`
- `counterfactualNonFloorEligibleSteps`
- `counterfactualNonFloorMeanScore`

Aggregate rows report the corresponding means/totals:

- `meanCounterfactualMeanEffectVsNoop`
- `meanCounterfactualMeanAbsEffectVsNoop`
- `meanCounterfactualMeanRawNormalizer`
- `minCounterfactualRawNormalizer`
- `totalCounterfactualNormalizerFloorHits`
- `meanCounterfactualNormalizerFloorRate`
- `meanCounterfactualFloorMeanEffectVsNoop`
- `meanCounterfactualFloorPositiveRate`
- `meanCounterfactualFloorMeanScore`
- `totalCounterfactualNonFloorEligibleSteps`
- `meanCounterfactualNonFloorMeanScore`

Primary readouts:

- Floor exposure: TRACK `meanCounterfactualNormalizerFloorRate`, separately for
  candidate and non-candidate aggregate rows.
- Hidden-positive test: non-candidate TRACK
  `meanCounterfactualFloorPositiveRate` and
  `meanCounterfactualFloorMeanEffectVsNoop`.
- Separation: TRACK floor-hit raw-effect signs versus `signal_delay`,
  `signal_shuffle`, `action_shuffle`, and `sign_flip`.
- Suppression check: compare `meanCounterfactualFloorMeanScore` with
  `meanCounterfactualNonFloorMeanScore`.

## 5. Pre-Registered Branches

**Floor-collapse supported:** in non-candidate TRACK rows, mean floor-hit rate is
`>= 0.20`, mean floor-hit raw effect is positive, and mean floor-hit positive
rate is `>= 0.60`, while at least two of the three mistimed/shuffled arms are at
least `0.10` lower in floor-hit positive rate. Interpretation: the Phase 15
one-step normalized score likely under-read useful action signal in a material
subset of rows. Next step is a separately locked multi-step counterfactual.

**Floor-collapse rejected:** TRACK mean floor-hit rate is `< 0.05`, or TRACK
floor-hit raw effect is non-positive in both candidate and non-candidate rows.
Interpretation: the Phase 15 magnitude miss is not explained by the denominator
floor; look next at horizon locality or the hazard score itself.

**Mixed / partial diagnostic:** floor-hit rate is material but sign separation is
weak, or floor-hit signal appears only in candidate rows. Interpretation:
denominator collapse is real but not sufficient to explain the Phase 15 miss.

All branches preserve the Phase 15 formal verdict.

## 6. Readback

After the smoke, record:

- command and wall-clock
- trial count
- presence of all new columns in `paired.csv`, `trial-outcomes.csv`, and
  `aggregate-envelope.csv`
- smoke TRACK floor-hit rate and sign read
- whether the lock command is worth staging unchanged

After the lock, record the branch, the candidate/non-candidate split, and whether
a multi-step counterfactual should be pre-registered next.
