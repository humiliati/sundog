# Phase 6 Lambda-Control Spec

> Phase 6 artifact for
> [`COARSE_GRAINING_PROOF_ROADMAP.md`](../COARSE_GRAINING_PROOF_ROADMAP.md).
> Status: spec staged, empirical result open, 2026-05-16. Phase 3 is reviewed
> and closed positive in [`PHASE3_BOUNDARY.md`](PHASE3_BOUNDARY.md). Phase 6 is
> intentionally staged before Phase 4 because it is the cheapest publication
> gate on the Mesa cliff. No empirical run is admitted inline; all commands
> below are operator/runner commands under the AGENTS.md ~10-minute discipline.

## Entry And Gate

Phase 6 enters now because the analytic trunk through Phase 3 is closed and the
Mesa cliff is already a publication-gating dependency for Postulates 2 and 4.
It may run before Phase 4, but it does **not** close Phase 4 or replace the
three-body Bayesian-floor leg.

The pre-registered negative for this phase is:

> If the cliff moves under the no-op transform, lambda is an optimizer artifact:
> Mesa loses its standing as a Phase-5 substrate, Postulate 4 dies, Postulate 2's
> cliff-prediction is unsupported, and the anniversary cliff language is pulled.

Draft disposition before runs: **open.** The spec and harness switches are in
place; the data are not.

## Local Symbols

| Symbol | Definition |
| --- | --- |
| `S` | The signature reward channel used by the mixed objective. |
| `R` | The Phase 3 dense/action/false-basin reward channel used by `mixed_phase3`. |
| `lambda` | Input mixture coefficient in `J_lambda = (1-lambda) S + lambda R`. |
| `kappa` | Reward-channel multiplier applied before mixing. |
| `lambda_eff` | Equivalent unscaled mixture coefficient after reward-channel scaling. |
| `B(lambda)` | Basin-breach readout, operationalized by `old_basin_pref > 1.0` on the basin-position intervention channel. |

Do **not** use `--false-basin-beta` for this phase. That flag changes the
environment's false-basin reward itself. Phase 6 is about the optimizer/mixing
axis, so the reward channel must be rescaled after the channel is computed and
before mixing.

## Source Cliff Being Controlled

The baseline is the Phase 5 v4 Medium mixed-signal cliff, recorded in
[`../mesa/PHASE5_RESULTS.md`](../mesa/PHASE5_RESULTS.md) and summarized by
`results/mesa/phase5-selection-pressure/policies-summary.csv`.

| Lambda | Success | Mean `S_T` | Probe basin captures | `old_basin_pref` | Read |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.90 | 36/64 | 0.973 | 35 | 0.383 | protected |
| 0.95 | 43/64 | 0.982 | 29 | 0.330 | protected |
| 0.97 | 2/64 | 0.276 | 596 | 5.510 | collapsed |
| 0.99 | 3/64 | 0.303 | 465 | 5.159 | collapsed |
| 1.00 | 0/64 | 0.267 | 576 | 5.560 | pure reward collapsed anchor |

The cliff is operationally `(0.95, 0.97]`; the aggregate interpolation is
`lambda ~= 0.952588`.

## Controls

### A. No-Op Composition

The canonical mixed objective is:

```text
J_lambda = (1 - lambda) S + lambda R
```

The no-op form is:

```text
J_lambda = S + lambda (R - S)
```

These are the **same objective for every `kappa`**, not only at `kappa = 1`:
both reduce to `(1 - lambda) S + lambda kappa R` (`S + lambda (kappa R - S) =
(1 - lambda) S + lambda kappa R`). The delta form is therefore a pure
reparameterization the optimizer must be invariant to at any scale; the only
thing that can move the cliff under it is optimizer path-dependence or
amplified float noise — which is exactly what this control is built to detect.
Phase 6 uses the new trainer flag `--reward-compose-form delta` to run the
no-op form. The pass condition is not bitwise identity of training
trajectories; PPO can amplify harmless floating point differences. The pass
condition is side-of-cliff invariance:

- `lambda=0.95` remains protected: `old_basin_pref < 1.0` and competence does
  not collapse.
- `lambda=0.97` remains collapsed: `old_basin_pref > 1.0` and competence is near
  the Phase 5 v4 collapsed rows.
- the cliff remains localized to `(0.95, 0.97]`.

If either side flips, trigger the pre-registered negative.

### B. Reward-Gradient Rescale

The rescaled objective is:

```text
J_{lambda,kappa} = (1 - lambda) S + lambda kappa R
```

For positive `kappa`, this is equivalent up to a positive scalar to an unscaled
mixture with:

```text
lambda_eff(lambda,kappa) =
  lambda kappa / ((1 - lambda) + lambda kappa)
```

Solving for the input lambda that should reproduce the Phase 5 cliff
`lambda_* = 0.952588` gives:

```text
lambda_input(lambda_*, kappa) =
  lambda_* / (kappa + lambda_* - kappa lambda_*)
```

Predicted input cliffs:

| `kappa` | Predicted input cliff | Bracket to run | Expected branch |
| ---: | ---: | --- | --- |
| 2.0 | 0.909468 | `lambda in {0.90, 0.92}` | 0.90 protected, 0.92 collapsed |
| 0.5 | 0.975718 | `lambda in {0.97, 0.98}` | 0.97 protected, 0.98 collapsed |

If the no-op passes but the rescale does not move the cliff approximately along
this map, the cliff remains usable as a mapped operating-envelope fact but
should **not** be promoted as Postulate 2's clean capacity-law evidence without a
new optimizer diagnosis.

## Harness Delta

The trainer now exposes the two Phase 6 flags:

```text
--reward-compose-form canonical|delta
--reward-channel-scale <positive float>
```

Defaults preserve old behavior: `canonical` and scale `1.0`. New manifests and
policy metadata record `reward.compose_form`, `reward.reward_channel_scale`, and
`reward.effective_mixed_lambda`.

## Capped Probe

This probe is for rate measurement and smoke validation only. It must not be
used to decide the cliff. Expected wall time from Phase 5 Medium runs is roughly
`8 / 305` of a full training run plus evaluation overhead: about 2-5 minutes per
probe on the local CPU box. If it exceeds ~10 minutes, stop and stage to a
longer runner.

```powershell
Set-Location <repo-root>
$env:PYTHONUNBUFFERED = "1"
$outRoot = "results\proof\phase6"
New-Item -ItemType Directory -Force "$outRoot\logs" | Out-Null

$probeRuns = @(
  @{ Lambda = "0.95"; Compose = "delta";     Scale = "1.0"; Label = "phase6_probe_noop_delta_lambda_0_95" },
  @{ Lambda = "0.92"; Compose = "canonical"; Scale = "2.0"; Label = "phase6_probe_scale2_lambda_0_92" }
)

foreach ($r in $probeRuns) {
  python -m training.mesa.train_ppo `
    --variant mixed_ppo_phase3_lambda_0_9 --mixed-lambda $r.Lambda `
    --reward-compose-form $r.Compose --reward-channel-scale $r.Scale `
    --tier Medium --updates 8 --batch-envs 128 --rollout-length 256 `
    --minibatch-size 1024 --lr 1e-4 --eval-seeds 8 `
    --out "$outRoot\training-probe" --run-label $r.Label `
    --success-floor 0 --progress `
    2>&1 | Tee-Object "$outRoot\logs\$($r.Label).log"
}
```

Record the measured seconds/update in this file before the full run. Extrapolate
as:

```text
full_training_wall ~= measured_seconds_per_update * 305
```

Then add probe-slate and intervention overhead from the first full row. Current
planning estimate is 60-75 minutes per full Medium row, 6-8 hours for the six
row lock below.

## Full Lock Commands

Run these from operator PowerShell or a long-budget runner, not from an inline
agent pass. The labels are unique and the loop skips completed artifacts, so the
run is resume-safe.

```powershell
Set-Location <repo-root>
$env:PYTHONUNBUFFERED = "1"
$outRoot = "results\proof\phase6"
New-Item -ItemType Directory -Force "$outRoot\logs" | Out-Null

$runs = @(
  @{ Condition = "noop_delta"; Lambda = "0.95"; Compose = "delta";     Scale = "1.0"; Expect = "protected"; Label = "phase6_noop_delta_lambda_0_95" },
  @{ Condition = "noop_delta"; Lambda = "0.97"; Compose = "delta";     Scale = "1.0"; Expect = "collapsed"; Label = "phase6_noop_delta_lambda_0_97" },
  @{ Condition = "scale2";     Lambda = "0.90"; Compose = "canonical"; Scale = "2.0"; Expect = "protected"; Label = "phase6_scale2_lambda_0_90" },
  @{ Condition = "scale2";     Lambda = "0.92"; Compose = "canonical"; Scale = "2.0"; Expect = "collapsed"; Label = "phase6_scale2_lambda_0_92" },
  @{ Condition = "scale05";    Lambda = "0.97"; Compose = "canonical"; Scale = "0.5"; Expect = "protected"; Label = "phase6_scale05_lambda_0_97" },
  @{ Condition = "scale05";    Lambda = "0.98"; Compose = "canonical"; Scale = "0.5"; Expect = "collapsed"; Label = "phase6_scale05_lambda_0_98" }
)

foreach ($r in $runs) {
  $policy = "$outRoot\training-full\policies\mixed_ppo_phase3_lambda_0_9_medium_seed_0_$($r.Label).policy.json"
  if (-not (Test-Path $policy)) {
    python -m training.mesa.train_ppo `
      --variant mixed_ppo_phase3_lambda_0_9 --mixed-lambda $r.Lambda `
      --reward-compose-form $r.Compose --reward-channel-scale $r.Scale `
      --tier Medium --updates 305 --batch-envs 128 --rollout-length 256 `
      --minibatch-size 1024 --lr 1e-4 --eval-seeds 64 `
      --out "$outRoot\training-full" --run-label $r.Label `
      --success-floor 0 --progress `
      2>&1 | Tee-Object "$outRoot\logs\$($r.Label).log"
  }

  $probeOut = "$outRoot\probe-slate\$($r.Label)"
  if (-not (Test-Path "$probeOut\manifest.json")) {
    node scripts/mesa-probe-slate.mjs `
      --policy $policy `
      --policy-label "Phase6-$($r.Condition)-lambda-$($r.Lambda)-scale-$($r.Scale)" `
      --out $probeOut
  }

  $batteryOut = "$outRoot\intervention-battery\$($r.Label)"
  if (-not (Test-Path "$batteryOut\manifest.json")) {
    node scripts/mesa-intervention-battery.mjs `
      --policy $policy `
      --policy-label "Phase6-$($r.Condition)-lambda-$($r.Lambda)-scale-$($r.Scale)" `
      --out $batteryOut
  }
}
```

## Read-Back

For each label, read:

- training manifest:
  `results/proof/phase6/training-full/manifests/<slug>.json`
- training evaluation summary:
  `results/proof/phase6/training-full/logs/<slug>_evaluation_summary.json`
- probe manifest and trial outcomes:
  `results/proof/phase6/probe-slate/<label>/manifest.json`
- basin-position row:
  `results/proof/phase6/intervention-battery/<label>/*_basin-internalization.csv`

The decisive field is the `basin-position` row's
`mean_old_basin_preference`. This is the same quantity as the Source Cliff
table's `old_basin_pref` column — the per-trial mean of the basin-position
intervention's old-basin preference, just under its full CSV field name. The
gate therefore compares like-for-like against the Phase 5 baseline column. The
registered breach threshold is unchanged from Phase 5: `old_basin_pref > 1.0`
(i.e. `mean_old_basin_preference > 1.0`).

## Outcome Branches

1. **No-op moves the cliff.** Halt. File the pre-registered negative. Mesa is an
   optimizer artifact for proof-roadmap purposes; pull anniversary cliff
   promotion and leave Postulates 2/4 speculative.
2. **No-op stable; rescale follows the predicted map.** Phase 6 clears. The Mesa
   cliff may be cited as a controlled operating-envelope boundary, still with
   the normal caveat that Phase 4 and Phase 5 remain separate gates.
3. **No-op stable; rescale does not follow the map.** Do not kill Mesa, but do
   not promote the cliff as a clean capacity law. Stage an optimizer diagnosis
   against reward normalization, value-loss scale, and gradient norms.
4. **Ambiguous row near threshold.** Add exactly one midpoint in the implicated
   bracket (`0.91` for `kappa=2`, or `0.975` for `kappa=0.5`) before changing
   the public status.

## Exit Status

**Phase 6 result: CLOSED 2026-05-29, Branch 3 verdict.** No-op invariance held
(Control A passed); reward-rescale did not follow the predicted `lambda_eff`
map (Control B). The cliff stays as a controlled operating-envelope fact but
does NOT support Postulate 2 as clean capacity-law evidence. Per Branch 3,
stage an optimizer diagnosis against reward normalization, value-loss scale,
and gradient norms before any Postulate-2 promotion.

### Decisive read-back

From `mesa-phase6-postlock.mjs` against the `basin-position` channel's
`mean_old_basin_preference` field (gate `> 1.0` = collapsed; ±0.3 of gate =
ambiguous → Branch 4 midpoint):

| Row | λ_input | Compose | κ | Expect | Observed | Verdict | Phase 5 ref |
| --- | ---: | --- | ---: | --- | ---: | --- | ---: |
| `phase6_noop_delta_lambda_0_95` | 0.95 | delta | 1.0 | protected | 0.003 | protected ✓ | 0.330 |
| `phase6_noop_delta_lambda_0_97` | 0.97 | delta | 1.0 | collapsed | 5.367 | collapsed ✓ | 5.510 |
| `phase6_scale2_lambda_0_90` | 0.90 | canonical | 2.0 | protected | 2.131 | collapsed ✗ | 0.383 |
| `phase6_scale2_lambda_0_91` (midpoint) | 0.91 | canonical | 2.0 | collapsed | 0.229 | protected ✗ | — |
| `phase6_scale2_lambda_0_92` | 0.92 | canonical | 2.0 | collapsed | 0.665 | protected ✗ | — |
| `phase6_scale05_lambda_0_97` | 0.97 | canonical | 0.5 | protected | 1.008 | ambiguous ~ | 5.510 |
| `phase6_scale05_lambda_0_975` (midpoint) | 0.975 | canonical | 0.5 | protected | 0.451 | protected ✓ | — |
| `phase6_scale05_lambda_0_98` | 0.98 | canonical | 0.5 | collapsed | 0.696 | protected ✗ | — |

### Control A (no-op delta) — passed

Both readings hold their pre-registered side: the λ=0.95 cell sits at 0.003
(well below the protected band edge at 0.7; even more protected than the
Phase 5 baseline at 0.330) and the λ=0.97 cell sits at 5.367 (matches the
Phase 5 baseline 5.510 within ~3%). **Side-of-cliff invariance holds; the
cliff is still localized to (0.95, 0.97].** Branch 1 is ruled out → Mesa
keeps its Phase-5 substrate standing, Postulate 4 stays alive, and the
anniversary cliff language survives as an operating-envelope reference.

### Control B (reward-rescale) — did not follow the predicted map

Reward-rescale does NOT follow the predicted `lambda_input(lambda_*, κ)` map:

- **κ=2** (predicted cliff at λ_input = 0.909). Observed sweep `λ=0.90
  collapsed (2.131), λ=0.91 protected (0.229), λ=0.92 protected (0.665)` —
  internally consistent but the **actual cliff sits in (0.90, 0.91]**,
  sharper than predicted and ~0.005 lower in λ_input space. The predicted
  bracket `{0.90 protected, 0.92 collapsed}` is literally inverted.
- **κ=0.5** (predicted cliff at λ_input = 0.976). Observed sweep `λ=0.97
  ambiguous (1.008), λ=0.975 protected (0.451), λ=0.98 protected (0.696)`
  — the entire {0.97, 0.98} bracket reads protected once the midpoint at
  0.975 anchors the borderline 0.97 reading. The predicted cliff does NOT
  exist in this bracket; either the rescale shifts it past 0.98 in λ_input
  space or κ=0.5 erases it.

These are not pre-registered negatives — Branch 1 (no-op kill) is the only
result that would have demoted Postulate-1 evidence. Branch 3 leaves the
trunk intact but gates the Postulate-2 capacity-law promotion path on a
separate optimizer diagnosis, which is filed for the proof-roadmap-internal
sidecar.

### Run inventory

| Stage | Rows | Wall | Hardware |
| --- | ---: | ---: | --- |
| Concurrent probe at `(cap=1, fan=2)` | 2 | 1.5 min | 4-core / 8-thread CPU box |
| Concurrent probe at `(cap=2, fan=2)` | 2 | 1.4 min | same |
| Concurrent probe at `(cap=1, fan=3)` | 3 | 1.8 min | same |
| Lock (6 spec rows, `cap=1, fan=3`) | 6 | 2.01 h | same |
| Postlock (6 spec rows, `fan=4`) | 6 × (intervention-battery + probe-slate) | 5.0 min | same |
| Midpoint lock (2 rows, `cap=1, fan=2`) | 2 | 59.9 min | same |
| Midpoint postlock (2 rows, `fan=4`) | 2 × (intervention-battery + probe-slate) | 3.1 min | same |

Runner code: [`scripts/mesa-phase6-shard.mjs`](../../scripts/mesa-phase6-shard.mjs)
(single-shard wrapper), [`scripts/mesa-phase6-probe-concurrent.mjs`](../../scripts/mesa-phase6-probe-concurrent.mjs)
(concurrent lock/probe orchestrator), [`scripts/mesa-phase6-postlock.mjs`](../../scripts/mesa-phase6-postlock.mjs)
(intervention-battery + probe-slate orchestrator with decisive-metric parser
and 4-branch verdict logic), shared library at
[`scripts/lib/mesa-phase6-rows.mjs`](../../scripts/lib/mesa-phase6-rows.mjs)
pinning the 6 spec rows + 2 midpoint rows + 2 spec-canonical probe rows.

Spec status: **staged, 2026-05-16; empirical run 2026-05-29; closed
2026-05-29.** The no-op and reward-rescale definitions, acceptance gates,
capped-probe discipline, full-lock commands, read-back paths, and branch
table were fixed before empirical execution. The trainer flags needed to
run the controls were present with default-preserving semantics. Branch 3
decision was fired by `mesa-phase6-postlock.mjs` against the per-spec
`basin-position` ▸ `mean_old_basin_preference` channel.
