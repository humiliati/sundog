# AT-4b Rollout Crossover Scope

> 2026-07-05. Scoping note after `AT4_SURFACE_SUFFICIENT`. Not a frozen
> pre-registration, not a run, and not a rescue of AT-4. This document records
> which AT-4b variant is worth lifting into a spec, and which tempting variant
> should stay closed.

## Recommendation

`AT4B_BALANCE_ONLY_NOT_RECOMMENDED`.

A balance-gated rerun of AT-4 at G=200 should not be registered. It fixes the
owned transplant hygiene gap but leaves the two mechanism failures intact:

1. The AT-4 label, `J_q(500)`, is persistence-readable. Its 500-step horizon
   sits inside the G=200 autocorrelation structure, so a trailing window and the
   forward max share the same information.
2. The G=200 cell has no effective noise floor. AT-6 and
   `AveragingDecodability` agree: in a noiseless near-regular cell, window
   averaging attenuates amplitude but does not destroy exact decodability.

The existing AT-4 receipt remains the verdict for that family:
`AT4_SURFACE_SUFFICIENT`.

## Mechanism-Bearing AT-4b

`AT4B_ROLLOUT_DETRENDED_G300` is the only recommended successor design.

The carrier should be the maintained ledger's own rollout forecast, not a
snapshot readout of `Phi_K3(v(s))`. From the sub-synchronizing ledger state
`v(s)`, free-run the PDE surrogate forward for a registered horizon and apply
the decision functional to that rollout. This tests whether the maintained
state is useful as a forecast machine. A snapshot carrier can only relay the
observed mode; a rollout carrier has to use the entrained dynamics.

The interesting observation budget is still `K_obs = 1`, below the measured
sync threshold `K_sync = 2`. A positive at `K_obs >= K_sync` would be ordinary
data assimilation, not the crossover form this lane is trying to test.

## Required Design Choices

These choices should be frozen in a later `AT4B_*_SPEC.md`; they are scoped
here but not yet binding.

- **Regime:** G=300 primary. G=200 is the near-regular/no-noise-floor regime
  that made AT-4 surface-sufficient.
- **Carrier:** AOT ledger at `K_obs = 1`; `mu = 10` as the primary inherited
  gain, with `mu = 50` only as a pre-registered sensitivity if the spec wants
  one. The carrier output is a direct rollout forecast, not a trained probe on
  the current ledger snapshot.
- **Rollout horizon:** strictly beyond the measured G=200 autocorrelation time;
  the spec should name one primary horizon before the run. A truth-only
  admission rung may choose from an ordered list such as `{1500, 2500, 5000}`
  by the first horizon that clears the balance and power gates.
- **Label:** drift-aware from the start. Example shape:
  `max_{t in (s, s+tau]} E_low(u_t)` exceeds the rolling `q=0.70` quantile of a
  trailing epoch ending before `s`. This is a new objective, not a repair of
  frozen `J_q(500)`.
- **Surface:** same observed stream as the ledger sees; order-blind trailing
  statistics get their strongest registered shot. Include moments, quantiles,
  and binned count/gram features, with the exact window grid frozen before any
  read.
- **Slice:** balance-gated by construction. The comparison slice must have
  held-out label balance in `[0.40, 0.60]` and sufficient test mass. If no
  margin band clears both, file the typed input failure rather than weakening
  the gate.

## Suggested Two-Rung Shape

**AT-4b-0: data/surface admission.** Truth stream only. Build the detrended
G=300 labels, test balance, build the balanced slice, and run the registered
surface suite. If the surface already reads the label on the hard regime, file
`AT4B_SURFACE_SUFFICIENT_ADMISSION` and stop before spending ledger compute.

**AT-4b-1: rollout carrier.** Only if AT-4b-0 leaves room. Run the `K_obs = 1`
ledger, the scrambled-observation ledger, and the synchronized `K_obs = 2`
control. Compare the direct ledger rollout forecast to `surface_max`,
scrambled, persistence, and the truth-rollout ceiling on the same split and
balanced slice.

## Rung 0 Status

`AT4B_UNPOWERED_INPUT` (slice stage) is filed in
`AT4B0_ADMISSION_RECEIPT.md`.

The v1.1 label fix worked: replacing the literal value-quantile threshold with
the rolling quantile of the matched lookahead-max functional powered the label
at `tau = 1500` (`damp = 0.395`) beyond the detrended autocorrelation scale
(about 500 steps). But the balanced slice could not be formed. Quintile damp
over the 500k window ran `0.684 -> 0.794 -> 0.496 -> 0.000 -> 0.000`; the test
block had zero positives. No surface or crossover number was read.

Diagnosis: the G=300 envelope is non-stationary at the 500k contiguous-window
scale, even under rolling matched-functional calibration. This is a window-scale
formation failure, not a regime impossibility: AT-2's G=300 blocks were 2.5M
steps and pinned damp at 0.30.

Owner selected the one bundled v1.2 formation amendment. It used a
2,000,000-step truth window and a blocked alternating split: 50,000-step train
block, 5,000-step guard gap, 50,000-step test block, 5,000-step guard gap,
repeated.

v1.2 final: `AT4B_UNPOWERED_INPUT` (horizon stage). No registered horizon
cleared the damp window (`tau=1500`: 0.091, `tau=2500`: 0.107, `tau=5000`:
0.103), all beyond the detrended autocorrelation scale. No slice, surface, or
crossover number was read. The pre-committed stop fires: no v1.3 and no rung 1.

## Branch Shapes

The frozen spec should keep these branches or make any deviations explicit.

| branch | meaning |
| --- | --- |
| `AT4B_CROSSOVER_CONFIRMED` | At `K_obs = 1`, ledger rollout accuracy on the balanced slice is at least `surface_max + delta`, at least scrambled/persistence floors plus `delta`, liveness holds, and the ledger remains state-unsynchronized. |
| `AT4B_SURFACE_SUFFICIENT` | The registered surface reads the detrended G=300 label within the old AT-4 tolerance of the ledger carrier. This would be the clean terminal negative for the crossover form, with every AT-4 lesson applied. |
| `AT4B_ROLLOUT_JOINT_INSUFFICIENT` | The sub-sync ledger rollout does not beat the registered floors. This says maintenance at `K_obs = 1` buys relay but not forecast skill. |
| `AT4B_SYNC_VACUOUS` | The advantage appears only at `K_obs >= K_sync`. Record as classical synchronization/forecasting, not a Sundog crossover result. |
| `AT4B_UNPOWERED_INPUT` | The drift-aware objective or balanced slice cannot be formed with registered mass and balance. |
| `AT4B_DEAD_APPARATUS` | Liveness or sync controls fail. Void and fix the apparatus. |
| `AT4B_NEG_B` | Any post-read change to horizon, rolling quantile, slice rule, surface family, or thresholds. |

## Claim Boundary

Even a positive would say only:

> On this finite G=300 Kolmogorov proxy, a sub-synchronizing maintained ledger's
> rollout forecasts the registered detrended decision better than the registered
> order-blind surface statistics allow.

It would not claim an infinite-dimensional NSE result, a world model, or that
the ledger reconstructs the state. A surface-sufficient result under this
design is equally valuable: it would close the crossover form cleanly rather
than by the G=200 persistence/no-noise artifact.
