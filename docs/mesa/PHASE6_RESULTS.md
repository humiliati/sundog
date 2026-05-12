# Mesa Phase 6 - Interpretability Probe Result Note (v1)

This document records the Phase 6 v1 result for
[`PHASE6_SPEC.md`](PHASE6_SPEC.md). Phase 6 tested whether the Phase 5 v4
Medium cliff between `lambda=0.95` and `lambda=0.97` is visible in
linear-probe feature availability or in causal activation patching.

Status: Axis A smoke probes are **complete and negative**. Axis B
cliff-pair activation patching is **complete at layer granularity** for
the Medium cliff pair.

## 1. Summary

Phase 6 gives the program three useful updates:

1. Linear-probe feature availability does not dissociate the protected
   and collapsed cliff policies. Both endpoint-conditioned targets and
   geometry ΔR² depth profiles failed their smoke gates.
2. The cliff is causally localizable at the final hidden activation
   layer. Patching `net.7` across the `lambda=0.95` / `lambda=0.97`
   pair clears the pre-registered P4 threshold in both directions.
3. Clean and basin-position-intervened patch batteries are exactly
   identical in this feed-forward policy setting, confirming that the
   live `x_false` move is not observed at inference. The fixed-attractor
   behavior is in the weights, not a response to live basin state.

The cleanest Phase 6 headline is now: linear probes saw availability,
not use; causal patching localized use to the last hidden layer.

## 2. Artifacts

Axis A smoke artifacts:

- `results/mesa/phase6-probes/axis-a-smoke-option3/`
- `results/mesa/phase6-probes/axis-a-depth-smoke-v15/`
- `results/mesa/phase6-probes/axis-a-depth-smoke-v15-medium/`
- `results/mesa/phase6-probes/axis-a-depth-smoke-v15-cliff/`

Axis B activation-patching artifacts:

- `results/mesa/phase6-probes/axis-b-smoke-net1-8seed/`
- `results/mesa/phase6-probes/axis-b-full-64seed/`

Primary CSVs:

- `results/mesa/phase6-probes/axis-b-full-64seed/axis-b-patch-smoke.csv`
- `results/mesa/phase6-probes/axis-b-full-64seed/axis-b-patch-smoke-aggregate.csv`

## 3. Negative Result: Axis A

Axis A tested two target designs.

### 3.1 Endpoint-Shaped Behavior Target

The v1.2 target was `basin_pref_intervened`: terminal old-basin
preference under a paired basin-position intervention, broadcast back
onto clean-rollout activations by seed.

Smoke result:

| Policy | Max hidden `basin_pref_intervened` R² | Expected |
| --- | ---: | --- |
| L-Reward-S | 0.766 | high |
| L-Sig-S-Integrated | 0.737 | low |
| Oracle-S | n/a; input/privileged ceiling 0.992 | low |

This failed because the target was endpoint-shaped. Oracle predicts it
from `x_goal`; L-Signature predicts it from inferred goal / terminal
position; L-Reward predicts it from basin capture. High R² therefore
means "terminal-position information is available," not "basin
attraction is represented."

### 3.2 ΔR² Depth Profile

The v1.5 pivot tested whether feature investment across depth separates
families. The headline metric was:

`delta_r2_vs_input = R²(layer) - R²(input.obs)`

Small smoke failed: L-Reward-S and L-Signature-S were nearly identical.

Medium smoke failed:

| Policy | Deepest F1 `dist_to_x_goal` ΔR² | Deepest F2 `dist_to_x_false` ΔR² |
| --- | ---: | ---: |
| L-Reward-M | 0.060 | 0.011 |
| L-Sig-M-Integrated | 0.049 | 0.213 |

Cliff-pair smoke also failed the predicted direction:

| Policy | Deepest F1 `dist_to_x_goal` ΔR² | Deepest F2 `dist_to_x_false` ΔR² |
| --- | ---: | ---: |
| L-Mixed-M λ=0.95 | -0.018 | 0.047 |
| L-Mixed-M λ=0.97 | 0.091 | 0.018 |

Interpretation: feature availability and feature use are decoupled in
this regime. Linear probes can recover spatial information, but they do
not show which available feature is routed to the action head.

## 4. Axis B: Activation Patching

Axis B patched the Medium cliff pair:

- Protected: `L-Mixed-M λ=0.95`
- Collapsed: `L-Mixed-M λ=0.97`

For each layer, seed, and condition, the harness ran four forwards:
clean protected, clean collapsed, protected→collapsed patch, and
collapsed→protected patch. Readout was old-basin preference.

Aggregate clean-condition results:

| Layer | Direction | Mean patch success | Median | Ratio of means |
| --- | --- | ---: | ---: | ---: |
| `net.1` | protected→collapsed | 2.659 | 0.061 | 0.219 |
| `net.1` | collapsed→protected | -1.566 | -0.992 | -0.885 |
| `net.3` | protected→collapsed | -0.498 | -0.260 | -0.237 |
| `net.3` | collapsed→protected | -0.237 | -0.004 | 0.005 |
| `net.5` | protected→collapsed | 0.125 | 0.412 | 0.429 |
| `net.5` | collapsed→protected | 0.417 | 0.647 | 0.625 |
| `net.7` | protected→collapsed | 0.894 | 0.944 | 0.899 |
| `net.7` | collapsed→protected | 0.934 | 0.860 | 0.854 |

`net.7` is the only layer that clears the P4 threshold robustly across
mean, median, and ratio-of-means in both directions. `net.1` has a large
mean only because per-seed normalization is unstable when the baseline
gap is tiny; median and ratio-of-means demote it.

## 5. Clean vs Intervened

The clean and basin-position-intervened batteries were exactly identical
for all logged fields (`max_delta = 0`). This is expected after Phase 4:
the learned feed-forward policies do not observe live `x_false` or live
reward at inference. The intervention changes the environment's basin
state, but not the policy input.

This strengthens the fixed-attractor interpretation. The patch changes
behavior by altering the policy computation, not by exposing the policy
to a different live basin signal.

## 6. Verdict

P4 confirms at layer granularity: the cliff localizes causally to the
final hidden activation (`net.7`) of the actor MLP.

Axis A should not be used as Phase 6 v1 evidence beyond the negative
methodological result. Phase 6 v2 should use direction-based or
sparse-autoencoder probes if the program needs representational
interpretability beyond layer-level causal patching.
