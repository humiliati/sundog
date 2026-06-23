# H3.0-a Body / Invariant Static Audit Results

Status: **`H3_0_A_STATIC_ADMITTED`**. Generated 2026-06-23T18:30:11.128911+00:00 by `scripts/mesa-h3-0-static-audit.py`.

This is an admission-only static audit. It tests H3.0 Gates 1-2: body resistance plus invariant determination. It does not run the H3.0 fixed-control task or any H3.1 controller.

## Configuration

| parameter | value |
| --- | ---: |
| `samples` | `8192` |
| `body_dim` | `96` |
| `linear_dim` | `12` |
| `invariant_bits` | `4` |
| `shadow_noise` | `0.05` |
| `cue_noise` | `0.04` |
| `cue_strength` | `4.0` |
| `seed` | `20260623` |
| `train_frac` | `0.7` |
| `ridge_alpha` | `0.001` |
| `body_epochs` | `120` |
| `inv_epochs` | `120` |
| `null_epochs` | `60` |
| `batch_size` | `256` |
| `hidden` | `128` |
| `torch_threads` | `1` |

## Body Resistance

- `PR_body`: **94.8713** (gate >= 20)
- best body `FVE_body_from_shadow`: **0.1214** (gate <= 0.80)
- best coordinate sign accuracy: **0.6118** (gate <= 0.75)

| probe | FVE | coord_sign_acc |
| --- | ---: | ---: |
| ridge | 0.1214 | 0.6118 |
| pca_ridge | 0.0492 | 0.5697 |
| mlp | 0.0405 | 0.5862 |

## Invariant Determination

- MLP bit accuracy: **0.9733** (gate >= 0.95)
- exact packet accuracy across all bits: **0.8987**
- majority null bit accuracy: **0.5002**
- shuffled-label MLP null bit accuracy: **0.5090**
- primary null accuracy: **0.5090** (gate <= majority + 0.05)

## Gates

- Gate 1 body resistance: **True**
- Gate 2 invariant determination: **True**

Decision: **`H3_0_A_STATIC_ADMITTED`**.

## Feature Schema

Allowed shadow features:

`linear_shadow_0`, `linear_shadow_1`, `linear_shadow_2`, `linear_shadow_3`, `linear_shadow_4`, `linear_shadow_5`, `linear_shadow_6`, `linear_shadow_7`, `linear_shadow_8`, `linear_shadow_9`, `linear_shadow_10`, `linear_shadow_11`, `certificate_cue_0`, `certificate_cue_1`, `certificate_cue_2`, `certificate_cue_3`, `nuisance_0`, `nuisance_1`, `nuisance_2`, `nuisance_3`

Forbidden feature classes: `body_coordinate_*`, `invariant_label_*`, `seed`, `terminal_outcome`, `cell_id`.

JSON receipt: `results/mesa/h3/body_invariant_static_audit/summary.json`.

## Interpretation

The default synthetic family is a continuous/high-entropy body carrying discrete pair-product certificate bits. The shadow carries low-dimensional linear projections plus noisy nonlinear certificate cues. Passing this audit means the Gate 1 / Gate 2 crux is at least constructible on the static family; H3.0-b must still prove a control-sufficient singleton dilemma before any H3.1 council test.
