# JEPA Phase 0 Noise-Carry Results

**Status:** DEBUG HOLD after failed smoke (2026-06-04). No phase verdict; no full JEPA
battery run.

Binding/draft spec: `docs/chatv2/JEPA_PHASE0_NOISE_CARRY_SPEC.md`.
Runner under debug: `scripts/jepa_phase0_noise_carry.py`.

## Smoke 0 - naive runner failed

Receipt: `results/chatv2/jepa-phase0-noise-carry/smoke.json`.

The first smoke did not pass:

| body | `u_det` | linear `noise_det` | nonlinear sidecar | effective rank |
| --- | ---: | ---: | ---: | ---: |
| GEN | 0.213 | 0.000 | -0.080 | 3.0 |
| JEPA | 0.120 | -0.240 | -0.220 | 32.5 |

Interpretation: this was not a JEPA-vs-GEN scientific result. The runner used fresh `Cfg`
defaults instead of the Phase-7b positive-control cell, so GEN under-trained and did not
recover the shared source `u`. Because neither body carried `u` strongly, the private-noise
read was uninterpretable.

## Debug repair 1 - Phase-7b config inheritance

The runner was patched to inherit the Phase-7b cell before any future JEPA read:
`bits_per_channel=24`, `delta=0.45`, `max_steps=6000`, `min_steps=3000`, `patience=10`,
fair readout, and `seed + 1000*H`.

Attempted inline command:

```powershell
python scripts/jepa_phase0_noise_carry.py --smoke --gen-only --out results/chatv2/jepa-phase0-noise-carry-debug-gen
```

This exceeded the ~10-minute inline rule on the CPU path and was stopped. The corrected GEN
retraining preflight must be staged for the operator/GPU if needed.

## Debug repair 2 - read-only Phase-7b GEN diagnostic

Command:

```powershell
python scripts/jepa_phase0_noise_carry.py --smoke --gen-only --read-phase7b-gen --out results/chatv2/jepa-phase0-noise-carry-debug-phase7b-read
```

Read-only result on the already banked Phase-7b GEN body:

| read | value |
| --- | ---: |
| `u_det` | 0.754 |
| local linear `noise_det` | -0.010 |
| global linear `noise_det` | -0.476 |
| support starved | false |

This separates the two failures. GEN source recovery is live on the known-good Phase-7b
body, but direct linear private-noise carry is not. Therefore the original primary statistic
`det(x_i | body)` is not a valid binding kill-gate as written.

## Side diagnostic - where `x_i` is visible

On the same Phase-7b seed0 arrays, a quick side diagnostic found:

| feature | linear `x_i` det | nonlinear `x_i` det |
| --- | ---: | ---: |
| true `z` | 0.000 | 0.448 |
| true `z` + true `u` | 0.000 | 0.997 |
| body `z`-score shadow | 0.000 | 0.010 |

Reading: `x_i = z_i xor parity(u,A_i)` is a nonlinear containment relation. It is visible
from true state variables, but not as a direct linear body probe and not from the quick
low-dimensional body-score sidecar. At this point the phase needed either a valid derived
noise-containment read with its own controls, or an uninterpretable-read branch before
interpreting JEPA. Debug repair 3 supplies that derived read.

## Debug repair 3 - flip-conditioned `z_i` read

User-proposed repair: do not probe the XOR-derived private bit `x_i` directly. Instead train a
normal linear probe `body -> z_i` on all training rows, then evaluate its held-out accuracy only
on the noise-flipped subset `{x_i=1}`.

Command:

```powershell
python scripts/jepa_phase0_noise_carry.py --smoke --gen-only --read-phase7b-gen --out results/chatv2/jepa-phase0-noise-carry-debug-phase7b-zflip-rerun
```

Read-only result on the same already banked Phase-7b GEN body:

| read | value |
| --- | ---: |
| `u_det` | 0.754 |
| direct local `noise_det` sidecar | -0.010 |
| **`z_flip_acc` primary preflight** | **0.7839** |
| held-out flip counts | 118-165 per latent |
| support starved | false |

This repairs the primary read. GEN visibly predicts the observed noisy `z_i` even on flipped
samples, while the failed direct `x_i` probe remains dead as expected.

A separate oracle sanity check on the same row split gives the intended polarity:

| oracle feature | all-row `z_i` acc | flip-only `z_i` acc | clean-row `z_i` acc |
| --- | ---: | ---: | ---: |
| true observed `z_i` | 1.000 | 1.000 | 1.000 |
| true clean component only | 0.902 | 0.000 | 1.000 |
| Phase-7b GEN body | 0.903 | 0.784 | 0.917 |

So a denoising/clean-only body is systematically wrong on flips, while a GEN-like observed-`z`
body is high on flips. That is exactly the JEPA-vs-GEN contrast the repaired statistic is meant
to test.

## Current disposition

`JEPA_PHASE0_NOISE_CARRY_SPEC.md` remains on DEBUG HOLD, but the read-design blocker is closed:
the flip-conditioned `z_i` statistic has a live GEN positive-control. The remaining blocker is
JEPA training/collapse smoke. No full battery has run.

Next staged smoke (not inline; corrected GEN-only training exceeded the ~10-minute CPU rule):

```powershell
python scripts/jepa_phase0_noise_carry.py --smoke --out results/chatv2/jepa-phase0-noise-carry-zflip-smoke
```

If that smoke passes, the operator-staged lock battery is:

```powershell
python scripts/jepa_phase0_noise_carry.py --out results/chatv2/jepa-phase0-noise-carry-zflip-lock
```
