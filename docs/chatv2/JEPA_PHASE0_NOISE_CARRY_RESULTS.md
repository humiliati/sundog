# JEPA Phase 0 Noise-Carry Results

**Status:** SMOKE VERDICT (2026-06-05) — `blocked_by_unfaithful_jepa`. Read-design repaired
(flip-conditioned `z_flip_acc`, GEN positive-control live). JEPA trains without collapse and
shows the predicted lower flip-conditioned noisy-bit read, but it does **not** keep the shared
source strongly enough: masked-context `u_det=0.368 < 0.70`. Per the verdict tree this branch
precedes capacity/confirmation; no `{128,256} × 3` battery is run.

Executed spec: `docs/chatv2/JEPA_PHASE0_NOISE_CARRY_SPEC.md`.
Runner: `scripts/jepa_phase0_noise_carry.py`.

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

## Disposition after repair 3

At this point `JEPA_PHASE0_NOISE_CARRY_SPEC.md` remained on DEBUG HOLD, but the read-design
blocker was closed: the flip-conditioned `z_i` statistic had a live GEN positive-control. The
next move was the staged JEPA training/collapse smoke recorded below.

## Debug repair 4 - JEPA training/collapse smoke (the staged z_flip smoke) RAN

Ran the staged smoke (d=128, 1 seed, full Phase-7b training, ~15.5 min). First launch died
silently at the `[cfg]` line — a background PowerShell `& py … 2>&1 | Tee-Object` promotes
python's first stderr line to a fatal `NativeCommandError`, killing the run with no traceback.
Re-launched **clean** (no `2>&1`, `$ErrorActionPreference='Continue'`, `$env:PYTHONWARNINGS='ignore'`).
Receipt: `results/chatv2/jepa-phase0-noise-carry/smoke.json`.

| read | GEN | JEPA |
| --- | ---: | ---: |
| `z_flip_acc` (primary) | 0.713 | 0.542 |
| `u_det` (control) | 0.622 | **0.256** |
| effective rank | 3.0 | 31.5 |
| collapsed | - | **False** |

`z_flip_gap = +0.170` (> `frozen_delta` 0.15); direct-`x_i` sidecar gap `+0.046`.

**Verdict: VIABLE-BUT-UNINTERPRETABLE.**
- ✓ **Buildable** — JEPA trains without collapse (`frac_std_ok=1.0`, eff-rank 31.5 ≥ the
  `max(8, 0.05·d)` floor). VICReg holds. This was the hard engineering gate.
- ✓ **Directional** — the noise-discard gap is `+0.170`, above the bar: JEPA tracks the private
  noise *less* than GEN, in the predicted direction.
- ✗ **`u_det` control fails** — JEPA `u_det = 0.256 ≪ 0.70`. Until JEPA visibly *keeps* `u`, the
  gap is **uninterpretable**: a lower `z_flip_acc` could be a weaker overall body rather than a
  *selective* noise-discard. (GEN's own `u_det=0.622` is soft vs the banked 0.754 — partly the
  smoke's `n=1000` vs 3000.)
- `support_starved=True` is a pure **smoke artifact** of `n=1000` (~45 flips/latent); the full
  `n=3000` gives ~150 and clears the floor.

**Most likely cause:** read-protocol mismatch — the JEPA context encoder is *trained* on
50%-masked input but *read* on full input (8 latents it never co-saw), so its full-input
representation is off-distribution and recovers `u` weakly.

## Debug repair 5 - masked-input read protocol patched and re-smoked

The read-design blocker is closed and the build is viable; the one open blocker is the JEPA
`u_det` control. The likely bug was read-protocol mismatch: the JEPA context encoder was trained
on 50%-masked input but read on full input. The runner now reads JEPA with
`masked_context_avg`: 50% latent-channel masks matching training, final-position context body,
averaged over 8 independently sampled mask patterns before probing.

Patch sanity:

```powershell
python -m py_compile scripts/jepa_phase0_noise_carry.py
python scripts/jepa_phase0_noise_carry.py --smoke --gen-only --read-phase7b-gen --out results/chatv2/jepa-phase0-noise-carry-debug-phase7b-zflip-rerun
```

The saved-GEN positive control still reproduces after the patch:

| read | value |
| --- | ---: |
| `u_det` | 0.7538 |
| `z_flip_acc` | 0.7839 |
| support starved | false |

Registered operator-staged re-smoke command:

```powershell
python scripts/jepa_phase0_noise_carry.py --smoke --read-n-fingerprint 3000 --jepa-mask-reads 8 --out results/chatv2/jepa-phase0-noise-carry-zflip-masked-smoke
```

The landed receipt was written to the canonical smoke path
`results/chatv2/jepa-phase0-noise-carry/smoke.json`, with `read_protocol=masked_context_avg`,
`mask_reads=8`, and `n_fingerprint=1000`.

| read | GEN | JEPA |
| --- | ---: | ---: |
| `z_flip_acc` (primary) | 0.713 | 0.504 |
| `z_flip_gap` |  | **+0.208** |
| `u_det` (control) | 0.622 | **0.368** |
| effective rank | 3.0 | **68.0** |
| collapsed | - | **False** |
| read protocol | full-input fair | masked-context avg ×8 |

**Verdict: `blocked_by_unfaithful_jepa`.**

The masked read helped (`u_det` rose from 0.256 to 0.368) and made the representation healthier
(effective rank rose from ~31 to ~68). The predicted flip-conditioned gap also strengthened
(`+0.170` → `+0.208`). But the load-bearing `u_det >= 0.70` control still fails badly. Even
granting the smoke-size deflation seen on GEN (~0.13 between `n=1000` and the banked `n=3000`
read), JEPA would remain around ~0.50, below the bar. Therefore the lower `z_flip_acc` cannot be
interpreted as selective noise discard rather than a weaker functional representation.

Per the spec, `blocked_by_unfaithful_jepa` precedes `blocked_by_capacity` and
`jepa_noise_discard_confirmed`, so the `{128,256} × 3-seed` battery is **not run**. This is a
banked negative: the small JEPA objective is buildable and does not collapse, but on this coupled
toy it does not keep the predictable source strongly enough for the noise-discard contrast to be
valid.
