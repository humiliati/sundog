# Chat-v2 Phase 1 — Seed-Stability + Contrast Decomposition Receipt

> 2026-06-01. The Amendment-A (`PHASE1_RESIDUAL_BODY_SCALING_SPEC.md` §13)
> seed-stability cell, run and adjudicated, plus the contrast shore-up the
> result motivated. **Status: receipt, not a promotion.** Synthetic toy;
> `d_dec ≈ 7.6 < 20` (does not clear the Gate-D high-dim bar).

## Cell (frozen)

`H=8`, computed pair-XOR latents, **fair readout** (`_lastpos`), `d_model=192`,
`δ=0.45`, `bits_per_channel=24`, `max_steps=6000`, grok-aware `min_steps=3000` /
`patience=10`, seeds **{0, 1, 2}**. Runner `scripts/chatv2_phase0_bodyresist.py`.
Out: `results/chatv2/phase1-seedstab/seed{0,1,2}/`. Adjudicator
`scripts/chatv2_phase1_adjudicate.py`; decomposition
`scripts/chatv2_phase1_contrast.py`.

## Per-seed verdicts

| seed | learned (`eval_loss`) | `d_dec` | `z1_acc` | `leak` | `body_carry` gen / twin | gap | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | yes (0.497) | 7.6 | 0.84 | 0.51 | 0.83 / 0.60 | 0.23 | SHARP |
| 1 | yes (0.494) | 7.5 | 0.86 | 0.52 | 0.88 / 0.67 | 0.21 | SHARP |
| 2 | yes (0.485) | 7.6 | 0.80 | 0.52 | 0.73 / 0.56 | 0.17 | MARGINAL |

**Gate E (≥2/3 same branch): PASS — `phase1_scaling_sharp_below_bar`** (2 of 3
SHARP; the third fails only the gap gate). `branch_adjudication.json`.

## What is robust vs at-the-bar

- **Robust across all three seeds (tight):** learned; high-dim (`d_dec` 7.5–7.6);
  control-sufficient (`z1` 0.80–0.86); resisting (`leak` 0.51–0.52). The
  earlier "seed-sensitivity" scare was a **config artifact** — the curriculum's
  MARGINAL was at `pos_h=16`, a different init regime; at the frozen `pos_h=8` +
  fair-readout cell the body-resistance is stable.
- **At-the-bar:** the binary SHARP/MARGINAL verdict is decided entirely by the
  gen-twin `body_carry` gap, which clusters on the 0.20 threshold (0.23 / 0.21 /
  0.17).

## Contrast decomposition (the shore-up)

Replacing the binary gap with a baseline ladder
(`contrast_decomposition.json`), aggregated across seeds:

```
chance 0.50  →  untrained 0.591±0.010  →  twin 0.608±0.047  →  gen 0.812±0.061
  architectural   (untrained − chance) = 0.091 ± 0.010
  incidental      (twin − untrained)   = 0.017 ± 0.037     <- ~ZERO
  objective_excess(gen − twin)          = 0.205 ± 0.022     <- ROBUST
  objective_fraction (gen−twin)/(gen−chance) = 0.67 ± 0.09
```

**Two findings that change the reading:**

1. **The twin floor is architectural, not training-incidental.** A *random-init*
   backbone already reads the non-decision XOR latents at 0.59 (random features
   partially compute them); control-only *training* adds essentially nothing
   (`incidental ≈ 0.017`, seed 2 negative). The hypothesis that the twin
   "incidentally computes" the latents is **falsified** — it's the architecture,
   not the control objective.
2. **The objective-driven contrast is a robust quantity, not a coin-flip.** The
   generative objective adds `0.205 ± 0.022` to the non-decision representation
   above the control objective — **~67% of the body's above-chance non-decision
   content is objective-driven.** The binary SHARP/MARGINAL flipped only because
   that stable ~0.205 straddles the 0.20 bar.

## Honest headline (shored)

**A high-dimensional resisting body that is robust *and robustly
objective-driven*:** across three seeds, high-dim + control-sufficient +
resisting, with the generative objective contributing a stable
**0.205 ± 0.022** of the non-decision representation (control training ≈ 0;
~0.09 architectural). The "SHARP coin-flip" dissolves into this continuous,
seed-stable measure. **Unpromoted**; `d_dec ≈ 7.6 < 20` (below the Gate-D bar).

## Disposition

- **Contrast shored** — reported continuously + decomposed; the result is
  seed-stable, so a harder-per-channel-computation task variant (to widen the
  raw gap) is now **optional**, not required: the contrast is already a robust
  quantified ~0.205, and the decomposition shows *why* it isn't larger (the
  baseline is architectural, near-irreducible).
- **Scaling re-opens** per the spec: `H=16` / `H=32` toward `d_dec ≥ 20`
  (Gate D — the ARC PR-11 body could not reach it). CPU-prohibitive at `H=32` →
  gated on the **LATTICE / GPU** on-ramp.
- **Cross-substrate note + Phase 0.2** to be updated to the shored framing
  (robust + robustly-objective-driven, the contrast as a continuous 0.205, the
  architectural-floor finding).
