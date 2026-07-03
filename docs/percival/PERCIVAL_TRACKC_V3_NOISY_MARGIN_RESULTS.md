# Percival Track-C v3 — The Noisy Margin (results)

Generated 2026-07-03T08:13:45.380Z by `scripts/percival-trackc-v3-noisy-margin.mjs`. Spec: [`PERCIVAL_TRACKC_V3_NOISY_MARGIN_SPEC.md`](PERCIVAL_TRACKC_V3_NOISY_MARGIN_SPEC.md).

## Verdict: **TCV3_CRISPNESS_IS_PAIRING**

Predictions: N1=true, N2=true, N3=true, N4=true.
Bounded-adversarial tolerances (h=2): β*_paired=0.425 vs β*_unpaired=0.0354 — ratio 12 = |S|/h = 12.

| mode | h | σ | MC recovery | probit analytic | margin var |
| --- | ---: | ---: | ---: | ---: | ---: |
| paired | 0 | 0.25 | 0 | 0 | 0 |
| paired | 0 | 1 | 0 | 0 | 0 |
| paired | 0 | 5 | 0 | 0 | 0 |
| paired | 1 | 0.25 | 0.9748 | 0.9761 | 0.1255 |
| paired | 1 | 1 | 0.6835 | 0.6897 | 2.0099 |
| paired | 1 | 5 | 0.5382 | 0.5394 | 49.6297 |
| paired | 2 | 0.25 | 0.9997 | 0.9997 | 0.2469 |
| paired | 2 | 1 | 0.8048 | 0.8023 | 4.0032 |
| paired | 2 | 5 | 0.5717 | 0.5675 | 98.0402 |
| paired | 4 | 0.25 | 1 | 1 | 0.5048 |
| paired | 4 | 1 | 0.905 | 0.9046 | 8.0178 |
| paired | 4 | 5 | 0.5987 | 0.6032 | 199.3003 |
| unpaired | 0 | 0.25 | 0.4268 | 0.4312 | 3.0124 |
| unpaired | 0 | 1 | 0.4817 | 0.4827 | 47.7076 |
| unpaired | 0 | 5 | 0.4964 | 0.4965 | 1192.9624 |
| unpaired | 1 | 0.25 | 0.6502 | 0.6569 | 3.0542 |
| unpaired | 1 | 1 | 0.5436 | 0.5402 | 48.5607 |
| unpaired | 1 | 5 | 0.5113 | 0.5081 | 1194.3801 |
| unpaired | 2 | 0.25 | 0.8409 | 0.8368 | 2.9856 |
| unpaired | 2 | 1 | 0.5973 | 0.5969 | 47.4094 |
| unpaired | 2 | 5 | 0.5189 | 0.5196 | 1216.6227 |
| unpaired | 4 | 0.25 | 0.9845 | 0.9837 | 2.9516 |
| unpaired | 4 | 1 | 0.7028 | 0.7033 | 47.9345 |
| unpaired | 4 | 5 | 0.5382 | 0.5425 | 1190.3855 |

## Reading

- **N1** paired, h=0: margin ≡ −0.3 with ZERO variance at σ=0.25, 1.0 and 5.0 alike — no noise process touches the zero-coverage margin, because shared behavior receives shared observations and cancels (T1). v2's inseparability is noise-robust, not a noiseless idealization.
- **N2** paired, h≥1: the v2 step function smears into a probit CENTERED on the same crisp inequality, width = per-hit noise only.
- **N3** unpaired: variance inflates from 2σ²h to 2σ²|S| (measured ratios match h/|S|); at h=0 the comparison degrades to a near-coin-flip. The 'noisy, uninterpretable margin' of standard evaluations is unpaired evaluation on a disagreement-sparse set — derived, not observed.
- **N4** worst-case noise tolerance: paired = margin/(2h) per observation, unpaired = margin/(2|S|) — the ratio is exactly the disagreement fraction. Pairing is why differential/targeted evaluation works; it is D-restriction at the evaluation layer, the same move interp makes at the hypothesis layer (v2 Q5).

## Honest boundary

MC verifies the Gaussian laws; the exact-cancellation and bounded-noise inequalities (T1–T3) are the Lean targets in sundogcert (`PercivalNoisyMargin.lean`). The B1/B2 bridge predictions (variance collapse on real checkpoint pairs; prior-driven near-determinism at behavioral agreement) remain registered, unrun — that is where this can die on real systems.

