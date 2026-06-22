# H1.3 Medium Trust Scaling Probe Results

Status: **INDICATIVE SUPPORT** on 2026-06-22. This H1.3-a probe is not the
binding H1.3 verdict; it is the capped Medium read that estimates throughput
and checks whether the Medium tier is worth binding.

Spec: [`H1_3_MEDIUM_TRUST_SCALING_SPEC.md`](H1_3_MEDIUM_TRUST_SCALING_SPEC.md)

Artifacts:

- dataset: `results/mesa/h1-pantheon/h1_3_medium_trust_probe/dataset/`
- supervised init: `results/mesa/h1-pantheon/h1_3_medium_trust_probe/models_sup/`
- PPO checkpoint: `results/mesa/h1-pantheon/h1_3_medium_trust_probe/models/`
- ablated eval: `results/mesa/h1-pantheon/h1_3_medium_trust_probe/eval_ablate/`
- intact eval: `results/mesa/h1-pantheon/h1_3_medium_trust_probe/eval/`

## Shape

- cells: `nominal,geometric-light,sensor-delay-light`
- train seeds: `64`; val seeds: `16`
- horizon: `200`
- PPO updates: `64`, PPO seed `0`
- eval seeds: `10000-10015`
- Medium frozen heads, 23 trust features, passive guard
- caps: `field=1.00 / reward=0.50 / guard=0.70`
- same-run equal-budget M-adapter with identical 23 controller features

## Readback

Dataset:

- rows: train `29462`, val `6779`, total `36241`
- basin rollouts in dataset: `50`
- elapsed: `30.592 s`; throughput: `1184.7 rows/s`
- feature audit: clean; no missing trust features; no privileged feature names

Supervised init:

- hidden size: `32`
- controller params: guard `1857`, arbiter `1955`, council total `3812`
- M-adapter params: `3852`; budget ratio `1.0105`; within 5%: `true`
- validation: guard AUC `0.8442`, guard calibration error `0.1543`,
  arbiter CE `1.0151`, M-adapter MSE `0.24534`

PPO:

- updates: `64/64`
- env steps: `41177`
- elapsed: `86.911 s`; throughput: `473.78 env-steps/s`
- checkpoint cadence: every `16` updates

Eval:

| controller | S_T | GI basin | success | bull breach |
| --- | ---: | ---: | ---: | ---: |
| Learned-P-Council | 0.78282 | 0.0625 | 0.3125 | 0 |
| M-Adapter | 0.82511 | 0.1667 | 0.3542 |  |
| Blind-Council | 0.83017 | 0.0625 | 0.125 | 0.3125 |

Gates:

- competence noninferior: `true` (council trails M by `0.04229`, inside the
  `0.05` band)
- GI proxy capture strict: `true` (`0.0625` council vs `0.1667` monolith)
- trust attribution: `true` (intact advantage `0.1042`; ablated advantage `0`;
  delta `0.1042`)
- bull discipline: `true` (`max_reward_w=0.41131`, no bull breaches)
- fairness: `true` (feature audit clean, budget ratio `1.0105`)

Branch selected by the probe: `H1_3_MEDIUM_SUPPORT`.

## Interpretation

The Medium three-cell probe keeps the H1.2f mechanism alive: with the trust
features intact, the council is competence-noninferior and cuts GI basin
capture below the equally enriched monolith; when the trust features are
zeroed, that advantage collapses to a tie. This is exactly enough to justify
the full H1.3-b binding run.

It is not enough to claim that the advantage scales. The probe covers only
three gradient-intact cells, `16` eval seeds, and one PPO seed. H1.3-b must
decide the branch on the full 13-cell slate.

## Binding Estimate

Measured H1.3-a rates:

- dataset: `1184.7 rows/s`
- PPO: `473.78 env-steps/s`
- eval: `144 trials / 2.177 s`

H1.3-a used a thin PPO probe budget (`2` rollouts/update). H1.3-b uses the
binding budget (`64` rollouts/update), so PPO scales by about `256x` from the
probe, not by `8x`.

Extrapolated H1.3-b local wall-clock: **about 6-7 hours** for dataset,
supervised init, 512 PPO updates, ablated eval, and intact eval. That exceeds
the repo's inline rule, so H1.3-b is staged for the operator in the spec rather
than run inline.
