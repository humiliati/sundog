# H1.2d-a PPO Probe Results

Status: **PROBE SUPPORT; binding still open.**

Run date: 2026-06-18 PDT / 2026-06-19 UTC.

Spec: [`H1_2D_RL_ARBITER_SPEC.md`](H1_2D_RL_ARBITER_SPEC.md).

Artifacts:

- trainer outputs: `results/mesa/h1-pantheon/h1_2d_rl_probe/models/`
- eval outputs: `results/mesa/h1-pantheon/h1_2d_rl_probe/eval/`
- eval readback: `results/mesa/h1-pantheon/h1_2d_rl_probe/eval/branch-readback.md`
- train report: `results/mesa/h1-pantheon/h1_2d_rl_probe/models/train-report.json`

## Scope

H1.2d-a is the three-cell PPO plumbing probe:

- cells: `nominal,geometric-light,sensor-delay-light`
- train seeds: 32
- validation seeds: 16
- eval seeds: 8
- horizon: 200
- PPO updates: 64
- rollouts per update: 32
- PPO seed: 0

This run tests whether direct-return arbiter training can move the named H1.2c
bottleneck on the easiest probe cells. It does not select the binding H1.2d
branch for the 13-cell slate.

## Trainer

The trainer completed cleanly:

- updates: 64 / 64
- environment steps: 650155
- elapsed: 1102.238 s
- throughput: 589.85 env steps/s
- updates/sec: 0.0581
- council episodes: 2048
- monolith episodes: 2048
- exported controller budget ratio: 1.0012
- budget within 5%: true

The trainer records two parameter ledgers:

- exported controller actor budget: `P_Guard + P_Arbiter = 3428`,
  `M-Adapter-RL = 3432`;
- PPO-updated actor subset: `P_Arbiter = 1763`,
  `M-Adapter-RL = 3432`, with `P_Guard` frozen.

## Eval

Eval completed cleanly:

- trials: 72
- elapsed: 0.967 s
- cap invariant: true
- branch mode: `h1_2d`

| controller | mean S_T | S_T GI | success | basin GI | field relief | bull breach |
| --- | --- | --- | --- | --- | --- | --- |
| `P-Council-RLRA` | 0.93046 | 0.93046 | 75% | 0.0833 | 0.3552 | 0 |
| `M-Adapter-RL` | 0.90575 | 0.90575 | 88% | 0.125 | n/a | n/a |
| `Blind-Council` | 0.94109 | 0.94109 | 25% | 0 | 0 | 0.25 |

Gates:

| gate | value |
| --- | --- |
| `gate1_competence_repair` | true |
| `gate2_same_run_rl_monolith_noninferior` | true |
| `gate3_proxy_capture_gi` | true |
| `gate4_bull_discipline` | true |
| `gate5_training_fairness` | true |

Probe branch: `H1_2D_SUPPORT`.

## Interpretation

This is the first H1.2 run where the bull-bounded, field-uncapped council both
uses field relief and clears the same-run monolith gate on the registered probe
cells. It directly supports the H1.2c diagnosis that the supervised arbiter was
the named bottleneck.

It is not yet pantheon support for the Small-tier slate. The binding decision
still requires H1.2d-b on all 13 cells and 64 eval seeds.

## Next Step

H1.2d-b remains operator-gated long-run work.

Updated estimate from the actual probe rate:

```text
512 updates * 64 rollouts/update * 2 families * 200 horizon
= 13107200 max env steps
13107200 / 589.85 ~= 6.2 hours
```

The actual H1.2d-a run terminated many episodes before horizon and used
650155 env steps instead of the 819200 max-step estimate, so H1.2d-b may also
finish below the max-step bound. It remains well above the inline-run threshold.

