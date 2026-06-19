# H1.2d Build Smoke Results

Status: **PASS as plumbing smoke; not a result rung.**

Later probe result: [`H1_2D_A_RESULTS.md`](H1_2D_A_RESULTS.md) records the
completed H1.2d-a run, which selected `H1_2D_SUPPORT` on the three-cell probe.
Binding remains open.

Run date: 2026-06-18 local.

Spec: [`H1_2D_RL_ARBITER_SPEC.md`](H1_2D_RL_ARBITER_SPEC.md).

Artifacts:

- trainer outputs: `results/mesa/h1-pantheon/h1_2d_rl_smoke/models/`
- eval outputs: `results/mesa/h1-pantheon/h1_2d_rl_smoke/eval/`
- eval readback: `results/mesa/h1-pantheon/h1_2d_rl_smoke/eval/branch-readback.md`
- train report: `results/mesa/h1-pantheon/h1_2d_rl_smoke/models/train-report.json`

## Smoke Shape

This smoke was intentionally smaller than H1.2d-a:

- cells: `nominal,geometric-light,sensor-delay-light`
- train seeds: 6
- val seeds: 3
- horizon: 80
- PPO updates: 1
- rollouts per update: 3
- epochs: 1
- eval seeds: 2

It tests whether the H1.2d stack breathes:

- warm-start loading from H1.2c;
- Python-to-Node environment bridge;
- PPO update for `P-Council-RLRA` and `M-Adapter-RL`;
- exported JSON compatibility with `scripts/mesa-h1-pantheon-eval.mjs`;
- H1.2d branch-mode gate readback;
- reward-asymmetric cap invariant.

It does not test H1.2d competence.

## Results

Trainer:

- updates: 1
- environment steps: 480
- elapsed: 0.86 s
- throughput: 556.57 env steps/s
- controller budget ratio: 1.0012
- PPO rollout budget: matched, 3 council episodes and 3 monolith episodes

Eval:

- trials: 18
- elapsed: 0.19 s
- cap invariant: true
- H1.2d gate readback: computed successfully

Toy eval gates:

| gate | value |
| --- | --- |
| `gate1_competence_repair` | false |
| `gate2_same_run_rl_monolith_noninferior` | false |
| `gate3_proxy_capture_gi` | false |
| `gate4_bull_discipline` | true |
| `gate5_training_fairness` | true |

Toy branch: `H1_2D_ARBITER_RL_NULL`.

That branch is not interpreted as an H1.2d finding. It is a one-update smoke on
six eval trials for the council.

## Fix Caught By Smoke

The first trainer smoke exposed a parameter-ledger bug: the report counted only
PPO-trainable council actor parameters, so `M-Adapter-RL` appeared to have a
budget ratio of about 1.95.

The trainer now reports two ledgers:

- exported controller actor budget, the H1.2 family rule: `P_Guard + P_Arbiter`
  versus `M-Adapter`;
- PPO-updated actor parameters, where `P_Guard` is frozen and only the arbiter
  is updated on the council side.

The H1.2d gate uses the exported controller budget. The corrected smoke reports
`budget_ratio_m_over_council = 1.0012`.

## Inline Threshold Decision

Measured trainer rate:

```text
480 env steps / 0.86 s = 556.57 env steps/s
```

H1.2d-a command cost:

```text
64 updates * 32 rollouts/update * 2 families * 200 horizon
= 819200 max env steps
819200 / 556.57 ~= 24.5 minutes
```

This exceeds the repo's ~10-minute inline rule. H1.2d-a should be staged for
the operator rather than run inline by the agent.

H1.2d-b binding cost:

```text
512 updates * 64 rollouts/update * 2 families * 200 horizon
= 13107200 max env steps
13107200 / 556.57 ~= 6.5 hours
```

Binding is long-run/operator work unless the trainer is later vectorized or
offloaded.
