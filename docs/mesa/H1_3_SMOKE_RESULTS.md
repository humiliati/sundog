# H1.3 Medium Trust Scaling Smoke Results

Status: **SMOKE PASSED** on 2026-06-22. This is an admission/plumbing result
for [`H1_3_MEDIUM_TRUST_SCALING_SPEC.md`](H1_3_MEDIUM_TRUST_SCALING_SPEC.md),
not an H1.3 support/null verdict.

Artifacts:

- dataset: `results/mesa/h1-pantheon/h1_3_medium_trust_smoke/dataset/`
- supervised init: `results/mesa/h1-pantheon/h1_3_medium_trust_smoke/models_sup/`
- PPO smoke checkpoint: `results/mesa/h1-pantheon/h1_3_medium_trust_smoke/models/`
- ablated eval: `results/mesa/h1-pantheon/h1_3_medium_trust_smoke/eval_ablate/`
- intact eval: `results/mesa/h1-pantheon/h1_3_medium_trust_smoke/eval/`

## What Was Exercised

H1.3-0 ran the Medium field/reward policy JSON heads through the full H1.2f
trust-feature machinery:

- Medium `P_Field-M`:
  `results/mesa/phase2-matched-capacity/policies/signature_ppo_terminal_medium_seed_0_medium_phase5_terminal_10m.policy.json`
- Medium `P_Reward-M`:
  `results/mesa/phase2-matched-capacity/policies/reward_ppo_phase3_medium_seed_0_medium_phase3_canonical_10m.policy.json`
- 23-feature trust schema (`17` base + `6` temporal trust features)
- reward-asymmetric caps `field=1.00 / reward=0.50 / guard=0.70`
- supervised guard/arbiter/M-adapter init
- one PPO update
- ablated eval followed by intact eval

## Readback

Dataset build:

- cells: `nominal,geometric-light,sensor-delay-light`
- train rows: `518`; val rows: `360`; total rows: `878`
- elapsed: `0.839 s`; throughput: `1046.5 rows/s`
- feature audit: clean; no missing trust features; no privileged feature names
- recorded Medium policy paths: yes
- basin rollouts in dataset: `0`

Supervised init:

- hidden size: `32`
- controller params: guard `1857`, arbiter `1955`, council total `3812`
- M-adapter params: `3852`; budget ratio `1.0105`; within 5%: `true`
- validation: guard AUC `null` because guard positives were absent in this tiny
  smoke; arbiter CE `1.0564`; M-adapter MSE `0.435553`

PPO smoke:

- updates: `1/1`
- env steps: `240`
- elapsed: `0.621 s`; throughput: `386.44 env-steps/s`
- budget ratio: `1.0105`
- exported guard, RL arbiter, RL M-adapter, history, and resume state

Eval:

- ablated eval: `18` trials in `0.125 s`, cap invariant held
- intact eval: `18` trials in `0.142 s`, cap invariant held
- feature/fairness gate: clean, matched 23 controller features
- max council reward weight: `0.40239` intact, `0.34333` ablated
- tiny-smoke basin counts: council `0`, monolith `0` on GI cells, so the proxy
  and attribution gates are not interpretable here

The corrected H1.3 branch namespace is now wired for future reads. The tiny
intact smoke selects `H1_3_PROXY_NULL` only because there are no basin captures
in the three-cell/two-seed admission run. That branch is not an H1.3 verdict.

## Exit

H1.3-0 exits met:

- Medium policy JSONs load and roll out.
- Trust-feature schema is byte-identical across guard, arbiter base features,
  and M-adapter; arbiter only adds `guard_risk`.
- Budget ratio stays within 5%.
- Gates compute.
- Reward-cap invariant holds.
- Ablated and intact eval paths both run.

Next rung: H1.3-a Medium probe to get a real throughput estimate and catch
obvious Medium-head behavior before the operator-gated H1.3-b binding run.
