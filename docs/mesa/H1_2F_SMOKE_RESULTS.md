# H1.2f-0 Trust-Features Smoke Results

Status: **SMOKE PASSED** on 2026-06-20. This is a plumbing/admission smoke,
not an H1.2f result.

Spec: [`H1_2F_TRUST_FEATURES_SPEC.md`](H1_2F_TRUST_FEATURES_SPEC.md).

Artifacts:

- dataset: `results/mesa/h1-pantheon/h1_2f_trust_smoke/dataset/`
- supervised init: `results/mesa/h1-pantheon/h1_2f_trust_smoke/models_sup/`
- PPO smoke models: `results/mesa/h1-pantheon/h1_2f_trust_smoke/models/`
- ablated eval: `results/mesa/h1-pantheon/h1_2f_trust_smoke/eval_ablate/`
- intact eval: `results/mesa/h1-pantheon/h1_2f_trust_smoke/eval/`

## What Was Exercised

- `feature_mode=trust` through all three required paths: Node dataset builder,
  Python PPO trainer, and Node eval.
- The six `K=8` temporal trust features were emitted as inference features:
  `sample_dispersion`, `sLocal_var_K`, `grad_norm_var_K`,
  `grad_dir_stability_K`, `disagree_mean_K`, `act_dir_consistency_K`.
- The supervised trainer consumed the schema-driven 23-feature set without a
  special path.
- PPO ran one update with the enriched feature map and exported resume state.
- Eval ran both trust-feature ablation (`--trust-ablation zero`) and intact
  inference, with the intact run reading the ablation artifact for gate 3.

## Readback

Dataset smoke:

- rows: 540 train / 360 val
- feature count: 23
- feature audit: no missing base features, no missing trust features, no
  forbidden/privileged feature names
- caps: field 1.00 / reward 0.50 / guard 0.70

Supervised init:

- params: guard 1857, arbiter 1955, council total 3812, monolith 3852
- budget ratio: 1.010 within 5%

PPO smoke:

- updates: 1
- env steps: 240
- elapsed: 0.66 s
- env steps/sec: 365.34
- checkpoint/resume artifact: `models/train_state.pt`

Eval smoke:

- intact eval: 3 controllers x 3 cells x 2 seeds = 18 trials
- `cap_ok=true`
- max council `w_reward=0.37351` (below 0.50 cap)
- feature fairness audit: guard 23, arbiter 24 (`guard_risk` only extra),
  monolith 23; same controller features; no privileged names
- ablation readback: intact GI advantage 0, ablated GI advantage 0,
  attribution delta 0

The smoke branch was `H1_2F_PROXY_NULL` because this tiny three-cell, two-seed
run produced zero GI basin captures for both council and monolith. That branch
is not interpreted as evidence; the smoke decision is only that the H1.2f
plumbing, fairness audit, cap invariant, PPO export, ablation readback, and gate
computation are live.

## Admission

H1.2f-0 exits met. Next rung is H1.2f-a: a three-cell supervised + RL probe with
enough PPO updates to estimate binding wall-clock and verify trust-feature usage
under a nontrivial controller.
