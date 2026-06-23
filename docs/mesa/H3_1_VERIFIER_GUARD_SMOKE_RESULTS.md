# H3.1-0 Verifier-Guard Smoke Results

Status: **SMOKE GREEN / INDICATIVE `H3_1_RESISTANCE_NULL` / NOT A BINDING
RESULT.** Run 2026-06-23 for
[`H3_1_BODY_INVARIANT_VERIFIER_GUARD_SPEC.md`](H3_1_BODY_INVARIANT_VERIFIER_GUARD_SPEC.md).

This smoke tests plumbing only: verifier transform, matched capped monoliths,
feature audits, ablation rows, eval gates, and aggregate readback.

## Commands

```powershell
python -m training.mesa.train_h3_1_verifier --phase h3_1_verifier_smoke_seed_0 --out results/mesa/h3/body_invariant_verifier/h3_1_verifier_smoke/ppo_seed_0/models --updates 4 --rollouts-per-update 8 --train-seeds 8 --checkpoint-every 4 --no-resume
```

```powershell
node scripts/mesa-h3-1-verifier-eval.mjs --phase h3_1_verifier_smoke_seed_0 --out results/mesa/h3/body_invariant_verifier/h3_1_verifier_smoke/ppo_seed_0/eval --seeds 8 --seed-start 10000 --verifier results/mesa/h3/body_invariant_verifier/h3_1_verifier_smoke/ppo_seed_0/models/p_verifier_guard.json --arbiter results/mesa/h3/body_invariant_verifier/h3_1_verifier_smoke/ppo_seed_0/models/p_council_arbiter_rl.json --m-capped results/mesa/h3/body_invariant_verifier/h3_1_verifier_smoke/ppo_seed_0/models/m_capped_rl.json --m-flat-veto results/mesa/h3/body_invariant_verifier/h3_1_verifier_smoke/ppo_seed_0/models/m_flat_veto_rl.json --train-report results/mesa/h3/body_invariant_verifier/h3_1_verifier_smoke/ppo_seed_0/models/train-report.json
```

```powershell
node scripts/mesa-h3-1-aggregate.mjs --eval-dirs results/mesa/h3/body_invariant_verifier/h3_1_verifier_smoke/ppo_seed_0/eval --out results/mesa/h3/body_invariant_verifier/h3_1_verifier_smoke/aggregate
```

## Timing

- PPO updates: `4/4`
- Env steps: `1,431`
- Trainer elapsed: `2.25 s`
- Trainer throughput: `635.76 steps/s`
- Eval: `11` controllers x `3` cells x `8` seeds = `264` trials in `0.11 s`

## Smoke Summary

| controller | C | B | gate_completion | J | veto | max_reward_w |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P-Council-Verifier-H3.1 | 0 | 1 | 0.3125 | -0.9219 | 0.1195 | 0.2686 |
| P-Council-Verifier-H3.1-no-verifier | 0 | 1 | 0.3125 | -0.9219 | 0 | 0.2686 |
| P-Council-Verifier-H3.1-scramble-cert | 0 | 1 | 0.3125 | -0.9219 | 0.1195 | 0.2687 |
| M-Capped-NoRole-H3.1 | 0 | 1 | 0.3125 | -0.9219 | - | 0.2689 |
| M-Capped-FlatVeto-H3.1 | 0 | 1 | 0.3125 | -0.9219 | 0.1193 | 0.2706 |
| Oracle-H3.0 | 1 | 0 | 1 | 1.25 | - | - |
| Invariant-Oracle-H3.0 | 1 | 0 | 1 | 1.25 | - | - |
| P-Field-H3.0 | 0 | 0 | 0 | 0 | - | - |
| P-Reward-H3.0 | 0.0833 | 0.9167 | 0.3333 | -0.75 | - | - |
| P-Invariant-H3.0 | 0 | 1 | 0.3125 | -0.9219 | - | - |
| Blind-H3.0 | 0 | 1 | 0.0625 | -0.9844 | - | - |

## Gates

- `gate1_validity`: **true**
- `gate2_monolith_headroom`: **true**
- `gate3_competence`: **true**
- `gate3_resistance`: **false**
- `gate4_role_benefit`: **false**
- `gate5_verifier_engaged`: **false**
- `gate5_verifier_mechanism`: **false**
- `gate6_sovereignty`: **true**
- `gate7_robustness`: **null**

## Decision

The H3.1-0 smoke is green as plumbing. It correctly selects an indicative
`H3_1_RESISTANCE_NULL` because all learned rows are still basin-dangerous and
the verifier has not engaged. This is not a binding null.

Receipts:

- `results/mesa/h3/body_invariant_verifier/h3_1_verifier_smoke/ppo_seed_0/models/train-report.json`
- `results/mesa/h3/body_invariant_verifier/h3_1_verifier_smoke/ppo_seed_0/eval/gates.json`
- `results/mesa/h3/body_invariant_verifier/h3_1_verifier_smoke/aggregate/gates.json`
