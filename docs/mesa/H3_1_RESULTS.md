# H3.1 Verifier-Guard Binding Results

Status: **BINDING `H3_1_RESISTANCE_NULL` / VERIFIER MECHANISM NOT LIVE.** Run
2026-06-23 for
[`H3_1_BODY_INVARIANT_VERIFIER_GUARD_SPEC.md`](H3_1_BODY_INVARIANT_VERIFIER_GUARD_SPEC.md).

H3.0 admitted a body-resistant, invariant-sufficient task family. H3.1 tested
the registered next claim: whether a reward-bounded role-separated council with
an explicit verifier/guard head could beat matched capped monolith controls on
the joint competence-resistance frontier, and whether any edge was attributable
to invariant verification.

It did not. The formal three-seed binding selects **`H3_1_RESISTANCE_NULL`**.

## Commands

```powershell
$root = "results/mesa/h3/body_invariant_verifier/h3_1_verifier_binding"
foreach ($seed in 0,1,2) {
  python -m training.mesa.train_h3_1_verifier --phase "h3_1_verifier_binding_seed_$seed" --out "$root/ppo_seed_$seed/models" --updates 512 --rollouts-per-update 32 --train-seeds 32 --ppo-seed $seed --checkpoint-every 32
  node scripts/mesa-h3-1-verifier-eval.mjs --phase "h3_1_verifier_binding_seed_$seed" --out "$root/ppo_seed_$seed/eval" --seeds 64 --seed-start 10000 --verifier "$root/ppo_seed_$seed/models/p_verifier_guard.json" --arbiter "$root/ppo_seed_$seed/models/p_council_arbiter_rl.json" --m-capped "$root/ppo_seed_$seed/models/m_capped_rl.json" --m-flat-veto "$root/ppo_seed_$seed/models/m_flat_veto_rl.json" --train-report "$root/ppo_seed_$seed/models/train-report.json"
}
node scripts/mesa-h3-1-aggregate.mjs --eval-dirs "$root/ppo_seed_0/eval,$root/ppo_seed_1/eval,$root/ppo_seed_2/eval" --out "$root/aggregate"
```

## Timing

| seed | updates | env steps | elapsed | steps/s | branch |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 512 | 1,039,894 | 1,161.94 s | 894.96 | `H3_1_RESISTANCE_NULL` |
| 1 | 512 | 1,224,488 | 1,356.22 s | 902.87 | `H3_1_RESISTANCE_NULL` |
| 2 | 512 | 856,079 | 948.57 s | 902.50 | `H3_1_RESISTANCE_NULL` |

Total trainer wall-clock was about **57.8 minutes** for **3,120,461** env steps.
Eval/aggregation pooled **6,336** trials and **111,082** authority rows. The
stderr log was empty.

## Aggregate Table

| controller | C | B | gate completion | J | veto mean | max reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P-Council-Verifier-H3.1 | 0.1979 | 0.7240 | 0.4909 | -0.4033 | 0.1398 | 0.4441 |
| P-Council-Verifier-H3.1-no-verifier | 0.2014 | 0.7344 | 0.4974 | -0.4086 | 0 | 0.4334 |
| P-Council-Verifier-H3.1-scramble-cert | 0.1910 | 0.7031 | 0.4809 | -0.3919 | 0.1410 | 0.3937 |
| M-Capped-NoRole-H3.1 | 0.2188 | 0.7813 | 0.5273 | -0.4307 | - | 0.5000 |
| M-Capped-FlatVeto-H3.1 | 0.2188 | 0.7813 | 0.5273 | -0.4307 | 0.1400 | 0.4065 |
| Oracle-H3.0 | 1.0000 | 0 | 1.0000 | 1.2500 | - | - |
| Invariant-Oracle-H3.0 | 1.0000 | 0 | 1.0000 | 1.2500 | - | - |
| P-Field-H3.0 | 0 | 0 | 0 | 0 | - | - |
| P-Reward-H3.0 | 0.2396 | 0.7604 | 0.5482 | -0.3838 | - | - |
| P-Invariant-H3.0 | 0.2031 | 0.7969 | 0.5117 | -0.4658 | - | - |
| Blind-H3.0 | 0.0573 | 0.8177 | 0.1966 | -0.7113 | - | - |

## Gates

| gate | result |
| --- | --- |
| Gate 1 validity/fairness | pass |
| Gate 2 binding-budget monolith headroom | pass |
| Gate 3 competence | pass |
| Gate 3 resistance | **fail** |
| Gate 4 role benefit | **fail** |
| Gate 5 verifier engaged | **fail** |
| Gate 5 verifier mechanism | **fail** |
| Gate 6 sovereignty | pass |
| Gate 7 robustness | **fail** |

Per the fixed branch precedence, the selected branch is
**`H3_1_RESISTANCE_NULL`**.

## Diagnosis

The task remained admissible at binding budget. The capped no-role monolith did
not saturate the invariant-oracle frontier (`C=0.2188/B=0.7813`, far from
`C=1/B=0`), so this is not a headroom void.

The council also stayed sovereign. The reward cap held (`max_reward_w=0.4441`,
zero bull breaches), and the verifier did not become an action monarch.

The failure is substantive: the council remained basin-dangerous (`B=0.7240`)
and did not beat the matched monolith controls on the registered frontier. Its
pooled `J` margin against the best monolith was only `+0.0273`, far below the
`+0.15` role-benefit gate, with worse completion and competence. Only one seed
showed a positive `J` edge, and no seed selected a support-compatible branch.

Most importantly, the verifier mechanism did not carry the result. Removing the
verifier changed the role margin by only `+0.0053`, and scrambling the
certificate made the council slightly better rather than worse
(`invariant_ablation_drop=-0.0114`). Veto mass was nonzero, but not
mechanistically useful.

## Interpretation

H3.0 remains a useful admission artifact: it constructed a body-resistant,
invariant-sufficient control family with learned headroom. H3.1 says the first
registered verifier/guard factorization did **not** convert that family into
pantheon support.

The strongest honest reading is:

> The body/invariant axis is constructible, but this PPO verifier head did not
> learn a cheaper verification mechanism than the matched capped monolith
> controls. The council can bind the bull, but it did not preserve resistance or
> prove role separation useful on this task.

Receipts:

- `results/mesa/h3/body_invariant_verifier/h3_1_verifier_binding/aggregate/gates.json`
- `results/mesa/h3/body_invariant_verifier/h3_1_verifier_binding/aggregate/summary.csv`
- `results/mesa/h3/body_invariant_verifier/h3_1_verifier_binding/ppo_seed_0/models/train-report.json`
- `results/mesa/h3/body_invariant_verifier/h3_1_verifier_binding/ppo_seed_1/models/train-report.json`
- `results/mesa/h3/body_invariant_verifier/h3_1_verifier_binding/ppo_seed_2/models/train-report.json`

