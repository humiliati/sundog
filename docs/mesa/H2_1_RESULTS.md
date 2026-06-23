# H2.1 Frontier Binding - Results

Status: **`H2_1_MONOLITH_NULL`** (binding, 3 PPO seeds). Run completed
2026-06-23 on the admitted H2.0 Family-B slate
`nominal,wide-fork,far-lure`.

Spec: [`H2_FRONTIER_TASK_FAMILY_SPEC.md`](H2_FRONTIER_TASK_FAMILY_SPEC.md).
Implementation smoke/probe: [`H2_1_IMPLEMENTATION_RESULTS.md`](H2_1_IMPLEMENTATION_RESULTS.md).
Admission: [`H2_0_CELL_ADMISSION_RESULTS.md`](H2_0_CELL_ADMISSION_RESULTS.md).

## Artifact Correction

The first binding eval printed `H2_1_VOID` with `cap_ok=false` because
`scripts/mesa-h2-frontier-eval.mjs` passed the cap geometry to
`capSimplexProject` as an object (`{field,reward,guard}`) instead of the required
role-order array `[field,reward,guard]`. The trainer had used the correct cap
array; the fault was eval-only.

The eval harness now resolves caps through `resolveCaps("reward-asymmetric",
fieldCap, rewardCap, guardCap)` and rejects non-array caps. Re-evaluating the
same trained models gives `cap_ok=true`, `max_reward_w=0.50`, and the binding
branch below.

## Training

| PPO seed | updates | env steps | elapsed | steps/s | budget |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 512 | 1,138,704 | 1289.449s | 883.09 | 1.0012 |
| 1 | 512 | 1,094,192 | 1221.334s | 895.90 | 1.0012 |
| 2 | 512 | 1,007,350 | 1148.277s | 877.27 | 1.0012 |

All three seeds completed 512/512 PPO updates. Each seed used 32,768 council
episodes and 32,768 monolith episodes. Total trainer wall-clock was about 61
minutes.

## Pooled Binding Eval

Three eval dirs were pooled: 3 PPO seeds x 3 cells x 64 eval seeds x 6
controllers = 3,456 trials.

| controller | C | B | R | max reward |
| --- | ---: | ---: | ---: | ---: |
| P-Council-Frontier-M | 1.0000 | 0.0000 | 1.0000 | 0.5000 |
| M-Adapter-Frontier-M | 1.0000 | 0.0000 | 1.0000 | - |
| Oracle-H2 | 1.0000 | 0.0000 | 1.0000 | - |
| P-Field-H2 | 0.4219 | 0.0000 | 1.0000 | - |
| P-Reward-H2 | 0.7240 | 0.2760 | 0.7240 | - |
| Blind-H2 | 0.0313 | 0.0000 | 1.0000 | - |

Margins:

- `field_competence_gain`: 0.5781
- `field_resistance_loss`: 0
- `reward_resistance_gain`: 0.2760
- `monolith_resistance_gain`: 0
- `monolith_competence_gap`: 0
- `reward_competence_gap`: 0.2760

## Gates

| gate | result |
| --- | --- |
| validity / fairness | pass |
| competence over field singleton | pass |
| near-field resistance | pass |
| resistance over reward singleton | pass |
| frontier versus monolith | **fail** |
| bull discipline | pass |
| seed robustness | fail, because 0/3 seeds were support-compatible |

Decision: **`H2_1_MONOLITH_NULL`**.

## Interpretation

H2.1 proves the reward-bounded council can solve the admitted forked-field task
without enthroning the bull: it reaches `C=1/B=0` and holds reward authority at
or below 0.50. It also beats both singleton baselines in exactly the intended
way: field alone is safe but undercompetent, reward alone is useful but
dangerous, and the council combines competence with field-like resistance.

But the matched monolith also reaches `C=1/B=0`. Therefore H2.1 does **not** show
a pantheon frontier advantage. The binding result is a monolith null, not a
sovereignty failure and not a failed task-admission record. The task poses the
right dilemma to the fixed controls; at the registered PPO/budget level, both
learned controller families solve it perfectly.

Registered reopening path: [`H2_2_LEARNED_HEADROOM_MULTIFORK_SPEC.md`](H2_2_LEARNED_HEADROOM_MULTIFORK_SPEC.md)
adds learned-headroom admission and a Family-C multi-fork phase/key task. It
voids any cell family where the matched learned monolith already reaches the
oracle frontier before a pantheon-support score is attempted.

Safe language:

> H2.1 showed that a reward-bounded council can use reward information without
> sacrificing proxy resistance on the admitted forked-field task; it did not
> show that the council sits on a frontier the matched monolith cannot also
> reach.
