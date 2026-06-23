# H2.1 Controller Integration - Implementation Smoke + Probe

Status: **IMPLEMENTED / H2.1-a PROBE = `H2_1_MONOLITH_NULL` (single PPO seed,
indicative).** This records the implementation smoke and capped probe. The later
three-seed binding is recorded in [`H2_1_RESULTS.md`](H2_1_RESULTS.md).

Parent spec: [`H2_FRONTIER_TASK_FAMILY_SPEC.md`](H2_FRONTIER_TASK_FAMILY_SPEC.md).
Admission record: [`H2_0_CELL_ADMISSION_RESULTS.md`](H2_0_CELL_ADMISSION_RESULTS.md).

## What Was Built

- `training/mesa/h2_forked_task.py` mirrors `scripts/h2-forked-task.mjs`
  Python-side for PPO rollouts.
- `scripts/mesa-h2-forked-fixtures.mjs` + `scripts/mesa-h2-forked-parity.py`
  generate and replay JS traces before training is trusted.
- `training/mesa/train_h2_frontier.py` trains the H2.1 council and matched
  monolith on terminal `competence - basin`.
- `scripts/mesa-h2-frontier-eval.mjs` evaluates council, monolith, Oracle,
  field singleton, reward singleton, and Blind rows on the canonical JS env.
- `scripts/mesa-h2-frontier-aggregate.mjs` pools eval directories across PPO
  seeds and selects the binding branch.

The H2 cell constants now live in `scripts/h2-forked-task.mjs`; the H2.0
admission script default was corrected to the admitted slate
`nominal,wide-fork,far-lure`.

## Parity Smoke

Command:

```powershell
node scripts/mesa-h2-forked-fixtures.mjs --out results/mesa/h2-frontier/h2_1_parity/fixtures.json --seeds 10000,10001,10002,10003 --cells nominal,wide-fork,far-lure,strong-lure,near-lure
python scripts/mesa-h2-forked-parity.py --fixtures results/mesa/h2-frontier/h2_1_parity/fixtures.json --tol 1e-9
```

Result:

- 80 JS fixture episodes.
- 1,711 replayed steps.
- `max_abs_diff=3.28e-13` at tolerance `1e-9`.
- Decision: Python mirror is fit for H2.1 PPO rollouts.

## Implementation Smoke

Command:

```powershell
python -m training.mesa.train_h2_frontier --phase h2_1_impl_smoke --out results/mesa/h2-frontier/h2_1_impl_smoke/models --cells nominal,wide-fork,far-lure --train-seeds 4 --train-seed-start 20000 --updates 2 --rollouts-per-update 4 --epochs 1 --minibatch-size 32 --checkpoint-every 1 --no-resume

node scripts/mesa-h2-frontier-eval.mjs --phase h2_1_impl_smoke --out results/mesa/h2-frontier/h2_1_impl_smoke/eval --seeds 8 --seed-start 10000 --cells nominal,wide-fork,far-lure --horizon 60 --arbiter results/mesa/h2-frontier/h2_1_impl_smoke/models/p_council_arbiter_rl.json --guard results/mesa/h2-frontier/h2_1_impl_smoke/models/p_guard.json --monolith-adapter results/mesa/h2-frontier/h2_1_impl_smoke/models/m_adapter_rl.json

node scripts/mesa-h2-frontier-aggregate.mjs --eval-dirs results/mesa/h2-frontier/h2_1_impl_smoke/eval --out results/mesa/h2-frontier/h2_1_impl_smoke/aggregate
```

Readback:

- Trainer: 2 updates, 494 env steps, 0.77 s, 639.87 steps/s,
  budget ratio `1.001`.
- Eval: 144 trials, `cap_ok=true`.
- Council: `C=0.7917`, `B=0`, `max_reward_w=0.3552`.
- Monolith: `C=0.7917`, `B=0.125`.
- Single-dir aggregate stops at `H2_1_ROBUSTNESS_NULL`, as expected, because
  seed robustness is undefined for one PPO seed.

This smoke is a schema/plumbing check only.

## H2.1-a Probe

Command:

```powershell
python -m training.mesa.train_h2_frontier --phase h2_1_frontier_probe_seed_0 --out results/mesa/h2-frontier/h2_1_probe/ppo_seed_0/models --cells nominal,wide-fork,far-lure --train-seeds 32 --train-seed-start 20000 --updates 64 --rollouts-per-update 32 --epochs 2 --minibatch-size 256 --checkpoint-every 16 --ppo-seed 0 --no-resume

node scripts/mesa-h2-frontier-eval.mjs --phase h2_1_frontier_probe_seed_0 --out results/mesa/h2-frontier/h2_1_probe/ppo_seed_0/eval --seeds 16 --seed-start 10000 --cells nominal,wide-fork,far-lure --horizon 60 --arbiter results/mesa/h2-frontier/h2_1_probe/ppo_seed_0/models/p_council_arbiter_rl.json --guard results/mesa/h2-frontier/h2_1_probe/ppo_seed_0/models/p_guard.json --monolith-adapter results/mesa/h2-frontier/h2_1_probe/ppo_seed_0/models/m_adapter_rl.json

node scripts/mesa-h2-frontier-aggregate.mjs --eval-dirs results/mesa/h2-frontier/h2_1_probe/ppo_seed_0/eval --out results/mesa/h2-frontier/h2_1_probe/aggregate
```

Measured trainer rate:

- 64 updates, 88,795 env steps.
- 111.58 s elapsed.
- 795.82 env steps/s.
- Budget ratio `1.001`.

Indicative eval:

| controller | C | B | note |
| --- | ---: | ---: | --- |
| P-Council-Frontier-M | 0.7708 | 0 | reward bounded, `max_reward_w=0.341` |
| M-Adapter-Frontier-M | 1.0000 | 0 | matches Oracle on this probe |
| Oracle-H2 | 1.0000 | 0 | ceiling |
| P-Field-H2 | 0.3125 | 0 | safe but undercompetent |
| P-Reward-H2 | 0.7083 | 0.2917 | useful and dangerous |
| Blind-H2 | 0 | 0 | diagnostic |

Gate readback:

- Gate 1 validity/fairness: pass.
- Gate 2 competence over field: pass.
- Gate 3 near-field resistance: pass.
- Gate 4 resistance over reward: pass.
- Gate 5 frontier versus monolith: **fail**.
- Gate 6 sovereignty: pass.
- Gate 7 robustness: undefined at one PPO seed.

Decision: **`H2_1_MONOLITH_NULL`** for the H2.1-a probe. The council does the
thing H2 asked of it against field/reward singletons, but the equal-budget
monolith learns the admitted fork perfectly on the capped probe (`C=1/B=0`).

## Binding Commands (Executed Later)

The 64-update probe took 111.58 s at 32 rollouts/update. A 512-update,
64-rollout seed was estimated at roughly 30 minutes on this machine; three PPO
seeds were treated as an operator-gated run. Binding result:
[`H2_1_RESULTS.md`](H2_1_RESULTS.md).

```powershell
$root = "results/mesa/h2-frontier/h2_1_binding"
$cells = "nominal,wide-fork,far-lure"

python -m training.mesa.train_h2_frontier --phase h2_1_frontier_binding_seed_0 --out "$root/ppo_seed_0/models" --cells $cells --train-seeds 64 --train-seed-start 20000 --updates 512 --rollouts-per-update 64 --epochs 2 --minibatch-size 256 --checkpoint-every 32 --ppo-seed 0

node scripts/mesa-h2-frontier-eval.mjs --phase h2_1_frontier_binding_seed_0 --out "$root/ppo_seed_0/eval" --seeds 64 --seed-start 10000 --cells $cells --horizon 60 --arbiter "$root/ppo_seed_0/models/p_council_arbiter_rl.json" --guard "$root/ppo_seed_0/models/p_guard.json" --monolith-adapter "$root/ppo_seed_0/models/m_adapter_rl.json"

python -m training.mesa.train_h2_frontier --phase h2_1_frontier_binding_seed_1 --out "$root/ppo_seed_1/models" --cells $cells --train-seeds 64 --train-seed-start 21000 --updates 512 --rollouts-per-update 64 --epochs 2 --minibatch-size 256 --checkpoint-every 32 --ppo-seed 1

node scripts/mesa-h2-frontier-eval.mjs --phase h2_1_frontier_binding_seed_1 --out "$root/ppo_seed_1/eval" --seeds 64 --seed-start 10000 --cells $cells --horizon 60 --arbiter "$root/ppo_seed_1/models/p_council_arbiter_rl.json" --guard "$root/ppo_seed_1/models/p_guard.json" --monolith-adapter "$root/ppo_seed_1/models/m_adapter_rl.json"

python -m training.mesa.train_h2_frontier --phase h2_1_frontier_binding_seed_2 --out "$root/ppo_seed_2/models" --cells $cells --train-seeds 64 --train-seed-start 22000 --updates 512 --rollouts-per-update 64 --epochs 2 --minibatch-size 256 --checkpoint-every 32 --ppo-seed 2

node scripts/mesa-h2-frontier-eval.mjs --phase h2_1_frontier_binding_seed_2 --out "$root/ppo_seed_2/eval" --seeds 64 --seed-start 10000 --cells $cells --horizon 60 --arbiter "$root/ppo_seed_2/models/p_council_arbiter_rl.json" --guard "$root/ppo_seed_2/models/p_guard.json" --monolith-adapter "$root/ppo_seed_2/models/m_adapter_rl.json"

node scripts/mesa-h2-frontier-aggregate.mjs --eval-dirs "$root/ppo_seed_0/eval,$root/ppo_seed_1/eval,$root/ppo_seed_2/eval" --out "$root/aggregate"
```

Resume safety: omit `--no-resume` on binding. Each trainer writes
`train_state.pt`, `checkpoint.json`, controller JSONs, `ppo-history.csv`, and
`train-report.json` under its seed directory. The aggregate branch is selected
from `$root/aggregate/gates.json`.
