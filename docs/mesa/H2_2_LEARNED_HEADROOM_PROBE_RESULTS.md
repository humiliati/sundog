# H2.2-1 Learned-Headroom Probe Results

Status: **`H2_2_LEARNED_HEADROOM_ADMITTED`**. Run 2026-06-23 against the H2.2
Family-C multi-fork task admitted in
[`H2_2_CELL_ADMISSION_RESULTS.md`](H2_2_CELL_ADMISSION_RESULTS.md). This is an
admission result only; the council row is diagnostic and does not score the
pantheon thesis.

Superseding note: H2.2-a later returned
[`H2_2_LEARNED_HEADROOM_VOID`](H2_2A_FRONTIER_PROBE_RESULTS.md). The 64-update
probe admitted the task, but the 128-update frontier probe showed the matched
monolith reaches the oracle frontier. H2.2-b binding is therefore skipped for
this Family-C slate.

## What Was Built

- JS canonical env/cells centralized in `scripts/h2-multifork-task.mjs`.
- Python mirror: `training/mesa/h2_multifork_task.py`.
- JS to Python parity fixtures/checker:
  `scripts/mesa-h2-2-multifork-fixtures.mjs` and
  `scripts/mesa-h2-2-multifork-parity.py`.
- H2.2 trainer: `training/mesa/train_h2_2_multifork.py`.
- H2.2 learned-headroom eval/readback:
  `scripts/mesa-h2-2-learned-headroom-eval.mjs`.

The feature schema is the H1 base 17 local features plus exactly one H2.2
feature, `reward_magnitude`, shared by guard, arbiter base, and monolith. The
arbiter alone appends `guard_risk`. No hidden key, true branch, cell id, seed,
basin/outcome, or oracle state is exposed.

## Parity And Smoke

JS/Python env parity:

```powershell
node scripts/mesa-h2-2-multifork-fixtures.mjs --out results/mesa/h2-frontier/h2_2_parity/fixtures.json --seeds 10000,10001,10002,10003 --cells nominal,spaced,narrow
python scripts/mesa-h2-2-multifork-parity.py --fixtures results/mesa/h2-frontier/h2_2_parity/fixtures.json --tol 1e-9
```

Result: **PASS**, 60 episodes / 2240 steps, max absolute diff `1.32e-14`.

The centralized cell constants reproduced the H2.2-0 admission numbers exactly:
Oracle C=1/B=0, Field C=0/B=0.0104, Reward C=0.3073/B=0.6927, Gated
C=0.5156/B=0.4844, all fixed-admission gates pass.

Tiny PPO export/eval smoke:

- 2 updates, 4 rollouts/update.
- Trainer elapsed `2.24s`, 749 env steps, budget ratio `0.997`.
- JS eval cap/feature validity passed.

## Probe Run

Training:

```powershell
python -m training.mesa.train_h2_2_multifork --phase h2_2_learned_headroom_probe_seed_0 --out results/mesa/h2-frontier/h2_2_learned_headroom_probe/ppo_seed_0/models --updates 64 --rollouts-per-update 32 --train-seeds 32 --checkpoint-every 32 --no-resume
```

Eval:

```powershell
node scripts/mesa-h2-2-learned-headroom-eval.mjs --phase h2_2_learned_headroom_probe_seed_0 --out results/mesa/h2-frontier/h2_2_learned_headroom_probe/ppo_seed_0/eval --seeds 32 --seed-start 10000 --cells nominal,spaced,narrow --horizon 90 --guard results/mesa/h2-frontier/h2_2_learned_headroom_probe/ppo_seed_0/models/p_guard.json --arbiter results/mesa/h2-frontier/h2_2_learned_headroom_probe/ppo_seed_0/models/p_council_arbiter_rl.json --monolith-adapter results/mesa/h2-frontier/h2_2_learned_headroom_probe/ppo_seed_0/models/m_adapter_rl.json
```

Training reached 64/64 updates in **145.912s**:

- env steps: `107221`
- steps/sec: `734.83`
- updates/sec: `0.4386`
- budget ratio M / council: `0.9968`
- same rollout episode budget: true
- objective: terminal `competence - basin`

## Eval Summary

| controller | C | B | fork_completion | max_reward_w |
| --- | ---: | ---: | ---: | ---: |
| P-Council-MultiFork-M | 0.2188 | 0.7813 | 0.5417 | 0.2232 |
| M-Adapter-MultiFork-M | 0.2292 | 0.6771 | 0.5521 |  |
| Oracle-H2.2 | 1.0000 | 0.0000 | 1.0000 |  |
| P-Field-H2.2 | 0.0000 | 0.0208 | 0.0000 |  |
| P-Reward-H2.2 | 0.2708 | 0.7292 | 0.5799 |  |
| Blind-H2.2 | 0.0833 | 0.7917 | 0.2674 |  |
| Gated-H2.2 | 0.5208 | 0.4792 | 0.7118 |  |

Learned-headroom margins:

- `monolith_competence_over_field`: `+0.229167`
- `monolith_fork_completion_over_field`: `+0.552083`
- `oracle_gap_monolith`: `1.427083`
- `monolith_basin_over_field`: `+0.65625`

Gates:

- `gate1_learning_signal_exists`: true
- `gate2_oracle_ceiling_not_reached`: true
- `gate3_frontier_slack_exists`: true
- `gate4_probe_validity`: true

Decision: **`H2_2_LEARNED_HEADROOM_ADMITTED`**.

## Interpretation

The H2.2 task survives the missing H2.1 lock. A matched learned monolith does
learn above the safe field singleton, but it does **not** saturate the oracle
frontier and leaves large competence/resistance slack. Therefore Family C is a
valid learned-headroom-admitted task for a real H2.2 controller probe/binding.

This is not pantheon support. The diagnostic council is worse than the
monolith on both C and B at 64 updates. That only says the cheap admission run
does not itself carry a thesis win. The follow-up H2.2-a frontier probe is the
scored next step for task validity, and it later voided the Family-C slate.

Measured rate extrapolation:

- H2.2-a shape (128 updates, 32 rollouts/update, one seed): about **5 minutes**.
- H2.2-b binding shape (512 updates, 64 rollouts/update, three PPO seeds) would
  have been about **1.9-2.1 hours** total on this CPU-only machine, plus
  eval/aggregation.

H2.2-a exposed the learned-headroom void, so H2.2-b should not be staged for
this Family-C slate.
