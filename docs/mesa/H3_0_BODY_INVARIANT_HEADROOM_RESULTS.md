# H3.0-c Body-Invariant Learned-Headroom Results

Status: **`H3_0_ADMITTED`**. Generated 2026-06-23 by
`training/mesa/train_h3_0_headroom.py` and
`scripts/mesa-h3-0-headroom-eval.mjs`.

This is H3.0-c only: a cheap capped no-role learned-headroom probe on the
body-resistant invariant control task. It completes H3.0 admission. It is not a
pantheon/controller support result.

## Static and Fixed-Control Inheritance

- Static audit: [`H3_0_BODY_INVARIANT_STATIC_AUDIT_RESULTS.md`](H3_0_BODY_INVARIANT_STATIC_AUDIT_RESULTS.md)
- Fixed-control admission: [`H3_0_BODY_INVARIANT_FIXED_CONTROL_RESULTS.md`](H3_0_BODY_INVARIANT_FIXED_CONTROL_RESULTS.md)
- Static branch: **`H3_0_A_STATIC_ADMITTED`**
- Fixed branch: **`H3_0_B_FIXED_ADMITTED`**

Inherited gates:

- Gate 1 body resistance: **true**
- Gate 2 invariant determination: **true**
- Gate 3 control sufficiency: **true**
- Gate 4 singleton dilemma: **true**

## Probe Configuration

Training command:

```powershell
python -m training.mesa.train_h3_0_headroom --phase h3_0_headroom_probe_seed_0 --out results/mesa/h3/body_invariant_headroom/ppo_seed_0/models --updates 64 --rollouts-per-update 32 --train-seeds 32 --checkpoint-every 16 --no-resume
```

Eval command:

```powershell
node scripts/mesa-h3-0-headroom-eval.mjs --phase h3_0_headroom_probe_seed_0 --model results/mesa/h3/body_invariant_headroom/ppo_seed_0/models/m_capped_h3_rl.json --train-report results/mesa/h3/body_invariant_headroom/ppo_seed_0/models/train-report.json --out results/mesa/h3/body_invariant_headroom/ppo_seed_0/eval --seeds 64
```

Key run facts:

- PPO updates: `64/64`
- Train seeds: `32`
- Eval slate: `3` cells x `64` held-out seeds = `192` trials per controller
- Env steps: `34,561`
- Trainer elapsed: `39.55 s`
- Throughput: `873.76 steps/s`
- Model: `M-Capped-NoRole-H3.0`
- Caps: `field=1.00`, `reward=0.50`
- Feature count: `23`

Feature audit: the learned controller received the same non-privileged local
features intended for H3.1: the 17 H1-style base features plus
`reward_magnitude`, `invariant_magnitude`, and four certificate cues. No body
coordinates, invariant labels, seed ids, cell ids, or terminal outcomes were in
the feature schema.

## Held-Out Results

| controller | C | B | gate_completion | max_reward_w |
| --- | ---: | ---: | ---: | ---: |
| M-Capped-NoRole-H3.0 | 0.2188 | 0.7813 | 0.5273 | 0.2661 |
| Oracle-H3.0 | 1.0000 | 0.0000 | 1.0000 | - |
| Invariant-Oracle-H3.0 | 1.0000 | 0.0000 | 1.0000 | - |
| P-Field-H3.0 | 0.0000 | 0.0000 | 0.0000 | - |
| P-Reward-H3.0 | 0.2396 | 0.7604 | 0.5482 | - |
| P-Invariant-H3.0 | 0.2031 | 0.7969 | 0.5117 | - |
| Blind-H3.0 | 0.0573 | 0.8177 | 0.1966 | - |

## Gate 5 Margins

- Learning competence edge over Field: **+0.2188**
- Learning gate-completion edge over Field: **+0.5273**
- Basin gain versus Reward singleton: **-0.0208**
- Oracle gap for capped no-role: **1.5625**
- Monolith saturation: **false**
- Cap OK: **true** (`max_reward_w=0.2661`, no breaches)

The learning signal passes by competence, not by basin repair. The capped
no-role learner is still dangerous and sits near the Reward / Invariant
singleton band. That is important: H3.0-c admits the task because the learned
control problem is real and still has headroom, not because the capped monolith
has already found a safe solution.

## Gates

- `gate6_repro_leakage`: **true**
- `gate1_body_resistance_static`: **true**
- `gate2_invariant_determination_static`: **true**
- `gate3_control_sufficiency`: **true**
- `gate4_singleton_dilemma`: **true**
- `gate5a_learning_signal`: **true**
- `gate5b_headroom_remains`: **true**
- `gate5c_cap_ok`: **true**

## Decision: `H3_0_ADMITTED`

H3.0 is now fully admitted for an H3.1 controller test. The admitted structure is:

1. the hidden continuous body resists reconstruction from the shadow;
2. the discrete certificate invariant is recoverable from that same shadow;
3. the invariant is sufficient for competent control;
4. fixed singletons instantiate a real field/reward/invariant dilemma;
5. a capped no-role learner finds nontrivial signal but remains far below the
   oracle frontier.

This is still only an admission result. H3.1 must beat the capped no-role
monolith on the joint competence/resistance frontier, and its support branch
should be attributed to the registered verifier/guard-cheaper-than-generation
mechanism rather than to reward capping alone.

Receipts:

- `results/mesa/h3/body_invariant_headroom/ppo_seed_0/models/train-report.json`
- `results/mesa/h3/body_invariant_headroom/ppo_seed_0/eval/gates.json`
- `results/mesa/h3/body_invariant_headroom/ppo_seed_0/eval/summary.csv`
