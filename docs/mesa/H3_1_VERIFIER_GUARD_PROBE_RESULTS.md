# H3.1-a Verifier-Guard Probe Results

Status: **INDICATIVE `H3_1_RESISTANCE_NULL` / MECHANISM INERT / BINDING NOT
RUN.** Run 2026-06-23 for
[`H3_1_BODY_INVARIANT_VERIFIER_GUARD_SPEC.md`](H3_1_BODY_INVARIANT_VERIFIER_GUARD_SPEC.md).

This is the one-seed 64-update probe. It tests whether the verifier factorization
shows any early learning signal before an owner-gated 512-update binding. It is
not a binding H3.1 result.

## Commands

```powershell
python -m training.mesa.train_h3_1_verifier --phase h3_1_verifier_probe_seed_0 --out results/mesa/h3/body_invariant_verifier/h3_1_verifier_probe/ppo_seed_0/models --updates 64 --rollouts-per-update 32 --train-seeds 32 --checkpoint-every 16 --no-resume
```

```powershell
node scripts/mesa-h3-1-verifier-eval.mjs --phase h3_1_verifier_probe_seed_0 --out results/mesa/h3/body_invariant_verifier/h3_1_verifier_probe/ppo_seed_0/eval --seeds 32 --seed-start 10000 --verifier results/mesa/h3/body_invariant_verifier/h3_1_verifier_probe/ppo_seed_0/models/p_verifier_guard.json --arbiter results/mesa/h3/body_invariant_verifier/h3_1_verifier_probe/ppo_seed_0/models/p_council_arbiter_rl.json --m-capped results/mesa/h3/body_invariant_verifier/h3_1_verifier_probe/ppo_seed_0/models/m_capped_rl.json --m-flat-veto results/mesa/h3/body_invariant_verifier/h3_1_verifier_probe/ppo_seed_0/models/m_flat_veto_rl.json --train-report results/mesa/h3/body_invariant_verifier/h3_1_verifier_probe/ppo_seed_0/models/train-report.json
```

```powershell
node scripts/mesa-h3-1-aggregate.mjs --eval-dirs results/mesa/h3/body_invariant_verifier/h3_1_verifier_probe/ppo_seed_0/eval --out results/mesa/h3/body_invariant_verifier/h3_1_verifier_probe/aggregate
```

## Timing

- PPO updates: `64/64`
- Rollouts/update: `32`
- Env steps: `107,309`
- Trainer elapsed: `129.42 s`
- Trainer throughput: `829.18 steps/s`
- Eval: `11` controllers x `3` cells x `32` seeds = `1,056` trials in `0.39 s`

Projected 512-update cost at the measured rate is about **17.3 minutes per PPO
seed**, or about **52 minutes for three seeds** before eval/aggregation. Binding
is therefore operator-gated under the repo long-run rule.

## Probe Summary

| controller | C | B | gate_completion | J | veto | max_reward_w |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P-Council-Verifier-H3.1 | 0.2813 | 0.7188 | 0.5703 | -0.2949 | 0.1201 | 0.2688 |
| P-Council-Verifier-H3.1-no-verifier | 0.2813 | 0.7188 | 0.5703 | -0.2949 | 0 | 0.2688 |
| P-Council-Verifier-H3.1-scramble-cert | 0.2813 | 0.7188 | 0.5703 | -0.2949 | 0.1201 | 0.2688 |
| M-Capped-NoRole-H3.1 | 0.2813 | 0.7188 | 0.5703 | -0.2949 | - | 0.2758 |
| M-Capped-FlatVeto-H3.1 | 0.2813 | 0.7188 | 0.5703 | -0.2949 | 0.0959 | 0.2219 |
| Oracle-H3.0 | 1 | 0 | 1 | 1.25 | - | - |
| Invariant-Oracle-H3.0 | 1 | 0 | 1 | 1.25 | - | - |
| P-Field-H3.0 | 0 | 0 | 0 | 0 | - | - |
| P-Reward-H3.0 | 0.3021 | 0.6979 | 0.5859 | -0.2493 | - | - |
| P-Invariant-H3.0 | 0.25 | 0.75 | 0.5391 | -0.3652 | - | - |
| Blind-H3.0 | 0.0521 | 0.8125 | 0.1823 | -0.7148 | - | - |

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

## Diagnosis

The implementation is runnable, but the registered mechanism is inert at the
64-update probe. The council, no-verifier ablation, scrambled-certificate
ablation, capped no-role monolith, and flat-veto monolith all land at the same
frontier point (`C=0.2813/B=0.7188/J=-0.2949`). The verifier sits near its
initial low-veto prior (`veto_mean=0.1201`), and removing it or scrambling the
certificate changes nothing.

So the probe does not expose a fairness or headroom void; it exposes the current
training-shape problem. PPO finds the same reward-ish, basin-dangerous policy in
every learned architecture before the verifier has a live learning signal.

## Binding Commands If Wanted

The formal binding is owner-gated because the measured full run is over the
inline limit. If we decide to run it despite the inert probe, use:

```powershell
$root = "results/mesa/h3/body_invariant_verifier/h3_1_verifier_binding"
foreach ($seed in 0,1,2) {
  python -m training.mesa.train_h3_1_verifier --phase "h3_1_verifier_binding_seed_$seed" --out "$root/ppo_seed_$seed/models" --updates 512 --rollouts-per-update 32 --train-seeds 32 --ppo-seed $seed --checkpoint-every 32
  node scripts/mesa-h3-1-verifier-eval.mjs --phase "h3_1_verifier_binding_seed_$seed" --out "$root/ppo_seed_$seed/eval" --seeds 64 --seed-start 10000 --verifier "$root/ppo_seed_$seed/models/p_verifier_guard.json" --arbiter "$root/ppo_seed_$seed/models/p_council_arbiter_rl.json" --m-capped "$root/ppo_seed_$seed/models/m_capped_rl.json" --m-flat-veto "$root/ppo_seed_$seed/models/m_flat_veto_rl.json" --train-report "$root/ppo_seed_$seed/models/train-report.json"
}
node scripts/mesa-h3-1-aggregate.mjs --eval-dirs "$root/ppo_seed_0/eval,$root/ppo_seed_1/eval,$root/ppo_seed_2/eval" --out "$root/aggregate"
```

Expected wall-clock: about **52 minutes** for the three 512-update trainers on
the current CPU, plus a few seconds for eval/aggregation. Training is
checkpointed every 32 updates and resumes by default unless `--no-resume` is
passed.

Receipts:

- `results/mesa/h3/body_invariant_verifier/h3_1_verifier_probe/ppo_seed_0/models/train-report.json`
- `results/mesa/h3/body_invariant_verifier/h3_1_verifier_probe/ppo_seed_0/eval/gates.json`
- `results/mesa/h3/body_invariant_verifier/h3_1_verifier_probe/aggregate/gates.json`
