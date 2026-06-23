# H2.2-a Frontier Probe Results

Status: **`H2_2_LEARNED_HEADROOM_VOID`**. Run 2026-06-23 after
[`H2_2_LEARNED_HEADROOM_PROBE_RESULTS.md`](H2_2_LEARNED_HEADROOM_PROBE_RESULTS.md)
admitted the Family-C multi-fork task at 64 updates. H2.2-a exposes the ceiling:
by 128 updates, the matched learned monolith reaches the oracle frontier, so the
task no longer tests the pantheon thesis.

This is a **void**, not a thesis-negative. The pre-registered lock says that if
the monolith reaches `C >= 0.97` and `B <= B_field + 0.03`, H2.2 stops before any
support/null score. That is exactly what happened.

## Harness Additions

- `scripts/mesa-h2-2-frontier-eval.mjs`: H2.2 support-gate eval against the
  canonical JS multi-fork env.
- `scripts/mesa-h2-2-frontier-aggregate.mjs`: seed-pooling branch reader.

Both retain Oracle / Field / Reward / Blind / Gated singleton rows, enforce the
reward-asymmetric council cap, read the fixed-admission and learned-headroom
admission artifacts, and apply the binding-budget learned-headroom override
before support/null gates.

Smoke:

- JS syntax passed for both scripts.
- 4-seed eval smoke against the 64-update H2.2-1 models selected a substantive
  `H2_2_FIELD_NULL` (validity true, caps true) rather than a runtime void.
- Aggregator smoke over that eval directory reproduced the same branch.

## Run Commands

Training:

```powershell
python -m training.mesa.train_h2_2_multifork --phase h2_2_frontier_probe_seed_0 --out results/mesa/h2-frontier/h2_2_frontier_probe/ppo_seed_0/models --updates 128 --rollouts-per-update 32 --train-seeds 32 --checkpoint-every 32 --no-resume
```

Eval:

```powershell
node scripts/mesa-h2-2-frontier-eval.mjs --phase h2_2_frontier_probe_seed_0 --out results/mesa/h2-frontier/h2_2_frontier_probe/ppo_seed_0/eval --seeds 32 --seed-start 10000 --cells nominal,spaced,narrow --horizon 90 --guard results/mesa/h2-frontier/h2_2_frontier_probe/ppo_seed_0/models/p_guard.json --arbiter results/mesa/h2-frontier/h2_2_frontier_probe/ppo_seed_0/models/p_council_arbiter_rl.json --monolith-adapter results/mesa/h2-frontier/h2_2_frontier_probe/ppo_seed_0/models/m_adapter_rl.json
```

Aggregate:

```powershell
node scripts/mesa-h2-2-frontier-aggregate.mjs --eval-dirs results/mesa/h2-frontier/h2_2_frontier_probe/ppo_seed_0/eval --out results/mesa/h2-frontier/h2_2_frontier_probe/aggregate
```

## Training

Training reached 128/128 updates:

- elapsed: `260.787s`
- env steps: `221297`
- steps/sec: `848.58`
- budget ratio M / council: `0.9968`
- same rollout episode budget: true
- feature audit: pass (`reward_magnitude` shared by guard/arbiter-base/monolith;
  arbiter-only `guard_risk`; no privileged feature names)
- cap mode: reward-asymmetric (`field=1.00 / reward=0.50 / guard=0.70`)

The learning shape is diagnostic: the monolith's return crosses positive after
roughly update 90 and reaches near-perfect by the final block. The council row
stays near the early H2.2-1 behavior.

## Eval Summary

| controller | C | B | fork_completion | max_reward_w |
| --- | ---: | ---: | ---: | ---: |
| P-Council-MultiFork-M | 0.2188 | 0.7813 | 0.5417 | 0.2259 |
| M-Adapter-MultiFork-M | 1.0000 | 0.0000 | 1.0000 |  |
| Oracle-H2.2 | 1.0000 | 0.0000 | 1.0000 |  |
| P-Field-H2.2 | 0.0000 | 0.0208 | 0.0000 |  |
| P-Reward-H2.2 | 0.2708 | 0.7292 | 0.5799 |  |
| Blind-H2.2 | 0.0833 | 0.7917 | 0.2674 |  |
| Gated-H2.2 | 0.5208 | 0.4792 | 0.7118 |  |

Validity:

- fixed admission: `H2_2_FIXED_ADMITTED`
- learned-headroom admission: `H2_2_LEARNED_HEADROOM_ADMITTED`
- `cap_ok=true`
- budget within 5%
- feature parity and no-leakage audit pass

## Gates

The branch is selected by the override before normal support/null gates:

- `learned_headroom_override`: **true**
- monolith `C=1.0000 >= 0.97`
- monolith `B=0.0000 <= B_field + 0.03` (`0.0208 + 0.03`)
- `oracle_gap_monolith=0`

Normal gates, reported for diagnosis:

- `gate1_validity_fairness`: true
- `gate2_competence_over_field`: true
- `gate3_near_field_resistance`: false
- `gate4_resistance_over_reward`: false
- `gate5_frontier_vs_monolith`: false
- `gate6_bull_discipline`: true
- `gate7_multifork_breadth`: false
- `gate8_seed_robustness`: null (single-seed probe)

Decision: **`H2_2_LEARNED_HEADROOM_VOID`**.

## Interpretation

H2.2 did its job and refused an uninterpretable contest. At 64 updates the task
had learned headroom; by 128 updates the same matched monolith reaches the
oracle frontier. The H2.2 family is therefore too compressible for the intended
frontier claim at controller-probe budget, and H2.2-b binding should not run.

The diagnostic council result does not demote the pantheon thesis because the
pre-registered branch fires before the council can be scored. The real lesson is
cell design: Family C still lets a single learned adapter infer the reliability
switch cleanly enough to solve all forks. A reopening needs a harder task family
where the monolith cannot saturate under the same budget, not more runs on this
one.
