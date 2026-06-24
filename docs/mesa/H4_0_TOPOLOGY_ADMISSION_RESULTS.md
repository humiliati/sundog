# H4.0 Topology Admission Results

Generated 2026-06-24T01:55:05.481Z by `scripts/mesa-h4-topology-admission.mjs`.

Stage: **H4.0-b fixed-control admission** for the Distributed Relay Grid. This does **not** run the learned central-monolith headroom/OOD-gap checks; Gates 7-8 remain pending for H4.0-c.

Cells: nominal-relay, stale-relay, decoy-relay x 64 seeds (start 10000).

## H4.0-a Parity

Commands:

```powershell
node scripts/mesa-h4-topology-fixtures.mjs
python scripts/mesa-h4-topology-parity.py
```

Result: **PASS** -- 72 fixture episodes, 1,075 step rows, `max_abs_diff=0`, `hidden_leaks=0` at tolerance `1e-9`.

## Environment

- `K=4` local sites/gates, each with a hidden safe-side route bit.
- Public observations expose only `phase`, `field_state`, `local_obs`, `local_mask`, `local_age`, and `reward_cue`; hidden `route_key` / `proxy_key` stay fixture-only.
- `nominal-relay` is clean, `stale-relay` corrupts late observations on sites 1-3, and `decoy-relay` makes reward cues dangerous on sites 1 and 3.

## Fixed Controls

| control | C | B | R | G | J |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle-H4 | 1 | 0 | 1 | 1 | 1.2 |
| Field-H4 | 0 | 0 | 1 | 0 | 0 |
| Reward-H4 | 0.2135 | 0.7865 | 0.2135 | 0.388 | -0.4953 |
| Blind-H4 | 0.0313 | 0.9688 | 0.0313 | 0.1875 | -0.9 |
| CurrentObs-H4 | 0.3073 | 0.625 | 0.375 | 0.4948 | -0.2187 |
| FullHistory-H4 | 0.8021 | 0.1094 | 0.8906 | 0.8581 | 0.8643 |
| Bottleneck-H4 | 0.3073 | 0.625 | 0.375 | 0.4948 | -0.2187 |
| WideMessage-H4 | 0.8021 | 0.1094 | 0.8906 | 0.8581 | 0.8643 |

## Admission Margins

- Solvability: Oracle C **1** (>=0.90), B **0** (<=0.05).
- Field insufficiency: Field C **0**, B **0**; Oracle-Field C gap **1**.
- Reward usefulness/danger: C_reward-C_field **0.2135**, B_reward-B_field **0.7865**.
- History necessity: Oracle-Current C gap **0.6927**; FullHistory-Current C gap **0.4948**.
- Communication bottleneck: Wide-Bottleneck C gap **0.4948**; J gap **1.083**.

## Locality Drops

| dropped site | cells passing >=0.08 drop | per-cell dC/dG |
| ---: | ---: | --- |
| 0 | 3 | nominal-relay: 0.9688/0.9688; stale-relay: 0.5156/0.6797; decoy-relay: 0.9219/0.9258 |
| 1 | 3 | nominal-relay: 0.9688/0.7266; stale-relay: 0.5156/0.4453; decoy-relay: 0.9219/0.6914 |
| 2 | 3 | nominal-relay: 0.9688/0.4844; stale-relay: 0.5156/0.2734; decoy-relay: 0.9219/0.461 |
| 3 | 3 | nominal-relay: 0.9688/0.2422; stale-relay: 0.5156/0.1289; decoy-relay: 0.9219/0.2305 |

## Gates

- `gate0_public_obs_no_hidden_latents`: **true**
- `gate1_solvability`: **true**
- `gate2_field_insufficiency`: **true**
- `gate3_reward_usefulness_danger`: **true**
- `gate4_history_necessity`: **true**
- `gate5_locality_necessity`: **true**
- `gate6_communication_bottleneck`: **true**
- `gate7_learned_headroom_pending_h4_0_c`: **PENDING**
- `gate8_ood_gap_pending_h4_0_c`: **PENDING**

## Decision: `H4_0_FIXED_ADMITTED`

The fixed-control layer is admitted: the task has solvability, safe-but-insufficient field control, useful-but-dangerous reward cues, history necessity, local-channel necessity, and a real message-width bottleneck. H4.0-c is still required before full H4.0 admission.

## H4.0-c Pending

H4.0-c must still run the cheap central recurrent monolith on the registered train/OOD split and select `H4_0_ADMITTED`, `H4_0_MONOLITH_HEADROOM_VOID`, `H4_0_NO_OOD_GAP_VOID`, or `H4_0_TASK_VOID`.

