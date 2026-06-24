# H4.0 Topology Admission Results

Generated 2026-06-24T01:55:05.481Z by `scripts/mesa-h4-topology-admission.mjs`; updated 2026-06-24 with H4.0-c learned-headroom/OOD-gap results from `training/mesa/train_h4_topology.py`.

Stage: **H4.0-a/b/c admission** for the Distributed Relay Grid. H4.0-a/b fixed admission passed; H4.0-c ran the cheap learned central-monolith headroom/OOD-gap gate and selected `H4_0_NO_OOD_GAP_VOID`.

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
- `gate7_learned_headroom`: **true**
- `gate7_not_oracle_saturated`: **true**
- `gate8_oracle_ood_solvable`: **true**
- `gate8_ood_generalization_gap`: **false**

## H4.0-c Learned Headroom / OOD Gap

Command:

```powershell
python -m training.mesa.train_h4_topology --phase h4_0c_learned_headroom --out results/mesa/h4-topology/h4_0c_learned_headroom --updates 64 --rollouts-per-update 32 --eval-seeds 32 --checkpoint-every 16
```

Runtime: 64 updates, 17,210 env steps, 82.46 s, 208.71 steps/s on CPU. Artifacts:

- `results/mesa/h4-topology/h4_0c_learned_headroom/train_report.json`
- `results/mesa/h4-topology/h4_0c_learned_headroom/train_log.csv`
- `results/mesa/h4-topology/h4_0c_learned_headroom/eval_trials.csv`
- `results/mesa/h4-topology/h4_0c_learned_headroom/models/m_central_rnn_h4.pt`

Train distribution: `train-nominal-relay`, `train-mild-stale-relay`, `train-mild-decoy-relay`. Held-out OOD: `ood-long-stale-relay`, `ood-drop-decoy-relay`, `ood-shifted-decoy-relay`.

| split | controller | C | B | G | J |
| --- | --- | ---: | ---: | ---: | ---: |
| in-distribution | M-Central-RNN-H4-cheap | 0.0625 | 0.9375 | 0.4297 | -0.7891 |
| held-out OOD | M-Central-RNN-H4-cheap | 0.0833 | 0.9167 | 0.2943 | -0.7745 |
| in-distribution | Oracle-H4 | 1 | 0 | 1 | 1.2 |
| held-out OOD | Oracle-H4 | 1 | 0 | 1 | 1.2 |
| in-distribution | Field-H4 | 0 | 0 | 0 | 0 |
| held-out OOD | Field-H4 | 0 | 0 | 0 | 0 |

Gate 7 technically passes only in the narrow admission sense: the cheap monolith finds a small amount of competence over Field (`C=0.0625` vs `0`) and does not saturate the Oracle frontier. It remains badly basin-dangerous (`B=0.9375`), so this is learned signal, not useful governance.

Gate 8 fails decisively: the registered OOD gap is `J_ID - J_OOD = -0.0146`, not `>= 0.10`. Held-out corruption was not harder for the cheap central monolith; if anything, it scored slightly less badly out of distribution.

## Decision: `H4_0_NO_OOD_GAP_VOID`

The fixed-control task is structurally interesting, but the registered held-out corruption split does not create the OOD generalization gap H4 needs. H4.1 must not proceed on this slate. Reopening requires a new pre-registered H4.0 design with a demonstrably harder held-out corruption regime, not a rescore of this run.

## H4.0-c 128-update Diagnostic (2026-06-24)

Re-ran the cheap central-monolith headroom probe at **128 updates** (2x budget) to disambiguate "monolith robust" (OOD premise dead) from "monolith at floor" (slate mis-tuned / too hard). Command: `python -m training.mesa.train_h4_topology --phase h4_0c_headroom_128 --updates 128 --rollouts-per-update 32 --eval-seeds 32` (179.84 s, 42,380 env steps).

Verdict: the monolith stays **floored** — ID `C=0.0625 / B=0.9375 / J=-0.7896`, OOD `C=0.0833 / B=0.9167 / J=-0.7750`, byte-identical to the 64-update row; training `C` bounced 0.03-0.13 without climbing. `ood_gap_J=-0.0146`, gate 8 still false -> `H4_0_NO_OOD_GAP_VOID` confirmed at 2x budget.

Interpretation: doubling the budget changed nothing. The relay POMDP is **solvable** (privileged `FullHistory-H4` reaches `C=0.80`) but **not learnable by a cheap RNN at probe budget** — a learning/optimization wall, not a robustness finding. The OOD premise cannot be tested on this slate at probe budget because no learned controller (monolith or distributed) reaches the competence where an ID->OOD gap could be measured. A valid reopening needs an **easier base task** (learnable at probe budget) that still carries a held-out gap, not a rescore or more budget on this one. Artifacts: `results/mesa/h4-topology/h4_0c_headroom_128/`.
