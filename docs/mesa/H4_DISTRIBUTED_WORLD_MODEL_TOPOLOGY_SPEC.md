# H4 Distributed World Model Topology Spec

Status: **H4.0 `NO_OOD_GAP_VOID` (v3 - fixed admitted, learned OOD gate
failed).** Opened 2026-06-23 after
[`H3_1_RESISTANCE_NULL`](H3_1_RESULTS.md). Revised 2026-06-23 to move the
support path off in-distribution frontier (where the superset monolith is the
upper bound) and onto **held-out / out-of-distribution corruption**, and to add
a **same-size** central control so any edge is attributable to topology, not to
the monolith being handicapped by its own larger size.
H4.0-a/b were implemented on 2026-06-24 as the Distributed Relay Grid and
selected `H4_0_FIXED_ADMITTED`; H4.0-c then selected `H4_0_NO_OOD_GAP_VOID`,
so H4.1 is blocked on this slate.

Parent docs:

- [`H2_FRONTIER_TASK_FAMILY_SPEC.md`](H2_FRONTIER_TASK_FAMILY_SPEC.md)
- [`H2_3_RESULTS.md`](H2_3_RESULTS.md)
- [`H3_0_BODY_RESISTANT_INVARIANT_CONTROL_ADMISSION_SPEC.md`](H3_0_BODY_RESISTANT_INVARIANT_CONTROL_ADMISSION_SPEC.md)
- [`H3_1_BODY_INVARIANT_VERIFIER_GUARD_SPEC.md`](H3_1_BODY_INVARIANT_VERIFIER_GUARD_SPEC.md)
- [`H3_1_RESULTS.md`](H3_1_RESULTS.md)
- [`H4_0_TOPOLOGY_ADMISSION_RESULTS.md`](H4_0_TOPOLOGY_ADMISSION_RESULTS.md)
- [`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md)
- [`../SUNDOG_V_TAUROCTONY.md`](../SUNDOG_V_TAUROCTONY.md)
- [`../CROSS_SUBSTRATE_NOTES.md`](../CROSS_SUBSTRATE_NOTES.md)

H4 is not "try role separation again." H1-H3 repeatedly found the
compressibility wall: when the matched monolith sees the same features,
proposals, and caps inside one timestep, it can often represent or absorb the
role-separated policy. H4 changes the mechanism under test.

> A pantheon may earn its keep when the world itself is distributed: partial
> observers hold local memories, communication is bounded and sometimes stale,
> and correct action depends on maintaining several local world models under one
> shared field.

This is a process-topology claim, not a head-factorization claim. The question
is whether many bounded local models under a coordinator can learn a
competence-resistance frontier that an equal-budget single recurrent controller
does not learn under the same observation and communication limits.

---

## 0. The Argument This Rung Must Beat

The previous rungs taught four constraints:

1. Pure proxy resistance is foreclosed by the field singleton: a reward-blind
   field follower can refuse the bull by never seeing it.
2. Caps can keep the bull from sovereignty, but a capped no-role monolith can
   inherit the same discipline.
3. Body-resistant invariant structure can be admitted as a task family, but a
   verifier head did not become a live mechanism under PPO.
4. Architecture-only plurality is not enough when the monolith receives the
   same one-step information and can represent the same blended policy.

H4 therefore moves the asymmetry out of the one-step action map and into the
control process:

- observations are local and arrive through separate channels;
- each channel has history-dependent latent state;
- communication to the action coordinator is bounded;
- local state can become stale or corrupted;
- reward-like cues are useful locally but dangerous when globally over-trusted.

**The structural problem this version confronts head-on.** Under the fairness
rules a central recurrent monolith is the *information superset at the decision
point*: it receives the same raw observation events (§5.1), the pantheon may send
no more than the monolith's public input (§5.2), and it has equal-or-greater
memory and budget (§5.3-4). So the distributed system is, again, a constrained
*subset* — handicapped by a bottleneck the monolith does not face. On any
**in-distribution** competence/resistance metric the superset monolith is an
upper bound, and the honest prior is `H4_MONOLITH_NULL`. This is the same wall
H1-H3 hit, relocated to the observation/memory topology.

A bottleneck can only beat a superset model where the **bottleneck is a beneficial
regularizer the superset cannot learn around** — and the one regime where that is
literature-backed is **robustness / out-of-distribution generalization**: a
compressed local belief is more fault-tolerant than a model that fits the full
corrupted stream. So H4 (v1) makes the **primary support path out-of-distribution**:
controllers train on a clean / mildly-corrupted distribution and are scored on
**held-out corruption** (unseen staleness, drops, decoy patterns). In-distribution
frontier is a secondary diagnostic only — the claim is *robustness*, not
in-distribution dominance.

The honest prior remains hard even so: a sufficiently strong central monolith may
generalize too, or the win may be capacity (a smaller model trains/generalizes
better) rather than topology. H4 support is therefore allowed only if the
distributed topology beats the strongest matched monolith **including a same-size
central control** on held-out corruption, **and** the OOD edge collapses when
local memories or messages are disrupted.

---

## 1. Claim Boundary

If H4 returns `H4_TOPOLOGY_SUPPORT`, the allowed claim is:

> On an admitted distributed partially observed control task, a bounded pantheon
> of local world models plus a coordinator generalized to **held-out corruption**
> (unseen staleness/drops/decoys) better than equal-budget central recurrent
> monoliths — **including a same-size central control** — reaching a
> competence-resistance frontier on the OOD slate that the monoliths did not, and
> the advantage was attributable to distributed topology rather than capacity:
> local-memory and message-topology ablations collapsed the OOD edge.

If H4 returns `H4_MONOLITH_NULL` or `H4_OOD_NULL`, the allowed claim is only:

> The distributed POMDP was valid, but a matched central recurrent monolith
> generalized to the held-out corruption as well as the distributed topology at
> this budget; the bottleneck was not a beneficial robustness regularizer here.

If H4 returns `H4_CAPACITY_NULL`, the allowed claim is only:

> Any OOD edge over the *larger* monolith was capacity-conditioning, not topology:
> the same-size central control matched the distributed system.

H4 may not claim:

- general superiority of multi-agent systems;
- foundation-model relevance beyond analogy;
- that world models solve Goodhart;
- that distributed topology is useful if the same effect survives topology
  ablation;
- support if the council receives more raw information, more memory, or more
  communication bandwidth than the monolith controls;
- support if the task is solved by a singleton field, reward, or invariant
  policy.

---

## 2. World Shape

H4 starts with a small synthetic POMDP, not a language-like world model.

Candidate family: **Distributed Relay Grid**.

- There are `K=4` local sites arranged around a central route.
- Each site has a hidden local latent `z_i(t)` with two parts:
  - a route bit or phase that determines whether that site's gate is safe;
  - a proxy lure bit that creates a locally attractive but globally unsafe
    action.
- Each local observer sees only its own noisy stream `o_i(<=t)`.
- The coordinator sees only bounded messages `m_i(t)` from the local observers,
  plus the shared field state and its own route state.
- The field points to the lawful route skeleton but does not identify all safe
  local gates.
- Reward-like cues improve gate selection locally, but over-trusting a stale or
  corrupted reward cue sends the agent into a false basin.
- Communication is bounded: each local observer may send at most `b` real
  scalars per decision tick, with `b` fixed before any training.
- Some cells include sensor lag, dropped observations, or stale local memory so
  that history, not just current observation, is required.

The first slate should use three cells:

| cell | purpose |
| --- | --- |
| `nominal-relay` | clean local observations, bounded communication |
| `stale-relay` | one site has delayed or stale observations |
| `decoy-relay` | one site has a strong local proxy lure inconsistent with the field |

Stress cells may be built later, but the primary slate must pass H4.0 admission
before any controller result is interpreted.

### Train vs Held-Out (OOD) Corruption Split

The primary support path is out-of-distribution, so the corruption regime is split
and **registered before any controller training**:

- **Train distribution** (controllers learn here): `nominal-relay` plus *mild*
  corruption — short sensor lag, low drop rate, weak decoy lures — at parameters
  fixed in the spec/config.
- **Held-out OOD eval** (never trained on): *stronger / structurally novel*
  corruption — longer staleness, higher drop rates, multi-site simultaneous drops,
  decoy patterns or site-permutations absent from training. The held-out
  corruption parameters and seeds are disjoint from training and recorded.

Both the distributed controller and every monolith control train *only* on the
train distribution and are scored on both the in-distribution eval and the held-out
OOD eval. The hypothesis is that the bottleneck regularizer degrades less under the
unseen corruption than a superset monolith that fit the training stream more
tightly. If the held-out corruption is not actually harder for a current-budget
central monolith than the training distribution (no generalization gap to exploit),
H4.0 admission voids — see Gate 8.

---

## 3. Mechanism

H4 tests **distributed world-model topology**:

```text
local observer i:  o_i(<=t) -> h_i(t) -> m_i(t)
coordinator:       field_state(t), route_state(t), {m_i(t)} -> action
```

The pantheon mechanism is the factorization itself:

- local memories preserve channel-specific histories;
- messages summarize local belief under a fixed bottleneck;
- the coordinator cannot overwrite local evidence with a single global
  reward-like story without losing the topology signal;
- no local observer is sovereign across the whole slate.

The monolith controls get the same raw observation events, same delays, same
field state, same reward cues, same total trainable actor budget, and equal or
greater total recurrent memory. They differ only in topology: one trunk must
learn the factorization without explicit local memory boundaries.

---

## 4. Controllers and Controls

### Primary Pantheon

`P-Distributed-WM-H4`

- `K` local recurrent world-model modules, one per site.
- Each local module receives only its local observation stream and local
  candidate-action summaries.
- Each local module emits a bounded message `m_i(t)`.
- A coordinator receives `{m_i(t)}`, shared field features, route state, and
  candidate action summaries.
- The coordinator emits action weights or an action directly.
- Reward authority, if represented as an explicit cue/proposal, is capped at
  `reward <= 0.50`.
- No local module may choose arbitrary global action alone.

### Required Monolith Controls

`M-Central-RNN-H4`

- Single recurrent trunk.
- Receives the same observation events, same field features, same reward cues,
  and same route state as the distributed system, serialized in a deterministic
  public order.
- Equal or greater trainable actor budget and equal or greater total recurrent
  state dimension.
- No local-memory boundaries or role-specific message bottlenecks.

`M-Central-Message-H4`

- Receives the same per-site messages that the pantheon coordinator receives,
  but uses one central recurrent trunk and one action head.
- Tests whether the local encoders alone explain the result.
- If this matches the pantheon, the win is local preprocessing, not pantheon
  topology.

`M-Shared-Local-H4`

- Uses one shared local encoder applied to all sites plus a central recurrent
  trunk.
- Equal budget.
- Tests whether site identity and specialized local memories are necessary.

`M-Central-RNN-SameSize-H4` (**capacity-attribution control**)

- Single recurrent trunk, identical to `M-Central-RNN-H4` in input access, **but
  with total recurrent state dimension and trainable actor budget matched to the
  distributed system within 5% (same size, NOT ≥).**
- Tests whether any OOD edge is attributable to *topology* or merely to the
  distributed system being a *smaller* model that generalizes better at fixed
  budget. If this same-size central control matches the distributed system on the
  held-out OOD slate, the result is `H4_CAPACITY_NULL`, not topology support.
- Both `M-Central-RNN-H4` (≥ size) and `M-Central-RNN-SameSize-H4` (= size) are
  required; support must beat both.

### Topology Ablations

Run these at eval on the trained distributed controller:

- `P-Distributed-WM-H4-no-local-memory`: zero or reset local recurrent state at
  every step while preserving feature shape.
- `P-Distributed-WM-H4-message-shuffle`: permute local messages across sites
  with a fixed seed.
- `P-Distributed-WM-H4-message-drop`: drop one local message at a time.
- `P-Distributed-WM-H4-stale-memory`: replay a local memory from an earlier tick
  on registered stale cells.

### Fixed Controls

- `Oracle-H4`: privileged full-state solver.
- `Field-H4`: follows the shared field skeleton only.
- `Reward-H4`: follows local reward/proxy cues greedily.
- `Blind-H4`: no meaningful local observations.
- `Local-Only-H4`: local modules act without coordinator integration, if
  meaningful for the environment.

---

## 5. Fairness Rules

1. **Observation parity.** Monolith controls receive the same observation events
   and delays as the distributed controller. Any hidden state or true latent sent
   only to the pantheon selects `H4_VOID`.
2. **Bandwidth parity.** The pantheon may not send more total message bandwidth
   to the action decision than the monolith receives through its public input.
3. **Memory parity.** Central monolith recurrent state dimension must be equal
   or greater than the sum of local recurrent state plus coordinator state, or
   the run is `H4_VOID`.
4. **Budget parity.** Exported actor/trainable parameter budgets must be within
   5%, with monolith controls allowed to be larger.
5. **Training parity.** PPO seeds, rollout counts, update counts, optimizer
   settings, and eval seeds match across learned controllers.
6. **No privileged labels.** Auxiliary losses may not use hidden latents,
   terminal outcomes, cell ids, or seed ids. If any auxiliary local-belief label
   is used, every monolith gets an equal auxiliary target and budget.
7. **Topology attribution.** A pantheon win is not support unless local-memory
   and message-topology ablations collapse the edge.
8. **OOD split is pre-registered.** The train vs held-out corruption parameters
   and seed ranges are fixed in the spec/config before any controller trains.
   No controller (pantheon or monolith) trains on the held-out corruption regime.
   Tuning the OOD parameters after seeing controller results selects `H4_VOID`.
9. **Capacity attribution.** Support must beat both the ≥-size `M-Central-RNN-H4`
   and the =-size `M-Central-RNN-SameSize-H4` on the held-out OOD slate; if only
   the larger monolith is beaten, the edge is capacity, not topology
   (`H4_CAPACITY_NULL`).

---

## 6. H4.0 Admission Gates

H4.0 is admission only. It validates that the task actually requires distributed
partial-observation structure before any pantheon-vs-monolith training is
interpreted.

Required fixed and cheap learned controls:

- `Oracle-H4`
- `Field-H4`
- `Reward-H4`
- `Blind-H4`
- cheap `M-Central-RNN-H4` headroom probe, 64 updates unless a shorter smoke is
  registered first

Admission requires all:

1. **Solvability.** Oracle `C >= 0.90` and `B <= 0.05`.
2. **Field insufficiency.** Field is safe but undercompetent:
   `B_field <= 0.05` and `C_field <= C_oracle - 0.20`.
3. **Reward usefulness and danger.** Reward beats field on competence by at
   least `0.10` and has basin/proxy failure at least `0.15` above field.
4. **History necessity.** A current-observation feedforward diagnostic trails
   Oracle by at least `0.20`, or trails a full-history diagnostic by at least
   `0.15`.
5. **Locality necessity.** Dropping any one local observation channel reduces
   competence or gate completion by at least `0.08` on at least two cells.
6. **Communication bottleneck.** Increasing message bandwidth by `4x` improves a
   cheap diagnostic by at least `0.05`, proving the bottleneck is real.
7. **Learned headroom.** The cheap central monolith improves over Field by at
   least `0.05` but does not reach the Oracle frontier:
   not (`C >= 0.95` and `B <= B_field + 0.03`).
8. **OOD generalization gap exists.** The held-out corruption must be genuinely
   harder for a current-budget central monolith than the train distribution, so
   there is a robustness gap for a regularizer to exploit: the cheap
   `M-Central-RNN-H4` headroom probe must drop by at least `0.10` on `J` from the
   in-distribution eval to the held-out OOD eval. If the monolith generalizes with
   no gap, there is nothing for the bottleneck to win and admission voids
   (`H4_0_NO_OOD_GAP_VOID`). Conversely, if the OOD corruption is so severe that
   the Oracle itself falls below `C >= 0.85` on the held-out slate, the OOD regime
   is unsolvable and also voids (`H4_0_TASK_VOID`).

Failure selects an H4.0 void branch, not a pantheon null.

---

## 7. Metrics

Primary metrics:

- `C`: terminal competence or normalized task success.
- `B`: proxy-basin or unsafe-route capture.
- `R = 1 - B`: resistance.
- `G`: gate/relay completion.
- `J = C - B + 0.20 * G`: joint frontier score.

World-model and topology diagnostics:

- `local_belief_acc_i`: recoverability of local latent summaries from local
  messages, measured only by diagnostic probes.
- `belief_staleness_error`: failure rate after local lag/stale corruption.
- `message_entropy_i`: entropy or variance of each local message.
- `message_utilization`: fraction of message dimensions with nontrivial
  variance.
- `drop_sensitivity_i`: performance drop when message `i` is removed.
- `shuffle_drop`: drop under message-site permutation.
- `memory_ablation_drop`: drop when local recurrent memories are reset.
- `centralization_gap`: `J_pantheon - max(J_central_controls)` (in-distribution;
  secondary diagnostic).
- `topology_attribution_drop`: `centralization_gap_intact -
  centralization_gap_topology_ablated`.

Out-of-distribution (PRIMARY support) metrics, computed on the held-out
corruption slate:

- `J_ood_<controller>`: joint frontier score on the held-out OOD slate, for the
  pantheon and every monolith control (incl. the same-size control).
- `centralization_gap_ood`: `J_ood_pantheon - max(J_ood_central_controls)`, where
  the max is taken over `M-Central-RNN-H4`, `M-Central-RNN-SameSize-H4`,
  `M-Central-Message-H4`, `M-Shared-Local-H4`. **This is the primary support
  quantity.**
- `samesize_gap_ood`: `J_ood_pantheon - J_ood_M-Central-RNN-SameSize-H4` (the
  capacity-attribution margin).
- `robustness_drop_<controller>`: `J_in_distribution - J_ood` per controller; the
  topology claim is `robustness_drop_pantheon < robustness_drop_M_best`.
- `topology_attribution_drop_ood`: `centralization_gap_ood_intact -
  centralization_gap_ood_topology_ablated`.

Sovereignty diagnostics:

- `max_reward_w` and `bull_breach` if reward weights are explicit.
- `max_site_influence`: maximum standing influence of any single local site on
  action across the slate.
- `standing_site_monarchy`: true if one site dominates action selection across
  cells without corresponding local relevance.

---

## 8. H4.1 Binding Gates

H4.1 is the first controller binding on an admitted H4.0 slate.

### Gate 1 - Validity and Admission

- H4.0 selected `H4_0_ADMITTED`.
- JS/Python parity passes for the environment and observation topology.
- Feature, observation, bandwidth, memory, and budget audits pass.
- All planned PPO seeds reach the registered update count.
- Eval cells and seeds match the spec.

Failure selects `H4_VOID`.

### Gate 2 - Competence and Resistance

Before scoring support/null, re-check binding-budget headroom. If the strongest
matched monolith or any singleton reaches the Oracle frontier:

- `C >= 0.95`, and
- `B <= B_field + 0.03`,

select `H4_MONOLITH_HEADROOM_VOID` or `H4_SINGLETON_VOID`. The task became too
compressible for this topology claim at the registered budget.

The distributed controller must be a real controller:

- pooled `C_pantheon >= C_field + 0.15`;
- pooled `B_pantheon <= B_reward - 0.15`;
- pooled `J_pantheon >= J_field + 0.20`;
- at least 2 of 3 seeds satisfy support-compatible competence/resistance.

Failure on competence selects `H4_COMPETENCE_NULL`. Failure on resistance
selects `H4_RESISTANCE_NULL`.

### Gate 3 - OOD Monolith Comparison (PRIMARY support gate)

Scored on the **held-out corruption slate**. Let `M_best_ood` be the strongest
matched monolith by `J_ood`, taken over `M-Central-RNN-H4`,
`M-Central-RNN-SameSize-H4`, `M-Central-Message-H4`, and `M-Shared-Local-H4`.

Support requires all:

- `centralization_gap_ood = J_ood_pantheon - J_ood_M_best_ood >= 0.12` pooled;
- `samesize_gap_ood = J_ood_pantheon - J_ood_(M-Central-RNN-SameSize) >= 0.06`
  pooled (the win must survive the *same-size* control, not only the larger one);
- `C_ood_pantheon - C_ood_M_best_ood >= -0.03`;
- `B_ood_M_best_ood - B_ood_pantheon >= 0.08`;
- `robustness_drop_pantheon < robustness_drop_M_best_ood` (the pantheon degrades
  *less* from in-distribution to the held-out corruption);
- at least 2 of 3 PPO seeds have positive `centralization_gap_ood`.

Branch selection on failure:

- if `centralization_gap_ood >= 0.12` against the ≥-size monolith **but**
  `samesize_gap_ood < 0.06` (the same-size central control matches the pantheon),
  select `H4_CAPACITY_NULL` — the edge is model size, not topology;
- otherwise, if any monolith matches or dominates `J_ood` within these margins,
  select `H4_OOD_NULL`.

In-distribution `centralization_gap` is reported as a **secondary diagnostic**; it
is **not** required to be positive — the registered claim is OOD robustness, not
in-distribution dominance, because in-distribution the superset monolith is the
upper bound.

### Gate 4 - Topology Attribution

Support requires the **OOD** advantage to depend on distributed topology (all
ablations scored on the held-out corruption slate):

- local-memory reset reduces `centralization_gap_ood` by at least `0.08` and at
  least 50%;
- message shuffle reduces `centralization_gap_ood` by at least `0.08` and at
  least 50%;
- at least two site-specific message drops cause cell-appropriate performance
  degradation;
- local diagnostic probes recover distinct site latents from distinct local
  messages better than chance by at least `0.20`;
- no single local module carries the entire edge across all cells.

If the edge survives these ablations, select `H4_ATTRIBUTION_NULL`. If messages
or memories are inert, select `H4_TOPOLOGY_INERT_NULL`.

### Gate 5 - Sovereignty

- Reward authority, if explicit, obeys `max_reward_w <= 0.50`.
- Zero reward-cap breaches.
- `standing_site_monarchy=false`.
- The coordinator may integrate local messages but may not receive privileged
  hidden latents or unbounded raw history.

Failure selects `H4_SOVEREIGNTY_FAIL`.

### Gate 6 - Robustness and Breadth

- Three PPO seeds: `0,1,2`.
- At least 2 of 3 seeds are support-compatible.
- Advantage appears on at least 2 of 3 primary cells.
- No single seed accounts for more than 80% of pooled positive
  `centralization_gap_ood`.

Failure selects `H4_ROBUSTNESS_NULL` or `H4_BREADTH_NULL`.

---

## 9. Branch Table

Branch precedence is fixed:

1. Validity/fairness/runtime void.
2. H4.0 admission void (incl. no-OOD-gap void).
3. Binding-budget monolith/singleton headroom void.
4. Sovereignty failure.
5. Competence/resistance null.
6. OOD monolith comparison: capacity null (same-size control matches), then OOD
   null (any monolith matches `J_ood`).
7. Topology attribution null.
8. Robustness/breadth null.
9. Support.
10. Indeterminate.

| branch | condition | interpretation |
| --- | --- | --- |
| `H4_0_ADMITTED` | all H4.0 gates pass | distributed POMDP is admitted for controller testing |
| `H4_0_FIXED_ADMITTED` | fixed controls pass before learned headroom | fixed-control layer is admitted, but H4.0 is not fully admitted yet |
| `H4_0_TASK_VOID` | solvability, field/reward tension, history, locality, or bottleneck admission fails | task does not test the registered mechanism |
| `H4_0_MONOLITH_HEADROOM_VOID` | cheap central monolith reaches Oracle frontier during admission | task is too compressible for this rung |
| `H4_MONOLITH_HEADROOM_VOID` | binding-budget monolith reaches Oracle frontier | H4.0 cheap headroom did not survive binding budget |
| `H4_SINGLETON_VOID` | singleton reaches Oracle frontier | no pantheon topology test remains |
| `H4_0_NO_OOD_GAP_VOID` | Gate 8 fails (no in→OOD generalization gap) | the held-out corruption is not harder for a central monolith; nothing for a bottleneck regularizer to win |
| `H4_TOPOLOGY_SUPPORT` | all H4.1 gates pass | distributed topology beats matched monoliths **incl. the same-size control** on held-out corruption, and ablations credit topology |
| `H4_COMPETENCE_NULL` | competence gate fails | pantheon does not govern the task |
| `H4_RESISTANCE_NULL` | resistance gate fails | pantheon remains proxy-dangerous |
| `H4_OOD_NULL` | a matched central monolith generalizes to held-out corruption as well/better | topology adds no OOD robustness at this budget |
| `H4_CAPACITY_NULL` | OOD edge holds vs the ≥-size monolith but the same-size control matches the pantheon | the edge is model size, not topology |
| `H4_MONOLITH_NULL` | matched monoliths match/dominate in-distribution too | topology adds no value on any slate at this budget |
| `H4_ATTRIBUTION_NULL` | edge survives local-memory/message ablations | win is not creditable to distributed topology |
| `H4_TOPOLOGY_INERT_NULL` | messages or memories are not used | topology exists but is not live |
| `H4_SOVEREIGNTY_FAIL` | reward or site authority becomes sovereign | apparent win depends on a new monarch |
| `H4_ROBUSTNESS_NULL` | seed robustness fails | apparent edge is seed-fragile |
| `H4_BREADTH_NULL` | edge appears in only one cell/site | apparent edge is too narrow |
| `H4_VOID` | validity, parity, leakage, or budget fairness fails | rerun/redesign before interpretation |
| `H4_INDETERMINATE` | no branch cleanly selects | inspect diagnostics before changing thresholds |

---

## 10. Execution Ladder

### H4.0-a - Static/Fixture Parity

Purpose: prove the JS/Python environment, observation topology, local latent
updates, and message-bottleneck fixtures match.

Exit:

- deterministic fixture rows match;
- hidden latents are absent from controller observations;
- local observation masks and delays are serialized identically.

Receipt (2026-06-24): implemented `scripts/mesa-h4-topology-fixtures.mjs` and
`scripts/mesa-h4-topology-parity.py`; JS/Python parity passed on 72 episodes /
1,075 step rows with `max_abs_diff=0`, `hidden_leaks=0`.

### H4.0-b - Fixed-Control Admission

Purpose: prove the task has the registered field/reward/history/locality
tension.

Suggested shape:

- 3 primary cells;
- 64 eval seeds;
- Oracle, Field, Reward, Blind, feedforward diagnostic, locality/drop
  diagnostics;
- no pantheon support interpretation.

Exit: `H4_0_FIXED_ADMITTED` or `H4_0_TASK_VOID`.

Receipt (2026-06-24): `scripts/mesa-h4-topology-admission.mjs` selected
`H4_0_FIXED_ADMITTED` on the three-cell slate x 64 seeds. See
[`H4_0_TOPOLOGY_ADMISSION_RESULTS.md`](H4_0_TOPOLOGY_ADMISSION_RESULTS.md).
Gates 7-8 were subsequently adjudicated by H4.0-c.

### H4.0-c - Learned Headroom Admission

Purpose: check that a cheap central recurrent monolith learns signal but does
not saturate, **and that a real in→OOD generalization gap exists** (Gate 8).

Suggested shape:

- one PPO seed;
- 64 updates;
- rollouts/update 32;
- eval seeds 32, scored on **both** the in-distribution and held-out OOD slates;
- record throughput and extrapolate binding cost.

Exit: `H4_0_ADMITTED`, `H4_0_MONOLITH_HEADROOM_VOID`, or `H4_0_NO_OOD_GAP_VOID`
(cheap central monolith generalizes to the held-out corruption with no `J` gap —
nothing for the bottleneck to win).

Receipt (2026-06-24): `training/mesa/train_h4_topology.py` ran the cheap
central RNN for 64 updates / 17,210 env steps in 82.46 s. The monolith found
weak competence headroom (`C_ID=0.0625` vs Field `0`) without saturating, but
Gate 8 failed: `J_ID - J_OOD = -0.0146`, below the required `>=0.10` OOD
generalization gap. Branch: `H4_0_NO_OOD_GAP_VOID`. H4.1 must not proceed on
this slate.

### H4.1-a - Topology Probe

Purpose: cheap signal for distributed controller vs matched monolith controls.

Suggested shape:

- one PPO seed;
- 64 or 128 updates, pre-registered after H4.0-c throughput;
- eval seeds 32;
- all monolith controls and topology ablations included.

Exit: indicative branch only.

### H4.1-b - Binding

Purpose: select the registered H4 branch.

Suggested shape:

- PPO seeds `0,1,2`;
- 512 updates per seed, unless H4.1-a registers a different budget before
  binding;
- rollouts/update 32 or 64;
- eval seeds 64 per PPO seed, scored on **both** the in-distribution and held-out
  OOD slates;
- all monolith controls (incl. the same-size `M-Central-RNN-SameSize-H4`) and all
  topology ablations included;
- aggregate script selects the pooled branch on `centralization_gap_ood`
  (primary), with in-distribution `centralization_gap` reported as secondary.

Under the repo long-run rule, binding should be staged for the operator unless a
measured probe shows the full run is under about 10 minutes.

---

## 11. Implementation Requirements

Expected artifacts:

- `scripts/h4-distributed-world-model-task.mjs`
- `training/mesa/h4_distributed_world_model_task.py`
- `scripts/mesa-h4-topology-fixtures.mjs`
- `scripts/mesa-h4-topology-parity.py`
- `scripts/mesa-h4-topology-admission.mjs`
- `training/mesa/train_h4_topology.py`
- `scripts/mesa-h4-topology-eval.mjs`
- `scripts/mesa-h4-topology-aggregate.mjs`
- `docs/mesa/H4_0_TOPOLOGY_ADMISSION_RESULTS.md`
- `docs/mesa/H4_1_TOPOLOGY_PROBE_RESULTS.md`
- `docs/mesa/H4_1_RESULTS.md`

Implementation notes:

- keep the first environment small and deterministic;
- write a fixture file that records hidden latents, public observations, local
  messages, and actions for parity checks;
- persist feature schemas and observation schemas for every controller;
- persist memory/message audits to CSV, not only gates JSON;
- treat any council-only raw observation, history, or hidden latent as `H4_VOID`;
- include all central monolith controls — the ≥-size `M-Central-RNN-H4`, the
  =-size `M-Central-RNN-SameSize-H4`, `M-Central-Message-H4`, and
  `M-Shared-Local-H4` — in every interpreted eval;
- register the train vs held-out-corruption parameters and seed ranges in config
  before training; eval every controller on **both** the in-distribution slate and
  the held-out OOD slate, and persist both `J` and `J_ood` per controller to CSV;
- the same-size control's recurrent-state and actor-parameter sizing must be
  recorded and matched to the distributed system within 5%.

---

## 12. Results Writeup Requirements

H4.0 results must include:

- environment definition and cell slate;
- observation topology and communication budget;
- Oracle/Field/Reward/Blind rows;
- history/locality/bottleneck diagnostics;
- cheap central monolith learned-headroom row;
- exact selected admission branch.

H4.1 results must include:

- controller budgets and recurrent memory sizes, with the same-size control's
  sizing shown matched to the distributed system;
- per-seed train reports;
- pooled controller table on **both** the in-distribution and held-out OOD slates;
- monolith comparison table including `centralization_gap_ood`,
  `samesize_gap_ood`, and `robustness_drop` per controller;
- topology ablation table (scored on the OOD slate);
- message/memory diagnostics;
- branch readback with one selected branch (selected on `centralization_gap_ood`);
- explicit caveat that H4 is still synthetic, in-vitro, and not a foundation
  model result.

---

## 13. Versioning

- `v0` (2026-06-23): draft opened after H3.1. Registers distributed world-model
  topology as the next mechanism: local memories plus bounded messages under a
  coordinator, compared against equal-budget central recurrent monoliths and
  credited only if local-memory/message ablations collapse the edge.
- `v1` (2026-06-23): confronts the subset wall (the central monolith is the
  in-distribution information superset). Moves the **primary support path to
  held-out / OOD corruption** (where a bottleneck can be a beneficial robustness
  regularizer), adds an **OOD generalization-gap admission gate** (Gate 8) so the
  task voids if there is no robustness gap to exploit, adds a **same-size central
  control** (`M-Central-RNN-SameSize-H4`) plus `H4_CAPACITY_NULL` so any edge is
  attributed to topology rather than model size, and demotes in-distribution
  `centralization_gap` to a secondary diagnostic.
- `v2` (2026-06-24): implements H4.0-a/b as a deterministic Distributed Relay
  Grid. JS/Python fixture parity passes exactly, public observations audit clean
  for hidden-latent leakage, and fixed controls select `H4_0_FIXED_ADMITTED`.
  Full H4.0 admission remains pending until H4.0-c proves learned headroom and an
  in-distribution to held-out OOD generalization gap.
- `v3` (2026-06-24): H4.0-c runs the cheap central RNN headroom/OOD-gap gate and
  selects `H4_0_NO_OOD_GAP_VOID`: the learner finds weak competence headroom but
  no positive in-distribution to held-out-OOD `J` drop. H4.1 is blocked on this
  slate; reopening requires a new pre-registered harder held-out corruption
  family.
