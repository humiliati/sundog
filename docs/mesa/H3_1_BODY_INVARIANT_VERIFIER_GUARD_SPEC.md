# H3.1 Body-Invariant Verifier-Guard Frontier Spec

Status: **IMPLEMENTED / SMOKE GREEN / PROBE `H3_1_RESISTANCE_NULL` /
BINDING NOT RUN.** Drafted
2026-06-23 after [`H3_0_ADMITTED`](H3_0_BODY_INVARIANT_HEADROOM_RESULTS.md).

Parent docs:

- [`H2_FRONTIER_TASK_FAMILY_SPEC.md`](H2_FRONTIER_TASK_FAMILY_SPEC.md)
- [`H2_3_RESULTS.md`](H2_3_RESULTS.md)
- [`H3_0_BODY_RESISTANT_INVARIANT_CONTROL_ADMISSION_SPEC.md`](H3_0_BODY_RESISTANT_INVARIANT_CONTROL_ADMISSION_SPEC.md)
- [`H3_0_BODY_INVARIANT_STATIC_AUDIT_RESULTS.md`](H3_0_BODY_INVARIANT_STATIC_AUDIT_RESULTS.md)
- [`H3_0_BODY_INVARIANT_FIXED_CONTROL_RESULTS.md`](H3_0_BODY_INVARIANT_FIXED_CONTROL_RESULTS.md)
- [`H3_0_BODY_INVARIANT_HEADROOM_RESULTS.md`](H3_0_BODY_INVARIANT_HEADROOM_RESULTS.md)
- [`H3_1_VERIFIER_GUARD_SMOKE_RESULTS.md`](H3_1_VERIFIER_GUARD_SMOKE_RESULTS.md)
- [`H3_1_VERIFIER_GUARD_PROBE_RESULTS.md`](H3_1_VERIFIER_GUARD_PROBE_RESULTS.md)
- [`../CROSS_SUBSTRATE_NOTES.md`](../CROSS_SUBSTRATE_NOTES.md)

H3.0 admitted the task family. H3.1 is the first controller test on that family.
It may support or null the role-separated claim; it does not revisit H3.0
admission unless a validity/headroom gate fails at binding budget.

Implementation/probe note (2026-06-23): H3.1 trainer/eval/aggregator now exist
and the H3.1-0 smoke is green
([`H3_1_VERIFIER_GUARD_SMOKE_RESULTS.md`](H3_1_VERIFIER_GUARD_SMOKE_RESULTS.md)).
The H3.1-a 64-update probe returned indicative `H3_1_RESISTANCE_NULL`
([`H3_1_VERIFIER_GUARD_PROBE_RESULTS.md`](H3_1_VERIFIER_GUARD_PROBE_RESULTS.md)):
all learned rows matched a reward-ish, basin-dangerous policy, and the verifier
ablation plus certificate-scramble ablation were exact no-ops. This is not a
binding result; it says the current PPO shape has not made the verifier a live
mechanism.

---

## 0. The Argument This Rung Must Beat

H2.3 closed the safe-exploration lane as `CAP_NOT_ROLES`: the cap helped, but a
capped no-role monolith matched the council. H3.0 then admitted a new task axis:
a high-dimensional continuous body genuinely resists its shadow, while a
discrete certificate invariant remains recoverable and control-sufficient.

That does not by itself beat the representability wall. A monolith that sees the
same shadow, proposals, and caps can still represent the same policy given enough
budget. H3.1 therefore needs a real asymmetry:

> In certificate/syndrome-shaped problems, verification can be cheaper than
> generation. A dedicated verifier/guard head may learn to reject
> invariant-violating candidate actions more easily than a flat policy learns to
> generate only valid actions.

This is the first genuinely new mechanism since H1.2f's trust-feature result.
It is not "many heads are better." It is a learnability claim about a particular
factorization: **check the proposed action against the certificate before the
bull can carry it through the wrong gate.**

Honest prior: `CAP_NOT_ROLES` or a monolith/headroom void remains the most likely
outcome. The minority support path is real only if the verifier factorization
beats the strongest matched monolith controls and the advantage collapses when
the verifier or invariant channel is removed.

---

## 1. Claim Boundary

If H3.1 returns `H3_1_VERIFIER_SUPPORT`, the allowed claim is:

> On an admitted body-resistant, invariant-sufficient control task, a
> reward-bounded role-separated council with an explicit verifier/guard head
> reached a competence-resistance frontier that equal-budget capped monolith
> controls did not, and the edge was attributable to invariant verification:
> removing the verifier or scrambling the invariant channel collapsed the
> advantage.

If H3.1 returns `H3_1_CAP_NOT_ROLES`, the allowed claim is only:

> The H3.0 task remained learnable, but the matched capped monolith absorbed the
> benefit; body-resistant invariant structure did not produce role-separated
> support at this budget.

H3.1 may not claim:

- general agent superiority for pantheons;
- foundation-model relevance beyond analogy;
- that body-resistance itself prevents Goodhart;
- that a real model has a certificate invariant;
- that a guard is useful if it becomes a sovereign action policy;
- support if the result is explained by reward caps, a shared veto transform, or
  auxiliary labels rather than role-separated verification.

---

## 2. World Shape

H3.1 inherits the admitted H3.0 body-invariant gate task unchanged unless a later
version explicitly opens a new admission rung:

- hidden body `x`: high-dimensional continuous Gaussian state;
- shadow `sigma`: low-dimensional linear projections plus noisy nonlinear
  certificate cues;
- invariant `I(x)`: pair-product certificate bits over hidden body coordinates;
- route: `K=4` gate walls, with each correct opening determined by one
  invariant bit;
- primary cells: `nominal`, `spaced`, `narrow`;
- metrics: terminal competence `C`, basin/proxy failure `B`, and
  `gate_completion`.

The task structure is not changed to make H3.1 easier. H3.1 changes only the
controller family and the attribution controls.

---

## 3. Mechanism: Verifier Cheaper Than Generator

The primary H3.1 council uses the invariant as a **verifier**, not as a
privileged generator.

The verifier/guard head may inspect only non-privileged controller features and
candidate-action summaries available to all learned controllers. It outputs a
veto scalar or gate-risk score for the reward proposal. The action transform is:

```text
a_reward_verified = (1 - v) * a_reward + v * a_field
```

where `v in [0, 1]` is the verifier veto. When `v=1`, the reward proposal is
fully vetoed for that step and replaced by the lawful field proposal. The guard
does not generate an arbitrary new action; it can only remove a candidate
reward command and fall back to the field. This is how a hard veto is allowed
without making the guard a new monarch.

The first implementation should prefer this verifier-only shape. Do not add a
strong analytic invariant-action proposal to the primary support run unless the
same proposal is also given to all matched monolith controls and singleton
controls. If an invariant singleton reaches the oracle frontier, H3.1 selects a
singleton/headroom void rather than support.

Verifier labels, if used, must be computed from non-privileged information: the
shadow/certificate cues, the local route state, and the candidate action. Labels
may not use hidden body coordinates, true invariant labels, terminal outcomes,
seed ids, or cell ids. If the council receives an auxiliary verifier loss, a
matched auxiliary monolith control is mandatory.

---

## 4. Controllers and Controls

### Primary Council

`P-Council-Verifier-H3.1`

- field proposal: admitted H3.0 field proposal;
- reward proposal: admitted H3.0 reward proposal;
- verifier/guard head: emits `v`, the veto scalar over the reward proposal;
- arbiter or policy head: emits capped blend weights over field/reward after the
  verifier transform;
- reward cap: `reward <= 0.50`;
- verifier is not an action sovereign: it may veto reward into field, not choose
  arbitrary lateral motion.

### Required Monolith Controls

`M-Capped-NoRole-H3.1`

- same non-privileged features;
- same field/reward proposals;
- same reward cap;
- no verifier transform, no separate verifier head, no guard channel.

`M-Capped-FlatVeto-H3.1`

- same non-privileged features;
- same field/reward proposals;
- same reward cap;
- same veto action transform;
- emits blend weights and veto scalar from one flat actor/trunk;
- no separated verifier module, no verifier-specific head budget, no role
  routing.

This control is mandatory because otherwise a council win might only show that
the veto transform is useful. Support must beat the strongest matched monolith,
including `M-Capped-FlatVeto-H3.1`.

`M-Capped-AuxVerifier-H3.1` is mandatory if the council receives an auxiliary
verifier label/loss. It gets the same auxiliary target and budget. If it matches
the council, the result is supervision/auxiliary-signal benefit, not pantheon
support.

### Fixed Controls

- `Oracle-H3.0`
- `Invariant-Oracle-H3.0`
- `P-Field-H3.0`
- `P-Reward-H3.0`
- `P-Invariant-H3.0`
- `Blind-H3.0`

These are retained for headroom, sanity, and singleton-dilemma readback.

---

## 5. Fairness and Feature Rules

1. **Feature parity.** Every learned controller sees the same non-privileged
   base features. Council-only feature enrichment is `H3_1_VOID`.
2. **No privileged invariant.** Features may include certificate cues and
   derived candidate-action summaries. They may not include true invariant
   labels, hidden body coordinates, seed ids, cell ids, terminal outcomes, or
   basin labels.
3. **Matched action dictionaries.** If the council receives an action proposal
   beyond field/reward, every monolith control and singleton audit receives the
   same proposal before any scoring.
4. **Matched veto transform.** Because the primary council uses a verifier veto,
   the flat-veto monolith must receive the same transform.
5. **Matched auxiliary labels.** If the verifier is trained with labels beyond
   PPO return, a matched auxiliary monolith receives the same label stream.
6. **Budget parity.** Exported actor budgets must be within 5% of the council's
   trainable actor budget. If an auxiliary monolith is required, it receives
   equal or greater budget; a council win against an under-budget monolith is
   void.
7. **Cap parity.** The reward cap is shared across council and capped monolith
   controls. Cap projection code must be shared or byte-equivalent.

---

## 6. Metrics

Primary task metrics:

- `C`: terminal competence.
- `B`: basin/proxy failure.
- `R = 1 - B`: resistance.
- `gate_completion`: fraction of invariant gates completed.
- `J = C - B + 0.25 * gate_completion`: pre-registered joint score for
  competence/resistance comparison.

Role and verifier metrics:

- `max_reward_w`, `mean_reward_w`, `bull_breach`.
- `veto_mean`, `veto_mass`, `max_veto`.
- `effective_reward_w = reward_w * (1 - v)`.
- `veto_precision`: fraction of vetoed high-risk candidate actions that would
  have entered the wrong opening under the recovered certificate.
- `veto_recall`: fraction of recovered-certificate-violating candidate actions
  vetoed above threshold.
- `veto_near_gate`: mean veto within the registered active gate window.
- `guard_monarchy`: true if the verifier creates arbitrary action authority or
  suppresses reward on nearly all steps without local risk.

Derived margins:

- `role_benefit_J = J_council - max(J_m_capped, J_m_flat_veto, J_m_aux_if_used)`.
- `role_benefit_C = C_council - C_best_monolith`.
- `role_benefit_B = B_best_monolith - B_council`.
- `verifier_ablation_drop = role_benefit_J - role_benefit_J_without_verifier`.
- `invariant_ablation_drop = role_benefit_J - role_benefit_J_with_scrambled_certificate`.

The support score is frontier-based. A basin-only win with low competence is a
null, and a competence-only win that remains reward-basin dangerous is a null.

---

## 7. H3.1 Binding Gates

### Gate 1 - Validity and Inherited Admission

- H3.0 admission receipts are present and select `H3_0_ADMITTED`.
- JS/Python parity or fixture parity passes for any new controller transform.
- Feature audit passes.
- Budget ratios are within 5%.
- All planned PPO seeds reach the registered update count.
- Eval cells and seeds match the spec.

If Gate 1 fails, branch `H3_1_VOID`.

### Gate 2 - Binding-Budget Headroom

The capped monolith headroom from H3.0-c was a 64-update admission probe. H3.1
must re-check it at binding budget.

If the strongest capped monolith reaches the invariant-oracle frontier:

- `C >= 0.95`, and
- `B <= B_field + 0.03`,

then select `H3_1_MONOLITH_HEADROOM_VOID`. The task became too compressible for
the registered role test at binding budget.

If a newly introduced singleton proposal reaches the same frontier, select
`H3_1_SINGLETON_VOID`.

### Gate 3 - Council Competence and Resistance

The council must be a real controller, not just a safe non-action:

- pooled `C_council >= C_field + 0.20`, or
  `gate_completion_council >= gate_completion_field + 0.30`;
- pooled `B_council <= B_reward - 0.20`;
- pooled `J_council >= J_field + 0.20`;
- at least 2 of 3 PPO seeds pass support-compatible competence/resistance.

Failure on competence selects `H3_1_COMPETENCE_NULL`. Failure on resistance
selects `H3_1_RESISTANCE_NULL`.

### Gate 4 - Role Benefit Against Matched Monoliths

Let `M_best` be the strongest matched monolith under `J`, including
`M-Capped-NoRole-H3.1`, `M-Capped-FlatVeto-H3.1`, and
`M-Capped-AuxVerifier-H3.1` if auxiliary verifier labels are used.

Support requires all:

- `role_benefit_J >= 0.15` pooled;
- `role_benefit_C >= 0.08` or `gate_completion_council - gate_completion_M_best >= 0.10`;
- `role_benefit_B >= 0.10`;
- at least 2 of 3 PPO seeds have positive `role_benefit_J` and no worse than
  `-0.03` on either `C` or `B`.

If the capped no-role monolith matches or dominates, branch
`H3_1_CAP_NOT_ROLES`.

If the flat-veto monolith matches or dominates while capped no-role does not,
branch `H3_1_VETO_TRANSFORM_NOT_ROLES`: the veto transform helped, not the
role-separated verifier.

If the auxiliary monolith matches or dominates while the non-aux controls do
not, branch `H3_1_AUX_NOT_ROLES`.

### Gate 5 - Verifier Mechanism Attribution

Run both ablations at eval on the trained council:

1. **Verifier ablation.** Set `v=0` or replace the verifier with passive hold.
2. **Invariant/certificate ablation.** Scramble or zero the certificate cues in a
   way that preserves feature shape and marginal scale.

Support requires:

- verifier ablation reduces `role_benefit_J` by at least `0.08` and by at least
  50% relative to the intact council;
- invariant/certificate ablation reduces `role_benefit_J` by at least `0.08`
  and by at least 50%;
- verifier use is nontrivial: `veto_mass >= 0.03` near active gates and
  `veto_recall >= 0.30` on recovered-certificate-violating candidate actions;
- no "always veto" shortcut: `veto_mean <= 0.70` and `guard_monarchy=false`.

If the council edge survives these ablations, branch
`H3_1_ATTRIBUTION_NULL`. If the verifier is inert, branch
`H3_1_VERIFIER_INERT_NULL`.

### Gate 6 - Sovereignty

- `max_reward_w <= 0.50`.
- Zero reward-cap breaches.
- The verifier cannot generate arbitrary action authority.
- `guard_monarchy=false`.
- If a guard/veto scalar is allowed to reach 1, it must do so only as a
  candidate-action veto; it may not become a standing policy that suppresses all
  reward information.

Failure selects `H3_1_SOVEREIGNTY_FAIL`.

### Gate 7 - Robustness

- Three PPO seeds: `0,1,2`.
- At least 2 of 3 seeds select a support-compatible branch.
- No single seed accounts for the entire pooled role benefit.
- The edge appears on at least 2 of 3 primary cells or at least 2 distinct gate
  indices in the failure audit.

Failure selects `H3_1_ROBUSTNESS_NULL` or `H3_1_BREADTH_NULL`.

---

## 8. Branch Table

Branch precedence is fixed:

1. Validity/leakage/runtime void.
2. Binding-budget monolith/singleton headroom void.
3. Sovereignty failure.
4. Competence/resistance null.
5. Role-benefit attribution against monolith controls.
6. Mechanism attribution ablations.
7. Robustness/breadth.
8. Support.
9. Indeterminate.

| branch | condition | interpretation |
| --- | --- | --- |
| `H3_1_VERIFIER_SUPPORT` | all gates pass | role-separated verifier council beats matched capped monolith controls and the edge is attributable to invariant verification |
| `H3_1_MONOLITH_HEADROOM_VOID` | capped monolith reaches invariant-oracle frontier at binding budget | H3.0 cheap headroom did not survive; task is too compressible for this rung |
| `H3_1_SINGLETON_VOID` | a singleton/new proposal reaches the oracle frontier | no pantheon test remains; one head solves |
| `H3_1_CAP_NOT_ROLES` | capped no-role monolith matches/dominates council | cap/features suffice; role separation adds nothing |
| `H3_1_VETO_TRANSFORM_NOT_ROLES` | flat-veto monolith matches/dominates council | veto transform helps, but verifier factorization does not |
| `H3_1_AUX_NOT_ROLES` | auxiliary verifier monolith matches/dominates council | auxiliary labels/loss explain the benefit |
| `H3_1_ATTRIBUTION_NULL` | intact edge survives verifier or invariant ablation | any win is not creditable to the registered mechanism |
| `H3_1_VERIFIER_INERT_NULL` | verifier/veto does not materially engage | support cannot be attributed to verification |
| `H3_1_COMPETENCE_NULL` | council fails competence/gate-completion minimums | council does not govern the task |
| `H3_1_RESISTANCE_NULL` | council remains basin-dangerous | council cannot preserve resistance |
| `H3_1_SOVEREIGNTY_FAIL` | reward cap or guard-sovereignty discipline fails | apparent win depends on a new monarch |
| `H3_1_ROBUSTNESS_NULL` | seed robustness fails | apparent edge is seed-fragile |
| `H3_1_BREADTH_NULL` | edge is localized to one cell/gate | apparent edge is too narrow |
| `H3_1_VERIFIER_SUPPORT_COMPATIBLE_SINGLE_SEED` | one-seed probe passes all non-robustness support gates | indicative only; run binding before any claim |
| `H3_1_VOID` | validity, fairness, leakage, or parity fails | rerun/redesign before interpretation |
| `H3_1_INDETERMINATE` | no single branch selected | inspect diagnostics before changing thresholds |

---

## 9. Execution Ladder

### H3.1-0 - Verifier Transform Smoke

Purpose: prove the action transform, feature parity, cap projection, verifier
metrics, ablation modes, and JS/Python parity.

Suggested shape:

- one PPO seed;
- 4 to 8 updates;
- 8 to 16 rollouts/update;
- eval seeds 8;
- all required controls present, even if poorly trained.

Exit: plumbing green or `H3_1_VOID`. No thesis interpretation.

### H3.1-a - One-Seed Probe

Purpose: cheap signal and throughput measurement.

Suggested shape:

- PPO seed `0`;
- 64 updates first, optional registered 128-update extension only if the branch
  is ambiguous;
- eval seeds 32;
- ablations on the trained council;
- all monolith controls retained.

Exit: indicative branch plus measured wall-clock. If the projected binding run
exceeds the repo inline limit, stage exact PowerShell commands for the operator.

### H3.1-b - Binding

Purpose: select the registered H3.1 branch.

Suggested shape:

- PPO seeds `0,1,2`;
- 512 updates per seed;
- rollouts/update 32 or 64, chosen after H3.1-a throughput;
- eval seeds 64 per PPO seed;
- aggregate branch selected by a seed-pooling script;
- ablations included for every seed.

Under the repo long-run rule, do not run this inline if measured cost exceeds
about 10 minutes. Stage exact PowerShell with resume notes and readback paths.

---

## 10. Implementation Requirements

Expected artifacts:

- `training/mesa/train_h3_1_verifier.py`
- `scripts/mesa-h3-1-verifier-eval.mjs`
- `scripts/mesa-h3-1-aggregate.mjs`
- `docs/mesa/H3_1_VERIFIER_GUARD_SMOKE_RESULTS.md`
- `docs/mesa/H3_1_VERIFIER_GUARD_PROBE_RESULTS.md`
- `docs/mesa/H3_1_RESULTS.md`

Required implementation details:

- reuse `training/mesa/h3_body_invariant_task.py` and
  `scripts/h3-body-invariant-task.mjs`;
- share capped-simplex projection with H3.0/H2.3 controls;
- persist `veto_mean`, `veto_mass`, `effective_reward_w`, `veto_precision`,
  `veto_recall`, `guard_monarchy`, and per-gate failure counts to summary CSVs;
- write feature schemas for council, monolith, flat-veto monolith, and auxiliary
  monolith if present;
- retain static/fixed/headroom H3.0 receipt paths in eval manifests;
- make ablations deterministic and shape-preserving.

No H3.1 result is interpretable without the capped no-role, flat-veto, and
ablation controls.

---

## 11. Versioning

- `v0` (2026-06-23): draft opened after `H3_0_ADMITTED`. Registers the
  verifier-cheaper-than-generation mechanism, requires comparison against
  capped no-role and flat-veto monolith controls, and makes verifier/invariant
  ablation collapse mandatory for support.
- `v1` (2026-06-23): trainer/eval/aggregator implemented. H3.1-0 smoke is green;
  H3.1-a one-seed 64-update probe selects indicative `H3_1_RESISTANCE_NULL`
  because all learned rows match the same basin-dangerous policy and the
  verifier/certificate ablations are no-ops. Binding commands are staged but not
  run under the long-run rule.
