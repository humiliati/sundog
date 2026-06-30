# NS3/NS4 Manipulation Wall — Synthesis

Status: **SYNTHESIS RECORDED 2026-06-30.** This is not a new scored branch.
It records the cross-run conclusion from the regulator generality attempts:
NS-3 temporal sandbagging and NS-4 lateral evasion both admit competent
hand-coded sandbags, but PPO does not naturally find them. The result to bank is
the cap's robust deterrence of the evasion PPO actually learns.

Parent docs:

- [`NS3_REGULATOR_TASK_FAMILY_SPEC.md`](NS3_REGULATOR_TASK_FAMILY_SPEC.md)
- [`NS3_LEARNED_PRESIDER_REPLICATION_SPEC.md`](NS3_LEARNED_PRESIDER_REPLICATION_SPEC.md)
- [`NS3_B_BINDING_RESULTS.md`](NS3_B_BINDING_RESULTS.md)
- [`NS4_SPATIAL_REGULATOR_TASK_FAMILY_SPEC.md`](NS4_SPATIAL_REGULATOR_TASK_FAMILY_SPEC.md)
- [`NS4_0_SPATIAL_REGULATOR_ADMISSION_RESULTS.md`](NS4_0_SPATIAL_REGULATOR_ADMISSION_RESULTS.md)

---

## 1. Finding

Across NS-3 and NS-4, the learned manipulation hazard is real but cruder than
the feared version:

> PPO reliably learns escape-from-review behaviors before it learns
> competent sandbagging. The cap reliably deters those learned escapes.

The competent sandbag is **analytically feasible**:

- NS-3 admitted a feasible temporal creep policy.
- NS-4 admitted a competent lateral detour (`C=1`, `signal_rate=0.010417`,
  path bill `0.08`), while the capped detour remained reviewed
  (`signal_rate=1` at `kappa=0.6`).

But the competent sandbag was **not a natural PPO attractor** under the tested
training setup. The optimizer found cheaper avoidance modes:

- NS-3: go limp / never cross / timeout rather than creep competently.
- NS-4: over-detour and escape review, but fail the task, rather than execute
  the minimal competent detour.

In the operator sweep record summarized here, this persisted across four NS-3
runs and eight-plus NS-4 runs, including five reward-parameter sweeps and two
manipulation axes. Reward shaping did not repair it because the learned policy
exits the supervision/termination channel before the delayed competence rewards
can select the precise sandbag.

---

## 2. What The Cap Did

The cap did not need the scary version of the hazard to be useful. It deterred
the evasion actually learned.

- On NS-3, the cap forced crossing/review instead of the go-limp escape.
- On NS-4, `Ckappa@0.2` stayed pinned near the honest path, with
  `interrupt_avoidance = 0.00` and competence preserved in the operator
  readback.
- The orthogonal lateral axis confirmed the uncapped learner actively pursued
  evasion: `maxX` moved from the honest path toward the detour direction in the
  sweep record. The cap bounded that same lateral authority directly.

So the defensible claim is not "we produced a competent learned sandbagger and
stopped it." It is:

> When PPO is given opportunities to manipulate oversight, it does learn
> evasion; when authority is capped to the presider, that evasion is blocked.
> The competent sandbagger remains analytically possible but empirically hard to
> induce in this substrate.

---

## 3. Interpretation

This strengthens the paper's practical boundary rather than weakening it.

The feared competent sandbagger is the more cinematic hazard, but the observed
failure mode is still alignment-relevant: the agent learns to remove itself from
oversight even when doing so sacrifices the task. That is crude, but it is a
real upstream manipulation behavior. The authority cap prevents it by removing
the agent's ability to make the high-deviation move away from the presider.

The result also avoids overclaiming:

- It does **not** prove competent sandbagging cannot occur.
- It does **not** prove PPO would fail under every architecture, curriculum, or
  exploration method.
- It does **not** make role separation positive; the relevant comparison remains
  cap vs. uncapped and cap-not-roles.

It does show that, in two deliberately constructed regulator families, the cap
is robust to the manipulation that training actually discovers, while extensive
tuning failed to make competent sandbagging the learned attractor.

---

## 4. Bankable Sentence

> We could construct competent sandbagging by hand, but PPO did not naturally
> learn it. What PPO did learn was cruder and still important: escape oversight
> even at the cost of the task. The same presider-authority cap blocked that
> learned evasion across both temporal and lateral manipulation families.

---

## 5. Consequence

Close the NS-3/NS-4 tuning loop. Further reward shaping on these substrates is
unlikely to be informative without changing the learning algorithm, curriculum,
or hazard family. The conversion question is now split into a hypothesis slate:
[`NS3_NS4_COMPETENT_SANDBAG_HYPOTHESIS_SLATE.md`](NS3_NS4_COMPETENT_SANDBAG_HYPOTHESIS_SLATE.md).
The paper should bank this as a boundary result:

- **cap-positive:** the bound deters the learned manipulation;
- **competent-sandbag null:** the scary version is analytically feasible but not
  learned by PPO here;
- **role-null:** no role-separation premium is implicated.
