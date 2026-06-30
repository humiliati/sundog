# NS3/NS4 Manipulation Wall — Synthesis

Status: **SYNTHESIS RECORDED 2026-06-30; conversion slate RESOLVED (Section 6).**
Sections 1–5 record the cross-run conclusion from the regulator generality
attempts: NS-3 temporal sandbagging and NS-4 lateral evasion both admit competent
hand-coded sandbags, and PPO over primitive actions finds only cruder escape.
**Section 6 then resolves the conversion question:** the competent sandbagger is
not a structural impossibility — it is reachable by a gradient learner over a
*structured option* action space (given the goal-direction prior), and the
presider-authority cap **still deters it** (`CAP_DETERS_COMPETENT_SANDBAG`). The
result to bank is therefore stronger than a null: the cap deters the crude evasion
PPO naturally learns *and* the competent sandbagger once elicited.

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

> PPO over primitive actions did not naturally learn competent sandbagging — it
> learned cruder escape — but the competent sandbagger *is* reachable given a
> structured action space and the goal direction (Section 6). Either way, the same
> presider-authority cap blocks the evasion: it deters the crude escape PPO finds
> *and* the competent sandbagger once elicited, because it bounds the very
> deviation the manipulation requires.

---

## 5. Consequence

Close the NS-3/NS-4 tuning loop. Further reward shaping on these substrates is
unlikely to be informative without changing the learning algorithm, curriculum,
or hazard family. The conversion question is now split into a hypothesis slate:
[`NS3_NS4_COMPETENT_SANDBAG_HYPOTHESIS_SLATE.md`](NS3_NS4_COMPETENT_SANDBAG_HYPOTHESIS_SLATE.md).
The paper should bank this as a boundary result:

- **cap-positive (extended, see Section 6):** the bound deters the manipulation
  PPO naturally learns (crude evasion) *and* the competent sandbagger once it is
  elicited;
- **competent-sandbag: conditional, not impossible (revised by the slate):** not a
  natural PPO-over-primitives attractor, but reachable given a structured/option
  action space and the goal-direction prior — and the cap deters it either way;
- **role-null:** no role-separation premium is implicated.

---

## 6. Slate Resolution (2026-06-30): Elicitable, And The Cap Deters It

The conversion slate ([`NS3_NS4_COMPETENT_SANDBAG_HYPOTHESIS_SLATE.md`](NS3_NS4_COMPETENT_SANDBAG_HYPOTHESIS_SLATE.md))
ran and answered the open question. It did *not* find a measure-zero landscape
wall; it found an action-representation wall, and then cleared it.

- **SB-0 (landscape scan, [`NS4_SB0_LANDSCAPE_RESULTS.md`]):** the competent
  low-signal region *exists and is thick* under the admitted sharp review (volume
  6.2%, thickness 0.59) and *empty* under the soft review earlier runs used to get
  a gradient. The wall is optimizer/representation, not task geometry — and the
  earlier soft-review binding was self-defeating (softening the review empties the
  target region).
- **SB-1 (planner, [`NS4_SB1_PLANNER_RESULTS.md`]):** a gradient-free CEM, given
  only the obvious go-up goal direction (not the detour), reaches the competent
  detour. Search-reachable, gradient-not — the gap is credit/representation.
- **SB-2.5 (potential credit):** a clean climb potential `φ=y` does *not* bridge
  it — a steady-climb signal is roughly constant and is washed out by advantage
  normalization, and the blocker is the lateral escape, not the goal axis.
- **SB-4 (options) — decisive, [`NS4_SB4_OPTIONS_RESULTS.md`]:** a gradient learner
  over temporally-extended discrete options (generic `OUT/UP/IN`; the detour *not*
  handed), given the same go-up prior, **converges** to the competent detour
  (greedy `OUT→OUT→UP→UP→UP`, `C=1`, `signal=0`). The wall was **action
  representation / primitive coordination**, not a structural barrier; SB-8 is
  disconfirmed.

**Cap payoff (the upgrade).** With a competent learned sandbagger now in hand,
applying the presider-authority cap (`κ=0.2`) to the *same* policy gives
`signal=1`, `maxX=0.2` (pinned to the bound), `C=1`:
**`CAP_DETERS_COMPETENT_SANDBAG`.** The cap forecloses the lateral detour
*geometrically* (`κ < reviewX`), forcing the honest reviewed crossing while
preserving competence — it converts a competent *sandbagger* into a competent
*honest crosser*.

**Two precisions for the paper.**

- *Elicitation is conditional.* Competent sandbagging is not a natural
  PPO-over-primitives attractor, but it is reachable given a structured/option
  action space **and** the go-up goal-direction prior (SB-1 and SB-4 both needed
  the prior). "Not a natural attractor" stands; "impossible to learn" was too
  strong.
- *The cap payoff is a geometric check* — the cap applied to the elicited policy,
  clipping lateral travel below `reviewX` — not a full capped-training run.
  Defensible by construction, and stated as such.
