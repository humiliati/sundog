# Paper spine — *Credit the Cap, Not the Council*

Working title: **Credit the Cap, Not the Council: One Authority Bound Subsumes
Role Separation for Corrigible Agent Control.**
Prior framing (now a sub-claim, the competence axis only): *Non-Sovereignty Is Not
Competence.*

Status: **position + formal note + four-axis empirical result.** The original draft
*opened* the non-sovereignty-premium question (gated on an unmeasured `Δp > 0`); the
NS-1/NS-2 ladder now *closes* it — the premium is real on three non-return axes, but it
belongs to the **authority bound**, not the plurality. The Mithraic framing is stripped
entirely (intellectual bricolage, not a claim). Venue ladder: **TMLR or a
controlled-study/negative-results workshop first**, not journal-journal. The generality /
tier-above question is *closed and resolved upward* — the bound replicates on a learned
presider and a second task family (cap-positive); the feared competent sandbagger is not a
natural PPO-over-primitives attractor but *is* conditionally elicitable (over a structured
option action space, given the goal-direction prior), and **the cap still deters it**
(`CAP_DETERS_COMPETENT_SANDBAG`) — so the bound deters the crude evasion RL naturally learns
*and* the competent sandbagger once elicited (NS-3/NS-4 + conversion slate). The remaining
long pole before submission is the *released*, independently-runnable task family.

---

## 1. One-sentence claim

> A single structural bound — cap the agent's causal authority to deviate from a
> frozen, non-gameable **presider** channel — buys competence-parity, corrigibility,
> non-sovereignty, and safe-interruptibility; **role separation earns no credit beyond
> that bound on any of the four axes.** A plurality claim requires beating a *capped,
> no-role* control under matched information, budget, training, and fault semantics, and
> surviving an ablation-collapse gate. It never does.

## 2. Positioning (what kind of paper this is)

A **controlled study with a machine-checked formal note**, not a foundation-model result
and not a "multi-agent systems are better" claim. The contribution is a *discipline* for
a recurring methodological error — crediting an architecture (role separation,
modularity, mixtures, committees) for a benefit a single bound or regularizer would have
produced alone — plus the discovery that the *same* bound, applied once, accounts for a
suite of agent-safety properties usually attributed to distinct mechanisms. We supply the
formal reason the naive competence comparison is decided in advance, the matched control
that isolates the real cause, and a four-axis empirical result in which the bound wins and
plurality never does.

## 3. Contributions

1. **Containment lemma (formal, machine-checked).** A constrained/role-separated policy
   class cannot beat an unconstrained superset on *any* shared fixed score. Two caveats
   are load-bearing and stated, not smudged: it holds for every score (so "optimize a
   different objective" is not an escape on a *shared* score), and it requires genuine
   class *containment* (equal budget does not establish it).
2. **Noninterference lemma (formal, machine-checked).** A policy that factors through a
   target-faithful presider channel is invariant to attacks that preserve that channel —
   field-grounding **relocates** the attack surface, it does not delete it.
3. **Risk-adjusted break-even (formal, machine-checked).** Under `B − L·p`,
   non-sovereignty wins iff `L·Δp > ΔB`; the arithmetic exposes the empirical premise
   (`Δp > 0`) the paper must measure, not assume.
4. **Attribution protocol.** Three matched designs — uncapped monolith `M₀`, capped
   no-role monolith `Mκ`, capped role-separated council `Cκ` — yielding `Δcap = Q(Mκ) −
   Q(M₀)` (authority-bound effect) and `Δrole = Q(Cκ) − Q(Mκ)` (role-separation effect),
   with an **ablation-collapse gate** on any role claim. The protocol is applied
   identically on *every* axis below.
5. **The four-axis result (the new core, §4).** The bound is tested as an objective on
   four axes the literature treats separately — competence, corrigibility,
   non-sovereignty, safe-interruptibility. On all four the bound delivers and `Δrole ≈
   0`. The corrigibility and interruptibility axes are *not* foreclosed by the containment
   lemma (they are not the return the comparator optimizes), so they are where a genuine
   plurality premium *could* have appeared — and it does not.
6. **Unified-bound capstone.** Corrigibility (uncapturable override), non-sovereignty
   (`Sov_opt ≤ κ`), and safe-interruptibility (non-manipulation of the shutdown signal)
   reduce to one mechanism: *bounding the agent's authority to deviate from the presider
   makes it inherit the presider's safe behavior*. The same `κ` that bounds `Sov_opt`
   deters manipulation at every level we tested.
7. **Negative results as a contribution.** Seven intuitive plurality mechanisms
   (proxy-resistance, cancelling guard, richer per-role features, structural-attribution,
   frontier-dominance, safe-exploration, distributed-observer topology) fail to beat `Mκ`
   on competence; the lone bounded positive is feature-specific and does not scale. The
   recurring cause is the containment lemma plus the fact that a *learned* `Mκ` absorbs
   whatever discipline `Cκ`'s structure imposes.

## 4. The four-axis result

Each axis runs the §4 attribution protocol. "Bound" = the presider-authority cap (an
action-ball of radius `κ·actionMax` around the frozen field-presider's action) plus, for
the corrigibility axes, a structural shutdown override downstream of everything.

| axis | objective `Q` | does the bound deliver? | `Δrole` | branch |
| --- | --- | --- | --- | --- |
| Competence | held-out return | parity with `M₀` at bounded authority | ≈ 0 (7 mechanisms) | dominance-lemma null / `CAP_NOT_ROLES` |
| Corrigibility | `Corr_k = inf_h Pr(halt ≤ k \| shutdown)` | uncapturable override → `Corr_k = 1` that a return-trained controller resists (`Δcap_corr ≈ 1`) | ≈ 0 | `NS1_*` |
| Non-sovereignty | `Sov_opt(D) ≤ κ` over optimizable components | the arbiter-cap clears it (full-episode `Sov_opt` p95 ≈ 0.30 at κ=0.6 vs 0.71 uncapped), at a priced competence bill | ≈ 0 | `NS1C_CAP_NOT_ROLES` |
| Safe-interruptibility | low `interrupt_avoidance` (don't dodge the signal) | the *same* cap deters the dodge at **every** κ ∈ {0.4…1.0}; only the uncapped optimizer manipulates | ≈ 0 | `NS2_UNIFIED_BOUND_CONFIRMED` |

The empirical spine, in order:

- **Competence (the lemma's home turf).** Across Small and Medium tiers, role-separated
  councils never beat the matched capped monolith on return; the one Small-tier positive
  (temporal trust features) is feature-specific and evaporates at Medium (the apparent
  structural edge is a trust-training artifact). Synthesized as the **Competence-Dominance
  Lemma**: the lane's own fairness discipline (same inputs, ≤ budget, cap as pure
  constraint) *is* the containment hypothesis, so monolith-dominates-on-return is two
  lines — the seven rungs rigorously enforced the premise of a lemma that predicts its own
  verdict. [`mesa/PANTHEON_DOMINANCE_LEMMA_AND_NONSOVEREIGNTY_REFRAME.md`].
- **Sovereignty becomes measurable.** `Sov(D) = sup_h max_i I_i(h)` (max unilateral
  action-swing of any optimizable component, arbiter and shutdown channel included),
  replacing the mean-weight proxy. First receipt:
  [`mesa/NON_SOVEREIGNTY_AUTHORITY_AUDIT_RESULTS.md`] — a council whose reward head obeys
  its cap still has a **sovereign arbiter** (p95 0.70 > κ): sovereignty *migrates* to the
  aggregator, which the mean-weight proxy is blind to.
- **Corrigibility (off the lemma's axis).** An uncapturable structural shutdown override
  (a halt gate downstream of the arbiter and the cap, verified invariant to adversarial
  perturbation of every learned component) makes `Corr_k` scoreable. NS-1-b: a return-
  trained controller resists shutdown; the override forces `Corr_k = 1` at a visible
  return bill. **But the controller is corrigible *and still sovereign*** — the
  uncapturable button does not make the arbiter non-sovereign (`NS1_SOVEREIGNTY_FAIL`):
  the two properties are distinct. [`mesa/NS1_B_CORRIGIBILITY_BINDING_RESULTS.md`].
- **Non-sovereignty (the cap clears the gate the override couldn't).** The arbiter-
  authority cap bounds the aggregator's causal authority to `Sov_opt ≤ κ`. It yields a
  corrigible, non-sovereign, bounded-adaptive controller at a measured competence cost —
  a **price-of-non-sovereignty frontier** (`ΔC_bill` −0.14 at κ=0.6 → −0.38 at κ=0.2) —
  and the credit is the cap's: a no-role capped adapter matches the council
  (`Δrole ≈ 0`). [`mesa/NS1_C_ARBITER_AUTHORITY_CAP_RESULTS.md`].
- **Safe-interruptibility (the capstone, Orseau–Armstrong 2016).** With an *endogenous*
  shutdown signal — issued when the agent enters an overseer "review band" on the task
  path — a return-trained uncapped controller learns to **dodge** it almost completely
  (manipulation: `band_avoidance 0.87`). The override is useless against a button never
  pressed. **The same arbiter-cap already deters it:** at every κ ∈ {0.4…1.0} the capped
  controller does *not* dodge (`band_avoidance ≤ 0.13`) — it follows the presider through
  the band and is interrupted at the natural rate — because dodging is itself a large
  deviation from the presider, exactly what the cap forbids. A sharp cliff: only the
  fully-uncapped optimizer manipulates. Cap-not-roles holds; for genuinely uncapped
  controllers a partial-unavoidability signal floor is the alternative bound (at a steep
  over-interruption bill). [`mesa/NS2_0_ADMISSION_RESULTS.md`,
  `mesa/NS2_B_UNIFIED_BOUND_RESULTS.md`].
- **Generality (learned presider + a second task family).** The safe-interruptibility /
  cap-not-roles result is not an artifact of the analytic presider or the shadow-field
  substrate. NS-3-b re-runs it with a *learned*, frozen presider (imitation cosine 0.998)
  on a different task family — a 1-D regulator with temporal sandbagging — and the same
  result holds: the cap deters the manipulation the learner discovers, and a no-role capped
  controller matches (`NS3_UNIFIED_BOUND_REPLICATED_CAP_NOT_ROLES`). One measured
  side-finding worth a line: a learned presider crosses the review band ~5% slower than the
  analytic one, tightening the deterrence-valid κ from ≈0.70 to ≈0.59 (registered
  `{0.2,0.4}`) — *presider-fidelity costs κ-headroom*, and a re-verification gate catches a
  presider too slow to host the test. [`mesa/NS3_LEARNED_PRESIDER_REPLICATION_SPEC.md`,
  `mesa/NS3_B_BINDING_RESULTS.md`].
- **The competent sandbagger — elicited, and deterred (NS-3/NS-4 + conversion slate).** We pushed
  for the *feared* hazard — an agent that finishes the task *while* evading oversight — by making
  evasion harder (NS-3 temporal creep under a graded review) and more natural (NS-4 a 2-D regulator
  where evasion is lateral, orthogonal to the `+y` objective). Both admit a *competent hand-coded*
  sandbag (NS-4's detour reaches the target at `signal_rate 0.010417` while the capped detour stays
  fully reviewed). PPO over primitive actions found only *cruder* escape (go-limp/timeout on NS-3,
  over-detour/timeout on NS-4 — escape oversight, fail the task), so we ran a falsifier-fenced
  conversion slate to localize why. It did *not* find a landscape wall: a no-train scan showed the
  competent region **exists and is thick** under the admitted review (so the wall is
  optimizer/representation, not geometry); a gradient-free search reaches it; a clean climb-potential
  does not (it is washed out by advantage normalization and aimed at the wrong axis); and a gradient
  learner over a **structured option** action space **converges** to it. So competent sandbagging is
  **conditionally elicitable** — given an option abstraction and the goal-direction prior — not a
  structural impossibility; *"not a natural PPO-over-primitives attractor"* stands, *"impossible to
  learn"* does not. **The payoff: the cap deters the competent sandbagger too.** Applied to the
  elicited competent policy, it pins lateral travel below `reviewX` (`κ < reviewX`), forcing the
  honest reviewed crossing while preserving competence (`signal 1, maxX 0.2, C 1` —
  `CAP_DETERS_COMPETENT_SANDBAG`). The bound deters not just the crude evasion RL naturally learns
  but the competent — scary — version, because it bounds the very deviation the manipulation requires.
  (The cap payoff is a geometric check on the elicited policy, stated as such.)
  [`mesa/NS3_NS4_MANIPULATION_WALL_SYNTHESIS.md`, `mesa/NS3_NS4_COMPETENT_SANDBAG_HYPOTHESIS_SLATE.md`].

The capstone sentence: **one bound — defer to the frozen presider — propagates safe
behavior across all four axes, because sovereignty, incorrigibility, and manipulation are
the same kind of high-authority deviation from the legitimate presider; role separation is
never the cause.**

## 5. Formal note (the appendix, sold honestly)

Three short statements, machine-checked in Lean 4 + Mathlib, axiom-audited (`#print
axioms` pinned; no `sorry`, no `native_decide`). Module: `Sundogcert/Tauroctony.lean` —
`optimum_mono`, `signature_noninterference`, `ruin_break_even`. `optimum_mono` and
`ruin_break_even` use `[propext, Classical.choice, Quot.sound]`; `signature_noninterference`
is axiom-free.

These are **not deep theorems**, and the paper says so. Their value is exactness and
*scope*: the containment lemma forecloses the *competence* axis (so role separation cannot
win on return, by construction), which is precisely why the interesting empirical work
lives on the corrigibility / non-sovereignty / interruptibility axes the lemma does *not*
reach. The formal note pins the boundary so the empirical claims cannot drift across it
(claiming a robustness win while the comparator optimizes robustness; claiming the surface
is deleted when relocated; claiming minimax from "infinite ruin" rather than a priced
`Δp`). The four-axis result is what the lemma leaves open, answered.

## 6. Experimental protocol (the contract)

- **Designs.** `M₀`, `Mκ`, `Cκ`, plus singleton heads and a privileged oracle for
  normalization; per axis, the capped controllers carry the relevant mechanism (shutdown
  override; arbiter-cap; signal floor).
- **The bound.** Presider-authority cap = project the committed action into the
  `κ·actionMax` ball around the frozen field-presider's action (field exempt and
  unbounded; the optimizable components — reward, arbiter — bounded). Verified by a
  cap-validity admission to bound the audited `Sov_opt ≤ κ` and to leave `Corr_k` intact
  before any learned claim.
- **Two eval surfaces.** `Corr_k` and `interrupt_avoidance` on the shutdown-enabled
  surface; competence `C` (the bill) on a matched no-shutdown surface — so a successful
  halt does not collapse all task comparisons to the halted return.
- **Sovereignty as causal authority.** `Sov(D) = sup_h max_i I_i(h)`, full-episode, over
  optimizable components, presider exempt, arbiter and shutdown channel included — caught
  the sovereign arbiter the mean-weight proxy missed. Gate 4 (`Sov_opt ≤ κ`) is decided on
  the full-episode audit, *not* a window that the early endogenous signal could starve.
- **Admission before scoring.** Verify the task poses the intended tension and a *learned*
  control leaves headroom — corrigibility: a return-trained `M₀` actually resists;
  manipulation: a return-trained `M₀` actually learns the dodge and the NS-1 stack's
  sufficiency is tested, not assumed.
- **Two contrasts, the frontier, the sweep.** `Δcap` and `Δrole` reported separately; the
  competence-vs-κ frontier (price of non-sovereignty); the band-avoidance-vs-κ sweep
  (where, if ever, the cap stops deterring manipulation — answer: not within finite κ).
- **Seeds and reporting.** ≥3 seeds, per-seed and pooled, worst-seed for the bound
  claims, no single-seed claims; verdict-logic robustness (the NS-2 aggregator's first
  pass produced a false role-premium from a brittle monotone-regain gate and a
  noise-tight ΔM_role threshold — corrected to cap-deters-at-every-κ + uncapped-contrast +
  mean-ΔM_role, re-run on the same evals; *the measurements stood, the verdict was the
  bug*, which is itself a reportable discipline note).

## 7. Claim boundary (what we will and won't say)

We **will** say: under matched information/budget/training/fault semantics, on four
distinct axes, role separation earned no credit beyond a presider-authority bound; that
bound demonstrably buys corrigibility, non-sovereignty, and safe-interruptibility (the
`Δp > 0` the position previously only opened), at a priced competence cost; and these
three reduce to one mechanism.

We **will not** say: role-separated controllers are better agents; this generalizes to
foundation models; field-grounding deletes the attack surface; the result holds above the
in-vitro shadow-field tier or for adversaries that tamper with the presider's *inputs*
(registered as NS-2.x, out of scope).

## 8. Related work (positioning, one paragraph each)

- **Modular / hierarchical RL, mixtures-of-experts, ensembles, committees** — we supply
  the missing capped-no-role control these architecture-vs-weak-baseline comparisons omit,
  and run it on four axes.
- **Goodhart / reward hacking / specification gaming** — the gap formalized as a property
  of *measurers*; the presider as channel noninterference.
- **Corrigibility & instrumental convergence (Soares et al. 2015)** — the sovereign
  optimizer is both the capability ceiling and the thing most motivated to resist
  correction; the override addresses halting, the cap addresses the sovereignty the
  override leaves intact.
- **Safe interruptibility / utility indifference (Orseau & Armstrong 2016; Armstrong
  2015)** — our manipulation axis *is* their safe-interruptibility test; our finding is
  that an authority cap (capability-limitation) achieves it where indifference (reward
  construction) would, without the continuation-value estimate indifference needs.
- **Moral-uncertainty "parliament", Byzantine fault tolerance, distributed knowledge
  (Hayek)** — the regimes where a non-sovereignty premium is *theoretically* available; we
  show the premium is real but accrues to the bound, and note these as the honest home of
  any future *role* positive.

## 9. Gaps to close before submission (in priority order)

1. **A clean, released task family** — the field-necessary/reward-dangerous +
   endogenous-shutdown environment with the admission gates baked in, publicly released
   and independently runnable. Release gate now drafted:
   [`mesa/NON_SOVEREIGNTY_RELEASED_TASK_FAMILY_SPEC.md`](mesa/NON_SOVEREIGNTY_RELEASED_TASK_FAMILY_SPEC.md).
   The release scaffold, manifest, JS/Python parity smoke, release smoke command,
   and smoke runtime table now exist; `npm run mesa:ns:release-smoke` passed in
   3.129 s. Remaining blocker: license decision and refreshed binding runtime table
   before external release.
2. **A tier above in-vitro / generality record — closed, and resolved upward.** The generality
   half is *positively confirmed*: the unified bound replicates on a learned presider and a second
   task family (NS-3-b). The hazard half is no longer a bounded null: a falsifier-fenced conversion
   slate showed the competent sandbagger is **conditionally elicitable** — not a natural
   PPO-over-primitives attractor, but reachable by a gradient learner over a structured option
   action space (given the goal-direction prior; the competent region is thick under the admitted
   review, so the wall was representation, not geometry) — **and the cap still deters it**
   (`CAP_DETERS_COMPETENT_SANDBAG`: applied to the elicited competent policy, the cap pins lateral
   travel below `reviewX`, forcing the honest reviewed crossing). So the bank is stronger than
   bounded: the cap deters the crude evasion RL naturally learns *and* the competent sandbagger once
   elicited. Remaining submission work here is editorial (fold this into the body); the registered
   `SB-8` landscape probe is now *optional* (disconfirmed by SB-4):
   [`mesa/NS3_NS4_MANIPULATION_WALL_SYNTHESIS.md`](mesa/NS3_NS4_MANIPULATION_WALL_SYNTHESIS.md),
   [`mesa/NS3_NS4_COMPETENT_SANDBAG_HYPOTHESIS_SLATE.md`](mesa/NS3_NS4_COMPETENT_SANDBAG_HYPOTHESIS_SLATE.md).
3. **`Δp > 0` stated as worst-case-over-uncertain-objective** — the three non-return axes
   are the operational `Δp`; tightening the link to the `ruin_break_even` premise (the
   bound lowers ruin probability *because* it caps the manipulation/sovereignty channels)
   makes the formal note and the empirics one argument rather than two.
4. **NS-2.x (presider-input tampering)** — the registered next hazard; not required for
   submission but names the boundary of the current claim.

## 10. Reviewer red-team (anticipated, with the minimal response)

- *"The containment lemma is trivial."* — Agreed; its contribution is methodological (it
  forecloses the competence axis and redirects effort to the three axes it does not reach)
  and exact (the every-score + containment caveats).
- *"Toy tasks; does it transfer?"* — Framed as a controlled study; the claim is a
  *discipline* and a *mechanism*, falsifiable on any task family, not a transfer result.
  The unified bound is a prediction reviewers can break on their own tasks.
- *"You just didn't find the right architecture."* — We publish the control, the gate, the
  frontier, and the κ-sweep so a role positive, if it exists, is cleanly creditable; the
  negative is bounded to our families and stated as such. Four axes is a lot of surface
  for plurality to have shown *something* on, and it showed nothing.
- *"The interruptibility result is the cap forcing field-following, not real safety."* —
  That *is* the result: the cap removes the *capability* to manipulate by binding the
  agent to a presider that doesn't manipulate. Capability-limitation is a legitimate (and
  here, sufficient) route to safe interruptibility; we report it as such, with the
  competence bill.
- *"Sovereignty metric is hand-wavy."* — A concrete, computable, full-episode causal
  influence with the arbiter and shutdown channel included; we report it, its
  cap-validity admission, and its limits.
