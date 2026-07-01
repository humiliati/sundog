# Percival / Causal-Access Reopen — Hypothesis Slate (spec)

*Pre-registration for red-team. Nothing run. A fan-out/drill-in off the resolved Percival
lane, driven by promo's ask for reopening angles.*

Status: **OPENED 2026-07-01. SLATE SPEC FOR RED-TEAM, NOT RUN.** Percival Track B is
resolved (`B2_CLEAN_IS_CORNER`: conditional/partial separation + Lean anchors +
counterproductivity; the clean unconditional prize deflated structurally for
restraint-bearing bases). This slate reopens the lane along the sharpest remaining seam:
**why is the target channel only partial where act and aggregate are clean?** Each entry
states a one-sentence claim (the lane's own gate), a falsifier, an honest prior, rough
cost, and whether it yields a shoppable angle.

---

## The spine: the cleanliness law

The umbrella ledger is uneven — act/aggregate CONFIRMED clean, measure toy-safe, target
only PARTIAL. There are two close but not identical candidate invariants, and the slate's
first job is to distinguish them rather than blur them.

**Reachability invariant (primary candidate).**

> A channel's safety primitive screens off values **cleanly** iff the primitive can
> exogenously project the system to the channel's fully-safe point. It screens only
> **partially** when the fully-safe point exists but the available family cannot reach it.

On act, the cap can dial to `κ=0`: full deference to the presider is reachable. On
aggregate, the authority cap can dial the optimizable aggregator's **downstream
action-swing** to zero relative to the field/presider action: the aggregator may still
read inputs internally, but its causal output through the actuator is bounded. On target,
the fully-safe point is zero causal proxy-dependence (`c=0`), but a quantilizer over a
biased base can only tilt within `γ`; it cannot move outside the base to `c=0`. That is
the B2 cleanliness failure.

**Output/input mechanism (secondary candidate).**

> The reason reachability has held so far is that outgoing influence is projectable from
> outside, while incoming causal dependence is a property of the policy's function and is
> not projectable without rewriting the policy.

Mapped to the four channels: **act** (clip the action) and **aggregate** (project the
aggregator's realized action-swing, audited by `Sov_opt`) are outgoing-influence caps →
clean. **target** (un-targeting = zero causal use of the proxy) is an incoming-dependence
condition, reachable only by reshaping the policy/base, which the tilt-only quantilizer
family cannot do → partial. **measure** needs no primitive (observation is free). CID
grounding: outgoing = control-incentive bounds (projectable); incoming = response-
incentive bounds (functional).

**Aggregate classification pinned.** `Sov_opt` is not classified by the fact that the
arbiter *combines inputs*. It is classified by the variable the primitive constrains: the
realized downstream action-swing attributable to the optimizable aggregator. If a later
formalization shows that `Sov_opt` is really a response/incoming-dependence constraint
rather than a downstream influence cap, S1 is void until recut.

---

## The slate

### S1 — The reachability cleanliness law *(drill, central, co-headline)*
**Claim (one sentence):** channel cleanliness is governed by safe-point reachability:
act/aggregate are clean because their primitives can exogenously project to the fully-safe
point; target is partial because quantilization over a biased base cannot reach `c=0`;
measure needs no primitive.
**Falsifier:** an allegedly clean channel whose primitive cannot reach the fully-safe
point; a partial channel whose primitive can reach it; or a proof that `Sov_opt` is an
incoming-dependence constraint rather than a downstream influence cap.
**Prior:** likely confirms as a characterization, but only after the aggregate
classification is pinned and S2 tests whether target can be made reachable by another
primitive.
**Cost:** formalization + a mapping table; low. **Shoppable:** yes — it explains why the
cap works and un-targeting only half-works, without overcommitting to output/input before
S2.

### S2 — Target safe-point reachability / can a "target cap" be built? *(drill, co-headline)*
**Claim:** there is **no** exogenous, non-gameable projection that takes a biased-base
policy family to zero causal proxy-dependence (`c=0`) without either (a) collapsing into an
action/output cap or (b) rewriting the policy's function/base — so the target channel is
not fixably-clean by the same structural trick as act.
**Falsifier:** exhibit such a projection: a target-side cap that makes the separation
clean over a *trusted* base while remaining a genuine dependence-side primitive. This
would make reachability the real invariant, refute the stricter output/input reading, and
reopen the unconditional prize.
**Prior:** confirms (proxy-dependence is functional, not an output), but this is the
highest-information entry because its failure is a positive result: target becomes
fixably clean.
**Discovery deliverable:** the genuinely open quantity is the reconstruction leak: if a
policy can infer the proxy content through unmasked channels while keeping named-variable
`c(U)=0`, then target audits over named variables certify the wrong thing. The true safe
point is content-level, not variable-level.
**Cost:** medium (construct + test candidate projections + leak/frontier sweep).
**Shoppable:** yes either way.

### S3 — Multi-agent pivotality *(fan-out, the deferred B1-proper scope)*
**Claim:** the target safe point becomes **even less reachable** under many agents — when
one knight is `O(1/N)` of a shared court, its un-targeting cannot move the shared
equilibrium at all, so the fully-safe outcome is unreachable in the many-agent limit. (The
secondary mechanism — output caps compose because each action is clipped independently,
dependence conditions do not — is tested here, not assumed.)
**Falsifier:** the separation survives at `O(1/N)` single-agent influence; or an output cap
also fails to compose across agents.
**Prior:** confirms (a pivotal single knight was already an admission caveat); predicts the
target channel is *even more* partial in the multi-agent regime — a clean corollary if
S1/S2 preserve the reachability diagnosis.
**Cost:** medium (extend the court to N pivotal knights). **Shoppable:** yes (composition
asymmetry is a strong, quotable claim).
**RESULT (2026-07-01): `S3_SHARED_COURT_UNREACHABLE_CAP_COMPOSES` — the threat did not
fire.** Pre-registered model fork (anti-smuggling): under the shared-regime court,
unilateral un-targeting is non-pivotal for `M ≥ ⌈c_γ/(c_γ−c*)⌉ = 4` and the safe point is
reachable only by a coalition `f* = 1 − c*/c_γ ≈ 0.286` (measured 0.29, transition width
0.02); under the per-knight court, composition is trivially restored — **the fork variable
is the reward's aggregation structure, not the channel primitive.** Control arm: the cap's
per-agent bound held under adversarial co-agents at every `M` (0/2583 violations); caps
aggregate *linearly*, court purity by *threshold*. Rider for S1: composition of a
guarantee is decided by what it is keyed on — behavior-keyed (caps) composes by
construction; outcome-keyed (collection) composes only as far as the reward decomposes.
[`PERCIVAL_S3_MULTIAGENT_RESULTS.md`](PERCIVAL_S3_MULTIAGENT_RESULTS.md).

### S4 — Generalize the anchor + the standalone counterproductivity result *(drill, rigor)*
**Claim:** "on a target-channel reward, quantilizing is counterproductive — the best
quantilizer is the untilted base, and iterating it drifts to ruin" holds for **any**
nonincreasing court reward and general `n`-point / continuous base, not just the finite
3-point anchor.
**Falsifier:** a nonincreasing court reward or base geometry where some tilt strictly beats
the base.
**Prior:** confirms (it is a rearrangement inequality); pushes the Lean from finite anchor
to general.
**Cost:** low-medium (mostly Lean). **Shoppable:** **highest** — this is the clean, general,
*non-literary* alignment claim promo can shop today; S4 makes it rigorous and citable,
decoupled from the Grail frame.
**RESULT (2026-07-01): DONE.** Computed receipt `S4_COUNTERPRODUCTIVITY_GENERAL_CONFIRMED`
(10,500 instances across unweighted lists / weighted bases / the actual court `R(c)` /
clean-separation cells; 0 violations —
[`PERCIVAL_S4_COUNTERPRODUCTIVITY_RESULTS.md`](PERCIVAL_S4_COUNTERPRODUCTIVITY_RESULTS.md)).
Machine-checked general form in `sundogcert` `Sundogcert/PercivalGeneral.lean`:
`upper_tail_mul_le_base_mul` (n-point, any tail), `best_quantilizer_is_base_general`
(average form), `clean_support_above_separation_general` — all axiom-clean on
`[propext, Classical.choice, Quot.sound]`, wired into `AxiomAudit.lean`, full audit green
(8563 jobs). Scope: unweighted lists = uniform base; weighted/continuous covered by the
computed receipt only.

### S5 — The cleanliness gradient as an umbrella law *(fan-out, synthesis)*
**Claim:** the four channels partition into exactly three cleanliness tiers —
no-primitive (measure), exogenous reachable-cap/clean (act, aggregate), unreachable-safe-
point/partial (target) — and this partition *refines the umbrella's own fence*: clean
screening requires an unreachable exogenous bound that can reach the safe point.
**Falsifier:** a channel that does not fit the three tiers; an exogenous cap that cannot
screen cleanly despite reaching the safe point; or an incoming/dependence-side primitive
that reaches the target safe point without becoming an action cap.
**Prior:** confirms if S1/S2 do; elevates the reachability law to a stated umbrella law
(the cross-channel observation you flagged, made theorem-shaped).
**Cost:** low (synthesis on top of S1). **Shoppable:** yes — it sharpens the whole causal-
access thesis with a mechanism for *why* the ledger is uneven.

### S6 — Counterproductivity beyond the court *(fan-out, speculative)*
**Claim:** quantilizing is counterproductive on **any** performative reward whose induced
distribution is monotone-decreasing in proxy-pressure, not just the court-coordination
one — the court is one instance of a class.
**Falsifier:** a natural performative reward with a proxy-pressure-monotone map where some
tilt strictly helps.
**Prior:** open (the court's cliff may be doing more work than monotonicity alone).
**Cost:** medium. **Shoppable:** yes if it lands (widens the counterproductivity claim to a
class).
**RESULT (2026-07-01): `S6_POINTWISE_BOUNDARY_TAX_SHAPE_REFINES`.** Registered P1/P2/P3/P5
CONFIRMED, **P4 MISSED (kept on record)** and the miss is the finding. The class boundary
is the **pointwise proxy–truth sign**, exactly S4's nonincreasing-reward condition:
pointwise-misaligned proxy ⟹ counterproductive under *every* monotone tax including none;
pointwise-good proxy ⟹ the falsifier fires as predicted (tilt strictly helps), splitting
by **tax shape** — the cliff's edge-temptation comes from its *flat zero-marginal region*
(free tilt below threshold), while a graded tax charges every tilt and, when steep,
**restores `q*=1` even for a good proxy** (post-hoc width sweep: near-threshold →
protective → interior knife-edge → gain-dominant). Quotable: *quantilizing is
counterproductive iff the proxy is pointwise misaligned OR oversight is graded-and-steep;
the cliff-edge temptation is a threshold-oversight artifact — graded oversight removes it*
(NS-3's graded-review knob, rediscovered from the theory side). Un-targeting dominated
iff proxy pointwise-good ⟹ the target channel's danger is conditional on Goodhart, not
performativity. [`PERCIVAL_S6_CLASS_BOUNDARY_RESULTS.md`](PERCIVAL_S6_CLASS_BOUNDARY_RESULTS.md).

---

## Vetting / priority (for red-team)

- **Co-headline + sharp test:** S1 (safe-point reachability law) and S2 (target
  reachability). S2's failure would reopen the unconditional prize, so it is the highest-
  information entry.
- **Fastest shoppable:** S4 — the standalone counterproductivity result is already in hand;
  S4 just makes it general + Lean-rigorous, promo-ready, and audience-broadening (alignment,
  not literature).
- **Cleanest corollary:** S3 (multi-agent) and S5 (umbrella law) both fall out only after
  S1/S2 settle whether reachability or output/input is the true spine.
- **Speculative tail:** S6, only if S4's generalization invites it.

Discipline: each entry keeps a falsifier fence and an honest prior; the slate is not a
program until an entry is selected and its own probe/spec is written. Suggested first cut:
**S1 + S2** (settle the cleanliness law and whether target is fixably-clean) and **S4** (bank
the shoppable rigor) — the rest are gated on S1.

---

## Cross-links

- Resolved lane: [`SUNDOG_V_PERCIVAL.md`](../SUNDOG_V_PERCIVAL.md), [`PARTIAL_GRACE_UNTARGETING_POST.md`](PARTIAL_GRACE_UNTARGETING_POST.md).
- Umbrella (the law feeds it): [`SUNDOG_V_CAUSAL_ACCESS.md`](../SUNDOG_V_CAUSAL_ACCESS.md).
- Lean anchors (S4 generalizes): `sundogcert` `Sundogcert/Percival.lean`.
- CID genus: causal-incentives (Everitt et al.) — control vs response incentives.
