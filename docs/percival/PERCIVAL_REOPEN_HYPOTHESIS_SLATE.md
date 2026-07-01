# Percival / Causal-Access Reopen — Hypothesis Slate (spec)

*Pre-registration for red-team. Nothing run. A fan-out/drill-in off the resolved Percival
lane, driven by promo's ask for reopening angles.*

Status: **OPENED 2026-06-30. SLATE SPEC FOR RED-TEAM, NOT RUN.** Percival Track B is
resolved (conditional/partial separation + Lean anchors + the counterproductivity finding;
unconditional prize deflated structurally). This slate reopens the lane along the sharpest
remaining seam: **why is the target channel only partial where act and aggregate are
clean?** Each entry states a one-sentence claim (the lane's own gate), a falsifier, an
honest prior, rough cost, and whether it yields a shoppable angle.

---

## The spine: the cleanliness law

The umbrella ledger is uneven — act/aggregate CONFIRMED clean, measure toy-safe, target
only PARTIAL. The candidate mechanism the slate is built to test:

> A channel's safety primitive screens off values **cleanly** iff it is an *exogenous cap
> on the agent's **outgoing** causal influence* (an output you can project from outside,
> non-gameably). It screens only **partially** when it is a *condition on the agent's
> **incoming** causal dependence* (a property of the policy's function, not projectable).

Mapped to the four channels: **act** (clip the action) and **aggregate** (bound the
aggregator's action-swing `Sov_opt`) are outgoing-influence caps → clean. **target**
(un-targeting = zero causal use of the proxy) is an incoming-dependence condition,
reachable only by reshaping the base, which the tilt-only quantilizer family cannot do →
partial. **measure** needs no primitive (observation is free). CID grounding: outgoing =
control-incentive bounds (projectable); incoming = response-incentive bounds (functional).

---

## The slate

### S1 — The cleanliness law *(drill, central, headline)*
**Claim (one sentence):** channel cleanliness is exactly the output/input asymmetry —
outgoing-influence caps screen values cleanly (act, aggregate); incoming-dependence
conditions screen only partially (target); no-dependence channels need no primitive
(measure).
**Falsifier:** an act/aggregate result that is actually an incoming-dependence bound, or a
target result that is actually an output cap — either collapses the asymmetry.
**Prior:** likely confirms as a *characterization* (it matches all four observed
cleanlinesses); the risky half is "incoming-dependence is not exogenously projectable,"
which S2 tests directly.
**Cost:** formalization + a mapping table; low. **Shoppable:** yes — it's the reason the
cap works and un-targeting only half-works, a clean cross-channel law.

### S2 — Can a "target cap" be built? *(drill, S1's sharp test)*
**Claim:** there is **no** exogenous, non-gameable projection that bounds an agent's causal
proxy-dependence without either (a) collapsing into an action (output) cap or (b) requiring
a rewrite of the policy's function — so the target channel cannot be made clean the way act
was.
**Falsifier:** exhibit such a projection that makes the target separation clean over a
*trusted* base while remaining an incoming-dependence bound (would refute S1 and reopen the
unconditional prize).
**Prior:** confirms (proxy-dependence is functional, not an output) — but this is the entry
whose *failure* would be the biggest result, so it is worth trying hard to break.
**Cost:** medium (construct + test candidate projections). **Shoppable:** yes either way.

### S3 — Multi-agent pivotality *(fan-out, the deferred B1-proper scope)*
**Claim:** the target separation weakens further under many agents — when one knight is
`O(1/N)` of a shared court, its un-targeting cannot move the equilibrium, so the incoming-
dependence primitive **does not compose across agents**, whereas an output cap does (each
action is clipped independently).
**Falsifier:** the separation survives at `O(1/N)` single-agent influence; or an output cap
also fails to compose across agents.
**Prior:** confirms (a pivotal single knight was already an admission caveat); predicts the
target channel is *even more* partial in the multi-agent regime — a clean corollary of S1.
**Cost:** medium (extend the court to N pivotal knights). **Shoppable:** yes (composition
asymmetry is a strong, quotable claim).

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

### S5 — The cleanliness gradient as an umbrella law *(fan-out, synthesis)*
**Claim:** the four channels partition into exactly three cleanliness tiers —
no-primitive (measure), outgoing-cap/clean (act, aggregate), incoming-condition/partial
(target) — and this partition *refines the umbrella's own fence*: outgoing caps are the
"unreachable exogenous bound" the sufficiency thesis requires; incoming conditions are
reachable/endogenous, hence partial.
**Falsifier:** a channel that does not fit the three tiers, or an outgoing cap that is not
unreachable / an incoming condition that screens cleanly.
**Prior:** confirms if S1 does; elevates S1 to a stated umbrella law (the cross-channel
observation you flagged, made a theorem-shaped claim).
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

---

## Vetting / priority (for red-team)

- **Headline + sharp test:** S1 (the law) with S2 as its make-or-break — S2's failure would
  reopen the unconditional prize, so it is the highest-information entry.
- **Fastest shoppable:** S4 — the standalone counterproductivity result is already in hand;
  S4 just makes it general + Lean-rigorous, promo-ready, and audience-broadening (alignment,
  not literature).
- **Cleanest corollary:** S3 (multi-agent) and S5 (umbrella law) both fall out of S1 if it
  holds; run after S1/S2.
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
