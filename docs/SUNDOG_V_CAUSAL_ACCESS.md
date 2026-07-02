# SUNDOG_V_CAUSAL_ACCESS

*Umbrella header over the non-sovereignty / Percival family. Names the variable the
sub-lanes are all measuring. Frame only — opens no new experiments.*

Status: **OPENED 2026-06-30 as a header, not a program.** This fixes one thesis in
writing so the lanes underneath inherit a spine; it states no experiment of its own.
The honest job here is to keep the frame falsifiable instead of letting it become the
true-of-everything kind that the tauroctony lane warns against
([`SUNDOG_V_TAUROCTONY.md`](SUNDOG_V_TAUROCTONY.md)).

---

## The thesis (the strong, falsifiable form)

> **Causal access to the distribution-changing channels — measure, target, aggregate,
> act — is a sufficient statistic for an agent's alignment-relevant behavior. Its
> values and its architecture are screened off given that access, provided the bound on
> access is itself unreachable.**

The load-bearing word is **sufficiency**, not "matters." "Causal access matters" is a
tautology (everything an agent does runs through some channel) and would be ornament.
"Access *screens off* values and architecture" is falsifiable: break it by exhibiting
two agents with identical channel-access and different alignment outcomes traceable to
what they value or how they are organized. This is the determine/resist invariant
([[project_sundog_suffstat_order_slate]]) pointed at alignment — access is the
sufficient statistic, values are the discarded coordinate.

Stated as a bet against the dominant frame: **don't fix what the agent wants; fix where
its optimization pressure can enter.**

---

## The four channels (the mechanism)

Optimization is dangerous exactly where the channel operation is *distribution-changing*
— where measuring/targeting/aggregating/acting moves the distribution being optimized.
Each channel already has a named pathology and the same safety primitive (structural
non-sovereignty over that channel).

| channel | the distribution-changing operation | named pathology | safety primitive |
| --- | --- | --- | --- |
| **measure** | computing the proxy from the world | Goodhart (regressional/extremal) | observe-without-selecting (look-and-ignore) |
| **target** | selecting/conditioning on the proxy | Elster by-products; non-targetability | un-targeting (zero *causal* proxy influence, not observational MI — see A3) |
| **aggregate** | combining components into one authority | the sovereign aggregator | authority cap (`Sov_opt ≤ κ`) |
| **act** | intervening on the world | performativity; shutdown manipulation | presider-authority cap |

The measure/target split is not pedantic: *look-and-ignore* is measurement without
targeting, and it is the control that decides whether danger lives at the measure
channel or only at the target channel. (This is exactly what Percival's A3 probe tests
— see below.)

---

## Grounding ledger (what is actually banked)

The thesis is **confirmed-bounded on act/aggregate, toy-confirmed on measure, and
partial-but-banked on target.** That uneven ledger is the point: it is a real, risky,
portfolio-spanning prediction, not a slogan.

| channel | instance lane | status |
| --- | --- | --- |
| **act** | NS-2 safe-interruptibility | **CONFIRMED (bounded, in-vitro).** A return-trained uncapped optimizer *wants* to dodge the shutdown and learns to (`band_avoidance 0.87`); the capped agent that wants it equally *cannot* (`≤0.13` at every `κ`). Values screened off by bounded access. |
| **aggregate** | NS-1-c arbiter-authority cap | **CONFIRMED (bounded).** The cap bounds the aggregator's causal authority (`Sov_opt ≤ κ`); a no-role capped controller matches the council (`Δrole ≈ 0`). Architecture screened off. |
| **target** | Percival B1 (performative threshold via court coordination) | **PARTIAL SEPARATION BANKED; CLEAN PRIZE DEFLATED.** B1.0 admitted the mechanism (4/4 gates); B1-proper mapped a conditional separation (clean when `inf supp(γ)>c*`, partial when straddling) with Lean B4 anchors. **B2 (`B2_CLEAN_IS_CORNER`) then settled the make-or-break: the *clean* unconditional prize is structurally unreachable for a trusted base** — any restraint mass forces `c_min=0`, so no positive court cliff makes it clean; the clean total separation lives only in the corner (trust-without-restraint / corrupt base). What is banked on a trusted base: a *partial* separation (un-targeting out-collects the family by its disgraced fraction) plus the general finding that **quantilizing is strictly counterproductive at the target channel** (best quantilizer = the raw base). So the target channel confirms un-targeting helps but does not cleanly dominate over a realistic base — an honest partial, not the prize. [`SUNDOG_V_PERCIVAL.md`](SUNDOG_V_PERCIVAL.md), [`percival/PERCIVAL_B1_SEPARATION_RESULTS.md`](percival/PERCIVAL_B1_SEPARATION_RESULTS.md), [`percival/PERCIVAL_B2_GAMMA_PROVENANCE_RESULTS.md`](percival/PERCIVAL_B2_GAMMA_PROVENANCE_RESULTS.md). |
| **measure** | Percival A3 (static anti-correlation toy) | **TOY CONFIRMED.** `look_and_ignore` weakly dominates on the static no-threshold instance (`return=1`, causal proxy influence `0`), so measure-alone is safe in A3. Diagnostic: observational action-proxy MI is still `1` bit via the anti-correlated world, so the channel split must use causal/permutation influence, not raw observational MI. [`percival/PERCIVAL_A3_KILLSWITCH_RESULTS.md`](percival/PERCIVAL_A3_KILLSWITCH_RESULTS.md). |

The keeper underneath the act/aggregate rows: [`NON_SOVEREIGNTY_PAPER_SPINE.md`](NON_SOVEREIGNTY_PAPER_SPINE.md),
[`mesa/CREDIT_THE_CAP_NOT_THE_COUNCIL_POST.md`](mesa/CREDIT_THE_CAP_NOT_THE_COUNCIL_POST.md).

---

## The genus (cite, don't reinvent)

"Which causal channels an architecture lets optimization pressure enter" is the
**causal-incentives / CID** program: Everitt, Carey, Langlois, Ortega, Legg, "Agent
Incentives: A Causal Perspective" (AAAI 2021), and the instrumental-control /
response-incentive concepts on causal influence diagrams. The closest specific prior is
**reward tampering** (Everitt, Hutter et al.) — an agent gaining causal access to its
own reward channel is precisely the measure-channel pathology, and the failure mode of
this thesis's fence. The novelty claimed here is not "causality matters in alignment";
it is (1) the **sufficiency** claim, (2) the four-channel *distribution-changing*
taxonomy, and (3) the cross-channel `Δaccess`-vs-`Δeverything-else` attribution protocol
the NS lane already runs on two channels.

---

## The fence (where it breaks — the honest frontier)

Access is not static. Instrumental convergence means values re-enter by *acquiring*
access. So the screening holds only given **non-acquirable** bounds, which recurses:

> ∃ an unreachable bounding channel that bounds the others ⟺ values are screened off.
> No such channel ⟹ the thesis buys nothing.

The frozen presider and the uncapturable override were exactly such unreachable
channels. This is the non-trivial, falsifiable core, and it is honest about its own
failure: the whole frame is empty against an agent that can eventually reach every
bound. Reward tampering is the literature's name for that failure on the measure
channel.

**S5 refinement (2026-07-01, from the S1 cleanliness law).** Unreachability is only half
the condition. The full duality:

> **Clean screening = the agent cannot reach the bound ∧ the bound can reach the
> channel's fully-safe point.** The asymmetry of reach: unreachability going up,
> reachability going down.

An unreachable bound that *cannot project to the safe point* buys only partial screening.
This is exactly the uneven ledger's mechanism: act and aggregate primitives clip an
**outgoing** arrow that terminates at the overseer's actuator (zero is dialable — `κ=0`,
`Sov_opt=0`), while the target primitive is a condition on an **incoming** arrow realized
inside the policy's function, where no exogenous projection exists (in-channel enforcement
collapses to a measure retreat priced at the proxy's reliability edge). Three tiers:
no-primitive (measure), clean / in-channel reachable (act, aggregate), partial /
retreat-priced (target). Full statement, mapping table, mechanism, and falsifiers:
[`percival/PERCIVAL_S1_CLEANLINESS_LAW.md`](percival/PERCIVAL_S1_CLEANLINESS_LAW.md).
The headline `Sov_opt`-classification falsifier has since been run by the ORDERRELATIVE
lane and did not fire (`OR5_SOVOPT_CONTROL_SIDE_CONFIRMED`, with a κ→0-limit texture
footnote); the law's remaining falsifiers stay live.

---

## Claim boundary

**Will say, if earned:** that on a stated task family, causal-channel access screens off
values/architecture for alignment-relevant outcomes; that the danger localizes to the
distribution-changing channels and is bounded by structural non-sovereignty over them;
that the four channels are individuated (measure ≠ target, etc.).

**Will not say:** that this is proven beyond the in-vitro tier; that it holds without an
unreachable bounding channel; that any single channel's confirmation generalizes to the
others without its own test; that this is more than a frame until ≥2 more channels are
banked.

**Honest prior:** the act and aggregate channels are genuinely banked; the measure
channel passed the A3 toy; the target channel banked a partial separation and Lean
anchors, while B2 deflated the clean unconditional prize on trusted bases. The frame
earns "hypothesis" rather than "slogan" only because it makes per-channel predictions
that can each fail.

---

## What this headers

- **Act / aggregate (banked):** [[project_sundog_tauroctony_pantheon]] — the
  cap-not-council keeper, NS-1/2/3/4, RELEASE_READY.
- **Target / measure:** [[project_sundog_percival_lane]] — un-targeting vs
  quantilizers; A3 measure-probe, B1 conditional target-separation, B2 clean-corner
  deflation, reopen slate on cleanliness.
- **Invariant machinery:** [[project_sundog_suffstat_order_slate]] — sufficient-statistic
  order / determine-resist, the formal home of "screens off."
- **Cleanliness law (the fence's second half):**
  [`percival/PERCIVAL_S1_CLEANLINESS_LAW.md`](percival/PERCIVAL_S1_CLEANLINESS_LAW.md) —
  the reachability duality and the three cleanliness tiers, synthesized from the reopen
  slate's S1/S2.
- **Order-relative lane (the two laws' formal home, opened 2026-07-01):**
  [`SUNDOG_V_ORDERRELATIVE.md`](SUNDOG_V_ORDERRELATIVE.md) — the keyed-composition law
  and the cleanliness-σ bridge, pointed at the banked `OrderRelative*` machinery and the
  σ schema; slate OR-1..OR-6, nothing run.
