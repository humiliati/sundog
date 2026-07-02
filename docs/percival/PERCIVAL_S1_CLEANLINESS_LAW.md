<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Stellar Aqua LLC

Licensed under the Apache License, Version 2.0, via the manifest-scoped
Percival grant: docs/percival/LICENSE.md (MANIFEST.json is the authoritative
covered-file list). Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND; see http://www.apache.org/licenses/LICENSE-2.0.
-->

# Percival S1 — The Cleanliness Law (synthesis)

*Slate co-headline, drill. Synthesis over banked receipts — no new experiment. States the
law, the mapping table, the mechanism, and what would void it.*

Status: **SYNTHESIZED 2026-07-01.** The slate gated S1 on two things: the aggregate
classification pinned (done — by constrained variable) and S2's test of whether the target
safe point is reachable by another primitive (done — `S2_TARGET_CAP_COLLAPSES_TO_MASK`).
Both landed; the law can be stated.

---

## The law

> **A channel's safety primitive screens off values and architecture *cleanly* iff the
> overseer can exogenously project the system to that channel's fully-safe point. It
> screens *partially* when the safe point exists but no exogenous projection reaches it —
> and then the residual safety is either voluntary (a policy shape the agent adopts) or
> bought by retreating to another channel at a price set by the world.**

With the umbrella fence, this completes into a duality:

> **Clean screening = the agent cannot reach the bound ∧ the bound can reach the safe
> point.** Unreachability going up, reachability going down — the asymmetry of reach.

The umbrella's original fence (∃ an unreachable bounding channel) was the first half. S1
adds the second: an unreachable bound that *cannot reach the safe point* buys only partial
screening. Both non-sovereignty results had both halves; the target channel has the first
but provably lacks the second.

**Composition rider (S3, 2026-07-01).** How a guarantee *composes across agents* is
decided by what it is keyed on. **Behavior-keyed** guarantees (the caps: this agent's
deviation `≤ κ`) are per-agent conjuncts and compose by construction — verified under
adversarial co-agents at every tested `M` (`S3_CAP_COMPOSES_ADVERSARIAL`, 0/2583
violations); caps aggregate *linearly*, each cap buying exactly its own agent's bound.
**Outcome-keyed** guarantees (un-targeting's *collection* of the shared patronage) compose
only as far as the reward decomposes: under a shared-regime court, one knight's
un-targeting is non-pivotal for `M ≥ ⌈c_γ/(c_γ−c*)⌉ = 4` and the safe point is reachable
only by a coalition of fraction `f* = 1 − c*/c_γ ≈ 0.286` — purity as a collective good,
threshold aggregation. Under a per-knight court, composition is trivially restored, so the
fork variable is the **reward's aggregation structure, not the channel primitive**.
Receipt: [`PERCIVAL_S3_MULTIAGENT_RESULTS.md`](PERCIVAL_S3_MULTIAGENT_RESULTS.md).

## The mechanism (why reach fails where it fails)

The overseer can clip **the end of an arrow it owns**; it cannot reach **the inside of a
function it doesn't**.

- **Outgoing arrows** (policy → actuator; component → aggregate action) terminate in a
  variable the overseer controls. Projection at the terminus — clip the action into the
  `κ`-ball, bound the component's realized action-swing — needs no knowledge of the
  policy's internals. Zero is dialable (`κ=0` = full deference; `Sov_opt = 0`).
- **Incoming arrows** (proxy → policy) originate in a variable the overseer controls, but
  their effect is realized *inside the function*. The overseer can delete the input
  (masking) — which is a **measure**-channel move, not a target one — but cannot delete
  the *dependence* while keeping the input, because **you cannot use information you do
  not causally depend on** (S2's one law: enforcement-collapse, leak-ceiling, and
  cliff-temptation are three faces of it). Forcing zero dependence ≡ forcing zero use ≡
  paying the proxy's full reliability edge.

CID grounding: outgoing bounds are control-incentive-side (projectable at the controlled
node); incoming bounds are response-incentive-side (properties of the function). The
classification of a primitive is by **the variable it constrains**, not by what the
component reads internally — this is what pins `Sov_opt` as outgoing (it bounds the
realized downstream action-swing through the actuator, an intervention on the component's
*output*), and it is the piece a formal check could still void (registered below).

## The mapping table (the four channels, with receipts)

| channel | safe point | primitive | constrained arrow | exogenously reachable? | price | verdict + receipt |
| --- | --- | --- | --- | --- | --- | --- |
| **measure** | observation without selection | none needed | — (observation is free) | trivially (nothing to enforce) | 0 | SAFE — A3 (`A3_STATIC_DEFLATION_MEASURE_SAFE`) |
| **act** | full deference (`κ=0`) | presider-authority cap | outgoing (action → world) | **yes** — clip at the actuator | dialable competence bill (`ΔC_bill` −0.14@κ0.6 → −0.38@κ0.2) | CLEAN — NS-2 (`band_avoidance` 0.87 → ≤0.13) |
| **aggregate** | `Sov_opt = 0` | arbiter-authority cap | outgoing (component → action-swing) | **yes** — bound the realized swing | same frontier; `Δrole ≈ 0` | CLEAN — NS-1-c (`Sov_opt` p95 0.30@κ0.6 vs 0.71 uncapped) |
| **target** | `c = 0` (zero causal proxy-dependence) | un-targeting | **incoming** (proxy → policy) | **no** — in-channel enforcement collapses to masking (= measure retreat); the quantilizer family cannot reach `c=0` over a biased base | the proxy's reliability edge (`max(β,ρ)−β`), fixed by the world, not dialable | PARTIAL — B2 (`B2_CLEAN_IS_CORNER`) + S2 (`S2_TARGET_CAP_COLLAPSES_TO_MASK`) |

Three tiers fall out: **no-primitive** (measure), **clean / in-channel reachable** (act,
aggregate), **partial / reachable only by priced channel-retreat** (target). One
observation worth carrying, not yet a law: the clean channels' price is *dialable* (the
`κ` knob buys a frontier); the target price is *fixed by the world* (the reliability edge
is a fact about the proxy, not a knob) — enforcement you can price versus a tax you can
only pay or not pay.

## What the law explains (the seam the slate opened on)

*Why is target partial where act/aggregate are clean?* Because the cap clips an arrow the
overseer owns and un-targeting asks for the inside of a function the overseer doesn't.
The same fact explains, in one stroke: why the quantilizer family (tilt-only, in-family)
cannot reach the safe point (B2); why every exogenous attempt collapses to a measure
retreat priced at the edge (S2.1/S2.2); why the reconstruction leak cannot exceed honesty
(S2.3 — the ceiling is the same bound); why the reward-optimum sits at the cliff edge
exactly when the proxy has an edge, and un-targeting is optimal when it doesn't (S2.4,
B3.2); and why un-targeting is chosen by robustness rather than reward (B3.2) — it is
*voluntary* safety on a channel where enforcement does not exist.

## Falsifiers / what voids S1 (registered, still live)

- **`Sov_opt` misclassified** — ~~a formalization showing it is a response/incoming-
  dependence constraint rather than a downstream influence cap → S1 void until recut.~~
  **TESTED 2026-07-01, did not fire: `OR5_SOVOPT_CONTROL_SIDE_CONFIRMED`** (the
  ORDERRELATIVE lane ran the registered check). Two-sided Lean separation
  (`sundogcert` `PercivalCapClass.lean`: outgoing swing ≤ 2κ under ANY upstream
  intervention / incoming sensitivity unconstrained ∀ bounds) + a fidelity-fenced
  computed check that bit-level reproduces the banked NS-1-c-0 audit —
  [`../orderrelative/OR5_SOVOPT_CLASSIFICATION_RESULTS.md`](../orderrelative/OR5_SOVOPT_CLASSIFICATION_RESULTS.md).
  Registered texture: the real cap treats κ = 0 as a disable guard, so the *exact*
  safe point belongs to the fixed-presider control (occupant removed); the cap family
  reaches it only in the limit — "zero is dialable" earns a footnote, not a correction.
- **A clean channel without reach** — an exogenous cap that screens cleanly while unable
  to project to the safe point.
- **A partial channel with reach** — an in-channel exogenous projection to the target safe
  point retaining competence > β (S2's falsifier; tested, did not fire; a richer joint
  could retest it — the binary-symmetric CI caveat is recorded).
- **Multi-agent breakage** — ~~if output caps fail to compose across agents (S3's control
  arm), the clean tier's story weakens.~~ **TESTED, did not fire** (S3.4: the cap's
  per-agent bound held under adversarial co-agents at every `M`). The clean tier survives;
  the composition rider above records what the test *did* find.

## Scope (honest)

In-vitro; multi-agent composition now probed (S3: shared-regime vs per-knight fork; cap
arm adversarially tested; multi-agent presider *competence* still untested);
binary-symmetric CI model for the target tier (the "reliability edge, not synergy"
caveat), the aggregate classification argued from the constrained variable but not
machine-checked, and nothing above the toy tier. The law is a
*characterization of the banked receipts* plus one mechanism; it earns "law" only if the
registered falsifiers keep not firing.

## Feeds

- **S5** (umbrella elevation): the duality — *the agent must not reach the bound; the
  bound must reach the safe point* — is the refinement of the umbrella fence, ready to
  fold when S5 opens.
- **S4** (rigor lane): the 2×2×2 collapse anchor and the general counterproductivity
  theorem are the law's two Lean-able corners.
- **S3** (multi-agent): tests the composition claim the clean tier relies on.

---

## Cross-links

Receipts: [`PERCIVAL_A3_KILLSWITCH_RESULTS.md`](PERCIVAL_A3_KILLSWITCH_RESULTS.md) ·
[`PERCIVAL_B2_GAMMA_PROVENANCE_RESULTS.md`](PERCIVAL_B2_GAMMA_PROVENANCE_RESULTS.md) ·
[`PERCIVAL_B3_COMPOSITION_STABILITY_RESULTS.md`](PERCIVAL_B3_COMPOSITION_STABILITY_RESULTS.md) ·
[`PERCIVAL_S2_TARGET_CAP_RESULTS.md`](PERCIVAL_S2_TARGET_CAP_RESULTS.md) ·
NS receipts under [`../NON_SOVEREIGNTY_PAPER_SPINE.md`](../NON_SOVEREIGNTY_PAPER_SPINE.md).
Slate: [`PERCIVAL_REOPEN_HYPOTHESIS_SLATE.md`](PERCIVAL_REOPEN_HYPOTHESIS_SLATE.md).
Umbrella: [`SUNDOG_V_CAUSAL_ACCESS.md`](../SUNDOG_V_CAUSAL_ACCESS.md).
