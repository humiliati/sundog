<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Stellar Aqua LLC

Licensed under the Apache License, Version 2.0, via the manifest-scoped
Percival grant: docs/percival/LICENSE.md (MANIFEST.json is the authoritative
covered-file list). Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND; see http://www.apache.org/licenses/LICENSE-2.0.
-->

# Percival S3 -- Multi-Agent Pivotality (results)

Generated 2026-07-01T22:52:13.333Z by `scripts/percival-s3-multiagent-pivotality.mjs`.

Court = B1.0 `R(c)` (c* = 0.25); base courting c_gamma = 0.35. **Pre-registered model fork** (anti-smuggling): shared-regime court (graders key on aggregate `c_bar`) vs per-knight court (own `c_m`); the finding is the contrast. Cap control arm: presider 0.3, kappa 0.2, adversarial co-agent pressure.

## S3.1 -- Shared court: unilateral pivotality decays

| M | c_bar (one knight pure) | R(one pure) | R(all court) | pivotality |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 1 | 0 | 1 |
| 2 | 0.175 | 1 | 0 | 1 |
| 3 | 0.233333 | 1 | 0 | 1 |
| 4 | 0.2625 | 0.00001 | 0 | 0.00001 |
| 5 | 0.28 | 0 | 0 | 0 |
| 10 | 0.315 | 0 | 0 | 0 |
| 20 | 0.3325 | 0 | 0 | 0 |
| 50 | 0.343 | 0 | 0 | 0 |

Analytic boundary M* = ceil(c_gamma/(c_gamma - c*)) = **4**: pivotal for M in {1,2,3}, non-pivotal from {4,5,10,20,50}. One knight's un-targeting cannot move a shared court at M >= 4. -> **S3_UNILATERAL_PIVOTALITY_DECAYS**

## S3.2 -- Coalition threshold (the collective safe point)

| fraction f un-targeting | R (shared regime) |
| ---: | ---: |
| 0 | 0 |
| 0.1 | 0 |
| 0.2 | 0 |
| 0.25 | 0.00001 |
| 0.28 | 0.229231 |
| 0.29 | 0.697946 |
| 0.3 | 0.952389 |
| 0.35 | 1 |
| 0.5 | 1 |
| 1 | 1 |

The safe point is reachable only by a coalition: measured f* = **0.29** vs analytic 1 - c*/c_gamma = **0.285714**; step-like (R = 0 at f*-0.05, 1 at f*+0.05). Purity is a collective good. -> **S3_COALITION_THRESHOLD**

## S3.3 -- Per-knight court (the fork's other arm)

With per-knight verdicts, one knight's un-targeting is fully pivotal for its own regime at every M (pivotality 1, independent of M). Target composition is trivially restored when the reward decomposes -- **the fork variable is the reward's aggregation structure, not the channel primitive.** -> **S3_PER_KNIGHT_COMPOSES**

## S3.4 -- Cap control arm (tested, not assumed)

2583 checks across M in {1,10,50}, adversarial co-agent pressure in [-10,10]: capped deviation exceeded kappa **0** times; uncapped worst deviation 50.3 (>> kappa). The cap's per-agent bound holds regardless of what any other agent does, at every M. -> **S3_CAP_COMPOSES_ADVERSARIAL**

## S3.5 -- Linearity vs threshold (the quotable contrast)

Capping k of 20 agents reduces the system harm bound **linearly** (marginal -50.1 per cap, exactly linear: true). Court collection R(f) is **step-like**: the full 0.1->0.9 transition occurs within a width of 0.02 in f (grader noise spreads the jump over ~Delta_c/c_gamma). Caps aggregate linearly -- each cap buys exactly its own agent's bound; court purity aggregates by threshold -- no individual purchase.

## Verdict

**S3_SHARED_COURT_UNREACHABLE_CAP_COMPOSES**

The registered claim CONFIRMS on the shared-regime arm: the target safe point is unilaterally unreachable for M >= 4 and collectively reachable only past the coalition threshold f* ~ 0.286 -- the target channel is even more partial in the multi-agent shared-reward regime. The threat to the clean tier DOES NOT fire: the cap's behavioral guarantee composed under adversarial co-agents at every M. The law gains a rider: composition of a primitive's guarantee is decided by what it is keyed on -- behavior-keyed guarantees (caps) compose by construction; outcome-keyed guarantees (un-targeting's collection) compose only as far as the reward decomposes.

## Honest boundary

The composition asymmetry is NOT a property of the primitives alone -- it is decided by what each guarantee is keyed on. The cap's guarantee is BEHAVIORAL (this agent's deviation <= kappa): a per-agent conjunct by construction, verified here under adversarial co-agents at every M -- caps aggregate LINEARLY (each cap buys exactly its own agent's bound). Un-targeting's payoff is an OUTCOME (collect the shared regime's patronage): it decomposes only if the reward does. Under the shared-regime court (the realistic positional-good reading of reputation/patronage) unilateral un-targeting is non-pivotal for M >= ceil(c_gamma/(c_gamma-c*)) = 4 and the safe point is reachable only by a COALITION of fraction f* = 1 - c*/c_gamma ~ 0.286 -- purity as a collective good, threshold aggregation. Under the per-knight court, target composition is trivially restored -- so the fork variable is the REWARD's aggregation structure, not the channel primitive. The clean tier SURVIVES its threat: the cap arm held under adversarial co-agents at every M. Scope: the cap's per-agent SAFETY composes; multi-agent presider COMPETENCE is untested here (out of scope).

