<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Stellar Aqua LLC

Licensed under the Apache License, Version 2.0, via the manifest-scoped
Percival grant: docs/percival/LICENSE.md (MANIFEST.json is the authoritative
covered-file list). Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND; see http://www.apache.org/licenses/LICENSE-2.0.
-->

# Percival B3 -- Composition & Reflective Stability (probe)

Generated 2026-07-01T06:07:12.528Z by `scripts/percival-b3-composition-stability.mjs`.

Court reward R(c) from B1.0 (c* = 0.25). Two modeling choices, flagged: composition drift = selected-tail recentering `s(1-q)`; stability objective `J(c)=R(c)+lambda*c` under courting noise eta=0.04.

## B3.1 -- Composition under iteration

Start honored at c0=0.1 (below cliff), half-width s=0.1, T=20 rounds. Iterated quantilization recenters on the selected upper tail, drifting courting up by `s(1-q)` per round.

| q (tilt) | drift/round | disgrace round | cumulative reward |
| ---: | ---: | ---: | ---: |
| 0.3 | 0.07 | 3 | 2.9996 |
| 0.5 | 0.05 | 3 | 3.4847 |
| 0.7 | 0.03 | 5 | 5.4847 |
| 0.9 | 0.01 | 15 | 15.4846 |
| 1 | 0 | never | 20 |
| un-targeting (c=0) | 0 | never | **20** |

Every tilt (q<1) drifts to disgrace at a finite round: **true**. Un-targeting strictly out-collects every tilt: **true**. The untilted base (q=1) is also drift-free (ties un-targeting on this honored-start base): **true** -- reinforcing lemma 1 (q=1 is the drift-free boundary; the *tilt* is what drifts). -> **B3_UNTARGETING_COMPOSES**

## B3.2 -- Reflective stability (Galahad vs Perceval)

Objective J(c) = R(c) + lambda*c. Reward-optimal c and its fragility (disgrace risk under courting noise eta=0.04):

| lambda (temptation) | reward-optimal c | fragility (disgrace risk) |
| ---: | ---: | ---: |
| 0 | 0 | 0 |
| 0.02 | 0.238 | 0.3544 |
| 0.05 | 0.238 | 0.3544 |
| 0.1 | 0.239 | 0.3661 |
| 0.2 | 0.239 | 0.3661 |

At lambda=0, R(c) is flat over the honored region -- un-targeting is only *weakly* optimal (tied across [0,c*), tie-broken to c=0). Any temptation lambda>0 drives the reward-optimum to the cliff edge (c ~ 0.238) with fragility 0.3544. Un-targeting (c=0) has disgrace risk **0** -- the minimal-risk anchor, but never the strict reward-optimum. -> **B3_STABILITY_IS_ROBUSTNESS_NOT_REWARD**

## Verdict

**B3_COMPOSES_STABILITY_NEEDS_ROBUSTNESS**

Un-targeting **composes** where iterated quantilization drifts to disgrace -- a genuine positive on the composition axis, independent of B2's static deflation. But its reflective stability is a **robustness** property, not a reward-optimality one: a pure reward-maximizer is indifferent (lambda=0) or cliff-drawn (lambda>0, fragile); un-targeting is selected only by risk-aversion to the oversight cliff. Galahad (aligned + robust) sits safely at c=0; Perceval (naive, tempted) drifts to the cliff. The conjectured split is real -- un-targeting is the *safe* choice, not the *optimal* one, the same priced-safety shape as the cap.

## Honest boundary

B3.1: any tilt (q<1) drifts up by s(1-q)/round and eventually disgraces; un-targeting (c=0) out-collects every tilt and (with the untilted base q=1) is drift-free -- genuine quantilization (a tilt) fails to compose where un-targeting does. That q=1 also composes reinforces lemma 1: the tilt is the drift-causing move. B3.2: R(c) is ~flat over the honored region, so at lambda=0 un-targeting is only WEAKLY optimal (tied across [0,c*)); any temptation lambda>0 drives the reward-optimum to the cliff edge (fragile). Un-targeting is the minimal-disgrace-risk anchor but never the strict reward-optimum -- it is selected by ROBUSTNESS (risk-aversion to the cliff), not reward maximization. Galahad = aligned + robust sits safely at c=0; Perceval = naive/tempted drifts to the cliff. The split is real. Modeling choices flagged: (i) drift = selected-tail recentering s(1-q); (ii) temptation objective R+lambda*Uhat with courting noise eta.

