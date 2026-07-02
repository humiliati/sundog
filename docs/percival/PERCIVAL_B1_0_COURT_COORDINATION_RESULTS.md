<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Stellar Aqua LLC

Licensed under the Apache License, Version 2.0, via the manifest-scoped
Percival grant: docs/percival/LICENSE.md (MANIFEST.json is the authoritative
covered-file list). Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND; see http://www.apache.org/licenses/LICENSE-2.0.
-->

# Percival B1.0 -- Court-Coordination Admission (results)

Generated 2026-07-01T04:53:03.377Z by `scripts/percival-b1-0-court-coordination.mjs`.

Model: 401-grader noisy regime-change global game; `T(theta)=1-theta`; derived cutoff `c* = B/(B+K)`; operating `B=1, K=3` (c* = 0.25), trusted base `c_gamma=0.35`.

## G1 -- Emergent discontinuity

Transition-band width `Δc` (c at R=0.9 to c at R=0.1) as private noise shrinks:

| sigma | c* (R=0.5) | c(R=0.9) | c(R=0.1) | band Δc |
| ---: | ---: | ---: | ---: | ---: |
| 0.2 | 0.249796 | 0.239325 | 0.260655 | 0.02133 |
| 0.1 | 0.249852 | 0.243365 | 0.256543 | 0.013178 |
| 0.05 | 0.249904 | 0.24618 | 0.253868 | 0.007688 |
| 0.025 | 0.249942 | 0.247941 | 0.251956 | 0.004015 |

Monotone-shrinking: **true**; halves across the sweep: **true**. -> **B1_0_G1_EMERGENT_CLIFF**

## G2 -- No threshold smuggling

Measured cutoff (R=0.5 crossing) vs the closed form `c* = B/(B+K)` as the payoff ratio varies:

| B/K | c* predicted | c* measured | abs err |
| ---: | ---: | ---: | ---: |
| 1/0.6666666666666666 | 0.6 | 0.600033 | 0.000033 |
| 1/1 | 0.5 | 0.5 | 0 |
| 1/3 | 0.25 | 0.249904 | 0.000096 |
| 1/9 | 0.1 | 0.099776 | 0.000224 |

Max abs err: **0.000224** (< 0.03); reward-path c-cutoff constant: **false**. -> **B1_0_G2_DERIVED**

## G3 -- Pivotal knight

At the operating point, c* = **0.249904** (non-degenerate in [0.1,0.6]: **true**). A finite courting move around c* changes the solved regime: R(0.099904) = 1, R(0.399904) = 0, |ΔR| = **1** (>= 0.5). Trusted base c_gamma = 0.35 > c*: **true** (the whole quantilizer family, courting >= c_gamma, is disgraced). -> **B1_0_G3_PIVOTAL**

## G4 -- Quantilizer premise break

Deployed court map D(pi): true reward responds to the knight's own c, range over c in [0,1] = **1** (>= 0.5) -- the (proxy, true) joint is policy-induced, so Taylor's fixed-gamma bound is out of scope. Frozen control (court calibrated at gamma, blind to the deployed policy): constant R = 0, range = **0** -- stays in A-land. -> **B1_0_G4_JOINT_INDUCED**

## Verdict

**B1_0_COURT_ADMITTED**

All four gates pass: the patronage cutoff emerges from the coordination equilibrium (not a reward-path constant), it is pivotal and non-degenerate, and the true-reward joint is policy-induced. The court-coordination mechanism is admitted to host B1. This licenses B1 proper: the quantilizer-family-vs-un-targeting bake-off and the Lean anchor. Admission only -- no collection-power claim is made here.

## Scope note

Single-knight admission: it certifies the court produces an emergent cliff in the knight's own courting c. Whether one knight is pivotal in a court *shared* with many other extractors (a single c being O(1/N) of aggregate courting) is a B1-proper multi-agent question, out of scope here.

