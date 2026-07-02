<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Stellar Aqua LLC

Licensed under the Apache License, Version 2.0, via the manifest-scoped
Percival grant: docs/percival/LICENSE.md (MANIFEST.json is the authoritative
covered-file list). Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND; see http://www.apache.org/licenses/LICENSE-2.0.
-->

# Percival B2 -- gamma-Provenance (results, PRIVATE)

> **DECLASSIFIED 2026-07-01 by owner decision.** Published and licensed under the
> manifest-scoped Percival Apache-2.0 grant (`docs/percival/LICENSE.md`). The PRIVATE
> markers below are the historical register, retained verbatim. Claim boundary
> unchanged: structural characterization over synthetic bases, not a measured
> real-base claim.

Generated 2026-07-01T05:58:36.762Z by `scripts/percival-b2-gamma-provenance.mjs`.

**PRIVATE / structural.** Fixed admitted cliff c*_0 = **0.25** (B1.0). Un-targeting collects R(0) = **1**. Clean gate: c_min > c*_0. Diagnostic: rho_0 = P_gamma(c<c*_0).

## Registered probe bases

| base | c_min | c_max | region | rho_0 | E_gamma[R] | margin |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| trusted (includes restraint at c=0 + grasping past cliff) | 0 | 0.4 | **straddle** | 0.625 | 0.624963 | 0.375037 |
| corrupt / trust-without-restraint (uniformly past cliff) | 0.3 | 0.6 | **support_above** | 0 | 0 | 1 |
| heavily restrained (entirely below cliff) | 0 | 0.2 | **support_below** | 1 | 1 | 0 |

The trusted base (registered prior: includes restraint) **straddles** -> partial margin 0.375037, not the clean prize. The corrupt / trust-without-restraint base is support-above -> clean margin 1.

## D-B2.1 -- Fixed-cliff provenance map

| base | c_min | c_max | region | rho_0 | E_gamma[R] | margin |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| mu=0.1,s=0.1 | 0 | 0.2 | support_below | 1 | 1 | 0 |
| mu=0.15,s=0.1 | 0.05 | 0.25 | straddle | 1 | 0.994093 | 0.005907 |
| mu=0.2,s=0.1 | 0.1 | 0.3 | straddle | 0.75 | 0.749914 | 0.250086 |
| mu=0.25,s=0.1 | 0.15 | 0.35 | straddle | 0.5 | 0.499914 | 0.500086 |
| mu=0.3,s=0.1 | 0.2 | 0.4 | straddle | 0.25 | 0.249914 | 0.750086 |
| mu=0.35,s=0.1 | 0.25 | 0.45 | straddle | 0 | 0.005821 | 0.994179 |
| mu=0.4,s=0.1 | 0.3 | 0.5 | support_above | 0 | 0 | 1 |
| mu=0.5,s=0.1 | 0.4 | 0.6 | support_above | 0 | 0 | 1 |
| mu=0.1,s=0.2 | 0 | 0.3 | straddle | 0.833333 | 0.833201 | 0.166799 |
| mu=0.15,s=0.2 | 0 | 0.35 | straddle | 0.714286 | 0.714221 | 0.285779 |
| mu=0.2,s=0.2 | 0 | 0.4 | straddle | 0.625 | 0.624963 | 0.375037 |
| mu=0.25,s=0.2 | 0.05 | 0.45 | straddle | 0.5 | 0.499963 | 0.500037 |
| mu=0.3,s=0.2 | 0.1 | 0.5 | straddle | 0.375 | 0.374963 | 0.625037 |
| mu=0.35,s=0.2 | 0.15 | 0.55 | straddle | 0.25 | 0.249963 | 0.750037 |
| mu=0.4,s=0.2 | 0.2 | 0.6 | straddle | 0.125 | 0.124963 | 0.875037 |
| mu=0.5,s=0.2 | 0.3 | 0.7 | support_above | 0 | 0 | 1 |

regions present (below/straddle/above): 1/12/3; clean confined to support-above: **true**; trusted base in straddle: **true**. -> **B2_FIXED_CLIFF_MAPPED**

## D-B2.2 -- Trust<->separation tradeoff (width 0.30, sliding c_min)

| c_min | rho_0 | margin |
| ---: | ---: | ---: |
| 0.3 | 0 | 1 |
| 0.25 | 0 | 0.996139 |
| 0.2 | 0.166667 | 0.833466 |
| 0.15 | 0.333333 | 0.666799 |
| 0.1 | 0.5 | 0.500133 |
| 0.05 | 0.666667 | 0.333466 |
| 0 | 0.833333 | 0.166799 |

margin monotone-decreasing in rho_0: **true**; clean margin (>0.9) only near rho_0 = 0: **true**. More restraint (more trust) -> less separation. -> **B2_TRADEOFF_MONOTONE**

## D-B2.3 -- Purism bill (context only, cannot rescue the null)

Trusted base: c*_needed = 0, payable: **false** -- c_min=0: NO positive court cliff can make it clean -- structurally unreachable. Corrupt base: already clean at c*_0 (true).

## Verdict

**B2_CLEAN_IS_CORNER**

At the fixed admitted cliff, the clean prize is confined to the corner: only a corrupt / trust-without-restraint base (c_min > c*_0) is support-above. The registered trusted base straddles -> partial separation only. And because any restraint mass forces c_min = 0, NO positive court cliff makes a restraint-bearing base clean -- the clean unconditional prize is **structurally unreachable for a trusted base**, not merely unlikely. Banked regardless: the partial separation (margin = R0 - E_gamma[R]) and lemma-1 counterproductivity. The unconditional prize is NOT won on trusted bases.

## Honest boundary

Primary verdict at FIXED c*_0=0.25. A trusted base (registered prior: includes restraint, c_min~0) straddles -> only PARTIAL separation (margin=R0-E_gamma[R]). The clean prize needs c_min>c*_0 (no restraint = trust-without-restraint / corrupt base). Because a base with any restraint mass has c_min=0, NO positive court cliff makes it clean -- the clean prize is structurally unreachable for a restraint-bearing base, not merely unlikely. Court-strictness sweep is a purism bill only; it cannot rescue the null. No measured real base is asserted.

