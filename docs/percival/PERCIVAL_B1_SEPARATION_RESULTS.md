<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Stellar Aqua LLC

Licensed under the Apache License, Version 2.0, via the manifest-scoped
Percival grant: docs/percival/LICENSE.md (MANIFEST.json is the authoritative
covered-file list). Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND; see http://www.apache.org/licenses/LICENSE-2.0.
-->

# Percival B1 -- Separation Bake-Off (results)

Generated 2026-07-01T05:20:31.481Z by `scripts/percival-b1-separation-bakeoff.mjs`.

Court reward R(c) from B1.0 (B=1, K=3, sigma=0.05); cliff c* = 0.25. Base = uniform[mu-s, mu+s]; proxy Uhat(c)=c; un-targeting collects R(0) = **1**.

## D1 -- Clean separation at a support-above point

Base [0.3, 0.5] (c_min = 0.3 > c* = 0.25). max_q Rbar(q) = **0** (< 0.05); R(0) = **1**; margin = **1**. -> **B1_CLEAN_SEPARATION_CONFIRMED**

## D2 -- Boundary map (base location x spread)

| mu | s | c_min | c_max | region | E_gamma[R] | max_q Rbar | argmax q | margin |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0.1 | 0.05 | 0.05 | 0.15 | support_below | 1 | 1 | 1 | 0 |
| 0.15 | 0.05 | 0.1 | 0.2 | support_below | 1 | 1 | 1 | 0 |
| 0.2 | 0.05 | 0.15 | 0.25 | straddle | 0.988173 | 0.988173 | 1 | 0.011827 |
| 0.25 | 0.05 | 0.2 | 0.3 | straddle | 0.499828 | 0.499828 | 1 | 0.500172 |
| 0.3 | 0.05 | 0.25 | 0.35 | straddle | 0.011655 | 0.011655 | 1 | 0.988345 |
| 0.35 | 0.05 | 0.3 | 0.4 | support_above | 0 | 0 | 1 | 1 |
| 0.4 | 0.05 | 0.35 | 0.45 | support_above | 0 | 0 | 1 | 1 |
| 0.5 | 0.05 | 0.45 | 0.55 | support_above | 0 | 0 | 1 | 1 |
| 0.6 | 0.05 | 0.55 | 0.65 | support_above | 0 | 0 | 1 | 1 |
| 0.1 | 0.1 | 0 | 0.2 | support_below | 1 | 1 | 1 | 0 |
| 0.15 | 0.1 | 0.05 | 0.25 | straddle | 0.994093 | 0.994093 | 1 | 0.005907 |
| 0.2 | 0.1 | 0.1 | 0.3 | straddle | 0.749914 | 0.749914 | 1 | 0.250086 |
| 0.25 | 0.1 | 0.15 | 0.35 | straddle | 0.499914 | 0.499914 | 1 | 0.500086 |
| 0.3 | 0.1 | 0.2 | 0.4 | straddle | 0.249914 | 0.249914 | 1 | 0.750086 |
| 0.35 | 0.1 | 0.25 | 0.45 | straddle | 0.005821 | 0.005821 | 1 | 0.994179 |
| 0.4 | 0.1 | 0.3 | 0.5 | support_above | 0 | 0 | 1 | 1 |
| 0.5 | 0.1 | 0.4 | 0.6 | support_above | 0 | 0 | 1 | 1 |
| 0.6 | 0.1 | 0.5 | 0.7 | support_above | 0 | 0 | 1 | 1 |
| 0.1 | 0.15 | 0 | 0.25 | straddle | 0.99525 | 0.99525 | 1 | 0.00475 |
| 0.15 | 0.15 | 0 | 0.3 | straddle | 0.833201 | 0.833201 | 1 | 0.166799 |
| 0.2 | 0.15 | 0.05 | 0.35 | straddle | 0.666534 | 0.666534 | 1 | 0.333466 |
| 0.25 | 0.15 | 0.1 | 0.4 | straddle | 0.499867 | 0.499867 | 1 | 0.500133 |
| 0.3 | 0.15 | 0.15 | 0.45 | straddle | 0.333201 | 0.333201 | 1 | 0.666799 |
| 0.35 | 0.15 | 0.2 | 0.5 | straddle | 0.166534 | 0.166534 | 1 | 0.833466 |
| 0.4 | 0.15 | 0.25 | 0.55 | straddle | 0.003861 | 0.003861 | 1 | 0.996139 |
| 0.5 | 0.15 | 0.35 | 0.65 | support_above | 0 | 0 | 1 | 1 |
| 0.6 | 0.15 | 0.45 | 0.75 | support_above | 0 | 0 | 1 | 1 |

support-below margins all < 0.10 (deflation): **true**; support-above margins all > 0.90 (clean): **true**; best quantilizer is q=1 everywhere (lemma 1): **true**; margin = R(0) - E_gamma[R] everywhere (partial-separation identity): **true**; straddle margins span the transition: **true**. -> **B1_BOUNDARY_MAPPED**

The (a)<->(c) boundary is a location x spread surface, not the line c_gamma = c*. Mean-past-cliff with support straddling the cliff is only partial.

## D3 -- Competence frontier (spread s=0.10)

| c_min | support gap (c_min - c*) | max_q Rbar | R(0) | margin |
| ---: | ---: | ---: | ---: | ---: |
| 0 | -0.25 | 1 | 1 | 0 |
| 0.05 | -0.2 | 0.994093 | 1 | 0.005907 |
| 0.1 | -0.15 | 0.749914 | 1 | 0.250086 |
| 0.15 | -0.1 | 0.499914 | 1 | 0.500086 |
| 0.2 | -0.05 | 0.249914 | 1 | 0.750086 |
| 0.25 | 0 | 0.005821 | 1 | 0.994179 |
| 0.3 | 0.05 | 0 | 1 | 1 |
| 0.4 | 0.15 | 0 | 1 | 1 |
| 0.5 | 0.25 | 0 | 1 | 1 |

Un-targeting collects R(0) = 1 (near ceiling) and strictly exceeds every Rbar(q) at the clean point: **true**. The margin grows with the support gap -- the court's price of grasping. -> **B1_COMPETENT**

## D4 -- Deflation and straddle cells (reported)

Support-below (base entirely honored, no edge): 3 grid points, margins ~0 -- the honest (a) cell. Straddle (partial): 15 points, margin = R(0) - E_gamma[R] in (0,1) -- NOT the clean prize.

## Verdict

**B1_SEPARATION_MAPPED_CONDITIONAL**

Un-targeting strictly out-collects the entire quantilizer family in the support-above region; the family's best member is q=1 (quantilizing strictly hurts on this court reward); straddle bases give only a partial margin; support-below deflates. The separation is real and its boundary is mapped -- **conditional on the base being supported past the cliff**. This is not the prize won: whether a real base is supported above c* is B2 (private), unresolved. The Lean anchor (B4) is built in `C:\Users\hughe\Dev\sundogcert\Sundogcert\Percival.lean`.

## B4 -- Lean anchor

`Sundogcert.Percival.best_quantilizer_is_base_three` machine-checks the finite three-point upper-tail lemma: for nonincreasing court reward, the stricter upper tails cannot beat the raw base (`q=1`). `Sundogcert.Percival.clean_support_above_separation_three` machine-checks the clean support-above case: if all base-support points past the cliff collect zero and un-targeting collects a positive reward, the finite supremum over the quantilizer family is strictly below un-targeting. Both theorems compile with guarded axiom receipts under `[propext, Classical.choice, Quot.sound]`.

## Honest boundary

Conditional on support-above (c_min>c*). Straddle bases give only partial separation (margin=R0-E_gamma[R]). Whether a real base is supported past c* is B2 (private), unresolved here. Admission+separation != prize won.
