# Percival S2 -- Target-Cap Probe (results)

Generated 2026-07-01T22:18:05.032Z by `scripts/percival-s2-target-cap-probe.mjs`.

Model: `V,U | G` conditionally independent, `P(V=G)=beta`, `P(U=G)=rho`; court = B1.0 `R(c)` at `c*=0.25`; `Total = Competence x R(c)`. Tax = `max(beta,rho)-beta` (reliability edge under the binary-symmetric CI joint -- not synergistic info; caveat recorded for richer joints).

## S2.1 + S2.2 -- Collapse and gap (per grid cell)

| beta | rho | kind | tax | best unconstrained | best projected (c=0) | gap | r_leak P(V=U) | advantage leak |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.55 | 0.7 | primary | 0.15 | 0.7 | 0.55 | 0.15 | 0.52 | 0 |
| 0.6 | 0.75 | primary | 0.15 | 0.75 | 0.6 | 0.15 | 0.55 | 0 |
| 0.6 | 0.9 | primary | 0.3 | 0.9 | 0.6 | 0.3 | 0.58 | 0 |
| 0.75 | 0.9 | primary | 0.15 | 0.9 | 0.75 | 0.15 | 0.7 | 0 |
| 0.75 | 0.6 | control | 0 | 0.75 | 0.75 | 0 | 0.55 | 0 |
| 0.7 | 0.7 | control | 0 | 0.7 | 0.7 | 0 | 0.58 | 0 |

Sample projection detail (beta=0.6, rho=0.9): every do(U)-invariance projection lands at c=0 with competence <= beta; only canonical V-only attains beta.

| base policy | projection | competence | c | <= beta |
| --- | --- | ---: | ---: | --- |
| proxy_user | mask0 | 0.5 | 0 | true |
| proxy_user | mask1 | 0.5 | 0 | true |
| proxy_user | avg | 0.5 | 0 | true |
| xor | mask0 | 0.4 | 0 | true |
| xor | mask1 | 0.6 | 0 | true |
| xor | avg | 0.5 | 0 | true |
| mix_0.5 | mask0 | 0.55 | 0 | true |
| mix_0.5 | mask1 | 0.55 | 0 | true |
| mix_0.5 | avg | 0.55 | 0 | true |

-> **S2_COLLAPSE_VERIFIED**, **S2_NO_EXOGENOUS_TARGET_CAP**

## S2.3 -- Reconstruction leak (named-variable vs content-level)

The Bayes reconstruction courtier (`Uhat = V` since `P(U=V) > 1/2`) has named-variable `c(U) = 0` -- court-honored -- while its behavior agrees with the full courter at `r_leak = P(V=U)` (large, rising with `beta*rho`). **But it is behaviorally identical to the honest V-user**: the courting *advantage* of reconstruction over honesty is exactly `0`. The same data-processing bound that forces the S2.1 collapse also **ceilings the leak** -- you cannot reconstruct more courting-power than your unmasked channels carry. So the named-variable audit *under-describes* (it cannot distinguish honest use from reconstructive courting -- there is no behavioral difference to detect) but does not *under-protect* here: at `c(U)=0` the incentive gradient toward `U` is genuinely absent. The scary version of the leak -- reconstruction that gains real courting advantage -- would require recovering `U`'s unique content from channels that do not carry it, which the bound forbids.

-> **S2_NAMED_VARIABLE_CAP_LEAKY_BUT_CEILINGED**

## S2.4 -- Court-vs-tax frontier (beta=0.6, rho=0.9 primary; beta=0.75, rho=0.6 control)

| w (=c) | competence | R(c) | Total |
| ---: | ---: | ---: | ---: |
| 0 | 0.6 | 1 | 0.6 |
| 0.1 | 0.63 | 1 | 0.63 |
| 0.2 | 0.66 | 1 | 0.66 |
| 0.24 | 0.672 | 0.999645 | 0.671762 |
| 0.25 | 0.675 | 0.484667 | 0.32715 |
| 0.3 | 0.69 | 0 | 0 |
| 0.5 | 0.75 | 0 | 0 |
| 1 | 0.9 | 0 | 0 |

Primary: optimum at w* = **0.24** (just under the cliff c*=0.25; B3.2 cliff-edge coherence CONFIRMED), Total(w*) = 0.671762 vs masking Total = 0.6; at w*+0.03 Total collapses to 0 (fragility). Control (zero-tax): w* = **0** -- un-targeting is OPTIMAL when the proxy has no reliability edge; the cliff-edge temptation exists only when it does.

-> **S2_FRONTIER_MAPPED**

## Verdict

**S2_TARGET_CAP_COLLAPSES_TO_MASK**

In-channel target enforcement collapses to measure masking (competence <= beta, only V-only attains it); the rho-beta gap is unreachable at c=0 by any exogenous projection; the reconstruction leak is real as *correlation* but ceilinged at zero *advantage* by the same information bound; and the reward-optimum courts to the cliff edge exactly when the proxy carries a reliability edge. **One law does all the work at the target channel: you cannot use information you do not causally depend on.** Enforcement (the collapse), evasion (the leak ceiling), and temptation (the frontier) are three faces of the same data-processing bound. Feeds S1: the target safe point is exogenously reachable only by priced channel-retreat; the price is the proxy's reliability edge; and named-variable audits are complete for *incentive* safety in this model even though they cannot express content-level distinctions.

## Honest boundary

Collapse verified: every do(U)-invariance projection lands at competence <= beta; only the canonical V-only policy attains beta; the rho-beta gap is unreachable at c=0 (data processing). RECONSTRUCTION LEAK, resolved honestly: the Bayes reconstruction courtier (Uhat=V) is BEHAVIORALLY IDENTICAL to the honest V-user -- named-variable c(U)=0 admits behavior correlated with the proxy up to P(V=U) (correlation-leak, large), but the courting ADVANTAGE of reconstruction over honesty is ZERO: the same data-processing bound that forces the collapse also ceilings the leak. You cannot reconstruct more courting-power than your unmasked channels carry. So the named-variable audit under-describes (cannot tell honest from reconstructive) but does not under-protect in this model (harm defined causally = both safe; the incentive gradient toward U is genuinely absent at c=0). Frontier: optimum courts to just under the cliff (w*~0.245, B3.2 coherence confirmed); un-targeting/masking is safe-not-optimal on primary cells and OPTIMAL on zero-tax controls (cliff-edge temptation exists only when the proxy carries a reliability edge). Feeds S1: enforcement, evasion, and reconstruction at the target channel are all governed by ONE information bound.

