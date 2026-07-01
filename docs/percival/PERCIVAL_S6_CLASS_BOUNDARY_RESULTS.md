# Percival S6 -- Class Boundary of Counterproductivity (results)

Generated 2026-07-01T23:06:45.893Z by `scripts/percival-s6-class-boundary.mjs`.

Two-factor family V(q) = E_tail_q[f_s] * g(p(q)); base uniform [0,0.4]; f_s(c)=1+s(c-0.2); pressure p(q)=0.4-0.2q crosses c*=0.25 at q=0.75. Predictions P1-P5 pre-registered in the script header.

## The (s, g) grid

| s (pointwise sign) | g (tax) | maximizer q set | q=1 attains max | V* | V(q=1) | un-targeting V | un-targeting dominated |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| -1 | const | {1} | true | 1 | 1 | 1.2 | false |
| -1 | smooth | {1} | true | 0.924142 | 0.924142 | 1.199996 | false |
| -1 | cliff | {1} | true | 1 | 1 | 1.2 | false |
| -0.5 | const | {1} | true | 1 | 1 | 1.1 | false |
| -0.5 | smooth | {1} | true | 0.924142 | 0.924142 | 1.099996 | false |
| -0.5 | cliff | {1} | true | 1 | 1 | 1.1 | false |
| 0 | const | {0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1} | true | 1 | 1 | 1 | false |
| 0 | smooth | {1} | true | 0.924142 | 0.924142 | 0.999996 | false |
| 0 | cliff | {0.8,0.9,1} | true | 1 | 1 | 1 | false |
| 0.5 | const | {0.05} | false | 1.095 | 1 | 0.9 | true |
| 0.5 | smooth | {1} | true | 0.924142 | 0.924142 | 0.899997 | true |
| 0.5 | cliff | {0.8} | false | 1.02 | 1 | 0.9 | true |
| 1 | const | {0.05} | false | 1.19 | 1 | 0.8 | true |
| 1 | smooth | {1} | true | 0.924142 | 0.924142 | 0.799997 | true |
| 1 | cliff | {0.8} | false | 1.04 | 1 | 0.8 | true |

## Prediction checks

- **P1** (s<=0: q*=1 for ALL g (counterproductivity; S4 extends multiplicatively)): **CONFIRMED**
- **P2** (s>0, no tax: max tilt best (Goodhart-free control)): **CONFIRMED**
- **P3** (s>0, cliff: optimum at cliff edge, STRICTLY beats q=1 -- counterproductivity fails on this half): **CONFIRMED**
- **P4** (s>0, smooth tax: interior optimum): **MISS**
- **P5** (un-targeting dominated iff s>0): **CONFIRMED**

## Post-hoc: tax-shape width sweep (labeled, NOT pre-registered; s=1 pointwise-good proxy)

| logistic width w | maximizer q set | regime |
| ---: | --- | --- |
| 0.001 | {0.8} | intermediate optimum (near-threshold backoff / interior continuum) |
| 0.005 | {0.9} | intermediate optimum (near-threshold backoff / interior continuum) |
| 0.02 | {1} | protective (q*=1) |
| 0.1 | {1} | protective (q*=1) |
| 0.5 | {0.9} | intermediate optimum (near-threshold backoff / interior continuum) |
| 2 | {0.05} | gain-dominant (max tilt) |
| 10 | {0.05} | gain-dominant (max tilt) |

## Verdict

**S6_POINTWISE_BOUNDARY_TAX_SHAPE_REFINES**

Registered core CONFIRMED (P1/P2/P3/P5), **P4 MISSED and the miss is the finding**: the class boundary of counterproductivity is the **pointwise proxy-truth sign** -- for a pointwise-misaligned proxy, quantilizing is counterproductive under every monotone tax including none (exactly the S4 condition) -- but the pointwise-good half splits by **tax shape**: the cliff's edge-temptation comes from its flat zero-marginal region (tilting under the cliff is free), while a smooth tax charges every tilt and, when its marginal rate dominates the pointwise gain, **restores q*=1 even for a good proxy**. Interior optima are knife-edges between regimes, which is why the registered 'generic interior' P4 missed. Quotable: **quantilizing is counterproductive iff the proxy is pointwise misaligned OR the oversight tax is graded and steep; the cliff-edge temptation is a threshold-oversight artifact -- graded oversight removes it.** (Resonates with NS-3's graded review: the same knob, rediscovered from the theory side.)

## Honest boundary

Registered predictions: P1/P2/P3/P5 CONFIRMED, P4 MISSED (kept on the record). The class boundary of counterproductivity IS the pointwise proxy-truth sign: for a pointwise-bad-or-neutral proxy (s<=0), quantilizing is counterproductive under every monotone tax including none (the S4 condition, extended multiplicatively). But the s>0 half splits by TAX SHAPE, not as P4 predicted: the cliff's edge-temptation comes from its FLAT, ZERO-MARGINAL region below threshold (tilting under the cliff is free), while a smooth tax charges every tilt at the margin -- and when that marginal tax dominates the pointwise gain, q*=1 is RESTORED even for a pointwise-good proxy. Post-hoc width sweep (labeled, not pre-registered): w->0 recovers near-threshold optima (the edge temptation, with a backoff margin scaling in w), an intermediate-w band is PROTECTIVE (q*=1 restored -- (1-g(base))/w > marginal gain), a narrow band around w~0.5 gives the genuinely INTERIOR optimum P4 predicted (it exists, but as a knife-edge, not generically), and large w is gain-dominant (max tilt). Policy-shaped consequence, resonant with NS-3's graded review: THRESHOLD oversight creates the cliff-edge temptation; GRADED oversight with sufficient marginal steepness removes it -- the optimal agent under a graded tax does not tilt at all. Un-targeting is strictly dominated exactly on the s>0 half: the target channel's danger is conditional on Goodhart (pointwise misalignment), which the court satisfied by construction.

