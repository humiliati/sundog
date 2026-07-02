# ME-5 -- the Priced Quadrant (sweep receipt)

Generated 2026-07-02T14:11:16.035Z by `scripts/orderrelative-me5-priced-quadrant.mjs` (deterministic seed 20260702).

Definitions: price = value(V,U) - value(V) at the joint's own prior (the collapse price);
edgeFormula = max(value V, value U) - value V (the S2/OR-4 reliability-edge formula);
deltaBayes = sup over priors of the full-vs-masked Bayes-value gap (1e-3 grid).

## Gates

- FLOOR (price >= edgeFormula, the 2-line theorem): **0 violations** over 5111 joints.
- TRIVIAL (price <= deltaBayes): **0 violations**.
- F1 binary-symmetric CI: formula exact to 0; price = rho - beta to 0 (the banked cell reproduced).
- F3 lambda = 1 anchor vs the Lean witness (`PercivalSynergy.lean`: price 1/2, formula 0): **MATCH**.

## Families

| family | n | formula exact | max(price - edge) | mean | price = delta | max(delta - price) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F1_ci_symmetric | 100 | 1 | 0 | 0 | 0.43 | 0.096 |
| F2_ci_asymmetric | 2000 | 0.695 | 0.1073 | 0.007603 | 0.001 | 0.157016 |
| F3_synergy_path | 11 | 0.181818 | 0.5 | 0.174545 | 0.090909 | 0.06464 |
| F4_random | 3000 | 0.372333 | 0.342146 | 0.034164 | 0.002333 | 0.314345 |

## The synergy path (lambda * XOR + (1 - lambda) * CI(0.7, 0.8))

| lambda | price | edgeFormula | deltaBayes |
| ---: | ---: | ---: | ---: |
| 0 | 0.1 | 0.1 | 0.126 |
| 0.1 | 0.09 | 0.09 | 0.14056 |
| 0.2 | 0.09 | 0.08 | 0.15464 |
| 0.3 | 0.11 | 0.07 | 0.16824 |
| 0.4 | 0.13 | 0.06 | 0.18136 |
| 0.5 | 0.15 | 0.05 | 0.194 |
| 0.6 | 0.22 | 0.04 | 0.22768 |
| 0.7 | 0.29 | 0.03 | 0.29432 |
| 0.8 | 0.36 | 0.02 | 0.36192 |
| 0.9 | 0.43 | 0.01 | 0.43048 |
| 1 | 0.5 | 0 | 0.5 |

## Verdict: `ME5_EDGE_FORMULA_IS_BINARY_SYMMETRIC_ARTIFACT + ME5_PRICE_IS_LOCAL_DEFICIENCY`

The reliability-edge FORMULA is exact exactly where it was banked (binary-symmetric CI)
and is a strict FLOOR elsewhere -- asymmetric-CI fusion and synergy joints price above it,
up to the machine-checked XOR maximum. The PRICE itself survives every family as the
local (operative-prior) Bayes gap, bounded by the full deficiency; equality with the
deficiency holds only where the operative decision problem attains the sup -- prices are
per-decision-problem, as sigma is per-filtration. Read-price: 0 in-model by replay
(premise; quantum contrast fenced).

