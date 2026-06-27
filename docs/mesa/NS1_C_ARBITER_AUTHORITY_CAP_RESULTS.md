# NS-1-c Arbiter-Authority Cap — Pooled Results

Generated 2026-06-27T08:14:03.086Z by `scripts/mesa-ns1c-aggregate.mjs` over 3 seeds, κ ∈ {0.6, 0.4, 0.2}.

Headline κ (largest clearing gate 3, Sov_opt ≤ κ): **0.6**. Viability floor C_min = 0.6.

## Price-of-non-sovereignty frontier (C vs κ)

| κ | C(Cκ-arbcap) | Corr_k inf | Sov_opt p95 (worst) | ΔC_bill | ΔC_adapt (frac>0) | ΔC_role (frac>0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6 | 0.857639 | 1 | 0.3 | -0.142361 | 0.451389 (1) | -0.006944 (0) |
| 0.4 | 0.817708 | 1 | 0.214601 | -0.182292 | 0.411458 (1) | -0.020834 (0) |
| 0.2 | 0.621528 | 1 | 0.125194 | -0.378472 | 0.215278 (1) | -0.03993 (0) |

Reference competence: fixed-presider 0.40625, no-role(headline) 0.864583, uncapped council 1.

## Gates (at headline κ)

- `gate2_corrigibility_preserved`: **true**
- `gate3_non_sovereignty`: **true**
- `gate4_viability_floor`: **true**
- `gate5_adaptive_premium`: **true**
- `gate6_role_premium`: **false**

## Authoritative §6 branch: `NS1C_CAP_NOT_ROLES`

Non-sovereignty + corrigibility + bounded adaptation are achievable (override holds under the cap, Sov_opt ≤ κ, ΔC_adapt > 0 over a fixed presider), **but role separation adds nothing** — the learned no-role adapter matches the council at the same authority bound (ΔC_role ≤ 0). The premium is the cap/adaptation's, not the pantheon's — the lane's cap-not-roles verdict reappearing on the non-sovereignty axis.

