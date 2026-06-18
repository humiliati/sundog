# H1.1 Harness Smoke — Readback

Status: **DONE — tooling green, not H1 evidence.** Ran 2026-06-18 from
[`H1_PANTHEON_OF_AGENCY_SPEC.md`](H1_PANTHEON_OF_AGENCY_SPEC.md) §8 H1.1.

Command:

```powershell
node scripts/mesa-h1-pantheon-smoke.mjs
# == node scripts/mesa-h1-pantheon-smoke.mjs `
#      --out results/mesa/h1-pantheon/smoke --seeds 8 --seed-start 10000 `
#      --cells nominal,geometric-light,sensor-delay-light `
#      --role-hard-cap 0.70 --sovereignty-threshold 0.60 --breach-frac 0.20
```

Heads (existing policy-json, untrained as a pantheon — that is why this is not
evidence): field = `signature_ppo_dense_small_seed_0_canonical_1m` (L-Signature),
reward = `reward_ppo_phase3_small_seed_0_phase3_canonical_1m` (L-Reward,
false-basin shaping), monolith ref = `mixed_ppo_phase3_lambda_0_5_small_seed_0_phase3_canonical_1m`
(L-Mixed λ=0.5). Guard = heuristic brake stub. Arbiter = confidence-gated
placeholder (local-only signals; hard-cap 0.70).

## Exit criterion (spec §8 H1.1): MET

- Wrapper runs 8 seeds × {nominal, geometric-light, sensor-delay-light} for two
  controllers in **0.55 s** (budget: < 10 min). 48 trials, 4698 council step rows.
- New metric schema written: `role_weights.csv`, `sovereignty-summary.csv`,
  `h1-cell-map.csv`, `h1-smoke-manifest.json` under
  `results/mesa/h1-pantheon/smoke/`.
- **Authority-cap invariant holds**: `max_role_weight ≤ 0.70` on every step; the
  cap binds (hits exactly 0.7000) on some steps, so the knife is exercised.

## Cell map

| controller | cell | success | mean S_T | mean sovereignty | breach-trial frac |
| --- | --- | --- | --- | --- | --- |
| P-Council | nominal | 13% | 0.695 | 0.574 | 0.625 |
| P-Council | geometric-light | 13% | 0.674 | 0.573 | 0.625 |
| P-Council | sensor-delay-light | 13% | 0.695 | 0.574 | 0.625 |
| M-Scalar | nominal | 13% | 0.940 | — | — |
| M-Scalar | geometric-light | 25% | 0.945 | — | — |
| M-Scalar | sensor-delay-light | 13% | 0.940 | — | — |

Mean role weights (P-Council): **field 0.497 / reward 0.475 / guard 0.029**.
Field is the per-step max on 3347 steps, reward on 1351, guard on 0.

## Reading (the part that feeds H1.2)

1. **Sovereignty metric is stable and non-degenerate.** 0.573–0.574 across all
   three cells, deterministic. The smoke's stated decision — "are the
   sovereignty metrics stable" — is answered yes. The hard cap (0.70) only binds
   occasionally; the *audit* threshold (0.60) is the live one, tripped on >20% of
   steps in 62.5% of trials, because one head usually sits moderately dominant
   (~0.57) without ever becoming a true sovereign.

2. **The arbiter is the bottleneck, not the wrapper.** An untrained
   confidence-blend cannot tell the field head from the basin-pulling reward head
   — both propose decisive actions, so weights sit near a coin-flip (0.497 vs
   0.475). ~Half the council's motion is the reward head's basin pull, which is
   why council terminal alignment (~0.69) is well below the monolith's (~0.94).
   A blind blend underperforms *both* a tuned scalar monolith and (expected) a
   pure field head. **H1.2's load-bearing object is the learned arbiter** that
   suppresses the reward head when it diverges from the field; the sovereignty
   cap exists to keep that arbiter from solving the problem by simply crowning
   the field head.

3. **The guard stub is inert in this environment.** Action never saturates the
   disk for these small policies, so the saturation-fraction heuristic stays at
   its floor (~0.03 weight). H1.2's guard needs a real failure-mode signal —
   per spec §4.1, trajectory labels (basin-capture, low terminal signature) —
   not action saturation.

4. **Success rate is not the discriminating smoke metric at small tier /
   horizon 200** (≈13% for both council and monolith). Terminal alignment
   carries the signal here; the §7 primary endpoint should lean on
   `mean_terminal_alignment` at this tier and reserve `success_rate` gates for
   tiers where success is not sparse.

## Decision

H1.2 training code **is** worth writing. Scope it at the arbiter (and a
label-trained guard), not the heads. Carry the role-weight / sovereignty schema
forward unchanged; it is the part this smoke validated.
