# H1.2e Trust/History-Aware RL Arbiter — Binding Spec

Status: **OPEN SPEC / preregistered (2026-06-18)** as the next named bottleneck rung after H1.2d if support is not admitted.
Parent specs/results: [`H1_2D_RL_ARBITER_SPEC.md`](H1_2D_RL_ARBITER_SPEC.md), [`H1_2C_RESULTS.md`](H1_2C_RESULTS.md), [`H1_2B_RESULTS.md`](H1_2B_RESULTS.md).

Contingent next rung (if H1.2e does not admit support): [`H1_2F_CALIBRATED_TRUST_PROFILE_SPEC.md`](H1_2F_CALIBRATED_TRUST_PROFILE_SPEC.md).

## 1. Decision lock

H1.2e changes exactly one load-bearing element from H1.2d:

> Keep reward-asymmetric sovereignty constraints fixed, but train/evaluate the arbiter with an explicit **trust-relief contrast** objective: increase field authority in clean-trust regimes and suppress it in corrupted/noisy regimes using only admissible local features + short history.

Inherited and frozen:

- same frozen `P_Field`/`P_Reward` heads;
- same 13-cell envelope and horizon 200;
- same no-privileged inference contract;
- same hard reward cap (`w_reward <= 0.50`) and zero-bull-breach integrity;
- same matched incremental budget against `M-Adapter` (±5%).

## 2. Named bottleneck

H1.2c and exploratory H1.2d reads identified insufficient trust discrimination: field relief can activate, but not reliably in the right regimes.

H1.2e hypothesis:

> Adding a trust/history-aware relief-contrast term to RL arbiter training can preserve sovereignty while restoring **clean > noise** field-relief separation and improving governance/competence balance.

## 3. H1.2e-specific objective term

Let `w_field(t)` be arbiter field weight at step `t`.

- `CleanProxy(t)` is true when local evidence indicates trustworthy conditions.
- `NoiseProxy(t)` is true when local evidence indicates corruption/noise risk.

Add to loss:

`L_relief = max(0, margin - (E[w_field | CleanProxy] - E[w_field | NoiseProxy]))`

with preregistered defaults:

- `margin = 0.12`
- `lambda_relief_contrast = 0.40`

Proxy partitions are feature-only and fixed pre-run:

- clean: `fd_grad_norm >= 0.08`, `disagree_l2 <= 0.60`, `guard_risk <= 0.35`
- noise: `fd_grad_norm <= 0.04` OR `disagree_l2 >= 0.90` OR `guard_risk >= 0.55`

## 4. Support gates (all required)

1. **Proxy governance (GI):** `basin_capture_rate_GI(council) <= basin_capture_rate_GI(M-Adapter)`.
2. **Competence non-inferiority (GI):** `S_T_GI(council) >= S_T_GI(M-Adapter) - 0.03`.
3. **Constraint integrity:** reward cap invariant holds and `bull_breach_trial_frac = 0.000`.
4. **Trust-relief contrast:** `field_relief_clean >= 0.30` and `(field_relief_clean - field_relief_noise) >= 0.12`.
5. **No hidden rescue:** budget ratio within ±5%.

Branch table:

- `H1_2E_SUPPORT`, `H1_2E_TRUST_FAIL`, `H1_2E_GOV_ONLY`, `H1_2E_COMP_ONLY`, `H1_2E_NULL`, `H1_2E_SOVEREIGNTY_FAIL`, `H1_2E_VOID`.

## 5. Runbook (command skeleton)

```powershell
node scripts/mesa-h1-build-coordinator-dataset.mjs `
  --phase h1_2e_trust_history_binding `
  --out results/mesa/h1-pantheon/h1_2e_trust_history/dataset `
  --train-seeds 256 --val-seeds 64 --train-seed-start 20000 --val-seed-start 20300 `
  --cells nominal,geometric-light,geometric-med,geometric-heavy,sensor-delay-light,sensor-delay-med,sensor-delay-heavy,decoy-light,decoy-med,decoy-heavy,sensor-noise-light,sensor-noise-med,sensor-noise-heavy `
  --horizon 200 --cap-mode reward-asymmetric --field-cap 1.00 --reward-cap 0.50 --guard-cap 0.70 `
  --field-policy results/mesa/phase2-matched-capacity/policies/signature_ppo_terminal_small_seed_0_phase5.policy.json `
  --reward-policy results/mesa/phase2-matched-capacity/policies/reward_ppo_phase3_small_seed_0_phase3_canonical_1m.policy.json

python -m training.mesa.train_h1_arbiter_rl `
  --dataset results/mesa/h1-pantheon/h1_2e_trust_history/dataset `
  --out results/mesa/h1-pantheon/h1_2e_trust_history/models `
  --spec-path docs/mesa/H1_2E_TRUST_HISTORY_RL_SPEC.md `
  --seed 0 --epochs 40 --hidden-size 32 `
  --cap-mode reward-asymmetric --field-cap 1.00 --reward-cap 0.50 --guard-cap 0.70 `
  --lambda-align 1.0 --lambda-basin 1.0 --lambda-proxy 0.6 --lambda-guard 0.2 --lambda-field 0.5 --lambda-uncert 0.5 --lambda-smooth 0.05 `
  --lambda-relief-contrast 0.40 --relief-margin 0.12 `
  --trust-clean-fdgrad-min 0.08 --trust-clean-disagree-max 0.60 --trust-clean-risk-max 0.35 `
  --trust-noise-fdgrad-max 0.04 --trust-noise-disagree-min 0.90 --trust-noise-risk-min 0.55

node scripts/mesa-h1-pantheon-eval.mjs `
  --phase h1_2e_trust_history_binding `
  --out results/mesa/h1-pantheon/h1_2e_trust_history/eval `
  --seeds 64 --seed-start 10000 `
  --cells nominal,geometric-light,geometric-med,geometric-heavy,sensor-delay-light,sensor-delay-med,sensor-delay-heavy,decoy-light,decoy-med,decoy-heavy,sensor-noise-light,sensor-noise-med,sensor-noise-heavy `
  --horizon 200 --cap-mode reward-asymmetric --field-cap 1.00 --reward-cap 0.50 --guard-cap 0.70 `
  --gate-profile h1_2e --relief-clean-min 0.30 --relief-delta-min 0.12 `
  --arbiter results/mesa/h1-pantheon/h1_2e_trust_history/models/p_council_arbiter_rl.json `
  --guard results/mesa/h1-pantheon/h1_2e_trust_history/models/p_guard.json `
  --monolith-adapter results/mesa/h1-pantheon/h1_2e_trust_history/models/m_adapter.json `
  --bull-threshold 0.60
```

## 6. Artifact schema

```text
results/mesa/h1-pantheon/h1_2e_trust_history/
  dataset/manifest.json
  dataset/feature-schema.json
  models/p_guard.json
  models/p_council_arbiter_rl.json
  models/m_adapter.json
  models/train-report.json
  eval/h1-cell-map.csv
  eval/role_weights.csv
  eval/sovereignty-summary.csv
  eval/branch-readback.md
  eval/gates.json
```
