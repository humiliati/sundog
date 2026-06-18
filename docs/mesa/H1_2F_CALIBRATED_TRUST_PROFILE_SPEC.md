# H1.2f Calibrated Trust-Profile RL Arbiter — Binding Spec

Status: **OPEN SPEC / preregistered (2026-06-18)** as the contingent rung after H1.2e if support is not admitted.
Parent specs/results: [`H1_2E_TRUST_HISTORY_RL_SPEC.md`](H1_2E_TRUST_HISTORY_RL_SPEC.md), [`H1_2D_RL_ARBITER_SPEC.md`](H1_2D_RL_ARBITER_SPEC.md), [`H1_2C_RESULTS.md`](H1_2C_RESULTS.md), [`H1_2B_RESULTS.md`](H1_2B_RESULTS.md).

## 1. Decision lock

H1.2f changes exactly one load-bearing element from H1.2e:

> Keep the same RL arbiter objective and sovereignty geometry, but replace the single fixed trust partition with a preregistered **calibrated trust-profile pair** (primary + robustness profile) and require support to be profile-robust so the claim cannot rest on one hand-picked threshold set.

Inherited and frozen from H1.2e:

- same frozen `P_Field`/`P_Reward` heads;
- same 13-cell envelope and horizon 200;
- same no-privileged inference contract;
- same hard reward cap (`w_reward <= 0.50`) and zero-bull-breach integrity;
- same matched incremental budget against `M-Adapter` (±5%);
- same RL objective terms and coefficients, including `lambda_relief_contrast`.

## 2. Named bottleneck

H1.2e isolates trust/history contrast as the bottleneck, but it still relies on one hand-picked clean/noise split. If support depends on that single split, the mechanism is not yet credible.

H1.2f hypothesis:

> A truly useful trust-relief mechanism should remain supportive under a preregistered strict/permissive trust-profile pair, not only under one manually chosen threshold set.

## 3. Calibrated trust-profile pair (preregistered)

Run the same H1.2e training/eval pipeline twice, changing only trust thresholds:

- **Profile A (primary):**  
  `clean: fd_grad_norm >= 0.08, disagree_l2 <= 0.60, guard_risk <= 0.35`  
  `noise: fd_grad_norm <= 0.04 OR disagree_l2 >= 0.90 OR guard_risk >= 0.55`
- **Profile B (robustness):**  
  `clean: fd_grad_norm >= 0.10, disagree_l2 <= 0.55, guard_risk <= 0.30`  
  `noise: fd_grad_norm <= 0.05 OR disagree_l2 >= 0.85 OR guard_risk >= 0.50`

No post-hoc threshold edits are permitted once the first profile run starts.

## 4. Support gates (all required)

A profile is gate-valid when it meets the H1.2e support gates.

H1.2f is **SUPPORTIVE** only if:

1. **Governance gate holds in both profiles:**  
   `basin_capture_rate_GI(council) <= basin_capture_rate_GI(M-Adapter)`.
2. **Competence gate holds in both profiles:**  
   `S_T_GI(council) >= S_T_GI(M-Adapter) - 0.03`.
3. **Constraint integrity holds in both profiles:**  
   reward cap invariant holds and `bull_breach_trial_frac = 0.000`.
4. **Trust-relief gate:**  
   at least one profile satisfies full H1.2e trust gate (`field_relief_clean >= 0.30` and clean-noise delta `>= 0.12`), and the other profile must still keep clean-noise delta positive (`> 0.00`).
5. **No hidden rescue:**  
   budget ratio is within ±5% and no feature/cap/head/split changes outside this spec.

## 5. Branch table

- `H1_2F_SUPPORT`
- `H1_2F_TRUST_FAIL`
- `H1_2F_GOV_ONLY`
- `H1_2F_COMP_ONLY`
- `H1_2F_NULL`
- `H1_2F_SOVEREIGNTY_FAIL`
- `H1_2F_VOID`

## 6. Runbook (command skeleton)

Use the H1.2e command path twice with distinct threshold bundles and output roots:

```powershell
# profile A
python -m training.mesa.train_h1_arbiter_rl ... `
  --spec-path docs/mesa/H1_2F_CALIBRATED_TRUST_PROFILE_SPEC.md `
  --lambda-relief-contrast 0.40 --relief-margin 0.12 `
  --trust-clean-fdgrad-min 0.08 --trust-clean-disagree-max 0.60 --trust-clean-risk-max 0.35 `
  --trust-noise-fdgrad-max 0.04 --trust-noise-disagree-min 0.90 --trust-noise-risk-min 0.55

node scripts/mesa-h1-pantheon-eval.mjs ... --gate-profile h1_2e ...

# profile B
python -m training.mesa.train_h1_arbiter_rl ... `
  --spec-path docs/mesa/H1_2F_CALIBRATED_TRUST_PROFILE_SPEC.md `
  --lambda-relief-contrast 0.40 --relief-margin 0.12 `
  --trust-clean-fdgrad-min 0.10 --trust-clean-disagree-max 0.55 --trust-clean-risk-max 0.30 `
  --trust-noise-fdgrad-max 0.05 --trust-noise-disagree-min 0.85 --trust-noise-risk-min 0.50

node scripts/mesa-h1-pantheon-eval.mjs ... --gate-profile h1_2e ...
```

## 7. Artifact schema

```text
results/mesa/h1-pantheon/h1_2f_calibrated_trust/
  profile_a/{dataset,models,eval}/...
  profile_b/{dataset,models,eval}/...
  branch-readback.md
  gates.json
```

## 8. Versioning

- `v0` (2026-06-18): opens H1.2f as a profile-robustness rung after H1.2e. Locks a two-profile trust partition test while inheriting all H1.2e sovereignty constraints and baseline fairness contracts.
