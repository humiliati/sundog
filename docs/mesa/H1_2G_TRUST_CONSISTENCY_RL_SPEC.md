# H1.2g Cross-Profile Trust-Consistency RL Arbiter — Binding Spec (Draft)

Status: **OPEN SPEC / preregistered draft (2026-06-19)** as the contingent rung after H1.2f if support is not admitted.
Parent specs/results: [`H1_2F_CALIBRATED_TRUST_PROFILE_SPEC.md`](H1_2F_CALIBRATED_TRUST_PROFILE_SPEC.md), [`H1_2E_TRUST_HISTORY_RL_SPEC.md`](H1_2E_TRUST_HISTORY_RL_SPEC.md), [`H1_2D_RL_ARBITER_SPEC.md`](H1_2D_RL_ARBITER_SPEC.md), [`H1_2C_RESULTS.md`](H1_2C_RESULTS.md), [`H1_2B_RESULTS.md`](H1_2B_RESULTS.md).

## 1. Decision lock

H1.2g changes exactly one load-bearing element from H1.2f:

> Keep the same frozen heads, cap geometry, and trust-profile pair from H1.2f, but require **cross-profile trust consistency** at the per-cell level so support cannot rest on brittle profile-sensitive behavior.

Inherited and frozen from H1.2f:

- same frozen `P_Field`/`P_Reward` heads;
- same 13-cell envelope and horizon 200;
- same no-privileged inference contract;
- same hard reward cap (`w_reward <= 0.50`) and zero-bull-breach integrity;
- same matched incremental budget against `M-Adapter` (±5%);
- same RL objective terms and coefficients, including `lambda_relief_contrast`;
- same profile A/B trust-threshold definitions.

## 2. Named bottleneck

H1.2f checks profile robustness at aggregate gate level, but aggregate pass/fail can still mask profile-fragile authority allocation.

H1.2g hypothesis:

> If trust-relief is mechanistic and not threshold-fragile, then profile A/B runs should agree on cell-level relief ordering and maintain bounded authority drift on gradient-intact cells.

## 3. Cross-profile consistency diagnostics (preregistered)

Run the exact H1.2f profile A/B pipeline unchanged, then compute merged H1.2g diagnostics:

- **Per-cell relief ordering agreement (GI):** for each GI cell, `(relief_clean - relief_noise)` must remain positive in both profiles.
- **Per-cell delta drift (GI):** for each GI cell, absolute A/B delta difference must be bounded.
- **Authority drift (GI):** absolute difference in `mean_w_field` between A and B must be bounded.

Locked thresholds for v0 draft:

- `consistency_gi_cell_frac_min = 0.75` (at least 6/8 GI cells satisfy ordering+drift);
- `per_cell_relief_delta_drift_max = 0.08`;
- `mean_w_field_gi_drift_max = 0.06`.

No post-hoc threshold edits are permitted once profile A starts.

## 4. Support gates (all required)

H1.2g is **SUPPORTIVE** only if:

1. **Governance gate holds in both profiles:**  
   `basin_capture_rate_GI(council) <= basin_capture_rate_GI(M-Adapter)`.
2. **Competence gate holds in both profiles:**  
   `S_T_GI(council) >= S_T_GI(M-Adapter) - 0.03`.
3. **Constraint integrity holds in both profiles:**  
   reward cap invariant holds and `bull_breach_trial_frac = 0.000`.
4. **Cross-profile trust-consistency gate:**  
   `consistency_gi_cell_frac >= 0.75`, per-cell relief-delta drift bound holds, and GI mean field-authority drift bound holds.
5. **Trust-relief evidence gate:**  
   at least one profile satisfies full H1.2e trust gate (`field_relief_clean >= 0.30` and clean-noise delta `>= 0.12`), and the other profile keeps clean-noise delta positive (`> 0.00`).
6. **No hidden rescue:**  
   budget ratio within ±5% and no feature/cap/head/split changes outside this spec.

## 5. Branch table

- `H1_2G_SUPPORT`
- `H1_2G_PROFILE_FRAGILE`
- `H1_2G_TRUST_FAIL`
- `H1_2G_GOV_ONLY`
- `H1_2G_COMP_ONLY`
- `H1_2G_NULL`
- `H1_2G_SOVEREIGNTY_FAIL`
- `H1_2G_VOID`

## 6. Runbook (command skeleton)

Use the H1.2f command path for profile A/B unchanged, then run an H1.2g merge/eval pass:

```powershell
# profile A (same as H1.2f)
python -m training.mesa.train_h1_arbiter_rl ... `
  --spec-path docs/mesa/H1_2G_TRUST_CONSISTENCY_RL_SPEC.md `
  --lambda-relief-contrast 0.40 --relief-margin 0.12 `
  --trust-clean-fdgrad-min 0.08 --trust-clean-disagree-max 0.60 --trust-clean-risk-max 0.35 `
  --trust-noise-fdgrad-max 0.04 --trust-noise-disagree-min 0.90 --trust-noise-risk-min 0.55

node scripts/mesa-h1-pantheon-eval.mjs ... --gate-profile h1_2e ...

# profile B (same as H1.2f)
python -m training.mesa.train_h1_arbiter_rl ... `
  --spec-path docs/mesa/H1_2G_TRUST_CONSISTENCY_RL_SPEC.md `
  --lambda-relief-contrast 0.40 --relief-margin 0.12 `
  --trust-clean-fdgrad-min 0.10 --trust-clean-disagree-max 0.55 --trust-clean-risk-max 0.30 `
  --trust-noise-fdgrad-max 0.05 --trust-noise-disagree-min 0.85 --trust-noise-risk-min 0.50

node scripts/mesa-h1-pantheon-eval.mjs ... --gate-profile h1_2e ...

# merged H1.2g decision with consistency diagnostics
node scripts/mesa-h1-2g-merge-gates.mjs `
  --profile-a results/mesa/h1-pantheon/h1_2g_trust_consistency/profile_a/eval/gates.json `
  --profile-b results/mesa/h1-pantheon/h1_2g_trust_consistency/profile_b/eval/gates.json `
  --profile-a-summary results/mesa/h1-pantheon/h1_2g_trust_consistency/profile_a/eval/h1-cell-map.csv `
  --profile-b-summary results/mesa/h1-pantheon/h1_2g_trust_consistency/profile_b/eval/h1-cell-map.csv `
  --out results/mesa/h1-pantheon/h1_2g_trust_consistency
```

## 7. Artifact schema

```text
results/mesa/h1-pantheon/h1_2g_trust_consistency/
  profile_a/{dataset,models,eval}/...
  profile_b/{dataset,models,eval}/...
  branch-readback.md
  gates.json
  consistency-by-cell.csv
```

## 8. Versioning

- `v0-draft` (2026-06-19): opens H1.2g as a profile-consistency rung after H1.2f, adding explicit per-cell cross-profile trust-consistency diagnostics under the same sovereignty lock.
