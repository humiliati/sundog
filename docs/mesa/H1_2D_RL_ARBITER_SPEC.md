# H1.2d RL-Trained Arbiter — Binding Spec

Status: **OPEN SPEC / preregistered (2026-06-18)** after two binding nulls:
[`H1_2B_RESULTS.md`](H1_2B_RESULTS.md) and [`H1_2C_RESULTS.md`](H1_2C_RESULTS.md).
Parent specs: [`H1_2_SMALL_BAKEOFF_SPEC.md`](H1_2_SMALL_BAKEOFF_SPEC.md),
[`H1_2C_REWARD_ASYMMETRIC_CAP_SPEC.md`](H1_2C_REWARD_ASYMMETRIC_CAP_SPEC.md).

H1.2c established the named bottleneck: reward-bounded field relief exists, but
the **supervised** arbiter does not use it reliably enough to recover competence.
H1.2d tests whether an **RL-trained** arbiter can use that relief while keeping
the bull/reward head structurally non-sovereign.

---

## 1. Decision Lock

H1.2d changes exactly one load-bearing element from H1.2c:

> Replace supervised privileged-target arbiter training with RL training of the
> arbiter policy under frozen heads and fixed reward-asymmetric cap geometry.

Inherited and frozen:

- same frozen head policies (`P_Field`, `P_Reward`);
- same guard model architecture and risk signal interface;
- same 13-cell envelope and horizon 200 unless versioned below;
- same feature leakage contract (no privileged outcomes at inference);
- same equal-incremental-budget `M-Adapter` baseline family;
- same reward-bound structural cap target: `w_reward <= 0.50`.

Out of scope:

- changing frozen heads;
- adding privileged inference features;
- changing envelope composition without version bump;
- post-hoc threshold edits after first binding run starts.

---

## 2. Hypothesis

Working diagnosis from H1.2c:

- cap relief is structurally available;
- supervised imitation target induces global hedging;
- local trust discrimination is insufficiently exploited.

H1.2d tests:

> An RL-trained arbiter, optimized directly on rollout outcomes with explicit
> governance and utilization terms, can improve governance and recover part of
> monolith competence while preserving bull-bound constraints.

---

## 3. Arbiter Interface Lock

### 3.1 State (observability)

Arbiter input remains strictly local and non-privileged:

- current observation channels used in H1.2b/c;
- `P_Field` and `P_Reward` proposed actions;
- proposal disagreement features (L2 / cosine / norm);
- finite-difference local field-strength features already admitted in H1.2;
- short action/weight history already admitted in H1.2;
- guard risk score and calibration bucket.

No added privileged channels are permitted in H1.2d-v0.

### 3.2 Action space

Arbiter emits 3 logits `[field, reward, guard]` each step.

Logits are projected with the **reward-asymmetric capped simplex**:

- `field <= 1.00`;
- `reward <= 0.50` (hard);
- `guard <= 0.70`;
- sum exactly `1.00`.

### 3.3 Behavior policy

On-policy stochastic training with categorical temperature on role logits.
Evaluation uses deterministic mean action (temperature off) with identical cap
projection.

---

## 4. RL Objective Composition (Preregistered)

Let `r_t` be per-step arbiter reward and `J = E[sum_t gamma^t r_t]`.

```text
r_t =
  + λ_align * Δterminal_alignment_proxy_t
  - λ_basin * basin_capture_risk_t
  - λ_proxy * reward_dominance_under_disagreement_t
  - λ_guard * excessive_guard_brake_t
  + λ_field * trustworthy_field_utilization_t
  - λ_uncert * overconfident_field_use_under_noise_t
  - λ_smooth * role_weight_churn_t
```

Locked objectives:

1. **Competence term** (`λ_align`) improves terminal alignment trajectory.
2. **Governance terms** (`λ_basin`, `λ_proxy`) penalize proxy-capture behavior.
3. **Constraint-aware utilization term** (`λ_field`) rewards high field authority
   when trust diagnostics indicate clean conditions.
4. **Uncertainty term** (`λ_uncert`) penalizes aggressive field authority in
   known corrupted regimes.
5. **Stability term** (`λ_smooth`) discourages high-frequency authority flapping.

`λ_*` are preregistered before binding run; no after-the-fact tuning.

---

## 5. Constraint Enforcement and Safety

Hard invariants:

1. Reward authority cap is structural (`w_reward <= 0.50`) in both train and eval.
2. Any step cap violation aborts run classification to `H1_2D_VOID`.
3. `bull_breach_trial_frac` must remain `0.000` for support.

Audit hooks (must be written to manifests/logs):

- cap mode and role caps;
- per-step projected role weights;
- pre-projection logits;
- violation counter;
- trial-level bull-breach tags.

No support claim is allowed if support depends on relaxing reward cap.

---

## 6. Training Curriculum

Three locked stages:

1. **Clean-first warm start**  
   nominal + geometric-light + sensor-delay-light, low corruption.
2. **Mixed-envelope curriculum**  
   progressively include all 13 cells with balanced sampling.
3. **Adversarial robustness pass**  
   oversample decoy-heavy and sensor-noise cells to suppress over-hedging and
   capture regressions.

Promotion between stages requires:

- no cap violations;
- non-degrading governance metrics on validation;
- no authority collapse (`mean w_field` and `mean w_guard` both in `(0.05,0.90)`).

---

## 7. Stability Controls

Required controls:

- entropy coefficient schedule: high → medium → low across curriculum stages;
- temperature anneal on arbiter policy logits;
- gradient clipping and KL early-stop against prior checkpoint;
- anti-mode-collapse monitors:
  - role entropy floor,
  - action diversity floor,
  - per-cell authority concentration alarms;
- uncertainty fallback: if trust diagnostics are low-confidence, route mass to
  guard + conservative field mix without violating reward cap.

---

## 8. Pre-Registered Metrics and Gates

### 8.1 Primary metrics

1. `mean_terminal_alignment` (slate-wide, GI subset).
2. `basin_capture_rate` (all and GI subset).
3. `bull_breach_trial_frac`.
4. `field_relief_frac_clean` = step fraction with `w_field > 0.70` on GI cells.

### 8.2 Secondary metrics

- `success_rate`;
- `reward_authority_frac` (`w_reward > 0.40`);
- `guard_brake_mass`;
- confidence intervals by seed bootstrap.

### 8.3 Diagnostic metrics

- per-cell authority traces;
- trust-conditioned authority curves;
- mode-collapse indicators (entropy/diversity alarms);
- clean-vs-corrupted utilization contrast.

### 8.4 Support gate (all required)

H1.2d is **SUPPORTIVE** only if all hold on the 13-cell binding slate:

1. **Proxy governance improvement:**  
   `basin_capture_rate_GI(H1.2d) <= basin_capture_rate_GI(H1.2c) - 0.01`.
2. **Competence recovery vs monolith:**  
   monolith gap closure on GI alignment is at least `35%` relative to H1.2c:
   `(gap_H1.2c - gap_H1.2d) / gap_H1.2c >= 0.35`.
3. **Constraint integrity:**  
   `bull_breach_trial_frac = 0.000` and no cap invariant violation.
4. **Mechanistic utilization evidence:**  
   `field_relief_frac_clean >= 0.30` and at least `0.12` higher than the same
   metric on corrupted sensor-noise cells.

If any single gate fails, support is not admitted.

---

## 9. Branch Table

| branch | condition | interpretation |
| --- | --- | --- |
| `H1_2D_SUPPORT` | all support gates pass | RL arbiter recovered governance/competence under bull-bound constraints |
| `H1_2D_GOV_ONLY` | gate 1,3,4 pass; gate 2 fails | governance improved but competence tax remains too large |
| `H1_2D_COMP_ONLY` | gate 2,3 pass; gate 1 fails | competence rose without governance advantage |
| `H1_2D_UTIL_FAIL` | gate 4 fails with otherwise valid run | RL did not materially use trustworthy field relief |
| `H1_2D_NULL` | valid run but support gate set not satisfied | pantheon reopening rung not supported at this tier |
| `H1_2D_SOVEREIGNTY_FAIL` | cap/bull integrity fails | invalid as pantheon evidence |
| `H1_2D_VOID` | leakage, seed overlap, budget mismatch, unregistered edits | rerun only after re-registration |

Demotion rule (locked): if `H1_2D_NULL`/`H1_2D_SOVEREIGNTY_FAIL` with no clear
non-redundant bottleneck, keep MESA-lane pantheon thesis at `[ORNAMENT]` for
this tier and require named new bottleneck before any H1.2e.

---

## 10. Runbook (Execution Plan)

### 10.1 Seed plan

- Train seeds: `20000-20255`;
- Validation seeds: `20300-20363`;
- Eval seeds: `10000-10063`;
- optional robustness reruns with seeds `11000-11063` (must be marked auxiliary).

Ranges are intentionally non-overlapping and offset by block to make split
audits and accidental overlap detection mechanically easy in manifests.

### 10.2 Budget and sweep matrix

Budget lock:

- one primary RL configuration + two preregistered ablations:
  1. `RL-Arbiter-Full` (all terms);
  2. `RL-Arbiter-noFieldUtil` (`λ_field = 0`);
  3. `RL-Arbiter-noProxyPenalty` (`λ_proxy = 0`).

Use matched incremental budget against `M-Adapter` within 5%.

### 10.3 Command skeleton (to be implemented/adapted in harness)

```powershell
node scripts/mesa-h1-build-coordinator-dataset.mjs `
  --phase h1_2d_rl_binding `
  --out results/mesa/h1-pantheon/h1_2d_rl/dataset `
  --train-seeds 256 --val-seeds 64 --train-seed-start 20000 --val-seed-start 20300 `
  --cells nominal,geometric-light,geometric-med,geometric-heavy,sensor-delay-light,sensor-delay-med,sensor-delay-heavy,decoy-light,decoy-med,decoy-heavy,sensor-noise-light,sensor-noise-med,sensor-noise-heavy `
  --horizon 200 --cap-mode reward-asymmetric --field-cap 1.00 --reward-cap 0.50 --guard-cap 0.70 `
  --field-policy results/mesa/phase2-matched-capacity/policies/signature_ppo_terminal_small_seed_0_phase5.policy.json `
  --reward-policy results/mesa/phase2-matched-capacity/policies/reward_ppo_phase3_small_seed_0_phase3_canonical_1m.policy.json

python -m training.mesa.train_h1_arbiter_rl `
  --dataset results/mesa/h1-pantheon/h1_2d_rl/dataset `
  --out results/mesa/h1-pantheon/h1_2d_rl/models `
  --seed 0 --epochs 40 --hidden-size 32 `
  --cap-mode reward-asymmetric --field-cap 1.00 --reward-cap 0.50 --guard-cap 0.70 `
  --lambda-align 1.0 --lambda-basin 1.0 --lambda-proxy 0.6 --lambda-guard 0.2 --lambda-field 0.5 --lambda-uncert 0.5 --lambda-smooth 0.05

node scripts/mesa-h1-pantheon-eval.mjs `
  --phase h1_2d_rl_binding `
  --out results/mesa/h1-pantheon/h1_2d_rl/eval `
  --seeds 64 --seed-start 10000 `
  --cells nominal,geometric-light,geometric-med,geometric-heavy,sensor-delay-light,sensor-delay-med,sensor-delay-heavy,decoy-light,decoy-med,decoy-heavy,sensor-noise-light,sensor-noise-med,sensor-noise-heavy `
  --horizon 200 --cap-mode reward-asymmetric --field-cap 1.00 --reward-cap 0.50 --guard-cap 0.70 `
  --arbiter results/mesa/h1-pantheon/h1_2d_rl/models/p_council_arbiter_rl.json `
  --guard results/mesa/h1-pantheon/h1_2d_rl/models/p_guard.json `
  --monolith-adapter results/mesa/h1-pantheon/h1_2d_rl/models/m_adapter.json `
  --bull-threshold 0.60
```

### 10.4 Artifact schema

```text
results/mesa/h1-pantheon/h1_2d_rl/
  dataset/manifest.json
  dataset/feature-schema.json
  models/p_guard.json
  models/p_council_arbiter_rl.json
  models/m_adapter.json
  models/training-report.json
  eval/h1-cell-map.csv
  eval/role_weights.csv
  eval/sovereignty-summary.csv
  eval/branch-readback.md
  eval/authority_traces.parquet
```

### 10.5 Result table template

| controller | mean S_T | S_T (GI) | basin (all) | basin (GI) | field-relief clean | bull-breach |
| --- | --- | --- | --- | --- | --- | --- |
| `P-Council-RL-RA50` | | | | | | |
| `P-Council-RA50` (H1.2c ref) | | | | | | |
| `M-Adapter` | | | | | — | — |

---

## 11. Reporting Checklist

- [ ] Spec committed (`docs/mesa/H1_2D_RL_ARBITER_SPEC.md`)
- [ ] Metrics preregistered and thresholds locked
- [ ] Budget + seed plan locked pre-run
- [ ] Baseline compatibility validated (`H1.2b/H1.2c`)
- [ ] Binding result table + confidence intervals produced
- [ ] Mechanistic authority-allocation analysis written
- [ ] Decision memo: `SUPPORT` or `NULL` with exact scope

---

## 12. Versioning

- `v0` (2026-06-18): opened after H1.2b and H1.2c binding nulls; registers
  RL-trained arbiter as the named-bottleneck rung; locks gates, demotion rule,
  and runbook scaffold before any H1.2d binding run.
