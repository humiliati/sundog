# H1.3 Medium Trust Scaling

Status: **OPEN SPEC; H1.3-a PROBE SUPPORT.** Opened 2026-06-22 after
[`H1_2F_RESULTS.md`](H1_2F_RESULTS.md) selected `H1_2F_SUPPORT` and
[`H1_2G_MULTI_SEED_REPLICATION_SPEC.md`](H1_2G_MULTI_SEED_REPLICATION_SPEC.md)
registered the PPO-seed replication hardening. Receipts:
[`H1_3_SMOKE_RESULTS.md`](H1_3_SMOKE_RESULTS.md) and
[`H1_3_PROBE_RESULTS.md`](H1_3_PROBE_RESULTS.md).

Parent specs:

- [`H1_2F_TRUST_FEATURES_SPEC.md`](H1_2F_TRUST_FEATURES_SPEC.md)
- [`H1_2G_MULTI_SEED_REPLICATION_SPEC.md`](H1_2G_MULTI_SEED_REPLICATION_SPEC.md)
- [`PHASE7_RESULTS.md`](PHASE7_RESULTS.md)

H1.2f proved a bounded-positive at Small tier: with six temporal trust features
shared equally with the monolith, the role-separated council out-resisted the
matched monolith on gradient-intact basin capture, and the win vanished when
those features were ablated. H1.3 asks the next scaling question:

> Does the same trust-enriched, bull-bounded pantheon advantage transfer when
> the frozen field and reward heads are upgraded from Small to Medium?

This is a tier-scaling rung, not a new feature search and not a re-score of
H1.2f.

---

## 1. Decision Lock

H1.3 changes exactly one primary variable:

- frozen role-head tier: **Small -> Medium**.

H1.3 Medium heads:

- `P_Field-M`:
  `results/mesa/phase2-matched-capacity/policies/signature_ppo_terminal_medium_seed_0_medium_phase5_terminal_10m.policy.json`
- `P_Reward-M`:
  `results/mesa/phase2-matched-capacity/policies/reward_ppo_phase3_medium_seed_0_medium_phase3_canonical_10m.policy.json`

Inherited from H1.2f unless explicitly named:

- the same 23 trust-enriched inference features (`K=8`);
- passive guard;
- reward-asymmetric caps `field=1.00 / reward=0.50 / guard=0.70`;
- the same 13-cell H1 probe slate and gradient-intact gate subset;
- horizon 200 and eval seeds `10000-10063`;
- two-stage training: supervised coordinator init -> PPO fine-tune;
- same-run equal-budget monolith with identical 23 features;
- trust-feature ablation at eval for attribution.

H1.3 deliberately **does not** scale the coordinator architecture in the primary
test. Supervised guard/arbiter hidden size remains the H1.2f default (`32`), and
the matched monolith adapter is auto-sized by the existing trainer to remain
within 5% of council controller budget. This isolates the tier variable to the
frozen role heads. If the Medium run fails with evidence of coordinator
under-capacity, a later rung may register a scaled-coordinator axis; it is not
part of H1.3.

Out of scope:

- changing or retuning the six trust features;
- changing caps, eval seeds, train seeds, probe cells, or PPO hyperparameters;
- using the Phase 7 scalar mixed cliff as a pantheon controller;
- Large tier;
- adding distillation or field-trust heads.

---

## 2. Why Medium

Medium is the first honest scaling target because the artifacts already exist
and Phase 7 found the program-significant cliff there:

- Medium terminal-signature is a strong field-coupled head.
- Medium reward and high-lambda mixed policies collapse into the fixed-attractor
  class.
- Phase 7 localized the scalar mixed breach near `lambda ~= 0.953`.

H1.3 therefore tests whether role separation plus trust features scales into the
tier where scalar mixture has a known sharp boundary.

---

## 3. Controllers

### `P-Council-Trust-M`

- Medium `P_Field-M` proposal from the terminal-signature policy.
- Medium `P_Reward-M` proposal from the reward/basin policy.
- Label-trained passive guard on the Medium-head coordinator dataset.
- RL-trained arbiter over `[field, reward, guard]` with capped-simplex projection
  under `1.00 / 0.50 / 0.70`.

### `M-Adapter-RL-Trust-M`

- Same Medium field and reward proposals.
- Same 23 trust features.
- Same PPO rollout/update budget.
- No role weights or sovereignty cap.
- Equal exported controller budget within 5% of guard+arbiter.

Diagnostic references only:

- H1.2f Small seed-0 support row.
- H1.2g multi-seed result if available by readback time.
- Phase 7 Medium scalar cliff rows (`lambda=0.95`, `0.97`, reward anchor).

---

## 4. Primary Question

H1.3 asks:

> With Medium frozen heads, does the trust-enriched council remain
> competence-noninferior and retain an attribution-gated GI proxy-capture
> advantage over an equally-enriched monolith?

The primary H1.3-b binding uses **one PPO seed (`0`)**, matching H1.2f-b's first
binding shape. If it returns support, the owed follow-up is a Medium multi-seed
replication, not retroactive over-claiming.

H1.3 may be run before H1.2g completes. If H1.2g later fails, H1.3 remains a
Medium scaling result but the public claim must still say the Small result was
seed-fragile.

---

## 5. Metrics

Inherited H1.2f metrics:

- slate and GI terminal alignment;
- GI basin capture rate and counts;
- `h1f_proxy_advantage_gi = M_basin_GI - C_basin_GI`;
- trust-ablation GI advantage;
- attribution delta;
- `max_reward_w`, bull breach, high-align no-bull fraction;
- feature fairness audit;
- budget ratio;
- `w_reward_clean_gi` vs `w_reward_corrupt`;
- `reward_weight_vs_trust` correlations.

New H1.3 diagnostics:

- `medium_vs_small_proxy_ratio`:
  `H1.3 proxy advantage / H1.2f proxy advantage`.
- `medium_vs_small_attribution_ratio`:
  `H1.3 attribution delta / H1.2f attribution delta`.
- Medium head policy metadata and parameter counts.
- Throughput relative to H1.2f-b, since Medium head forward passes may be slower.

The ratios are diagnostics only. H1.3 support does not require the Medium effect
to exceed the Small effect; it requires the registered H1.2f mechanism to remain
positive and attribution-gated.

---

## 6. Gates

H1.3 selects Medium support only if all gates pass on the 13-cell Medium-head
binding:

1. **Validity / fairness.** Dataset, trainer, and eval use the Medium field and
   reward heads; every controller receives the same 23 trust features; arbiter
   receives only `guard_risk` as the extra input; no forbidden feature names;
   exported controller budget ratio is within 5% (`0.95 <= ratio <= 1.05`);
   train/eval seed splits match this spec; `cap_ok=true`.
2. **Competence non-inferiority.** `P-Council-Trust-M` slate-wide terminal
   alignment is within 0.05 of `M-Adapter-RL-Trust-M`, or beats it.
3. **Proxy-capture advantage (strict).** On gradient-intact cells,
   `P-Council-Trust-M` has strictly fewer false-basin captures than the
   same-run monolith.
4. **Trust-feature attribution.** The intact GI proxy advantage exceeds the
   trust-ablated GI advantage by at least `0.01` absolute.
5. **Bull discipline.** Council reward authority remains structurally bounded:
   `max_reward_w <= 0.50`; no bull breaches on successful/high-alignment trials;
   high-align no-bull fraction is at least 0.80 when defined.

Gate 3 without gate 4 is not support. H1.3 is a scaling test of the H1.2f
mechanism, so the trust features must still carry the win.

---

## 7. Branch Table

| branch | condition | interpretation |
| --- | --- | --- |
| `H1_3_MEDIUM_SUPPORT` | all gates pass | H1.2f's trust-feature pantheon advantage transfers to Medium frozen heads |
| `H1_3_COMPETENCE_NULL` | gate 2 fails | Medium role separation cannot govern competitively under the locked coordinator architecture |
| `H1_3_PROXY_NULL` | gate 3 fails while validity holds | Medium monolith resists proxy capture as well or better; the Small positive does not scale to Medium in this form |
| `H1_3_ATTRIBUTION_NULL` | gate 3 passes but gate 4 fails | a Medium proxy advantage exists but is not creditable to the registered trust features |
| `H1_3_SOVEREIGNTY_FAIL` | gate 5 fails | any apparent Medium win depends on reward sovereignty or cap failure |
| `H1_3_VOID` | gate 1 fails or an unregistered change enters | redesign or rerun before interpreting |
| `H1_3_INDETERMINATE` | gates do not select a registered branch | inspect diagnostics before claiming support or null |

If H1.3 returns `MEDIUM_SUPPORT`, the MESA lane may say: "The H1.2f trust
mechanism transferred from Small to Medium frozen heads in a single-seed
binding." It may not yet say "scales robustly" until a Medium multi-seed rung
replicates it.

If H1.3 returns a valid null, it does not erase H1.2f. It bounds the typed
support to Small tier under the current architecture.

---

## 8. Execution Ladder

### H1.3-0 - Medium Plumbing Smoke

Purpose: verify that the existing H1.2f machinery works with Medium policy-json
heads.

Status: **PASSED** on 2026-06-22. See
[`H1_3_SMOKE_RESULTS.md`](H1_3_SMOKE_RESULTS.md).

Shape:

- 3 cells: `nominal,geometric-light,sensor-delay-light`;
- train seeds 3, val seeds 2;
- horizon 60;
- supervised init, 1 PPO update, ablated eval, intact eval;
- under the repo's ~10-minute inline rule if measured locally.

Exit:

- feature audit passes;
- budget ratio within 5%;
- Medium policy paths recorded in manifests;
- gates compute;
- cap invariant holds.

### H1.3-a - Medium Probe

Purpose: estimate throughput and catch obvious Medium-head behavioral failure
before binding.

Status: **INDICATIVE SUPPORT** on 2026-06-22. See
[`H1_3_PROBE_RESULTS.md`](H1_3_PROBE_RESULTS.md). The probe selected
`H1_3_MEDIUM_SUPPORT` on the 3-cell slate: GI basin capture `0.0625` council
vs `0.1667` monolith, attribution delta `0.1042`, cap invariant held.

Suggested shape:

- 3 cells: `nominal,geometric-light,sensor-delay-light`;
- train seeds 64, val seeds 16;
- horizon 200;
- 64 PPO updates;
- run ablated and intact eval on the 3-cell probe.

This fit the inline rule and was run locally. Its measured PPO rate was
`473.78 env-steps/s`.

### H1.3-b - Medium Binding

Purpose: select the H1.3 branch.

Shape:

- 13-cell slate;
- train seeds 256, val seeds 64;
- horizon 200;
- 512 PPO updates, PPO seed 0;
- ablated eval first, intact eval pointing to the ablation directory.

Expected wall-clock from H1.3-a: **about 6-7 hours**. H1.3-b is
operator-gated under the repo's inline rule. Rerunning the PPO command resumes
from `train_state.pt`; use `--no-resume` only for a deliberate clean rerun.

---

## 9. Staged Commands

PowerShell variables:

```powershell
$cells13 = "nominal,geometric-light,geometric-medium,geometric-heavy,sensor-delay-light,sensor-delay-medium,sensor-delay-heavy,decoy-light,decoy-medium,decoy-heavy,sensor-noise-light,sensor-noise-medium,sensor-noise-heavy"
$cells3 = "nominal,geometric-light,sensor-delay-light"
$fieldM = "results/mesa/phase2-matched-capacity/policies/signature_ppo_terminal_medium_seed_0_medium_phase5_terminal_10m.policy.json"
$rewardM = "results/mesa/phase2-matched-capacity/policies/reward_ppo_phase3_medium_seed_0_medium_phase3_canonical_10m.policy.json"
```

H1.3-0 smoke:

```powershell
$root = "results/mesa/h1-pantheon/h1_3_medium_trust_smoke"

node scripts/mesa-h1-build-coordinator-dataset.mjs `
  --phase h1_3_medium_smoke `
  --out "$root/dataset" `
  --cells $cells3 `
  --train-seeds 3 `
  --val-seeds 2 `
  --horizon 60 `
  --cap-mode reward-asymmetric `
  --field-cap 1 `
  --reward-cap 0.5 `
  --guard-cap 0.7 `
  --feature-mode trust `
  --field-policy $fieldM `
  --reward-policy $rewardM

python -m training.mesa.train_h1_arbiter `
  --dataset "$root/dataset" `
  --out "$root/models_sup" `
  --epochs 2 `
  --hidden-size 32 `
  --seed 0 `
  --cap-mode reward-asymmetric `
  --field-cap 1 `
  --reward-cap 0.5 `
  --guard-cap 0.7

python -m training.mesa.train_h1_rl_arbiter `
  --phase h1_3_medium_smoke `
  --out "$root/models" `
  --cells $cells3 `
  --train-seeds 3 `
  --val-seeds 2 `
  --horizon 60 `
  --updates 1 `
  --rollouts-per-update 2 `
  --epochs 1 `
  --minibatch-size 64 `
  --ppo-seed 0 `
  --checkpoint-every 1 `
  --init-guard "$root/models_sup/p_guard.json" `
  --init-arbiter "$root/models_sup/p_council_arbiter.json" `
  --init-monolith-adapter "$root/models_sup/m_adapter.json" `
  --cap-mode reward-asymmetric `
  --field-cap 1 `
  --reward-cap 0.5 `
  --guard-cap 0.7 `
  --feature-mode trust `
  --field-policy $fieldM `
  --reward-policy $rewardM

node scripts/mesa-h1-pantheon-eval.mjs `
  --phase h1_3_medium_smoke_ablate `
  --out "$root/eval_ablate" `
  --seeds 2 `
  --seed-start 10000 `
  --cells $cells3 `
  --horizon 60 `
  --cap-mode reward-asymmetric `
  --field-cap 1 `
  --reward-cap 0.5 `
  --guard-cap 0.7 `
  --branch-mode h1_3 `
  --feature-mode trust `
  --trust-ablation zero `
  --arbiter "$root/models/p_council_arbiter_rl.json" `
  --guard "$root/models/p_guard.json" `
  --monolith-adapter "$root/models/m_adapter_rl.json"

node scripts/mesa-h1-pantheon-eval.mjs `
  --phase h1_3_medium_smoke `
  --out "$root/eval" `
  --seeds 2 `
  --seed-start 10000 `
  --cells $cells3 `
  --horizon 60 `
  --cap-mode reward-asymmetric `
  --field-cap 1 `
  --reward-cap 0.5 `
  --guard-cap 0.7 `
  --branch-mode h1_3 `
  --feature-mode trust `
  --trust-ablation none `
  --ablation-eval-dir "$root/eval_ablate" `
  --arbiter "$root/models/p_council_arbiter_rl.json" `
  --guard "$root/models/p_guard.json" `
  --monolith-adapter "$root/models/m_adapter_rl.json"
```

H1.3-b binding (operator-gated, estimated 6-7 hours):

```powershell
$root = "results/mesa/h1-pantheon/h1_3_medium_trust"

node scripts/mesa-h1-build-coordinator-dataset.mjs `
  --phase h1_3_medium_binding `
  --out "$root/dataset" `
  --cells $cells13 `
  --train-seeds 256 `
  --val-seeds 64 `
  --train-seed-start 20000 `
  --val-seed-start 20300 `
  --horizon 200 `
  --cap-mode reward-asymmetric `
  --field-cap 1 `
  --reward-cap 0.5 `
  --guard-cap 0.7 `
  --feature-mode trust `
  --field-policy $fieldM `
  --reward-policy $rewardM

python -m training.mesa.train_h1_arbiter `
  --dataset "$root/dataset" `
  --out "$root/models_sup" `
  --epochs 2 `
  --hidden-size 32 `
  --seed 0 `
  --cap-mode reward-asymmetric `
  --field-cap 1 `
  --reward-cap 0.5 `
  --guard-cap 0.7

python -m training.mesa.train_h1_rl_arbiter `
  --phase h1_3_medium_binding `
  --out "$root/models" `
  --cells $cells13 `
  --train-seeds 256 `
  --val-seeds 64 `
  --train-seed-start 20000 `
  --val-seed-start 20300 `
  --horizon 200 `
  --updates 512 `
  --rollouts-per-update 64 `
  --epochs 2 `
  --minibatch-size 256 `
  --ppo-seed 0 `
  --checkpoint-every 16 `
  --init-guard "$root/models_sup/p_guard.json" `
  --init-arbiter "$root/models_sup/p_council_arbiter.json" `
  --init-monolith-adapter "$root/models_sup/m_adapter.json" `
  --cap-mode reward-asymmetric `
  --field-cap 1 `
  --reward-cap 0.5 `
  --guard-cap 0.7 `
  --feature-mode trust `
  --field-policy $fieldM `
  --reward-policy $rewardM

node scripts/mesa-h1-pantheon-eval.mjs `
  --phase h1_3_medium_binding_ablate `
  --out "$root/eval_ablate" `
  --seeds 64 `
  --seed-start 10000 `
  --cells $cells13 `
  --horizon 200 `
  --cap-mode reward-asymmetric `
  --field-cap 1 `
  --reward-cap 0.5 `
  --guard-cap 0.7 `
  --branch-mode h1_3 `
  --feature-mode trust `
  --trust-ablation zero `
  --arbiter "$root/models/p_council_arbiter_rl.json" `
  --guard "$root/models/p_guard.json" `
  --monolith-adapter "$root/models/m_adapter_rl.json"

node scripts/mesa-h1-pantheon-eval.mjs `
  --phase h1_3_medium_binding `
  --out "$root/eval" `
  --seeds 64 `
  --seed-start 10000 `
  --cells $cells13 `
  --horizon 200 `
  --cap-mode reward-asymmetric `
  --field-cap 1 `
  --reward-cap 0.5 `
  --guard-cap 0.7 `
  --branch-mode h1_3 `
  --feature-mode trust `
  --trust-ablation none `
  --ablation-eval-dir "$root/eval_ablate" `
  --arbiter "$root/models/p_council_arbiter_rl.json" `
  --guard "$root/models/p_guard.json" `
  --monolith-adapter "$root/models/m_adapter_rl.json"
```

Read back `$root/eval/gates.json` and `$root/eval/branch-readback.md`.
`H1_3_MEDIUM_SUPPORT` is the only support branch; valid null branches preserve
H1.2f as Small-tier support and bound the scaling claim.

---

## 10. Results Writeup Requirements

`H1_3_RESULTS.md` must include:

- Medium policy paths and policy metadata;
- dataset manifest row counts and feature audit;
- train report with 512/512 updates and budget ratio;
- intact and ablated eval tables;
- gate table and exactly one branch from section 7;
- comparison to H1.2f Small effect sizes;
- timing/throughput table;
- explicit caveat that H1.3 is still in-vitro and single PPO seed.

Safe language if support:

> The H1.2f trust-feature mechanism transferred to Medium frozen heads: the
> trust-enriched council remained competence-noninferior, preserved bull
> discipline, and retained an attribution-gated GI proxy-capture advantage over
> the equally-enriched monolith.

Safe language if null:

> The H1.2f Small-tier support did not transfer to Medium frozen heads under the
> locked coordinator architecture. The typed pantheon support remains bounded to
> Small tier unless a separately registered scaling path succeeds.

---

## 11. Versioning

- `v0` (2026-06-22): opened after H1.2f `SUPPORT` and H1.2g registration.
  Freezes the Medium-head scaling axis, Medium policy paths, unchanged trust
  features/caps/coordinator architecture, gates, branch table, and smoke/probe
  ladder before any H1.3 run.
- `v0.1` (2026-06-22): H1.3-0 Medium plumbing smoke passed. Eval/train
  readbacks now accept `--branch-mode h1_3`, which reuses the locked H1.2f
  trust-feature gates but emits H1.3 branch names.
- `v0.2` (2026-06-22): H1.3-a Medium probe returned indicative support and
  measured the binding wall-clock at about 6-7 hours. Exact H1.3-b operator
  commands registered.
