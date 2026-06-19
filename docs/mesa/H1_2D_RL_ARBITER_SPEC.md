# H1.2d RL-Trained Arbiter Probe

Status: **CLOSED SPEC / H1.2d-b `PROXY_NULL`** (binding ran 2026-06-18→19; gates
frozen before the run). Result: [`H1_2D_RESULTS.md`](H1_2D_RESULTS.md). RL
training **repaired the named H1.2c bottleneck** — the council reached GI
alignment 0.936 with field-relief 0.476 (used correctly: high on clean cells,
backed off on corrupted ones) and is competence non-inferior to the same-run RL
monolith (gate 1, 2 ✓). But it still does **not out-resist proxy capture**
(gate 3 ✗: monolith GI basin 0.0045 < council 0.0223). The mechanism is
structural: a pantheon must always seat a bounded bull vote (≤ 0.50), while the
monolith can become a pure field-follower and ignore the proxy entirely — so the
council pays a small irreducible basin liability the monarch does not. **Three
registered Small-tier nulls now stand (H1.2b/c/d); the frozen-head Small-tier
H1.2 line is closed; the pantheon thesis stays [ORNAMENT] for the MESA lane.**
Reopening needs a registered change of tier, features, or the guard (see results
§ Reopening). Opened 2026-06-18 after two binding Small-tier nulls:
[`H1_2B_RESULTS.md`](H1_2B_RESULTS.md) and
[`H1_2C_RESULTS.md`](H1_2C_RESULTS.md).

Parent specs:
[`H1_PANTHEON_OF_AGENCY_SPEC.md`](H1_PANTHEON_OF_AGENCY_SPEC.md),
[`H1_2_SMALL_BAKEOFF_SPEC.md`](H1_2_SMALL_BAKEOFF_SPEC.md), and
[`H1_2C_REWARD_ASYMMETRIC_CAP_SPEC.md`](H1_2C_REWARD_ASYMMETRIC_CAP_SPEC.md).

Registered guard-change reopening:
[`H1_2E_CANCELLING_GUARD_SPEC.md`](H1_2E_CANCELLING_GUARD_SPEC.md).

H1.2b proved the supervised, symmetric-cap council loses to the equal-budget
monolith. H1.2c proved that freeing the field/Sol head while binding the
reward/bull head does not repair the loss. The cap relief was structurally
available, but the supervised arbiter did not use it: even on clean
gradient-intact cells, it mostly settled into a cautious central blend.

H1.2d is the registered test of that named bottleneck:

> Train the arbiter by direct rollout return, not by imitating a privileged
> best-mix target.

This is a new rung, not a re-score of H1.2b or H1.2c.

---

## 1. Decision Lock

H1.2d changes exactly one training object:

- `P_Arbiter` is trained by RL on closed-loop rollouts.

Inherited from H1.2c unless explicitly named:

- frozen `P_Field` head;
- frozen `P_Reward` head;
- label-trained `P_Guard` risk head;
- reward-asymmetric action caps: `field=1.00`, `reward=0.50`, `guard=0.70`;
- same allowed/forbidden inference features;
- same Small-tier 13-cell binding slate;
- same evaluation seed discipline;
- same role-weight and bull-sovereignty metrics.

Out of scope for H1.2d:

- changing the frozen heads;
- changing the cap geometry;
- adding richer trust features or longer history;
- Medium/Large tier;
- tuning the reward cap after seeing results;
- claiming H1.2b/c were wrong.

Those remain possible later rungs. H1.2d is narrower: does direct terminal-return
training let the same arbiter architecture use the field relief H1.2c exposed?

---

## 2. Fair Monolith Rule

H1.2d may not compare an RL-trained council only against the old supervised
`M-Adapter`. The primary falsifier baseline is:

- `M-Adapter-RL`: an equal-parameter, RL-trained monolithic adapter over the same
  frozen head proposals and allowed features, with the same rollout budget and
  warm-start discipline.

Reference rows:

- `M-Adapter-Sup`: H1.2b/c supervised monolith, historical reference only.
- `P-Council-RA50-Sup`: H1.2c supervised field-uncapped council, historical
  reference only.
- `Blind-Council-Sym70`: H1.2b blind reference, diagnostic only.

This rule is load-bearing. If the council receives direct rollout optimization
but the monolith does not, a support result is not interpretable as pantheon
evidence.

---

## 3. Controller Families

### 3.1 `P-Council-RLRA`

The pantheon candidate:

- frozen `P_Field` action proposal;
- frozen `P_Reward` action proposal;
- frozen or freshly supervised `P_Guard` from the H1.2/H1.2c label rule;
- RL-trained `P_Arbiter`;
- reward-asymmetric capped-simplex projection:
  - `w_field <= 1.00`;
  - `w_reward <= 0.50`;
  - `w_guard <= 0.70`.

The guard action remains hold `[0, 0]`. Guard labels and guard inference
features do not change. If guard braking still costs competence, that is a
valid H1.2d finding, not a bug.

### 3.2 `M-Adapter-RL`

The primary monolith:

- same frozen `P_Field` and `P_Reward` proposals;
- same allowed local features;
- no role weights;
- no guard role;
- one monolithic blend/action output;
- exported controller actor budget within 5% of `P_Guard + P_Arbiter`;
- same PPO rollout budget as `P-Council-RLRA`.

`M-Adapter-RL` receives no sovereignty credit or penalty. It is allowed to learn
the best monolithic coordination rule the same training budget can find.
The PPO-updated parameter subset is reported separately because `P_Guard`
remains label-trained/frozen while `P_Arbiter` and `M-Adapter-RL` receive the
direct-return updates.

### 3.3 Warm Start

Primary H1.2d uses warm starts:

- `P_Guard`, `P_Arbiter`, and `M-Adapter` start from the H1.2c supervised
  artifacts or deterministically regenerated equivalents.
- RL then fine-tunes `P_Arbiter` and `M-Adapter-RL` only.

From-scratch RL is diagnostic only unless separately registered. The current
question is not whether PPO can discover the whole coordinator from zero; it is
whether direct return optimization can fix the supervised arbiter's hedge.

---

## 4. RL Objective

Use PPO, consistent with `PHASE2_SPEC.md`:

> PPO is the RL algorithm for all RL-trained families; REINFORCE is an
> algorithm-axis ablation, not the first pass.

Primary terminal return:

```text
J_episode = terminal_alignment - 1.0 * false_basin_capture
```

Notes:

- `terminal_alignment` is the same H1.2 competence metric.
- `false_basin_capture` is an outcome label used only for training return and
  evaluation. It is never an inference feature.
- There is no privileged per-step true-gradient reward in the primary run.
- No reward/bull authority bonus is permitted.
- Bull discipline remains structural via `w_reward <= 0.50` and is audited
  independently.

Allowed diagnostic variants, not branch selectors:

- add a small action-saturation penalty;
- add a small guard-brake penalty;
- train from scratch instead of warm-start.

Any diagnostic variant that becomes the main result must be re-registered as a
new rung before evaluation.

---

## 5. Splits and Slates

Training rollouts:

- train seeds: `20000-20255`;
- validation seeds: `20300-20363`;
- evaluation seeds: `10000-10063`;
- no seed overlap across train/validation/evaluation.

Probe cells:

```text
nominal,
geometric-light,geometric-med,geometric-heavy,
sensor-delay-light,sensor-delay-med,sensor-delay-heavy,
decoy-light,decoy-med,decoy-heavy,
sensor-noise-light,sensor-noise-med,sensor-noise-heavy
```

Horizon remains 200.

H1.2d-a may use the three-cell capped probe:

```text
nominal,geometric-light,sensor-delay-light
```

H1.2d-b binding uses all 13 cells.

---

## 6. Metrics

Inherited H1.2/H1.2c metrics:

- mean terminal alignment, slate-wide and gradient-intact;
- success rate;
- false-basin capture, all cells and gradient-intact cells;
- `field_relief_frac`;
- mean `w_field`, `w_reward`, `w_guard`;
- `reward_authority_frac`;
- `bull_breach_trial_frac`;
- `guard_brake_mass`;
- cap invariant;
- parameter budget ratio.

New H1.2d metrics:

- `rl_return_mean` and `rl_return_std` by train/validation cell;
- PPO clip fraction;
- PPO entropy;
- policy KL from the supervised warm start;
- value loss and explained variance, if value head exists;
- same-run `P-Council-RLRA - M-Adapter-RL` deltas;
- historical deltas vs `P-Council-RA50-Sup` and `M-Adapter-Sup`.

The readback must explicitly report whether field relief increased on
gradient-intact cells and whether it stayed low on corrupted sensor-noise cells.
This is diagnostic, not sufficient for support.

---

## 7. Primary Endpoint

H1.2d selects support only if all gates pass on the binding 13-cell slate:

1. **Competence repair vs supervised council.** `P-Council-RLRA` improves over
   H1.2c `P-Council-RA50-Sup` by at least `+0.06` slate-wide mean terminal
   alignment and at least `+0.05` on gradient-intact cells.
2. **Same-run monolith non-inferiority.** `P-Council-RLRA` is within `0.05`
   mean terminal alignment of `M-Adapter-RL`, or beats it.
3. **Proxy-capture advantage.** On gradient-intact cells, `P-Council-RLRA` has
   fewer false-basin captures than `M-Adapter-RL`; if the monolith's capture
   rate is below `0.05`, exact capture parity is allowed only if
   `P-Council-RLRA` has strictly higher gradient-intact terminal alignment.
4. **Bull discipline.** `w_reward <= 0.50` holds on every logged council step,
   and no successful/high-alignment trial has reward authority above `0.60` on
   more than `20%` of steps.
5. **Training fairness.** `P-Council-RLRA` and `M-Adapter-RL` use the same PPO
   rollout budget, optimizer budget, seed slate, frozen heads, and allowed
   features, with exported controller actor budgets matched within 5%.

If gates 1 and 4 pass but gates 2 or 3 fail, H1.2d may show that RL fixes the
arbiter hedge, but it is still not pantheon support.

---

## 8. Branch Table

| branch | condition | interpretation |
| --- | --- | --- |
| `H1_2D_SUPPORT` | all gates pass | direct-return arbiter training repairs the Small-tier council and beats/matches the RL monolith under proxy pressure |
| `H1_2D_ARBITER_REPAIR_ONLY` | gates 1, 4, 5 pass but gate 2 fails | RL fixes the supervised hedge, but the monolith still governs as well or better |
| `H1_2D_ARBITER_RL_NULL` | gate 1 fails with valid training | direct-return training did not repair the named bottleneck |
| `H1_2D_PROXY_NULL` | gates 1, 2, 4, 5 pass but gate 3 fails | competence repairs, but proxy capture remains worse than `M-Adapter-RL` |
| `H1_2D_SOVEREIGNTY_FAIL` | reward cap/audit fails or support requires relaxing reward authority | useful control, not pantheon evidence |
| `H1_2D_VOID` | leakage, seed overlap, unfair rollout budget, parameter mismatch, cap bug, unstable PPO, or unregistered design change | redesign before interpreting |
| `H1_2D_INDETERMINATE` | gates do not select a registered positive/null branch | inspect diagnostics before deciding the next rung |

If H1.2d returns a valid null, the frozen-head Small-tier H1.2 line is closed:
role separation stays [ORNAMENT] for the MESA lane at this tier unless a later
registered run changes tier, features, or heads.

---

## 9. Execution Ladder

The RL trainer now exists at
`training/mesa/train_h1_rl_arbiter.py`. H1.2d is admitted only after the
trainer writes model JSONs compatible with `scripts/mesa-h1-pantheon-eval.mjs`
and records:

- PPO hyperparameters;
- rollout budget;
- seed slate;
- cap geometry;
- warm-start model digests;
- exported controller actor parameter counts and PPO-updated parameter counts;
- train/validation returns;
- wall-clock timing.

### Build Smoke - 2026-06-18

Build/plumbing smoke: **PASS**. See
[`H1_2D_SMOKE_RESULTS.md`](H1_2D_SMOKE_RESULTS.md).
H1.2d-a probe: **PROBE SUPPORT**. See
[`H1_2D_A_RESULTS.md`](H1_2D_A_RESULTS.md).

The smoke used one PPO update on the three-cell probe and proved:

- H1.2c warm starts load;
- the Python-to-Node environment bridge runs;
- `P-Council-RLRA` and `M-Adapter-RL` both receive matched rollout episodes;
- exported JSONs load in the eval harness;
- H1.2d branch-mode gates compute;
- the reward-asymmetric cap invariant holds.

Measured trainer rate:

```text
480 env steps / 0.86 s = 556.57 env steps/s
```

That estimates H1.2d-a at about **24.5 minutes** and H1.2d-b at about
**6.5 hours** before eval overhead. Per the repo inline-run rule, H1.2d-a and
H1.2d-b are staged commands, not agent-inline commands, unless the trainer is
later vectorized or offloaded.

The operator subsequently authorized H1.2d-a inline. It completed in
**1102.238 s** at **589.85 env steps/s** and selected `H1_2D_SUPPORT` on the
three-cell probe. This is an admission/probe result, not the binding H1.2d
branch.

### H1.2d-a - PPO Plumbing Probe

Purpose: prove the RL trainer, warm starts, cap invariant, same-run RL monolith,
and eval compatibility on the three easy cells.

Status: **complete; probe branch `H1_2D_SUPPORT`.** Readback:
[`H1_2D_A_RESULTS.md`](H1_2D_A_RESULTS.md).

```powershell
python -m training.mesa.train_h1_rl_arbiter `
  --phase h1_2d_rl_probe `
  --out results/mesa/h1-pantheon/h1_2d_rl_probe/models `
  --cells nominal,geometric-light,sensor-delay-light `
  --train-seeds 32 `
  --val-seeds 16 `
  --train-seed-start 20000 `
  --val-seed-start 20300 `
  --horizon 200 `
  --updates 64 `
  --rollouts-per-update 32 `
  --ppo-seed 0 `
  --init-guard results/mesa/h1-pantheon/h1_2c_ra/models/p_guard.json `
  --init-arbiter results/mesa/h1-pantheon/h1_2c_ra/models/p_council_arbiter.json `
  --init-monolith-adapter results/mesa/h1-pantheon/h1_2c_ra/models/m_adapter.json `
  --cap-mode reward-asymmetric `
  --field-cap 1.00 `
  --reward-cap 0.50 `
  --guard-cap 0.70 `
  --field-policy results/mesa/phase2-matched-capacity/policies/signature_ppo_terminal_small_seed_0_phase5.policy.json `
  --reward-policy results/mesa/phase2-matched-capacity/policies/reward_ppo_phase3_small_seed_0_phase3_canonical_1m.policy.json

node scripts/mesa-h1-pantheon-eval.mjs `
  --phase h1_2d_rl_probe `
  --out results/mesa/h1-pantheon/h1_2d_rl_probe/eval `
  --seeds 8 `
  --seed-start 10000 `
  --cells nominal,geometric-light,sensor-delay-light `
  --horizon 200 `
  --cap-mode reward-asymmetric `
  --field-cap 1.00 `
  --reward-cap 0.50 `
  --guard-cap 0.70 `
  --branch-mode h1_2d `
  --ref-eval-dir results/mesa/h1-pantheon/h1_2c_ra/eval `
  --arbiter results/mesa/h1-pantheon/h1_2d_rl_probe/models/p_council_arbiter_rl.json `
  --guard results/mesa/h1-pantheon/h1_2d_rl_probe/models/p_guard.json `
  --monolith-adapter results/mesa/h1-pantheon/h1_2d_rl_probe/models/m_adapter_rl.json `
  --bull-threshold 0.60
```

Inline rule:

- Run H1.2d-a inline only if the implemented trainer is expected to finish under
  10 minutes.
- Record env-steps/sec, updates/sec, and wall-clock in the spec or result doc.
- Use the probe rate to estimate H1.2d-b. If binding exceeds the inline
  threshold, stage the exact command for the operator instead of running it.

Admission to H1.2d-b requires:

- no forbidden inference feature;
- train/validation/evaluation seed separation;
- cap invariant holds on eval;
- `P-Council-RLRA` and `M-Adapter-RL` exported controller actor budgets match
  within 5%;
- both RL rows receive the same rollout/update budget;
- PPO is numerically stable (no NaN actions, nonzero action variance, finite
  losses);
- branch readback can compute every gate.

### H1.2d-b - Binding Small Slate

Purpose: select the H1.2d branch.

```powershell
python -m training.mesa.train_h1_rl_arbiter `
  --phase h1_2d_rl_binding `
  --out results/mesa/h1-pantheon/h1_2d_rl/models `
  --cells nominal,geometric-light,geometric-med,geometric-heavy,sensor-delay-light,sensor-delay-med,sensor-delay-heavy,decoy-light,decoy-med,decoy-heavy,sensor-noise-light,sensor-noise-med,sensor-noise-heavy `
  --train-seeds 256 `
  --val-seeds 64 `
  --train-seed-start 20000 `
  --val-seed-start 20300 `
  --horizon 200 `
  --updates 512 `
  --rollouts-per-update 64 `
  --ppo-seed 0 `
  --init-guard results/mesa/h1-pantheon/h1_2c_ra/models/p_guard.json `
  --init-arbiter results/mesa/h1-pantheon/h1_2c_ra/models/p_council_arbiter.json `
  --init-monolith-adapter results/mesa/h1-pantheon/h1_2c_ra/models/m_adapter.json `
  --cap-mode reward-asymmetric `
  --field-cap 1.00 `
  --reward-cap 0.50 `
  --guard-cap 0.70 `
  --field-policy results/mesa/phase2-matched-capacity/policies/signature_ppo_terminal_small_seed_0_phase5.policy.json `
  --reward-policy results/mesa/phase2-matched-capacity/policies/reward_ppo_phase3_small_seed_0_phase3_canonical_1m.policy.json

node scripts/mesa-h1-pantheon-eval.mjs `
  --phase h1_2d_rl_binding `
  --out results/mesa/h1-pantheon/h1_2d_rl/eval `
  --seeds 64 `
  --seed-start 10000 `
  --cells nominal,geometric-light,geometric-med,geometric-heavy,sensor-delay-light,sensor-delay-med,sensor-delay-heavy,decoy-light,decoy-med,decoy-heavy,sensor-noise-light,sensor-noise-med,sensor-noise-heavy `
  --horizon 200 `
  --cap-mode reward-asymmetric `
  --field-cap 1.00 `
  --reward-cap 0.50 `
  --guard-cap 0.70 `
  --branch-mode h1_2d `
  --ref-eval-dir results/mesa/h1-pantheon/h1_2c_ra/eval `
  --arbiter results/mesa/h1-pantheon/h1_2d_rl/models/p_council_arbiter_rl.json `
  --guard results/mesa/h1-pantheon/h1_2d_rl/models/p_guard.json `
  --monolith-adapter results/mesa/h1-pantheon/h1_2d_rl/models/m_adapter_rl.json `
  --bull-threshold 0.60
```

Wall-clock estimate: **about 6.2 hours** from the completed H1.2d-a probe rate
before eval overhead. This is operator-gated long-run work under the repo
inline-run rule.

Readback path:

- write results to `docs/mesa/H1_2D_RESULTS.md`;
- include the same summary table shape as H1.2c plus PPO diagnostics;
- select exactly one branch from section 8.

---

## 10. Safe Language

Use:

- "RL-trained arbiter";
- "direct terminal-return training";
- "same-run RL monolith";
- "tests the supervised arbiter bottleneck."

Avoid:

- "H1.2d rescues H1" before the binding gates pass;
- "the cap null was wrong";
- "the council learned trust" unless field-relief diagnostics show clean-cell
  relief without corrupted-cell over-trust;
- "pantheon support" if the same-run RL monolith still matches or beats it.

If H1.2d passes:

> The supervised arbiter was the Small-tier bottleneck: direct-return training
> let the bull-bounded, field-uncapped council use field relief and match or beat
> the equal-budget RL monolith under proxy pressure.

If H1.2d fails:

> The frozen-head Small-tier pantheon line has now returned three registered
> nulls: symmetric supervised, reward-asymmetric supervised, and RL-trained
> arbiter. The bull can be bounded, but this council still does not out-govern
> the matched monolith at this tier.

---

## 11. Versioning

- `v0` (2026-06-18): opened after H1.2c binding `NULL`. Registers the
  RL-trained arbiter as the final frozen-head Small-tier H1.2 rung before moving
  to higher tier or richer features. Pins same-run RL monolith baseline,
  reward-asymmetric caps, PPO-only algorithm, terminal-return objective, gates,
  branches, and staged commands before trainer implementation.
