# H1.2 Small-Tier Pantheon Bake-Off - Learned Arbiter and Guard

Status: **OPEN SPEC.** Opened 2026-06-18 after the H1.1 harness smoke.
Parent: [`H1_PANTHEON_OF_AGENCY_SPEC.md`](H1_PANTHEON_OF_AGENCY_SPEC.md).

H1.1 answered the admission question: the wrapper, role-weight schema, and
sovereignty metric work. It also named the actual experimental object. A blind
confidence blend does not know when the reward head is basin-pulling, so the
council loses terminal alignment to the tuned scalar monolith. H1.2 therefore
tests the **learned arbiter plus label-trained guard**, not the heads.

---

## 1. Decision Lock

H1.2 is a **frozen-head, learned-coordinator** bake-off.

Frozen heads:

- `P_Field`: terminal-signature Small policy, preferred path
  `results/mesa/phase2-matched-capacity/policies/signature_ppo_terminal_small_seed_0_phase5.policy.json`.
- `P_Reward`: canonical Phase 3 reward policy,
  `results/mesa/phase2-matched-capacity/policies/reward_ppo_phase3_small_seed_0_phase3_canonical_1m.policy.json`.

Trainable H1.2 objects:

- `P_Guard`: label-trained risk head.
- `P_Arbiter`: learned bounded role-weight head.

Primary matched monolith:

- `M-Adapter`: same frozen head proposals and local features, same incremental
  trainable parameter budget as `P_Guard + P_Arbiter`, but one monolithic action
  output and no role-separation constraint. It may learn a single action or
  unconstrained head blend; it may not emit role weights or claim pantheon
  support.

Reference monolith:

- `M-Scalar`: existing scalar-mixture policy used by H1.1,
  `mixed_ppo_phase3_lambda_0_5_small_seed_0_phase3_canonical_1m.policy.json`,
  plus any Phase 7 Small scalar row selected as best reference. This row is
  prior/readback, not the fairness-critical baseline.

This means H1.2 is not the final full-capacity pantheon result. It is the first
clean test of whether learned role arbitration can beat scalar coordination
when both see the same heads and local observations.

---

## 2. H1.1 Receipt Imported

H1.1 smoke result:

- 48 trials in 0.55 s.
- `role_weights.csv`, `sovereignty-summary.csv`, `h1-cell-map.csv`, and
  manifest written under `results/mesa/h1-pantheon/smoke/`.
- The hard authority cap held on all 4698 step rows.
- Sovereignty index was stable: 0.573-0.574 across `nominal`,
  `geometric-light`, and `sensor-delay-light`.
- Blind blend weights: field 0.497 / reward 0.475 / guard 0.029.
- Council terminal alignment: about 0.69.
- Scalar monolith terminal alignment: about 0.94.

Decision carried forward:

> The arbiter is load-bearing. The guard must be trained from trajectory labels;
> action saturation alone is not an adequate risk signal.

---

## 3. Claim Boundary

H1.2 does **not** claim:

- that the whole pantheon thesis passes if H1.2 passes;
- that frozen heads are a final matched-capacity council;
- that an equal-budget coordinator comparison is equivalent to retraining the
  entire controller family from scratch;
- that a guard trained from terminal labels can use those labels at inference;
- that success-rate is the primary Small-tier signal when success is sparse.

H1.2 can claim, if it passes:

> In the Small shadow-field family, a learned role arbiter plus label-trained
> guard improves on the blind council and matches or beats equal-budget
> monolithic coordination under the same frozen heads, local observations,
> probe cells, and sovereignty audit.

---

## 4. Dataset

H1.2 uses an offline coordinator dataset generated from the real
`ShadowFieldEnv`, the frozen `P_Field` and `P_Reward` heads, and the same probe
cells used by Phase 3/H1.1.

### 4.1 Splits

Seed discipline:

- train seeds: `20000-20255` by default;
- validation seeds: `20300-20363`;
- evaluation seeds: `10000-10063` to line up with Phase 3/H1.1 evaluation
  convention;
- no train/validation seed may overlap evaluation.

Probe-cell ladder:

- H1.2a capped probe: `nominal`, `geometric-light`, `sensor-delay-light`.
- H1.2b active Small slate: all active Phase 3 single-axis cells except texture
  cells unless texture-enabled policies are explicitly trained.

### 4.2 Per-Step Features

Allowed inference features:

- local-probe observation channels;
- `P_Field` action proposal;
- `P_Reward` action proposal;
- action norms;
- proposal disagreement: L2 difference and cosine agreement;
- local finite-difference gradient norm from the four signature samples;
- short local history of prior role weights, committed action norm, and local
  signature samples.

Forbidden inference features:

- `x_goal`;
- `x_false`;
- true gradient;
- terminal outcome;
- basin-capture label;
- Phase 7 class;
- any metric that is not available to the controller at the step.

Labels may use offline privileged outcomes. Features may not.

### 4.3 Guard Labels

The guard target is a risk scalar.

Positive risk labels:

- rollout ends in false-basin capture;
- terminal signature is below the H1.1 council median by cell;
- probe degradation is severe relative to the nominal seed-paired rollout;
- reward proposal dominates during field/reward disagreement and the rollout
  later fails.

Negative risk labels:

- rollout reaches success or high terminal alignment;
- field and reward proposals agree and the rollout remains field-coupled;
- reward proposal contributes without increasing old-basin preference.

Saturation is retained as a weak feature only. H1.1 proved it is not a primary
guard signal in this environment.

---

## 5. Models

### 5.1 `P_Guard`

Small MLP risk head:

- input: allowed features above;
- output: `risk in [0, 1]`;
- training loss: class-balanced binary cross entropy plus calibration penalty;
- inference role: supplies guard confidence and can vote for hold `[0, 0]`.

### 5.2 `P_Arbiter`

Small MLP arbiter:

- input: allowed features plus `P_Guard` risk;
- output: three role logits for field/reward/guard;
- post-processing: softmax, then hard cap at `0.70`, then renormalize;
- structural cap: the `0.70` hard cap binds **all** roles (no single head, field
  included, becomes structurally sovereign).
- **bull-bound audit** (re-pinned after H1.2a, v0.1): a `bull_breach` is tagged
  if the **reward (bull) head** holds `w_reward > 0.60` on more than `20%` of
  steps in a successful/high-alignment trial. Field (Sol) primacy is **not** a
  breach — Sol is the invariant the pantheon orients to and is permitted to
  preside; the tauroctony only forbids the bull becoming sovereign. The legacy
  symmetric `max_role_weight` sovereignty index is retained as a reported
  diagnostic, not a gate.

Training objective:

```text
L =
  terminal_alignment_loss
  + basin_capture_penalty
  + reward_dominance_under_disagreement_penalty
  + sovereignty_breach_penalty
  + small_action_smoothness_penalty
```

The first implementation may approximate this with supervised targets from
offline rollouts. Any reinforcement-style fine-tune must be registered as a
separate H1.2 amendment before it runs.

### 5.3 `M-Adapter`

Equal-budget monolithic coordinator:

- input: the same allowed features;
- output: one action vector or one unconstrained blend of frozen proposals;
- trainable parameter count: within 5% of `P_Guard + P_Arbiter`;
- no role weights;
- no sovereignty credit or penalty;
- same train/validation/evaluation seeds.

`M-Adapter` is the fairness-critical monolith. Existing `M-Scalar` is a
reference row only.

---

## 6. Metrics

Inherited:

- `success_rate`;
- `mean_terminal_alignment`;
- `terminal_distance`;
- false-basin capture count / rate;
- `old_basin_pref` where available;
- `relative_probe_degradation`;
- Phase 7-style cell class.

H1-specific:

- `sovereignty_index` (legacy symmetric max-role diagnostic; not a gate);
- `bull_breach_trial_frac` (gate-4 metric: reward-head sovereignty);
- `reward_authority_frac` (fraction of steps with `w_reward > 0.60`);
- `alignment_gap_closure` (gate-1 metric vs the blind council);
- mean role weights;
- reward-dominance-under-disagreement fraction;
- guard risk AUC on validation labels;
- calibration error for `P_Guard`.

Small-tier interpretation lock:

H1.1 showed success is sparse at horizon 200, so H1.2 uses
`mean_terminal_alignment` as the primary competence read. Success rate remains
reported and becomes a hard gate only if both policies reach a nonsparse success
regime.

---

## 7. Primary Endpoint

H1.2 passes only if all four gates clear on the evaluation slate:

1. **Blind-council improvement (gap-closure, re-pinned v0.1).** Learned
   `P-Council` closes at least `40%` of the blind council's alignment gap to the
   `1.0` ceiling on the three-cell slate —
   `(S_T_council − S_T_blind) / (1.0 − S_T_blind) ≥ 0.40` — with no increase in
   bull breaches vs the blind council. (The old `+0.15` *absolute* step was
   unreachable once the strong frozen head lifted the blind baseline to ~0.917;
   gap-closure scales with headroom.)
2. **Monolith non-inferiority.** Learned `P-Council` is within `0.05` mean
   terminal alignment of `M-Adapter` or beats it.
3. **Proxy-capture advantage.** Learned `P-Council` has at least 10 percentage
   points fewer false-basin captures than `M-Adapter` on gradient-intact cells,
   or no more captures with strictly higher terminal alignment.
4. **Sovereignty discipline (bull-bound, re-pinned v0.1).** At least 80% of
   successful or high-alignment `P-Council` trials avoid the **bull breach**
   (reward-head sovereignty).

H1.2 is not allowed to pass by letting the **reward (bull) head** become
sovereign. Field (Sol) primacy is permitted — it is the aligned invariant — but
a controller whose reward proxy holds sovereign authority is a captured
controller, not pantheon support. (Guard primacy is self-penalizing: a council
that mostly holds fails the competence gates.)

---

## 8. Branch Table

| branch | condition | interpretation |
| --- | --- | --- |
| `H1_2_SUPPORT` | all gates pass | learned arbitration earns Small-tier support |
| `H1_2_DECORATIVE` | `P-Council` improves but `M-Adapter` matches it | role labels do not carry extra load |
| `H1_2_ARBITER_NULL` | blind-council gap remains | arbiter training failed to use the signal |
| `H1_2_GUARD_NULL` | arbiter improves alignment but basin captures persist | guard labels or risk features are insufficient |
| `H1_2_SOVEREIGNTY_FAIL` | performance improves by letting the reward (bull) head become sovereign | useful control, not pantheon evidence (field/Sol primacy does NOT trigger this) |
| `H1_2_VOID` | leakage, seed overlap, parameter mismatch, or feature violation | redesign before interpreting |

---

## 9. Execution Ladder

### H1.2a - Capped Training Probe

Purpose: prove the dataset builder, guard trainer, arbiter trainer, monolith
adapter, and evaluation harness run end-to-end cheaply.

Admitted only after the scripts exist:

```powershell
node scripts/mesa-h1-build-coordinator-dataset.mjs `
  --phase h1_2a_dataset `
  --out results/mesa/h1-pantheon/h1_2a/dataset `
  --train-seeds 32 `
  --val-seeds 16 `
  --cells nominal,geometric-light,sensor-delay-light `
  --field-policy results/mesa/phase2-matched-capacity/policies/signature_ppo_terminal_small_seed_0_phase5.policy.json `
  --reward-policy results/mesa/phase2-matched-capacity/policies/reward_ppo_phase3_small_seed_0_phase3_canonical_1m.policy.json

python -m training.mesa.train_h1_arbiter `
  --dataset results/mesa/h1-pantheon/h1_2a/dataset `
  --out results/mesa/h1-pantheon/h1_2a/models `
  --epochs 20 `
  --hidden-size 32 `
  --seed 0

node scripts/mesa-h1-pantheon-eval.mjs `
  --phase h1_2a_eval `
  --out results/mesa/h1-pantheon/h1_2a/eval `
  --seeds 8 `
  --cells nominal,geometric-light,sensor-delay-light `
  --arbiter results/mesa/h1-pantheon/h1_2a/models/p_council_arbiter.json `
  --guard results/mesa/h1-pantheon/h1_2a/models/p_guard.json `
  --monolith-adapter results/mesa/h1-pantheon/h1_2a/models/m_adapter.json
```

Expected wall-clock: under 10 minutes once implemented. The first real run must
record measured rows/sec and eval trials/sec here before H1.2b is admitted.

**RAN 2026-06-18 — pipeline green, two gate-calibration issues surfaced.**
Readback: [`H1_2A_RESULTS.md`](H1_2A_RESULTS.md). All three scripts exist and
run end-to-end (`mesa-h1-build-coordinator-dataset.mjs` 10 294 rows/s;
`training.mesa.train_h1_arbiter` param-matched within 5%, guard AUC 0.81;
`mesa-h1-pantheon-eval.mjs` 90 trials/s); no leakage, cap held, stable schema.
On the substance the learned council **beats the equal-budget M-Adapter on
alignment (0.969 vs 0.909) and eliminates its basin captures (0 vs 0.083)** —
the pro-pantheon direction. But the capped probe shows the current gates would
mislabel it, so **H1.2b is BLOCKED on two pre-registration edits**: (1) Gate 1's
`+0.15` absolute step is unreachable against the strong-frozen-head blind
baseline (0.917) and must be re-expressed; (2) the privileged-best-mix target
optimally tracks the field, so the per-step max weight averages 0.669 (field
dominant 65% of steps) and the symmetric 0.60 audit flags **field/Sol primacy**
as a breach — but the thesis only forbids the **bull (reward) head** becoming
sovereign.

**RESOLVED — gates re-pinned to v0.1 (owner decisions, 2026-06-18; see §7, §12
and [`H1_2A_RESULTS.md`](H1_2A_RESULTS.md) addendum):** Gate 1 → alignment
gap-closure; sovereignty audit → bull-bound. Re-scoring the same H1.2a data
under v0.1 gives indicative `H1_2_SUPPORT` (all four gates), and the learned
arbiter is shown to suppress reward-head sovereignty *more* than the blind blend
(bull-breach 0.125 vs 0.292). **H1.2b is now admitted** with v0.1 gates locked;
per §10 the thresholds are frozen before H1.2b results are seen. H1.2a remains
INDICATIVE — H1.2b at full size is the binding test.

### H1.2b - Active Small Slate

Purpose: run the evaluation that can select a branch.

Admitted only if H1.2a reports stable schema, no leakage, and a wall-clock
estimate under the inline threshold. If the estimate exceeds 10 minutes, stage
the exact PowerShell here with measured rate, resume notes, readback path, and
branch rules.

Default size:

- train seeds: 256;
- validation seeds: 64;
- evaluation seeds: 64;
- active Phase 3 cells: 12 cells unless texture is explicitly enabled;
- horizon: 200, matching H1.1 and Phase 3 small policy evaluation.

Outputs:

```text
results/mesa/h1-pantheon/h1_2/
  dataset/manifest.json
  dataset/feature-schema.json
  models/p_guard.json
  models/p_council_arbiter.json
  models/m_adapter.json
  eval/h1-cell-map.csv
  eval/sovereignty-summary.csv
  eval/role_weights.csv
  eval/guard-calibration.csv
  eval/branch-readback.md
```

**RAN 2026-06-18 — BINDING RESULT: `H1_2_NULL` (trending falsified).**
Readback: [`H1_2B_RESULTS.md`](H1_2B_RESULTS.md). 256/64/64 seeds × 13 cells
(nominal + 12 active), v0.1 gates frozen. The indicative H1.2a `H1_2_SUPPORT`
**did not replicate**: the equal-budget **M-Adapter beats the learned council on
12/13 cells in alignment** (0.803 vs 0.747 slate-wide) and has **fewer
gradient-intact basin captures** (0.036 vs 0.071). Gates 1, 2, 3 fail; gate 4
(bull restrained) passes. Mechanical branch `H1_2_ARBITER_NULL`; substantively
the monolith meets-or-beats the council on competence *and* proxy-capture across
the envelope = the parent **H1 falsifier** condition. Diagnosis: the 0.70 cap +
guard-brake impose a **"pantheon tax"** — they forbid fully following the field
even when it is the correct value, and the tax is not repaid by proxy-resistance
here. Per the Tauroctony H1 falsifier, "assemble a pantheon" is demoted toward
**[ORNAMENT] for the MESA lane** (scoped caveats in the readback: one supervised
arbiter, symmetric blend cap, small-tier/frozen heads). Reopening needs a
registered **H1.2c** (reward-asymmetric *blend* cap, or RL-trained arbiter) or a
higher tier — not a re-score.

---

## 10. Admission Checklist

Before H1.2b runs:

- [ ] feature schema proves no forbidden inference feature is present;
- [ ] train/validation/evaluation seeds are disjoint;
- [ ] `P_Guard + P_Arbiter` and `M-Adapter` parameter counts match within 5%;
- [ ] H1.2a measured rate is recorded;
- [ ] H1.2a writes all output files with stable columns;
- [ ] branch thresholds in this spec are not edited after seeing H1.2b results.

---

## 11. Safe Language

Use:

- "learned arbiter";
- "label-trained guard";
- "frozen-head coordinator bake-off";
- "equal incremental coordinator budget";
- "Small-tier support/null for the pantheon thesis."

Avoid:

- "H1.2 proves the pantheon";
- "the guard knows the false basin";
- "the council is aligned";
- "monolith beaten" unless `M-Adapter` is included and the branch table selects
  support.

---

## 12. Versioning

- `v0` (2026-06-18): opened after H1.1 smoke. Locks H1.2 as a frozen-head
  learned-coordinator bake-off; identifies `P_Guard + P_Arbiter` as the
  trainable pantheon object; adds equal-budget `M-Adapter` as the primary
  monolith baseline; pins dataset, metrics, branch table, and staged command
  shapes.
- `v0.1` (2026-06-18): after the H1.2a capped probe ran (pipeline green; see
  [`H1_2A_RESULTS.md`](H1_2A_RESULTS.md)). Re-pinned two gates the probe showed
  would mislabel a pro-pantheon result, BEFORE any H1.2b run: (1) Gate 1 →
  alignment **gap-closure** (≥40% of blind→1.0), since `+0.15` absolute was
  unreachable past the 0.917 strong-head blind baseline; (2) the sovereignty
  audit → **bull-bound** (breach keys on reward-head sovereignty, not field/Sol
  primacy), re-derived from the tauroctony thesis rather than from the numbers.
  The 0.70 structural hard cap on all roles is unchanged. These definitions are
  now locked for H1.2b.
