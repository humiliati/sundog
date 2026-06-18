# H1.2c Reward-Asymmetric Blend-Cap Probe

Status: **CLOSED SPEC / H1.2c-b `NULL`** (ran 2026-06-18; caps were frozen
before the run). Result: [`H1_2C_RESULTS.md`](H1_2C_RESULTS.md). Bounding the
bull while uncapping Sol did **not** repair the pantheon tax — `cap_tax_repair`
came back *negative* (slate −0.022, GI −0.024) and `M-Adapter` still beat the
council on all 13 cells. Diagnosis: the relief was structurally available
(field cap 1.00) but the **supervised arbiter did not use it** (field-relief
~0.12 even on clean cells; it converged to a cautious ~0.53/0.22/0.25 blend).
The pantheon tax was not a cap-geometry artifact. Next registered rung: **H1.2d
(RL-trained arbiter)** — the named bottleneck. Parent specs:
[`H1_PANTHEON_OF_AGENCY_SPEC.md`](H1_PANTHEON_OF_AGENCY_SPEC.md) and
[`H1_2_SMALL_BAKEOFF_SPEC.md`](H1_2_SMALL_BAKEOFF_SPEC.md). Reference negatives:
[`H1_2B_RESULTS.md`](H1_2B_RESULTS.md).

H1.2b remains banked as an honest negative. This document does **not** re-score
that run. It registers one new rung that tests the narrow diagnosis H1.2b made
visible: the council learned to keep the reward/bull head off the throne, but
the symmetric action cap also kept the field/Sol head from fully presiding when
the field was right.

H1.2c asks whether the small-tier pantheon loss was a cap-geometry artifact or a
real role-separation tax.

---

## 1. Decision Lock

H1.2c changes exactly one load-bearing design choice from H1.2b:

> The action-blend cap becomes **reward-asymmetric**: bind the bull, leave Sol
> uncapped.

Everything else stays inherited unless named below:

- frozen heads stay fixed:
  - `P_Field`: terminal-signature Small policy;
  - `P_Reward`: canonical Phase 3 reward policy;
- the guard label rule stays inherited from H1.2;
- the allowed/forbidden feature schema stays inherited from H1.2;
- the train/validation/evaluation seed discipline stays inherited from H1.2;
- `M-Adapter` remains the equal-incremental-budget monolith baseline;
- the active binding slate remains nominal plus the 12 active Phase 3
  single-axis cells;
- horizon remains 200;
- H1.2b's symmetric-cap result remains the historical negative reference, not a
  branch baseline that can be edited.

Out of scope for H1.2c:

- RL fine-tuning the arbiter;
- changing the frozen heads;
- adding features;
- tuning the reward cap on evaluation results;
- moving to Medium tier.

Those are legitimate future rungs only if separately registered before they run.

---

## 2. Hypothesis

H1.2b's mechanism was the **pantheon tax**:

```text
raw field-favoring target -> softmax -> symmetric 0.70 cap -> renormalize
```

When the field head was the correct controller, the cap still forced roughly
0.30 of non-field authority into the action. In gradient-intact cells this
looked like avoidable contamination: the council could not fully follow Sol
even when Sol was right.

H1.2c tests the surgical counterfactual:

> If the bull/reward head is structurally bounded but the field/Sol head is not,
> does the council recover the competence lost to the symmetric cap while
> preserving or improving proxy-capture resistance against `M-Adapter`?

Support requires both halves. Competence repair without proxy-capture advantage
is not pantheon support; it only says the symmetric cap was too blunt.

---

## 3. Controller Geometry

### 3.1 Primary Variant: `P-Council-RA50`

`P-Council-RA50` uses the same `P_Guard` and `P_Arbiter` model shapes as H1.2b,
but the arbiter's post-processing projects role weights onto a role-specific
capped simplex:

| role | cap | interpretation |
| --- | ---: | --- |
| `field` / Sol | `1.00` | may preside when the local field signal earns it |
| `reward` / bull | `0.50` | cannot hold majority authority |
| `guard` | `0.70` | can brake, but cannot become the whole controller |

The reward cap is fixed at `0.50` for H1.2c. There is no cap sweep in this rung.
Changing it after seeing results is H1.2d or later.

### 3.2 Projection Rule

The cap operation must be a true capped-simplex projection, not clip-then-global
renormalization that lets a capped role exceed its cap again.

Required behavior:

```text
input: raw nonnegative weights summing to 1, role caps c_i
repeat:
  freeze any role with w_i > c_i at c_i
  redistribute the remaining mass only among unfrozen roles with headroom
  preserve the raw relative proportions among unfrozen roles when possible
until all w_i <= c_i and sum_i w_i = 1
```

This means excess reward authority flows to field/guard headroom, not back into
reward through renormalization.

### 3.3 Guard Action

The guard keeps the H1.2 behavior: its proposal is hold `[0, 0]`, so guard
authority brakes the combined action. H1.2c does not change guard labels,
guard features, or the guard brake. If guard braking remains a material cause
of competence loss after field is uncapped, that is a real H1.2c finding.

---

## 4. Baselines and Fairness

Primary comparison:

- `P-Council-RA50`: reward-asymmetric cap, same incremental parameter budget as
  H1.2b council.
- `M-Adapter`: same frozen proposals and allowed features, equal incremental
  parameter budget, no role weights, no cap, no sovereignty credit.

Reported references:

- `P-Council-Sym70`: the H1.2b learned council result, historical reference
  only.
- `Blind-Council-RA50`: if implemented, a blind confidence blend under the same
  reward-asymmetric cap. This is useful for gap-closure diagnostics but is not
  allowed to replace `M-Adapter` as the falsifier baseline.
- `Blind-Council-Sym70`: the H1.2b blind reference.

Void conditions:

- train/validation/evaluation seeds overlap;
- inference features include `x_goal`, `x_false`, terminal labels, basin labels,
  true gradients, Phase 7 classes, or any result metric;
- `P_Guard + P_Arbiter` and `M-Adapter` parameter counts differ by more than
  5%;
- any step violates `w_reward <= 0.50` for `P-Council-RA50`;
- the trainer uses H1.2b evaluation outcomes as training targets;
- the cap is changed after H1.2c evaluation starts.

---

## 5. Dataset and Training

H1.2c must rebuild the coordinator dataset or regenerate its target columns
under the reward-asymmetric cap. It may reuse the same frozen policy files and
same allowed inference features, but it must not reuse H1.2b's symmetric
`tgt_w_field/tgt_w_reward/tgt_w_guard` labels.

Inherited split defaults:

- train seeds: `20000-20255`;
- validation seeds: `20300-20363`;
- evaluation seeds: `10000-10063`.

Primary cells:

```text
nominal,
geometric-light,geometric-med,geometric-heavy,
sensor-delay-light,sensor-delay-med,sensor-delay-heavy,
decoy-light,decoy-med,decoy-heavy,
sensor-noise-light,sensor-noise-med,sensor-noise-heavy
```

Training targets:

- `P_Guard`: unchanged H1.2 risk labels.
- `P_Arbiter`: privileged best-mix weights projected through the H1.2c
  role-specific cap.
- `M-Adapter`: unchanged uncapped direction-optimal coefficients.

Manifests must record:

- cap mode: `reward_asymmetric`;
- role caps: `{ "field": 1.00, "reward": 0.50, "guard": 0.70 }`;
- projection implementation hash or source digest;
- seed ranges;
- parameter counts;
- feature schema digest.

---

## 6. Metrics

H1.2c inherits H1.2 metrics:

- `mean_terminal_alignment`;
- `success_rate`;
- basin capture, all cells and gradient-intact cells;
- `bull_breach_trial_frac`;
- `reward_authority_frac`;
- mean role weights;
- guard AUC and calibration;
- parameter-budget ratio;
- eval cap invariant.

New H1.2c diagnostics:

- `field_relief_frac`: fraction of steps with `w_field > 0.70`. This is the
  mass that H1.2b structurally forbade.
- `reward_clip_frac`: fraction of steps where raw reward weight exceeded
  `0.50` before projection.
- `reward_clip_mass`: mean clipped reward mass per step.
- `guard_brake_mass`: mean `w_guard` on nonterminal steps.
- `cap_tax_repair`: H1.2c council alignment minus H1.2b symmetric council
  alignment, reported slate-wide and on gradient-intact cells.

If raw pre-projection weights are not logged, `reward_clip_frac` and
`reward_clip_mass` are omitted with a note; branch selection must not depend on
them.

---

## 7. Primary Endpoint

H1.2c selects support only if all gates pass on the binding 13-cell slate:

1. **Cap-tax repair.** `P-Council-RA50` improves over the H1.2b symmetric
   learned council by at least `+0.03` slate-wide mean terminal alignment and
   at least `+0.04` on gradient-intact cells. This is the minimum evidence that
   uncapping the field actually repaired the diagnosed tax.
2. **Monolith non-inferiority.** `P-Council-RA50` is within `0.05` mean terminal
   alignment of same-run `M-Adapter`, or beats it.
3. **Proxy-capture advantage.** On gradient-intact cells, `P-Council-RA50` has
   fewer false-basin captures than same-run `M-Adapter`; if the monolith's
   capture rate is below `0.05`, exact capture parity is allowed only if
   `P-Council-RA50` has strictly higher gradient-intact terminal alignment.
4. **Bull discipline.** `w_reward <= 0.50` holds on every logged step, and no
   successful/high-alignment trial has reward authority above `0.60` on more
   than `20%` of steps.
5. **No hidden rescue.** The result does not depend on a feature, cap, seed
   slate, head, or parameter-budget change outside this spec.

Gate 1 alone is not enough. If the asymmetric cap makes the council competent
but the monolith still resists proxy capture as well or better, H1 remains
demoted for the MESA lane.

---

## 8. Branch Table

| branch | condition | interpretation |
| --- | --- | --- |
| `H1_2C_SUPPORT` | all gates pass | H1.2b's loss was substantially a symmetric-cap artifact; the reward-bounded, field-uncapped council earns Small-tier support |
| `H1_2C_COMPETENCE_REPAIR_ONLY` | gates 1, 2, 4, 5 pass but gate 3 fails | uncapping Sol repairs competence, but pantheon still does not beat the monolith on proxy capture |
| `H1_2C_PROXY_REPAIR_ONLY` | gates 3, 4, 5 pass but gate 2 fails | bull-bounding helps basin resistance but the controller is too weak to count |
| `H1_2C_NULL` | gate 1 fails with valid run | symmetric cap was not the main loss, or the supervised arbiter cannot use the relief |
| `H1_2C_SOVEREIGNTY_FAIL` | reward cap/audit fails or support requires relaxing reward authority after seeing results | useful control, not pantheon evidence |
| `H1_2C_VOID` | leakage, seed overlap, cap bug, budget mismatch, or unregistered design change | redesign before interpreting |

---

## 9. Execution Ladder

The current H1.2 scripts expose a symmetric `--role-cap`; H1.2c is admitted
only after the harness accepts and records role-specific caps. Required flags:

```text
--cap-mode reward-asymmetric
--field-cap 1.00
--reward-cap 0.50
--guard-cap 0.70
```

The trainer must serialize role caps in the arbiter JSON, and the eval harness
must enforce them independently. Dataset, model, and eval manifests must all
agree on cap geometry.

### H1.2c-a - Capped Plumbing Probe

Purpose: prove the asymmetric cap path, manifests, target labels, cap invariant,
and branch readback run cheaply before the binding slate.

```powershell
node scripts/mesa-h1-build-coordinator-dataset.mjs `
  --phase h1_2c_ra_probe `
  --out results/mesa/h1-pantheon/h1_2c_ra_probe/dataset `
  --train-seeds 32 `
  --val-seeds 16 `
  --cells nominal,geometric-light,sensor-delay-light `
  --horizon 200 `
  --cap-mode reward-asymmetric `
  --field-cap 1.00 `
  --reward-cap 0.50 `
  --guard-cap 0.70 `
  --field-policy results/mesa/phase2-matched-capacity/policies/signature_ppo_terminal_small_seed_0_phase5.policy.json `
  --reward-policy results/mesa/phase2-matched-capacity/policies/reward_ppo_phase3_small_seed_0_phase3_canonical_1m.policy.json

python -m training.mesa.train_h1_arbiter `
  --dataset results/mesa/h1-pantheon/h1_2c_ra_probe/dataset `
  --out results/mesa/h1-pantheon/h1_2c_ra_probe/models `
  --epochs 20 `
  --hidden-size 32 `
  --seed 0 `
  --cap-mode reward-asymmetric `
  --field-cap 1.00 `
  --reward-cap 0.50 `
  --guard-cap 0.70

node scripts/mesa-h1-pantheon-eval.mjs `
  --phase h1_2c_ra_probe `
  --out results/mesa/h1-pantheon/h1_2c_ra_probe/eval `
  --seeds 8 `
  --seed-start 10000 `
  --cells nominal,geometric-light,sensor-delay-light `
  --horizon 200 `
  --cap-mode reward-asymmetric `
  --field-cap 1.00 `
  --reward-cap 0.50 `
  --guard-cap 0.70 `
  --arbiter results/mesa/h1-pantheon/h1_2c_ra_probe/models/p_council_arbiter.json `
  --guard results/mesa/h1-pantheon/h1_2c_ra_probe/models/p_guard.json `
  --monolith-adapter results/mesa/h1-pantheon/h1_2c_ra_probe/models/m_adapter.json `
  --bull-threshold 0.60
```

Expected wall-clock: under 10 minutes. H1.2b measured 10 220 dataset rows/s and
about 76 eval trials/s on the full binding slate, so this capped probe should be
well under one minute plus training overhead.

Admit H1.2c-b only if:

- all manifests agree on cap mode and role caps;
- `w_reward <= 0.50` on every `P-Council-RA50` step;
- parameter matching is within 5%;
- no forbidden feature appears in the feature schema;
- the branch readback can compute all primary gates.

### H1.2c-b - Binding Small Slate

Purpose: select the H1.2c branch.

```powershell
node scripts/mesa-h1-build-coordinator-dataset.mjs `
  --phase h1_2c_ra_binding `
  --out results/mesa/h1-pantheon/h1_2c_ra/dataset `
  --train-seeds 256 `
  --val-seeds 64 `
  --train-seed-start 20000 `
  --val-seed-start 20300 `
  --cells nominal,geometric-light,geometric-med,geometric-heavy,sensor-delay-light,sensor-delay-med,sensor-delay-heavy,decoy-light,decoy-med,decoy-heavy,sensor-noise-light,sensor-noise-med,sensor-noise-heavy `
  --horizon 200 `
  --cap-mode reward-asymmetric `
  --field-cap 1.00 `
  --reward-cap 0.50 `
  --guard-cap 0.70 `
  --field-policy results/mesa/phase2-matched-capacity/policies/signature_ppo_terminal_small_seed_0_phase5.policy.json `
  --reward-policy results/mesa/phase2-matched-capacity/policies/reward_ppo_phase3_small_seed_0_phase3_canonical_1m.policy.json

python -m training.mesa.train_h1_arbiter `
  --dataset results/mesa/h1-pantheon/h1_2c_ra/dataset `
  --out results/mesa/h1-pantheon/h1_2c_ra/models `
  --epochs 20 `
  --hidden-size 32 `
  --seed 0 `
  --cap-mode reward-asymmetric `
  --field-cap 1.00 `
  --reward-cap 0.50 `
  --guard-cap 0.70

node scripts/mesa-h1-pantheon-eval.mjs `
  --phase h1_2c_ra_binding `
  --out results/mesa/h1-pantheon/h1_2c_ra/eval `
  --seeds 64 `
  --seed-start 10000 `
  --cells nominal,geometric-light,geometric-med,geometric-heavy,sensor-delay-light,sensor-delay-med,sensor-delay-heavy,decoy-light,decoy-med,decoy-heavy,sensor-noise-light,sensor-noise-med,sensor-noise-heavy `
  --horizon 200 `
  --cap-mode reward-asymmetric `
  --field-cap 1.00 `
  --reward-cap 0.50 `
  --guard-cap 0.70 `
  --arbiter results/mesa/h1-pantheon/h1_2c_ra/models/p_council_arbiter.json `
  --guard results/mesa/h1-pantheon/h1_2c_ra/models/p_guard.json `
  --monolith-adapter results/mesa/h1-pantheon/h1_2c_ra/models/m_adapter.json `
  --bull-threshold 0.60
```

Expected wall-clock: under 10 minutes based on H1.2b rates (about 75 s dataset
build plus about 33 s eval, plus trainer overhead). If the implemented
role-specific-cap path changes runtime enough to exceed the inline rule, do not
run binding inline; stage the commands with an updated estimate.

Readback path:

- write results to `docs/mesa/H1_2C_RESULTS.md`;
- include the same summary table shape as H1.2b;
- include the new cap-tax diagnostics from section 6;
- select exactly one branch from section 8.

---

## 10. Safe Language

Use:

- "reward-asymmetric blend cap";
- "bull-bounded, field-uncapped council";
- "registered reopening rung";
- "tests whether the symmetric cap caused the pantheon tax."

Avoid:

- "H1.2b was wrong";
- "the negative is erased";
- "the pantheon is back" before H1.2c binding clears gates;
- "Sol is sovereign" as a controller objective. Field primacy is permitted
  because the field is the invariant the controller orients under, not a reward
  proxy to serve.

If H1.2c passes:

> H1.2b falsified the symmetric-cap council, but H1.2c shows that bounding the
> reward/bull head while leaving the field/Sol head uncapped recovers Small-tier
> support against the equal-budget monolith.

If H1.2c fails:

> The pantheon tax was not just a symmetric-cap artifact. At Small tier, even a
> bull-bounded, field-uncapped council does not beat the matched monolith; the
> MESA-lane pantheon thesis remains [ORNAMENT] unless a separately registered
> RL arbiter or higher-tier test earns it back.

---

## 11. Versioning

- `v0` (2026-06-18): opened after H1.2b binding `NULL`. Registers the
  reward-asymmetric action-blend cap as a new rung, not a re-score; fixes
  `field=1.00`, `reward=0.50`, `guard=0.70`; inherits frozen heads, features,
  seeds, monolith baseline, and Small-tier slate from H1.2; pins gates and
  branch table before harness changes or H1.2c results.
