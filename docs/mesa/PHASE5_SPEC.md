# Mesa Phase 5 — Selection-Pressure Curriculum

This document is the implementation-grade spec for Phase 5 of
[`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md). Phases 1-4 produced a single
canonical comparison point per family at two capacity tiers, plus the
Phase 4 fixed-attractor receipt. Phase 5 maps the *training-regime* space:
how do basin absorption, Sundog-cost, and probe-resistance vary as we
sweep the shape of the training signal at fixed architecture?

Where this spec and the roadmap disagree, the roadmap wins. Where both
are silent, this spec is authoritative for Phase 5.

## 1. Decision Lock

Phase 5 v1 starts with seven pinned calls:

- **Three axes only.** Axis A: L-Mixed λ sweep. Axis B: L-Signature
  objective shape. Axis C: curriculum order. Other selection-pressure
  variants from PHASE0_SPEC §5 (reward-shape adversary, sparse-vs-dense
  reward, hindsight relabeling) are deferred to Phase 5 v2 once v1
  surfaces where the protection-breach threshold sits.
- **Axis A pins λ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}** at Small tier (5 runs).
  λ=0.5 is the existing Phase 3 canonical and is reused, not retrained.
- **Axis B pins three signature objective shapes** at Small tier:
  terminal, integrated (existing canonical), threshold.
- **Axis C pins two curriculum orders** with 50%+50% phase split (500K
  pretrain + 500K fine-tune env steps each).
- **Medium-tier coverage is gated on Small λ-sweep results.** After
  Small Axis A lands, run Medium for the most-informative subset
  (default: λ ∈ {0.3, 0.5, 0.7}).
- **Evaluation reuses Phase 3 probe slate and Phase 4 intervention
  battery without modification.** Same seed slate (10000-10063), same
  cells, same channels.
- **Headline metric is the protection curve**: `old_basin_pref` as a
  function of λ, with the breach threshold (the λ at which
  `old_basin_pref` crosses 1.0) reported as a single program-level
  number.

Total v1 compute: 10 Small-tier training runs + 3 Medium-tier follow-up
runs ≈ ~80 minutes total wall-clock.

## 2. Scope

Phase 5 v1 owns:

- L-Mixed λ sweep at Small (5 runs, with λ=0.5 reused from Phase 3).
- L-Signature objective shape sweep at Small (3 runs, with integrated
  reused from Phase 2).
- Curriculum order sweep at Small (2 runs).
- Medium-tier follow-up for the most-informative λ subset (3 runs).
- Trainer additions: `--mixed-lambda` flag, `--signature-shape` flag,
  `--load-checkpoint` flag for curriculum runs.
- Probe-slate evaluation of every new policy (reuses Phase 3 harness).
- Intervention-battery evaluation of every new policy (reuses Phase 4
  harness).
- Phase 5 aggregate report producing the protection curve, breach
  threshold, and per-axis tables.

Phase 5 does **not** own:

- Phase 6 interpretability probes.
- Phase 7 operating-envelope cross-product.
- Reward-shape adversary during training (deferred to Phase 5 v2).
- λ × signature-shape product runs (deferred to Phase 5 v2).
- Texture-channel selection-pressure variants (deferred to Phase 5 v3
  per PHASE3_SPEC §5.3).
- Large tier runs (deferred indefinitely; Medium covers the
  capacity-scaling question for v1).

## 3. Axes

### 3.1 Axis A — L-Mixed λ sweep

The headline axis. Maps how the signature anchor's protection scales
with the proportion of canonical reward (basin-corrupted) in the
mixed training signal.

**Training signal:**

```
R_mixed(s, a) = (1 - λ) · J_signature(s) + λ · R_canonical(s, a)
```

where `J_signature(s) = signature(s)` (per-step integrated, canonical
shape) and `R_canonical(s, a) = dense(s) − α · ||a||² + β · false_basin(s)`
with α=0.05, β=2.0, x_false=(-2.5, -2.5) as Phase 3 pinned.

**Pinned λ values:**

| λ | Family slug | Existing? |
| ---: | --- | --- |
| 0.1 | `mixed_ppo_phase3_lambda_0_1` | new |
| 0.3 | `mixed_ppo_phase3_lambda_0_3` | new |
| 0.5 | `mixed_ppo_phase3_lambda_0_5` | reuse Phase 3 canonical |
| 0.7 | `mixed_ppo_phase3_lambda_0_7` | new |
| 0.9 | `mixed_ppo_phase3_lambda_0_9` | new |

**Tier coverage:**

- Small (1M env steps): all 5 λ values.
- Medium (10M env steps): λ ∈ {0.3, 0.5, 0.7} only. λ=0.1 and λ=0.9 at
  Medium are likely close to L-Signature Medium and L-Reward canonical
  Medium respectively; running them at Medium adds little marginal
  information.

**Hyperparameters:** match Phase 3 canonical (lr=1e-4 Medium / 3e-4
Small, batch_envs=128/64, rollout=256/64, minibatch=1024/256, updates=305/122
Medium/Small).

### 3.2 Axis B — L-Signature objective shape

Tests whether the Phase 2 Sundog-cost finding (capacity doesn't help)
is signature-shape-bound or architecture-bound.

**Three shape variants:**

```
J_S_terminal(τ)     = S(x_T)                              # terminal only
J_S_integrated(τ)   = (1/T) · Σ_t S(x_t)                  # current canonical
J_S_threshold(τ)    = (1/T) · Σ_t 1{S(x_t) > τ_S}         # binary at τ_S
```

with `τ_S = 0.5` (per PHASE0_SPEC §3.5).

**Implementation:** the JS env's `rewardChannels.signature` returns
`S(x_t)` per step regardless. Shape transformations happen in the Python
trainer's `reward_from_channels` function based on the `--signature-shape`
flag and the `is_terminal_step` runtime context. No JS change.

**Pinned variants:**

| Shape | Family slug | Existing? |
| --- | --- | --- |
| `terminal` | `signature_ppo_terminal_small` | new |
| `integrated` | `signature_ppo_dense_small` | reuse Phase 2 canonical |
| `threshold` | `signature_ppo_threshold_small` | new |

**Tier coverage:** Small only for v1. Medium deferred (lifted to Phase 5
v1.1 — see versioning).

**Post-v1 correction (2026-05-12):** Phase 5 Small slate falsified prediction
B1 in the program-significant direction. Terminal-only signature reached 37/64
success at Small vs integrated's 5/64, demonstrating that the Phase 2-3
"Sundog-cost gap" was a signal-shape artifact, not a structural property of
state-only training. Terminal-only is now the recommended canonical L-Signature
shape for future Phase 5+ work. Integrated is retained as the historical
canonical for backward compatibility with Phase 2-4 result notes; new work
should adopt terminal-only and re-canonicalize when convenient.

### 3.3 Axis C — Curriculum order

Tests whether the basin attractor (Phase 4 fixed-attractor finding)
persists across subsequent training in a different signal regime, and
whether starting with signature-only protects against later
basin-corrupted fine-tuning.

**Two variants:**

1. **Signature-first → reward fine-tune.**
   - Pretrain: 500K env steps on L-Signature integrated.
   - Fine-tune: 500K env steps on L-Reward canonical (β=2.0).
   - Family slug: `curriculum_sig_then_reward_small`.

2. **Reward-first → signature fine-tune.**
   - Pretrain: 500K env steps on L-Reward canonical.
   - Fine-tune: 500K env steps on L-Signature integrated.
   - Family slug: `curriculum_reward_then_sig_small`.

**Phase split rationale:** 50%+50% gives clean per-phase budget and
matches the canonical 1M total. Alternative would be full pretrain (1M)
+ short fine-tune (200K), but that's confounded with "pretrain
saturation"; the 50%+50% scheme keeps the pretrain and fine-tune phases
comparable in influence.

**Implementation:** new `--load-checkpoint <path>` flag in `train_ppo.py`
that loads optimizer state + actor/critic weights and continues training
under the new `--variant`. Reset the optimizer's running statistics
optionally via `--reset-optimizer` (default off, so Adam momentum
carries over — this is the most realistic continuation).

**Tier coverage:** Small only for v1.

## 4. Pre-Registered Predictions

Four load-bearing predictions, all checkable from probe-slate +
intervention-battery outputs.

### 4.1 (A1) L-Mixed `old_basin_pref` is monotone in λ

At λ=0.1, the signature anchor dominates: expect `old_basin_pref` near
zero (≤ 0.5), comparable to L-Signature's -0.092. At λ=0.9, the reward
signal dominates: expect `old_basin_pref` near 3.0+, approaching
L-Reward canonical Small's 3.413. The intermediate values should map a
*smooth* interpolation.

**Falsifier:** a cliff between two consecutive λ values (e.g.,
`old_basin_pref` jumps from 0.2 to 2.5 between λ=0.6 and λ=0.7) would
suggest threshold effects worth investigating in Phase 6.

### 4.2 (A2) Protection-breach threshold lies in (0.5, 0.7]

Phase 3 found L-Mixed Small λ=0.5 has zero basin captures and
`old_basin_pref = -0.39`. The breach threshold — the λ at which
`old_basin_pref` first crosses 1.0 — should lie between 0.5 and 0.7 at
Small tier.

**Falsifier:** breach threshold below 0.3 would suggest the signature
anchor is much weaker than Phase 4 implied; threshold above 0.7 would
suggest the anchor is unexpectedly strong and the Phase 3 0.5 result
generalizes further than expected.

### 4.3 (B1) Terminal-only signature shaping worsens Sundog-cost

Per-step `S(x_t)` (current canonical) provides gradient information
throughout the episode. Terminal-only `S(x_T)` is sparse-reward-
equivalent. Predict at Small tier:

```
terminal_success / integrated_success ≤ 0.5
```

Specifically: integrated L-Signature reaches 5/64 success at Small;
predict terminal-only L-Signature reaches ≤ 2/64 success.

**Falsifier:** terminal-only matching or beating integrated would
indicate the dense per-step signal isn't doing useful learning work,
which would itself be an interesting finding about what the signature
training is actually optimizing.

### 4.4 (C1) Reward-first → signature fine-tune retains basin attraction

If the basin attractor is in the weights (Phase 4 confirmed),
signature fine-tuning on a basin-corrupted starting point shouldn't
fully erase it. Predict:

```
old_basin_pref(reward-first → sig-fine-tune) > 1.0
```

vs L-Signature-from-scratch's -0.092. The post-fine-tune policy should
retain measurable basin attraction even after 500K env steps of
signature-only training.

**Falsifier:** `old_basin_pref ≤ 0.5` would indicate signature
fine-tuning *can* erase the basin attractor, which would be a
program-significant finding for alignment-research framing — it would
suggest that fixing a Goodharted policy is tractable via clean-signal
fine-tuning.

The symmetric prediction for signature-first → reward-fine-tune is
*not* pre-registered, because the result is harder to interpret: the
reward-corruption phase will likely overwrite the signature pretraining
regardless of order. Worth measuring, but doesn't carry a load-bearing
prediction.

## 5. Naming Convention

Slug convention extends existing `<variant>_<tier>_seed_<N>_<run-label>`
pattern.

### 5.1 L-Mixed λ sweep

```
mixed_ppo_phase3_lambda_0_1_small_seed_0_phase5_lambda_0_1.policy.json
mixed_ppo_phase3_lambda_0_3_small_seed_0_phase5_lambda_0_3.policy.json
mixed_ppo_phase3_lambda_0_5_small_seed_0_phase3_canonical_1m.policy.json   # reused
mixed_ppo_phase3_lambda_0_7_small_seed_0_phase5_lambda_0_7.policy.json
mixed_ppo_phase3_lambda_0_9_small_seed_0_phase5_lambda_0_9.policy.json
```

Probe-slate / intervention-battery output slugs:

```
results/mesa/phase3-probe-slate/l_mixed_lambda_0_1_small/
results/mesa/phase4-intervention-battery/l_mixed_lambda_0_1_small/
... and so on
```

Policy labels for CSVs: `L-Mixed-λ0.1`, `L-Mixed-λ0.3`, etc.

### 5.2 L-Signature objective shape

```
signature_ppo_terminal_small_seed_0_phase5.policy.json
signature_ppo_dense_small_seed_0_canonical_1m.policy.json                  # reused (integrated)
signature_ppo_threshold_small_seed_0_phase5.policy.json
```

Policy labels: `L-Signature-Terminal`, `L-Signature-Integrated`,
`L-Signature-Threshold`.

### 5.3 Curriculum

```
curriculum_sig_then_reward_small_seed_0_phase5.policy.json
curriculum_reward_then_sig_small_seed_0_phase5.policy.json
```

Policy labels: `Curriculum-Sig-Then-Reward`, `Curriculum-Reward-Then-Sig`.

## 6. Metrics

No new metrics. Phase 5 reuses:

- **Phase 3 probe slate** (12 cells × 64 seeds): success rate, mean
  S_T, relative degradation per cell, failure-pattern classification
  (especially basin captures).
- **Phase 4 intervention battery** (5 channels × 64 seeds):
  action_response_L2, terminal_position_divergence, old_basin_pref
  (the key Phase 5 number).

The Phase 5-specific aggregates are *cross-policy*:

- **Protection curve**: `old_basin_pref` as a function of λ across the
  Axis A sweep.
- **Breach threshold**: smallest λ where `old_basin_pref > 1.0` (linear
  interpolation between sampled values if needed).
- **Sundog-cost sensitivity table**: success rate × signature shape
  across the Axis B sweep.
- **Curriculum persistence table**: `old_basin_pref` post-curriculum
  vs pretrain-only baselines, plus probe-slate basin-capture rates.

## 7. Harness — Trainer Changes

Three small additions to `training/mesa/train_ppo.py`.

### 7.1 `--mixed-lambda <float>` flag

Currently `mixed_ppo_phase3_lambda_0_5` has λ baked into the variant
config. Replace with a flag:

```python
parser.add_argument("--mixed-lambda", type=float, default=None,
                    help="override λ for mixed variants; ignored if variant is not mixed")
```

Validate `0 < λ < 1` if provided. When `--variant mixed_ppo_phase3_lambda_0_5
--mixed-lambda 0.3`, the trainer uses λ=0.3 but logs the variant name
as the slug component. Cleaner alternative: rename the variant to just
`mixed_ppo_phase3` and require `--mixed-lambda` as a positional arg.
Either works; pick whichever is easier in the current trainer code.

### 7.2 `--signature-shape {terminal,integrated,threshold}` flag

In `reward_from_channels(channels, *, reward_mode, mixed_lambda,
signature_shape, is_terminal_step, signature_threshold=0.5)`:

```python
def signature_reward(channels, shape, is_terminal, threshold):
    s = float(channels["signature"])
    if shape == "terminal":
        return s if is_terminal else 0.0
    if shape == "integrated":
        return s  # current canonical
    if shape == "threshold":
        return 1.0 if s > threshold else 0.0
    raise ValueError(f"unknown signature_shape: {shape}")
```

Default: `integrated` (preserves Phase 2 canonical). The trainer must
pass `is_terminal_step` from the env step's `done` flag.

### 7.3 `--load-checkpoint <path>` flag

For curriculum runs. Load actor + critic weights from a previous
checkpoint and continue training under the current variant.

```python
parser.add_argument("--load-checkpoint", type=Path, default=None,
                    help="load actor+critic weights from a previous .pt file before training")
parser.add_argument("--reset-optimizer", action="store_true",
                    help="reset Adam optimizer state when loading checkpoint (default: carry over)")
```

When `--load-checkpoint` is provided:

1. Construct policy with current `policy_config_for_tier(args.tier)`.
2. Load state dict from checkpoint (assert architecture matches).
3. Optionally reset optimizer state.
4. Continue training under the new variant for `--updates` more updates.

Critical discipline: log the loaded checkpoint's slug in the manifest
so curriculum policies are traceable to their pretrain source.

### 7.4 No JS-side changes

The env, reward channels, and intervention affordances are unchanged.
Probe-slate and intervention-battery harnesses run against the new
policies without modification.

## 8. Outputs

```
results/mesa/phase5-selection-pressure/
  manifest.json
  policies-summary.csv                # one row per Phase 5 policy
  axis-a-lambda-sweep.csv             # rows: λ × tier; cols: nominal + key probe + key intervention
  axis-b-signature-shape.csv          # rows: shape; cols: nominal + key probe + key intervention
  axis-c-curriculum.csv               # rows: order; cols: nominal + key probe + key intervention
  reports/
    protection-curve.csv              # λ → old_basin_pref, with linear interpolation
    breach-threshold.json             # { "tier": "small", "lambda_breach": 0.6 }
    sundog-cost-by-shape.csv          # signature shape → success rate
    curriculum-persistence.csv        # curriculum order → old_basin_pref pre/post
```

The `manifest.json` lists every policy run as part of Phase 5, with
slug, source checkpoint (for curriculum runs), tier, λ (for Axis A),
signature shape (for Axis B), pretrain checkpoint (for Axis C), and
seed.

## 9. Execution Order

Recommended sequencing for Phase 5 v1:

1. **Trainer additions land first** (§7.1, §7.2, §7.3). Smoke-test each
   with a 4-update Small dry run before launching real jobs.
2. **Axis A Small λ sweep** (4 new runs: λ ∈ {0.1, 0.3, 0.7, 0.9}; λ=0.5
   reused). ~20 min total.
3. **Axis B shape sweep** (2 new runs: terminal, threshold; integrated
   reused). ~10 min total.
4. **Axis C curriculum** (2 runs, each is pretrain + fine-tune = 2
   phases). ~20 min total.
5. **Probe-slate evaluation** on all 8 new policies + 2 reused. ~25
   sec each = ~5 min total.
6. **Intervention-battery evaluation** on all 8 new policies + 2 reused.
   ~5 sec each = ~2 min total.
7. **Aggregate report generation** via new
   `scripts/mesa-phase5-aggregate.mjs` (mirrors
   `mesa-phase4-aggregate.mjs`).
8. **Medium-tier follow-up** for λ ∈ {0.3, 0.7} (λ=0.5 reused). ~60
   min training + ~5 min evaluation each.

Total wall-clock: ~80 min v1 Small slate, ~125 min counting Medium
follow-up.

## 10. Exit Criterion

Phase 5 v1 complete when:

- All 8 new Small training runs land checkpoints + policy.json.
- All 3 new Medium training runs land checkpoints + policy.json.
- Probe-slate and intervention-battery CSVs exist for every Phase 5
  policy.
- Protection curve (`old_basin_pref` vs λ) is plotted and the breach
  threshold is identified.
- At least one of the four pre-registered predictions in §4 is
  confirmed or falsified.
- Phase 5 result note (`docs/mesa/PHASE5_RESULTS.md`) is written with
  the protection curve as the headline.

## 11. What Phase 6 Inherits

Phase 6 (interpretability) is most valuable on a *diverse* policy zoo.
Phase 5 v1 produces:

- 5 L-Mixed policies spanning the (signature → reward) spectrum.
- 3 L-Signature policies under different objective shapes.
- 2 curricular policies that have been through two training phases.

This is 10 new policies (plus the 5 existing canonical ones = 15
total) covering a structured grid of training-regime variations. Phase
6 can:

- Compare hidden-layer representations across the λ sweep (does the
  "x_false detector" feature emerge at a specific λ?).
- Apply activation patching from L-Signature into L-Reward canonical
  at various depths (where does the basin attractor live?).
- Test linear-probe accuracy for x_false position as a function of
  training regime (which selection-pressures induce the feature?).
- Compare curriculum-end vs from-scratch policies at matched final
  training-signal regime (does the path through training space
  matter for representation, even when the endpoint signal is the
  same?).

Phase 5's job is to make the zoo *exist*. Phase 6 does the probing.

## 12. Non-Goals (Deferred Variants)

Phase 5 v1 deliberately omits:

- **Reward-shape adversary**: injecting reward shaping perturbations
  during training. Deferred to v2.
- **Sparse-vs-dense reward**: testing whether sparse success-only
  rewards behave differently from dense distance rewards. Deferred to
  v2.
- **Hindsight relabeling**: training on relabeled trajectories.
  Deferred to v2.
- **λ × signature-shape product**: e.g., L-Mixed with terminal-only
  signature half. Combinatorial; deferred to v2 if any Axis A or Axis
  B finding motivates it.
- **Per-seed variance**: only seed=0 is used at Small per existing
  conventions. Multi-seed runs deferred.
- **β sensitivity at Medium tier**: Phase 3 §β covered Small.
- **Large tier**: deferred indefinitely.

## 13. Open Questions

Resolved during Phase 5 implementation; if they shift the spec, the
spec is revised and re-versioned.

- **Curriculum pretrain budget**: 50%+50% pinned. If pretrain phase
  doesn't sufficiently absorb the basin attractor (Phase 4-style)
  before fine-tuning starts, the Axis C result is uninformative. Worth
  monitoring: pretrain phase's `old_basin_pref` at 500K should be
  comparable to L-Reward canonical Small's 3.4. If it's much lower,
  the pretrain budget was too short and the curriculum test is
  preempted.

- **λ resolution**: 5 values pinned for v1. If §4.1 shows monotone
  behavior with a sharp breach in a narrow λ band, v2 should resample
  at λ ∈ {0.5, 0.55, 0.6, 0.65, 0.7} for finer resolution. Defer this
  decision until v1 lands.

- **Medium tier subset selection**: pinned λ ∈ {0.3, 0.5, 0.7} as
  default, but if Small λ-sweep shows the breach is near λ=0.4 or
  λ=0.8, swap subsets to bracket the breach at Medium. Decision made
  after Small v1 lands.

- **Probe-slate sampling**: 64 seeds is Phase 3 convention. If λ-sweep
  reveals close-to-zero `old_basin_pref` differences between adjacent
  policies (e.g., λ=0.3 and λ=0.5 both show -0.1 ± 0.2), a larger seed
  slate (128 or 256) may be needed for statistical confidence. Defer
  this call until v1 lands.

## 14. Cross-References

- PHASE0_SPEC §5 — original selection-pressure axes (pinned 2026-05-10).
- PHASE2_SPEC — matched-capacity training infrastructure (L-Mixed, L-Signature variants).
- PHASE3_SPEC §10.4 — three-point capacity-dependence picture motivating Phase 5.
- PHASE3_RESULTS.md v2 — Phase 3 Small + Medium results with the λ=0.5 anchor point.
- PHASE4_SPEC + PHASE4_RESULTS.md — fixed-attractor receipt motivating the curriculum-persistence prediction.
- `scripts/mesa-probe-slate.mjs` — reused unchanged.
- `scripts/mesa-intervention-battery.mjs` — reused unchanged.
- `training/mesa/train_ppo.py` — gets §7.1, §7.2, §7.3 additions.
- `training/mesa/policy.py` — unchanged.

## 15. Versioning

- `v1` (2026-05-12): initial Phase 5 spec. Pinned three axes (λ sweep,
  signature shape, curriculum order), 10 Small training runs + 3
  Medium follow-up, protection-curve as headline metric, breach
  threshold as single program number. Predictions A1, A2, B1, C1
  pre-registered. Three trainer additions (mixed-lambda flag,
  signature-shape flag, load-checkpoint flag) scoped; no JS-side
  changes.

- `v1.1` (2026-05-12): post-Small-slate amendment. Pre-registered prediction
  outcomes recorded:
  - (A1) λ-monotonicity **confirmed**. Protection curve well-behaved across
    λ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}.
  - (A2) Breach threshold **confirmed** at λ ≈ 0.660 (interpolated), inside
    the predicted (0.5, 0.7] interval and toward the upper end. Signature
    anchor fully prevents basin absorption up to ~2:1 reward:signature mix
    at Small tier.
  - (B1) **Falsified in program-significant direction.** Terminal-only
    signature reaches 37/64 success at Small vs integrated's 5/64 — 7.4×
    the success rate. Reframes the Sundog-cost gap from 61 pp to 11 pp at
    Small, and re-canonicalizes terminal-only as the recommended L-Signature
    shape (§3.2 correction note).
  - (C1) **Partial confirmation.** Reward → signature curriculum erases
    basin attraction (`old_basin_pref = -0.585`) but does not recover
    task competence (0/64 success). Phase 5 v2 will test
    reward-pretrain → **terminal-signature**-fine-tune to disambiguate
    whether the recovery failure was a signal-shape artifact (likely,
    given B1) or a structural property of post-Goodhart fine-tuning.
  Signature → reward curriculum confirms symmetric prediction: clean
  signature pretrain offers no protection against later basin-corrupted
  reward fine-tuning (`old_basin_pref = 2.613`, 62 probe basin captures).
  Full result note: [`PHASE5_RESULTS.md`](PHASE5_RESULTS.md).

- `v1.2` (2026-05-12): Medium slate complete. Three new Medium runs
  (L-Mixed λ=0.3, L-Mixed λ=0.7, L-Signature Terminal Medium) plus the
  existing λ=0.5 Medium anchor. **Headline:** L-Signature Medium
  Terminal reaches 64/64 success at mean S_T = 0.9986 — Oracle ceiling
  performance, exceeding L-Reward-Clean Medium (49/64) by 23 pp. The
  Sundog-cost framing from Phase 2-3 is overturned: state-only
  signature training at correct shape is **better than** dense reward
  training at Medium scale. Integrated-signature canonical is
  deprecated; terminal-only is the new canonical. Medium λ protection
  curve is non-monotone (`old_basin_pref` 0.823 → 0.889 → 0.613 → 5.560
  across λ ∈ {0.3, 0.5, 0.7, 1.0}). Pre-registered Medium prediction
  outcomes: (A1-M) confirmed marginally, (A2-M) falsified in unexpected
  direction (λ=0.7 less basin-attracted than λ=0.5; success higher than
  λ=0.5), (B-M) confirmed and exceeded (64/64 vs predicted ≥45/64).
  Phase 5 v3 candidates: λ ∈ {0.8, 0.9} Medium curve characterization,
  reward-pretrain → terminal-signature-fine-tune curriculum (C1
  follow-up), and Small L-Sig-Terminal evaluation slate completion.
  Full result note: [`PHASE5_RESULTS.md`](PHASE5_RESULTS.md) v2 §8.
